"""Web server FastAPI: giao diện tải PDF lên, theo dõi tiến độ và tải bản dịch."""

from __future__ import annotations

import logging
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from . import __version__
from .config import settings
from .jobs import manager
from .ir import Document
from .jobs import quality_summary
from .pdf_processor import (
    EXTRA_SUFFIXES,
    IR_FILENAME,
    PdfError,
    TranslateOptions,
    analyze_pdf,
    build_glossary,
    is_doubtful,
    local_llm,
    render_outputs,
)
from .pipeline import MATH_END, MATH_START, SUP_END, SUP_START
from .pipeline import TextTranslator
from .scientific import join_lines, load_builtin_glossary
from .terms import extract_terms
from .tm import get_tm
from .translators import TranslatorError, default_engine_key, get_engine, list_engines

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for _noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

STATIC_DIR = Path(__file__).parent / "static"
DOWNLOAD_NAMES = {
    "translated": ("application/pdf", "_vi.pdf"),
    "bilingual": ("application/pdf", "_song_ngu.pdf"),
    "markdown": ("text/markdown; charset=utf-8", "_vi.md"),
    "docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "_vi.docx"),
}
for _kind, _suffix in EXTRA_SUFFIXES.items():
    DOWNLOAD_NAMES[_kind] = (
        "text/html; charset=utf-8" if _suffix.endswith(".html") else
        "text/markdown; charset=utf-8" if _suffix.endswith(".md") else
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        _suffix,
    )



@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    manager.start()
    yield


app = FastAPI(title="Dịch PDF khoa học Anh → Việt", version=__version__, lifespan=lifespan)


@app.get("/api/engines")
def engines() -> dict:
    items = []
    for e in list_engines():
        ok, why = e.available()
        items.append({"key": e.key, "label": e.label, "description": e.description, "available": ok, "reason": why})
    return {"default": default_engine_key(), "engines": items}


@app.get("/api/glossary")
def glossary() -> dict:
    return {"entries": load_builtin_glossary()}


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    engine: str = Form(None),
    page_from: int | None = Form(None),
    page_to: int | None = Form(None),
    skip_references: bool = Form(True),
    use_glossary: bool = Form(True),
    custom_glossary: str = Form(""),
    make_bilingual: bool = Form(True),
    post_edit: bool = Form(True),
    highlight_doubtful: bool = Form(True),
    ocr: bool = Form(True),
    summary: bool = Form(False),
    plain: bool = Form(False),
) -> dict:
    engine = engine or default_engine_key()
    try:
        get_engine(engine)
    except TranslatorError as e:
        raise HTTPException(400, str(e)) from e
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File quá lớn (tối đa {settings.max_upload_mb} MB).")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "File tải lên không phải PDF.")
    opts = TranslateOptions(
        engine=engine,
        page_from=page_from or None,
        page_to=page_to or None,
        skip_references=skip_references,
        use_glossary=use_glossary,
        custom_glossary=custom_glossary[:200_000],
        make_bilingual=make_bilingual,
        post_edit=post_edit,
        highlight_doubtful=highlight_doubtful,
        ocr=ocr,
        summary=summary,
        plain=plain,
    )
    job = manager.submit(file.filename or "document.pdf", data, opts)
    return job.public()


class ExtrasRequest(BaseModel):
    summary: bool = True
    plain: bool = False


@app.post("/api/jobs/{job_id}/extras")
def create_extras(job_id: str, req: ExtrasRequest) -> dict:
    """Tạo thêm tóm tắt bài báo / bản dễ hiểu cho bài đã dịch."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Không tìm thấy công việc.")
    if job.status in ("queued", "running"):
        raise HTTPException(409, "Công việc đang chạy, hãy đợi xong.")
    if not (job.dir / IR_FILENAME).exists():
        raise HTTPException(400, "Bài này chưa dịch xong.")
    if not (req.summary or req.plain):
        raise HTTPException(400, "Chọn ít nhất một loại.")
    return manager.submit_extras(job, req.summary, req.plain).public()


@app.get("/api/llm")
def llm_status() -> dict:
    """LLM cục bộ có đang chạy không (để giao diện báo chế độ tóm tắt)."""
    llm = local_llm()
    return {"available": llm is not None, "model": settings.llm_model, "base_url": settings.llm_base_url}


@app.post("/api/terms")
async def analyze_terms(
    file: UploadFile = File(...),
    page_from: int | None = Form(None),
    page_to: int | None = Form(None),
) -> dict:
    """Phân tích nhanh (không dịch) để đề xuất thuật ngữ của bài cho người dùng duyệt."""
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File quá lớn (tối đa {settings.max_upload_mb} MB).")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "File tải lên không phải PDF.")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "input.pdf"
        path.write_bytes(data)
        try:
            ir = await run_in_threadpool(
                analyze_pdf, path, TranslateOptions(page_from=page_from or None, page_to=page_to or None)
            )
        except PdfError as e:
            raise HTTPException(400, str(e)) from e
    return {"terms": [t.to_dict() for t in extract_terms(ir)]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Không tìm thấy công việc.")
    d = job.public()
    d["queue_position"] = manager.queue_position(job)
    return d


@app.get("/api/jobs/{job_id}/download/{kind}")
def download(job_id: str, kind: str, inline: bool = False) -> FileResponse:
    job = manager.get(job_id)
    if job is None or kind not in job.outputs or kind not in DOWNLOAD_NAMES:
        raise HTTPException(404, "Không có file.")
    media, suffix = DOWNLOAD_NAMES[kind]
    stem = Path(job.filename).stem or "document"
    return FileResponse(
        job.outputs[kind],
        media_type=media,
        filename=stem + suffix,
        content_disposition_type="inline" if inline else "attachment",
    )


# ---------------------------------------------------------------------------
# Trang sửa bản dịch
# ---------------------------------------------------------------------------

_SUP_RE = re.compile(r"\^\{([^{}]*)\}")


def _to_editable(text: str | None) -> str | None:
    """Ký hiệu chỉ số trên nội bộ → dạng gõ được ^{...} trên trang sửa."""
    if text is None:
        return None
    return text.replace(SUP_START, "^{").replace(SUP_END, "}").replace(MATH_START, "").replace(MATH_END, "")


def _from_editable(text: str) -> str:
    return _SUP_RE.sub(lambda m: SUP_START + m.group(1) + SUP_END, text.strip())


def _job_ir(job_id: str):
    job = manager.get(job_id)
    if job is None or job.status != "done":
        raise HTTPException(404, "Không tìm thấy bản dịch (công việc chưa xong hoặc đã bị xóa).")
    path = job.dir / IR_FILENAME
    if not path.exists():
        raise HTTPException(404, "Không có dữ liệu bản dịch để sửa.")
    return job, path, Document.load(path)


@app.get("/api/jobs/{job_id}/document")
def get_document(job_id: str) -> dict:
    job, _, ir = _job_ir(job_id)
    regions = [
        {
            "id": r.id, "page": p.number, "kind": r.kind, "source": r.plain,
            "translation": _to_editable(r.translation), "flags": r.flags, "score": r.score,
            "notes": r.notes, "doubtful": is_doubtful(r),
        }
        for p in ir.pages for r in p.regions if r.kind == "text"
    ]
    return {"job": job.public(), "engine": ir.engine, "regions": regions}


class Edit(BaseModel):
    id: str
    translation: str


class EditRequest(BaseModel):
    edits: list[Edit]
    save_tm: bool = True


@app.put("/api/jobs/{job_id}/document")
async def save_document(job_id: str, req: EditRequest) -> dict:
    job, path, ir = _job_ir(job_id)
    by_id = {r.id: r for r in ir.regions}
    learned: dict[str, str] = {}
    for e in req.edits:
        r = by_id.get(e.id)
        if r is None or r.kind != "text":
            raise HTTPException(400, f"Không có đoạn {e.id}")
        new = _from_editable(e.translation)
        if not new:
            continue
        r.translation = new
        if "edited" not in r.flags:
            r.flags.append("edited")
        learned[r.plain] = new
    ir.save(path)
    if req.save_tm and learned:
        get_tm().put_many(learned)
    outputs = await run_in_threadpool(
        render_outputs, ir, job.input_path, job.dir, job.options.make_bilingual, None, job.options.highlight_doubtful
    )
    job.outputs = {k: str(v) for k, v in outputs.items()}
    job.summary = quality_summary(path)
    return {"job": job.public(), "saved": len(learned)}


class TextRequest(BaseModel):
    text: str
    engine: str | None = None
    use_glossary: bool = True
    custom_glossary: str = ""


@app.post("/api/translate-text")
def translate_text(req: TextRequest) -> dict:
    text = req.text.strip()
    if not text:
        return {"translation": ""}
    if len(text) > 20_000:
        raise HTTPException(413, "Văn bản quá dài (tối đa 20.000 ký tự).")
    try:
        engine = get_engine(req.engine or default_engine_key())
        glossary = build_glossary(
            TranslateOptions(use_glossary=req.use_glossary, custom_glossary=req.custom_glossary)
        )
        tr = TextTranslator(engine, glossary)
        # Giữ cấu trúc đoạn: mỗi đoạn (cách nhau dòng trống) dịch riêng.
        paras = text.replace("\r\n", "\n").split("\n\n")
        joined = [join_lines(p.split("\n")) for p in paras]
        out = tr.translate_texts(joined)
    except TranslatorError as e:
        raise HTTPException(503, str(e)) from e
    return {"translation": "\n\n".join(out)}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
