"""Web server FastAPI: giao diện tải PDF lên, theo dõi tiến độ và tải bản dịch."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .config import settings
from .jobs import manager
from .pdf_processor import TranslateOptions, build_glossary
from .pipeline import TextTranslator
from .scientific import join_lines, load_builtin_glossary
from .translators import TranslatorError, default_engine_key, get_engine, list_engines

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC_DIR = Path(__file__).parent / "static"
DOWNLOAD_NAMES = {
    "translated": ("application/pdf", "_vi.pdf"),
    "bilingual": ("application/pdf", "_song_ngu.pdf"),
    "markdown": ("text/markdown; charset=utf-8", "_vi.md"),
}



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
    )
    job = manager.submit(file.filename or "document.pdf", data, opts)
    return job.public()


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
