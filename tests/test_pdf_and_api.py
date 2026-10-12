import time

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.pdf_processor import PdfError, TranslateOptions, extract_blocks, mark_references, translate_pdf

from .make_sample_pdf import make


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "sample.pdf"
    make(str(path))
    return path


def test_block_classification(sample):
    doc = pymupdf.open(str(sample))
    blocks = extract_blocks(doc[0], 0)
    mark_references(blocks)
    kinds = {b.plain[:20]: b.kind for b in blocks}
    assert kinds["1 Introduction"] == "text"
    assert kinds["Attention(Q, K, V) ="] == "math"
    assert kinds["Full 91.2 12.5"] == "table"
    assert kinds["[1] A. Vaswani, N. S"] == "reference"
    assert kinds["Figure 1: Overview o"] == "text"  # chú thích hình không bị coi là tài liệu tham khảo
    assert kinds["arXiv:2401.12345v1 ["] == "rotated"


def test_translate_pdf_outputs(sample, tmp_path):
    out = translate_pdf(sample, tmp_path, TranslateOptions(engine="mock", highlight_doubtful=False))
    assert set(out) == {"translated", "bilingual", "markdown"}
    vi = pymupdf.open(str(out["translated"]))
    text = vi[0].get_text()
    assert "Giới thiệu" in text and "Tài liệu tham khảo" in text
    assert "ạ ế ữ ở đ" in text  # font hiển thị được tiếng Việt
    assert "Attention(Q, K, V)" in text  # công thức khối giữ nguyên
    assert "Attention is all you need" in text  # tài liệu tham khảo giữ nguyên
    assert "[VI] Học sâu has transformed" in text  # glossary: deep learning → học sâu
    bi = pymupdf.open(str(out["bilingual"]))
    assert bi[0].rect.width == pytest.approx(vi[0].rect.width * 2)
    # Hình vẽ vector (khung hình minh họa, kẻ bảng) được giữ nguyên.
    assert len(vi[0].get_drawings()) == len(pymupdf.open(str(sample))[0].get_drawings())


def test_scanned_pdf_rejected(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(50, 50, 300, 300), fill=(0, 0, 0))
    p = tmp_path / "scan.pdf"
    doc.save(str(p))
    with pytest.raises(PdfError, match="OCR"):
        translate_pdf(p, tmp_path / "o", TranslateOptions(engine="mock"))


def test_api_job_flow(sample):
    with TestClient(app) as client:
        engines = client.get("/api/engines").json()
        assert any(e["key"] == "mock" for e in engines["engines"])

        r = client.post(
            "/api/jobs",
            files={"file": ("paper.pdf", sample.read_bytes(), "application/pdf")},
            data={"engine": "mock", "custom_glossary": "sparse attention = chú ý thưa"},
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["id"]
        for _ in range(100):
            st = client.get(f"/api/jobs/{job_id}").json()
            if st["status"] in ("done", "error"):
                break
            time.sleep(0.1)
        assert st["status"] == "done", st
        pdf = client.get(f"/api/jobs/{job_id}/download/translated")
        assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
        md = client.get(f"/api/jobs/{job_id}/download/markdown").text
        assert "chú ý thưa" in md

        assert client.post("/api/jobs", files={"file": ("x.pdf", b"hello", "application/pdf")}).status_code == 400
        assert client.get("/").status_code == 200
        t = client.post("/api/translate-text", json={"text": "Abstract", "engine": "mock"}).json()
        assert t["translation"] == "Tóm tắt"


def test_wiley_text_font_is_not_math():
    from app.scientific import has_prose, is_math_font

    assert not is_math_font("STIXTwoText")
    assert not is_math_font("STIXTwoText-Italic")
    assert is_math_font("STIXTwoMath-Regular")
    assert is_math_font("CMMI10")
    assert has_prose("Perovskite solar cells have attracted considerable attention")
    assert not has_prose("Attention(Q, K, V) = softmax(QKT / √d) V (1)")


def test_merged_two_column_block_is_split():
    from app.pdf_processor import _segment_blocks

    def line(x0, x1, y, text):
        sp = {"bbox": (x0, y, x1, y + 10), "text": text, "size": 9, "font": "STIXTwoText", "flags": 4, "color": 0}
        return {"bbox": sp["bbox"], "spans": [sp], "dir": (1, 0)}

    lines = []
    for i in range(6):
        lines.append(line(50, 290, 100 + 11 * i, f"left {i}"))
        lines.append(line(304, 544, 100 + 11 * i, f"right {i}"))
    out = _segment_blocks([{"type": 0, "bbox": (50, 100, 544, 170), "lines": lines}], pymupdf.Rect(0, 0, 595, 842))
    assert len(out) == 2
    assert [ln["spans"][0]["text"] for ln in out[0]["lines"]] == [f"left {i}" for i in range(6)]
    assert [ln["spans"][0]["text"] for ln in out[1]["lines"]] == [f"right {i}" for i in range(6)]


def test_pdf_without_paragraphs_reports_error(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(30):
        page.insert_text((50, 50 + 20 * i), "12.5   33.1   0.98   41.0", fontsize=9)
    p = tmp_path / "numbers.pdf"
    doc.save(str(p))
    with pytest.raises(PdfError, match="Không tìm thấy đoạn văn"):
        translate_pdf(p, tmp_path / "o", TranslateOptions(engine="mock"))


def test_doubtful_regions_highlighted(sample, tmp_path):
    from app.ir import Document
    from app.pdf_processor import IR_FILENAME

    # Engine giả lập trả về câu còn tiếng Anh → bị chấm điểm thấp → tô vàng
    out = translate_pdf(sample, tmp_path, TranslateOptions(engine="mock"))
    ir = Document.load(tmp_path / IR_FILENAME)
    low = [r for r in ir.regions if "low_quality" in r.flags]
    assert low and all(r.score is not None and r.score < 0.6 and r.notes for r in low)
    vi = pymupdf.open(str(out["translated"]))
    annots = [(a.info["title"], a.info["content"]) for a in vi[0].annots()]
    assert len(annots) == len(low)
    assert annots[0][0] == "Cần kiểm tra lại" and "điểm" in annots[0][1]
    bi = pymupdf.open(str(out["bilingual"]))
    half = pymupdf.open(str(sample))[0].rect.width
    bi_rects = [a.rect for a in bi[0].annots()]
    assert len(bi_rects) == len(low) and all(r.x0 >= half for r in bi_rects)  # nằm ở nửa bản dịch
