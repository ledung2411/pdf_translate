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
    out = translate_pdf(sample, tmp_path, TranslateOptions(engine="mock"))
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
