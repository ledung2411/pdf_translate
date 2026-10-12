import pymupdf
import pytest

from app.ocr import find_tessdata, is_scanned
from app.pdf_processor import PdfError, TranslateOptions, analyze_pdf, translate_pdf

from .make_sample_pdf import make


def _tessdata_or_skip():
    try:
        return find_tessdata(download=False)
    except Exception:  # noqa: BLE001
        pytest.skip("chưa có dữ liệu Tesseract")


@pytest.fixture(scope="module")
def scan_pdf(tmp_path_factory):
    d = tmp_path_factory.mktemp("scan")
    src = d / "paper.pdf"
    make(str(src))
    pix = pymupdf.open(str(src))[0].get_pixmap(dpi=150)
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, pixmap=pix)
    path = d / "scan.pdf"
    doc.save(str(path))
    return path


def test_is_scanned(scan_pdf, tmp_path):
    assert is_scanned(pymupdf.open(str(scan_pdf))[0])
    born_digital = tmp_path / "d.pdf"
    make(str(born_digital))
    assert not is_scanned(pymupdf.open(str(born_digital))[0])


def test_ocr_analyze_and_render(scan_pdf, tmp_path):
    _tessdata_or_skip()
    ir = analyze_pdf(scan_pdf, TranslateOptions())
    assert ir.pages[0].scanned
    texts = [r for r in ir.regions if r.kind == "text"]
    assert any("Efficient Attention" in r.plain for r in texts)
    assert any(r.plain.startswith("Abstract") for r in texts)
    assert not any("arXiv" in r.plain for r in texts)  # watermark xoay không bị dịch

    out = translate_pdf(scan_pdf, tmp_path / "o", TranslateOptions(engine="mock", make_bilingual=False))
    page = pymupdf.open(str(out["translated"]))[0]
    assert "[VI] Efficient" in page.get_text()
    # Vùng tiêu đề trong ảnh đã được tô trắng (chữ tiếng Anh trong ảnh bị xóa)
    title = next(r for r in texts if "Efficient Attention" in r.plain)
    clip = pymupdf.Rect(title.bbox)
    before = pymupdf.open(str(scan_pdf))[0].get_pixmap(clip=clip, dpi=50)
    assert before.samples != page.get_pixmap(clip=clip, dpi=50).samples


def test_scanned_without_ocr_reports(scan_pdf, tmp_path):
    with pytest.raises(PdfError, match="OCR"):
        analyze_pdf(scan_pdf, TranslateOptions(ocr=False))
