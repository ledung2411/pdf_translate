import pymupdf

from app.ir import Document
from app.pdf_processor import IR_FILENAME, TranslateOptions, analyze_pdf, render_outputs, translate_pdf

from .make_sample_pdf import make


def test_ir_roundtrip_and_rerender(tmp_path):
    src = tmp_path / "paper.pdf"
    make(str(src))
    out = tmp_path / "out"
    translate_pdf(src, out, TranslateOptions(engine="mock", make_bilingual=False))

    ir = Document.load(out / IR_FILENAME)
    assert ir.engine == "mock" and ir.page_numbers == [1]
    texts = [r for r in ir.regions if r.kind == "text"]
    assert texts and all(r.id.startswith("p1-r") for r in ir.regions)
    assert sum(r.translation is not None for r in texts) >= len(texts) - 2

    # Người dùng sửa 1 đoạn → dựng lại PDF từ IR, không dịch lại.
    target = next(r for r in texts if r.plain.startswith("We presented"))
    target.translation = "Đoạn kết luận do người dùng sửa."
    target.flags.append("edited")
    ir.save(out / IR_FILENAME)
    again = Document.load(out / IR_FILENAME)
    outputs = render_outputs(again, src, tmp_path / "re", make_bilingual_pdf=False)
    assert "Đoạn kết luận do người dùng sửa." in pymupdf.open(str(outputs["translated"]))[0].get_text()


def test_analyze_page_range(tmp_path):
    doc = pymupdf.open()
    for i in range(3):
        doc.new_page().insert_text((72, 72), f"This is the text of page number {i + 1} in the document. " * 3)
    p = tmp_path / "three.pdf"
    doc.save(str(p))
    ir = analyze_pdf(p, TranslateOptions(page_from=2, page_to=3))
    assert ir.page_numbers == [2, 3]
    assert "page number 2" in ir.pages[0].regions[0].plain
