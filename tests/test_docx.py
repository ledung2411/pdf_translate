import pytest

from app.pdf_processor import TranslateOptions, translate_pdf

from .make_sample_pdf import make

docx = pytest.importorskip("docx")


def test_docx_export(tmp_path):
    src = tmp_path / "paper.pdf"
    make(str(src))
    out = translate_pdf(src, tmp_path / "o", TranslateOptions(engine="mock", make_bilingual=False))
    assert "docx" in out
    d = docx.Document(str(out["docx"]))
    paras = [(p.style.name, p.text) for p in d.paragraphs if p.text.strip()]
    texts = [t for _, t in paras]
    heads = [t for style, t in paras if style.startswith("Heading") or style == "Title"]
    assert any("Giới thiệu" in h for h in heads) and any("Thực nghiệm" in h for h in heads)
    # Thứ tự đọc: hết cột trái (Giới thiệu, Các công trình liên quan) rồi mới tới cột phải (Thực nghiệm)
    idx = lambda s: next(i for i, t in enumerate(texts) if s in t)  # noqa: E731
    assert idx("Giới thiệu") < idx("Các công trình liên quan") < idx("Thực nghiệm") < idx("Kết luận")
    # Công thức khối, bảng... là ảnh cắt từ PDF gốc
    assert len(d.inline_shapes) >= 2
    # Tài liệu tham khảo giữ nguyên tiếng Anh
    assert any(t.startswith("[1] A. Vaswani") for t in texts)
    # Đoạn nghi ngờ được tô vàng
    assert any(r.font.highlight_color is not None for p in d.paragraphs for r in p.runs)
