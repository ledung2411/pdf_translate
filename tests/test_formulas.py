import pymupdf

from app.pdf_processor import _split_math_lines, _to_html, extract_blocks

from .make_sample_pdf import make


def _line(y, text, font="CMR10"):
    sp = {"bbox": (72, y, 300, y + 10), "text": text, "size": 10, "font": font, "flags": 4, "color": 0}
    return {"bbox": sp["bbox"], "spans": [sp], "dir": (1, 0)}


def test_display_equation_split_from_paragraph():
    block = {"type": 0, "bbox": (72, 100, 300, 160), "lines": [
        _line(100, "We minimize the loss defined as follows, which combines"),
        _line(112, "three terms weighted by hyperparameters:"),
        _line(126, "L = Lc + λ1 Ln + λ2 Le (1)"),
        _line(140, "where Lc is the Chamfer distance between the surfaces."),
    ]}
    parts = _split_math_lines(block)
    texts = [" ".join(sp["text"] for ln in p["lines"] for sp in ln["spans"]) for p in parts]
    assert len(parts) == 3 and texts[1] == "L = Lc + λ1 Ln + λ2 Le (1)"


def test_math_font_line_split():
    block = {"type": 0, "bbox": (72, 100, 300, 140), "lines": [
        _line(100, "The attention score is computed as shown below."),
        _line(114, "softmax(QK / d) V", font="CMMI10"),
    ]}
    assert len(_split_math_lines(block)) == 2


def test_inline_math_rendered_italic():
    html = _to_html("Cho x ∈ ℝd là đầu vào, với ma trận W và hệ số α.", ["x ∈ ℝd", "W", "α", "="])
    assert '<span class="m">x ∈ ℝd</span>' in html
    assert '<span class="m">W</span>' in html and '<span class="m">α</span>' in html
    assert "Cho" in html and "<span class=\"m\">=" not in html
    # Chữ cái đơn chỉ khớp khi đứng riêng, không khớp trong từ
    assert _to_html("Với x = 2", ["x"]).count('class="m"') == 1
    assert _to_html("xin chào", ["x"]).count('class="m"') == 0


def test_inline_math_collected(tmp_path):
    p = tmp_path / "s.pdf"
    make(str(p))
    blocks = extract_blocks(pymupdf.open(str(p))[0], 0)
    let = next(b for b in blocks if b.plain.startswith("Let"))
    assert any("ℝ" in m for m in let.math_spans) and "Q" in let.math_spans
