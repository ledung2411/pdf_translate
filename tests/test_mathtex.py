"""Dựng LaTeX từ vị trí / cỡ chữ / font của từng ký tự (toạ độ kiểu pdfTeX và STIX)."""

import re

import pymupdf

from app.mathtex import find_math, formula_latex, inline_latex


def span(text, x0, oy, size=10.0, font="CMMI10", width=None):
    """Span kiểu ``rawdict``: mỗi ký tự rộng 0,5 cỡ chữ, đặt liền nhau từ ``x0``."""
    w = width or 0.5 * size
    chars = [{"c": c, "origin": (x0 + i * w, oy), "bbox": (x0 + i * w, oy - 0.75 * size, x0 + (i + 1) * w, oy + 0.25 * size)}
             for i, c in enumerate(text)]
    return {"text": text, "font": font, "size": size, "flags": 0, "chars": chars,
            "origin": (x0, oy), "bbox": (x0, oy - 0.75 * size, x0 + len(text) * w, oy + 0.25 * size)}


def test_scripts_fonts_and_nesting_cm():
    # W_i^Q ∈ ℝ^{d_model × d_k}  (pdfTeX: CMMI10/CMSY10/MSBM10/CMR7)
    spans = [
        span("W", 50, 28.5), span("Q", 60.5, 23.7, 7), span("i", 59, 32.2, 7),
        span("∈", 68.6, 28.5, font="CMSY10"), span("R", 78, 28.5, font="MSBM10"),
        span("d", 86, 24.4, 7), span("model", 89.5, 25.5, 4.9, font="CMR5"),
        span("×", 102, 24.4, 7, font="CMSY7"), span("d", 107, 24.4, 7), span("k", 110.5, 25.5, 4.9),
    ]
    tex, html = inline_latex(spans, 10.0, 28.5)
    assert tex == r"W_{i}^{Q} \in \mathbb{R}^{d_{\mathrm{model}} \times d_{k}}"
    assert '<span class="sup">' in html and '<span class="sub">' in html


def test_unicode_math_italic_and_prime_stix():
    # x′_β = x_β ⊙ σ(f(x_α))  (STIX Two Math: chữ nghiêng toán học Unicode)
    f = "STIXTwoMath"
    spans = [span("𝑥", 10, 100, font=f), span("′", 15.2, 96, 7, font=f), span("𝛽", 15.2, 101.5, 7, font=f),
             span("=", 22, 100, font=f), span("𝑥", 30, 100, font=f), span("𝛽", 35, 101.5, 7, font=f),
             span("⊙", 41, 100, font=f), span("𝜎(𝑓(𝑥", 48, 100, font=f), span("𝛼", 73, 101.5, 7, font=f),
             span("))", 77, 100, font=f)]
    tex, _ = inline_latex(spans, 10.0, 100)
    assert re.sub(r"\s", "", tex) in (r"x'_{\beta}=x_{\beta}\odot\sigma(f(x_{\alpha}))",
                                      r"x_{\beta}'=x_{\beta}\odot\sigma(f(x_{\alpha}))")


def test_fraction_sqrt_and_tag():
    # Attention(Q,K,V) = softmax( QK^T / √d_k ) V   (1)
    rm, it = "CMR10", "CMMI10"
    spans = [span("Attention(", 10, 71, font=rm), span("Q,K,V", 55, 71, font=it), span(")=softmax(", 80, 71, font=rm),
             span("QK", 142, 66.2, 7, font=it), span("T", 149, 63.5, 4.9, font=it),
             span("√", 142, 75, 7, font="CMEX10", width=7), span("d", 149.5, 77, 7, font=it),
             span("k", 153, 78, 4.9, font=it),
             span(")V", 160, 71, font=rm), span("(1)", 260, 71, font=rm)]
    rules = [(142, 68.5, 158, 69.0), (149, 70.0, 157.5, 70.4)]  # gạch phân số, nét trên dấu căn
    tex = formula_latex(spans, rules)
    assert r"\frac{QK^{T}}{\sqrt{d_{k}}}" in tex
    assert tex.startswith(r"\mathrm{Attention}(Q,K,V) = \mathrm{softmax}(")
    assert tex.endswith(r"\tag{1}")


def test_limits_under_min_and_sum():
    it, rm = "CMMI10", "CMR10"
    spans = [span("∑", 10, 100, font="CMEX10"), span("i", 11, 108, 7, font=it), span("n", 11, 90, 7, font=it),
             span("min", 24, 100, font=rm), span("x", 26, 108, 7, font=it),
             span("f", 42, 100, font=it), span("(", 47, 100, font=rm), span("x", 52, 100, font=it),
             span(")", 57, 100, font=rm)]
    tex = formula_latex(spans, [])
    assert tex.replace(" ", "") == r"\sum_{i}^{n}\min_{x}f(x)"


def test_combining_hat_and_unknown_glyph():
    f = "STIXTwoMath"
    spans = [span("𝑦", 10, 100, font=f), span("̂", 16, 100, font=f, width=0.001), span("𝑃", 16, 100, font=f)]
    assert inline_latex(spans, 10.0, 100)[0] == r"y\hat{P}"
    # Ký tự riêng của font (vùng Private Use) → không đoán, giữ ảnh công thức
    assert formula_latex([span("", 10, 100, font="STIXMathCalligraphy")], []) is None


def test_find_math_word_boundaries():
    keys = ["x", "x ∈ ℝd", "W"]
    marks = find_math("Cho x ∈ ℝd và xin chào W.", keys)
    assert [m[2] for m in marks] == ["x ∈ ℝd", "W"]


def test_sample_pdf_exports_latex(tmp_path):
    from app.pdf_processor import TranslateOptions, translate_pdf

    from .make_sample_pdf import make

    pdf = tmp_path / "sample.pdf"
    make(str(pdf))
    out = translate_pdf(pdf, tmp_path / "out", TranslateOptions(engine="mock", make_bilingual=False))
    md = out["markdown"].read_text(encoding="utf-8")
    assert r"\tag{1}" in md and "$$" in md  # công thức đứng riêng
    assert r"\mathbb{R}^{" in md  # công thức nội dòng x ∈ ℝ^d
    from docx import Document as Docx

    text = "\n".join(p.text for p in Docx(str(out["docx"])).paragraphs)
    assert r"\frac" not in text or "$$" in text
    assert r"$$" in text and r"\mathbb{R}" in text
    # PDF dịch: chỉ số trên của ℝ^d được vẽ nhỏ và nâng lên
    page = pymupdf.open(str(out["translated"]))[0]
    sizes = {round(s["size"], 1) for b in page.get_text("dict")["blocks"] for l in b.get("lines", [])
             for s in l["spans"] if s["text"].strip() == "d"}
    assert sizes, "không thấy chỉ số d trong PDF dịch"
