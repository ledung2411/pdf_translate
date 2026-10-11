from app.scientific import (
    BlockStats,
    Glossary,
    Protector,
    classify_block,
    join_lines,
    load_builtin_glossary,
    parse_glossary,
    split_sentences,
    translate_heading,
)


def protect(text: str, glossary: Glossary | None = None) -> tuple[str, Protector]:
    p = Protector()
    out = p.protect_patterns(text)
    if glossary is not None:
        out = glossary.protect(out, p)
    return p.merge_adjacent(out), p


def test_protect_restore_roundtrip():
    text = (
        "Let α ∈ ℝ^d and x = Wx + b (p < 0.05) [3, 7]. See https://example.com/a.b, "
        "doi:10.1000/xyz123 and getValue() with batch_size."
    )
    out, p = protect(text)
    for frag in ("α", "ℝ", "https://", "doi:", "getValue()", "batch_size", "[3, 7]", "p < 0.05", "x = Wx + b"):
        assert frag not in out, frag
    assert p.restore(out) == text


def test_url_trailing_period_kept_outside():
    out, p = protect("Code is at https://github.com/x/y.")
    assert out.endswith("}.")
    assert p.restore(out) == "Code is at https://github.com/x/y."


def test_existing_braces_not_confused_with_placeholders():
    out, p = protect("The set {0} is empty and {1, 2} has two elements.")
    assert p.restore(out) == "The set {0} is empty and {1, 2} has two elements."


def test_glossary_terms_and_labels():
    g = Glossary(load_builtin_glossary())
    out, p = protect("As shown in Fig. 3 and Table 2), the loss function of neural networks converges.", g)
    restored = p.restore(out)
    assert "Hình 3" in restored
    assert "Bảng 2)" in restored
    assert "hàm mất mát" in restored
    assert "mạng nơ-ron" in restored


def test_glossary_label_only_terms_not_translated_without_number():
    g = Glossary(load_builtin_glossary())
    out, p = protect("The table lists figures of merit.", g)
    assert p.restore(out) == "The table lists figures of merit."


def test_glossary_capitalization_and_custom():
    g = Glossary(parse_glossary("# comment\nknowledge distillation = chưng cất tri thức\nPyTorch = PyTorch"))
    out, p = protect("Knowledge distillation in PyTorch.", g)
    assert p.restore(out) == "Chưng cất tri thức in PyTorch."


def test_split_sentences_respects_abbreviations():
    s = split_sentences(
        "We follow Smith et al. in this work. Results (see Fig. 2) are good. "
        "J. Doe proposed it, e.g. in 2020. Accuracy was 91.2%. Finally!"
    )
    assert s == [
        "We follow Smith et al. in this work.",
        "Results (see Fig. 2) are good.",
        "J. Doe proposed it, e.g. in 2020.",
        "Accuracy was 91.2%.",
        "Finally!",
    ]


def test_split_long_sentence():
    long = "; ".join(["this is a clause with several words"] * 40) + "."
    parts = split_sentences(long, max_len=200)
    assert all(len(p) <= 200 for p in parts)
    assert len(parts) > 1


def test_headings():
    assert translate_heading("1 Introduction") == "1 Giới thiệu"
    assert translate_heading("2.1. Related Work") == "2.1. Các công trình liên quan"
    assert translate_heading("IV. CONCLUSIONS") == "IV. KẾT LUẬN"
    assert translate_heading("References") == "Tài liệu tham khảo"
    assert translate_heading("2 | Methods") == "2 | Phương pháp"
    assert translate_heading("We introduce a method") is None


def test_join_lines_dehyphenates():
    assert join_lines(["The trans-", "former model", "works."]) == "The transformer model works."
    assert join_lines(["State-of-the-", "Art"]) == "State-of-the-Art"
    assert join_lines(["a dMRI-", "specific pipeline"]) == "a dMRI-specific pipeline"


def test_classify_block():
    assert classify_block(BlockStats("This is a normal paragraph of text.", 35, 0, 0)) == "text"
    assert classify_block(BlockStats("91.2 12.5 0.33", 14, 0, 0)) == "data"
    assert classify_block(BlockStats("x y", 30, 25, 0)) == "math"
    assert classify_block(BlockStats("for i in range(10): print(i)", 28, 0, 28)) == "code"
