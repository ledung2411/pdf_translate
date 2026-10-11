import re

from app.pipeline import MATH_END, MATH_START, SUP_END, SUP_START, TextTranslator
from app.scientific import Glossary, load_builtin_glossary
from app.translators import Translator


class EchoTranslator(Translator):
    """Giả lập mô hình: giữ nguyên placeholder, đánh dấu phần đã dịch."""

    key = "echo"

    def __init__(self):
        self.calls: list[str] = []

    def translate_batch(self, texts, glossary=None):
        self.calls.extend(texts)
        return [f"VI<{t}>" for t in texts]


class DroppingTranslator(EchoTranslator):
    """Giả lập mô hình làm mất placeholder trong câu dài."""

    key = "drop"

    def translate_batch(self, texts, glossary=None):
        self.calls.extend(texts)
        return [re.sub(r"\{\d+\}", "", t) if len(t) > 20 else f"VI<{t}>" for t in texts]


def test_math_markers_and_glossary_restored():
    eng = EchoTranslator()
    tr = TextTranslator(eng, Glossary(load_builtin_glossary()), use_cache=False)
    src = f"Let {MATH_START}x ∈ ℝ{MATH_END} be the input of the neural network{SUP_START}1{SUP_END}."
    (out,) = tr.translate_texts([src])
    assert "x ∈ ℝ" in out
    assert "mạng nơ-ron" in out
    assert f"{SUP_START}1{SUP_END}" in out
    # Mô hình chỉ thấy placeholder, không thấy công thức
    assert all("ℝ" not in c for c in eng.calls)


def test_heading_shortcut_skips_engine():
    eng = EchoTranslator()
    tr = TextTranslator(eng, None, use_cache=False)
    assert tr.translate_texts(["3 Experiments"]) == ["3 Thực nghiệm"]
    assert eng.calls == []


def test_placeholder_loss_falls_back_to_piecewise():
    eng = DroppingTranslator()
    tr = TextTranslator(eng, None, use_cache=False)
    (out,) = tr.translate_texts(["The value p < 0.05 indicates a significant improvement overall."])
    assert "p < 0.05" in out
    assert "VI<" in out


def test_text_without_words_is_kept():
    eng = EchoTranslator()
    tr = TextTranslator(eng, None, use_cache=False)
    assert tr.translate_texts(["[12, 15]"]) == ["[12, 15]"]
    assert eng.calls == []


def test_duplicate_sentences_translated_once():
    eng = EchoTranslator()
    tr = TextTranslator(eng, None, use_cache=False)
    tr.translate_texts(["Preprint under review.", "Preprint under review."])
    assert eng.calls.count("Preprint under review.") == 1
