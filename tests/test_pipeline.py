import re

from app.pipeline import MATH_END, MATH_START, SUP_END, SUP_START, TextTranslator
from app.scientific import Glossary, load_builtin_glossary
from app.translators import Translator


class EchoTranslator(Translator):
    """Giả lập LLM: giữ nguyên placeholder, đánh dấu phần đã dịch."""

    key = "echo"
    supports_placeholders = True

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


# ---------------- Mô hình dịch máy (OPUS / VinAI): không dùng placeholder ----------------


class NMTEcho(Translator):
    key = "nmt"

    def __init__(self, mapping=None):
        self.calls: list[str] = []
        self.mapping = mapping or {}

    def translate_batch(self, texts, glossary=None):
        self.calls.extend(texts)
        return [self.mapping.get(t, f"VI<{t}>") for t in texts]


def test_nmt_never_sees_placeholders_or_markers():
    eng = NMTEcho()
    tr = TextTranslator(eng, Glossary(load_builtin_glossary()), use_cache=False)
    src = (
        f"We train a deep learning model on {MATH_START}x ∈ ℝ{MATH_END} with area 5 cm{SUP_START}2{SUP_END} "
        "(see Fig. 3) [20]. Contact fan.zhang@uestc.edu.cn."
    )
    (out,) = tr.translate_texts([src])
    assert eng.calls and all("{" not in c and "\ue000" not in c and "\ue002" not in c for c in eng.calls)
    assert "x ∈ ℝ" in eng.calls[0] and "cm²" in eng.calls[0]
    # Thuật ngữ & nhãn được Việt hóa sau khi dịch
    assert "học sâu" in out and "Hình 3" in out


def test_upper_and_title_case_normalized_for_nmt():
    eng = NMTEcho({"Research article": "Bài báo nghiên cứu"})
    tr = TextTranslator(eng, None, use_cache=False)
    out = tr.translate_texts([
        "RESEARCH ARTICLE",
        "DDSurfer: A Weakly-Supervised Framework for Cortical Surface Reconstruction From Diffusion MRI",
        "1 University of Electronic Science and Technology of China, Chengdu, China 3 School of Medicine, Tel Aviv, Israel",
        "ABSTRACT Cortical surface reconstruction is critical.",
    ])
    assert out[0] == "BÀI BÁO NGHIÊN CỨU"
    assert eng.calls[1] == "DDSurfer: a weakly-supervised framework for cortical surface reconstruction from diffusion MRI"
    assert "China, Chengdu" in eng.calls[2]  # tên riêng không bị viết thường
    assert out[3].startswith("TÓM TẮT VI<Cortical")
    assert tr.translate_texts(["Jane Doe, John Smith, Alan Turing"]) == ["VI<Jane Doe, John Smith, Alan Turing>"]


def test_degenerate_output_is_repaired_or_kept_english():
    bad = "Hệ Lục Hệ Hệ Hệ Hệ Hệ Hệ Hệ Hệ"
    eng = NMTEcho({
        "Advanced science journal is great, and it is open": bad,
        "Advanced science journal is great,": "Tạp chí rất tốt,",
        "and it is open": "chúng ta tính chúng ta tính chúng ta tính chúng ta tính",
    })
    tr = TextTranslator(eng, None, use_cache=False)
    (out,) = tr.translate_texts(["Advanced science journal is great, and it is open"])
    assert out == "Tạp chí rất tốt, and it is open"


def test_is_degenerate():
    from app.pipeline import is_degenerate

    assert is_degenerate("Advanced Science", "Hệ Lục Hệ Hệ Hệ Hệ Hệ Hệ")
    assert is_degenerate("RESEARCH ARTICLE", "thay_ hạ_ _ _ bitS_ ____")
    assert is_degenerate("We first compute x.", "Chúng ta đầu tiên tiên chúng ta tính chúng ta tính chúng ta tính chúng ta tính")
    assert is_degenerate("Data model.", "model model model model deep")
    assert not is_degenerate("A 1 1 1 matrix", "ma trận 1 1 1")
    assert not is_degenerate(
        "We propose a novel framework for cortical surface reconstruction.",
        "Chúng tôi đề xuất một khung mới để tái tạo bề mặt vỏ não.",
    )


class TermNMT(NMTEcho):
    """Giả lập mô hình dịch máy dịch sai thuật ngữ ghép nhưng chép nguyên mã viết tắt."""

    key = "termnmt"

    def __init__(self, drop_tokens=False):
        super().__init__()
        self.drop_tokens = drop_tokens

    def translate_batch(self, texts, glossary=None):
        self.calls.extend(texts)
        out = []
        for t in texts:
            t = re.sub(r"(?i)loss function", "chức năng mất", t)
            if self.drop_tokens:
                t = re.sub(r"ZQX[A-Z]", "", t)
            out.append(f"VI<{t}>")
        return out


def test_nmt_terms_off_by_default_keeps_wrong_term():
    tr = TextTranslator(TermNMT(), Glossary(load_builtin_glossary()), use_cache=False)
    (out,) = tr.translate_texts(["We minimize the loss function."])
    assert "chức năng mất" in out


def test_nmt_terms_token_mode(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "nmt_terms", "token")
    eng = TermNMT()
    tr = TextTranslator(eng, Glossary(load_builtin_glossary()), use_cache=False)
    (out,) = tr.translate_texts(["2.2.4 Loss Function Design", "We minimize the loss function of the matrix."])[:1]
    assert "Hàm mất mát" in out or "hàm mất mát" in out
    assert all("loss function" not in c.lower() for c in eng.calls)
    assert any("ZQXA" in c for c in eng.calls)
    # Thuật ngữ một từ của bảng mặc định ("matrix") không bị thay
    assert any("matrix" in c for c in eng.calls)


def test_nmt_terms_token_lost_falls_back(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "nmt_terms", "token")
    eng = TermNMT(drop_tokens=True)
    tr = TextTranslator(eng, Glossary(load_builtin_glossary()), use_cache=False)
    (out,) = tr.translate_texts(["We minimize the loss function."])
    assert "ZQX" not in out and "chức năng mất" in out  # dịch lại câu gốc, không mất nội dung


def test_nmt_terms_codeswitch_mode(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "nmt_terms", "codeswitch")
    eng = TermNMT()
    tr = TextTranslator(eng, Glossary({"loss function": "hàm mất mát", "attention head": "đầu chú ý"}), use_cache=False)
    (out,) = tr.translate_texts(["Each attention head minimizes the loss function."])
    assert "hàm mất mát" in out and "đầu chú ý" in out
