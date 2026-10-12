from app.config import settings
from app.pipeline import TextTranslator
from app.quality import chrf, combine, rule_score
from app.scientific import Glossary, load_builtin_glossary
from app.translators import Translator

SRC = "The optimized device achieves an efficiency of 24.1% and retains 95% after 1000 h [12]."


def test_rule_score():
    g = Glossary(load_builtin_glossary())
    good = rule_score(SRC, "Thiết bị tối ưu đạt hiệu suất 24,1% và giữ 95% sau 1.000 giờ [12].", g)
    assert good.score == 1.0 and not good.reasons
    lost = rule_score(SRC, "Thiết bị tối ưu đạt hiệu suất cao.", g)
    assert lost.score < 0.3 and any("số liệu" in r for r in lost.reasons)
    untranslated = rule_score(SRC, SRC, g)
    assert untranslated.score < 0.6 and "còn sót tiếng Anh" in untranslated.reasons
    assert rule_score("Advanced Science", "Hệ Hệ Hệ Hệ").score == 0
    term = rule_score("We train a neural network.", "Chúng tôi huấn luyện một mạng lưới thần kinh.", g)
    assert 0.6 <= term.score < 1 and any("thuật ngữ" in r for r in term.reasons)


def test_chrf_and_combine():
    assert chrf("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert chrf("abc", "xyz") == 0.0
    assert 0.5 < chrf("the cat sat on the mat", "the cat sat on a mat") < 1
    q = combine(rule_score("a b c", "x y z"), 0.1)
    assert q.score < 1 and any("dịch ngược" in r for r in q.reasons)


class NMT(Translator):
    key = "nmt"

    def __init__(self, mapping):
        self.mapping = mapping

    def translate_batch(self, texts, glossary=None):
        return [self.mapping.get(t, "Bản dịch tốt của câu này.") for t in texts]


class FakeLLM(Translator):
    key = "llm"

    def __init__(self, fix):
        self.fix, self.calls = fix, []

    def translate_batch(self, texts, glossary=None):
        raise AssertionError("không dùng để dịch")

    def post_edit(self, source, draft, glossary=None):
        self.calls.append((source, draft, glossary))
        return self.fix.get(source, draft)


def test_selective_post_edit(monkeypatch):
    monkeypatch.setattr(settings, "postedit_max_ratio", 1.0)
    bad_src = "We train a neural network for 10 epochs."
    nmt = NMT({bad_src: "Chúng tôi huấn luyện một mạng lưới thần kinh."})  # mất số + sai thuật ngữ
    llm = FakeLLM({bad_src: "Chúng tôi huấn luyện một mạng nơ-ron trong 10 epoch."})
    g = Glossary(load_builtin_glossary())
    tr = TextTranslator(nmt, g, use_cache=False, post_editor=llm)
    out = tr.translate_texts([bad_src, "The weather is nice today."])
    assert [c[0] for c in llm.calls] == [bad_src]  # chỉ câu điểm thấp được gửi cho LLM
    assert llm.calls[0][2] == {"neural network": "mạng nơ-ron"}  # kèm thuật ngữ bắt buộc
    assert out[0] == "Chúng tôi huấn luyện một mạng nơ-ron trong 10 epoch."
    assert tr.post_edited == [True, False]
    assert tr.scores[0] == 1.0 and tr.scores[1] == 1.0


def test_post_edit_rejected_if_worse(monkeypatch):
    monkeypatch.setattr(settings, "postedit_max_ratio", 1.0)
    src = "It lasts 10 epochs."
    nmt = NMT({src: "Kéo dài một thời gian."})  # mất số → điểm thấp
    llm = FakeLLM({src: "Hệ Hệ Hệ Hệ"})  # LLM còn tệ hơn
    tr = TextTranslator(nmt, None, use_cache=False, post_editor=llm)
    assert tr.translate_texts([src]) == ["Kéo dài một thời gian."]
    assert tr.post_edited == [False] and tr.scores[0] < 0.7 and tr.reasons[0]
