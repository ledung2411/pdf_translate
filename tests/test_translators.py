"""Kiểm tra lớp HFSeq2SeqTranslator với torch/transformers giả lập (không cần tải mô hình)."""

import contextlib
import sys
import types

import pytest

from app.translators import HFSeq2SeqTranslator, TranslationCache


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), 12)

    def to(self, device):
        return self


class FakeTokenizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, texts, **kw):
        return {"input_ids": FakeTensor(list(texts))}

    def convert_tokens_to_ids(self, tok):
        return {"vi_VN": 250024, "vie_Latn": 256184}[tok]

    def batch_decode(self, out, skip_special_tokens=True):
        return [f"vi:{t}" for t in out]


class FakeModel:
    last_kwargs: dict = {}

    def half(self):
        return self

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, input_ids, **kwargs):
        FakeModel.last_kwargs = kwargs
        return input_ids.rows


@pytest.fixture
def fake_hf(monkeypatch, tmp_path):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch.inference_mode = contextlib.nullcontext
    tf = types.ModuleType("transformers")
    created = {}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kw):
            created["tok"] = FakeTokenizer(name=name, **kw)
            return created["tok"]

    class AutoModelForSeq2SeqLM:
        @staticmethod
        def from_pretrained(name):
            return FakeModel()

    tf.AutoTokenizer, tf.AutoModelForSeq2SeqLM = AutoTokenizer, AutoModelForSeq2SeqLM
    tf.logging = types.SimpleNamespace(set_verbosity_error=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "sentencepiece", types.ModuleType("sentencepiece"))
    monkeypatch.setitem(sys.modules, "ctranslate2", None)  # backend transformers+torch
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "pytorch_model.bin").write_bytes(b"")
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(name, **kw):
        created.setdefault("snap", []).append(kw)
        return str(snap)

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    created["snap_dir"] = str(snap)
    return created


def test_mbart_uses_decoder_start_token_and_keeps_order(fake_hf):
    t = HFSeq2SeqTranslator("vinai", "", "", "vinai/x", src_lang="en_XX", tgt_lang="vi_VN", tgt_mode="decoder_start")
    texts = ["a much longer sentence here", "short", "medium length"]
    assert t.translate_batch(texts) == [f"vi:{x}" for x in texts]
    assert FakeModel.last_kwargs["decoder_start_token_id"] == 250024
    assert fake_hf["tok"].kwargs["src_lang"] == "en_XX"


def test_loads_from_local_cache_without_network(fake_hf):
    HFSeq2SeqTranslator("opus", "", "", "opus/x").prepare()
    assert fake_hf["snap"] == [{"local_files_only": True}]
    assert fake_hf["tok"].kwargs["name"] == fake_hf["snap_dir"]


def test_generation_guards_against_loops(fake_hf):
    HFSeq2SeqTranslator("opus", "", "", "opus/x").translate_batch(["hello"])
    assert FakeModel.last_kwargs["no_repeat_ngram_size"] == 4
    assert FakeModel.last_kwargs["max_length"] == 40  # 12 token đầu vào * 2.5 + 10


def test_nllb_uses_forced_bos(fake_hf):
    t = HFSeq2SeqTranslator("nllb", "", "", "nllb/x", src_lang="eng_Latn", tgt_lang="vie_Latn", tgt_mode="forced_bos")
    t.translate_batch(["hello"])
    assert FakeModel.last_kwargs["forced_bos_token_id"] == 256184
    assert "decoder_start_token_id" not in FakeModel.last_kwargs


def test_marian_has_no_lang_tokens(fake_hf):
    t = HFSeq2SeqTranslator("opus", "", "", "opus/x")
    assert t.translate_batch(["hello"]) == ["vi:hello"]
    assert "forced_bos_token_id" not in FakeModel.last_kwargs


def test_missing_dependencies_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    monkeypatch.setitem(sys.modules, "ctranslate2", None)
    ok, why = HFSeq2SeqTranslator("opus", "", "", "opus/x").available()
    assert not ok and "torch" in why


def test_prepare_reports_status(fake_hf):
    msgs = []
    HFSeq2SeqTranslator("opus", "", "", "opus/x").prepare(msgs.append)
    assert any(m.startswith("Đang nạp mô hình") for m in msgs) and "CPU" in msgs[-1]


def test_cache_roundtrip(tmp_path):
    c = TranslationCache(tmp_path / "c.sqlite")
    c.put_many("e", {"a": "x", "b": "y"})
    assert c.get_many("e", ["a", "b", "c"]) == {"a": "x", "b": "y"}
    assert c.get_many("other", ["a"]) == {}


def test_llm_translator_openai_compatible(monkeypatch):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from app.config import settings
    from app.translators import LLMTranslator

    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"data": []})

        def do_POST(self):
            seen["req"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            content = "<think>hmm</think>Bản dịch: \"Mạng {0} hội tụ.\""
            self._send({"choices": [{"message": {"content": content}}]})

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(settings, "llm_base_url", f"http://127.0.0.1:{srv.server_port}/v1")
    try:
        t = LLMTranslator()
        assert t.available()[0]
        out = t.translate_batch(["The {0} network converges."], glossary={"neural network": "mạng nơ-ron"})
        assert out == ["Mạng {0} hội tụ."]
        user_msg = seen["req"]["messages"][1]["content"]
        assert "neural network: mạng nơ-ron" in user_msg and "The {0} network converges." in user_msg
    finally:
        srv.shutdown()


# ---------------- Backend CTranslate2 (giả lập) ----------------


class FakeCT2Tokenizer:
    def __init__(self, **kw):
        self.kw = kw

    def encode(self, text, **kw):
        return [len(w) for w in text.split()] + [2]

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids]

    def convert_tokens_to_ids(self, toks):
        return toks

    def decode(self, toks, skip_special_tokens=True):
        return " ".join(toks)


@pytest.fixture
def fake_ct2(monkeypatch, tmp_path):
    from app.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    calls = {}
    ct2 = types.ModuleType("ctranslate2")
    ct2.get_cuda_device_count = lambda: 0

    class Result:
        def __init__(self, hyp):
            self.hypotheses = [hyp]

    class Translator:
        def __init__(self, path, **kw):
            calls["init"] = (path, kw)

        def translate_batch(self, sources, **kw):
            calls["kw"] = kw
            pre = kw.get("target_prefix")
            return [Result((pre[i] if pre else []) + ["vi"] + s[:-1]) for i, s in enumerate(sources)]

    ct2.Translator = Translator
    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda p, **kw: FakeCT2Tokenizer(path=p, **kw))
    tf.logging = types.SimpleNamespace(set_verbosity_error=lambda: None)
    monkeypatch.setitem(sys.modules, "ctranslate2", ct2)
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "sentencepiece", types.ModuleType("sentencepiece"))
    monkeypatch.setitem(sys.modules, "torch", None)  # không cần torch khi đã có bản CT2
    return calls


def _make_ct2_dir(t):
    d = t.ct2_dir()
    d.mkdir(parents=True)
    (d / "model.bin").write_bytes(b"")
    return d


def test_ct2_backend_without_torch(fake_ct2):
    t = HFSeq2SeqTranslator("opus", "", "", "Helsinki-NLP/opus-mt-en-vi")
    _make_ct2_dir(t)
    assert t.available() == (True, "")
    out = t.translate_batch(["hello big world", "hi"])
    assert out == ["vi t5 t3 t5", "vi t2"]
    path, kw = fake_ct2["init"]
    assert kw["compute_type"] == "int8" and kw["device"] == "cpu"
    assert fake_ct2["kw"]["no_repeat_ngram_size"] == 4 and fake_ct2["kw"]["target_prefix"] is None
    assert t.cache_id().endswith(":ct2")


def test_ct2_nllb_uses_target_prefix_and_strips_it(fake_ct2):
    t = HFSeq2SeqTranslator("nllb", "", "", "facebook/nllb", src_lang="eng_Latn", tgt_lang="vie_Latn", tgt_mode="forced_bos")
    _make_ct2_dir(t)
    assert t.translate_batch(["hello"]) == ["vi t5"]
    assert fake_ct2["kw"]["target_prefix"] == [["vie_Latn"]]


def test_ct2_missing_conversion_without_torch_explains(fake_ct2):
    ok, why = HFSeq2SeqTranslator("vinai", "", "", "vinai/x").available()
    assert not ok and "convert_ct2.py vinai" in why
