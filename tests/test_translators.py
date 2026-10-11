"""Kiểm tra lớp HFSeq2SeqTranslator với torch/transformers giả lập (không cần tải mô hình)."""

import contextlib
import sys
import types

import pytest

from app.translators import HFSeq2SeqTranslator, TranslationCache


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows

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
def fake_hf(monkeypatch):
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
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda name, **kw: created.setdefault("snap", []).append(kw) or f"/cache/{name}"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
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
    assert fake_hf["tok"].kwargs["name"] == "/cache/opus/x"


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
    ok, why = HFSeq2SeqTranslator("opus", "", "", "opus/x").available()
    assert not ok and "torch" in why


def test_prepare_reports_status(fake_hf):
    msgs = []
    HFSeq2SeqTranslator("opus", "", "", "opus/x").prepare(msgs.append)
    assert msgs[0].startswith("Đang nạp mô hình") and "CPU" in msgs[-1]


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
