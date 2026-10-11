"""Các engine dịch máy chạy cục bộ, miễn phí.

- ``vinai``  : vinai/vinai-translate-en2vi-v2 (mBART-50, chất lượng tốt nhất cho Anh→Việt).
- ``opus``   : Helsinki-NLP/opus-mt-en-vi (MarianMT, nhẹ ~300 MB, nhanh trên CPU).
- ``nllb``   : facebook/nllb-200-distilled-600M (đa ngôn ngữ, giấy phép phi thương mại).
- ``llm``    : LLM tự host qua API tương thích OpenAI (Ollama, llama.cpp, LM Studio...).
- ``mock``   : engine giả lập để kiểm thử, không dịch thật.

Mô hình Hugging Face được tải về một lần rồi chạy offline hoàn toàn.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from .config import settings

log = logging.getLogger(__name__)

# Windows không hỗ trợ symlink nếu không bật Developer Mode: cache vẫn chạy được,
# chỉ là cảnh báo gây rối — tắt đi.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class TranslatorError(RuntimeError):
    pass


StatusFn = Callable[[str], None]


def _dir_size_mb(path: Path) -> float:
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total / 1024 / 1024


class Translator(ABC):
    key: str = ""
    label: str = ""
    description: str = ""
    # LLM dịch tốt hơn khi nhận cả đoạn văn và tự xử lý glossary qua prompt.
    prefers_paragraphs: bool = False
    uses_glossary_prompt: bool = False
    # Chỉ LLM giữ được placeholder {n}; mô hình dịch máy biến chúng thành chữ rác.
    supports_placeholders: bool = False

    @abstractmethod
    def translate_batch(self, texts: list[str], glossary: dict[str, str] | None = None) -> list[str]:
        """Dịch một danh sách câu/đoạn tiếng Anh sang tiếng Việt."""

    def available(self) -> tuple[bool, str]:
        return True, ""

    def prepare(self, status: StatusFn | None = None) -> None:
        """Nạp mô hình trước khi dịch (có thể mất vài phút ở lần đầu)."""

    def cache_id(self) -> str:
        return self.key


# ---------------------------------------------------------------------------
# Hugging Face seq2seq
# ---------------------------------------------------------------------------


_TOKENIZER_FILES = [
    "config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json",
    "vocab*", "*.spm", "*.model", "sentencepiece*", "added_tokens.json",
]


class HFSeq2SeqTranslator(Translator):
    def __init__(
        self,
        key: str,
        label: str,
        description: str,
        model_name: str,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        tgt_mode: str | None = None,  # "decoder_start" (mBART-50) | "forced_bos" (NLLB)
    ):
        self.key = key
        self.label = label
        self.description = description
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tgt_mode = tgt_mode
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._lock = threading.Lock()

    def cache_id(self) -> str:
        return f"{self.key}:{self.model_name}:b{settings.num_beams}"

    def available(self) -> tuple[bool, str]:
        missing = []
        for mod in ("torch", "transformers", "sentencepiece"):
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)
            except Exception as e:  # noqa: BLE001  (vd. lỗi DLL của torch trên Windows)
                return False, f"Lỗi khi nạp {mod}: {e}"
        if missing:
            return False, f"Chưa cài {', '.join(missing)} (pip install -r requirements-ml.txt)"
        return True, ""

    def prepare(self, status: StatusFn | None = None) -> None:
        with self._lock:
            self._load(status)

    def _local_path(self, download: bool) -> str:
        """Đường dẫn thư mục mô hình trên máy.

        Nạp từ thư mục cục bộ thay vì tên repo để ``transformers`` không gọi mạng
        mỗi lần khởi động (kiểm tra cập nhật, tự tải thêm bản safetensors ~1,5 GB).
        Chỉ khi chưa có mô hình mới tải về — đúng một lần, chỉ các file cần thiết.
        """
        if Path(self.model_name).exists():
            return self.model_name
        from huggingface_hub import snapshot_download

        if not download:
            return snapshot_download(self.model_name, local_files_only=True)
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(self.model_name)
        if any(f.endswith(".safetensors") for f in files):
            weights = ["model*.safetensors", "model.safetensors.index.json"]
        else:
            weights = ["pytorch_model*.bin", "pytorch_model.bin.index.json"]
        return snapshot_download(self.model_name, allow_patterns=[*_TOKENIZER_FILES, *weights])

    def _watch_download(self, status: StatusFn | None) -> Callable[[], None]:
        """Báo số MB đã tải trong lúc Hugging Face tải mô hình lần đầu."""
        if status is None or Path(self.model_name).exists():
            return lambda: None
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
        except ImportError:
            return lambda: None
        folder = Path(HF_HUB_CACHE) / ("models--" + self.model_name.replace("/", "--"))
        stop = threading.Event()

        def run() -> None:
            last = -1.0
            while not stop.wait(1.5):
                mb = _dir_size_mb(folder) if folder.exists() else 0.0
                if mb != last:
                    status(f"Đang tải mô hình {self.model_name} (chỉ lần đầu): {mb:,.0f} MB...")
                    last = mb

        threading.Thread(target=run, daemon=True).start()
        return stop.set

    def _load(self, status: StatusFn | None = None) -> None:
        if self._model is not None:
            return
        ok, why = self.available()
        if not ok:
            raise TranslatorError(why)
        import torch
        import transformers
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        transformers.logging.set_verbosity_error()

        device = settings.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        log.info("Đang nạp mô hình %s trên %s ...", self.model_name, device)
        if status:
            status(f"Đang nạp mô hình {self.model_name}...")
        kwargs = {}
        if self.src_lang:
            kwargs["src_lang"] = self.src_lang
        stop_watch = self._watch_download(status)
        try:
            try:
                path = self._local_path(download=False)
                tok = AutoTokenizer.from_pretrained(path, **kwargs)
                model = AutoModelForSeq2SeqLM.from_pretrained(path)
            except Exception:  # noqa: BLE001  (chưa có trong cache / cache thiếu file)
                path = self._local_path(download=True)
                tok = AutoTokenizer.from_pretrained(path, **kwargs)
                model = AutoModelForSeq2SeqLM.from_pretrained(path)
        except Exception as e:  # noqa: BLE001
            raise TranslatorError(
                f"Không nạp được mô hình {self.model_name}: {e}. "
                "Lần chạy đầu cần Internet để tải mô hình (hoặc chạy scripts/download_models.py)."
            ) from e
        finally:
            stop_watch()
        if device == "cuda":
            model = model.half()
        model.to(device).eval()
        self._tokenizer, self._model, self._device = tok, model, device
        log.info("Đã nạp xong %s", self.model_name)
        if status:
            status(f"Đã nạp mô hình ({device.upper()})")

    def translate_batch(self, texts: list[str], glossary: dict[str, str] | None = None) -> list[str]:
        if not texts:
            return []
        with self._lock:
            self._load()
            import torch

            tok, model = self._tokenizer, self._model
            gen_kwargs: dict = {
                "num_beams": settings.num_beams,
                "early_stopping": True,
                # Chặn vòng lặp kiểu "chúng ta tính chúng ta tính..." của mô hình nhỏ.
                "no_repeat_ngram_size": 4,
            }
            if self.tgt_lang:
                tgt_id = tok.convert_tokens_to_ids(self.tgt_lang)
                if self.tgt_mode == "decoder_start":
                    gen_kwargs["decoder_start_token_id"] = tgt_id
                else:
                    gen_kwargs["forced_bos_token_id"] = tgt_id

            # Sắp theo độ dài để giảm padding, rồi trả về đúng thứ tự.
            order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            results: list[str] = [""] * len(texts)
            bs = max(1, settings.batch_size)
            for start in range(0, len(order), bs):
                idx = order[start : start + bs]
                batch = [texts[i] for i in idx]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                enc = {k: v.to(self._device) for k, v in enc.items()}
                # Bản dịch không thể dài hơn ~2,5 lần câu gốc: chặn sinh lan man tới 512 token.
                in_len = int(enc["input_ids"].shape[1])
                with torch.inference_mode():
                    out = model.generate(**enc, **gen_kwargs, max_length=min(512, int(in_len * 2.5) + 10))
                decoded = tok.batch_decode(out, skip_special_tokens=True)
                for i, d in zip(idx, decoded):
                    results[i] = d.strip()
            return results


# ---------------------------------------------------------------------------
# LLM cục bộ qua API tương thích OpenAI (Ollama / llama.cpp / LM Studio / vLLM)
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = """Bạn là biên dịch viên chuyên dịch tài liệu khoa học từ tiếng Anh sang tiếng Việt.
Quy tắc:
1. Dịch chính xác, trôi chảy, văn phong học thuật tiếng Việt; dùng thuật ngữ chuyên ngành chuẩn.
2. GIỮ NGUYÊN mọi placeholder dạng {0}, {1}, {12}... đúng vị trí ngữ pháp phù hợp — không dịch, không xóa, không thêm.
3. Giữ nguyên công thức, ký hiệu, số liệu, đơn vị, tên riêng, tên phần mềm và trích dẫn.
4. Chỉ trả về bản dịch, không giải thích, không thêm ngoặc kép hay tiền tố."""


class LLMTranslator(Translator):
    key = "llm"
    label = "LLM cục bộ (Ollama / llama.cpp / LM Studio)"
    description = "Dịch theo ngữ cảnh cả đoạn, hiểu thuật ngữ tốt; cần tự chạy một LLM mã nguồn mở."
    prefers_paragraphs = True
    uses_glossary_prompt = True
    supports_placeholders = True

    def cache_id(self) -> str:
        return f"llm:{settings.llm_model}"

    def available(self) -> tuple[bool, str]:
        try:
            self._request("GET", "/models", timeout=3)
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, f"Không kết nối được LLM tại {settings.llm_base_url} ({e.__class__.__name__})"

    def _request(self, method: str, path: str, payload: dict | None = None, timeout: float = 600):
        url = settings.llm_base_url.rstrip("/") + path
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if settings.llm_api_key:
            req.add_header("Authorization", f"Bearer {settings.llm_api_key}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _translate_one(self, text: str, glossary: dict[str, str] | None) -> str:
        user = ""
        if glossary:
            terms = "\n".join(f"- {en}: {vi}" for en, vi in list(glossary.items())[:40])
            user += f"Thuật ngữ bắt buộc dùng:\n{terms}\n\n"
        user += f"Dịch đoạn sau sang tiếng Việt:\n\n{text}"
        payload = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "stream": False,
        }
        try:
            res = self._request("POST", "/chat/completions", payload)
        except urllib.error.URLError as e:
            raise TranslatorError(f"Lỗi gọi LLM tại {settings.llm_base_url}: {e}") from e
        out = res["choices"][0]["message"]["content"] or ""
        # Bỏ phần "suy nghĩ" của các mô hình reasoning (Qwen3, DeepSeek-R1...).
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
        out = re.sub(r"^(bản dịch|translation)\s*:\s*", "", out, flags=re.I)
        if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'“”":
            out = out[1:-1]
        return out.strip()

    def translate_batch(self, texts: list[str], glossary: dict[str, str] | None = None) -> list[str]:
        return [self._translate_one(t, glossary) for t in texts]


# ---------------------------------------------------------------------------
# Engine giả lập để kiểm thử
# ---------------------------------------------------------------------------


class MockTranslator(Translator):
    key = "mock"
    label = "Giả lập (kiểm thử)"
    description = "Không dịch thật — chỉ dùng để kiểm tra bố cục."

    def translate_batch(self, texts: list[str], glossary: dict[str, str] | None = None) -> list[str]:
        return [f"[VI] {t} — bản dịch thử nghiệm có dấu: ạ ế ữ ở đ" for t in texts]


# ---------------------------------------------------------------------------
# Bộ nhớ đệm bản dịch (SQLite)
# ---------------------------------------------------------------------------


class TranslationCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._conn.commit()
        self._lock = threading.Lock()

    @staticmethod
    def _key(engine_id: str, text: str, extra: str) -> str:
        return hashlib.sha256(f"{engine_id}\x00{extra}\x00{text}".encode()).hexdigest()

    def get_many(self, engine_id: str, texts: list[str], extra: str = "") -> dict[str, str]:
        if not texts:
            return {}
        keys = {self._key(engine_id, t, extra): t for t in texts}
        found: dict[str, str] = {}
        with self._lock:
            items = list(keys)
            for i in range(0, len(items), 500):
                chunk = items[i : i + 500]
                q = "SELECT k, v FROM cache WHERE k IN (%s)" % ",".join("?" * len(chunk))
                for k, v in self._conn.execute(q, chunk):
                    found[keys[k]] = v
        return found

    def put_many(self, engine_id: str, pairs: dict[str, str], extra: str = "") -> None:
        if not pairs:
            return
        rows = [(self._key(engine_id, s, extra), t) for s, t in pairs.items()]
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)", rows)
            self._conn.commit()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ENGINES: dict[str, Translator] = {}
_cache: TranslationCache | None = None


def _build_registry() -> None:
    if _ENGINES:
        return
    engines: list[Translator] = [
        HFSeq2SeqTranslator(
            "vinai",
            "VinAI Translate En→Vi (khuyên dùng)",
            "Mô hình của VinAI, chất lượng tốt nhất trong nhóm (~1,5 GB). Chạy được trên CPU, nhanh hơn nhiều nếu có GPU.",
            os.environ.get("PDFTR_VINAI_MODEL", "vinai/vinai-translate-en2vi-v2"),
            src_lang="en_XX",
            tgt_lang="vi_VN",
            tgt_mode="decoder_start",
        ),
        HFSeq2SeqTranslator(
            "opus",
            "OPUS-MT En→Vi (nhanh, chất lượng thấp)",
            "Helsinki-NLP MarianMT ~300 MB, nhanh trên CPU nhưng hay dịch sai thuật ngữ — chỉ để xem nhanh.",
            os.environ.get("PDFTR_OPUS_MODEL", "Helsinki-NLP/opus-mt-en-vi"),
        ),
        HFSeq2SeqTranslator(
            "nllb",
            "NLLB-200 600M (Meta)",
            "Mô hình đa ngôn ngữ của Meta (~2,4 GB), giấy phép CC-BY-NC (phi thương mại).",
            os.environ.get("PDFTR_NLLB_MODEL", "facebook/nllb-200-distilled-600M"),
            src_lang="eng_Latn",
            tgt_lang="vie_Latn",
            tgt_mode="forced_bos",
        ),
        LLMTranslator(),
    ]
    if settings.enable_mock:
        engines.append(MockTranslator())
    for e in engines:
        _ENGINES[e.key] = e


def default_engine_key() -> str:
    if settings.default_engine != "auto":
        return settings.default_engine
    try:
        import torch

        # VinAI chất lượng tốt hơn OPUS rất nhiều với văn bản khoa học, nên dùng
        # mặc định kể cả trên CPU (chậm hơn nhưng bản dịch đọc được).
        return "vinai"
    except Exception:  # noqa: BLE001
        # Chưa cài torch: dùng LLM cục bộ nếu đang chạy, nếu không vẫn trả về
        # "opus" để thông báo lỗi hướng dẫn cài đặt.
        _build_registry()
        llm = _ENGINES.get("llm")
        if llm is not None and llm.available()[0]:
            return "llm"
        return "opus"


def get_engine(key: str) -> Translator:
    _build_registry()
    if key in ("", "auto"):
        key = default_engine_key()
    if key not in _ENGINES:
        raise TranslatorError(f"Engine không hợp lệ: {key}")
    return _ENGINES[key]


def list_engines() -> list[Translator]:
    _build_registry()
    return list(_ENGINES.values())


def get_cache() -> TranslationCache:
    global _cache
    if _cache is None:
        _cache = TranslationCache(settings.data_dir / "cache.sqlite")
    return _cache
