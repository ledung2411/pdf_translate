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


def _patch_transformers_for_ct2() -> None:
    """transformers 5.x bỏ ``normalize_before`` khỏi BartConfig/MBartConfig nhưng bộ chuyển đổi
    CTranslate2 (≤4.8) vẫn đọc thuộc tính này. mBART (VinAI, NLLB-style) dùng pre-norm, BART post-norm."""
    import transformers

    for name, value in (("MBartConfig", True), ("BartConfig", False)):
        cls = getattr(transformers, name, None)
        if cls is not None and not hasattr(cls(), "normalize_before"):
            setattr(cls, "normalize_before", value)


def _has_module(name: str) -> tuple[bool, str]:
    try:
        __import__(name)
        return True, ""
    except ImportError:
        return False, ""
    except Exception as e:  # noqa: BLE001  (vd. lỗi DLL của torch trên Windows)
        return False, f"Lỗi khi nạp {name}: {e}"


class HFSeq2SeqTranslator(Translator):
    """Mô hình dịch máy Hugging Face, chạy bằng một trong hai backend:

    - ``ct2``: CTranslate2 int8 — nhanh gấp ~3 lần trên CPU (đo bằng scripts/benchmark.py), ít RAM, KHÔNG cần torch.
      Lần đầu tự chuyển đổi từ mô hình Hugging Face (cần torch đúng một lần).
    - ``hf`` : transformers + torch (dự phòng khi chưa cài ctranslate2).
    """

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
        self._backend = ""
        self._lock = threading.Lock()

    # -- trạng thái ------------------------------------------------------------

    def ct2_dir(self) -> Path:
        name = self.model_name.strip("/\\").replace("/", "--").replace("\\", "--").replace(":", "")
        return settings.data_dir / "models" / "ct2" / name

    def _ct2_ready(self) -> bool:
        return (self.ct2_dir() / "model.bin").exists()

    def _choose_backend(self) -> tuple[str, str]:
        """Trả về (backend, lý do nếu không dùng được)."""
        want = settings.backend
        tf_ok, tf_err = _has_module("transformers")
        sp_ok, sp_err = _has_module("sentencepiece")
        ct2_ok, ct2_err = _has_module("ctranslate2")
        torch_ok, torch_err = _has_module("torch")
        err = tf_err or sp_err or ct2_err or torch_err
        if err:
            return "", err
        missing = [n for n, ok in (("transformers", tf_ok), ("sentencepiece", sp_ok)) if not ok]
        if not (ct2_ok or torch_ok):
            missing.append("ctranslate2 (hoặc torch)")
        if missing:
            return "", f"Chưa cài {', '.join(missing)} (pip install -r requirements-ml.txt)"
        if want in ("auto", "ct2") and ct2_ok and (self._ct2_ready() or torch_ok):
            return "ct2", ""
        if want in ("auto", "hf") and torch_ok:
            return "hf", ""
        if want == "ct2" or (ct2_ok and not torch_ok):
            return "", (
                "Chưa có bản CTranslate2 của mô hình. Cài torch một lần để chuyển đổi: "
                "pip install torch --index-url https://download.pytorch.org/whl/cpu, "
                f"rồi chạy python scripts/convert_ct2.py {self.key}"
            )
        return "", f"Backend '{want}' không dùng được (PDFTR_BACKEND)"

    def cache_id(self) -> str:
        backend = self._backend or self._choose_backend()[0]
        return f"{self.key}:{self.model_name}:b{settings.num_beams}:{backend}"

    def available(self) -> tuple[bool, str]:
        backend, why = self._choose_backend()
        return (bool(backend), why)

    def prepare(self, status: StatusFn | None = None) -> None:
        with self._lock:
            self._load(status)

    # -- tải mô hình gốc từ Hugging Face ----------------------------------------

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

    def _hf_source(self, status: StatusFn | None) -> str:
        """Thư mục mô hình Hugging Face trên máy (tải nếu chưa có)."""
        stop_watch = self._watch_download(status)
        try:
            try:
                path = self._local_path(download=False)
                if any(Path(path).glob("*.bin")) or any(Path(path).glob("*.safetensors")):
                    return path
            except Exception:  # noqa: BLE001
                pass
            return self._local_path(download=True)
        except Exception as e:  # noqa: BLE001
            raise TranslatorError(
                f"Không tải được mô hình {self.model_name}: {e}. "
                "Lần chạy đầu cần Internet để tải mô hình (hoặc chạy scripts/download_models.py)."
            ) from e
        finally:
            stop_watch()

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

    def _tokenizer_kwargs(self) -> dict:
        return {"src_lang": self.src_lang} if self.src_lang else {}

    # -- CTranslate2 -----------------------------------------------------------

    def convert_ct2(self, status: StatusFn | None = None) -> Path:
        """Chuyển mô hình Hugging Face sang CTranslate2 int8 (cần torch, chỉ một lần)."""
        from ctranslate2.converters import TransformersConverter
        from transformers import AutoTokenizer

        src = self._hf_source(status)
        out = self.ct2_dir()
        tmp = out.with_name(out.name + ".tmp")
        if status:
            status(f"Đang chuyển {self.model_name} sang CTranslate2 int8 (chỉ lần đầu, 1–3 phút)...")
        log.info("Chuyển đổi %s -> %s", src, out)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        _patch_transformers_for_ct2()
        TransformersConverter(src).convert(str(tmp), quantization="int8", force=True)
        tok = AutoTokenizer.from_pretrained(src, **self._tokenizer_kwargs())
        tok.save_pretrained(str(tmp))
        cfg_path = tmp / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        # Một số tokenizer không khai báo bos/unk → CTranslate2 từ chối nạp config có giá trị null.
        defaults = {
            "bos_token": tok.bos_token or "<s>",
            "eos_token": tok.eos_token or "</s>",
            "unk_token": tok.unk_token or "<unk>",
            "decoder_start_token": tok.eos_token or "</s>",
        }
        for key, value in defaults.items():
            if key in cfg and cfg[key] is None:
                cfg[key] = value
        if self.tgt_mode == "decoder_start" and self.tgt_lang:
            # VinAI (mBART-50) bắt đầu giải mã trực tiếp bằng mã ngôn ngữ đích.
            cfg["decoder_start_token"] = self.tgt_lang
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        if out.exists():
            import shutil

            shutil.rmtree(out)
        tmp.rename(out)
        return out

    def _load_ct2(self, status: StatusFn | None) -> None:
        import ctranslate2
        from transformers import AutoTokenizer

        if not self._ct2_ready():
            self.convert_ct2(status)
        device = settings.device
        if device in ("auto", "cuda"):
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        if device not in ("cuda", "cpu"):
            device = "cpu"
        compute = "int8_float16" if device == "cuda" else "int8"
        if status:
            status(f"Đang nạp mô hình {self.model_name} (CTranslate2 int8)...")
        self._model = ctranslate2.Translator(
            str(self.ct2_dir()), device=device, compute_type=compute,
            intra_threads=settings.threads or 0,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.ct2_dir()), **self._tokenizer_kwargs())
        self._device = device

    def _translate_ct2(self, batch: list[str]) -> list[str]:
        tok = self._tokenizer
        sources = [
            tok.convert_ids_to_tokens(tok.encode(t, truncation=True, max_length=512)) for t in batch
        ]
        prefix = None
        if self.tgt_mode == "forced_bos" and self.tgt_lang:
            prefix = [[self.tgt_lang] for _ in batch]
        longest = max(len(x) for x in sources)
        results = self._model.translate_batch(
            sources,
            target_prefix=prefix,
            beam_size=settings.num_beams,
            max_batch_size=max(1, settings.batch_size),
            max_decoding_length=min(512, int(longest * 2.5) + 10),
            no_repeat_ngram_size=4,
        )
        out = []
        for r in results:
            hyp = r.hypotheses[0]
            if prefix and hyp and hyp[0] == self.tgt_lang:
                hyp = hyp[1:]
            out.append(tok.decode(tok.convert_tokens_to_ids(hyp), skip_special_tokens=True).strip())
        return out

    # -- transformers + torch ---------------------------------------------------

    def _load_hf(self, status: StatusFn | None) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = settings.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        path = self._hf_source(status)
        if status:
            status(f"Đang nạp mô hình {self.model_name}...")
        try:
            tok = AutoTokenizer.from_pretrained(path, **self._tokenizer_kwargs())
            model = AutoModelForSeq2SeqLM.from_pretrained(path)
        except Exception as e:  # noqa: BLE001
            raise TranslatorError(f"Không nạp được mô hình {self.model_name}: {e}") from e
        if device == "cuda":
            model = model.half()
        model.to(device).eval()
        self._tokenizer, self._model, self._device = tok, model, device

    def _translate_hf(self, batch: list[str]) -> list[str]:
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
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(self._device) for k, v in enc.items()}
        # Bản dịch không thể dài hơn ~2,5 lần câu gốc: chặn sinh lan man tới 512 token.
        in_len = int(enc["input_ids"].shape[1])
        with torch.inference_mode():
            out = model.generate(**enc, **gen_kwargs, max_length=min(512, int(in_len * 2.5) + 10))
        return [d.strip() for d in tok.batch_decode(out, skip_special_tokens=True)]

    # -- chung -------------------------------------------------------------------

    def _load(self, status: StatusFn | None = None) -> None:
        if self._model is not None:
            return
        backend, why = self._choose_backend()
        if not backend:
            raise TranslatorError(why)
        import transformers

        transformers.logging.set_verbosity_error()
        log.info("Nạp %s bằng backend %s", self.model_name, backend)
        try:
            if backend == "ct2":
                self._load_ct2(status)
            else:
                self._load_hf(status)
        except TranslatorError:
            raise
        except Exception as e:  # noqa: BLE001
            raise TranslatorError(f"Không nạp được mô hình {self.model_name} ({backend}): {e}") from e
        self._backend = backend
        log.info("Đã nạp xong %s (%s, %s)", self.model_name, backend, self._device)
        if status:
            status(f"Đã nạp mô hình ({backend.upper()}, {self._device.upper()})")

    def translate_batch(self, texts: list[str], glossary: dict[str, str] | None = None) -> list[str]:
        if not texts:
            return []
        with self._lock:
            self._load()
            run = self._translate_ct2 if self._backend == "ct2" else self._translate_hf
            # Sắp theo độ dài để giảm padding, rồi trả về đúng thứ tự.
            order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            results: list[str] = [""] * len(texts)
            bs = max(1, settings.batch_size)
            for start in range(0, len(order), bs):
                idx = order[start : start + bs]
                for i, d in zip(idx, run([texts[i] for i in idx])):
                    results[i] = d
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


POST_EDIT_SYSTEM_PROMPT = """Bạn là biên tập viên hiệu đính bản dịch tài liệu khoa học Anh → Việt.
Sửa bản dịch máy cho đúng nghĩa câu gốc, dùng thuật ngữ chuyên ngành chuẩn, văn phong học thuật.
GIỮ NGUYÊN mọi placeholder dạng {0}, {1}... nếu có. Chỉ trả về bản dịch đã sửa, không giải thích."""


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
        return self._chat(LLM_SYSTEM_PROMPT, user)

    def _chat(self, system: str, user: str, max_tokens: int | None = None) -> str:
        payload = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens  # chặn mô hình nhỏ lặp chữ vô tận
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

    def post_edit(self, source: str, draft: str, glossary: dict[str, str] | None = None) -> str:
        """Hiệu đính bản dịch máy của một câu (dùng cho các câu bị chấm điểm thấp)."""
        user = ""
        if glossary:
            terms = "\n".join(f"- {en}: {vi}" for en, vi in list(glossary.items())[:40])
            user += f"Thuật ngữ bắt buộc dùng:\n{terms}\n\n"
        user += (
            f"Câu gốc tiếng Anh:\n{source}\n\n"
            f"Bản dịch máy (có thể sai nghĩa, sai thuật ngữ hoặc còn sót tiếng Anh):\n{draft}\n\n"
            "Hãy viết lại bản dịch tiếng Việt cho đúng nghĩa và trôi chảy. Giữ nguyên số liệu, ký hiệu, "
            "trích dẫn và tên riêng. Chỉ trả về bản dịch đã sửa."
        )
        return self._chat(POST_EDIT_SYSTEM_PROMPT, user)


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
    """Engine mặc định: mô hình tốt nhất đang dùng được ngay (VinAI → OPUS → LLM cục bộ).

    Nếu chưa có engine nào sẵn sàng, trả về "vinai" để thông báo lỗi hướng dẫn cài đặt.
    """
    if settings.default_engine != "auto":
        return settings.default_engine
    _build_registry()
    for key in ("vinai", "opus", "llm"):
        eng = _ENGINES.get(key)
        if eng is not None and eng.available()[0]:
            return key
    return "vinai"


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
