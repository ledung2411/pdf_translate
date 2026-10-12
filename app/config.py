"""Cấu hình qua biến môi trường (tiền tố PDFTR_)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("PDFTR_DATA_DIR", "data")).resolve())
    # "auto": dùng VinAI (chất lượng tốt nhất) nếu đã cài torch, nếu không thì LLM cục bộ.
    default_engine: str = os.environ.get("PDFTR_DEFAULT_ENGINE", "auto")
    device: str = os.environ.get("PDFTR_DEVICE", "auto")  # auto | cpu | cuda | mps
    # Backend mô hình dịch máy: auto (ưu tiên CTranslate2) | ct2 | hf (transformers+torch)
    backend: str = os.environ.get("PDFTR_BACKEND", "auto")
    threads: int = int(os.environ.get("PDFTR_THREADS", "0"))  # 0 = tự chọn
    batch_size: int = int(os.environ.get("PDFTR_BATCH_SIZE", "8"))
    num_beams: int = int(os.environ.get("PDFTR_NUM_BEAMS", "4"))
    max_upload_mb: int = int(os.environ.get("PDFTR_MAX_UPLOAD_MB", "100"))
    max_pages: int = int(os.environ.get("PDFTR_MAX_PAGES", "500"))
    job_ttl_hours: float = float(os.environ.get("PDFTR_JOB_TTL_HOURS", "24"))
    enable_mock: bool = _env_bool("PDFTR_ENABLE_MOCK", False)
    # Font TTF hỗ trợ tiếng Việt (tùy chọn). Mặc định tự tìm font hệ thống / Noto.
    font_serif: str = os.environ.get("PDFTR_FONT_SERIF", "")
    font_sans: str = os.environ.get("PDFTR_FONT_SANS", "")
    # LLM tự host qua API tương thích OpenAI. Mặc định là Ollama chạy trên máy.
    llm_base_url: str = os.environ.get("PDFTR_LLM_BASE_URL", "http://localhost:11434/v1")
    llm_model: str = os.environ.get("PDFTR_LLM_MODEL", "qwen2.5:7b")
    llm_api_key: str = os.environ.get("PDFTR_LLM_API_KEY", "")


settings = Settings()
