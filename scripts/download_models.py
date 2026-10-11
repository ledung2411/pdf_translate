"""Tải trước mô hình dịch về máy để dùng offline.

    python scripts/download_models.py            # mặc định: vinai
    python scripts/download_models.py opus nllb  # chọn mô hình
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODELS = {
    "vinai": os.environ.get("PDFTR_VINAI_MODEL", "vinai/vinai-translate-en2vi-v2"),
    "opus": os.environ.get("PDFTR_OPUS_MODEL", "Helsinki-NLP/opus-mt-en-vi"),
    "nllb": os.environ.get("PDFTR_NLLB_MODEL", "facebook/nllb-200-distilled-600M"),
}

if __name__ == "__main__":
    keys = sys.argv[1:] or ["vinai"]
    for k in keys:
        if k not in MODELS:
            sys.exit(f"Không biết mô hình '{k}'. Chọn trong: {', '.join(MODELS)}")
        from app.translators import get_engine

        print(f"Đang tải {MODELS[k]} ...")
        get_engine(k).prepare(print)
    print("Xong. Từ giờ mô hình được nạp từ máy, không cần Internet.")
