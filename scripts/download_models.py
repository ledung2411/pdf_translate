"""Tải trước mô hình dịch về máy để dùng offline.

    python scripts/download_models.py            # mặc định: vinai
    python scripts/download_models.py opus nllb  # chọn mô hình
"""

import os
import sys

from huggingface_hub import snapshot_download

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
        print(f"Đang tải {MODELS[k]} ...")
        path = snapshot_download(
            MODELS[k],
            allow_patterns=["*.json", "*.model", "*.spm", "*.txt", "*.safetensors", "*.bin", "*.py", "sentencepiece*"],
        )
        print(f"  -> {path}")
    print("Xong. Có thể đặt HF_HUB_OFFLINE=1 để chạy hoàn toàn offline.")
