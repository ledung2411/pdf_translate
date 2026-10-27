"""Chuyển mô hình dịch sang CTranslate2 int8 (nhanh gấp ~3 lần trên CPU, không cần torch khi dịch).

    python scripts/convert_ct2.py            # mặc định: vinai
    python scripts/convert_ct2.py vinai opus

Cần torch + transformers + ctranslate2 khi chuyển đổi (chỉ một lần). Sau đó có thể gỡ torch.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.translators import HFSeq2SeqTranslator, get_engine  # noqa: E402

if __name__ == "__main__":
    for key in sys.argv[1:] or ["vinai"]:
        eng = get_engine(key)
        if not isinstance(eng, HFSeq2SeqTranslator):
            sys.exit(f"'{key}' không phải mô hình dịch máy Hugging Face.")
        t0 = time.time()
        out = eng.convert_ct2(print)
        print(f"Xong {key} sau {time.time() - t0:.0f}s -> {out}")
