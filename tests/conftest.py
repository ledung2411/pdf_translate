import os
import sys
import tempfile
from pathlib import Path

# Cấu hình trước khi import app (settings đọc biến môi trường lúc import).
os.environ.setdefault("PDFTR_ENABLE_MOCK", "1")
os.environ.setdefault("PDFTR_DATA_DIR", tempfile.mkdtemp(prefix="pdftr-test-"))
os.environ.setdefault("PDFTR_DEFAULT_ENGINE", "mock")
os.environ.setdefault("PDFTR_LAYOUT", "off")  # test mô hình bố cục bật riêng bằng mô hình giả lập
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
