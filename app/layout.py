"""Nhận diện bố cục trang bằng mô hình DocLayout-YOLO (ONNX, chạy CPU bằng onnxruntime).

Mô hình nhìn ảnh trang và khoanh vùng: tiêu đề, đoạn văn, công thức đứng riêng, bảng,
hình, chú thích, đầu/chân trang... Kết quả được dùng để *sửa phân loại* các vùng chữ trích
từ PyMuPDF (vd. đoạn bị tưởng là bảng, công thức bị tưởng là chữ). Mọi lỗi (chưa cài
onnxruntime, không tải được mô hình...) đều quay về phân loại bằng luật như cũ.

Mô hình mặc định: DocLayout-YOLO huấn luyện trên DocStructBench, bản ONNX (~75 MB),
tải một lần từ Hugging Face. Đổi bằng ``PDFTR_LAYOUT_MODEL`` (``repo_id/tên_file.onnx``
hoặc đường dẫn file .onnx trên máy).
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .config import settings
from .scientific import has_prose

log = logging.getLogger(__name__)

# Nhãn của DocStructBench
LABELS = {
    0: "title", 1: "text", 2: "abandon", 3: "figure", 4: "figure_caption", 5: "table",
    6: "table_caption", 7: "table_footnote", 8: "formula", 9: "formula_caption",
}


@dataclass
class Detection:
    rect: pymupdf.Rect
    label: str
    score: float


class LayoutError(RuntimeError):
    pass


def _nms(boxes, scores, iou: float = 0.5):
    import numpy as np

    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        ov = inter / (area(boxes[i]) + area(boxes[order[1:]]) - inter + 1e-9)
        order = order[1:][ov < iou]
    return keep


class LayoutDetector:
    def __init__(self, model_path: str, imgsz: int = 1024, conf: float = 0.25):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if settings.threads:
            opts.intra_op_num_threads = settings.threads
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape  # [1, 3, H, W] — có thể là số cố định hoặc tên chiều động
        self.fixed = (shape[2], shape[3]) if all(isinstance(x, int) for x in shape[2:]) else None
        self.imgsz = max(self.fixed) if self.fixed else imgsz
        self.conf = conf

    def _prepare(self, page: pymupdf.Page):
        import numpy as np

        zoom = self.imgsz / max(page.rect.width, page.rect.height)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        if self.fixed:
            th, tw = self.fixed
        else:  # bội số của 32 (stride của YOLO)
            th, tw = (-(-pix.height // 32) * 32, -(-pix.width // 32) * 32)
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        canvas[: min(th, pix.height), : min(tw, pix.width)] = img[:th, :tw]
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return x, zoom

    def _decode(self, out):
        """Hỗ trợ 2 dạng đầu ra: YOLOv10 [1,N,6] (x1,y1,x2,y2,score,class) và YOLOv8 [1,4+C,N]."""
        import numpy as np

        out = np.asarray(out)
        if out.ndim == 3 and out.shape[-1] == 6:
            rows = out[0]
            return [(r[:4], float(r[4]), int(r[5])) for r in rows if r[4] >= self.conf]
        if out.ndim == 3 and out.shape[1] == 6 and out.shape[2] != 6:
            return self._decode(out.transpose(0, 2, 1))
        if out.ndim == 3:  # YOLOv8: [1, 4 + C, N], box dạng cx, cy, w, h
            pred = out[0].T
            cls_scores = pred[:, 4:]
            cls = cls_scores.argmax(1)
            score = cls_scores.max(1)
            m = score >= self.conf
            pred, cls, score = pred[m], cls[m], score[m]
            boxes = np.stack([pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2,
                              pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2], 1)
            keep = _nms(boxes, score) if len(score) else []
            return [(boxes[i], float(score[i]), int(cls[i])) for i in keep]
        raise LayoutError(f"Không hiểu định dạng đầu ra của mô hình bố cục: {out.shape}")

    def detect(self, page: pymupdf.Page) -> list[Detection]:
        x, zoom = self._prepare(page)
        out = self.session.run(None, {self.input_name: x})[0]
        dets = []
        for box, score, cls in self._decode(out):
            r = pymupdf.Rect(*(float(v) / zoom for v in box)) & page.rect
            if not r.is_empty:
                dets.append(Detection(r, LABELS.get(cls, str(cls)), score))
        return dets


# ---------------------------------------------------------------------------
# Hợp nhất kết quả mô hình với các vùng chữ
# ---------------------------------------------------------------------------

# Vùng mô hình xác định là KHÔNG dịch → loại vùng trong DocIR
_KEEP = {"formula": "math", "table": "table", "figure": "figure", "abandon": "abandon"}
_TEXTUAL = {"title", "text", "figure_caption", "table_caption", "table_footnote", "formula_caption"}


def _coverage(region: pymupdf.Rect, box: pymupdf.Rect) -> float:
    area = abs(region)
    return abs(region & box) / area if area else 0.0


def apply_layout(regions: list, detections: list[Detection]) -> int:
    """Sửa ``kind`` của vùng theo nhãn mô hình. Trả về số vùng bị đổi loại."""
    changed = 0
    for r in regions:
        if r.kind in ("rotated", "empty", "reference"):
            continue
        best, best_cov = None, 0.0
        for d in detections:
            cov = _coverage(r.bbox, d.rect)
            if cov > best_cov:
                best, best_cov = d, cov
        if best is None or best_cov < 0.6:
            continue
        words = len(re.findall(r"[A-Za-z]{3,}", r.plain))
        new = r.kind
        if best.label in _KEEP:
            # Không bỏ dịch một đoạn văn dài chỉ vì khung mô hình chồng lên (khung bảng/hình lớn)
            if best.label in ("table", "figure") and words > 25 and has_prose(r.plain):
                continue
            if best.label == "formula" and has_prose(r.plain) and words > 8:
                continue
            new = _KEEP[best.label]
        elif best.label in _TEXTUAL and r.kind in ("table", "data", "code") and has_prose(r.plain):
            new = "text"  # luật tưởng là bảng/dữ liệu nhưng mô hình thấy là đoạn văn
        if new != r.kind:
            r.kind = new
            changed += 1
    return changed


# ---------------------------------------------------------------------------
# Nạp mô hình (một lần, dùng chung)
# ---------------------------------------------------------------------------

_detector: LayoutDetector | None = None
_failed: str | None = None
_lock = threading.Lock()


def _model_path() -> str:
    spec = settings.layout_model
    if Path(spec).is_file():
        return spec
    repo, _, filename = spec.rpartition("/")
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo, filename, local_files_only=True)
    except Exception:  # noqa: BLE001
        log.info("Tải mô hình bố cục %s (chỉ lần đầu)...", spec)
        return hf_hub_download(repo, filename)


def get_detector() -> LayoutDetector | None:
    """Trả về bộ nhận diện bố cục, hoặc None nếu tắt / không dùng được (dùng luật thay thế)."""
    global _detector, _failed
    if settings.layout == "off":
        return None
    with _lock:
        if _detector is not None or _failed is not None:
            return _detector
        try:
            import onnxruntime  # noqa: F401  (chỉ kiểm tra đã cài)
        except ImportError:
            _failed = "chưa cài onnxruntime"
            log.info("Nhận diện bố cục bằng mô hình: tắt (%s)", _failed)
            return None
        try:
            _detector = LayoutDetector(_model_path(), imgsz=settings.layout_imgsz)
            log.info("Đã nạp mô hình bố cục %s", settings.layout_model)
        except Exception as e:  # noqa: BLE001
            _failed = str(e)
            log.warning("Không dùng được mô hình bố cục (%s) — dùng phân loại bằng luật", e)
        return _detector


def reset() -> None:
    global _detector, _failed
    _detector, _failed = None, None


def status() -> tuple[bool, str]:
    det = get_detector()
    if det is not None:
        return True, settings.layout_model
    return False, _failed or "đã tắt (PDFTR_LAYOUT=off)"
