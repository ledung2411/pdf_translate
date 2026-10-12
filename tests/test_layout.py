"""Kiểm thử nhận diện bố cục với mô hình ONNX giả lập (trả về khung cố định)."""

import numpy as np
import pymupdf
import pytest

from app.ir import Region
from app.layout import LABELS, Detection, LayoutDetector, apply_layout

from .make_sample_pdf import make

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
LABEL_ID = {v: k for k, v in LABELS.items()}


def fake_model(path, output: np.ndarray, fixed: int | None = None):
    from onnx import TensorProto, helper, numpy_helper

    dims = [1, 3, fixed, fixed] if fixed else [1, 3, "h", "w"]
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, dims)
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(output.shape))
    const = helper.make_node("Constant", [], ["output0"], value=numpy_helper.from_array(output.astype(np.float32)))
    graph = helper.make_graph([const], "fake", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return str(path)


def _region(text, kind="text", bbox=(100, 100, 300, 140)):
    return Region(page_index=0, bbox=pymupdf.Rect(bbox), text=text, kind=kind, font_size=10,
                  bold=False, italic=False, serif=True, color=0, align="left")


def test_apply_layout_rules():
    para = "This is a long paragraph of normal scientific prose that should never be skipped " * 3
    regs = [
        _region("x = y + z (1)", bbox=(100, 100, 300, 120)),
        _region(para, bbox=(100, 200, 500, 300)),
        _region("Method Dice HD95 U-Net results are shown here", kind="table", bbox=(100, 400, 300, 440)),
        _region("Advanced Science, 2026", bbox=(50, 800, 200, 810)),
    ]
    dets = [
        Detection(pymupdf.Rect(90, 95, 310, 125), "formula", 0.9),
        Detection(pymupdf.Rect(90, 190, 510, 310), "table", 0.6),  # khung bảng lớn chồng lên đoạn văn dài
        Detection(pymupdf.Rect(90, 395, 310, 445), "text", 0.8),
        Detection(pymupdf.Rect(40, 795, 210, 815), "abandon", 0.8),
    ]
    assert apply_layout(regs, dets) == 3
    assert [r.kind for r in regs] == ["math", "text", "text", "abandon"]


@pytest.mark.parametrize("fixed", [None, 1024])
def test_detector_yolov10_output_maps_to_page(tmp_path, fixed):
    pdf = tmp_path / "s.pdf"
    make(str(pdf))
    page = pymupdf.open(str(pdf))[0]
    zoom = 1024 / max(page.rect.width, page.rect.height)
    target = pymupdf.Rect(253, 102, 342, 117)  # "University of Science"
    rows = np.zeros((1, 300, 6), np.float32)
    rows[0, 0] = [*(np.array(list(target)) * zoom), 0.9, LABEL_ID["abandon"]]
    rows[0, 1] = [10, 10, 20, 20, 0.1, LABEL_ID["figure"]]  # dưới ngưỡng tin cậy → bỏ
    det = LayoutDetector(fake_model(tmp_path / "m.onnx", rows, fixed))
    found = det.detect(page)
    assert len(found) == 1 and found[0].label == "abandon"
    assert all(abs(a - b) < 0.5 for a, b in zip(found[0].rect, target))


def test_detector_yolov8_output_with_nms(tmp_path):
    pdf = tmp_path / "s.pdf"
    make(str(pdf))
    page = pymupdf.open(str(pdf))[0]
    out = np.zeros((1, 4 + len(LABELS), 3), np.float32)
    # 2 khung gần trùng nhau (NMS giữ 1) + 1 khung khác
    for j, (cx, cy, w, h, cls, sc) in enumerate([(200, 200, 100, 50, 5, 0.9), (202, 201, 100, 50, 5, 0.8),
                                                  (600, 600, 80, 40, 8, 0.7)]):
        out[0, :4, j] = [cx, cy, w, h]
        out[0, 4 + cls, j] = sc
    found = LayoutDetector(fake_model(tmp_path / "m8.onnx", out)).detect(page)
    assert sorted(d.label for d in found) == ["formula", "table"]


def test_analyze_pdf_uses_layout_model(tmp_path, monkeypatch):
    from app import layout
    from app.config import settings
    from app.pdf_processor import TranslateOptions, analyze_pdf

    pdf = tmp_path / "s.pdf"
    make(str(pdf))
    page = pymupdf.open(str(pdf))[0]
    zoom = 1024 / max(page.rect.width, page.rect.height)
    rows = np.zeros((1, 300, 6), np.float32)
    rows[0, 0] = [*(np.array([253, 102, 342, 117]) * zoom), 0.9, LABEL_ID["abandon"]]
    monkeypatch.setattr(settings, "layout", "auto")
    monkeypatch.setattr(settings, "layout_model", fake_model(tmp_path / "m.onnx", rows))
    layout.reset()
    try:
        ir = analyze_pdf(pdf, TranslateOptions())
        uni = next(r for r in ir.regions if r.plain.startswith("University of Science"))
        assert uni.kind == "abandon"
        assert layout.status()[0]
        # Tắt bằng tùy chọn → giữ phân loại bằng luật
        ir2 = analyze_pdf(pdf, TranslateOptions(layout_model=False))
        assert next(r for r in ir2.regions if r.plain.startswith("University")).kind == "text"
    finally:
        layout.reset()
