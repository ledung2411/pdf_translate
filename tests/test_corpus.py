"""Kiểm thử hồi quy trên bộ PDF mô phỏng nhiều nhà xuất bản (+ PDF riêng trong tests/corpus/local/).

PDF thật của bạn (không đưa lên git) đặt vào ``tests/corpus/local/``; tùy chọn kèm
``<tên>.expect.json`` cùng định dạng ``expect`` trong ``tests/corpus/docs.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf
import pytest

from app.ir import Document
from app.pdf_processor import TranslateOptions, analyze_pdf, translate_pdf

from .corpus.docs import ALL

LOCAL = Path(__file__).parent / "corpus" / "local"


def _find(ir: Document, prefix: str):
    norm = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731
    return [r for r in ir.regions if norm(r.plain).startswith(norm(prefix))]


def check_expectations(ir: Document, expect: dict) -> None:
    for prefix in expect.get("text", []):
        found = _find(ir, prefix)
        assert found, f"không thấy vùng bắt đầu bằng {prefix!r}"
        assert found[0].kind == "text", f"{prefix!r} bị phân loại {found[0].kind}"
    for prefix, kind in expect.get("kinds", {}).items():
        found = _find(ir, prefix)
        assert found and found[0].kind == kind, f"{prefix!r}: {[r.kind for r in found]} != {kind}"
    for prefix in expect.get("not_text", []):
        assert all(r.kind != "text" for r in _find(ir, prefix)), f"{prefix!r} không được dịch"
    for a, b in expect.get("no_mix", []):
        for r in ir.regions:
            assert not (a in r.plain and b in r.plain), f"trộn cột: {a!r} và {b!r} chung một vùng"
    for s in expect.get("contains", []):
        assert any(s in r.plain for r in ir.regions), f"thiếu {s!r}"
    if "pages" in expect:
        assert len(ir.pages) == expect["pages"]


@pytest.mark.parametrize("name", sorted(ALL))
def test_synthetic_publishers(name, tmp_path):
    pdf = tmp_path / f"{name}.pdf"
    expect = ALL[name](str(pdf))
    ir = analyze_pdf(pdf, TranslateOptions())
    check_expectations(ir, expect)

    # Toàn quy trình với engine giả lập: không lỗi, tiêu đề được dịch đúng.
    out = translate_pdf(pdf, tmp_path / "out", TranslateOptions(engine="mock", make_bilingual=False))
    text = "\n".join(p.get_text() for p in pymupdf.open(str(out["translated"])))
    text = re.sub(r"\s+", " ", text)
    for vi in expect.get("headings", {}).values():
        assert vi in text, f"thiếu tiêu đề đã dịch {vi!r}"


def _local_pdfs() -> list[Path]:
    return sorted(LOCAL.glob("*.pdf")) if LOCAL.is_dir() else []


@pytest.mark.parametrize("pdf", _local_pdfs(), ids=lambda p: p.name)
def test_local_pdfs(pdf):
    ir = analyze_pdf(pdf, TranslateOptions())
    chars = {k: 0 for k in ("text", "other")}
    for r in ir.regions:
        chars["text" if r.kind == "text" else "other"] += len(r.plain)
        # Ký hiệu đánh dấu nội bộ (công thức/chỉ số trên) không được lọt ra văn bản thuần
        assert not re.search("[\\ue000-\\ue003]", r.plain), f"ký hiệu nội bộ lọt ra trong {r.id}"
    # Phần lớn chữ (trừ tài liệu tham khảo) phải được nhận là đoạn văn cần dịch.
    refs = sum(len(r.plain) for r in ir.regions if r.kind == "reference")
    ratio = chars["text"] / max(chars["text"] + chars["other"] - refs, 1)
    assert ratio > 0.6, f"chỉ {ratio:.0%} chữ được nhận là đoạn văn ({ir.stats()})"
    expect_file = pdf.with_suffix(".expect.json")
    if expect_file.exists():
        check_expectations(ir, json.loads(expect_file.read_text(encoding="utf-8")))
