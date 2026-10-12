"""Đọc PDF, tách khối văn bản, dịch và ghi bản dịch vào đúng vị trí cũ.

Giữ nguyên hình ảnh, hình vẽ vector, công thức dạng khối và bảng số liệu; chỉ thay
phần chữ của các đoạn văn được dịch. Xuất 3 dạng: PDF đã dịch, PDF song ngữ
(bản gốc | bản dịch cạnh nhau) và văn bản Markdown.
"""

from __future__ import annotations

import html
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pymupdf

from .config import settings
from .ir import Document, Page, Region
from .pipeline import MATH_END, MATH_START, SUP_END, SUP_START, TextTranslator, strip_markers
from .scientific import (
    APPENDIX_RE,
    REFERENCES_RE,
    BlockStats,
    Glossary,
    classify_block,
    is_math_font,
    has_prose,
    is_mono_font,
    join_lines,
    load_builtin_glossary,
    looks_mathy,
    parse_glossary,
)
from .layout import apply_layout, get_detector
from .ocr import OcrError, find_tessdata, is_scanned, ocr_textpage
from .tm import get_tm
from .translators import TranslatorError, get_engine

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


class PdfError(RuntimeError):
    pass


@dataclass
class TranslateOptions:
    engine: str = "auto"
    page_from: int | None = None  # 1-based, bao gồm
    page_to: int | None = None
    skip_references: bool = True
    use_glossary: bool = True
    custom_glossary: str = ""
    make_bilingual: bool = True
    post_edit: bool = True  # dùng LLM cục bộ (nếu đang chạy) sửa các câu điểm thấp
    highlight_doubtful: bool = True  # tô vàng đoạn nghi ngờ trong PDF
    use_tm: bool = True  # dùng bản dịch đã duyệt trong bộ nhớ dịch (TM)
    ocr: bool = True  # tự OCR các trang ảnh scan
    layout_model: bool = True  # dùng mô hình nhận diện bố cục nếu có


# Khối văn bản = vùng trong DocIR (giữ tên cũ cho dễ đọc).
TextBlock = Region


# ---------------------------------------------------------------------------
# Trích xuất
# ---------------------------------------------------------------------------

_SERIF_NAME_RE = re.compile(r"Times|Roman|Serif|CMR|SFRM|Nimbus(?!Sans)|Georgia|Garamond|Palatino|Cambria|LMRoman|Minion|STIX|Charter|Utopia|Bookman|Computer Modern", re.I)
_SANS_NAME_RE = re.compile(r"Sans|Arial|Helvetica|Calibri|Verdana|Segoe|Roboto|Inter|Lato|Open|Source|Fira|Ubuntu|Tahoma", re.I)


def _is_serif(span: dict) -> bool:
    name = span.get("font", "")
    if _SANS_NAME_RE.search(name):
        return False
    if _SERIF_NAME_RE.search(name):
        return True
    return bool(span.get("flags", 0) & 4)


def _line_bbox(spans: list[dict]) -> tuple[float, float, float, float]:
    return (
        min(sp["bbox"][0] for sp in spans), min(sp["bbox"][1] for sp in spans),
        max(sp["bbox"][2] for sp in spans), max(sp["bbox"][3] for sp in spans),
    )


def _split_line_at_gaps(line: dict) -> list[dict]:
    """Tách một "dòng" MuPDF chứa chữ của 2 cột (khoảng trống lớn ở giữa)."""
    spans = [sp for sp in line.get("spans", []) if sp.get("text", "")]
    if len(spans) < 2:
        return [line]
    parts: list[list[dict]] = [[]]
    prev = None
    for sp in spans:
        size = sp.get("size", 10)
        width = sp["bbox"][2] - sp["bbox"][0]
        if not sp["text"].strip() and width > max(size * 1.5, 8):
            # Span "dấu cách" rộng bất thường = khe giữa hai cột (OCR, một số trình xuất PDF)
            if parts[-1]:
                parts.append([])
            prev = None
            continue
        if prev is not None and sp["bbox"][0] - prev["bbox"][2] > max(prev.get("size", 10) * 2.5, 12):
            parts.append([])
        parts[-1].append(sp)
        prev = sp
    parts = [p for p in parts if any(x["text"].strip() for x in p)]
    if len(parts) <= 1:
        return [line]
    return [{**line, "spans": p, "bbox": _line_bbox(p)} for p in parts]


def _find_gutter(lines: list[dict]) -> tuple[float, float] | None:
    """Khe trống dọc giữa hai cột trong một khối, suy ra từ các dòng hẹp (đã thuộc một cột)."""
    x0 = min(ln["bbox"][0] for ln in lines)
    x1 = max(ln["bbox"][2] for ln in lines)
    width = x1 - x0
    if width < 100:
        return None
    narrow = [ln["bbox"] for ln in lines if ln["bbox"][2] - ln["bbox"][0] < 0.6 * width]
    if len(narrow) < 4:
        return None
    cover = [0] * (int(width) + 1)
    for bx0, _, bx1, _ in narrow:
        for x in range(int(bx0 - x0), int(bx1 - x0) + 1):
            cover[x] += 1
    best: tuple[int, int] | None = None
    start = None
    lo, hi = int(0.3 * width), int(0.7 * width)
    for x in range(lo, hi + 1):
        if cover[x] == 0 and start is None:
            start = x
        if (cover[x] != 0 or x == hi) and start is not None:
            if best is None or x - start > best[1] - best[0]:
                best = (start, x)
            start = None
    if best is None or best[1] - best[0] < 5:
        return None
    # Mỗi bên khe phải có đủ dòng → đúng là 2 cột chứ không phải chỗ trống ngẫu nhiên
    g0, g1 = x0 + best[0], x0 + best[1]
    if g1 - g0 > 0.2 * width:
        return None
    # Mỗi bên khe phải là các DÒNG CỘT thật (đủ dài), không phải vài từ lẻ của dòng căn đều.
    left = [b for b in narrow if b[2] <= g0 + 1 and b[2] - b[0] > 0.3 * (g0 - x0)]
    right = [b for b in narrow if b[0] >= g1 - 1 and b[2] - b[0] > 0.3 * (x1 - g1)]
    return (g0, g1) if len(left) >= 3 and len(right) >= 3 else None


def _split_at_gutter(lines: list[dict]) -> list[dict]:
    """Cắt các dòng chạy ngang qua khe giữa hai cột (OCR hay gộp dòng của 2 cột)."""
    gutter = _find_gutter(lines)
    if gutter is None:
        return lines
    mid = (gutter[0] + gutter[1]) / 2
    out: list[dict] = []
    for ln in lines:
        bx0, _, bx1, _ = ln["bbox"]
        if not (bx0 < gutter[0] - 1 and bx1 > gutter[1] + 1):
            out.append(ln)
            continue
        left = [sp for sp in ln["spans"] if (sp["bbox"][0] + sp["bbox"][2]) / 2 < mid]
        right = [sp for sp in ln["spans"] if (sp["bbox"][0] + sp["bbox"][2]) / 2 >= mid]
        for part in (left, right):
            part = [sp for sp in part if sp["text"].strip()] and part
            if part and any(sp["text"].strip() for sp in part):
                out.append({**ln, "spans": part, "bbox": _line_bbox(part)})
    return out


def _x_overlap(a, b) -> float:
    return min(a[2], b[2]) - max(a[0], b[0])


def _segment_blocks(raw_blocks: list[dict], page_rect: pymupdf.Rect) -> list[dict]:
    """Tách các khối MuPDF bị gộp nhầm nhiều cột.

    Với bài báo 2 cột có khoảng giữa hẹp (Wiley, Elsevier...), MuPDF đôi khi gộp
    cả 2 cột vào một khối với các dòng xen kẽ trái/phải. Ở đây nhóm lại các dòng
    theo cột (chồng lấn theo chiều ngang) rồi cắt tiếp theo khoảng trống dọc.
    Bảng thật (≥3 cột hoặc cột hẹp) vẫn giữ nguyên một khối.
    """
    out: list[dict] = []
    for b in raw_blocks:
        if b.get("type") != 0:
            continue
        lines = [pl for ln in b.get("lines", []) for pl in _split_line_at_gaps(ln)]
        if not lines:
            continue
        lines = _split_at_gutter(lines)
        # Gom dòng theo cột: hai dòng cùng cột nếu chồng lấn ngang đủ nhiều.
        cols: list[dict] = []
        for ln in sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0])):
            lb = ln["bbox"]
            for c in cols:
                cb = c["x"]
                if _x_overlap(lb, cb) > 0.3 * min(lb[2] - lb[0], cb[2] - cb[0]):
                    c["lines"].append(ln)
                    c["x"] = (min(cb[0], lb[0]), 0, max(cb[2], lb[2]), 0)
                    break
            else:
                cols.append({"x": (lb[0], 0, lb[2], 0), "lines": [ln]})
        if len(cols) == 1:
            out.append({**b, "lines": lines})
            continue
        # Mẩu chữ hẹp ở lề (watermark arXiv, số dòng, ký tự OCR lạc) tách riêng, không tính là cột.
        W = page_rect.width
        narrow = [
            c for c in cols
            if c["x"][2] - c["x"][0] < 0.12 * W and (c["x"][2] < 0.15 * W or c["x"][0] > 0.85 * W)
        ]
        if narrow and len(cols) - len(narrow) >= 1:
            for c in narrow:
                out.append({"type": 0, "bbox": _union([ln["bbox"] for ln in c["lines"]]), "lines": c["lines"]})
            cols = [c for c in cols if c not in narrow]
            dropped = {id(ln) for c in narrow for ln in c["lines"]}
            lines = [ln for ln in lines if id(ln) not in dropped]
            if len(cols) == 1:
                out.append({"type": 0, "bbox": _union([ln["bbox"] for ln in lines]), "lines": lines})
                continue
        ys = [ln["bbox"][1] for ln in lines]
        size = max((sp.get("size", 10) for ln in lines for sp in ln["spans"]), default=10)
        def _wordy(c: dict) -> bool:
            txt = " ".join(sp["text"] for ln in c["lines"] for sp in ln["spans"])
            return len(re.findall(r"[A-Za-z]{2,}", txt)) >= 1 and not re.search(r"\d+\.\d+", txt)

        if max(ys) - min(ys) < size * 0.8 and len(cols) == 2 and all(_wordy(c) for c in cols):
            # Chỉ một hàng (vd. hai tiêu đề mục ở hai cột cùng độ cao) → không phải bảng
            for c in sorted(cols, key=lambda c: c["x"][0]):
                out.append({"type": 0, "bbox": _union([ln["bbox"] for ln in c["lines"]]), "lines": c["lines"]})
            continue
        widths = [c["x"][2] - c["x"][0] for c in cols]
        if len(cols) >= 3 or min(widths) < 0.25 * page_rect.width:
            # Cột chữ rộng nhiều dòng (vd. cột trái) nằm chung khối với bảng → tách riêng.
            prose = [
                c for c in cols
                if c["x"][2] - c["x"][0] >= 0.3 * W and len(c["lines"]) >= 3 and _wordy(c)
            ]
            others = [c for c in cols if c not in prose]
            def _numeric(c: dict) -> bool:
                txt = "".join(sp["text"] for ln in c["lines"] for sp in ln["spans"]).replace(" ", "")
                return bool(txt) and sum(ch.isdigit() for ch in txt) / len(txt) > 0.3

            if not (prose and others and any(_numeric(c) for c in others)):
                prose = []  # không có số liệu bên cạnh → không phải "cột chữ + bảng", giữ nguyên
            for c in prose:
                out.append({"type": 0, "bbox": _union([ln["bbox"] for ln in c["lines"]]), "lines": c["lines"]})
            taken = {id(ln) for c in prose for ln in c["lines"]}
            rest = [ln for ln in lines if id(ln) not in taken]  # giữ thứ tự dòng gốc
            if rest:  # bảng: giữ nguyên để không dịch vỡ bố cục
                out.append({"type": 0, "bbox": _union([ln["bbox"] for ln in rest]), "lines": rest})
            continue
        for c in sorted(cols, key=lambda c: c["x"][0]):
            group: list[dict] = []
            for ln in c["lines"]:
                if group:
                    prev = group[-1]["bbox"]
                    h = max(prev[3] - prev[1], 1)
                    if ln["bbox"][1] - prev[3] > 0.8 * h:  # khoảng trống giữa đoạn
                        out.append({"type": 0, "bbox": _union([g["bbox"] for g in group]), "lines": group})
                        group = []
                group.append(ln)
            if group:
                out.append({"type": 0, "bbox": _union([g["bbox"] for g in group]), "lines": group})
    return [part for blk in out for part in _split_math_lines(blk)]


_EQ_NUMBER_RE = re.compile(r"\(\s*\d+(?:\.\d+)?[a-z]?\s*\)\s*$")


def _is_math_line(line: dict) -> bool:
    """Dòng công thức đứng riêng: phần lớn ký tự thuộc font toán, hoặc dạng "… = … (3)" không có câu chữ."""
    spans = [sp for sp in line.get("spans", []) if sp.get("text", "").strip()]
    text = "".join(sp["text"] for sp in spans).strip()
    if not text or has_prose(text):
        return False
    total = sum(len(sp["text"].strip()) for sp in spans)
    math = sum(len(sp["text"].strip()) for sp in spans if is_math_font(sp.get("font", "")) or looks_mathy(sp["text"]))
    if total and math / total > 0.6 and len(re.findall(r"[A-Za-z]{4,}", text)) <= 2:
        return True
    return bool(_EQ_NUMBER_RE.search(text) and re.search(r"[=<>≤≥≈]", text))


def _split_math_lines(block: dict) -> list[dict]:
    """Tách các dòng công thức khỏi đoạn văn bị gộp chung khối → công thức giữ nguyên bản gốc."""
    lines = block.get("lines", [])
    if len(lines) < 2:
        return [block]
    flags = [_is_math_line(ln) for ln in lines]
    if not any(flags) or all(flags):
        return [block]
    groups: list[tuple[bool, list[dict]]] = []
    for ln, is_math in zip(lines, flags):
        if groups and groups[-1][0] == is_math:
            groups[-1][1].append(ln)
        else:
            groups.append((is_math, [ln]))
    return [{"type": 0, "bbox": _union([ln["bbox"] for ln in g]), "lines": g} for _, g in groups]


def _union(boxes: list) -> tuple[float, float, float, float]:
    return (
        min(x[0] for x in boxes), min(x[1] for x in boxes),
        max(x[2] for x in boxes), max(x[3] for x in boxes),
    )


def extract_blocks(
    page: pymupdf.Page, page_index: int, textpage: pymupdf.TextPage | None = None
) -> list[TextBlock]:
    flags = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP
    data = page.get_text("dict", flags=flags, textpage=textpage)
    blocks: list[TextBlock] = []
    for b in _segment_blocks(data.get("blocks", []), page.rect):
        lines_text: list[str] = []
        span_rects: list[pymupdf.Rect] = []
        total = math_chars = mono_chars = 0
        size_w: dict[float, int] = {}
        bold_w = italic_w = serif_w = 0
        color_w: dict[int, int] = {}
        rotated = False
        line_boxes: list[pymupdf.Rect] = []
        table_lines = 0
        math_pieces: list[str] = []

        for line in b.get("lines", []):
            dx, dy = line.get("dir", (1, 0))
            if abs(dy) > 0.05 or dx < 0:
                rotated = True
            parts: list[str] = []
            if _has_column_gap(line):
                table_lines += 1
            for sp in line.get("spans", []):
                t = sp.get("text", "")
                if not t:
                    continue
                # Wiley chèn dấu cách ("dummy-space") sau gạch nối: "Weakly- Supervised".
                if not t.strip() and parts and parts[-1].rstrip(MATH_END + SUP_END).endswith("-"):
                    continue
                n = len(t.strip())
                font = sp.get("font", "")
                rect = pymupdf.Rect(sp["bbox"])
                if t.strip():
                    span_rects.append(rect)
                total += n
                is_math = (is_math_font(font) and not has_prose(t)) or looks_mathy(t)
                is_sup = bool(sp.get("flags", 0) & 1) and t.strip() and not is_math
                if is_math:
                    math_chars += n
                    parts.append(f"{MATH_START}{t}{MATH_END}")
                elif is_sup:
                    parts.append(f"{SUP_START}{t.strip()}{SUP_END}")
                else:
                    parts.append(t)
                    # Chữ OCR (GlyphLessFont) luôn mang cờ monospace — không phải mã nguồn.
                    if font != "GlyphLessFont" and (is_mono_font(font) or sp.get("flags", 0) & 8):
                        mono_chars += n
                    size = round(sp.get("size", 10), 1)
                    size_w[size] = size_w.get(size, 0) + n
                    f = sp.get("flags", 0)
                    if f & 16 or re.search(r"Bold|Black|Heavy|Semibold|CMBX", font, re.I):
                        bold_w += n
                    if f & 2 or re.search(r"Italic|Oblique|CMTI", font, re.I):
                        italic_w += n
                    if _is_serif(sp):
                        serif_w += n
                    c = sp.get("color", 0)
                    color_w[c] = color_w.get(c, 0) + n
            line_text = "".join(parts)
            # Gộp các span công thức liền nhau.
            line_text = re.sub(f"{MATH_END}(\\s*){MATH_START}", r"\1", line_text)
            if line_text.strip():
                lines_text.append(line_text)
                line_boxes.append(pymupdf.Rect(line["bbox"]))
                math_pieces.extend(m.strip() for m in re.findall(f"{MATH_START}(.*?){MATH_END}", line_text) if m.strip())

        if not lines_text:
            continue
        text = join_lines(lines_text)
        normal_chars = max(sum(size_w.values()), 1)
        font_size = max(size_w, key=size_w.get) if size_w else 10.0
        stats = BlockStats(
            text=strip_markers(re.sub(f"{MATH_START}.*?{MATH_END}", " ", text)),
            total_chars=total,
            math_chars=math_chars,
            mono_chars=mono_chars,
        )
        if rotated:
            kind = "rotated"
        elif _is_table_like(line_boxes, table_lines, font_size):
            kind = "table"
        elif all(_is_math_line(ln) for ln in b.get("lines", []) if "".join(sp.get("text", "") for sp in ln.get("spans", [])).strip()):
            kind = "math"  # dòng công thức đứng riêng (kể cả chữ OCR không có font toán)
        elif math_chars and not stats.text.strip():
            kind = "math"
        else:
            kind = classify_block(stats)
        blocks.append(
            TextBlock(
                page_index=page_index,
                bbox=pymupdf.Rect(b["bbox"]),
                text=text,
                kind=kind,
                font_size=font_size,
                bold=bold_w / normal_chars > 0.6,
                italic=italic_w / normal_chars > 0.6,
                serif=serif_w / normal_chars >= 0.5,
                color=max(color_w, key=color_w.get) if color_w else 0,
                align=_detect_align(line_boxes, pymupdf.Rect(b["bbox"])),
                span_rects=span_rects,
                line_rects=line_boxes,
                math_spans=list(dict.fromkeys(math_pieces)),
            )
        )
    return merge_blocks(blocks)


def _is_table_like(lines: list[pymupdf.Rect], gap_lines: int, size: float) -> bool:
    if gap_lines and gap_lines >= len(lines) / 2:
        return True
    # Nhiều "dòng" nằm cùng hàng ngang, cách xa nhau → các ô của bảng.
    rows: dict[int, list[pymupdf.Rect]] = {}
    for r in lines:
        rows.setdefault(round((r.y0 + r.y1) / 2 / max(size * 0.5, 1)), []).append(r)
    multi = 0
    for row in rows.values():
        row.sort(key=lambda r: r.x0)
        if any(b.x0 - a.x1 > size * 1.2 for a, b in zip(row, row[1:])):
            multi += 1
    return multi > 0 and multi >= len(rows) / 2


_TERMINAL_RE = re.compile(r"[.!?:;]\W*$")


def _can_merge(a: TextBlock, b: TextBlock) -> bool:
    if a.kind != "text" or b.kind != "text" or a.page_index != b.page_index:
        return False
    if abs(a.font_size - b.font_size) > 0.3 or a.bold != b.bold:
        return False
    gap = b.bbox.y0 - a.bbox.y1
    if gap < -a.font_size * 0.5 or gap > a.font_size * 0.6:
        return False
    overlap = min(a.bbox.x1, b.bbox.x1) - max(a.bbox.x0, b.bbox.x0)
    if overlap < 0.5 * min(a.bbox.width, b.bbox.width):
        return False
    first = b.plain.lstrip()[:1]
    # Nối khi khối sau bắt đầu bằng chữ thường, hoặc khối trước (nhiều dòng) chưa hết câu.
    return first.islower() or (len(a.line_rects) >= 2 and not _TERMINAL_RE.search(a.plain))


def merge_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    """Gộp các khối bị PDF tách rời giữa chừng một câu (vd. chú thích hình nhiều dòng)."""
    out: list[TextBlock] = []
    for b in blocks:
        prev = out[-1] if out else None
        if prev is not None and _can_merge(prev, b):
            prev.text = join_lines([prev.text, b.text])
            prev.bbox = prev.bbox | b.bbox
            prev.span_rects += b.span_rects
            prev.line_rects += b.line_rects
            prev.align = _detect_align(prev.line_rects, prev.bbox)
            continue
        out.append(b)
    return out


def _has_column_gap(line: dict) -> bool:
    """Dòng có khoảng trống lớn giữa các span → khả năng là hàng của bảng."""
    spans = [sp for sp in line.get("spans", []) if sp.get("text", "").strip()]
    for a, b in zip(spans, spans[1:]):
        gap = b["bbox"][0] - a["bbox"][2]
        if gap > max(a.get("size", 10), 6) * 1.6:
            return True
    return False


def _detect_align(lines: list[pymupdf.Rect], bbox: pymupdf.Rect) -> str:
    if len(lines) < 2:
        return "left"
    cx = (bbox.x0 + bbox.x1) / 2
    lefts_vary = max(r.x0 for r in lines) - min(r.x0 for r in lines) > 3
    centered = all(abs((r.x0 + r.x1) / 2 - cx) < 3 for r in lines)
    if centered and lefts_vary:
        return "center"
    return "justify"


_CAPTION_RE = re.compile(r"^\s*(Figure|Fig\.|Table|Algorithm|Listing)\s*[A-Z]?\d", re.I)


def mark_references(blocks: list[TextBlock]) -> None:
    """Đánh dấu các khối thuộc mục Tài liệu tham khảo (giữ nguyên, không dịch).

    Trên trang chứa tiêu đề "References", chỉ tính các khối nằm dưới tiêu đề
    (cùng cột) hoặc ở cột bên phải; các trang sau tính toàn bộ cho tới Phụ lục.
    """
    heading: TextBlock | None = None
    for b in blocks:
        plain = b.plain.strip()
        if REFERENCES_RE.match(plain):
            heading = b
            continue  # tiêu đề "References" vẫn được dịch
        if heading is None:
            continue
        if APPENDIX_RE.match(plain) and len(plain) < 80:
            heading = None
            continue
        if b.kind != "text" or _CAPTION_RE.match(plain):
            continue
        if b.page_index == heading.page_index:
            h = heading.bbox
            same_col = b.bbox.x0 < h.x1 and b.bbox.x1 > h.x0 and b.bbox.y0 >= h.y0 - 2
            right_col = b.bbox.x0 >= h.x1 - 5
            if not (same_col or right_col):
                continue
        b.kind = "reference"


# ---------------------------------------------------------------------------
# Font tiếng Việt
# ---------------------------------------------------------------------------

_FONT_DIRS = [
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    str(Path.home() / ".fonts"),
    str(Path.home() / ".local/share/fonts"),
    "C:/Windows/Fonts",
    "/System/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
    "/Library/Fonts",
]
_STYLES = ("regular", "bold", "italic", "bolditalic")
_SERIF_CANDIDATES = [
    ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf", "LiberationSerif-Italic.ttf", "LiberationSerif-BoldItalic.ttf"),
    ("NotoSerif-Regular.ttf", "NotoSerif-Bold.ttf", "NotoSerif-Italic.ttf", "NotoSerif-BoldItalic.ttf"),
    ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"),
    ("Times New Roman.ttf", "Times New Roman Bold.ttf", "Times New Roman Italic.ttf", "Times New Roman Bold Italic.ttf"),
    ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf", "DejaVuSerif-Italic.ttf", "DejaVuSerif-BoldItalic.ttf"),
    ("FreeSerif.ttf", "FreeSerifBold.ttf", "FreeSerifItalic.ttf", "FreeSerifBoldItalic.ttf"),
]
_SANS_CANDIDATES = [
    ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf", "LiberationSans-BoldItalic.ttf"),
    ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf", "NotoSans-Italic.ttf", "NotoSans-BoldItalic.ttf"),
    ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"),
    ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf"),
]
_VI_TEST = "ạếữởđĐỹỷẫặƯơ"


def _supports_vietnamese(path: str) -> bool:
    try:
        f = pymupdf.Font(fontfile=path)
        return all(f.has_glyph(ord(c)) for c in _VI_TEST)
    except Exception:  # noqa: BLE001
        return False


def _index_font_files() -> dict[str, str]:
    found: dict[str, str] = {}
    for d in _FONT_DIRS:
        p = Path(d)
        if not p.is_dir():
            continue
        try:
            for f in p.rglob("*.tt[fc]"):
                found.setdefault(f.name.lower(), str(f))
        except OSError:
            continue
    return found


class FontKit:
    """Tạo CSS @font-face + Archive cho insert_htmlbox với font có dấu tiếng Việt."""

    def __init__(self) -> None:
        self.archive = pymupdf.Archive()
        css: list[str] = []
        index = _index_font_files()
        self.families = {}
        for fam, override, candidates, fallback in (
            ("vnserif", settings.font_serif, _SERIF_CANDIDATES, "notos"),
            ("vnsans", settings.font_sans, _SANS_CANDIDATES, "notos"),
        ):
            files = self._pick(override, candidates, index)
            if files:
                for style, path in files.items():
                    name = f"{fam}-{style}{Path(path).suffix.lower()}"
                    self.archive.add((Path(path).read_bytes(), name))
                    weight = "bold" if "bold" in style else "normal"
                    fstyle = "italic" if "italic" in style else "normal"
                    css.append(
                        f"@font-face{{font-family:{fam};src:url({name});"
                        f"font-weight:{weight};font-style:{fstyle};}}"
                    )
                self.families[fam] = Path(files["regular"]).name
            else:
                # Dùng font Noto Sans đóng gói sẵn trong pymupdf-fonts.
                css.append(pymupdf.css_for_pymupdf_font(fallback, archive=self.archive, name=fam))
                self.families[fam] = "pymupdf-fonts:" + fallback
        self.css = "\n".join(css)
        log.info("Font sử dụng: %s", self.families)

    @staticmethod
    def _pick(override: str, candidates, index: dict[str, str]) -> dict[str, str] | None:
        if override:
            if Path(override).is_file() and _supports_vietnamese(override):
                return {"regular": override}
            log.warning("Font %s không dùng được, bỏ qua", override)
        for names in candidates:
            reg = index.get(names[0].lower())
            if not reg or not _supports_vietnamese(reg):
                continue
            files = {"regular": reg}
            for style, n in zip(_STYLES[1:], names[1:]):
                p = index.get(n.lower())
                if p:
                    files[style] = p
            return files
        return None


_fontkit: FontKit | None = None


def get_fontkit() -> FontKit:
    global _fontkit
    if _fontkit is None:
        _fontkit = FontKit()
    return _fontkit


# ---------------------------------------------------------------------------
# Ghi bản dịch
# ---------------------------------------------------------------------------


def _to_html(text: str, math_spans: list[str] | None = None) -> str:
    """Bản dịch → HTML cho insert_htmlbox; công thức nội dòng in nghiêng kiểu LaTeX."""
    text = text.replace(MATH_START, "").replace(MATH_END, "")
    marks: list[tuple[int, int]] = []
    for m in sorted(set(math_spans or []), key=len, reverse=True):
        if len(m) == 1 and not (m.isalpha() or m in "αβγδεθλμσφψω"):
            continue  # ký hiệu đơn lẻ (=, +) không cần đổi kiểu chữ
        pat = re.escape(m)
        if m[:1].isalnum():
            pat = r"(?<![\w])" + pat
        if m[-1:].isalnum():
            pat = pat + r"(?![\w])"
        for hit in re.finditer(pat, text):
            if all(hit.end() <= a or hit.start() >= b for a, b in marks):
                marks.append((hit.start(), hit.end()))
    out, pos = [], 0
    for a, b in sorted(marks):
        out.append(html.escape(text[pos:a], quote=False))
        out.append(f'<span class="m">{html.escape(text[a:b], quote=False)}</span>')
        pos = b
    out.append(html.escape(text[pos:], quote=False))
    s = "".join(out)
    return s.replace(SUP_START, "<sup>").replace(SUP_END, "</sup>")


def _color_hex(c: int) -> str:
    return f"#{c & 0xFFFFFF:06x}"


def _column_bounds(b: TextBlock, blocks: list[TextBlock], page_rect: pymupdf.Rect) -> tuple[float, float]:
    """Ước lượng lề trái/phải của cột chứa khối ``b`` từ các đoạn văn nhiều dòng."""
    paras = [o for o in blocks if len(o.line_rects) >= 2 and o.kind in ("text", "reference")]
    same = [
        o for o in paras
        if min(o.bbox.x1, b.bbox.x1) - max(o.bbox.x0, b.bbox.x0) > 0.5 * b.bbox.width
    ]
    if same:
        # Ưu tiên cột hẹp nhất (tránh lấy nhầm khối Abstract trải rộng 2 cột).
        narrow = min(o.bbox.width for o in same)
        same = [o for o in same if o.bbox.width <= narrow * 1.3]
        return min(o.bbox.x0 for o in same), max(o.bbox.x1 for o in same)
    if paras:
        return min(o.bbox.x0 for o in paras), max(o.bbox.x1 for o in paras)
    return page_rect.x0 + 36, page_rect.x1 - 36


def _is_centered_in_column(b: TextBlock, blocks: list[TextBlock], page_rect: pymupdf.Rect) -> bool:
    left, right = _column_bounds(b, blocks, page_rect)
    if b.bbox.x0 - left < 10:
        return False
    return abs((b.bbox.x0 + b.bbox.x1) / 2 - (left + right) / 2) < 0.04 * (right - left)


def _target_rect(b: TextBlock, blocks: list[TextBlock], page_rect: pymupdf.Rect) -> pymupdf.Rect:
    """Khung để ghi bản dịch. Khối 1 dòng (tiêu đề, tên tác giả...) được nới ngang
    tới mép cột để bản dịch tiếng Việt (thường dài hơn) không bị xuống dòng/thu nhỏ."""
    rect = pymupdf.Rect(b.bbox)
    if len(b.line_rects) > 1:
        return rect
    left, right = _column_bounds(b, blocks, page_rect)
    if _is_centered_in_column(b, blocks, page_rect):
        rect.x0 = min(rect.x0, left)
    rect.x1 = max(rect.x1, right)
    # Không lấn sang khối khác nằm cùng hàng.
    for o in blocks:
        if o is b or o.bbox.y1 <= rect.y0 + 1 or o.bbox.y0 >= rect.y1 - 1:
            continue
        if o.bbox.x0 >= b.bbox.x1 - 1:
            rect.x1 = min(rect.x1, o.bbox.x0 - 2)
        if o.bbox.x1 <= b.bbox.x0 + 1:
            rect.x0 = max(rect.x0, o.bbox.x1 + 2)
    rect.x0 = min(rect.x0, b.bbox.x0)
    rect.x1 = max(rect.x1, b.bbox.x1)
    return rect


def render_page(page: pymupdf.Page, blocks: list[TextBlock], fontkit: FontKit, scanned: bool = False) -> None:
    todo = [b for b in blocks if b.translation is not None]
    if not todo:
        return
    if scanned:
        # Trang scan: chữ tiếng Anh nằm trong ảnh → tô trắng vùng ảnh dưới mỗi đoạn được dịch.
        for b in todo:
            for r in b.line_rects or [b.bbox]:
                page.add_redact_annot(pymupdf.Rect(r) + (-1, -1, 1, 1), fill=(1, 1, 1), cross_out=False)
        page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_PIXELS,
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            text=pymupdf.PDF_REDACT_TEXT_REMOVE,
        )
    else:
        # 1) Xóa chữ gốc (chỉ chữ, giữ hình ảnh và nét vẽ).
        for b in todo:
            for r in b.span_rects:
                h = r.height
                shrink = pymupdf.Rect(r.x0, r.y0 + h * 0.2, r.x1, r.y1 - h * 0.2)
                if shrink.is_empty:
                    shrink = r
                page.add_redact_annot(shrink, fill=False, cross_out=False)
        page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_NONE,
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            text=pymupdf.PDF_REDACT_TEXT_REMOVE,
        )
    # 2) Chèn bản dịch, tự thu nhỏ cỡ chữ nếu bản dịch dài hơn khung.
    for b in todo:
        fam = "vnserif" if b.serif else "vnsans"
        align = b.align
        if len(b.line_rects) <= 1 and _is_centered_in_column(b, blocks, page.rect):
            align = "center"
        style = (
            f"font-family:{fam};font-size:{b.font_size:.2f}px;line-height:1.18;"
            f"color:{_color_hex(b.color)};text-align:{align};"
            f"font-weight:{'bold' if b.bold else 'normal'};"
            f"font-style:{'italic' if b.italic else 'normal'};"
        )
        css = fontkit.css + f"\nbody{{margin:0;padding:0;}} .b{{{style}}} sup{{font-size:70%;}} .m{{font-style:italic;}}"
        rect = _target_rect(b, blocks, page.rect)
        # Nới nhẹ khung để chữ có dấu tiếng Việt không bị cắt.
        rect.y1 += b.font_size * 0.15
        try:
            page.insert_htmlbox(
                rect, f'<div class="b">{_to_html(b.translation, b.math_spans)}</div>', css=css,
                archive=fontkit.archive, scale_low=0,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Không ghi được khối trên trang %d: %s", b.page_index + 1, e)


def is_doubtful(r: TextBlock) -> bool:
    """Đoạn nên được người đọc kiểm tra lại (điểm thấp / phải giữ tiếng Anh), trừ khi đã sửa tay."""
    return (
        r.translation is not None
        and "edited" not in r.flags
        and ("low_quality" in r.flags or "kept_english" in r.flags)
    )


def annotate_doubtful(page: pymupdf.Page, regions: list[TextBlock], dx: float = 0.0) -> int:
    """Tô vàng (chú thích PDF, có thể ẩn khi in) các đoạn nghi ngờ; ``dx`` để dời sang nửa phải bản song ngữ."""
    n = 0
    for r in regions:
        if not is_doubtful(r):
            continue
        rect = pymupdf.Rect(r.bbox) + (dx, 0, dx, 0)
        annot = page.add_rect_annot(rect)
        annot.set_colors(stroke=(1, 0.75, 0), fill=(1, 0.92, 0.25))
        annot.set_border(width=0.5)
        annot.set_opacity(0.28)
        reason = "; ".join(r.notes) if r.notes else "bản dịch có thể chưa chính xác"
        if "kept_english" in r.flags:
            reason = "Có câu giữ nguyên tiếng Anh vì bản dịch máy bị hỏng. " + reason
        score = f" (điểm {r.score:.2f})" if r.score is not None else ""
        annot.set_info(title="Cần kiểm tra lại", content=reason + score)
        annot.update()
        n += 1
    return n


def make_bilingual(
    original: pymupdf.Document, translated: pymupdf.Document,
    doubtful: list[list[TextBlock]] | None = None,
) -> pymupdf.Document:
    out = pymupdf.open()
    for i in range(len(translated)):
        r = original[i].rect
        page = out.new_page(width=r.width * 2, height=r.height)
        page.show_pdf_page(pymupdf.Rect(0, 0, r.width, r.height), original, i)
        page.show_pdf_page(pymupdf.Rect(r.width, 0, r.width * 2, r.height), translated, i)
        if doubtful is not None and i < len(doubtful):
            annotate_doubtful(page, doubtful[i], dx=r.width)
    return out


def blocks_to_markdown(pages: list[list[TextBlock]], page_numbers: list[int]) -> str:
    out: list[str] = []
    for num, blocks in zip(page_numbers, pages):
        out.append(f"\n\n## Trang {num}\n")
        for b in blocks:
            if b.translation is not None:
                text = b.translation.replace(SUP_START, "^").replace(SUP_END, "")
                out.append(f"**{text}**" if b.bold and len(text) < 150 else text)
            elif b.kind in ("math", "code", "reference", "data", "table"):
                out.append(f"`{b.plain}`" if len(b.plain) < 200 else b.plain)
    return "\n\n".join(out).strip() + "\n"


# ---------------------------------------------------------------------------
# Hàm chính
# ---------------------------------------------------------------------------


def _fmt_duration(sec: float) -> str:
    sec = int(sec)
    if sec < 60:
        return f"{sec} giây"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m} phút {s} giây"
    h, m = divmod(m, 60)
    return f"{h} giờ {m} phút"


def build_glossary(opts: TranslateOptions) -> Glossary | None:
    entries: dict[str, str] = {}
    if opts.use_glossary:
        entries.update(load_builtin_glossary())
    if opts.custom_glossary.strip():
        entries.update(parse_glossary(opts.custom_glossary))
    return Glossary(entries) if entries else None


def _open_pdf(input_path: Path) -> pymupdf.Document:
    try:
        doc = pymupdf.open(str(input_path))
    except Exception as e:  # noqa: BLE001
        raise PdfError(f"Không mở được file PDF: {e}") from e
    if doc.needs_pass and not doc.authenticate(""):
        raise PdfError("PDF được bảo vệ bằng mật khẩu.")
    if not doc.is_pdf:
        raise PdfError("File không phải PDF.")
    return doc


def _select_pages(doc: pymupdf.Document, page_numbers: list[int]) -> None:
    if page_numbers != list(range(1, len(doc) + 1)):
        doc.select([n - 1 for n in page_numbers])


def analyze_pdf(input_path: Path, opts: TranslateOptions, progress: ProgressFn | None = None) -> Document:
    """Tầng phân tích: PDF → DocIR (các vùng đã phân loại, chưa dịch)."""
    doc = _open_pdf(input_path)
    n = len(doc)
    start = max(1, opts.page_from or 1)
    end = min(n, opts.page_to or n)
    if start > end:
        raise PdfError(f"Khoảng trang không hợp lệ (tài liệu có {n} trang).")
    if end - start + 1 > settings.max_pages:
        raise PdfError(f"Tối đa {settings.max_pages} trang mỗi lần dịch.")
    numbers = list(range(start, end + 1))
    _select_pages(doc, numbers)

    ir = Document(source=input_path.name)
    tessdata: str | None = None
    detector = get_detector() if opts.layout_model else None
    for i, num in enumerate(numbers):
        page = doc[i]
        scanned = opts.ocr and is_scanned(page)
        textpage = None
        if scanned:
            if tessdata is None:
                try:
                    tessdata = find_tessdata()
                except OcrError as e:
                    raise PdfError(f"Trang {num} là ảnh scan, cần OCR nhưng: {e}") from e
            if progress:
                progress(0.01, f"Đang nhận dạng chữ (OCR) trang {num}...")
            textpage = ocr_textpage(page, tessdata)
        regions = extract_blocks(page, i, textpage)
        if scanned:
            for r in regions:
                r.serif, r.color = True, 0  # OCR không biết font gốc: dùng chữ có chân, màu đen
                # Dải chữ hẹp và cao (watermark xoay, số dòng ở lề) → OCR ra rác, bỏ qua
                if r.bbox.width < 0.06 * page.rect.width and r.bbox.height > 3 * r.bbox.width:
                    r.kind = "rotated"
        if detector is not None:
            try:
                n_changed = apply_layout(regions, detector.detect(page))
                if n_changed:
                    log.info("Trang %d: mô hình bố cục sửa phân loại %d vùng", num, n_changed)
            except Exception as e:  # noqa: BLE001  (mô hình lỗi → giữ phân loại bằng luật)
                log.warning("Nhận diện bố cục trang %d lỗi: %s", num, e)
        ir.pages.append(
            Page(index=i, number=num, width=page.rect.width, height=page.rect.height,
                 regions=regions, scanned=scanned)
        )
    if sum(len(r.plain) for r in ir.regions) < 30 * len(ir.pages):
        raise PdfError(
            "PDF gần như không có lớp chữ (có thể là bản scan/ảnh) và OCR không nhận được chữ. "
            "Hãy thử bản scan rõ nét hơn."
            if opts.ocr else
            "PDF gần như không có lớp chữ (có thể là bản scan/ảnh). Hãy bật OCR rồi dịch lại."
        )
    if opts.skip_references:
        mark_references(ir.regions)
    ir.assign_ids()
    log.info("Thống kê khối: %s", ir.stats())
    return ir


def _post_editor(engine, opts: TranslateOptions):
    """LLM cục bộ để sửa câu điểm thấp — chỉ khi được bật, đang chạy, và engine chính không phải LLM."""
    if not opts.post_edit or engine.key == "llm":
        return None
    try:
        llm = get_engine("llm")
    except TranslatorError:
        return None
    ok, _ = llm.available()
    return llm if ok else None


def _back_translator():
    """Mô hình Việt→Anh để chấm điểm bằng dịch ngược (PDFTR_QE_BACKTRANS=1)."""
    if not settings.qe_backtrans:
        return None
    from .translators import HFSeq2SeqTranslator

    t = HFSeq2SeqTranslator("opus-vi-en", "OPUS-MT Vi→En", "", settings.backtrans_model)
    ok, why = t.available()
    if not ok:
        log.warning("Không dùng được dịch ngược để chấm điểm: %s", why)
        return None
    return t


def translate_document(ir: Document, opts: TranslateOptions, report: ProgressFn) -> None:
    """Tầng dịch: điền ``translation`` cho các vùng văn bản trong DocIR."""
    to_translate = [r for r in ir.regions if r.kind == "text"]
    if to_translate and opts.use_tm:
        hits = get_tm().lookup([r.plain for r in to_translate])
        for r in to_translate:
            if r.plain in hits:
                r.translation, r.score = hits[r.plain], 1.0
                r.flags.append("tm")
        if hits:
            log.info("Bộ nhớ dịch: dùng lại %d đoạn đã duyệt", sum(1 for r in to_translate if "tm" in r.flags))
            report(0.02, f"Dùng lại {len(hits)} đoạn đã duyệt từ bộ nhớ dịch")
        pending = [r for r in to_translate if "tm" not in r.flags]
        if not pending:
            ir.engine = opts.engine
            return
        to_translate = pending
    if not to_translate:
        raise PdfError(
            f"Không tìm thấy đoạn văn nào cần dịch (phân loại khối: {ir.stats()}). "
            "Hãy chạy: python -m app.inspect <file.pdf> và gửi kết quả để được hỗ trợ."
        )
    engine = get_engine(opts.engine)
    ok, why = engine.available()
    if not ok:
        raise TranslatorError(f"{engine.label}: {why}")
    engine.prepare(lambda msg: report(0.02, msg))
    glossary = build_glossary(opts)
    ir.engine = engine.key
    ir.glossary = dict(glossary.entries) if glossary is not None and opts.custom_glossary.strip() else {}
    translator = TextTranslator(
        engine, glossary,
        post_editor=_post_editor(engine, opts),
        back_translator=_back_translator(),
        status=lambda msg: report(0.82, msg),
    )
    report(0.03, f"Đang dịch {len(to_translate)} đoạn văn bằng {engine.label}...")
    t_start = time.time()

    def on_tr(done: int, total: int) -> None:
        frac = 0.03 + 0.77 * (done / total if total else 1)
        msg = f"Đang dịch: {done}/{total} câu"
        if 0 < done < total:
            remain = (time.time() - t_start) / done * (total - done)
            msg += f" — còn khoảng {_fmt_duration(remain)}"
        report(frac, msg)

    translations = translator.translate_texts([r.text for r in to_translate], on_tr)
    for i, (r, t) in enumerate(zip(to_translate, translations)):
        r.score = translator.scores[i]
        r.notes = translator.reasons[i]
        if translator.kept_english[i]:
            r.flags.append("kept_english")
        if translator.post_edited[i]:
            r.flags.append("post_edited")
        if r.score < settings.qe_threshold:
            r.flags.append("low_quality")
        if t and t.strip() != r.plain.strip():
            r.translation = t


def render_outputs(
    ir: Document, input_path: Path, out_dir: Path, make_bilingual_pdf: bool = True,
    report: ProgressFn | None = None, highlight: bool = True,
) -> dict[str, Path]:
    """Tầng dựng: DocIR + PDF gốc → PDF dịch, PDF song ngữ, Markdown."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = _open_pdf(input_path)
    _select_pages(doc, ir.page_numbers)
    original = pymupdf.open()
    original.insert_pdf(doc)

    fontkit = get_fontkit()
    for p in ir.pages:
        if report:
            report(0.88 + 0.09 * p.index / max(len(ir.pages), 1), f"Đang dựng trang {p.index + 1}/{len(ir.pages)}...")
        render_page(doc[p.index], p.regions, fontkit, p.scanned)
        if highlight:
            annotate_doubtful(doc[p.index], p.regions)

    if report:
        report(0.97, "Đang lưu kết quả...")
    stem = re.sub(r"[^\w\-]+", "_", Path(ir.source).stem)[:80] or "document"
    outputs: dict[str, Path] = {}
    path = out_dir / f"{stem}_vi.pdf"
    save_compact(doc, path)
    outputs["translated"] = path
    if make_bilingual_pdf:
        bi = make_bilingual(
            original, pymupdf.open(str(path)), [p.regions for p in ir.pages] if highlight else None
        )
        p2 = out_dir / f"{stem}_song_ngu.pdf"
        save_compact(bi, p2, subset=False)  # font trong bản dịch đã được cắt gọn
        outputs["bilingual"] = p2
    try:
        from .export_docx import build_docx

        outputs["docx"] = build_docx(
            ir, original, out_dir / f"{stem}_vi.docx", highlight=highlight, is_doubtful=is_doubtful
        )
    except ImportError:
        log.info("Chưa cài python-docx — bỏ qua xuất Word")
    p3 = out_dir / f"{stem}_vi.md"
    p3.write_text(blocks_to_markdown([p.regions for p in ir.pages], ir.page_numbers), encoding="utf-8")
    outputs["markdown"] = p3
    return outputs


IR_FILENAME = "document.json"


def save_compact(doc: pymupdf.Document, path: Path, subset: bool = True) -> None:
    """Lưu PDF gọn nhẹ.

    Mỗi lần ghi một đoạn dịch (insert_htmlbox), PyMuPDF nhúng lại TOÀN BỘ file font → bài
    19 trang có thể phình từ 7 MB lên ~90 MB. Cắt font chỉ giữ ký tự đã dùng (subset_fonts,
    cần fonttools) rồi gộp các đối tượng trùng nhau (garbage=4) đưa file về cỡ bản gốc.
    """
    if subset:
        try:
            doc.subset_fonts()
        except Exception as e:  # noqa: BLE001  (thiếu fonttools / font lạ → vẫn gộp trùng được)
            log.info("Không cắt gọn được font (%s) — chỉ gộp font trùng", e)
    doc.save(str(path), garbage=4, deflate=True, use_objstms=1)


def translate_pdf(
    input_path: Path,
    out_dir: Path,
    opts: TranslateOptions,
    progress: ProgressFn | None = None,
) -> dict[str, Path]:
    """Toàn bộ quy trình: phân tích → dịch → lưu DocIR → dựng kết quả."""

    def report(frac: float, msg: str) -> None:
        if progress:
            progress(max(0.0, min(1.0, frac)), msg)

    out_dir.mkdir(parents=True, exist_ok=True)
    report(0.01, "Đang phân tích bố cục PDF...")
    ir = analyze_pdf(input_path, opts, report)
    translate_document(ir, opts, report)
    ir.save(out_dir / IR_FILENAME)
    outputs = render_outputs(ir, input_path, out_dir, opts.make_bilingual, report, opts.highlight_doubtful)
    report(1.0, "Hoàn tất")
    return outputs
