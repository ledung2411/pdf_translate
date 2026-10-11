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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pymupdf

from .config import settings
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


@dataclass
class TextBlock:
    page_index: int
    bbox: pymupdf.Rect
    text: str  # có ký tự đánh dấu công thức / chỉ số trên
    kind: str
    font_size: float
    bold: bool
    italic: bool
    serif: bool
    color: int
    align: str
    span_rects: list[pymupdf.Rect] = field(default_factory=list)
    line_rects: list[pymupdf.Rect] = field(default_factory=list)
    translation: str | None = None

    @property
    def plain(self) -> str:
        return strip_markers(self.text)


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
    parts: list[list[dict]] = [[spans[0]]]
    for a, b in zip(spans, spans[1:]):
        if b["bbox"][0] - a["bbox"][2] > max(a.get("size", 10) * 2.5, 12):
            parts.append([b])
        else:
            parts[-1].append(b)
    if len(parts) == 1:
        return [line]
    return [{**line, "spans": p, "bbox": _line_bbox(p)} for p in parts]


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
        widths = [c["x"][2] - c["x"][0] for c in cols]
        if len(cols) >= 3 or min(widths) < 0.25 * page_rect.width:
            out.append({**b, "lines": lines})  # bảng: giữ nguyên để không dịch vỡ bố cục
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
    return out


def _union(boxes: list) -> tuple[float, float, float, float]:
    return (
        min(x[0] for x in boxes), min(x[1] for x in boxes),
        max(x[2] for x in boxes), max(x[3] for x in boxes),
    )


def extract_blocks(page: pymupdf.Page, page_index: int) -> list[TextBlock]:
    flags = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP
    data = page.get_text("dict", flags=flags)
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
                    if is_mono_font(font) or sp.get("flags", 0) & 8:
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


def _to_html(text: str) -> str:
    s = html.escape(text, quote=False)
    s = s.replace(SUP_START, "<sup>").replace(SUP_END, "</sup>")
    s = s.replace(MATH_START, "").replace(MATH_END, "")
    return s


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


def render_page(page: pymupdf.Page, blocks: list[TextBlock], fontkit: FontKit) -> None:
    todo = [b for b in blocks if b.translation is not None]
    if not todo:
        return
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
        css = fontkit.css + f"\nbody{{margin:0;padding:0;}} .b{{{style}}} sup{{font-size:70%;}}"
        rect = _target_rect(b, blocks, page.rect)
        # Nới nhẹ khung để chữ có dấu tiếng Việt không bị cắt.
        rect.y1 += b.font_size * 0.15
        try:
            page.insert_htmlbox(
                rect, f'<div class="b">{_to_html(b.translation)}</div>', css=css,
                archive=fontkit.archive, scale_low=0,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Không ghi được khối trên trang %d: %s", b.page_index + 1, e)


def make_bilingual(original: pymupdf.Document, translated: pymupdf.Document) -> pymupdf.Document:
    out = pymupdf.open()
    for i in range(len(translated)):
        r = original[i].rect
        page = out.new_page(width=r.width * 2, height=r.height)
        page.show_pdf_page(pymupdf.Rect(0, 0, r.width, r.height), original, i)
        page.show_pdf_page(pymupdf.Rect(r.width, 0, r.width * 2, r.height), translated, i)
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


def translate_pdf(
    input_path: Path,
    out_dir: Path,
    opts: TranslateOptions,
    progress: ProgressFn | None = None,
) -> dict[str, Path]:
    def report(frac: float, msg: str) -> None:
        if progress:
            progress(max(0.0, min(1.0, frac)), msg)

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        doc = pymupdf.open(str(input_path))
    except Exception as e:  # noqa: BLE001
        raise PdfError(f"Không mở được file PDF: {e}") from e
    if doc.needs_pass and not doc.authenticate(""):
        raise PdfError("PDF được bảo vệ bằng mật khẩu.")
    if not doc.is_pdf:
        raise PdfError("File không phải PDF.")

    n = len(doc)
    start = max(1, opts.page_from or 1)
    end = min(n, opts.page_to or n)
    if start > end:
        raise PdfError(f"Khoảng trang không hợp lệ (tài liệu có {n} trang).")
    if end - start + 1 > settings.max_pages:
        raise PdfError(f"Tối đa {settings.max_pages} trang mỗi lần dịch.")
    if start != 1 or end != n:
        doc.select(list(range(start - 1, end)))
    page_numbers = list(range(start, end + 1))
    original = pymupdf.open()
    original.insert_pdf(doc)

    report(0.01, "Đang phân tích bố cục PDF...")
    pages_blocks = [extract_blocks(doc[i], i) for i in range(len(doc))]
    all_blocks = [b for blocks in pages_blocks for b in blocks]
    if sum(len(b.plain) for b in all_blocks) < 30 * len(doc):
        raise PdfError(
            "PDF gần như không có lớp chữ (có thể là bản scan/ảnh). "
            "Hãy chạy OCR trước (ví dụ: ocrmypdf input.pdf output.pdf) rồi dịch lại."
        )
    if opts.skip_references:
        mark_references(all_blocks)

    to_translate = [b for b in all_blocks if b.kind == "text"]
    stats = {k: sum(1 for b in all_blocks if b.kind == k) for k in {b.kind for b in all_blocks}}
    log.info("Thống kê khối: %s", stats)
    if not to_translate:
        raise PdfError(
            f"Không tìm thấy đoạn văn nào cần dịch (phân loại khối: {stats}). "
            "Hãy chạy: python -m app.inspect <file.pdf> và gửi kết quả để được hỗ trợ."
        )
    engine = get_engine(opts.engine)
    ok, why = engine.available()
    if not ok:
        raise TranslatorError(f"{engine.label}: {why}")
    engine.prepare(lambda msg: report(0.02, msg))
    translator = TextTranslator(engine, build_glossary(opts))
    report(0.03, f"Đang dịch {len(to_translate)} đoạn văn bằng {engine.label}...")
    t_start = time.time()

    def on_tr(done: int, total: int) -> None:
        frac = 0.03 + 0.85 * (done / total if total else 1)
        msg = f"Đang dịch: {done}/{total} câu"
        if 0 < done < total:
            remain = (time.time() - t_start) / done * (total - done)
            msg += f" — còn khoảng {_fmt_duration(remain)}"
        report(frac, msg)

    translations = translator.translate_texts([b.text for b in to_translate], on_tr)
    for b, t in zip(to_translate, translations):
        if t and t.strip() != b.plain.strip():
            b.translation = t

    fontkit = get_fontkit()
    for i, blocks in enumerate(pages_blocks):
        report(0.88 + 0.09 * i / max(len(doc), 1), f"Đang dựng trang {i + 1}/{len(doc)}...")
        render_page(doc[i], blocks, fontkit)

    report(0.97, "Đang lưu kết quả...")
    stem = re.sub(r"[^\w\-]+", "_", input_path.stem)[:80] or "document"
    outputs: dict[str, Path] = {}
    p = out_dir / f"{stem}_vi.pdf"
    doc.save(str(p), garbage=3, deflate=True)
    outputs["translated"] = p
    if opts.make_bilingual:
        translated = pymupdf.open(str(p))
        bi = make_bilingual(original, translated)
        p2 = out_dir / f"{stem}_song_ngu.pdf"
        bi.save(str(p2), garbage=3, deflate=True)
        outputs["bilingual"] = p2
    p3 = out_dir / f"{stem}_vi.md"
    p3.write_text(blocks_to_markdown(pages_blocks, page_numbers), encoding="utf-8")
    outputs["markdown"] = p3
    report(1.0, "Hoàn tất")
    return outputs
