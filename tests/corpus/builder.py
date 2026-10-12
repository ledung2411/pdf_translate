"""Bộ dựng PDF mô phỏng đặc điểm của các nhà xuất bản (font, cột, khoảng trắng lạ...).

Dùng font có sẵn (Liberation, Noto Sans Math) rồi đổi tên font trong PDF thành
tên font thật của nhà xuất bản (CMR10, STIXTwoText, TimesNewRomanPSMT...), vì tool
nhận diện công thức/kiểu chữ dựa trên tên font.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf
import pymupdf_fonts  # noqa: F401  (đăng ký font "math")

LIB = "/usr/share/fonts/truetype/liberation/"
FONT_FILES = {
    "serif": LIB + "LiberationSerif-Regular.ttf",
    "serif-bold": LIB + "LiberationSerif-Bold.ttf",
    "serif-italic": LIB + "LiberationSerif-Italic.ttf",
    "sans": LIB + "LiberationSans-Regular.ttf",
    "sans-bold": LIB + "LiberationSans-Bold.ttf",
}
# Tên gốc (đã bỏ khoảng trắng/gạch) của font nhúng → để đổi tên sau khi dựng.
ORIGINAL_NAMES = {
    "serif": "LiberationSerif", "serif-bold": "LiberationSerifBold", "serif-italic": "LiberationSerifItalic",
    "sans": "LiberationSans", "sans-bold": "LiberationSansBold", "math": "NotoSansMath",
}


@dataclass
class Style:
    font: str  # khóa trong FONT_FILES hoặc "math"
    size: float = 9.5
    leading: float = 1.22


@dataclass
class Builder:
    """Ghi chữ theo từng từ (để điều khiển chính xác span, font, khoảng trắng)."""

    rename: dict[str, str]  # khóa font → tên font giả lập trong PDF
    width: float = 595
    height: float = 842
    space_font: str | None = None  # Wiley: dấu cách nằm trong span font riêng "dummy-space"
    doc: pymupdf.Document = field(default_factory=pymupdf.open)

    def __post_init__(self) -> None:
        self.page = self.doc.new_page(width=self.width, height=self.height)
        self._fonts: dict[str, pymupdf.Font] = {}

    def new_page(self) -> None:
        self.page = self.doc.new_page(width=self.width, height=self.height)

    def _font(self, key: str) -> tuple[str, pymupdf.Font]:
        alias = "F" + re.sub(r"\W", "", key)
        if key not in self._fonts:
            self._fonts[key] = pymupdf.Font("math") if key == "math" else pymupdf.Font(fontfile=FONT_FILES[key])
        try:
            if key == "math":
                self.page.insert_font(fontname=alias, fontbuffer=self._fonts[key].buffer)
            else:
                self.page.insert_font(fontname=alias, fontfile=FONT_FILES[key])
        except Exception:  # noqa: BLE001  (đã chèn trên trang này)
            pass
        return alias, self._fonts[key]

    def text(self, x: float, y: float, text: str, style: Style, rotate: int = 0) -> None:
        alias, _ = self._font(style.font)
        self.page.insert_text((x, y), text, fontname=alias, fontsize=style.size, rotate=rotate)

    def paragraph(self, x0: float, x1: float, y: float, text: str, style: Style, math: Style | None = None,
                  align: str = "justify") -> float:
        """Ghi đoạn văn tự xuống dòng. ``$...$`` là công thức nội dòng (font toán).

        Trả về toạ độ y sau đoạn.
        """
        alias, font = self._font(style.font)
        tokens = re.findall(r"\$[^$]+\$|\S+", text)
        words = [(t.strip("$"), math if t.startswith("$") and math else style) for t in tokens]
        space = font.text_length(" ", style.size)
        lines: list[list[tuple[str, Style, float]]] = [[]]
        width = 0.0
        for w, st in words:
            _, f = self._font(st.font)
            wl = f.text_length(w, st.size)
            if lines[-1] and width + space + wl > x1 - x0:
                lines.append([])
                width = 0.0
            width += (space if lines[-1] else 0) + wl
            lines[-1].append((w, st, wl))
        line_h = style.size * style.leading
        for i, line in enumerate(lines):
            natural = sum(wl for _, _, wl in line) + space * (len(line) - 1)
            gap = space
            x = x0
            if align == "justify" and i < len(lines) - 1 and len(line) > 1:
                gap = space + (x1 - x0 - natural) / (len(line) - 1)
            elif align == "center":
                x = x0 + (x1 - x0 - natural) / 2
            for j, (w, st, wl) in enumerate(line):
                a, _ = self._font(st.font)
                self.page.insert_text((x, y), w, fontname=a, fontsize=st.size)
                x += wl
                if j < len(line) - 1:
                    if self.space_font:
                        sa, _ = self._font(self.space_font)
                        self.page.insert_text((x, y), " ", fontname=sa, fontsize=st.size)
                    x += gap
            y += line_h
        return y + style.size * 0.6

    def save(self, path: str) -> None:
        targets = {ORIGINAL_NAMES[k]: v for k, v in self.rename.items()}
        for xref in range(1, self.doc.xref_length()):
            for key in ("BaseFont", "FontName"):
                typ, val = self.doc.xref_get_key(xref, key)
                if typ != "name":
                    continue
                norm = re.sub(r"[\s\-/]|Regular", "", val)
                norm = norm.split("+")[-1]
                for orig in sorted(targets, key=len, reverse=True):
                    if norm.startswith(orig):
                        self.doc.xref_set_key(xref, key, "/" + targets[orig])
                        break
        self.doc.save(path, garbage=3, deflate=True)
