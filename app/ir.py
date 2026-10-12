"""DocIR — mô hình dữ liệu trung tâm của một tài liệu đang dịch.

Mọi tầng đọc/ghi cùng cấu trúc này: phân tích bố cục tạo ra ``Region``, tầng dịch
điền ``translation``, tầng dựng chỉ đọc IR để xuất PDF/Markdown. Nhờ đó có thể lưu
IR ra ``document.json`` và dựng lại PDF (vd. sau khi người dùng sửa bản dịch) mà
không cần phân tích hay dịch lại.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

from .pipeline import strip_markers

IR_VERSION = 1

# Loại vùng: text | math | table | data | code | reference | rotated | empty
TRANSLATABLE = "text"


@dataclass
class Region:
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
    # Cờ chất lượng: "kept_english" (giữ tiếng Anh vì bản dịch hỏng), "edited" (người dùng sửa)...
    flags: list[str] = field(default_factory=list)
    id: str = ""
    score: float | None = None  # điểm chất lượng bản dịch 0..1 (thấp nhất trong các câu của vùng)
    notes: list[str] = field(default_factory=list)  # lý do bị trừ điểm

    @property
    def plain(self) -> str:
        return strip_markers(self.text)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        d["span_rects"] = [list(r) for r in self.span_rects]
        d["line_rects"] = [list(r) for r in self.line_rects]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Region:
        d = dict(d)
        d["bbox"] = pymupdf.Rect(d["bbox"])
        d["span_rects"] = [pymupdf.Rect(r) for r in d.get("span_rects", [])]
        d["line_rects"] = [pymupdf.Rect(r) for r in d.get("line_rects", [])]
        return cls(**d)


@dataclass
class Page:
    index: int  # vị trí trong tài liệu đã chọn trang (0-based)
    number: int  # số trang trong file gốc (1-based)
    width: float
    height: float
    regions: list[Region] = field(default_factory=list)
    scanned: bool = False  # trang ảnh scan, chữ lấy bằng OCR


@dataclass
class Document:
    source: str
    pages: list[Page] = field(default_factory=list)
    engine: str = ""
    glossary: dict[str, str] = field(default_factory=dict)
    version: int = IR_VERSION

    @property
    def regions(self) -> list[Region]:
        return [r for p in self.pages for r in p.regions]

    @property
    def page_numbers(self) -> list[int]:
        return [p.number for p in self.pages]

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.regions:
            out[r.kind] = out.get(r.kind, 0) + 1
        return out

    def assign_ids(self) -> None:
        for p in self.pages:
            for i, r in enumerate(p.regions):
                r.id = f"p{p.number}-r{i}"

    # -- JSON ---------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "source": self.source,
            "engine": self.engine,
            "glossary": self.glossary,
            "pages": [
                {
                    "index": p.index,
                    "number": p.number,
                    "width": p.width,
                    "height": p.height,
                    "regions": [r.to_dict() for r in p.regions],
                    "scanned": p.scanned,
                }
                for p in self.pages
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Document:
        if d.get("version") != IR_VERSION:
            raise ValueError(f"Phiên bản DocIR không hỗ trợ: {d.get('version')}")
        pages = [
            Page(
                index=p["index"], number=p["number"], width=p["width"], height=p["height"],
                regions=[Region.from_dict(r) for r in p["regions"]], scanned=p.get("scanned", False),
            )
            for p in d["pages"]
        ]
        return cls(source=d["source"], pages=pages, engine=d.get("engine", ""), glossary=d.get("glossary", {}))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Document:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
