"""Xem tool phân loại các khối chữ trong PDF thế nào (để gỡ lỗi khi không dịch).

    python -m app.inspect paper.pdf            # 2 trang đầu
    python -m app.inspect paper.pdf --page 3
"""

from __future__ import annotations

import argparse
from collections import Counter

import pymupdf

from .pdf_processor import extract_blocks, mark_references


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--page", type=int, help="chỉ xem trang này (đánh số từ 1)")
    args = ap.parse_args(argv)
    doc = pymupdf.open(args.pdf)
    pages = [args.page - 1] if args.page else list(range(min(2, len(doc))))
    for i in pages:
        page = doc[i]
        blocks = extract_blocks(page, i)
        mark_references(blocks)
        fonts = Counter()
        for b in page.get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                for sp in ln["spans"]:
                    fonts[sp["font"]] += len(sp["text"])
        print(f"===== Trang {i + 1}: {Counter(b.kind for b in blocks)}")
        print("Font:", ", ".join(f"{f} ({n})" for f, n in fonts.most_common(8)))
        for b in blocks:
            r = b.bbox
            print(f"[{b.kind:9}] ({r.x0:5.0f},{r.y0:5.0f},{r.x1:5.0f},{r.y1:5.0f}) {b.plain[:90]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
