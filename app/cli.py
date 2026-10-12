"""Dịch PDF từ dòng lệnh.

Ví dụ:
    python -m app.cli paper.pdf -o out/ --engine opus
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .pdf_processor import PdfError, TranslateOptions, translate_pdf
from .translators import TranslatorError, list_engines


def main(argv: list[str] | None = None) -> int:
    keys = [e.key for e in list_engines()]
    ap = argparse.ArgumentParser(description="Dịch PDF khoa học Anh → Việt (chạy cục bộ).")
    ap.add_argument("pdf", type=Path, help="file PDF đầu vào")
    ap.add_argument("-o", "--out", type=Path, default=Path("output"), help="thư mục kết quả")
    ap.add_argument("-e", "--engine", default="auto", choices=["auto", *keys])
    ap.add_argument("--pages", help="khoảng trang, vd. 1-5")
    ap.add_argument("--translate-references", action="store_true", help="dịch cả mục Tài liệu tham khảo")
    ap.add_argument("--no-glossary", action="store_true", help="không dùng bảng thuật ngữ mặc định")
    ap.add_argument("--glossary", type=Path, help="file thuật ngữ riêng (term = bản dịch)")
    ap.add_argument("--no-bilingual", action="store_true", help="không tạo PDF song ngữ")
    ap.add_argument("--no-ocr", action="store_true", help="không OCR trang ảnh scan")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    page_from = page_to = None
    if args.pages:
        a, _, b = args.pages.partition("-")
        page_from = int(a) if a else None
        page_to = int(b) if b else page_from
    opts = TranslateOptions(
        engine=args.engine,
        page_from=page_from,
        page_to=page_to,
        skip_references=not args.translate_references,
        use_glossary=not args.no_glossary,
        custom_glossary=args.glossary.read_text(encoding="utf-8") if args.glossary else "",
        make_bilingual=not args.no_bilingual,
        ocr=not args.no_ocr,
    )

    last = [""]

    def progress(frac: float, msg: str) -> None:
        line = f"\r[{frac * 100:5.1f}%] {msg}"
        sys.stderr.write(line + " " * max(0, len(last[0]) - len(line)))
        sys.stderr.flush()
        last[0] = line

    t0 = time.time()
    try:
        outputs = translate_pdf(args.pdf, args.out, opts, progress)
    except (PdfError, TranslatorError) as e:
        sys.stderr.write(f"\nLỗi: {e}\n")
        return 1
    sys.stderr.write(f"\nXong sau {time.time() - t0:.1f} giây.\n")
    for kind, path in outputs.items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
