"""OCR cho trang PDF dạng ảnh scan.

PyMuPDF đã tích hợp sẵn thư viện Tesseract; chỉ cần dữ liệu ngôn ngữ (``eng.traineddata``).
Thứ tự tìm dữ liệu: biến ``TESSDATA_PREFIX`` / Tesseract cài trên máy → thư mục
``data/tessdata`` → tự tải ``eng.traineddata`` (bản "fast", ~4 MB) một lần.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

import pymupdf

from .config import settings

log = logging.getLogger(__name__)

TESSDATA_URL = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/{lang}.traineddata"
_WINDOWS_DIRS = [r"C:\Program Files\Tesseract-OCR\tessdata", r"C:\Program Files (x86)\Tesseract-OCR\tessdata"]


class OcrError(RuntimeError):
    pass


def _has_langs(folder: Path, langs: list[str]) -> bool:
    return all((folder / f"{lang}.traineddata").exists() for lang in langs)


def find_tessdata(download: bool = True) -> str:
    langs = settings.ocr_lang.split("+")
    candidates: list[Path] = []
    try:
        candidates.append(Path(pymupdf.get_tessdata()))
    except Exception:  # noqa: BLE001  (Tesseract không cài sẵn)
        pass
    if os.environ.get("TESSDATA_PREFIX"):
        candidates.append(Path(os.environ["TESSDATA_PREFIX"]))
    candidates += [Path(p) for p in _WINDOWS_DIRS]
    local = settings.data_dir / "tessdata"
    candidates.append(local)
    for c in candidates:
        if c.is_dir() and _has_langs(c, langs):
            return str(c)
    if not download:
        raise OcrError("Chưa có dữ liệu OCR")
    local.mkdir(parents=True, exist_ok=True)
    for lang in langs:
        target = local / f"{lang}.traineddata"
        if target.exists():
            continue
        url = TESSDATA_URL.format(lang=lang)
        log.info("Tải dữ liệu OCR %s ...", url)
        try:
            tmp = target.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(target)
        except Exception as e:  # noqa: BLE001
            raise OcrError(
                f"Không tải được dữ liệu OCR ({url}): {e}. Tải thủ công file {lang}.traineddata "
                f"vào thư mục {local} hoặc cài Tesseract OCR."
            ) from e
    return str(local)


def is_scanned(page: pymupdf.Page) -> bool:
    """Trang gần như không có lớp chữ nhưng có ảnh phủ phần lớn trang."""
    if len(page.get_text().strip()) >= 30:
        return False
    area = abs(page.rect)
    covered = 0.0
    for info in page.get_image_info():
        covered += abs(pymupdf.Rect(info["bbox"]) & page.rect)
    return covered > 0.3 * area


def ocr_textpage(page: pymupdf.Page, tessdata: str) -> pymupdf.TextPage:
    return page.get_textpage_ocr(language=settings.ocr_lang, dpi=settings.ocr_dpi, full=True, tessdata=tessdata)
