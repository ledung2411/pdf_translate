"""Bộ nhớ dịch (Translation Memory): lưu các đoạn người dùng đã sửa tay.

Lần dịch sau, đoạn nào trùng (sau khi chuẩn hóa khoảng trắng) với một đoạn đã sửa sẽ
dùng ngay bản dịch đã duyệt thay vì dịch máy — hữu ích khi dịch lại cùng tài liệu,
bản sửa đổi của bài báo, hoặc các đoạn lặp (chú thích bản quyền, tiêu đề trang...).
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from pathlib import Path

from .config import settings


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class TranslationMemory:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS tm (src TEXT PRIMARY KEY, tgt TEXT NOT NULL, updated REAL, uses INTEGER DEFAULT 0)"
        )
        self._conn.commit()
        self._lock = threading.Lock()

    def lookup(self, sources: list[str]) -> dict[str, str]:
        keys = list({normalize(s) for s in sources if s.strip()})
        found: dict[str, str] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i : i + 500]
                q = "SELECT src, tgt FROM tm WHERE src IN (%s)" % ",".join("?" * len(chunk))
                found.update(dict(self._conn.execute(q, chunk)))
            if found:
                self._conn.executemany("UPDATE tm SET uses = uses + 1 WHERE src = ?", [(k,) for k in found])
                self._conn.commit()
        return {s: found[normalize(s)] for s in sources if normalize(s) in found}

    def put_many(self, pairs: dict[str, str]) -> None:
        rows = [(normalize(s), t, time.time()) for s, t in pairs.items() if s.strip() and t.strip()]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO tm (src, tgt, updated) VALUES (?, ?, ?) "
                "ON CONFLICT(src) DO UPDATE SET tgt = excluded.tgt, updated = excluded.updated",
                rows,
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM tm").fetchone()[0]


_tm: TranslationMemory | None = None


def get_tm() -> TranslationMemory:
    global _tm
    if _tm is None or settings.data_dir / "tm.sqlite" != getattr(_tm, "_path", None):
        _tm = TranslationMemory(settings.data_dir / "tm.sqlite")
        _tm._path = settings.data_dir / "tm.sqlite"  # type: ignore[attr-defined]
    return _tm
