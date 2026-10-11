"""Quy trình dịch văn bản: bảo vệ ký hiệu → tách câu → dịch (có cache) → khôi phục."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

from .scientific import (
    PLACEHOLDER_RE,
    Glossary,
    Protector,
    has_translatable_text,
    placeholder_ids,
    split_sentences,
    translate_heading,
)
from .translators import Translator, get_cache

log = logging.getLogger(__name__)

# Ký tự đánh dấu do pdf_processor chèn quanh span công thức / chỉ số trên.
MATH_START, MATH_END = "", ""
SUP_START, SUP_END = "", ""
_MARKED_RE = re.compile(f"{MATH_START}(.*?){MATH_END}|{SUP_START}(.*?){SUP_END}", re.S)

ProgressFn = Callable[[int, int], None]


def strip_markers(text: str) -> str:
    return re.sub(f"[{MATH_START}{MATH_END}{SUP_START}{SUP_END}]", "", text)


@dataclass
class _Unit:
    source: str
    protector: Protector = field(default_factory=Protector)
    segments: list[str] = field(default_factory=list)
    result: str | None = None


def postprocess_vi(text: str) -> str:
    text = re.sub(f"[{MATH_START}{MATH_END}]", "", text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[]) ", r"\1", text)
    return text.strip()


class TextTranslator:
    """Dịch danh sách đoạn văn khoa học bằng một engine bất kỳ."""

    def __init__(self, engine: Translator, glossary: Glossary | None = None, use_cache: bool = True):
        self.engine = engine
        self.glossary = glossary if glossary is not None and len(glossary) else None
        self.cache = get_cache() if use_cache else None

    # -- chuẩn bị ----------------------------------------------------------

    def _prepare(self, text: str) -> _Unit:
        unit = _Unit(source=text)
        plain = strip_markers(text)
        heading = translate_heading(plain)
        if heading is not None:
            unit.result = heading
            return unit

        prot = unit.protector

        def mark(m: re.Match) -> str:
            if m.group(1) is not None:
                return prot.token(m.group(1))
            return prot.token(SUP_START + m.group(2) + SUP_END)

        # Ngoặc nhọn có sẵn trong văn bản được bảo vệ trước để không lẫn với placeholder.
        p = prot.protect_braces(text)
        p = _MARKED_RE.sub(mark, p)
        p = prot.protect_patterns(p, braces=False)
        if self.glossary is not None and not self.engine.uses_glossary_prompt:
            p = self.glossary.protect(p, prot)
        p = prot.merge_adjacent(p)

        if not has_translatable_text(p):
            unit.result = prot.restore(p)
            return unit
        if self.engine.prefers_paragraphs:
            unit.segments = split_sentences(p, max_len=2000) if len(p) > 2000 else [p]
        else:
            unit.segments = split_sentences(p)
        return unit

    # -- dịch -------------------------------------------------------------

    def _extra_key(self, seg: str) -> tuple[str, dict[str, str] | None]:
        if self.glossary is not None and self.engine.uses_glossary_prompt:
            rel = self.glossary.relevant(seg)
            return json.dumps(rel, sort_keys=True, ensure_ascii=False), rel or None
        return "", None

    def _run_engine(self, segs: list[str]) -> dict[str, str]:
        """Dịch các câu chưa có trong cache; trả về map nguồn → đích."""
        out: dict[str, str] = {}
        by_extra: dict[str, tuple[dict | None, list[str]]] = {}
        for s in segs:
            extra, rel = self._extra_key(s)
            by_extra.setdefault(extra, (rel, []))[1].append(s)

        for extra, (rel, items) in by_extra.items():
            todo = items
            if self.cache is not None:
                hit = self.cache.get_many(self.engine.cache_id(), items, extra)
                out.update(hit)
                todo = [s for s in items if s not in hit]
            if not todo:
                continue
            res = self.engine.translate_batch(todo, glossary=rel)
            pairs = dict(zip(todo, res))
            out.update(pairs)
            if self.cache is not None:
                self.cache.put_many(self.engine.cache_id(), pairs, extra)
        return out

    def _translate_piecewise(self, seg: str) -> str:
        """Dự phòng khi mô hình làm mất placeholder: dịch từng đoạn giữa các placeholder."""
        pieces = re.split(r"(\{\d{1,3}\})", seg)
        to_tr = [p.strip() for p in pieces if not PLACEHOLDER_RE.fullmatch(p) and has_translatable_text(p)]
        tr = self._run_engine(list(dict.fromkeys(to_tr))) if to_tr else {}
        out = []
        for p in pieces:
            key = p.strip()
            if key in tr:
                t = PLACEHOLDER_RE.sub("", tr[key])  # bỏ placeholder do mô hình tự bịa
                lead = " " if p[:1].isspace() else ""
                trail = " " if p[-1:].isspace() else ""
                out.append(lead + t.strip() + trail)
            else:
                out.append(p)
        return "".join(out)

    def translate_texts(self, texts: list[str], progress: ProgressFn | None = None) -> list[str]:
        units = [self._prepare(t) for t in texts]
        all_segs = list(dict.fromkeys(s for u in units for s in u.segments))
        total = len(all_segs)
        results: dict[str, str] = {}

        # Dịch theo từng lô nhỏ để báo tiến độ đều đặn.
        chunk = 1 if self.engine.prefers_paragraphs else 32
        done = 0
        if progress:
            progress(0, total)
        for i in range(0, total, chunk):
            batch = all_segs[i : i + chunk]
            translated = self._run_engine(batch)
            for s in batch:
                t = translated.get(s, s)
                if placeholder_ids(t) != placeholder_ids(s):
                    log.debug("Placeholder lệch, dịch từng phần: %r -> %r", s, t)
                    t = self._translate_piecewise(s)
                results[s] = t
            done += len(batch)
            if progress:
                progress(done, total)

        out = []
        for u in units:
            if u.result is None:
                joined = " ".join(results.get(s, s) for s in u.segments)
                u.result = u.protector.restore(joined)
            out.append(postprocess_vi(u.result))
        return out
