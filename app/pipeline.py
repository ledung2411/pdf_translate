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
from .config import settings
from .translators import Translator, get_cache

log = logging.getLogger(__name__)

# Ký tự đánh dấu do pdf_processor chèn quanh span công thức / chỉ số trên.
MATH_START, MATH_END = "", ""
SUP_START, SUP_END = "", ""
_MARKED_RE = re.compile(f"{MATH_START}(.*?){MATH_END}|{SUP_START}(.*?){SUP_END}", re.S)

ProgressFn = Callable[[int, int], None]

# Dùng khi người dùng tắt bảng thuật ngữ: vẫn Việt hóa nhãn Hình/Bảng/Phương trình.
_LABEL_GLOSSARY = Glossary(
    {"Fig.": "Hình", "Figs.": "Hình", "Figure": "Hình", "Table": "Bảng", "Eq.": "Phương trình",
     "Eqs.": "Phương trình", "Equation": "Phương trình", "Section": "Mục", "Sec.": "Mục", "et al.": "và cộng sự"}
)


def strip_markers(text: str) -> str:
    return re.sub(f"[{MATH_START}{MATH_END}{SUP_START}{SUP_END}]", "", text)


@dataclass
class _Unit:
    source: str
    protector: Protector = field(default_factory=Protector)
    segments: list[str] = field(default_factory=list)
    cases: list[str] = field(default_factory=list)  # cách viết hoa gốc của từng câu
    prefix: str = ""  # nhãn đầu đoạn đã dịch sẵn ("TÓM TẮT ", "Từ khóa: ")
    result: str | None = None


_LEAD_LABEL_RE = re.compile(r"^(ABSTRACT|Abstract|KEYWORDS|Keywords|Key words|Index Terms)([.:—–\s]+)(?=\S)")
_SUPERSCRIPT = str.maketrans("0123456789+-−=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾ⁿ")
_ACRONYM_RE = re.compile(r"^(?:[A-Z0-9]{2,}|[a-z]+[A-Z][\w]*|[A-Z][a-z]*[A-Z][\w]*)$")


def _sup_to_plain(content: str) -> str:
    """Chỉ số trên → ký tự Unicode (cm² , 10⁻³) để mô hình dịch máy không gặp ký tự lạ."""
    c = content.strip()
    if c and all(ch in "0123456789+-−=()n" for ch in c):
        return c.translate(_SUPERSCRIPT)
    return c


def normalize_case(text: str) -> tuple[str, str]:
    """Mô hình dịch máy dịch rất tệ chữ IN HOA và Viết Hoa Mọi Từ (tiêu đề).

    Đưa về dạng câu thường trước khi dịch; trả về (văn bản, kiểu) để khôi phục sau.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.85:
        low = text.lower()
        return low[:1].upper() + low[1:], "upper"
    words = re.findall(r"[A-Za-z][\w’']*", text)
    # Chỉ áp dụng cho tiêu đề ngắn; danh sách địa chỉ/tên riêng (nhiều dấu phẩy) giữ nguyên
    # để không biến "China" thành "china".
    if 3 <= len(words) <= 25 and text.count(",") <= 1:
        caps = [w for w in words if w[0].isupper()]
        if len(caps) / len(words) >= 0.7:
            def lower_word(m: re.Match) -> str:
                w = m.group(0)
                if _ACRONYM_RE.match(w) or not (w[0].isupper() and w[1:] == w[1:].lower()):
                    return w
                return w.lower()

            out = re.sub(r"[A-Za-z][\w’']*", lower_word, text)
            return out[:1].upper() + out[1:], "title"
    return text, ""


def restore_case(text: str, mode: str) -> str:
    if mode == "upper":
        return text.upper()
    if mode == "title":
        return text[:1].upper() + text[1:]
    return text


def is_degenerate(src: str, out: str) -> bool:
    """Phát hiện bản dịch hỏng: lặp vô hạn, toàn gạch dưới, dài bất thường."""
    if not out.strip():
        return True
    if "___" in out or len(out) > 3 * len(src) + 40:
        return True
    words = out.split()
    if len(words) >= 6:
        run = 1
        for a, b in zip(words, words[1:]):
            run = run + 1 if a.lower() == b.lower() else 1
            if run >= 4:
                return True
        for n in (2, 3, 4):
            grams = [tuple(w.lower() for w in words[i : i + n]) for i in range(len(words) - n + 1)]
            if grams and max(grams.count(g) for g in set(grams)) >= 4 and len(words) < 12 * n:
                return True
            for i in range(len(grams) - 2 * n):
                if grams[i] == grams[i + n] == grams[i + 2 * n]:
                    return True
    if len(words) > 15 and len({w.lower() for w in words}) / len(words) < 0.3:
        return True
    return False


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

        if not self.engine.supports_placeholders:
            return self._prepare_plain(unit, text)

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
        unit.cases = [""] * len(unit.segments)
        return unit

    def _prepare_plain(self, unit: _Unit, text: str) -> _Unit:
        """Cho mô hình dịch máy (OPUS, VinAI, NLLB): KHÔNG chèn placeholder {n}.

        Các mô hình này không biết ký hiệu "{" và sinh ra chữ rác ("Name", "Comment").
        Công thức, trích dẫn, URL được giữ nguyên trong câu (mô hình thường chép lại),
        thuật ngữ được áp dụng sau khi dịch.
        """

        def plain(m: re.Match) -> str:
            return m.group(1) if m.group(1) is not None else _sup_to_plain(m.group(2))

        t = _MARKED_RE.sub(plain, text)
        if not re.search(r"[A-Za-z]{2,}", t) or not has_translatable_text(t):
            unit.result = t
            return unit
        # "ABSTRACT Cortical surface..." / "Keywords: a | b" → dịch nhãn riêng.
        m = _LEAD_LABEL_RE.match(t)
        if m:
            unit.prefix = (translate_heading(m.group(1)) or m.group(1)) + m.group(2)
            t = t[m.end():]
        for seg in split_sentences(t, max_len=400):
            seg, mode = normalize_case(seg)
            unit.segments.append(seg)
            unit.cases.append(mode)
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

    def _repair(self, seg: str) -> str:
        """Bản dịch câu bị hỏng: dịch lại từng vế ngắn; vế nào vẫn hỏng thì giữ tiếng Anh."""
        parts = [p for p in re.split(r"(?<=[,;:])\s+", seg) if p.strip()]
        if len(parts) < 2:
            return seg
        tr = self._run_engine(parts)
        out = []
        for p in parts:
            t = tr.get(p, p)
            out.append(p if is_degenerate(p, t) else t)
        return " ".join(out)

    def translate_texts(self, texts: list[str], progress: ProgressFn | None = None) -> list[str]:
        units = [self._prepare(t) for t in texts]
        all_segs = list(dict.fromkeys(s for u in units for s in u.segments))
        total = len(all_segs)
        results: dict[str, str] = {}

        # Dịch theo từng lô nhỏ để báo tiến độ đều đặn.
        chunk = 1 if self.engine.prefers_paragraphs else max(4, settings.batch_size)
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
                if is_degenerate(s, t):
                    log.info("Bản dịch hỏng, dịch lại từng vế: %r -> %r", s[:80], t[:80])
                    t = self._repair(s)
                results[s] = t
            done += len(batch)
            if progress:
                progress(done, total)

        out = []
        for u in units:
            if u.result is None:
                cases = u.cases or [""] * len(u.segments)
                joined = " ".join(restore_case(results.get(s, s), c) for s, c in zip(u.segments, cases))
                u.result = u.prefix + u.protector.restore(joined)
                if not self.engine.supports_placeholders:
                    u.result = self._fix_terms(u.result)
            out.append(postprocess_vi(u.result))
        return out

    def _fix_terms(self, text: str) -> str:
        """Sau khi dịch: Việt hóa nhãn còn sót (Fig. 3 → Hình 3) và thuật ngữ còn tiếng Anh."""
        glossary = self.glossary or _LABEL_GLOSSARY
        return glossary.apply_to_output(text)
