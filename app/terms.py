"""Tự trích thuật ngữ của một bài báo trước khi dịch.

Ba nguồn ứng viên:
1. Định nghĩa viết tắt: "diffusion MRI (dMRI)", "pseudo-ground-truth (pGT)".
2. Từ viết tắt / tên riêng kỹ thuật lặp lại: WM, SDF, DDSurfer.
3. Cụm danh từ chuyên ngành lặp lại ≥3 lần: "cortical surface reconstruction".

Người dùng duyệt / điền bản dịch, các mục có bản dịch được đưa vào glossary của lần dịch đó.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass

from .ir import Document
from .scientific import Glossary, load_builtin_glossary

STOPWORDS = set(
    """a an the of and or for to in on at by with from as is are was were be been being this that these
    those it its their our we which who whom whose than then there here into onto over under between
    among through during after before about against via per not no can may might will would should could
    also such each both all any some more most less least very only other same however thus hence while
    where when what how using used use based show shows shown proposed propose new novel two three one
    first second third fig figure table section eq et al i e g""".split()
)
_ROMAN = re.compile(r"^[IVXLC]+$")
_COMMON_CAPS = {"USA", "UK", "PhD", "MSc", "BSc", "OK", "PDF", "ID", "IEEE", "ACM", "URL", "DOI", "CC", "BY"}


# Từ quá chung chung, không làm đầu/cuối một thuật ngữ.
GENERIC = set(
    """result results better worse good best data value values means mean includes include including
    directly compared method methods approach approaches paper work study studies performance model models
    subject subjects test tests testing across previous following respectively overall different similar
    various several many large small high low higher lower main major key important significant
    significantly given total average obtained proposed existing""".split()
)


def _good_phrase(gram: list[str]) -> bool:
    if gram[0] in STOPWORDS or gram[-1] in STOPWORDS or gram[0] in GENERIC or gram[-1] in GENERIC:
        return False
    if any(len(w) < 3 for w in gram) or gram[-1].endswith("ly"):
        return False
    # Chỉ cho phép "of" ở giữa ("region of interest"), không cho "and/from/for/the..."
    return all(w == "of" or w not in STOPWORDS for w in gram[1:-1])


@dataclass
class TermCandidate:
    term: str
    count: int
    kind: str  # "definition" | "acronym" | "phrase"
    expansion: str = ""  # dạng đầy đủ của từ viết tắt (nếu có)
    suggestion: str = ""  # gợi ý bản dịch (từ bảng thuật ngữ có sẵn / giữ nguyên viết tắt)

    def to_dict(self) -> dict:
        return asdict(self)


def _letters(abbr: str) -> list[str]:
    a = abbr[:-1] if abbr.endswith("s") and abbr[:-1].isupper() and len(abbr) > 2 else abbr
    return [c.lower() for c in a if c.isalpha()]


def _match_expansion(before: str, abbr: str) -> str | None:
    """Tìm cụm từ ngay trước "(ABBR)" có chữ cái đầu khớp với từ viết tắt.

    Trả về đúng chuỗi trong văn bản gốc (giữ gạch nối: "Desikan-Killiany", "T1-weighted").
    """
    letters = _letters(abbr)
    if not 2 <= len(letters) <= 8:
        return None
    matches = list(re.finditer(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", before))[-(len(letters) + 4):]
    words = [m.group(0) for m in matches]
    for start in range(len(words) - 1, -1, -1):
        cand = words[start:]
        if not cand[0][0].isalpha() or cand[0].lower() in STOPWORDS:
            continue
        initials: list[str] = []
        for i, w in enumerate(cand):
            if i and w.lower() in STOPWORDS:
                continue
            for part in w.split("-"):
                if part.isupper() and len(part) > 1 and part.isalpha():
                    initials.extend(c.lower() for c in part)  # "diffusion MRI" → d,m,r,i
                elif part[0].isalpha():
                    initials.append(part[0].lower())
        if initials == letters:
            return before[matches[start].start():].strip()
        if len(initials) > len(letters):
            break
    return None


def _strip_subphrases(counts: Counter) -> Counter:
    """Bỏ cụm con nếu gần như luôn xuất hiện trong một cụm dài hơn."""
    keep = Counter(counts)
    for longer, n_long in counts.items():
        for shorter in list(keep):
            if shorter != longer and f" {shorter} " in f" {longer} " and keep[shorter] <= n_long + 1:
                del keep[shorter]
    return keep


def extract_terms(ir: Document, max_phrases: int = 40) -> list[TermCandidate]:
    texts = [r.plain for r in ir.regions if r.kind == "text"]
    full = "\n".join(texts)
    builtin = Glossary(load_builtin_glossary())
    lookup = {k.lower(): v for k, v in builtin.entries.items()}
    out: list[TermCandidate] = []
    seen: set[str] = set()

    # 1. Định nghĩa viết tắt
    for m in re.finditer(r"([^()\n]{3,160}?)\s*\(\s*([A-Za-z][A-Za-z0-9]{1,9})\s*\)", full):
        abbr = m.group(2)
        if sum(c.isupper() for c in abbr) < 1 or abbr.lower() in seen:
            continue
        exp = _match_expansion(m.group(1), abbr)
        if not exp:
            continue
        seen.add(abbr.lower())
        n = len(re.findall(re.escape(exp), full, re.IGNORECASE))
        out.append(TermCandidate(
            term=exp, count=max(n, 1), kind="definition", expansion=abbr,
            suggestion=lookup.get(exp.lower(), ""),
        ))

    # 2. Từ viết tắt / tên kỹ thuật lặp lại (≥2 chữ hoa)
    no_urls = re.sub(r"(https?://|www\.)\S+|\S+@\S+", " ", full)
    acr = Counter(
        w for w in re.findall(r"\b[A-Za-z]*[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b", no_urls)
        if not _ROMAN.match(w) and w not in _COMMON_CAPS and len(w) <= 15
        # "FIGURE", "TABLE": chữ thường viết hoa toàn bộ, không phải viết tắt
        and not (w.isupper() and len(w) > 3 and re.search(rf"\b{w.capitalize()}\b|\b{w.lower()}\b", no_urls))
    )
    for w, n in acr.most_common():
        if n < 2 or w.lower() in seen or (w.endswith("s") and w[:-1].lower() in seen):
            continue
        seen.add(w.lower())
        out.append(TermCandidate(term=w, count=n, kind="acronym", suggestion=w))

    # 3. Cụm danh từ lặp lại (chỉ từ thường; từ viết tắt/CamelCase ngắt cụm)
    phrases: Counter = Counter()
    for t in texts:
        for sent in re.split(r"[.;:!?()\[\],]", t):
            runs: list[list[str]] = [[]]
            for w in re.findall(r"[A-Za-z][A-Za-z-]*", sent):
                if w[1:] == w[1:].lower() and len(w) > 1:
                    runs[-1].append(w.lower())
                else:
                    runs.append([])
            for words in runs:
                for n in (2, 3, 4):
                    for i in range(len(words) - n + 1):
                        gram = words[i : i + n]
                        if _good_phrase(gram):
                            phrases[" ".join(gram)] += 1
    # Gộp số nhiều vào số ít: "pial surfaces" + "pial surface"
    for p in list(phrases):
        if p.endswith("s") and p[:-1] in phrases:
            phrases[p[:-1]] += phrases.pop(p)
    phrases = Counter({p: c for p, c in phrases.items() if c >= 3})
    phrases = _strip_subphrases(phrases)
    ranked = sorted(phrases.items(), key=lambda x: (-(x[1] * len(x[0].split())), x[0]))
    for p, n in ranked[:max_phrases]:
        if p in seen or any(p == c.term.lower() for c in out):
            continue
        out.append(TermCandidate(term=p, count=n, kind="phrase", suggestion=lookup.get(p, "")))
    return out
