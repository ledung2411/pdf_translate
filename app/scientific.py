"""Xử lý đặc thù cho tài liệu khoa học.

- Bảo vệ công thức, ký hiệu toán, trích dẫn, URL, DOI, mã nguồn... bằng placeholder
  ``{n}`` để mô hình dịch không làm hỏng chúng, rồi khôi phục sau khi dịch.
- Áp dụng bảng thuật ngữ (glossary) Anh–Việt để dịch thuật ngữ nhất quán.
- Tách câu có xét các chữ viết tắt hay gặp trong bài báo (e.g., et al., Fig., Eq.).
- Nhận diện tiêu đề mục chuẩn (Abstract, Introduction, References...).
- Phân loại khối văn bản: đoạn văn cần dịch / công thức / bảng số liệu / mã nguồn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Placeholder
# ---------------------------------------------------------------------------

# Chấp nhận cả "{ 3 }" vì một số mô hình chèn thêm khoảng trắng.
PLACEHOLDER_RE = re.compile(r"\{\s*(\d{1,3})\s*\}")

_GREEK_MATH_CHARS = (
    "\u0370-\u03FF"  # Hy Lạp
    "\u2100-\u214F"  # ký hiệu dạng chữ (ℝ, ℕ, ℓ...)
    "\u2190-\u22FF"  # mũi tên, toán tử
    "\u27C0-\u27EF\u2980-\u2AFF"  # toán tử bổ sung
    "\U0001D400-\U0001D7FF"  # chữ toán học
)
_REL_OPS = "=<>≤≥≈≠≡∝≪≫∼"
_OPERAND = r"[\w′'^()\[\].+\-−*/|]"

# Ngoặc nhọn có sẵn trong văn bản (tập hợp, LaTeX) — để không lẫn với placeholder.
BRACE_RE = re.compile(r"\{[^{}\n]{0,60}\}")
_PROTECT_PATTERNS = [
    r"https?://[^\s<>\"]+",
    r"\bwww\.[^\s<>\"]+",
    r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
    r"\b(?:doi:\s*)?10\.\d{4,9}/[^\s<>\"]+",
    r"\barXiv:\s*\d{4}\.\d{4,5}(?:v\d+)?",
    r"\$[^$\n]{1,200}\$",
    r"\\[A-Za-z]+(?:\{[^{}]*\})*",
    # Trích dẫn dạng số: [1], [2-5], [1, 3, 7]
    r"\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]",
    # Biểu thức có quan hệ: x = 2, p < 0.05, a ≤ b, y = Wx + b
    rf"{_OPERAND}*\s?[{_REL_OPS}]\s?{_OPERAND}+(?:\s?[+\-−*/·×]\s?{_OPERAND}+)*",
    # Token chứa chữ Hy Lạp / ký hiệu toán: α, ∇f, x∈ℝ, 5μm
    rf"[^\s,;]*[{_GREEK_MATH_CHARS}][^\s,;]*",
    # Định danh trong mã nguồn: hàm(), camelCase, snake_case
    r"\b[A-Za-z_]\w*\(\)",
    r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b",
    r"\b[A-Za-z]+_[A-Za-z0-9_]+\b",
]
_PROTECT_RE = re.compile("|".join(f"(?:{p})" for p in _PROTECT_PATTERNS))
_TRAILING_PUNCT = ".,;:"


def _split_trailing(s: str) -> tuple[str, str]:
    """Tách dấu câu cuối không thuộc biểu thức (vd. dấu chấm hết câu sau URL)."""
    tail = ""
    while s:
        c = s[-1]
        if c in _TRAILING_PUNCT or (c == ")" and s.count(")") > s.count("(")) or (
            c == "]" and s.count("]") > s.count("[")
        ):
            tail = c + tail
            s = s[:-1]
        else:
            break
    return s, tail


@dataclass
class Protector:
    """Thay các đoạn không được dịch bằng placeholder và khôi phục lại sau."""

    values: list[str] = field(default_factory=list)

    def token(self, value: str) -> str:
        # Nếu đoạn cần bảo vệ chứa placeholder cũ thì mở rộng ra trước,
        # nhờ vậy giá trị lưu không bao giờ chứa placeholder lồng nhau.
        value = self.restore(value)
        self.values.append(value)
        return "{%d}" % (len(self.values) - 1)

    def restore(self, text: str) -> str:
        def rep(m: re.Match) -> str:
            i = int(m.group(1))
            return self.values[i] if i < len(self.values) else m.group(0)

        return PLACEHOLDER_RE.sub(rep, text)

    def protect_braces(self, text: str) -> str:
        """Phải gọi trước mọi bước tạo placeholder khác."""
        return BRACE_RE.sub(lambda m: self.token(m.group(0)), text)

    def protect_patterns(self, text: str, braces: bool = True) -> str:
        if braces:
            text = self.protect_braces(text)

        def rep(m: re.Match) -> str:
            body, tail = _split_trailing(m.group(0))
            if not body or PLACEHOLDER_RE.fullmatch(body):
                return m.group(0)
            return self.token(body) + tail

        return _PROTECT_RE.sub(rep, text)

    def merge_adjacent(self, text: str) -> str:
        """Gộp các placeholder đứng sát nhau thành một để mô hình dễ giữ nguyên."""
        pat = re.compile(r"\{(\d{1,3})\}([ \t]*(?:[-+*/=,]?[ \t]*)?)\{(\d{1,3})\}")
        while True:
            m = pat.search(text)
            if not m:
                return text
            merged = self.values[int(m.group(1))] + m.group(2) + self.values[int(m.group(3))]
            self.values.append(merged)
            text = text[: m.start()] + "{%d}" % (len(self.values) - 1) + text[m.end() :]


def placeholder_ids(text: str) -> list[int]:
    return sorted(int(x) for x in PLACEHOLDER_RE.findall(text))


def has_translatable_text(text: str) -> bool:
    """Còn từ tiếng Anh nào để dịch sau khi bỏ placeholder không?"""
    stripped = PLACEHOLDER_RE.sub(" ", text)
    return re.search(r"[A-Za-z]{2,}", stripped) is not None


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------


def parse_glossary(text: str) -> dict[str, str]:
    """Đọc glossary dạng mỗi dòng ``term = bản dịch`` (hoặc ngăn cách bằng TAB).

    Dòng bắt đầu bằng ``#`` là chú thích.
    """
    result: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            en, _, vi = line.partition("\t")
        elif "=" in line:
            en, _, vi = line.partition("=")
        else:
            continue
        en, vi = en.strip(), vi.strip()
        if en and vi:
            result[en] = vi
    return result


def load_builtin_glossary() -> dict[str, str]:
    path = DATA_DIR / "glossary_en_vi.txt"
    if not path.exists():
        return {}
    return parse_glossary(path.read_text(encoding="utf-8"))


# Nhãn đánh số trong bài báo: "Fig. 3", "Table 2", "Theorem 1"...
_LABEL_TERMS = {
    "fig.", "figs.", "figure", "figures", "table", "tables", "eq.", "eqs.", "equation", "equations",
    "sec.", "section", "sections", "theorem", "theorems", "lemma", "lemmas", "corollary",
    "proposition", "definition", "algorithm", "algorithms", "appendix", "remark",
}


# Chỉ dịch theo glossary khi đi kèm số ("Figure 2"), vì "figures of merit",
# "look-up table"... mang nghĩa khác.
_LABEL_ONLY = {"figure", "figures", "fig.", "figs.", "table", "tables"}


def _variants(term: str) -> set[str]:
    v = {term}
    if term[-1:].isalpha() and term[-1:].islower() and not term.endswith("."):
        v.add(term + "s")
        if term.endswith(("s", "x", "ch", "sh")):
            v.add(term + "es")
        if term.endswith("y") and len(term) > 2 and term[-2] not in "aeiou":
            v.add(term[:-1] + "ies")
    return v


class Glossary:
    def __init__(self, entries: dict[str, str]):
        self.entries = dict(entries)
        lookup: dict[str, str] = {}
        for en, vi in self.entries.items():
            for var in _variants(en):
                lookup.setdefault(var.lower(), vi)
        self._lookup = lookup
        generic = [k for k in lookup if k not in _LABEL_ONLY]
        if generic:
            alts = sorted(generic, key=len, reverse=True)
            body = "|".join(re.escape(a) for a in alts)
            self._re = re.compile(rf"(?<![\w{{\-])(?:{body})(?![\w}}\-])", re.IGNORECASE)
        else:
            self._re = None
        labels = sorted((k for k in lookup if k in _LABEL_TERMS), key=len, reverse=True)
        self._label_re = (
            re.compile(
                rf"(?<![\w{{])({'|'.join(re.escape(k) for k in labels)})\s*"
                r"(\(?[A-Z]?\d+(?:\.\d+)*[a-z]?\)?(?:\s*(?:and|,|–|-)\s*\(?\d+(?:\.\d+)*[a-z]?\)?)*)(?![\w}])",
                re.IGNORECASE,
            )
            if labels
            else None
        )

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _match_case(src: str, vi: str) -> str:
        letters = [c for c in src if c.isalpha()]
        if len(letters) > 1 and all(c.isupper() for c in letters) and src.lower() != src:
            # Không viết hoa toàn bộ nếu thuật ngữ gốc vốn là viết tắt (vd. CNN)
            return vi.upper() if any(c.islower() for c in vi) and len(letters) > 4 else vi
        if src[:1].isupper() and vi[:1].islower():
            return vi[:1].upper() + vi[1:]
        return vi

    def protect(self, text: str, protector: Protector) -> str:
        if self._label_re is not None:
            # "Fig. 3", "Table 2", "Eq. (1)" → một placeholder duy nhất "Hình 3"...
            def rep_label(m: re.Match) -> str:
                vi = self._match_case(m.group(1), self._lookup[m.group(1).lower()])
                num, tail = _split_trailing(m.group(2))
                num = re.sub(r"\s+and\s+", " và ", num)
                return protector.token(f"{vi} {num}") + tail

            text = self._label_re.sub(rep_label, text)
        if self._re is None:
            return text

        def rep(m: re.Match) -> str:
            src = m.group(0)
            vi = self._lookup[src.lower()]
            return protector.token(self._match_case(src, vi))

        return self._re.sub(rep, text)

    def relevant(self, text: str) -> dict[str, str]:
        """Các mục glossary xuất hiện trong ``text`` (dùng làm gợi ý cho LLM)."""
        if self._re is None:
            return {}
        out: dict[str, str] = {}
        for m in self._re.finditer(text):
            out[m.group(0)] = self._lookup[m.group(0).lower()]
        return out


# ---------------------------------------------------------------------------
# Tiêu đề mục chuẩn
# ---------------------------------------------------------------------------

SECTION_HEADINGS = {
    "abstract": "Tóm tắt",
    "introduction": "Giới thiệu",
    "background": "Cơ sở lý thuyết",
    "preliminaries": "Kiến thức chuẩn bị",
    "related work": "Các công trình liên quan",
    "related works": "Các công trình liên quan",
    "literature review": "Tổng quan tài liệu",
    "method": "Phương pháp",
    "methods": "Phương pháp",
    "methodology": "Phương pháp luận",
    "materials and methods": "Vật liệu và phương pháp",
    "experimental setup": "Thiết lập thực nghiệm",
    "experiments": "Thực nghiệm",
    "experiment": "Thực nghiệm",
    "evaluation": "Đánh giá",
    "results": "Kết quả",
    "results and discussion": "Kết quả và thảo luận",
    "discussion": "Thảo luận",
    "analysis": "Phân tích",
    "limitations": "Hạn chế",
    "conclusion": "Kết luận",
    "conclusions": "Kết luận",
    "conclusion and future work": "Kết luận và hướng phát triển",
    "conclusions and future work": "Kết luận và hướng phát triển",
    "future work": "Hướng phát triển",
    "acknowledgment": "Lời cảm ơn",
    "acknowledgments": "Lời cảm ơn",
    "acknowledgement": "Lời cảm ơn",
    "acknowledgements": "Lời cảm ơn",
    "references": "Tài liệu tham khảo",
    "bibliography": "Tài liệu tham khảo",
    "appendix": "Phụ lục",
    "appendices": "Phụ lục",
    "supplementary material": "Tài liệu bổ sung",
    "keywords": "Từ khóa",
    "index terms": "Từ khóa",
    "contents": "Mục lục",
    "table of contents": "Mục lục",
}

_HEADING_NUM_RE = re.compile(
    r"^\s*((?:[0-9]+(?:\.[0-9]+)*\.?|[IVXLC]+\.|[A-Z]\.|Appendix\s+[A-Z0-9]+[.:]?)\s+)?(.*?)\s*[.:]?\s*$"
)
REFERENCES_RE = re.compile(
    r"^\s*(?:[0-9]+\.?|[IVXLC]+\.)?\s*(references|bibliography|literature cited|works cited|reference list)\s*$",
    re.IGNORECASE,
)
APPENDIX_RE = re.compile(
    r"^\s*(?:[A-Z0-9]+\.?\s+)?(appendix|appendices|supplementary material)\b", re.IGNORECASE
)


def translate_heading(text: str) -> str | None:
    """Dịch nhanh tiêu đề mục chuẩn (giữ nguyên số thứ tự và kiểu viết hoa)."""
    if len(text) > 60:
        return None
    m = _HEADING_NUM_RE.match(text)
    if not m:
        return None
    prefix, title = m.group(1) or "", m.group(2)
    vi = SECTION_HEADINGS.get(title.lower())
    if vi is None:
        return None
    if title.isupper() and len(title) > 1:
        vi = vi.upper()
    return prefix + vi


# ---------------------------------------------------------------------------
# Tách câu
# ---------------------------------------------------------------------------

_ABBREVIATIONS = {
    "e.g", "i.e", "et al", "al", "etc", "fig", "figs", "eq", "eqs", "vs", "cf", "resp",
    "approx", "no", "nos", "vol", "pp", "p", "dr", "mr", "mrs", "ms", "prof", "sec",
    "ref", "refs", "tab", "ch", "st", "jr", "inc", "ltd", "co", "viz", "ca", "w.r.t",
    "wrt", "thm", "def", "prop", "lem", "cor", "alg", "app", "approx", "max", "min",
    "dept", "univ", "est", "avg", "std", "var", "corr", "eqn", "sect", "chap",
}
_SPLIT_RE = re.compile(r"(?<=[.!?])[\"”’)]?\s+(?=[\"“(\[{]?[A-Z0-9{])")


def split_sentences(text: str, max_len: int = 600) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts: list[str] = []
    last = 0
    for m in _SPLIT_RE.finditer(text):
        candidate = text[last : m.start()].rstrip()
        word_m = re.search(r"(\S+)[.!?][\"”’)]?$", candidate)
        prev = word_m.group(1).lower() if word_m else ""
        prev_clean = prev.lstrip("([\"“")
        if (
            prev_clean in _ABBREVIATIONS
            or (len(prev_clean) == 1 and prev_clean.isalpha())  # chữ cái đầu tên: J. Smith
            or re.fullmatch(r"[a-z](?:\.[a-z])+", prev_clean)  # i.e / e.g dạng khác
        ):
            continue
        parts.append(text[last : m.end()].strip())
        last = m.end()
    parts.append(text[last:].strip())
    parts = [p for p in parts if p]

    # Câu quá dài: cắt tiếp theo dấu ';' rồi ',' để không vượt giới hạn mô hình.
    out: list[str] = []
    for p in parts:
        out.extend(_split_long(p, max_len))
    return out


def _split_long(sentence: str, max_len: int) -> list[str]:
    if len(sentence) <= max_len:
        return [sentence]
    for sep in ("; ", ", ", " "):
        pieces = sentence.split(sep)
        if len(pieces) < 2:
            continue
        chunks: list[str] = []
        cur = ""
        for i, piece in enumerate(pieces):
            add = piece + (sep.rstrip() if i < len(pieces) - 1 else "")
            if cur and len(cur) + len(add) + 1 > max_len:
                chunks.append(cur.strip())
                cur = add
            else:
                cur = f"{cur} {add}" if cur else add
        if cur.strip():
            chunks.append(cur.strip())
        if all(len(c) <= max_len for c in chunks):
            return chunks
    return [sentence[i : i + max_len] for i in range(0, len(sentence), max_len)]


# ---------------------------------------------------------------------------
# Nối dòng / phân loại khối
# ---------------------------------------------------------------------------


def join_lines(lines: list[str]) -> str:
    """Nối các dòng của một đoạn, xử lý từ bị ngắt bằng gạch nối cuối dòng."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif re.search(r"[A-Za-z]-$", out) and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out} {line}"
    return re.sub(r"[ \t]+", " ", out)


MATH_FONT_RE = re.compile(
    r"CMMI|CMSY|CMEX|CMBSY|MSAM|MSBM|EUFM|EUSM|EUEX|RSFS|wasy|stmary|esint|"
    # Lưu ý: "STIX Two Text" / "STIXGeneral" là font CHỮ thường (Wiley, Elsevier...) —
    # chỉ các biến thể Math/Size/Integrals... mới là font toán.
    r"Symbol|Math|STIX\w*(?:Size|Integrals|NonUnicode|Variants)|TeXGyre.*Math|"
    r"MTMI|MTSY|MTEX|txmi|txsy|txex|pxmi|pxsy|rtxmi|ntxmi|LMMath|Cambria.?Math|"
    r"Euclid(?:Symbol|Math|Extra|Fraktur)|MathJax|KaTeX_(?!Main)",
    re.IGNORECASE,
)
MONO_FONT_RE = re.compile(
    r"Courier|Mono|CMTT|SFTT|Consolas|Inconsolata|Menlo|Typewriter|txtt|Code", re.IGNORECASE
)


def is_math_font(fontname: str) -> bool:
    return bool(MATH_FONT_RE.search(fontname or ""))


def has_prose(text: str) -> bool:
    """Đoạn chủ yếu là từ tiếng Anh thật (≥3 từ) → là chữ, không phải công thức.

    Dùng khi font bị nhận nhầm là font toán; công thức kiểu ``softmax(QK^T)`` vẫn là công thức.
    """
    words = re.findall(r"\b[A-Za-z][a-z]{2,}\b(?!\s*\()", text)
    non_space = sum(1 for c in text if not c.isspace())
    return len(words) >= 3 and sum(map(len, words)) / max(non_space, 1) > 0.6


def is_mono_font(fontname: str) -> bool:
    return bool(MONO_FONT_RE.search(fontname or ""))


def looks_mathy(text: str) -> bool:
    """Đoạn ký tự gần như toàn ký hiệu toán (dùng cho span không có font toán)."""
    t = text.strip()
    if not t:
        return False
    sym = sum(1 for c in t if re.match(rf"[{_GREEK_MATH_CHARS}{_REL_OPS}+\-−×·/^_∑∫]", c))
    return sym / len(t) > 0.5


@dataclass
class BlockStats:
    text: str
    total_chars: int
    math_chars: int
    mono_chars: int


def classify_block(stats: BlockStats) -> str:
    """Trả về ``"text"`` (cần dịch) hoặc lý do bỏ qua: math/code/data/empty."""
    text = stats.text.strip()
    if not text:
        return "empty"
    non_space = [c for c in text if not c.isspace()]
    letters = sum(1 for c in non_space if c.isalpha() and c.isascii())
    if stats.total_chars and stats.mono_chars / stats.total_chars > 0.6:
        return "code"
    if stats.total_chars and stats.math_chars / stats.total_chars > 0.4:
        return "math"
    words = re.findall(r"[A-Za-z]{2,}", text)
    if letters < 3 or not words:
        return "data"
    if letters / max(len(non_space), 1) < 0.45:
        # Phần lớn là số / ký hiệu: bảng số liệu, công thức
        return "data"
    # Ô bảng ngắn chỉ gồm 1 từ viết tắt / đơn vị: giữ nguyên
    if len(words) == 1 and len(text) <= 6 and not text[:1].isupper():
        return "data"
    return "text"
