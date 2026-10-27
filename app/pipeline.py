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
    _variants,
    has_translatable_text,
    load_builtin_glossary,
    placeholder_ids,
    split_sentences,
    translate_heading,
)
from .config import settings
from .quality import QE, chrf, combine, is_degenerate, rule_score
from .translators import Translator, get_cache

log = logging.getLogger(__name__)

# Ký tự đánh dấu do pdf_processor chèn quanh span công thức / chỉ số trên.
MATH_START, MATH_END = "\ue000", "\ue001"
SUP_START, SUP_END = "\ue002", "\ue003"
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


def _match_first_letter(src: str, out: str) -> str:
    """Chữ cái đầu của bản dịch viết hoa nếu câu gốc viết hoa (thuật ngữ thay vào có thể đứng đầu)."""
    s = next((c for c in src if c.isalpha()), "")
    for i, c in enumerate(out):
        if c.isalpha():
            if s.isupper() and c.islower():
                return out[:i] + c.upper() + out[i + 1:]
            return out
    return out


_MATH_PH_START, _MATH_PH_END = "\ue010", "\ue011"  # chỗ giữ công thức trong lúc tách câu
_MATH_PH_RE = re.compile(f"{_MATH_PH_START}(\\d+){_MATH_PH_END}")
_TERM_TOKEN = "ZQX{}"
_TERM_TOKEN_RE = re.compile(r"ZQX[A-Z]")


_TITLE_FUNCTION_WORDS = {
    "of", "for", "from", "with", "in", "on", "to", "the", "a", "an", "via", "using", "by", "towards",
    "toward", "into", "under", "over", "through", "based", "is", "are",
}
_NUMBERED_HEADING_RE = re.compile(r"^(\s*(?:\d+(?:\.\d+)*\.?|[IVX]+\.|[A-Z]\.)\s+)(\S.*)$")
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
    # Tiêu đề mục có đánh số ("2.2.4 Loss Function Design") chắc chắn là tiêu đề, kể cả khi
    # không có từ chức năng: không chuẩn hóa thì mô hình dịch từng từ ("Thiết kế chức năng mất").
    def lower_title(t: str) -> str:
        def lower_word(m: re.Match) -> str:
            w = m.group(0)
            if _ACRONYM_RE.match(w) or not (w[0].isupper() and w[1:] == w[1:].lower()):
                return w
            return w.lower()

        out = re.sub(r"[A-Za-z][\w’']*", lower_word, t)
        return out[:1].upper() + out[1:]

    caps_ratio = sum(w[0].isupper() for w in words) / len(words) if words else 0.0
    numbered = _NUMBERED_HEADING_RE.match(text)
    if numbered and 2 <= len(words) <= 12 and "," not in text and caps_ratio >= 0.7:
        return numbered.group(1) + lower_title(numbered.group(2)), "title"
    # Chỉ áp dụng cho tiêu đề ngắn; danh sách địa chỉ/tên riêng (nhiều dấu phẩy) giữ nguyên
    # để không biến "China" thành "china".
    # Tiêu đề hầu như luôn có từ chức năng ("for", "of", "from"...); danh sách tên tác giả thì không.
    has_function_word = any(w.lower() in _TITLE_FUNCTION_WORDS for w in words)
    if 3 <= len(words) <= 25 and text.count(",") <= 1 and has_function_word and caps_ratio >= 0.7:
        return lower_title(text), "title"
    return text, ""


def restore_case(text: str, mode: str) -> str:
    if mode == "upper":
        return text.upper()
    if mode == "title":
        # Viết hoa chữ cái đầu tiên (bỏ qua số mục "2.2.4 " đứng trước).
        return re.sub(r"^([\d.\s]*)(\w)", lambda m: m.group(1) + m.group(2).upper(), text, count=1)
    return text




def postprocess_vi(text: str) -> str:
    text = re.sub(f"[{MATH_START}{MATH_END}]", "", text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[]) ", r"\1", text)
    return text.strip()

_EN_FUNCTION_WORDS = set(
    """the of and is are was were with for that this which these those from by into than then
    have has had been being will would should could can may might not but also such their our we they
    it its as at on in to an or if when while where whose because however therefore""".split()
)
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_CITE_RE = re.compile(r"\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]")


class TextTranslator:
    """Dịch danh sách đoạn văn khoa học bằng một engine bất kỳ."""

    def __init__(
        self,
        engine: Translator,
        glossary: Glossary | None = None,
        use_cache: bool = True,
        post_editor: Translator | None = None,
        back_translator: Translator | None = None,
        status: Callable[[str], None] | None = None,
    ):
        self.engine = engine
        self.glossary = glossary if glossary is not None and len(glossary) else None
        self.cache = get_cache() if use_cache else None
        # LLM sửa lại các câu bị chấm điểm thấp (chỉ dùng khi engine chính là mô hình dịch máy).
        self.post_editor = post_editor if post_editor is not None and post_editor is not engine else None
        # Mô hình Việt→Anh để chấm điểm bằng dịch ngược (tùy chọn).
        self.back_translator = back_translator
        self.status = status
        self._math_store: list[str] = []  # công thức nội dòng đang được giữ chỗ
        self._term_src: dict[str, str] = {}  # câu đã thay thuật ngữ → câu gốc
        self._term_tokens: dict[str, dict[str, str]] = {}  # câu đã thay → {mã: tiếng Việt}
        self._builtin_terms = {v.lower() for en in load_builtin_glossary() for v in _variants(en)}
        # Kết quả theo từng đoạn của lần translate_texts gần nhất:
        self.kept_english: list[bool] = []  # có câu phải giữ tiếng Anh (bản dịch hỏng)
        self.scores: list[float] = []  # điểm chất lượng thấp nhất trong đoạn (0..1)
        self.reasons: list[list[str]] = []  # lý do bị trừ điểm
        self.post_edited: list[bool] = []
        self._kept: set[str] = set()

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

        protect_math = settings.nmt_terms == "token"

        def plain(m: re.Match) -> str:
            if m.group(1) is None:
                return _sup_to_plain(m.group(2))
            raw = m.group(1)
            if not protect_math or not raw.strip():
                return raw
            # Công thức đi qua mô hình dưới dạng mã viết tắt (gắn ở _sub_terms) để không bị
            # dịch/làm hỏng; sau khi dịch đổi lại đúng nguyên văn → tìm được để thay LaTeX.
            self._math_store.append(raw.strip())
            lead = " " if raw[:1].isspace() else ""
            tail = " " if raw[-1:].isspace() else ""
            return f"{lead}{_MATH_PH_START}{len(self._math_store) - 1}{_MATH_PH_END}{tail}"

        t = _MARKED_RE.sub(plain, text)
        if not re.search(r"[A-Za-z]{2,}", self._unmath(t)) or not has_translatable_text(self._unmath(t)):
            unit.result = self._unmath(t)
            return unit
        # "ABSTRACT Cortical surface..." / "Keywords: a | b" → dịch nhãn riêng.
        m = _LEAD_LABEL_RE.match(t)
        if m:
            unit.prefix = (translate_heading(m.group(1)) or m.group(1)) + m.group(2)
            t = t[m.end():]
        for seg in split_sentences(t, max_len=400):
            seg, mode = normalize_case(seg)
            seg = self._sub_terms(seg)
            unit.segments.append(seg)
            unit.cases.append(mode)
        return unit

    # -- thuật ngữ cho mô hình dịch máy ---------------------------------------

    def _sub_terms(self, seg: str) -> str:
        """Áp thuật ngữ ngay trong câu nguồn (PDFTR_NMT_TERMS) vì mô hình dịch máy hay dịch
        sai thuật ngữ ghép ("loss function" → "chức năng mất") mà sửa sau không được nữa.

        Chỉ thay thuật ngữ nhiều từ của bảng mặc định và mọi thuật ngữ người dùng tự thêm.
        """
        mode = settings.nmt_terms
        if mode not in ("codeswitch", "token"):
            return self._unmath(seg)
        builtin = self._builtin_terms
        tokens: dict[str, str] = {}

        def rep(m: re.Match) -> str:
            src = m.group(0)
            if " " not in src.strip() and "-" not in src and src.lower() in builtin:
                return src
            # Viết hoa theo vị trí trong câu dịch (sửa sau), không theo câu gốc: "Loss Function
            # Design" → "Thiết kế hàm mất mát" chứ không phải "Thiết kế Hàm mất mát".
            vi = self.glossary._lookup[src.lower()]
            if src.isupper() and len(src) > 1:
                vi = self.glossary._match_case(src, vi)  # viết tắt (CNN, MRI) giữ như bảng
            if mode == "codeswitch":
                return vi
            if len(tokens) >= 26:
                return src
            tok = _TERM_TOKEN.format(chr(65 + len(tokens)))
            tokens[tok] = vi
            return tok

        out = seg
        if self.glossary is not None and self.glossary._re is not None:
            out = self.glossary._re.sub(rep, seg)

        def rep_math(m: re.Match) -> str:
            raw = self._math_store[int(m.group(1))]
            if len(tokens) >= 26:
                return raw
            tok = _TERM_TOKEN.format(chr(65 + len(tokens)))
            tokens[tok] = raw
            return tok

        out = _MATH_PH_RE.sub(rep_math, out)
        original = self._unmath(seg)
        if out != original:
            self._term_src[out] = original
            if tokens:
                self._term_tokens[out] = tokens
        return out

    def _unmath(self, text: str) -> str:
        """Đổi chỗ giữ công thức về nguyên văn công thức."""
        return _MATH_PH_RE.sub(lambda m: self._math_store[int(m.group(1))], text)

    def _restore_terms(self, seg: str, result: str) -> str | None:
        """Đổi mã thuật ngữ về tiếng Việt; None nếu mô hình làm mất/sai mã."""
        tokens = self._term_tokens.get(seg)
        if not tokens:
            return result
        if sorted(_TERM_TOKEN_RE.findall(result)) != sorted(tokens):
            return None
        return _TERM_TOKEN_RE.sub(lambda m: tokens[m.group(0)], result)

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
            self._kept.add(seg)
            return seg
        tr = self._run_engine(parts)
        out = []
        for p in parts:
            t = tr.get(p, p)
            if is_degenerate(p, t):
                self._kept.add(seg)
                out.append(p)
            else:
                out.append(t)
        return " ".join(out)

    def translate_texts(self, texts: list[str], progress: ProgressFn | None = None) -> list[str]:
        self._kept = set()
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
                if s in self._term_tokens:
                    restored = self._restore_terms(s, t)
                    if restored is None:  # mô hình làm mất mã thuật ngữ → dịch câu gốc như thường
                        orig = self._term_src[s]
                        t = self._run_engine([orig]).get(orig, orig)
                    else:
                        t = restored
                if s in self._term_src:
                    t = _match_first_letter(self._term_src[s], t)
                if is_degenerate(s, t):
                    log.info("Bản dịch hỏng, dịch lại từng vế: %r -> %r", s[:80], t[:80])
                    t = self._repair(self._term_src.get(s, s))
                results[s] = t
            done += len(batch)
            if progress:
                progress(done, total)

        seg_qe = self._score_all(all_segs, results)
        edited = self._post_edit(all_segs, results, seg_qe) if self.post_editor else set()

        self.kept_english = [any(s in self._kept for s in u.segments) for u in units]
        self.scores = [min((seg_qe[s].score for s in u.segments), default=1.0) for u in units]
        self.reasons = [sorted({r for s in u.segments for r in seg_qe[s].reasons}) for u in units]
        self.post_edited = [any(s in edited for s in u.segments) for u in units]
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

    # -- chất lượng ---------------------------------------------------------

    def _qe(self, src: str, out: str, back: str | None = None) -> QE:
        src_c = PLACEHOLDER_RE.sub(" ", src)
        out_c = PLACEHOLDER_RE.sub(" ", out)
        if not self.engine.supports_placeholders:
            out_c = self._fix_terms(out_c)
        rule = rule_score(src_c, out_c, self.glossary)
        return combine(rule, chrf(back, src_c) if back is not None else None)

    def _score_all(self, segs: list[str], results: dict[str, str]) -> dict[str, QE]:
        backs: dict[str, str] = {}
        if self.back_translator is not None and segs:
            if self.status:
                self.status("Đang chấm điểm bản dịch (dịch ngược)...")
            outs = [results.get(s, s) for s in segs]
            try:
                backs = dict(zip(segs, self.back_translator.translate_batch(outs)))
            except Exception as e:  # noqa: BLE001  (chấm điểm là phụ, không làm hỏng cả job)
                log.warning("Không dịch ngược được để chấm điểm: %s", e)
        return {s: self._qe(self._term_src.get(s, s), results.get(s, s), backs.get(s)) for s in segs}

    def _post_edit(self, segs: list[str], results: dict[str, str], seg_qe: dict[str, QE]) -> set[str]:
        """Gửi các câu điểm thấp nhất cho LLM sửa; chỉ nhận bản sửa nếu điểm không giảm."""
        low = sorted((s for s in segs if seg_qe[s].score < settings.qe_threshold), key=lambda s: seg_qe[s].score)
        limit = max(1, int(len(segs) * settings.postedit_max_ratio)) if segs else 0
        low = low[:limit]
        edited: set[str] = set()
        for i, s in enumerate(low, 1):
            if self.status:
                self.status(f"LLM đang sửa câu dịch kém: {i}/{len(low)}")
            src = self._term_src.get(s, s)  # LLM xem câu gốc, không xem câu đã thay mã thuật ngữ
            rel = self.glossary.relevant(src) if self.glossary is not None else None
            try:
                new = self.post_editor.post_edit(src, results.get(s, s), rel)
            except Exception as e:  # noqa: BLE001
                log.warning("LLM sửa câu thất bại, giữ bản dịch máy: %s", e)
                break
            if not new or placeholder_ids(new) != placeholder_ids(src):
                continue
            q = self._qe(src, new)
            if q.score >= seg_qe[s].score:
                results[s], seg_qe[s] = new, q
                self._kept.discard(s)
                edited.add(s)
        return edited

    def _fix_terms(self, text: str) -> str:
        """Sau khi dịch: Việt hóa nhãn còn sót (Fig. 3 → Hình 3) và thuật ngữ còn tiếng Anh."""
        glossary = self.glossary or _LABEL_GLOSSARY
        return glossary.apply_to_output(text)
