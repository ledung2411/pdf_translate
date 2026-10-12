"""Chấm điểm chất lượng bản dịch từng câu (Quality Estimation) — không cần bản dịch chuẩn.

Điểm 0..1 = 1 − tổng mức phạt từ các luật kiểm tra:
- bản dịch hỏng (lặp, rác)                → 0 điểm
- mất số liệu / trích dẫn của câu gốc
- độ dài bất thường so với câu gốc
- còn sót nhiều từ tiếng Anh (chưa dịch)
- không dùng thuật ngữ trong bảng thuật ngữ

Tùy chọn (``PDFTR_QE_BACKTRANS=1``): dịch ngược Việt→Anh bằng mô hình cục bộ và so độ
giống với câu gốc (chrF), trộn 50/50 với điểm luật.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .scientific import Glossary

LOW_QUALITY = 0.6

_EN_FUNCTION_WORDS = set(
    """the of and is are was were with for that this which these those from by into than then
    have has had been being will would should could can may might not but also such their our we they
    it its as at on in to an or if when while where whose because however therefore""".split()
)
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_CITE_RE = re.compile(r"\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]")


def is_degenerate(src: str, out: str) -> bool:
    """Phát hiện bản dịch hỏng: lặp vô hạn, toàn gạch dưới, dài bất thường."""
    if not out.strip():
        return True
    if "___" in out or len(out) > 3 * len(src) + 40:
        return True
    words = out.split()
    # Một từ lặp liền 3 lần ("model model model") gần như chắc chắn là lỗi sinh lặp.
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a.lower() == b.lower() and any(c.isalpha() for c in a) else 1
        if run >= 3:
            return True
    if len(words) >= 6:
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


@dataclass
class QE:
    score: float
    reasons: list[str] = field(default_factory=list)


def _norm_num(n: str) -> str:
    # Bỏ dấu phân cách: "24.1" ≡ "24,1" (thập phân kiểu Việt), "1000" ≡ "1.000"
    return re.sub(r"[.,]", "", n)


def rule_score(src: str, out: str, glossary: Glossary | None = None) -> QE:
    if is_degenerate(src, out):
        return QE(0.0, ["hỏng/lặp"])
    penalty = 0.0
    reasons: list[str] = []

    nums = [_norm_num(n) for n in _NUM_RE.findall(src)]
    if nums:
        have = Counter(_norm_num(n) for n in _NUM_RE.findall(out))
        missing = 0
        for n in nums:
            if have[n] > 0:
                have[n] -= 1
            else:
                missing += 1
        if missing:
            penalty += 0.4 * missing / len(nums)
            reasons.append(f"mất {missing} số liệu")

    cites = _CITE_RE.findall(src)
    if cites:
        out_digits = re.sub(r"\s", "", out)
        lost = [c for c in cites if re.sub(r"\s", "", c) not in out_digits]
        if lost:
            penalty += 0.2
            reasons.append("mất trích dẫn")

    if len(src) >= 20:
        ratio = len(out) / len(src)
        if ratio < 0.5 or ratio > 2.6:
            penalty += 0.3
            reasons.append(f"độ dài bất thường ({ratio:.1f}x)")

    words = re.findall(r"[A-Za-zÀ-ỹđĐ]+", out)
    if words:
        english = sum(1 for w in words if w.lower() in _EN_FUNCTION_WORDS)
        if english >= 2 and english / len(words) > 0.08:
            penalty += min(0.5, english / len(words) * 2)
            reasons.append("còn sót tiếng Anh")

    if glossary is not None:
        expected = glossary.relevant(src)
        miss = [vi for vi in set(expected.values()) if vi.lower() not in out.lower()]
        if miss:
            penalty += min(0.3, 0.1 * len(miss))
            reasons.append("sai thuật ngữ: " + ", ".join(miss[:3]))

    return QE(max(0.0, round(1.0 - penalty, 3)), reasons)


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """chrF (Popović 2015) — độ giống ký tự n-gram, 0..1."""
    hyp, ref = re.sub(r"\s+", " ", hyp.lower()).strip(), re.sub(r"\s+", " ", ref.lower()).strip()
    if not hyp or not ref:
        return 0.0
    precs, recs = [], []
    for k in range(1, n + 1):
        h = Counter(hyp[i : i + k] for i in range(len(hyp) - k + 1))
        r = Counter(ref[i : i + k] for i in range(len(ref) - k + 1))
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precs.append(overlap / sum(h.values()))
        recs.append(overlap / sum(r.values()))
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + r == 0:
        return 0.0
    return (1 + beta**2) * p * r / (beta**2 * p + r)


def combine(rule: QE, back_sim: float | None) -> QE:
    if back_sim is None:
        return rule
    score = round(0.5 * rule.score + 0.5 * min(1.0, back_sim / 0.6), 3)
    reasons = list(rule.reasons)
    if back_sim < 0.35:
        reasons.append(f"dịch ngược khác nhiều (chrF {back_sim:.2f})")
    return QE(min(rule.score, score) if rule.score == 0 else score, reasons)
