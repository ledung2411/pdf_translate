"""Bản dễ hiểu cho người không chuyên.

- Có LLM cục bộ: viết lại từng mục bằng tiếng Việt đơn giản (câu ngắn, giải thích thuật ngữ,
  giữ nguyên số liệu, tên riêng, công thức), dựa trên cả đoạn gốc tiếng Anh lẫn bản dịch.
- Không có LLM: giữ bản dịch, chèn lời giải thích ngắn (từ ``data/explain_vi.txt``) cho thuật
  ngữ ở lần xuất hiện đầu tiên và thêm khung "Khái niệm cần biết".
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

from .insight import Section, split_sections, with_latex
from .ir import Document, Region
from .mathtex import find_math
from .pipeline import MATH_END, MATH_START, SUP_END, SUP_START
from .scientific import translate_heading

log = logging.getLogger(__name__)
DATA = Path(__file__).parent / "data" / "explain_vi.txt"

PLAIN_SYSTEM = """Bạn là người viết phổ biến khoa học. Viết lại phần bài báo khoa học dưới đây bằng tiếng Việt dễ hiểu cho người không chuyên (sinh viên năm đầu, người đọc phổ thông).
Quy tắc:
- Giữ đúng ý và mọi con số, tên riêng, tên phương pháp/dataset, trích dẫn như [12], công thức $...$. Không thêm thông tin không có trong bài.
- Câu ngắn, rõ ràng. Giải thích thuật ngữ chuyên môn trong ngoặc ở lần đầu xuất hiện. Có thể thêm một ví dụ so sánh đời thường ngắn nếu giúp hiểu.
- Lược bớt chi tiết kỹ thuật quá sâu (thông số máy, chi tiết cài đặt) nhưng vẫn nói ý chính của chúng là gì.
- Chỉ trả về phần đã viết lại (có thể chia đoạn), không lời dẫn, không tiêu đề."""


@lru_cache(maxsize=1)
def load_explanations() -> list[tuple[str, str, str]]:
    """[(thuật ngữ Việt, thuật ngữ Anh, giải thích)], thuật ngữ dài trước."""
    out = []
    if DATA.exists():
        for line in DATA.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            left, expl = line.split("=", 1)
            vi, _, en = left.partition("|")
            out.append((vi.strip(), en.strip(), expl.strip()))
    return sorted(out, key=lambda t: -len(t[0]))


def vi_text(r: Region) -> str:
    """Bản dịch của vùng (công thức nội dòng dạng LaTeX); vùng chưa dịch → nguyên văn."""
    if r.translation is None:
        return with_latex(r)
    text = r.translation.replace(MATH_START, "").replace(MATH_END, "")
    if r.math_latex:
        out, pos = [], 0
        for a, b, key in find_math(text, list(r.math_latex)):
            out += [text[pos:a], f"${r.math_latex[key]}$"]
            pos = b
        text = "".join(out) + text[pos:]
    return text.replace(SUP_START, "^").replace(SUP_END, "").strip()


def _heading_vi(s: Section, ir: Document) -> str:
    for r in ir.regions:
        if r.kind == "text" and r.plain.strip() == s.title and r.translation:
            return vi_text(r)
    return translate_heading(s.title) or s.title


def explain_terms(text: str, used: set[str]) -> tuple[str, list[tuple[str, str]]]:
    """Chèn giải thích cho thuật ngữ xuất hiện lần đầu; trả về (văn bản, các thuật ngữ đã giải thích)."""
    added: list[tuple[str, str]] = []
    taken: list[tuple[int, int]] = []
    inserts: list[tuple[int, str]] = []
    for vi, _en, expl in load_explanations():
        if vi.lower() in used:
            continue
        m = re.search(r"(?<![\w])" + re.escape(vi) + r"(?![\w])", text, re.I)
        if not m or any(a < m.end() and m.start() < b for a, b in taken):
            continue
        # đã có ngoặc giải thích ngay sau thì thôi
        if text[m.end():m.end() + 2].startswith(" ("):
            continue
        taken.append((m.start(), m.end()))
        inserts.append((m.end(), f" (*{expl}*)"))
        used.add(vi.lower())
        added.append((m.group(0), expl))
    for pos, ins in sorted(inserts, reverse=True):
        text = text[:pos] + ins + text[pos:]
    return text, added


def _chunks(paras: list[tuple[str, str]], limit: int = 2200) -> list[list[tuple[str, str]]]:
    out: list[list[tuple[str, str]]] = []
    size = 0
    for en, vi in paras:
        if out and size + len(en) <= limit:
            out[-1].append((en, vi))
            size += len(en)
        else:
            out.append([(en, vi)])
            size = len(en)
    return out


def make_plain(ir: Document, llm=None, status=None) -> str:
    """Bản dễ hiểu dạng Markdown."""
    from .config import settings
    from .insight import _guess_title

    sections = split_sections(ir)
    title = _guess_title(ir)
    if llm is not None:
        note = (f"_Bản viết lại cho người không chuyên bằng LLM cục bộ `{settings.llm_model}`. Có thể lược bớt chi "
                f"tiết kỹ thuật — khi cần độ chính xác, xem bản dịch đầy đủ._")
    else:
        note = ("_Bản dịch kèm giải thích thuật ngữ cho người không chuyên (chưa có LLM cục bộ để viết lại câu chữ; "
                "bật Ollama/LM Studio để có bản diễn giải đơn giản hơn)._")
    out = [f"# Bản dễ hiểu\n\n**{title}**\n\n{note}\n"]
    used: set[str] = set()
    glossary: list[tuple[str, str]] = []
    total = sum(1 for s in sections for r in s.regions if r.kind == "text")
    done = 0
    # Bỏ phần đầu bài (tên tạp chí, địa chỉ tác giả, ngày nhận bài) trước Tóm tắt/Giới thiệu.
    start = next((i for i, x in enumerate(sections) if x.kind in ("abstract", "intro")), 0)
    sections = sections[start:]
    for s in sections:
        body: list[str] = []
        paras = [(with_latex(r), vi_text(r)) for r in s.regions if r.kind == "text" and len(r.plain.strip()) > 1]
        if llm is not None and paras:
            for chunk in _chunks(paras):
                done += len(chunk)
                if status:
                    status(f"Bản dễ hiểu: {_heading_vi(s, ir)[:40]} ({done}/{total} đoạn)")
                en = "\n\n".join(p[0] for p in chunk)
                vi = "\n\n".join(p[1] for p in chunk)
                try:
                    text = llm._chat(PLAIN_SYSTEM, f"MỤC: {s.title}\n\nĐOẠN GỐC (tiếng Anh):\n{en}\n\n"
                                                   f"BẢN DỊCH MÁY (để tham khảo):\n{vi}")
                except Exception as e:  # noqa: BLE001  (LLM lỗi → giữ bản dịch kèm giải thích)
                    log.warning("LLM lỗi khi viết bản dễ hiểu: %s", e)
                    text, added = explain_terms(vi, used)
                    glossary += added
                body.append(text.strip())
        else:
            for _en, vi in paras:
                text, added = explain_terms(vi, used)
                glossary += added
                body.append(text)
        for r in s.regions:
            if r.kind == "math" and r.latex:
                body.append(f"$${r.latex}$$")
        if body:
            heading = _heading_vi(s, ir) if s.kind != "front" else ""
            out.append((f"## {heading}\n\n" if heading else "") + "\n\n".join(body) + "\n")
    if glossary:
        box = "\n".join(f"- **{t}**: {e}" for t, e in glossary)
        out.insert(1, f"## Khái niệm cần biết\n\n{box}\n")
    return "\n".join(out).strip() + "\n"
