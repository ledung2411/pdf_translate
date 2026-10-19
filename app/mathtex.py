"""Dựng lại công thức dạng LaTeX (và HTML) từ các ký tự trong PDF.

PDF không lưu công thức mà chỉ lưu từng ký tự với font, cỡ chữ và vị trí. Cấu trúc được
suy ra từ hình học:

- chỉ số trên / dưới: ký tự nhỏ hơn hẳn, đường chân chữ cao hơn / thấp hơn ký tự trước;
- phân số: nét ngang có chữ ở cả trên và dưới; căn: dấu √ kèm nét ngang phía trên;
- kiểu chữ: font (CMMI/chữ nghiêng toán = biến, CMR = chữ đứng, MSBM = ℝ, CMSY = 𝒜…)
  hoặc tên ký tự Unicode (𝑥 = MATHEMATICAL ITALIC SMALL X).

Kết quả là LaTeX để xuất Word/Markdown và HTML (sub/sup) để vẽ lại trong PDF dịch.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Bảng ký hiệu
# ---------------------------------------------------------------------------

_GREEK = {
    "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma", "delta": r"\delta", "epsilon": r"\epsilon",
    "zeta": r"\zeta", "eta": r"\eta", "theta": r"\theta", "iota": r"\iota", "kappa": r"\kappa",
    "lamda": r"\lambda", "lambda": r"\lambda", "mu": r"\mu", "nu": r"\nu", "xi": r"\xi", "omicron": "o",
    "pi": r"\pi", "rho": r"\rho", "final sigma": r"\varsigma", "sigma": r"\sigma", "tau": r"\tau",
    "upsilon": r"\upsilon", "phi": r"\phi", "chi": r"\chi", "psi": r"\psi", "omega": r"\omega",
}
_GREEK_UPPER_LATIN = {"alpha": "A", "beta": "B", "epsilon": "E", "zeta": "Z", "eta": "H", "iota": "I",
                      "kappa": "K", "mu": "M", "nu": "N", "omicron": "O", "rho": "P", "tau": "T", "chi": "X"}
_GREEK_VARIANTS = {"ϵ": r"\epsilon", "ε": r"\varepsilon", "ϑ": r"\vartheta", "ϕ": r"\phi", "φ": r"\varphi",
                   "ϖ": r"\varpi", "ϱ": r"\varrho", "ς": r"\varsigma", "ϰ": r"\varkappa"}

_SYMBOLS = {
    "∈": r"\in", "∉": r"\notin", "∋": r"\ni", "×": r"\times", "·": r"\cdot", "⋅": r"\cdot", "∙": r"\cdot",
    "⊙": r"\odot", "⊗": r"\otimes", "⊕": r"\oplus", "⊖": r"\ominus", "∘": r"\circ", "∗": "*", "⋆": r"\star",
    "≤": r"\leq", "≥": r"\geq", "⩽": r"\leq", "⩾": r"\geq", "≠": r"\neq", "≈": r"\approx", "≃": r"\simeq",
    "≅": r"\cong", "≡": r"\equiv", "∼": r"\sim", "∝": r"\propto", "≪": r"\ll", "≫": r"\gg", "≜": r"\triangleq",
    "≔": ":=", "→": r"\to", "←": r"\leftarrow", "↔": r"\leftrightarrow", "↦": r"\mapsto", "⇒": r"\Rightarrow",
    "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow", "⟶": r"\longrightarrow", "↑": r"\uparrow", "↓": r"\downarrow",
    "∞": r"\infty", "∂": r"\partial", "∇": r"\nabla", "∑": r"\sum", "∏": r"\prod", "∫": r"\int", "∮": r"\oint",
    "√": r"\sqrt", "±": r"\pm", "∓": r"\mp", "∀": r"\forall", "∃": r"\exists", "∄": r"\nexists",
    "∩": r"\cap", "∪": r"\cup", "⊂": r"\subset", "⊆": r"\subseteq", "⊃": r"\supset", "⊇": r"\supseteq",
    "∅": r"\emptyset", "∧": r"\wedge", "∨": r"\vee", "¬": r"\neg", "∥": r"\|", "‖": r"\|", "∣": r"\mid",
    "⟨": r"\langle", "⟩": r"\rangle", "〈": r"\langle", "〉": r"\rangle", "⌊": r"\lfloor", "⌋": r"\rfloor",
    "⌈": r"\lceil", "⌉": r"\rceil", "…": r"\ldots", "⋯": r"\cdots", "⋮": r"\vdots", "⋱": r"\ddots",
    "′": "'", "″": "''", "‴": "'''", "−": "-", "–": "-", "†": r"\dagger", "‡": r"\ddagger", "ℓ": r"\ell",
    "ℏ": r"\hbar", "ħ": r"\hbar", "∠": r"\angle", "△": r"\triangle", "⊤": r"\top", "⊺": r"\top",
    "⊥": r"\perp", "÷": r"\div", "°": r"^{\circ}", "∖": r"\setminus", "⊎": r"\uplus", "□": r"\square",
    "{": r"\{", "}": r"\}", "%": r"\%", "#": r"\#", "&": r"\&", "_": r"\_", "$": r"\$", "~": r"\sim",
    "ˆ": r"\hat{}", "˜": r"\tilde{}", "¯": r"\bar{}", "ı": r"\imath", "ȷ": r"\jmath", "ℵ": r"\aleph",
}
_LETTERLIKE = {
    "ℝ": ("bb", "R"), "ℕ": ("bb", "N"), "ℤ": ("bb", "Z"), "ℚ": ("bb", "Q"), "ℂ": ("bb", "C"), "ℙ": ("bb", "P"),
    "ℍ": ("bb", "H"), "ℒ": ("cal", "L"), "ℱ": ("cal", "F"), "ℋ": ("cal", "H"), "ℐ": ("cal", "I"),
    "ℛ": ("cal", "R"), "ℬ": ("cal", "B"), "ℰ": ("cal", "E"), "ℳ": ("cal", "M"), "ℊ": ("cal", "g"),
    "ℯ": ("cal", "e"), "ℴ": ("cal", "o"), "ℎ": ("it", "h"), "ℭ": ("frak", "C"), "ℌ": ("frak", "H"),
    "ℑ": ("frak", "I"), "ℜ": ("frak", "R"), "ℨ": ("frak", "Z"),
}
_STYLE_CMD = {"rm": r"\mathrm", "bf": r"\mathbf", "bb": r"\mathbb", "cal": r"\mathcal", "frak": r"\mathfrak",
              "tt": r"\mathtt", "sf": r"\mathsf", "bfit": r"\boldsymbol"}
_OPERATORS = {"sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "lg", "exp", "max", "min", "sup", "inf",
              "lim", "det", "arg", "tanh", "sinh", "cosh", "dim", "ker", "deg", "gcd", "Pr", "liminf", "limsup"}
_BIG_OPS = {r"\sum", r"\prod", r"\int", r"\oint"}

# Font toán theo tên (pdfTeX, STIX, Cambria...)
_FONT_BB = re.compile(r"msbm|bbold|doublestruck|dsrom|bbm|stixdoublestruck", re.I)
_FONT_CAL = re.compile(r"^(?:[A-Z]{6}\+)?(cmsy|rsfs|eusm|eus\d|.*script|.*calligr)", re.I)
_FONT_FRAK = re.compile(r"eufm|eufb|fraktur", re.I)
_FONT_TT = re.compile(r"cmtt|mono|courier", re.I)
_FONT_BOLD = re.compile(r"cmbx|cmmib|bold|black|heavy", re.I)
_FONT_ITALIC = re.compile(r"cmmi|italic|oblique|cmti", re.I)
_FONT_RADICAL = re.compile(r"cmex|stixsize|size\d|symbol", re.I)


class UnknownGlyph(ValueError):
    """Ký tự riêng của font (vùng Private Use) — không biết là chữ gì."""


@dataclass
class Char:
    """Một ký tự (hoặc một khối đã dựng sẵn như phân số) trong công thức."""

    c: str
    font: str
    size: float
    x0: float
    y0: float
    x1: float
    y1: float
    oy: float  # đường chân chữ (origin y)
    flags: int = 0
    tex: str | None = None  # khối dựng sẵn: \frac{..}{..}, \sqrt{..}
    html: str | None = None

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


def chars_from_span(span: dict) -> list[Char]:
    """Ký tự của một span ``rawdict`` (có "chars") hoặc ``dict`` (chia đều bề ngang)."""
    font, size, flags = span.get("font", ""), float(span.get("size", 10)), int(span.get("flags", 0))
    out: list[Char] = []
    if "chars" in span:
        for ch in span["chars"]:
            x0, y0, x1, y1 = ch["bbox"]
            out.append(Char(ch["c"], font, size, x0, y0, x1, y1, ch["origin"][1], flags))
        return out
    text = span.get("text", "")
    x0, y0, x1, y1 = span["bbox"]
    w = (x1 - x0) / max(len(text), 1)
    for i, c in enumerate(text):
        out.append(Char(c, font, size, x0 + i * w, y0, x0 + (i + 1) * w, y1, span["origin"][1], flags))
    return out


# ---------------------------------------------------------------------------
# Một ký tự → (lệnh LaTeX, kiểu chữ, HTML)
# ---------------------------------------------------------------------------


def _font_style(font: str, flags: int) -> str:
    if _FONT_BB.search(font):
        return "bb"
    if _FONT_FRAK.search(font):
        return "frak"
    if _FONT_CAL.search(font):
        return "cal"
    if _FONT_TT.search(font):
        return "tt"
    bold = bool(_FONT_BOLD.search(font)) or bool(flags & 16)
    italic = bool(_FONT_ITALIC.search(font)) or bool(flags & 2)
    if bold:
        return "bfit" if italic else "bf"
    return "it" if italic else "rm"


def classify(ch: Char) -> tuple[str, str, str]:
    """(LaTeX, loại, HTML). Loại: it/rm/bf/bb/cal/... cho chữ cái, "dig", "sym", "op" (∑, ∫), "sp"."""
    c = ch.c
    if not c.strip():
        return "", "sp", ""
    if "\ue000" <= c <= "\uf8ff":
        raise UnknownGlyph(c)
    esc = html.escape(c, quote=False)
    if c in _LETTERLIKE:
        style, base = _LETTERLIKE[c]
        return base, style, esc
    if c in _GREEK_VARIANTS:
        return _GREEK_VARIANTS[c], "sym", f"<i>{esc}</i>"
    if c in _SYMBOLS:
        cmd = _SYMBOLS[c]
        return cmd, ("op" if cmd in _BIG_OPS else "sym"), esc
    if c.isascii():
        if c.isalpha():
            style = _font_style(ch.font, ch.flags)
            if style == "cal" and not c.isupper():
                style = "it"  # CMSY chỉ có chữ hoa dạng thư pháp
            return c, style, (f"<i>{esc}</i>" if style in ("it", "bfit") else esc)
        if c.isdigit():
            return c, "dig", esc
        return c, "sym", esc
    try:
        name = unicodedata.name(c)
    except ValueError:
        return c, "sym", esc
    m = re.match(r"MATHEMATICAL ((?:BOLD |ITALIC |SCRIPT |FRAKTUR |DOUBLE-STRUCK |SANS-SERIF |MONOSPACE )+)"
                 r"(CAPITAL |SMALL )?(.+)$", name)
    if m:
        styles, case, rest = m.group(1).split(), (m.group(2) or "").strip(), m.group(3)
        style = ("bb" if "DOUBLE-STRUCK" in styles else "cal" if "SCRIPT" in styles else
                 "frak" if "FRAKTUR" in styles else "tt" if "MONOSPACE" in styles else
                 "sf" if "SANS-SERIF" in styles else
                 "bfit" if "BOLD" in styles and "ITALIC" in styles else "bf" if "BOLD" in styles else
                 "it" if "ITALIC" in styles else "rm")
        if rest.startswith("DIGIT "):
            d = str(["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"].index(rest[6:]))
            return d, "dig", d
        if len(rest) == 1:
            letter = rest if case == "CAPITAL" else rest.lower()
            out_html = f"<i>{letter}</i>" if style in ("it", "bfit") else html.escape(c, quote=False)
            return letter, style, out_html
        g = rest.lower().replace(" symbol", "")
        if rest in ("PARTIAL DIFFERENTIAL",):
            return r"\partial", "sym", "∂"
        if rest == "NABLA":
            return r"\nabla", "sym", "∇"
        if g in _GREEK:
            if case == "CAPITAL":
                cmd = _GREEK_UPPER_LATIN.get(g) or "\\" + g.capitalize().replace("Lamda", "Lambda")
            else:
                cmd = {"epsilon": r"\epsilon" if "SYMBOL" in rest else r"\varepsilon"}.get(g, _GREEK[g])
            return cmd, "sym", f"<i>{unicodedata.lookup(('GREEK CAPITAL LETTER ' if case == 'CAPITAL' else 'GREEK SMALL LETTER ') + g.upper().replace('LAMBDA', 'LAMDA'))}</i>"
        return c, "sym", esc
    m = re.match(r"GREEK (CAPITAL|SMALL) LETTER (.+)$", name)
    if m:
        g = m.group(2).lower()
        if m.group(1) == "CAPITAL":
            cmd = _GREEK_UPPER_LATIN.get(g) or "\\" + g.capitalize().replace("Lamda", "Lambda")
            return cmd, "sym", esc
        if g in _GREEK:
            return _GREEK[g], "sym", f"<i>{esc}</i>"
    return c, "sym", esc


# ---------------------------------------------------------------------------
# Dựng LaTeX
# ---------------------------------------------------------------------------


_SPACED = {"=", "<", ">", "+", r"\in", r"\notin", r"\leq", r"\geq", r"\neq", r"\approx", r"\equiv", r"\sim",
           r"\simeq", r"\propto", r"\times", r"\cdot", r"\odot", r"\otimes", r"\oplus", r"\pm", r"\mp",
           r"\to", r"\mapsto", r"\Rightarrow", r"\Leftrightarrow", r"\subset", r"\subseteq", r"\cup", r"\cap",
           r"\ll", r"\gg", ":=", r"\triangleq", r"\leftarrow"}


def _join(parts: list[str]) -> str:
    """Nối các mẩu LaTeX: cách quanh quan hệ/phép toán, cách sau lệnh khi mẩu sau là chữ cái."""
    out = ""
    for p in parts:
        if not p:
            continue
        if p in _SPACED and out:
            out = out.rstrip() + " " + p + " "
            continue
        if out and re.search(r"\\[A-Za-z]+$", out) and re.match(r"[A-Za-z0-9]", p):
            out += " "
        out += p
    return out.strip()


def _level(ch: Char, base: float, baseline: float) -> str:
    """Tầng của ký tự so với dòng chính: "base", "sup" (chỉ số trên) hay "sub" (chỉ số dưới)."""
    if ch.tex is not None:
        return "base"
    r = ch.size / base if base else 1.0
    d = ch.oy - baseline
    if r >= 0.92:
        return "base"
    if d < -0.1 * base:
        return "sup"
    if r < 0.82:
        # Chữ nhỏ hẳn mà không nhô lên → chỉ số dưới (ở cỡ nhỏ, độ hạ chỉ vài phần chục pt).
        return "sub" if d > -0.02 * base else "base"
    # Nhỏ hơn chút (STIX Two thu nhỏ chỉ số cấp hai ~83%): phải hạ xuống rõ mới là chỉ số dưới.
    return "sub" if d > 0.12 * base else "base"


def _base_of(chars: list[Char]) -> tuple[float, float]:
    """Cỡ chữ chính (nhiều ký tự nhất) và đường chân chữ của nó."""
    weights: dict[float, int] = {}
    for ch in chars:
        if ch.c.strip() and ch.tex is None:
            weights[round(ch.size, 1)] = weights.get(round(ch.size, 1), 0) + 1
    if not weights:
        sizes = [ch.size for ch in chars] or [10.0]
        return max(sizes), sorted(ch.oy for ch in chars)[len(chars) // 2] if chars else 0.0
    total = sum(weights.values())
    big = [s for s in weights if weights[s] >= 0.15 * total]
    base = max(big) if big else max(weights, key=lambda s: (weights[s], s))
    oys = sorted(ch.oy for ch in chars if ch.tex is None and abs(ch.size - base) < 0.15 and ch.c.strip())
    return base, oys[len(oys) // 2]


def _render(chars: list[Char], base: float | None = None, baseline: float | None = None) -> tuple[str, str]:
    """Dựng LaTeX và HTML cho một dãy ký tự cùng một tầng (đệ quy cho chỉ số)."""
    chars = [ch for ch in chars if ch.tex is not None or ch.c.strip()]
    if not chars:
        return "", ""
    if base is None or baseline is None:
        base, baseline = _base_of(chars)
    chars = _attach_accents(chars)
    chars = _attach_limits(chars, base, baseline)
    chars = sorted(chars, key=lambda ch: (round(ch.x0, 1), ch.oy))
    tex: list[str] = []
    out_html: list[str] = []
    word: list[str] = []  # các chữ cái đứng liền nhau cùng kiểu (→ \mathrm{softmax})
    word_style = ""

    def flush_word() -> None:
        nonlocal word, word_style
        if not word:
            return
        w = "".join(word).strip()
        if word_style == "rm" and " " in w:
            tex.append(r"\text{" + re.sub(r"\s+", " ", w) + "}")
        elif word_style == "it":
            tex.append(w)
        elif word_style == "rm" and w in _OPERATORS:
            tex.append("\\" + w)
        elif word_style == "rm" and len(w) > 1 and w in ("argmax", "argmin"):
            tex.append(r"\operatorname{" + w + "}")
        else:
            tex.append(_STYLE_CMD[word_style] + "{" + w + "}")
        word, word_style = [], ""

    i = 0
    while i < len(chars):
        ch = chars[i]
        lvl = _level(ch, base, baseline)
        if lvl != "base":
            group = []
            while i < len(chars) and _level(chars[i], base, baseline) == lvl:
                group.append(chars[i])
                i += 1
            flush_word()
            g_base = max(c.size for c in group)
            g_line = sorted(c.oy for c in group if c.size >= g_base - 0.15)
            g_tex, g_html = _render(group, g_base, g_line[len(g_line) // 2])
            if lvl == "sup" and g_tex and set(g_tex) <= {"'"}:
                tex.append(g_tex)  # x′ → x'
                out_html.append("′" * len(g_tex))
                continue
            mark = "^" if lvl == "sup" else "_"
            if not tex:
                tex.append("{}")
            tex.append(mark + "{" + g_tex + "}")
            out_html.append(f'<span class="{lvl}">{g_html}</span>')
            continue
        i += 1
        if ch.tex is not None:
            flush_word()
            tex.append(ch.tex)
            out_html.append(ch.html or html.escape(ch.tex, quote=False))
            continue
        t, kind, h = classify(ch)
        if kind == "sp":
            if word and word_style == "rm":
                word.append(" ")  # "if otherwise" trong công thức → \text{if otherwise}
            continue
        if kind in _STYLE_CMD or kind == "it":
            prev = chars[i - 2] if i >= 2 else None
            gap = ch.x0 - prev.x1 if prev is not None else 0.0
            if word and (kind != word_style or (gap > 0.2 * ch.size and word_style == "rm")):
                if gap > 0.2 * ch.size and kind == word_style == "rm":
                    word.append(" ")
                else:
                    flush_word()
            word.append(t)
            word_style = kind
            out_html.append(h)
            continue
        flush_word()
        tex.append(t)
        out_html.append(h)
    flush_word()
    # "√d" không có nét ngang phía trên: căn của ký hiệu ngay sau → \sqrt{d}
    k = 0
    while k < len(tex) - 1:
        if tex[k] == r"\sqrt" and tex[k + 1] and not tex[k + 1].startswith(("^", "_")):
            tex[k:k + 2] = [r"\sqrt{" + tex[k + 1] + "}"]
        k += 1
    return _join(tex), "".join(out_html)


_ACCENTS = {"^": r"\hat", "ˆ": r"\hat", "\u0302": r"\hat", "~": r"\tilde", "˜": r"\tilde", "\u0303": r"\tilde",
            "¯": r"\bar", "\u0304": r"\bar", "ˉ": r"\bar", "˙": r"\dot", "\u0307": r"\dot", "¨": r"\ddot",
            "\u0308": r"\ddot", "ˇ": r"\check", "\u030c": r"\check", "→": r"\vec", "\u20d7": r"\vec"}


def _attach_accents(chars: list[Char]) -> list[Char]:
    """Dấu mũ/ngã/gạch đặt trên một chữ (P̂, x̃, x̄) → \\hat{P}..."""
    out = list(chars)
    for acc in [c for c in chars if c.tex is None and c.c in _ACCENTS]:
        if unicodedata.combining(acc.c):
            # Dấu tổ hợp (U+0302...): bề ngang 0, nằm sát mép trái chữ được đánh dấu.
            near = [c for c in out if c is not acc and c.tex is None and c.c.strip() and c.c not in _ACCENTS
                    and c.x0 - 2 <= acc.x0 <= c.x1 + 0.5 and abs(c.oy - acc.oy) < 0.3 * c.size]
            if not near:
                out = [c for c in out if c is not acc]
                continue
            below = [min(near, key=lambda c: (abs(c.x0 - acc.x0), c.x0 < acc.x0 - 0.5))]
        else:
            below = [c for c in out if c is not acc and c.tex is None and c.c.strip() and c.c not in _ACCENTS
                     and c.x0 - 1 <= acc.xc <= c.x1 + 1 and acc.yc < c.yc and c.y0 - acc.y1 < 0.6 * c.size]
        if acc.c == "→" and not below:
            continue
        if len(below) != 1:
            if acc.c in ("~", "→"):
                continue  # dấu ngã / mũi tên bình thường trên dòng
            if not below:
                continue
        target = min(below, key=lambda c: abs(c.xc - acc.xc))
        t, h = _render([target], target.size, target.oy)
        atom = Char("", "", target.size, target.x0, acc.y0, target.x1, target.y1, target.oy,
                    tex=_ACCENTS[acc.c] + "{" + t + "}", html=h + "\u0302" if _ACCENTS[acc.c] == r"\hat" else h)
        out = [c for c in out if c is not acc and c is not target] + [atom]
    return out


def _attach_limits(chars: list[Char], base: float, baseline: float) -> list[Char]:
    """Chữ nhỏ nằm ngay dưới/trên ∑, ∏, min, max, lim → giới hạn: \\min_{x \\in P}, \\sum_{i=1}^{n}."""
    ordered = sorted((c for c in chars if c.tex is not None or c.size >= 0.82 * base), key=lambda c: c.x0)
    ops: list[list[Char]] = []
    k = 0
    while k < len(ordered):
        c = ordered[k]
        if c.tex is None and c.c in ("∑", "∏", "⋃", "⋂") and c.size >= 0.82 * base:
            ops.append([c])
        elif c.tex is None and c.c.isascii() and c.c.isalpha() and _font_style(c.font, c.flags) == "rm":
            # tên toán tử: chữ đứng, liền nhau, cùng đường chân chữ ("min", không lấy "f" nghiêng sau nó)
            j = k + 1
            while j < len(ordered) and ordered[j].tex is None and ordered[j].c.isascii() and ordered[j].c.isalpha() \
                    and _font_style(ordered[j].font, ordered[j].flags) == "rm" \
                    and abs(ordered[j].oy - c.oy) < 0.2 * base and ordered[j].x0 - ordered[j - 1].x1 < 0.3 * base:
                j += 1
            word = "".join(x.c for x in ordered[k:j])
            if word in ("min", "max", "lim", "sup", "inf", "argmin", "argmax", "limsup", "liminf"):
                ops.append(ordered[k:j])
                k = j
                continue
        k += 1
    out = list(chars)
    centres = [(min(c.x0 for c in op) + max(c.x1 for c in op)) / 2 for op in ops]

    def owner(c: Char) -> int:
        return min(range(len(ops)), key=lambda k: abs(c.xc - centres[k]))

    for k_op, op in enumerate(ops):
        x0, x1 = min(c.x0 for c in op), max(c.x1 for c in op)
        top, bottom = min(c.y0 for c in op), max(c.y1 for c in op)
        w = x1 - x0

        def near(c: Char) -> bool:
            return (c not in op and c.size < 0.82 * base and x0 - 0.6 * w - 1 <= c.xc <= x1 + 0.6 * w + 1
                    and owner(c) == k_op)
        under = [c for c in out if near(c) and c.y0 >= bottom - 0.25 * base]
        over = [c for c in out if near(c) and c.y1 <= top + 0.25 * base and c not in under]
        if not under and not over:
            continue
        name = "".join(c.c for c in op)
        head = {"∑": r"\sum", "∏": r"\prod", "⋃": r"\bigcup", "⋂": r"\bigcap"}.get(name)
        head = head or (r"\operatorname*{" + name + "}" if name.startswith("arg") else "\\" + name)
        t = head
        h = html.escape(name, quote=False)
        if under:
            tu, hu = _render(under)
            t += "_{" + tu + "}"
            h += f'<span class="sub">{hu}</span>'
        if over:
            to, ho = _render(over)
            t += "^{" + to + "}"
            h += f'<span class="sup">{ho}</span>'
        atom = Char("", "", base, x0, min(top, *(c.y0 for c in over)) if over else top, x1,
                    max(bottom, *(c.y1 for c in under)) if under else bottom, op[0].oy, tex=t, html=h)
        out = [c for c in out if c not in op and c not in under and c not in over] + [atom]
    return out


def inline_latex(spans: list[dict], base: float, baseline: float) -> tuple[str, str]:
    """LaTeX + HTML cho một công thức nội dòng (các span liền nhau trên một dòng chữ)."""
    chars = [c for sp in spans for c in chars_from_span(sp)]
    return _render(chars, base, baseline)


# ---------------------------------------------------------------------------
# Công thức đứng riêng: phân số, căn, số thứ tự phương trình, nhiều dòng
# ---------------------------------------------------------------------------

_EQ_NUM_RE = re.compile(r"^\(\s*\d+(?:\.\d+)?[a-z]?\s*\)$")


def _rule_atoms(chars: list[Char], rules: list[tuple[float, float, float, float]]) -> list[Char]:
    """Thay các cụm phân số / căn bằng một khối dựng sẵn (xử lý từ trong ra ngoài)."""
    chars = list(chars)
    for rx0, ry0, rx1, ry1 in sorted(rules, key=lambda r: r[2] - r[0]):
        ry = (ry0 + ry1) / 2
        size = max((ch.size for ch in chars if rx0 - 1 <= ch.xc <= rx1 + 1), default=10.0)
        # Căn: dấu √ sát bên trái nét ngang, nét ngang nằm trên các ký tự bên trong.
        rad = next(
            (ch for ch in chars if ch.tex is None and (ch.c == "√" or (_FONT_RADICAL.search(ch.font) and ch.c in "pqrstu√"))
             and abs(ch.x1 - rx0) < max(2.5, 0.35 * ch.size) and ch.y0 <= ry + max(1.5, 0.3 * ch.size)
             and ch.y1 > ry), None)
        if rad is not None:
            inside = [ch for ch in chars if ch is not rad and rx0 - 0.5 <= ch.xc <= rx1 + 0.5
                      and ch.yc > ry and ch.y0 < ry + 2.5 * size]
            if inside:
                t, h = _render(inside)
                atom = Char("", "", max(c.size for c in inside), rad.x0, ry0, rx1, max(c.y1 for c in inside),
                            sorted(c.oy for c in inside)[len(inside) // 2], tex=r"\sqrt{" + t + "}",
                            html=f"√<span style='text-decoration:overline'>{h}</span>")
                chars = [ch for ch in chars if ch is not rad and ch not in inside] + [atom]
                continue
        num = [ch for ch in chars if rx0 - 1 <= ch.xc <= rx1 + 1 and ch.yc < ry and ry - ch.y1 < 1.6 * size]
        den = [ch for ch in chars if rx0 - 1 <= ch.xc <= rx1 + 1 and ch.yc > ry and ch.y0 - ry < 1.6 * size]
        if num and den:
            tn, hn = _render(num)
            td, hd = _render(den)
            atom = Char("", "", max(c.size for c in num + den), rx0, min(c.y0 for c in num), rx1,
                        max(c.y1 for c in den), ry + 0.3 * size, tex=r"\frac{" + tn + "}{" + td + "}",
                        html=f"({hn})/({hd})")
            chars = [ch for ch in chars if ch not in num and ch not in den] + [atom]
    return chars


def formula_latex(spans: list[dict], rules: list[tuple[float, float, float, float]] | None = None) -> str | None:
    """LaTeX cho một công thức đứng riêng; None nếu không dựng được (giữ ảnh cắt từ PDF)."""
    try:
        return _formula_latex(spans, rules)
    except UnknownGlyph:
        return None


def _formula_latex(spans: list[dict], rules: list[tuple[float, float, float, float]] | None) -> str | None:
    chars = [c for sp in spans for c in chars_from_span(sp) if c.c.strip()]
    if not chars:
        return None
    chars = _rule_atoms(chars, rules or [])
    base, _ = _base_of(chars)
    # Số thứ tự phương trình "(3)" ở tận cùng bên phải, cách xa phần còn lại → \tag{3}
    tag = None
    plain = [c for c in chars if c.tex is None]
    tail = sorted(plain, key=lambda c: c.x0)
    for k in range(len(tail) - 1, 0, -1):
        cand = "".join(c.c for c in tail[k:])
        if tail[k].c == "(" and _EQ_NUM_RE.match(cand.strip()):
            rest = [c for c in chars if c not in tail[k:]]
            if rest and tail[k].x0 - max(c.x1 for c in rest) > 0.6 * base:
                tag = cand.strip()[1:-1].strip()
                chars = rest
            break
    # Nhiều dòng (hệ phương trình, công thức xuống dòng): gom theo đường chân chữ.
    rows: list[list[Char]] = []
    for c in sorted(chars, key=lambda c: c.oy):
        if c.tex is None and abs(c.size - base) < 0.15 * base and rows:
            row_oy = sorted(x.oy for x in rows[-1] if x.tex is None and abs(x.size - base) < 0.15 * base) or [rows[-1][0].oy]
            if c.oy - row_oy[-1] > 1.3 * base:
                rows.append([c])
                continue
        if not rows:
            rows.append([c])
        else:
            rows[-1].append(c)
    if len(rows) > 1:
        # gán lại ký tự nhỏ (chỉ số) về dòng có đường chân chữ gần nhất
        bases = [sorted(x.oy for x in r)[len(r) // 2] for r in rows]
        rows2: list[list[Char]] = [[] for _ in rows]
        for c in chars:
            k = min(range(len(bases)), key=lambda j: abs(c.oy - bases[j]))
            rows2[k].append(c)
        body = r"\begin{gathered} " + r" \\ ".join(_render(r)[0] for r in rows2 if r) + r" \end{gathered}"
    else:
        body = _render(chars)[0]
    if not body.strip():
        return None
    return body + (rf" \tag{{{tag}}}" if tag else "")


def page_rules(page) -> list[tuple[float, float, float, float]]:
    """Các nét ngang mảnh trên trang (gạch phân số, nét trên của dấu căn)."""
    out = []
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001
        return out
    for d in drawings:
        r = d["rect"]
        if r.height <= 1.2 and 2 < r.width < 0.6 * page.rect.width:
            out.append((r.x0, r.y0, r.x1, r.y1))
    return out


# ---------------------------------------------------------------------------
# Tìm công thức trong bản dịch
# ---------------------------------------------------------------------------

_SINGLE_OK = set("αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ")


def find_math(text: str, keys) -> list[tuple[int, int, str]]:
    """Vị trí (đầu, cuối, công thức gốc) các công thức nội dòng trong ``text``.

    Ưu tiên công thức dài trước; chữ cái đơn chỉ khớp khi đứng riêng (không khớp "x" trong
    "xin"); ký hiệu đơn (=, +) bỏ qua.
    """
    marks: list[tuple[int, int, str]] = []
    for m in sorted(set(k for k in keys if k), key=len, reverse=True):
        if len(m) == 1 and not (m.isalpha() or m in _SINGLE_OK):
            continue
        pat = re.escape(m)
        if m.isascii() and len(m) <= 3:  # chữ cái / ký hiệu ngắn: phải đứng riêng
            if m[:1].isalnum():
                pat = r"(?<![\w])" + pat
            if m[-1:].isalnum():
                pat = pat + r"(?![\w])"
        for hit in re.finditer(pat, text):
            if all(hit.end() <= a or hit.start() >= b for a, b, _ in marks):
                marks.append((hit.start(), hit.end(), m))
    return sorted(marks)
