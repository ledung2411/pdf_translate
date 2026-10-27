"""Chữ dịch đè lên nhau: MuPDF cắt một đoạn văn thành nhiều khối, mỗi khối dịch riêng.

Toạ độ lấy từ một bài báo Advanced Science (Wiley) thật.
"""

import pymupdf

from app.ir import Region
from app.pdf_processor import _join_row_pieces, _segment_blocks, find_overlaps, resolve_overlaps


def span(text, x0, y0, x1, y1, size=9.0, font="STIXTwoText"):
    return {"text": text, "bbox": (x0, y0, x1, y1), "size": size, "font": font, "flags": 4, "color": 0}


def line(*spans):
    return {"spans": list(spans), "bbox": (min(s["bbox"][0] for s in spans), min(s["bbox"][1] for s in spans),
            max(s["bbox"][2] for s in spans), max(s["bbox"][3] for s in spans)), "dir": (1, 0), "wmode": 0}


def block(*lines):
    return {"type": 0, "lines": list(lines), "bbox": (min(l["bbox"][0] for l in lines), min(l["bbox"][1] for l in lines),
            max(l["bbox"][2] for l in lines), max(l["bbox"][3] for l in lines))}


def all_text(b):
    return "".join(s["text"] for l in b["lines"] for s in l["spans"])


def test_superscript_continuation_joined():
    # "... a b-value of 1000 s/mm" | "²" + ", 65 directions" (MuPDF mở khối mới ở chỉ số trên)
    a = block(line(span("with acquisition parameters including ", 53.6, 155.5, 300.9, 164.5)),
              line(span("a b-value of 1000 s/mm ", 53.6, 167.0, 145.2, 176.0)))
    b = block(line(span("2", 141.9, 163.3, 145.5, 169.6, size=6.4), span(", 65 directions, and TR/TE", 145.5, 167.0, 301.0, 176.0)),
              line(span("with a resolution of 3 mm isotropic.", 53.6, 178.5, 301.0, 187.4)))
    out = _join_row_pieces([a, b])
    assert len(out) == 1
    assert "s/mm 2, 65 directions" in all_text(out[0])
    assert len(out[0]["lines"]) == 3  # dòng chung được nối lại thành một


def test_inline_formula_pieces_joined():
    # "features x′" | "β = x_β ⊙ σ" | "(" | "f(x_α)" | ")" | ". Similarly, ..." trên cùng một hàng
    a = block(line(span("that modulates the Diffusivity stream", 53.6, 133.0, 301.0, 142.0)),
              line(span("tures x′", 53.6, 145.1, 79.0, 155.2)))
    pieces = [block(line(span(t, x0, y0, x1, y1, font="STIXTwoMath-Regular")))
              for t, x0, y0, x1, y1 in [("β = xβ ⊙ σ", 79.0, 144.6, 120.1, 156.6), ("(", 118.8, 137.4, 123.0, 146.4),
                                         ("f(xα)", 123.0, 145.1, 144.2, 155.2), (")", 144.2, 137.4, 148.4, 146.4)]]
    tail = block(line(span(". Similarly, xβ generates a corresponding", 148.4, 145.1, 300.9, 155.2)))
    out = _join_row_pieces([a, *pieces, tail])
    assert len(out) == 1
    assert all_text(out[0]).endswith("tures x′β = xβ ⊙ σ(f(xα)). Similarly, xβ generates a corresponding")


def test_two_columns_not_joined():
    # Hai cột thật: khe giữa cột (13 pt) lớn hơn hẳn khe nối mảnh → không ghép.
    left = block(line(span("end of a left column line", 53.6, 100, 301.0, 109)))
    right = block(line(span("start of right column", 314.0, 100, 562.0, 109)))
    assert len(_join_row_pieces([left, right])) == 2


def test_inline_affiliations_stay_one_block():
    # Danh sách địa chỉ ngăn bằng "|" chạy liền nhiều hàng — không phải 2 cột.
    lines = [line(span(t, x0, y0, x1, y1, size=8)) for t, x0, y0, x1, y1 in [
        ("1 University of Electronic Science and Technology of China, Chengdu, China |", 53.6, 261.7, 296.6, 271.4),
        ("2 Brigham and Women's Hospital, Harvard Medical School, Boston,", 301.3, 261.7, 515.4, 271.4),
        ("Massachusetts, USA |", 53.6, 274.9, 120.8, 282.3),
        ("3 School of Mathematical Sciences, University of Tel Aviv, Tel Aviv, Israel |", 125.5, 272.7, 357.6, 282.3),
        ("4 Department of Anatomy and Neurobiology, Boston", 362.2, 272.7, 529.7, 282.3),
        ("University School of Medicine, Boston, Massachusetts, USA", 53.6, 285.8, 245.2, 293.3)]]
    out = _segment_blocks([block(*lines)], pymupdf.Rect(0, 0, 612, 792))
    assert len(out) == 1


def _region(text, bbox, kind="text"):
    return Region(page_index=0, bbox=pymupdf.Rect(bbox), text=text, kind=kind, font_size=9.0, bold=False,
                  italic=False, serif=True, color=0, align="justify",
                  line_rects=[pymupdf.Rect(bbox)])


def test_overlapping_text_regions_merged():
    a = _region("used for the dMRI data are ... a b-value of 1000 s/mm", (54, 40.5, 301, 176.0))
    b = _region("2, 65 directions, and TR/TE = 5200/78 ms", (54, 163.3, 301, 244.7))
    c = _region("Next paragraph.", (54, 260, 301, 300))
    assert len(find_overlaps([a, b, c])) == 1
    out = resolve_overlaps([b, a, c])
    assert len(out) == 2 and not find_overlaps(out)
    merged = next(r for r in out if r is not c)
    assert merged.plain.startswith("used for the dMRI") and "65 directions" in merged.plain


def test_adjacent_regions_not_overlapping():
    # Tiêu đề sát đoạn văn, chạm nhau vài phần pt → không tính là chồng lấn.
    head = _region("2.2.4 Loss Function Design", (314, 225, 445, 234.5))
    para = _region("The loss function ...", (314, 234, 562, 300))
    assert find_overlaps([head, para]) == []
