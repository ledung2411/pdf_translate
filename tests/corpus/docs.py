"""Các tài liệu mẫu mô phỏng nhà xuất bản + đáp án phân tích mong đợi.

Mỗi hàm trả về ``expect`` gồm:
- ``text``: các đoạn (tiền tố) phải được nhận là đoạn văn cần dịch
- ``kinds``: tiền tố → loại vùng bắt buộc (math, table, reference, rotated...)
- ``no_mix``: cặp chuỗi không được nằm chung một vùng (chống trộn 2 cột)
- ``headings``: tiền tố tiêu đề → bản dịch mong đợi sau khi dịch
"""

from __future__ import annotations

from .builder import Builder, Style

P1 = (
    "Perovskite solar cells have attracted considerable attention because of their high power conversion "
    "efficiency and low fabrication cost. However, the long-term stability of these devices remains a major "
    "challenge for commercialization, especially under humid conditions and continuous illumination."
)
P2 = (
    "Deep neural networks have achieved remarkable results in medical image analysis. In this work, we train "
    "a convolutional neural network on magnetic resonance images and evaluate the model on three public "
    "datasets, where it consistently outperforms the baseline methods reported in previous studies."
)
P3 = (
    "The optimized device achieves an efficiency of 24.1% and retains 95% of its initial performance after "
    "1000 h of continuous illumination, which is among the best stability reported for this class of materials."
)
REFS = [
    "[1] A. Vaswani, N. Shazeer, N. Parmar, et al. Attention is all you need. In NeurIPS, 2017.",
    "[2] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.",
]


def latex_arxiv(path: str) -> dict:
    """LaTeX/arXiv: CMR10 + CMMI10, 2 cột khoảng giữa rộng, công thức khối, watermark xoay."""
    b = Builder(rename={"serif": "CMR10", "serif-bold": "CMBX12", "math": "CMMI10"})
    body, math = Style("serif", 10), Style("math", 10)
    b.paragraph(72, 523, 80, "Sparse Attention for Long Documents", Style("serif-bold", 16), align="center")
    b.text(30, 650, "arXiv:2401.12345v1 [cs.CL] 10 Jan 2024", Style("serif", 14), rotate=90)
    y = b.paragraph(72, 287, 140, "1 Introduction", Style("serif-bold", 12), align="left")
    y = b.paragraph(72, 287, y, P2 + " Let $x$ denote the input and $W$ the weight matrix of each layer.", body, math)
    y = b.paragraph(110, 250, y + 4, "$y = softmax ( W x + b )$", math, align="center")
    y = b.paragraph(72, 287, y + 4, P1, body)
    y2 = b.paragraph(308, 523, 140, P3, body)
    y2 = b.paragraph(308, 523, y2, "References", Style("serif-bold", 12), align="left")
    for r in REFS:
        y2 = b.paragraph(308, 523, y2, r, Style("serif", 8.5), align="left")
    b.save(path)
    return {
        "text": ["Sparse Attention", "Deep neural networks", "Perovskite solar", "The optimized device"],
        "kinds": {"y = softmax": "math", "arXiv:2401": "rotated", "[1] A. Vaswani": "reference"},
        "no_mix": [("Deep neural networks", "The optimized device")],
        "headings": {"1 Introduction": "1 Giới thiệu", "References": "Tài liệu tham khảo"},
    }


def wiley(path: str) -> dict:
    """Wiley (Advanced Science): STIX Two Text, dấu cách 'dummy-space', khoảng giữa 2 cột hẹp,
    tiêu đề '2 | Methods', 'RESEARCH ARTICLE', 'ABSTRACT' dính đầu đoạn."""
    b = Builder(rename={"serif": "STIXTwoText", "serif-bold": "STIXTwoText-Bold", "sans": "dummy-space"},
                space_font="sans")
    body = Style("serif", 9)
    b.paragraph(54, 300, 70, "RESEARCH ARTICLE", Style("serif-bold", 9), align="left")
    b.paragraph(54, 560, 95, "DDSurfer: A Weakly-Supervised Framework for Cortical Surface Reconstruction",
                Style("serif-bold", 16), align="left")
    y = b.paragraph(64, 550, 160, "ABSTRACT " + P2, body)
    # Hai cột, khoảng giữa chỉ 12pt, các dòng cùng hàng → MuPDF dễ gộp thành một khối
    y_l = b.paragraph(54, 296, y + 20, "1 | Introduction", Style("serif-bold", 11), align="left")
    y_l = b.paragraph(54, 296, y_l, P1 + " " + P3, body)
    y_r = b.paragraph(308, 550, y + 20, "2 | Methods", Style("serif-bold", 11), align="left")
    y_r = b.paragraph(308, 550, y_r, P2 + " " + P2.replace("Deep neural", "Recurrent"), body)
    b.save(path)
    return {
        "text": ["RESEARCH ARTICLE", "DDSurfer: A Weakly-Supervised", "ABSTRACT Deep neural", "Perovskite solar",
                 "Deep neural networks have achieved remarkable results in medical image analysis. In this work"],
        "kinds": {},
        "no_mix": [("Perovskite solar cells have attracted", "In this work, we train a convolutional neural network on magnetic")],
        "headings": {"1 | Introduction": "1 | Giới thiệu", "2 | Methods": "2 | Phương pháp"},
        "contains": ["Weakly-Supervised"],  # không còn "Weakly- Supervised"
    }


def ieee(path: str) -> dict:
    """IEEE: Times New Roman, tiêu đề La Mã viết hoa, 'Index Terms—', bảng số liệu."""
    b = Builder(rename={"serif": "TimesNewRomanPSMT", "serif-bold": "TimesNewRomanPS-BoldMT",
                        "serif-italic": "TimesNewRomanPS-ItalicMT"})
    body = Style("serif", 10)
    b.paragraph(60, 535, 70, "A Robust Method for Image Segmentation", Style("serif", 22), align="center")
    y = b.paragraph(60, 290, 130, "Abstract—" + P2, Style("serif-bold", 9))
    y = b.paragraph(60, 290, y, "Index Terms—image segmentation, deep learning, medical imaging.", Style("serif-italic", 9))
    y = b.paragraph(60, 290, y + 6, "I. INTRODUCTION", Style("serif", 10), align="center")
    y = b.paragraph(60, 290, y, P1, body)
    # Bảng số liệu ở cột phải
    yt = 130
    b.paragraph(305, 535, yt, "TABLE I", Style("serif", 8), align="center")
    for row in (("Method", "Dice", "HD95"), ("U-Net", "0.85", "4.21"), ("Ours", "0.91", "2.87")):
        yt += 14
        for k, cell in enumerate(row):
            b.text(320 + k * 75, yt, cell, Style("serif", 8))
    yr = b.paragraph(305, 535, yt + 25, "II. RELATED WORK", Style("serif", 10), align="center")
    b.paragraph(305, 535, yr, P3, body)
    b.save(path)
    return {
        "text": ["A Robust Method", "Perovskite solar", "The optimized device"],
        "kinds": {"Ours": "table"},
        "not_text": ["0.85", "U-Net"],
        "headings": {"I. INTRODUCTION": "I. GIỚI THIỆU", "II. RELATED WORK": "II. CÁC CÔNG TRÌNH LIÊN QUAN"},
    }


def single_column(path: str) -> dict:
    """Một cột (luận văn / Springer): đoạn dài nhiều trang, chú thích hình, chỉ số trên."""
    b = Builder(rename={"serif": "NimbusRomNo9L-Regu", "serif-bold": "NimbusRomNo9L-Medi"})
    body = Style("serif", 11)
    y = b.paragraph(72, 523, 90, "Chapter 2 Background", Style("serif-bold", 16), align="left")
    for p in (P1, P2, P3, P1):
        y = b.paragraph(72, 523, y, p, body)
    b.page.draw_rect((150, y + 10, 445, y + 120), color=(0, 0, 0.6), fill=(0.9, 0.9, 1))
    y = b.paragraph(72, 523, y + 130, "Figure 2.1: Schematic of the experimental setup used in this study.",
                    Style("serif", 9.5), align="center")
    b.new_page()
    b.paragraph(72, 523, 90, P2 + " " + P3, body)
    b.save(path)
    return {
        "text": ["Chapter 2 Background", "Perovskite solar", "Figure 2.1", "Deep neural networks"],
        "kinds": {},
        "pages": 2,
    }


def elsevier(path: str) -> dict:
    """Elsevier: tiêu đề sans-serif, 'Keywords:', mục '1. Introduction', 2 cột chuẩn."""
    b = Builder(rename={"serif": "Gulliver", "sans-bold": "Helvetica-Bold", "sans": "Helvetica"})
    body = Style("serif", 9)
    b.paragraph(48, 547, 80, "Stable perovskite solar cells via surface passivation", Style("sans-bold", 15), align="left")
    y = b.paragraph(48, 547, 120, "Keywords: perovskite, stability, passivation, solar cells", Style("sans", 8), align="left")
    y_l = b.paragraph(48, 290, y + 12, "1. Introduction", Style("sans-bold", 10), align="left")
    b.paragraph(48, 290, y_l, P1 + " " + P3, body)
    y_r = b.paragraph(305, 547, y + 12, "2. Results and discussion", Style("sans-bold", 10), align="left")
    b.paragraph(305, 547, y_r, P2, body)
    b.save(path)
    return {
        "text": ["Stable perovskite", "Keywords: perovskite", "Perovskite solar", "Deep neural networks"],
        "kinds": {},
        "no_mix": [("Perovskite solar cells", "Deep neural networks")],
        "headings": {"1. Introduction": "1. Giới thiệu", "2. Results and discussion": "2. Kết quả và thảo luận"},
    }


ALL = {"latex_arxiv": latex_arxiv, "wiley": wiley, "ieee": ieee, "single_column": single_column, "elsevier": elsevier}
