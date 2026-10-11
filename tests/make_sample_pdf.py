"""Tạo một PDF giả lập bài báo khoa học 2 cột để kiểm thử bố cục.

Chạy: python tests/make_sample_pdf.py sample.pdf
"""

from __future__ import annotations

import sys

import pymupdf

TITLE = "Efficient Attention Mechanisms for Long Document Understanding"
AUTHORS = "Jane Doe<sup>1</sup>, John Smith<sup>2</sup><br/>University of Science"
ABSTRACT = (
    "<b>Abstract.</b> Transformer models have achieved state-of-the-art results in natural language "
    "processing. However, the self-attention mechanism scales quadratically with sequence length "
    "<span class='m'>n</span>, which limits its use on long documents. In this paper, we propose a sparse "
    "attention method that reduces the complexity to <span class='m'>O(n log n)</span> while keeping "
    "accuracy within 1% of the full model. Experiments on three benchmark datasets show that our "
    "approach is 4.2 times faster (p &lt; 0.05). Code is available at https://github.com/example/sparse-attn."
)
LEFT = """
<h3>1 Introduction</h3>
<p>Deep learning has transformed many fields, including computer vision and natural language processing [1, 2].
Large language models are trained on massive datasets using stochastic gradient descent. Recent work by
Smith et al. [3] showed that the attention weights <span class='m'>α<sub>ij</sub></span> are often sparse,
i.e., most tokens attend to only a few others.</p>
<p>Let <span class='m'>x ∈ ℝ<sup>d</sup></span> be the input embedding. The attention score is computed as
shown in Eq. 1, where <span class='m'>Q</span>, <span class='m'>K</span> and <span class='m'>V</span> denote the query,
key and value matrices, respectively.</p>
<p class='eq'><span class='m'>Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d) V&#160;&#160;&#160;(1)</span></p>
<p>We make the following contributions. First, we introduce a learnable sparsity pattern. Second, we
prove an upper bound on the approximation error (Theorem 1). Third, we evaluate our method on long
document classification and question answering.</p>
<h3>2 Related Work</h3>
<p>Several efficient Transformer variants have been proposed, such as Longformer [4] and BigBird [5].
These models use fixed attention patterns, whereas our approach learns the pattern from data.</p>
"""
RIGHT = """
<h3>3 Experiments</h3>
<p>We trained all models for 10 epochs with a learning rate of 3e-4 and a batch size of 32.
Table 1 reports the accuracy and the training time. The proposed method outperforms the baseline
on all datasets, and the improvement is statistically significant.</p>
<table>
<tr><th>Model</th><th>Acc. (%)</th><th>Time (h)</th></tr>
<tr><td>Full</td><td>91.2</td><td>12.5</td></tr>
<tr><td>Ours</td><td>90.8</td><td>2.9</td></tr>
</table>
<p class='cap'>Table 1: Accuracy and training time on the test set.</p>
<h3>4 Conclusion</h3>
<p>We presented a sparse attention mechanism for long documents. Future work will explore
applications to genomics and time series forecasting.</p>
<h3>References</h3>
<p class='ref'>[1] A. Vaswani, N. Shazeer, et al. Attention is all you need. In NeurIPS, 2017.</p>
<p class='ref'>[2] J. Devlin, M. Chang, K. Lee, and K. Toutanova. BERT: Pre-training of deep bidirectional
transformers for language understanding. In NAACL, 2019.</p>
<p class='ref'>[3] J. Smith and J. Doe. Sparse patterns in attention. arXiv:2101.00001, 2021.</p>
"""
CSS = """
@font-face {font-family: mathf; src: url(math.ttf);}
body {font-family: serif; font-size: 9.5px; line-height: 1.25; text-align: justify;}
h3 {font-size: 11px; margin: 8px 0 4px 0;}
p {margin: 0 0 5px 0;}
.m {font-family: mathf; font-style: italic;}
.eq {text-align: center; margin: 6px 0;}
.cap {text-align: center; font-size: 8.5px;}
.ref {font-size: 8.5px; text-align: left;}
table {margin: 4px auto; border-collapse: collapse; font-size: 9px;}
td, th {padding: 1px 10px; border-bottom: 0.5px solid black;}
"""


def make(path: str) -> None:
    import pymupdf_fonts  # noqa: F401  (font toán học)

    arch = pymupdf.Archive()
    arch.add((pymupdf.Font("math").buffer, "math.ttf"))
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_htmlbox(pymupdf.Rect(60, 50, 535, 90), f"<h1 style='font-size:17px;text-align:center'>{TITLE}</h1>", css=CSS, archive=arch)
    page.insert_htmlbox(pymupdf.Rect(60, 92, 535, 125), f"<p style='text-align:center'>{AUTHORS}</p>", css=CSS, archive=arch)
    page.insert_htmlbox(pymupdf.Rect(80, 130, 515, 230), f"<p>{ABSTRACT}</p>", css=CSS, archive=arch)
    page.insert_htmlbox(pymupdf.Rect(60, 240, 290, 790), LEFT, css=CSS, archive=arch)
    page.insert_htmlbox(pymupdf.Rect(305, 240, 535, 790), RIGHT, css=CSS, archive=arch)
    # Hình minh họa + chú thích
    page.draw_rect(pymupdf.Rect(80, 640, 270, 720), color=(0.2, 0.3, 0.8), fill=(0.85, 0.9, 1))
    page.insert_htmlbox(pymupdf.Rect(60, 725, 290, 760), "<p class='cap'>Figure 1: Overview of the proposed sparse attention architecture.</p>", css=CSS, archive=arch)
    # Watermark xoay kiểu arXiv
    page.insert_text((30, 600), "arXiv:2401.12345v1 [cs.CL] 10 Jan 2024", fontsize=14, rotate=90, color=(0.5, 0.5, 0.5))
    page.insert_text((290, 815), "1", fontsize=9)
    doc.save(path)


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "sample.pdf")
