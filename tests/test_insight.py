"""Tóm tắt theo câu hỏi nghiên cứu và bản dễ hiểu (LLM giả lập + chế độ không LLM)."""

import time

import pymupdf
from fastapi.testclient import TestClient

from app.insight import QUESTIONS, extract_facts, split_sections, summarize, summary_markdown
from app.ir import Document, Page, Region
from app.plain import clean_rewrite, explain_terms, make_plain


def region(text, y, size=10.0, bold=False, translation=None, kind="text"):
    return Region(page_index=0, bbox=pymupdf.Rect(50, y, 550, y + 12), text=text, kind=kind, font_size=size,
                  bold=bold, italic=False, serif=True, color=0, align="justify",
                  line_rects=[pymupdf.Rect(50, y, 550, y + 12)], translation=translation)


def paper() -> Document:
    regs = [
        region("FastSeg: Efficient Brain Segmentation With Lightweight Networks", 40, 16, True),
        region("Abstract. Manual segmentation is time-consuming and expensive. We propose FastSeg, a lightweight "
               "network for brain segmentation. FastSeg improves Dice by 3.2% over nnU-Net while running 5× faster.", 70),
        region("1 Introduction", 100, 11, True),
        region("Brain segmentation is critical for clinical diagnosis. However, existing methods are slow and "
               "require large GPUs. Our contributions are threefold: we propose a novel encoder with depthwise "
               "convolutions, inspired by MobileNet, and we release our code at https://github.com/lab/fastseg.", 120),
        region("2 Method", 150, 11, True),
        region("The network takes a 3D MRI volume as input and outputs a label map with 4 classes. The encoder "
               "consists of five blocks; because depthwise convolutions reduce parameters, the model fits on a CPU.", 170),
        region("3 Experiments", 200, 11, True),
        region("We evaluate on the BraTS dataset and the OASIS dataset with Dice and HD95 metrics. We compare "
               "against nnU-Net, SwinUNETR and TransUNet baselines.", 220),
        region("4 Discussion", 250, 11, True),
        region("A limitation is that FastSeg was only tested on adult brains; future work will extend it to infants.", 270),
        region("References", 300, 11, True),
        region("[1] O. Ronneberger. U-Net. 2015.", 320),
    ]
    return Document(source="fastseg.pdf", pages=[Page(index=0, number=1, width=600, height=800, regions=regs)])


class FakeLLM:
    def __init__(self):
        self.calls = []

    def _chat(self, system, user, max_tokens=None):
        self.calls.append((system, user))
        return "- Câu trả lời của LLM"


def test_sections_and_facts():
    ir = paper()
    secs = split_sections(ir)
    kinds = [s.kind for s in secs]
    assert "intro" in kinds and "method" in kinds and "experiment" in kinds and "discussion" in kinds
    assert not any("Ronneberger" in r.plain for s in secs for r in s.regions)  # dừng ở References
    facts = extract_facts(ir, secs)
    assert facts["code"] == ["https://github.com/lab/fastseg"]
    assert "BraTS" in facts["datasets"] and "OASIS" in facts["datasets"]
    assert {"nnU-Net", "SwinUNETR", "TransUNet"} <= set(facts["baselines"])


def test_summary_without_llm_picks_evidence():
    s = summarize(paper(), llm=None, translate=lambda xs: [f"VI: {x}" for x in xs])
    by_key = {a.question.key: a for a in s.answers}
    assert len(s.answers) == len(QUESTIONS) and s.mode == "extract"
    assert "we propose" in by_key["contribution"].text.lower()
    assert "inspired by MobileNet" in by_key["origin"].text
    assert "takes a 3D MRI volume" in by_key["io"].text
    assert "limitation" in by_key["cons"].text.lower()
    assert "https://github.com/lab/fastseg" in by_key["code"].text
    assert by_key["contribution"].text.startswith("- VI:")  # câu trích được dịch
    md = summary_markdown(s)
    assert md.startswith("# Tóm tắt bài báo") and "## Đóng góp chính" in md


def test_summary_with_llm_uses_grounded_context():
    llm = FakeLLM()
    s = summarize(paper(), llm=llm)
    assert s.mode == "llm" and len(llm.calls) == len(QUESTIONS)
    system, user = llm.calls[0]
    assert "không bịa" in system and "Bài báo không nêu rõ" in system
    assert "TRÍCH ĐOẠN LIÊN QUAN" in user and "CÂU HỎI:" in user and "Abstract" in user
    code_prompt = next(u for _, u in llm.calls if "Mã nguồn" in u or "công bố công khai" in u)
    assert "github.com/lab/fastseg" in code_prompt
    assert all(a.text.startswith("- Câu trả lời của LLM") for a in s.answers)


def test_plain_without_llm_explains_terms_once():
    used: set[str] = set()
    text, added = explain_terms("Mô hình học sâu tối ưu hàm mất mát; học sâu rất mạnh.", used)
    assert text.count("học sâu (*") == 1 and "hàm mất mát (*" in text
    assert {t.lower() for t, _ in added} == {"học sâu", "hàm mất mát"}
    ir = paper()
    for r in ir.regions:
        r.translation = "Chúng tôi dùng học sâu và tập dữ liệu BraTS." if r.kind == "text" and len(r.plain) > 60 else r.plain
    md = make_plain(ir)
    assert "## Khái niệm cần biết" in md and "**học sâu**" in md
    assert "FastSeg: Efficient" not in md.split("## Khái niệm cần biết")[1].split("##")[0]


def test_plain_with_llm_rewrites_sections():
    llm = FakeLLM()
    ir = paper()
    for r in ir.regions:
        r.translation = r.plain
    md = make_plain(ir, llm=llm)
    assert llm.calls and "<<<" in llm.calls[0][1] and "MỤC:" not in llm.calls[0][1]
    assert "Câu trả lời của LLM" in md


def attention_like() -> Document:
    regs = [
        region("Attention Is All You Need", 40, 17, True),
        region("Abstract", 70, 12, True),
        region("The dominant sequence transduction models are based on complex recurrent networks. We propose the "
               "Transformer, based solely on attention mechanisms.", 90),
        region("∗Equal contribution. Jakob proposed replacing RNNs with self-attention and started the effort.",
               700, 8),
        region("31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.", 760, 8),
    ]
    regs2 = [
        region("1 Introduction", 130, 12, True),
        region("Recurrent models preclude parallelization within training examples, which becomes critical at longer "
               "sequence lengths. Attention mechanisms have become an integral part of sequence models.", 150),
        region("4 Why Self-Attention", 200, 12, True),
        region("Self-attention layers connect all positions with a constant number of sequential operations, which "
               "allows for significantly more parallelization and makes it easier to learn long-range dependencies.",
               220),
        region("6 Results", 260, 12, True),
        region("On the WMT 2014 English-to-German translation task, the big transformer outperforms the best "
               "previously reported models including GNMT and ConvS2S by more than 2.0 BLEU, training on 8 P100 GPUs "
               "for EN-DE.", 280),
        region("7 Conclusion", 320, 12, True),
        region("The code we used to train and evaluate our models is available at https://github.com/ "
               "tensorflow/tensor2tensor.", 340),
    ]
    for r in regs2:
        r.page_index = 1
    return Document(source="attention.pdf", pages=[Page(index=0, number=1, width=600, height=800, regions=regs),
                                                   Page(index=1, number=2, width=600, height=800, regions=regs2)])


def test_footnotes_rationale_urls_and_baselines():
    ir = attention_like()
    secs = split_sections(ir)
    text = " ".join(r.plain for s in secs for r in s.regions)
    assert "Equal contribution" not in text and "NIPS 2017" not in text  # chú thích chân trang bị loại
    assert {s.title: s.kind for s in secs}["4 Why Self-Attention"] == "rationale"
    by_key = {a.question.key: a for a in summarize(ir, llm=None).answers}
    assert "allows for significantly more parallelization" in by_key["why"].text
    facts = extract_facts(ir, secs)
    assert facts["code"] == ["https://github.com/tensorflow/tensor2tensor"]
    assert {"GNMT", "ConvS2S"} <= set(facts["baselines"])
    assert not {"BLEU", "GPUs", "P100", "EN-DE"} & set(facts["baselines"])


class RefusingLLM(FakeLLM):
    def __init__(self, stubborn=False):
        super().__init__()
        self.stubborn = stubborn

    def _chat(self, system, user, max_tokens=None):
        self.calls.append((system, user))
        if system.startswith("Dịch các câu"):
            return "- Câu đã dịch 表 1"
        if "Lưu ý:" in user and not self.stubborn:
            return "- [6.1 Machine Translation] [7 Conclusion] Có trả lời"
        return "Bài báo không nêu rõ."


def test_refusal_is_retried_once():
    llm = RefusingLLM()
    s = summarize(paper(), llm=llm)
    retried = [u for _, u in llm.calls if "Lưu ý:" in u]
    assert retried and len(llm.calls) == len(QUESTIONS) + len(retried)
    assert any(a.text.startswith("- Có trả lời") for a in s.answers)  # thẻ "[1 Introduction]" đã bỏ
    assert "không nêu rõ" not in llm.calls[len(QUESTIONS)][0].split("Quy tắc:")[1]  # lần hỏi lại không cho né


def test_stubborn_refusal_falls_back_to_translated_evidence():
    s = summarize(paper(), llm=RefusingLLM(stubborn=True))
    by_key = {a.question.key: a for a in s.answers}
    assert "**Link tìm thấy trong bài:**" in by_key["code"].text and "không nêu rõ" not in by_key["code"].text
    fb = [a for a in s.answers if "không tự tổng hợp" in a.text]
    assert fb and all("- Câu đã dịch Bảng 1" in a.text and "表" not in a.text for a in fb)


def test_english_or_chinese_llm_output_is_rejected():
    from app.insight import has_cjk_run, is_mostly_english, strip_cjk

    assert is_mostly_english("- quality also drops off with too many heads. - our model achieves 28.4 BLEU on the task")
    assert not is_mostly_english("- Transformer đạt 28.4 BLEU trên WMT 2014, nhanh hơn RNN (https://github.com/x/y).")
    assert has_cjk_run("một hiệu ứng mà chúng tôi đối抗通过多头注意力") and strip_cjk("trong表 3") == "trong Bảng 3"


def test_clean_rewrite_strips_leaked_labels():
    assert clean_rewrite('Đoạn trích thuộc mục "3.2 Attention" của một bài báo nói về X. Cơ chế chú ý.') \
        == "Cơ chế chú ý."
    out = clean_rewrite("MỤC: Tóm tắt\n\nĐOẠN GỐC (tiếng Anh):\nMô hình mới đơn giản hơn.\n\n"
                        "BẢN DỊCH MÁY (để tham khảo):\n<<<\nNó nhanh hơn.\n>>>")
    assert out == "Mô hình mới đơn giản hơn.\n\nNó nhanh hơn."


def test_api_extras_for_finished_job(tmp_path):
    from app.main import app

    from .make_sample_pdf import make

    pdf = tmp_path / "s.pdf"
    make(str(pdf))
    with TestClient(app) as client:
        r = client.post("/api/jobs", files={"file": ("paper.pdf", pdf.read_bytes(), "application/pdf")},
                        data={"engine": "mock", "make_bilingual": "false"})
        job_id = r.json()["id"]

        def wait():
            for _ in range(200):
                st = client.get(f"/api/jobs/{job_id}").json()
                if st["status"] in ("done", "error"):
                    return st
                time.sleep(0.1)
            return st

        assert wait()["status"] == "done"
        assert client.get("/api/llm").json()["available"] in (True, False)
        r = client.post(f"/api/jobs/{job_id}/extras", json={"summary": True, "plain": True})
        assert r.status_code == 200, r.text
        st = wait()
        assert st["status"] == "done", st
        assert {"summary_html", "summary_docx", "summary_md", "plain_html", "plain_md"} <= set(st["downloads"])
        page = client.get(f"/api/jobs/{job_id}/download/summary_html?inline=true")
        assert page.status_code == 200 and "Tóm tắt bài báo" in page.text and "text/html" in page.headers["content-type"]
