"""Đo chất lượng và tốc độ dịch trên CPU, in bảng kết quả (Markdown) và lưu JSON.

    python scripts/benchmark.py --src eng.devtest --ref vie.devtest --pdf paper.pdf

Ba phép đo:
1. Chất lượng: BLEU và chrF (sacrebleu) của mô hình dịch (CTranslate2 int8, đúng cấu hình
   tool dùng: beam 4, chặn lặp 4-gram) trên tập thử song ngữ ``--src`` / ``--ref``.
2. Tăng tốc nhờ int8: cùng ``--speed-n`` câu, dịch bằng transformers fp32 và CTranslate2 int8.
3. Số giây mỗi trang: chạy trọn quy trình (phân tích bố cục → dịch → dựng PDF/Word) trên
   ``--pdf`` sau khi đã nạp mô hình, không dùng bộ nhớ đệm hay bộ nhớ dịch.

Cần: pip install -r requirements-ml.txt sacrebleu, và torch (cho phép đo fp32 / chuyển đổi).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import translators  # noqa: E402
from app.config import settings  # noqa: E402
from app.translators import HFSeq2SeqTranslator, TranslationCache, get_engine  # noqa: E402


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def read_lines(path: Path, n: int | None) -> list[str]:
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines()]
    return lines[:n] if n else lines


def use_backend(eng: HFSeq2SeqTranslator, backend: str) -> float:
    """Chuyển engine sang backend khác, trả về thời gian nạp mô hình (giây)."""
    settings.backend = backend
    eng._model, eng._tokenizer, eng._backend = None, None, ""
    t0 = time.perf_counter()
    eng.prepare()
    if eng._backend != backend:
        raise SystemExit(f"Không nạp được backend {backend} (đang là {eng._backend!r})")
    return time.perf_counter() - t0


def timed_translate(eng: HFSeq2SeqTranslator, src: list[str]) -> tuple[list[str], float]:
    eng.translate_batch(src[: settings.batch_size])  # khởi động (bỏ qua lần chạy đầu)
    t0 = time.perf_counter()
    hyp = eng.translate_batch(src)
    return hyp, time.perf_counter() - t0


def scores(hyp: list[str], ref: list[str]) -> dict[str, float]:
    import sacrebleu

    return {
        "bleu": round(sacrebleu.corpus_bleu(hyp, [ref]).score, 2),
        "chrf": round(sacrebleu.corpus_chrf(hyp, [ref]).score, 2),
    }


def bench_pdf(engine_key: str, pdf: Path) -> dict:
    import pymupdf

    from app.pdf_processor import IR_FILENAME, TranslateOptions, analyze_pdf, render_outputs, translate_document

    out = Path(tempfile.mkdtemp(prefix="bench_"))
    translators._cache = TranslationCache(out / "cache.sqlite")  # đệm trống: dịch thật mọi câu
    opts = TranslateOptions(engine=engine_key, post_edit=False, use_tm=False)
    pages = len(pymupdf.open(str(pdf)))
    t0 = time.perf_counter()
    ir = analyze_pdf(pdf, opts)
    t1 = time.perf_counter()
    translate_document(ir, opts, lambda *_: None)
    t2 = time.perf_counter()
    ir.save(out / IR_FILENAME)
    render_outputs(ir, pdf, out, True, None, opts.highlight_doubtful)
    t3 = time.perf_counter()
    words = sum(len(r.plain.split()) for r in ir.regions if r.kind == "text")
    return {
        "file": pdf.name,
        "pages": pages,
        "words_translated": words,
        "analyze_s": round(t1 - t0, 1),
        "translate_s": round(t2 - t1, 1),
        "render_s": round(t3 - t2, 1),
        "total_s": round(t3 - t0, 1),
        "s_per_page": round((t3 - t0) / pages, 2),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="vinai")
    ap.add_argument("--src", type=Path, help="tập thử: câu tiếng Anh, mỗi dòng một câu")
    ap.add_argument("--ref", type=Path, help="tập thử: bản dịch tham chiếu tiếng Việt")
    ap.add_argument("--n", type=int, default=0, help="chỉ lấy n câu đầu (0 = tất cả)")
    ap.add_argument("--speed-n", type=int, default=100, help="số câu để so tốc độ fp32 và int8 (0 = bỏ qua)")
    ap.add_argument("--pdf", type=Path, action="append", default=[], help="PDF để đo giây/trang")
    ap.add_argument("--json", type=Path, default=Path("benchmark.json"))
    args = ap.parse_args(argv)

    eng = get_engine(args.engine)
    if not isinstance(eng, HFSeq2SeqTranslator):
        raise SystemExit(f"'{args.engine}' không phải mô hình dịch máy Hugging Face")
    import ctranslate2

    res: dict = {
        "engine": args.engine,
        "model": eng.model_name,
        "beams": settings.num_beams,
        "batch_size": settings.batch_size,
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "ctranslate2": ctranslate2.__version__,
    }
    settings.device = "cpu"
    res["load_ct2_s"] = round(use_backend(eng, "ct2"), 1)  # chuyển đổi int8 nếu chưa có

    if args.src and args.ref:
        src, ref = read_lines(args.src, args.n), read_lines(args.ref, args.n)
        assert len(src) == len(ref), "src và ref phải cùng số dòng"
        print(f"[1/3] Dịch {len(src)} câu của tập thử (CTranslate2 int8)...", flush=True)
        hyp, sec = timed_translate(eng, src)
        res["quality"] = {"testset": args.src.name, "sentences": len(src), **scores(hyp, ref),
                          "sentences_per_s": round(len(src) / sec, 2)}
        (args.json.with_suffix(".hyp.txt")).write_text("\n".join(hyp) + "\n", encoding="utf-8")
        print("     ", res["quality"], flush=True)

        if args.speed_n:
            sub_src, sub_ref = src[: args.speed_n], ref[: args.speed_n]
            print(f"[2/3] So tốc độ int8 và fp32 trên {len(sub_src)} câu...", flush=True)
            hyp8, t8 = timed_translate(eng, sub_src)
            import torch

            res["torch_threads"] = torch.get_num_threads()
            use_backend(eng, "hf")
            hyp32, t32 = timed_translate(eng, sub_src)
            res["speed"] = {
                "sentences": len(sub_src),
                "fp32_s": round(t32, 1),
                "int8_s": round(t8, 1),
                "speedup": round(t32 / t8, 2),
                "fp32_chrf": scores(hyp32, sub_ref)["chrf"],
                "int8_chrf": scores(hyp8, sub_ref)["chrf"],
            }
            print("     ", res["speed"], flush=True)
            use_backend(eng, "ct2")

    if args.pdf:
        from app.layout import get_detector

        res["layout_model"] = get_detector() is not None  # nạp sẵn để không tính vào thời gian
        res["pdf"] = []
        for pdf in args.pdf:
            print(f"[3/3] Dịch trọn PDF {pdf.name}...", flush=True)
            r = bench_pdf(args.engine, pdf)
            res["pdf"].append(r)
            print("     ", r, flush=True)

    args.json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(to_markdown(res))
    return 0


def to_markdown(res: dict) -> str:
    lines = [f"### Kết quả đo: {res['model']} trên CPU {res['cpu']} ({res['cpu_count']} luồng)", "",
             "| Chỉ số | Giá trị |", "|---|---|"]
    if q := res.get("quality"):
        lines += [f"| chrF trên {q['testset']} ({q['sentences']} câu) | **{q['chrf']}** |",
                  f"| BLEU trên {q['testset']} | {q['bleu']} |",
                  f"| Tốc độ dịch câu (int8, beam {res['beams']}) | {q['sentences_per_s']} câu/giây |"]
    if s := res.get("speed"):
        lines += [f"| Tăng tốc int8 so với fp32 ({s['sentences']} câu) | **{s['speedup']}×** "
                  f"({s['fp32_s']} s → {s['int8_s']} s) |",
                  f"| chrF fp32 → int8 (cùng {s['sentences']} câu) | {s['fp32_chrf']} → {s['int8_chrf']} |"]
    for p in res.get("pdf", []):
        lines += [f"| Giây mỗi trang, {p['file']} ({p['pages']} trang, {p['words_translated']} từ) | "
                  f"**{p['s_per_page']} s/trang** (phân tích {p['analyze_s']} s, dịch {p['translate_s']} s, "
                  f"dựng {p['render_s']} s) |"]
    lines.append(f"| Nạp mô hình int8 (một lần) | {res['load_ct2_s']} s |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
