"""Tự chẩn đoán môi trường khi dịch không chạy được.

    python -m app.doctor            # kiểm tra thư viện, GPU, mô hình
    python -m app.doctor --try opus        # thử dịch vài câu khoa học bằng 'opus'
    python -m app.doctor --try opus,vinai  # so sánh chất lượng & tốc độ 2 mô hình
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
import time

OK, BAD, WARN = "[OK]  ", "[LỖI]", "[!]   "

SAMPLES = [
    "Deep neural networks have achieved remarkable results in computer vision.",
    "As shown in Fig. 3, the proposed method outperforms the baseline (p < 0.05) [12].",
    "We fine-tune the pre-trained model with a learning rate of 3e-4 and a batch size of 32.",
    "The catalyst exhibits high thermal stability and a narrow band gap of 2.1 eV.",
    "Patients in the control group received a placebo for 12 weeks.",
]


def check_module(name: str, required: bool = True) -> bool:
    try:
        mod = importlib.import_module(name)
        print(f"{OK} {name} {getattr(mod, '__version__', '')}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"{BAD if required else WARN} {name}: {e.__class__.__name__}: {e}")
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--try", dest="engine", help="thử dịch vài câu bằng các engine này, cách nhau dấu phẩy (vd. opus,vinai)")
    args = ap.parse_args(argv)

    print(f"Python {sys.version.split()[0]} — {platform.platform()}")
    if sys.version_info < (3, 10):
        print(f"{BAD} Cần Python 3.10 trở lên.")
    good = True
    print("\n# Thư viện lõi")
    for m in ("pymupdf", "pymupdf_fonts", "fastapi", "uvicorn", "multipart"):
        good &= check_module(m)
    print("\n# Thư viện mô hình (cho vinai/opus/nllb)")
    ml = all([check_module("torch"), check_module("transformers"), check_module("sentencepiece")])
    check_module("sacremoses", required=False)
    if ml:
        import torch

        if torch.cuda.is_available():
            print(f"{OK} GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"{WARN} Không có GPU — dịch bằng CPU (chậm hơn, có thể đặt PDFTR_NUM_BEAMS=2 để nhanh hơn).")
        try:
            from huggingface_hub import scan_cache_dir

            repos = [r.repo_id for r in scan_cache_dir().repos]
            print(f"{OK} Mô hình đã tải: {', '.join(repos) or '(chưa có — sẽ tự tải ở lần dịch đầu)'}")
        except Exception:  # noqa: BLE001
            pass
    else:
        print(f"{BAD} Thiếu thư viện mô hình. Cài: pip install -r requirements-ml.txt")

    from .translators import default_engine_key, list_engines

    print("\n# Engine")
    for e in list_engines():
        ok, why = e.available()
        print(f"{OK if ok else WARN} {e.key:6} {e.label}" + ("" if ok else f" — {why}"))
    print(f"      Mặc định: {default_engine_key()}")

    if args.engine:
        from .pipeline import TextTranslator
        from .scientific import Glossary, load_builtin_glossary
        from .translators import get_engine

        glossary = Glossary(load_builtin_glossary())
        for key in [k.strip() for k in args.engine.split(",") if k.strip()]:
            print(f"\n# Thử dịch bằng '{key}' (có bảng thuật ngữ, giống khi dịch PDF)")
            try:
                eng = get_engine(key)
                t0 = time.time()
                eng.prepare(lambda m: print("   ", m))
                print(f"    Nạp mô hình: {time.time() - t0:.1f}s")
                tr = TextTranslator(eng, glossary, use_cache=False)
                t1 = time.time()
                outs = tr.translate_texts(SAMPLES)
                per = (time.time() - t1) / len(SAMPLES)
                for en, vi in zip(SAMPLES, outs):
                    print(f"  EN: {en}\n  VI: {vi}\n")
                print(f"{OK} Tốc độ: {per:.1f} giây/câu")
            except Exception as e:  # noqa: BLE001
                good = False
                print(f"{BAD} {e.__class__.__name__}: {e}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
