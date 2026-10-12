"""Dựng mô hình CTranslate2 tí hon (trọng số ngẫu nhiên) + tokenizer HF — không cần torch.

Dùng để kiểm thử thật luồng CTranslate2 (API, cấu hình decoder_start_token, giải mã)
mà không phải tải mô hình vài trăm MB.
"""

from __future__ import annotations

import json
import os

import numpy as np

WORDS = "hello world the cell solar model data neural network deep learning is good".split()
SPECIAL = ["<pad>", "</s>", "<unk>", "vi_VN", "en_XX"]


def build(out: str, dec_start: str | None = None) -> None:
    from ctranslate2.specs import TransformerSpec, model_spec
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    vocab = SPECIAL + WORDS
    rng = np.random.default_rng(0)
    d, ff, v = 16, 32, len(vocab)
    spec = TransformerSpec.from_config(1, 2)

    def shape(path: str) -> tuple[int, ...]:
        if ("embeddings" in path and path.endswith("weight")) or path.endswith("projection.weight"):
            return (v, d)
        if "layer_norm" in path:
            return (d,)
        if "linear_0" in path:
            return (ff, d)
        if "linear_1" in path:
            return (d, ff)
        if "self_attention.linear.0" in path:
            return (3 * d, d)  # q, k, v gộp
        if "attention.linear.1" in path and "self_attention" not in path:
            return (2 * d, d)  # k, v của cross-attention
        return (d, d)

    def fill(obj, path: str = "") -> None:
        for name, val in list(vars(obj).items()):
            if isinstance(val, model_spec.LayerSpec):
                fill(val, path + name + ".")
            elif isinstance(val, list) and val and isinstance(val[0], model_spec.LayerSpec):
                for i, item in enumerate(val):
                    fill(item, f"{path}{name}.{i}.")
            elif val is None:
                p = path + name
                if p.endswith("gamma"):
                    arr = np.ones(shape(p), np.float32)
                elif p.endswith("beta"):
                    arr = np.zeros(shape(p), np.float32)
                else:
                    arr = rng.normal(0, 0.3, shape(p)).astype(np.float32)
                setattr(obj, name, arr)

    fill(spec)
    spec.register_source_vocabulary(vocab)
    spec.register_target_vocabulary(vocab)
    spec.config.bos_token = "</s>"
    spec.config.eos_token = "</s>"
    spec.config.unk_token = "<unk>"
    spec.config.decoder_start_token = "</s>"
    os.makedirs(out, exist_ok=True)
    spec.validate()
    spec.optimize(quantization="int8")
    spec.save(out)

    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocab)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.WordPiece()
    tok.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
        additional_special_tokens=["vi_VN", "en_XX"],
    )
    fast.save_pretrained(out)
    if dec_start:
        cfg_path = os.path.join(out, "config.json")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["decoder_start_token"] = dec_start
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
