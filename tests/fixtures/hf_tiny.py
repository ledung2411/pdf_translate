"""Dựng mô hình Hugging Face tí hon (Marian kiểu OPUS, mBART-50 kiểu VinAI) với trọng số ngẫu nhiên.

Cần torch + transformers + sentencepiece. Dùng để kiểm thử thật bước chuyển đổi sang
CTranslate2 và so khớp backend ct2 với backend transformers.
"""

from __future__ import annotations

import json
from pathlib import Path

CORPUS = [
    "the solar cell is good", "deep learning model works well", "neural network training data",
    "hello world from the lab", "cortical surface reconstruction from diffusion images",
    "chúng tôi đề xuất một phương pháp mới", "mô hình học sâu cho ảnh y tế",
] * 20


def _train_spm(prefix: Path, vocab_size: int = 60) -> Path:
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(CORPUS), model_prefix=str(prefix), vocab_size=vocab_size,
        character_coverage=1.0, hard_vocab_limit=False, model_type="unigram",
        pad_id=-1, bos_id=-1, eos_id=2, unk_id=1, minloglevel=2,
    )
    return prefix.with_suffix(".model")


def build_marian(out: Path) -> Path:
    import torch
    from transformers import MarianConfig, MarianMTModel, MarianTokenizer

    out.mkdir(parents=True, exist_ok=True)
    spm_path = _train_spm(out / "spm")
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(spm_path))
    pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    vocab = {"</s>": 0, "<unk>": 1}
    for p in pieces:
        vocab.setdefault(p, len(vocab))
    vocab["<pad>"] = len(vocab)
    (out / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    for name in ("source.spm", "target.spm"):
        (out / name).write_bytes(spm_path.read_bytes())
    tok = MarianTokenizer(str(out / "source.spm"), str(out / "target.spm"), str(out / "vocab.json"))
    torch.manual_seed(0)
    cfg = MarianConfig(
        vocab_size=len(vocab), d_model=16, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
        decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=64,
        pad_token_id=vocab["<pad>"], eos_token_id=0, decoder_start_token_id=vocab["<pad>"],
        forced_eos_token_id=0,
    )
    MarianMTModel(cfg).eval().save_pretrained(str(out))
    tok.save_pretrained(str(out))
    return out


def build_mbart50(out: Path) -> Path:
    import torch
    from transformers import MBart50Tokenizer, MBartConfig, MBartForConditionalGeneration

    out.mkdir(parents=True, exist_ok=True)
    spm_path = _train_spm(out / "sentencepiece.bpe")
    tok = MBart50Tokenizer(str(spm_path), src_lang="en_XX", tgt_lang="vi_VN")
    torch.manual_seed(0)
    cfg = MBartConfig(
        vocab_size=len(tok), d_model=16, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
        decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=64,
        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        decoder_start_token_id=tok.convert_tokens_to_ids("vi_VN"),
    )
    MBartForConditionalGeneration(cfg).eval().save_pretrained(str(out))
    tok.save_pretrained(str(out))
    return out
