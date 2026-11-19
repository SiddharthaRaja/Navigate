"""CPU smoke test for the three model families, with tiny randomly-initialised networks.

    python tests/check_models.py /path/to/bench/hyderabad

It does NOT download anything and does NOT produce meaningful accuracy: the point is to prove that the
training loop, decoding and scoring code all run end to end before any GPU quota is spent.
Takes about two minutes on a laptop.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import metrics as mt
from bench.geo import CITY_CENTRE
from models import addressbind as ab, h3_classifier as hc, llm_geocoder as lg
from models.common import set_seed


def tiny_bert(texts, folder):
    """A 2-layer BERT with a word-piece vocabulary learned from the given texts, saved like a hub model."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import BertConfig, BertModel, BertTokenizerFast
    t = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    t.normalizer = normalizers.BertNormalizer(lowercase=True)
    t.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    t.train_from_iterator(texts, trainers.WordPieceTrainer(vocab_size=3000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tok = BertTokenizerFast(tokenizer_object=t, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    tok.save_pretrained(folder)
    BertModel(BertConfig(vocab_size=t.get_vocab_size(), hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128)).save_pretrained(folder)
    return folder


def tiny_gpt(texts, folder):
    """A 2-layer GPT-2 with a chat template, standing in for Qwen."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    t = Tokenizer(models.BPE(unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=2000, special_tokens=["<unk>", "<pad>", "<eos>", "<|im_start|>", "<|im_end|>"],
                                                     initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    tok.chat_template = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    tok.save_pretrained(folder)
    GPT2LMHeadModel(GPT2Config(vocab_size=t.get_vocab_size(), n_positions=1024, n_embd=64, n_layer=2, n_head=2, eos_token_id=2, pad_token_id=1)).save_pretrained(folder)
    return folder


def main(bench_dir):
    set_seed(0)
    train = pd.read_parquet(f"{bench_dir}/train.parquet")
    test = pd.read_parquet(f"{bench_dir}/test.parquet")
    train = train[~train["abstain"]].head(3000).reset_index(drop=True)
    test = test[test["split"] == "test_warm"].head(200).reset_index(drop=True)
    centre = CITY_CENTRE["hyderabad"]
    tmp = tempfile.mkdtemp()
    bert = tiny_bert(train["text"].tolist(), f"{tmp}/bert")

    print("\n--- A: H3 classifier")
    cells = hc.build_cells(train, resolutions=(7, 8, 9))
    model, tok = hc.train_model(train, cells, encoder_name=bert, epochs=2, batch_size=64, lr=1e-3, log_every=20)
    pred = hc.predict(model, tok, test, cells)
    _, res = mt.evaluate(test, pred, "h3_classifier", slices=False)
    print(res[["model", "n", "coverage", "acc@1000", "p50"]].to_string(index=False))
    assert pred["lat"].notna().all() and len(pred) == len(test)

    print("\n--- A: H3 classifier with the character n-gram encoder (the default)")
    model, tok = hc.train_model(train, cells, encoder_name="char-ngram", epochs=2, batch_size=128, lr=2e-3, head_lr=2e-3, log_every=20)
    pred = hc.predict(model, tok, test, cells)
    assert pred["lat"].notna().all() and pred[["lat", "lon"]].round(5).drop_duplicates().shape[0] > 5
    print("n-gram encoder: ok,", pred[["lat", "lon"]].round(5).drop_duplicates().shape[0], "different answers")

    print("\n--- B: AddressBind")
    model, tok = ab.train_model(train, centre, encoder_name=bert, epochs=2, batch_size=128, lr=1e-3, n_extra=64, log_every=10)
    cand = train.drop_duplicates("h3_10")
    emb = ab.embed_candidates(model, cand["lat"], cand["lon"], centre)
    pred, topk = ab.predict(model, tok, test, emb, cand["lat"].to_numpy(), cand["lon"].to_numpy())
    _, res = mt.evaluate(test, pred, "addressbind", slices=False)
    print(res[["model", "n", "coverage", "acc@1000", "p50"]].to_string(index=False), ab.hit_at_k(test, topk, cand["lat"], cand["lon"], radius_m=1000))
    assert pred["lat"].notna().all() and topk.shape == (len(test), 5)

    print("\n--- C: LLM geocoder (supervised fine-tuning with LoRA)")
    for fmt in ("digipin", "geohash", "coords"):
        la, lo = lg.decode_target(lg.encode_target(17.41234, 78.45678, fmt), fmt)
        assert abs(la - 17.41234) < 2e-4 and abs(lo - 78.45678) < 2e-4, fmt
    gpt = tiny_gpt(train["text"].tolist() + [lg.encode_target(a, b) for a, b in zip(train["lat"], train["lon"])], f"{tmp}/gpt")
    model, tok = lg.load_model(gpt, lora_targets=("c_attn",))
    ctx = ["Some Place = 422-56L-7M2K"] * len(train)
    small = train.head(600)
    lg.train_sft(model, tok, small, context=ctx[:600], epochs=1, batch_size=16, lr=5e-3, log_every=10)
    pred = lg.predict(model, tok, test.head(40), context=ctx[:40], batch_size=20)
    print(pred[["raw", "lat", "lon"]].head(4).to_string(index=False))
    model.save_pretrained(f"{tmp}/adapter")
    model2, _ = lg.load_model(gpt, lora_targets=("c_attn",), adapter=f"{tmp}/adapter")
    assert len(pred) == 40
    print("\nall three families ran end to end")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/bench/hyderabad")
