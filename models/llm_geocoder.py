"""Model family C: a small language model that writes the location as a short code.

Follows the generative geocoding idea in ReaGeo (Alibaba Amap, 2026): fine-tune an LLM so that, given an
address, it writes a hierarchical location code one character at a time. Each extra character it gets right
shrinks the error, and it can use knowledge an encoder lacks ("opposite" means the other side of the road).

Output formats (pick one, compare them):
    digipin  India Post's 10-character grid code, e.g. 422-56L-7M2K   (each character = a 4x4 subdivision)
    geohash  8-character geohash, e.g. tepfbq2k                        (each character = 32 subdivisions)
    coords   plain "17.41234, 78.45678"

Optional retrieval: places from the public gazetteer whose names appear in the address are listed in the
prompt with their codes, so the model can copy a prefix and adjust. That is its route into cold areas.

Training is supervised fine-tuning with LoRA adapters (only ~1% of the weights change).
The reinforcement-learning stage (GRPO with a distance reward) comes after this and is a separate notebook.
"""
import re
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench.geo import digipin_decode, digipin_encode, geohash_decode, geohash_encode
from .common import batches, device, warmup_linear

SYSTEM = {"digipin": "You are a geocoder for Hyderabad, India. Reply with the DIGIPIN code of the address and nothing else.",
          "geohash": "You are a geocoder for Hyderabad, India. Reply with the 8-character geohash of the address and nothing else.",
          "coords": "You are a geocoder for Hyderabad, India. Reply with the latitude and longitude of the address as two decimal numbers and nothing else."}


# ------------------------------------------------------------------ location <-> text
def encode_target(lat, lon, fmt="digipin"):
    if fmt == "digipin":
        c = digipin_encode(lat, lon)
        return f"{c[:3]}-{c[3:6]}-{c[6:]}"
    if fmt == "geohash":
        return geohash_encode(lat, lon, 8)
    return f"{lat:.5f}, {lon:.5f}"


def decode_target(text, fmt="digipin"):
    """Model output -> (lat, lon), or None if it is not a usable code. A valid prefix still gives a (coarser) point."""
    text = (text or "").strip().split("\n")[0]
    try:
        if fmt == "digipin":                               # must START with a well-formed code; stray letters do not count
            m = re.match(r"\s*([FCJKLMPT2-9]{3})-?([FCJKLMPT2-9]{3})-?([FCJKLMPT2-9]{0,4})", text.upper())
            return digipin_decode("".join(m.groups())) if m else None
        if fmt == "geohash":
            m = re.match(r"\s*([0-9b-hjkmnp-z]{5,8})(?![a-z0-9])", text.lower())
            return geohash_decode(m.group(1)) if m else None
        m = re.findall(r"-?\d+\.\d+", text)
        if len(m) >= 2:
            lat, lon = float(m[0]), float(m[1])
            return (lat, lon) if 6 < lat < 38 and 68 < lon < 98 else None
    except Exception:
        return None
    return None


def make_context(gazetteer_geocoder, df, fmt="digipin", k=4):
    """For each address: up to k gazetteer places named in it, written as 'Name = code'. Uses bench.baselines."""
    hits = gazetteer_geocoder.lookup(df, k=k)
    return ["; ".join(f"{name} = {encode_target(lat, lon, fmt)}" for name, kind, lat, lon in h) for h in hits]


def build_prompt(tok, address, fmt="digipin", context=None):
    user = f"Address: {address}"
    if context:
        user += f"\nKnown places: {context}"
    msgs = [{"role": "system", "content": SYSTEM[fmt]}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ------------------------------------------------------------------ model
def _remove_old_torchao():
    """Kaggle ships torchao 0.10, and recent peft refuses to start when it finds a torchao older than it supports.
    Nothing here uses torchao, so an old copy is uninstalled before peft is imported."""
    import importlib
    import importlib.metadata as md
    import subprocess
    import sys
    try:
        v = md.version("torchao")
    except md.PackageNotFoundError:
        return
    major, minor = (int(x) for x in v.split(".")[:2])
    if (major, minor) < (0, 16):
        print(f"removing torchao {v} (too old for peft, not used here)")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
        importlib.invalidate_caches()


def load_model(name="Qwen/Qwen2.5-0.5B-Instruct", lora_r=16, lora_alpha=32, lora_targets=("q_proj", "k_proj", "v_proj", "o_proj"), adapter=None):
    """Base model + LoRA adapters. Pass adapter=<folder> to load adapters trained earlier."""
    _remove_old_torchao()
    from peft import LoraConfig, PeftModel, get_peft_model
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(name).to(device())
    if adapter:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, target_modules=list(lora_targets), task_type="CAUSAL_LM"))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"{name}: training {n_train / 1e6:.1f}M of {n_all / 1e6:.0f}M weights ({n_train / n_all:.2%})")
    return model, tok


def _encode_batch(tok, prompts, answers, max_len):
    """Token ids for prompt + answer; the loss is computed on the answer tokens only (labels = -100 elsewhere)."""
    ids, labels = [], []
    for p, a in zip(prompts, answers):
        pi = tok(p, add_special_tokens=False)["input_ids"][-(max_len - 24):]
        ai = tok(a, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids.append(pi + ai)
        labels.append([-100] * len(pi) + ai)
    n = max(len(x) for x in ids)
    pad = tok.pad_token_id
    x = torch.tensor([s + [pad] * (n - len(s)) for s in ids])
    y = torch.tensor([s + [-100] * (n - len(s)) for s in labels])
    return x, (x != pad) | (y != -100), y


def train_sft(model, tok, train, fmt="digipin", context=None, epochs=2, batch_size=16, lr=2e-4, max_len=160, log_every=100, seed=7):
    """Supervised fine-tuning. context: optional list of retrieval strings, one per training row."""
    dev = device()
    prompts = [build_prompt(tok, t, fmt, c) for t, c in zip(train["text"], context or [None] * len(train))]
    answers = [encode_target(a, b, fmt) for a, b in zip(train["lat"], train["lon"])]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = epochs * int(np.ceil(len(prompts) / batch_size))
    sched = warmup_linear(opt, steps, warmup=0.03)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    step, run, t0 = 0, 0.0, time.time()
    model.train()
    for ep in range(epochs):
        for idx in batches(len(prompts), batch_size, shuffle=True, seed=seed + ep):
            x, mask, y = _encode_batch(tok, [prompts[i] for i in idx], [answers[i] for i in idx], max_len)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = model(input_ids=x.to(dev), attention_mask=mask.to(dev), labels=y.to(dev)).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run += loss.item()
            if step % log_every == 0:
                print(f"epoch {ep + 1} step {step}/{steps} loss {run / log_every:.3f} ({time.time() - t0:.0f}s)")
                run = 0.0
    return model


@torch.no_grad()
def predict(model, tok, df, fmt="digipin", context=None, batch_size=64, max_len=160, max_new_tokens=16):
    """Greedy decoding. Returns DataFrame id, lat, lon, raw (the text the model wrote), latency_ms."""
    dev = device()
    model.eval()
    tok.padding_side = "left"                                   # generation needs the prompt flush against the answer
    prompts = [build_prompt(tok, t, fmt, c) for t, c in zip(df["text"].fillna(""), context or [None] * len(df))]
    raw, ms = [], []
    for idx in batches(len(prompts), batch_size):
        t0 = time.time()
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=max_len, add_special_tokens=False).to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        raw += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        ms += [(time.time() - t0) * 1000 / len(idx)] * len(idx)
    ll = [decode_target(r, fmt) for r in raw]
    return pd.DataFrame({"id": df["id"].to_numpy(), "lat": [p[0] if p else np.nan for p in ll], "lon": [p[1] if p else np.nan for p in ll],
                         "raw": raw, "latency_ms": ms})
