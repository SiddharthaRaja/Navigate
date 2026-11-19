"""Model family A: address text -> H3 cell, by classification.

Follows the idea in Flipkart's COLING 2025 paper: a text encoder with one classification head per H3
resolution, trained together. Coarse heads are easy and stabilise training; the fine head gives the answer.

text --encoder--> vector --+--> head r7 (which ~5 sq km cell?)
                           +--> head r8 (which ~0.7 sq km cell?)
                           +--> head r9 (which ~0.1 sq km cell?)

Prediction walks down the hierarchy: best r7 cell, then the best r8 cell INSIDE it, then the best r9 cell
inside that. The answer is the average position of the training buildings in the chosen cell.

Known limit, by design: it can only ever answer with a cell it saw in training, so it cannot be right in a
cold area. The cold split measures exactly that.
"""
import time

import h3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .common import batches, device, mean_pool, warmup_linear


def build_cells(train, resolutions=(7, 8, 9)):
    """Number the H3 cells seen in training and remember each fine cell's parent and centre."""
    lat, lon = train["lat"].to_numpy(), train["lon"].to_numpy()
    # the fine cell comes from the point; coarser cells are its parents. (H3 cells are not perfectly nested in
    # space, so asking for each resolution separately can give a coarse cell that is not the fine cell's parent.)
    fine_cells = [h3.latlng_to_cell(a, b, resolutions[-1]) for a, b in zip(lat, lon)]
    cells = {r: np.array([h3.cell_to_parent(c, r) for c in fine_cells]) for r in resolutions}
    vocab = {r: {c: i for i, c in enumerate(sorted(set(cells[r])))} for r in resolutions}
    labels = {r: np.array([vocab[r][c] for c in cells[r]]) for r in resolutions}
    parent = {}
    for hi, lo in zip(resolutions[:-1], resolutions[1:]):          # for each fine cell: index of its coarse parent
        parent[lo] = torch.tensor([vocab[hi][h3.cell_to_parent(c, hi)] for c in vocab[lo]], dtype=torch.long)
    fine = resolutions[-1]
    cen = pd.DataFrame({"c": labels[fine], "lat": lat, "lon": lon}).groupby("c")[["lat", "lon"]].mean().reindex(range(len(vocab[fine])))
    return dict(resolutions=tuple(resolutions), vocab=vocab, labels=labels, parent=parent, centre=cen.to_numpy())


# ------------------------------------------------------------------ character n-gram encoder (fastText-style)
class NgramTokenizer:
    """Turns each address into hashed ids of its words and character 2-4-grams (no download, no vocabulary file).
    Spelling slips change only a few n-grams, so 'Kukatpaly' and 'Kukatpally' still share most of their ids."""

    def __init__(self, n_buckets=2 ** 19, nmin=2, nmax=4, max_ids=256):
        self.n_buckets, self.nmin, self.nmax, self.max_ids = n_buckets, nmin, nmax, max_ids

    def ids(self, text):
        import re
        import zlib
        out = []
        for w in re.findall(r"[a-z0-9]+", (text or "").lower()):
            out.append(zlib.crc32(("w:" + w).encode()))
            g = f"<{w}>"
            for n in range(self.nmin, self.nmax + 1):
                out += [zlib.crc32(g[i:i + n].encode()) for i in range(len(g) - n + 1)]
        return [1 + x % (self.n_buckets - 1) for x in out[: self.max_ids]] or [1]

    def __call__(self, texts, **kw):
        from transformers import BatchEncoding
        rows = [self.ids(t) for t in texts]
        n = max(len(r) for r in rows)
        x = torch.zeros(len(rows), n, dtype=torch.long)
        for i, r in enumerate(rows):
            x[i, :len(r)] = torch.tensor(r)
        return BatchEncoding({"input_ids": x, "attention_mask": (x > 0).long()})


class NgramEncoder(nn.Module):
    def __init__(self, n_buckets=2 ** 19, dim=128):
        super().__init__()
        self.emb = nn.Embedding(n_buckets, dim, padding_idx=0)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.hidden_size = dim

    def forward(self, input_ids, attention_mask):
        from types import SimpleNamespace
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


def load_tokenizer(encoder_name):
    return NgramTokenizer() if encoder_name == "char-ngram" else AutoTokenizer.from_pretrained(encoder_name)


class H3Classifier(nn.Module):
    def __init__(self, encoder_name, n_cells, dropout=0.1):
        super().__init__()
        if encoder_name == "char-ngram":
            self.encoder = NgramEncoder()
            d = self.encoder.hidden_size
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name)
            d = self.encoder.config.hidden_size
        # Pretrained encoders give sentence vectors that are almost the same for every input (a few dimensions
        # carry large constant values). The heads then learn "always answer the commonest cell" and the text is
        # ignored. BatchNorm centres and scales every dimension across the batch, so only what differs between
        # addresses is left for the heads to use.
        self.norm = nn.BatchNorm1d(d)
        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(dropout)) if encoder_name == "char-ngram" else nn.Identity()
        d = 512 if encoder_name == "char-ngram" else d
        self.heads = nn.ModuleDict({str(r): nn.Linear(d, n) for r, n in n_cells.items()})

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        v = self.mlp(self.drop(self.norm(mean_pool(h, attention_mask).float())))
        return {int(r): head(v) for r, head in self.heads.items()}


def train_model(train, cells, encoder_name="google/muril-base-cased", epochs=3, batch_size=64, lr=3e-5, head_lr=1e-3,
                max_len=64, loss_weights=None, log_every=200, val=None, seed=7, fp16=True, check_after=300):
    """Returns (model, tokenizer). Prints the loss as it goes, and validation error each epoch if val is given.

    Safety check: after `check_after` steps the model is asked about 500 validation addresses. If it gives fewer
    than 20 different answers it is ignoring the text, and training stops with an error instead of burning GPU time.
    """
    dev = device()
    tok = load_tokenizer(encoder_name)
    model = H3Classifier(encoder_name, {r: len(v) for r, v in cells["vocab"].items()}).to(dev)
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": lr},
                             {"params": list(model.heads.parameters()) + list(model.norm.parameters()) + list(model.mlp.parameters()),
                              "lr": head_lr}], weight_decay=0.01)
    texts = train["text"].tolist()
    steps = epochs * int(np.ceil(len(texts) / batch_size))
    sched = warmup_linear(opt, steps)
    amp = fp16 and dev.type == "cuda"                  # 16-bit maths: faster on GPU, occasionally unstable
    scaler = torch.amp.GradScaler(enabled=amp)
    # coarse cells have many examples each and are learned first; fine cells have few, so they weigh less
    w = loss_weights or dict(zip(cells["resolutions"], (1.0, 0.5, 0.25, 0.25, 0.25)))
    y = {r: torch.tensor(cells["labels"][r]) for r in cells["resolutions"]}
    step, run, t0 = 0, 0.0, time.time()
    for ep in range(epochs):
        model.train()
        for idx in batches(len(texts), batch_size, shuffle=True, seed=seed + ep):
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
            with torch.autocast(dev.type, enabled=amp):
                out = model(enc["input_ids"], enc["attention_mask"])
                loss = sum(w[r] * F.cross_entropy(out[r].float(), y[r][idx].to(dev)) for r in cells["resolutions"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run += loss.item()
            if step % log_every == 0:
                print(f"epoch {ep + 1} step {step}/{steps} loss {run / log_every:.3f} ({time.time() - t0:.0f}s)")
                run = 0.0
            if val is not None and len(val) and step == min(check_after, steps):
                p = predict(model, tok, val.head(500), cells, max_len=max_len)
                spread = p[["lat", "lon"]].round(5).drop_duplicates().shape[0]
                print(f"safety check at step {step}: {spread} different answers for {min(500, len(val))} validation addresses")
                if spread < 20:
                    raise RuntimeError("The model gives the same few answers for every address, so it is ignoring the text. "
                                       "Training stopped to save GPU time. Send the training output for a fix.")
                model.train()
        if val is not None and len(val):
            p = predict(model, tok, val, cells, max_len=max_len)
            from bench.geo import haversine_m
            e = haversine_m(val["lat"].to_numpy(), val["lon"].to_numpy(), p["lat"].to_numpy(), p["lon"].to_numpy())
            spread = p[["lat", "lon"]].round(5).drop_duplicates().shape[0]
            print(f"epoch {ep + 1} done: validation median error {np.median(e):.0f} m, within 500 m {np.mean(e <= 500):.1%}, "
                  f"{spread} different answers for {len(p)} addresses (a handful means the model is ignoring the text)")
    return model, tok


@torch.no_grad()
def predict(model, tok, df, cells, batch_size=256, max_len=64, min_conf=0.0):
    """Hierarchical decode. Returns DataFrame id, lat, lon, conf, latency_ms. conf = probability of the chosen r7 cell;
    rows below min_conf get no answer."""
    dev = next(model.parameters()).device
    model.eval()
    res = cells["resolutions"]
    texts = df["text"].fillna("").tolist()
    out_idx, out_conf, ms = [], [], []
    for idx in batches(len(texts), batch_size):
        t0 = time.time()
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            logits = model(enc["input_ids"], enc["attention_mask"])
        p0 = logits[res[0]].float().softmax(-1)
        conf, chosen = p0.max(-1)
        for r in res[1:]:
            lg = logits[r].float()
            inside = cells["parent"][r].to(dev)[None, :] == chosen[:, None]       # only cells inside the chosen parent
            chosen = lg.masked_fill(~inside, -1e9).argmax(-1)
        out_idx.append(chosen.cpu().numpy())
        out_conf.append(conf.cpu().numpy())
        ms += [(time.time() - t0) * 1000 / len(idx)] * len(idx)
    k, conf = np.concatenate(out_idx), np.concatenate(out_conf)
    ll = cells["centre"][k]
    ok = conf >= min_conf
    return pd.DataFrame({"id": df["id"].to_numpy(), "lat": np.where(ok, ll[:, 0], np.nan), "lon": np.where(ok, ll[:, 1], np.nan),
                         "conf": conf, "latency_ms": ms})
