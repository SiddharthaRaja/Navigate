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


class H3Classifier(nn.Module):
    def __init__(self, encoder_name, n_cells, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        d = self.encoder.config.hidden_size
        # Pretrained encoders give sentence vectors that are almost the same for every input (a few dimensions
        # carry large constant values). The heads then learn "always answer the commonest cell" and the text is
        # ignored. BatchNorm centres and scales every dimension across the batch, so only what differs between
        # addresses is left for the heads to use.
        self.norm = nn.BatchNorm1d(d)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({str(r): nn.Linear(d, n) for r, n in n_cells.items()})

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        v = self.drop(self.norm(mean_pool(h, attention_mask).float()))
        return {int(r): head(v) for r, head in self.heads.items()}


def train_model(train, cells, encoder_name="google/muril-base-cased", epochs=3, batch_size=64, lr=3e-5, head_lr=1e-3,
                max_len=64, loss_weights=None, log_every=200, val=None, seed=7, fp16=True):
    """Returns (model, tokenizer). Prints the loss as it goes, and validation error each epoch if val is given."""
    dev = device()
    tok = AutoTokenizer.from_pretrained(encoder_name)
    model = H3Classifier(encoder_name, {r: len(v) for r, v in cells["vocab"].items()}).to(dev)
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": lr},
                             {"params": list(model.heads.parameters()) + list(model.norm.parameters()), "lr": head_lr}], weight_decay=0.01)
    texts = train["text"].tolist()
    steps = epochs * int(np.ceil(len(texts) / batch_size))
    sched = warmup_linear(opt, steps)
    amp = fp16 and dev.type == "cuda"                  # 16-bit maths: faster on GPU, occasionally unstable
    scaler = torch.amp.GradScaler(enabled=amp)
    w = loss_weights or {r: 1.0 for r in cells["resolutions"]}
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
