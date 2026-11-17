"""Model family B: put address text and map locations in the same vector space, then search.

Follows Amazon's AddressBind (SIGSPATIAL 2025): two encoders trained with a contrastive loss so that an
address lands next to its own location and far from other locations.

text     --MiniLM--------------------------> vector  \
                                                       cosine similarity, InfoNCE loss
location --Fourier features--MLP-----------> vector  /

Geocoding = embed every candidate point of the city once, embed the address, take the best-scoring point.
Unlike the classifier, the location encoder is a smooth function of position, so it can score places it never
saw during training: the question for the cold split is how well.
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .common import batches, device, mean_pool, warmup_linear

KM_PER_DEG = 111.32


def to_km(lat, lon, centre):
    """Degrees -> kilometres east / north of the city centre (flat-earth is fine at city scale)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    x = (lon - centre[1]) * KM_PER_DEG * np.cos(np.radians(centre[0]))
    y = (lat - centre[0]) * KM_PER_DEG
    return np.stack([x, y], -1).astype(np.float32)


class LocationEncoder(nn.Module):
    """Random Fourier features at several length scales, then a small MLP.
    Each scale lets the network tell apart places that far apart: 20 km (which side of town) down to 50 m."""

    def __init__(self, dim=384, scales_km=(20, 5, 1.5, 0.5, 0.15, 0.05), n_freq=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        B = torch.cat([torch.randn(n_freq, 2, generator=g) / s for s in scales_km])           # frequencies, fixed (not trained)
        self.register_buffer("B", B)
        self.mlp = nn.Sequential(nn.Linear(2 * len(B), 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, dim))

    def forward(self, xy_km):
        z = xy_km @ self.B.T
        return self.mlp(torch.cat([z.sin(), z.cos()], -1))


class AddressBind(nn.Module):
    def __init__(self, encoder_name, dim=384):
        super().__init__()
        self.text = AutoModel.from_pretrained(encoder_name)
        self.proj = nn.Linear(self.text.config.hidden_size, dim)
        self.loc = LocationEncoder(dim)
        self.log_temp = nn.Parameter(torch.tensor(np.log(1 / 0.05), dtype=torch.float32))      # learned temperature

    def embed_text(self, input_ids, attention_mask):
        h = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return F.normalize(self.proj(mean_pool(h, attention_mask)).float(), dim=-1)

    def embed_loc(self, xy_km):
        return F.normalize(self.loc(xy_km).float(), dim=-1)


def info_nce(t, l, l_extra, log_temp):
    """Each address must pick its own location out of: the other locations in the batch + extra random points."""
    scale = log_temp.exp().clamp(max=100)
    logits_t = scale * t @ torch.cat([l, l_extra]).T          # text -> locations
    logits_l = scale * l @ t.T                                 # location -> texts
    target = torch.arange(len(t), device=t.device)
    return 0.5 * (F.cross_entropy(logits_t, target) + F.cross_entropy(logits_l, target))


def train_model(train, centre, encoder_name="sentence-transformers/all-MiniLM-L6-v2", epochs=4, batch_size=256, lr=5e-5, loc_lr=5e-4,
                max_len=64, n_extra=512, hard_radius_km=1.0, log_every=100, seed=7):
    """n_extra random negatives per step: half anywhere in the city, half within hard_radius_km of a batch
    address (near misses are what teach fine position)."""
    dev = device()
    tok = AutoTokenizer.from_pretrained(encoder_name)
    model = AddressBind(encoder_name).to(dev)
    opt = torch.optim.AdamW([{"params": list(model.text.parameters()) + list(model.proj.parameters()), "lr": lr},
                             {"params": list(model.loc.parameters()) + [model.log_temp], "lr": loc_lr}], weight_decay=0.01)
    texts = train["text"].tolist()
    xy = to_km(train["lat"], train["lon"], centre)
    lo, hi = xy.min(0), xy.max(0)
    steps = epochs * int(np.ceil(len(texts) / batch_size))
    sched = warmup_linear(opt, steps)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    rng = np.random.default_rng(seed)
    step, run, t0 = 0, 0.0, time.time()
    for ep in range(epochs):
        model.train()
        for idx in batches(len(texts), batch_size, shuffle=True, seed=seed + ep):
            if len(idx) < 8:
                continue
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
            anywhere = rng.uniform(lo, hi, size=(n_extra // 2, 2))
            nearby = xy[rng.choice(idx, n_extra - n_extra // 2)] + rng.normal(0, hard_radius_km / 2, size=(n_extra - n_extra // 2, 2))
            extra = torch.tensor(np.concatenate([anywhere, nearby]), dtype=torch.float32, device=dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                t = model.embed_text(enc["input_ids"], enc["attention_mask"])
                l = model.embed_loc(torch.tensor(xy[idx], device=dev))
                loss = info_nce(t, l, model.embed_loc(extra), model.log_temp)
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
                print(f"epoch {ep + 1} step {step}/{steps} loss {run / log_every:.3f} temp {1 / model.log_temp.exp().item():.3f} ({time.time() - t0:.0f}s)")
                run = 0.0
    return model, tok


@torch.no_grad()
def embed_candidates(model, cand_lat, cand_lon, centre, batch_size=65536):
    """Embed every candidate point once. Candidates = centres of the H3 cells that contain buildings."""
    dev = next(model.parameters()).device
    model.eval()
    xy = torch.tensor(to_km(cand_lat, cand_lon, centre))
    return torch.cat([model.embed_loc(xy[a:a + batch_size].to(dev)).half() for a in range(0, len(xy), batch_size)])


@torch.no_grad()
def predict(model, tok, df, cand_emb, cand_lat, cand_lon, batch_size=256, max_len=64, top_k=5, min_score=-1.0):
    """Best-scoring candidate for each address. Also returns the top_k candidate indices (for hit@K) and the score."""
    dev = next(model.parameters()).device
    model.eval()
    texts = df["text"].fillna("").tolist()
    best, score, topk, ms = [], [], [], []
    for idx in batches(len(texts), batch_size):
        t0 = time.time()
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            t = model.embed_text(enc["input_ids"], enc["attention_mask"])
        s = t.to(cand_emb.dtype) @ cand_emb.T
        v, k = s.topk(top_k, dim=-1)
        best.append(k[:, 0].cpu().numpy()); score.append(v[:, 0].float().cpu().numpy()); topk.append(k.cpu().numpy())
        ms += [(time.time() - t0) * 1000 / len(idx)] * len(idx)
    best, score, topk = np.concatenate(best), np.concatenate(score), np.concatenate(topk)
    ok = score >= min_score
    cand_lat, cand_lon = np.asarray(cand_lat), np.asarray(cand_lon)
    pred = pd.DataFrame({"id": df["id"].to_numpy(), "lat": np.where(ok, cand_lat[best], np.nan), "lon": np.where(ok, cand_lon[best], np.nan),
                         "conf": score, "latency_ms": ms})
    return pred, topk


def hit_at_k(df, topk, cand_lat, cand_lon, radius_m=100, ks=(1, 5)):
    """AddressBind's retrieval metric: is any of the top-k candidates within radius_m of the truth?"""
    from bench.geo import haversine_m
    cand_lat, cand_lon = np.asarray(cand_lat), np.asarray(cand_lon)
    d = haversine_m(df["lat"].to_numpy()[:, None], df["lon"].to_numpy()[:, None], cand_lat[topk], cand_lon[topk])
    return {f"hit@{k}": float((d[:, :k] <= radius_m).any(1).mean()) for k in ks}
