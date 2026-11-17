"""Small helpers shared by the three model files."""
import os
import time

import numpy as np
import pandas as pd
import torch


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=7):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_bench(bench_dir, train_rows=None, seed=7):
    """Read the benchmark files written by notebook 06 (or 07 if the paraphrased version exists)."""
    from bench import io
    train, val, test = io.read(bench_dir, "train"), io.read(bench_dir, "val"), io.read(bench_dir, "test")
    train = train[~train["abstain"]].reset_index(drop=True)
    if train_rows and len(train) > train_rows:
        train = train.sample(train_rows, random_state=seed).reset_index(drop=True)
    return train, val[~val["abstain"]].reset_index(drop=True), test


def mean_pool(last_hidden, mask):
    """Average the token vectors, ignoring padding."""
    m = mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)


def batches(n, size, shuffle=False, seed=0):
    idx = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    for a in range(0, n, size):
        yield idx[a:a + size]


def warmup_linear(opt, total_steps, warmup=0.06):
    w = max(1, int(total_steps * warmup))
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / w if s < w else max(0.0, (total_steps - s) / max(1, total_steps - w)))


def median_error_m(lat, lon, plat, plon):
    from bench.geo import haversine_m
    ok = ~np.isnan(plat)
    return float(np.median(haversine_m(lat[ok], lon[ok], plat[ok], plon[ok]))) if ok.any() else float("nan")


class Stopwatch:
    def __init__(self):
        self.t = time.time()

    def lap(self):
        t, self.t = time.time() - self.t, time.time()
        return t
