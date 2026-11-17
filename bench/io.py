"""Reading and writing the benchmark files, so every notebook uses the same version of the text."""
import os

import pandas as pd


def read(bench_dir, name):
    """name: 'train', 'val' or 'test'. If notebook 07 wrote a paraphrased version, that one is used."""
    for f in (f"{name}_paraphrased.parquet", f"{name}.parquet"):
        p = os.path.join(bench_dir, f)
        if os.path.exists(p):
            return pd.read_parquet(p)
    raise FileNotFoundError(f"{name}.parquet not found in {bench_dir}: run notebook 06 first")


def save_run(runs_dir, model, pred=None, scores=None, kind="local", about=""):
    """Store one model's output. pred (id, lat, lon) for our own models and open geocoders;
    scores only (no coordinates) for paid geocoders."""
    import json
    d = os.path.join(runs_dir, model)
    os.makedirs(d, exist_ok=True)
    if pred is not None:
        assert kind != "paid", "paid geocoders: save scores only, never coordinates"
        pred.to_parquet(f"{d}/pred_test.parquet", index=False)
    if scores is not None:
        scores.to_parquet(f"{d}/scores_test.parquet", index=False)
    json.dump({"model": model, "kind": kind, "about": about}, open(f"{d}/run.json", "w"))
    return d


def list_runs(runs_dir):
    """Every saved run: list of dicts with model, kind, about, pred (or None), scores (or None)."""
    import json
    out = []
    for m in sorted(os.listdir(runs_dir)) if os.path.exists(runs_dir) else []:
        d = os.path.join(runs_dir, m)
        if not os.path.exists(f"{d}/run.json"):
            continue
        r = json.load(open(f"{d}/run.json"))
        r["pred"] = pd.read_parquet(f"{d}/pred_test.parquet") if os.path.exists(f"{d}/pred_test.parquet") else None
        r["scores"] = pd.read_parquet(f"{d}/scores_test.parquet") if os.path.exists(f"{d}/scores_test.parquet") else None
        r["name"] = r["model"]
        out.append(r)
    return out
