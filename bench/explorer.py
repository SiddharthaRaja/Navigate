"""Export benchmark results as one JSON file for the error explorer (docs/index.html).

The explorer is a static web page: open it locally or serve it with GitHub Pages. It reads
docs/data/explorer.json, which this module writes.

What goes in:   test addresses (text, true point, labels), each model's error per address,
                and each model's predicted point where we are allowed to show it.
What stays out: predicted points of paid geocoders (kind="paid"): distances only.
                Anything from private data: pass only public test rows.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from .metrics import add_slices

ROW_COLS = ["id", "split", "text", "style", "relation", "noise_level", "noise_ops", "lang", "source", "ring", "pin_state",
            "ref_name", "ref_in_gazetteer", "components"]


def _r(a, nd):
    a = np.asarray(a, float)
    return [None if np.isnan(v) else round(float(v), nd) for v in a]


def export(test, runs, results, out_path, city="hyderabad", note=""):
    """runs: list of dicts {name, kind, pred (DataFrame or None), scores (DataFrame)}
            kind is one of: "local" (our baselines / models), "open" (Nominatim, Photon), "paid", "reference"
    results: the tidy table from metrics.summarise, all models concatenated
    """
    t = add_slices(test).reset_index(drop=True)
    for c in ROW_COLS:
        if c not in t:
            t[c] = None
    rows = {c: [None if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in t[c].tolist()] for c in ROW_COLS}
    rows["lat"], rows["lon"] = _r(t["lat"], 5), _r(t["lon"], 5)
    rows["abstain"] = [bool(v) for v in t["abstain"]]
    rows["oracle"] = _r(t.get("oracle_err_m", np.nan), 0)
    rows["ref_lat"], rows["ref_lon"] = _r(t.get("ref_lat", np.nan), 5), _r(t.get("ref_lon", np.nan), 5)
    rows["ref_dist"] = _r(t.get("ref_dist_m", np.nan), 0)

    models, pred = [], {}
    for run in runs:
        s = run["scores"].set_index("id").reindex(t["id"])
        entry = {"err": _r(s["err_m"], 0)}
        show = run["kind"] in ("local", "open") and run.get("pred") is not None
        if show:
            p = run["pred"].drop_duplicates("id").set_index("id").reindex(t["id"])
            entry["lat"], entry["lon"] = _r(p["lat"], 5), _r(p["lon"], 5)
            if "level" in p:
                entry["level"] = [None if not isinstance(v, str) else v for v in p["level"]]
        pred[run["name"]] = entry
        models.append({"name": run["name"], "kind": run["kind"], "points": bool(show), "about": run.get("about", "")})

    keep = ["model", "split", "slice", "value", "n", "coverage", "acc@100", "acc@200", "acc@500", "acc@1000", "acc@2000",
            "mean_m", "p50", "p90", "p95", "misroute", "wrong_side", "access_p50", "abstain_p", "abstain_r"]
    res = results[[c for c in keep if c in results]].copy()
    res = res.astype(object).where(res.notna(), None)
    table = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in res.to_dict("records")]

    out = {"meta": {"city": city, "generated": time.strftime("%Y-%m-%d"), "n": len(t), "note": note,
                    "splits": sorted(t["split"].unique().tolist())},
           "models": models, "rows": rows, "pred": pred, "results": table}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    return os.path.getsize(out_path)
