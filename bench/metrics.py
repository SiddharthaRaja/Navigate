"""One scoring harness for every model and every geocoder.

Two steps, kept separate on purpose:

score_rows(truth, pred)   -> one row per test address: error in metres and a few yes/no checks
summarise(truth, scores)  -> the tables: overall per split, and per slice

Commercial geocoders are scored in memory and only the output of score_rows is saved (distances and flags,
never their coordinates).

Metric names and where they come from
-------------------------------------
coverage        share of answerable addresses that got any answer
acc@100..2000   share of answerable addresses predicted within N metres (no answer counts as a miss)
                Flipkart reports @100/500/1000/2000, ReaGeo @100/200/500
mean_m, p25..p95  error distance over answered rows (Amazon reports p25/p50/p95, ReaGeo the mean)
defect_1km      share of answerable addresses that are more than 1 km off or unanswered
geohash6        share where prediction and truth fall in the same geohash-6 cell (about 1.2 km x 0.6 km)
misroute        share where the prediction falls in a different pincode area from the truth
wrong_side      of predictions within 150 m of the truth: share on the wrong side of the building's road
access_p50      median distance from the prediction to the building's road access point
abstain_p / _r  precision and recall of "no answer" on rows that truly have no answer (gibberish)
"""
import numpy as np
import pandas as pd

from .geo import UTM, geohash_encode, haversine_m

ACC = [100, 200, 500, 1000, 2000]
SLICES = ["style", "relation", "noise_level", "lang", "ref_in_gazetteer", "ring", "source", "pin_state", "cold_depth", "has_house_no", "paraphrased", "area_seen"]
NOISE_OPS = ["typo", "abbrev", "merge", "junk", "shuffle", "omit", "lang", "name_variant", "instructions", "wrong_pin_far", "wrong_pin_adjacent"]


def _to_utm(lat, lon):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    return tr.transform(np.asarray(lon, float), np.asarray(lat, float))


def score_rows(truth, pred, pin_lookup=None, targets=None):
    """truth: benchmark rows.  pred: DataFrame with columns id, lat, lon (NaN = the model gave no answer).

    targets (optional): the targets table from notebook 04, used for the wrong-side-of-road check.
    """
    p = pred.drop_duplicates("id").set_index("id").reindex(truth["id"])
    plat, plon = p["lat"].to_numpy(float), p["lon"].to_numpy(float)
    tlat, tlon = truth["lat"].to_numpy(float), truth["lon"].to_numpy(float)
    answered = ~np.isnan(plat) & ~np.isnan(plon)
    s = pd.DataFrame({"id": truth["id"].to_numpy(), "answered": answered})
    s["err_m"] = np.where(answered, haversine_m(tlat, tlon, np.nan_to_num(plat), np.nan_to_num(plon)), np.nan).round(1)
    gh_t = np.array([geohash_encode(a, b, 6) for a, b in zip(tlat, tlon)])
    gh_p = np.array([geohash_encode(a, b, 6) if ok else "" for a, b, ok in zip(plat, plon, answered)])
    s["gh6"] = np.where(answered, gh_t == gh_p, np.nan)

    alat, alon = truth["access_lat"].to_numpy(float), truth["access_lon"].to_numpy(float)
    ok = answered & ~np.isnan(alat)
    s["access_m"] = np.where(ok, haversine_m(np.nan_to_num(alat), np.nan_to_num(alon), np.nan_to_num(plat), np.nan_to_num(plon)), np.nan).round(1)

    s["misroute"] = np.nan
    if pin_lookup is not None:
        have = answered & truth["pincode_true"].notna().to_numpy()
        if have.any():
            pp, _ = pin_lookup(plat[have], plon[have])
            s.loc[have, "misroute"] = (np.asarray(pp, object) != truth["pincode_true"].to_numpy()[have]).astype(float)

    s["wrong_side"] = np.nan
    if targets is not None:
        r = targets.drop_duplicates("building_id").set_index("building_id").reindex(truth["building_id"])
        near = answered & (s["err_m"].to_numpy() <= 150) & r["r_side"].notna().to_numpy() & (r["r_dist"].to_numpy() <= 60)
        if near.any():
            px, py = _to_utm(plat[near], plon[near])
            rr = r[near]
            side = np.sign(rr["r_dx"].to_numpy() * (py - rr["r_sy"].to_numpy()) - rr["r_dy"].to_numpy() * (px - rr["r_sx"].to_numpy()))
            s.loc[near, "wrong_side"] = (side != rr["r_side"].to_numpy()).astype(float)
    if "latency_ms" in p:
        s["latency_ms"] = p["latency_ms"].to_numpy()
    return s


def add_slices(truth):
    """Derived columns that results are sliced by."""
    t = truth.copy()
    pit, ptr = t.get("pincode_in_text"), t.get("pincode_true")
    t["pin_state"] = np.where(pit.isna(), "no pincode", np.where(ptr.isna(), "unknown", np.where(pit == ptr, "correct pincode", "wrong pincode")))
    if "nn_train_m" in t:
        nn = t["nn_train_m"].astype(float)
        t["cold_depth"] = np.where(~t["split"].isin(["test_cold"]), None, np.where(nn > 300, "cold interior (>300 m from train)", "cold border"))
    t["has_house_no"] = t.get("synthetic_tokens", pd.Series("", index=t.index)).fillna("").str.contains("house_no").map({True: "house no", False: "no house no"})
    if "ref_in_gazetteer" in t:
        t["ref_in_gazetteer"] = t["ref_in_gazetteer"].map({True: "landmark in gazetteer", False: "landmark hidden", 1.0: "landmark in gazetteer", 0.0: "landmark hidden"})
    if "area_seen" in t:
        t["area_seen"] = t["area_seen"].map({True: "area seen in training", False: "area never seen"})
    if "paraphrased" in t:
        t["paraphrased"] = t["paraphrased"].fillna(False).map({True: "rewritten by LLM", False: "template text"})
    ops = t.get("noise_ops", pd.Series("", index=t.index)).fillna("")
    for o in NOISE_OPS:
        t[f"op_{o}"] = ops.str.split(",").map(lambda x: o in x)
    return t


def _stats(g):
    """g: scores joined with truth for one group."""
    real = ~g["abstain"].to_numpy(bool)
    ans = g["answered"].to_numpy(bool)
    e = g["err_m"].to_numpy(float)
    n = int(real.sum())
    out = {"n": n, "coverage": ans[real].mean() if n else np.nan}
    hit = real & ans
    for k in ACC:
        out[f"acc@{k}"] = (e[hit] <= k).sum() / n if n else np.nan
    if hit.any():
        q = np.percentile(e[hit], [25, 50, 75, 90, 95])
        out.update(mean_m=e[hit].mean(), p25=q[0], p50=q[1], p75=q[2], p90=q[3], p95=q[4])
    out["defect_1km"] = 1 - out["acc@1000"] if n else np.nan
    for col, name, fn in (("gh6", "geohash6", "mean"), ("misroute", "misroute", "mean"), ("wrong_side", "wrong_side", "mean"), ("access_m", "access_p50", "median")):
        v = g.loc[hit, col].astype(float) if col in g else pd.Series(dtype=float)
        out[name] = getattr(v, fn)() if v.notna().any() else np.nan
    n_ab = int((~real).sum())
    if n_ab:
        said_no = ~ans
        out["abstain_r"] = said_no[~real].mean()
        out["abstain_p"] = (~real)[said_no].mean() if said_no.any() else np.nan
        out["n_abstain"] = n_ab
    if "latency_ms" in g and g["latency_ms"].notna().any():
        out["latency_p50_ms"] = g["latency_ms"].median()
    return out


def summarise(truth, scores, model, slices=True, min_n=30):
    """Tidy results table: one row per (split, slice, value)."""
    t = add_slices(truth)
    g = t.merge(scores, on="id", how="left")
    g["answered"] = g["answered"].fillna(False).astype(bool)
    rows = []
    for split, gs in g.groupby("split"):
        rows.append({"model": model, "split": split, "slice": "overall", "value": "all", **_stats(gs)})
        if not slices:
            continue
        for col in SLICES:
            if col not in gs:
                continue
            for val, gv in gs[~gs["abstain"].astype(bool)].groupby(col):
                if len(gv) >= min_n:
                    rows.append({"model": model, "split": split, "slice": col, "value": str(val), **_stats(gv)})
        for o in NOISE_OPS:
            gv = gs[gs[f"op_{o}"] & ~gs["abstain"].astype(bool)]
            if len(gv) >= min_n:
                rows.append({"model": model, "split": split, "slice": "noise_op", "value": o, **_stats(gv)})
    return pd.DataFrame(rows)


def evaluate(truth, pred, model, pin_lookup=None, targets=None, slices=True):
    """Shortcut: score and summarise in one go. Returns (per-row scores, results table)."""
    s = score_rows(truth, pred, pin_lookup, targets)
    return s, summarise(truth, s, model, slices)


def scores_from_errors(truth, err_m):
    """Wrap a plain list of error distances (for the oracle row) as a scores table."""
    e = np.asarray(err_m, float)
    return pd.DataFrame({"id": truth["id"].to_numpy(), "answered": ~np.isnan(e), "err_m": e})


def leaderboard(results, cols=("n", "coverage", "acc@100", "acc@500", "acc@1000", "p50", "p90", "misroute")):
    """Overall rows only, one line per model and split."""
    r = results[results["slice"] == "overall"]
    return r.set_index(["split", "model"])[[c for c in cols if c in r]].sort_index().round(3)


def bootstrap_ci(err_m, stat=np.median, n_boot=500, seed=0):
    """95% interval for a statistic of the error distances, by resampling rows."""
    e = np.asarray(err_m, float)
    e = e[~np.isnan(e)]
    if len(e) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = [stat(e[rng.integers(0, len(e), len(e))]) for _ in range(n_boot)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return (round(float(lo), 1), round(float(hi), 1))
