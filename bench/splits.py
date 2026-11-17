"""Train / validation / test splits, decided BEFORE any text is written.

Two kinds of held-out data:

warm  : buildings the model has never seen, in parts of the city it HAS seen during training
cold  : buildings in H3 resolution-7 cells (about 5 sq km each) that are completely absent from training

Plus one flag on landmarks:

lm_public : True for ~80% of landmarks. Only these are handed to gazetteer-style baselines.
            The hidden 20% still appear in address text, the way people cite shops that no map lists.
            A method that only looks things up cannot resolve them; a method that learns from data can.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import ring_km


def ring_label(lat, lon, city="hyderabad"):
    """Distance band from the city centre: core (<5 km), mid (5-12 km), outer (12-20 km), fringe (>20 km)."""
    km = ring_km(lat, lon, city)
    return np.select([km < 5, km < 12, km < 20], ["core", "mid", "outer"], "fringe")


def assign_splits(targets, cold_frac=0.15, holdout_frac=0.08, val_frac=0.02, seed=7, min_cell=200, city="hyderabad"):
    """Give every target building one label: train, val, holdout_warm or holdout_cold.

    Cold cells are sampled within each distance ring, so the cold set is not all city centre or all outskirts.
    Cells with fewer than min_cell target buildings are never chosen as cold (too little to test on).
    """
    rng = np.random.default_rng(seed)
    t = targets.copy()
    t["ring"] = ring_label(t["lat"].to_numpy(), t["lon"].to_numpy(), city)

    cells = t.groupby("h3_7").agg(n=("building_id", "size"), ring=("ring", lambda s: s.mode().iat[0])).reset_index()
    cold = []
    for _, g in cells[cells["n"] >= min_cell].groupby("ring"):
        k = max(1, int(round(len(g) * cold_frac)))
        cold += list(rng.choice(g["h3_7"].to_numpy(), size=min(k, len(g)), replace=False))
    is_cold = t["h3_7"].isin(cold).to_numpy()

    u = rng.random(len(t))
    t["split"] = np.where(is_cold, "holdout_cold",
                 np.where(u < holdout_frac, "holdout_warm",
                 np.where(u < holdout_frac + val_frac, "val", "train")))

    # how far is each building from the nearest TRAIN building? (lets us separate cold-interior from cold-border)
    tr = t["split"].to_numpy() == "train"
    tree = cKDTree(t.loc[tr, ["x", "y"]].to_numpy())
    d, _ = tree.query(t[["x", "y"]].to_numpy(), k=2)
    t["nn_train_m"] = np.where(tr, d[:, 1], d[:, 0]).round(1)
    return t, sorted(cold)


def flag_landmarks(lm, cold_cells, hidden_frac=0.20, seed=7, h3_res=7):
    """Mark which landmarks are public, and which sit in cold cells (those are never used in training text)."""
    import h3
    rng = np.random.default_rng(seed)
    lm = lm.copy()
    cell = [h3.latlng_to_cell(a, b, h3_res) for a, b in zip(lm["lat"], lm["lon"])]
    lm["h3_7"] = cell
    lm["in_cold_cell"] = lm["h3_7"].isin(set(cold_cells))
    lm["lm_public"] = rng.random(len(lm)) >= hidden_frac
    return lm


def split_report(t):
    r = t.groupby(["split", "ring"]).size().unstack(fill_value=0)
    r["total"] = r.sum(axis=1)
    return r
