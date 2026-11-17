"""Address text generator: turn a building's context into the kind of address a person would type.

Every row records exactly what went into the text (components, landmark, relation, noise operations),
so results can be sliced by any of them later.

What is real and what is invented
---------------------------------
real      : building point, building name, street, landmark, complex / colony, locality, suburb, pincode
invented  : flat / floor numbers (harmless: no geocoder resolves them) and municipal-style house numbers.
            House numbers are flagged in `synthetic_tokens`, because an invented "8-2-293" can collide with a
            real house somewhere else. Rows carrying one are kept out of the commercial-API comparison.
never     : invented building, street or landmark names.
"""
import json
import re
import zlib
import numpy as np
import pandas as pd

from . import noise as nz
from .geo import COMPASS_DEG, haversine_m, offset_point

CITY = {"hyderabad": dict(names=["Hyderabad", "Hyderabad", "Hyderabad", "Hyd", "hyderabad", "HYDERABAD"],
                          state=["Telangana", "Telangana", "TS", "TG", "Telangana State"])}

PHRASES = {
    "en": {
        "NEAR": ["near {x}", "near {x}", "near {x}", "nr {x}", "near to {x}", "close to {x}", "nearby {x}", "landmark {x}", "landmark: {x}", "near by {x}"],
        "ACROSS_THE_ROAD": ["opposite {x}", "opp {x}", "opp. {x}", "opposite to {x}", "opp to {x}", "across the road from {x}", "opposite side of {x}"],
        "BESIDE": ["beside {x}", "next to {x}", "adjacent to {x}", "besides {x}", "adj to {x}", "beside to {x}", "right next to {x}"],
        "BEHIND": ["behind {x}", "behind {x}", "back side of {x}", "backside of {x}", "behind the {x}", "{x} back side"],
        "DOWN_THE_ROAD": ["{x} road", "{x} lane", "same road as {x}", "same lane as {x}", "down the road from {x}", "on {x} road", "{x} line"],
        "AROUND_THE_CORNER": ["lane beside {x}", "lane adjacent to {x}", "lane next to {x}", "around the corner from {x}", "{x} side lane", "turning at {x}", "take turn at {x}"],
        "WITHIN": ["inside {x}", "in {x}", "{x}", "{x}", "{x} campus", "within {x}"],
    },
    "hi": {
        "NEAR": ["{x} ke paas", "{x} ke pass", "{x} k paas", "{x} ke nazdeek"],
        "ACROSS_THE_ROAD": ["{x} ke saamne", "{x} ke samne", "{x} k samne"],
        "BESIDE": ["{x} ke bagal mein", "{x} ke bagal me", "{x} ke baju mein"],
        "BEHIND": ["{x} ke peeche", "{x} ke piche", "{x} k peeche"],
        "DOWN_THE_ROAD": ["{x} wali road pe", "{x} wali road par", "{x} wale raste pe"],
        "AROUND_THE_CORNER": ["{x} wali gali", "{x} ke bagal wali gali", "{x} ke baju wali gali mein"],
        "WITHIN": ["{x} ke andar", "{x} ke ander", "{x} mein"],
    },
    "te": {
        "NEAR": ["{x} daggara", "{x} daggaralo", "{x} dagara"],
        "ACROSS_THE_ROAD": ["{x} eduruga", "{x} ki eduruga", "{x} edurugaa"],
        "BESIDE": ["{x} pakkana", "{x} pakkane", "{x} pakkana undi"],
        "BEHIND": ["{x} venuka", "{x} venakala", "{x} venaka"],
        "DOWN_THE_ROAD": ["{x} road lo", "{x} line lo", "{x} dari lo"],
        "AROUND_THE_CORNER": ["{x} pakka sandhu lo", "{x} galli lo", "{x} pakka galli lo"],
        "WITHIN": ["{x} lopala", "{x} lo", "{x} lopali"],
    },
}
OFFSET_PHRASES = ["{d} m {dir} of {x}", "{d}m {dir} of {x}", "{d} mtrs {dir} of {x}", "about {d} metres {dir} of {x}",
                  "{d} meters {dir} from {x}", "{d} m towards {dir} from {x}", "approx {d}m {dir} side of {x}"]
DIR_SHORT = {"north": "N", "north-east": "NE", "east": "E", "south-east": "SE", "south": "S", "south-west": "SW", "west": "W", "north-west": "NW"}

STYLE_MIX = {"formal": 0.34, "landmark": 0.36, "offset": 0.10, "minimal": 0.10, "complex": 0.10}      # complex only if available
NOISE_MIX = {"clean": 0.35, "light": 0.40, "heavy": 0.25}
REL_WEIGHT = {"NEAR": 1.0, "ACROSS_THE_ROAD": 6.0, "BESIDE": 6.0, "BEHIND": 6.0, "DOWN_THE_ROAD": 3.0, "AROUND_THE_CORNER": 3.0}
SUBSTANTIVE = ["typo", "abbrev", "merge", "junk", "shuffle", "omit", "lang", "wrong_pin_far", "wrong_pin_adjacent", "instructions", "name_variant"]
OP_WEIGHT = {"typo": 3, "abbrev": 3, "merge": 1.5, "junk": 1, "shuffle": 1.5, "omit": 2, "lang": 1.5, "wrong_pin_far": 0.5,
             "wrong_pin_adjacent": 0.7, "instructions": 1, "name_variant": 1.5}


def _stable(s, mod):
    return zlib.crc32(str(s).encode()) % mod


def house_sequence(t):
    """A running door number for every building: buildings are numbered in order along their road inside each
    H3 resolution-8 cell, odd numbers on one side and even on the other. Real door numbers behave like this
    (neighbours have neighbouring numbers), which is what lets a model learn anything from them."""
    order = t.sort_values(["h3_8", "r_road", "r_s"])
    rank = order.groupby("h3_8").cumcount().to_numpy()
    seq = pd.Series(2 * rank + np.where(order["r_side"].to_numpy() > 0, 1, 2), index=order.index)
    return seq.reindex(t.index).to_numpy()


def house_number(h3_8, building_id, rng, seq=None):
    """Invented municipal-style door number, e.g. "H.No 8-2-293/A". The ward-block prefix is shared by every
    building in one H3 resolution-8 cell (about 0.7 sq km); the last number runs along the road."""
    ward, block = 1 + _stable(h3_8, 16), 1 + _stable("b" + str(h3_8), 9)
    n = int(seq) if seq is not None else 1 + _stable(building_id, 990)
    tail = ["", "", "", "/A", "/B", f"/{1 + _stable('t' + str(building_id), 40)}", f"/{1 + _stable('t' + str(building_id), 9)}/A"][rng.integers(7)]
    core = f"{ward}-{block}-{n}{tail}"
    form = ["H.No {c}", "H.No. {c}", "H No {c}", "{c}", "{c}", "House No {c}", "D.No {c}", "Hno {c}", "H.No: {c}", "Plot No {n}", "Plot {n}", "#{c}"][rng.integers(12)]
    return form.format(c=core, n=n)


def unit_number(rng):
    floor, flat = int(rng.integers(0, 12)), int(rng.integers(1, 9))
    n = floor * 100 + flat
    ordinal = ["Ground", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th"][floor]
    return [f"Flat {n}", f"Flat No {n}", f"Flat No. {n}", f"F-{n}", f"#{n}", f"{n}", f"{ordinal} Floor", f"Flat {n}, {ordinal} Floor",
            f"Block {'ABCD'[rng.integers(4)]}, Flat {n}", f"{'ABCDEFG'[rng.integers(7)]}-{n}"][rng.integers(10)]


def name_variant(name, rng):
    """How a person shortens or mangles a place name (beyond spelling slips)."""
    toks = name.split()
    opts = []
    if len(toks) >= 2 and toks[0].lower().strip(".") in {"sri", "shri", "sree", "shree", "the", "new", "sri."}:
        opts.append(" ".join(toks[1:]))
    if len(toks) >= 3:
        opts.append(" ".join(toks[:-1]))
        opts.append(" ".join(toks[:2]))
    if len(toks) >= 2:
        opts.append(toks[0] + toks[1] + (" " + " ".join(toks[2:]) if len(toks) > 2 else ""))     # "SBI Bank" -> "SBIBank"
    opts.append(re.sub(r"[^\w\s]", "", name))
    out = opts[rng.integers(len(opts))]
    return out if len(out) >= 3 else name


class Generator:
    """Holds the lookup tables once, then writes rows for any set of target buildings."""

    def __init__(self, targets, relations, landmarks, city="hyderabad", seed=7):
        self.city = city
        self.t = targets.reset_index(drop=True)
        self.house_seq = dict(zip(self.t["building_id"], house_sequence(self.t)))
        lm = landmarks.set_index("landmark_id")
        self.lm = lm
        rel = relations[relations["landmark_id"].isin(lm.index)].sort_values("building_id").reset_index(drop=True)
        self.rel = rel
        self.rel_lm = rel["landmark_id"].to_numpy()
        self.rel_type = rel["relation"].to_numpy()
        self.rel_d = rel["d_m"].to_numpy()
        self.rel_dir = rel["direction"].to_numpy()
        lmr = lm.reindex(rel["landmark_id"])
        self.rel_cold = lmr["in_cold_cell"].to_numpy() if "in_cold_cell" in lm else np.zeros(len(rel), bool)
        w = lmr["salience"].to_numpy() / np.sqrt(lmr["name_count"].to_numpy()) * np.exp(-self.rel_d / 150.0)
        self.rel_w = w * np.array([REL_WEIGHT.get(r, 1.0) for r in self.rel_type])
        bid = rel["building_id"].to_numpy()
        starts = np.flatnonzero(np.r_[True, bid[1:] != bid[:-1]])
        ends = np.r_[starts[1:], len(bid)]
        self.span = {b: (s, e) for b, s, e in zip(bid[starts], starts, ends)}
        # pincode tables for the wrong-pincode noise
        pc = self.t.dropna(subset=["pincode"]).groupby("pincode")[["lat", "lon"]].mean()
        self.pins = pc.index.to_numpy()
        self.pin_ll = pc.to_numpy()
        self.seed = seed

    # ------------------------------------------------------------ pieces
    def _pick_landmark(self, building_id, rng, allow_cold, want=None, exclude=None):
        se = self.span.get(building_id)
        if se is None:
            return None
        idx = np.arange(se[0], se[1])
        ok = np.ones(len(idx), bool)
        if not allow_cold:
            ok &= ~self.rel_cold[idx]
        if want is not None:
            ok &= self.rel_type[idx] == want
        if exclude is not None:
            ok &= self.rel_lm[idx] != exclude
        idx = idx[ok]
        if len(idx) == 0:
            return None
        w = self.rel_w[idx]
        return int(idx[rng.choice(len(idx), p=w / w.sum())])

    def _wrong_pin(self, pin, lat, lon, rng, adjacent):
        if pin is None or len(self.pins) < 6:
            return None
        d = haversine_m(lat, lon, self.pin_ll[:, 0], self.pin_ll[:, 1])
        order = np.argsort(d)
        order = order[self.pins[order] != pin]
        pick = order[rng.integers(0, 4)] if adjacent else order[rng.integers(min(10, len(order) - 1), len(order))]
        return str(self.pins[pick])

    # ------------------------------------------------------------ one row
    def row(self, r, rng, allow_cold=True, style=None, level=None, want_relation=None):
        city = CITY[self.city]
        has_area = isinstance(r.area_name, str)
        if style is None:
            mix = dict(STYLE_MIX)
            if not has_area:
                mix.pop("complex")
            if want_relation == "WITHIN":
                style = "complex"
            elif want_relation is not None:
                style = "landmark"
            else:
                k = list(mix)
                p = np.array([mix[s] for s in k])
                style = k[rng.choice(len(k), p=p / p.sum())]
        level = level or list(NOISE_MIX)[rng.choice(3, p=list(NOISE_MIX.values()))]

        # which noise operations this row gets
        n_ops = {"clean": 0, "light": int(rng.integers(1, 3)), "heavy": int(rng.integers(3, 6))}[level]
        ops = []
        if n_ops:
            p = np.array([OP_WEIGHT[o] for o in SUBSTANTIVE])
            ops = list(rng.choice(SUBSTANTIVE, size=n_ops, replace=False, p=p / p.sum()))
            if "wrong_pin_far" in ops and "wrong_pin_adjacent" in ops:
                ops.remove("wrong_pin_far")

        # landmark reference
        ref = ref2 = None
        need_lm = style in ("landmark", "offset") or (style == "formal" and rng.random() < 0.35) or \
                  (style == "complex" and rng.random() < 0.2) or (style == "minimal" and rng.random() < 0.4)
        if need_lm and want_relation != "WITHIN":
            ref = self._pick_landmark(r.building_id, rng, allow_cold, want=want_relation)
            if ref is not None and style == "landmark" and rng.random() < 0.15:
                ref2 = self._pick_landmark(r.building_id, rng, allow_cold, exclude=self.rel_lm[ref])
        if ref is None and style in ("landmark", "offset"):
            style = "complex" if has_area else "formal"

        comp, synth, oracle = {}, [], []
        lang = "en"
        relation = None

        def lm_phrase(i, offset=False):
            nonlocal lang, relation
            row = self.lm.loc[self.rel_lm[i]]
            nm = row["name"]
            if "name_variant" in ops:
                nm = name_variant(nm, rng)
            rel_t = self.rel_type[i]
            if offset:
                d = self.rel_d[i]
                dr = int(round(d / 10.0) * 10) if d < 100 else int(round(d / 50.0) * 50)
                dr = max(dr, 10)
                direction = self.rel_dir[i]
                dtxt = direction if rng.random() < 0.7 else DIR_SHORT[direction]
                txt = OFFSET_PHRASES[rng.integers(len(OFFSET_PHRASES))].format(d=dr, dir=dtxt, x=nm)
                olat, olon = offset_point(row["lat"], row["lon"], COMPASS_DEG[direction], dr)
                oracle.append(float(haversine_m(r.lat, r.lon, olat, olon)))
                return txt, "OFFSET"
            lg = "en"
            if "lang" in ops:
                lg = "hi" if rng.random() < 0.5 else "te"
                lang = lg
            ph = PHRASES[lg][rel_t]
            oracle.append(float(self.rel_d[i]))
            return ph[rng.integers(len(ph))].format(x=nm), rel_t

        # ---- components by style
        big = r.area_m2 >= 150
        if style == "formal":
            if rng.random() < (0.35 if big else 0.08):
                comp["unit"] = unit_number(rng)
            if rng.random() < 0.75:
                comp["house_no"] = house_number(r.h3_8, r.building_id, rng, self.house_seq.get(r.building_id)); synth.append("house_no")
            if isinstance(r.bname, str) and rng.random() < 0.9:
                comp["building"] = r.bname; oracle.append(0.0)
            if isinstance(r.street, str) and rng.random() < 0.9:
                comp["street"] = r.street
            if ref is not None:
                comp["landmark"], relation = lm_phrase(ref)
            if has_area and rng.random() < 0.95:
                comp["area"] = r.area_name
            if isinstance(r.locality, str):
                comp["locality"] = r.locality
            if isinstance(r.suburb, str) and rng.random() < 0.5:
                comp["suburb"] = r.suburb
            p_city, p_state, p_pin = 0.9, 0.25, 0.75
        elif style == "landmark":
            if rng.random() < 0.25:
                comp["unit"] = unit_number(rng)
            if rng.random() < 0.35:
                comp["house_no"] = house_number(r.h3_8, r.building_id, rng, self.house_seq.get(r.building_id)); synth.append("house_no")
            if isinstance(r.bname, str) and rng.random() < 0.5:
                comp["building"] = r.bname; oracle.append(0.0)
            comp["landmark"], relation = lm_phrase(ref)
            if ref2 is not None:
                comp["landmark2"], _ = lm_phrase(ref2)
            if isinstance(r.street, str) and rng.random() < 0.4:
                comp["street"] = r.street
            if isinstance(r.locality, str) and rng.random() < 0.85:
                comp["locality"] = r.locality
            if isinstance(r.suburb, str) and rng.random() < 0.3:
                comp["suburb"] = r.suburb
            p_city, p_state, p_pin = 0.7, 0.1, 0.5
        elif style == "complex":
            if rng.random() < 0.9:
                comp["unit"] = unit_number(rng)
            ph = PHRASES["en"]["WITHIN"] if r.area_kind == "campus" else ["{x}"]
            comp["area"] = ph[rng.integers(len(ph))].format(x=r.area_name)
            relation = "WITHIN"
            if isinstance(r.street, str) and rng.random() < 0.5:
                comp["street"] = r.street
            if ref is not None:
                comp["landmark"], _ = lm_phrase(ref)
            if isinstance(r.locality, str) and rng.random() < 0.9:
                comp["locality"] = r.locality
            if isinstance(r.suburb, str) and rng.random() < 0.3:
                comp["suburb"] = r.suburb
            p_city, p_state, p_pin = 0.9, 0.2, 0.7
        elif style == "offset":
            comp["landmark"], relation = lm_phrase(ref, offset=True)
            if isinstance(r.locality, str) and rng.random() < 0.8:
                comp["locality"] = r.locality
            p_city, p_state, p_pin = 0.7, 0.05, 0.3
        else:                                           # minimal: what a hurried person types
            kind = rng.integers(0, 4)
            if ref is not None:
                comp["landmark"], relation = lm_phrase(ref)
            elif kind == 0 and isinstance(r.street, str):
                comp["street"] = r.street
            elif kind == 1 and has_area:
                comp["area"] = r.area_name
            if isinstance(r.locality, str):
                comp["locality"] = r.locality
            elif isinstance(r.suburb, str):
                comp["suburb"] = r.suburb
            p_city, p_state, p_pin = 0.5, 0.0, 0.25

        # every address must say SOMETHING about where it is
        if not any(k in comp for k in ("building", "street", "landmark", "area", "locality", "suburb")):
            if isinstance(r.locality, str):
                comp["locality"] = r.locality
            elif isinstance(r.suburb, str):
                comp["suburb"] = r.suburb
            elif isinstance(r.pincode, str):
                p_pin = 1.0
            else:
                return None                              # nothing truthful to write for this building

        if "area" in comp:
            oracle.append(float(haversine_m(r.lat, r.lon, r.area_lat, r.area_lon)))
        if "locality" in comp and not np.isnan(r.locality_lat):
            oracle.append(float(haversine_m(r.lat, r.lon, r.locality_lat, r.locality_lon)))

        if rng.random() < p_city:
            comp["city"] = city["names"][rng.integers(len(city["names"]))]
        if rng.random() < p_state:
            comp["state"] = city["state"][rng.integers(len(city["state"]))]
        pin_true = r.pincode if isinstance(r.pincode, str) else None
        pin_text = None
        if pin_true and (rng.random() < p_pin or "wrong_pin_far" in ops or "wrong_pin_adjacent" in ops):
            pin_text = pin_true
            for op, adj in (("wrong_pin_far", False), ("wrong_pin_adjacent", True)):
                if op in ops:
                    pin_text = self._wrong_pin(pin_true, r.lat, r.lon, rng, adj) or pin_true
            comp["pincode"] = pin_text

        # ---- component-level noise
        applied = []
        if "omit" in ops:
            drop = [k for k in ("suburb", "city", "state", "pincode", "street", "unit") if k in comp]
            keep_any = [k for k in comp if k not in ("city", "state", "unit")]
            if drop and len(keep_any) > 1:
                k = drop[rng.integers(len(drop))]
                comp.pop(k)
                if k == "pincode":
                    pin_text = None
                applied.append("omit")
        order = [k for k in ("unit", "house_no", "building", "street", "landmark", "landmark2", "area", "locality", "suburb", "city", "state", "pincode") if k in comp]
        if style in ("landmark", "minimal") and "landmark" in order and rng.random() < 0.35:        # landmark first is common
            order.remove("landmark"); order.insert(0, "landmark")
        if "shuffle" in ops and len(order) >= 3:
            i = int(rng.integers(0, len(order) - 1))
            j = int(rng.integers(0, len(order)))
            order.insert(j, order.pop(i))
            applied.append("shuffle")
        parts = [str(comp[k]) for k in order]
        if "city" in comp and "pincode" in comp and order.index("pincode") == order.index("city") + 1 and rng.random() < 0.5:
            i = order.index("city")
            parts[i:i + 2] = [f"{parts[i]}{[' - ', '-', ' ', ' – '][rng.integers(4)]}{parts[i + 1]}"]
        text_clean = ", ".join(parts)

        # ---- string-level noise
        text = text_clean
        for op in ("abbrev", "typo", "merge", "junk"):
            if op in ops:
                new = nz.STRING_OPS[op](text, rng)
                if new != text:
                    applied.append(op)
                text = new
        for op in ("lang", "name_variant", "wrong_pin_far", "wrong_pin_adjacent"):
            if op in ops:
                if (op == "lang" and lang != "en") or (op == "name_variant" and ("landmark" in comp)) or \
                   (op.startswith("wrong_pin") and pin_text is not None and pin_text != pin_true):
                    applied.append(op)
        fmt = []
        u = rng.random()
        if u < 0.30:
            text = nz.recase(text, rng); fmt.append("case")
        if rng.random() < 0.12:
            text = nz.strip_commas(text, rng); fmt.append("nocomma")
        if "instructions" in ops:
            text = nz.instructions(text, rng); applied.append("instructions")
        n_sub = len(applied)
        level_real = "clean" if n_sub == 0 else "light" if n_sub <= 2 else "heavy"

        out = dict(
            text=text, text_clean=text_clean, lat=r.lat, lon=r.lon, building_id=r.building_id,
            h3_7=r.h3_7, h3_8=r.h3_8, h3_9=r.h3_9, h3_10=r.h3_10, digipin=r.digipin,
            access_lat=r.access_lat, access_lon=r.access_lon,
            pincode_true=pin_true, pincode_in_text=pin_text, style=style, relation=relation, lang=lang,
            components=json.dumps(comp, ensure_ascii=False), noise_ops=",".join(applied), format_ops=",".join(fmt),
            noise_level=level_real, synthetic_tokens=",".join(synth),
            oracle_err_m=round(min(oracle), 1) if oracle else np.nan, abstain=False, source="synthetic",
            ring=r.ring, nn_train_m=r.nn_train_m, locality=r.locality if isinstance(r.locality, str) else None,
            area_seen=getattr(r, "split", "train") != "holdout_cold",       # False = the building is in a cold cell
        )
        for tag, i in (("ref", ref), ("ref2", ref2)):
            if i is not None and ("landmark" if tag == "ref" else "landmark2") in comp:
                lmrow = self.lm.loc[self.rel_lm[i]]
                out.update({f"{tag}_landmark_id": self.rel_lm[i], f"{tag}_name": lmrow["name"], f"{tag}_lat": lmrow["lat"],
                            f"{tag}_lon": lmrow["lon"], f"{tag}_dist_m": float(self.rel_d[i]),
                            f"{tag}_in_gazetteer": bool(lmrow.get("lm_public", True))})
        if relation == "WITHIN" and "ref_name" not in out:
            out.update(ref_landmark_id=r.area_id, ref_name=r.area_name, ref_lat=r.area_lat, ref_lon=r.area_lon,
                       ref_dist_m=float(haversine_m(r.lat, r.lon, r.area_lat, r.area_lon)), ref_in_gazetteer=True)
        return out

    # ------------------------------------------------------------ many rows
    def generate(self, targets, split, n_variants=1, allow_cold=True, seed=0, abstain_frac=0.0, want_relation=None):
        """Write n_variants addresses for each building in `targets`. Deterministic for a given seed."""
        rng = np.random.default_rng([self.seed, seed])
        rows = []
        for r in targets.itertuples(index=False):
            for v in range(n_variants):
                row = self.row(r, rng, allow_cold=allow_cold, want_relation=want_relation)
                if row is None:
                    continue
                if abstain_frac and rng.random() < abstain_frac:
                    row.update(text=nz.gibberish(rng), text_clean="", style="gibberish", relation=None, abstain=True,
                               components="{}", noise_ops="gibberish", noise_level="heavy", oracle_err_m=np.nan,
                               pincode_in_text=None, synthetic_tokens="")
                    for k in [k for k in row if k.startswith("ref")]:
                        row.pop(k)
                row["variant"] = v
                rows.append(row)
        df = pd.DataFrame(rows)
        df.insert(0, "split", split)
        df.insert(0, "id", [f"{split}_{i:06d}" for i in range(len(df))])
        return df


def relation_test_set(gen, holdout, relations, per_relation=250, seed=11):
    """A test set with a fixed quota per relation, so the rare ones (ACROSS, BESIDE, BEHIND) have enough rows."""
    rng = np.random.default_rng(seed)
    hold = holdout.set_index("building_id", drop=False)
    rel = relations[relations["building_id"].isin(hold.index)]
    used, out = set(), []
    order = ["ACROSS_THE_ROAD", "BESIDE", "BEHIND", "DOWN_THE_ROAD", "AROUND_THE_CORNER", "NEAR"]
    for k, name in enumerate(order):
        ids = rel.loc[rel["relation"] == name, "building_id"].unique()
        ids = np.array([i for i in ids if i not in used])
        pick = rng.choice(ids, size=min(per_relation, len(ids)), replace=False) if len(ids) else []
        used.update(pick)
        if len(pick):
            out.append(gen.generate(hold.loc[pick], "tmp", seed=100 + k, want_relation=name))
    within = hold[hold["area_name"].notna() & ~hold["building_id"].isin(used)]
    if len(within):
        pick = within.sample(min(per_relation, len(within)), random_state=seed)
        out.append(gen.generate(pick, "tmp", seed=199, want_relation="WITHIN"))
    df = pd.concat(out, ignore_index=True)
    df["split"] = "test_rel"
    df["id"] = [f"test_rel_{i:06d}" for i in range(len(df))]
    return df


def mark_api_subset(df, per_split=None, seed=5):
    """Choose a stratified subset for the paid geocoders: no invented house numbers, no abstain rows,
    spread evenly over styles. Keeps the whole comparison inside the free tiers."""
    per_split = per_split or {"test_warm": 900, "test_cold": 900, "test_rel": 450, "test_real": 300}
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["api_subset"] = False
    ok = ~df["abstain"] & ~df["synthetic_tokens"].fillna("").str.contains("house_no")
    for sp, n in per_split.items():
        pool = df[ok & (df["split"] == sp)]
        if pool.empty:
            continue
        groups = pool.groupby(pool["relation"].fillna("none") if sp == "test_rel" else pool["style"])
        quota = int(np.ceil(n / groups.ngroups))
        pick = []
        for _, g in groups:
            pick += list(rng.choice(g.index.to_numpy(), size=min(quota, len(g)), replace=False))
        pick = pick[:n]
        if len(pick) < n:                                # small groups ran out: top up at random
            rest = pool.index.difference(pick).to_numpy()
            pick += list(rng.choice(rest, size=min(n - len(pick), len(rest)), replace=False))
        df.loc[pick, "api_subset"] = True
    return df


# ---------------------------------------------------------------- real public text
def real_osm_rows(gaz_dir, cold_cells=(), city="hyderabad"):
    """Addresses that mappers typed into OpenStreetMap address tags: real text, real building point."""
    import pyarrow.parquet as pq
    cols = [c for c in pq.read_schema(f"{gaz_dir}/buildings.parquet").names if c.startswith("addr:")] + \
           ["osm_id", "name", "lat", "lon", "h3_7", "h3_8", "h3_9", "h3_10", "digipin", "access_lat", "access_lon"]
    b = pd.read_parquet(f"{gaz_dir}/buildings.parquet", columns=cols).drop_duplicates("osm_id")
    def s(v):
        return v.strip() if isinstance(v, str) and v.strip() else None
    rows = []
    for r in b.to_dict("records"):
        hn, st, full = s(r.get("addr:housenumber")), s(r.get("addr:street")), s(r.get("addr:full"))
        sub, pin = s(r.get("addr:suburb")) or s(r.get("addr:place")), s(r.get("addr:postcode"))
        if full and len(full) >= 15:
            text = full
        elif hn and (st or sub):
            parts = [hn, s(r.get("addr:housename")), st, sub, s(r.get("addr:city")) or "Hyderabad", pin]
            text = ", ".join(dict.fromkeys(p for p in parts if p))
        else:
            continue
        if len(text) < 12 or not re.search(r"\d", text):
            continue
        pin6 = re.sub(r"\D", "", pin or "")
        rows.append(dict(text=text, text_clean=text, lat=r["lat"], lon=r["lon"], building_id=r["osm_id"], h3_7=r["h3_7"], h3_8=r["h3_8"],
                         h3_9=r["h3_9"], h3_10=r["h3_10"], digipin=r["digipin"], access_lat=r["access_lat"], access_lon=r["access_lon"],
                         pincode_true=None, pincode_in_text=pin6 if len(pin6) == 6 else None, style="real", relation=None, lang="en",
                         components="{}", noise_ops="", format_ops="", noise_level="real", synthetic_tokens="", oracle_err_m=np.nan,
                         abstain=False, source="osm_addr", in_cold_cell=r["h3_7"] in set(cold_cells)))
    df = pd.DataFrame(rows).drop_duplicates("text")
    return df


def real_overture_rows(gaz_dir, cold_cells=()):
    """Business addresses from Overture Places that notebook 02 verified against OSM. The business NAME is left
    out of the text on purpose: with the name in, the task becomes place search, not address geocoding."""
    import os
    import h3
    p = f"{gaz_dir}/real_addresses_overture.parquet"
    if not os.path.exists(p):
        return pd.DataFrame()
    o = pd.read_parquet(p, columns=["overture_id", "addr_text", "addr_postcode", "lat", "lon", "h3_7"])
    from .geo import digipin_encode
    pin = o["addr_postcode"].astype("string").str.replace(r"\D", "", regex=True)
    df = pd.DataFrame(dict(
        text=o["addr_text"], text_clean=o["addr_text"], lat=o["lat"], lon=o["lon"], building_id="ov_" + o["overture_id"].astype(str),
        h3_7=o["h3_7"], h3_8=[h3.latlng_to_cell(a, b, 8) for a, b in zip(o["lat"], o["lon"])],
        h3_9=[h3.latlng_to_cell(a, b, 9) for a, b in zip(o["lat"], o["lon"])],
        h3_10=[h3.latlng_to_cell(a, b, 10) for a, b in zip(o["lat"], o["lon"])],
        digipin=[digipin_encode(a, b) for a, b in zip(o["lat"], o["lon"])], access_lat=np.nan, access_lon=np.nan,
        pincode_true=None, pincode_in_text=pin.where(pin.str.len() == 6), style="real", relation=None, lang="en", components="{}",
        noise_ops="", format_ops="", noise_level="real", synthetic_tokens="", oracle_err_m=np.nan, abstain=False, source="overture_addr"))
    df["in_cold_cell"] = df["h3_7"].isin(set(cold_cells))
    return df.drop_duplicates("text")


def real_test_set(gaz_dir, cold_cells=(), max_rows=3000, seed=3):
    parts = [d for d in (real_osm_rows(gaz_dir, cold_cells), real_overture_rows(gaz_dir, cold_cells)) if len(d)]
    df = pd.concat(parts, ignore_index=True)
    if len(df) > max_rows:
        df = df.groupby("source", group_keys=False).apply(lambda g: g.sample(min(len(g), max_rows // df["source"].nunique()), random_state=seed))
    df = df.reset_index(drop=True)
    df["area_seen"] = ~df["in_cold_cell"].astype(bool)
    df.insert(0, "split", "test_real")
    df.insert(0, "id", [f"test_real_{i:06d}" for i in range(len(df))])
    return df
