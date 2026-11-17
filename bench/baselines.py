"""Local baselines: no neural network, no paid API. They set the floor every model has to beat.

city_centroid       always answers the city centre
pincode_centroid    reads a 6-digit pincode from the text and answers that pincode's centre
gazetteer_rules     looks for known place names in the text (fuzzy), answers the most specific one it finds;
                    it only sees the PUBLIC gazetteer, so hidden landmarks are invisible to it
tfidf_knn           finds the most similar training address by character n-grams, answers its location

Each returns a DataFrame with columns id, lat, lon (NaN where it declines to answer) plus `level`.
"""
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

from .context import name_key
from .geo import CITY_CENTRE, COMPASS_DEG, haversine_m, offset_point

DIR_WORDS = {"north": "north", "south": "south", "east": "east", "west": "west", "n": "north", "s": "south", "e": "east", "w": "west",
             "north-east": "north-east", "northeast": "north-east", "ne": "north-east", "north-west": "north-west", "northwest": "north-west",
             "nw": "north-west", "south-east": "south-east", "southeast": "south-east", "se": "south-east",
             "south-west": "south-west", "southwest": "south-west", "sw": "south-west"}
OFFSET_RE = re.compile(r"(\d{2,4})\s*(?:m|mtrs?|metres?|meters?)\b[^,]*?\b(north[- ]?east|north[- ]?west|south[- ]?east|south[- ]?west|north|south|east|west|ne|nw|se|sw|n|s|e|w)\b", re.I)


TEXT_PIN_RE = re.compile(r"(?<!\d)5\d{2}\s?\d{3}(?!\d)")     # a 6-digit number starting with 5, not part of a longer number
STOP_NAMES = {"hyderabad", "secunderabad", "telangana", "india", "andhra pradesh", "main road", "service road", "bus stop", "temple", "park"}


def find_pin(text):
    """Last pincode-looking number in the text (people put it at the end)."""
    m = TEXT_PIN_RE.findall(text or "")
    return m[-1].replace(" ", "") if m else None


def city_centroid(df, city="hyderabad"):
    lat, lon = CITY_CENTRE[city]
    return pd.DataFrame({"id": df["id"].to_numpy(), "lat": lat, "lon": lon, "level": "city"})


def pincode_table(targets):
    """Pincode -> centre, from buildings with a known pincode."""
    return targets.dropna(subset=["pincode"]).groupby("pincode")[["lat", "lon"]].median()


def pincode_centroid(df, pins, city="hyderabad", fallback=True):
    clat, clon = CITY_CENTRE[city]
    out = []
    for i, text in zip(df["id"], df["text"]):
        p = find_pin(text)
        if p in pins.index:
            out.append((i, pins.at[p, "lat"], pins.at[p, "lon"], "pincode"))
        elif fallback:
            out.append((i, clat, clon, "city"))
        else:
            out.append((i, np.nan, np.nan, "none"))
    return pd.DataFrame(out, columns=["id", "lat", "lon", "level"])


# ------------------------------------------------------------------ fuzzy gazetteer
def build_gazetteer(landmarks, areas=None, localities=None, suburbs=None, roads=None):
    """One table of everything the rules geocoder may look up: name, kind, point.

    landmarks  : only rows with lm_public == True are used
    areas      : complexes / colonies (centre point)        localities, suburbs: named points
    roads      : named roads (mid point of each piece)
    """
    parts = []
    lm = landmarks[landmarks["lm_public"]] if "lm_public" in landmarks else landmarks
    parts.append(pd.DataFrame({"name": lm["name"], "kind": "landmark", "lat": lm["lat"], "lon": lm["lon"]}))
    if areas is not None and len(areas):
        parts.append(pd.DataFrame({"name": areas["name"], "kind": "area", "lat": areas["lat"], "lon": areas["lon"]}))
    for g, kind in ((localities, "locality"), (suburbs, "suburb")):
        if g is not None and len(g):
            ll = g.to_crs("EPSG:4326").geometry
            parts.append(pd.DataFrame({"name": g["name"].to_numpy(), "kind": kind, "lat": ll.y.to_numpy(), "lon": ll.x.to_numpy()}))
    if roads is not None and len(roads):
        r = roads.dropna(subset=["name"])
        mid = r.geometry.interpolate(0.5, normalized=True).to_crs("EPSG:4326")
        parts.append(pd.DataFrame({"name": r["name"].to_numpy(), "kind": "street", "lat": mid.y.to_numpy(), "lon": mid.x.to_numpy()}))
    g = pd.concat(parts, ignore_index=True)
    g["key"] = g["name"].map(name_key)
    g = g[(g["key"].str.len() >= 4) & ~g["key"].isin(STOP_NAMES)].reset_index(drop=True)
    return g


class GazetteerGeocoder:
    """Fuzzy lookup against a public gazetteer.

    1. shortlist names whose character 3-grams are mostly present in the address
    2. keep those that really occur in it (fuzzy partial match >= min_score)
    3. anchor = matched locality / suburb, else the pincode centre
    4. answer = the most specific match within 4 km of the anchor (landmark or complex > street > locality > pincode), taking the copy
       nearest the anchor when a name exists in several places; apply "150 m north of" if present
    """
    RANK = {"landmark": 4, "area": 4, "street": 3, "locality": 2, "suburb": 1}

    def __init__(self, gazetteer, pins=None, city="hyderabad", min_score=88, shortlist=40, max_from_anchor_m=4000):
        self.g = gazetteer.reset_index(drop=True)
        self.keys = self.g["key"].to_numpy()
        self.count = self.g.groupby("key")["key"].transform("size").to_numpy()
        self.kind = self.g["kind"].to_numpy()
        self.ll = self.g[["lat", "lon"]].to_numpy(float)
        self.pins, self.city = pins, city
        self.min_score, self.shortlist, self.max_anchor = min_score, shortlist, max_from_anchor_m
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, dtype=np.float32)
        X = self.vec.fit_transform(self.keys)
        X.data = X.data ** 2                                    # rows are L2-normalised, so squared weights sum to 1
        self.X = X.tocsc()

    def _candidates(self, keys):
        """For each query: indices of names with most of their 3-gram weight present in the query."""
        Q = self.vec.transform(keys)
        Q.data[:] = 1.0
        S = (Q @ self.X.T).tocsr()                              # share of each name's weight found in the query
        out = []
        for i in range(S.shape[0]):
            row = S.indices[S.indptr[i]:S.indptr[i + 1]]
            val = S.data[S.indptr[i]:S.indptr[i + 1]]
            keep = val >= 0.55
            row, val = row[keep], val[keep]
            out.append(row[np.argsort(-val)[:self.shortlist]])
        return out

    def geocode_one(self, text, cand):
        q = name_key(text)
        qtok = set(q.split())
        pin = find_pin(text)
        hits = []
        for j in cand:
            k = self.keys[j]
            if len(k) > len(q) or any(tok.isdigit() and tok not in qtok for tok in k.split()):     # "Road No 1" is not "Road No 13"
                continue
            sc = fuzz.partial_ratio(k, q)
            if sc >= self.min_score:
                hits.append((j, sc, len(k)))
        anchor, level = None, "none"
        if self.pins is not None and pin in self.pins.index:
            anchor, level = (self.pins.at[pin, "lat"], self.pins.at[pin, "lon"]), "pincode"
        loc = [h for h in hits if self.kind[h[0]] in ("locality", "suburb")]
        if loc:
            best = max(loc, key=lambda h: (h[1] >= 95, h[2], h[1]))
            same = [h for h in loc if self.keys[h[0]] == self.keys[best[0]]]
            if anchor is not None and len(same) > 1:              # same locality name in two places: take the one near the pincode
                best = min(same, key=lambda h: haversine_m(anchor[0], anchor[1], *self.ll[h[0]]))
            far = anchor is not None and haversine_m(anchor[0], anchor[1], *self.ll[best[0]]) > self.max_anchor + 1000
            if not far:                                           # a locality far from the pincode is probably a namesake
                anchor, level = tuple(self.ll[best[0]]), self.kind[best[0]]
        spec = [h for h in hits if self.kind[h[0]] in ("landmark", "area", "street")]
        if anchor is not None:
            spec = [h for h in spec if haversine_m(anchor[0], anchor[1], *self.ll[h[0]]) <= self.max_anchor]
        if spec:
            # most specific kind first, then the longest well-matched name, then nearest the anchor
            def order(h):
                d = haversine_m(anchor[0], anchor[1], *self.ll[h[0]]) if anchor is not None else 0.0
                return (-self.RANK[self.kind[h[0]]], -h[2] * (h[1] / 100.0), d)
            if anchor is None:                                    # nothing to disambiguate with: only trust distinctive names
                spec = [h for h in spec if h[2] >= 8 and self.count[h[0]] == 1]
            if spec:
                best = min(spec, key=order)
                lat, lon = self.ll[best[0]]
                level = self.kind[best[0]]
                m = OFFSET_RE.search(text) if level in ("landmark", "area") else None
                if m:
                    d = DIR_WORDS.get(m.group(2).lower().replace(" ", "-"))
                    if d:
                        lat, lon = offset_point(lat, lon, COMPASS_DEG[d], float(m.group(1)))
                        level = "landmark+offset"
                return lat, lon, level
        if anchor is not None:
            return anchor[0], anchor[1], level
        return np.nan, np.nan, "none"

    def lookup(self, df, k=4, chunk=2000):
        """For each address: up to k gazetteer entries named in it, most specific and longest first.
        Returns a list (one per row) of lists of (name, kind, lat, lon). Used as retrieval context for the LLM."""
        names = self.g["name"].to_numpy()
        out = []
        texts = df["text"].fillna("").to_numpy()
        for a in range(0, len(df), chunk):
            keys = [name_key(t) for t in texts[a:a + chunk]]
            for q, cand in zip(keys, self._candidates(keys)):
                qtok, hits, seen = set(q.split()), [], set()
                for j in cand:
                    kk = self.keys[j]
                    if len(kk) > len(q) or kk in seen or any(t.isdigit() and t not in qtok for t in kk.split()):
                        continue
                    if fuzz.partial_ratio(kk, q) >= self.min_score:
                        hits.append(j); seen.add(kk)
                hits.sort(key=lambda j: (-self.RANK[self.kind[j]], -len(self.keys[j])))
                out.append([(names[j], self.kind[j], float(self.ll[j][0]), float(self.ll[j][1])) for j in hits[:k]])
        return out

    def geocode(self, df, chunk=2000):
        out = []
        texts, ids = df["text"].fillna("").to_numpy(), df["id"].to_numpy()
        for a in range(0, len(df), chunk):
            keys = [name_key(t) for t in texts[a:a + chunk]]
            for i, t, c in zip(ids[a:a + chunk], texts[a:a + chunk], self._candidates(keys)):
                out.append((i, *self.geocode_one(t, c)))
        return pd.DataFrame(out, columns=["id", "lat", "lon", "level"])


# ------------------------------------------------------------------ nearest training address
class TfidfKnn:
    """Nearest neighbour over training addresses using character n-grams. Abstains when nothing is similar."""

    def __init__(self, min_sim=0.25, max_features=400_000):
        self.min_sim = min_sim
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2, max_features=max_features, dtype=np.float32, lowercase=True)

    def fit(self, train):
        self.X = self.vec.fit_transform(train["text"].fillna("")).tocsr()
        self.ll = train[["lat", "lon"]].to_numpy(float)
        return self

    def geocode(self, df, chunk=256):
        Q = self.vec.transform(df["text"].fillna(""))
        XT = self.X.T.tocsr()
        best, sim = np.zeros(Q.shape[0], int), np.zeros(Q.shape[0], np.float32)
        for a in range(0, Q.shape[0], chunk):
            S = np.asarray((Q[a:a + chunk] @ XT).todense())
            best[a:a + chunk] = S.argmax(1)
            sim[a:a + chunk] = S.max(1)
        ok = sim >= self.min_sim
        return pd.DataFrame({"id": df["id"].to_numpy(), "lat": np.where(ok, self.ll[best, 0], np.nan),
                             "lon": np.where(ok, self.ll[best, 1], np.nan), "level": np.where(ok, "train_neighbour", "none"), "sim": sim})
