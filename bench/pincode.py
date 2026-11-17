"""Pincode areas.

OpenStreetMap has no pincode boundaries for Hyderabad, so we get them one of two ways:

1. Official boundaries (India Post's "All India Pincode Boundary" GeoJSON, or a simplified copy),
   clipped to the city box.  -> exact polygons.
2. Fallback: points whose pincode we know (OSM address tags, Overture places) vote for the pincode
   of nearby locations (k nearest neighbours).  -> approximate, but good enough for the
   wrong-pincode slice and the misroute metric.

Either way the result is one function: lookup(lat, lon) -> pincode string (or None).
"""
import re
import numpy as np
import pandas as pd

PIN_RE = re.compile(r"^5[0-9]{5}$")          # Telangana / Andhra pincodes start with 5


def clean_pin(s):
    """Pull a 6-digit pincode out of a messy tag value; None if there isn't one."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    m = re.search(r"(?<!\d)(5\d{2})\s?(\d{3})(?!\d)", str(s))
    return m.group(1) + m.group(2) if m else None


def find_pin_column(gdf):
    """Find the column of an unknown boundary file that holds 6-digit pincodes."""
    best, best_share = None, 0.0
    for c in gdf.columns:
        if c == "geometry":
            continue
        vals = gdf[c].dropna().astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        if len(vals) == 0:
            continue
        share = vals.str.fullmatch(r"[1-9]\d{5}").mean()
        if share > best_share:
            best, best_share = c, share
    return (best, best_share) if best_share >= 0.9 else (None, best_share)


def labelled_points(gaz_dir):
    """All points with a known pincode: OSM address tags, plus Overture places if notebook 02 ran."""
    import os
    import geopandas as gpd
    rows = []
    a = gpd.read_parquet(f"{gaz_dir}/addr_tagged.parquet", columns=["addr:postcode", "geometry"])
    p = a.geometry.representative_point()
    rows.append(pd.DataFrame({"pin": a["addr:postcode"].map(clean_pin), "lat": p.y, "lon": p.x, "src": "osm"}))
    ov = f"{gaz_dir}/overture_places.parquet"
    if os.path.exists(ov):
        o = pd.read_parquet(ov, columns=["addr_postcode", "lat", "lon", "confidence"])
        o = o[o["confidence"] >= 0.6]
        rows.append(pd.DataFrame({"pin": o["addr_postcode"].map(clean_pin), "lat": o["lat"], "lon": o["lon"], "src": "overture"}))
    pts = pd.concat(rows, ignore_index=True).dropna(subset=["pin"])
    return pts[pts["pin"].str.match(PIN_RE)].reset_index(drop=True)


def _xy(lat, lon):
    """Degrees -> approximate metres on a local flat grid (fine at city scale)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return np.c_[lon * 111320.0 * np.cos(np.radians(17.4)), lat * 110540.0]


def knn_lookup(pts, k=7, max_dist_m=2500.0):
    """Build lookup(lat, lon) -> (pincode, confidence) from labelled points by distance-weighted vote."""
    from scipy.spatial import cKDTree
    tree = cKDTree(_xy(pts["lat"], pts["lon"]))
    pins = pts["pin"].to_numpy()

    def lookup(lat, lon, exclude_self=False):
        kk = k + 1 if exclude_self else k
        d, i = tree.query(_xy(lat, lon), k=kk)
        if exclude_self:
            d, i = d[:, 1:], i[:, 1:]
        out, conf = [], []
        for dr, ir in zip(d, i):
            ok = dr <= max_dist_m
            if not ok.any():
                out.append(None); conf.append(0.0); continue
            w = 1.0 / (50.0 + dr[ok])
            votes = {}
            for pin, wi in zip(pins[ir[ok]], w):
                votes[pin] = votes.get(pin, 0.0) + wi
            best = max(votes, key=votes.get)
            out.append(best); conf.append(votes[best] / w.sum())
        return np.array(out, dtype=object), np.array(conf)

    return lookup


def polygon_lookup(areas):
    """Build lookup(lat, lon) -> (pincode, confidence) from a GeoDataFrame with columns pin + geometry."""
    import shapely
    geoms = areas.geometry.values
    pins = areas["pin"].to_numpy()
    tree = shapely.STRtree(geoms)

    def lookup(lat, lon, exclude_self=False):
        pts = shapely.points(np.asarray(lon, float), np.asarray(lat, float))
        qi, gi = tree.query(pts, predicate="within")
        out = np.full(len(pts), None, dtype=object)
        out[qi] = pins[gi]                       # if polygons overlap, the last match wins
        return out, (out != None).astype(float)  # noqa: E711

    return lookup


def make_lookup(gaz_dir):
    """Best available lookup: official polygons if notebook 03 saved them, else the kNN estimate."""
    import os
    import geopandas as gpd
    poly = f"{gaz_dir}/pincode_areas.parquet"
    if os.path.exists(poly):
        return polygon_lookup(gpd.read_parquet(poly)), "polygons"
    return knn_lookup(labelled_points(gaz_dir)), "knn"


def neighbours(centroids, n=4):
    """centroids: DataFrame pin, lat, lon -> dict pin -> list of the n nearest other pincodes."""
    from scipy.spatial import cKDTree
    xy = _xy(centroids["lat"], centroids["lon"])
    _, idx = cKDTree(xy).query(xy, k=min(n + 1, len(centroids)))
    pins = centroids["pin"].to_numpy()
    return {pins[i]: [pins[j] for j in row[1:]] for i, row in enumerate(idx)}
