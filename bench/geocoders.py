"""Clients for outside geocoders, with a cache so an interrupted run can resume without paying twice.

Open services  : nominatim (OpenStreetMap), photon (OpenStreetMap)       -> coordinates may be kept
Paid services  : google, ola, mappls                                     -> see the rules below

Rules for paid services (their terms of service, not optional):
  * API keys come from Colab / Kaggle secrets or environment variables, never from a file in the repo.
  * Their coordinates are scored straight away; only the scores (distance in metres, yes/no flags) are saved
    to the results folder. Raw answers sit in a private cache only so a crashed run can resume, and
    purge_cache() deletes anything older than 30 days.
  * Their output is never used to train, tune or validate a model. Comparison only.
  * Never plot their points on a non-Google / non-vendor map. The explorer shows the error distance only.

Response formats were written from each vendor's public documentation and have NOT been run against live
keys here. parse_* functions are separated out so a format surprise is a one-line fix; run check_provider()
on three addresses before spending quota.
"""
import hashlib
import json
import os
import time
import uuid

import numpy as np
import pandas as pd
import requests

BBOX = {"hyderabad": (78.25, 17.25, 78.70, 17.60)}          # west, south, east, north
PAID = {"google", "ola", "mappls"}
UA = "navigate-benchmark/0.1 (research; https://github.com/SiddharthaRaja/Navigate)"


# ------------------------------------------------------------------ parsers (pure functions, unit-tested with sample payloads)
def parse_nominatim(js):
    if isinstance(js, list) and js:
        r = js[0]
        return float(r["lat"]), float(r["lon"]), str(r.get("addresstype") or r.get("type") or "")
    return None


def parse_photon(js):
    f = (js or {}).get("features") or []
    if f:
        lon, lat = f[0]["geometry"]["coordinates"][:2]
        p = f[0].get("properties", {})
        return float(lat), float(lon), str(p.get("type") or p.get("osm_value") or "")
    return None


def parse_google(js):
    if (js or {}).get("status") == "OK" and js.get("results"):
        r = js["results"][0]
        loc = r["geometry"]["location"]
        level = r["geometry"].get("location_type", "") + ("|partial" if r.get("partial_match") else "")
        return float(loc["lat"]), float(loc["lng"]), level
    return None


def parse_ola(js):
    res = (js or {}).get("geocodingResults") or (js or {}).get("results") or []
    if res:
        r = res[0]
        loc = (r.get("geometry") or {}).get("location") or {}
        if "lat" in loc:
            return float(loc["lat"]), float(loc.get("lng", loc.get("lon"))), str((r.get("geometry") or {}).get("location_type", ""))
    return None


def parse_mappls(js):
    r = (js or {}).get("copResults") or (js or {}).get("results") or js
    if isinstance(r, list):
        r = r[0] if r else None
    if isinstance(r, dict) and r.get("latitude") not in (None, "", 0) and r.get("longitude") not in (None, "", 0):
        return float(r["latitude"]), float(r["longitude"]), str(r.get("geocodeLevel", ""))
    return None


# ------------------------------------------------------------------ one request per provider
def _get(url, params=None, headers=None, timeout=20):
    r = requests.get(url, params=params, headers={"User-Agent": UA, **(headers or {})}, timeout=timeout)
    if r.status_code == 429:
        raise RuntimeError("rate limited (HTTP 429)")
    if r.status_code in (401, 403):
        raise PermissionError(f"HTTP {r.status_code}: key missing, wrong, or this API is not enabled for it")
    r.raise_for_status()
    return r.json()


def call_nominatim(text, key=None, city="hyderabad", base="https://nominatim.openstreetmap.org"):
    w, s, e, n = BBOX[city]
    return _get(f"{base}/search", {"q": text, "format": "jsonv2", "limit": 1, "countrycodes": "in", "viewbox": f"{w},{n},{e},{s}", "bounded": 0})


def call_photon(text, key=None, city="hyderabad", base="https://photon.komoot.io"):
    w, s, e, n = BBOX[city]
    return _get(f"{base}/api/", {"q": text, "limit": 1, "lat": (s + n) / 2, "lon": (w + e) / 2, "bbox": f"{w - 0.3},{s - 0.3},{e + 0.3},{n + 0.3}"})


def call_google(text, key, city="hyderabad"):
    w, s, e, n = BBOX[city]
    return _get("https://maps.googleapis.com/maps/api/geocode/json",
                {"address": text, "key": key, "region": "in", "components": "country:IN", "bounds": f"{s},{w}|{n},{e}"})


def call_ola(text, key, city="hyderabad"):
    return _get("https://api.olamaps.io/places/v1/geocode", {"address": text, "api_key": key}, headers={"X-Request-Id": str(uuid.uuid4())})


def call_mappls(text, key, city="hyderabad"):
    return _get("https://search.mappls.com/search/address/geocode", {"address": text, "access_token": key, "itemCount": 1})


PROVIDERS = {
    "nominatim": dict(call=call_nominatim, parse=parse_nominatim, secret=None, rps=1.0),
    "photon": dict(call=call_photon, parse=parse_photon, secret=None, rps=1.0),
    "google": dict(call=call_google, parse=parse_google, secret="GOOGLE_MAPS_KEY", rps=10.0),
    "ola": dict(call=call_ola, parse=parse_ola, secret="OLA_MAPS_KEY", rps=4.0),
    "mappls": dict(call=call_mappls, parse=parse_mappls, secret="MAPPLS_KEY", rps=4.0),
}


# ------------------------------------------------------------------ cache
def _h(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def load_cache(path):
    c = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    c[r["h"]] = r
                except Exception:
                    pass
    return c


def purge_cache(path, max_age_days=30):
    """Delete cached answers older than max_age_days. Returns how many were removed."""
    c = load_cache(path)
    now = time.time()
    keep = [r for r in c.values() if now - r.get("ts", 0) <= max_age_days * 86400]
    if os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")
    return len(c) - len(keep)


def check_provider(name, key=None, samples=("Charminar, Hyderabad", "Road No 12, Banjara Hills, Hyderabad 500034", "near Kukatpally bus stop, KPHB")):
    """Send three easy addresses and show the raw and parsed answers. Do this before a full run."""
    p = PROVIDERS[name]
    for t in samples:
        js = p["call"](t, key)
        print(t, "->", p["parse"](js), "| raw:", json.dumps(js)[:300])
        time.sleep(1.1)


def run_provider(name, df, cache_path, key=None, city="hyderabad", limit=None, max_errors=20, progress=200):
    """Geocode df['text'] with one provider. Returns DataFrame id, lat, lon, level, latency_ms.

    Safe to re-run: answers already in the cache file are not requested again.
    """
    p = PROVIDERS[name]
    if p["secret"] and not key:
        raise ValueError(f"{name} needs an API key (secret {p['secret']})")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = load_cache(cache_path)
    rows = df if limit is None else df.head(limit)
    gap, errors, new = 1.0 / p["rps"], 0, 0
    with open(cache_path, "a", encoding="utf-8") as f:
        for i, text in enumerate(rows["text"].fillna("")):
            h = _h(text)
            if h in cache or not text.strip():
                continue
            t0 = time.time()
            try:
                js = p["call"](text, key, city)
                got = p["parse"](js)
            except PermissionError:
                raise
            except Exception as ex:                       # network hiccup or rate limit: wait, count, move on
                errors += 1
                if errors > max_errors:
                    raise RuntimeError(f"{name}: too many errors, last one: {ex}")
                time.sleep(5)
                continue
            ms = (time.time() - t0) * 1000
            rec = {"h": h, "ts": time.time(), "ms": round(ms, 1), "lat": got[0] if got else None, "lon": got[1] if got else None, "level": got[2] if got else ""}
            cache[h] = rec
            f.write(json.dumps(rec) + "\n")
            f.flush()
            new += 1
            if progress and new % progress == 0:
                print(f"  {name}: {new} new answers, {errors} errors")
            time.sleep(max(0.0, gap - (time.time() - t0)))
    out = []
    for i, text in zip(rows["id"], rows["text"].fillna("")):
        r = cache.get(_h(text))
        out.append((i, r["lat"] if r and r["lat"] is not None else np.nan, r["lon"] if r and r["lon"] is not None else np.nan,
                    r["level"] if r else "", r["ms"] if r else np.nan))
    print(f"{name}: {new} requested now, {len(out)} rows returned, {errors} errors")
    return pd.DataFrame(out, columns=["id", "lat", "lon", "level", "latency_ms"])


def scores_only(scores, pred):
    """What may be saved for a paid provider: the per-row scores plus the match level. No coordinates."""
    keep = scores.merge(pred[["id", "level"]], on="id", how="left")
    assert "lat" not in keep and "lon" not in keep
    return keep
