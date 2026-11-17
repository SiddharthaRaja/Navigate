"""Quick self-checks for the bench/ code. No test framework needed:

    python tests/check_bench.py

Every check prints OK or stops with an error. None of them touch the network.
"""
import json
import os
import sys
import tempfile
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import baselines as bl, geo, geocoders as gc, metrics as mt, noise as nz, paraphrase as pp


def ok(name):
    print(f"OK  {name}")


# ---- geo
assert geo.digipin_encode(28.622788, 77.213033) == "39J49LL8T4"
la, lo = geo.digipin_decode("39J49LL8T4")
assert geo.haversine_m(la, lo, 28.622788, 77.213033) < 5
la, lo = geo.offset_point(17.4, 78.5, 90, 100)
assert abs(geo.haversine_m(17.4, 78.5, la, lo) - 100) < 1 and lo > 78.5
assert geo.compass_8(44) == "north-east"
ok("geo: digipin, offset, compass")

# ---- noise
rng = np.random.default_rng(0)
t = "H.No 8-2-293, Road No 12, opposite Apollo Hospital, Banjara Hills, Hyderabad 500034"
assert nz.typos(t, rng) != t and nz.abbreviate(t, rng) != t and len(nz.merge_tokens(t, rng)) < len(t)
assert all(len(nz.gibberish(rng)) > 0 for _ in range(50))
assert nz.fake_phone(rng).startswith("90000")
ok("noise ops")

# ---- pincode in text
assert bl.find_pin("near X, Hyd 500034 call 9000012345") == "500034"
assert bl.find_pin("ph 9000050003") is None and bl.find_pin("Flat 500, Road 5") is None
ok("pincode parsing ignores phone numbers")

# ---- gazetteer geocoder on a toy city
lm = pd.DataFrame({"name": ["Apollo Hospital", "Apollo Hospital", "Ratnadeep Supermarket"], "lat": [17.415, 17.50, 17.416],
                   "lon": [78.41, 78.55, 78.412], "lm_public": [True, True, True]})
g = bl.build_gazetteer(lm)
g = pd.concat([g, pd.DataFrame({"name": ["Banjara Hills"], "kind": ["locality"], "lat": [17.414], "lon": [78.43], "key": ["banjara hills"]})], ignore_index=True)
pins = pd.DataFrame({"lat": [17.41], "lon": [78.42]}, index=["500034"])
gg = bl.GazetteerGeocoder(g, pins)
q = pd.DataFrame({"id": ["a", "b", "c", "d"], "text": ["opp apolo hospital, banjara hills", "150 m north of Ratnadeep Supermarket, Banjara Hills 500034",
                                                 "asdf qwer", "Flat 2, Hyderabad 500034"]})
r = gg.geocode(q).set_index("id")
assert abs(r.at["a", "lat"] - 17.415) < 1e-6, r                         # picks the Apollo near Banjara Hills, despite the typo
assert r.at["b", "level"] == "landmark+offset" and r.at["b", "lat"] > 17.416
assert np.isnan(r.at["c", "lat"]) and r.at["d", "level"] == "pincode"
ok("gazetteer geocoder: typo, namesake, offset, abstain, pincode fallback")

# ---- metrics
truth = pd.DataFrame({"id": ["a", "b", "c", "d"], "split": "test_warm", "lat": [17.40, 17.41, 17.42, 17.43], "lon": [78.40] * 4,
                      "access_lat": [17.40, 17.41, 17.42, 17.43], "access_lon": [78.4001] * 4, "abstain": [False, False, False, True],
                      "pincode_true": [None] * 4, "pincode_in_text": [None] * 4, "building_id": list("wxyz"), "style": "formal"})
pred = pd.DataFrame({"id": ["a", "b", "c", "d"], "lat": [17.40, 17.4109, np.nan, np.nan], "lon": [78.40, 78.40, np.nan, np.nan]})
s, res = mt.evaluate(truth, pred, "toy")
o = res[res["slice"] == "overall"].iloc[0]
assert o["n"] == 3 and abs(o["coverage"] - 2 / 3) < 1e-9 and abs(o["acc@100"] - 1 / 3) < 1e-9 and abs(o["acc@200"] - 2 / 3) < 1e-9
assert o["abstain_r"] == 1.0 and o["abstain_p"] == 0.5
ok("metrics: coverage, acc@k count unanswered rows as misses, abstain precision / recall")

# ---- geocoder response parsers (sample payloads in each vendor's documented shape)
assert gc.parse_nominatim([{"lat": "17.36", "lon": "78.47", "addresstype": "road"}]) == (17.36, 78.47, "road")
assert gc.parse_nominatim([]) is None
assert gc.parse_photon({"features": [{"geometry": {"coordinates": [78.47, 17.36]}, "properties": {"type": "street"}}]}) == (17.36, 78.47, "street")
assert gc.parse_google({"status": "OK", "results": [{"geometry": {"location": {"lat": 17.36, "lng": 78.47}, "location_type": "ROOFTOP"}}]}) == (17.36, 78.47, "ROOFTOP")
assert gc.parse_google({"status": "ZERO_RESULTS", "results": []}) is None
assert gc.parse_ola({"geocodingResults": [{"geometry": {"location": {"lat": 17.36, "lng": 78.47}}}]})[:2] == (17.36, 78.47)
assert gc.parse_mappls({"copResults": {"latitude": 17.36, "longitude": 78.47, "geocodeLevel": "houseNumber"}}) == (17.36, 78.47, "houseNumber")
ok("geocoder parsers")

# ---- run_provider: cache, resume, scores-only
calls = []
def fake_get(url, params=None, headers=None, timeout=20):
    calls.append(params["q"])
    return [{"lat": "17.40", "lon": "78.40", "addresstype": "road"}] if "road" in params["q"] else []
df = pd.DataFrame({"id": ["a", "b"], "text": ["some road, hyd", "nowhere"]})
with tempfile.TemporaryDirectory() as d, mock.patch.object(gc, "_get", fake_get), mock.patch.dict(gc.PROVIDERS["nominatim"], {"rps": 1000.0}):
    path = f"{d}/cache/nominatim.jsonl"
    p1 = gc.run_provider("nominatim", df, path)
    p2 = gc.run_provider("nominatim", df, path)                         # second run must not call the service again
    assert len(calls) == 2 and p1["lat"].notna().tolist() == [True, False] and p2.equals(p1)
    assert gc.purge_cache(path, max_age_days=30) == 0 and gc.purge_cache(path, max_age_days=-1) == 2
tr2 = truth.iloc[:2]
sc = gc.scores_only(mt.score_rows(tr2, p1.assign(id=["a", "b"])), p1)
assert "lat" not in sc.columns and "err_m" in sc.columns
ok("run_provider: cache, resume, purge, scores-only")

# ---- paraphrase fact check
comp = {"house_no": "H.No 8-2-293", "landmark": "opposite Apollo Hospital", "locality": "Banjara Hills", "pincode": "500034"}
orig = "H.No 8-2-293, opposite Apollo Hospital, Banjara Hills, Hyderabad 500034"
k = lambda rw, **kw: pp.keep(orig, rw, comp, relation="ACROSS_THE_ROAD", ref_name="Apollo Hospital", **kw)
assert k("8-2-293 opp apollo hospital banjara hills hyd 500034")[0]
assert k("8-2-293 behind apollo hospital banjara hills hyd 500034") == (False, "relation word lost")
assert k("8-2-294 opp apollo hospital banjara hills hyd 500034") == (False, "numbers changed")
assert k("8-2-293 opp care hospital banjara hills hyd 500034") == (False, "landmark lost")
assert k("8-2-293 opp apollo hospital jubilee hills hyd 500034") == (False, "locality lost")
o2 = "150 m north of City Mall, Ameerpet"
assert pp.keep(o2, "150m N of city mall ameerpet", {"locality": "Ameerpet"}, "OFFSET", "City Mall")[0]
assert pp.keep(o2, "150m south of city mall ameerpet", {"locality": "Ameerpet"}, "OFFSET", "City Mall") == (False, "direction changed")
assert pp.clean_output('Rewritten: "opp apollo, banjara hills"\nextra') == "opp apollo, banjara hills"
ok("paraphrase fact check")
print("\nall checks passed")
