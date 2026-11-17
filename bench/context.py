"""Building context: for each target building, what could a person truthfully write about where it is?

Two tables come out of this module:

targets  : one row per building  - its point, locality, suburb, complex, street, pincode
relations: one row per (building, nearby landmark) pair - the spatial relation between them

Relations use the seven labels from Google's Address Descriptors
(NEAR, WITHIN, BESIDE, ACROSS_THE_ROAD, DOWN_THE_ROAD, AROUND_THE_CORNER, BEHIND),
worked out from OpenStreetMap geometry.  The rules are deliberately strict: a pair only gets a
specific label when the geometry clearly supports it, otherwise it stays NEAR.
"""
import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

from .geo import UTM, WGS, bearing_deg, compass_8, haversine_m

DRIVABLE = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "service",
            "living_street", "road", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}

# Thresholds (metres) for the relation rules. Kept in one place so they are easy to defend and tune.
R = dict(near=300, across_along=35, across_d=80, road_close=40, beside_along=35, beside_d=45, touching=18,
         behind_d=55, behind_along=30, behind_extra=20, down_max=200, corner_path=150, cross_check=90, parallel_cos=0.87, turn_cos=0.70)

RELATIONS = ["WITHIN", "ACROSS_THE_ROAD", "BESIDE", "BEHIND", "DOWN_THE_ROAD", "AROUND_THE_CORNER", "NEAR"]


# ------------------------------------------------------------------ loading and cleaning
def _latin(s):
    return isinstance(s, str) and len(s) > 0 and sum(c.isascii() for c in s) / len(s) >= 0.9


def clean_name(s):
    """Tidy a map name; return None if it is not something a person would write in an address."""
    if not isinstance(s, str):
        return None
    s = re.sub(r"\s+", " ", s).strip(" ,.-")
    if not (3 <= len(s) <= 45) or not _latin(s):
        return None
    if re.search(r"[|;]|https?:|www\.", s) or re.fullmatch(r"[\d\W]+", s):
        return None
    return s


def name_key(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip() if isinstance(s, str) else ""


_HIGH = r"worship|temple|mandir|mosque|masjid|church|hospital|school|college|university|bank|fuel|petrol|gas_station|police|post_office|cinema|theatre|marketplace|mall|supermarket|department_store|park|stadium|station|bus_stop|hotel|museum|attraction|townhall"
_LOW = r"childcare|atm|health_post|hostel|vending|toilets|parking|bench|waste|telephone|other$"


def salience(category):
    """How likely people are to use this kind of place as a landmark (sampling weight)."""
    c = str(category).lower()
    if re.search(_LOW, c):
        return 0.3
    if re.search(_HIGH, c):
        return 3.0
    return 1.0


def load_buildings(gaz_dir):
    cols = ["osm_id", "name", "area_m2", "lat", "lon", "h3_7", "h3_8", "h3_9", "h3_10", "h3_11", "digipin", "dist_to_road_m"]
    b = pd.read_parquet(f"{gaz_dir}/buildings.parquet", columns=cols)
    return b.drop_duplicates("osm_id").reset_index(drop=True)


def load_landmarks(gaz_dir, buildings=None):
    """Landmarks from notebook 02 (OSM + Overture) if present, else OSM only. One clean row per landmark."""
    merged = f"{gaz_dir}/landmarks_merged.parquet"
    if os.path.exists(merged):
        g = gpd.read_parquet(merged)
        lm = pd.DataFrame({"landmark_id": g["landmark_id"].astype(str), "name": g["name"], "category": g["category"],
                           "source": g["source"], "lat": g.geometry.y, "lon": g.geometry.x})
    else:
        g = gpd.read_parquet(f"{gaz_dir}/landmarks.parquet")
        lm = pd.DataFrame({"landmark_id": "osm_" + g["osm_id"].astype(str), "name": g["name"], "category": g["category"],
                           "source": "osm", "lat": g.geometry.y, "lon": g.geometry.x})
    lm = lm.drop_duplicates("landmark_id")
    lm["name"] = lm["name"].map(clean_name)
    lm = lm.dropna(subset=["name"]).reset_index(drop=True)
    if buildings is not None:                       # a landmark that is a building sits at the building's point
        bpt = buildings.set_index("osm_id")[["lat", "lon"]]
        key = lm["landmark_id"].str.replace("osm_", "", n=1)
        hit = key.isin(bpt.index).to_numpy()
        lm.loc[hit, ["lat", "lon"]] = bpt.loc[key[hit]].to_numpy()
        lm["building_id"] = np.where(hit, key, None)
    lm["name_key"] = lm["name"].map(name_key)
    lm = lm[lm["name_key"].str.len() >= 3]
    # the same place often appears twice (node + building, or OSM + Overture): keep one per name within ~60 m
    lm["_cell"] = (lm["lat"] / 0.0006).round().astype(int).astype(str) + "_" + (lm["lon"] / 0.0006).round().astype(int).astype(str)
    lm = lm.sort_values("source").drop_duplicates(["name_key", "_cell"]).drop(columns="_cell").reset_index(drop=True)
    lm["name_count"] = lm.groupby("name_key")["name_key"].transform("size")
    lm["salience"] = lm["category"].map(salience)
    return lm


def load_roads(gaz_dir):
    r = gpd.read_parquet(f"{gaz_dir}/roads.parquet")
    r = r[r["highway"].isin(DRIVABLE)].drop_duplicates("osm_id").reset_index(drop=True)
    r["name"] = r["name"].map(clean_name)
    return r.to_crs(UTM)


def load_areas(gaz_dir):
    """Named polygons a building can be 'inside', and named points for localities and suburbs."""
    a = gpd.read_parquet(f"{gaz_dir}/areas.parquet")
    a["name"] = a["name"].map(clean_name)
    a = a.dropna(subset=["name"])
    a["_poly"] = a.geom_type.isin(["Polygon", "MultiPolygon"])
    a = a.sort_values("_poly", ascending=False).drop_duplicates("osm_id")
    au = a.to_crs(UTM)

    poly = au[au["_poly"] & (au.get("landuse") == "residential")].copy()
    poly["area_m2"] = poly.area
    poly = poly[(poly["area_m2"] >= 2000) & (poly["area_m2"] <= 1.5e6)]
    poly = poly[~poly["name"].str.contains(r"\b(?:store|stores|kirana|shop|mart|bakery|hotel)\b", case=False, regex=True)]
    poly["kind"] = np.where(poly["name"].str.contains(
        r"apartment|residenc|tower|villa|heights|homes|enclave|county|meadows|greens|gardens|park|palace|plaza|nest|arcade|block",
        case=False, regex=True), "complex", "colony")
    poly["area_id"] = "osm_" + poly["osm_id"].astype(str)
    cen = poly.geometry.representative_point().to_crs(WGS)
    poly["lat"], poly["lon"] = cen.y.values, cen.x.values
    extra = f"{gaz_dir}/landmark_areas.parquet"          # written by notebook 01 (campuses, malls, parks)
    if os.path.exists(extra):
        e = gpd.read_parquet(extra).to_crs(UTM)
        e["name"] = e["name"].map(clean_name)
        e = e.dropna(subset=["name"])
        e["area_m2"] = e.area
        e = e[(e["area_m2"] >= 2000) & (e["area_m2"] <= 1.5e6)]
        e["kind"] = "campus"
        e["area_id"] = "osm_" + e["osm_id"].astype(str)
        cen = e.geometry.representative_point().to_crs(WGS)
        e["lat"], e["lon"] = cen.y.values, cen.x.values
        poly = pd.concat([poly, e[~e["area_id"].isin(poly["area_id"])]], ignore_index=True)
    poly = gpd.GeoDataFrame(poly[["area_id", "name", "kind", "area_m2", "lat", "lon", "geometry"]], crs=UTM).reset_index(drop=True)

    pts = au.copy()
    pts["geometry"] = pts.geometry.representative_point()
    place = pts.get("place")
    # place names only: OSM place nodes are sometimes misused to advertise a business
    spam = pts["name"].str.contains(r" - | in hyderabad|pvt|ltd|services|designer|solutions|clinic|\d{4,}", case=False, regex=True)
    pts = pts[~spam & (pts["name"].str.len() <= 32) & (pts["name"].str.split().str.len() <= 4)]
    place = pts.get("place")
    locality = pts[place.isin(["neighbourhood", "locality", "quarter", "village", "hamlet"])][["name", "geometry"]].reset_index(drop=True)
    suburb = pts[place.isin(["suburb", "town"])][["name", "geometry"]].reset_index(drop=True)
    return poly, locality, suburb


# ------------------------------------------------------------------ geometry helpers
def snap_to_roads(x, y, roads, max_dist=150.0):
    """Snap points (UTM metres) to the nearest drivable road.

    Returns a dict of arrays, one entry per point: road index (-1 if none within max_dist), distance along the
    road, snapped x/y, distance to the road, unit direction of the road there, and which side the point is on (+1/-1).
    """
    n = len(x)
    pts = shapely.points(x, y)
    lines_all = roads.geometry.values
    tree = shapely.STRtree(lines_all)
    (src, dst), dist = tree.query_nearest(pts, max_distance=max_dist, return_distance=True, all_matches=False)
    out = dict(road=np.full(n, -1), s=np.full(n, np.nan), sx=np.full(n, np.nan), sy=np.full(n, np.nan),
               dist=np.full(n, np.nan), dx=np.full(n, np.nan), dy=np.full(n, np.nan), side=np.zeros(n))
    if len(src) == 0:
        return out
    lines = lines_all[dst]
    s = shapely.line_locate_point(lines, pts[src])
    length = shapely.length(lines)
    snap = shapely.line_interpolate_point(lines, s)
    a = shapely.get_coordinates(shapely.line_interpolate_point(lines, np.clip(s - 3, 0, None)))
    b = shapely.get_coordinates(shapely.line_interpolate_point(lines, np.minimum(s + 3, length)))
    d = b - a
    norm = np.hypot(d[:, 0], d[:, 1]) + 1e-9
    d = d / norm[:, None]
    sxy = shapely.get_coordinates(snap)
    side = np.sign(d[:, 0] * (y[src] - sxy[:, 1]) - d[:, 1] * (x[src] - sxy[:, 0]))
    out["road"][src] = dst
    out["s"][src] = s
    out["sx"][src], out["sy"][src] = sxy[:, 0], sxy[:, 1]
    out["dist"][src] = dist
    out["dx"][src], out["dy"][src] = d[:, 0], d[:, 1]
    out["side"][src] = side
    return out


def road_junctions(roads):
    """Pairs of drivable roads that share a node. Returns DataFrame a, b (road indices), nx, ny (the shared node)."""
    coords, idx = shapely.get_coordinates(roads.geometry.values, return_index=True)
    key = np.round(coords[:, 0] * 10).astype(np.int64) * 100_000_000 + np.round(coords[:, 1] * 10).astype(np.int64)
    df = pd.DataFrame({"key": key, "road": idx, "nx": coords[:, 0], "ny": coords[:, 1]}).drop_duplicates(["key", "road"])
    df = df[df.groupby("key")["road"].transform("size") > 1]
    j = df.merge(df, on="key", suffixes=("_a", "_b"))
    j = j[j["road_a"] != j["road_b"]]
    return pd.DataFrame({"a": j["road_a"].to_numpy(), "b": j["road_b"].to_numpy(),
                         "nx": j["nx_a"].to_numpy(), "ny": j["ny_a"].to_numpy()})


def _nearest_named(points_xy, named, max_dist):
    """For each point, the nearest named place (GeoDataFrame of points) within max_dist. Returns name, dist, idx."""
    n = len(points_xy)
    name, dist, idx = np.full(n, None, dtype=object), np.full(n, np.nan), np.full(n, -1)
    if len(named) == 0:
        return name, dist, idx
    tree = shapely.STRtree(named.geometry.values)
    (src, dst), d = tree.query_nearest(shapely.points(points_xy[:, 0], points_xy[:, 1]), max_distance=max_dist,
                                       return_distance=True, all_matches=False)
    name[src], dist[src], idx[src] = named["name"].to_numpy()[dst], d, dst
    return name, dist, idx


# ------------------------------------------------------------------ main steps
def pick_targets(buildings, n_targets=200_000, seed=7):
    """Choose the buildings we will write addresses for."""
    ok = buildings["area_m2"].between(30, 20000) & (buildings["dist_to_road_m"].fillna(1e9) <= 60)
    b = buildings[ok]
    return b.sample(min(n_targets, len(b)), random_state=seed).reset_index(drop=True)


def build_targets(targets, roads, area_poly, locality, suburb, pin_lookup=None):
    """Add locality, suburb, complex, street and pincode to each target building."""
    t = targets.copy()
    xy = shapely.get_coordinates(gpd.GeoSeries(gpd.points_from_xy(t["lon"], t["lat"]), crs=WGS).to_crs(UTM).values)
    t["x"], t["y"] = xy[:, 0], xy[:, 1]
    t["bname"] = t["name"].map(clean_name)

    snap = snap_to_roads(xy[:, 0], xy[:, 1], roads)
    for k, v in snap.items():
        t[f"r_{k}"] = v
    rn = roads["name"].to_numpy()
    t["street"] = np.where((snap["road"] >= 0) & (snap["dist"] <= 40), rn[np.clip(snap["road"], 0, None)], None)
    acc = gpd.GeoSeries(gpd.points_from_xy(np.nan_to_num(snap["sx"], nan=xy[:, 0]), np.nan_to_num(snap["sy"], nan=xy[:, 1])), crs=UTM).to_crs(WGS)
    t["access_lat"], t["access_lon"] = acc.y.values, acc.x.values

    t["locality"], t["locality_dist_m"], li = _nearest_named(xy, locality, 1500)
    loc_ll = locality.to_crs(WGS).geometry if len(locality) else None
    t["locality_lat"] = np.where(li >= 0, loc_ll.y.to_numpy()[np.clip(li, 0, None)], np.nan) if loc_ll is not None else np.nan
    t["locality_lon"] = np.where(li >= 0, loc_ll.x.to_numpy()[np.clip(li, 0, None)], np.nan) if loc_ll is not None else np.nan
    t["suburb"], t["suburb_dist_m"], _ = _nearest_named(xy, suburb, 5000)
    t.loc[t["suburb"] == t["locality"], "suburb"] = None

    # the smallest named polygon that contains the building
    for c in ("area_id", "area_name", "area_kind"):
        t[c] = None
    t["area_m2_in"], t["area_lat"], t["area_lon"] = np.nan, np.nan, np.nan
    if len(area_poly):
        tree = shapely.STRtree(area_poly.geometry.values)
        qi, gi = tree.query(shapely.points(xy[:, 0], xy[:, 1]), predicate="within")
        if len(qi):
            m = pd.DataFrame({"q": qi, "g": gi, "a": area_poly["area_m2"].to_numpy()[gi]}).sort_values("a").drop_duplicates("q")
            q, g = m["q"].to_numpy(), m["g"].to_numpy()
            t.loc[q, "area_id"] = area_poly["area_id"].to_numpy()[g]
            t.loc[q, "area_name"] = area_poly["name"].to_numpy()[g]
            t.loc[q, "area_kind"] = area_poly["kind"].to_numpy()[g]
            t.loc[q, "area_m2_in"] = area_poly["area_m2"].to_numpy()[g]
            t.loc[q, "area_lat"] = area_poly["lat"].to_numpy()[g]
            t.loc[q, "area_lon"] = area_poly["lon"].to_numpy()[g]

    if pin_lookup is not None:
        t["pincode"], t["pincode_conf"] = pin_lookup(t["lat"].to_numpy(), t["lon"].to_numpy())
    else:
        t["pincode"], t["pincode_conf"] = None, 0.0
    return t.rename(columns={"osm_id": "building_id"})


def label_relations(t, lm, roads):
    """All (target, landmark) pairs within 300 m, each with one relation label.

    t  : output of build_targets          lm : output of load_landmarks          roads: output of load_roads
    """
    lxy = shapely.get_coordinates(gpd.GeoSeries(gpd.points_from_xy(lm["lon"], lm["lat"]), crs=WGS).to_crs(UTM).values)
    ls = snap_to_roads(lxy[:, 0], lxy[:, 1], roads)
    tree = shapely.STRtree(shapely.points(lxy[:, 0], lxy[:, 1]))
    ti, li = tree.query(shapely.points(t["x"].to_numpy(), t["y"].to_numpy()), predicate="dwithin", distance=R["near"])

    tx, ty = t["x"].to_numpy()[ti], t["y"].to_numpy()[ti]
    lx, ly = lxy[li, 0], lxy[li, 1]
    d = np.hypot(tx - lx, ty - ly)
    t_road, l_road = t["r_road"].to_numpy()[ti], ls["road"][li]
    t_s, l_s = t["r_s"].to_numpy()[ti], ls["s"][li]
    t_sx, t_sy = t["r_sx"].to_numpy()[ti], t["r_sy"].to_numpy()[ti]
    l_sx, l_sy = ls["sx"][li], ls["sy"][li]
    t_rd, l_rd = t["r_dist"].to_numpy()[ti], ls["dist"][li]
    ldx, ldy, l_side = ls["dx"][li], ls["dy"][li], ls["side"][li]
    tdx, tdy = t["r_dx"].to_numpy()[ti], t["r_dy"].to_numpy()[ti]
    have = (t_road >= 0) & (l_road >= 0)

    # where the target sits relative to the LANDMARK's road: which side, how far along, how far away
    t_side_on_l = np.sign(ldx * (ty - l_sy) - ldy * (tx - l_sx))
    t_along_on_l = np.abs(ldx * (tx - l_sx) + ldy * (ty - l_sy))
    t_perp_on_l = np.abs(ldx * (ty - l_sy) - ldy * (tx - l_sx))
    cosang = np.abs(tdx * ldx + tdy * ldy)

    same_way = have & (t_road == l_road)
    rn = roads["name"].to_numpy()
    tn, ln = rn[np.clip(t_road, 0, None)], rn[np.clip(l_road, 0, None)]
    snap_gap = np.hypot(t_sx - l_sx, t_sy - l_sy)
    same_name = have & ~same_way & (tn == ln) & pd.notna(tn) & (snap_gap <= 400)

    # roads that meet at a junction: straight on = same road continuing, a turn = around the corner
    junc = road_junctions(roads)
    pair = pd.DataFrame({"a": t_road, "b": l_road, "row": np.arange(len(ti))})
    pair = pair[have & ~same_way].merge(junc, on=["a", "b"], how="inner")
    r = pair["row"].to_numpy()
    path = np.hypot(t_sx[r] - pair["nx"], t_sy[r] - pair["ny"]) + np.hypot(l_sx[r] - pair["nx"], l_sy[r] - pair["ny"])
    best = pd.DataFrame({"row": r, "path": path.to_numpy()}).groupby("row")["path"].min()
    via = np.full(len(ti), np.nan)
    via[best.index.to_numpy()] = best.to_numpy()
    joined = ~np.isnan(via)
    continuation = joined & (cosang >= R["parallel_cos"])
    turn = joined & (cosang <= R["turn_cos"])

    same_road = same_way | same_name | continuation
    along = np.where(same_way, np.abs(t_s - l_s), np.where(continuation, via, snap_gap))
    close_to_road = (t_rd <= R["road_close"]) & (l_rd <= R["road_close"])

    # how many drivable roads does the straight line from building to landmark cross? (only checked for close pairs)
    n_cross = np.full(len(ti), -1)
    chk = np.flatnonzero(d <= R["cross_check"])
    if len(chk):
        seg = shapely.linestrings(np.stack([np.c_[tx[chk], ty[chk]], np.c_[lx[chk], ly[chk]]], axis=1))
        qi, _ = shapely.STRtree(roads.geometry.values).query(seg, predicate="intersects")
        n_cross[chk] = np.bincount(qi, minlength=len(chk))

    across = (same_road & (t_side_on_l * l_side < 0) & (along <= R["across_along"]) & (d <= R["across_d"]) & close_to_road
              & (n_cross >= 1) & (n_cross <= 2))
    beside = ((same_road & (t_side_on_l * l_side > 0) & (along <= R["beside_along"]) & (d <= R["beside_d"])) | (d <= R["touching"])) & (n_cross == 0)
    behind = (have & ~same_road & (d <= R["behind_d"]) & (t_along_on_l <= R["behind_along"]) & (t_side_on_l * l_side > 0)
              & (t_perp_on_l >= l_rd + R["behind_extra"]) & (l_rd <= R["road_close"]) & (n_cross == 0))
    down = same_road & (along > R["across_along"]) & (along <= R["down_max"]) & close_to_road
    corner = turn & ~same_road & (via <= R["corner_path"]) & close_to_road

    rel = np.select([across, beside, behind, down, corner], ["ACROSS_THE_ROAD", "BESIDE", "BEHIND", "DOWN_THE_ROAD", "AROUND_THE_CORNER"], "NEAR")

    tlat, tlon = t["lat"].to_numpy()[ti], t["lon"].to_numpy()[ti]
    llat, llon = lm["lat"].to_numpy()[li], lm["lon"].to_numpy()[li]
    brg = bearing_deg(llat, llon, tlat, tlon)                 # direction FROM the landmark TO the building
    out = pd.DataFrame({
        "building_id": t["building_id"].to_numpy()[ti], "landmark_id": lm["landmark_id"].to_numpy()[li],
        "relation": rel, "d_m": haversine_m(tlat, tlon, llat, llon).round(1), "along_m": np.round(along, 1),
        "bearing": brg.round(1), "direction": compass_8(brg),
    })
    # a building is not a landmark for itself
    if "building_id" in lm:
        own = lm["building_id"].to_numpy()[li] == out["building_id"].to_numpy()
        out = out[~own & (out["d_m"] > 3)]
    return out.reset_index(drop=True)
