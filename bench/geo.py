"""Small geographic helpers used everywhere.

Everything here is plain numpy so it runs the same on Colab, Kaggle and a laptop.
"""
import numpy as np

EARTH_R = 6371008.8          # metres
UTM = "EPSG:32644"           # UTM zone 44N: metres, covers Hyderabad and Bengaluru
WGS = "EPSG:4326"
CITY_CENTRE = {"hyderabad": (17.385, 78.4867), "bengaluru": (12.9716, 77.5946)}


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres. Works on scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Compass bearing from point 1 to point 2, in degrees (0 = north, 90 = east)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


COMPASS_8 = np.array(["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"])
COMPASS_DEG = {name: i * 45.0 for i, name in enumerate(COMPASS_8)}


def compass_8(bearing):
    """Bearing in degrees -> one of eight direction words."""
    return COMPASS_8[(np.round(np.asarray(bearing) / 45.0).astype(int)) % 8]


def offset_point(lat, lon, bearing, dist_m):
    """Move dist_m metres from (lat, lon) along a compass bearing. Returns (lat, lon)."""
    lat1, lon1, brg = np.radians(lat), np.radians(lon), np.radians(bearing)
    d = np.asarray(dist_m) / EARTH_R
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(brg))
    lon2 = lon1 + np.arctan2(np.sin(brg) * np.sin(d) * np.cos(lat1), np.cos(d) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lat2), np.degrees(lon2)


# ---------------------------------------------------------------- DIGIPIN
# India Post's open grid code: 10 characters, each step splits the cell 4 x 4.
_DIGI = [["F", "C", "9", "8"], ["J", "3", "2", "7"], ["K", "4", "5", "6"], ["L", "M", "P", "T"]]
_DIGI_POS = {ch: (r, c) for r, row in enumerate(_DIGI) for c, ch in enumerate(row)}


def digipin_encode(lat, lon, length=10):
    min_lat, max_lat, min_lon, max_lon = 2.5, 38.5, 63.5, 99.5
    out = ""
    for _ in range(length):
        dlat, dlon = (max_lat - min_lat) / 4, (max_lon - min_lon) / 4
        r = min(max(3 - int((lat - min_lat) // dlat), 0), 3)
        c = min(max(int((lon - min_lon) // dlon), 0), 3)
        out += _DIGI[r][c]
        max_lat = min_lat + dlat * (4 - r)
        min_lat = max_lat - dlat
        min_lon = min_lon + dlon * c
        max_lon = min_lon + dlon
    return out


def digipin_decode(code):
    """DIGIPIN (any length up to 10, dashes ignored) -> centre (lat, lon) of its cell."""
    min_lat, max_lat, min_lon, max_lon = 2.5, 38.5, 63.5, 99.5
    for ch in code.replace("-", ""):
        r, c = _DIGI_POS[ch]
        dlat, dlon = (max_lat - min_lat) / 4, (max_lon - min_lon) / 4
        max_lat = min_lat + dlat * (4 - r)
        min_lat = max_lat - dlat
        min_lon = min_lon + dlon * c
        max_lon = min_lon + dlon
    return (min_lat + max_lat) / 2, (min_lon + max_lon) / 2


# ---------------------------------------------------------------- geohash
_GH = "0123456789bcdefghjkmnpqrstuvwxyz"
_GH_POS = {ch: i for i, ch in enumerate(_GH)}


def geohash_encode(lat, lon, precision=8):
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    out, bits, ch, even = [], 0, 0, True
    while len(out) < precision:
        if even:
            mid = (lon_lo + lon_hi) / 2
            if lon >= mid:
                ch, lon_lo = ch * 2 + 1, mid
            else:
                ch, lon_hi = ch * 2, mid
        else:
            mid = (lat_lo + lat_hi) / 2
            if lat >= mid:
                ch, lat_lo = ch * 2 + 1, mid
            else:
                ch, lat_hi = ch * 2, mid
        even = not even
        bits += 1
        if bits == 5:
            out.append(_GH[ch])
            bits, ch = 0, 0
    return "".join(out)


def geohash_decode(code):
    """Geohash -> centre (lat, lon) of its cell. Raises KeyError on an invalid character."""
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    even = True
    for c in code:
        v = _GH_POS[c]
        for shift in (4, 3, 2, 1, 0):
            bit = (v >> shift) & 1
            if even:
                mid = (lon_lo + lon_hi) / 2
                lon_lo, lon_hi = (mid, lon_hi) if bit else (lon_lo, mid)
            else:
                mid = (lat_lo + lat_hi) / 2
                lat_lo, lat_hi = (mid, lat_hi) if bit else (lat_lo, mid)
            even = not even
    return (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2


def ring_km(lat, lon, city="hyderabad"):
    """Rough distance from the city centre in km (used to stratify by centre vs outskirts)."""
    clat, clon = CITY_CENTRE[city]
    return haversine_m(lat, lon, clat, clon) / 1000.0
