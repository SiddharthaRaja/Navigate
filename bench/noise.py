"""Noise operations: the ways real people mangle an address.

Each function takes text (or a list of components) plus a numpy random generator and returns the changed
version. The generator records which operations were applied, so every benchmark row carries exact labels.

String-level ops follow the six corruption types Amazon's GeoGround paper trains with
(phonetic typo, token shuffle, token merge, abbreviation drift, omission, junk characters);
the rest are India-specific (Hinglish / Telugu-English phrasing, wrong pincode, appended instructions).
"""
import re
import numpy as np

ABBREV = {
    "road": ["rd", "rd."], "street": ["st", "str"], "colony": ["cly", "col", "clny"], "nagar": ["ngr", "ngr."],
    "apartments": ["apts", "appts", "apt"], "apartment": ["apt", "appt"], "residency": ["resi", "rsdncy"],
    "opposite": ["opp", "opp."], "near": ["nr", "nr."], "behind": ["bhnd", "bhd"], "beside": ["bsd", "bside"],
    "number": ["no", "no."], "hyderabad": ["hyd", "hyd."], "bengaluru": ["blr", "bglr"], "telangana": ["ts", "tg"],
    "building": ["bldg", "bld"], "floor": ["flr", "fl"], "hospital": ["hosp", "hsptl"], "school": ["schl", "sch"],
    "temple": ["tmpl", "temp"], "main": ["mn"], "cross": ["x", "crs"], "junction": ["jn", "jnc"], "station": ["stn"],
    "complex": ["cmplx", "cplx"], "enclave": ["encl", "enc"], "phase": ["ph"], "sector": ["sec"], "market": ["mkt"],
    "house": ["hse", "h"], "flat": ["flt", "f"], "block": ["blk"], "plot": ["plt", "p"], "lane": ["ln"],
    "police": ["ps", "pol"], "university": ["univ"], "college": ["clg", "coll"], "towers": ["twrs"], "bank": ["bnk"],
}

_VOWEL_SWAP = {"a": "e", "e": "a", "i": "e", "o": "u", "u": "o"}
_PHONETIC = [("th", "t"), ("sh", "s"), ("ph", "f"), ("ee", "i"), ("oo", "u"), ("aa", "a"), ("v", "w"), ("w", "v"),
             ("ch", "c"), ("ks", "x"), ("z", "j"), ("y", "i"), ("dh", "d"), ("bh", "b"), ("ck", "k")]
_KEYS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_JUNK = ["..", ",,", " #", " @", " --", " //", "***", " .", ";;", "~", "__"]

INSTRUCTIONS = [
    "call before coming", "call before delivery", "pls call on arrival", "call when you reach", "leave with watchman",
    "leave at security", "dont ring bell", "do not ring the bell baby sleeping", "deliver after 6pm", "deliver before 10 am",
    "come to 2nd floor", "use lift", "hand over to security guard", "give missed call", "office hours only",
    "keep at door", "contact {ph}", "ph {ph}", "mob: {ph}", "call {ph}", "alternate no {ph}",
]


def fake_phone(rng):
    """A phone-shaped number that is obviously not a real one (fixed 90000 prefix)."""
    return "90000" + "".join(str(int(d)) for d in rng.integers(0, 10, 5))


def typo_word(w, rng):
    """One phonetic or keyboard slip in a word."""
    if len(w) < 4:
        return w
    lw = w.lower()
    kind = rng.integers(0, 6)
    if kind == 0:                                   # phonetic respelling
        opts = [(a, b) for a, b in _PHONETIC if a in lw]
        if opts:
            a, b = opts[rng.integers(len(opts))]
            i = lw.find(a)
            return w[:i] + b + w[i + len(a):]
    if kind == 1:                                   # vowel swap
        idx = [i for i, c in enumerate(lw) if c in _VOWEL_SWAP and i > 0]
        if idx:
            i = idx[rng.integers(len(idx))]
            return w[:i] + _VOWEL_SWAP[lw[i]] + w[i + 1:]
    if kind == 2:                                   # drop a vowel
        idx = [i for i, c in enumerate(lw) if c in "aeiou" and 0 < i < len(w) - 1]
        if idx:
            i = idx[rng.integers(len(idx))]
            return w[:i] + w[i + 1:]
    if kind == 3:                                   # double or un-double a letter
        for i in range(1, len(w)):
            if lw[i] == lw[i - 1]:
                return w[:i] + w[i + 1:]
        i = int(rng.integers(1, len(w) - 1))
        return w[:i] + w[i] + w[i:]
    if kind == 4:                                   # swap two neighbouring letters
        i = int(rng.integers(1, len(w) - 2)) if len(w) > 4 else 1
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    i = int(rng.integers(1, len(w)))               # neighbouring key
    for row in _KEYS:
        j = row.find(lw[i])
        if j >= 0:
            k = min(max(j + (1 if rng.random() < 0.5 else -1), 0), len(row) - 1)
            return w[:i] + row[k] + w[i + 1:]
    return w


def typos(text, rng, n=None):
    """Misspell 1-3 of the longer alphabetic words."""
    parts = re.split(r"(\W+)", text)
    idx = [i for i, p in enumerate(parts) if p.isalpha() and len(p) >= 5]
    if not idx:
        return text
    n = n or int(rng.integers(1, 4))
    for i in rng.choice(idx, size=min(n, len(idx)), replace=False):
        parts[i] = typo_word(parts[i], rng)
    return "".join(parts)


def abbreviate(text, rng, p=0.8):
    """Shorten common address words (Road -> Rd, Opposite -> opp ...)."""
    def sub(m):
        w = m.group(0)
        opts = ABBREV.get(w.lower())
        if not opts or rng.random() > p:
            return w
        a = opts[rng.integers(len(opts))]
        return a.upper() if w.isupper() else a.capitalize() if w[0].isupper() else a
    return re.sub(r"[A-Za-z]+", sub, text)


def merge_tokens(text, rng):
    """Glue two neighbouring chunks together, as people do when they skip the space or comma."""
    seps = [m for m in re.finditer(r",\s*| ", text)]
    if not seps:
        return text
    m = seps[rng.integers(len(seps))]
    return text[:m.start()] + text[m.end():]


def junk(text, rng):
    """Stray punctuation and symbols."""
    out = text
    for _ in range(int(rng.integers(1, 3))):
        j = _JUNK[rng.integers(len(_JUNK))]
        spots = [m.start() for m in re.finditer(r"[ ,]", out)] + [len(out)]
        i = spots[rng.integers(len(spots))]
        out = out[:i] + j + out[i:]
    return out


def recase(text, rng):
    r = rng.random()
    return text.lower() if r < 0.6 else text.upper() if r < 0.85 else text.title()


def strip_commas(text, rng):
    return re.sub(r"\s+", " ", text.replace(",", " ")).strip()


def instructions(text, rng):
    """Append a delivery instruction or phone number (no location information in it)."""
    ins = INSTRUCTIONS[rng.integers(len(INSTRUCTIONS))].replace("{ph}", fake_phone(rng))
    sep = [", ", " - ", ". ", " (", " "][rng.integers(5)]
    return text + sep + ins + (")" if sep == " (" else "")


def gibberish(rng):
    """Monkey-typed or empty-of-meaning text: the right response is to abstain."""
    kind = rng.integers(0, 7)
    if kind == 0:
        row = _KEYS[rng.integers(3)]
        i = int(rng.integers(0, max(1, len(row) - 5)))
        return " ".join(row[i:i + int(rng.integers(3, 7))] for _ in range(int(rng.integers(1, 4))))
    if kind == 1:
        c = "abcdefghijklmnopqrstuvwxyz"[rng.integers(26)]
        return c * int(rng.integers(4, 12))
    if kind == 2:
        return "".join("bcdfghjklmnpqrstvwxz"[rng.integers(20)] for _ in range(int(rng.integers(6, 16))))
    if kind == 3:
        return "".join(str(int(d)) for d in rng.integers(0, 10, int(rng.integers(3, 9))))
    if kind == 4:
        return ["na", "n/a", "test", "xxx", "same as above", "address", "home", "my house", "none", ".", "-", "as before"][rng.integers(12)]
    if kind == 5:
        return INSTRUCTIONS[rng.integers(len(INSTRUCTIONS))].replace("{ph}", fake_phone(rng))
    return " ".join("".join("asdfghjklqwertyuiop"[rng.integers(19)] for _ in range(int(rng.integers(3, 8)))) for _ in range(int(rng.integers(2, 5))))


STRING_OPS = {"typo": typos, "abbrev": abbreviate, "merge": merge_tokens, "junk": junk, "case": recase,
              "nocomma": strip_commas, "instructions": instructions}
