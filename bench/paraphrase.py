"""LLM paraphrasing of generated addresses, with a fact check.

Templates produce correct but stiff text. A language model rewrites each address the way a real person might
type it. The danger is that it also changes facts (drops the landmark, alters a number, flips "opposite" to
"behind"), which would silently corrupt the labels. So every rewrite goes through keep(): if any fact is lost
or changed, the rewrite is thrown away and the template text is kept.

Nothing here needs a GPU; the model call itself lives in notebook 07.
"""
import json
import re

from rapidfuzz import fuzz

from .context import name_key

SYSTEM = ("You rewrite Indian postal addresses the way real people type them into a delivery app. "
          "Keep every fact exactly: all numbers, every place name, and the meaning of position words "
          "(near / opposite / behind / beside / inside / lane next to). You may change word order, casing, "
          "punctuation and abbreviations, and you may use casual phrasing. Do not add any place, number or "
          "detail that is not in the input. Do not explain. Output the rewritten address on one line.")

STYLES = ["typed quickly on a phone, lowercase, few commas",
          "neat and complete, as on a courier label",
          "casual, the way you would text it to a delivery rider",
          "short, with common abbreviations like opp, nr, rd, apt",
          "written by an older relative, formal and a bit long"]

# Words that signal each relation. A rewrite must keep the same family and not introduce another one.
REL_WORDS = {
    "ACROSS_THE_ROAD": ["opp", "opposite", "across", "saamne", "samne", "eduruga", "edurugaa", "eduru", "facing"],
    "BEHIND": ["behind", "back side", "backside", "peeche", "piche", "venuka", "venaka", "venakala", "bhnd", "rear"],
    "BESIDE": ["beside", "besides", "next to", "adjacent", "adj", "bagal", "baju", "pakkana", "pakkane", "next door"],
    "WITHIN": ["inside", "within", "andar", "ander", "lopala", "lopali", "campus"],
    "AROUND_THE_CORNER": ["lane beside", "lane adjacent", "lane next", "corner", "side lane", "turning", "turn at", "gali", "galli", "sandhu"],
    "DOWN_THE_ROAD": ["same road", "same lane", "down the road", "road", "lane", "line", "raste", "dari"],
    "NEAR": ["near", "nr", "close to", "nearby", "landmark", "paas", "pass", "nazdeek", "daggara", "daggaralo", "dagara"],
}
STRICT = ["ACROSS_THE_ROAD", "BEHIND", "BESIDE", "WITHIN"]            # families that must not appear unless they were there


def build_prompt(text, style):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Style: {style}\nAddress: {text}\nRewritten:"}]


def _numbers(s):
    return sorted(re.findall(r"\d+", s))


def _has(words, key):
    return any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", key) for w in words)


def keep(original, rewrite, components, relation=None, ref_name=None, min_name=85):
    """Return (ok, reason). ok is True only if the rewrite preserves every fact in the original.

    components: the dict (or JSON string) the generator stored for this row.
    """
    if not isinstance(rewrite, str):
        return False, "empty"
    rw = rewrite.strip().strip('"').split("\n")[0].strip()
    if not (8 <= len(rw) <= max(60, 2.2 * len(original))):
        return False, "length"
    comp = json.loads(components) if isinstance(components, str) else dict(components or {})
    if _numbers(rw) != _numbers(original):
        return False, "numbers changed"
    key = name_key(rw)
    for part in ("building", "street", "area", "locality", "suburb"):
        v = comp.get(part)
        if v and fuzz.partial_ratio(name_key(v), key) < min_name:
            return False, f"{part} lost"
    if ref_name and fuzz.partial_ratio(name_key(ref_name), key) < min_name:
        return False, "landmark lost"
    okey = name_key(original)
    if relation in REL_WORDS and relation not in ("NEAR", "DOWN_THE_ROAD") and _has(REL_WORDS[relation], okey):
        if not _has(REL_WORDS[relation], key):                 # the original said "opposite"; the rewrite must still say so
            return False, "relation word lost"
    if relation == "OFFSET":                                   # "150 m north of X": distance and direction must survive
        from .baselines import DIR_WORDS, OFFSET_RE
        m_old, m_new = OFFSET_RE.search(original), OFFSET_RE.search(rw)
        if not m_new:
            return False, "offset lost"
        canon = lambda m: DIR_WORDS.get(m.group(2).lower().replace(" ", "-"))
        if m_old and (canon(m_old) != canon(m_new) or m_old.group(1) != m_new.group(1)):
            return False, "direction changed"
    for fam in STRICT:
        if fam != relation and _has(REL_WORDS[fam], key) and not _has(REL_WORDS[fam], okey):
            return False, f"relation changed to {fam}"
    return True, "ok"


def clean_output(s):
    """Models like to add quotes or a label; strip them."""
    s = (s or "").strip().split("\n")[0]
    s = re.sub(r"^(rewritten|address|output)\s*[:\-]\s*", "", s, flags=re.I).strip().strip('"').strip("'").strip()
    return s
