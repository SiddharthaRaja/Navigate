# Benchmark card

What the Navigate benchmark contains, how it was made, and where it is weak.

## Task

Input: one address string for a place in Hyderabad. Output: a latitude and longitude, or "no answer". Score: distance in metres from the answer to the true building.

## How a test address is made

1. **Truth.** A building footprint from OpenStreetMap (OSM). Its point is a location guaranteed to be inside the footprint.
2. **Context** (notebook 04). What can truthfully be said about it: street (if within 40 m), named complex or colony it sits inside, nearest locality and suburb, pincode, and its relation to every named landmark within 300 m.
3. **Split** (notebook 05). Decided before any text exists.
4. **Text** (notebook 06). A template picks some of that context and writes it in one of five styles, then noise is applied. Every choice is recorded on the row.
5. **Rewrite** (notebook 07, optional). An LLM rewrites a share of rows; code checks that no fact changed.

## What is real and what is invented

| Part of the text | Source |
|---|---|
| Building name, street, landmark, complex, locality, suburb | OSM (and Overture Places), always real |
| Pincode | boundary file if usable, else a vote of nearby OSM / Overture points with a pincode (about 92% correct, measured by leave-one-out) |
| Flat and floor numbers | invented; harmless, no geocoder resolves them |
| Door numbers ("H.No 8-2-293/A") | invented, and flagged in `synthetic_tokens`. The prefix is shared within an H3 resolution-8 cell and the last number runs along the road, as real numbering does |

Rows with an invented door number are excluded from the comparison with outside geocoders (`api_subset`), because an invented number can match a real house somewhere else.

## Test sets

| `split` | Rows (full build) | Content |
|---|---|---|
| `test_warm` | 3,000 | unseen buildings in H3-7 cells that have training data |
| `test_cold` | 3,000 | buildings in H3-7 cells with no training data at all (about 15% of cells, spread across centre and outskirts) |
| `test_rel` | up to 250 per relation | fixed quota per landmark relation; mixes seen and unseen areas (see column `area_seen`) |
| `test_real` | 1,400 or more | address tags typed by OSM mappers, plus verified Overture business addresses with the business name removed |

About 2% of warm and cold rows are gibberish with `abstain = True`. The right response is no answer.

## Columns on every row

| Column | Meaning |
|---|---|
| `id`, `split`, `text` | row id, test set, the address as the model sees it |
| `lat`, `lon`, `building_id` | the true point and OSM building |
| `access_lat`, `access_lon` | nearest point on a drivable road |
| `h3_7` … `h3_10`, `digipin` | grid codes of the true point |
| `pincode_true`, `pincode_in_text` | they differ when wrong-pincode noise was applied |
| `style` | formal, landmark, complex, offset, minimal, real, gibberish |
| `relation` | NEAR, WITHIN, BESIDE, ACROSS_THE_ROAD, BEHIND, DOWN_THE_ROAD, AROUND_THE_CORNER, OFFSET |
| `ref_name`, `ref_lat`, `ref_lon`, `ref_dist_m` | the landmark named in the text |
| `ref_in_gazetteer` | False if that landmark is one of the hidden 20% |
| `noise_ops`, `noise_level` | which noise operations were applied; clean / light (1–2) / heavy (3+) |
| `lang` | en, hi or te (the relation phrase is in Roman-letter Hindi or Telugu) |
| `components` | the parts that went into the text, as JSON |
| `oracle_err_m` | distance from the building to the best place named in the text |
| `nn_train_m`, `area_seen` | distance to the nearest training building; whether the cell had training data |
| `api_subset` | row is in the stratified subset sent to paid geocoders |
| `text_clean` | the text before string-level noise |

## Relation rules

Worked out in `bench/context.py` from OSM road and building geometry, thresholds in one dictionary at the top of the file. A pair gets a specific label only when the geometry clearly supports it (for instance, "beside" also requires that no road passes between the two). Everything else stays NEAR. In the Hyderabad trial about 24% of buildings have at least one specific relation.

These labels are produced by rules, not by people. They were checked by plotting samples, not by a formal annotation study.

## Metrics

Defined in `bench/metrics.py`. Accuracy within 100 / 200 / 500 / 1,000 / 2,000 m counts unanswered rows as misses; error percentiles are over answered rows, so always read them next to coverage. Also: share in the wrong pincode area (misroute), share on the wrong side of the building's road, distance to the road access point, and precision / recall of abstaining on gibberish.

## Known weaknesses

1. **The text is generated from the same map that lookup baselines search.** A fuzzy matcher can partly invert the generator. Counter-measures: hidden landmarks, name shortening, the real-text set, LLM rewrites, and reporting the gap between generated and real text. The gap is a result to report, not to hide.
2. **Real-text set is small and skewed** towards gated communities and commercial buildings that mappers bothered to tag.
3. **OSM coverage is uneven.** Only about 1% of buildings have a name and 8% a named street nearby, so many formal addresses are locality-level at best (see the oracle row).
4. **Pincode truth is an estimate** unless the boundary file loads.
5. **Cold cells share borders with training cells.** Use `nn_train_m` (or the "cold interior" slice) for the strict reading.
6. **One city.** Nothing here shows the findings transfer to other cities.
7. **Outside geocoders are scored on a subset** (about 2,500 rows), on one date; they change over time.

## Outside geocoders: what is and is not stored

Per their terms: only error distances and yes/no flags are saved and published. Their coordinates are not stored in this repo, not drawn on the explorer map, and not used to train, tune or validate any model. The date, endpoint and parameters of each run are recorded with the scores.

## Licences

OSM-derived data: ODbL 1.0, © OpenStreetMap contributors. Overture Places: per-source open licences as published by Overture.
