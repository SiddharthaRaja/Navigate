# Navigate

An open benchmark and model suite for geocoding messy Indian addresses, starting with Hyderabad.

Geocoding means turning the text of an address into a point on the map. Indian addresses make that hard: they lean on landmarks ("opp SBI, behind the temple"), mix languages, skip parts, and often carry the wrong pincode. Delivery and ride companies publish papers on the problem, but each tests on private data, so results cannot be compared. Navigate is a public test bed for it.

**Live error explorer:** https://siddhartharaja.github.io/Navigate/ (every test address on a map, coloured by how far each geocoder missed)

## What is here

| Part | What it is | State |
|---|---|---|
| P1 · Benchmark | Building-level test sets for Hyderabad with labelled noise, landmark relations, a cold-area split, and real address text. One scoring harness. Baselines and outside geocoders. | built; trial run below |
| P2 · Models | Three model families trained on identical data: H3 cell classifier, address–location dual encoder, small LLM that writes a location code. | code written and smoke-tested on CPU; GPU runs pending |
| P3 · Pin checker | Detects when a customer's GPS pin does not match the typed address, and corrects it. | not started |

## What makes the benchmark different

- **Building-level truth.** Every address is tied to one building footprint from OpenStreetMap, not to a village or pincode.
- **Warm and cold test sets.** *Warm* = unseen buildings in neighbourhoods the model trained on. *Cold* = whole H3 resolution-7 cells (about 5 sq km each) that are absent from training. Methods that memorise addresses collapse on cold.
- **Seven landmark relations**, labelled from map geometry using the vocabulary of Google's Address Descriptors: near, within, beside, across the road, behind, down the road, around the corner. A separate test set holds a fixed number of each.
- **Every kind of noise is labelled per row:** spelling slips, abbreviations, missing spaces, shuffled order, missing parts, Hindi or Telugu phrasing in Roman letters, shortened landmark names, wrong pincode (far or neighbouring), appended delivery instructions, and gibberish that should get no answer.
- **Hidden landmarks.** One in five landmarks is withheld from the public gazetteer but still appears in address text, the way people cite shops no map lists. This keeps the task from being a lookup exercise.
- **A real-text test set** (addresses typed by OpenStreetMap mappers; Overture business addresses) to check that results on generated text carry over.
- **An oracle reference row:** the distance from each building to the best place actually named in its address. It shows how much the text itself pins down.

Full detail, including what is invented and the known weaknesses: [docs/benchmark_card.md](docs/benchmark_card.md).

## Trial results

From a trial run on a 60,000-building sample (the full build uses 200,000), local baselines only. Median error in metres, with the share of addresses answered within 500 m in brackets.

| Method | Warm | Cold | Relations | Real text |
|---|---|---|---|---|
| Pincode centre | 3,408 (8%) | 3,313 (2%) | 3,743 (3%) | 2,112 (7%) |
| Fuzzy gazetteer rules (no learning) | 291 (58%) | 344 (53%) | 141 (70%) | 244 (47%) |
| Nearest training address (TF-IDF) | 240 (76%) | 4,070 (14%) | 4,228 (23%) | 2,209 (34%) |
| *Oracle reference (not a method)* | *169 (86%)* | *171 (87%)* | *68 (100%)* | – |

What the trial already shows:

- The method that learns from training addresses wins on warm data and fails on cold data (240 m against 4,070 m). The lookup method barely changes. A useful model has to do both.
- When the pincode in the text is wrong and far away, the rules method is thrown 15 km off; the learned one is not.
- When the landmark is hidden from the gazetteer, the rules method's median error roughly triples.

Trained models and outside geocoders (Nominatim, Photon, Google, Ola Maps, Mappls) are added to [eval/leaderboard.md](eval/leaderboard.md) as they are run.

## Layout

```
notebooks/   numbered Colab / Kaggle notebooks, run in order
bench/       benchmark code: map context, relations, splits, generator, noise, metrics, baselines, geocoder clients
models/      the three model families, one plain PyTorch file each
tests/       self-checks that run on a laptop CPU with no downloads
eval/        results tables
docs/        the error explorer (served by GitHub Pages) and the benchmark card
data/        local scratch only; real data lives on Google Drive and is never committed
```

## Notebooks

`NN_task_device.ipynb`: `NN` is the run order, `device` is `cpu` or `gpu`. Each one opens with a setup cell that fetches this repo, so the notebooks stay thin and the logic stays in `bench/` and `models/`.

| # | Notebook | What it does | Device |
|---|---|---|---|
| 01 | `01_osm_gazetteer_cpu` | OpenStreetMap extract for Hyderabad: buildings, landmarks, roads, areas | CPU |
| 02 | `02_overture_places_cpu` | Adds Overture Maps places and their address strings | CPU |
| 03 | `03_pincode_areas_cpu` | Pincode for every building (boundary file, or vote of nearby labelled points) | CPU |
| 04 | `04_relations_cpu` | Context for each target building and its relation to every landmark within 300 m | CPU |
| 05 | `05_splits_cpu` | Train / validation / warm / cold split, and which landmarks are hidden | CPU |
| 06 | `06_address_generator_cpu` | Writes the address text and the four test sets | CPU |
| 07 | `07_llm_paraphrase_gpu` | An LLM rewrites a share of addresses; a fact check rejects any rewrite that changes a fact | GPU |
| 08 | `08_baselines_apis_cpu` | Local baselines, open geocoders, paid geocoders (scores only) | CPU |
| 09 | `09_evaluate_cpu` | Scores everything, writes the leaderboard and the explorer data | CPU |
| 10 | `10_h3_classifier_gpu` | Model A: text encoder with H3 classification heads | GPU |
| 11 | `11_addressbind_gpu` | Model B: contrastive text–location dual encoder | GPU |
| 12 | `12_llm_sft_gpu` | Model C: small LLM fine-tuned to write a DIGIPIN code, with gazetteer retrieval | GPU |

## Run it

On Colab, open the notebooks in order from 01. Data goes to `MyDrive/Navigate/data`. Nothing needs a paid service; the paid geocoders in notebook 08 are skipped unless you add a key.

On a laptop:

```
pip install -r requirements.txt
python tests/check_bench.py                         # benchmark code, no data needed
python tests/check_models.py data/bench/hyderabad   # all three models end to end on tiny networks (after notebook 06)
python -m http.server --directory docs 8000         # the explorer, at http://localhost:8000
```

## Rules this project keeps

- API keys live in Colab or Kaggle secrets or environment variables, never in the repo.
- Coordinates returned by paid geocoders are never stored here, never drawn on the map, and never used to train or tune a model. Only error distances are published, together with everything needed to reproduce the run.
- No private addresses. All text is either generated from open map data or already public.

## Sources this builds on

- Flipkart, multi-resolution H3 classification for Indian address geocoding (COLING 2025)
- Amazon, AddressBind: contrastive address–location embeddings (SIGSPATIAL 2025); GeoGround noise types
- Alibaba Amap, ReaGeo: generative geocoding with a small LLM (2026)
- Google Maps Platform, Address Descriptors (the seven relations)
- Uber H3, India Post DIGIPIN

## Data and licences

Map data © OpenStreetMap contributors, ODbL 1.0. Overture Maps Places under its own open licences. Leaflet (BSD-2-Clause) is vendored in `docs/vendor/`.
