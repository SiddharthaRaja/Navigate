# Navigate

An open benchmark and model suite for geocoding messy Indian addresses, starting with Hyderabad.

- **P1 · Benchmark:** urban, building-level test sets with labelled noise and relation slices, warm and cold-area splits, scored against commercial and open geocoders.
- **P2 · Models:** three model families on identical data: H3 cell classifier, address–location dual encoder, and a small LLM trained with SFT then GRPO.
- **P3 · Pin checker:** detects when a customer's GPS pin does not match the typed address, and corrects it.

## Layout

```
notebooks/   numbered Colab/Kaggle notebooks (see naming below)
bench/       benchmark code: generator, relation labeller, splits
models/      model code for P2 and P3
eval/        evaluation harness and metrics
docs/        notes, figures, dataset card
data/        local scratch only; real data lives on Google Drive (not committed)
```

## Notebook naming

`NN_task_device.ipynb`, where `NN` is the run order and `device` is `cpu` or `gpu`.

| # | Notebook | Device |
|---|---|---|
| 01 | `01_osm_gazetteer_cpu` | CPU |
| 02 | `02_overture_places_cpu` | CPU |
| 03 | `03_pincode_areas_cpu` | CPU |
| 04 | `04_relation_labeller_cpu` | CPU |
| 05 | `05_address_generator_cpu` | CPU |
| 06 | `06_splits_cpu` | CPU |
| 07 | `07_llm_paraphrase_gpu` | GPU |
| 08 | `08_baselines_apis_cpu` | CPU |
| 09 | `09_evaluate_cpu` | CPU |

Later notebooks (P2, P3) continue from 10.

## Data and licences

Map data © OpenStreetMap contributors, ODbL 1.0. Commercial geocoder outputs are never stored in this repo; only aggregate scores are published.
