# Madurai Pincode Search

A small, reproducible example of filtered product-vector search for Madurai-area pincodes. It includes an exact baseline, a NumPy IVF-style index, a simple probe planner, recall benchmarks, and mutation-oriented checks.

## Quick start

```powershell
cd madurai-pincode-search
python -m pip install -r requirements.txt
python -m src.search --query-pincode 625001 --top-k 5
```

To generate a larger deterministic catalog:

```powershell
python -m src.generate_data --output data/products.csv --count 1000
```

Run tests with `python -m pytest`. Benchmark scripts are under `benchmarks/`; see their `--help` options for input and output paths.

## Search model

Each product has a pincode, category, price, and numeric embedding. Search applies metadata filters before ranking by cosine distance. Exact search scores every eligible item; IVF search scores items in the nearest selected clusters. The planner increases the probe count for selective filters and small collections.

The sample catalog uses synthetic embeddings and is intended for experimentation, not as a real product directory or geographic distance dataset.

## Layout

- `src/`: data generation, index, filtering, planning, ranking, and baselines
- `benchmarks/`: reproducible query generation, timing, and recall evaluation
- `experiments/`: small demonstrations of planner edge cases and mutations
- `tests/`: filtering, planner, recall, and mutation regression checks

See [the architecture diagram](docs/architecture.png).