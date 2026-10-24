"""Generate deterministic vector queries from catalog products."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.search import load_products


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sample_products.csv"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/queries.json"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    products = load_products(args.data)
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(len(products), min(args.count, len(products)), replace=False)
    queries = [{"query": products[int(i)]["embedding"], "pincode": products[int(i)]["pincode"]} for i in selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(queries, indent=2), encoding="utf-8")
    print(f"Wrote {len(queries)} queries to {args.output}")


if __name__ == "__main__":
    main()