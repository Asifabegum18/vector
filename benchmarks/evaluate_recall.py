"""Evaluate IVF top-k overlap against exact filtered search."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.baselines import exact_search
from src.search import load_products, search_products


def recall_at_k(expected, actual, k):
    expected_ids = {item["product_id"] for item in expected[:k]}
    if not expected_ids:
        return 1.0
    actual_ids = {item["product_id"] for item in actual[:k]}
    return len(expected_ids & actual_ids) / len(expected_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sample_products.csv"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    products = load_products(args.data)
    query = np.asarray(products[0]["embedding"])
    expected = exact_search(products, query, args.top_k, pincode="625001")
    actual = search_products(products, query, args.top_k, mode="ivf", pincode="625001")
    print(json.dumps({"recall_at_k": recall_at_k(expected, actual, args.top_k), "exact_count": len(expected), "ivf_count": len(actual)}, indent=2))


if __name__ == "__main__":
    main()