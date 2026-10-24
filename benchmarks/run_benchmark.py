
"""Benchmark exact, IVF, and adaptive product search."""

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

from src.search import load_products, search_products


ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "benchmarks" / "results" / "benchmark_results.csv"


def percentile(values, percent):
    """Calculate a percentile without requiring an extra dependency."""
    return float(np.percentile(values, percent))


def recall_at_k(expected, actual):
    """Calculate recall against exact-search product IDs."""
    expected_ids = {item["product_id"] for item in expected}

    if not expected_ids:
        return 1.0 if not actual else 0.0

    actual_ids = {item["product_id"] for item in actual}
    return len(expected_ids & actual_ids) / len(expected_ids)


def run_benchmark(data_path, n_queries=10, nlist=4):
    products = load_products(data_path)

    if not products:
        raise ValueError("The dataset contains no products")

    query_count = min(n_queries, len(products))
    modes = ("exact", "ivf", "adaptive")
    measurements = {
        mode: {"latency_ms": [], "recall": []}
        for mode in modes
    }

    print(f"Products: {len(products)}")
    print(f"Queries: {query_count}")
    print(f"IVF partitions: {nlist}")
    print("Running benchmark...")

    for product in products[:query_count]:
        query = np.asarray(product["embedding"], dtype=float)
        filters = {"pincode": product["pincode"]}

        # Exact search provides the reference results for this query.
        expected = search_products(
            products, query, 10, mode="exact", **filters
        )

        for mode in modes:
            start = time.perf_counter()

            results = search_products(
                products,
                query,
                10,
                mode=mode,
                nlist=nlist,
                nprobe=nlist if mode == "ivf" else None,
                **filters,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000

            measurements[mode]["latency_ms"].append(elapsed_ms)
            measurements[mode]["recall"].append(
                recall_at_k(expected, results)
            )

    rows = []

    for mode in modes:
        latencies = measurements[mode]["latency_ms"]
        recalls = measurements[mode]["recall"]

        row = {
            "mode": mode,
            "queries": query_count,
            "mean_recall": round(statistics.mean(recalls), 4),
            "p50_latency_ms": round(percentile(latencies, 50), 3),
            "p95_latency_ms": round(percentile(latencies, 95), 3),
        }
        rows.append(row)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nBenchmark results")
    print("-" * 76)
    print(
        f"{'Mode':<12}{'Recall@10':>12}"
        f"{'p50 (ms)':>16}{'p95 (ms)':>16}"
    )

    for row in rows:
        print(
            f"{row['mode']:<12}"
            f"{row['mean_recall']:>12.4f}"
            f"{row['p50_latency_ms']:>16.3f}"
            f"{row['p95_latency_ms']:>16.3f}"
        )

    print(f"\nSaved results to: {RESULTS_FILE}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Madurai Pincode Search"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "sample_products.csv",
    )
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--nlist", type=int, default=4)
    args = parser.parse_args()

    if args.queries < 1:
        parser.error("--queries must be positive")

    if args.nlist < 1:
        parser.error("--nlist must be positive")

    run_benchmark(args.data, args.queries, args.nlist)


if __name__ == "__main__":
    main()
