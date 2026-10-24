
"""Reproducible catalogue mutation test for Madurai Pincode Search."""

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.metadata_estimator import MetadataSelectivityEstimator
from src.ivf_index import IVFIndex
from src.planner import choose_strategy

from benchmarks.run_large_benchmark import (
    load_dataset,
    make_mask,
    top_exact_ids,
    top_ivf_ids,
    recall_at_k,
)


SEED = 42
EXACT_THRESHOLD = 10_000
TOP_K = 10


def mutate_dataset(products, vectors):
    """Delete 10% of the busiest zone and add 5% shifted products."""
    rng = np.random.default_rng(SEED)

    if len(products) != len(vectors):
        raise ValueError(
            "Product count must match the number of vectors."
        )

    if len(products) == 0:
        raise ValueError("Cannot mutate an empty catalogue.")

    # Find the busiest zone dynamically.
    busiest_zone = products["zone"].value_counts().idxmax()

    zone_rows = np.flatnonzero(
        products["zone"].to_numpy() == busiest_zone
    )

    remove_count = int(round(0.10 * len(zone_rows)))

    if remove_count > 0:
        removed_rows = rng.choice(
            zone_rows,
            size=remove_count,
            replace=False,
        )
    else:
        removed_rows = np.array([], dtype=int)

    keep = np.ones(len(products), dtype=bool)
    keep[removed_rows] = False

    kept_products = products.loc[keep].copy().reset_index(drop=True)
    kept_vectors = vectors[keep].copy()

    # Add 5% of the original catalogue size.
    added_count = max(1, int(round(0.05 * len(products))))

    categories = rng.choice(
        ["electronics", "skincare", "jewellery"],
        size=added_count,
        p=[0.70, 0.20, 0.10],
    )

    zones = rng.choice(
        ["Salem", "Madurai", "Tirunelveli"],
        size=added_count,
        p=[0.50, 0.30, 0.20],
    )

    prices = np.round(
        rng.uniform(5000, 30000, size=added_count),
        2,
    )

    # Reuse source vectors and shift their distribution.
    source_rows = rng.integers(
        0,
        len(vectors),
        size=added_count,
    )

    added_vectors = vectors[source_rows].copy()
    added_vectors[:, :min(8, added_vectors.shape[1])] += 2.5

    added_vectors += rng.normal(
        0,
        0.10,
        size=added_vectors.shape,
    ).astype(np.float32)

    added_products = pd.DataFrame({
        "product_id": np.arange(
            len(kept_products),
            len(kept_products) + added_count,
        ),
        "title": [
            f"mutation test shifted product {i:05d}"
            for i in range(added_count)
        ],
        "category": categories,
        "zone": zones,
        "price_inr": prices,
        "active": True,
    })

    mutated_products = pd.concat(
        [kept_products, added_products],
        ignore_index=True,
    )

    mutated_vectors = np.concatenate(
        [kept_vectors, added_vectors],
        axis=0,
    ).astype(np.float32, copy=False)

    # Keep vector positions and product IDs aligned.
    mutated_products["product_id"] = np.arange(
        len(mutated_products)
    )

    details = {
        "busiest_zone": busiest_zone,
        "original_zone_count": len(zone_rows),
        "removed_count": remove_count,
        "added_count": added_count,
        "original_count": len(products),
        "mutated_count": len(mutated_products),
    }

    return mutated_products, mutated_vectors, details


def benchmark_dataset(
    name,
    products,
    vectors,
    queries,
    nlist,
    nprobe,
):
    """Compare exact, IVF, and adaptive search for one catalogue."""

    print(f"\n{'=' * 72}")
    print(f"DATASET: {name}")
    print(f"Products: {len(products):,}")

    if len(products) != len(vectors):
        raise ValueError(
            f"{name}: product count does not match vector count."
        )

    # Rebuild the index for this catalogue.
    index = IVFIndex(n_partitions=nlist)

    start = time.perf_counter()
    index.fit(
        vectors,
        product_ids=np.arange(len(vectors)),
    )
    build_seconds = time.perf_counter() - start

    # Rebuild the estimator for this catalogue too.
    estimator = MetadataSelectivityEstimator(products)

    print(f"Index build: {build_seconds:.3f} seconds")
    print("Metadata estimator: rebuilt for current catalogue")

    # Choose cases based on available catalogue metadata.
    busiest_zone = products["zone"].value_counts().idxmax()

    filters_to_test = [
        ("all_products", {}),
        ("busiest_zone", {"zone": busiest_zone}),
        (
            "saree_in_erode",
            {"category": "saree", "zone": "Erode"},
        ),
        (
            "electronics_in_trichy",
            {"category": "electronics", "zone": "Trichy"},
        ),
        (
            "electronics_in_salem",
            {"category": "electronics", "zone": "Salem"},
        ),
    ]

    rng = np.random.default_rng(SEED)
    records = []

    for case_name, filters in filters_to_test:
        # Actual filtered products, used for evaluation and ground truth.
        mask = make_mask(products, filters)
        eligible_ids = np.flatnonzero(mask)
        actual_count = len(eligible_ids)

        # Metadata-based count used by the planner.
        estimated_count = estimator.estimate_count(**filters)

        # Verify estimator correctness for the supported filters.
        if estimated_count != actual_count:
            raise AssertionError(
                f"{name}/{case_name}: metadata estimator returned "
                f"{estimated_count}, but actual count is {actual_count}. "
                f"Filters: {filters}"
            )

        selectivity = (
            estimated_count / estimator.total_products
            if estimator.total_products > 0
            else 0.0
        )

        if actual_count == 0:
            print(
                f"\n{case_name}: no eligible products; skipping."
            )
            continue

        # Planner decision uses the estimator, not the filtered row count.
        strategy = choose_strategy(
            total_products=estimator.total_products,
            estimated_count=estimated_count,
            exact_threshold=EXACT_THRESHOLD,
        )

        print(
            f"\n{case_name}:"
            f"\n  Filters: {filters}"
            f"\n  Estimated count: {estimated_count:,}"
            f"\n  Actual count: {actual_count:,}"
            f"\n  Selectivity: {selectivity:.3%}"
            f"\n  Adaptive strategy: {strategy}"
        )

        for query_number in range(queries):
            query_id = int(rng.choice(eligible_ids))
            query = vectors[query_id]

            # Exact search provides the ground truth.
            start = time.perf_counter()
            truth = top_exact_ids(
                vectors,
                query,
                eligible_ids,
                TOP_K,
            )
            exact_ms = (time.perf_counter() - start) * 1000

            # IVF search followed by the same metadata filter.
            start = time.perf_counter()
            ivf = top_ivf_ids(
                index,
                vectors,
                query,
                mask,
                TOP_K,
                nprobe,
            )
            ivf_ms = (time.perf_counter() - start) * 1000

            if strategy == "exact":
                adaptive_ids = truth
                adaptive_ms = exact_ms
            else:
                adaptive_ids = ivf
                adaptive_ms = ivf_ms

            modes = [
                ("exact", truth, exact_ms),
                ("ivf", ivf, ivf_ms),
                ("adaptive", adaptive_ids, adaptive_ms),
            ]

            for mode, found, latency in modes:
                records.append({
                    "dataset": name,
                    "case": case_name,
                    "filters": str(filters),
                    "estimated_count": estimated_count,
                    "actual_count": actual_count,
                    "eligible_count": actual_count,
                    "selectivity": selectivity,
                    "mode": mode,
                    "recall_at_10": recall_at_k(found, truth),
                    "latency_ms": latency,
                    "adaptive_strategy": strategy,
                    "query_number": query_number,
                    "index_build_seconds": build_seconds,
                })

    result = pd.DataFrame(records)

    if result.empty:
        raise RuntimeError(
            f"No benchmark results were generated for {name}."
        )

    summary = (
        result.groupby(["case", "mode"])
        .agg(
            estimated_count=("estimated_count", "first"),
            actual_count=("actual_count", "first"),
            selectivity=("selectivity", "first"),
            recall_at_10=("recall_at_10", "mean"),
            p50_ms=("latency_ms", "median"),
            p95_ms=(
                "latency_ms",
                lambda values: values.quantile(0.95),
            ),
        )
        .reset_index()
    )

    print("\nSUMMARY")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark exact, IVF, and adaptive search before "
            "and after a reproducible catalogue mutation."
        )
    )

    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--nlist", type=int, default=128)
    parser.add_argument("--nprobe", type=int, default=8)

    args = parser.parse_args()

    if args.queries < 1:
        parser.error("--queries must be positive")

    if args.nlist < 1:
        parser.error("--nlist must be at least 1")

    if not 1 <= args.nprobe <= args.nlist:
        parser.error("--nprobe must be between 1 and nlist")

    # Load the reproducible original catalogue.
    products, vectors = load_dataset()

    # Create a deterministic mutated catalogue.
    mutated_products, mutated_vectors, details = mutate_dataset(
        products,
        vectors,
    )

    print("MUTATION DETAILS")
    for key, value in details.items():
        print(f"{key}: {value}")

    # Benchmark both catalogues independently.
    original_results = benchmark_dataset(
        "original",
        products,
        vectors,
        args.queries,
        args.nlist,
        args.nprobe,
    )

    mutated_results = benchmark_dataset(
        "mutated",
        mutated_products,
        mutated_vectors,
        args.queries,
        args.nlist,
        args.nprobe,
    )

    output_dir = "benchmarks/results"
    os.makedirs(output_dir, exist_ok=True)

    original_path = os.path.join(
        output_dir,
        "mutation_original.csv",
    )
    mutated_path = os.path.join(
        output_dir,
        "mutation_mutated.csv",
    )
    comparison_path = os.path.join(
        output_dir,
        "mutation_comparison.csv",
    )

    original_results.to_csv(original_path, index=False)
    mutated_results.to_csv(mutated_path, index=False)

    combined = pd.concat(
        [original_results, mutated_results],
        ignore_index=True,
    )
    combined.to_csv(comparison_path, index=False)

    print("\nSaved benchmark results:")
    print(original_path)
    print(mutated_path)
    print(comparison_path)


if __name__ == "__main__":
    main()
