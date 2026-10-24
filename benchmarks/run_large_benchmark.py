
"""Benchmark exact, IVF, and adaptive search on the full dataset."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.ivf_index import IVFIndex
from src.planner import choose_strategy
from src.metadata_estimator import MetadataSelectivityEstimator


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "generated"
OUTPUT = ROOT / "benchmarks" / "results" / "large_benchmark.csv"


def load_dataset():
    products = pd.read_csv(DATA_DIR / "products.csv")
    vectors = np.load(DATA_DIR / "vectors.npy")

    if len(products) != len(vectors):
        raise ValueError("Product count and vector count do not match")

    if not np.array_equal(
        products["product_id"].to_numpy(),
        np.arange(len(products)),
    ):
        raise ValueError("Expected product IDs from 0 to N-1")

    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Vectors must be a finite 2D array")

    return products, vectors.astype(np.float32, copy=False)


def make_mask(products, filters):
    """Build the actual eligibility mask for benchmark ground truth."""
    mask = np.ones(len(products), dtype=bool)

    if filters.get("category") is not None:
        mask &= products["category"].to_numpy() == filters["category"]

    if filters.get("zone") is not None:
        mask &= products["zone"].to_numpy() == filters["zone"]

    prices = products["price_inr"].to_numpy()

    if filters.get("min_price") is not None:
        mask &= prices >= filters["min_price"]

    if filters.get("max_price") is not None:
        mask &= prices <= filters["max_price"]

    return mask


def choose_filter_cases(products):
    """Find repeatable filter cases across different selectivities."""
    categories = products["category"].to_numpy()
    zones = products["zone"].to_numpy()
    prices = products["price_inr"].to_numpy()
    total = len(products)
    candidates = []

    def add(filters, count):
        if count > 0:
            candidates.append((filters, int(count)))

    add({}, total)

    for category in sorted(products["category"].unique()):
        count = int(np.count_nonzero(categories == category))
        add({"category": category}, count)

    for zone in sorted(products["zone"].unique()):
        count = int(np.count_nonzero(zones == zone))
        add({"zone": zone}, count)

    # This exhaustive work is only for choosing benchmark cases.
    # The planner does not use these precomputed counts.
    for category in sorted(products["category"].unique()):
        for zone in sorted(products["zone"].unique()):
            base = (categories == category) & (zones == zone)
            local_prices = prices[base]

            if len(local_prices) == 0:
                continue

            add(
                {"category": category, "zone": zone},
                len(local_prices),
            )

            for quantile in (0.1, 0.5, 0.9):
                center = float(np.quantile(local_prices, quantile))

                for width in (100, 25, 5, 1, 0.25, 0.05):
                    count = int(np.count_nonzero(
                        (local_prices >= center - width)
                        & (local_prices <= center + width)
                    ))

                    add({
                        "category": category,
                        "zone": zone,
                        "min_price": center - width,
                        "max_price": center + width,
                    }, count)

    ranges = {
        "around_10_percent": (0.05 * total, 0.20 * total),
        "around_1_percent": (0.005 * total, 0.02 * total),
        "around_0.1_percent": (0.0005 * total, 0.002 * total),
        "handful": (1, 10),
    }

    cases = [("all_products", {}, total)]

    for label, (low, high) in ranges.items():
        matching = [
            item for item in candidates
            if low <= item[1] <= high
        ]

        if not matching:
            target = (low + high) / 2
            matching = sorted(
                candidates,
                key=lambda item: abs(item[1] - target),
            )[:1]

        filters, count = min(
            matching,
            key=lambda item: abs(item[1] - (low + high) / 2),
        )
        cases.append((label, filters, count))

    unique = []
    seen = set()

    for label, filters, count in cases:
        key = tuple(sorted(filters.items()))

        if key not in seen:
            seen.add(key)
            unique.append((label, filters, count))

    return unique


def top_exact_ids(vectors, query, eligible_ids, k):
    """Return exact top-k IDs using squared Euclidean distance."""
    if len(eligible_ids) == 0:
        return np.array([], dtype=int)

    differences = vectors[eligible_ids] - query
    distances = np.einsum("ij,ij->i", differences, differences)
    order = np.argsort(distances, kind="stable")[:k]

    return eligible_ids[order]


def top_ivf_ids(index, vectors, query, mask, k, nprobe):
    """Search IVF candidates, apply filters, and rerank by squared L2."""
    result = index.search(
        query,
        k=len(vectors),
        nprobe=nprobe,
    )

    candidate_ids = np.asarray(result.product_ids, dtype=int)
    candidate_ids = candidate_ids[
        (candidate_ids >= 0) & (candidate_ids < len(vectors))
    ]

    candidate_ids = candidate_ids[mask[candidate_ids]]

    if len(candidate_ids) == 0:
        return np.array([], dtype=int)

    differences = vectors[candidate_ids] - query
    distances = np.einsum("ij,ij->i", differences, differences)
    order = np.argsort(distances, kind="stable")[:k]

    return candidate_ids[order]


def recall_at_k(found, truth):
    if len(truth) == 0:
        return 1.0 if len(found) == 0 else 0.0

    return (
        len(set(found.tolist()) & set(truth.tolist()))
        / len(truth)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--nlist", type=int, default=128)
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.queries < 1 or args.top_k < 1:
        parser.error("--queries and --top-k must be positive")

    if args.nlist < 1 or not 1 <= args.nprobe <= args.nlist:
        parser.error("Require nlist >= 1 and 1 <= nprobe <= nlist")

    products, vectors = load_dataset()

    print(f"Products: {len(products):,}")
    print(f"Vector dimensions: {vectors.shape[1]}")
    print(f"Categories: {products['category'].nunique()}")
    print(f"Zones: {products['zone'].nunique()}")

    # Build metadata statistics once for the current catalogue.
    estimator = MetadataSelectivityEstimator(products)

    print("\nBuilding IVF index once...")
    index = IVFIndex(n_partitions=args.nlist)

    build_start = time.perf_counter()
    index.fit(
        vectors,
        product_ids=products["product_id"].to_numpy(),
    )
    build_seconds = time.perf_counter() - build_start

    print(f"Index build time: {build_seconds:.3f} seconds")

    cases = choose_filter_cases(products)
    rng = np.random.default_rng(42)
    records = []

    for label, filters, _setup_count in cases:
        # The actual mask is only used for evaluation and ground truth.
        mask = make_mask(products, filters)
        eligible_ids = np.flatnonzero(mask)
        actual_count = len(eligible_ids)

        if actual_count == 0:
            continue

        # Planner input comes from the metadata estimator.
        estimated_count = estimator.estimate_count(**filters)

        # This dataset's estimator is exact for these supported predicates.
        # Keep this assertion to catch stale or incorrect statistics.
        if estimated_count != actual_count:
            raise AssertionError(
                f"Estimator mismatch for {filters}: "
                f"estimated={estimated_count}, actual={actual_count}"
            )

        strategy = choose_strategy(
            total_products=estimator.total_products,
            estimated_count=estimated_count,
        )

        print(
            f"\n{label}: {actual_count:,} eligible "
            f"({100 * actual_count / len(products):.3f}%)"
            f" | estimated={estimated_count}"
            f" | strategy={strategy}"
            f" | filters={filters}"
        )

        for query_number in range(args.queries):
            query_id = int(rng.choice(eligible_ids))
            query = vectors[query_id]

            start = time.perf_counter()
            truth = top_exact_ids(
                vectors, query, eligible_ids, args.top_k
            )
            exact_ms = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            ivf_ids = top_ivf_ids(
                index, vectors, query, mask, args.top_k, args.nprobe
            )
            ivf_ms = (time.perf_counter() - start) * 1000

            if strategy == "exact":
                adaptive_ids, adaptive_ms = truth, exact_ms
            else:
                adaptive_ids, adaptive_ms = ivf_ids, ivf_ms

            for mode, found, elapsed in (
                ("exact", truth, exact_ms),
                ("ivf", ivf_ids, ivf_ms),
                ("adaptive", adaptive_ids, adaptive_ms),
            ):
                records.append({
                    "bucket": label,
                    "filters": repr(filters),
                    "eligible_count": actual_count,
                    "estimated_count": estimated_count,
                    "selectivity": actual_count / len(products),
                    "mode": mode,
                    "query_number": query_number,
                    "recall_at_k": recall_at_k(found, truth),
                    "query_latency_ms": elapsed,
                    "planner_strategy": (
                        strategy if mode == "adaptive" else mode
                    ),
                    "index_build_seconds": build_seconds,
                    "nlist": args.nlist,
                    "nprobe": args.nprobe,
                    "top_k": args.top_k,
                })

    if not records:
        raise RuntimeError("No benchmark records were produced")

    results = pd.DataFrame(records)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    summary = results.groupby(["bucket", "mode"]).agg(
        eligible_count=("eligible_count", "first"),
        estimated_count=("estimated_count", "first"),
        selectivity=("selectivity", "first"),
        recall_at_k=("recall_at_k", "mean"),
        p50_ms=("query_latency_ms", "median"),
        p95_ms=("query_latency_ms", lambda values: values.quantile(0.95)),
    ).reset_index()

    print("\nBENCHMARK SUMMARY")
    print(summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))
    print(f"\nDetailed results saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
