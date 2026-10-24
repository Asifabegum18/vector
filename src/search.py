
"""Command-line entry point and reusable product search functions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.baselines import exact_search
from src.distance import cosine_distance
from src.filters import filter_products
from src.ivf_index import IVFIndex
from src.planner import choose_nprobe, choose_strategy
from src.statistics import MetadataStatistics

DEFAULT_DATA = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "sample_products.csv"
)


def load_products(path=DEFAULT_DATA):
    """Load product metadata and convert price and embedding fields."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        products = list(csv.DictReader(handle))

    for product in products:
        product["price"] = float(product["price"])
        product["embedding"] = json.loads(product["embedding"])

    return products


def search_products(
    products,
    query,
    k=5,
    *,
    mode="exact",
    nlist=4,
    nprobe=None,
    **filters,
):
    """Search products with exact, IVF, or adaptive retrieval."""
    if k < 1:
        raise ValueError("k must be at least 1")

    if not products:
        return []

    query = np.asarray(query, dtype=float)

    if query.ndim != 1:
        raise ValueError("query must be a one-dimensional vector")

    if mode not in ("exact", "ivf", "adaptive"):
        raise ValueError("mode must be 'exact', 'ivf', or 'adaptive'")

    # Exact mode doesn't need to build an IVF index.
    if mode == "exact":
        return exact_search(products, query, k, **filters)

    if nlist < 1:
        raise ValueError("nlist must be at least 1")

    vectors = np.asarray(
        [item["embedding"] for item in products],
        dtype=float,
    )

    if vectors.ndim != 2 or vectors.shape[1] != query.shape[0]:
        raise ValueError(
            "Query and product embeddings must have equal dimensions"
        )

    # Apply metadata filters.
    eligible = filter_products(products, **filters)

    if not eligible:
        return []

    # Adaptive mode estimates the eligible count and chooses a strategy.
    if mode == "adaptive":
        stats = MetadataStatistics(products)
        estimated_count = stats.estimate_count(**filters)

        strategy = choose_strategy(
            total_products=len(products),
            estimated_count=estimated_count,
        )

        print(
            f"Planner selected: {strategy}; "
            f"estimated eligible products: {estimated_count}"
        )

        if strategy == "exact":
            return exact_search(products, query, k, **filters)

    selectivity = len(eligible) / len(products)

    if nprobe is None:
        probes = choose_nprobe(
            len(products),
            nlist,
            selectivity=selectivity,
        )
    else:
        probes = nprobe

    if not 1 <= probes <= nlist:
        raise ValueError("nprobe must be between 1 and nlist")

    # Build the IVF index using its existing API.
    index = IVFIndex(n_partitions=nlist)
    index.fit(vectors)

    search_result = index.search(
        query,
        k=len(products),
        nprobe=probes,
    )

    # SearchResult.product_ids contains indices assigned by fit().
    candidate_indices = search_result.product_ids
    eligible_ids = {item["product_id"] for item in eligible}

    candidate_products = [
        products[int(candidate_index)]
        for candidate_index in candidate_indices
        if 0 <= int(candidate_index) < len(products)
    ]

    filtered = [
        item
        for item in candidate_products
        if item["product_id"] in eligible_ids
    ]

    if not filtered:
        return []

    filtered_vectors = np.asarray(
        [item["embedding"] for item in filtered],
        dtype=float,
    )

    distances = cosine_distance(query, filtered_vectors)
    order = np.argsort(distances, kind="stable")[:k]

    return [
        {
            **filtered[int(position)],
            "distance": float(distances[int(position)]),
        }
        for position in order
    ]


def main():
    """Run product search from the command line."""
    parser = argparse.ArgumentParser(
        description="Search synthetic Madurai catalogue products"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--query-pincode", default="625001")
    parser.add_argument("--category")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=("exact", "ivf", "adaptive"),
        default="exact",
    )
    parser.add_argument("--nlist", type=int, default=4)
    parser.add_argument("--nprobe", type=int)

    args = parser.parse_args()
    products = load_products(args.data)

    if not products:
        print("No products found.")
        return

    query = np.asarray(products[0]["embedding"], dtype=float)

    filters = {"pincode": args.query_pincode}
    if args.category is not None:
        filters["category"] = args.category

    results = search_products(
        products,
        query,
        args.top_k,
        mode=args.mode,
        nlist=args.nlist,
        nprobe=args.nprobe,
        **filters,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
