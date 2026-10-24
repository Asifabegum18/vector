"""Create deterministic query vectors and filters for later benchmark phases."""
from typing import Iterable
import numpy as np
import pandas as pd

from src.schema import SearchQuery

def make_queries(
    products: pd.DataFrame,
    vectors: np.ndarray,
    seed: int = 2026,
    per_bucket: int = 20,
) -> list[SearchQuery]:
    """Generate query cases targeting five approximate selectivity buckets.

    The bucket is a target label only. Later evaluation must compute actual
    selectivity from matching counts and re-bin queries based on observed values.
    """
    if per_bucket < 1:
        raise ValueError("per_bucket must be >= 1")
    rng = np.random.default_rng(seed)
    vector_by_id = {int(pid): vectors[i] for i, pid in enumerate(products.product_id)}
    queries: list[SearchQuery] = []
    categories = sorted(products.category.unique())
    zones = sorted(products.zone.unique())
    # Queries are sampled from real catalogue vectors so ground truth is meaningful.
    for bucket in ("100%", "10%", "1%", "0.1%", "handful"):
        for j in range(per_bucket):
            source = products.iloc[int(rng.integers(0, len(products)))]
            qvec = vector_by_id[int(source.product_id)].copy()
            # Add controlled noise; harder queries are not always exact product vectors.
            qvec += rng.normal(0, 0.15, size=qvec.shape).astype(np.float32)
            category = None
            zone = None
            max_price = None
            if bucket == "100%":
                pass
            elif bucket == "10%":
                # Single common predicate: choose a category with roughly 10% share
                # where possible; actual bucket is determined from real counts.
                category = str(rng.choice(categories))
            elif bucket == "1%":
                category = str(rng.choice(categories))
                zone = str(rng.choice(zones))
            elif bucket == "0.1%":
                # Combine all three supported predicates to produce selective cases.
                category = str(rng.choice(categories))
                zone = str(rng.choice(zones))
                cat_rows = products[products.category == category]
                if len(cat_rows):
                    max_price = float(cat_rows.price_inr.quantile(float(rng.uniform(0.08, 0.35))))
            else:
                # Construct a narrow price ceiling within a category-zone pair.
                category = str(source.category)
                zone = str(source.zone)
                subset = products[(products.category == category) & (products.zone == zone)]
                if len(subset):
                    max_price = float(subset.price_inr.quantile(float(rng.uniform(0.02, 0.15))))
            queries.append(SearchQuery(
                query_id=len(queries),
                vector=qvec,
                category=category,
                zone=zone,
                max_price_inr=max_price,
                target_bucket=bucket,
            ))
    return queries
