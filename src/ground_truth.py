"""Exact filtered nearest-neighbour search used as benchmark ground truth."""
from typing import Iterable
import numpy as np

from src.schema import Product, SearchQuery

def exact_search(
    products: Iterable[Product],
    query: SearchQuery,
    k: int = 10,
) -> list[tuple[int, float]]:
    """Return [(product_id, squared_euclidean_distance), ...] sorted nearest first.

    This is deliberately exhaustive over every active product that matches all
    query predicates. It is a correctness baseline, not the optimized index.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    eligible = [p for p in products if query.matches(p)]
    if not eligible:
        return []
    q = np.asarray(query.vector, dtype=np.float32)
    matrix = np.stack([np.asarray(p.vector, dtype=np.float32) for p in eligible])
    if matrix.ndim != 2 or q.ndim != 1 or matrix.shape[1] != q.shape[0]:
        raise ValueError("query and product vectors must have the same dimension")
    # Squared L2 distance preserves nearest-neighbour ordering without sqrt.
    distances = np.sum((matrix - q) ** 2, axis=1)
    # Stable sorting makes ties reproducible; product ID breaks exact distance ties.
    order = np.lexsort((np.array([p.product_id for p in eligible]), distances))
    return [(eligible[int(i)].product_id, float(distances[int(i)]))
            for i in order[:k]]
