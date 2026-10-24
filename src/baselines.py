"""Exact and simple linear-scan baselines."""

import numpy as np

from src.distance import cosine_distance
from src.filters import filter_products


def exact_search(products, query, k, **filters):
    matches = filter_products(products, **filters)
    if not matches:
        return []
    distances = cosine_distance(query, np.asarray([item["embedding"] for item in matches]))
    order = np.argsort(distances)[:k]
    return [{**matches[index], "distance": float(distances[index])} for index in order]


def linear_scan(products, query, k, **filters):
    """Readable baseline alias; exact_search is the shared implementation."""
    return exact_search(products, query, k, **filters)