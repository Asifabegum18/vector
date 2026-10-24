
"""Metadata-based selectivity estimation for product search."""

from bisect import bisect_left, bisect_right

import numpy as np


class MetadataSelectivityEstimator:
    """Estimate eligible counts using precomputed category-zone price groups."""

    def __init__(self, products):
        required = {"category", "zone", "price_inr"}
        missing = required.difference(products.columns)

        if missing:
            raise ValueError(
                f"Missing required metadata columns: {sorted(missing)}"
            )

        self.total_products = len(products)
        self.groups = {}

        for (category, zone), group in products.groupby(
            ["category", "zone"], dropna=False, sort=False
        ):
            prices = np.sort(
                group["price_inr"].to_numpy(dtype=float)
            )
            self.groups[(category, zone)] = prices

    def estimate_count(
        self,
        *,
        category=None,
        zone=None,
        min_price=None,
        max_price=None,
    ):
        """Count eligible products without scanning individual product rows."""

        if (
            min_price is not None
            and max_price is not None
            and min_price > max_price
        ):
            raise ValueError("min_price cannot exceed max_price")

        count = 0

        for (group_category, group_zone), prices in self.groups.items():
            if category is not None and group_category != category:
                continue

            if zone is not None and group_zone != zone:
                continue

            left = (
                0
                if min_price is None
                else bisect_left(prices, float(min_price))
            )

            right = (
                len(prices)
                if max_price is None
                else bisect_right(prices, float(max_price))
            )

            count += max(0, right - left)

        return count

    def estimate_selectivity(self, **filters):
        """Return the fraction of products expected to match the filters."""

        if self.total_products == 0:
            return 0.0

        return self.estimate_count(**filters) / self.total_products
