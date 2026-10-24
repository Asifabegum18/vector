
"""Metadata statistics for selectivity-aware product search."""

from bisect import bisect_left, bisect_right
from collections import defaultdict


class MetadataStatistics:
    """Estimate matching product counts using grouped price statistics."""

    def __init__(self, products):
        self.total_products = len(products)
        self.groups = defaultdict(list)

        for product in products:
            category = product.get("category")
            pincode = product.get("pincode", product.get("zone"))
            active = product.get("active", True)
            price = float(product.get("price", 0))

            key = (category, pincode, active)
            self.groups[key].append(price)

        # Sort each group's prices once for efficient range counting.
        for key in self.groups:
            self.groups[key].sort()

    def estimate_count(
        self,
        *,
        category=None,
        pincode=None,
        zone=None,
        min_price=None,
        max_price=None,
        active=True,
    ):
        """Estimate the number of products matching the supplied filters."""

        if pincode is None:
            pincode = zone

        lower = float("-inf") if min_price is None else float(min_price)
        upper = float("inf") if max_price is None else float(max_price)

        if lower > upper:
            return 0

        count = 0

        # Iterate over metadata groups, not individual products.
        for (group_category, group_pincode, group_active), prices in self.groups.items():
            if category is not None and group_category != category:
                continue

            if pincode is not None and group_pincode != pincode:
                continue

            if active is not None and group_active != active:
                continue

            first = bisect_left(prices, lower)
            last = bisect_right(prices, upper)
            count += last - first

        return count

    def estimate_selectivity(self, **filters):
        """Return the estimated matching fraction between 0 and 1."""
        if self.total_products == 0:
            return 0.0

        count = self.estimate_count(**filters)
        return count / self.total_products
