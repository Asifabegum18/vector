"""Metadata filtering shared by exact and approximate search."""

from collections.abc import Iterable, Mapping


def filter_products(
    products: Iterable[Mapping],
    pincode: str | None = None,
    category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
) -> list[Mapping]:
    """Return products satisfying all supplied metadata constraints."""
    if min_price is not None and max_price is not None and min_price > max_price:
        raise ValueError("min_price cannot be greater than max_price")
    result = []
    for product in products:
        if pincode is not None and str(product["pincode"]) != str(pincode):
            continue
        if category is not None and product["category"] != category:
            continue
        price = float(product["price"])
        if min_price is not None and price < min_price:
            continue
        if max_price is not None and price > max_price:
            continue
        result.append(product)
    return result