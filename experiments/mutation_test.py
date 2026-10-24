"""Demonstrate that regression checks catch a pincode-filter mutation."""

from src.filters import filter_products


def mutated_filter(products, pincode):
    """Representative mutation: mistakenly accept every product."""
    return [product for product in products if str(product["pincode"]) != str(pincode)]


def main():
    catalog = [{"pincode": "625001"}, {"pincode": "625002"}]
    expected = filter_products(catalog, pincode="625001")
    mutated = mutated_filter(catalog, "625001")
    print({"correct_ids": [item["pincode"] for item in expected], "mutated_ids": [item["pincode"] for item in mutated], "mutation_detected": expected != mutated})


if __name__ == "__main__":
    main()