from experiments.mutation_test import mutated_filter
from src.filters import filter_products


def test_pincode_filter_regression_detects_wrong_comparison():
    catalog = [
        {"pincode": "625001", "price": 10},
        {"pincode": "625002", "price": 20},
    ]
    assert filter_products(catalog, pincode="625001") != mutated_filter(catalog, "625001")