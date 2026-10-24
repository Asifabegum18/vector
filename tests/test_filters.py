import pytest

from src.filters import filter_products


@pytest.fixture
def catalog():
    return [
        {"pincode": "625001", "category": "home", "price": 100},
        {"pincode": "625002", "category": "home", "price": 200},
        {"pincode": "625001", "category": "kitchen", "price": 300},
    ]


def test_combines_pincode_category_and_price_filters(catalog):
    result = filter_products(catalog, pincode="625001", category="home", max_price=150)
    assert result == [catalog[0]]


def test_rejects_inverted_price_range(catalog):
    with pytest.raises(ValueError, match="min_price"):
        filter_products(catalog, min_price=300, max_price=100)