
from src.statistics import MetadataStatistics


def sample_products():
    return [
        {
            "product_id": "1",
            "category": "home",
            "pincode": "625001",
            "price": 500,
            "active": True,
        },
        {
            "product_id": "2",
            "category": "home",
            "pincode": "625001",
            "price": 1500,
            "active": True,
        },
        {
            "product_id": "3",
            "category": "electronics",
            "pincode": "625002",
            "price": 2500,
            "active": True,
        },
        {
            "product_id": "4",
            "category": "home",
            "pincode": "625001",
            "price": 700,
            "active": False,
        },
    ]


def test_count_by_pincode_and_category():
    stats = MetadataStatistics(sample_products())

    assert stats.estimate_count(
        category="home",
        pincode="625001",
    ) == 2


def test_count_by_price_range():
    stats = MetadataStatistics(sample_products())

    assert stats.estimate_count(
        min_price=400,
        max_price=1000,
        active=None,
    ) == 2


def test_active_filter():
    stats = MetadataStatistics(sample_products())

    assert stats.estimate_count(
        category="home",
        pincode="625001",
        active=True,
    ) == 2


def test_selectivity():
    stats = MetadataStatistics(sample_products())

    result = stats.estimate_selectivity(
        category="home",
        pincode="625001",
    )

    assert result == 0.5


def test_empty_catalogue():
    stats = MetadataStatistics([])

    assert stats.estimate_count() == 0
    assert stats.estimate_selectivity() == 0.0
