
import pandas as pd
import pytest

from src.metadata_estimator import MetadataSelectivityEstimator


@pytest.fixture
def products():
    return pd.DataFrame({
        "category": [
            "electronics", "electronics", "electronics",
            "saree", "saree",
        ],
        "zone": ["Salem", "Salem", "Trichy", "Erode", "Erode"],
        "price_inr": [100.0, 200.0, 300.0, 500.0, 700.0],
    })


def test_count_all_products(products):
    estimator = MetadataSelectivityEstimator(products)
    assert estimator.estimate_count() == 5


def test_category_and_zone(products):
    estimator = MetadataSelectivityEstimator(products)
    assert estimator.estimate_count(
        category="electronics", zone="Salem"
    ) == 2


def test_inclusive_price_range(products):
    estimator = MetadataSelectivityEstimator(products)
    assert estimator.estimate_count(
        min_price=200, max_price=500
    ) == 3


def test_combined_filters(products):
    estimator = MetadataSelectivityEstimator(products)
    assert estimator.estimate_count(
        category="saree",
        zone="Erode",
        min_price=600,
        max_price=800,
    ) == 1


def test_selectivity(products):
    estimator = MetadataSelectivityEstimator(products)
    assert estimator.estimate_selectivity(
        zone="Salem"
    ) == pytest.approx(0.4)


def test_invalid_price_range(products):
    estimator = MetadataSelectivityEstimator(products)
    with pytest.raises(ValueError):
        estimator.estimate_count(min_price=500, max_price=100)


def test_rebuilt_estimator_reflects_new_products(products):
    original = MetadataSelectivityEstimator(products)
    assert original.estimate_count(zone="Salem") == 2

    updated = pd.concat([
        products,
        pd.DataFrame({
            "category": ["electronics"],
            "zone": ["Salem"],
            "price_inr": [900.0],
        }),
    ], ignore_index=True)

    rebuilt = MetadataSelectivityEstimator(updated)
    assert rebuilt.estimate_count(zone="Salem") == 3
