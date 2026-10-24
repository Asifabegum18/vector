import numpy as np

from src.baselines import exact_search
from src.search import load_products, search_products


def test_ivf_recall_is_one_when_all_lists_are_probed():
    products = load_products()
    query = np.asarray(products[0]["embedding"])
    exact = exact_search(products, query, 5, pincode="625001")
    approximate = search_products(products, query, 5, mode="ivf", nlist=4, nprobe=4, pincode="625001")
    assert [item["product_id"] for item in approximate] == [item["product_id"] for item in exact]