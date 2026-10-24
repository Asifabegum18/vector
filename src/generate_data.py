"""Generate a deterministic, correlated synthetic Indian e-commerce catalogue.

Example:
    python -m src.generate_data --count 60000 --seed 42 --out data/generated
Writes products.csv, vectors.npy, and metadata.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = [
    "saree", "kurta", "salwar_suit", "dress", "shirt", "trousers",
    "jeans", "tshirt", "footwear", "handbag", "jewellery", "watch",
    "home_decor", "bedsheet", "curtain", "kitchenware", "electronics",
    "books", "toys", "skincare",
]
ZONES = [
    "Madurai", "Chennai", "Coimbatore", "Trichy", "Salem", "Tirunelveli",
    "Erode", "Bengaluru", "Hyderabad", "Mumbai", "Pune", "Delhi",
    "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Kochi", "Mysuru",
    "Vijayawada", "Visakhapatnam", "Thanjavur", "Dindigul", "Karur",
    "Sivakasi", "Ramanathapuram", "Nagapattinam", "Nilgiris", "Kanniyakumari",
    "Andaman", "Lakshadweep",
]

# Unequal zone sizes. Zipf-like weights make a few zones much larger than others.
ZONE_WEIGHTS = np.array([
    0.15, 0.14, 0.10, 0.075, 0.06, 0.045, 0.04, 0.075, 0.055, 0.06,
    0.035, 0.035, 0.025, 0.02, 0.015, 0.012, 0.012, 0.008, 0.006, 0.006,
    0.004, 0.004, 0.003, 0.003, 0.002, 0.002, 0.0015, 0.001, 0.0005, 0.0005
], dtype=float)
ZONE_WEIGHTS /= ZONE_WEIGHTS.sum()

# Category mixture differs by zone; rows are constructed dynamically below.
BASE_CATEGORY_WEIGHTS = np.array([
    0.11, 0.09, 0.07, 0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0.04,
    0.04, 0.03, 0.04, 0.035, 0.025, 0.035, 0.025, 0.02, 0.02, 0.025
], dtype=float)
BASE_CATEGORY_WEIGHTS /= BASE_CATEGORY_WEIGHTS.sum()

# Price ranges intentionally overlap, but category medians differ.
CATEGORY_PRICE = {
    "saree": (350, 4500), "kurta": (250, 2200), "salwar_suit": (450, 3500),
    "dress": (400, 4000), "shirt": (250, 1800), "trousers": (350, 2400),
    "jeans": (500, 3000), "tshirt": (180, 1200), "footwear": (300, 4500),
    "handbag": (350, 5000), "jewellery": (100, 2500), "watch": (300, 8000),
    "home_decor": (150, 3500), "bedsheet": (250, 2500), "curtain": (250, 3000),
    "kitchenware": (100, 3000), "electronics": (500, 25000), "books": (80, 1500),
    "toys": (100, 2500), "skincare": (100, 2200),
}
CATEGORY_WORDS = {
    "saree": ["cotton saree", "silk saree", "printed saree", "daily wear saree"],
    "kurta": ["cotton kurta", "embroidered kurta", "casual kurta"],
    "salwar_suit": ["cotton salwar suit", "festive salwar suit"],
    "dress": ["casual dress", "printed dress", "party dress"],
    "shirt": ["formal shirt", "cotton shirt", "checked shirt"],
    "trousers": ["formal trousers", "cotton trousers"],
    "jeans": ["slim fit jeans", "straight fit jeans"],
    "tshirt": ["cotton t-shirt", "printed t-shirt"],
    "footwear": ["walking sandals", "casual shoes", "traditional footwear"],
    "handbag": ["everyday handbag", "shoulder bag"],
    "jewellery": ["silver earrings", "fashion necklace"],
    "watch": ["digital watch", "analog watch"],
    "home_decor": ["wall decor", "decorative vase"],
    "bedsheet": ["cotton bedsheet", "printed bedsheet"],
    "curtain": ["blackout curtain", "cotton curtain"],
    "kitchenware": ["steel storage container", "cooking pan"],
    "electronics": ["wireless earbuds", "phone charger"],
    "books": ["fiction book", "exam preparation book"],
    "toys": ["building blocks", "soft toy"],
    "skincare": ["moisturizer", "sunscreen lotion"],
}
ZONE_TITLE_WORDS = {
    "Madurai": ["Madurai delivery", "Tamil cotton", "southern style"],
    "Chennai": ["Chennai delivery", "urban collection"],
    "Coimbatore": ["Coimbatore delivery", "cotton collection"],
}

def _category_weights_for_zone(zone: str) -> np.ndarray:
    """Return zone-specific category probabilities, adding intentional correlation."""
    w = BASE_CATEGORY_WEIGHTS.copy()
    if zone in {"Madurai", "Chennai", "Coimbatore", "Trichy", "Salem",
                "Tirunelveli", "Erode", "Thanjavur", "Dindigul", "Karur",
                "Sivakasi", "Ramanathapuram", "Nagapattinam", "Kanniyakumari"}:
        w[0] *= 4.0   # sarees more common in Tamil Nadu zones
        w[1] *= 1.8   # kurtas also more common
    if zone in {"Mumbai", "Delhi", "Bengaluru", "Hyderabad"}:
        w[16] *= 2.2  # electronics more common in metro zones
        w[9] *= 1.5   # handbags
    return w / w.sum()

def generate_catalogue(count: int = 60000, seed: int = 42, dimensions: int = 64):
    if count < 100:
        raise ValueError("count must be at least 100 to make meaningful skewed zones")
    if dimensions < 8:
        raise ValueError("dimensions must be at least 8")
    rng = np.random.default_rng(seed)

    # Each category has a latent semantic centre. Zones add a smaller shift,
    # creating clusters and some category-zone correlation.
    category_centres = rng.normal(0, 3.0, size=(len(CATEGORIES), dimensions)).astype(np.float32)
    zone_centres = rng.normal(0, 0.45, size=(len(ZONES), dimensions)).astype(np.float32)

    zone_idx = rng.choice(len(ZONES), size=count, p=ZONE_WEIGHTS)
    cat_idx = np.empty(count, dtype=np.int32)
    for zi, zone in enumerate(ZONES):
        positions = np.flatnonzero(zone_idx == zi)
        if positions.size:
            cat_idx[positions] = rng.choice(
                len(CATEGORIES), size=positions.size,
                p=_category_weights_for_zone(zone)
            )

    vectors = np.empty((count, dimensions), dtype=np.float32)
    prices = np.empty(count, dtype=np.float32)
    titles = []
    for i in range(count):
        ci = int(cat_idx[i])
        zi = int(zone_idx[i])
        category = CATEGORIES[ci]
        low, high = CATEGORY_PRICE[category]

        # Log-uniform prices model many inexpensive products and fewer premium ones.
        prices[i] = np.exp(rng.uniform(np.log(low), np.log(high)))
        # Category-cluster vector with modest zone influence and independent noise.
        vectors[i] = (
            category_centres[ci]
            + zone_centres[zi]
            + rng.normal(0, 0.8, size=dimensions)
        ).astype(np.float32)

        title = rng.choice(CATEGORY_WORDS[category])
        if rng.random() < 0.35:
            title += " " + rng.choice(["budget", "premium", "festive", "everyday", "new"])
        if rng.random() < 0.20:
            title += " " + rng.choice(ZONE_TITLE_WORDS.get(ZONES[zi], ["Indian collection"]))
        titles.append(str(title))

    frame = pd.DataFrame({
        "product_id": np.arange(count, dtype=np.int64),
        "title": titles,
        "category": [CATEGORIES[i] for i in cat_idx],
        "zone": [ZONES[i] for i in zone_idx],
        "price_inr": np.round(prices, 2),
        "active": np.ones(count, dtype=bool),
    })
    metadata = {
        "seed": seed,
        "count": count,
        "dimensions": dimensions,
        "categories": CATEGORIES,
        "zones": ZONES,
        "vector_file": "vectors.npy",
        "vector_dtype": "float32",
        "distance_metric": "squared_euclidean",
        "data_note": "Synthetic, correlated e-commerce catalogue; not real customer or product data.",
    }
    return frame, vectors, metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("data/generated"))
    args = parser.parse_args()

    frame, vectors, metadata = generate_catalogue(args.count, args.seed, args.dimensions)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "products.csv", index=False)
    np.save(args.out / "vectors.npy", vectors)
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(frame):,} products and vectors {vectors.shape} to {args.out}")
    print("Seed:", args.seed)
    print("Zone counts (top 10):")
    print(frame["zone"].value_counts().head(10).to_string())
    print("Category-zone correlation example: saree share by zone:")
    print((frame.assign(is_saree=frame.category.eq("saree"))
           .groupby("zone").is_saree.mean().sort_values(ascending=False).head(10)
           .round(3).to_string()))

if __name__ == "__main__":
    main()
