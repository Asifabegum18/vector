"""Shared product and query schema for Madurai Adaptive Search Engine (MASE)."""
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Product:
    product_id: int
    title: str
    category: str
    zone: str
    price_inr: float
    vector: object  # NumPy ndarray; kept loose here to avoid runtime typing overhead.
    active: bool = True

@dataclass(frozen=True)
class SearchQuery:
    query_id: int
    vector: object
    category: Optional[str] = None
    zone: Optional[str] = None
    max_price_inr: Optional[float] = None

    def matches(self, product: Product) -> bool:
        """All supplied predicates are combined with AND."""
        if not product.active:
            return False
        if self.category is not None and product.category != self.category:
            return False
        if self.zone is not None and product.zone != self.zone:
            return False
        if self.max_price_inr is not None and product.price_inr > self.max_price_inr:
            return False
        return True
