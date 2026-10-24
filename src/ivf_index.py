"""Custom IVF-style vector index. KMeans only builds partitions; search is ours."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from src.distance import squared_l2

@dataclass
class SearchResult:
    product_ids: np.ndarray
    distances: np.ndarray
    candidates_examined: int
    partitions_probed: int

class IVFIndex:
    def __init__(self, n_partitions=128, random_state=42, batch_size=2048, max_iter=100):
        if n_partitions < 1:
            raise ValueError("n_partitions must be >= 1")
        self.n_partitions = n_partitions
        self.random_state = random_state
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.centroids = self.inverted_lists = self.product_ids = None
        self.vectors = self.assignments = None

    def fit(self, vectors, product_ids=None):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) == 0 or not np.isfinite(vectors).all():
            raise ValueError("vectors must be a non-empty finite 2D array")
        if self.n_partitions > len(vectors):
            raise ValueError("n_partitions cannot exceed number of vectors")
        if product_ids is None:
            product_ids = np.arange(len(vectors), dtype=np.int64)
        product_ids = np.asarray(product_ids, dtype=np.int64)
        if product_ids.ndim != 1 or len(product_ids) != len(vectors):
            raise ValueError("provide one product ID per vector")
        if len(np.unique(product_ids)) != len(product_ids):
            raise ValueError("product IDs must be unique")

        model = MiniBatchKMeans(
            n_clusters=self.n_partitions, random_state=self.random_state,
            batch_size=max(self.batch_size, self.n_partitions),
            max_iter=self.max_iter, n_init=3
        )
        assignments = model.fit_predict(vectors).astype(np.int32)
        self.centroids = model.cluster_centers_.astype(np.float32)
        self.vectors = vectors.copy()
        self.product_ids = product_ids.copy()
        self.assignments = assignments
        self.inverted_lists = [
            np.flatnonzero(assignments == p).astype(np.int64)
            for p in range(self.n_partitions)
        ]
        return self

    def _check_fitted(self):
        if self.centroids is None:
            raise RuntimeError("call fit() before search()")

    def nearest_partitions(self, query, nprobe=8):
        self._check_fitted()
        if not 1 <= nprobe <= self.n_partitions:
            raise ValueError("nprobe must be between 1 and n_partitions")
        return np.argsort(squared_l2(query, self.centroids), kind="stable")[:nprobe]

    def search(self, query, k=10, nprobe=8):
        self._check_fitted()
        if k < 1:
            raise ValueError("k must be >= 1")
        query = np.asarray(query, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.vectors.shape[1]:
            raise ValueError("query has wrong vector dimension")
        partitions = self.nearest_partitions(query, nprobe)
        lists = [self.inverted_lists[int(p)] for p in partitions if len(self.inverted_lists[int(p)])]
        if not lists:
            return SearchResult(np.array([], dtype=np.int64), np.array([], dtype=np.float32), 0, len(partitions))
        rows = np.concatenate(lists)
        distances = squared_l2(query, self.vectors[rows])
        order = np.lexsort((self.product_ids[rows], distances))[:min(k, len(rows))]
        return SearchResult(self.product_ids[rows][order], distances[order], len(rows), len(partitions))

    def save(self, directory):
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, arr in [("centroids", self.centroids), ("product_ids", self.product_ids),
                          ("vectors", self.vectors), ("assignments", self.assignments)]:
            np.save(directory / f"{name}.npy", arr)
        meta = {"n_partitions": self.n_partitions, "random_state": self.random_state,
                "batch_size": self.batch_size, "max_iter": self.max_iter}
        (directory / "index_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / "index_metadata.json").read_text(encoding="utf-8"))
        obj = cls(**meta)
        obj.centroids = np.load(directory / "centroids.npy")
        obj.product_ids = np.load(directory / "product_ids.npy")
        obj.vectors = np.load(directory / "vectors.npy")
        obj.assignments = np.load(directory / "assignments.npy")
        obj.inverted_lists = [np.flatnonzero(obj.assignments == p).astype(np.int64)
                              for p in range(obj.n_partitions)]
        return obj
