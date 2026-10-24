"""Vector distance helpers implemented for this project."""
import numpy as np

def squared_l2(query, vectors):
    q = np.asarray(query, dtype=np.float32)
    x = np.asarray(vectors, dtype=np.float32)
    if q.ndim != 1 or x.ndim != 2:
        raise ValueError("query must be 1D and vectors must be 2D")
    if x.shape[1] != q.shape[0]:
        raise ValueError("query and vectors must have equal dimensions")
    return np.sum((x - q) ** 2, axis=1, dtype=np.float32)

def nearest_indices(query, vectors, k):
    if k < 1:
        raise ValueError("k must be >= 1")
    distances = squared_l2(query, vectors)
    order = np.argsort(distances, kind="stable")[:min(k, len(distances))]
    return order.astype(np.int64), distances[order]
def cosine_distance(query, vectors):
    """Calculate cosine distance between one query and each vector."""
    import numpy as np

    query = np.asarray(query, dtype=np.float32)
    vectors = np.asarray(vectors, dtype=np.float32)

    if query.ndim != 1 or vectors.ndim != 2:
        raise ValueError("Query must be 1D and vectors must be 2D")

    if vectors.shape[1] != query.shape[0]:
        raise ValueError("Query and vectors must have equal dimensions")

    query_norm = np.linalg.norm(query)
    vector_norms = np.linalg.norm(vectors, axis=1)

    denominator = vector_norms * query_norm
    similarities = np.zeros(len(vectors), dtype=np.float32)

    valid = denominator > 0
    similarities[valid] = (
        vectors[valid] @ query
    ) / denominator[valid]

    return 1.0 - similarities