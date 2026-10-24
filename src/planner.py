"""Small heuristic planner for selecting an IVF probe count."""

import math


def choose_nprobe(
    total_items: int,
    nlist: int,
    *,
    selectivity: float = 1.0,
    target_candidates: int = 20,
) -> int:
    """Choose a bounded probe count, increasing it for selective filters."""
    if total_items < 0 or nlist < 1 or target_candidates < 1:
        raise ValueError("total_items must be non-negative; nlist and target_candidates must be positive")
    if not 0.0 <= selectivity <= 1.0:
        raise ValueError("selectivity must be between 0 and 1")
    if total_items == 0 or selectivity == 0:
        return 1
    expected_per_list = total_items / nlist
    candidates_needed = min(total_items, target_candidates / selectivity)
    needed = math.ceil(candidates_needed / expected_per_list)
    return min(nlist, max(1, needed))

def choose_strategy(total_products, estimated_count, exact_threshold=10000):
    if total_products <= 0 or estimated_count <= 0:
        return "exact"

    if estimated_count <= exact_threshold:
        return "exact"

    return "ivf"



def choose_nprobe(total_products, nlist, selectivity=1.0):
    """Choose an IVF probe budget based on estimated selectivity."""

    if total_products <= 0:
        raise ValueError("total_products must be positive")

    if nlist <= 0:
        raise ValueError("nlist must be positive")

    if not 0.0 <= selectivity <= 1.0:
        raise ValueError("selectivity must be between 0 and 1")

    if selectivity <= 0.01:
        probes = nlist
    elif selectivity <= 0.10:
        probes = max(1, (3 * nlist + 3) // 4)
    elif selectivity <= 0.50:
        probes = max(1, nlist // 2)
    else:
        probes = max(1, nlist // 4)

    return min(nlist, probes)
