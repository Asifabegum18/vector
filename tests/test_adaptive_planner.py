
from src.planner import choose_strategy, choose_nprobe


def test_small_eligible_subset_uses_exact():
    assert choose_strategy(60000, 100) == "exact"


def test_large_eligible_subset_uses_ivf():
    assert choose_strategy(60000, 30000) == "ivf"


def test_empty_catalogue_uses_exact():
    assert choose_strategy(0, 0) == "exact"


def test_probe_count_is_valid():
    probes = choose_nprobe(60000, 16, selectivity=0.05)

    assert 1 <= probes <= 16


def test_rare_selectivity_probes_all_partitions():
    assert choose_nprobe(60000, 16, selectivity=0.01) == 16
