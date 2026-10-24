import pytest

from src.planner import choose_nprobe


def test_probe_count_is_bounded_and_selectivity_aware():
    broad = choose_nprobe(1000, 20, selectivity=1.0)
    selective = choose_nprobe(1000, 20, selectivity=0.01)
    assert 1 <= broad <= 20
    assert 1 <= selective <= 20
    assert selective >= broad


@pytest.mark.parametrize("args", [(0, 0), (-1, 2), (10, 2)])
def test_invalid_planner_inputs(args):
    if args == (10, 2):
        with pytest.raises(ValueError, match="selectivity"):
            choose_nprobe(*args, selectivity=1.1)
    else:
        with pytest.raises(ValueError):
            choose_nprobe(*args)