"""Show a planner boundary case and its selected probe count."""

from src.planner import choose_nprobe


def main():
    cases = [
        {"name": "empty_catalog", "total_items": 0, "nlist": 1, "selectivity": 1.0},
        {"name": "highly_selective_filter", "total_items": 10000, "nlist": 100, "selectivity": 0.001},
    ]
    for case in cases:
        probes = choose_nprobe(case["total_items"], case["nlist"], selectivity=case["selectivity"])
        print({**case, "nprobe": probes})


if __name__ == "__main__":
    main()