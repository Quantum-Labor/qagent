"""Generate the qagent-mini-50 benchmark dataset.

Seeded and reproducible: 25 small tasks (N=8, k=3) + 25 full tasks (N=16, k=5).
Each task has random tool relevances and a sparse synergy matrix; the optimal
size-k subset is filled in by the brute-force oracle so the dataset is its own
ground truth. Synthetic by construction (no real tool metadata is claimed).

Run:  python scripts/_gen_qagent_mini.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from qagent.oracle.bruteforce import brute_force_best
from qagent.qaoa.encoding import ToolScoring

OUT = Path(__file__).resolve().parent.parent / "benchmarks" / "qagent_mini" / "dataset.json"
SEED = 20260608
PAIR_DENSITY = 0.4  # fraction of tool pairs with a non-zero interaction


def _make_scoring(rng: random.Random, n: int) -> tuple[list[float], list[list[float]]]:
    weights = [round(rng.uniform(0.05, 1.0), 3) for _ in range(n)]
    synergy = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < PAIR_DENSITY:
                v = round(rng.uniform(-0.6, 0.9), 3)
                synergy[i][j] = synergy[j][i] = v
    return weights, synergy


def build() -> list[dict[str, object]]:
    rng = random.Random(SEED)
    entries: list[dict[str, object]] = []
    specs = [("small", 8, 3, 25), ("full", 16, 5, 25)]
    counter = 0
    for tier, n, k, count in specs:
        for c in range(count):
            counter += 1
            weights, synergy = _make_scoring(rng, n)
            scoring = ToolScoring(
                weights=tuple(weights),
                synergy=tuple(tuple(row) for row in synergy),
            )
            opt_subset, opt_score = brute_force_best(scoring, k)
            entries.append(
                {
                    "id": f"{tier[0]}{c + 1:02d}",
                    "description": (
                        f"Task {counter}: choose the {k} most useful tools from "
                        f"{n} candidates ({tier} pool)."
                    ),
                    "tier": tier,
                    "n_tools": n,
                    "k": k,
                    "tools": [{"index": i, "name": f"tool_{i:02d}"} for i in range(n)],
                    "weights": weights,
                    "synergy": synergy,
                    "optimal_subset": sorted(opt_subset),
                    "optimal_score": round(opt_score, 6),
                }
            )
    return entries


def main() -> None:
    entries = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    n_small = sum(1 for e in entries if e["tier"] == "small")
    n_full = sum(1 for e in entries if e["tier"] == "full")
    print(f"wrote {len(entries)} entries ({n_small} small N=8, {n_full} full N=16) to {OUT}")


if __name__ == "__main__":
    main()
