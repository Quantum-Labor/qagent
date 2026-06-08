"""Bake QAOA / greedy / brute-force results for all qagent-mini-50 tasks into a
single JSON, so the HF Space serves instant results with no live QAOA on CPU.

For each task we store the brute-force optimum, the greedy baseline, the QAOA
result (p=4, 160 steps, seed 0 -- the documented v0.2 config), and a histogram of
the scores of every size-k subset (for the Space's score-landscape chart, which
shows where QAOA / greedy / optimal fall in the full distribution).

Run:  python scripts/precompute_results.py   (slow: full QAOA per task, ~1h)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from qagent.oracle.bruteforce import brute_force_best
from qagent.oracle.greedy import greedy_top_k
from qagent.qaoa.circuit import solve_qaoa
from qagent.qaoa.encoding import ToolScoring, all_subsets_of_size, score_subset

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "benchmarks" / "qagent_mini" / "dataset.json"
OUT = ROOT / "space" / "precomputed" / "benchmark_results.json"
P, STEPS, SHOTS, SEED, MIXER = 4, 160, 1024, 0, "x"
EPS = 1e-6


def _scoring(entry: dict[str, Any]) -> ToolScoring:
    return ToolScoring(
        weights=tuple(entry["weights"]),
        synergy=tuple(tuple(row) for row in entry["synergy"]),
    )


def _score_histogram(scoring: ToolScoring, k: int, nbins: int = 40) -> dict[str, Any]:
    scores = [score_subset(scoring, s) for s in all_subsets_of_size(scoring.n_tools, k)]
    counts, edges = np.histogram(scores, bins=nbins)
    return {
        "bin_edges": [round(float(e), 4) for e in edges],
        "counts": [int(c) for c in counts],
        "min": round(float(min(scores)), 4),
        "max": round(float(max(scores)), 4),
        "n_subsets": len(scores),
    }


def main() -> None:
    entries = json.loads(DATA.read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = []
    t0 = time.monotonic()
    for e in entries:
        sc = _scoring(e)
        k = e["k"]
        opt_sub, opt = brute_force_best(sc, k)
        gd_sub, gd = greedy_top_k(sc, k)
        res = solve_qaoa(sc, k, p=P, steps=STEPS, shots=SHOTS, seed=SEED, mixer=MIXER)
        tasks.append(
            {
                "id": e["id"],
                "tier": e["tier"],
                "n_tools": e["n_tools"],
                "k": k,
                "description": e["description"],
                "tools": [
                    {"index": i, "name": f"tool_{i:02d}", "weight": round(e["weights"][i], 3)}
                    for i in range(e["n_tools"])
                ],
                "optimal": {"subset": sorted(opt_sub), "score": round(opt, 4)},
                "greedy": {
                    "subset": sorted(gd_sub),
                    "score": round(gd, 4),
                    "approx_ratio": round(gd / opt, 4) if opt else 1.0,
                    "exact_match": abs(gd - opt) <= EPS,
                },
                "qaoa": {
                    "subset": sorted(res.subset),
                    "score": round(res.score, 4),
                    "approx_ratio": round(res.score / opt, 4) if opt else 1.0,
                    "exact_match": abs(res.score - opt) <= EPS,
                    "p": P,
                    "steps": STEPS,
                    "seed": SEED,
                },
                "score_hist": _score_histogram(sc, k),
            }
        )
        print(
            f"{e['id']:>4} ({e['tier']:<5}) qaoa {res.score:.3f}/{opt:.3f} "
            f"ratio {res.score / opt:.3f} exact={abs(res.score - opt) <= EPS}",
            flush=True,
        )

    def _agg(tier: str | None) -> dict[str, Any]:
        rows = [t for t in tasks if tier is None or t["tier"] == tier]
        n = len(rows)
        return {
            "n": n,
            "qaoa_exact": sum(1 for t in rows if t["qaoa"]["exact_match"]),
            "qaoa_ratio": round(sum(t["qaoa"]["approx_ratio"] for t in rows) / n, 4),
            "greedy_exact": sum(1 for t in rows if t["greedy"]["exact_match"]),
            "greedy_ratio": round(sum(t["greedy"]["approx_ratio"] for t in rows) / n, 4),
        }

    summary = {"all": _agg(None), "small": _agg("small"), "full": _agg("full")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"p": P, "steps": STEPS, "shots": SHOTS, "seed": SEED, "mixer": MIXER},
        "summary": summary,
        "tasks": tasks,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT} ({len(tasks)} tasks) in {time.monotonic() - t0:.0f}s")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
