"""Run qagent-mini-50: brute-force vs greedy vs QAOA accuracy.

Accuracy = fraction of tasks whose selected subset matches the optimal score
(score-match, so ties between equally optimal subsets count as correct).
Brute-force is the oracle (100% by construction).

Examples:
    python scripts/run_benchmarks.py                 # all backends, full suite
    python scripts/run_benchmarks.py --no-qaoa       # classical baselines only
    python scripts/run_benchmarks.py --max-entries 6 # quick subset
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from qagent.oracle.bruteforce import brute_force_best
from qagent.oracle.greedy import greedy_top_k
from qagent.qaoa.encoding import ToolScoring

DATA = Path(__file__).resolve().parent.parent / "benchmarks" / "qagent_mini" / "dataset.json"
EPS = 1e-6


def _scoring(entry: dict[str, Any]) -> ToolScoring:
    return ToolScoring(
        weights=tuple(entry["weights"]),
        synergy=tuple(tuple(row) for row in entry["synergy"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="qagent-mini-50 accuracy")
    parser.add_argument("--dataset", type=Path, default=DATA)
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--no-qaoa", action="store_true", help="skip the QAOA backend")
    parser.add_argument("--p", type=int, default=4, help="QAOA layers")
    parser.add_argument("--steps", type=int, default=160, help="optimizer steps")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixer", choices=["x", "xy"], default="x", help="QAOA mixer")
    args = parser.parse_args()

    entries = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.max_entries is not None:
        entries = entries[: args.max_entries]

    solve_qaoa = None
    if not args.no_qaoa:
        from qagent.qaoa.circuit import solve_qaoa as _sq

        solve_qaoa = _sq

    rows: list[dict[str, Any]] = []
    t0 = time.monotonic()
    for e in entries:
        sc = _scoring(e)
        k = e["k"]
        opt = float(e["optimal_score"])

        bf_ok = abs(brute_force_best(sc, k)[1] - opt) <= EPS
        gd_ok = abs(greedy_top_k(sc, k)[1] - opt) <= EPS
        qa_ok: bool | None = None
        if solve_qaoa is not None:
            res = solve_qaoa(
                sc,
                k,
                p=args.p,
                steps=args.steps,
                shots=args.shots,
                seed=args.seed,
                mixer=args.mixer,
            )
            qa_ok = abs(res.score - opt) <= EPS

        rows.append({"id": e["id"], "tier": e["tier"], "bf": bf_ok, "gd": gd_ok, "qa": qa_ok})

    elapsed = time.monotonic() - t0

    def pct(rows_: list[dict[str, Any]], key: str) -> str:
        vals = [r[key] for r in rows_ if r[key] is not None]
        if not vals:
            return "n/a"
        return f"{100.0 * sum(vals) / len(vals):.1f}% ({sum(vals)}/{len(vals)})"

    tiers = ["small", "full"]
    print(f"\nqagent-mini  ({len(rows)} tasks, {elapsed:.1f}s)")
    print(f"{'group':<10} {'brute-force':<16} {'greedy':<16} {'qaoa':<16}")
    print("-" * 58)
    for t in tiers:
        sub = [r for r in rows if r["tier"] == t]
        if sub:
            print(f"{t:<10} {pct(sub, 'bf'):<16} {pct(sub, 'gd'):<16} {pct(sub, 'qa'):<16}")
    print(f"{'ALL':<10} {pct(rows, 'bf'):<16} {pct(rows, 'gd'):<16} {pct(rows, 'qa'):<16}")


if __name__ == "__main__":
    main()
