# Changelog

All notable changes to QAgent are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
semantic versioning.

## [0.2.0] - Unreleased

Targeted the N=16 failure of v0.1 (0% exact-match) with three fixes and measured
each honestly (seed 0, no cherry-picking).

- `fix(qaoa)` tighter cardinality penalty (2x best single-tool marginal instead of
  the sum of all weights+synergy), which stops the penalty from washing out the
  score signal at N=16.
- `perf(qaoa)` diagonal-energy training (the cost is diagonal in Z), default
  `p=4` / 160 steps, and a `--mixer` CLI flag.
- `feat(qaoa)` XY ring mixer (`mixer="xy"`), Hamming-weight-preserving.
- `run_benchmarks.py` now reports mean approximation ratio alongside exact-match.

Result: the N=16 approximation ratio rises from ~0.735 to **0.915** (x mixer) and
overall exact-match from 50% to 52%, but **exact-match at N=16 stays below the 50%
target (4%)**. The XY mixer underperformed the X mixer at `p=4` (needs Dicke-state
init). See docs/benchmarks.md.

## [0.1.0] - Unreleased

First baseline. QAOA-based tool selection (pick k tools from N) on the PennyLane
simulator, with classical brute-force and greedy baselines and a hand-crafted
benchmark.

- `qagent.qaoa` — one-hot encoding, cost Hamiltonian (relevance + pair synergy +
  cardinality penalty), QAOA ansatz, and a PennyLane/PyTorch solver.
- `qagent.oracle` — exact brute-force solver and a greedy top-k baseline.
- `qagent.selector` — high-level task-to-tools API with `qaoa` and `classical`
  backends.
- `benchmarks/qagent_mini` — qagent-mini-50, 50 seeded tasks (25 at N=8, 25 at
  N=16) with brute-force-verified optimal subsets.
