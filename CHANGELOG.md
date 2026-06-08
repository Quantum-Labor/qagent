# Changelog

All notable changes to QAgent are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
semantic versioning.

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
