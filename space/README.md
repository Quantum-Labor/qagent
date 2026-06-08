---
title: QAgent
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
hardware: cpu-basic
tags:
  - quantum-computing
  - qaoa
  - pennylane
  - llm-agents
  - tool-selection
  - auto-deploy
---

# QAgent - quantum tool selection for LLM agents

This Space demonstrates [QAgent](https://github.com/Quantum-Labor/qagent): picking
the best subset of `k` tools from `N` candidates for an agent task, solved with the
Quantum Approximate Optimization Algorithm (QAOA). It is project 2 of 3 in the
Quantum Co-Processor program (after QVerify, alongside QRoute).

## What you can do here

- **Explore the qagent-mini-50 tasks.** Pick any of the 50 benchmark tasks; the
  4x4 (or 4x2) tool grid highlights which tools the brute-force optimum, QAOA, and
  greedy each select, with score cards and approximation ratios.
- **See the score landscape.** A chart plots the scores of every size-`k` subset
  with markers showing exactly where QAOA, greedy, and the optimum fall - so you
  can see how close QAOA gets and how far greedy misses.
- **Build an exploration history.** As you browse tasks, the session tracks the
  QAOA vs greedy approximation ratio.
- **Verify live.** A button runs the pure-Python brute-force and greedy solvers
  live (no precomputed lookup) and confirms they match the served numbers.
- **Read the leaderboard.** The qagent-mini-50 summary: exact-match and mean
  approximation ratio for brute-force, greedy, and QAOA.

## Design notes

- **Precomputed QAOA.** QAOA on 16 qubits is slow on CPU, so the QAOA results are
  precomputed with the documented v0.2 config (p=4, 160 steps, 1024 shots, seed 0,
  x mixer) and served from JSON for an instant experience. The classical solvers
  run live (they are pure Python and finish in milliseconds), which keeps the
  served numbers honest and the Docker image small (no torch / pennylane).
- **Fully open.** There is no IBM hardware path and no quota to protect, so the
  Space has no OAuth gating - anyone can use every feature.
- **Honest scope.** Simulator only; no quantum advantage is claimed. QAOA matches
  the optimum on every N=8 task and reaches a 0.915 mean approximation ratio at
  N=16, but exact-match at N=16 remains hard (the next step is Dicke-initialised
  XY-QAOA). Full benchmarks and source: the GitHub repo.

Auto-deployed from the `qagent` repository `main` branch via a GitHub Action
(`HfApi.upload_folder`); see `docs/deploy.md` in the repo.
