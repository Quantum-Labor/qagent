# Benchmarks

## qagent-mini-50

A seeded, brute-force-verified benchmark for select-`k`-from-`N` tool selection.
See [benchmarks/qagent_mini/README.md](../benchmarks/qagent_mini/README.md) for
the dataset schema; this page covers methodology and results.

### Methodology

- **Tasks:** 50 total — 25 small (`N=8, k=3`) and 25 full (`N=16, k=5`).
- **Ground truth:** each task's optimal subset is computed by the exact
  brute-force oracle (`qagent.oracle.bruteforce`). Brute force enumerates all
  `C(N, k)` subsets — trivial here (`C(16,5)=4368`).
- **Metric:** *score-match accuracy*. A solver is correct on a task if its chosen
  subset achieves the optimal score (within `1e-6`). Score-match — rather than
  exact-subset-match — is used so that ties between equally optimal subsets count
  as correct.
- **Solvers compared:**
  - *brute-force* — the oracle (100% by construction; included as a sanity check).
  - *greedy top-k* — pick the `k` highest-relevance tools, ignoring interactions.
  - *QAOA* — `p=2` layers, 80 Adam steps, 1024 shots, seed 0, on
    `default.qubit`. Answer = highest-scoring measured size-`k` subset.

Reproduce:

```bash
python scripts/_gen_qagent_mini.py    # regenerate the dataset (deterministic)
python scripts/run_benchmarks.py      # full table (QAOA included; minutes)
python scripts/run_benchmarks.py --no-qaoa   # classical baselines only (fast)
```

### Results

| Group | Brute-force | Greedy top-k | QAOA |
| --- | --- | --- | --- |
| small (N=8, k=3) | 100% (25/25) | 24% (6/25) | 100% (25/25) |
| full (N=16, k=5) | 100% (25/25) | 0% (0/25) | 0% (0/25) |
| **all** | **100% (50/50)** | **12% (6/50)** | **50% (25/50)** |

(Full run: 50 tasks, ~34 min wall-clock on CPU `default.qubit`.)

### Reading the numbers

- **Greedy collapses** as the problem grows: at `N=16, k=5` a relevance-only
  ranking never recovers the interaction-optimal subset. This is the gap that
  motivates an interaction-aware solver.
- **QAOA solves all small (N=8) tasks** exactly and beats greedy overall (50% vs
  12%), but **solves none of the N=16/k=5 tasks** at `p=2`. This is a real,
  reported limitation of the v0.1 baseline, with a clear cause: the cardinality
  penalty (`default_penalty` ~ sum of all |weights| + |synergy|) is large relative
  to the score differences that distinguish size-5 subsets, so the QAOA landscape
  is dominated by "is `|S|=5`?" and the fine "which 5?" signal is washed out;
  1024 shots also cannot cover the 4368 size-5 candidates for the classical
  post-check. At N=8 (56 size-3 subsets) both effects are mild, hence 100%.
- **v0.2 levers** (not in this phase): deeper circuits (`p>=4`); a *tighter*
  penalty (only large enough to beat the best single-tool marginal, not the sum of
  everything), which restores the score signal; and a Hamming-weight-preserving
  **XY mixer** that enforces `|S|=k` by construction so no penalty is needed at
  all. The honest takeaway for v0.1 is the architecture and a measured baseline —
  the same `select_tools` interface that runs QAOA here is what would run a larger
  optimiser where brute force is no longer free.

## v0.2 results (the three N=16 fixes)

v0.2 applied the three levers above: a tighter penalty (`2 x` best single-tool
marginal instead of the sum of everything), deeper circuits (`p=4`, 160 steps),
and an XY ring mixer option. Full runs below are seed 0, 1024 shots, no seed
cherry-picking. The runner now also reports **mean approximation ratio**
(`score / optimal`), which is the metric that actually moves at N=16.

### x mixer (default), tighter penalty, p=4

| Group | Brute-force | Greedy | QAOA exact-match | QAOA approx ratio |
| --- | --- | --- | --- | --- |
| small (N=8, k=3) | 100% | 24% | 100% (25/25) | 1.000 |
| full (N=16, k=5) | 100% | 0% | 4% (1/25) | 0.915 |
| **all** | **100%** | **12%** | **52% (26/50)** | **0.957** |

### xy mixer, p=4

| Group | QAOA exact-match | QAOA approx ratio |
| --- | --- | --- |
| small (N=8, k=3) | 56% (14/25) | 0.919 |
| full (N=16, k=5) | 8% (2/25) | 0.737 |
| **all** | **32% (16/50)** | **0.828** |

### Honest reading

- **The headline goal was not met.** Exact-match on N=16/k=5 is still far below
  50% (4% with the x mixer, 8% with the xy mixer). Finding the single global
  optimum among 4368 size-5 subsets is hard for QAOA at this depth, and the
  classical post-check only sees the sampled candidates.
- **The fixes did help, on the right metric.** The tighter penalty lifts the N=16
  *approximation ratio* from ~0.735 (v0.1, indistinguishable from greedy because
  the penalty washed out the score signal) to **0.915** — the x-mixer QAOA now
  recovers ~92% of the optimal score at N=16, well above greedy. Overall
  exact-match ticks up from 50% to 52%.
- **The XY mixer underperformed the X mixer.** With a single feasible basis-state
  initialisation and a first-order ring Trotterisation, the XY mixer mixes too
  slowly: it needs very deep `p` to escape the initial state (it only solved the
  6-tool toy case at `p=8`), so at `p=4` it is worse than the plain X mixer even on
  the small tasks. The documented next step is **Dicke-state initialisation**
  (equal superposition over all weight-k states), which is what Wang et al. 2020
  use and what the simple init skips.
- **Conclusion.** v0.2 is a genuine improvement in solution *quality*
  (approximation ratio) and fixes the v0.1 penalty design flaw, but it does **not**
  reach the exact-match target at N=16. Per the project's stop criterion, no
  further seeds or fixes were tried.
