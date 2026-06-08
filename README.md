# QAgent

**Quantum-optimized tool selection for LLM agents: pick the best subset of tools
for a task with QAOA.**

> An agent with many tools must choose a few. The best choice is not just "the
> top-k most relevant" — tools interact (some synergise, some are redundant), so
> the optimal subset is a combinatorial optimization, not a ranking. QAgent casts
> "pick k of N tools" as a constrained quadratic objective and solves it with the
> Quantum Approximate Optimization Algorithm (QAOA) on a simulator, with exact and
> greedy classical baselines to measure against.

[![tests](https://github.com/Quantum-Labor/qagent/actions/workflows/tests.yml/badge.svg)](https://github.com/Quantum-Labor/qagent/actions/workflows/tests.yml)
[![lint](https://github.com/Quantum-Labor/qagent/actions/workflows/lint.yml/badge.svg)](https://github.com/Quantum-Labor/qagent/actions/workflows/lint.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Built with PennyLane](https://img.shields.io/badge/Built%20with-PennyLane-3E00FF)](https://pennylane.ai)

**Project 2 of 3** in the Quantum Co-Processor research program. Project 1 is
[QVerify](https://github.com/Quantum-Labor/qverify) (quantum-assisted verification
of LLM reasoning, shipped). QAgent applies the same "quantum subroutine behind a
stable classical interface" pattern to tool selection.

## What is this?

When an LLM agent has access to `N` tools but should only be handed `k` of them
(to fit a context budget, reduce confusion, or cut cost), which `k`? Picking the
`k` individually-most-relevant tools is greedy and often wrong: tools have
**pairwise interactions** — a retriever plus a summarizer synergise; two
overlapping search tools are redundant. The value of a tool set is

```
score(S) = sum_{i in S} relevance[i] + sum_{i<j in S} interaction[i][j]
```

and the task is to maximise it subject to `|S| = k`. That is a quadratic binary
optimization with a cardinality constraint — exactly the structure QAOA is built
for.

## How it works (in 30 seconds)

1. **Encode** the task as a scoring model: per-tool relevances and a pairwise
   interaction matrix.
2. **Map** "select k of N" to `N` qubits (one-hot: qubit `i` = tool `i` in/out),
   with the cardinality constraint folded into a cost Hamiltonian as a penalty.
3. **Optimize** a QAOA ansatz (cost layer + mixer) with PyTorch on the PennyLane
   simulator; measure and keep the best valid subset.
4. **Compare** against an exact brute-force oracle and a greedy top-k baseline.

```python
from qagent import ToolScoring, select_tools

# 4 tools, pick 2. Tools 2 and 3 are mid-relevance but strongly synergistic,
# so {2, 3} beats the two highest-relevance tools {0, 1}.
scoring = ToolScoring(
    weights=(0.9, 0.85, 0.5, 0.5),
    synergy=((0.0, 0.0, 0.0, 0.0),
             (0.0, 0.0, 0.0, 0.0),
             (0.0, 0.0, 0.0, 1.0),
             (0.0, 0.0, 1.0, 0.0)),
)

exact = select_tools(scoring, k=2, backend="classical")   # brute-force optimum
qaoa = select_tools(scoring, k=2, backend="qaoa", p=2, steps=80)

print(exact.subset, exact.score)   # frozenset({2, 3}) 2.0
print(qaoa.subset, qaoa.score)     # frozenset({2, 3}) 2.0
```

## Install

```bash
git clone https://github.com/Quantum-Labor/qagent
cd qagent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -m "not slow"        # fast unit tests
```

## Benchmarks

`qagent-mini-50` is a seeded, hand-crafted benchmark: 25 small tasks (N=8, k=3) +
25 full tasks (N=16, k=5). Every task has relevances and a sparse interaction
matrix (about 40% of pairs interact), and its optimal subset is filled in by the
brute-force oracle. Accuracy is **score-match**: a solver is correct on a task if
its selected subset achieves the optimal score (ties count as correct).

| Group | Brute-force (oracle) | Greedy top-k | QAOA |
| --- | --- | --- | --- |
| small (N=8, k=3) | 100% (25/25) | 24% (6/25) | 100% (25/25) |
| full (N=16, k=5) | 100% (25/25) | 0% (0/25) | 0% (0/25) |
| **all** | **100% (50/50)** | **12% (6/50)** | **50% (25/50)** |

(QAOA: `p=2`, 80 optimizer steps, 1024 shots, seed 0. Reproduce with
`python scripts/run_benchmarks.py`.)

Honest read: greedy is weak by construction — because the optima depend on tool
interactions, a relevance-only ranking rarely finds them, and never at N=16/k=5.
QAOA recovers the exact optimum on **every** small (N=8) task and beats greedy
overall (50% vs 12%), but at p=2 it does **not** yet solve any of the N=16/k=5
tasks: with 16 qubits the cardinality penalty dominates the cost landscape and the
fine score differences that pick the best size-5 subset are washed out, and 1024
shots cannot cover the 4368 candidate subsets. Deeper circuits, a tighter penalty,
and a Hamming-weight-preserving (XY) mixer are the v0.2 levers. The point of v0.1
is the architecture and an honest baseline, not a quantum win at N=16. See
[docs/benchmarks.md](docs/benchmarks.md) for methodology and
[docs/qaoa-explained.md](docs/qaoa-explained.md) for the formulation.

## Limits (honest scope)

- **Simulator only.** `default.qubit` statevector; this phase targets `N <= 16`.
  No real quantum hardware yet (QVerify covers the hardware-execution story).
- **QAOA is a heuristic, and v0.1 is an honest baseline.** It solves every N=8 task
  but **0/25 of the N=16/k=5 tasks** at p=2 (the cardinality penalty dominates the
  16-qubit landscape; see [docs/benchmarks.md](docs/benchmarks.md)). At these sizes
  brute force is trivial, so the value today is the architecture, not a speed-up:
  the same `select_tools` interface scales to regimes where exhaustive search does
  not. Improving N=16 (deeper p, tighter penalty, XY mixer) is the v0.2 target.
- **Synthetic benchmark.** `qagent-mini-50` is seeded random data measuring
  interaction-aware selection quality, not real-world tool utility.
- **Out of scope for v0.1:** Hugging Face Space, IBM hardware, OAuth, GPU, larger
  encodings, and integration into a live agent loop.

## Repository layout

```
qagent/
  qaoa/        # one-hot encoding, cost Hamiltonian, QAOA ansatz + solver
  oracle/      # brute-force exact solver, greedy top-k baseline
  selector.py  # high-level select_tools(scoring, k, backend=...)
  utils/       # config defaults
benchmarks/qagent_mini/   # qagent-mini-50 dataset + methodology
scripts/                  # dataset generator, benchmark runner
docs/                     # qaoa-explained, benchmarks
tests/                    # unit tests (slow QAOA runs gated)
```

## Citation

```bibtex
@misc{brinza2026qagent,
  author       = {Serghei Brinza},
  title        = {QAgent: QAOA-based tool selection for LLM agents},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Quantum-Labor/qagent}},
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).

## Author

Serghei Brinza ([@SergheiBrinza](https://github.com/SergheiBrinza))
