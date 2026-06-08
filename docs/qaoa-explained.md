# QAOA for tool selection, explained

QAgent picks the best size-`k` subset of tools for a task. This document explains
the formulation: the scoring model, the one-hot encoding, the cost Hamiltonian,
and why this is more than a greedy top-k.

## The problem

A task comes with `N` candidate tools. Each tool `i` has a standalone relevance
`w_i`. Each pair `(i, j)` has an interaction `J_ij` that is positive when the
tools are synergistic (better together) and negative when they are redundant
(overlapping capability). The value of a chosen subset `S` is

```
score(S) = sum_{i in S} w_i  +  sum_{i<j in S} J_ij
```

The goal: maximise `score(S)` subject to `|S| = k`.

This is a constrained quadratic binary optimisation. The pair term is what makes
it interesting — and what a greedy "take the k highest `w_i`" approach gets wrong
whenever the optimal subset hinges on interactions rather than standalone scores.

## One-hot encoding

We use one qubit per tool: `N` qubits, where qubit `i` holds a binary variable
`x_i in {0, 1}` that is 1 iff tool `i` is selected. A computational basis state is
therefore a bitstring that *directly* names a subset (tool 0 is the leftmost bit).
For `N = 16` that is 16 qubits and `2^16 = 65536` basis states — one per possible
subset of any size.

We deliberately do **not** use a more compact encoding (e.g. `log2(C(N,k))`
qubits indexing only size-`k` subsets). One-hot keeps the cost local (single- and
two-qubit terms only), makes the circuit and the readout trivial to interpret, and
matches how an agent thinks about tools ("is this one in or out?"). The cost is
that cardinality is not automatic — so we enforce it inside the cost Hamiltonian.

## Cost Hamiltonian

We turn the objective into an energy to minimise:

```
C(x) = -score(x)  +  lambda * (sum_i x_i - k)^2
```

The first term rewards high-scoring subsets; the second is a soft penalty that
costs energy whenever the number of selected tools differs from `k`. We pick
`lambda` larger than the largest achievable score swing (see `default_penalty`),
so the global minimum of `C` is guaranteed to have exactly `|S| = k`. The
cardinality constraint thus lives entirely in the cost Hamiltonian — the mixer
stays a standard transverse field.

To run on a quantum device we express `C` in the Pauli-Z basis using
`x_i = (1 - Z_i) / 2`. Expanding the linear, quadratic, and penalty terms (and
dropping the constant offset, which does not move the minimiser) gives

```
H_C = sum_i h_i Z_i  +  sum_{i<j} J'_ij Z_i Z_j
```

with the coefficients computed in `qagent.qaoa.circuit.cost_coefficients`:

```
a_i   = -w_i + lambda * (1 - 2k)          # linear x-coefficient
b_ij  = -J_ij + 2 * lambda                # quadratic x-coefficient (i<j)
h_i   = -a_i / 2  -  (1/4) * sum_{m != i} b_im
J'_ij =  b_ij / 4
```

This is a standard Ising form: single-`Z` fields plus `ZZ` couplings. Its diagonal
is exactly the (offset) classical cost of each subset, so the ground state is the
optimal size-`k` selection. The unit test
`test_cost_hamiltonian_ground_state_is_the_optimal_subset` checks precisely this:
the lowest-energy basis state has `|S| = k` and matches the brute-force optimum.

## The QAOA circuit

A depth-`p` QAOA alternates the cost and mixer unitaries on the uniform
superposition:

```
|psi(gamma, beta)> = prod_{l=1..p} [ U_M(beta_l) U_C(gamma_l) ]  H^{⊗N} |0>
```

- **Cost layer** `U_C(gamma) = exp(-i gamma H_C)` is implemented with `RZ(2*gamma*h_i)`
  on each qubit and `IsingZZ(2*gamma*J'_ij)` on each interacting pair.
- **Mixer layer** `U_M(beta) = exp(-i beta sum_i X_i)` is `RX(2*beta)` on every qubit.

The `2p` angles `(gamma_l, beta_l)` are trained with PyTorch/Adam against the
analytic expectation `<psi| H_C |psi>` (PennyLane `default.qubit` backprop). After
training we sample the circuit and return the highest-scoring measured subset of
size `k` — a classical post-check over the measured candidates, mirroring how an
agent would pick the best valid option it observed. (If no size-`k` state is
sampled, we fall back to the most probable size-`k` basis state analytically.)

## Why this is more than greedy

Greedy top-k ignores `J_ij` entirely: it takes the `k` tools with the largest
`w_i`. Whenever the optimum depends on pair interactions — two mid-relevance tools
that synergise, or two high-relevance tools that are redundant — greedy is wrong.
QAOA optimises over the full quadratic objective, so it can find those
interaction-driven optima. The `qagent-mini-50` benchmark is built to expose this
gap: greedy scores far below 100% (12% overall, 0% at N=16). In v0.1 QAOA recovers
the exact optimum on every N=8 task and beats greedy overall, but does not yet
solve the N=16/k=5 tasks at `p=2` — a reported baseline limitation, not a quantum
win at scale. See [benchmarks.md](benchmarks.md) for the numbers and the cause.

## Honest limits

- The simulator caps practical `N` (statevector is `2^N`); this phase targets
  `N <= 16`, well within `default.qubit`.
- QAOA is a heuristic: at fixed depth and finite optimisation it does not match the
  exact brute-force oracle on every instance. For the sizes here brute force is
  trivial, so the value is the architecture (the same `select_tools` interface
  scales to regimes where exhaustive search does not), not a speed-up today.
- The benchmark is synthetic and seeded; it measures interaction-aware selection
  quality, not real-world tool utility.
