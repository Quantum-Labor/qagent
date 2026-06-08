"""QAOA ansatz for select-k-from-N tool selection (one-hot, PennyLane).

The cost Hamiltonian encodes ``-score(S) + penalty * (|S| - k)^2`` in the Pauli-Z
basis via ``x_i = (1 - Z_i) / 2``. Minimising its expectation drives amplitude
onto the highest-scoring size-``k`` subset. A standard transverse-field X mixer is
used; the cardinality constraint lives entirely in the cost Hamiltonian as a soft
penalty (see docs/qaoa-explained.md).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pennylane as qml

from qagent.qaoa.encoding import (
    ToolScoring,
    bitstring_to_subset,
    score_subset,
)


def default_penalty(scoring: ToolScoring) -> float:
    """Cardinality penalty just large enough to make ``|S| = k`` optimal.

    Moving from a size-k subset to size k+-1 changes the penalty by ``lambda`` and
    the score by at most the best single-tool marginal contribution. So any
    ``lambda`` above that marginal forces the optimum to have exactly k tools; we
    use twice the marginal for margin. This is far tighter than summing every
    weight and synergy (the v0.1 heuristic), which swamped the score differences
    that distinguish size-k subsets and made QAOA fail at N=16.
    """
    n = scoring.n_tools
    best_marginal = 0.0
    for i in range(n):
        marginal = scoring.weights[i] + sum(
            max(scoring.synergy[i][j], 0.0) for j in range(n) if j != i
        )
        best_marginal = max(best_marginal, marginal)
    return 2.0 * best_marginal if best_marginal > 0.0 else 1.0


def cost_coefficients(
    scoring: ToolScoring, k: int, penalty: float
) -> tuple[list[float], dict[tuple[int, int], float]]:
    """Return ``(h, zz)``: single-qubit ``Z_i`` coefficients and ``Z_i Z_j``
    coefficients for the cost Hamiltonian, derived from the QUBO objective
    ``-score(x) + penalty * (sum_i x_i - k)^2`` with ``x_i = (1 - Z_i)/2``.
    The constant offset is dropped (it does not change the minimiser)."""
    n = scoring.n_tools
    lam = penalty
    w = scoring.weights
    syn = scoring.synergy

    # Objective in x-space: linear a_i x_i + quadratic b_ij x_i x_j (i<j).
    a = [-w[i] + lam * (1 - 2 * k) for i in range(n)]

    def b(i: int, m: int) -> float:
        return -syn[i][m] + 2.0 * lam

    h: list[float] = []
    for i in range(n):
        hi = -a[i] / 2.0
        for m in range(n):
            if m != i:
                hi += -b(i, m) / 4.0
        h.append(hi)

    zz: dict[tuple[int, int], float] = {}
    for i in range(n):
        for m in range(i + 1, n):
            zz[(i, m)] = b(i, m) / 4.0
    return h, zz


def cost_hamiltonian(scoring: ToolScoring, k: int, penalty: float | None = None) -> Any:
    """Build the cost Hamiltonian as a ``qml.Hamiltonian`` over Z and ZZ terms."""
    if penalty is None:
        penalty = default_penalty(scoring)
    h, zz = cost_coefficients(scoring, k, penalty)
    coeffs: list[float] = []
    ops: list[Any] = []
    for i, hi in enumerate(h):
        if abs(hi) > 1e-12:
            coeffs.append(hi)
            ops.append(qml.PauliZ(i))
    for (i, m), c in zz.items():
        if abs(c) > 1e-12:
            coeffs.append(c)
            ops.append(qml.PauliZ(i) @ qml.PauliZ(m))
    return qml.Hamiltonian(coeffs, ops)


@dataclass(frozen=True)
class QAOAResult:
    """Outcome of a QAOA solve: the best size-``k`` subset measured."""

    subset: frozenset[int]
    score: float
    bitstring: str
    probability: float
    n_layers: int
    n_steps: int
    final_cost: float


def _apply_ansatz(
    gammas: Any, betas: Any, n: int, p: int, h: list[float], zz: dict[tuple[int, int], float]
) -> None:
    for w in range(n):
        qml.Hadamard(wires=w)
    for layer in range(p):
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qml.RZ(2.0 * gammas[layer] * h[i], wires=i)
        for (i, m), c in zz.items():
            if abs(c) > 1e-12:
                qml.IsingZZ(2.0 * gammas[layer] * c, wires=[i, m])
        for i in range(n):
            qml.RX(2.0 * betas[layer], wires=i)


def solve_qaoa(
    scoring: ToolScoring,
    k: int,
    *,
    p: int = 2,
    steps: int = 60,
    lr: float = 0.1,
    shots: int = 1024,
    penalty: float | None = None,
    seed: int = 0,
) -> QAOAResult:
    """Optimise a ``p``-layer QAOA for select-``k`` and return the best measured
    size-``k`` subset.

    Parameters are trained with PyTorch/Adam against the analytic cost expectation
    (``default.qubit`` backprop). The answer is read out by sampling and keeping
    the highest-scoring measured subset of size ``k`` (a classical post-check over
    the measured candidates); if sampling produces no size-``k`` candidate, the
    most-probable size-``k`` basis state is returned analytically.
    """
    import torch

    n = scoring.n_tools
    if penalty is None:
        penalty = default_penalty(scoring)
    hamiltonian = cost_hamiltonian(scoring, k, penalty)
    h, zz = cost_coefficients(scoring, k, penalty)

    dev = qml.device("default.qubit", wires=n, seed=seed)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def cost_expval(gammas: Any, betas: Any) -> Any:
        _apply_ansatz(gammas, betas, n, p, h, zz)
        return qml.expval(hamiltonian)

    torch.manual_seed(seed)
    gammas = torch.full((p,), 0.1, dtype=torch.float64, requires_grad=True)
    betas = torch.full((p,), 0.1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([gammas, betas], lr=lr)

    final_cost = 0.0
    for _ in range(steps):
        opt.zero_grad()
        loss = cost_expval(gammas, betas)
        loss.backward()
        opt.step()
        final_cost = float(loss.detach())

    g = [float(x) for x in gammas.detach().numpy()]
    bt = [float(x) for x in betas.detach().numpy()]

    sample_dev = qml.device("default.qubit", wires=n, seed=seed)

    @qml.set_shots(shots=shots)
    @qml.qnode(sample_dev)
    def sample_circuit() -> Any:
        _apply_ansatz(g, bt, n, p, h, zz)
        return qml.sample(wires=list(range(n)))

    samples = np.atleast_2d(np.asarray(sample_circuit()))
    counter: Counter[str] = Counter("".join(str(int(x)) for x in row) for row in samples)

    best: tuple[frozenset[int], float, str, int] | None = None
    for bits, cnt in counter.most_common():
        sub = bitstring_to_subset(bits)
        if len(sub) == k:
            sc = score_subset(scoring, sub)
            if best is None or sc > best[1]:
                best = (sub, sc, bits, cnt)

    if best is not None:
        subset, sc, bits, cnt = best
        return QAOAResult(
            subset=subset,
            score=sc,
            bitstring=bits,
            probability=cnt / shots,
            n_layers=p,
            n_steps=steps,
            final_cost=final_cost,
        )

    # Fallback: no size-k state sampled -> pick the most probable size-k state.
    prob_dev = qml.device("default.qubit", wires=n)

    @qml.qnode(prob_dev)
    def probs_circuit() -> Any:
        _apply_ansatz(g, bt, n, p, h, zz)
        return qml.probs(wires=list(range(n)))

    probs = np.asarray(probs_circuit())
    best_idx = -1
    best_prob = -1.0
    for idx in range(len(probs)):
        bits = format(idx, f"0{n}b")
        if bits.count("1") == k and probs[idx] > best_prob:
            best_prob = float(probs[idx])
            best_idx = idx
    bits = format(best_idx, f"0{n}b")
    subset = bitstring_to_subset(bits)
    return QAOAResult(
        subset=subset,
        score=score_subset(scoring, subset),
        bitstring=bits,
        probability=best_prob,
        n_layers=p,
        n_steps=steps,
        final_cost=final_cost,
    )
