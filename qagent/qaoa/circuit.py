"""QAOA ansatz for select-k-from-N tool selection (one-hot, PennyLane).

The cost is diagonal in the Pauli-Z basis (single-Z and ZZ terms only, via
``x_i = (1 - Z_i) / 2``), so its expectation is ``sum_x p(x) E(x)`` for a
precomputed energy vector ``E`` -- which is what the optimiser minimises (much
cheaper than a term-by-term Hamiltonian expectation, and what makes deeper p
practical).

Two mixers are supported:

* ``"x"`` (default) -- transverse-field X mixer. Does not conserve cardinality, so
  the constraint ``|S| = k`` lives in the cost Hamiltonian as a soft penalty.
* ``"xy"`` -- ring XY mixer (see :mod:`qagent.qaoa.xy_mixer`). Preserves Hamming
  weight, so starting from a feasible weight-``k`` state needs **no penalty**
  (cost = -score) and every measured bitstring already has ``|S| = k``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pennylane as qml

from qagent.qaoa.encoding import ToolScoring, bitstring_to_subset, score_subset
from qagent.qaoa.xy_mixer import apply_xy_ring_mixer, feasible_init


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
    """Return ``(h, zz)``: single-qubit ``Z_i`` and ``Z_i Z_j`` coefficients for
    the cost ``-score(x) + penalty * (sum_i x_i - k)^2`` with ``x_i = (1 - Z_i)/2``
    (constant offset dropped). With ``penalty = 0`` this is just ``-score``."""
    n = scoring.n_tools
    lam = penalty
    w = scoring.weights
    syn = scoring.synergy
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


def _diagonal_energies(h: list[float], zz: dict[tuple[int, int], float], n: int) -> Any:
    """Energy E(x) of every computational basis state (Z eigenvalues), ordered to
    match ``qml.probs(wires=0..n-1)`` (wire 0 is the most significant bit)."""
    dim = 1 << n
    idx = np.arange(dim)
    z = np.empty((dim, n), dtype=np.float64)
    for i in range(n):
        z[:, i] = 1.0 - 2.0 * ((idx >> (n - 1 - i)) & 1)
    energies = z @ np.asarray(h, dtype=np.float64)
    for (i, m), c in zz.items():
        energies += c * z[:, i] * z[:, m]
    return energies


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
    mixer: str


def _apply_cost_layer(gamma: Any, n: int, h: list[float], zz: dict[tuple[int, int], float]) -> None:
    for i in range(n):
        if abs(h[i]) > 1e-12:
            qml.RZ(2.0 * gamma * h[i], wires=i)
    for (i, m), c in zz.items():
        if abs(c) > 1e-12:
            qml.IsingZZ(2.0 * gamma * c, wires=[i, m])


def _apply_ansatz(
    gammas: Any,
    betas: Any,
    n: int,
    k: int,
    p: int,
    h: list[float],
    zz: dict[tuple[int, int], float],
    mixer: str,
) -> None:
    if mixer == "xy":
        feasible_init(k, n)
    else:
        for w in range(n):
            qml.Hadamard(wires=w)
    for layer in range(p):
        _apply_cost_layer(gammas[layer], n, h, zz)
        if mixer == "xy":
            apply_xy_ring_mixer(betas[layer], n)
        else:
            for i in range(n):
                qml.RX(2.0 * betas[layer], wires=i)


def solve_qaoa(
    scoring: ToolScoring,
    k: int,
    *,
    p: int = 4,
    steps: int = 160,
    lr: float = 0.1,
    shots: int = 1024,
    penalty: float | None = None,
    seed: int = 0,
    mixer: str = "x",
) -> QAOAResult:
    """Optimise a ``p``-layer QAOA for select-``k`` and return the best measured
    size-``k`` subset.

    ``mixer="x"`` uses the X mixer with a cardinality penalty; ``mixer="xy"`` uses
    the Hamming-weight-preserving ring XY mixer with no penalty. Parameters are
    trained with PyTorch/Adam against the (diagonal) cost expectation, then the
    answer is read out by sampling and keeping the highest-scoring measured size-k
    subset.
    """
    import torch

    if mixer not in ("x", "xy"):
        raise ValueError(f"unknown mixer {mixer!r}; use 'x' or 'xy'")

    n = scoring.n_tools
    penalty_used = (
        0.0 if mixer == "xy" else (default_penalty(scoring) if penalty is None else penalty)
    )
    h, zz = cost_coefficients(scoring, k, penalty_used)
    energies = torch.tensor(_diagonal_energies(h, zz, n), dtype=torch.float64)

    dev = qml.device("default.qubit", wires=n, seed=seed)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def probs_qnode(gammas: Any, betas: Any) -> Any:
        _apply_ansatz(gammas, betas, n, k, p, h, zz, mixer)
        return qml.probs(wires=list(range(n)))

    torch.manual_seed(seed)
    gammas = torch.full((p,), 0.1, dtype=torch.float64, requires_grad=True)
    betas = torch.full((p,), 0.1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([gammas, betas], lr=lr)

    final_cost = 0.0
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.dot(probs_qnode(gammas, betas), energies)
        loss.backward()
        opt.step()
        final_cost = float(loss.detach())

    g = [float(x) for x in gammas.detach().numpy()]
    bt = [float(x) for x in betas.detach().numpy()]

    sample_dev = qml.device("default.qubit", wires=n, seed=seed)

    @qml.set_shots(shots=shots)
    @qml.qnode(sample_dev)
    def sample_circuit() -> Any:
        _apply_ansatz(g, bt, n, k, p, h, zz, mixer)
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
            mixer=mixer,
        )

    # Fallback: no size-k state sampled -> most probable size-k basis state.
    @qml.qnode(qml.device("default.qubit", wires=n))
    def probs_only() -> Any:
        _apply_ansatz(g, bt, n, k, p, h, zz, mixer)
        return qml.probs(wires=list(range(n)))

    probs = np.asarray(probs_only())
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
        mixer=mixer,
    )
