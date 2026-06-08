"""Tests for the QAOA cost Hamiltonian and ansatz."""

from __future__ import annotations

import itertools
import random

import pytest

from qagent.oracle.bruteforce import brute_force_best
from qagent.qaoa.circuit import (
    cost_coefficients,
    cost_hamiltonian,
    default_penalty,
)
from qagent.qaoa.encoding import ToolScoring, bitstring_to_subset, score_subset


def _make(n: int, seed: int) -> ToolScoring:
    rng = random.Random(seed)
    w = tuple(round(rng.uniform(0.0, 1.0), 3) for _ in range(n))
    syn = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            v = round(rng.uniform(-0.5, 0.8), 3)
            syn[i][j] = syn[j][i] = v
    return ToolScoring(weights=w, synergy=tuple(tuple(r) for r in syn))


def _classical_energy(h: list[float], zz: dict[tuple[int, int], float], bits: str) -> float:
    # Z eigenvalue: bit 0 -> +1, bit 1 -> -1.
    z = [1 - 2 * int(c) for c in bits]
    e = sum(h[i] * z[i] for i in range(len(h)))
    for (i, j), c in zz.items():
        e += c * z[i] * z[j]
    return e


@pytest.mark.parametrize("n,k", [(6, 2), (7, 3), (8, 4)])
def test_cost_hamiltonian_ground_state_is_the_optimal_subset(n: int, k: int) -> None:
    """The diagonal of the cost Hamiltonian must rank the optimal size-k subset
    lowest, and the penalty must make the global minimum have exactly |S| = k."""
    scoring = _make(n, seed=n * 13 + k)
    h, zz = cost_coefficients(scoring, k, default_penalty(scoring))

    best_bits = ""
    best_e = float("inf")
    for combo in itertools.product("01", repeat=n):
        bits = "".join(combo)
        e = _classical_energy(h, zz, bits)
        if e < best_e:
            best_e = e
            best_bits = bits

    subset = bitstring_to_subset(best_bits)
    assert len(subset) == k, "penalty did not force cardinality k"
    _, bf_score = brute_force_best(scoring, k)
    assert score_subset(scoring, subset) == pytest.approx(bf_score)


def test_default_penalty_is_positive() -> None:
    assert default_penalty(_make(5, 1)) > 0.0


def test_cost_hamiltonian_builds_nonempty() -> None:
    h_op = cost_hamiltonian(_make(4, 2), 2)
    assert len(h_op.ops) >= 1


def test_qaoa_smoke_returns_size_k() -> None:
    """Fast path coverage: tiny QAOA returns a valid size-k subset (not asserting
    convergence in so few steps)."""
    from qagent.qaoa.circuit import solve_qaoa

    res = solve_qaoa(_make(4, 7), 2, p=1, steps=3, shots=128, seed=1)
    assert len(res.subset) == 2
    assert res.n_layers == 1


@pytest.mark.slow
def test_qaoa_finds_optimum_on_synergy_case() -> None:
    w = (0.9, 0.85, 0.5, 0.5, 0.05, 0.05)
    syn = [[0.0] * 6 for _ in range(6)]
    syn[2][3] = syn[3][2] = 1.2
    scoring = ToolScoring(weights=w, synergy=tuple(tuple(r) for r in syn))
    from qagent.qaoa.circuit import solve_qaoa

    res = solve_qaoa(scoring, 2, p=2, steps=80, shots=512, seed=3)
    _, bf_score = brute_force_best(scoring, 2)
    assert res.score == pytest.approx(bf_score)
    assert set(res.subset) == {2, 3}


# --- XY mixer (Hamming-weight preserving) -----------------------------------


def test_xy_ring_bonds_cover_the_ring() -> None:
    from qagent.qaoa.xy_mixer import ring_bonds

    bonds = ring_bonds(6)
    assert len(bonds) == 6
    nodes = {x for bond in bonds for x in bond}
    assert nodes == set(range(6))


def test_xy_mixer_preserves_cardinality() -> None:
    """By construction every measured subset has size k (weight is preserved),
    regardless of depth -- so even an untrained shallow run returns a valid set."""
    from qagent.qaoa.circuit import solve_qaoa

    res = solve_qaoa(_make(6, 5), 3, p=1, steps=3, shots=128, seed=0, mixer="xy")
    assert len(res.subset) == 3
    assert res.mixer == "xy"


@pytest.mark.slow
def test_xy_mixer_finds_optimum_when_deep() -> None:
    # The single-basis-state init needs deep p to mix (Dicke init would help).
    w = (0.9, 0.85, 0.5, 0.5, 0.05, 0.05)
    syn = [[0.0] * 6 for _ in range(6)]
    syn[2][3] = syn[3][2] = 1.2
    scoring = ToolScoring(weights=w, synergy=tuple(tuple(r) for r in syn))
    from qagent.qaoa.circuit import solve_qaoa

    res = solve_qaoa(scoring, 2, p=8, steps=300, lr=0.25, shots=512, seed=0, mixer="xy")
    assert set(res.subset) == {2, 3}
