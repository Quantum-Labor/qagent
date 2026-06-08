"""Ring XY mixer for QAOA tool selection (Hamming-weight preserving).

A standard transverse-field X mixer flips individual qubits, so it does not
conserve the number of selected tools and the cardinality constraint must be
enforced with a soft penalty in the cost Hamiltonian. The XY mixer

    H_XY = sum_i (X_i X_{i+1} + Y_i Y_{i+1})      (indices mod n: a ring)

only swaps excitations between neighbouring qubits, so it preserves the Hamming
weight. Initialised on a feasible state of weight ``k``, QAOA then explores only
the size-``k`` subspace and **no cardinality penalty is needed at all** (cost =
-score). See Wang et al. 2020, "XY mixers: analytical and numerical results for
the quantum alternating operator ansatz".

The ring mixer is applied as a first-order Trotter step: one parameterised
``IsingXY`` rotation per ring bond (even bonds then odd bonds to reduce ordering
bias). ``IsingXY`` is exactly the (XX+YY) two-qubit rotation, so each bond term
preserves Hamming weight individually.
"""

from __future__ import annotations

from typing import Any

import pennylane as qml


def feasible_init(k: int, n: int) -> None:
    """Prepare the computational basis state with exactly ``k`` ones (tools
    0..k-1 selected) -- a valid weight-``k`` starting point for the XY mixer."""
    if not 0 <= k <= n:
        raise ValueError(f"k={k} out of range [0, {n}]")
    for i in range(k):
        qml.PauliX(wires=i)


def ring_bonds(n: int) -> list[tuple[int, int]]:
    """Even ring bonds followed by odd ring bonds (mod n)."""
    even = [(i, (i + 1) % n) for i in range(0, n, 2)]
    odd = [(i, (i + 1) % n) for i in range(1, n, 2)]
    return even + odd


def apply_xy_ring_mixer(beta: Any, n: int) -> None:
    """Apply one Trotter step of ``exp(-i beta H_XY)`` over the ring (n >= 2).

    The overall sign/scale of ``beta`` is absorbed by the variational optimiser,
    so a single trained angle per layer is shared across all bonds.
    """
    if n < 2:
        raise ValueError("XY mixer needs at least 2 qubits")
    for i, j in ring_bonds(n):
        qml.IsingXY(2.0 * beta, wires=[i, j])
