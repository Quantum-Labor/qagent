"""QAOA tool-selection: one-hot encoding, cost Hamiltonian, and ansatz.

The circuit module (:mod:`qagent.qaoa.circuit`) imports PennyLane/PyTorch, so it is
not imported here — import it explicitly when you need the solver.
"""

from __future__ import annotations

from qagent.qaoa.encoding import (
    ToolScoring,
    all_subsets_of_size,
    bitstring_to_subset,
    score_subset,
    subset_to_bitstring,
)

__all__ = [
    "ToolScoring",
    "all_subsets_of_size",
    "bitstring_to_subset",
    "score_subset",
    "subset_to_bitstring",
]
