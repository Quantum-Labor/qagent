"""High-level tool selector: a scoring model plus k, in, selected tools, out.

Two backends:

* ``"classical"`` — exact brute-force optimum (the ground truth).
* ``"qaoa"`` — the QAOA solver on the PennyLane simulator.

The greedy top-k baseline lives in :mod:`qagent.oracle.greedy` and is exposed for
benchmarking, not as a selector backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from qagent.oracle.bruteforce import brute_force_best
from qagent.qaoa.encoding import ToolScoring

Backend = Literal["qaoa", "classical"]


@dataclass(frozen=True)
class Selection:
    """A chosen set of tools and its score, tagged with the backend that found it."""

    subset: frozenset[int]
    score: float
    backend: str


def select_tools(
    scoring: ToolScoring,
    k: int,
    *,
    backend: Backend = "classical",
    **qaoa_kwargs: Any,
) -> Selection:
    """Select ``k`` tools under ``scoring`` using ``backend``.

    ``qaoa_kwargs`` (p, steps, lr, shots, penalty, seed) are forwarded to
    :func:`qagent.qaoa.circuit.solve_qaoa` and ignored for the classical backend.
    """
    if backend == "classical":
        subset, score = brute_force_best(scoring, k)
        return Selection(subset=subset, score=score, backend="classical")
    if backend == "qaoa":
        from qagent.qaoa.circuit import solve_qaoa

        result = solve_qaoa(scoring, k, **qaoa_kwargs)
        return Selection(subset=result.subset, score=result.score, backend="qaoa")
    raise ValueError(f"unknown backend {backend!r}; use 'qaoa' or 'classical'")
