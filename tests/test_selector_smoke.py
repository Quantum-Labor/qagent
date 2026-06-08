"""Smoke tests for the high-level selector and package surface."""

from __future__ import annotations

import pytest

import qagent
from qagent.oracle.bruteforce import brute_force_best
from qagent.qaoa.encoding import ToolScoring
from qagent.selector import select_tools


def _scoring() -> ToolScoring:
    w = (0.9, 0.85, 0.5, 0.5, 0.1)
    syn = [[0.0] * 5 for _ in range(5)]
    syn[2][3] = syn[3][2] = 1.0
    return ToolScoring(weights=w, synergy=tuple(tuple(r) for r in syn))


def test_version_is_semver() -> None:
    assert isinstance(qagent.__version__, str)
    assert qagent.__version__.count(".") == 2


def test_classical_backend_is_exact() -> None:
    sc = _scoring()
    sel = select_tools(sc, 2, backend="classical")
    bf_sub, bf_score = brute_force_best(sc, 2)
    assert sel.backend == "classical"
    assert set(sel.subset) == set(bf_sub)
    assert sel.score == pytest.approx(bf_score)


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError):
        select_tools(_scoring(), 2, backend="bogus")  # type: ignore[arg-type]


def test_qaoa_backend_smoke() -> None:
    """Fast: the qaoa backend wires through and returns a size-k selection."""
    sel = select_tools(_scoring(), 2, backend="qaoa", p=1, steps=3, shots=128, seed=2)
    assert sel.backend == "qaoa"
    assert len(sel.subset) == 2


@pytest.mark.slow
def test_qaoa_backend_finds_optimum() -> None:
    sc = _scoring()
    sel = select_tools(sc, 2, backend="qaoa", p=2, steps=80, shots=512, seed=4)
    assert sel.score == pytest.approx(brute_force_best(sc, 2)[1])
