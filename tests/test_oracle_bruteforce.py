"""Tests for the scoring model, encoding, and classical baselines."""

from __future__ import annotations

import pytest

from qagent.oracle.bruteforce import brute_force_best
from qagent.oracle.greedy import greedy_top_k
from qagent.qaoa.encoding import (
    ToolScoring,
    all_subsets_of_size,
    bitstring_to_subset,
    score_subset,
    subset_to_bitstring,
)


def test_score_subset_linear_plus_pairs() -> None:
    w = (1.0, 2.0, 3.0)
    syn = ((0.0, 0.5, 0.0), (0.5, 0.0, -1.0), (0.0, -1.0, 0.0))
    sc = ToolScoring(weights=w, synergy=syn)
    assert score_subset(sc, set()) == 0.0
    assert score_subset(sc, {0, 1}) == pytest.approx(1 + 2 + 0.5)
    assert score_subset(sc, {1, 2}) == pytest.approx(2 + 3 - 1.0)
    assert score_subset(sc, {0, 1, 2}) == pytest.approx(6 + 0.5 - 1.0)


def test_encoding_roundtrip() -> None:
    for sub in [set(), {0}, {0, 2}, {1, 2}]:
        bits = subset_to_bitstring(sub, 3)
        assert bitstring_to_subset(bits) == frozenset(sub)
    assert subset_to_bitstring({0, 2}, 3) == "101"  # tool 0 leftmost


def test_all_subsets_counts() -> None:
    assert sum(1 for _ in all_subsets_of_size(16, 5)) == 4368
    assert sum(1 for _ in all_subsets_of_size(8, 3)) == 56


def test_bruteforce_beats_greedy_on_synergy() -> None:
    w = (0.9, 0.85, 0.5, 0.5)
    syn = [[0.0] * 4 for _ in range(4)]
    syn[2][3] = syn[3][2] = 1.0  # mid pair {2,3} synergises past top-2 {0,1}
    sc = ToolScoring(weights=w, synergy=tuple(tuple(r) for r in syn))
    bf_sub, bf_score = brute_force_best(sc, 2)
    gd_sub, gd_score = greedy_top_k(sc, 2)
    assert set(bf_sub) == {2, 3} and bf_score == pytest.approx(2.0)
    assert set(gd_sub) == {0, 1} and gd_score == pytest.approx(1.75)
    assert bf_score > gd_score


def test_greedy_optimal_without_synergy() -> None:
    w = (0.1, 0.9, 0.5, 0.7)
    zero = tuple(tuple(0.0 for _ in range(4)) for _ in range(4))
    sc = ToolScoring(weights=w, synergy=zero)
    assert greedy_top_k(sc, 2)[1] == pytest.approx(brute_force_best(sc, 2)[1])


def test_validation_errors() -> None:
    with pytest.raises(ValueError):
        ToolScoring(weights=(), synergy=())
    with pytest.raises(ValueError):
        ToolScoring(weights=(1.0,), synergy=((0.0, 0.0),))  # wrong shape
    sc = ToolScoring(weights=(1.0, 2.0), synergy=((0.0, 0.0), (0.0, 0.0)))
    with pytest.raises(ValueError):
        score_subset(sc, {5})
    with pytest.raises(ValueError):
        list(all_subsets_of_size(3, 5))
    with pytest.raises(ValueError):
        bitstring_to_subset("012")
    with pytest.raises(ValueError):
        greedy_top_k(sc, 9)
