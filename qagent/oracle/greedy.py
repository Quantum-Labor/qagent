"""Greedy top-k baseline.

Picks the ``k`` tools with the highest standalone relevance, ignoring pairwise
synergy entirely. This is the weak baseline QAOA is meant to beat: whenever the
optimal subset depends on tool-pair interactions, greedy gets it wrong.
"""

from __future__ import annotations

from qagent.qaoa.encoding import ToolScoring, score_subset


def greedy_top_k(scoring: ToolScoring, k: int) -> tuple[frozenset[int], float]:
    """Return ``(subset, score)`` for the ``k`` highest-weight tools.

    Ties on weight are broken by tool index (lower index first) for determinism.
    """
    if not 0 <= k <= scoring.n_tools:
        raise ValueError(f"k={k} out of range [0, {scoring.n_tools}]")
    order = sorted(range(scoring.n_tools), key=lambda i: (-scoring.weights[i], i))
    subset = frozenset(order[:k])
    return subset, score_subset(scoring, subset)
