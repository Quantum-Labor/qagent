"""Exact brute-force solver for select-k tool selection.

Enumerates every size-``k`` subset and returns the highest-scoring one. For the
sizes QAgent targets this is trivial (C(16, 5) = 4368), which makes it the perfect
ground-truth oracle for validating QAOA and greedy.
"""

from __future__ import annotations

from qagent.qaoa.encoding import ToolScoring, all_subsets_of_size, score_subset


def brute_force_best(scoring: ToolScoring, k: int) -> tuple[frozenset[int], float]:
    """Return ``(best_subset, best_score)`` over all size-``k`` subsets.

    Ties are broken by the lexicographic order of :func:`all_subsets_of_size`, so
    the result is deterministic. Callers that compare solvers should compare the
    *score* (ties mean several subsets are equally optimal).
    """
    best_subset: frozenset[int] | None = None
    best_score = float("-inf")
    for subset in all_subsets_of_size(scoring.n_tools, k):
        s = score_subset(scoring, subset)
        if s > best_score:
            best_score = s
            best_subset = subset
    if best_subset is None:
        raise ValueError("no subsets enumerated; check n_tools and k")
    return best_subset, best_score
