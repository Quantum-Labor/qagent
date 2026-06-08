"""One-hot encoding of tool subsets and the subset scoring model.

A tool selection is a subset ``S`` of ``{0, ..., n_tools - 1}``. The one-hot
encoding maps it to an ``n_tools``-bit string where bit ``i`` is ``"1"`` iff tool
``i`` is in ``S`` (tool 0 is the leftmost character, matching the qubit-0-leftmost
convention used throughout the package).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class ToolScoring:
    """Scoring model for one tool-selection task.

    ``weights[i]`` is the standalone relevance of tool ``i`` to the task.
    ``synergy[i][j]`` (symmetric, zero diagonal) is the pairwise bonus (positive)
    or redundancy penalty (negative) for selecting both tools ``i`` and ``j``.

    The score of a subset ``S`` is::

        score(S) = sum_{i in S} weights[i] + sum_{i<j in S} synergy[i][j]

    The pair term is what makes optimal selection more than a greedy top-k: two
    mid-relevance synergistic tools can beat two high-relevance redundant ones.
    """

    weights: tuple[float, ...]
    synergy: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        n = len(self.weights)
        if n == 0:
            raise ValueError("weights must be non-empty")
        if len(self.synergy) != n or any(len(row) != n for row in self.synergy):
            raise ValueError(f"synergy must be {n}x{n} to match {n} weights")

    @property
    def n_tools(self) -> int:
        return len(self.weights)


def score_subset(scoring: ToolScoring, subset: Iterable[int]) -> float:
    """Return ``score(subset)`` under ``scoring``. Raises on out-of-range indices."""
    s = sorted(set(subset))
    n = scoring.n_tools
    for i in s:
        if not 0 <= i < n:
            raise ValueError(f"tool index {i} out of range [0, {n})")
    total = sum(scoring.weights[i] for i in s)
    for a in range(len(s)):
        for b in range(a + 1, len(s)):
            total += scoring.synergy[s[a]][s[b]]
    return float(total)


def subset_to_bitstring(subset: Iterable[int], n_tools: int) -> str:
    """Render ``subset`` as an ``n_tools``-bit one-hot string (tool 0 leftmost)."""
    sel = set(subset)
    return "".join("1" if i in sel else "0" for i in range(n_tools))


def bitstring_to_subset(bits: str) -> frozenset[int]:
    """Inverse of :func:`subset_to_bitstring`."""
    if any(c not in "01" for c in bits):
        raise ValueError(f"bitstring may only contain '0'/'1', got {bits!r}")
    return frozenset(i for i, c in enumerate(bits) if c == "1")


def all_subsets_of_size(n_tools: int, k: int) -> Iterator[frozenset[int]]:
    """Yield every size-``k`` subset of ``{0, ..., n_tools - 1}``."""
    if not 0 <= k <= n_tools:
        raise ValueError(f"k={k} out of range [0, {n_tools}]")
    for combo in combinations(range(n_tools), k):
        yield frozenset(combo)
