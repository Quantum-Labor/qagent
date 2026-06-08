"""Classical baselines: exact brute-force solver and greedy top-k."""

from __future__ import annotations

from qagent.oracle.bruteforce import brute_force_best
from qagent.oracle.greedy import greedy_top_k

__all__ = ["brute_force_best", "greedy_top_k"]
