"""QAgent — QAOA-based tool selection for LLM agents.

Project 2 of 3 in the Quantum Co-Processor research program (after QVerify).
Importing this package is cheap: PennyLane/PyTorch are pulled in lazily only when
the QAOA backend actually runs.
"""

from __future__ import annotations

__version__ = "0.1.0"

from qagent.qaoa.encoding import ToolScoring, score_subset
from qagent.selector import Selection, select_tools

__all__ = ["Selection", "ToolScoring", "__version__", "score_subset", "select_tools"]
