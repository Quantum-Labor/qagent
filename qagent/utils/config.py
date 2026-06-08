"""Default configuration for QAgent tool selection."""

from __future__ import annotations

from dataclasses import dataclass

# The locked Phase-1 task: pick k=5 tools out of N=16 candidates.
DEFAULT_N_TOOLS = 16
DEFAULT_K = 5


@dataclass(frozen=True)
class QAOAConfig:
    """Hyper-parameters for the QAOA solver (see qagent.qaoa.circuit.solve_qaoa)."""

    n_layers: int = 2
    steps: int = 60
    learning_rate: float = 0.1
    shots: int = 1024
    seed: int = 0
