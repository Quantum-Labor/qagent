"""Smoke tests for the HuggingFace Space app (space/app.py).

space/ is not a package; app.py is loaded by file path with space/ prepended to
sys.path so its ``from safety import ...`` resolves. Marked ``unit`` so the
default suite (``-m "not slow"``) covers them.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

_SPACE = Path(__file__).resolve().parent.parent / "space"


@pytest.fixture(scope="module")
def app() -> Any:
    if str(_SPACE) not in sys.path:
        sys.path.insert(0, str(_SPACE))
    spec = importlib.util.spec_from_file_location("_qa_space_app", _SPACE / "app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_qa_space_app"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_app_builds_demo(app: Any) -> None:
    assert app.demo is not None


@pytest.mark.unit
def test_precomputed_has_all_50_tasks(app: Any) -> None:
    assert len(app.TASKS) == 50
    tiers = [t["tier"] for t in app.TASKS]
    assert tiers.count("small") == 25
    assert tiers.count("full") == 25
    for t in app.TASKS:
        assert {"id", "optimal", "qaoa", "greedy", "score_hist", "tools"} <= set(t)
        assert len(t["tools"]) == t["n_tools"]


@pytest.mark.unit
def test_dataset_bundled_for_live_verify(app: Any) -> None:
    # Every task must have its source scoring bundled so live verification works.
    assert len(app.DATASET_BY_ID) == 50
    for t in app.TASKS:
        assert t["id"] in app.DATASET_BY_ID


@pytest.mark.unit
def test_render_helpers_produce_markup(app: Any) -> None:
    task = app.TASKS[0]
    grid = app.render_tool_grid(task)
    assert "qa-tool-grid" in grid and "qa-tool-card" in grid
    assert "qa-score-card" in app.render_scores(task)
    assert app.task_choices()[0].startswith(app.TASKS[0]["id"])


@pytest.mark.unit
def test_rate_limiter_blocks_rapid_calls(app: Any) -> None:
    rl = app.RateLimiter(window_seconds=300, daily_cap=10)
    t0 = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)
    assert rl.check_and_register(ip="1.1.1.1", now=t0).allowed
    v2 = rl.check_and_register(ip="1.1.1.1", now=t0 + timedelta(seconds=30))
    assert not v2.allowed
    assert v2.reason == "rate_limited"


@pytest.mark.unit
def test_leaderboard_markdown(app: Any) -> None:
    md = app.leaderboard_markdown()
    assert "QAOA" in md and "Brute-force" in md


@pytest.mark.unit
def test_verify_live_matches_served_optimal(app: Any) -> None:
    # Live brute-force must reproduce the served optimal score (faithfulness).
    label = app.task_choices()[0]
    out = app.verify_live(label, request=None)
    assert "matches" in out and "DIFFERS" not in out
