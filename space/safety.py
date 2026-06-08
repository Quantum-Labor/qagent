"""Lightweight rate limiting for the Space's live-solve path.

The Space serves precomputed QAOA results instantly, but it also runs the
pure-Python classical solvers (brute-force, greedy) live so visitors can confirm
the served numbers are real. Those solves are cheap (milliseconds), but a public
endpoint still warrants a guard against rapid-fire hammering. Two layers:

1. **Per-IP rate limit** - at most one live solve per ``window_seconds`` per
   visitor IP (read from ``x-forwarded-for``, set by HF's reverse proxy).
2. **Global daily cap** - a generous ceiling on live solves per UTC day across
   all visitors, persisted to JSON when a writable dir is available.

There is no IBM-quota guard here (this Space has no hardware path). Verdicts are
returned as a dataclass so the Gradio layer stays thin and the logic is trivially
unit-testable.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

SafetyReason = Literal["rate_limited", "daily_cap"]


@dataclass(frozen=True)
class SafetyVerdict:
    """Outcome of a :meth:`RateLimiter.check_and_register` call."""

    allowed: bool
    reason: SafetyReason | None
    detail: str
    daily_remaining: int
    daily_cap: int


class RateLimiter:
    """Per-IP + daily gate for the live-solve path. Thread-safe via one lock."""

    def __init__(
        self,
        *,
        window_seconds: int = 3,
        daily_cap: int = 5000,
        persist_path: Path | None = None,
    ) -> None:
        if window_seconds < 0:
            raise ValueError("window_seconds must be >= 0")
        if daily_cap < 0:
            raise ValueError("daily_cap must be >= 0")
        self._window = window_seconds
        self._cap = daily_cap
        self._persist_path = persist_path
        self._lock = threading.Lock()
        self._last_ip: dict[str, float] = {}
        self._day: date | None = None
        self._count: int = 0
        self._load_persisted()

    def check_and_register(self, *, ip: str, now: datetime) -> SafetyVerdict:
        """Decide whether ``ip`` may run a live solve at ``now`` (commit-on-allow)."""
        with self._lock:
            self._roll_day_if_needed(now)
            self._evict_stale_ips(now.timestamp())

            if self._count >= self._cap:
                return SafetyVerdict(
                    allowed=False,
                    reason="daily_cap",
                    detail=f"Daily limit of {self._cap} live solves reached. Resets midnight UTC.",
                    daily_remaining=0,
                    daily_cap=self._cap,
                )

            last = self._last_ip.get(ip)
            if last is not None and (now.timestamp() - last) < self._window:
                wait = self._window - int(now.timestamp() - last)
                return SafetyVerdict(
                    allowed=False,
                    reason="rate_limited",
                    detail=f"Please wait {max(wait, 1)}s between live solves.",
                    daily_remaining=max(0, self._cap - self._count),
                    daily_cap=self._cap,
                )

            self._last_ip[ip] = now.timestamp()
            self._count += 1
            self._persist()
            return SafetyVerdict(
                allowed=True,
                reason=None,
                detail="ok",
                daily_remaining=max(0, self._cap - self._count),
                daily_cap=self._cap,
            )

    def daily_remaining(self, now: datetime) -> int:
        with self._lock:
            self._roll_day_if_needed(now)
            return max(0, self._cap - self._count)

    def daily_cap(self) -> int:
        return self._cap

    def _roll_day_if_needed(self, now: datetime) -> None:
        today = now.astimezone(UTC).date()
        if self._day != today:
            self._day = today
            self._count = 0

    def _evict_stale_ips(self, now_ts: float) -> None:
        if self._window <= 0:
            self._last_ip.clear()
            return
        cutoff = now_ts - self._window
        self._last_ip = {ip: ts for ip, ts in self._last_ip.items() if ts > cutoff}

    def _load_persisted(self) -> None:
        if self._persist_path is None:
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        try:
            self._day = date.fromisoformat(str(raw.get("date")))
            self._count = int(raw.get("count", 0))
        except (TypeError, ValueError):
            self._day = None
            self._count = 0

    def _persist(self) -> None:
        if self._persist_path is None or self._day is None:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(
                json.dumps({"date": self._day.isoformat(), "count": self._count}),
                encoding="utf-8",
            )
        except OSError:
            self._persist_path = None


def default_persist_path() -> Path | None:
    """Return ``/data/qagent_quota.json`` when HF Persistent Storage is mounted."""
    candidate = Path("/data")
    if candidate.is_dir():
        try:
            test = candidate / ".qa_write_test"
            test.write_text("x")
            test.unlink()
        except OSError:
            return None
        return candidate / "qagent_quota.json"
    return None
