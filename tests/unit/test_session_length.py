"""Tests for iter-054 — session length + turns/min metric.

Metric 1.15 from docs/perf-metrics-taxonomy.md. Adds total wall-
clock time of the session (computed by mic_chat from
``time.monotonic`` deltas, passed as a kwarg) and derives the
turns-per-minute rate.

Header now reads:
    Session Summary (3 turns over 4m 30s)
        ...
        Turns/min:        0.7
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    TurnMetrics,
    print_session_summary,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _m(ttfs=0.5):
    return TurnMetrics(ttfs=ttfs)


# ---- Header duration formatting -----------------------------------------


class TestHeaderDuration:
    def test_no_session_seconds_omits_duration(self):
        out = io.StringIO()
        print_session_summary(
            [_m()], {"model": "stub"}, file=out,
            # No session_seconds passed.
        )
        plain = _strip_ansi(out.getvalue())
        assert "Session Summary (1 turn)" in plain
        assert "over" not in plain

    def test_zero_session_seconds_omits_duration(self):
        out = io.StringIO()
        print_session_summary(
            [_m()], {"model": "stub"}, file=out, session_seconds=0.0,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Session Summary (1 turn)" in plain
        assert "over" not in plain

    def test_short_session_seconds(self):
        # <60s → "Ns".
        out = io.StringIO()
        print_session_summary(
            [_m(), _m()], {"model": "stub"}, file=out, session_seconds=42.0,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Session Summary (2 turns over 42s)" in plain

    def test_minutes_no_seconds_remainder(self):
        # 5*60 = 300 → "5m".
        out = io.StringIO()
        print_session_summary(
            [_m()], {"model": "stub"}, file=out, session_seconds=300.0,
        )
        plain = _strip_ansi(out.getvalue())
        assert "(1 turn over 5m)" in plain

    def test_minutes_and_seconds(self):
        # 4m 30s.
        out = io.StringIO()
        print_session_summary(
            [_m()], {"model": "stub"}, file=out, session_seconds=270.0,
        )
        plain = _strip_ansi(out.getvalue())
        assert "(1 turn over 4m 30s)" in plain

    def test_hours(self):
        # 1h 30m.
        out = io.StringIO()
        print_session_summary(
            [_m()], {"model": "stub"}, file=out, session_seconds=5400.0,
        )
        plain = _strip_ansi(out.getvalue())
        assert "(1 turn over 1h 30m)" in plain


# ---- Turns/min ------------------------------------------------------------


class TestTurnsPerMin:
    def test_no_session_seconds_omits_tpm(self):
        out = io.StringIO()
        print_session_summary([_m()], {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        assert "Turns/min" not in plain

    def test_short_session_omits_tpm(self):
        # <1s session → too short to make a rate meaningful.
        out = io.StringIO()
        print_session_summary(
            [_m()], {"model": "stub"}, file=out, session_seconds=0.5,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Turns/min" not in plain

    def test_tpm_computed(self):
        # 3 turns over 60s = 3.0 turns/min.
        out = io.StringIO()
        print_session_summary(
            [_m(), _m(), _m()], {"model": "stub"}, file=out, session_seconds=60.0,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Turns/min:        3.0" in plain

    def test_tpm_low_rate(self):
        # 1 turn over 5m = 0.2 turns/min.
        out = io.StringIO()
        print_session_summary(
            [_m()], {"model": "stub"}, file=out, session_seconds=300.0,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Turns/min:        0.2" in plain

    def test_tpm_with_no_metrics(self):
        # Empty metrics list → early-return placeholder, no
        # Turns/min. Documents the existing behavior.
        out = io.StringIO()
        print_session_summary(
            [], {"model": "stub"}, file=out, session_seconds=120.0,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Turns/min" not in plain
        assert "no completed turns" in plain
