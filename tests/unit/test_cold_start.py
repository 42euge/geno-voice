"""Tests for iter-066 — cold-start latency penalty metric.

Metric 1.20 from docs/perf-metrics-taxonomy.md.

    cold_start_penalty = metrics_list[0].ttfs - median(m.ttfs for m in metrics_list[1:] if m.ttfs > 0)

Captures lazy initialization that hits turn 1 disproportionately —
model load, speaker open, TTS warmup, lazy imports — and would
otherwise get buried in the overall TTFS median.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    TurnMetrics,
    print_session_summary,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _m(ttfs: float = 0.0) -> TurnMetrics:
    return TurnMetrics(ttfs=ttfs)


def _summary(metrics_list, **kwargs):
    out = io.StringIO()
    print_session_summary(
        metrics_list, {"model": "stub"}, file=out, **kwargs,
    )
    return _strip_ansi(out.getvalue())


# ---- Boundary cases -----------------------------------------------------


class TestNoEmit:
    def test_single_turn_omits(self):
        # Need ≥2 measurable TTFS values total (turn 1 + steady state).
        plain = _summary([_m(ttfs=0.5)])
        assert "Cold start" not in plain

    def test_no_turns_omits(self):
        # No-completed-turn header path.
        plain = _summary([])
        assert "Cold start" not in plain

    def test_turn1_no_ttfs_omits(self):
        # Turn 1 without measurable TTFS (worker error / barge-in
        # before audio) → can't compute the penalty.
        plain = _summary([_m(ttfs=0.0), _m(ttfs=0.5), _m(ttfs=0.5)])
        assert "Cold start" not in plain

    def test_no_steady_state_omits(self):
        # Turn 1 has TTFS but turns 2:N all 0 → no comparison.
        plain = _summary([_m(ttfs=0.5), _m(ttfs=0.0), _m(ttfs=0.0)])
        assert "Cold start" not in plain

    def test_below_jitter_floor_omits(self):
        # |penalty| ≤ 50ms is within natural turn-to-turn jitter.
        plain = _summary([_m(ttfs=0.520), _m(ttfs=0.500), _m(ttfs=0.500)])
        assert "Cold start" not in plain


# ---- Positive penalty (typical cold-start) -----------------------------


class TestColdStartPenalty:
    def test_two_turn_session_emits(self):
        # Just turn 1 + 1 steady-state turn — minimum viable.
        plain = _summary([_m(ttfs=0.800), _m(ttfs=0.500)])
        # 800 - 500 = +300ms penalty.
        assert "Cold start:       +300ms vs steady state" in plain

    def test_multi_turn_uses_median(self):
        # Median of [0.50, 0.55, 0.60] = 0.55. Penalty = 0.90 - 0.55 = +350ms.
        plain = _summary([
            _m(ttfs=0.900),
            _m(ttfs=0.500),
            _m(ttfs=0.550),
            _m(ttfs=0.600),
        ])
        assert "Cold start:       +350ms vs steady state" in plain

    def test_zero_steady_state_filtered(self):
        # Steady-state turns with ttfs=0 (worker error / barge) are
        # excluded from the median, not treated as 0.
        plain = _summary([
            _m(ttfs=0.800),  # turn 1
            _m(ttfs=0.0),    # excluded
            _m(ttfs=0.500),  # included
            _m(ttfs=0.500),  # included
        ])
        # Median of [0.500, 0.500] = 0.500. 800 - 500 = +300.
        assert "Cold start:       +300ms vs steady state" in plain


# ---- Negative penalty (rare — turn 1 was faster) ----------------------


class TestNegativePenalty:
    def test_turn1_faster_than_steady_state(self):
        # Turn 1: 400ms. Steady: 500ms. Penalty = -100ms — turn 1
        # was actually faster. Sign is preserved with a leading "-".
        plain = _summary([_m(ttfs=0.400), _m(ttfs=0.500), _m(ttfs=0.500)])
        assert "Cold start:       -100ms vs steady state" in plain


# ---- Boundary at threshold ------------------------------------------


class TestThresholdBoundary:
    def test_just_below_threshold_omits(self):
        # |penalty| ≤ 50ms is below the chunk-noise floor.
        # Use 49ms to dodge float-precision noise around the boundary.
        plain = _summary([_m(ttfs=0.549), _m(ttfs=0.500), _m(ttfs=0.500)])
        assert "Cold start" not in plain

    def test_just_above_emits(self):
        # 51ms above floor → emits.
        plain = _summary([_m(ttfs=0.551), _m(ttfs=0.500), _m(ttfs=0.500)])
        assert "Cold start:       +51ms vs steady state" in plain
