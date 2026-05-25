"""Tests for iter-068 — TTFS jitter (turn-taking jitter).

Metric 1.12 from docs/perf-metrics-taxonomy.md.

    jitter = stdev(ttfs for ttfs in metrics_list if ttfs > 0)

The same stdev that feeds iter-055's rhythm score, promoted to
its own line so operators can read it as an absolute number when
tuning. Humans tolerate consistent slow turn-taking better than
inconsistent fast turn-taking — the jitter is the more actionable
number even when the rhythm score looks fine.
"""

from __future__ import annotations

import io
import re
import statistics
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


# ---- No-emit cases ---------------------------------------------------


class TestNoEmit:
    def test_zero_turns_omits(self):
        plain = _summary([])
        assert "TTFS jitter" not in plain

    def test_single_measurable_turn_omits(self):
        # Stdev requires ≥2 samples — same gating as the rhythm score.
        plain = _summary([_m(ttfs=0.5)])
        assert "TTFS jitter" not in plain

    def test_no_audio_turns_omits(self):
        # All turns ended without audio → ttfs_times is empty →
        # the whole block is omitted.
        plain = _summary([_m(ttfs=0.0), _m(ttfs=0.0)])
        assert "TTFS jitter" not in plain


# ---- Emit cases ------------------------------------------------------


class TestEmit:
    def test_two_turn_jitter(self):
        # Stdev of [400, 600] = 141.42... → rounds to 141ms.
        plain = _summary([_m(ttfs=0.400), _m(ttfs=0.600)])
        # Compute expected to keep the test rounding-tolerant.
        expected = round(statistics.stdev([0.400, 0.600]) * 1000)
        assert f"TTFS jitter:      ±{expected}ms" in plain

    def test_uniform_turns_zero_jitter(self):
        # All identical → stdev = 0 → "±0ms".
        plain = _summary([_m(ttfs=0.5), _m(ttfs=0.5), _m(ttfs=0.5)])
        assert "TTFS jitter:      ±0ms" in plain

    def test_high_variance(self):
        plain = _summary([
            _m(ttfs=0.300),
            _m(ttfs=0.500),
            _m(ttfs=0.700),
            _m(ttfs=1.500),
        ])
        expected = round(statistics.stdev([0.3, 0.5, 0.7, 1.5]) * 1000)
        assert f"TTFS jitter:      ±{expected}ms" in plain

    def test_zero_ttfs_filtered(self):
        # Same filter as ttfs_times — turns with ttfs=0 (worker
        # error / barge before audio) excluded from the stdev.
        plain = _summary([
            _m(ttfs=0.0),     # excluded
            _m(ttfs=0.400),
            _m(ttfs=0.600),
        ])
        # Only [0.4, 0.6] in the sample.
        expected = round(statistics.stdev([0.4, 0.6]) * 1000)
        assert f"TTFS jitter:      ±{expected}ms" in plain


# ---- Co-emission with rhythm score -----------------------------------


class TestCoEmissionWithRhythmScore:
    def test_both_lines_present_on_2plus_turns(self):
        plain = _summary([_m(ttfs=0.400), _m(ttfs=0.600)])
        assert "Rhythm score" in plain
        assert "TTFS jitter" in plain

    def test_neither_when_under_2_turns(self):
        plain = _summary([_m(ttfs=0.5)])
        assert "Rhythm score" not in plain
        assert "TTFS jitter" not in plain
