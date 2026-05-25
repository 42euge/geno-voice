"""Tests for iter-055 — conversation rhythm score.

Metric 3.2 from docs/perf-metrics-taxonomy.md ("Novel/speculative").
Captures cadence consistency, not raw speed:

    rhythm = 1 - stdev(ttfs) / median(ttfs)
    clamped to [0, 1]

A bot with consistent ~300ms TTFS has high rhythm; one that
oscillates between 100ms and 800ms has low rhythm even if its
median is the same. The taxonomy notes: "consistency feels like
a personality; jitter feels like a system."
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


def _m(ttfs):
    return TurnMetrics(ttfs=ttfs)


# ---- Suppression conditions ---------------------------------------------


class TestSuppression:
    def test_no_ttfs_omits_score(self):
        # All turns have ttfs=0 → no Best TTFS block reached, no rhythm.
        out = io.StringIO()
        print_session_summary([_m(0.0), _m(0.0)], {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        assert "Rhythm score" not in plain

    def test_single_turn_omits_score(self):
        # Need ≥2 turns for stdev to be defined.
        out = io.StringIO()
        print_session_summary([_m(0.5)], {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        assert "Rhythm score" not in plain


# ---- Score computation --------------------------------------------------


class TestScoreValues:
    def test_perfectly_consistent_high_score(self):
        # All TTFS equal → stdev = 0 → rhythm = 1.00.
        out = io.StringIO()
        print_session_summary(
            [_m(0.3), _m(0.3), _m(0.3)], {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Rhythm score:     1.00" in plain

    def test_moderate_jitter_mid_score(self):
        # 200ms, 300ms, 400ms → median 300ms, stdev=100ms → 1 - 100/300 = 0.67.
        out = io.StringIO()
        print_session_summary(
            [_m(0.2), _m(0.3), _m(0.4)], {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Rhythm score:     0.67" in plain

    def test_high_jitter_clamped_low(self):
        # Wide spread: 50ms, 100ms, 800ms → median 100ms, stdev~410ms.
        # Raw = 1 - 410/100 = -3.10 → clamped to 0.
        out = io.StringIO()
        print_session_summary(
            [_m(0.05), _m(0.1), _m(0.8)], {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Rhythm score:     0.00" in plain

    def test_two_turn_session(self):
        # 2 turns: 200ms, 400ms. Median 300ms, stdev = 141ms (population
        # stdev would be 100; statistics.stdev uses sample stdev).
        # Raw = 1 - 141/300 = 0.53.
        out = io.StringIO()
        print_session_summary(
            [_m(0.2), _m(0.4)], {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Rhythm score:     0.53" in plain


# ---- Integration with other summary lines -------------------------------


class TestIntegrationWithSummary:
    def test_appears_between_best_ttfs_and_naturalness(self):
        # Set naturalness_bucket explicitly so the Naturalness line
        # also emits — confirms rhythm and naturalness coexist in
        # the output in the right order.
        def _m_natural(ttfs):
            return TurnMetrics(ttfs=ttfs, naturalness_bucket="natural")

        out = io.StringIO()
        print_session_summary(
            [
                _m_natural(0.3),
                _m_natural(0.35),
                _m_natural(0.32),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Order: Median TTFS → Best TTFS → Rhythm score → Naturalness.
        # All three of these landmarks should appear.
        idx_best = plain.find("Best TTFS:")
        idx_rhythm = plain.find("Rhythm score:")
        idx_naturalness = plain.find("Naturalness:")
        assert idx_best != -1
        assert idx_rhythm != -1
        assert idx_naturalness != -1
        assert idx_best < idx_rhythm < idx_naturalness
