"""Tests for iter-069 — interruption rate.

Metric 1.18 from docs/perf-metrics-taxonomy.md.

    interruption_rate = barges / completed_turns

Industry single-number UX KPI: "what fraction of bot turns did
the user feel they had to interrupt?" Distinct from the existing
mid-stream % (denominator there is total barges, not turns).
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


def _m(barge=False, ttfs=0.5):
    # ttfs > 0 puts the metric in the standard print path.
    return TurnMetrics(ttfs=ttfs, barge_in=barge)


def _summary(metrics_list, **kwargs):
    out = io.StringIO()
    print_session_summary(
        metrics_list, {"model": "stub"}, file=out, **kwargs,
    )
    return _strip_ansi(out.getvalue())


# ---- No-emit cases ---------------------------------------------------


class TestNoEmit:
    def test_no_barges_omits(self):
        # Whole barge block is gated on barges_total > 0.
        plain = _summary([_m(), _m(), _m()])
        assert "Interruption rate" not in plain

    def test_zero_turns_omits(self):
        plain = _summary([])
        assert "Interruption rate" not in plain


# ---- Emit cases ------------------------------------------------------


class TestEmit:
    def test_one_of_three(self):
        plain = _summary([_m(barge=True), _m(), _m()])
        assert "Interruption rate: 1/3 turns (33%)" in plain

    def test_all_turns_interrupted(self):
        plain = _summary([_m(barge=True), _m(barge=True)])
        assert "Interruption rate: 2/2 turns (100%)" in plain

    def test_low_rate(self):
        plain = _summary([_m(barge=True)] + [_m()] * 9)
        assert "Interruption rate: 1/10 turns (10%)" in plain


# ---- Co-emission with existing barge block ----------------------------


class TestCoEmission:
    def test_with_mid_stream_line(self):
        # When sentences_cancelled is present, the existing line
        # shows mid-stream %; the rate line is a separate, additive
        # signal.
        m1 = TurnMetrics(ttfs=0.5, barge_in=True, sentences_cancelled=1)
        m2 = TurnMetrics(ttfs=0.5)
        plain = _summary([m1, m2])
        assert "Barge-ins:        1 (1 mid-stream, 100%)" in plain
        assert "Interruption rate: 1/2 turns (50%)" in plain

    def test_distinct_denominators(self):
        # The mid-stream % uses barges as denominator; interruption
        # rate uses total turns as denominator. Verify they're not
        # accidentally equal in a case where they should differ.
        # 4 turns, 2 barges, 1 of those mid-stream:
        #   mid_cancels / barges_total = 50%
        #   barges_total / n           = 50%
        # OK these collide. Pick a case where they differ:
        # 8 turns, 4 barges, 1 mid-stream:
        #   mid_cancels/barges = 25%, barges/n = 50%.
        m_mid = TurnMetrics(ttfs=0.5, barge_in=True, sentences_cancelled=1)
        m_clean_barge = TurnMetrics(ttfs=0.5, barge_in=True)
        m_normal = TurnMetrics(ttfs=0.5)
        plain = _summary([m_mid, m_clean_barge, m_clean_barge, m_clean_barge,
                          m_normal, m_normal, m_normal, m_normal])
        assert "Barge-ins:        4 (1 mid-stream, 25%)" in plain
        assert "Interruption rate: 4/8 turns (50%)" in plain
