"""Tests for iter-067 — worker error-recovery success rate.

Metric 2.15 from docs/perf-metrics-taxonomy.md.

    recovery_rate = turns_with_errors_that_produced_audio / turns_with_errors

Of the turns where the SentenceWorker raised at least one
synth/play exception, what fraction still produced audio
(ttfs > 0)? 100% recovery is silent partial degradation — a user
heard a complete-sounding response but a sentence inside was
dropped. 0% recovery means every error killed the whole turn.
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


def _m(ttfs=0.0, errors=0):
    return TurnMetrics(ttfs=ttfs, worker_errors=errors)


def _summary(metrics_list, **kwargs):
    out = io.StringIO()
    print_session_summary(
        metrics_list, {"model": "stub"}, file=out, **kwargs,
    )
    return _strip_ansi(out.getvalue())


# ---- No-emit cases ---------------------------------------------------


class TestNoEmit:
    def test_no_errors_anywhere_omits(self):
        plain = _summary([_m(ttfs=0.5), _m(ttfs=0.5)])
        assert "Worker recovery" not in plain
        assert "Errors" not in plain

    def test_only_llm_errors_omits_recovery(self):
        # The Errors block emits, but the recovery line is gated on
        # at least one turn having worker_errors > 0.
        plain = _summary([_m(ttfs=0.5)], llm_errors=2)
        assert "Errors" in plain
        assert "Worker recovery" not in plain


# ---- Emit cases ------------------------------------------------------


class TestEmit:
    def test_full_recovery_silent_degradation(self):
        # 2 error turns, both produced audio → 100% — silent partial
        # degradation; this is exactly the case the metric is for.
        plain = _summary([
            _m(ttfs=0.5, errors=1),
            _m(ttfs=0.6, errors=2),
            _m(ttfs=0.5),  # no errors, ignored
        ])
        assert "Worker recovery:  2/2 turns produced audio (100%)" in plain
        assert "partial degradation" in plain

    def test_zero_recovery_loud_failure(self):
        # Errors kill audio every time → 0%.
        plain = _summary([
            _m(ttfs=0.0, errors=1),
            _m(ttfs=0.0, errors=1),
            _m(ttfs=0.5),
        ])
        assert "Worker recovery:  0/2 turns produced audio (0%)" in plain

    def test_partial_recovery(self):
        # Mixed: 1 of 3 error turns recovered.
        plain = _summary([
            _m(ttfs=0.5, errors=1),  # recovered
            _m(ttfs=0.0, errors=1),
            _m(ttfs=0.0, errors=2),
        ])
        assert "Worker recovery:  1/3 turns produced audio (33%)" in plain

    def test_clean_turns_excluded_from_denominator(self):
        # Turns without errors don't dilute the rate.
        plain = _summary([
            _m(ttfs=0.5),
            _m(ttfs=0.5),
            _m(ttfs=0.5),
            _m(ttfs=0.5, errors=1),  # the only error turn → 1/1.
        ])
        assert "Worker recovery:  1/1 turns produced audio (100%)" in plain

    def test_multiple_errors_in_single_turn_count_once(self):
        # Denominator is "turns with errors", not "errors". A single
        # turn with 5 errors is still 1 turn.
        plain = _summary([
            _m(ttfs=0.5, errors=5),
        ])
        assert "Worker recovery:  1/1 turns produced audio (100%)" in plain
        # Errors block also shows the total error count for context.
        assert "5 worker" in plain
