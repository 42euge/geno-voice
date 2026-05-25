"""Tests for iter-079 — silent-turn rate.

Metric 3.11 from docs/perf-metrics-taxonomy.md.

    silent = transcript != "" AND ttfs == 0

Counts turns where the user spoke (transcript captured) but the
bot produced no audio. The invisible failure mode — user thinks
the bot is broken but no exception fired.
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


def _summary(metrics_list, **kwargs):
    out = io.StringIO()
    print_session_summary(metrics_list, {"model": "stub"}, file=out, **kwargs)
    return _strip_ansi(out.getvalue())


def _normal(transcript="hi", ttfs=0.5):
    return TurnMetrics(transcript=transcript, ttfs=ttfs)


def _silent(transcript="hello"):
    # User spoke but bot stayed silent.
    return TurnMetrics(transcript=transcript, ttfs=0.0)


def _no_speech(ttfs=0.5):
    # No transcript captured (e.g. recorder bypassed via test stub
    # that didn't set _last_text) — bot may or may not have audio.
    return TurnMetrics(transcript="", ttfs=ttfs)


# ---- No-emit boundaries ------------------------------------------


class TestNoEmit:
    def test_clean_session_omits(self):
        plain = _summary([_normal(), _normal(), _normal()])
        assert "Silent turns" not in plain

    def test_zero_turns_omits(self):
        # Empty metrics list — the "no completed turns" path runs
        # before the silent-turn block, so nothing emits.
        plain = _summary([])
        assert "Silent turns" not in plain

    def test_no_transcript_no_silent(self):
        # ttfs==0 + no transcript = "false trigger" not "silent",
        # excluded from the count.
        plain = _summary([_normal(), _no_speech(ttfs=0.0)])
        assert "Silent turns" not in plain


# ---- Emit cases --------------------------------------------------


class TestEmit:
    def test_one_silent(self):
        plain = _summary([_normal(), _silent(), _normal()])
        assert "Silent turns:     1/3 (33%) — bot produced no audio" in plain

    def test_all_silent(self):
        plain = _summary([_silent(), _silent()])
        assert "Silent turns:     2/2 (100%)" in plain

    def test_one_in_ten(self):
        plain = _summary([_silent()] + [_normal()] * 9)
        assert "Silent turns:     1/10 (10%)" in plain


# ---- Distinction from worker_errors ------------------------------


class TestDistinctFromErrors:
    def test_silent_with_no_worker_errors(self):
        # Pure "silent failure" — the failure mode this metric
        # specifically targets. No worker errors but no audio.
        plain = _summary([_silent()])
        assert "Silent turns" in plain
        # No Errors block since worker_errors_total == 0.
        assert "Errors:" not in plain

    def test_silent_with_worker_errors_both_emit(self):
        # Errored turn that ALSO produced no audio: counted by
        # both metrics — the worker recovery rate (iter-067) AND
        # silent-turn rate (this).
        m = TurnMetrics(transcript="hi", ttfs=0.0, worker_errors=1)
        plain = _summary([m])
        # Both lines present.
        assert "Silent turns" in plain
        assert "Worker recovery" in plain
        # Recovery on a single error turn that didn't produce
        # audio = 0/1 (0%).
        assert "0/1 turns produced audio" in plain
