"""Tests for iter-058 — error rate per stage metric.

Metric 1.16 from docs/perf-metrics-taxonomy.md.

Two-layer structure:
- Per-turn: ``TurnMetrics.worker_errors`` = len(worker.errors)
  at turn end. Captures partial-success turns where some
  sentences synthed but others raised.
- Session: ``llm_errors`` kwarg = count of turns with
  ``had_error=True`` (entire turn killed by LLM exception).

Why both layers: an LLM error kills the whole turn (no metrics);
a worker error keeps the turn but loses one sentence's audio.
Different reliability stories.
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


def _m(worker_errors=0):
    return TurnMetrics(ttfs=0.5, worker_errors=worker_errors)


# ---- Per-turn field -------------------------------------------------------


class TestPerTurnField:
    def test_default_zero(self):
        assert TurnMetrics().worker_errors == 0

    def test_can_be_set(self):
        m = TurnMetrics(worker_errors=3)
        assert m.worker_errors == 3


# ---- Session aggregate kwargs --------------------------------------------


class TestNoErrors:
    def test_clean_session_omits_block(self):
        out = io.StringIO()
        print_session_summary(
            [_m()], {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Errors:" not in plain


class TestLLMOnly:
    def test_llm_errors_only(self):
        # 2 successful turns + 1 LLM error → "1 LLM (over 3 attempts)".
        out = io.StringIO()
        print_session_summary(
            [_m(), _m()], {"model": "stub"}, file=out,
            llm_errors=1,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Errors:           1 LLM (over 3 attempts)" in plain

    def test_llm_errors_with_false_triggers_in_attempts(self):
        # 1 success + 1 LLM error + 2 false triggers = 4 attempts.
        out = io.StringIO()
        print_session_summary(
            [_m()], {"model": "stub"}, file=out,
            llm_errors=1, false_triggers=2,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Errors:           1 LLM (over 4 attempts)" in plain


class TestWorkerOnly:
    def test_one_turn_with_worker_errors(self):
        out = io.StringIO()
        print_session_summary(
            [_m(worker_errors=2), _m()],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # 2 attempts (no LLM errors / false triggers); 2 worker errors.
        assert "Errors:           2 worker (over 2 attempts)" in plain

    def test_workers_summed_across_turns(self):
        out = io.StringIO()
        print_session_summary(
            [_m(worker_errors=2), _m(worker_errors=3), _m()],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Errors:           5 worker (over 3 attempts)" in plain


class TestBothStages:
    def test_llm_and_worker_combined(self):
        out = io.StringIO()
        print_session_summary(
            [_m(worker_errors=1), _m()],
            {"model": "stub"}, file=out,
            llm_errors=2,
        )
        plain = _strip_ansi(out.getvalue())
        # 2 successful turns + 2 LLM errors = 4 attempts. 2 LLM, 1 worker.
        assert "Errors:           2 LLM, 1 worker (over 4 attempts)" in plain

    def test_singular_attempts(self):
        out = io.StringIO()
        print_session_summary(
            [], {"model": "stub"}, file=out,
            llm_errors=1,
        )
        plain = _strip_ansi(out.getvalue())
        # 0 + 1 = 1 attempt. The empty metrics_list path early-returns
        # the "no completed turns" placeholder before reaching the
        # error block — document the behavior.
        assert "no completed turns" in plain
        assert "Errors:" not in plain
