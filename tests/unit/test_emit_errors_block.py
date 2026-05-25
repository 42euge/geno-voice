"""Tests for iter-092 — _emit_errors_block helper.

Mirrors the iter-089/090/091 pattern: extract a multi-line
session-summary block into a helper, test it directly with
synthetic inputs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    ErrorStats,
    _emit_errors_block,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line: str = "") -> None:
        lines.append(_strip_ansi(line))

    return emit, lines


# ---- ErrorStats defaults ---------------------------------------


class TestDefaults:
    def test_all_zero(self):
        s = ErrorStats()
        assert s.llm_errors == 0
        assert s.worker_errors_total == 0
        assert s.error_turns_with_audio == 0
        assert s.error_turns_total == 0
        assert s.n == 0
        assert s.false_triggers == 0
        assert s.silent_turns == 0


# ---- No-data path ----------------------------------------------


class TestClean:
    def test_emits_nothing(self):
        emit, lines = _capture()
        _emit_errors_block(emit, ErrorStats(n=10))
        assert lines == []


# ---- LLM errors only -------------------------------------------


class TestLlmErrors:
    def test_emits_count_and_attempts(self):
        emit, lines = _capture()
        _emit_errors_block(
            emit,
            ErrorStats(llm_errors=2, n=8, false_triggers=1),
        )
        # 8 + 2 + 1 = 11 attempts.
        assert any(
            "Errors:           2 LLM (over 11 attempts)" in ln
            for ln in lines
        )
        # No worker recovery line since no worker errors.
        assert not any("Worker recovery:" in ln for ln in lines)

    def test_singular_attempt(self):
        # Single attempt → "1 attempt" (not "1 attempts").
        emit, lines = _capture()
        _emit_errors_block(
            emit,
            ErrorStats(llm_errors=1, n=0, false_triggers=0),
        )
        assert any(
            "Errors:           1 LLM (over 1 attempt)" in ln for ln in lines
        )


# ---- Worker errors + recovery rate -----------------------------


class TestWorkerErrors:
    def test_emits_count_and_recovery(self):
        emit, lines = _capture()
        _emit_errors_block(
            emit,
            ErrorStats(
                worker_errors_total=3,
                error_turns_with_audio=2,
                error_turns_total=3,
                n=10,
            ),
        )
        # Errors line: "3 worker (over 10 attempts)".
        assert any(
            "Errors:           3 worker (over 10 attempts)" in ln
            for ln in lines
        )
        # Recovery: 2/3 = 67%.
        assert any(
            "Worker recovery:  2/3 turns produced audio (67%)" in ln
            for ln in lines
        )

    def test_zero_recovery_loud_failure(self):
        emit, lines = _capture()
        _emit_errors_block(
            emit,
            ErrorStats(
                worker_errors_total=2,
                error_turns_with_audio=0,
                error_turns_total=2,
                n=5,
            ),
        )
        assert any(
            "Worker recovery:  0/2 turns produced audio (0%)" in ln
            for ln in lines
        )


# ---- Combined LLM + worker -----------------------------------


class TestCombined:
    def test_both_kinds_emit_in_one_line(self):
        emit, lines = _capture()
        _emit_errors_block(
            emit,
            ErrorStats(
                llm_errors=1, worker_errors_total=2,
                error_turns_with_audio=1, error_turns_total=1,
                n=5, false_triggers=0,
            ),
        )
        # Single Errors line with both kinds.
        line = next(ln for ln in lines if "Errors:" in ln)
        assert "1 LLM" in line
        assert "2 worker" in line


# ---- Silent turns ---------------------------------------------


class TestSilentTurns:
    def test_emits_when_present(self):
        emit, lines = _capture()
        _emit_errors_block(
            emit,
            ErrorStats(silent_turns=2, n=10),
        )
        assert any(
            "Silent turns:     2/10 (20%) — bot produced no audio" in ln
            for ln in lines
        )

    def test_omits_when_zero(self):
        emit, lines = _capture()
        _emit_errors_block(emit, ErrorStats(n=10))
        assert not any("Silent turns:" in ln for ln in lines)

    def test_silent_independent_of_errors(self):
        # Silent turns can fire without any errors — the user spoke
        # but the bot stayed silent without throwing.
        emit, lines = _capture()
        _emit_errors_block(emit, ErrorStats(silent_turns=1, n=3))
        assert any("Silent turns:" in ln for ln in lines)
        assert not any("Errors:" in ln for ln in lines)


# ---- Ordering invariant ---------------------------------------


class TestOrdering:
    def test_errors_then_recovery_then_silent(self):
        emit, lines = _capture()
        _emit_errors_block(
            emit,
            ErrorStats(
                llm_errors=1,
                worker_errors_total=2,
                error_turns_with_audio=1,
                error_turns_total=1,
                n=5,
                silent_turns=1,
            ),
        )

        def _idx(label: str) -> int:
            for i, ln in enumerate(lines):
                if label in ln:
                    return i
            return -1

        e_i = _idx("Errors:")
        r_i = _idx("Worker recovery:")
        s_i = _idx("Silent turns:")
        assert all(i >= 0 for i in (e_i, r_i, s_i))
        assert e_i < r_i < s_i
