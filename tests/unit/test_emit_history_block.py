"""Tests for iter-097 — _emit_history_block helper."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    HistoryStats,
    _emit_history_block,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line=""):
        lines.append(_strip_ansi(line))

    return emit, lines


# ---- HistoryStats defaults ------------------------------------


class TestDefaults:
    def test_empty(self):
        s = HistoryStats()
        assert s.context_token_counts == []
        assert s.trim_events == 0
        assert s.trim_messages_evicted == 0


# ---- No-data path ----------------------------------------------


class TestEmpty:
    def test_emits_nothing(self):
        emit, lines = _capture()
        _emit_history_block(emit, HistoryStats())
        assert lines == []


# ---- Context tokens block ------------------------------------


class TestContextTokens:
    def test_two_turns_emits_median_max_no_growth(self):
        # Below 3-turn threshold — no growth line.
        emit, lines = _capture()
        _emit_history_block(
            emit, HistoryStats(context_token_counts=[20, 40]),
        )
        assert any("Context tokens:   30 median, 40 max" in ln for ln in lines)
        assert not any("Context growth:" in ln for ln in lines)

    def test_three_turns_emits_growth(self):
        emit, lines = _capture()
        _emit_history_block(
            emit, HistoryStats(context_token_counts=[20, 30, 80]),
        )
        # Median of [20,30,80]=30; max=80; growth = 80-20 = +60.
        assert any("Context tokens:   30 median, 80 max" in ln for ln in lines)
        assert any(
            "Context growth:   +60 tokens (turn 1 → turn 3)" in ln
            for ln in lines
        )

    def test_negative_growth(self):
        # Trim caused context to shrink across the session.
        emit, lines = _capture()
        _emit_history_block(
            emit, HistoryStats(context_token_counts=[100, 80, 50]),
        )
        assert any(
            "Context growth:   -50 tokens (turn 1 → turn 3)" in ln
            for ln in lines
        )


# ---- Trim events block ----------------------------------------


class TestTrimEvents:
    def test_emits_with_ratio(self):
        emit, lines = _capture()
        _emit_history_block(
            emit,
            HistoryStats(
                context_token_counts=[20], trim_events=2,
                trim_messages_evicted=4,
            ),
        )
        # 4/2 = 2.0/event.
        assert any(
            "Trim events:      2 (4 evicted, 2.0/event)" in ln for ln in lines
        )

    def test_omits_when_zero(self):
        emit, lines = _capture()
        _emit_history_block(
            emit, HistoryStats(context_token_counts=[20]),
        )
        assert not any("Trim events:" in ln for ln in lines)

    def test_steady_state_one_per_event(self):
        emit, lines = _capture()
        _emit_history_block(
            emit,
            HistoryStats(trim_events=5, trim_messages_evicted=5),
        )
        assert any(
            "Trim events:      5 (5 evicted, 1.0/event)" in ln for ln in lines
        )


# ---- Independence ---------------------------------------------


class TestIndependence:
    def test_trim_emits_without_context_data(self):
        # Edge case: trim happened but no context tokens recorded.
        emit, lines = _capture()
        _emit_history_block(
            emit,
            HistoryStats(trim_events=1, trim_messages_evicted=2),
        )
        assert not any("Context tokens:" in ln for ln in lines)
        assert any("Trim events:" in ln for ln in lines)


# ---- Ordering invariant ---------------------------------------


class TestOrdering:
    def test_context_then_growth_then_trim(self):
        emit, lines = _capture()
        _emit_history_block(
            emit,
            HistoryStats(
                context_token_counts=[10, 20, 30],
                trim_events=2,
                trim_messages_evicted=2,
            ),
        )

        def _idx(label: str) -> int:
            for i, ln in enumerate(lines):
                if label in ln:
                    return i
            return -1

        ctx_i = _idx("Context tokens:")
        grow_i = _idx("Context growth:")
        trim_i = _idx("Trim events:")
        assert all(i >= 0 for i in (ctx_i, grow_i, trim_i))
        assert ctx_i < grow_i < trim_i
