"""Tests for iter-090 — _emit_barge_block helper.

Mirrors iter-089's pattern: extract a multi-line session-summary
block into a helper, test it directly with synthetic inputs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    BargeStats,
    _emit_barge_block,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line: str = "") -> None:
        lines.append(_strip_ansi(line))

    return emit, lines


# ---- BargeStats defaults ----------------------------------------


class TestBargeStatsDefaults:
    def test_all_zero(self):
        s = BargeStats()
        assert s.barges_total == 0
        assert s.mid_cancels == 0
        assert s.n == 0
        assert s.barge_latencies == []
        assert s.cancel_close_lats == []
        assert s.llm_phase_barges == 0
        assert s.playback_phase_barges == 0
        assert s.regret_barges == 0
        assert s.preempted_total == 0
        assert s.barge_turns_with_loss == 0


# ---- No-data path -----------------------------------------------


class TestNoBarges:
    def test_zero_barges_omits_everything(self):
        emit, lines = _capture()
        _emit_barge_block(emit, BargeStats(n=10))
        assert lines == []


# ---- Single barge ---------------------------------------------


class TestSingleBarge:
    def test_clean_barge_emits_count_and_rate(self):
        # 1 barge, no mid-stream cuts → "all between sentences" form.
        # n=10 → interruption rate 10%.
        emit, lines = _capture()
        _emit_barge_block(emit, BargeStats(barges_total=1, n=10))
        assert any(
            "Barge-ins:        1 (all between sentences)" in ln
            for ln in lines
        )
        assert any(
            "Interruption rate: 1/10 turns (10%)" in ln for ln in lines
        )

    def test_mid_stream_cut_uses_mid_stream_form(self):
        emit, lines = _capture()
        _emit_barge_block(
            emit,
            BargeStats(barges_total=2, mid_cancels=1, n=10),
        )
        # 1 of 2 mid-stream = 50%.
        assert any(
            "Barge-ins:        2 (1 mid-stream, 50%)" in ln
            for ln in lines
        )


# ---- Latency lines ---------------------------------------------


class TestLatencies:
    def test_emits_median_and_worst(self):
        emit, lines = _capture()
        _emit_barge_block(
            emit,
            BargeStats(
                barges_total=3,
                n=5,
                barge_latencies=[0.080, 0.150, 0.250],
            ),
        )
        # Median of [80, 150, 250] = 150ms; worst = 250ms.
        assert any("Median barge:     150ms" in ln for ln in lines)
        assert any("Worst barge:      250ms" in ln for ln in lines)

    def test_emits_cancel_close_when_present(self):
        emit, lines = _capture()
        _emit_barge_block(
            emit,
            BargeStats(
                barges_total=2, n=5,
                cancel_close_lats=[0.300, 0.700],
            ),
        )
        # Median of [300, 700] = 500ms.
        assert any("Median LLM canc:  500ms" in ln for ln in lines)


# ---- Phase distribution ----------------------------------------


class TestPhases:
    def test_emits_when_either_set(self):
        emit, lines = _capture()
        _emit_barge_block(
            emit,
            BargeStats(
                barges_total=3, n=5,
                llm_phase_barges=2,
                playback_phase_barges=1,
            ),
        )
        assert any(
            "Barge phases:     2 LLM-stream, 1 playback" in ln
            for ln in lines
        )

    def test_omits_when_both_zero(self):
        emit, lines = _capture()
        _emit_barge_block(emit, BargeStats(barges_total=1, n=5))
        assert not any("Barge phases:" in ln for ln in lines)


# ---- Regret rate -----------------------------------------------


class TestRegret:
    def test_emits_with_pct(self):
        emit, lines = _capture()
        _emit_barge_block(
            emit,
            BargeStats(barges_total=4, n=10, regret_barges=2),
        )
        assert "Regret rate:      2/4 (50%)" in lines[2]

    def test_omits_when_zero(self):
        emit, lines = _capture()
        _emit_barge_block(emit, BargeStats(barges_total=1, n=5))
        assert not any("Regret rate:" in ln for ln in lines)


# ---- Pre-empted words ------------------------------------------


class TestPreempted:
    def test_emits_with_avg(self):
        emit, lines = _capture()
        _emit_barge_block(
            emit,
            BargeStats(
                barges_total=3, n=5,
                preempted_total=42,
                barge_turns_with_loss=2,
            ),
        )
        # 42 / 2 = 21 avg/loss.
        line = next(ln for ln in lines if "Pre-empted words:" in ln)
        assert "42 total" in line
        assert "2/3 barges" in line
        assert "21 avg/loss" in line

    def test_omits_when_zero(self):
        emit, lines = _capture()
        _emit_barge_block(emit, BargeStats(barges_total=1, n=5))
        assert not any("Pre-empted words:" in ln for ln in lines)


# ---- Full sample emit-order invariant -------------------------


class TestEmitOrdering:
    def test_lines_emit_in_documented_order(self):
        emit, lines = _capture()
        _emit_barge_block(
            emit,
            BargeStats(
                barges_total=3,
                mid_cancels=1,
                n=10,
                barge_latencies=[0.1, 0.2],
                cancel_close_lats=[0.3, 0.5],
                llm_phase_barges=1,
                playback_phase_barges=2,
                regret_barges=1,
                preempted_total=15,
                barge_turns_with_loss=1,
            ),
        )

        def _idx(label: str) -> int:
            for i, ln in enumerate(lines):
                if label in ln:
                    return i
            return -1

        order = [
            _idx("Barge-ins:"),
            _idx("Interruption rate:"),
            _idx("Median barge:"),
            _idx("Worst barge:"),
            _idx("Median LLM canc:"),
            _idx("Barge phases:"),
            _idx("Regret rate:"),
            _idx("Pre-empted words:"),
        ]
        assert all(i >= 0 for i in order), order
        assert order == sorted(order), f"Out-of-order emit: {order}"
