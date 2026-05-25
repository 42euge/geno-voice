"""Tests for iter-089 — _emit_ttfs_block helper.

The helper extracts ~80 lines of TTFS-related session-summary
output that previously lived inline in print_session_summary.
This test suite exercises it directly without going through the
full session-summary path, faster and more focused.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    TurnMetrics,
    _emit_ttfs_block,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    """Returns (emit_fn, lines list).

    The helper takes any callable as its emit. We collect each
    call's argument into a list so tests can assert ordering AND
    content cleanly.
    """
    lines: list[str] = []

    def emit(line: str = "") -> None:
        lines.append(_strip_ansi(line))

    return emit, lines


def _no_naturalness():
    return {"rushed": 0, "natural": 0, "slow": 0}


# ---- No-data path ------------------------------------------------


class TestEmpty:
    def test_empty_ttfs_times_emits_na(self):
        emit, lines = _capture()
        _emit_ttfs_block(emit, [], [], _no_naturalness())
        assert lines == [
            "    Median TTFS:      n/a",
            "    Best TTFS:        n/a",
        ]


# ---- Single-turn path -------------------------------------------


class TestSingleTurn:
    def test_emits_median_best_sub_second(self):
        emit, lines = _capture()
        _emit_ttfs_block(
            emit,
            [0.5],
            [TurnMetrics(ttfs=0.5)],
            _no_naturalness(),
        )
        # Single-turn: rhythm/jitter/cold-start blocks all gated
        # on ≥2 turns and skip.
        assert lines == [
            "    Median TTFS:      500ms",
            "    Best TTFS:        500ms",
            "    Sub-second TTFS:  1/1 (100%)",
        ]

    def test_above_one_second_emits_zero_pct(self):
        emit, lines = _capture()
        _emit_ttfs_block(
            emit,
            [1.5],
            [TurnMetrics(ttfs=1.5)],
            _no_naturalness(),
        )
        assert "    Sub-second TTFS:  0/1 (0%)" in lines


# ---- Multi-turn path ---------------------------------------------


class TestMultiTurn:
    def test_emits_rhythm_and_jitter(self):
        emit, lines = _capture()
        _emit_ttfs_block(
            emit,
            [0.5, 0.6],
            [TurnMetrics(ttfs=0.5), TurnMetrics(ttfs=0.6)],
            _no_naturalness(),
        )
        # Rhythm + jitter lines present.
        assert any("Rhythm score:" in ln for ln in lines)
        assert any("TTFS jitter:" in ln for ln in lines)

    def test_emits_cold_start_when_penalty_above_floor(self):
        # Turn 1 = 800ms, steady = 500ms median → +300ms penalty.
        emit, lines = _capture()
        _emit_ttfs_block(
            emit,
            [0.800, 0.500, 0.500, 0.500],
            [
                TurnMetrics(ttfs=0.800),
                TurnMetrics(ttfs=0.500),
                TurnMetrics(ttfs=0.500),
                TurnMetrics(ttfs=0.500),
            ],
            _no_naturalness(),
        )
        cold_lines = [ln for ln in lines if "Cold start:" in ln]
        assert len(cold_lines) == 1
        assert "+300ms" in cold_lines[0]

    def test_skips_cold_start_below_floor(self):
        # Turn 1 = 520ms, steady = 500ms → +20ms (below 50ms floor).
        emit, lines = _capture()
        _emit_ttfs_block(
            emit,
            [0.520, 0.500, 0.500],
            [
                TurnMetrics(ttfs=0.520),
                TurnMetrics(ttfs=0.500),
                TurnMetrics(ttfs=0.500),
            ],
            _no_naturalness(),
        )
        assert not any("Cold start:" in ln for ln in lines)


# ---- Naturalness distribution -----------------------------------


class TestNaturalness:
    def test_omits_when_no_buckets(self):
        emit, lines = _capture()
        _emit_ttfs_block(emit, [0.5], [TurnMetrics(ttfs=0.5)], _no_naturalness())
        assert not any("Naturalness:" in ln for ln in lines)

    def test_emits_when_any_bucket_set(self):
        emit, lines = _capture()
        _emit_ttfs_block(
            emit,
            [0.5],
            [TurnMetrics(ttfs=0.5)],
            {"rushed": 1, "natural": 2, "slow": 0},
        )
        nat = [ln for ln in lines if "Naturalness:" in ln]
        assert len(nat) == 1
        assert "1 rushed, 2 natural, 0 slow" in nat[0]


# ---- Ordering invariant ----------------------------------------


class TestEmitOrdering:
    def test_lines_emit_in_documented_order(self):
        # The helper guarantees a stable order:
        # Median → Best → Sub-second → Rhythm → TTFS jitter →
        # Cold start → Naturalness.
        emit, lines = _capture()
        _emit_ttfs_block(
            emit,
            [0.800, 0.500, 0.500],
            [
                TurnMetrics(ttfs=0.800, naturalness_bucket="slow"),
                TurnMetrics(ttfs=0.500, naturalness_bucket="natural"),
                TurnMetrics(ttfs=0.500, naturalness_bucket="natural"),
            ],
            {"rushed": 0, "natural": 2, "slow": 1},
        )

        def _idx(label: str) -> int:
            for i, ln in enumerate(lines):
                if label in ln:
                    return i
            return -1

        median_i = _idx("Median TTFS:")
        best_i = _idx("Best TTFS:")
        sub_i = _idx("Sub-second TTFS:")
        rhythm_i = _idx("Rhythm score:")
        jitter_i = _idx("TTFS jitter:")
        cold_i = _idx("Cold start:")
        nat_i = _idx("Naturalness:")

        # All emitted in order, all >= 0.
        order = [median_i, best_i, sub_i, rhythm_i, jitter_i, cold_i, nat_i]
        assert all(i >= 0 for i in order), order
        assert order == sorted(order), f"Out-of-order emit: {order}"
