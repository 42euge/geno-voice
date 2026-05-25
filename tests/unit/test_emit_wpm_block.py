"""Tests for iter-094 — _emit_wpm_block helper.

Mirrors the iter-089/090/091/092 pattern.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    WpmStats,
    _emit_wpm_block,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line: str = "") -> None:
        lines.append(_strip_ansi(line))

    return emit, lines


# ---- WpmStats defaults -----------------------------------------


class TestDefaults:
    def test_empty_lists(self):
        s = WpmStats()
        assert s.user_wpms == []
        assert s.bot_wpms == []


# ---- No-data path ----------------------------------------------


class TestEmpty:
    def test_emits_nothing(self):
        emit, lines = _capture()
        _emit_wpm_block(emit, WpmStats())
        assert lines == []


# ---- User-only / bot-only ---------------------------------------


class TestUserOnly:
    def test_emits_median_user_no_mirror_gap(self):
        emit, lines = _capture()
        _emit_wpm_block(emit, WpmStats(user_wpms=[140, 160]))
        assert any("Median user WPM:  150" in ln for ln in lines)
        assert not any("Median bot WPM:" in ln for ln in lines)
        assert not any("Mirror gap:" in ln for ln in lines)


class TestBotOnly:
    def test_emits_median_bot_no_mirror_gap(self):
        emit, lines = _capture()
        _emit_wpm_block(emit, WpmStats(bot_wpms=[170, 190]))
        assert any("Median bot WPM:   180" in ln for ln in lines)
        assert not any("Median user WPM:" in ln for ln in lines)
        assert not any("Mirror gap:" in ln for ln in lines)


# ---- Both present → mirror gap emits ---------------------------


class TestBoth:
    def test_emits_all_three_lines(self):
        emit, lines = _capture()
        _emit_wpm_block(
            emit,
            WpmStats(user_wpms=[140, 150], bot_wpms=[170, 180]),
        )
        # Median user = 145, median bot = 175, gap = +30.
        assert any("Median user WPM:  145" in ln for ln in lines)
        assert any("Median bot WPM:   175" in ln for ln in lines)
        assert any("Mirror gap:       +30 WPM (bot − user)" in ln for ln in lines)

    def test_negative_gap(self):
        emit, lines = _capture()
        _emit_wpm_block(
            emit,
            WpmStats(user_wpms=[200], bot_wpms=[150]),
        )
        assert any("Mirror gap:       -50 WPM" in ln for ln in lines)


# ---- Ordering invariant ----------------------------------------


class TestOrdering:
    def test_user_then_bot_then_gap(self):
        emit, lines = _capture()
        _emit_wpm_block(
            emit,
            WpmStats(user_wpms=[140], bot_wpms=[170]),
        )

        def _idx(label: str) -> int:
            for i, ln in enumerate(lines):
                if label in ln:
                    return i
            return -1

        u_i = _idx("Median user WPM:")
        b_i = _idx("Median bot WPM:")
        g_i = _idx("Mirror gap:")
        assert all(i >= 0 for i in (u_i, b_i, g_i))
        assert u_i < b_i < g_i
