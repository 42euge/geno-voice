"""Tests for iter-095 — _emit_sentence_block helper."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    SentenceStats,
    _emit_sentence_block,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line: str = "") -> None:
        lines.append(_strip_ansi(line))

    return emit, lines


# ---- SentenceStats defaults ------------------------------------


class TestDefaults:
    def test_empty_defaults(self):
        s = SentenceStats()
        assert s.sentence_lens == []
        assert s.min_chars_seen == 0
        assert s.max_chars_seen == 0
        assert s.coverage_values == []


# ---- No-data path ----------------------------------------------


class TestEmpty:
    def test_emits_nothing(self):
        emit, lines = _capture()
        _emit_sentence_block(emit, SentenceStats())
        assert lines == []


# ---- Mean-only ------------------------------------------------


class TestMean:
    def test_emits_mean_when_lens_present(self):
        emit, lines = _capture()
        _emit_sentence_block(
            emit,
            SentenceStats(sentence_lens=[60, 80]),
        )
        assert any("Mean sentence:    70 chars" in ln for ln in lines)
        # Range gated on max_chars_seen > 0; not set here.
        assert not any("Sentence range:" in ln for ln in lines)

    def test_emits_range_when_min_lt_max(self):
        emit, lines = _capture()
        _emit_sentence_block(
            emit,
            SentenceStats(
                sentence_lens=[60, 80],
                min_chars_seen=10,
                max_chars_seen=130,
            ),
        )
        assert any(
            "Sentence range:   [10..130] chars (session)" in ln
            for ln in lines
        )

    def test_omits_range_when_min_equals_max(self):
        emit, lines = _capture()
        _emit_sentence_block(
            emit,
            SentenceStats(
                sentence_lens=[60, 80],
                min_chars_seen=70,
                max_chars_seen=70,
            ),
        )
        assert not any("Sentence range:" in ln for ln in lines)


# ---- Coverage ---------------------------------------------------


class TestCoverage:
    def test_emits_when_present(self):
        emit, lines = _capture()
        _emit_sentence_block(
            emit,
            SentenceStats(coverage_values=[0.80, 0.90]),
        )
        # Median of [0.80, 0.90] = 0.85 → 85%.
        assert any("Split coverage:   85%" in ln for ln in lines)

    def test_omits_when_empty(self):
        emit, lines = _capture()
        _emit_sentence_block(emit, SentenceStats())
        assert not any("Split coverage:" in ln for ln in lines)

    def test_independent_of_mean(self):
        # Coverage emits even when sentence_lens is empty.
        emit, lines = _capture()
        _emit_sentence_block(
            emit,
            SentenceStats(coverage_values=[0.95]),
        )
        assert any("Split coverage:   95%" in ln for ln in lines)
        assert not any("Mean sentence:" in ln for ln in lines)


# ---- Ordering invariant ---------------------------------------


class TestOrdering:
    def test_mean_then_range_then_coverage(self):
        emit, lines = _capture()
        _emit_sentence_block(
            emit,
            SentenceStats(
                sentence_lens=[70],
                min_chars_seen=10,
                max_chars_seen=140,
                coverage_values=[0.92],
            ),
        )

        def _idx(label: str) -> int:
            for i, ln in enumerate(lines):
                if label in ln:
                    return i
            return -1

        m_i = _idx("Mean sentence:")
        r_i = _idx("Sentence range:")
        c_i = _idx("Split coverage:")
        assert all(i >= 0 for i in (m_i, r_i, c_i))
        assert m_i < r_i < c_i
