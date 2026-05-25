"""Tests for iter-091 — _emit_filler_block helper.

Mirrors the iter-089/iter-090 pattern: extract a multi-line
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
    FillerStats,
    _emit_filler_block,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line: str = "") -> None:
        lines.append(_strip_ansi(line))

    return emit, lines


# ---- FillerStats defaults --------------------------------------


class TestDefaults:
    def test_all_zero(self):
        s = FillerStats()
        assert s.fillers_total == 0
        assert s.filler_turns == 0
        assert s.filler_false_positives == 0
        assert s.unique_filler_count == 0


# ---- No-data path ----------------------------------------------


class TestNoFillers:
    def test_zero_fillers_omits_everything(self):
        emit, lines = _capture()
        _emit_filler_block(emit, FillerStats())
        assert lines == []


# ---- Single-play sessions --------------------------------------


class TestSingleFiller:
    def test_emits_count_only(self):
        emit, lines = _capture()
        _emit_filler_block(
            emit,
            FillerStats(fillers_total=1, filler_turns=1, unique_filler_count=1),
        )
        # Only "Fillers played: 1". No FP rate (no FPs). No novelty
        # (single play is trivially 100%).
        assert any("Fillers played:   1" in ln for ln in lines)
        assert not any("Filler FP rate" in ln for ln in lines)
        assert not any("Filler novelty" in ln for ln in lines)


# ---- False-positive rate ---------------------------------------


class TestFalsePositiveRate:
    def test_emits_when_any_fp(self):
        emit, lines = _capture()
        _emit_filler_block(
            emit,
            FillerStats(
                fillers_total=4, filler_turns=4,
                filler_false_positives=2,
                unique_filler_count=2,
            ),
        )
        # 2/4 = 50%.
        assert any(
            "Filler FP rate:   2/4 (50%)" in ln for ln in lines
        )

    def test_omits_when_zero_fp(self):
        emit, lines = _capture()
        _emit_filler_block(
            emit,
            FillerStats(
                fillers_total=2, filler_turns=2,
                filler_false_positives=0,
                unique_filler_count=2,
            ),
        )
        assert not any("Filler FP rate" in ln for ln in lines)


# ---- Novelty index ---------------------------------------------


class TestNovelty:
    def test_emits_at_two_plays(self):
        emit, lines = _capture()
        _emit_filler_block(
            emit,
            FillerStats(
                fillers_total=2, filler_turns=2,
                unique_filler_count=2,
            ),
        )
        assert any(
            "Filler novelty:   2 unique / 2 (100%)" in ln for ln in lines
        )

    def test_emits_partial_diversity(self):
        emit, lines = _capture()
        _emit_filler_block(
            emit,
            FillerStats(
                fillers_total=4, filler_turns=4,
                unique_filler_count=2,
            ),
        )
        # 2 unique / 4 = 50%.
        assert any(
            "Filler novelty:   2 unique / 4 (50%)" in ln for ln in lines
        )

    def test_omits_at_single_play(self):
        emit, lines = _capture()
        _emit_filler_block(
            emit,
            FillerStats(
                fillers_total=1, filler_turns=1,
                unique_filler_count=1,
            ),
        )
        assert not any("Filler novelty" in ln for ln in lines)


# ---- Ordering invariant ---------------------------------------


class TestOrdering:
    def test_ordered_emit(self):
        # Played > FP > Novelty in stable order.
        emit, lines = _capture()
        _emit_filler_block(
            emit,
            FillerStats(
                fillers_total=4, filler_turns=4,
                filler_false_positives=1,
                unique_filler_count=3,
            ),
        )

        def _idx(label: str) -> int:
            for i, ln in enumerate(lines):
                if label in ln:
                    return i
            return -1

        played_i = _idx("Fillers played:")
        fp_i = _idx("Filler FP rate:")
        nov_i = _idx("Filler novelty:")
        assert all(i >= 0 for i in (played_i, fp_i, nov_i))
        assert played_i < fp_i < nov_i
