"""Tests for iter-096 — filler idle_threshold recommendation.

When filler false positives fire AND we know the current
idle_threshold AND we have llm_first_token observations, the
session summary appends a concrete recommended new threshold to
the FP-rate line: "tune idle_threshold up to N.Ns".
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
    FillerStats,
    SessionMeta,
    TurnMetrics,
    _emit_filler_block,
    print_session_summary,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _summary(metrics_list, **kwargs):
    out = io.StringIO()
    print_session_summary(metrics_list, {"model": "stub"}, file=out, **kwargs)
    return _strip_ansi(out.getvalue())


# ---- FillerStats field ----------------------------------------


class TestFillerStatsField:
    def test_default_zero(self):
        s = FillerStats()
        assert s.recommended_idle_threshold == 0.0


# ---- _emit_filler_block: recommendation rendering ------------


class TestRender:
    def _capture(self):
        lines: list[str] = []

        def emit(line=""):
            lines.append(_strip_ansi(line))

        return emit, lines

    def test_no_recommendation_legacy_text(self):
        emit, lines = self._capture()
        _emit_filler_block(
            emit,
            FillerStats(
                fillers_total=2, filler_turns=2, filler_false_positives=1,
                # recommended_idle_threshold defaults to 0.
            ),
        )
        # Legacy "tune idle_threshold up" suffix preserved when no
        # recommendation provided.
        line = next(ln for ln in lines if "Filler FP rate:" in ln)
        assert "— tune idle_threshold up" in line
        assert "to" not in line.split("up")[-1]  # No "up to ..."

    def test_recommendation_appended(self):
        emit, lines = self._capture()
        _emit_filler_block(
            emit,
            FillerStats(
                fillers_total=2, filler_turns=2, filler_false_positives=1,
                recommended_idle_threshold=1.2,
            ),
        )
        line = next(ln for ln in lines if "Filler FP rate:" in ln)
        assert "tune idle_threshold up to 1.2s" in line

    def test_no_fp_no_recommendation_either(self):
        # If FPs are zero, no FP rate line — recommendation moot.
        emit, lines = self._capture()
        _emit_filler_block(
            emit,
            FillerStats(
                fillers_total=2, filler_turns=2, filler_false_positives=0,
                recommended_idle_threshold=1.5,
            ),
        )
        assert not any("Filler FP rate:" in ln for ln in lines)


# ---- print_session_summary: end-to-end recommendation ---------


def _m(*, fillers_played=0, last_filler_id=0, filler_false_positive=False,
       llm_first_token=0.0, ttfs=0.5):
    return TurnMetrics(
        ttfs=ttfs,
        fillers_played=fillers_played,
        last_filler_id=last_filler_id,
        filler_false_positive=filler_false_positive,
        llm_first_token=llm_first_token,
    )


class TestEndToEnd:
    def test_no_threshold_no_recommendation(self):
        # idle_threshold not provided → no recommendation rendered
        # even when FPs fired. Falls back to legacy text.
        plain = _summary(
            [
                _m(fillers_played=1, last_filler_id=111,
                   filler_false_positive=True, llm_first_token=0.3),
                _m(fillers_played=1, last_filler_id=222,
                   filler_false_positive=True, llm_first_token=0.4),
            ],
        )
        assert "tune idle_threshold up" in plain
        assert "tune idle_threshold up to" not in plain

    def test_with_threshold_emits_recommendation(self):
        # Current threshold = 0.5s. Two turns with first_token at
        # 0.3s and 0.4s — both well below the threshold (FPs fired).
        # 75th percentile = 0.4s. Recommended = max(0.5*1.2=0.6,
        # 0.4+0.1=0.5) = 0.6s rounded.
        plain = _summary(
            [
                _m(fillers_played=1, last_filler_id=111,
                   filler_false_positive=True, llm_first_token=0.3),
                _m(fillers_played=1, last_filler_id=222,
                   filler_false_positive=True, llm_first_token=0.4),
            ],
            meta=SessionMeta(idle_threshold=0.5),
        )
        assert "tune idle_threshold up to 0.6s" in plain

    def test_high_first_token_drives_p75_branch(self):
        # When 75th percentile is much larger than current
        # threshold, the p75 + 0.1 branch wins over 1.2x current.
        plain = _summary(
            [
                _m(fillers_played=1, last_filler_id=111,
                   filler_false_positive=True, llm_first_token=0.5),
                _m(fillers_played=1, last_filler_id=222,
                   filler_false_positive=True, llm_first_token=1.5),
                _m(fillers_played=1, last_filler_id=333,
                   filler_false_positive=True, llm_first_token=2.0),
            ],
            meta=SessionMeta(idle_threshold=0.6),
        )
        # Sorted: [0.5, 1.5, 2.0]. p75 idx = round(0.75 * 2) = 2 →
        # value 2.0. Recommended = max(0.6*1.2=0.72, 2.0+0.1=2.1) =
        # 2.1s.
        assert "tune idle_threshold up to 2.1s" in plain

    def test_no_first_token_data_no_recommendation(self):
        # If we have FPs but no llm_first_token observations
        # (very unusual — every FP turn would have a first_token),
        # the p75 path can't compute and we fall back to legacy
        # text.
        plain = _summary(
            [
                _m(fillers_played=1, last_filler_id=111,
                   filler_false_positive=True, llm_first_token=0.0),
            ],
            meta=SessionMeta(idle_threshold=0.5),
        )
        assert "tune idle_threshold up" in plain
        assert "to 0." not in plain

    def test_clean_session_no_change(self):
        # No FPs → no FP-rate line at all. Legacy regression cover.
        plain = _summary(
            [
                _m(fillers_played=1, last_filler_id=111,
                   llm_first_token=0.4),
            ],
            meta=SessionMeta(idle_threshold=0.5),
        )
        assert "Filler FP rate" not in plain
