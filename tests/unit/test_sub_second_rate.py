"""Tests for iter-084 — sub-second turn rate.

Metric 3.19 from docs/perf-metrics-taxonomy.md.

    sub_second_rate = sum(1 for t in ttfs if 0 < t < 1.0) / len(ttfs)

Single human-feel threshold across the session — what fraction of
turns hit the snappy bar. Easier to track than median when
comparing across sessions or model swaps.
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


def _m(ttfs=0.5):
    return TurnMetrics(ttfs=ttfs)


# ---- No-emit boundaries -----------------------------------------


class TestNoEmit:
    def test_zero_turns_omits(self):
        plain = _summary([])
        assert "Sub-second TTFS" not in plain

    def test_no_audio_omits(self):
        # All turns ended without audio (ttfs == 0). The TTFS
        # block hits the n/a branch and the sub-second line is
        # gated on having at least one measurable TTFS.
        plain = _summary([_m(ttfs=0.0), _m(ttfs=0.0)])
        assert "Sub-second TTFS" not in plain


# ---- Emit cases -------------------------------------------------


class TestEmit:
    def test_all_sub_second(self):
        plain = _summary([_m(ttfs=0.5), _m(ttfs=0.7), _m(ttfs=0.9)])
        assert "Sub-second TTFS:  3/3 (100%)" in plain

    def test_none_sub_second(self):
        plain = _summary([_m(ttfs=1.5), _m(ttfs=2.0)])
        assert "Sub-second TTFS:  0/2 (0%)" in plain

    def test_mixed(self):
        plain = _summary([
            _m(ttfs=0.5),
            _m(ttfs=1.2),
            _m(ttfs=0.8),
            _m(ttfs=1.5),
        ])
        # 2 of 4 below 1.0s.
        assert "Sub-second TTFS:  2/4 (50%)" in plain

    def test_boundary_just_below_1s_counts(self):
        # Exactly 0.999s should count (strict less-than).
        plain = _summary([_m(ttfs=0.999), _m(ttfs=1.0)])
        # 1 of 2 (the 1.0 itself doesn't count).
        assert "Sub-second TTFS:  1/2 (50%)" in plain

    def test_zero_ttfs_excluded(self):
        # Filter mirrors ttfs_times in print_session_summary —
        # zero-TTFS turns are excluded from the denominator
        # because they're "no audio" not "slow".
        plain = _summary([_m(ttfs=0.0), _m(ttfs=0.5), _m(ttfs=0.6)])
        # 2 of 2 (zero excluded from denom, both remaining < 1.0).
        assert "Sub-second TTFS:  2/2 (100%)" in plain


# ---- Co-emission with neighboring lines -------------------------


class TestCoEmission:
    def test_after_best_ttfs(self):
        plain = _summary([_m(ttfs=0.5), _m(ttfs=0.7)])
        # Both Best TTFS and Sub-second TTFS lines present.
        assert "Best TTFS:" in plain
        assert "Sub-second TTFS:" in plain

    def test_with_rhythm_score(self):
        # ≥2 turns triggers rhythm score AND sub-second.
        plain = _summary([_m(ttfs=0.5), _m(ttfs=0.7)])
        assert "Sub-second TTFS:" in plain
        assert "Rhythm score:" in plain
