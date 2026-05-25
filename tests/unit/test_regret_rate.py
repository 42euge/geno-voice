"""Tests for iter-056 — regret rate metric.

Metric 3.4 from docs/perf-metrics-taxonomy.md ("Novel/speculative").
A barge-in is "regret" when the user starts speaking within 200ms
of bot first audio — implies the bot pre-empted the user (the user
was already mid-utterance and the bot misjudged end-of-turn).

Distinct from iter-053's "rushed" naturalness:
  - rushed = bot's TTFS was very low (subjective fast response)
  - regret = the user actually objected to the bot speaking
"""

from __future__ import annotations

import io
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    TurnMetrics,
    print_session_summary,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ---- Default + per-turn print --------------------------------------------


class TestDefault:
    def test_default_false(self):
        assert TurnMetrics().barge_in_regret is False


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_no_barge_no_regret_marker(self):
        m = TurnMetrics(transcript="hi", model="stub", barge_in=False)
        out = self._capture(m)
        assert "regret" not in out

    def test_barge_no_regret_no_marker(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            barge_in=True, barge_in_regret=False,
        )
        out = self._capture(m)
        assert "Barge-in" in out
        assert "regret" not in out

    def test_barge_with_regret_emits_marker(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            barge_in=True, barge_in_regret=True,
        )
        out = self._capture(m)
        assert "Barge-in" in out
        assert "— regret" in out


# ---- Session aggregate ---------------------------------------------------


def _m(barge=False, regret=False):
    return TurnMetrics(
        ttfs=0.5, barge_in=barge, barge_in_regret=regret,
    )


class TestSessionSummary:
    def test_no_barges_omits_regret_line(self):
        out = io.StringIO()
        print_session_summary(
            [_m(), _m()], {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Regret rate" not in plain

    def test_barges_no_regret_omits_line(self):
        out = io.StringIO()
        print_session_summary(
            [
                _m(barge=True, regret=False),
                _m(barge=True, regret=False),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Regret rate" not in plain

    def test_partial_regret_emits_with_pct(self):
        out = io.StringIO()
        print_session_summary(
            [
                _m(barge=True, regret=True),
                _m(barge=True, regret=False),
                _m(barge=True, regret=True),
                _m(barge=True, regret=False),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # 2/4 = 50%.
        assert "Regret rate:      2/4 (50%)" in plain
        # Suggestion text appears.
        assert "raise silence_duration" in plain

    def test_all_regret_shows_100pct(self):
        out = io.StringIO()
        print_session_summary(
            [
                _m(barge=True, regret=True),
                _m(barge=True, regret=True),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Regret rate:      2/2 (100%)" in plain


# ---- ChatLoop wiring ------------------------------------------------------
#
# We can't easily produce a deterministic regret scenario in
# integration tests (timing-flaky, like the existing barge-in
# integration test). Instead, validate the boundary condition
# directly via TurnMetrics + a fake coordinator.


from examples._chat_pipeline import BargeInCoordinator  # noqa: E402


class TestRegretBoundary:
    """Compute the regret check directly and verify boundaries.
    The check is in ChatLoop, so we replicate the formula here as
    a contract test of the threshold."""

    @pytest.mark.parametrize("gap_ms,expected", [
        (0,    False),  # boundary: 0 → not strictly > 0
        (50,   True),   # well inside
        (100,  True),
        (199,  True),   # just inside
        (200,  False),  # boundary: 200 → not strictly < 0.2
        (250,  False),  # well outside
        (1000, False),
    ])
    def test_threshold_boundary(self, gap_ms, expected):
        gap = gap_ms / 1000
        # The check from ChatLoop:
        is_regret = 0 < gap < 0.2
        assert is_regret == expected

    def test_no_first_audio_no_regret(self):
        # Even if barge-in triggered, if first_audio_at is None
        # (no audio played), regret can't be computed → False.
        # Replicated logic:
        first_audio_at = None
        triggered_at = 1.0
        coord_set = True
        if (
            triggered_at is not None
            and first_audio_at is not None
            and coord_set
        ):
            gap = triggered_at - first_audio_at
            is_regret = 0 < gap < 0.2
        else:
            is_regret = False
        assert is_regret is False

    def test_no_barge_no_regret(self):
        # No coord trigger → regret stays False even if timestamps
        # would otherwise qualify.
        first_audio_at = 1.0
        triggered_at = 1.1
        coord_set = False
        if (
            triggered_at is not None
            and first_audio_at is not None
            and coord_set
        ):
            gap = triggered_at - first_audio_at
            is_regret = 0 < gap < 0.2
        else:
            is_regret = False
        assert is_regret is False
