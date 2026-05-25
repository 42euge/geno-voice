"""Tests for iter-057 — primed-frames replay duration metric.

Metric 2.12 from docs/perf-metrics-taxonomy.md. The chat_loop
already prints `len(next_primed) * chunk / rate` on barge-in
turns; iter-057 promotes that to a TurnMetrics field so the
session summary can aggregate it.

Diagnostic value: validates the iter-025 lead-in (BargeInWatcher's
ring buffer of pre-detection frames). High totals = the watcher
is preserving meaningful user audio for the next STT pass; near-
zero = the lead-in is barely contributing.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

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
    def test_default_zero(self):
        assert TurnMetrics().primed_frames_seconds == 0.0


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_zero_omits_line(self):
        m = TurnMetrics(transcript="hi", model="stub", primed_frames_seconds=0.0)
        out = self._capture(m)
        assert "Primed frames" not in out

    def test_nonzero_emits_line(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            barge_in=True, primed_frames_seconds=0.45,  # 450ms
        )
        out = self._capture(m)
        assert "Primed frames:" in out
        assert "450ms" in out
        assert "carried into next turn" in out

    def test_no_barge_with_zero_omits(self):
        # Sanity: clean turn, no primed frames, no line.
        m = TurnMetrics(transcript="hi", model="stub", barge_in=False)
        out = self._capture(m)
        assert "Primed frames" not in out


# ---- Session aggregate ---------------------------------------------------


def _m(seconds=0.0):
    return TurnMetrics(ttfs=0.5, primed_frames_seconds=seconds)


class TestSessionSummary:
    def test_no_primed_omits_line(self):
        out = io.StringIO()
        print_session_summary([_m(), _m()], {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        assert "Primed audio" not in plain

    def test_some_primed_emits_total(self):
        out = io.StringIO()
        print_session_summary(
            [_m(0.3), _m(0.5), _m(0.0)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Total: 0.3 + 0.5 = 0.8s.
        assert "Primed audio:     0.8s" in plain
        assert "validates iter-025" in plain

    def test_single_primed_turn(self):
        out = io.StringIO()
        print_session_summary(
            [_m(1.5)], {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Primed audio:     1.5s" in plain


# ---- Computation contract -------------------------------------------------


class TestComputation:
    """Document the formula used in ChatLoop:
        primed_frames_seconds = len(next_primed) * chunk / rate

    With CHUNK=1024 and RATE=16000:
        1 frame  = 64ms
        16 frames = 1024ms = 1.024s
    """

    def test_formula_at_default_chunk_rate(self):
        from examples._chat_recording import CHUNK, RATE
        # 16 frames at 1024 chunk / 16000 rate = 1.024 sec.
        n_frames = 16
        seconds = n_frames * CHUNK / RATE
        assert seconds == pytest.approx(1.024, rel=0.001)

    def test_formula_zero_frames(self):
        from examples._chat_recording import CHUNK, RATE
        seconds = 0 * CHUNK / RATE
        assert seconds == 0.0


# pytest module-level helper.
import pytest  # noqa: E402
