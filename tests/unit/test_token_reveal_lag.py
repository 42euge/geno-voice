"""Tests for iter-071 — token-reveal lag.

Metric 2.17 from docs/perf-metrics-taxonomy.md.

For each token printed during play_aligned, lag is:

    lag = (clock_at_emit - t0) - token["start"]

Positive = text falls behind audio (UX feels broken).
Negative = text leads audio (spoils the bot before it speaks).
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

from examples._chat_loop import ChatLoop  # noqa: E402
from examples._chat_metrics import (  # noqa: E402
    TurnMetrics,
    print_session_summary,
)
from examples._chat_pipeline import (  # noqa: E402
    SentenceWorker,
    _play_fn_accepts_lag_out,
)
from examples._chat_playback import play_aligned  # noqa: E402
from examples._chat_recording import CHUNK, RATE  # noqa: E402
from examples.virtual_audio import (  # noqa: E402
    VirtualMicStream,
    VirtualSpeakerStream,
    concat,
    make_silence,
    make_tone_burst,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ---- play_aligned lag_out contract ----------------------------------


class TestPlayAlignedLagOut:
    def _audio(self, seconds=0.4, rate=24000):
        return np.zeros(int(rate * seconds), dtype=np.float32)

    def test_lag_out_optional(self):
        # Default behavior — no lag_out passed → no error.
        speaker = VirtualSpeakerStream(rate=24000)
        elapsed = play_aligned(
            speaker, self._audio(), [],
            output=io.StringIO(),
            clock=lambda: 0.0,
        )
        assert isinstance(elapsed, float)

    def test_lag_out_populated_with_tokens(self):
        # Two tokens whose start times the test clock matches:
        # we control the clock so the lag is deterministic.
        clock_seq = iter([0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
        clock = lambda: next(clock_seq)
        tokens = [
            {"text": "one", "start": 0.05},
            {"text": "two", "start": 0.15},
        ]
        speaker = VirtualSpeakerStream(rate=24000)
        out = {}
        play_aligned(
            speaker, self._audio(seconds=0.5), tokens,
            output=io.StringIO(),
            clock=clock,
            lag_out=out,
        )
        # We just assert the dict was populated with a sensible
        # shape — exact values depend on clock-call order which is
        # an implementation detail.
        assert "sum" in out
        assert "count" in out
        assert "max" in out
        assert out["count"] == 2

    def test_lag_out_empty_when_no_tokens(self):
        speaker = VirtualSpeakerStream(rate=24000)
        out = {}
        play_aligned(
            speaker, self._audio(), [],
            output=io.StringIO(),
            clock=lambda: 0.0,
            lag_out=out,
        )
        # play_aligned only writes the dict when at least one token
        # was emitted — empty signals "no data."
        assert out == {}


# ---- Worker accepts/sniffs lag_out ----------------------------------


class TestPlayFnSniff:
    def test_supports_explicit_kwarg(self):
        def play(speaker, audio, tokens, *, is_first_sentence=False, lag_out=None):
            return 0.0
        assert _play_fn_accepts_lag_out(play) is True

    def test_supports_var_keyword(self):
        def play(speaker, audio, tokens, *, is_first_sentence=False, **kwargs):
            return 0.0
        assert _play_fn_accepts_lag_out(play) is True

    def test_no_support(self):
        def play(speaker, audio, tokens, *, is_first_sentence=False):
            return 0.0
        assert _play_fn_accepts_lag_out(play) is False


# ---- Worker accumulates per-call lag stats ---------------------------


class TestWorkerAccumulates:
    def _make(self, play_fn):
        return SentenceWorker(
            speaker_factory=lambda: SimpleNamespace(
                write=lambda b: None, close=lambda: None,
            ),
            synth_fn=lambda s: (np.zeros(8, dtype=np.float32), []),
            play_fn=play_fn,
        )

    def test_no_support_leaves_zero(self):
        def play(speaker, audio, tokens, *, is_first_sentence=False):
            return 0.0
        w = self._make(play)
        w.start()
        w.submit("hi")
        w.submit_done()
        w.wait_done(timeout=2.0)
        assert w.token_reveal_lag_count == 0
        assert w.token_reveal_lag_sum == 0.0

    def test_accumulates_across_calls(self):
        # Stub play_fn populates the dict deterministically each call.
        per_call = [
            {"sum": 0.10, "count": 2, "max": 0.08},   # call 1
            {"sum": 0.05, "count": 1, "max": 0.05},   # call 2
        ]
        call_idx = [0]

        def play(speaker, audio, tokens, *, is_first_sentence=False, lag_out=None):
            if lag_out is not None:
                payload = per_call[call_idx[0]]
                lag_out["sum"] = payload["sum"]
                lag_out["count"] = payload["count"]
                lag_out["max"] = payload["max"]
                call_idx[0] += 1
            return 0.0

        w = self._make(play)
        w.start()
        w.submit("first")
        w.submit("second")
        w.submit_done()
        w.wait_done(timeout=2.0)

        assert w.token_reveal_lag_count == 3
        assert w.token_reveal_lag_sum == pytest.approx(0.15)
        # max is the largest abs across calls.
        assert w.token_reveal_lag_max == pytest.approx(0.08)

    def test_max_picks_largest_absolute(self):
        # Negative-lag-larger-than-positive should win.
        calls = [
            {"sum": 0.05, "count": 1, "max": 0.05},
            {"sum": -0.20, "count": 1, "max": -0.20},  # bigger |·|
            {"sum": 0.10, "count": 1, "max": 0.10},
        ]
        idx = [0]

        def play(speaker, audio, tokens, *, is_first_sentence=False, lag_out=None):
            if lag_out is not None:
                p = calls[idx[0]]
                lag_out.update(p)
                idx[0] += 1
            return 0.0

        w = self._make(play)
        w.start()
        for s in ("a", "b", "c"):
            w.submit(s)
        w.submit_done()
        w.wait_done(timeout=2.0)

        assert w.token_reveal_lag_max == pytest.approx(-0.20)


# ---- TurnMetrics + print -------------------------------------------


class TestTurnMetricsAndPrint:
    def _capture(self, m):
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_default_zero(self):
        m = TurnMetrics()
        assert m.mean_token_reveal_lag == 0.0
        assert m.max_token_reveal_lag == 0.0

    def test_zero_omits_line(self):
        m = TurnMetrics(transcript="hi", model="stub")
        assert "Token-reveal" not in self._capture(m)

    def test_positive_lag_emits(self):
        m = TurnMetrics(transcript="hi", model="stub",
                        mean_token_reveal_lag=0.030,
                        max_token_reveal_lag=0.080)
        out = self._capture(m)
        assert "Token-reveal:" in out
        assert "+30ms mean" in out
        assert "+80ms peak" in out

    def test_negative_lag_emits_with_sign(self):
        m = TurnMetrics(transcript="hi", model="stub",
                        mean_token_reveal_lag=-0.040,
                        max_token_reveal_lag=-0.120)
        out = self._capture(m)
        assert "-40ms mean" in out
        assert "-120ms peak" in out


# ---- Session aggregate ---------------------------------------------


def _m(mean=0.0, mx=0.0):
    return TurnMetrics(ttfs=0.5,
                       mean_token_reveal_lag=mean,
                       max_token_reveal_lag=mx)


class TestSessionAggregate:
    def test_no_data_omits(self):
        out = io.StringIO()
        print_session_summary([_m(), _m()], {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        assert "Token lag" not in plain

    def test_emits_median_and_worst(self):
        out = io.StringIO()
        print_session_summary(
            [_m(mean=0.030, mx=0.060),
             _m(mean=0.040, mx=0.090),
             _m(mean=0.020, mx=0.040)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Median of [30, 40, 20] = 30. Worst peak across [60, 90, 40] = 90.
        assert "Token lag:        +30ms median, +90ms worst peak" in plain

    def test_negative_worst_picks_largest_abs(self):
        out = io.StringIO()
        print_session_summary(
            [_m(mean=0.020, mx=0.050),
             _m(mean=0.010, mx=-0.150)],  # |−150|=150 > 50
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "-150ms worst peak" in plain
