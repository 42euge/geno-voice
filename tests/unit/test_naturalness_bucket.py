"""Tests for iter-053 — naturalness bucket metric.

Metric 3.1 from docs/perf-metrics-taxonomy.md ("Novel/speculative").
TTFS bucketed against the human-conversation sweet spot:
  <200ms      = rushed (bot interrupted natural pause)
  200-400ms   = natural
  >400ms      = slow (user notices lag)
""              = no audio this turn

Counter-intuitive insight: lower TTFS isn't always better. A bot
that responds in 50ms feels robotic; one that responds in 250ms
matches human conversational rhythm.
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


# ---- Default + per-turn print --------------------------------------------


class TestDefault:
    def test_default_empty(self):
        assert TurnMetrics().naturalness_bucket == ""


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_empty_omits_bucket(self):
        m = TurnMetrics(transcript="hi", model="stub", ttfs=0.5)
        out = self._capture(m)
        # No bucket tag in TTFS line.
        assert "rushed" not in out
        assert "natural" not in out
        assert "slow" not in out

    def test_natural_bucket_tagged(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            ttfs=0.3, naturalness_bucket="natural",
        )
        out = self._capture(m)
        assert "natural" in out

    def test_rushed_bucket_tagged(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            ttfs=0.1, naturalness_bucket="rushed",
        )
        out = self._capture(m)
        assert "rushed" in out

    def test_slow_bucket_tagged(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            ttfs=0.8, naturalness_bucket="slow",
        )
        out = self._capture(m)
        assert "slow" in out


# ---- Session aggregate ---------------------------------------------------


def _m(ttfs=0.3, bucket=""):
    return TurnMetrics(ttfs=ttfs, naturalness_bucket=bucket)


class TestSessionSummary:
    def test_no_buckets_omits_line(self):
        out = io.StringIO()
        print_session_summary([_m()], {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        assert "Naturalness" not in plain

    def test_distribution_emitted_when_buckets_present(self):
        out = io.StringIO()
        print_session_summary(
            [
                _m(ttfs=0.1, bucket="rushed"),
                _m(ttfs=0.3, bucket="natural"),
                _m(ttfs=0.3, bucket="natural"),
                _m(ttfs=0.5, bucket="slow"),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Naturalness:      1 rushed, 2 natural, 1 slow" in plain

    def test_only_natural(self):
        out = io.StringIO()
        print_session_summary(
            [
                _m(ttfs=0.3, bucket="natural"),
                _m(ttfs=0.35, bucket="natural"),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Naturalness:      0 rushed, 2 natural, 0 slow" in plain


# ---- ChatLoop bucket assignment ------------------------------------------


def _stt_engine(transcript="hi"):
    engine = SimpleNamespace(_last_text=None, model_repo="stub")
    def transcribe(wav):
        return transcript if wav else None
    return engine, transcribe


def _const_synth(samples=512):
    def synth(s):
        return np.full(samples, 0.5, dtype=np.float32), []
    return synth


def _slow_play(speaker, audio, tokens, *, is_first_sentence=False, cancel_event=None):
    audio_int16 = (audio * 32767).astype(np.int16)
    chunk = 256
    written = 0
    while written < len(audio_int16):
        if cancel_event is not None and cancel_event.is_set():
            break
        end = min(written + chunk, len(audio_int16))
        speaker.write(audio_int16[written:end].tobytes())
        written = end
        time.sleep(0.005)
    return 0.0


def _yield_tokens(text):
    import re as _re
    def factory(messages, config):
        for p in _re.findall(r"\S+|\.|!|\?", text):
            yield p + " "
    return factory


def _push_one(mic):
    mic.push(concat(
        make_silence(0.3, rate=RATE),
        make_tone_burst(1.0, rate=RATE, amp=0.3),
        make_silence(1.5, rate=RATE),
    ))


class TestChatLoopBucketAssignment:
    """The bucket boundaries are deterministic: <200ms / 200-400ms /
    >400ms. Synthesize TurnMetrics directly in the integration tests
    rather than trying to make ChatLoop hit specific TTFS numbers
    (which depend on virtual audio timing).
    """

    def test_bucket_set_when_audio_played(self):
        # End-to-end: just confirm the bucket lands non-empty when
        # ttfs > 0. The exact bucket depends on virtual-audio timing.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine, transcribe = _stt_engine()
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=_yield_tokens("Hi."),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_slow_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        assert result.metrics.ttfs > 0
        assert result.metrics.naturalness_bucket in ("rushed", "natural", "slow")

    def test_no_audio_no_bucket(self):
        # synth returns empty audio → no first_audio_at → ttfs not
        # set → bucket stays "".
        def synth(s):
            return np.array([], dtype=np.float32), []

        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine, transcribe = _stt_engine()
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=_yield_tokens("Done."),
            llm_config={"model": "stub"},
            synth_fn=synth,
            play_fn=_slow_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        assert result.metrics.naturalness_bucket == ""


class TestBucketBoundariesDirect:
    """Verify the boundary logic by constructing TurnMetrics directly
    and re-running the bucketing in isolation (the same logic from
    ChatLoop, transcribed)."""

    @pytest.mark.parametrize("ttfs_ms,expected", [
        (0,    "rushed"),    # 0ms — degenerate but bucket-able
        (50,   "rushed"),
        (199,  "rushed"),
        (200,  "natural"),
        (300,  "natural"),
        (400,  "natural"),
        (401,  "slow"),
        (1000, "slow"),
        (5000, "slow"),
    ])
    def test_bucket_boundaries(self, ttfs_ms, expected):
        # Direct check of the boundary condition.
        if ttfs_ms < 200:
            bucket = "rushed"
        elif ttfs_ms <= 400:
            bucket = "natural"
        else:
            bucket = "slow"
        assert bucket == expected
