"""Tests for iter-061 — speaker open overhead metric.

Metric 2.8 from docs/perf-metrics-taxonomy.md.

    speaker_open_seconds = clock_after - clock_before

Time spent inside ``speaker_factory()`` in the SentenceWorker
thread when it opens the per-turn persistent output device. The
iter-008 win was holding ONE speaker across all sentences of a
turn (vs reopening per sentence) — if open cost balloons (driver
change, Bluetooth pairing, SDL/PortAudio init) TTFS regresses
silently. This metric makes that regression visible.
"""

from __future__ import annotations

import io
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_loop import ChatLoop  # noqa: E402
from examples._chat_metrics import (  # noqa: E402
    TurnMetrics,
    print_session_summary,
)
from examples._chat_pipeline import SentenceWorker  # noqa: E402
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
    def test_turnmetrics_default_zero(self):
        assert TurnMetrics().speaker_open_seconds == 0.0

    def test_worker_default_zero(self):
        # Constructor sets the field to 0.0 before _run() ever fires.
        w = SentenceWorker(
            speaker_factory=lambda: object(),
            synth_fn=lambda s: (np.zeros(8, dtype=np.float32), []),
            play_fn=lambda *a, **k: 0.0,
        )
        assert w.speaker_open_seconds == 0.0


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
        m = TurnMetrics(transcript="hi", model="stub", speaker_open_seconds=0.0)
        out = self._capture(m)
        assert "Speaker open" not in out

    def test_nonzero_emits_line(self):
        m = TurnMetrics(
            transcript="hi", model="stub", speaker_open_seconds=0.012,  # 12ms
        )
        out = self._capture(m)
        assert "Speaker open" in out
        assert "12ms" in out
        assert "device init" in out

    def test_above_threshold_emits(self):
        m = TurnMetrics(
            transcript="hi", model="stub", speaker_open_seconds=0.080,  # 80ms
        )
        out = self._capture(m)
        assert "Speaker open" in out
        assert "80ms" in out


# ---- Session aggregate ---------------------------------------------------


def _m(open_s=0.0):
    # ttfs > 0 keeps print_session_summary happy on the TTFS branch.
    return TurnMetrics(ttfs=0.5, speaker_open_seconds=open_s)


class TestSessionSummary:
    def test_no_data_omits_line(self):
        out = io.StringIO()
        print_session_summary([_m(), _m()], {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        assert "Speaker open" not in plain

    def test_single_value_no_median_label(self):
        out = io.StringIO()
        print_session_summary([_m(open_s=0.025)], {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        # Single observation: emit raw ms, no "median" / "worst" decoration
        # on the speaker-open line specifically (other rows still have
        # their own medians).
        speaker_lines = [ln for ln in plain.splitlines() if "Speaker open" in ln]
        assert len(speaker_lines) == 1
        assert "25ms" in speaker_lines[0]
        assert "median" not in speaker_lines[0]
        assert "worst" not in speaker_lines[0]

    def test_multi_value_emits_median_and_worst(self):
        out = io.StringIO()
        print_session_summary(
            [_m(open_s=0.020), _m(open_s=0.040), _m(open_s=0.080)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Median of [20, 40, 80] = 40. Worst = 80.
        assert "Speaker open:" in plain
        assert "median 40ms" in plain
        assert "worst 80ms" in plain

    def test_zeros_filtered(self):
        out = io.StringIO()
        print_session_summary(
            [_m(open_s=0.0), _m(open_s=0.030), _m(open_s=0.050)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Median of [30, 50] = 40 (zeros excluded).
        assert "median 40ms" in plain
        assert "worst 50ms" in plain


# ---- SentenceWorker timing --------------------------------------------


class TestWorkerTiming:
    def test_slow_speaker_factory_recorded(self):
        # Real clock: factory sleeps for a known duration, so the
        # measurement is dominated by that sleep. Generous tolerance
        # to keep the test resilient on a busy host.
        slept = 0.030

        def slow_factory():
            time.sleep(slept)
            return SimpleNamespace(write=lambda b: None, close=lambda: None)

        w = SentenceWorker(
            speaker_factory=slow_factory,
            synth_fn=lambda s: (np.zeros(8, dtype=np.float32), []),
            play_fn=lambda *a, **k: 0.0,
        )
        w.start()
        w.submit_done()
        w.wait_done(timeout=2.0)

        # Open captured the sleep window; tolerate scheduler jitter.
        assert w.speaker_open_seconds >= slept * 0.8
        assert w.speaker_open_seconds < slept + 0.5

    def test_factory_failure_leaves_zero(self):
        # If the factory raises, _run records the error and exits BEFORE
        # the post-open clock read — speaker_open_seconds stays at the
        # default 0.0. The test guards against accidentally surfacing a
        # spurious overhead value on failed-open turns.
        def boom():
            raise RuntimeError("no audio device")

        w = SentenceWorker(
            speaker_factory=boom,
            synth_fn=lambda s: (np.zeros(8, dtype=np.float32), []),
            play_fn=lambda *a, **k: 0.0,
        )
        w.start()
        w.submit_done()
        w.wait_done(timeout=1.0)

        assert w.speaker_open_seconds == 0.0
        assert len(w.errors) == 1
        assert "no audio device" in str(w.errors[0])


# ---- ChatLoop wiring (end-to-end through TurnMetrics) ----------------


def _stt_engine(transcript="hi"):
    engine = SimpleNamespace(_last_text=None, model_repo="stub")
    def transcribe(wav):
        return transcript if wav else None
    return engine, transcribe


def _const_synth(samples=2048):
    def synth(s):
        return np.full(samples, 0.5, dtype=np.float32), []
    return synth


def _fast_play(speaker, audio, tokens, *, is_first_sentence=False, cancel_event=None):
    audio_int16 = (audio * 32767).astype(np.int16)
    speaker.write(audio_int16.tobytes())
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


class TestChatLoopWiring:
    def test_speaker_open_lands_on_metrics(self):
        # Build a deliberately slow speaker factory and verify the
        # measurement bubbles all the way through to TurnMetrics.
        slow_open_seconds = 0.04

        def slow_factory():
            time.sleep(slow_open_seconds)
            return VirtualSpeakerStream(rate=24000)

        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine, transcribe = _stt_engine()
        loop = ChatLoop(
            mic=mic,
            speaker_factory=slow_factory,
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=_yield_tokens("Done."),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_fast_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Should be at least the sleep we forced. Generous upper bound
        # to keep the test resilient on slow CI.
        assert result.metrics.speaker_open_seconds >= slow_open_seconds * 0.8
        assert result.metrics.speaker_open_seconds < 1.0
