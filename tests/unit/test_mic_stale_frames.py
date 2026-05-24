"""Tests for iter-037 — mic stale-frame count surfaced on TurnMetrics.

Metric 2.19 in the perf-metrics taxonomy. ``flush_pending_audio``
already returned the drained count, but ChatLoop discarded it.
iter-037 captures it on ``TurnMetrics.mic_stale_frames`` and adds
both per-turn and session-aggregate display.

These tests verify:
  - The default value is 0 on TurnMetrics.
  - ChatLoop sets the field from flush_pending_audio's return.
  - The per-turn print() emits a "Mic stale: ..." line only when > 0.
  - The session summary aggregates and emits only when > 0.
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


# ---- Default value ---------------------------------------------------------


class TestDefault:
    def test_turnmetrics_defaults_to_zero(self):
        m = TurnMetrics()
        assert m.mic_stale_frames == 0


# ---- Per-turn print --------------------------------------------------------


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        # `TurnMetrics.print` writes via `print()` to stdout. Redirect.
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_zero_omits_line(self):
        m = TurnMetrics(transcript="hi", model="stub", mic_stale_frames=0)
        out = self._capture(m)
        assert "Mic stale" not in out

    def test_nonzero_emits_line(self):
        m = TurnMetrics(transcript="hi", model="stub", mic_stale_frames=320)
        out = self._capture(m)
        assert "Mic stale" in out
        assert "320 frames" in out

    def test_seconds_displayed(self):
        # 16000 frames at RATE=16000 = 1.0 seconds.
        m = TurnMetrics(
            transcript="hi", model="stub", mic_stale_frames=16000,
        )
        out = self._capture(m)
        assert "1.0s" in out


# ---- Session summary aggregate ---------------------------------------------


def _make(stale: int = 0) -> TurnMetrics:
    return TurnMetrics(
        ttfs=0.5, mic_stale_frames=stale,
    )


class TestSessionSummary:
    def test_zero_total_omits_line(self):
        out = io.StringIO()
        print_session_summary([_make(0), _make(0)], {"model": "stub"}, file=out)
        assert "Mic stale" not in _strip_ansi(out.getvalue())

    def test_nonzero_total_emits_aggregate(self):
        out = io.StringIO()
        print_session_summary(
            [_make(320), _make(640), _make(0)],
            {"model": "stub"},
            file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Mic stale:        960 frames" in plain
        # Suggestion text appears.
        assert "echo cancellation" in plain

    def test_seconds_total_in_aggregate(self):
        # 32000 frames total = 2.0s
        out = io.StringIO()
        print_session_summary(
            [_make(16000), _make(16000)],
            {"model": "stub"},
            file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "(2.0s)" in plain


# ---- ChatLoop wiring -------------------------------------------------------


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


def _yield_tokens(text, *, per_token_delay=0.0):
    import re

    def factory(messages, config):
        for p in re.findall(r"\S+|\.|!|\?", text):
            if per_token_delay > 0:
                time.sleep(per_token_delay)
            yield p + " "

    return factory


class TestChatLoopCapturesStaleFrames:
    """Drive a turn through ChatLoop with extra audio sitting in the
    mic buffer. ``flush_pending_audio`` should drain it AFTER the
    recorder consumes the utterance, and the count should land on
    metrics.

    Note: the count includes trailing silence that wasn't read by
    the recorder before VAD's DONE_OK fired. This is honest behavior
    — every audio chunk in the buffer is "stale" relative to the
    just-completed utterance. Future work could distinguish silent
    stale (harmless) from voiced stale (real echo) via RMS, but
    iter-037 just exposes the raw count.
    """

    def test_field_is_populated_on_metrics(self):
        # Even a clean utterance leaves trailing silence in the
        # buffer that the recorder didn't consume before DONE_OK.
        # The count must be set (>=0), not None or unset.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(concat(
            make_silence(0.3, rate=RATE),
            make_tone_burst(1.0, rate=RATE, amp=0.3),
            make_silence(1.5, rate=RATE),
        ))
        engine, transcribe = _stt_engine()

        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=_yield_tokens("OK."),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_slow_play,
        )

        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Field is set (an int, not None / not the default-without-write).
        assert isinstance(result.metrics.mic_stale_frames, int)
        assert result.metrics.mic_stale_frames >= 0

    def test_extra_audio_is_counted(self):
        # Simulate echo / leak: push utterance + BIG extra tone burst
        # right after, with no silence gap. The recorder VAD will
        # see "still speaking" and try to keep recording, but the
        # trailing silence eventually triggers DONE_OK and the burst
        # extras pile up. flush_pending_audio drains them.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(concat(
            make_silence(0.3, rate=RATE),
            make_tone_burst(0.5, rate=RATE, amp=0.3),
            make_silence(1.5, rate=RATE),  # closes the recording window
            # Extra audio that'll be flushed:
            make_tone_burst(1.0, rate=RATE, amp=0.3),
        ))
        engine, transcribe = _stt_engine()

        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=_yield_tokens("OK."),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_slow_play,
        )

        result = loop.run_one_turn([])
        assert result.metrics is not None
        # flush_pending_audio drained at least one chunk worth of
        # extra audio. Exact count depends on how fast the watcher
        # picks it up afterwards, so just assert >0.
        assert result.metrics.mic_stale_frames > 0

    def test_extra_audio_strictly_more_than_baseline(self):
        # Compare two runs: one with just trailing silence, one
        # with an extra burst on top. The extra-burst run should
        # leave strictly more frames in the buffer for the flush.
        def _run(extra_burst: bool) -> int:
            mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
            chunks = [
                make_silence(0.3, rate=RATE),
                make_tone_burst(0.5, rate=RATE, amp=0.3),
                make_silence(1.5, rate=RATE),
            ]
            if extra_burst:
                chunks.append(make_tone_burst(1.0, rate=RATE, amp=0.3))
            mic.push(concat(*chunks))
            engine, transcribe = _stt_engine()
            loop = ChatLoop(
                mic=mic,
                speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
                stt_engine=engine,
                transcribe_fn=transcribe,
                llm_stream_fn=_yield_tokens("OK."),
                llm_config={"model": "stub"},
                synth_fn=_const_synth(),
                play_fn=_slow_play,
            )
            result = loop.run_one_turn([])
            assert result.metrics is not None
            return result.metrics.mic_stale_frames

        baseline = _run(extra_burst=False)
        with_extra = _run(extra_burst=True)
        # With the extra burst, MORE bytes survived. Some watcher /
        # post-DONE_OK timing wiggle is normal, so use strictly
        # greater rather than exact deltas.
        assert with_extra > baseline, (
            f"extra_burst run did not exceed baseline: "
            f"baseline={baseline}, with_extra={with_extra}"
        )
