"""Tests for iter-049 — STT real-time factor metric.

Metric 1.7 from docs/perf-metrics-taxonomy.md. Trivial derivation:
    stt_rtf = stt_time / speech_duration

  <1 = STT runs faster than realtime (can be invoked inline).
  >1 = STT is the bottleneck (need streaming partial transcription).
  0  = speech_duration was 0 (false trigger turn).
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
    def test_default_zero(self):
        assert TurnMetrics().stt_rtf == 0.0


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_zero_rtf_falls_back_to_plain_stt_line(self):
        # Without RTF, the STT line should still render the ms value
        # exactly as before iter-049.
        m = TurnMetrics(transcript="hi", model="stub", stt_time=0.05, stt_rtf=0.0)
        out = self._capture(m)
        assert "STT:" in out
        assert "RTF" not in out

    def test_subreal_rtf_shown(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            stt_time=0.05, stt_rtf=0.05,  # 50ms STT on 1s speech
        )
        out = self._capture(m)
        assert "RTF 0.05x" in out

    def test_realtime_threshold_color_change(self):
        # Just verify the line contains the value — color codes are
        # stripped by _strip_ansi. We're not asserting on the color
        # itself here, just that >=1 RTF still renders.
        m = TurnMetrics(
            transcript="hi", model="stub",
            stt_time=2.0, stt_rtf=2.0,
        )
        out = self._capture(m)
        assert "RTF 2.00x" in out


# ---- Session aggregate ---------------------------------------------------


def _m(rtf):
    return TurnMetrics(ttfs=0.5, stt_time=0.05, stt_rtf=rtf)


class TestSessionSummary:
    def test_no_rtf_omits_line(self):
        out = io.StringIO()
        print_session_summary([_m(0.0), _m(0.0)], {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        assert "Median STT RTF" not in plain

    def test_some_rtfs_emit_median(self):
        out = io.StringIO()
        print_session_summary(
            [_m(0.05), _m(0.10), _m(0.08)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Median of [0.05, 0.08, 0.10] = 0.08.
        assert "Median STT RTF:   0.08x" in plain

    def test_zero_rtfs_filtered(self):
        out = io.StringIO()
        print_session_summary(
            [_m(0.0), _m(0.20), _m(0.30)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Median of [0.20, 0.30] = 0.25.
        assert "Median STT RTF:   0.25x" in plain


# ---- ChatLoop arithmetic -------------------------------------------------


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


class TestChatLoopArithmetic:
    def test_rtf_computed_when_speech_duration_positive(self):
        # Use a transcribe_fn with controlled latency so RTF is
        # measurable AND non-trivial.
        slow_transcribe_started = False

        def slow_transcribe(wav):
            nonlocal slow_transcribe_started
            if not wav:
                return None
            time.sleep(0.05)  # 50ms STT
            return "hi"

        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine = SimpleNamespace(_last_text=None, model_repo="stub")
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=slow_transcribe,
            llm_stream_fn=_yield_tokens("Done."),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_slow_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        assert result.metrics.speech_duration > 0
        assert result.metrics.stt_time > 0
        # RTF > 0; should equal stt_time / speech_duration.
        expected = result.metrics.stt_time / result.metrics.speech_duration
        assert result.metrics.stt_rtf == pytest.approx(expected, rel=0.01)
        # And the recorded RTF is small (50ms STT on ~1s speech).
        assert result.metrics.stt_rtf < 0.2
