"""Tests for iter-050 — TTS real-time factor metric.

Metric 1.11 from docs/perf-metrics-taxonomy.md. Symmetric to
iter-049's STT RTF:
    tts_rtf = tts_time / audio_seconds_total

  <1 = synth faster than realtime (overlap-friendly).
  >1 = synth is the bottleneck.
  0  = no audio produced.
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
        assert TurnMetrics().tts_rtf == 0.0


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_zero_rtf_no_rtf_in_tts_line(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            tts_time=0.05, sentences_spoken=1, tts_rtf=0.0,
        )
        out = self._capture(m)
        assert "TTS:" in out
        # Nothing matching the per-turn TTS RTF format on TTS line
        # (note: STT line might have RTF for other reasons; we
        # check for "RTF" in the output as a whole — should be
        # absent since we didn't set stt_rtf either).
        assert "RTF" not in out

    def test_nonzero_rtf_in_tts_line(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            tts_time=0.05, sentences_spoken=1, tts_rtf=0.10,
        )
        out = self._capture(m)
        assert "RTF 0.10x" in out

    def test_high_rtf_still_shown(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            tts_time=2.0, sentences_spoken=1, tts_rtf=2.0,
        )
        out = self._capture(m)
        assert "RTF 2.00x" in out


# ---- Session aggregate ---------------------------------------------------


def _m(rtf):
    return TurnMetrics(ttfs=0.5, tts_time=0.1, tts_rtf=rtf)


class TestSessionSummary:
    def test_no_rtf_omits_line(self):
        out = io.StringIO()
        print_session_summary([_m(0.0), _m(0.0)], {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        assert "Median TTS RTF" not in plain

    def test_with_data_emits_median(self):
        out = io.StringIO()
        print_session_summary(
            [_m(0.05), _m(0.10), _m(0.08)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Median TTS RTF:   0.08x" in plain

    def test_zero_filtered(self):
        out = io.StringIO()
        print_session_summary(
            [_m(0.0), _m(0.30), _m(0.20)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Median of [0.20, 0.30] = 0.25.
        assert "Median TTS RTF:   0.25x" in plain


# ---- ChatLoop arithmetic -------------------------------------------------


def _stt_engine(transcript="hi"):
    engine = SimpleNamespace(_last_text=None, model_repo="stub")
    def transcribe(wav):
        return transcript if wav else None
    return engine, transcribe


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
    def test_rtf_matches_tts_time_over_audio(self):
        # synth produces 1 sec of audio per call but takes 50ms.
        # RTF should be ~0.05.
        def synth(s):
            time.sleep(0.05)
            return np.full(24000, 0.5, dtype=np.float32), []  # 1 sec @ 24kHz

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
        # tts_time ~ 0.05s (one sentence), audio ~ 1s. RTF ~ 0.05.
        assert 0 < result.metrics.tts_rtf < 0.2
        # And the field equals tts_time / 1 second.
        assert result.metrics.tts_rtf == pytest.approx(
            result.metrics.tts_time / 1.0, rel=0.01,
        )

    def test_zero_when_no_audio(self):
        # synth returns empty audio → no audio_seconds → tts_rtf stays 0.
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
        assert result.metrics.tts_rtf == 0.0
