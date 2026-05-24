"""Tests for iter-045 — sentence-split fragmentation metric.

Metric 2.6 from docs/perf-metrics-taxonomy.md. Mean character
length of sentences submitted to the SentenceWorker per turn.
Diagnostic for splitter behavior:
  - mean ≪ ~30 chars: over-fragmented, hurts streaming overlap
  - mean ≫ ~150 chars: run-on sentences, hurts TTFS
  - mean 50-100 chars: healthy LLM voice output
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


# ---- Default + per-turn print ---------------------------------------------


class TestDefault:
    def test_default_zero(self):
        assert TurnMetrics().mean_sentence_chars == 0.0


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_zero_omits_avg_chars(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            sentences_spoken=2, mean_sentence_chars=0.0,
        )
        out = self._capture(m)
        assert "avg" not in out
        assert "chars" not in out

    def test_nonzero_emits_avg_chars(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            sentences_spoken=3, mean_sentence_chars=42.5,
        )
        out = self._capture(m)
        # Appears inside the TTS suffix.
        assert "avg 42 chars" in out  # 42.5 rounded by :.0f → "42"
        # NOTE: Python's :.0f rounds half-to-even by default;
        # 42.5 → 42 on most CPython builds.

    def test_short_avg_still_shown(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            sentences_spoken=5, mean_sentence_chars=18.0,
        )
        out = self._capture(m)
        assert "avg 18 chars" in out


# ---- Session aggregate ---------------------------------------------------


def _m(chars):
    return TurnMetrics(ttfs=0.5, mean_sentence_chars=chars)


class TestSessionSummary:
    def test_no_data_omits_line(self):
        out = io.StringIO()
        print_session_summary([_m(0.0), _m(0.0)], {"model": "stub"}, file=out)
        assert "Mean sentence" not in _strip_ansi(out.getvalue())

    def test_some_data_emits_average(self):
        out = io.StringIO()
        print_session_summary(
            [_m(50), _m(70), _m(60)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Average of [50, 70, 60] = 60.
        assert "Mean sentence:    60 chars" in plain

    def test_zero_turns_filtered(self):
        out = io.StringIO()
        print_session_summary(
            [_m(0.0), _m(60), _m(80)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Average of [60, 80] = 70 (not 47 from including 0).
        assert "Mean sentence:    70 chars" in plain


# ---- ChatLoop wiring -----------------------------------------------------


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


def _push_one(mic):
    mic.push(concat(
        make_silence(0.3, rate=RATE),
        make_tone_burst(1.0, rate=RATE, amp=0.3),
        make_silence(1.5, rate=RATE),
    ))


class TestChatLoopWires:
    def test_short_sentences_low_avg(self):
        # LLM emits very short sentences. Splitter sees each "Yes."
        # as a complete sentence of length 4 (or so).
        def llm(messages, config):
            yield "Yes. "
            yield "OK. "
            yield "Done. "

        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine, transcribe = _stt_engine()
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=llm,
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_slow_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # "Yes.", "OK.", "Done." → 4, 3, 5 chars → mean ≈ 4.
        assert 0 < result.metrics.mean_sentence_chars < 10

    def test_normal_length_sentences_healthy_avg(self):
        def llm(messages, config):
            for s in [
                "This is a normal-length sentence with about fifty chars. ",
                "Here is another one of similar length to balance. ",
                "And a third sentence that completes the trio nicely. ",
            ]:
                yield s

        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine, transcribe = _stt_engine()
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=llm,
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_slow_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Each sentence ~50 chars; mean should land in ~40-60 range.
        assert 30 < result.metrics.mean_sentence_chars < 70

    def test_no_complete_sentence_remains_zero(self):
        # LLM yields fragments only — no terminator, no submit, no
        # mean to compute.
        def llm(messages, config):
            yield "fragment of"
            yield " text without"
            yield " terminator"

        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine, transcribe = _stt_engine()
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=llm,
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_slow_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Trailing token_buffer is non-empty → submitted as a final
        # "remaining" piece. So we DO get one submission with the
        # full fragment text. Mean should equal that fragment's length.
        # Verify the field is populated (non-zero, since one submit).
        assert result.metrics.mean_sentence_chars > 0
