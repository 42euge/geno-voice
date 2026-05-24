"""Tests for iter-038 — LLM time-to-first-sentence metric.

Metric 1.10 from docs/perf-metrics-taxonomy.md. Distinct from
``llm_first_token``: the LLM may stream chatty preamble for a
while before a sentence terminator arrives. ``llm_first_sentence``
captures the moment the first complete sentence reaches the TTS
worker.

These tests verify:
  - TurnMetrics defaults to 0.
  - Per-turn print emits the line only when > 0; shows preamble gap.
  - Session summary aggregates only when at least one turn has > 0.
  - ChatLoop sets the field from the splitter's first non-empty
    yield.
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


# ---- Default value ----------------------------------------------------------


class TestDefault:
    def test_turnmetrics_defaults_to_zero(self):
        m = TurnMetrics()
        assert m.llm_first_sentence == 0.0


# ---- Per-turn print --------------------------------------------------------


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
        m = TurnMetrics(transcript="hi", model="stub", llm_first_sentence=0.0)
        out = self._capture(m)
        assert "LLM 1st sent" not in out

    def test_nonzero_emits_line_with_preamble_gap(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            llm_first_token=0.05,        # 50ms
            llm_first_sentence=0.20,     # 200ms
        )
        out = self._capture(m)
        assert "LLM 1st sent" in out
        assert "200ms" in out
        # Preamble gap = 200ms - 50ms = 150ms, shown in parens.
        assert "+150ms preamble" in out

    def test_zero_preamble_when_first_sentence_equals_first_token(self):
        # Edge case: the very first token is itself a sentence
        # terminator (e.g. LLM yielded "OK." as one chunk).
        m = TurnMetrics(
            transcript="hi", model="stub",
            llm_first_token=0.05,
            llm_first_sentence=0.05,
        )
        out = self._capture(m)
        assert "LLM 1st sent" in out
        # Preamble gap = 0ms.
        assert "+0ms preamble" in out


# ---- Session summary aggregate ---------------------------------------------


class TestSessionSummary:
    def test_no_turns_omits_line(self):
        out = io.StringIO()
        print_session_summary(
            [TurnMetrics(ttfs=0.5, llm_first_sentence=0.0)],
            {"model": "stub"}, file=out,
        )
        assert "Median LLM sent" not in _strip_ansi(out.getvalue())

    def test_some_turns_emits_median(self):
        # Median of [0.1, 0.3] = 0.2 → 200ms.
        out = io.StringIO()
        print_session_summary(
            [
                TurnMetrics(ttfs=0.5, llm_first_sentence=0.1),
                TurnMetrics(ttfs=0.6, llm_first_sentence=0.3),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Median LLM sent:  200ms" in plain

    def test_zero_turns_filtered_from_median(self):
        # Two turns at 0.4 + one at 0.0 → filter the 0, median of
        # [0.4, 0.4] = 400ms (not biased toward 0).
        out = io.StringIO()
        print_session_summary(
            [
                TurnMetrics(ttfs=0.5, llm_first_sentence=0.4),
                TurnMetrics(ttfs=0.5, llm_first_sentence=0.0),  # excluded
                TurnMetrics(ttfs=0.5, llm_first_sentence=0.4),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Median LLM sent:  400ms" in plain


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


class TestChatLoopCapturesFirstSentence:
    def test_first_sentence_recorded_when_terminator_arrives(self):
        # LLM streams "Hello. Done." — first sentence at the period
        # after "Hello".
        def llm(messages, config):
            yield "Hello"
            yield ". "
            yield "Done"
            yield ". "

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
            llm_stream_fn=llm,
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_slow_play,
        )

        result = loop.run_one_turn([])
        assert result.metrics is not None
        # First sentence emerged; metric is set.
        assert result.metrics.llm_first_sentence > 0
        # And it's >= first_token (sentence emerged at-or-after first
        # token).
        assert (
            result.metrics.llm_first_sentence
            >= result.metrics.llm_first_token
        )

    def test_zero_when_no_complete_sentence(self):
        # LLM yields tokens but never a terminator — first_sentence
        # never gets stamped.
        def llm(messages, config):
            yield "fragment"
            yield " of"
            yield " text"
            # No '.', '!', '?' followed by space.

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
            llm_stream_fn=llm,
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_slow_play,
        )

        result = loop.run_one_turn([])
        assert result.metrics is not None
        assert result.metrics.llm_first_sentence == 0

    def test_first_sentence_with_preamble_is_later_than_first_token(self):
        # LLM yields a long preamble of fragments then a terminator.
        # first_sentence MUST be measurably after first_token.
        def llm(messages, config):
            yield "alpha "
            time.sleep(0.02)
            yield "beta "
            time.sleep(0.02)
            yield "gamma. "  # terminator here
            yield "Done."

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
            llm_stream_fn=llm,
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_slow_play,
        )

        result = loop.run_one_turn([])
        assert result.metrics is not None
        assert result.metrics.llm_first_sentence > 0
        # Preamble took ≥40ms; first_sentence should be at least
        # that much later than first_token.
        gap = result.metrics.llm_first_sentence - result.metrics.llm_first_token
        assert gap >= 0.03, f"Expected preamble gap, got {gap*1000:.1f}ms"
