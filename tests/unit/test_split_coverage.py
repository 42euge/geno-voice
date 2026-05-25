"""Tests for iter-059 — sentence-split coverage metric.

Metric 2.5 from docs/perf-metrics-taxonomy.md.

    coverage = complete_sentence_chars / (complete + remainder)
    range [0, 1]

  1.0 = LLM always ended responses with punctuation; every char
        went to the worker as part of a complete sentence
        (overlap-friendly).
  0.5 = half the chars came as remainder (forced through at
        end-of-stream — can't overlap with next sentence).
  0.0 = LLM produced fragments only (no terminator); all chars
        flushed as remainder.
  0   = no chars submitted this turn.
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
        assert TurnMetrics().sentence_split_coverage == 0.0


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_perfect_coverage_omits_marker(self):
        # Don't clutter the line on perfect 100% — that's expected.
        m = TurnMetrics(
            transcript="hi", model="stub",
            sentences_spoken=2,
            sentence_split_coverage=1.0,
        )
        out = self._capture(m)
        assert "% complete" not in out

    def test_zero_coverage_omits_marker(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            sentences_spoken=0,
            sentence_split_coverage=0.0,
        )
        out = self._capture(m)
        assert "% complete" not in out

    def test_partial_coverage_emits_marker(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            sentences_spoken=2,
            sentence_split_coverage=0.75,
        )
        out = self._capture(m)
        assert "75% complete" in out


# ---- Session aggregate ---------------------------------------------------


def _m(coverage):
    return TurnMetrics(ttfs=0.5, sentence_split_coverage=coverage)


class TestSessionSummary:
    def test_no_data_omits_line(self):
        out = io.StringIO()
        print_session_summary([_m(0.0), _m(0.0)], {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        assert "Split coverage" not in plain

    def test_with_data_emits_median(self):
        out = io.StringIO()
        print_session_summary(
            [_m(0.5), _m(0.7), _m(1.0)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Median of [0.5, 0.7, 1.0] = 0.7 → 70%.
        assert "Split coverage:   70%" in plain

    def test_zero_filtered(self):
        out = io.StringIO()
        print_session_summary(
            [_m(0.0), _m(0.6), _m(0.8)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Median of [0.6, 0.8] = 0.7.
        assert "Split coverage:   70%" in plain


# ---- ChatLoop arithmetic --------------------------------------------------


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


class TestChatLoopArithmetic:
    def test_perfect_coverage_when_terminated(self):
        # LLM yields a complete sentence terminated cleanly. No
        # remainder → 100%.
        def llm(messages, config):
            yield "Hello world. "

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
        # All chars went through as a complete sentence — coverage 1.0.
        assert result.metrics.sentence_split_coverage == 1.0

    def test_zero_coverage_when_no_terminator(self):
        # LLM yields fragments only — no terminator. All chars get
        # flushed as the trailing remainder.
        def llm(messages, config):
            yield "fragment "
            yield "without "
            yield "terminator"

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
        # All chars were flushed as remainder → coverage 0.
        # Note: the field uses 0.0 as both "no data" AND "all-remainder"
        # — these are operationally distinct but the field can't
        # distinguish them. The session summary's >0 filter would
        # exclude this turn, which is acceptable since the meaningful
        # signal is "median > 0.5 = healthy".
        assert result.metrics.sentence_split_coverage == 0.0

    def test_partial_coverage_with_trailing_fragment(self):
        # LLM yields one complete sentence + a trailing fragment.
        def llm(messages, config):
            yield "Done."
            yield " "          # whitespace after period → split fires
            yield "trailing"   # no terminator → remainder

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
        # "Done." is 5 chars, "trailing" is 8 chars after .strip().
        # Coverage = 5 / (5 + 8) ≈ 0.385.
        assert 0 < result.metrics.sentence_split_coverage < 1.0
        # Specifically:
        assert result.metrics.sentence_split_coverage == pytest.approx(
            5 / 13, rel=0.01,
        )
