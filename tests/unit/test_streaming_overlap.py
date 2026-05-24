"""Tests for iter-043 — streaming overlap ratio metric.

Metric 2.1 from docs/perf-metrics-taxonomy.md. The whole point of
iter-008's SentenceWorker is to run TTS in parallel with LLM token
receipt. This metric makes that parallelism observable.

Definition: max(0, llm_stream_done_at - first_audio_at) / llm_total.
  1.0 (cap) = audio overlapped the whole LLM stream.
  0.5      = audio overlapped half the LLM stream.
  0        = audio only started AFTER LLM finished (sequential).
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
        assert TurnMetrics().streaming_overlap_ratio == 0.0


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
        m = TurnMetrics(transcript="hi", model="stub", streaming_overlap_ratio=0.0)
        out = self._capture(m)
        assert "Overlap" not in out

    def test_nonzero_emits_pct(self):
        m = TurnMetrics(
            transcript="hi", model="stub", streaming_overlap_ratio=0.65,
        )
        out = self._capture(m)
        assert "Overlap" in out
        assert "65%" in out
        # And the explainer.
        assert "LLM↔TTS concurrency" in out

    def test_full_overlap_shown_as_100pct(self):
        m = TurnMetrics(
            transcript="hi", model="stub", streaming_overlap_ratio=1.0,
        )
        out = self._capture(m)
        assert "100%" in out


# ---- Session aggregate ----------------------------------------------------


def _m(ratio):
    return TurnMetrics(ttfs=0.5, streaming_overlap_ratio=ratio)


class TestSessionSummary:
    def test_no_overlap_data_omits_line(self):
        out = io.StringIO()
        print_session_summary([_m(0.0), _m(0.0)], {"model": "stub"}, file=out)
        assert "Median overlap" not in _strip_ansi(out.getvalue())

    def test_some_data_emits_median(self):
        out = io.StringIO()
        print_session_summary(
            [_m(0.4), _m(0.6), _m(0.5)],
            {"model": "stub"}, file=out,
        )
        assert "Median overlap:   50%" in _strip_ansi(out.getvalue())

    def test_zero_turns_filtered(self):
        # 0.0 turns are filtered (sequential turns, not failures —
        # but "median including zeros" misleads in the same way as
        # iter-031's TTFS filter).
        out = io.StringIO()
        print_session_summary(
            [_m(0.0), _m(0.7), _m(0.7)],
            {"model": "stub"}, file=out,
        )
        # Median of [0.7, 0.7] = 70%, not 47% (which would be median
        # of [0, 0.7, 0.7]).
        assert "Median overlap:   70%" in _strip_ansi(out.getvalue())


# ---- ChatLoop arithmetic ---------------------------------------------------


def _stt_engine(transcript="hi"):
    engine = SimpleNamespace(_last_text=None, model_repo="stub")

    def transcribe(wav):
        return transcript if wav else None

    return engine, transcribe


def _const_synth(samples=2048):
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
    import re as _re

    def factory(messages, config):
        for p in _re.findall(r"\S+|\.|!|\?", text):
            if per_token_delay > 0:
                time.sleep(per_token_delay)
            yield p + " "

    return factory


def _push_one(mic):
    mic.push(concat(
        make_silence(0.3, rate=RATE),
        make_tone_burst(1.0, rate=RATE, amp=0.3),
        make_silence(1.5, rate=RATE),
    ))


class TestChatLoopOverlap:
    def test_overlap_set_when_audio_played_during_llm(self):
        # Slow LLM (per-token sleep) + small synth → audio starts
        # well before LLM stream ends.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine, transcribe = _stt_engine()
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=_yield_tokens(
                "First sentence. Second one. Third here.",
                per_token_delay=0.02,
            ),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(samples=512),
            play_fn=_slow_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Some overlap. Exact ratio is timing-dependent; just assert > 0.
        assert result.metrics.streaming_overlap_ratio > 0
        # And is bounded in [0, 1].
        assert result.metrics.streaming_overlap_ratio <= 1.0

    def test_zero_when_no_audio(self):
        # LLM yields fragments only, no terminator → no sentence,
        # no synth, no audio → first_audio_at stays None → ratio = 0.
        def llm(messages, config):
            yield "no"
            yield " terminator"
            yield " here"

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
        assert result.metrics.streaming_overlap_ratio == 0.0
