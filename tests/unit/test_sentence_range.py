"""Tests for iter-070 — sentence-length min/max range.

Metric 2.6 from docs/perf-metrics-taxonomy.md (histogram form).

The mean sentence length (iter-045) hides bimodal patterns: a turn
with sentences [10, 130] and a turn with [70, 70] both report
mean=70, but only the first has the "short interjection followed
by long monologue" fragmentation profile that defeats streaming
overlap. Min/max surface that.
"""

from __future__ import annotations

import io
import re
import sys
import threading
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
    def test_turnmetrics_defaults_zero(self):
        m = TurnMetrics()
        assert m.min_sentence_chars == 0
        assert m.max_sentence_chars == 0


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_no_mean_no_range(self):
        # Range only renders inside the mean's TTS suffix.
        m = TurnMetrics(transcript="hi", model="stub",
                        sentences_spoken=1,
                        mean_sentence_chars=0.0,
                        min_sentence_chars=10, max_sentence_chars=130)
        out = self._capture(m)
        assert "[10..130]" not in out

    def test_uniform_sentences_no_range_suffix(self):
        # min == max → no range suffix; would just be "[70..70]" noise.
        m = TurnMetrics(transcript="hi", model="stub",
                        sentences_spoken=2,
                        mean_sentence_chars=70.0,
                        min_sentence_chars=70, max_sentence_chars=70)
        out = self._capture(m)
        assert "avg 70 chars" in out
        assert "[70..70]" not in out

    def test_diverging_range_emits_suffix(self):
        m = TurnMetrics(transcript="hi", model="stub",
                        sentences_spoken=2,
                        mean_sentence_chars=70.0,
                        min_sentence_chars=10, max_sentence_chars=130)
        out = self._capture(m)
        assert "avg 70 chars [10..130]" in out

    def test_min_zero_skips_range(self):
        # Defensive: min_sentence_chars==0 is the unset signal.
        # max could still be set but the check requires both to be
        # consistent, so skip the suffix.
        m = TurnMetrics(transcript="hi", model="stub",
                        sentences_spoken=1,
                        mean_sentence_chars=70.0,
                        min_sentence_chars=0, max_sentence_chars=70)
        out = self._capture(m)
        assert "[" not in out.split("chars")[1].split("\n")[0]


# ---- Session aggregate ---------------------------------------------------


def _m(mean=0.0, lo=0, hi=0):
    return TurnMetrics(
        ttfs=0.5,
        mean_sentence_chars=mean,
        min_sentence_chars=lo, max_sentence_chars=hi,
    )


def _summary(metrics_list, **kwargs):
    out = io.StringIO()
    print_session_summary(
        metrics_list, {"model": "stub"}, file=out, **kwargs,
    )
    return _strip_ansi(out.getvalue())


class TestSessionSummary:
    def test_no_data_omits(self):
        plain = _summary([_m(), _m()])
        assert "Sentence range" not in plain

    def test_uniform_omits(self):
        plain = _summary([
            _m(mean=70, lo=70, hi=70),
            _m(mean=70, lo=70, hi=70),
        ])
        assert "Mean sentence:" in plain
        assert "Sentence range" not in plain

    def test_session_range_aggregates(self):
        # Across turns: turn 1 [10..50], turn 2 [60..200].
        # Session: shortest=10, longest=200.
        plain = _summary([
            _m(mean=30, lo=10, hi=50),
            _m(mean=130, lo=60, hi=200),
        ])
        assert "Sentence range:   [10..200] chars (session)" in plain

    def test_filters_zero_per_turn(self):
        # Turns with 0 (no submissions) excluded from min/max.
        plain = _summary([
            _m(mean=0, lo=0, hi=0),       # excluded
            _m(mean=80, lo=20, hi=140),
        ])
        assert "Sentence range:   [20..140] chars (session)" in plain


# ---- ChatLoop wiring -------------------------------------------------


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
    def test_bimodal_response_diverges(self):
        # Force a bimodal LLM response: one short sentence + one long.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine, transcribe = _stt_engine()
        # Short ("Yes.") + long (~80 chars).
        response = (
            "Yes. The quick brown fox jumped over the lazy dog "
            "which was quite a sight to behold."
        )
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=_yield_tokens(response),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_fast_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Min should be small (<10), max should be larger (>50). The
        # split happens on ". " so the 4-char "Yes." short sentence
        # lands first.
        assert result.metrics.min_sentence_chars > 0
        assert result.metrics.min_sentence_chars < 10
        assert result.metrics.max_sentence_chars > 50

    def test_uniform_response_min_equals_max(self):
        # Single-sentence response → min == max.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine, transcribe = _stt_engine()
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=_yield_tokens("OK ready."),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_fast_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # One submission → min == max.
        assert result.metrics.min_sentence_chars == result.metrics.max_sentence_chars
        assert result.metrics.min_sentence_chars > 0
