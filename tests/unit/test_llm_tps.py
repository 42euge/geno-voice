"""Tests for iter-052 — LLM tokens-per-second metric.

Metric 1.9 from docs/perf-metrics-taxonomy.md.

Definition: tokens/sec measured AFTER first token.
    llm_tps = (token_count - 1) / (llm_stream_done_at - first_token_at)

Excluding the first-token wait avoids biasing TPS down on slow-
warmup endpoints.
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
        assert TurnMetrics().llm_tps == 0.0


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_zero_omits_tps(self):
        m = TurnMetrics(transcript="hi", model="stub", llm_total=0.5, llm_tps=0)
        out = self._capture(m)
        assert "tps" not in out

    def test_nonzero_emits_tps(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            llm_total=0.5, llm_tps=42.0,
        )
        out = self._capture(m)
        assert "42 tps" in out

    def test_high_tps_rounded(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            llm_total=0.5, llm_tps=85.7,
        )
        out = self._capture(m)
        assert "86 tps" in out  # :.0f rounds


# ---- Session aggregate ---------------------------------------------------


def _m(tps):
    return TurnMetrics(ttfs=0.5, llm_tps=tps)


class TestSessionSummary:
    def test_no_data_omits_line(self):
        out = io.StringIO()
        print_session_summary([_m(0.0), _m(0.0)], {"model": "stub"}, file=out)
        assert "Median LLM TPS" not in _strip_ansi(out.getvalue())

    def test_with_data_emits_median(self):
        out = io.StringIO()
        print_session_summary(
            [_m(40.0), _m(50.0), _m(60.0)],
            {"model": "stub"}, file=out,
        )
        assert "Median LLM TPS:   50" in _strip_ansi(out.getvalue())

    def test_zero_filtered(self):
        # 0 turns are filtered from the median calculation.
        out = io.StringIO()
        print_session_summary(
            [_m(0.0), _m(40.0), _m(60.0)],
            {"model": "stub"}, file=out,
        )
        assert "Median LLM TPS:   50" in _strip_ansi(out.getvalue())


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
    def test_tps_computed_when_multi_token(self):
        # LLM yields 6 tokens with 20ms each. After first token the
        # remaining 5 take 100ms → 5 / 0.1 = 50 tps.
        def llm(messages, config):
            for tok in ["alpha ", "beta ", "gamma ", "delta. ", "Done", "."]:
                time.sleep(0.02)
                yield tok

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
        assert result.metrics.llm_tps > 0
        # With 5 tokens spread over ~100ms+, TPS lands ~30-60. Loose bound.
        assert 20 < result.metrics.llm_tps < 100

    def test_zero_when_single_token(self):
        # One token only → can't measure TPS (need ≥2).
        def llm(messages, config):
            yield "Hi."

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
        assert result.metrics.llm_tps == 0.0

    def test_zero_when_no_tokens(self):
        # Empty stream — no tokens at all.
        def llm(messages, config):
            return
            yield

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
        assert result.metrics.llm_tps == 0.0
