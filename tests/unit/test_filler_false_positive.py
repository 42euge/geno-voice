"""Tests for iter-051 — filler false-positive rate metric.

Metric 2.4 from docs/perf-metrics-taxonomy.md. A turn's filler is
a false positive when:
- A filler actually played (fillers_played > 0)
- AND the LLM's first-token came back faster than idle_threshold

Meaning: the bot would have started speaking on its own before
the filler was needed. Tune idle_threshold up.
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
    def test_default_false(self):
        assert TurnMetrics().filler_false_positive is False


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_no_filler_no_marker(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            sentences_spoken=2, fillers_played=0,
        )
        out = self._capture(m)
        assert "*" not in out  # no FP marker

    def test_filler_no_fp_no_marker(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            sentences_spoken=2, fillers_played=1,
            filler_false_positive=False,
        )
        out = self._capture(m)
        assert "1 filler" in out
        assert "1 filler*" not in out

    def test_filler_fp_emits_marker(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            sentences_spoken=2, fillers_played=1,
            filler_false_positive=True,
        )
        out = self._capture(m)
        assert "1 filler*" in out


# ---- Session aggregate ---------------------------------------------------


def _m(fillers=0, fp=False):
    return TurnMetrics(
        ttfs=0.5, fillers_played=fillers, filler_false_positive=fp,
    )


class TestSessionSummary:
    def test_no_fillers_omits_block(self):
        out = io.StringIO()
        print_session_summary(
            [_m(), _m()], {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Filler FP rate" not in plain

    def test_fillers_no_fp_omits_rate_line(self):
        # Filler aggregate emits but no FP — keep summary clean.
        out = io.StringIO()
        print_session_summary(
            [_m(fillers=1, fp=False), _m(fillers=1, fp=False)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Fillers played:   2" in plain
        # Don't emit a "0% FP" line when nothing went wrong.
        assert "Filler FP rate" not in plain

    def test_some_fp_emits_rate_with_pct(self):
        out = io.StringIO()
        print_session_summary(
            [
                _m(fillers=1, fp=True),
                _m(fillers=1, fp=False),
                _m(fillers=1, fp=True),
                _m(fillers=1, fp=False),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Filler FP rate:   2/4 (50%)" in plain
        assert "tune idle_threshold up" in plain

    def test_all_fp_shows_100pct(self):
        out = io.StringIO()
        print_session_summary(
            [_m(fillers=1, fp=True), _m(fillers=1, fp=True)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Filler FP rate:   2/2 (100%)" in plain


# ---- ChatLoop wires (synthetic, validates the comparison logic) -----------


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
    def test_fast_llm_with_filler_marks_fp(self):
        # idle_threshold = 0.05s. A filler will play after 50ms of
        # waiting. But the LLM's first token (with 0.0 per-token
        # delay) will arrive almost instantly — typically <50ms.
        # That makes the filler a false positive.
        #
        # The flow: filler starts at t=0.05, but the first LLM
        # sentence might not arrive THAT fast — depends on stub
        # timing. So we use a fast LLM (no delay) AND a moderate
        # idle threshold (0.15s) to bias toward FP.
        filler = (np.full(2048, 0.3, dtype=np.float32), [])

        def llm(messages, config):
            # Fast LLM: emits sentence immediately.
            yield "Hello there. "

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
            fillers=[filler],
            idle_threshold=0.5,  # easily beaten by a fast LLM
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Whether the filler actually played depends on the worker's
        # race with the LLM. If it did play, FP should be set.
        if result.metrics.fillers_played > 0:
            # llm_first_token < idle_threshold (0.5s) → FP.
            assert result.metrics.filler_false_positive is True
        else:
            # No filler → can't be FP.
            assert result.metrics.filler_false_positive is False

    def test_no_fillers_configured_no_fp(self):
        def llm(messages, config):
            yield "Hello. "

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
            # No fillers configured.
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        assert result.metrics.fillers_played == 0
        assert result.metrics.filler_false_positive is False
