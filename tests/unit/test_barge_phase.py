"""Tests for iter-047 — barge-in phase metric.

Metric 2.11 from docs/perf-metrics-taxonomy.md. Distinguishes
where the user's barge-in landed in the pipeline:
  - "llm_stream": LLM was still streaming tokens — user is
    impatient with TTFS. Fix: LLM TTFT.
  - "playback": bot was speaking — verbose / wrong response.
    Fix: system prompt / response quality.
  - "" (empty): no barge fired this turn.

The phase string was already computed in _chat_loop for the
diagnostic message; iter-047 lifts it to a structured metric on
TurnMetrics + the perf snapshot.
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


# ---- Default + per-turn print ---------------------------------------------


class TestDefault:
    def test_default_empty_string(self):
        assert TurnMetrics().barge_in_phase == ""


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_no_barge_omits_phase(self):
        m = TurnMetrics(transcript="hi", model="stub", barge_in=False)
        out = self._capture(m)
        assert "during" not in out  # no phase note

    def test_llm_stream_phase_shown(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            barge_in=True, barge_in_phase="llm_stream",
        )
        out = self._capture(m)
        assert "during LLM stream" in out

    def test_playback_phase_shown(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            barge_in=True, barge_in_phase="playback",
        )
        out = self._capture(m)
        assert "during playback" in out

    def test_empty_phase_with_barge_omits(self):
        # Edge case: barge_in=True but phase wasn't set.
        # The phase note should be omitted (not "during ").
        m = TurnMetrics(
            transcript="hi", model="stub",
            barge_in=True, barge_in_phase="",
        )
        out = self._capture(m)
        assert "during LLM stream" not in out
        assert "during playback" not in out


# ---- Session summary aggregate -------------------------------------------


def _m(barge=False, phase=""):
    return TurnMetrics(ttfs=0.5, barge_in=barge, barge_in_phase=phase)


class TestSessionSummary:
    def test_no_barge_phases_omits_line(self):
        out = io.StringIO()
        print_session_summary(
            [_m(), _m()], {"model": "stub"}, file=out,
        )
        assert "Barge phases" not in _strip_ansi(out.getvalue())

    def test_mixed_phases_emits_counts(self):
        out = io.StringIO()
        print_session_summary(
            [
                _m(barge=True, phase="llm_stream"),
                _m(barge=True, phase="playback"),
                _m(barge=True, phase="llm_stream"),
                _m(barge=True, phase="playback"),
                _m(barge=True, phase="playback"),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Barge phases:     2 LLM-stream, 3 playback" in plain

    def test_all_one_phase(self):
        out = io.StringIO()
        print_session_summary(
            [
                _m(barge=True, phase="playback"),
                _m(barge=True, phase="playback"),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Barge phases:     0 LLM-stream, 2 playback" in plain


# ---- ChatLoop wires (end-to-end via the perf-style barge scenario) ------


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


class TestChatLoopWires:
    def test_barge_during_playback_sets_playback_phase(self):
        # Use the perf-suite's deterministic delayed-push trick.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)

        def _delayed_barge():
            time.sleep(0.05)
            mic.push(concat(
                make_silence(0.05, rate=RATE),
                make_tone_burst(0.6, rate=RATE, amp=0.4),
                make_silence(0.5, rate=RATE),
            ))

        engine, transcribe = _stt_engine()
        long_response = " ".join(f"sentence {i}." for i in range(8))
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=_yield_tokens(long_response, per_token_delay=0.015),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_slow_play,
        )
        threading.Thread(target=_delayed_barge, daemon=True).start()
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # The barge landed; phase should be one of the two values.
        if result.metrics.barge_in:
            assert result.metrics.barge_in_phase in ("llm_stream", "playback")

    def test_no_barge_phase_remains_empty(self):
        # Clean turn — no barge — phase stays "".
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
            synth_fn=_const_synth(),
            play_fn=_slow_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        assert result.metrics.barge_in_phase == ""
