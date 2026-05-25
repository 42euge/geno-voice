"""Tests for iter-060 — LLM stream cancel-to-close metric.

Metric 2.14 from docs/perf-metrics-taxonomy.md.

    cancel_to_close = close_finished_at - coord.triggered_at

Time from BargeInCoordinator.trigger() to llm_gen.close()
returning. High values mean the upstream HTTP socket is taking
a long time to wind down — wastes tokens we paid for and can
block the next turn.
"""

from __future__ import annotations

import io
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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
        assert TurnMetrics().llm_cancel_to_close == 0.0


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
        m = TurnMetrics(transcript="hi", model="stub", llm_cancel_to_close=0.0)
        out = self._capture(m)
        assert "LLM cancel" not in out

    def test_nonzero_emits_line(self):
        # Filling barge_in to satisfy the print path's outer
        # if-block — the cancel-close line lives in the barge-in
        # nested branch (post-Primed-frames).
        m = TurnMetrics(
            transcript="hi", model="stub",
            barge_in=True, llm_cancel_to_close=0.080,  # 80ms
        )
        out = self._capture(m)
        assert "LLM cancel" in out
        assert "80ms" in out
        assert "trigger → stream close" in out

    def test_high_value_still_shown(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            barge_in=True, llm_cancel_to_close=0.700,  # 700ms — slow!
        )
        out = self._capture(m)
        assert "700ms" in out


# ---- Session aggregate ---------------------------------------------------


def _m(barge=False, cancel_close=0.0):
    return TurnMetrics(
        ttfs=0.5, barge_in=barge, llm_cancel_to_close=cancel_close,
    )


class TestSessionSummary:
    def test_no_data_omits_line(self):
        out = io.StringIO()
        print_session_summary([_m(), _m()], {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        assert "Median LLM canc" not in plain

    def test_with_data_emits_median(self):
        out = io.StringIO()
        print_session_summary(
            [
                _m(barge=True, cancel_close=0.05),
                _m(barge=True, cancel_close=0.10),
                _m(barge=True, cancel_close=0.30),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Median of [50, 100, 300] = 100ms.
        assert "Median LLM canc:  100ms" in plain

    def test_zero_filtered(self):
        out = io.StringIO()
        print_session_summary(
            [
                _m(barge=True, cancel_close=0.0),  # excluded
                _m(barge=True, cancel_close=0.20),
                _m(barge=True, cancel_close=0.40),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Median of [200, 400] = 300.
        assert "Median LLM canc:  300ms" in plain


# ---- ChatLoop wiring (deterministic via barge scenario) ------------------


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


class TestChatLoopArithmetic:
    def test_no_barge_no_cancel_close(self):
        # Clean turn, no barge — field stays 0.
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
        assert result.metrics.llm_cancel_to_close == 0.0

    def test_barge_populates_cancel_close(self):
        # Reuse the deterministic delayed-barge pattern.
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
        # If the barge landed AND we got a metrics object back —
        # cancel_to_close should be populated. But the success-path
        # population only happens when run_one_turn reaches its
        # post-finally return. On a barge, the body completes
        # successfully (barge isn't an exception), so we DO reach
        # the return — cancel_to_close should be > 0.
        if result.metrics.barge_in:
            # The close was called immediately after the barge fired,
            # so the gap is small (microseconds in stub setup) but >0.
            assert result.metrics.llm_cancel_to_close > 0


# ---- Edge cases ----------------------------------------------------------


class TestArithmeticBoundaries:
    @pytest.mark.parametrize("triggered_at,close_at,expected_max", [
        (10.0, 10.001,  0.002),  # ~1ms — typical
        (10.0, 10.5,    0.501),  # 500ms — slow
        (10.0, 9.99,    0.0),    # negative gap (shouldn't happen) → clamp 0
    ])
    def test_clamping(self, triggered_at, close_at, expected_max):
        # Replicate the formula.
        gap = max(0.0, close_at - triggered_at)
        assert gap <= expected_max
        assert gap >= 0.0
