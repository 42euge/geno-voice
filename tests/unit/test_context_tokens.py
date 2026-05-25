"""Tests for iter-077 — conversation history grow rate.

Metric 2.23 from docs/perf-metrics-taxonomy.md.

    context_tokens = sum(len(m["content"].split()) for m in messages)

Whitespace-split estimator across the entire messages list, sampled
right before each LLM call. Tracks how much context is being sent
to the LLM each turn — late-session turns get progressively slower
as context grows, so creep here predicts an LLM TTFB regression
before llm_first_token measurably worsens.
"""

from __future__ import annotations

import io
import re
import sys
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


# ---- Default + per-turn print -------------------------------------


class TestDefault:
    def test_turnmetrics_default_zero(self):
        assert TurnMetrics().context_tokens == 0


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_zero_omits_suffix(self):
        m = TurnMetrics(transcript="hi", model="stub",
                        llm_total=0.5, context_tokens=0)
        out = self._capture(m)
        # Find the LLM total line; should NOT contain " ctx".
        llm_lines = [ln for ln in out.splitlines() if "LLM total:" in ln]
        assert len(llm_lines) == 1
        assert "ctx" not in llm_lines[0]

    def test_nonzero_appends_suffix(self):
        m = TurnMetrics(transcript="hi", model="stub",
                        llm_total=0.5, context_tokens=42)
        out = self._capture(m)
        llm_lines = [ln for ln in out.splitlines() if "LLM total:" in ln]
        assert "42 ctx" in llm_lines[0]

    def test_combined_with_tps(self):
        m = TurnMetrics(transcript="hi", model="stub",
                        llm_total=0.5, llm_tps=45, context_tokens=120)
        out = self._capture(m)
        llm_lines = [ln for ln in out.splitlines() if "LLM total:" in ln]
        assert "45 tps" in llm_lines[0]
        assert "120 ctx" in llm_lines[0]


# ---- Session aggregate --------------------------------------------


def _m(ctx=0):
    return TurnMetrics(ttfs=0.5, llm_total=0.5, context_tokens=ctx)


def _summary(metrics_list, **kwargs):
    out = io.StringIO()
    print_session_summary(metrics_list, {"model": "stub"}, file=out, **kwargs)
    return _strip_ansi(out.getvalue())


class TestSessionSummary:
    def test_no_data_omits(self):
        plain = _summary([_m(), _m()])
        assert "Context tokens" not in plain
        assert "Context growth" not in plain

    def test_two_turns_emits_median_max_no_growth(self):
        # 2 turns is below the growth-line threshold of 3.
        plain = _summary([_m(ctx=20), _m(ctx=40)])
        assert "Context tokens:   30 median, 40 max" in plain
        assert "Context growth" not in plain

    def test_three_turns_emits_growth(self):
        plain = _summary([_m(ctx=20), _m(ctx=30), _m(ctx=80)])
        # Median of [20, 30, 80] = 30. Max = 80. Growth = 60.
        assert "Context tokens:   30 median, 80 max" in plain
        assert "Context growth:   +60 tokens (turn 1 → turn 3)" in plain

    def test_negative_growth(self):
        # Hypothetical: trim aggressive enough that turn 1 had more
        # context than turn 3 (rare but possible if system prompt
        # was edited mid-session).
        plain = _summary([_m(ctx=100), _m(ctx=80), _m(ctx=50)])
        assert "Context growth:   -50 tokens (turn 1 → turn 3)" in plain

    def test_zeros_filtered(self):
        # Turn that errored before LLM (context_tokens=0) excluded.
        plain = _summary([_m(ctx=0), _m(ctx=20), _m(ctx=40)])
        # Median of [20, 40] = 30, max = 40.
        assert "Context tokens:   30 median, 40 max" in plain


# ---- ChatLoop arithmetic ------------------------------------------


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
        make_tone_burst(0.6, rate=RATE, amp=0.3),
        make_silence(1.5, rate=RATE),
    ))


class TestChatLoopArithmetic:
    def _build(self, *, mic, transcript, response="Done."):
        engine = SimpleNamespace(_last_text=None, model_repo="stub")
        return ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=lambda w: transcript if w else None,
            llm_stream_fn=_yield_tokens(response),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_fast_play,
        )

    def test_short_transcript_low_context(self):
        # System prompt + 2-word user message.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        loop = self._build(mic=mic, transcript="hi there")
        messages = [{"role": "system", "content": "You are a helpful bot."}]
        result = loop.run_one_turn(messages)
        assert result.metrics is not None
        # 5 (system: "You are a helpful bot.") + 2 (user) = 7.
        assert result.metrics.context_tokens == 7

    def test_long_transcript_higher_context(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        loop = self._build(
            mic=mic,
            transcript="this is a longer message with several words",
        )
        messages = [{"role": "system", "content": "Be concise."}]
        result = loop.run_one_turn(messages)
        assert result.metrics is not None
        # 2 (system) + 8 (user) = 10.
        assert result.metrics.context_tokens == 10

    def test_growing_messages_grow_context(self):
        # Pre-fill messages with prior turns; metric reflects total.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        loop = self._build(mic=mic, transcript="hello")
        messages = [
            {"role": "system", "content": "You are a bot."},   # 4
            {"role": "user", "content": "first question"},     # 2
            {"role": "assistant", "content": "first answer"},  # 2
            {"role": "user", "content": "second question"},    # 2
            {"role": "assistant", "content": "second answer"}, # 2
        ]
        # +1 (current "hello") = 13 total.
        result = loop.run_one_turn(messages)
        assert result.metrics is not None
        assert result.metrics.context_tokens == 13
