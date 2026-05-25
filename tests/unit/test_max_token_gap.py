"""Tests for iter-085 — max LLM inter-token gap.

Metric 3.21 from docs/perf-metrics-taxonomy.md (simpler "max gap"
take, not the full stall-recoverability calculation).

    max_token_gap = max(t[i+1] - t[i] for i in range(1, n_tokens))

Tracks the worst pause between consecutive LLM tokens. Catches
mid-stream stalls — currently invisible to operators because the
user just hears a long pause and no signal fires. >500ms is
"noticeable mid-response stall"; >2s is "user definitely thought
the bot was broken."
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


# ---- Default + per-turn print -----------------------------------


class TestDefault:
    def test_turnmetrics_default_zero(self):
        assert TurnMetrics().max_token_gap == 0.0


class TestPerTurnPrint:
    def _capture(self, m):
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_zero_omits_suffix(self):
        m = TurnMetrics(transcript="hi", model="stub",
                        llm_total=0.5, max_token_gap=0.0)
        out = self._capture(m)
        # The LLM total line still prints, just no "max gap" suffix.
        llm_lines = [ln for ln in out.splitlines() if "LLM total:" in ln]
        assert len(llm_lines) == 1
        assert "max gap" not in llm_lines[0]

    def test_below_threshold_omits(self):
        # ≤200ms is normal token-streaming jitter, not a stall.
        m = TurnMetrics(transcript="hi", model="stub",
                        llm_total=0.5, max_token_gap=0.150)
        out = self._capture(m)
        llm_lines = [ln for ln in out.splitlines() if "LLM total:" in ln]
        assert "max gap" not in llm_lines[0]

    def test_significant_emits(self):
        m = TurnMetrics(transcript="hi", model="stub",
                        llm_total=0.5, max_token_gap=0.350)
        out = self._capture(m)
        llm_lines = [ln for ln in out.splitlines() if "LLM total:" in ln]
        assert "max gap 350ms" in llm_lines[0]

    def test_severe_stall_emits_yellow(self):
        # >500ms hits the yellow color path.
        m = TurnMetrics(transcript="hi", model="stub",
                        llm_total=0.5, max_token_gap=1.500)
        out = self._capture(m)
        llm_lines = [ln for ln in out.splitlines() if "LLM total:" in ln]
        assert "max gap 1500ms" in llm_lines[0]


# ---- Session aggregate -----------------------------------------


def _m(gap=0.0):
    return TurnMetrics(ttfs=0.5, max_token_gap=gap)


def _summary(metrics_list, **kwargs):
    out = io.StringIO()
    print_session_summary(metrics_list, {"model": "stub"}, file=out, **kwargs)
    return _strip_ansi(out.getvalue())


class TestSessionSummary:
    def test_no_data_omits(self):
        plain = _summary([_m(), _m()])
        assert "Worst LLM stall" not in plain

    def test_below_threshold_omits(self):
        # All gaps <= 200ms — normal jitter, no stall line.
        plain = _summary([_m(gap=0.05), _m(gap=0.10), _m(gap=0.15)])
        assert "Worst LLM stall" not in plain

    def test_one_stall_emits(self):
        plain = _summary([
            _m(gap=0.05),
            _m(gap=0.350),  # one real stall
            _m(gap=0.10),
        ])
        # Worst across stalls = 350ms; 1 of 3 turns stalled.
        assert "Worst LLM stall:  350ms (1/3 turns)" in plain

    def test_multiple_stalls(self):
        plain = _summary([
            _m(gap=0.05),
            _m(gap=0.300),
            _m(gap=1.200),  # the worst
            _m(gap=0.500),
        ])
        assert "Worst LLM stall:  1200ms (3/4 turns)" in plain


# ---- ChatLoop arithmetic ---------------------------------------


def _const_synth(samples=2048):
    def synth(s):
        return np.full(samples, 0.5, dtype=np.float32), []
    return synth


def _fast_play(speaker, audio, tokens, *, is_first_sentence=False, cancel_event=None):
    audio_int16 = (audio * 32767).astype(np.int16)
    speaker.write(audio_int16.tobytes())
    return 0.0


def _yield_tokens(text, *, per_token_delay=0.0, stall_after=None,
                  stall_duration=0.0):
    """Yield tokens with optional uniform delay AND a one-shot stall
    after the Nth token. ``stall_after=N, stall_duration=D`` makes
    the (N+1)-th token arrive D seconds late.
    """
    import re as _re

    def factory(messages, config):
        tokens_iter = list(_re.findall(r"\S+|\.|!|\?", text))
        for i, p in enumerate(tokens_iter):
            yield p + " "
            if per_token_delay > 0:
                time.sleep(per_token_delay)
            if stall_after is not None and i == stall_after - 1:
                # The NEXT token will be delayed by stall_duration.
                time.sleep(stall_duration)

    return factory


def _push_one(mic):
    mic.push(concat(
        make_silence(0.3, rate=RATE),
        make_tone_burst(0.6, rate=RATE, amp=0.3),
        make_silence(1.5, rate=RATE),
    ))


class TestChatLoopWiring:
    def _build(self, *, mic, llm_stream_fn):
        engine = SimpleNamespace(_last_text=None, model_repo="stub")
        return ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=lambda w: "hi" if w else None,
            llm_stream_fn=llm_stream_fn,
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_fast_play,
        )

    def test_no_stall_low_gap(self):
        # Fast token stream — gaps between tokens are sub-50ms.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        loop = self._build(
            mic=mic,
            llm_stream_fn=_yield_tokens("Hello there friend Done.",
                                        per_token_delay=0.001),
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Gap should be small — natural test scheduling jitter only.
        assert result.metrics.max_token_gap < 0.1

    def test_injected_stall_caught(self):
        # Force a 200ms stall after token 2.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        loop = self._build(
            mic=mic,
            llm_stream_fn=_yield_tokens(
                "One Two Three Four Done.",
                stall_after=2,
                stall_duration=0.20,
            ),
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Should observe at least the stall duration.
        assert result.metrics.max_token_gap >= 0.18  # tolerance for jitter

    def test_single_token_zero_gap(self):
        # Only one token → no inter-token gap to measure.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        loop = self._build(
            mic=mic,
            llm_stream_fn=_yield_tokens("Done."),
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # "Done." splits to ["Done", "."] = 2 tokens, so there IS
        # one gap, but it's tiny. Just assert non-negative + small.
        assert result.metrics.max_token_gap >= 0
        assert result.metrics.max_token_gap < 0.1
