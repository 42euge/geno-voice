"""Tests for iter-076 — TTFS attribution breakdown.

Metric 2.22 from docs/perf-metrics-taxonomy.md.

    synth_dispatch_seconds = max(0, ttfs - stt_time - llm_first_sentence)

Decomposes TTFS into three accounting buckets that sum to 100%:
  - STT: speech-end → STT done.
  - LLM: STT done → first complete sentence reaches the worker.
  - synth+dispatch: first sentence at worker → first audio played.
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


# ---- Default + per-turn print --------------------------------------


class TestDefault:
    def test_turnmetrics_default_zero(self):
        assert TurnMetrics().synth_dispatch_seconds == 0.0


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_zero_omits(self):
        # All zero — no breakdown to display.
        m = TurnMetrics(transcript="hi", model="stub")
        assert "Attribution" not in self._capture(m)

    def test_partial_data_omits(self):
        # Missing one of the three legs (e.g. no LLM time
        # measured) → omit the breakdown.
        m = TurnMetrics(transcript="hi", model="stub",
                        ttfs=0.5, stt_time=0.1,
                        llm_first_sentence=0.0,
                        synth_dispatch_seconds=0.4)
        assert "Attribution" not in self._capture(m)

    def test_full_breakdown_emits(self):
        # All three legs populated → emit breakdown.
        m = TurnMetrics(transcript="hi", model="stub",
                        ttfs=1.0, stt_time=0.2,
                        llm_first_sentence=0.5,
                        synth_dispatch_seconds=0.3)
        out = self._capture(m)
        assert "Attribution:" in out
        # 200/1000 = 20%, 500/1000 = 50%, 300/1000 = 30%.
        assert "STT 20%" in out
        assert "LLM 50%" in out
        assert "synth 30%" in out


# ---- ChatLoop arithmetic -------------------------------------------


def _const_synth(samples=2048, delay=0.0):
    def synth(s):
        if delay > 0:
            import time as _t
            _t.sleep(delay)
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
    def _build(self, *, mic, response="Done.", synth_delay=0.0):
        engine = SimpleNamespace(_last_text=None, model_repo="stub")
        return ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=lambda w: "hi" if w else None,
            llm_stream_fn=_yield_tokens(response),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(delay=synth_delay),
            play_fn=_fast_play,
        )

    def test_residual_non_negative(self):
        # Even on micro-jitter clock skew, the field clamps at 0.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        loop = self._build(mic=mic)
        result = loop.run_one_turn([])
        assert result.metrics is not None
        assert result.metrics.synth_dispatch_seconds >= 0.0

    def test_residual_below_ttfs(self):
        # Residual = ttfs - stt - llm_first_sentence ≤ ttfs.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        loop = self._build(mic=mic)
        result = loop.run_one_turn([])
        m = result.metrics
        assert m is not None
        assert m.synth_dispatch_seconds <= m.ttfs + 1e-9

    def test_three_legs_sum_to_ttfs(self):
        # Within tolerance, stt + llm + synth_dispatch ≈ ttfs.
        # Tolerance for clock skew across the recorder/loop clocks.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        loop = self._build(mic=mic, synth_delay=0.02)
        result = loop.run_one_turn([])
        m = result.metrics
        assert m is not None
        if m.ttfs > 0 and m.llm_first_sentence > 0:
            sum_legs = (
                m.stt_time + m.llm_first_sentence + m.synth_dispatch_seconds
            )
            # max(0, ...) means sum can be ≤ ttfs but not > ttfs.
            assert sum_legs <= m.ttfs + 1e-6
            # And generally close to ttfs.
            assert sum_legs >= m.ttfs - 0.05  # 50ms tolerance


# ---- Session aggregate ---------------------------------------------


def _m(ttfs=1.0, stt=0.2, llm=0.5, synth=0.3):
    return TurnMetrics(
        ttfs=ttfs, stt_time=stt,
        llm_first_sentence=llm,
        synth_dispatch_seconds=synth,
    )


def _summary(metrics_list, **kwargs):
    out = io.StringIO()
    print_session_summary(metrics_list, {"model": "stub"}, file=out, **kwargs)
    return _strip_ansi(out.getvalue())


class TestSessionSummary:
    def test_no_data_omits(self):
        plain = _summary([])
        assert "TTFS breakdown" not in plain

    def test_no_attribution_data_omits(self):
        # ttfs but missing one leg → not in attribution_turns.
        plain = _summary([_m(ttfs=0.5, stt=0.1, llm=0.0, synth=0.4)])
        assert "TTFS breakdown" not in plain

    def test_uniform_breakdown(self):
        plain = _summary([_m()])
        # 20 / 50 / 30.
        assert "TTFS breakdown:   STT 20% + LLM 50% + synth 30%" in plain

    def test_filters_zero_legs(self):
        # Mix of complete and incomplete turns — only the complete
        # ones contribute to the median.
        plain = _summary([
            _m(),                          # 20/50/30
            _m(ttfs=0.5, stt=0.1, llm=0.0, synth=0.4),  # excluded
            _m(ttfs=2.0, stt=0.4, llm=1.0, synth=0.6),  # 20/50/30
        ])
        # Median across the two complete turns: still 20/50/30.
        assert "STT 20% + LLM 50% + synth 30%" in plain
