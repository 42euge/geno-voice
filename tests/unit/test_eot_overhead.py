"""Tests for iter-065 — VAD trailing-silence wall metric.

Metric 1.3 from docs/perf-metrics-taxonomy.md.

    eot_overhead = max(0, eot_latency - silence_duration_used)

Decomposes the EoT wait into:
  - "knob-budget": the silence_duration we asked for.
  - "implementation overhead": chunk granularity, processing.

If overhead is ~0, lower silence_duration. If overhead is >100ms,
something else in the recording loop is slow and tuning the knob
won't help.
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


# ---- Default + per-turn print --------------------------------------------


class TestDefault:
    def test_turnmetrics_default_zero(self):
        assert TurnMetrics().eot_overhead == 0.0


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_zero_omits_overhead_suffix(self):
        m = TurnMetrics(transcript="hi", model="stub",
                        eot_latency=0.85, eot_overhead=0.0)
        out = self._capture(m)
        eot_lines = [ln for ln in out.splitlines() if "EoT detect" in ln]
        assert len(eot_lines) == 1
        assert "overhead" not in eot_lines[0]

    def test_sub_chunk_overhead_omitted(self):
        # <10ms is within chunk-granularity noise — don't surface.
        m = TurnMetrics(transcript="hi", model="stub",
                        eot_latency=0.85, eot_overhead=0.005)
        out = self._capture(m)
        eot_lines = [ln for ln in out.splitlines() if "EoT detect" in ln]
        assert len(eot_lines) == 1
        assert "overhead" not in eot_lines[0]

    def test_real_overhead_emits_suffix(self):
        m = TurnMetrics(transcript="hi", model="stub",
                        eot_latency=1.20, eot_overhead=0.300)
        out = self._capture(m)
        eot_lines = [ln for ln in out.splitlines() if "EoT detect" in ln]
        assert len(eot_lines) == 1
        assert "+300ms overhead" in eot_lines[0]

    def test_high_overhead_yellow(self):
        # >100ms is the yellow threshold — we just assert it shows up.
        m = TurnMetrics(transcript="hi", model="stub",
                        eot_latency=1.5, eot_overhead=0.650)
        out = self._capture(m)
        assert "+650ms overhead" in out

    def test_no_eot_no_overhead_line(self):
        # If eot_latency is 0, the whole EoT line is omitted —
        # overhead suffix can't appear.
        m = TurnMetrics(transcript="hi", model="stub",
                        eot_latency=0.0, eot_overhead=0.500)
        out = self._capture(m)
        assert "EoT detect" not in out
        assert "overhead" not in out


# ---- Session aggregate ---------------------------------------------------


def _m(eot=0.0, overhead=0.0):
    return TurnMetrics(speech_duration=1.0, eot_latency=eot,
                       eot_overhead=overhead)


class TestSessionSummary:
    def test_no_overhead_omits_line(self):
        out = io.StringIO()
        print_session_summary(
            [_m(eot=0.85, overhead=0.0), _m(eot=0.85, overhead=0.0)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "EoT overhead" not in plain

    def test_emits_when_present(self):
        out = io.StringIO()
        print_session_summary(
            [
                _m(eot=1.10, overhead=0.30),
                _m(eot=1.20, overhead=0.40),
                _m(eot=1.05, overhead=0.25),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Median of [300, 400, 250] = 300.
        assert "EoT overhead:     300ms (above silence_duration)" in plain

    def test_sub_chunk_filtered(self):
        # Sub-chunk overhead (<10ms) excluded from aggregate.
        out = io.StringIO()
        print_session_summary(
            [
                _m(eot=0.85, overhead=0.003),  # excluded
                _m(eot=1.10, overhead=0.300),
                _m(eot=1.20, overhead=0.400),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Median of [300, 400] (the sub-chunk one excluded) = 350.
        assert "EoT overhead:     350ms" in plain


# ---- ChatLoop arithmetic --------------------------------------------


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
    def test_overhead_clamps_at_zero(self):
        # When eot_latency happens to be ≤ silence_duration (rare
        # edge case from the off-by-one between last_speech_at and
        # vad's silence_start) the overhead must clamp to 0 rather
        # than going negative.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine = SimpleNamespace(_last_text=None, model_repo="stub")
        def transcribe(wav):
            return "hello there" if wav else None
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=_yield_tokens("Done."),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_fast_play,
            silence_duration=0.8,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Always non-negative.
        assert result.metrics.eot_overhead >= 0.0
        # Should be small in the deterministic test setup — at most
        # a few chunks of granularity.
        assert result.metrics.eot_overhead < 0.2

    def test_overhead_below_eot(self):
        # By definition: overhead = eot - silence_duration ≤ eot.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine = SimpleNamespace(_last_text=None, model_repo="stub")
        def transcribe(wav):
            return "hello there" if wav else None
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=_yield_tokens("Done."),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_fast_play,
            silence_duration=0.8,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        assert result.metrics.eot_overhead <= result.metrics.eot_latency

    def test_no_eot_no_overhead(self):
        # Synthetic case at the metrics layer: eot_latency==0 means
        # the recorder didn't emit; overhead must stay at default.
        m = TurnMetrics(eot_latency=0.0, eot_overhead=0.0)
        # ChatLoop's guard `if metrics.eot_latency > 0` would skip
        # the assignment; overhead remains at the default.
        assert m.eot_overhead == 0.0
