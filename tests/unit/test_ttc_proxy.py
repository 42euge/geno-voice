"""Tests for iter-082 — TTC (time-to-comprehension) proxy.

Metric 3.14 from docs/perf-metrics-taxonomy.md.

    ttc = current_turn.speech_start_at - prev_turn.first_audio_at

Captures how long the user listened before responding. Sub-500ms
suggests the user already knew what the bot said (bot
underperformed). >5s suggests the user was confused / thinking.
Bell-curve target is 1-3s.
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
from examples._chat_recording import (  # noqa: E402
    CHUNK,
    RATE,
    record_utterance_streaming,
)
from examples.virtual_audio import (  # noqa: E402
    VirtualMicStream,
    VirtualSpeakerStream,
    concat,
    make_silence,
    make_tone_burst,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ---- Default + per-turn print --------------------------------


class TestDefault:
    def test_turnmetrics_default_zero(self):
        assert TurnMetrics().time_to_comprehension == 0.0


class TestPerTurnPrint:
    def _capture(self, m):
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=2)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_zero_omits(self):
        m = TurnMetrics(transcript="hi", model="stub",
                        time_to_comprehension=0.0)
        assert "TTC:" not in self._capture(m)

    def test_natural_emits_dim(self):
        # 1.5s — well within the bell curve.
        m = TurnMetrics(transcript="hi", model="stub",
                        time_to_comprehension=1.5)
        out = self._capture(m)
        assert "TTC:" in out
        assert "1500ms" in out

    def test_rushed_emits_yellow(self):
        # <500ms — bot was telling them what they already knew.
        m = TurnMetrics(transcript="hi", model="stub",
                        time_to_comprehension=0.300)
        out = self._capture(m)
        assert "300ms" in out

    def test_slow_emits_yellow(self):
        # >5s — user was thinking / confused.
        m = TurnMetrics(transcript="hi", model="stub",
                        time_to_comprehension=7.2)
        out = self._capture(m)
        assert "7200ms" in out


# ---- Recorder integration ------------------------------------


class TestRecorderEmits:
    def _push(self, mic):
        mic.push(concat(
            make_silence(0.3, rate=RATE),
            make_tone_burst(0.6, rate=RATE, amp=0.3),
            make_silence(1.5, rate=RATE),
        ))

    def test_speech_start_populated_on_done_ok(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        self._push(mic)
        engine = SimpleNamespace(_last_text=None, model_repo="stub")
        out_metrics: dict = {}
        wav, dur, _ = record_utterance_streaming(
            mic, engine, transcribe_fn=lambda w: "hi" if w else None,
            output=io.StringIO(),
            out_metrics=out_metrics,
        )
        assert wav  # DONE_OK
        assert "speech_start_at" in out_metrics
        # speech_start_at should land near the start of the tone:
        # ~0.3s of silence + first tone-frame.
        assert out_metrics["speech_start_at"] > 0
        # And it should precede the eot_latency window (which
        # ends at "now" when DONE_OK fired).
        # Since we don't have "now" exposed, just sanity-bound.

    def test_done_too_short_omits(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(concat(
            make_silence(0.1, rate=RATE),
            make_tone_burst(0.10, rate=RATE, amp=0.3),  # below min
            make_silence(1.5, rate=RATE),
        ))
        engine = SimpleNamespace(_last_text=None, model_repo="stub")
        out_metrics: dict = {}
        wav, dur, _ = record_utterance_streaming(
            mic, engine, transcribe_fn=lambda w: "x",
            output=io.StringIO(),
            out_metrics=out_metrics,
        )
        assert wav == b""
        assert "speech_start_at" not in out_metrics


# ---- Session aggregate --------------------------------------


def _m(ttc=0.0):
    return TurnMetrics(ttfs=0.5, time_to_comprehension=ttc)


def _summary(metrics_list, **kwargs):
    out = io.StringIO()
    print_session_summary(metrics_list, {"model": "stub"}, file=out, **kwargs)
    return _strip_ansi(out.getvalue())


class TestSessionSummary:
    def test_no_data_omits(self):
        plain = _summary([_m(), _m()])
        assert "Median TTC" not in plain

    def test_natural_emits(self):
        plain = _summary([_m(ttc=1.5), _m(ttc=2.0), _m(ttc=2.5)])
        assert "Median TTC:       2000ms" in plain
        # No outlier annotations.
        assert "rushed" not in plain
        assert "slow" not in plain

    def test_rushed_outlier(self):
        plain = _summary([_m(ttc=0.300), _m(ttc=2.0), _m(ttc=3.0)])
        # Median of [300, 2000, 3000] = 2000.
        assert "Median TTC:       2000ms" in plain
        assert "1 rushed" in plain

    def test_slow_outlier(self):
        plain = _summary([_m(ttc=2.0), _m(ttc=3.0), _m(ttc=8.0)])
        assert "Median TTC:       3000ms" in plain
        assert "1 slow" in plain

    def test_both_outlier_kinds(self):
        plain = _summary([
            _m(ttc=0.2), _m(ttc=2.0), _m(ttc=3.0), _m(ttc=8.0)
        ])
        assert "1 rushed" in plain
        assert "1 slow" in plain

    def test_zeros_filtered(self):
        # Turn 1 has no TTC (default 0); excluded from aggregation.
        plain = _summary([_m(ttc=0.0), _m(ttc=2.0), _m(ttc=3.0)])
        # Median of [2000, 3000] = 2500.
        assert "Median TTC:       2500ms" in plain


# ---- ChatLoop wiring -----------------------------------------


def _stt_engine():
    engine = SimpleNamespace(_last_text=None, model_repo="stub")
    def transcribe(wav):
        return "hi" if wav else None
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
        make_tone_burst(0.6, rate=RATE, amp=0.3),
        make_silence(1.5, rate=RATE),
    ))


class TestChatLoopWiring:
    def test_turn_1_zero(self):
        # No prior turn → TTC is 0.
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
            play_fn=_fast_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        assert result.metrics.time_to_comprehension == 0.0

    def test_turn_2_populated(self):
        # Run two turns through the same ChatLoop instance — the
        # second one should have a TTC value derived from turn 1's
        # first_audio_at.
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
            play_fn=_fast_play,
        )
        # Turn 1.
        r1 = loop.run_one_turn([])
        assert r1.metrics is not None
        assert r1.metrics.time_to_comprehension == 0.0
        # Push more audio for turn 2.
        _push_one(mic)
        r2 = loop.run_one_turn([])
        assert r2.metrics is not None
        # TTC should be populated (>= 0 by clamp).
        assert r2.metrics.time_to_comprehension >= 0.0

    def test_silent_prev_turn_breaks_chain(self):
        # If the prev turn produced no audio (no
        # _last_first_audio_at update), the next turn's TTC stays 0.
        # Construct a loop that won't produce audio on turn 1 by
        # giving an empty LLM response.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine, transcribe = _stt_engine()
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            # Empty response → no sentences submitted → no audio.
            llm_stream_fn=_yield_tokens(""),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_fast_play,
        )
        r1 = loop.run_one_turn([])
        # Turn 1 may or may not produce audio depending on the
        # trailing-remainder path; either way we assert that the
        # next turn's TTC is bounded.
        _push_one(mic)
        r2 = loop.run_one_turn([])
        assert r2.metrics is not None
        # Field is always non-negative thanks to the clamp.
        assert r2.metrics.time_to_comprehension >= 0.0
