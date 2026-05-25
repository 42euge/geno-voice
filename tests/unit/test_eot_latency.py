"""Tests for iter-063 — EoT detection latency metric.

Metric 1.2 from docs/perf-metrics-taxonomy.md.

    eot_latency = clock_at_DONE_OK - clock_at_last_in_speech_frame

Time from the user's last in-speech frame to the VAD declaring
DONE_OK. Lower bound is roughly ``silence_duration`` (the VAD has
to wait that long); the gap above that is implementation overhead.
Critical UX number — "the agent feels slow" complaints map directly
to this.
"""

from __future__ import annotations

import io
import re
import sys
import wave
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


# ---- Default + per-turn print --------------------------------------------


class TestDefault:
    def test_turnmetrics_default_zero(self):
        assert TurnMetrics().eot_latency == 0.0


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
        m = TurnMetrics(transcript="hi", model="stub", eot_latency=0.0)
        assert "EoT detect" not in self._capture(m)

    def test_normal_value_emits_dim(self):
        # 850ms is typical (silence_duration default = 800ms + chunk
        # processing).
        m = TurnMetrics(transcript="hi", model="stub", eot_latency=0.850)
        out = self._capture(m)
        assert "EoT detect" in out
        assert "850ms" in out
        assert "silence wait" in out

    def test_high_value_yellow(self):
        m = TurnMetrics(transcript="hi", model="stub", eot_latency=1.500)
        out = self._capture(m)
        assert "EoT detect" in out
        assert "1500ms" in out


# ---- Session aggregate ---------------------------------------------------


def _m(eot=0.0):
    # speech_duration > 0 to emit the standard line; eot is what we care.
    return TurnMetrics(speech_duration=1.0, eot_latency=eot)


class TestSessionSummary:
    def test_no_data_omits_lines(self):
        out = io.StringIO()
        print_session_summary([_m(), _m()], {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        assert "Median EoT" not in plain
        assert "Worst EoT" not in plain

    def test_emit_median_only_when_uniform(self):
        # All same value → no spread → no "Worst EoT" line.
        out = io.StringIO()
        print_session_summary(
            [_m(eot=0.85), _m(eot=0.85)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Median EoT:       850ms" in plain
        assert "Worst EoT" not in plain

    def test_emit_median_and_worst_on_spread(self):
        out = io.StringIO()
        print_session_summary(
            [_m(eot=0.85), _m(eot=0.90), _m(eot=1.40)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Median of [850, 900, 1400] = 900.
        assert "Median EoT:       900ms" in plain
        assert "Worst EoT:        1400ms" in plain

    def test_zeros_filtered(self):
        out = io.StringIO()
        print_session_summary(
            [_m(eot=0.0), _m(eot=0.85), _m(eot=0.95)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Median of [850, 950] = 900 (zeros excluded).
        assert "Median EoT:       900ms" in plain


# ---- record_utterance_streaming integration ---------------------------


def _stt_engine():
    engine = SimpleNamespace(_last_text=None, model_repo="stub")
    def transcribe(wav):
        return "hello there" if wav else None
    return engine, transcribe


class TestRecorderEmits:
    """End-to-end through the actual recorder using virtual mic."""

    def test_eot_populated_on_done_ok(self):
        # 0.3s pre-silence, 0.6s tone, 1.5s post-silence.
        # silence_duration default = 0.8s, so DONE_OK fires ~0.8s
        # after the tone ends. Last-speech-at = ~tone_end. EoT
        # latency should land ~800ms (plus ≤ chunk granularity).
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(concat(
            make_silence(0.3, rate=RATE),
            make_tone_burst(0.6, rate=RATE, amp=0.3),
            make_silence(1.5, rate=RATE),
        ))
        engine, transcribe = _stt_engine()
        out_metrics: dict = {}
        wav, dur, _ = record_utterance_streaming(
            mic, engine, transcribe_fn=transcribe,
            output=io.StringIO(),
            out_metrics=out_metrics,
        )
        assert wav  # DONE_OK path was reached
        assert "eot_latency" in out_metrics
        # Allow a generous window: silence_duration + a few chunks.
        eot = out_metrics["eot_latency"]
        # Should be at least silence_duration (0.8s) — VAD waited
        # that long before deciding the user really stopped.
        assert eot >= 0.7
        # Hard upper bound: silence_duration + a couple chunks.
        assert eot < 1.2

    def test_eot_omitted_when_too_short(self):
        # Brief tone < min_speech_duration → DONE_TOO_SHORT, not
        # DONE_OK. Recorder returns early without populating.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(concat(
            make_silence(0.1, rate=RATE),
            make_tone_burst(0.10, rate=RATE, amp=0.3),  # below 0.3s default
            make_silence(1.5, rate=RATE),
        ))
        engine, _ = _stt_engine()
        out_metrics: dict = {}
        wav, dur, stt_time = record_utterance_streaming(
            mic, engine, transcribe_fn=lambda w: "x",
            output=io.StringIO(),
            out_metrics=out_metrics,
        )
        # DONE_TOO_SHORT returns empty bytes.
        assert wav == b""
        assert "eot_latency" not in out_metrics

    def test_param_optional(self):
        # Calling without out_metrics still works (backwards compat).
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(concat(
            make_silence(0.3, rate=RATE),
            make_tone_burst(0.6, rate=RATE, amp=0.3),
            make_silence(1.5, rate=RATE),
        ))
        engine, transcribe = _stt_engine()
        wav, dur, _ = record_utterance_streaming(
            mic, engine, transcribe_fn=transcribe,
            output=io.StringIO(),
        )
        assert wav  # DONE_OK reached without error


# ---- ChatLoop wiring -------------------------------------------------


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


class TestChatLoopWiring:
    def test_eot_lands_on_metrics(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(concat(
            make_silence(0.3, rate=RATE),
            make_tone_burst(0.6, rate=RATE, amp=0.3),
            make_silence(1.5, rate=RATE),
        ))
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
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Should be populated and within sanity bounds (silence_duration
        # default ≈ 0.8s, tolerate up to ~1.2s for chunk granularity).
        assert result.metrics.eot_latency >= 0.7
        assert result.metrics.eot_latency < 1.2
