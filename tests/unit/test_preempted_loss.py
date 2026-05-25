"""Tests for iter-080 — pre-empted-content loss.

Metric 3.7 from docs/perf-metrics-taxonomy.md.

    preempted_words = max(0, len(response.split()) - worker.word_count_total)

Words the LLM generated but the user never heard because of a
mid-stream barge. 0 on non-barge turns. High values signal the
bot was being verbose enough that the user interrupted; pairs with
iter-069 interruption rate and iter-047 barge phase to localize
the cause.
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


# ---- Default + per-turn print ----------------------------------


class TestDefault:
    def test_turnmetrics_default_zero(self):
        assert TurnMetrics().preempted_words == 0


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
        m = TurnMetrics(transcript="hi", model="stub",
                        barge_in=True, preempted_words=0)
        assert "Pre-empted" not in self._capture(m)

    def test_no_barge_no_preempted_line(self):
        # The block is nested under barge_in; without barge, no line.
        m = TurnMetrics(transcript="hi", model="stub",
                        barge_in=False, preempted_words=15)
        assert "Pre-empted" not in self._capture(m)

    def test_low_preempted_dim(self):
        # ≤10 words = clean cut-off mid-sentence.
        m = TurnMetrics(transcript="hi", model="stub",
                        barge_in=True, preempted_words=5)
        out = self._capture(m)
        assert "Pre-empted:" in out
        assert "5 words" in out
        assert "generated but not played" in out

    def test_high_preempted_yellow(self):
        # >10 words = bot was being verbose.
        m = TurnMetrics(transcript="hi", model="stub",
                        barge_in=True, preempted_words=42)
        out = self._capture(m)
        assert "Pre-empted:" in out
        assert "42 words" in out


# ---- Session aggregate -----------------------------------------


def _m(barge=False, preempted=0):
    return TurnMetrics(
        ttfs=0.5, barge_in=barge, preempted_words=preempted,
    )


def _summary(metrics_list, **kwargs):
    out = io.StringIO()
    print_session_summary(metrics_list, {"model": "stub"}, file=out, **kwargs)
    return _strip_ansi(out.getvalue())


class TestSessionSummary:
    def test_no_barges_omits(self):
        # Whole barge sub-block is gated; no Pre-empted line.
        plain = _summary([_m(), _m()])
        assert "Pre-empted words" not in plain

    def test_barge_with_no_loss_omits(self):
        # Clean cut between sentences — preempted == 0 across all
        # barge turns. Block emits Barge-ins line but not Pre-empted.
        plain = _summary([_m(barge=True, preempted=0), _m()])
        assert "Pre-empted words" not in plain

    def test_one_lossy_barge(self):
        plain = _summary([
            _m(barge=True, preempted=18),  # mid-content cut
            _m(),
        ])
        # 18 total, 1/1 barges with loss, avg 18.
        assert "Pre-empted words: 18 total (1/1 barges, 18 avg/loss)" in plain

    def test_mix_lossy_clean(self):
        plain = _summary([
            _m(barge=True, preempted=0),    # clean cut
            _m(barge=True, preempted=15),
            _m(barge=True, preempted=25),
            _m(),
        ])
        # 40 total, 2/3 barges with loss, avg 20 (40/2).
        assert "Pre-empted words: 40 total (2/3 barges, 20 avg/loss)" in plain


# ---- ChatLoop wiring (deterministic via barge scenario) -----------


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


class TestChatLoopWiring:
    def test_no_barge_zero(self):
        # Clean turn — full response, no barge → no preempted.
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
        assert result.metrics.preempted_words == 0

    def test_barge_loss_non_negative(self):
        # Barge mid-response: preempted_words >= 0 always (clamp).
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
        # >= 0 guaranteed by the max(0, ...) clamp.
        assert result.metrics.preempted_words >= 0
        # Bounded above by total response word count.
        if result.metrics.barge_in:
            response_words = len(result.metrics.response.split())
            assert result.metrics.preempted_words <= response_words
