"""Tests for iter-044 — SentenceWorker idle-gap metric.

Metric 2.16 from docs/perf-metrics-taxonomy.md. Cumulative time
the worker spent blocked on ``self._queue.get(...)`` between
sentences. Excludes the first wait (= TTFsent, iter-038).

Combined with iter-043's streaming_overlap_ratio:
  - low overlap + high idle gap → LLM is the bottleneck.
  - low overlap + low idle gap  → synth is the bottleneck.
  - high overlap → pipeline is healthy.
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
from examples._chat_metrics import TurnMetrics  # noqa: E402
from examples._chat_pipeline import SentenceWorker  # noqa: E402
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


# ---- Default + per-turn print ----------------------------------------------


class TestDefault:
    def test_worker_default_zero(self):
        w = SentenceWorker(
            speaker_factory=lambda: _FakeSpeaker(),
            synth_fn=lambda s: (np.full(256, 0.5, dtype=np.float32), []),
            play_fn=_noop_play,
        )
        assert w.idle_gap_total == 0.0

    def test_turnmetrics_default_zero(self):
        assert TurnMetrics().worker_idle_gap_total == 0.0


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
        m = TurnMetrics(transcript="hi", model="stub", worker_idle_gap_total=0.0)
        out = self._capture(m)
        assert "Idle gap" not in out

    def test_nonzero_emits_ms(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            worker_idle_gap_total=0.180,  # 180ms
        )
        out = self._capture(m)
        assert "Idle gap" in out
        assert "180ms" in out
        assert "worker waited" in out


# ---- Worker behavior -------------------------------------------------------


class _FakeSpeaker:
    def __init__(self):
        self.captured: list[bytes] = []
    def write(self, data): self.captured.append(data)
    def stop_stream(self): pass
    def close(self): pass


def _noop_play(speaker, audio, tokens, *, is_first_sentence=False, cancel_event=None):
    speaker.write((audio * 32767).astype(np.int16).tobytes())
    return 0.001


def _slow_synth(samples=256, delay=0.0):
    def synth(s):
        if delay > 0:
            time.sleep(delay)
        return np.full(samples, 0.5, dtype=np.float32), []
    return synth


class TestWorkerIdleGap:
    def test_first_wait_does_not_count(self):
        # Worker started, no sentences submitted yet — submit one
        # immediately after a sleep, then submit_done. The wait
        # before the first sentence should NOT count toward idle_gap.
        w = SentenceWorker(
            speaker_factory=lambda: _FakeSpeaker(),
            synth_fn=_slow_synth(),
            play_fn=_noop_play,
        )
        w.start()
        # Sleep here represents "the LLM hasn't started streaming yet."
        time.sleep(0.05)
        w.submit("first")
        w.submit_done()
        w.wait_done(timeout=2.0)
        # Only one sentence was spoken; the gap before it doesn't
        # count (that's TTFsent territory). idle_gap should be 0.
        assert w.sentences_spoken == 1
        assert w.idle_gap_total == 0.0

    def test_between_sentence_gap_counted(self):
        # Submit sentence 1, wait, then submit sentence 2. The gap
        # between them is what we measure.
        w = SentenceWorker(
            speaker_factory=lambda: _FakeSpeaker(),
            synth_fn=_slow_synth(),
            play_fn=_noop_play,
        )
        w.start()
        w.submit("first")
        # Let the worker finish the first sentence + start blocking
        # on the next get().
        time.sleep(0.08)
        w.submit("second")
        w.submit_done()
        w.wait_done(timeout=2.0)
        assert w.sentences_spoken == 2
        # Gap should be approximately the sleep we did, minus
        # synth+play time of the first sentence (~0). Allow a wide
        # window since the worker thread + pytest overhead vary.
        assert w.idle_gap_total > 0
        # Don't assert tight upper bound — overhead varies. Just
        # confirm it's at least a meaningful chunk of the sleep.

    def test_back_to_back_sentences_zero_gap(self):
        # Both sentences submitted before worker starts. No
        # between-sentence wait — worker immediately picks up
        # the second after finishing the first.
        w = SentenceWorker(
            speaker_factory=lambda: _FakeSpeaker(),
            synth_fn=_slow_synth(),
            play_fn=_noop_play,
        )
        w.submit("first")
        w.submit("second")
        w.submit_done()
        w.start()
        w.wait_done(timeout=2.0)
        assert w.sentences_spoken == 2
        # idle_gap should be very small (microseconds — the time
        # between play returning and the next get() returning).
        # Bound loosely.
        assert w.idle_gap_total < 0.05


# ---- ChatLoop wiring -------------------------------------------------------


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
    def test_metric_lands_on_metrics(self):
        # End-to-end: real ChatLoop, real worker, slow LLM.
        # idle_gap should land on metrics (likely 0 here since the
        # synth is fast and the LLM is fast — but the FIELD must
        # be set, not None).
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine, transcribe = _stt_engine()
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=_yield_tokens("First. Second. Third."),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_slow_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Field is a float, not None.
        assert isinstance(result.metrics.worker_idle_gap_total, float)
        assert result.metrics.worker_idle_gap_total >= 0.0
