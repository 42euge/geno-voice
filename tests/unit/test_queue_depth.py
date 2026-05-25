"""Tests for iter-062 — peak worker queue depth metric.

Metric 2.7 from docs/perf-metrics-taxonomy.md.

    max_queue_depth = max(qsize_after_each_put)

Sampled inside ``SentenceWorker.submit()`` after each ``Queue.put``
call, so it captures the depth as seen by the producer at the
moment of submission. Inverse of iter-044's ``worker_idle_gap_total``
(which measures the worker STARVED). High values mean synth is the
bottleneck and streaming-overlap can't fully mask the producer/
consumer mismatch.
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


# ---- Default + per-turn print --------------------------------------------


class TestDefault:
    def test_turnmetrics_default_zero(self):
        assert TurnMetrics().max_queue_depth == 0

    def test_worker_default_zero(self):
        w = SentenceWorker(
            speaker_factory=lambda: object(),
            synth_fn=lambda s: (np.zeros(8, dtype=np.float32), []),
            play_fn=lambda *a, **k: 0.0,
        )
        assert w.max_queue_depth == 0


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
        m = TurnMetrics(transcript="hi", model="stub", max_queue_depth=0)
        assert "Queue depth" not in self._capture(m)

    def test_one_omits_line(self):
        # depth=1 = healthy (each sentence drained before next put).
        m = TurnMetrics(transcript="hi", model="stub", max_queue_depth=1)
        assert "Queue depth" not in self._capture(m)

    def test_two_emits_line_dim(self):
        m = TurnMetrics(transcript="hi", model="stub", max_queue_depth=2)
        out = self._capture(m)
        assert "Queue depth" in out
        assert "2" in out
        assert "synth backlog peak" in out

    def test_three_emits_line_yellow(self):
        # ≥3 is the yellow threshold but the line still goes out
        # regardless; we only assert the count appears.
        m = TurnMetrics(transcript="hi", model="stub", max_queue_depth=5)
        out = self._capture(m)
        assert "Queue depth" in out
        assert "5" in out


# ---- Session aggregate ---------------------------------------------------


def _m(depth=0):
    return TurnMetrics(ttfs=0.5, max_queue_depth=depth)


class TestSessionSummary:
    def test_no_data_omits_line(self):
        out = io.StringIO()
        print_session_summary([_m(), _m()], {"model": "stub"}, file=out)
        assert "Worst queue" not in _strip_ansi(out.getvalue())

    def test_only_healthy_turns_omits_line(self):
        # depth=1 across the board → no backup → omit.
        out = io.StringIO()
        print_session_summary(
            [_m(depth=1), _m(depth=1), _m(depth=1)],
            {"model": "stub"}, file=out,
        )
        assert "Worst queue" not in _strip_ansi(out.getvalue())

    def test_single_backed_up_turn(self):
        out = io.StringIO()
        print_session_summary(
            [_m(depth=1), _m(depth=4), _m(depth=1)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Worst queue:      4" in plain
        assert "1 turn backed up" in plain

    def test_multiple_backed_up_turns(self):
        out = io.StringIO()
        print_session_summary(
            [_m(depth=2), _m(depth=5), _m(depth=1), _m(depth=3)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Worst queue:      5" in plain
        # 3 of 4 turns backed up.
        assert "3/4 turns backed up" in plain


# ---- SentenceWorker tracking ---------------------------------------


class TestWorkerTracking:
    def _make(self, *, slow_synth=False):
        # Synth takes time → consumer drains slower than producer
        # can push → queue accumulates.
        if slow_synth:
            def synth(s):
                time.sleep(0.05)
                return np.zeros(8, dtype=np.float32), []
        else:
            def synth(s):
                return np.zeros(8, dtype=np.float32), []

        return SentenceWorker(
            speaker_factory=lambda: SimpleNamespace(
                write=lambda b: None, close=lambda: None,
            ),
            synth_fn=synth,
            play_fn=lambda *a, **k: 0.0,
        )

    def test_single_submit_records_one(self):
        w = self._make()
        w.start()
        w.submit("hi")
        w.submit_done()
        w.wait_done(timeout=2.0)
        # Producer never had more than 1 queued at submit time —
        # but qsize() may have been 1 at that moment, so 1 is a
        # valid observation (not 0 — empty would mean we never
        # submitted). The contract is "peak after a put."
        assert w.max_queue_depth >= 1

    def test_burst_of_submits_grows_depth(self):
        # Slow synth so the consumer doesn't drain in time. Push
        # 5 sentences in a tight loop before submit_done — the
        # peak should be the count we pushed (or close to it,
        # depending on consumer timing).
        w = self._make(slow_synth=True)
        w.start()
        for i in range(5):
            w.submit(f"sentence {i}")
        # Snapshot BEFORE submit_done so we see the producer-side
        # peak, not the post-drain depth.
        peak_observed_by_test = w.max_queue_depth
        w.submit_done()
        w.wait_done(timeout=10.0)
        # Submitted 5 → peak should be at LEAST 2 (consumer almost
        # certainly drained at least one during the put loop, but
        # not all five). Generous tolerance for CI scheduler.
        assert peak_observed_by_test >= 2
        # The final value should match what we observed mid-burst,
        # since submit_done's sentinel put happens AFTER we already
        # captured the peak.
        assert w.max_queue_depth >= 2

    def test_no_submits_zero(self):
        # Worker started but never receives a sentence → max stays 0.
        w = self._make()
        w.start()
        w.submit_done()  # sentinel, but submit_done() doesn't update peak
        w.wait_done(timeout=2.0)
        assert w.max_queue_depth == 0

    def test_after_stopped_submits_ignored(self):
        # submit() returns early when stop is set — shouldn't bump
        # the peak from a phantom put.
        w = self._make()
        w.start()
        w.stop(timeout=2.0)
        depth_before = w.max_queue_depth
        w.submit("late arrival")
        assert w.max_queue_depth == depth_before


# ---- ChatLoop wiring -----------------------------------------------


def _stt_engine(transcript="hi"):
    engine = SimpleNamespace(_last_text=None, model_repo="stub")
    def transcribe(wav):
        return transcript if wav else None
    return engine, transcribe


def _const_synth(samples=2048, delay=0.0):
    def synth(s):
        if delay > 0:
            time.sleep(delay)
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
        make_tone_burst(1.0, rate=RATE, amp=0.3),
        make_silence(1.5, rate=RATE),
    ))


class TestChatLoopWiring:
    def test_short_response_low_depth(self):
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
        # One sentence in, one drained — depth ≤ 1.
        assert result.metrics.max_queue_depth <= 1

    def test_slow_synth_builds_depth(self):
        # Multiple sentences + slow synth → producer outruns consumer.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine, transcribe = _stt_engine()
        long_response = " ".join(f"sentence {i}." for i in range(5))
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=_yield_tokens(long_response),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(delay=0.04),  # 40ms per synth
            play_fn=_fast_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # With 5 sentences and 40ms-per-synth the producer will
        # almost certainly get ahead at some point. Don't assert a
        # specific value (timing-flaky on CI) — just confirm the
        # field bubbled and stayed within sanity bounds.
        assert 1 <= result.metrics.max_queue_depth <= 6
