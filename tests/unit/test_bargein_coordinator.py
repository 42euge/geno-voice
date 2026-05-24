"""Tests for ``examples._chat_pipeline.BargeInCoordinator``.

The coordinator is a single-shot signal that bundles together the
actions that need to happen on user barge-in: stop the LLM-token
consumer, cancel the SentenceWorker, fire any caller-provided
hangup hook (e.g. close an HTTP requests stream).

These tests use a stub worker so we don't need a real
SentenceWorker, then a separate "with real worker" test that
verifies cancel actually flows through.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_pipeline import BargeInCoordinator, SentenceWorker  # noqa: E402
from examples.virtual_audio import VirtualSpeakerStream  # noqa: E402


class StubWorker:
    """Minimal SentenceWorker-shape that just records cancel calls."""

    def __init__(self, *, cancel_raises=False):
        self.cancel_calls = 0
        self.cancel_raises = cancel_raises

    def cancel(self, timeout: float = 5.0) -> None:
        self.cancel_calls += 1
        if self.cancel_raises:
            raise RuntimeError("worker boom in cancel")


# ---- Basic semantics ---------------------------------------------------------


class TestCoordinatorBasic:
    def test_is_set_starts_false(self):
        c = BargeInCoordinator()
        assert c.is_set() is False
        assert c.triggered_at is None

    def test_trigger_sets_event_and_timestamps(self):
        c = BargeInCoordinator()
        before = time.monotonic()
        c.trigger()
        after = time.monotonic()
        assert c.is_set() is True
        assert c.triggered_at is not None
        assert before <= c.triggered_at <= after

    def test_event_property_returns_underlying_threading_event(self):
        c = BargeInCoordinator()
        ev = c.event
        assert isinstance(ev, threading.Event)
        assert ev.is_set() is False
        c.trigger()
        assert ev.is_set() is True


# ---- Idempotency -------------------------------------------------------------


class TestCoordinatorIdempotent:
    def test_trigger_twice_only_fires_once(self):
        worker = StubWorker()
        hits = {"n": 0}

        def hook():
            hits["n"] += 1

        c = BargeInCoordinator(worker=worker, on_trigger=hook)
        c.trigger()
        c.trigger()
        c.trigger()
        assert worker.cancel_calls == 1
        assert hits["n"] == 1

    def test_triggered_at_does_not_update_on_second_trigger(self):
        c = BargeInCoordinator()
        c.trigger()
        first_ts = c.triggered_at
        time.sleep(0.005)
        c.trigger()
        assert c.triggered_at == first_ts


# ---- Wiring ------------------------------------------------------------------


class TestCoordinatorWiring:
    def test_calls_worker_cancel(self):
        worker = StubWorker()
        c = BargeInCoordinator(worker=worker)
        c.trigger()
        assert worker.cancel_calls == 1

    def test_no_worker_means_no_cancel(self):
        # No worker bound — trigger should still set the event.
        c = BargeInCoordinator(worker=None)
        c.trigger()
        assert c.is_set() is True

    def test_on_trigger_hook_fires(self):
        hits = {"n": 0}

        def hook():
            hits["n"] += 1

        c = BargeInCoordinator(on_trigger=hook)
        c.trigger()
        assert hits["n"] == 1

    def test_event_set_before_worker_cancel_returns(self):
        """The event must be set BEFORE the worker.cancel call so the
        for-token loop can see the flag immediately, even if the
        worker's cancel does a long-ish join.
        """
        events_seen_in_cancel: list[bool] = []

        class SlowCancelWorker:
            def cancel(self, timeout=5.0):
                # Note: we can't easily check the coordinator's event
                # from inside without a back-reference; instead, we
                # block briefly so a concurrent reader could observe.
                time.sleep(0.005)
                events_seen_in_cancel.append(True)

        c = BargeInCoordinator(worker=SlowCancelWorker())

        # Concurrent reader: spin in a thread, check is_set() until
        # the trigger completes.
        observed = {"set_before_complete": False}
        ready = threading.Event()
        done = threading.Event()

        def watcher():
            ready.set()
            while not done.is_set():
                if c.is_set():
                    observed["set_before_complete"] = True
                    return
                time.sleep(0.0005)

        t = threading.Thread(target=watcher, daemon=True)
        t.start()
        ready.wait(timeout=1.0)
        c.trigger()
        done.set()
        t.join(timeout=2.0)
        assert observed["set_before_complete"] is True


# ---- Robustness --------------------------------------------------------------


class TestCoordinatorRobustness:
    def test_worker_cancel_exception_does_not_break_trigger(self):
        worker = StubWorker(cancel_raises=True)
        hits = {"n": 0}

        def hook():
            hits["n"] += 1

        c = BargeInCoordinator(worker=worker, on_trigger=hook)
        c.trigger()  # must not raise
        assert c.is_set() is True
        # Hook still fires even though cancel raised.
        assert hits["n"] == 1

    def test_on_trigger_exception_does_not_break_trigger(self):
        worker = StubWorker()

        def bad_hook():
            raise RuntimeError("hook boom")

        c = BargeInCoordinator(worker=worker, on_trigger=bad_hook)
        c.trigger()  # must not raise
        assert c.is_set() is True
        assert worker.cancel_calls == 1


# ---- Real SentenceWorker integration -----------------------------------------


def _const_synth(samples: int = 2048):
    def synth(text):
        return np.full(samples, 0.5, dtype=np.float32), []
    return synth


def _slow_play(speaker, audio_np, tokens, *, is_first_sentence=False, cancel_event=None):
    audio_int16 = (audio_np * 32767).astype(np.int16)
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


class TestCoordinatorWithRealWorker:
    def test_trigger_cancels_worker_for_real(self):
        spk_holder = {"spk": None}

        def factory():
            spk_holder["spk"] = VirtualSpeakerStream(rate=24000)
            return spk_holder["spk"]

        worker = SentenceWorker(
            speaker_factory=factory,
            synth_fn=_const_synth(samples=20000),
            play_fn=_slow_play,
        )
        worker.start()
        worker.submit("a")
        worker.submit("b (gets dropped)")

        c = BargeInCoordinator(worker=worker)
        time.sleep(0.05)  # let worker start sentence "a"
        c.trigger()

        worker.wait_done(timeout=5.0)
        assert worker.cancelled is True
        # Less than full first-sentence audio was written (interrupted).
        assert 0 < len(spk_holder["spk"].captured) < 20000 * 2
