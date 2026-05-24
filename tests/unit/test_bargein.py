"""Tests for the iter-009 barge-in primitives.

Three pieces:
  1. ``examples._chat_playback.play_aligned`` — checks ``cancel_event``
     between chunks and breaks the play loop early when set.
  2. ``examples._chat_pipeline.SentenceWorker.cancel`` — sets the
     cancel event (interrupts current play_fn), drains the queue,
     joins.
  3. ``examples._chat_pipeline.BargeInWatcher`` — listens on a mic
     stream, runs ``VadState``, fires a callback when user speech
     crosses the threshold.

Plus an integration test that wires all three together with a
deterministic test play_fn that respects cancel_event.
"""

from __future__ import annotations

import io
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_helpers import VadEvent  # noqa: E402
from examples._chat_pipeline import BargeInWatcher, SentenceWorker  # noqa: E402
from examples._chat_playback import TTS_RATE, play_aligned  # noqa: E402
from examples.virtual_audio import (  # noqa: E402
    VirtualMicStream,
    VirtualSpeakerStream,
    concat,
    make_silence,
    make_tone_burst,
)


# ---- 1) play_aligned cancel_event --------------------------------------------


def _short_audio(samples: int = 4096) -> np.ndarray:
    return np.full(samples, 0.5, dtype=np.float32)


class StepClock:
    def __init__(self, step: float = 0.001):
        self._step = step
        self._t = 0.0

    def __call__(self) -> float:
        t = self._t
        self._t += self._step
        return t


class TestPlayAlignedCancel:
    def test_cancel_event_set_before_play_breaks_immediately(self):
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        ev = threading.Event()
        ev.set()  # already set before we even start

        elapsed = play_aligned(
            spk,
            _short_audio(4096),
            tokens=[],
            output=io.StringIO(),
            clock=StepClock(),
            cancel_event=ev,
            play_chunk=512,
        )
        # No bytes should have been written.
        assert spk.captured == bytearray()
        assert elapsed >= 0

    def test_cancel_event_set_mid_loop_breaks_between_chunks(self):
        """Set the event after exactly one chunk's worth of writes have
        landed. The next iteration should break, leaving a partial write.
        """
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        ev = threading.Event()

        # Wrap the speaker so we can set the cancel event after the
        # first write, simulating what a barge-in watcher would do.
        original_write = spk.write
        write_count = {"n": 0}

        def hook(data):
            original_write(data)
            write_count["n"] += 1
            if write_count["n"] == 1:
                ev.set()

        spk.write = hook  # type: ignore[method-assign]

        play_aligned(
            spk,
            _short_audio(4096),
            tokens=[],
            output=io.StringIO(),
            clock=StepClock(),
            cancel_event=ev,
            play_chunk=512,
        )
        # Only the first 512 samples (1024 bytes) should have been written.
        assert len(spk.captured) == 512 * 2
        assert write_count["n"] == 1

    def test_cancel_event_unset_completes_full_audio(self):
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        ev = threading.Event()  # never set
        play_aligned(
            spk,
            _short_audio(2048),
            tokens=[],
            output=io.StringIO(),
            clock=StepClock(),
            cancel_event=ev,
            play_chunk=512,
        )
        assert len(spk.captured) == 2048 * 2

    def test_no_cancel_event_argument_works_unchanged(self):
        """Backward compat: callers that don't pass cancel_event are
        unaffected.
        """
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        play_aligned(
            spk,
            _short_audio(1024),
            tokens=[],
            output=io.StringIO(),
            clock=StepClock(),
            play_chunk=256,
        )
        assert len(spk.captured) == 1024 * 2


# ---- 2) SentenceWorker.cancel ------------------------------------------------


def _factory():
    return VirtualSpeakerStream(rate=TTS_RATE)


def _const_synth(samples: int = 2048):
    def synth(text: str):
        return np.full(samples, 0.5, dtype=np.float32), []
    return synth


class TestSentenceWorkerCancel:
    def test_cancel_interrupts_current_play_via_event(self):
        """play_fn that respects cancel_event: writes audio chunk-by-
        chunk, breaks when the event is set. Verify cancel() drops
        pending sentences and the in-flight one stops mid-way.
        """
        cancel_received_at = threading.Event()
        proceed_to_finish = threading.Event()

        def cancellable_play(speaker, audio_np, tokens, *,
                             is_first_sentence=False, cancel_event=None):
            audio_int16 = (audio_np * 32767).astype(np.int16)
            written = 0
            chunk = 512
            t0 = time.monotonic()
            while written < len(audio_int16):
                if cancel_event is not None and cancel_event.is_set():
                    cancel_received_at.set()
                    break
                end = min(written + chunk, len(audio_int16))
                speaker.write(audio_int16[written:end].tobytes())
                written = end
                # Yield briefly so the test thread can call cancel() between
                # chunks. Without this, the worker thread races through.
                time.sleep(0.005)
            return time.monotonic() - t0

        w = SentenceWorker(
            speaker_factory=_factory,
            synth_fn=_const_synth(samples=10000),  # 10k samples → many chunks
            play_fn=cancellable_play,
        )
        w.start()
        w.submit("first")
        w.submit("second")  # should be dropped
        w.submit("third")  # should be dropped

        # Give the worker time to start the first sentence.
        time.sleep(0.05)
        w.cancel(timeout=5.0)

        assert w.cancelled is True
        # cancellable_play recorded that it saw cancel_event.is_set()
        assert cancel_received_at.is_set()
        # Only the first sentence was started; second and third never ran.
        assert w.sentences_spoken <= 1

    def test_cancel_is_idempotent(self):
        w = SentenceWorker(
            speaker_factory=_factory,
            synth_fn=_const_synth(),
            play_fn=lambda s, a, t, **kw: 0.0,
        )
        w.start()
        w.submit_done()
        w.wait_done(timeout=5.0)
        # Cancel after natural completion — no-op (idempotent).
        w.cancel(timeout=2.0)
        w.cancel(timeout=2.0)

    def test_cancel_before_start_is_noop(self):
        w = SentenceWorker(
            speaker_factory=_factory,
            synth_fn=_const_synth(),
            play_fn=lambda s, a, t, **kw: 0.0,
        )
        w.cancel(timeout=1.0)  # should not raise
        assert w.cancelled is False  # never even started

    def test_play_fn_without_cancel_event_kwarg_still_runs(self):
        """Backward compat: a play_fn that doesn't accept cancel_event
        still works — the worker falls back to the iter-008 contract.
        """
        spk_holder = {"spk": None}

        def factory():
            spk_holder["spk"] = VirtualSpeakerStream(rate=TTS_RATE)
            return spk_holder["spk"]

        def old_style_play(speaker, audio_np, tokens, *, is_first_sentence=False):
            speaker.write((audio_np * 32767).astype(np.int16).tobytes())
            return 0.05

        w = SentenceWorker(
            speaker_factory=factory,
            synth_fn=_const_synth(samples=512),
            play_fn=old_style_play,
        )
        w.start()
        w.submit("hello")
        w.submit_done()
        w.wait_done(timeout=5.0)
        assert w.sentences_spoken == 1
        assert len(spk_holder["spk"].captured) == 512 * 2


# ---- 3) BargeInWatcher -------------------------------------------------------


class TestBargeInWatcher:
    def test_silence_does_not_trigger_callback(self):
        mic = VirtualMicStream(rate=16000, chunk_size=1024)
        # Push a few seconds of pure silence so the watcher has data
        # to read but VAD never crosses threshold.
        mic.push(make_silence(2.0, rate=16000))
        called = threading.Event()

        watcher = BargeInWatcher(
            mic=mic,
            on_speech_detected=called.set,
            chunk_size=1024,
            rate=16000,
            poll_interval=0.001,
        )
        watcher.start()
        # Wait for all silence frames to drain.
        deadline = time.monotonic() + 1.0
        while mic.frames_buffered >= 1024 and time.monotonic() < deadline:
            time.sleep(0.01)
        watcher.stop(timeout=2.0)
        assert called.is_set() is False
        assert watcher.detected is False
        # All events should be IDLE for pure silence input.
        assert all(e is VadEvent.IDLE for e in watcher.events)

    def test_speech_burst_triggers_callback(self):
        mic = VirtualMicStream(rate=16000, chunk_size=1024)
        # Silence then speech — watcher should fire on first ACTIVE.
        mic.push(concat(
            make_silence(0.3, rate=16000),
            make_tone_burst(0.5, rate=16000, amp=0.3),
            make_silence(0.5, rate=16000),
        ))
        called = threading.Event()

        watcher = BargeInWatcher(
            mic=mic,
            on_speech_detected=called.set,
            chunk_size=1024,
            rate=16000,
            poll_interval=0.001,
        )
        watcher.start()
        # The watcher should set `called` quickly once it processes a
        # tone-burst frame.
        triggered = called.wait(timeout=2.0)
        watcher.stop(timeout=2.0)
        assert triggered is True
        assert watcher.detected is True
        assert watcher.frame_idx_at_trigger is not None
        # At least one ACTIVE event must have been observed.
        assert any(e is VadEvent.ACTIVE for e in watcher.events)

    def test_callback_only_fires_once_even_during_long_speech(self):
        mic = VirtualMicStream(rate=16000, chunk_size=1024)
        mic.push(concat(
            make_silence(0.3, rate=16000),
            make_tone_burst(2.0, rate=16000, amp=0.3),  # 2s tone
            make_silence(1.2, rate=16000),
        ))
        call_count = {"n": 0}

        def cb():
            call_count["n"] += 1

        watcher = BargeInWatcher(
            mic=mic,
            on_speech_detected=cb,
            chunk_size=1024,
            rate=16000,
            poll_interval=0.001,
        )
        watcher.start()
        # Wait for everything to drain.
        deadline = time.monotonic() + 3.0
        while mic.frames_buffered >= 1024 and time.monotonic() < deadline:
            time.sleep(0.01)
        watcher.stop(timeout=2.0)
        assert call_count["n"] == 1

    def test_trigger_on_done_ok_waits_for_silence_window(self):
        """trigger_on='done_ok' fires only after a complete utterance —
        useful when you don't want to interrupt on every cough.

        Uses a frame-aligned clock so the VAD's silence window
        elapses in virtual time as the watcher consumes pre-pushed
        audio. Without this, all the fast-served frames arrive in
        milliseconds of wall-clock and the 0.8s silence window
        never closes.
        """
        mic = VirtualMicStream(rate=16000, chunk_size=1024)
        mic.push(concat(
            make_silence(0.3, rate=16000),
            make_tone_burst(0.6, rate=16000, amp=0.3),
            # 0.8s+ silence forces VAD to fire DONE_OK.
            make_silence(1.2, rate=16000),
        ))
        called = threading.Event()

        # FrameClock-style: advances by chunk/rate per call, so VAD
        # sees the same time progression it would with real audio.
        frame_dt = 1024 / 16000.0
        t = [0.0]

        def frame_clock():
            now = t[0]
            t[0] += frame_dt
            return now

        watcher = BargeInWatcher(
            mic=mic,
            on_speech_detected=called.set,
            trigger_on="done_ok",
            chunk_size=1024,
            rate=16000,
            poll_interval=0.001,
            clock=frame_clock,
        )
        watcher.start()
        triggered = called.wait(timeout=3.0)
        watcher.stop(timeout=2.0)
        assert triggered is True
        # Make sure DONE_OK was actually in the event log.
        assert VadEvent.DONE_OK in watcher.events

    def test_invalid_trigger_on_raises(self):
        with pytest.raises(ValueError):
            BargeInWatcher(
                mic=VirtualMicStream(),
                on_speech_detected=lambda: None,
                trigger_on="bogus",
            )

    def test_double_start_raises(self):
        mic = VirtualMicStream(rate=16000)
        watcher = BargeInWatcher(mic=mic, on_speech_detected=lambda: None)
        watcher.start()
        try:
            with pytest.raises(RuntimeError):
                watcher.start()
        finally:
            watcher.stop(timeout=1.0)

    def test_stop_before_start_is_noop(self):
        watcher = BargeInWatcher(
            mic=VirtualMicStream(),
            on_speech_detected=lambda: None,
        )
        watcher.stop(timeout=0.5)  # should not raise

    def test_frames_captured_for_replay(self):
        """The watcher records each mic frame so the orchestrator can
        replay the user's first syllables into the next record loop.
        """
        mic = VirtualMicStream(rate=16000, chunk_size=1024)
        mic.push(concat(
            make_silence(0.1, rate=16000),
            make_tone_burst(0.3, rate=16000, amp=0.3),
            make_silence(0.3, rate=16000),
        ))
        called = threading.Event()
        watcher = BargeInWatcher(
            mic=mic,
            on_speech_detected=called.set,
            chunk_size=1024,
            rate=16000,
            poll_interval=0.001,
        )
        watcher.start()
        called.wait(timeout=2.0)
        # Let it consume the rest, then stop.
        deadline = time.monotonic() + 1.0
        while mic.frames_buffered >= 1024 and time.monotonic() < deadline:
            time.sleep(0.01)
        watcher.stop(timeout=2.0)
        # We should have captured several frames covering the audio.
        assert len(watcher.frames) > 0
        # Each frame is chunk_size * 2 bytes (int16).
        for f in watcher.frames:
            assert len(f) == 1024 * 2


# ---- 4) Integration: barge-in cancels worker mid-sentence --------------------


class TestBargeInIntegration:
    def test_user_speech_cancels_worker_mid_sentence(self):
        """End-to-end barge-in:
          1. SentenceWorker is playing a long sentence.
          2. User audio appears on the mic.
          3. BargeInWatcher detects ACTIVE, calls worker.cancel.
          4. play_fn sees cancel_event, breaks mid-stream.
          5. Pending queued sentences are dropped.
        """
        # Worker side: speaker + slow play_fn that respects cancel_event.
        spk_holder = {"spk": None}

        def factory():
            spk_holder["spk"] = VirtualSpeakerStream(rate=TTS_RATE)
            return spk_holder["spk"]

        def cancellable_play(speaker, audio_np, tokens, *,
                             is_first_sentence=False, cancel_event=None):
            audio_int16 = (audio_np * 32767).astype(np.int16)
            written = 0
            chunk = 1024
            t0 = time.monotonic()
            while written < len(audio_int16):
                if cancel_event is not None and cancel_event.is_set():
                    break
                end = min(written + chunk, len(audio_int16))
                speaker.write(audio_int16[written:end].tobytes())
                written = end
                time.sleep(0.005)  # ~5ms per chunk so the test races finish
            return time.monotonic() - t0

        worker = SentenceWorker(
            speaker_factory=factory,
            synth_fn=_const_synth(samples=20000),  # 20k samples → ~20 chunks
            play_fn=cancellable_play,
        )

        # Mic side: virtual mic, watcher pointed at it.
        mic = VirtualMicStream(rate=16000, chunk_size=1024)
        # Initially silent — push speech burst once worker has started.
        watcher = BargeInWatcher(
            mic=mic,
            on_speech_detected=lambda: worker.cancel(timeout=2.0),
            chunk_size=1024,
            rate=16000,
            poll_interval=0.001,
        )

        worker.start()
        watcher.start()

        worker.submit("first long sentence")
        worker.submit("second sentence (will be dropped)")
        worker.submit("third sentence (will be dropped)")
        # Don't submit_done — we want cancel() to be the terminator.

        # Give the worker time to start the first sentence's play loop.
        time.sleep(0.05)

        # User starts talking — push speech to the mic.
        mic.push(make_tone_burst(0.5, rate=16000, amp=0.3))

        # Wait for the worker to be cancelled (watcher fires cancel,
        # worker thread joins).
        worker.wait_done(timeout=5.0)
        watcher.stop(timeout=2.0)

        # Assertions:
        assert worker.cancelled is True
        assert watcher.detected is True
        # First sentence was interrupted: less than full 20000 samples
        # were written (mid-stream break).
        full_bytes = 20000 * 2
        assert 0 < len(spk_holder["spk"].captured) < full_bytes
        # Pending sentences were dropped.
        assert worker.sentences_spoken <= 1
