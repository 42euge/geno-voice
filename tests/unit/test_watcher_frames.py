"""Tests for iter-025 — BargeInWatcher captures only post-detection frames.

Pre-iter-025 the watcher stored every frame from .start() to .stop(),
including pre-detection silence/noise/feedback. When fed back into
``record_utterance_streaming`` as ``primed_frames``, this could include
bot acoustic feedback in production with speakers — STT would
transcribe both bot and user voice.

Now frames stores only:
  - the trigger frame itself (the chunk where speech first crossed
    threshold; contains the user's first syllable)
  - all subsequent frames until stop()
  - optionally, the most recent N pre-detection frames if the
    caller asks for ``lead_in_chunks > 0``

Tests verify:
  - Default lead_in=0: pre-detection frames are dropped
  - lead_in=N > 0: ring buffer captures up to N pre-detection
    frames, flushed to frames at detection
  - The trigger frame is always included
  - Negative lead_in raises
  - Behavioral parity for tests that just check `frames > 0`
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_pipeline import BargeInWatcher  # noqa: E402
from examples.virtual_audio import (  # noqa: E402
    VirtualMicStream,
    concat,
    make_silence,
    make_tone_burst,
)


def _frame_clock(chunk: int = 1024, rate: int = 16000):
    """Same FrameClock pattern from earlier tests."""
    dt = chunk / rate
    t = [0.0]

    def clock():
        v = t[0]
        t[0] += dt
        return v

    return clock


def _drive_watcher_to_completion(watcher, mic, timeout=2.0):
    """Wait until the mic buffer is drained or timeout expires."""
    deadline = time.monotonic() + timeout
    while mic.frames_buffered >= watcher._chunk and time.monotonic() < deadline:
        time.sleep(0.01)


# ---- default lead_in_chunks=0 -----------------------------------------------


class TestPostDetectionOnly:
    def test_no_pre_detection_frames_stored_by_default(self):
        # Push significant silence (no speech) — watcher reads it,
        # but never detects → frames stays empty.
        mic = VirtualMicStream(rate=16000, chunk_size=1024)
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
        _drive_watcher_to_completion(watcher, mic)
        watcher.stop(timeout=2.0)

        assert watcher.detected is False
        assert called.is_set() is False
        # No detection → no frames stored.
        assert watcher.frames == []
        # But events were fed — VAD got every frame.
        assert len(watcher.events) > 0

    def test_pre_detection_silence_dropped(self):
        # 0.5s silence (drops), 0.5s tone (triggers + stored), 0.5s
        # silence (post-trigger, stored).
        mic = VirtualMicStream(rate=16000, chunk_size=1024)
        mic.push(concat(
            make_silence(0.5, rate=16000),
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
        called.wait(timeout=2.0)
        _drive_watcher_to_completion(watcher, mic)
        watcher.stop(timeout=2.0)

        assert watcher.detected is True
        # Pre-detection silence (~0.5s = 8 chunks) is dropped, so
        # frames should be substantially less than total events.
        assert len(watcher.frames) < len(watcher.events)
        # And less than the total 1.5s of audio (24 chunks).
        assert len(watcher.frames) < 24

    def test_trigger_frame_is_included(self):
        # First tone-burst frame should be in frames (it's the
        # one that fired the trigger).
        mic = VirtualMicStream(rate=16000, chunk_size=1024)
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
        called.wait(timeout=2.0)
        _drive_watcher_to_completion(watcher, mic)
        watcher.stop(timeout=2.0)

        assert watcher.detected is True
        assert len(watcher.frames) >= 1
        # First stored frame should be high-RMS (trigger frame
        # is the first loud one).
        first = np.frombuffer(watcher.frames[0], dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(first ** 2)))
        assert rms > 0.05  # tone burst, not silence


# ---- lead_in_chunks > 0 -----------------------------------------------------


class TestLeadInBuffer:
    def test_lead_in_captures_pre_detection_frames(self):
        # 0.5s silence + 0.5s tone + 0.5s silence. With
        # lead_in_chunks=3, the 3 most recent silence chunks
        # before detection should be captured.
        mic = VirtualMicStream(rate=16000, chunk_size=1024)
        mic.push(concat(
            make_silence(0.5, rate=16000),
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
            lead_in_chunks=3,
        )
        watcher.start()
        called.wait(timeout=2.0)
        _drive_watcher_to_completion(watcher, mic)
        watcher.stop(timeout=2.0)

        assert watcher.detected is True
        # Without lead-in: trigger frame onwards (~12 chunks for
        # 0.5s tone + 0.5s silence). With lead_in=3: 3 silence
        # frames + trigger frame + post-trigger frames.
        # So frames should be at least 4 (3 lead-in + 1 trigger).
        assert len(watcher.frames) >= 4
        # The first 3 should be silence (RMS ≈ 0).
        for i in range(3):
            f = np.frombuffer(watcher.frames[i], dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(f ** 2)))
            assert rms < 0.01, f"Frame {i} should be silence (lead-in)"
        # The 4th frame onwards should have signal.
        f3 = np.frombuffer(watcher.frames[3], dtype=np.int16).astype(np.float32) / 32768.0
        rms_f3 = float(np.sqrt(np.mean(f3 ** 2)))
        assert rms_f3 > 0.05, "Trigger frame should have signal"

    def test_lead_in_zero_equivalent_to_no_kwarg(self):
        # Explicit lead_in_chunks=0 should behave identically
        # to omitting it.
        mic_a = VirtualMicStream(rate=16000, chunk_size=1024)
        mic_b = VirtualMicStream(rate=16000, chunk_size=1024)
        audio = concat(
            make_silence(0.5, rate=16000),
            make_tone_burst(0.5, rate=16000, amp=0.3),
            make_silence(0.5, rate=16000),
        )
        mic_a.push(audio)
        mic_b.push(audio)

        called_a = threading.Event()
        watcher_a = BargeInWatcher(
            mic=mic_a, on_speech_detected=called_a.set,
            poll_interval=0.001,
        )
        called_b = threading.Event()
        watcher_b = BargeInWatcher(
            mic=mic_b, on_speech_detected=called_b.set,
            poll_interval=0.001,
            lead_in_chunks=0,
        )
        watcher_a.start()
        watcher_b.start()
        called_a.wait(timeout=2.0)
        called_b.wait(timeout=2.0)
        _drive_watcher_to_completion(watcher_a, mic_a)
        _drive_watcher_to_completion(watcher_b, mic_b)
        watcher_a.stop(timeout=2.0)
        watcher_b.stop(timeout=2.0)

        # Same number of stored frames (assuming similar timing).
        # Allow off-by-one for race-condition timing.
        assert abs(len(watcher_a.frames) - len(watcher_b.frames)) <= 1

    def test_lead_in_ring_buffer_caps_at_max_size(self):
        # 1.0s silence (10 chunks) + 0.5s tone. With
        # lead_in_chunks=2, only the last 2 silence chunks should
        # be carried forward, not all 10.
        mic = VirtualMicStream(rate=16000, chunk_size=1024)
        mic.push(concat(
            make_silence(1.0, rate=16000),
            make_tone_burst(0.5, rate=16000, amp=0.3),
            make_silence(0.5, rate=16000),
        ))

        called = threading.Event()
        watcher = BargeInWatcher(
            mic=mic,
            on_speech_detected=called.set,
            poll_interval=0.001,
            lead_in_chunks=2,
        )
        watcher.start()
        called.wait(timeout=2.0)
        _drive_watcher_to_completion(watcher, mic)
        watcher.stop(timeout=2.0)

        assert watcher.detected is True
        # Total frames < 10 (full silence) + tone_chunks. Should
        # be roughly 2 (lead-in) + post-trigger frames. Let's
        # bound it: < 20 chunks total.
        assert len(watcher.frames) < 20
        # First 2 frames are the silence-ring-buffer.
        for i in range(2):
            f = np.frombuffer(watcher.frames[i], dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(f ** 2)))
            assert rms < 0.01

    def test_negative_lead_in_chunks_raises(self):
        with pytest.raises(ValueError):
            BargeInWatcher(
                mic=VirtualMicStream(rate=16000),
                on_speech_detected=lambda: None,
                lead_in_chunks=-1,
            )

    def test_lead_in_internal_buffer_cleared_after_flush(self):
        # After detection fires, the lead-in buffer should be
        # empty (its contents went into frames). Subsequent frames
        # go directly to frames, not through the buffer.
        mic = VirtualMicStream(rate=16000, chunk_size=1024)
        mic.push(concat(
            make_silence(0.3, rate=16000),
            make_tone_burst(0.5, rate=16000, amp=0.3),
            make_silence(0.5, rate=16000),
        ))

        called = threading.Event()
        watcher = BargeInWatcher(
            mic=mic, on_speech_detected=called.set,
            poll_interval=0.001,
            lead_in_chunks=5,
        )
        watcher.start()
        called.wait(timeout=2.0)
        _drive_watcher_to_completion(watcher, mic)
        watcher.stop(timeout=2.0)

        # Internal buffer should be cleared post-detection.
        assert watcher._lead_in_buffer == []
