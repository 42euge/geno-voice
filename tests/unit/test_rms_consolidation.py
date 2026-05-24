"""Tests for iter-024 RMS consolidation.

BargeInWatcher used to inline the RMS computation (with the
iter-014 NaN-on-empty guard duplicated alongside the version in
``_chat_recording.rms``). Now both call the same ``rms()`` helper.

These tests verify:
  - The watcher binds the rms helper on construction.
  - Behavioral parity: same audio input → same VAD event sequence
    as before the consolidation (regression cover).
  - The empty-bytes path is handled (would NaN before iter-014;
    same fix now lives in one place).
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
from examples._chat_recording import rms  # noqa: E402
from examples.virtual_audio import (  # noqa: E402
    VirtualMicStream,
    concat,
    make_silence,
    make_tone_burst,
)


class TestWatcherUsesConsolidatedRms:
    def test_watcher_binds_rms_helper_at_construction(self):
        # The watcher should hold a reference to the centralized
        # rms function. If a future refactor drifts back to inline
        # logic, this assert catches it immediately.
        mic = VirtualMicStream(rate=16000)
        watcher = BargeInWatcher(
            mic=mic,
            on_speech_detected=lambda: None,
        )
        assert watcher._rms is rms

    def test_silence_does_not_trigger_callback(self):
        """Pre-iter-024 behavioral test that should still pass.
        Verifies the consolidated rms returns 0.0 for silence and
        doesn't accidentally trigger the watcher.
        """
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
        deadline = time.monotonic() + 1.0
        while mic.frames_buffered >= 1024 and time.monotonic() < deadline:
            time.sleep(0.01)
        watcher.stop(timeout=2.0)
        assert called.is_set() is False
        assert watcher.detected is False

    def test_speech_triggers_via_consolidated_rms(self):
        """Same input that should trigger ACTIVE — previously fired
        via inline RMS, now via the centralized helper. Behavioral
        parity confirmed.
        """
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
        triggered = called.wait(timeout=2.0)
        watcher.stop(timeout=2.0)
        assert triggered is True
        assert watcher.detected is True


class TestSingleSourceOfTruthForRms:
    """Belt-and-suspenders: if the rms function gets a future fix
    (e.g. clipping detection), the test below ensures the watcher
    picks it up automatically — there's only one definition.
    """

    def test_same_function_object_in_recording_and_watcher(self):
        # Import from both call sites and verify identity, not
        # just equality. If someone copy-pastes a "fix" into
        # _chat_pipeline as a private helper, this fails.
        from examples._chat_recording import rms as rms_recording
        mic = VirtualMicStream(rate=16000)
        watcher = BargeInWatcher(
            mic=mic,
            on_speech_detected=lambda: None,
        )
        assert watcher._rms is rms_recording

    @pytest.mark.parametrize("samples,expected", [
        (np.zeros(1024, dtype=np.float32), 0.0),
        (np.full(1024, 0.5, dtype=np.float32), 0.5),
        (np.array([], dtype=np.float32), 0.0),  # iter-014 empty guard
    ])
    def test_rms_well_defined_on_canonical_inputs(self, samples, expected):
        # Exercise the centralized rms directly so any future
        # behavioral change is caught here too.
        assert rms(samples) == pytest.approx(expected)
