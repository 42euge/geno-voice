"""Tests for iter-028 — VAD config flows from ChatLoop to BargeInWatcher.

Pre-iter-028: ChatLoop accepted silence_threshold / silence_duration
/ min_speech_duration kwargs (iter-020) and forwarded them to
record_utterance_streaming. But the BargeInWatcher inside
run_one_turn was constructed with no `vad=` arg, so it built its
own VadState() with default values (silence_threshold=0.02 etc.).

A user setting ``chat.vad.silence_threshold = 0.05`` for a noisy
room would get the recorder threshold tuned but the watcher
still triggering on background noise — false barge-ins.

iter-028 builds a VadState from the ChatLoop's VAD params and
passes it to the watcher. These tests verify the wiring.
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

from examples._chat_helpers import VadState  # noqa: E402
from examples._chat_loop import ChatLoop  # noqa: E402
from examples._chat_pipeline import BargeInWatcher  # noqa: E402
from examples._chat_recording import CHUNK, RATE  # noqa: E402
from examples.virtual_audio import (  # noqa: E402
    VirtualMicStream,
    VirtualSpeakerStream,
    concat,
    make_silence,
    make_tone_burst,
)


# ---- Helpers ---------------------------------------------------------------


def _stt(transcript="hi"):
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
    import re

    def factory(messages, config):
        parts = re.findall(r"\S+|\.|!|\?", text)
        for p in parts:
            if per_token_delay > 0:
                time.sleep(per_token_delay)
            yield p + " "

    return factory


# ---- iter-028 wiring --------------------------------------------------------


class TestWatcherReceivesChatLoopVadConfig:
    """Capture the VadState that ChatLoop passes to BargeInWatcher
    and verify the parameters match the ChatLoop's VAD config.
    """

    def test_default_threshold_passed_through(self):
        captured = []

        original_init = BargeInWatcher.__init__

        def hook(self, *args, **kwargs):
            captured.append(kwargs.get("vad"))
            original_init(self, *args, **kwargs)

        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(concat(
            make_silence(0.3, rate=RATE),
            make_tone_burst(1.0, rate=RATE, amp=0.3),
            make_silence(1.5, rate=RATE),
        ))
        engine, transcribe = _stt()

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

        BargeInWatcher.__init__ = hook  # type: ignore[method-assign]
        try:
            loop.run_one_turn([])
        finally:
            BargeInWatcher.__init__ = original_init  # type: ignore[method-assign]

        assert len(captured) == 1
        vad = captured[0]
        assert isinstance(vad, VadState)
        # Default ChatLoop values match VadState defaults.
        assert vad.silence_threshold == 0.02
        assert vad.silence_duration == 0.8
        assert vad.min_speech_duration == 0.3

    def test_custom_threshold_passed_through(self):
        captured = []

        original_init = BargeInWatcher.__init__

        def hook(self, *args, **kwargs):
            captured.append(kwargs.get("vad"))
            original_init(self, *args, **kwargs)

        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(concat(
            make_silence(0.3, rate=RATE),
            make_tone_burst(1.0, rate=RATE, amp=0.3),
            make_silence(1.5, rate=RATE),
        ))
        engine, transcribe = _stt()

        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=_yield_tokens("Done."),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_slow_play,
            silence_threshold=0.05,
            silence_duration=0.5,
            min_speech_duration=0.5,
        )

        BargeInWatcher.__init__ = hook  # type: ignore[method-assign]
        try:
            loop.run_one_turn([])
        finally:
            BargeInWatcher.__init__ = original_init  # type: ignore[method-assign]

        assert len(captured) == 1
        vad = captured[0]
        assert isinstance(vad, VadState)
        # User-configured values flow through.
        assert vad.silence_threshold == 0.05
        assert vad.silence_duration == 0.5
        assert vad.min_speech_duration == 0.5


class TestWatcherWithCustomThresholdRejectsBackgroundNoise:
    """Behavioral test: with a high silence_threshold, the watcher
    no longer fires on quiet background-noise audio that the default
    threshold would catch.
    """

    def test_high_threshold_watcher_ignores_quiet_noise(self):
        # Build the watcher directly with a high threshold; push
        # quiet noise; verify no detection.
        mic = VirtualMicStream(rate=16000, chunk_size=1024)
        # Quiet "noise": amp 0.03 → RMS ≈ 0.021. Default threshold
        # 0.02 catches it; threshold 0.05 doesn't.
        mic.push(make_tone_burst(2.0, rate=16000, amp=0.03))

        called = threading.Event()
        watcher = BargeInWatcher(
            mic=mic,
            on_speech_detected=called.set,
            chunk_size=1024,
            rate=16000,
            poll_interval=0.001,
            vad=VadState(silence_threshold=0.05),
        )
        watcher.start()
        time.sleep(0.2)
        watcher.stop(timeout=2.0)

        # High threshold rejected the quiet "noise" — no callback.
        assert called.is_set() is False
        assert watcher.detected is False

    def test_default_threshold_watcher_does_catch_quiet_noise(self):
        # Same audio, default threshold (0.02) — should catch it.
        mic = VirtualMicStream(rate=16000, chunk_size=1024)
        mic.push(make_tone_burst(2.0, rate=16000, amp=0.03))

        called = threading.Event()
        watcher = BargeInWatcher(
            mic=mic,
            on_speech_detected=called.set,
            chunk_size=1024,
            rate=16000,
            poll_interval=0.001,
            vad=VadState(silence_threshold=0.02),
        )
        watcher.start()
        triggered = called.wait(timeout=1.0)
        watcher.stop(timeout=2.0)

        assert triggered is True
        assert watcher.detected is True
