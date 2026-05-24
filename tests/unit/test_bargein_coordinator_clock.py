"""Tests for iter-030 — BargeInCoordinator accepts an injected clock.

Pre-iter-030: ``BargeInCoordinator.trigger()`` stamped
``self.triggered_at = time.monotonic()`` directly. Meanwhile ChatLoop
samples ``llm_stream_done_at = self._clock()``. Production matched
because both happen to be ``time.monotonic`` — but a deterministic
test that injected a fake clock saw ``triggered_at`` in real wall
time and ``llm_stream_done_at`` on the fake clock, making the
``triggered_at < llm_stream_done_at`` phase comparison meaningless.

iter-030 wires a ``clock`` kwarg through. ChatLoop passes its own
clock down. These tests verify the wiring and the phase decision
becomes deterministic under a mocked clock.
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

from examples._chat_loop import ChatLoop  # noqa: E402
from examples._chat_pipeline import BargeInCoordinator  # noqa: E402
from examples._chat_recording import CHUNK, RATE  # noqa: E402
from examples.virtual_audio import (  # noqa: E402
    VirtualMicStream,
    VirtualSpeakerStream,
    concat,
    make_silence,
    make_tone_burst,
)


# ---- Direct tests on the coordinator ---------------------------------------


class TestCoordinatorAcceptsClock:
    def test_default_clock_is_monotonic(self):
        # Backwards compat — without an explicit clock, behavior matches
        # pre-iter-030 (``time.monotonic`` under the hood).
        before = time.monotonic()
        c = BargeInCoordinator()
        c.trigger()
        after = time.monotonic()
        assert c.triggered_at is not None
        assert before <= c.triggered_at <= after

    def test_injected_clock_is_used_for_triggered_at(self):
        # A fake clock returns deterministic values. ``triggered_at``
        # should pull from the fake clock, not the real one.
        fake_now = [100.0]
        c = BargeInCoordinator(clock=lambda: fake_now[0])
        fake_now[0] = 100.5
        c.trigger()
        assert c.triggered_at == 100.5

    def test_idempotent_with_injected_clock(self):
        # Second trigger doesn't overwrite triggered_at even though
        # the clock has advanced. Pre-iter-030 behavior preserved.
        fake_now = [50.0]
        c = BargeInCoordinator(clock=lambda: fake_now[0])
        c.trigger()
        first = c.triggered_at
        fake_now[0] = 999.0
        c.trigger()  # second call — no-op
        assert c.triggered_at == first

    def test_clock_callable_signature_matches_other_components(self):
        # Sanity: same shape as ChatLoop / SentenceWorker / etc. —
        # a zero-arg callable returning float. A counter-style callable
        # should work too.
        ticks = iter([1.0, 2.0, 3.0, 4.0])
        c = BargeInCoordinator(clock=lambda: next(ticks))
        c.trigger()
        # First call inside trigger pulls ``1.0`` from the iterator.
        assert c.triggered_at == 1.0


# ---- Integration: ChatLoop forwards its clock ------------------------------


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


class TestChatLoopForwardsClockToCoord:
    """Hook BargeInCoordinator.__init__ to capture the clock kwarg
    and verify ChatLoop passed its own clock down.
    """

    def test_chatloop_passes_its_clock_to_coordinator(self):
        captured = []

        original_init = BargeInCoordinator.__init__

        def hook(self, *args, **kwargs):
            captured.append(kwargs.get("clock"))
            original_init(self, *args, **kwargs)

        # A unique clock callable so we can identity-compare.
        my_clock_calls = [0]

        def my_clock():
            my_clock_calls[0] += 1
            return float(my_clock_calls[0])

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
            clock=my_clock,
        )

        BargeInCoordinator.__init__ = hook  # type: ignore[method-assign]
        try:
            loop.run_one_turn([])
        finally:
            BargeInCoordinator.__init__ = original_init  # type: ignore[method-assign]

        assert len(captured) == 1
        assert captured[0] is my_clock


class TestPhaseDecisionDeterministicUnderMockedClock:
    """The phase string ("LLM-stream phase" vs "playback phase") is
    chosen by comparing ``coord.triggered_at < llm_stream_done_at``.

    Pre-iter-030, with a mocked clock, this comparison was unreliable
    because triggered_at was real wall-clock time and llm_stream_done_at
    was the mock clock. With iter-030 both come from the same clock,
    so the comparison is deterministic.

    We don't easily get to assert on the phase string from outside
    ChatLoop, but we can construct a coordinator + sample timestamps
    with the same fake clock and verify the comparison goes the
    expected way.
    """

    def test_trigger_before_stream_end_is_lt(self):
        clock_val = [0.0]

        def fake_clock():
            return clock_val[0]

        c = BargeInCoordinator(clock=fake_clock)
        clock_val[0] = 10.0
        c.trigger()  # triggered_at = 10.0
        clock_val[0] = 20.0
        llm_stream_done_at = fake_clock()  # 20.0
        # Phase decision: trigger fired first → LLM-stream phase.
        assert c.triggered_at < llm_stream_done_at

    def test_trigger_after_stream_end_is_gt(self):
        clock_val = [0.0]

        def fake_clock():
            return clock_val[0]

        c = BargeInCoordinator(clock=fake_clock)
        clock_val[0] = 30.0
        llm_stream_done_at = fake_clock()  # 30.0
        clock_val[0] = 40.0
        c.trigger()  # triggered_at = 40.0
        # Phase decision: trigger fired later → playback phase.
        assert c.triggered_at > llm_stream_done_at
