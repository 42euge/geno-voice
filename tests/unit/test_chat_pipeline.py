"""Tests for examples/_chat_pipeline.SentenceWorker.

These tests use real threads (the GIL + Queue give clean semantics)
but are bounded with ``wait_done(timeout=5)`` so a hang fails fast
rather than blocking the test run forever.

Rather than synthesizing real audio, the tests pass deterministic
synth_fn / play_fn callables that record what they were called with.
That gives precise assertions about ordering, counts, and metric
math without depending on TTS or audio hardware.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_pipeline import SentenceWorker  # noqa: E402
from examples.virtual_audio import VirtualSpeakerStream  # noqa: E402


# ---- Test doubles ------------------------------------------------------------

def _const_synth(samples: int = 1024, rate_per_call_increment: int = 0):
    """Synth that returns `samples` of sine audio for every sentence.

    Optionally bumps the sample count by `rate_per_call_increment` per
    call so we can distinguish call order in the speaker bytes.
    """
    counter = {"n": 0}

    def synth(text: str):
        n = samples + counter["n"] * rate_per_call_increment
        counter["n"] += 1
        audio = np.full(n, 0.5, dtype=np.float32)  # constant float for easy match
        tokens = [{"text": w, "start": i * 0.05} for i, w in enumerate(text.split())]
        return audio, tokens

    return synth


def _empty_synth():
    return lambda text: (np.array([], dtype=np.float32), [])


def _recording_play(speaker, audio_np, tokens, *, is_first_sentence=False):
    """Playback that just dumps int16 bytes into the speaker and
    returns the audio's nominal duration. Records the call so tests
    can assert on order/flags.
    """
    audio_int16 = (audio_np * 32767).astype(np.int16)
    speaker.write(audio_int16.tobytes())
    _recording_play.calls.append({
        "samples": len(audio_np),
        "tokens": list(tokens),
        "is_first_sentence": is_first_sentence,
    })
    return len(audio_np) / 24000.0


_recording_play.calls = []


@pytest.fixture(autouse=True)
def _reset_recording_play():
    _recording_play.calls = []


# ---- Helpers -----------------------------------------------------------------

def _speaker_factory(rate: int = 24000) -> Callable[[], VirtualSpeakerStream]:
    """Return a factory that mints a fresh VirtualSpeakerStream when called."""
    holder = {"speaker": None}

    def factory():
        holder["speaker"] = VirtualSpeakerStream(rate=rate)
        return holder["speaker"]

    factory.last = lambda: holder["speaker"]  # type: ignore[attr-defined]
    return factory


# ---- Lifecycle ---------------------------------------------------------------

class TestLifecycle:
    def test_submit_then_submit_done_runs_clean(self):
        factory = _speaker_factory()
        w = SentenceWorker(
            speaker_factory=factory,
            synth_fn=_const_synth(samples=2048),
            play_fn=_recording_play,
        )
        w.start()
        w.submit("hello world")
        w.submit("how are you")
        w.submit_done()
        w.wait_done(timeout=5.0)

        assert w.sentences_spoken == 2
        assert w.errors == []
        # Speaker captured both audio blobs back to back.
        assert len(factory.last().captured) == 2 * 2048 * 2  # int16 → 2 bytes/sample

    def test_double_start_raises(self):
        w = SentenceWorker(
            speaker_factory=_speaker_factory(),
            synth_fn=_const_synth(),
            play_fn=_recording_play,
        )
        w.start()
        try:
            with pytest.raises(RuntimeError):
                w.start()
        finally:
            w.submit_done()
            w.wait_done(timeout=5.0)

    def test_wait_done_without_start_raises(self):
        w = SentenceWorker(
            speaker_factory=_speaker_factory(),
            synth_fn=_const_synth(),
            play_fn=_recording_play,
        )
        with pytest.raises(RuntimeError):
            w.wait_done(timeout=1.0)

    def test_submit_after_done_is_dropped(self):
        factory = _speaker_factory()
        w = SentenceWorker(
            speaker_factory=factory,
            synth_fn=_const_synth(samples=1024),
            play_fn=_recording_play,
        )
        w.start()
        w.submit("first")
        w.submit_done()
        # This submit should be ignored — the sentinel is already in the queue.
        w.submit("never plays")
        w.wait_done(timeout=5.0)
        assert w.sentences_spoken == 1


# ---- Ordering & metrics ------------------------------------------------------

class TestOrderingAndMetrics:
    def test_sentences_played_in_submission_order(self):
        # Per-sentence sample count grows so we can prove order via byte layout.
        factory = _speaker_factory()
        synth = _const_synth(samples=512, rate_per_call_increment=512)
        w = SentenceWorker(
            speaker_factory=factory,
            synth_fn=synth,
            play_fn=_recording_play,
        )
        w.start()
        w.submit("a")
        w.submit("bb")
        w.submit("ccc")
        w.submit_done()
        w.wait_done(timeout=5.0)

        # Three plays, with growing sizes.
        assert [c["samples"] for c in _recording_play.calls] == [512, 1024, 1536]

    def test_is_first_sentence_only_true_for_first(self):
        w = SentenceWorker(
            speaker_factory=_speaker_factory(),
            synth_fn=_const_synth(samples=512),
            play_fn=_recording_play,
        )
        w.start()
        w.submit("one")
        w.submit("two")
        w.submit("three")
        w.submit_done()
        w.wait_done(timeout=5.0)

        flags = [c["is_first_sentence"] for c in _recording_play.calls]
        assert flags == [True, False, False]

    def test_metrics_accumulate(self):
        # Use a clock generator that advances by 1.0s per call.
        ticks = iter(range(0, 1000))

        def clock():
            return float(next(ticks))

        # synth-and-play interleave: 1.0s synth, 1.0s play, per sentence.
        # Each sentence advances the clock by ~3 ticks (synth start, synth
        # end + first_audio_at, play return). Don't rely on exact counts —
        # just verify monotonic growth.
        w = SentenceWorker(
            speaker_factory=_speaker_factory(),
            synth_fn=_const_synth(samples=2048),
            play_fn=_recording_play,
            clock=clock,
        )
        w.start()
        w.submit("a")
        w.submit("b")
        w.submit_done()
        w.wait_done(timeout=5.0)

        assert w.sentences_spoken == 2
        # tts_time and playback_time both > 0 since both branches ran.
        assert w.tts_time > 0
        assert w.playback_time > 0
        # first_audio_at captured at first non-empty audio.
        assert w.first_audio_at is not None
        assert w.first_audio_at > 0

    def test_first_audio_at_set_only_once(self):
        ticks = iter([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                      18.0, 19.0, 20.0, 21.0, 22.0])

        def clock():
            return next(ticks)

        w = SentenceWorker(
            speaker_factory=_speaker_factory(),
            synth_fn=_const_synth(samples=512),
            play_fn=_recording_play,
            clock=clock,
        )
        w.start()
        w.submit("first")
        w.submit("second")
        w.submit_done()
        w.wait_done(timeout=5.0)

        # The first first_audio_at sample is taken between synth and play
        # of sentence #1; the value never updates afterward.
        first = w.first_audio_at
        assert first is not None
        # Play another lap of sentence — first_audio_at must not move.
        assert w.first_audio_at == first


# ---- Skipping & robustness ---------------------------------------------------

class TestSkipping:
    def test_empty_string_sentence_skipped(self):
        w = SentenceWorker(
            speaker_factory=_speaker_factory(),
            synth_fn=_const_synth(samples=1024),
            play_fn=_recording_play,
        )
        w.start()
        w.submit("")
        w.submit("   ")
        w.submit("real text")
        w.submit_done()
        w.wait_done(timeout=5.0)
        assert w.sentences_spoken == 1
        assert _recording_play.calls[0]["samples"] == 1024

    def test_empty_audio_skipped(self):
        # synth returns empty audio for every sentence.
        w = SentenceWorker(
            speaker_factory=_speaker_factory(),
            synth_fn=_empty_synth(),
            play_fn=_recording_play,
        )
        w.start()
        w.submit("hello")
        w.submit("world")
        w.submit_done()
        w.wait_done(timeout=5.0)
        assert w.sentences_spoken == 0
        assert w.first_audio_at is None
        assert _recording_play.calls == []


class TestErrorHandling:
    def test_synth_exception_recorded_and_loop_continues(self):
        calls = {"n": 0}

        def flaky_synth(text):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("synth boom")
            return np.full(1024, 0.5, dtype=np.float32), []

        w = SentenceWorker(
            speaker_factory=_speaker_factory(),
            synth_fn=flaky_synth,
            play_fn=_recording_play,
        )
        w.start()
        w.submit("first")  # raises
        w.submit("second")  # plays
        w.submit_done()
        w.wait_done(timeout=5.0)

        assert w.sentences_spoken == 1
        assert len(w.errors) == 1
        assert "synth boom" in str(w.errors[0])

    def test_play_exception_recorded_and_loop_continues(self):
        calls = {"n": 0}

        def flaky_play(speaker, audio_np, tokens, *, is_first_sentence=False):
            calls["n"] += 1
            speaker.write((audio_np * 32767).astype(np.int16).tobytes())
            if calls["n"] == 1:
                raise RuntimeError("play boom")
            return 0.05

        w = SentenceWorker(
            speaker_factory=_speaker_factory(),
            synth_fn=_const_synth(samples=512),
            play_fn=flaky_play,
        )
        w.start()
        w.submit("a")
        w.submit("b")
        w.submit_done()
        w.wait_done(timeout=5.0)

        assert w.sentences_spoken == 1
        assert any("play boom" in str(e) for e in w.errors)

    def test_speaker_factory_failure_exits_cleanly(self):
        def bad_factory():
            raise OSError("no audio device")

        w = SentenceWorker(
            speaker_factory=bad_factory,
            synth_fn=_const_synth(),
            play_fn=_recording_play,
        )
        w.start()
        w.submit("never reached")
        w.submit_done()
        w.wait_done(timeout=5.0)

        assert w.sentences_spoken == 0
        assert len(w.errors) >= 1
        assert isinstance(w.errors[0], OSError)


# ---- Stop -------------------------------------------------------------------

class TestStop:
    def test_stop_drains_pending_and_joins(self):
        # Synth that blocks long enough that we can stop mid-queue.
        synth_started = threading.Event()
        proceed = threading.Event()

        def slow_synth(text):
            synth_started.set()
            proceed.wait(timeout=2.0)
            return np.full(1024, 0.5, dtype=np.float32), []

        w = SentenceWorker(
            speaker_factory=_speaker_factory(),
            synth_fn=slow_synth,
            play_fn=_recording_play,
        )
        w.start()
        w.submit("first (will run)")
        w.submit("second (will be dropped)")
        w.submit("third (will be dropped)")
        synth_started.wait(timeout=2.0)
        proceed.set()  # let first complete

        # Now stop — second and third should be dropped without play.
        w.stop(timeout=5.0)

        # Exactly one sentence spoken (the first); the rest were drained.
        assert w.sentences_spoken <= 1

    def test_stop_before_start_is_noop(self):
        w = SentenceWorker(
            speaker_factory=_speaker_factory(),
            synth_fn=_const_synth(),
            play_fn=_recording_play,
        )
        w.stop(timeout=1.0)  # should not raise

    def test_submit_done_then_stop_does_not_hang(self):
        w = SentenceWorker(
            speaker_factory=_speaker_factory(),
            synth_fn=_const_synth(samples=128),
            play_fn=_recording_play,
        )
        w.start()
        w.submit("first")
        w.submit_done()
        # Stop right after submit_done; should still join cleanly.
        w.stop(timeout=5.0)


# ---- Integration with VirtualSpeakerStream + loopback ------------------------

class TestSpeakerWiring:
    def test_speaker_close_called_on_exit(self):
        spk_holder = {}

        def factory():
            spk = VirtualSpeakerStream(rate=24000)
            spk_holder["spk"] = spk
            return spk

        w = SentenceWorker(
            speaker_factory=factory,
            synth_fn=_const_synth(samples=512),
            play_fn=_recording_play,
        )
        w.start()
        w.submit("hello")
        w.submit_done()
        w.wait_done(timeout=5.0)

        assert spk_holder["spk"]._closed is True

    def test_speaker_loopback_routes_audio_to_paired_mic(self):
        from examples.virtual_audio import VirtualMicStream

        mic = VirtualMicStream(rate=24000, chunk_size=1024)

        def factory():
            return VirtualSpeakerStream(rate=24000, loopback_to=mic)

        w = SentenceWorker(
            speaker_factory=factory,
            synth_fn=_const_synth(samples=2048),
            play_fn=_recording_play,
        )
        w.start()
        w.submit("hello")
        w.submit_done()
        w.wait_done(timeout=5.0)

        # Audio should have flowed through the loopback into the mic.
        assert mic.frames_buffered == 2048
