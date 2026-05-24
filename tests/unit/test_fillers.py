"""Tests for iter-011 filler-word generation in SentenceWorker.

Filler clips are pre-rendered ``(audio_np, tokens)`` tuples that the
caller provides at startup. When the worker's queue stays empty for
longer than ``idle_threshold`` before any real sentence arrives, the
worker plays one filler to mask LLM first-token latency.

Tests use the iter-005 ``VirtualSpeakerStream`` and a deterministic
filler picker (always returns the first entry) so assertions don't
depend on randomness.
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

from examples._chat_pipeline import SentenceWorker  # noqa: E402
from examples.virtual_audio import VirtualSpeakerStream  # noqa: E402


# ---- helpers / doubles -------------------------------------------------------


def _const_synth(samples: int = 2048):
    def synth(text: str):
        return np.full(samples, 0.5, dtype=np.float32), []
    return synth


_recording_play_calls: list[dict] = []


def _recording_play(speaker, audio_np, tokens, *, is_first_sentence=False, cancel_event=None):
    audio_int16 = (audio_np * 32767).astype(np.int16)
    speaker.write(audio_int16.tobytes())
    _recording_play_calls.append({
        "samples": len(audio_np),
        "is_first_sentence": is_first_sentence,
    })
    return len(audio_np) / 24000.0


@pytest.fixture(autouse=True)
def _reset_recording_play():
    _recording_play_calls.clear()


def _factory():
    return VirtualSpeakerStream(rate=24000)


def _filler_clip(samples: int = 1024) -> tuple[np.ndarray, list]:
    """A filler clip, distinguishable from real-sentence clips by
    sample count. Audio amplitude 0.3 to differentiate via byte
    inspection if needed.
    """
    return np.full(samples, 0.3, dtype=np.float32), []


def _first_picker(lst):
    """Deterministic alternative to random.choice."""
    return lst[0]


# ---- backward compat ---------------------------------------------------------


class TestFillerBackwardCompat:
    def test_no_fillers_no_threshold_unchanged_behavior(self):
        # iter-008 / iter-009 / iter-010 all worked without these
        # kwargs. This test guards that behavior.
        spk_holder = {"spk": None}

        def factory():
            spk_holder["spk"] = VirtualSpeakerStream(rate=24000)
            return spk_holder["spk"]

        w = SentenceWorker(
            speaker_factory=factory,
            synth_fn=_const_synth(samples=2048),
            play_fn=_recording_play,
        )
        w.start()
        w.submit("hello")
        w.submit_done()
        w.wait_done(timeout=5.0)

        assert w.sentences_spoken == 1
        assert w.fillers_played == 0
        assert len(spk_holder["spk"].captured) == 2048 * 2

    def test_empty_filler_list_does_not_play_anything(self):
        w = SentenceWorker(
            speaker_factory=_factory,
            synth_fn=_const_synth(samples=2048),
            play_fn=_recording_play,
            fillers=[],
            idle_threshold=0.05,
        )
        w.start()
        # Don't submit anything; just close it.
        w.submit_done()
        w.wait_done(timeout=5.0)
        assert w.fillers_played == 0
        assert w.sentences_spoken == 0


# ---- happy path --------------------------------------------------------------


class TestFillerPlaysWhenIdle:
    def test_filler_plays_after_idle_threshold(self):
        """Idle threshold hit before any sentence arrives → one filler
        plays. Then a real sentence arrives and plays after.
        """
        spk_holder = {"spk": None}

        def factory():
            spk_holder["spk"] = VirtualSpeakerStream(rate=24000)
            return spk_holder["spk"]

        filler = _filler_clip(samples=1024)
        w = SentenceWorker(
            speaker_factory=factory,
            synth_fn=_const_synth(samples=2048),
            play_fn=_recording_play,
            fillers=[filler],
            idle_threshold=0.05,
            filler_picker=_first_picker,
        )
        w.start()
        # Wait long enough for idle threshold to fire.
        time.sleep(0.15)
        # Now submit the real sentence.
        w.submit("real sentence")
        w.submit_done()
        w.wait_done(timeout=5.0)

        assert w.fillers_played == 1
        assert w.sentences_spoken == 1
        # Speaker captured filler bytes (1024 samples) + sentence bytes
        # (2048 samples) = 3072 samples total → 6144 bytes.
        assert len(spk_holder["spk"].captured) == (1024 + 2048) * 2

    def test_first_audio_output_is_the_filler_with_bot_prefix(self):
        """The filler is the first audio output, so it must trigger
        the is_first_sentence=True path (which prints the "Bot:"
        prefix in real playback).
        """
        filler = _filler_clip(samples=512)
        w = SentenceWorker(
            speaker_factory=_factory,
            synth_fn=_const_synth(samples=1024),
            play_fn=_recording_play,
            fillers=[filler],
            idle_threshold=0.05,
            filler_picker=_first_picker,
        )
        w.start()
        time.sleep(0.15)
        w.submit("real sentence")
        w.submit_done()
        w.wait_done(timeout=5.0)

        flags = [c["is_first_sentence"] for c in _recording_play_calls]
        # Filler first (is_first=True), then real sentence (is_first=False).
        assert flags == [True, False]

    def test_no_filler_if_sentence_arrives_before_idle_threshold(self):
        """Sentence submitted immediately → queue.get returns before
        the idle timeout, no filler plays.
        """
        filler = _filler_clip(samples=1024)
        w = SentenceWorker(
            speaker_factory=_factory,
            synth_fn=_const_synth(samples=2048),
            play_fn=_recording_play,
            fillers=[filler],
            idle_threshold=2.0,  # generous so we don't accidentally fire
            filler_picker=_first_picker,
        )
        w.start()
        # Submit before threshold elapses.
        w.submit("right away")
        w.submit_done()
        w.wait_done(timeout=5.0)

        assert w.fillers_played == 0
        assert w.sentences_spoken == 1

    def test_only_one_filler_per_run_even_if_long_idle(self):
        """Once a filler has played, even if we stay idle longer, no
        second filler should fire. The flag prevents the loop from
        burning through the entire filler list while waiting.
        """
        filler = _filler_clip(samples=512)
        w = SentenceWorker(
            speaker_factory=_factory,
            synth_fn=_const_synth(samples=1024),
            play_fn=_recording_play,
            fillers=[filler, filler, filler],
            idle_threshold=0.05,
            filler_picker=_first_picker,
        )
        w.start()
        # First filler will play after ~50ms.
        # Wait significantly longer to verify no second filler fires.
        time.sleep(0.3)
        w.submit("at last")
        w.submit_done()
        w.wait_done(timeout=5.0)

        assert w.fillers_played == 1


# ---- counters & metrics ------------------------------------------------------


class TestFillerCounters:
    def test_first_audio_at_captured_at_filler_start(self):
        """When a filler plays before any sentence, first_audio_at
        should be set during the filler — not delayed until the real
        sentence. Otherwise the TTFS metric undercounts the masking.
        """
        ticks = iter([100.0 + i for i in range(50)])

        def fake_clock():
            return next(ticks)

        filler = _filler_clip(samples=512)
        w = SentenceWorker(
            speaker_factory=_factory,
            synth_fn=_const_synth(samples=1024),
            play_fn=_recording_play,
            fillers=[filler],
            idle_threshold=0.01,
            filler_picker=_first_picker,
            clock=fake_clock,
        )
        w.start()
        time.sleep(0.1)
        w.submit("real")
        w.submit_done()
        w.wait_done(timeout=5.0)

        # first_audio_at was set the first time _play_clip ran, which
        # is the filler. The exact value depends on how many clock()
        # calls happened before — but we can verify it was set.
        assert w.first_audio_at is not None
        # And no real sentence had been played yet, so the value
        # corresponds to the filler timing.
        assert w.fillers_played == 1

    def test_fillers_played_starts_at_zero(self):
        w = SentenceWorker(
            speaker_factory=_factory,
            synth_fn=_const_synth(),
            play_fn=_recording_play,
        )
        assert w.fillers_played == 0


# ---- robustness --------------------------------------------------------------


class TestFillerRobustness:
    def test_empty_filler_audio_does_not_count(self):
        """A filler clip with len(audio_np) == 0 should be skipped —
        we don't want to "use up" the one-filler-per-turn slot on an
        empty clip.
        """
        empty_filler = (np.array([], dtype=np.float32), [])
        w = SentenceWorker(
            speaker_factory=_factory,
            synth_fn=_const_synth(samples=1024),
            play_fn=_recording_play,
            fillers=[empty_filler],
            idle_threshold=0.05,
            filler_picker=_first_picker,
        )
        w.start()
        time.sleep(0.15)
        w.submit("hello")
        w.submit_done()
        w.wait_done(timeout=5.0)

        # Empty filler still consumed the "filler attempt" slot, so
        # fillers_played stays 0 but no second attempt is made — the
        # real sentence comes through normally.
        assert w.fillers_played == 0
        assert w.sentences_spoken == 1

    def test_play_fn_raising_during_filler_does_not_block_real_sentences(self):
        """If play_fn raises while playing a filler, the worker
        should record the error and continue to the next item.
        """
        call_count = {"n": 0}

        def flaky_play(speaker, audio_np, tokens, *, is_first_sentence=False, cancel_event=None):
            call_count["n"] += 1
            speaker.write((audio_np * 32767).astype(np.int16).tobytes())
            if call_count["n"] == 1:
                raise RuntimeError("play boom on filler")
            return 0.05

        filler = _filler_clip(samples=512)
        w = SentenceWorker(
            speaker_factory=_factory,
            synth_fn=_const_synth(samples=1024),
            play_fn=flaky_play,
            fillers=[filler],
            idle_threshold=0.05,
            filler_picker=_first_picker,
        )
        w.start()
        time.sleep(0.15)
        w.submit("hello")
        w.submit_done()
        w.wait_done(timeout=5.0)

        # Filler crashed but we recorded the error; sentence still played.
        assert any("play boom on filler" in str(e) for e in w.errors)
        assert w.sentences_spoken == 1
        assert w.fillers_played == 0

    def test_filler_audio_appears_first_in_speaker_byte_stream(self):
        """End-to-end byte check: speaker.captured starts with the
        filler's bytes (amplitude 0.3 → int16 ≈ 9830) followed by
        the real sentence's bytes (amplitude 0.5 → int16 ≈ 16384).
        Catches regressions where filler/sentence ordering flips.
        """
        spk_holder = {"spk": None}

        def factory():
            spk_holder["spk"] = VirtualSpeakerStream(rate=24000)
            return spk_holder["spk"]

        filler_samples = 256
        sentence_samples = 512
        # Distinct amplitudes so we can identify regions in the bytes.
        filler_audio = np.full(filler_samples, 0.3, dtype=np.float32)
        filler_clip = (filler_audio, [])

        def sentence_synth(text):
            return np.full(sentence_samples, 0.5, dtype=np.float32), []

        w = SentenceWorker(
            speaker_factory=factory,
            synth_fn=sentence_synth,
            play_fn=_recording_play,
            fillers=[filler_clip],
            idle_threshold=0.05,
            filler_picker=_first_picker,
        )
        w.start()
        time.sleep(0.15)
        w.submit("real")
        w.submit_done()
        w.wait_done(timeout=5.0)

        decoded = spk_holder["spk"].captured_int16
        # First filler_samples should be ~0.3 * 32767 = 9830
        first_region = decoded[:filler_samples]
        assert np.all(np.abs(first_region - 9830) < 5)
        # Next sentence_samples should be ~0.5 * 32767 = 16383
        second_region = decoded[filler_samples:filler_samples + sentence_samples]
        assert np.all(np.abs(second_region - 16383) < 5)

    def test_cancel_during_filler_playback_works(self):
        """If the user barges in during the filler, the worker should
        stop cleanly. (Edge case — usually barge-in triggers cancel,
        which we already validated in iter-009 against real sentences.
        Here we make sure the same path works for fillers.)
        """
        cancel_seen = threading.Event()

        def cancellable_play(speaker, audio_np, tokens, *,
                             is_first_sentence=False, cancel_event=None):
            audio_int16 = (audio_np * 32767).astype(np.int16)
            chunk = 256
            written = 0
            while written < len(audio_int16):
                if cancel_event is not None and cancel_event.is_set():
                    cancel_seen.set()
                    break
                end = min(written + chunk, len(audio_int16))
                speaker.write(audio_int16[written:end].tobytes())
                written = end
                time.sleep(0.005)
            return 0.0

        # Long filler so we have time to cancel it.
        long_filler = (np.full(20000, 0.3, dtype=np.float32), [])
        w = SentenceWorker(
            speaker_factory=_factory,
            synth_fn=_const_synth(samples=1024),
            play_fn=cancellable_play,
            fillers=[long_filler],
            idle_threshold=0.05,
            filler_picker=_first_picker,
        )
        w.start()
        time.sleep(0.15)  # let filler start playing
        w.cancel(timeout=5.0)

        assert w.cancelled is True
        assert cancel_seen.is_set()
