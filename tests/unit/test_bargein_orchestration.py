"""Full-loop orchestration test for iter-010 barge-in wiring.

This is the closest we can get on x86_64 Linux to exercising what
``mic_chat.run_chat`` actually does during a barge-in:

  1. The bot is speaking — SentenceWorker plays sentences via a
     persistent VirtualSpeakerStream.
  2. A BargeInWatcher listens on a VirtualMicStream while the bot
     plays. The watcher's ``on_speech_detected`` is wired to
     ``worker.cancel`` so that user speech interrupts playback
     mid-sentence.
  3. After the worker exits, the watcher's captured frames are
     fed as ``primed_frames`` to ``record_utterance_streaming``,
     which closes the loop and produces a wav covering the user's
     barge-in audio.

The components stay completely virtual — no pyaudio, no kokoro
inference at the LLM level, no real wall clock. The only blocking
operation is the worker thread itself; tests bound it with
``wait_done(timeout=...)``.
"""

from __future__ import annotations

import io
import sys
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_pipeline import BargeInWatcher, SentenceWorker  # noqa: E402
from examples._chat_playback import TTS_RATE  # noqa: E402
from examples._chat_recording import (  # noqa: E402
    CHUNK,
    RATE,
    record_utterance_streaming,
)
from examples.virtual_audio import (  # noqa: E402
    VirtualMicStream,
    VirtualSpeakerStream,
    concat,
    make_silence,
    make_tone_burst,
)


class FrameClock:
    """Same clock as the iter-006 recording tests — virtual time
    advancing at audio rate per call."""

    def __init__(self, chunk: int = CHUNK, rate: int = RATE):
        self._dt = chunk / rate
        self._t = 0.0

    def __call__(self) -> float:
        t = self._t
        self._t += self._dt
        return t


def _stub_engine() -> SimpleNamespace:
    return SimpleNamespace(_last_text=None, model_repo="stub")


def _const_synth(samples: int = 30000):
    """Synth that returns a long enough audio so the worker has time
    to be cancelled mid-stream during the barge-in test.
    """
    def synth(text: str):
        return np.full(samples, 0.5, dtype=np.float32), []
    return synth


def _cancellable_play(speaker, audio_np, tokens, *,
                     is_first_sentence=False, cancel_event=None):
    """Play_fn that respects cancel_event between chunks. Sleeps
    briefly per chunk so the watcher thread has time to flip the
    flag — real PyAudio playback blocks at audio rate, so this
    matches reality.
    """
    audio_int16 = (audio_np * 32767).astype(np.int16)
    chunk = 1024
    written = 0
    t0 = time.monotonic()
    while written < len(audio_int16):
        if cancel_event is not None and cancel_event.is_set():
            break
        end = min(written + chunk, len(audio_int16))
        speaker.write(audio_int16[written:end].tobytes())
        written = end
        time.sleep(0.005)
    return time.monotonic() - t0


def test_full_loop_bot_speaks_user_barges_in_record_replays_capture():
    """End-to-end barge-in flow:

      - Bot is speaking via SentenceWorker.
      - User starts speaking on the mic.
      - BargeInWatcher detects, fires worker.cancel.
      - record_utterance_streaming runs with primed_frames =
        watcher.frames and produces a valid wav covering the
        user's barge-in audio.
    """
    # ---- Phase 1: bot is speaking ----
    spk_holder = {"spk": None}

    def speaker_factory():
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        spk_holder["spk"] = spk
        return spk

    worker = SentenceWorker(
        speaker_factory=speaker_factory,
        synth_fn=_const_synth(samples=30000),  # ~30 chunks
        play_fn=_cancellable_play,
    )

    mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
    watcher = BargeInWatcher(
        mic=mic,
        on_speech_detected=lambda: worker.cancel(timeout=5.0),
        chunk_size=CHUNK,
        rate=RATE,
        poll_interval=0.001,
    )

    worker.start()
    watcher.start()

    worker.submit("first long bot sentence")
    worker.submit("second sentence (will be dropped)")
    worker.submit("third sentence (will be dropped)")

    # Let the worker actually start the first sentence.
    time.sleep(0.05)

    # User starts talking — push speech to the mic.
    user_speech = make_tone_burst(0.8, rate=RATE, amp=0.3)
    mic.push(user_speech)

    # Worker should be cancelled by the watcher; wait for it to exit.
    worker.wait_done(timeout=5.0)
    watcher.stop(timeout=2.0)

    assert worker.cancelled is True
    assert watcher.detected is True
    # Speaker received some audio but not all of the first sentence.
    full_first_bytes = 30000 * 2
    assert 0 < len(spk_holder["spk"].captured) < full_first_bytes
    # Pending sentences were dropped.
    assert worker.sentences_spoken <= 1
    # Watcher captured at least one frame of user audio for replay.
    assert len(watcher.frames) > 0

    # ---- Phase 2: feed captured frames into next record ----
    # Push more silence so the recording loop's VAD eventually closes.
    mic.push(make_silence(1.5, rate=RATE))
    engine = _stub_engine()

    primed = list(watcher.frames)
    wav, dur, _ = record_utterance_streaming(
        mic,
        engine,
        transcribe_fn=lambda w: "user-barge-in-text",
        clock=FrameClock(),
        output=io.StringIO(),
        primed_frames=primed,
    )

    # The wav should be non-empty and parseable.
    assert len(wav) > 0
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == RATE
        recorded_n_frames = wf.getnframes()
    # Recorded audio should be at least roughly the size of the
    # primed frames (the user's barge-in audio is preserved).
    assert recorded_n_frames >= len(primed) * CHUNK - CHUNK
    assert dur > 0.0
    # Engine got the canned transcript.
    assert engine._last_text == "user-barge-in-text"


def test_no_barge_in_means_no_priming_next_turn():
    """If the watcher never detects speech, no primed_frames are
    carried forward. The next record_utterance call should behave
    as if nothing happened.
    """
    spk_holder = {"spk": None}

    def speaker_factory():
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        spk_holder["spk"] = spk
        return spk

    worker = SentenceWorker(
        speaker_factory=speaker_factory,
        synth_fn=_const_synth(samples=4096),
        play_fn=_cancellable_play,
    )

    mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
    # No user speech pushed — mic stays silent.
    watcher = BargeInWatcher(
        mic=mic,
        on_speech_detected=lambda: worker.cancel(timeout=5.0),
        chunk_size=CHUNK,
        rate=RATE,
        poll_interval=0.001,
    )

    worker.start()
    watcher.start()
    worker.submit("just one short sentence")
    worker.submit_done()
    worker.wait_done(timeout=5.0)
    watcher.stop(timeout=2.0)

    assert worker.cancelled is False
    assert watcher.detected is False
    # Whole sentence played.
    assert len(spk_holder["spk"].captured) == 4096 * 2
    assert worker.sentences_spoken == 1


def test_barge_in_during_first_sentence_drops_remaining_queue():
    """Worker has 5 sentences queued; user barges in during the
    first. The remaining 4 should be dropped; the speaker should
    contain only a partial first sentence's worth of audio.
    """
    spk_holder = {"spk": None}

    def speaker_factory():
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        spk_holder["spk"] = spk
        return spk

    samples = 50000  # very long sentence so cancel happens early
    worker = SentenceWorker(
        speaker_factory=speaker_factory,
        synth_fn=_const_synth(samples=samples),
        play_fn=_cancellable_play,
    )
    mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
    watcher = BargeInWatcher(
        mic=mic,
        on_speech_detected=lambda: worker.cancel(timeout=5.0),
        chunk_size=CHUNK,
        rate=RATE,
        poll_interval=0.001,
    )

    worker.start()
    watcher.start()

    for i in range(5):
        worker.submit(f"sentence {i}")

    time.sleep(0.05)
    # User starts talking.
    mic.push(make_tone_burst(0.5, rate=RATE, amp=0.3))

    worker.wait_done(timeout=5.0)
    watcher.stop(timeout=2.0)

    assert watcher.detected is True
    assert worker.cancelled is True
    # Less than even one full sentence was played.
    full_first = samples * 2
    assert 0 < len(spk_holder["spk"].captured) < full_first
    assert worker.sentences_spoken <= 1
