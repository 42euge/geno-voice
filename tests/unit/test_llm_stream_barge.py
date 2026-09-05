"""Orchestration tests for iter-012 — barge-in during LLM streaming.

The chat loop's for-token block is exposed indirectly: we replicate
its essential structure (token receipt + coord check + sentence
submit) in a small helper, then drive that helper with a faked LLM
token iterator. The test asserts the loop exits early when a
``BargeInCoordinator`` is triggered mid-stream and that the worker
gets cancelled.

This is the closest we can get to testing
``examples/mic_chat.run_chat`` without spinning up a real
OpenAI-compatible LLM server. The components themselves
(``BargeInCoordinator``, ``BargeInWatcher``, ``SentenceWorker``)
have full unit coverage already — this just verifies the
orchestration shape.
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

from examples._chat_helpers import split_complete_sentences  # noqa: E402
from examples._chat_pipeline import (  # noqa: E402
    BargeInCoordinator,
    BargeInWatcher,
    SentenceWorker,
)
from examples._chat_playback import TTS_RATE  # noqa: E402
from examples.virtual_audio import (  # noqa: E402
    VirtualMicStream,
    VirtualSpeakerStream,
    make_silence,
    make_tone_burst,
)


# ---- Faked LLM token stream -------------------------------------------------


def _yield_tokens(text: str, *, per_token_delay: float = 0.0):
    """Generator that yields tokens one at a time. Optional delay
    between tokens so a watcher thread has time to fire mid-stream.
    """
    # Naive: split on whitespace + punctuation, keep delimiters so
    # the resulting buffer has periods.
    import re
    parts = re.findall(r"\S+|\.|!|\?", text)
    for i, p in enumerate(parts):
        if per_token_delay > 0:
            time.sleep(per_token_delay)
        # Emit each token with a trailing space so split_complete_sentences
        # can detect sentence ends.
        yield p + " "


def _consume_with_coord(token_iter, *, coord, on_token):
    """Replicates the essential shape of mic_chat.run_chat's for-token
    loop: yields tokens through, calls on_token(complete_sentence) on
    each completed sentence, breaks early if coord.is_set().
    Returns (interrupted, full_response).
    """
    buffer = ""
    full = ""
    interrupted = False
    for token in token_iter:
        if coord.is_set():
            interrupted = True
            break
        buffer += token
        full += token
        complete, buffer = split_complete_sentences(buffer)
        for s in complete:
            on_token(s)
    return interrupted, full


# ---- Helpers / doubles -------------------------------------------------------


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


# ---- Tests ------------------------------------------------------------------


class TestForTokenLoopEarlyExit:
    """Verifies the consumer loop respects coord.is_set()."""

    def test_no_barge_in_consumes_all_tokens(self):
        coord = BargeInCoordinator()
        seen: list[str] = []
        interrupted, full = _consume_with_coord(
            _yield_tokens("Hello world. How are you?"),
            coord=coord,
            on_token=seen.append,
        )
        assert interrupted is False
        assert "Hello" in full
        assert len(seen) >= 1

    def test_coord_already_set_yields_zero_tokens(self):
        coord = BargeInCoordinator()
        coord.trigger()
        seen: list[str] = []
        interrupted, _ = _consume_with_coord(
            _yield_tokens("never reached"),
            coord=coord,
            on_token=seen.append,
        )
        assert interrupted is True
        assert seen == []

    def test_coord_set_mid_stream_breaks_early(self):
        coord = BargeInCoordinator()
        seen: list[str] = []

        def trigger_after_two(token_iter):
            n = 0
            for t in token_iter:
                yield t
                n += 1
                if n == 2:
                    coord.trigger()

        full_text = "First sentence. Second sentence. Third sentence."
        interrupted, _ = _consume_with_coord(
            trigger_after_two(_yield_tokens(full_text)),
            coord=coord,
            on_token=seen.append,
        )
        assert interrupted is True
        # We MAY have submitted at most one complete sentence
        # before the trigger fired. The exact count depends on
        # how tokens map to sentences, but it's bounded.
        assert len(seen) <= 1


class TestFullBargeInDuringLlmStream:
    """End-to-end: watcher running, faked LLM stream emits tokens
    slowly, user speaks on virtual mic, watcher fires coord, the
    consumer loop and the worker both stop.
    """

    def test_user_speech_during_llm_stream_cancels_everything(self):
        spk_holder = {"spk": None}

        def factory():
            spk_holder["spk"] = VirtualSpeakerStream(rate=TTS_RATE)
            return spk_holder["spk"]

        worker = SentenceWorker(
            speaker_factory=factory,
            synth_fn=_const_synth(samples=4096),
            play_fn=_slow_play,
        )
        worker.start()

        mic = VirtualMicStream(rate=16000, chunk_size=1024)
        coord = BargeInCoordinator(worker=worker)
        watcher = BargeInWatcher(
            mic=mic,
            on_speech_detected=coord.trigger,
            chunk_size=1024,
            rate=16000,
            poll_interval=0.001,
        )
        watcher.start()

        # Schedule user speech to arrive a bit after the loop starts.
        def push_user_speech():
            time.sleep(0.05)
            mic.push(make_tone_burst(0.5, rate=16000, amp=0.3))

        threading.Thread(target=push_user_speech, daemon=True).start()

        # Faked LLM stream: long bot reply, slow per-token to give
        # the user time to barge in.
        seen: list[str] = []
        try:
            interrupted, full = _consume_with_coord(
                _yield_tokens(
                    "First long sentence. Second long sentence. "
                    "Third long sentence. Fourth long sentence.",
                    per_token_delay=0.02,
                ),
                coord=coord,
                on_token=lambda s: (seen.append(s), worker.submit(s)),
            )
        finally:
            # Whether the barge-in fired or the stream ran out, wait
            # cleanly; coord.trigger has already cancelled the worker.
            worker.wait_done(timeout=5.0)
            watcher.stop(timeout=2.0)

        assert interrupted is True
        assert coord.is_set() is True
        assert worker.cancelled is True
        assert watcher.detected is True
        # Watcher captured frames the user spoke; ready for replay.
        assert len(watcher.frames) > 0


class TestNoBargeInPlaysAllSentences:
    """Sanity check: no user speech, the whole faked stream plays."""

    def test_clean_completion(self):
        spk_holder = {"spk": None}

        def factory():
            spk_holder["spk"] = VirtualSpeakerStream(rate=TTS_RATE)
            return spk_holder["spk"]

        worker = SentenceWorker(
            speaker_factory=factory,
            synth_fn=_const_synth(samples=1024),
            play_fn=_slow_play,
        )
        worker.start()

        mic = VirtualMicStream(rate=16000, chunk_size=1024)
        # No user audio pushed.
        coord = BargeInCoordinator(worker=worker)
        watcher = BargeInWatcher(
            mic=mic,
            on_speech_detected=coord.trigger,
            chunk_size=1024,
            rate=16000,
            poll_interval=0.001,
        )
        watcher.start()

        seen: list[str] = []
        interrupted, _ = _consume_with_coord(
            _yield_tokens("Short reply.", per_token_delay=0.01),
            coord=coord,
            on_token=lambda s: (seen.append(s), worker.submit(s)),
        )
        worker.submit_done()
        worker.wait_done(timeout=5.0)
        watcher.stop(timeout=2.0)

        assert interrupted is False
        assert coord.is_set() is False
        assert worker.cancelled is False
        assert watcher.detected is False
        # At least one sentence was submitted to the worker.
        assert len(seen) >= 1
        # Worker actually played a sentence.
        assert worker.sentences_spoken >= 1
