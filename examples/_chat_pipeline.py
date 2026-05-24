"""Streaming pipeline pieces — keep the LLM token stream moving while
TTS synthesis and audio playback run in the background.

Until iter-007, the chat loop synthesized and played each sentence
synchronously inside the for-token loop. While that audio was being
produced, the LLM stream sat idle: every byte of audio playback was a
byte of token-receipt latency we could have been eating. With one
small bot reply this looks fine; with anything multi-sentence the
playback time stacks linearly.

This module hosts the producer/consumer boundary:

    main thread                          SentenceWorker thread
    -----------                          ---------------------
    for token in llm_stream:             pull sentence from queue
        accumulate buffer                synth via tts_engine
        if complete sentence:            play via speaker stream
            worker.submit(sentence)      track metrics
    worker.submit_done()                 exit cleanly on sentinel
    worker.wait_done()

The worker holds a single persistent speaker (passed via
speaker_factory) instead of opening one per sentence — that's the
other latency win. PyAudio open/close adds a few ms per call which
shows up in TTFS for the first sentence of every turn.

Both `synth_fn` and `play_fn` are injected, so tests drive the
worker against the iter-005 VirtualSpeakerStream and a stub
synthesizer with deterministic audio.
"""

from __future__ import annotations

import sys
import threading
import time
from queue import Empty, Queue
from typing import Callable, Optional

import numpy as np

# Sentinel marking "no more sentences will be submitted." Using a unique
# object ensures it can never collide with a real sentence string.
_SENTINEL = object()

SynthFn = Callable[[str], "tuple[np.ndarray, list[dict]]"]
SpeakerFactory = Callable[[], object]
PlayFn = Callable[..., float]


class SentenceWorker:
    """Background sentence player.

    Submit complete sentences via ``submit(text)``. The worker thread
    pulls from the queue, calls ``synth_fn(text)`` to produce
    ``(audio_np, tokens)``, then calls ``play_fn(speaker, audio_np,
    tokens, is_first_sentence=...)`` to emit them.

    Lifecycle:
        worker = SentenceWorker(...)
        worker.start()
        worker.submit(...); worker.submit(...); ...
        worker.submit_done()    # signal "no more" — worker drains and exits
        worker.wait_done()       # block for the worker thread

    For early termination (LLM error, user barge-in):
        worker.stop()            # drain queue, signal sentinel, join

    Metrics exposed for the caller after wait_done / stop:
        sentences_spoken — count of successfully-played sentences
        tts_time         — cumulative seconds spent inside synth_fn
        playback_time    — cumulative seconds reported by play_fn
        first_audio_at   — clock reading at first non-empty audio
        errors           — list of any exceptions caught inside the
                           worker loop (synth / play / speaker open)
    """

    def __init__(
        self,
        *,
        speaker_factory: SpeakerFactory,
        synth_fn: SynthFn,
        play_fn: PlayFn,
        clock: Callable[[], float] = time.monotonic,
        output=None,
    ):
        self._speaker_factory = speaker_factory
        self._synth_fn = synth_fn
        self._play_fn = play_fn
        self._clock = clock
        self._output = output if output is not None else sys.stdout

        self._queue: Queue = Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False
        self._submit_done_called = False

        # Public metrics — read after wait_done() / stop() returns.
        self.sentences_spoken: int = 0
        self.tts_time: float = 0.0
        self.playback_time: float = 0.0
        self.first_audio_at: Optional[float] = None
        self.errors: list[Exception] = []

    # --- lifecycle ---------------------------------------------------

    def start(self) -> None:
        if self._started:
            raise RuntimeError("SentenceWorker already started")
        self._started = True
        self._thread = threading.Thread(
            target=self._run,
            name="SentenceWorker",
            daemon=True,
        )
        self._thread.start()

    def submit(self, sentence: str) -> None:
        """Queue a sentence for synth + play. No-op if the worker has
        been stopped.
        """
        if self._stop_event.is_set() or self._submit_done_called:
            return
        self._queue.put(sentence)

    def submit_done(self) -> None:
        """Signal that no more sentences will be submitted. The worker
        drains whatever's queued and then exits cleanly.
        """
        if self._submit_done_called:
            return
        self._submit_done_called = True
        self._queue.put(_SENTINEL)

    def wait_done(self, timeout: Optional[float] = None) -> None:
        """Join the worker thread. Caller is expected to have already
        called ``submit_done()`` (otherwise the worker is still happily
        waiting on the queue).
        """
        if self._thread is None:
            raise RuntimeError("SentenceWorker not started")
        self._thread.join(timeout=timeout)

    def stop(self, timeout: float = 5.0) -> None:
        """Drain the queue, signal stop, and join the thread.

        Pending unplayed sentences are dropped. The currently-playing
        sentence is allowed to finish (mid-sentence cancellation is
        iter-009 barge-in territory).
        """
        if not self._started:
            return
        self._stop_event.set()
        # Empty out anything queued so the worker reaches the sentinel
        # quickly without playing extras.
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break
        # Always push a sentinel so the worker's blocking get() unblocks
        # even if submit_done() was never called.
        self._queue.put(_SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # --- worker body -------------------------------------------------

    def _run(self) -> None:
        # Open the persistent speaker. If this fails (no audio device,
        # virtual interface terminated, etc.) record the error and exit.
        try:
            speaker = self._speaker_factory()
        except Exception as e:  # pragma: no cover — exercised by test
            self.errors.append(e)
            return

        try:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    break
                if self._stop_event.is_set():
                    # Drain remaining without playing.
                    continue

                sentence = item
                if not isinstance(sentence, str) or not sentence.strip():
                    continue

                # Synth
                try:
                    t = self._clock()
                    audio_np, tokens = self._synth_fn(sentence)
                    self.tts_time += self._clock() - t
                except Exception as e:
                    self.errors.append(e)
                    continue

                if audio_np is None or len(audio_np) == 0:
                    continue

                if self.first_audio_at is None:
                    self.first_audio_at = self._clock()

                # Play
                try:
                    is_first = self.sentences_spoken == 0
                    elapsed = self._play_fn(
                        speaker,
                        audio_np,
                        tokens,
                        is_first_sentence=is_first,
                    )
                    self.playback_time += float(elapsed) if elapsed else 0.0
                    self.sentences_spoken += 1
                except Exception as e:
                    self.errors.append(e)
                    continue
        finally:
            for method in ("stop_stream", "close"):
                fn = getattr(speaker, method, None)
                if fn is not None:
                    try:
                        fn()
                    except Exception:
                        pass
