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

import inspect
import sys
import threading
import time
from queue import Empty, Queue
from typing import Any, Callable, Optional

import numpy as np


def _play_fn_accepts_cancel_event(play_fn) -> bool:
    """Return True if ``play_fn`` looks like it can accept a
    ``cancel_event`` keyword argument.

    Detected via ``inspect.signature`` once at worker construction
    rather than per-call ``try/except TypeError``. The old
    per-call approach masked real bugs: a play_fn whose BODY
    raised ``TypeError`` (for any reason) would be retried
    without ``cancel_event``, causing the function to be invoked
    *twice* per sentence and writing partial audio twice.

    Conservative on errors: if the callable's signature can't
    be inspected (some C extensions, builtin functions), assume
    no cancel_event support — same fallback the old code provided
    for old-style play_fns.
    """
    try:
        sig = inspect.signature(play_fn)
    except (ValueError, TypeError):
        return False
    params = sig.parameters
    if "cancel_event" in params:
        return True
    # ``**kwargs`` accepts anything, including cancel_event.
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

def _play_fn_accepts_lag_out(play_fn) -> bool:
    """iter-071: Return True if ``play_fn`` accepts ``lag_out``.

    Same sniff pattern as ``_play_fn_accepts_cancel_event``: detect
    once at worker construction; pass the kwarg only when supported
    so older / minimal play_fns continue to work unchanged.
    """
    try:
        sig = inspect.signature(play_fn)
    except (ValueError, TypeError):
        return False
    params = sig.parameters
    if "lag_out" in params:
        return True
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


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
        fillers: Optional[list] = None,
        idle_threshold: float = 0.0,
        filler_picker: Optional[Callable[[list], object]] = None,
        recent_filler_ids: Optional[Any] = None,
    ):
        self._speaker_factory = speaker_factory
        self._synth_fn = synth_fn
        self._play_fn = play_fn
        # iter-023: detect once at construction whether the
        # play_fn accepts ``cancel_event``, instead of swallowing
        # TypeError per-call (which masked real bugs).
        self._play_fn_supports_cancel = _play_fn_accepts_cancel_event(play_fn)
        # iter-071: same detection pattern for the optional lag_out
        # kwarg. When supported, the worker passes a fresh dict per
        # play call and folds the stats into worker-level totals.
        self._play_fn_supports_lag_out = _play_fn_accepts_lag_out(play_fn)
        self._clock = clock
        self._output = output if output is not None else sys.stdout

        # Pre-rendered filler clips. Each entry is an ``(audio_np,
        # tokens)`` tuple — the same shape ``synth_fn`` returns. The
        # caller is responsible for synthesizing them once at startup
        # so we don't pay TTS latency exactly when we're trying to
        # mask it. iter-011.
        self._fillers = list(fillers) if fillers else []
        self._idle_threshold = float(idle_threshold)
        # Picker so tests can be deterministic (random.choice in prod,
        # ``lambda lst: lst[0]`` in tests).
        if filler_picker is None:
            import random as _r
            filler_picker = _r.choice
        self._filler_picker = filler_picker
        # iter-113: cross-turn filler variety. The caller (ChatLoop)
        # passes a bounded FIFO of recently-played filler IDs that
        # persists across turns. The picker prefers fillers NOT in
        # this set; falls back to the full available list when
        # everything's recent. Default None = empty (no cross-turn
        # tracking, equivalent to pre-iter-113 behavior).
        self._recent_filler_ids = recent_filler_ids

        self._queue: Queue = Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # Hard-cancel signal forwarded to play_fn so it can break
        # mid-chunk. iter-009 barge-in primitive — distinct from
        # _stop_event, which is the polite "drain and exit" signal.
        self._cancel_event = threading.Event()
        self._started = False
        self._submit_done_called = False
        self.cancelled: bool = False

        # Public metrics — read after wait_done() / stop() returns.
        self.sentences_spoken: int = 0
        # iter-040: count of sentences whose play_fn was interrupted
        # by ``cancel_event`` mid-stream (vs completed naturally).
        # Detected by sampling cancel_event before and after each
        # play_fn call — if it transitioned from clear to set DURING
        # the call, that sentence was cut mid-stream. Tighter than
        # sampling only after, which would false-positive when
        # cancel fires in the microseconds after natural completion.
        # Metric 2.18 in the perf-metrics taxonomy. Validates the
        # iter-009 / iter-026 cancel plumbing.
        self.cancelled_sentences: int = 0
        # iter-044: cumulative seconds the worker spent blocked on
        # ``self._queue.get(...)`` between sentences. Excludes the
        # first wait (that's TTFsent — covered by iter-038). High
        # idle gap = LLM didn't produce complete sentences fast
        # enough; low gap = synth is the bottleneck. Combined with
        # iter-043's streaming_overlap_ratio, points at where the
        # pipeline is actually slow. Metric 2.16 in the perf-metrics
        # taxonomy.
        self.idle_gap_total: float = 0.0
        # iter-046: count of non-punctuation tokens emitted (across
        # all sentences this run) and cumulative audio seconds. Used
        # to derive bot WPM. Metric 1.13 in the perf-metrics taxonomy.
        # Healthy voice agents land 150-180 WPM; outside that range
        # is either too fast (user can't follow) or too slow (user
        # interrupts). Excludes filler clips — those have no tokens.
        self.word_count_total: int = 0
        self.audio_seconds_total: float = 0.0
        self.fillers_played: int = 0
        # iter-081: identity (id()) of the filler clip the picker
        # selected for this turn. None on turns where no filler
        # played. Mic_chat aggregates these across turns into a
        # session-wide diversity count — `id()` is stable across
        # worker instances because the rendered_fillers list lives
        # in mic_chat's scope and is passed by reference. Metric 3.8
        # in the perf-metrics taxonomy ("Novel/speculative").
        self.last_filler_id: Optional[int] = None
        self.tts_time: float = 0.0
        self.playback_time: float = 0.0
        # iter-061: time spent opening the persistent speaker
        # (taxonomy 2.8). Validates the iter-008 win — if this
        # creeps back, TTFS regresses silently.
        self.speaker_open_seconds: float = 0.0
        # iter-071: token-reveal lag accumulators across the per-
        # sentence play_fn calls. ``mean = sum / count`` is computed
        # by the consumer (ChatLoop) since ``count`` is the more
        # useful number for downstream filtering. ``max`` is the
        # single-token worst-case across all sentences this turn.
        # All zero when play_fn doesn't support lag_out (test stubs)
        # or no sentences played. Metric 2.17 in the perf-metrics
        # taxonomy.
        self.token_reveal_lag_sum: float = 0.0
        self.token_reveal_lag_count: int = 0
        self.token_reveal_lag_max: float = 0.0
        # iter-073: first-sentence synth window. Latched once on
        # the first successful synth. ChatLoop intersects this with
        # the LLM-stream window to compute first-sentence overlap
        # savings — the actual TTFS win from iter-008's streaming-
        # sentence-dispatch design (vs iter-043's whole-stream
        # ratio). Both None on turns where no synth ever ran or
        # every synth raised. Metric 2.2 in the perf-metrics
        # taxonomy.
        self.first_synth_start_at: Optional[float] = None
        self.first_synth_done_at: Optional[float] = None
        # iter-062: peak queue depth observed during the turn —
        # number of sentences waiting for synth at the moment of
        # the deepest backlog. >1 means the producer (LLM splitter)
        # got ahead of the consumer (synth+play); high values mean
        # synth is the bottleneck and streaming-overlap can't keep
        # up with token rate. Inverse of iter-044's idle_gap (which
        # measures the worker STARVED). Metric 2.7 in the perf-
        # metrics taxonomy.
        self.max_queue_depth: int = 0
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
        # iter-062: peak queue-depth tracker. ``qsize()`` is documented
        # as approximate but the only race is with the consumer thread,
        # which only DRAINS — so reads here are at most a slight
        # over-count, fine for a peak metric. Sample after put so the
        # just-queued sentence is included in the depth.
        depth = self._queue.qsize()
        if depth > self.max_queue_depth:
            self.max_queue_depth = depth

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
        sentence is allowed to finish — for mid-sentence cancellation
        use ``cancel()`` instead.
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

    def cancel(self, timeout: float = 5.0) -> None:
        """Hard-cancel: interrupt the currently-playing sentence
        mid-stream and drop everything queued behind it.

        Implementation: set ``_cancel_event`` so play_fn breaks between
        chunks (see examples/_chat_playback.play_aligned), then drain
        and join just like ``stop()``.

        Idempotent — calling twice is safe.
        """
        if not self._started:
            return
        self.cancelled = True
        self._cancel_event.set()
        self.stop(timeout=timeout)

    # --- worker body -------------------------------------------------

    def _play_clip(self, speaker, audio_np, tokens, *, is_first: bool) -> bool:
        """Play one audio clip via play_fn. Updates first_audio_at and
        playback_time; appends to errors on failure. Returns True if a
        non-empty clip was actually played, False if it was skipped or
        crashed.

        Used for both real sentences (from the queue) and pre-rendered
        fillers, so the bookkeeping stays in one place.
        """
        if audio_np is None or len(audio_np) == 0:
            return False
        if self.first_audio_at is None:
            self.first_audio_at = self._clock()
        # iter-040: capture cancel_event state BEFORE the play call.
        # If it transitions to set DURING the call, the play_fn
        # exited because of cancel_event (vs completing naturally).
        # The before/after pair tightens the race vs sampling only
        # after — a cancel firing in the microseconds after natural
        # completion would otherwise count as a false positive.
        cancel_was_set_before = self._cancel_event.is_set()
        try:
            # iter-023: signature was inspected at construction;
            # call with the right kwargs once. A TypeError raised
            # by the play_fn body now surfaces correctly via the
            # outer ``except Exception`` instead of triggering a
            # silent retry.
            # iter-071: pass a fresh lag dict when the play_fn
            # supports it; fold per-call stats into worker totals
            # after each call.
            # iter-075: assemble optional kwargs into a dict instead
            # of branching on each combination. Each new optional
            # kwarg adds one entry rather than doubling the branch
            # count. ``base_kwargs`` is the always-passed set;
            # extras are appended only when the play_fn signature
            # accepts them.
            lag_out: dict | None = (
                {} if self._play_fn_supports_lag_out else None
            )
            base_kwargs: dict = {"is_first_sentence": is_first}
            if self._play_fn_supports_cancel:
                base_kwargs["cancel_event"] = self._cancel_event
            if lag_out is not None:
                base_kwargs["lag_out"] = lag_out
            elapsed = self._play_fn(
                speaker, audio_np, tokens, **base_kwargs,
            )
            self.playback_time += float(elapsed) if elapsed else 0.0
            # iter-071: fold this call's lag stats into worker-level
            # totals. play_aligned only writes the dict when at
            # least one token was emitted, so an empty dict here is
            # the legitimate "no data" signal.
            if lag_out:
                self.token_reveal_lag_sum += lag_out.get("sum", 0.0)
                self.token_reveal_lag_count += lag_out.get("count", 0)
                call_max = lag_out.get("max", 0.0)
                if abs(call_max) > abs(self.token_reveal_lag_max):
                    self.token_reveal_lag_max = call_max
            # iter-040: cancel transitioned during the call —
            # play_fn exited mid-stream.
            if (
                not cancel_was_set_before
                and self._cancel_event.is_set()
            ):
                self.cancelled_sentences += 1
            return True
        except Exception as e:
            self.errors.append(e)
            return False

    def _run(self) -> None:
        # Open the persistent speaker. If this fails (no audio device,
        # virtual interface terminated, etc.) record the error and exit.
        # iter-061: time the open so the chat loop can surface
        # speaker-open overhead per turn. Validates the iter-008
        # persistent-speaker win — if this creeps back (driver
        # change, Bluetooth, SDL), TTFS regresses silently.
        open_t0 = self._clock()
        try:
            speaker = self._speaker_factory()
        except Exception as e:  # pragma: no cover — exercised by test
            self.errors.append(e)
            return
        self.speaker_open_seconds = self._clock() - open_t0

        # Tracks "have we written any audio output yet" — drives the
        # is_first_sentence flag for the play_fn (controls the "Bot:"
        # prefix). Includes both fillers and real sentences.
        is_first_audio = True
        # iter-087: track filler clip ids played this turn so the
        # picker can pick from the still-unplayed subset on the
        # second attempt. Cap at MAX_FILLERS to avoid stacking.
        played_filler_ids: set[int] = set()
        MAX_FILLERS = 2

        try:
            while True:
                # iter-087: filler can still fire if we haven't hit
                # MAX_FILLERS AND there's at least one clip the
                # picker hasn't seen yet. Single-shot turns that
                # play one filler then receive a real sentence
                # naturally exit this branch — sentences_spoken > 0
                # below disables the timeout.
                filler_can_fire = (
                    bool(self._fillers)
                    and len(played_filler_ids) < MAX_FILLERS
                    and len(played_filler_ids) < len(self._fillers)
                )
                # Decide whether to wait with a timeout so we can play
                # a filler if the LLM stalls. Only applies before the
                # first real sentence and only if fillers are
                # configured.
                use_filler_timeout = (
                    filler_can_fire
                    and self._idle_threshold > 0
                    and self.sentences_spoken == 0
                    and not self._submit_done_called
                )
                # iter-044: time the queue.get call. Skip the first
                # wait — that's TTFsent (already iter-038). After
                # the first sentence has been spoken, the gap
                # between sentences is the metric we want.
                gap_t0 = self._clock()
                try:
                    if use_filler_timeout:
                        item = self._queue.get(timeout=self._idle_threshold)
                    else:
                        item = self._queue.get()
                except Empty:
                    # iter-087: idle threshold hit before any sentence
                    # arrived. Play a filler clip to mask LLM first-
                    # token latency. Up to MAX_FILLERS fire per turn,
                    # picking from clips not already played to avoid
                    # repetition (the "umm umm" pattern).
                    if filler_can_fire:
                        available = [
                            c for c in self._fillers
                            if id(c) not in played_filler_ids
                        ]
                        if not available:
                            # Defensive: filler_can_fire said yes but
                            # filtering left nothing. Skip this round.
                            continue
                        # iter-113: cross-turn variety. Prefer clips
                        # NOT in the recent-IDs FIFO. Fall back to
                        # the full available list when every
                        # remaining clip was used recently — better
                        # to repeat than to drop the filler entirely.
                        if self._recent_filler_ids:
                            fresh = [
                                c for c in available
                                if id(c) not in self._recent_filler_ids
                            ]
                            if fresh:
                                available = fresh
                        clip = self._filler_picker(available)
                        audio_np, tokens = clip
                        played = self._play_clip(
                            speaker, audio_np, tokens, is_first=is_first_audio,
                        )
                        # Whether or not it played, record this clip
                        # as "attempted" so the next loop iteration
                        # picks something different.
                        played_filler_ids.add(id(clip))
                        if played:
                            is_first_audio = False
                            self.fillers_played += 1
                            # iter-081: record which filler we picked
                            # (by tuple id, stable across workers
                            # because mic_chat holds the canonical
                            # rendered_fillers list). Multi-shot
                            # iter-087: this remains the LAST filler
                            # played; session-wide diversity comes
                            # from print_session_summary aggregating
                            # across turns.
                            self.last_filler_id = id(clip)
                            # iter-113: append to the cross-turn FIFO
                            # so the NEXT turn's picker prefers
                            # something different. Append-only; the
                            # caller's bounded deque handles eviction.
                            if self._recent_filler_ids is not None:
                                # Duck-type: deque has .append, set has
                                # .add, list has .append. Try .append
                                # first (most common shape), fall back
                                # to .add for set callers.
                                if hasattr(self._recent_filler_ids, "append"):
                                    self._recent_filler_ids.append(id(clip))
                                else:
                                    self._recent_filler_ids.add(id(clip))
                    continue

                if item is _SENTINEL:
                    break
                if self._stop_event.is_set():
                    # Drain remaining without playing.
                    continue

                # iter-044: stamp the gap. Only count gaps AFTER the
                # first sentence — the first wait is TTFsent
                # territory, not "between-sentence stall."
                if self.sentences_spoken > 0:
                    self.idle_gap_total += self._clock() - gap_t0

                sentence = item
                if not isinstance(sentence, str) or not sentence.strip():
                    continue

                # Synth
                try:
                    t = self._clock()
                    audio_np, tokens = self._synth_fn(sentence)
                    elapsed = self._clock() - t
                    self.tts_time += elapsed
                    # iter-073: latch first-sentence synth window the
                    # FIRST time synth succeeds. A failed first synth
                    # (raise → continue) means the next successful
                    # synth becomes "first" for TTFS accounting,
                    # which matches the user's perception.
                    if self.first_synth_start_at is None:
                        self.first_synth_start_at = t
                        self.first_synth_done_at = t + elapsed
                except Exception as e:
                    self.errors.append(e)
                    continue

                played = self._play_clip(
                    speaker, audio_np, tokens, is_first=is_first_audio,
                )
                if played:
                    is_first_audio = False
                    self.sentences_spoken += 1
                    # iter-046: accumulate words + audio seconds for WPM.
                    # Words = non-punctuation tokens. Falls back to a
                    # whitespace split on the sentence text if tokens
                    # is empty — kokoro's alignment can be missing on
                    # some configurations.
                    if tokens:
                        for tok in tokens:
                            text = tok.get("text", "") if isinstance(tok, dict) else getattr(tok, "text", "")
                            stripped = text.strip()
                            if stripped and not all(c in ".,!?;:" for c in stripped):
                                self.word_count_total += 1
                    else:
                        self.word_count_total += len(sentence.split())
                    # Audio duration: sample count / TTS rate. The
                    # play_fn writes int16 PCM at 24kHz (TTS_RATE);
                    # use that here. We have audio_np in float32
                    # already so its length is the sample count.
                    self.audio_seconds_total += len(audio_np) / 24000.0
        finally:
            for method in ("stop_stream", "close"):
                fn = getattr(speaker, method, None)
                if fn is not None:
                    try:
                        fn()
                    except Exception:
                        pass


# ---- Barge-in watcher --------------------------------------------------------

class BargeInWatcher:
    """Background thread that listens for user speech on a mic stream
    while the bot is speaking.

    Run a ``VadState`` over chunks read from the mic. When the VAD
    transitions into the configured trigger event (default:
    ``ACTIVE`` — i.e. the moment user speech crosses the threshold),
    invoke the user-supplied callback once. Continue capturing frames
    after triggering so the orchestrating code can replay them into
    the next ``record_utterance_streaming`` call (so the user's first
    syllables aren't lost).

    Designed to be paired with ``SentenceWorker.cancel`` — the typical
    setup is:

        watcher = BargeInWatcher(
            mic=mic_stream,
            on_speech_detected=worker.cancel,
        )
        watcher.start()
        worker.submit(...); worker.submit(...)
        worker.submit_done()
        worker.wait_done()
        watcher.stop()
        if watcher.detected:
            # feed watcher.frames into the next record loop

    Tests inject deterministic audio via the iter-005
    ``VirtualMicStream`` and verify the callback fires at the right
    instant relative to pushed audio.
    """

    def __init__(
        self,
        *,
        mic,
        on_speech_detected: Callable[[], None],
        vad=None,
        chunk_size: int = 1024,
        rate: int = 16000,
        trigger_on: str = "active",
        clock: Callable[[], float] = time.monotonic,
        poll_interval: float = 0.005,
        lead_in_chunks: int = 0,
    ):
        # Local imports keep this module independent of the helpers
        # / recording module import paths during type-checking.
        from examples._chat_helpers import VadEvent, VadState
        from examples._chat_recording import rms

        self._mic = mic
        self._callback = on_speech_detected
        self._vad = vad if vad is not None else VadState()
        self._chunk = chunk_size
        self._rate = rate
        # iter-024: bind the centralized rms helper at construction
        # so the per-frame loop doesn't pay an import cost.
        self._rms = rms
        if trigger_on not in ("active", "done_ok"):
            raise ValueError(
                f"trigger_on must be 'active' or 'done_ok', got {trigger_on!r}"
            )
        self._trigger_on = trigger_on
        self._clock = clock
        self._poll = poll_interval
        self._VadEvent = VadEvent

        # iter-025: ring buffer of the most recent N pre-detection
        # frames. When detection fires, the ring buffer's contents
        # are flushed into ``frames`` followed by all subsequent
        # frames. Default 0 means no pre-detection capture — only
        # the detection frame and onwards are stored.
        if lead_in_chunks < 0:
            raise ValueError("lead_in_chunks must be >= 0")
        self._lead_in_chunks = lead_in_chunks
        self._lead_in_buffer: list[bytes] = []

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False

        # Public observable state (read after stop()).
        self.detected: bool = False
        self.frames: list[bytes] = []
        self.events: list = []  # full sequence for assertions
        self.frame_idx_at_trigger: Optional[int] = None
        # iter-074: timestamps for bargeable-time fraction. The
        # watcher is "active" between started_at and stopped_at.
        # ChatLoop intersects this with the bot-speech window
        # (worker.first_audio_at → end of last audio) to surface
        # what fraction of bot speech could actually have been
        # interrupted. Both None until the corresponding lifecycle
        # call fires. Metric 1.19 in the perf-metrics taxonomy.
        self.started_at: Optional[float] = None
        self.stopped_at: Optional[float] = None

    def start(self) -> None:
        if self._started:
            raise RuntimeError("BargeInWatcher already started")
        self._started = True
        # iter-074: stamp before launching the thread so callers
        # racing for the timestamp can't catch a None.
        self.started_at = self._clock()
        self._thread = threading.Thread(
            target=self._run, name="BargeInWatcher", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        if not self._started:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        # iter-074: stamp after the thread joins so stopped_at
        # reflects the moment the watcher is genuinely no longer
        # listening (matching the start_at semantics).
        self.stopped_at = self._clock()

    def _run(self) -> None:
        frame_idx = 0
        while not self._stop_event.is_set():
            try:
                # Only read if data is actually available, so the
                # watcher doesn't grab the silence-padding zeros that
                # VirtualMicStream serves on underflow (which would
                # generate noisy IDLE events at the rate of the poll
                # loop). For real PyAudio mics, get_read_available()
                # returns the count buffered in PortAudio.
                avail = getattr(self._mic, "get_read_available", lambda: self._chunk)()
            except Exception:
                avail = 0

            if avail < self._chunk:
                # Sleep briefly so we don't pin a core; this is also
                # the granularity at which the watcher reacts.
                if self._stop_event.wait(timeout=self._poll):
                    break
                continue

            try:
                data = self._mic.read(self._chunk, exception_on_overflow=False)
            except Exception:
                break

            if not data:
                continue

            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            # iter-024: centralized rms helper (single source of truth
            # with _chat_recording.rms; same iter-014 NaN-empty guard).
            level = self._rms(audio)
            now = self._clock()
            event = self._vad.feed(level, now)
            self.events.append(event)

            # iter-025: only store frames from detection onwards
            # (plus an optional pre-detection lead-in maintained as
            # a ring buffer). Pre-detection frames in production are
            # likely bot acoustic feedback or silence — feeding them
            # into the next record_utterance via primed_frames would
            # have STT transcribe bot voice as user speech.
            if self.detected:
                self.frames.append(data)
            elif self._trigger_matches(event):
                # Trigger frame: flush lead-in buffer (pre-detection
                # context) followed by this frame (the one that
                # actually crossed threshold). Order matters —
                # trigger frame is the user's first audible syllable.
                self.detected = True
                self.frame_idx_at_trigger = frame_idx
                self.frames.extend(self._lead_in_buffer)
                self.frames.append(data)
                self._lead_in_buffer.clear()
                try:
                    self._callback()
                except Exception:
                    # Caller's callback shouldn't take down the watcher.
                    pass
            else:
                # Pre-detection, no trigger this frame: maintain the
                # ring buffer. When detection eventually fires, the
                # buffer's contents will be flushed.
                if self._lead_in_chunks > 0:
                    self._lead_in_buffer.append(data)
                    if len(self._lead_in_buffer) > self._lead_in_chunks:
                        self._lead_in_buffer.pop(0)

            frame_idx += 1

    def _trigger_matches(self, event) -> bool:
        if self._trigger_on == "active":
            return event is self._VadEvent.ACTIVE
        if self._trigger_on == "done_ok":
            return event is self._VadEvent.DONE_OK
        return False


# ---- Barge-in coordinator ----------------------------------------------------

class BargeInCoordinator:
    """Single-shot barge-in signal that bundles together the actions
    that need to happen when the user interrupts.

    The chat loop creates one per turn and wires it into:
      - ``BargeInWatcher`` callback: ``coord.trigger``
      - The LLM for-token loop: ``if coord.is_set(): break``
      - SentenceWorker: cancelled inside ``trigger()``

    Idempotent — multiple calls to ``trigger()`` are safe; only the
    first one does anything. That matters because the watcher might
    fire on multiple ACTIVE events, or because cleanup paths might
    call ``trigger()`` defensively.

    `on_trigger` is an optional hook for hangups that can't go
    through the worker — closing an open HTTP requests stream, for
    instance, so we stop pulling tokens we'll never use.
    """

    def __init__(
        self,
        worker=None,
        *,
        on_trigger: Optional[Callable[[], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._worker = worker
        self._on_trigger = on_trigger
        # iter-030: the chat loop compares ``triggered_at`` against
        # timestamps stamped by its own injected clock to decide
        # phase ("LLM-stream" vs "playback") for the barge-in
        # diagnostic message. Calling ``time.monotonic()`` here
        # would make those two values incomparable under any test
        # that mocks the clock — the test would see ``triggered_at``
        # in real wall-clock time but ``llm_stream_done_at`` on the
        # fake clock. Accept the same clock the caller already uses.
        self._clock = clock
        self._event = threading.Event()
        self._lock = threading.Lock()
        # When the trigger fired, sampled from ``clock``. None until set.
        self.triggered_at: Optional[float] = None
        # iter-041: when worker.cancel() returned (i.e. the moment
        # the SentenceWorker thread had been joined and playback
        # was actually stopped). None until trigger() runs and
        # reaches that point.
        # Latency = playback_stopped_at - triggered_at. Metric 2.10
        # in the perf-metrics taxonomy. Barge-in feel >200ms is the
        # moment the user thinks the bot is ignoring them.
        self.playback_stopped_at: Optional[float] = None

    @property
    def event(self) -> threading.Event:
        """The underlying ``threading.Event`` — useful when callers
        want to ``.wait()`` on it directly.
        """
        return self._event

    def is_set(self) -> bool:
        return self._event.is_set()

    def trigger(self) -> None:
        """Idempotent. The first call:
            1. flips the event (so the for-token loop sees it)
            2. timestamps ``triggered_at``
            3. calls ``worker.cancel()`` if a worker is bound
            4. calls ``on_trigger`` if one is bound

        Subsequent calls are no-ops. Exceptions in step 3 / 4 are
        swallowed so a bad hook can't leave the event un-set.
        """
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            self.triggered_at = self._clock()
        # Outside the lock — worker.cancel takes its own lock and
        # may join a thread; we don't want to hold ours that long.
        if self._worker is not None:
            try:
                self._worker.cancel(timeout=5.0)
            except Exception:
                pass
        # iter-041: stamp playback_stopped_at AFTER worker.cancel
        # has joined the thread — that's the moment playback is
        # truly halted. If no worker was bound, stamp now anyway
        # (the trigger itself counts as "stopped" — there was
        # nothing to stop).
        self.playback_stopped_at = self._clock()
        if self._on_trigger is not None:
            try:
                self._on_trigger()
            except Exception:
                pass
