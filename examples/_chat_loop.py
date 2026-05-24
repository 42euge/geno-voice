"""ChatLoop — per-turn chat orchestration extracted from
``mic_chat.run_chat``.

Until iter-014 the per-turn body lived inline inside
``run_chat``: a 200-line block that records an utterance, opens
an LLM stream, spins up a worker for synth + play, runs a
barge-in watcher, and stitches everything back together with
metrics. The orchestration tests in iter-009 / iter-010 / iter-012
*approximated* this shape with helper functions, but the real
function was untested.

Pulling it into a class with all its dependencies injected lets
us drive the actual production code path with stub STT, stub LLM,
and virtual audio. mic_chat.run_chat is now a thin shim that
constructs real dependencies (PyAudio mic, kokoro TTS, requests
LLM stream) and delegates to ChatLoop.

The split:
  ``ChatLoop.__init__`` — accept dependencies as callables /
    plain values. Don't construct anything that requires
    real I/O.
  ``ChatLoop.run_one_turn(messages, primed_frames=None)`` —
    one full turn. Returns ``(metrics, next_primed_frames)``,
    where ``metrics`` is None on no-transcription / LLM error
    and ``next_primed_frames`` is non-None only when the user
    barged in (so the caller can pass them back in for the
    next turn).

Same observable behavior as the inline run_chat body, including
the iter-014 hardening (rms-empty, error-path frame carryover)
and the iter-013 LLM-stream cleanup.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from examples._chat_helpers import (
    flush_pending_audio,
    split_complete_sentences,
    trim_history,
)
from examples._chat_metrics import TurnMetrics
from examples._chat_pipeline import (
    BargeInCoordinator,
    BargeInWatcher,
    SentenceWorker,
)
from examples._chat_recording import (
    CHUNK,
    RATE,
    SILENCE_DURATION,
    record_utterance_streaming,
)

# ANSI codes — duplicated from mic_chat for status prints. Keeps
# this module a clean leaf with no dependency back on mic_chat.
_DIM = "\033[2m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


@dataclass
class TurnResult:
    """Outcome of ``ChatLoop.run_one_turn``.

    Fields:
      ``metrics`` — the populated TurnMetrics on a successful or
        barged-in turn; None on no-transcription / too-short-utterance
        / LLM error.
      ``next_primed_frames`` — frames captured by the barge-in
        watcher to feed into the *next* turn's record. None when
        no barge-in occurred.
      ``had_error`` — True if the LLM call raised. The caller
        should not append the assistant message to history in
        this case (already popped inside the loop).
    """
    metrics: Optional[TurnMetrics]
    next_primed_frames: Optional[list]
    had_error: bool = False


class ChatLoop:
    """Per-turn chat orchestrator with all dependencies injected.

    Construct once at startup with concrete real-world bindings
    (mic stream, speaker factory, real LLM streaming function,
    real synth/play). Call ``run_one_turn`` in a loop, threading
    ``next_primed_frames`` through to preserve barge-in audio.

    Tests construct one with stubs (VirtualMicStream + virtual
    speaker factory + canned LLM token list + stub synth) and
    drive a single turn through ``run_one_turn`` to verify the
    full per-turn behavior.
    """

    def __init__(
        self,
        *,
        # Audio
        mic,
        speaker_factory: Callable[[], object],
        rate: int = RATE,
        chunk: int = CHUNK,
        silence_duration: float = SILENCE_DURATION,
        # STT
        stt_engine,
        transcribe_fn: Optional[Callable[[bytes], Optional[str]]] = None,
        # LLM
        llm_stream_fn: Callable[[list, dict], Iterator[str]],
        llm_config: dict,
        # TTS / playback
        synth_fn: Callable[[str], tuple],
        play_fn: Callable[..., float],
        # Filler config (iter-011)
        fillers: Optional[list] = None,
        idle_threshold: float = 0.0,
        # Tunables / I/O
        clock: Callable[[], float] = time.monotonic,
        output=None,
        wait_done_timeout: float = 120.0,
        cancel_wait_timeout: float = 5.0,
    ):
        self._mic = mic
        self._speaker_factory = speaker_factory
        self._rate = rate
        self._chunk = chunk
        self._silence_duration = silence_duration

        self._stt_engine = stt_engine
        self._transcribe_fn = transcribe_fn

        self._llm_stream_fn = llm_stream_fn
        self._llm_config = llm_config

        self._synth_fn = synth_fn
        self._play_fn = play_fn

        self._fillers = list(fillers) if fillers else []
        self._idle_threshold = idle_threshold

        self._clock = clock
        self._output = output  # passed to record_utterance_streaming
        self._wait_done_timeout = wait_done_timeout
        self._cancel_wait_timeout = cancel_wait_timeout

    def _print(self, msg: str) -> None:
        """Status print — go to the same output the recording loop
        uses if the caller injected one (tests), else stdout.
        """
        if self._output is not None:
            self._output.write(msg + "\n")
            self._output.flush()
        else:
            print(msg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_one_turn(
        self,
        messages: list[dict],
        *,
        primed_frames: Optional[list] = None,
    ) -> TurnResult:
        """Run one full chat turn.

        Mutates ``messages`` in place — appends a user message after
        successful STT, an assistant message after successful LLM
        completion, and pops the user message on LLM error.

        Returns a ``TurnResult`` with metrics, optional captured
        frames for the next turn, and an error flag.
        """
        # ---- Phase 1: record user utterance ----
        wav_bytes, speech_dur, stt_time = record_utterance_streaming(
            self._mic,
            self._stt_engine,
            transcribe_fn=self._transcribe_fn,
            clock=self._clock,
            output=self._output,
            primed_frames=primed_frames,
        )
        if not wav_bytes:
            return TurnResult(metrics=None, next_primed_frames=None)

        text = getattr(self._stt_engine, "_last_text", None)
        if not text or len(text.strip()) < 2:
            self._print(f"  {_YELLOW}(no transcription){_RESET}")
            return TurnResult(metrics=None, next_primed_frames=None)

        metrics = TurnMetrics(
            speech_duration=speech_dur,
            model=self._llm_config.get("model", ""),
        )
        speech_ended_at = self._clock() - self._silence_duration
        turn_start = self._clock()
        metrics.stt_time = stt_time
        metrics.transcript = text.strip()

        # ---- Phase 2: LLM stream + worker + watcher ----
        messages.append({"role": "user", "content": metrics.transcript})

        worker = SentenceWorker(
            speaker_factory=self._speaker_factory,
            synth_fn=self._synth_fn,
            play_fn=self._play_fn,
            fillers=self._fillers,
            idle_threshold=self._idle_threshold if self._fillers else 0.0,
            clock=self._clock,
        )
        worker.start()

        flush_pending_audio(self._mic, chunk_size=self._chunk)

        llm_gen = self._llm_stream_fn(messages, self._llm_config)
        coord = BargeInCoordinator(worker=worker)
        watcher = BargeInWatcher(
            mic=self._mic,
            on_speech_detected=coord.trigger,
            chunk_size=self._chunk,
            rate=self._rate,
        )
        watcher.start()

        llm_start = self._clock()
        first_token_at: Optional[float] = None
        token_buffer = ""
        full_response = ""
        llm_stream_done_at: Optional[float] = None
        next_primed: Optional[list] = None
        had_error = False

        try:
            for token in llm_gen:
                if coord.is_set():
                    break  # iter-012: barge-in during LLM streaming
                if first_token_at is None:
                    first_token_at = self._clock()
                token_buffer += token
                full_response += token

                complete, token_buffer = split_complete_sentences(token_buffer)
                for sentence in complete:
                    worker.submit(sentence)

            llm_stream_done_at = self._clock()

            if not coord.is_set():
                remaining = token_buffer.strip()
                if remaining:
                    worker.submit(remaining)
                worker.submit_done()
                worker.wait_done(timeout=self._wait_done_timeout)
            else:
                worker.wait_done(timeout=self._cancel_wait_timeout)

            watcher.stop(timeout=2.0)
            if watcher.detected:
                next_primed = list(watcher.frames)
                phase = (
                    "LLM-stream phase"
                    if coord.triggered_at is not None
                    and llm_stream_done_at is not None
                    and coord.triggered_at < llm_stream_done_at
                    else "playback phase"
                )
                self._print(
                    f"\n  {_DIM}barge-in during {phase}: replaying "
                    f"{len(next_primed)} captured frames "
                    f"({len(next_primed) * self._chunk / self._rate:.1f}s){_RESET}"
                )

            # Populate metrics from the worker.
            if worker.first_audio_at is not None:
                metrics.ttfs = worker.first_audio_at - speech_ended_at
            metrics.llm_first_token = (
                (first_token_at - llm_start) if first_token_at else 0
            )
            metrics.llm_total = (
                (llm_stream_done_at - llm_start) if llm_stream_done_at else 0
            )
            metrics.tts_time = worker.tts_time
            metrics.playback_time = worker.playback_time
            metrics.sentences_spoken = worker.sentences_spoken
            metrics.fillers_played = worker.fillers_played
            metrics.barge_in = coord.is_set()
            metrics.response = full_response.strip()
            metrics.total_e2e = self._clock() - turn_start

            messages.append({"role": "assistant", "content": metrics.response})

            for err in worker.errors:
                self._print(f"  {_YELLOW}worker error: {err}{_RESET}")

        except Exception as e:
            had_error = True
            self._print(f"\n  {_YELLOW}LLM error: {e}{_RESET}")
            messages.pop()  # remove the user message we appended
            watcher.stop(timeout=2.0)
            if watcher.detected:
                next_primed = list(watcher.frames)
                self._print(
                    f"  {_DIM}barge-in during failed LLM call: "
                    f"replaying {len(next_primed)} captured frames "
                    f"({len(next_primed) * self._chunk / self._rate:.1f}s){_RESET}"
                )
            worker.stop(timeout=self._cancel_wait_timeout)
            drained = flush_pending_audio(self._mic, chunk_size=self._chunk)
            if drained:
                self._print(
                    f"  {_DIM}flushed {drained} stale audio frames "
                    f"({drained / self._rate:.1f}s){_RESET}"
                )
            return TurnResult(
                metrics=None,
                next_primed_frames=next_primed,
                had_error=True,
            )

        finally:
            # iter-013: explicit close so the upstream HTTP
            # response is released promptly even on barge-in /
            # error paths. Idempotent.
            try:
                llm_gen.close()
            except Exception:
                pass

        return TurnResult(
            metrics=metrics,
            next_primed_frames=next_primed,
            had_error=False,
        )

    @staticmethod
    def trim_messages(messages: list[dict], max_user_assistant: int = 20) -> list[dict]:
        """Convenience pass-through to the helper used by run_chat."""
        return trim_history(messages, max_user_assistant=max_user_assistant)
