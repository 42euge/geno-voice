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
    MIN_SPEECH_DURATION,
    RATE,
    SILENCE_DURATION,
    SILENCE_THRESHOLD,
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
        # VAD tuning (iter-020). silence_duration also used to
        # compute speech_ended_at for TTFS measurement.
        silence_threshold: float = SILENCE_THRESHOLD,
        silence_duration: float = SILENCE_DURATION,
        min_speech_duration: float = MIN_SPEECH_DURATION,
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
        # iter-088: aggressive first-sentence splitter. When True,
        # the splitter accepts comma+whitespace as a terminator for
        # the FIRST sentence only, reducing TTFS on long-preamble
        # responses at the cost of some prosody. Strict splitter
        # resumes for sentence 2+. Default False — opt in via
        # chat.aggressive_first_sentence in config.local.yaml.
        aggressive_first_sentence: bool = False,
        # iter-093: auto-aggressive on stall. When >0, a mid-stream
        # token gap exceeding this threshold (seconds) flips the
        # splitter into aggressive mode mid-turn even if the static
        # aggressive_first_sentence config was False. Rationale:
        # the user has waited longer than expected; getting some
        # audio out faster (with imperfect prosody) beats continued
        # silence. Default 0.0 = disabled. Recommended 0.5-1.0s —
        # comfortably above normal token-streaming jitter, well
        # below the user's "is this broken" threshold.
        auto_aggressive_threshold: float = 0.0,
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
        self._silence_threshold = silence_threshold
        self._silence_duration = silence_duration
        self._min_speech_duration = min_speech_duration

        self._stt_engine = stt_engine
        self._transcribe_fn = transcribe_fn

        self._llm_stream_fn = llm_stream_fn
        self._llm_config = llm_config

        self._synth_fn = synth_fn
        self._play_fn = play_fn

        self._fillers = list(fillers) if fillers else []
        self._idle_threshold = idle_threshold
        # iter-113: cross-turn filler variety. Bounded FIFO of recently-
        # played filler IDs, threaded down to the per-turn SentenceWorker
        # so the picker can prefer fillers NOT recently used. maxlen
        # is set to len(fillers) - 1 (or 1 minimum) — keeps "the last
        # one" out of the picker's preferred set, but allows everything
        # to cycle when there are 2+ fillers. With only one filler
        # configured, the FIFO is irrelevant (picker has nothing to
        # vary).
        from collections import deque as _deque
        n_fillers = len(self._fillers)
        self._recent_filler_ids = (
            _deque(maxlen=max(1, n_fillers - 1))
            if n_fillers > 0
            else None
        )
        # iter-088: aggressive first-sentence splitter config.
        self._aggressive_first_sentence = aggressive_first_sentence
        # iter-093: auto-aggressive-on-stall threshold (seconds).
        self._auto_aggressive_threshold = auto_aggressive_threshold

        self._clock = clock
        self._output = output  # passed to record_utterance_streaming
        self._wait_done_timeout = wait_done_timeout
        self._cancel_wait_timeout = cancel_wait_timeout
        # iter-082: cross-turn state for the TTC proxy. Records the
        # ``worker.first_audio_at`` of the previous turn so the
        # NEXT turn can compute "user speech start - prev bot first
        # audio" = how long the user listened before responding.
        # None until turn 1 produces audio.
        self._last_first_audio_at: Optional[float] = None

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
        # iter-063: collect side-band metrics (currently EoT detection
        # latency) via the new ``out_metrics`` parameter. Old return
        # signature is unchanged.
        rec_metrics: dict = {}
        wav_bytes, speech_dur, stt_time = record_utterance_streaming(
            self._mic,
            self._stt_engine,
            transcribe_fn=self._transcribe_fn,
            clock=self._clock,
            output=self._output,
            primed_frames=primed_frames,
            silence_threshold=self._silence_threshold,
            silence_duration=self._silence_duration,
            min_speech_duration=self._min_speech_duration,
            out_metrics=rec_metrics,
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
        # iter-063: copy the EoT detection latency over from
        # record_utterance_streaming's side-band dict. Defaults to 0.0
        # when the recorder didn't emit (DONE_TOO_SHORT path), which
        # the per-turn print + session aggregate both filter on.
        metrics.eot_latency = float(rec_metrics.get("eot_latency", 0.0))
        # iter-072: STT preview-vs-final divergence (taxonomy 1.8).
        # Same dict; default 0.0 when the recorder didn't populate
        # (no preview emerged or final empty).
        metrics.stt_preview_divergence = float(
            rec_metrics.get("stt_preview_divergence", 0.0)
        )
        # iter-082: TTC (time-to-comprehension) proxy. Cross-turn
        # gap from the PREVIOUS turn's first bot audio to THIS
        # turn's first speech-detected frame. Captures "how long
        # did the user listen before responding." Skip on turn 1
        # (no prev) and on turns where the recorder didn't emit
        # speech_start_at.
        speech_start_at = rec_metrics.get("speech_start_at")
        if (
            self._last_first_audio_at is not None
            and speech_start_at is not None
        ):
            ttc = speech_start_at - self._last_first_audio_at
            # Negative TTC is a clock-skew artifact (the recorder
            # uses its own t_origin clock; the worker uses
            # self._clock); clamp at 0.
            metrics.time_to_comprehension = max(0.0, ttc)
        # iter-065: trailing-silence wall. The part of EoT NOT
        # explained by the configured silence_duration. ``max(0, ...)``
        # because the EoT measurement uses the actual last-speech
        # frame timestamp while VadState's silence window starts one
        # frame later (the first sub-threshold frame); on rare turns
        # this gap is a hair smaller than silence_duration. Clamp
        # negative numbers to 0 so the metric stays interpretable.
        if metrics.eot_latency > 0:
            metrics.eot_overhead = max(
                0.0, metrics.eot_latency - self._silence_duration
            )
        speech_ended_at = self._clock() - self._silence_duration
        turn_start = self._clock()
        metrics.stt_time = stt_time
        # iter-049: STT real-time factor. Only meaningful when
        # speech_duration > 0 — guard against div-by-zero.
        if speech_dur > 0:
            metrics.stt_rtf = stt_time / speech_dur
        metrics.transcript = text.strip()
        # iter-064: user speaking rate. Symmetric to iter-046's
        # bot_wpm. Whitespace-split word count is a decent proxy —
        # Whisper transcripts use space-separated tokens, and the
        # error versus a true tokenization is dwarfed by natural
        # variance in human speech rates.
        if speech_dur > 0:
            n_words = len(metrics.transcript.split())
            if n_words > 0:
                metrics.user_wpm = (n_words / speech_dur) * 60.0

        # ---- Phase 2: LLM stream + worker + watcher ----
        messages.append({"role": "user", "content": metrics.transcript})
        # iter-077: count approximate context tokens being sent to
        # the LLM. Whitespace-split is a rough but consistent
        # estimator — actual tokenizer-aware counts vary by model
        # but the per-turn TREND is what matters here. LLM TTFB
        # scales with input context; without aggressive trimming
        # late-session turns get progressively slower.
        metrics.context_tokens = sum(
            len(str(m.get("content", "")).split())
            for m in messages
        )

        worker = SentenceWorker(
            speaker_factory=self._speaker_factory,
            synth_fn=self._synth_fn,
            play_fn=self._play_fn,
            fillers=self._fillers,
            idle_threshold=self._idle_threshold if self._fillers else 0.0,
            clock=self._clock,
            # iter-113: pass the loop-level FIFO so the picker can
            # avoid clips played in recent turns.
            recent_filler_ids=self._recent_filler_ids,
        )
        worker.start()

        # iter-037: capture the drained count so we can surface it
        # via TurnMetrics. Metric 2.19 in the perf-metrics taxonomy.
        # Many stale frames each turn means the mic accumulated bot
        # audio between turns (acoustic echo / OS loopback / Bluetooth
        # duplex). Reliable signal the user needs echo cancellation.
        stale_frames = flush_pending_audio(self._mic, chunk_size=self._chunk)

        llm_gen = self._llm_stream_fn(messages, self._llm_config)
        # iter-030: pass the same clock so ``coord.triggered_at`` is
        # comparable against ``llm_stream_done_at`` below (both sampled
        # from ``self._clock``). Without this, mocked-clock tests
        # couldn't verify the phase decision.
        coord = BargeInCoordinator(worker=worker, clock=self._clock)
        # iter-028: build the watcher's VadState from the same VAD
        # config that record_utterance_streaming uses, so a user who
        # tunes ``chat.vad.silence_threshold`` for a noisy room gets
        # the threshold applied to barge-in detection too. Without
        # this, the watcher kept defaults — false barge-ins from
        # background noise even when the recorder was tuned to ignore
        # it.
        from examples._chat_helpers import VadState
        watcher = BargeInWatcher(
            mic=self._mic,
            on_speech_detected=coord.trigger,
            chunk_size=self._chunk,
            rate=self._rate,
            vad=VadState(
                silence_threshold=self._silence_threshold,
                silence_duration=self._silence_duration,
                min_speech_duration=self._min_speech_duration,
            ),
        )
        watcher.start()

        llm_start = self._clock()
        first_token_at: Optional[float] = None
        # iter-038: stamp the moment the first complete sentence
        # leaves the splitter — i.e. the moment TTS can start. This
        # is distinct from first_token_at: the LLM may stream chatty
        # preamble for a while before terminating a sentence, which
        # delays TTFS even if first-token was fast. Metric 1.10 in
        # the perf-metrics taxonomy.
        first_sentence_at: Optional[float] = None
        token_buffer = ""
        full_response = ""
        llm_stream_done_at: Optional[float] = None
        next_primed: Optional[list] = None
        had_error = False
        # iter-045: accumulate sentence character lengths so we can
        # report the mean as a fragmentation diagnostic. Track total
        # + count separately so the average is computed once at end.
        sentence_chars_total = 0
        sentence_chars_count = 0
        # iter-070: also track min/max for the per-turn range. Mean
        # alone hides bimodal patterns ("Yes." + a 150-char sentence
        # both look fine at mean=80). ``None`` for min until the
        # first sentence so the first observation can land regardless
        # of size.
        sentence_min_chars: int | None = None
        sentence_max_chars = 0
        # iter-059: split coverage — chars submitted as complete
        # sentences (overlap-friendly) vs the trailing remainder
        # forced through at end-of-stream (can't overlap).
        complete_sentence_chars = 0
        remainder_chars = 0
        # iter-052: count tokens received from the LLM. Used to
        # compute TPS (tokens/sec) post-stream.
        token_count = 0
        # iter-085: track inter-token gap to catch mid-stream LLM
        # stalls. Only the FIRST gap (first_token wait) is excluded
        # — that's already covered by iter-052's llm_first_token.
        # All subsequent gaps feed into max_token_gap.
        prev_token_at: Optional[float] = None
        max_token_gap = 0.0
        # iter-088: track whether the aggressive first-sentence
        # splitter is still active for this turn. Starts True only
        # if the loop's config enabled it; flips False as soon as
        # the splitter returns ANY complete sentence (so subsequent
        # iterations use strict splitting).
        aggressive_active = self._aggressive_first_sentence

        try:
            for token in llm_gen:
                if coord.is_set():
                    break  # iter-012: barge-in during LLM streaming
                now = self._clock()
                if first_token_at is None:
                    first_token_at = now
                else:
                    # iter-085: gap between consecutive tokens.
                    # Skipped for the first token (prev is None
                    # at that point — first_token_at has the
                    # initial timestamp instead).
                    if prev_token_at is not None:
                        gap = now - prev_token_at
                        if gap > max_token_gap:
                            max_token_gap = gap
                        # iter-093: auto-aggressive on stall. If
                        # the inter-token gap exceeded the
                        # configured threshold AND we haven't
                        # produced a complete sentence yet, flip
                        # the splitter to aggressive mode so the
                        # NEXT iteration can comma-split. Once
                        # first_sentence_at is set, the strict
                        # splitter is fine again — we already got
                        # audio out. Threshold of 0.0 means
                        # disabled (pre-iter-093 behavior).
                        if (
                            self._auto_aggressive_threshold > 0
                            and gap > self._auto_aggressive_threshold
                            and not aggressive_active
                            and first_sentence_at is None
                        ):
                            aggressive_active = True
                prev_token_at = now
                token_buffer += token
                full_response += token
                token_count += 1

                complete, token_buffer = split_complete_sentences(
                    token_buffer,
                    aggressive_first=aggressive_active,
                )
                if complete:
                    # iter-088: first sentence(s) emerged — flip off
                    # aggressive splitting so subsequent iterations
                    # require strict ``.!?`` terminators.
                    aggressive_active = False
                    if first_sentence_at is None:
                        first_sentence_at = self._clock()
                for sentence in complete:
                    sentence_chars_total += len(sentence)
                    sentence_chars_count += 1
                    # iter-070: per-turn min/max range tracking.
                    n = len(sentence)
                    if sentence_min_chars is None or n < sentence_min_chars:
                        sentence_min_chars = n
                    if n > sentence_max_chars:
                        sentence_max_chars = n
                    # iter-059: track complete-sentence chars
                    # separately to compute split coverage at end.
                    complete_sentence_chars += len(sentence)
                    worker.submit(sentence)

            llm_stream_done_at = self._clock()

            if not coord.is_set():
                remaining = token_buffer.strip()
                if remaining:
                    sentence_chars_total += len(remaining)
                    sentence_chars_count += 1
                    # iter-070: include the trailing remainder in
                    # range tracking — it's a real submitted unit
                    # the worker has to synthesize.
                    n = len(remaining)
                    if sentence_min_chars is None or n < sentence_min_chars:
                        sentence_min_chars = n
                    if n > sentence_max_chars:
                        sentence_max_chars = n
                    # iter-059: trailing remainder — can't overlap
                    # with anything (synth happens after stream done).
                    remainder_chars += len(remaining)
                    worker.submit(remaining)
                worker.submit_done()
                worker.wait_done(timeout=self._wait_done_timeout)
            else:
                worker.wait_done(timeout=self._cancel_wait_timeout)

            watcher.stop(timeout=2.0)
            # iter-074: bargeable-time fraction. Of the time the
            # bot was producing audio, what fraction was the
            # watcher active (i.e. barge-in was actually possible)?
            # 1.0 is the architectural default — watcher.start
            # precedes worker.first_audio_at and watcher.stop is
            # called right after worker.wait_done. Sentinel: a
            # future change that pauses the watcher (e.g. during
            # fillers) would push this below 1.0 — visible
            # regression. Skip when no audio played.
            if (
                worker.first_audio_at is not None
                and watcher.started_at is not None
                and watcher.stopped_at is not None
                and watcher.stopped_at > worker.first_audio_at
            ):
                bot_speech_dur = watcher.stopped_at - worker.first_audio_at
                inter_start = max(watcher.started_at, worker.first_audio_at)
                inter_end = watcher.stopped_at
                intersection = max(0.0, inter_end - inter_start)
                metrics.bargeable_fraction = min(
                    1.0, intersection / bot_speech_dur
                )
            if watcher.detected:
                next_primed = list(watcher.frames)
                # iter-047: structured phase. Used both for the
                # diagnostic string AND for the metric on TurnMetrics
                # (where the value is "llm_stream" or "playback" —
                # the user-facing string adds " phase" suffix).
                phase_key = (
                    "llm_stream"
                    if coord.triggered_at is not None
                    and llm_stream_done_at is not None
                    and coord.triggered_at < llm_stream_done_at
                    else "playback"
                )
                metrics.barge_in_phase = phase_key
                # iter-057: primed-frames replay duration as a metric.
                metrics.primed_frames_seconds = (
                    len(next_primed) * self._chunk / self._rate
                )
                phase = f"{phase_key.replace('_', '-')} phase"
                self._print(
                    f"\n  {_DIM}barge-in during {phase}: replaying "
                    f"{len(next_primed)} captured frames "
                    f"({metrics.primed_frames_seconds:.1f}s){_RESET}"
                )

            # Populate metrics from the worker.
            if worker.first_audio_at is not None:
                # iter-082: stash this turn's first-audio timestamp
                # so the NEXT turn's TTC computation has its
                # left-hand operand. Update only when audio
                # actually played — silent turns don't anchor TTC.
                self._last_first_audio_at = worker.first_audio_at
                metrics.ttfs = worker.first_audio_at - speech_ended_at
                # iter-053: bucket TTFS against the human-conversation
                # sweet spot. <200ms feels rushed (bot interrupted the
                # natural pause); 200-400ms is comfortable; >400ms is
                # noticeable lag.
                ttfs_ms = metrics.ttfs * 1000
                if ttfs_ms < 200:
                    metrics.naturalness_bucket = "rushed"
                elif ttfs_ms <= 400:
                    metrics.naturalness_bucket = "natural"
                else:
                    metrics.naturalness_bucket = "slow"
                # iter-076: TTFS attribution residual = ttfs minus
                # the stt and LLM-to-first-sentence terms. Captures
                # everything from "first complete sentence reached
                # the worker" through "first audio chunk played":
                # synth, queue dispatch, audio device buffering.
                # Defensively clamps negative residuals (which
                # would mean parts add to more than the whole —
                # only happens via microsecond clock-skew between
                # the recorder and loop clocks).
                llm_first_sent_local = (
                    (first_sentence_at - llm_start) if first_sentence_at else 0.0
                )
                metrics.synth_dispatch_seconds = max(
                    0.0,
                    metrics.ttfs - stt_time - llm_first_sent_local,
                )
            metrics.llm_first_token = (
                (first_token_at - llm_start) if first_token_at else 0
            )
            # iter-083: first-token-to-audio gap. Both ends must
            # exist — first_token_at (LLM produced anything) AND
            # worker.first_audio_at (any sentence reached the
            # speaker). Clamp to 0 against tiny clock-skew negatives.
            if (
                first_token_at is not None
                and worker.first_audio_at is not None
            ):
                metrics.first_token_to_audio = max(
                    0.0, worker.first_audio_at - first_token_at
                )
            # iter-085: max inter-token gap during the LLM stream.
            # Only meaningful when token_count >= 2 (need at least
            # two tokens to have a gap). 0 on single-token responses.
            metrics.max_token_gap = max_token_gap
            # iter-052: LLM TPS — tokens/sec measured AFTER first
            # token (excludes first-token wait). Need ≥2 tokens
            # AND a positive interval. (token_count - 1) tokens
            # were received over (done - first_token_at) seconds.
            if (
                token_count >= 2
                and first_token_at is not None
                and llm_stream_done_at is not None
                and llm_stream_done_at > first_token_at
            ):
                metrics.llm_tps = (
                    (token_count - 1)
                    / (llm_stream_done_at - first_token_at)
                )
            # iter-038: time from LLM start to the first complete
            # sentence reaching the worker. 0 if no complete sentence
            # ever emerged (LLM yielded fragments only, or stream was
            # cut off before a terminator).
            metrics.llm_first_sentence = (
                (first_sentence_at - llm_start) if first_sentence_at else 0
            )
            metrics.llm_total = (
                (llm_stream_done_at - llm_start) if llm_stream_done_at else 0
            )
            # iter-043: streaming overlap ratio. Defined as the
            # fraction of the LLM-stream window during which audio
            # was already playing. Only computable when both ends
            # of each interval exist — first_audio_at on the worker
            # (audio actually started) and llm_stream_done_at
            # (LLM finished). Clamps:
            #   - If first_audio_at is None (no audio played):
            #     ratio = 0 (no overlap by definition).
            #   - If first_audio_at >= llm_stream_done_at (audio
            #     started AFTER LLM finished, fully sequential):
            #     ratio = 0.
            #   - If llm_total is 0 (no LLM time, edge case):
            #     ratio = 0.
            #   - Else ratio in (0, 1].
            if (
                worker.first_audio_at is not None
                and llm_stream_done_at is not None
                and metrics.llm_total > 0
            ):
                overlap = max(0.0, llm_stream_done_at - worker.first_audio_at)
                metrics.streaming_overlap_ratio = min(
                    1.0, overlap / metrics.llm_total
                )
            # iter-073: first-sentence overlap savings — how much
            # of the FIRST synth was masked by ongoing LLM streaming.
            # Distinct from the iter-043 whole-stream ratio: this
            # scopes to the first sentence specifically because
            # that's what gates TTFS. Standard interval-overlap:
            #     overlap = max(0, min(synth_done, llm_done) -
            #                       max(synth_start, llm_start))
            # 0 means first synth ran entirely after LLM finished
            # (sequential — streaming bought nothing for TTFS).
            # Equal to first-synth duration means first synth ran
            # entirely under LLM streaming (best case — synth was
            # fully masked).
            if (
                worker.first_synth_start_at is not None
                and worker.first_synth_done_at is not None
                and llm_stream_done_at is not None
            ):
                overlap_first = max(
                    0.0,
                    min(worker.first_synth_done_at, llm_stream_done_at)
                    - max(worker.first_synth_start_at, llm_start),
                )
                metrics.first_synth_overlap_seconds = overlap_first
            # iter-044: cumulative between-sentence idle gap.
            metrics.worker_idle_gap_total = worker.idle_gap_total
            # iter-045: mean character length of sentences submitted.
            if sentence_chars_count > 0:
                metrics.mean_sentence_chars = (
                    sentence_chars_total / sentence_chars_count
                )
            # iter-070: per-turn min/max sentence lengths. Both
            # default to 0; populating only when at least one
            # sentence was submitted keeps the "no submissions"
            # signal distinguishable from "all submissions had
            # length 0" (which can't happen — empty sentences are
            # filtered upstream).
            if sentence_min_chars is not None:
                metrics.min_sentence_chars = sentence_min_chars
                metrics.max_sentence_chars = sentence_max_chars
            # iter-059: sentence-split coverage. Only meaningful
            # when at least some chars were submitted to the worker.
            total_submitted = complete_sentence_chars + remainder_chars
            if total_submitted > 0:
                metrics.sentence_split_coverage = (
                    complete_sentence_chars / total_submitted
                )
            # iter-046: bot WPM. Both components must be >0 — a
            # turn with no audio (worker errored, no sentences
            # submitted) or no words (alignment broken) leaves
            # bot_wpm at 0.
            if (
                worker.audio_seconds_total > 0
                and worker.word_count_total > 0
            ):
                metrics.bot_wpm = (
                    worker.word_count_total
                    / (worker.audio_seconds_total / 60.0)
                )
            metrics.tts_time = worker.tts_time
            # iter-050: TTS real-time factor. Same shape as iter-049's
            # STT RTF — guard div-by-zero, only meaningful when audio
            # was produced.
            if worker.audio_seconds_total > 0:
                metrics.tts_rtf = worker.tts_time / worker.audio_seconds_total
            metrics.playback_time = worker.playback_time
            # iter-061: speaker-open overhead (taxonomy 2.8). The worker
            # only opens the speaker once (first turn) — on later turns
            # it's a no-op and the field stays at 0.0 for those metrics.
            metrics.speaker_open_seconds = worker.speaker_open_seconds
            # iter-062: peak queue depth (taxonomy 2.7). Sampled in
            # SentenceWorker.submit() after each put.
            metrics.max_queue_depth = worker.max_queue_depth
            # iter-071: token-reveal lag — worker accumulates raw
            # sums; ChatLoop computes the mean at the turn boundary
            # so the denominator is stable.
            if worker.token_reveal_lag_count > 0:
                metrics.mean_token_reveal_lag = (
                    worker.token_reveal_lag_sum
                    / worker.token_reveal_lag_count
                )
                metrics.max_token_reveal_lag = worker.token_reveal_lag_max
            metrics.sentences_spoken = worker.sentences_spoken
            # iter-040: count of sentences cut mid-stream by cancel_event.
            metrics.sentences_cancelled = worker.cancelled_sentences
            metrics.fillers_played = worker.fillers_played
            # iter-081: filler clip ID for session-wide diversity
            # aggregation. 0 when no filler played this turn.
            if worker.last_filler_id is not None:
                metrics.last_filler_id = worker.last_filler_id
            # iter-051: filler false-positive flag. The filler is
            # unnecessary if the LLM's first token actually arrived
            # before the idle_threshold window would have elapsed.
            # Only meaningful when a filler actually played AND
            # both first_token + threshold are positive — otherwise
            # the comparison is undefined.
            if (
                metrics.fillers_played > 0
                and self._idle_threshold > 0
                and 0 < metrics.llm_first_token < self._idle_threshold
            ):
                metrics.filler_false_positive = True
            metrics.barge_in = coord.is_set()
            # iter-041: barge-in latency from coordinator timestamps.
            # Both pieces only valid when the trigger actually fired.
            if (
                coord.triggered_at is not None
                and coord.playback_stopped_at is not None
            ):
                metrics.barge_in_latency = max(
                    0.0,
                    coord.playback_stopped_at - coord.triggered_at,
                )
            # iter-056: regret flag. The user started speaking within
            # 200ms of bot first audio — implies the bot misjudged
            # end-of-turn and pre-empted the user. Different signal
            # than iter-053's "rushed" naturalness: rushed measures
            # the bot's response latency from the user's perspective;
            # regret measures whether the bot interrupted real speech.
            if (
                coord.triggered_at is not None
                and worker.first_audio_at is not None
                and coord.is_set()
            ):
                gap = coord.triggered_at - worker.first_audio_at
                if 0 < gap < 0.2:
                    metrics.barge_in_regret = True
            metrics.mic_stale_frames = stale_frames
            metrics.response = full_response.strip()
            # iter-080: pre-empted-content loss. Compute on barge
            # turns only — non-barge turns naturally have small
            # diffs from splitter remainder + alignment edge cases
            # that aren't really "lost content."
            if coord.is_set():
                response_words = len(metrics.response.split())
                played_words = worker.word_count_total
                metrics.preempted_words = max(
                    0, response_words - played_words
                )
            metrics.total_e2e = self._clock() - turn_start

            messages.append({"role": "assistant", "content": metrics.response})

            # iter-058: lift worker error count to TurnMetrics.
            metrics.worker_errors = len(worker.errors)
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
            # iter-060: time the close so we can report
            # cancel-to-close latency on barge turns. Captures the
            # time between coord.trigger() and llm_gen.close()
            # completing — high values mean the HTTP socket is
            # winding down slowly.
            close_started_at = self._clock()
            try:
                llm_gen.close()
            except Exception:
                pass
            close_finished_at = self._clock()
            # iter-027: also stop worker + watcher in finally so a
            # KeyboardInterrupt during the for-token loop cleans
            # them up. ``except Exception`` doesn't catch
            # KeyboardInterrupt (which inherits from BaseException),
            # so without these the worker thread keeps running with
            # the speaker open until the daemon thread dies on
            # process exit. Idempotent — already-stopped workers /
            # watchers no-op.
            try:
                watcher.stop(timeout=1.0)
            except Exception:
                pass
            try:
                worker.stop(timeout=self._cancel_wait_timeout)
            except Exception:
                pass

        # iter-060: only meaningful on barge turns — populate
        # llm_cancel_to_close as the gap between trigger and the
        # close() finishing. Both timestamps were captured during
        # the finally block above. Non-barge turns leave the field
        # at 0 (also handles "close was instantaneous" — sub-ms).
        if (
            coord.triggered_at is not None
            and coord.is_set()
        ):
            metrics.llm_cancel_to_close = max(
                0.0, close_finished_at - coord.triggered_at,
            )
        return TurnResult(
            metrics=metrics,
            next_primed_frames=next_primed,
            had_error=False,
        )

    @staticmethod
    def trim_messages(messages: list[dict], max_user_assistant: int = 20) -> list[dict]:
        """Convenience pass-through to the helper used by run_chat."""
        return trim_history(messages, max_user_assistant=max_user_assistant)
