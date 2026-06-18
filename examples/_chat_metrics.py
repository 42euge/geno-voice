"""Per-turn metrics struct + printer extracted from mic_chat.

Lives in its own module so tests can import ``TurnMetrics`` without
pulling in mic_chat's top-level ``import pyaudio`` (which is
unavailable on x86_64 Linux without ALSA dev headers, and on most
CI runners).

Same pattern as iter-006/007 — pull pure-Python primitives out of
the pyaudio-bound entry point.

Also hosts ``print_session_summary`` (iter-017): the
KeyboardInterrupt summary block previously inlined in
``mic_chat.run_chat``, now testable + using ``statistics.median``
for proper even-length handling.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional

# ANSI codes — duplicated from mic_chat so this module remains a
# clean leaf with no dependency back on mic_chat itself.
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


@dataclass
class TurnMetrics:
    speech_duration: float = 0.0
    stt_time: float = 0.0
    # iter-049: STT real-time factor — stt_time / speech_duration.
    # <1 = STT runs faster than realtime (can be invoked inline at
    # end-of-turn). >1 = STT is the bottleneck and needs streaming
    # partial transcription to overlap with speech. Mlx-whisper-large
    # on Apple Silicon lands ~0.1-0.3 on M-series. 0 = either
    # speech_duration was 0 (false trigger turn — though those don't
    # produce TurnMetrics post-iter-031) or stt_time wasn't measured.
    # Metric 1.7 in the perf-metrics taxonomy.
    stt_rtf: float = 0.0
    # iter-072: STT preview-vs-final divergence in [0.0, 1.0].
    # 0 = the live preview transcript matched the final perfectly
    # (incremental Whisper output was already correct — live STT
    # was useful). 1 = totally different — the user had to wait for
    # the final transcript to know if they were understood. >0.3
    # is "preview UX is broken." Computed via
    # difflib.SequenceMatcher in record_utterance_streaming.
    # Metric 1.8 in the perf-metrics taxonomy.
    stt_preview_divergence: float = 0.0
    llm_first_token: float = 0.0
    # iter-052: stream throughput of the LLM in tokens/sec, measured
    # AFTER first token (so the first-token wait doesn't bias TPS
    # downward). Local 7B-13B models on Apple Silicon land 30-80 tps;
    # cloud APIs typically 20-60 tps. Directly gates how fast
    # complete sentences arrive at the worker. Metric 1.9 in the
    # perf-metrics taxonomy.
    llm_tps: float = 0.0
    # iter-038: time from LLM start to the first complete sentence
    # reaching the TTS worker. Distinct from llm_first_token: the
    # LLM may stream chatty preamble for a while before a terminator
    # arrives. Metric 1.10 in the perf-metrics taxonomy. 0 means
    # no complete sentence ever emerged this turn (rare).
    llm_first_sentence: float = 0.0
    llm_total: float = 0.0
    tts_time: float = 0.0
    # iter-050: TTS real-time factor — tts_time / audio_seconds_total.
    # Symmetric to iter-049's STT RTF. <1 = synth runs faster than
    # the audio it produces (can stream overlap usefully); >1 = TTS
    # is the bottleneck and synth-overlap won't help. Kokoro on
    # Apple Silicon lands ~0.1-0.3. 0 = no audio produced this turn.
    # Metric 1.11 in the perf-metrics taxonomy.
    tts_rtf: float = 0.0
    playback_time: float = 0.0
    # iter-043: fraction of the LLM-stream window during which the
    # worker was already playing audio (vs synth + play happening
    # serially after the stream completed). 1.0 = first audio
    # landed at llm_start (impossible — there's at least the
    # first-sentence wait + first synth). Realistic 0.4-0.8 on
    # multi-sentence responses with a fast-enough LLM. 0 means
    # the worker only played AFTER the LLM finished — sequential,
    # iter-008 streaming-overlap not buying us anything that turn.
    # Metric 2.1 in the perf-metrics taxonomy.
    streaming_overlap_ratio: float = 0.0
    # iter-073: first-sentence overlap savings — seconds of first
    # synth that ran concurrently with LLM streaming. Distinct from
    # streaming_overlap_ratio (whole-stream ratio): this scopes to
    # the FIRST sentence because that's what gates TTFS. 0 = first
    # synth was entirely sequential with LLM (no TTFS savings from
    # the iter-008 streaming-sentence-dispatch design). Equal to
    # first-synth duration = first synth was fully masked.
    # Metric 2.2 in the perf-metrics taxonomy.
    first_synth_overlap_seconds: float = 0.0
    # iter-074: bargeable-time fraction in [0.0, 1.0]. Of the
    # window during which the bot was producing audio, what
    # fraction had the BargeInWatcher active? 1.0 is the
    # architectural default — watcher starts before bot speech
    # and stops right after. Anything below 1.0 means the bot
    # was functionally uninterruptible for some fraction of its
    # speech (would happen if a future change paused the watcher
    # mid-turn — e.g. during fillers). Sentinel for that regression.
    # 0.0 default = no audio played this turn or watcher lifecycle
    # didn't fire. Metric 1.19 in the perf-metrics taxonomy.
    bargeable_fraction: float = 0.0
    # iter-044: cumulative seconds the SentenceWorker spent blocked
    # waiting for the next sentence, AFTER the first sentence
    # (excludes TTFsent). High idle gap = LLM didn't keep up with
    # synth+playback. Combined with streaming_overlap_ratio,
    # localizes pipeline bottlenecks. Metric 2.16 in the
    # perf-metrics taxonomy.
    worker_idle_gap_total: float = 0.0
    ttfs: float = 0.0
    # iter-076: TTFS attribution residual — the part of TTFS not
    # already accounted for by stt_time + llm_first_sentence.
    # Captures everything from "first complete sentence reached
    # the worker" through "first audio chunk played": synth time,
    # speaker open, queue dispatch, audio device buffering.
    # Computed as ``max(0, ttfs - stt_time - llm_first_sentence)``;
    # the clamp is defensive against tiny clock-skew negatives.
    # Combined with stt_time + llm_first_sentence, the three terms
    # decompose TTFS into a 100% breakdown. Metric 2.22 in the
    # perf-metrics taxonomy.
    synth_dispatch_seconds: float = 0.0
    # iter-077: approximate count of context tokens sent to the
    # LLM this turn (whitespace-split estimator across the entire
    # messages list). The per-turn TREND is the actionable signal:
    # late-session turns get progressively slower as context
    # grows, so a creep here predicts an LLM TTFB regression even
    # before llm_first_token measurably worsens. Pairs with
    # iter-024's trim_history(max_user_assistant=20) — if context
    # keeps growing despite the trim cap, system-prompt bloat is
    # the culprit. Metric 2.23 in the perf-metrics taxonomy.
    context_tokens: int = 0
    # iter-080: words the LLM generated but the user never heard
    # (cancelled mid-stream by a barge). Computed on barge turns
    # only: ``max(0, len(response.split()) - worker.word_count_total)``.
    # 0 on non-barge turns and on barge turns where the cut
    # happened cleanly between sentences (no words were lost).
    # High values pair with iter-069's interruption rate and
    # iter-047's barge-in phase to localize the cause: "bot was
    # being verbose, user cut it off" vs "bot was being slow, user
    # got impatient before any audio." Metric 3.7 in the perf-
    # metrics taxonomy ("Novel/speculative").
    preempted_words: int = 0
    # iter-053: TTFS bucketed against the human-conversation
    # sweet spot. "rushed" (<200ms): bot interrupted natural
    # turn-taking pause; "natural" (200-400ms): matches human
    # conversational rhythm; "slow" (>400ms): user notices
    # latency. Counter-intuitive: lower TTFS isn't always better.
    # "" when no audio played this turn. Metric 3.1 in the
    # perf-metrics taxonomy ("Novel/speculative").
    naturalness_bucket: str = ""
    # iter-105: Word Error Rate against an optional reference
    # transcript, computed via examples._chat_wer.compute_wer.
    # 0.0 means "not measured" — the field defaults silent
    # because most turns don't have a ground-truth reference.
    # When >= 0 AND a reference is supplied at the call site,
    # this carries the per-turn WER for the session-summary
    # aggregator (median + max). Audio-fixture corpus is a
    # follow-up iteration; this field exists so future eval runs
    # can populate it without changing the dataclass shape.
    wer: float = 0.0
    # iter-105: True when wer is meaningful (a reference was
    # supplied this turn). Distinguishes "0.0 = perfect" from
    # "0.0 = not measured" — the session-summary helper filters
    # on this flag, not on wer == 0.0.
    wer_measured: bool = False
    # iter-056: True when the user barged in within 200ms of bot
    # first audio. Implies the bot pre-empted the user — the user
    # was already mid-utterance when bot speech started, suggesting
    # iter-001's end-of-turn detection (or iter-020's silence_duration
    # config) fired too early. Distinct from "rushed" naturalness:
    # rushed = bot felt fast (subjective, latency-based); regret =
    # user was actually still talking. Metric 3.4 in the perf-metrics
    # taxonomy ("Novel/speculative").
    barge_in_regret: bool = False
    # iter-060: time between BargeInCoordinator.trigger() firing and
    # the LLM HTTP stream's `.close()` actually returning. Only set
    # on barge turns. High value (>500ms) means the upstream HTTP
    # connection is taking a long time to wind down — wastes tokens
    # we paid for and can block the next turn. Metric 2.14 in the
    # perf-metrics taxonomy.
    llm_cancel_to_close: float = 0.0
    # iter-057: audio seconds carried over via next_primed_frames
    # into the next turn. Validates iter-025 lead-in: how much of
    # the user's first words would have been lost without the
    # watcher's frame buffer. 0 on non-barge turns. Metric 2.12 in
    # the perf-metrics taxonomy.
    primed_frames_seconds: float = 0.0
    # iter-058: count of worker errors observed during this turn —
    # synth failures, play_fn raises, speaker_factory crashes. Lifted
    # from len(worker.errors) at turn end. A non-zero count means
    # the turn produced PARTIAL audio (some sentences succeeded,
    # others raised). Distinct from session-level LLM errors which
    # kill the entire turn. Metric 1.16 in the perf-metrics taxonomy.
    worker_errors: int = 0
    total_e2e: float = 0.0
    sentences_spoken: int = 0
    # iter-045: mean character length of sentences submitted to the
    # worker this turn. Diagnostic for splitter fragmentation:
    # mean ≪ ~30 means lots of short fragments ("Yes.", "I see.")
    # which synth fast but defeat streaming-overlap; mean ≫ ~150
    # means run-on sentences that delay TTFS. Healthy LLM output
    # in voice context lands ~50-100 chars / sentence. Metric 2.6
    # in the perf-metrics taxonomy.
    mean_sentence_chars: float = 0.0
    # iter-070: min and max sentence length submitted this turn.
    # The mean alone hides bimodal patterns ("Yes." + a 150-char
    # sentence both look fine at mean=80). Range = max - min surfaces
    # those — a wide range with a centered mean tells a different
    # story than a narrow range. 0 on turns with no submissions.
    # Metric 2.6 in the perf-metrics taxonomy (histogram form).
    min_sentence_chars: int = 0
    max_sentence_chars: int = 0
    # iter-059: fraction of LLM token chars submitted to the worker
    # as part of a complete sentence (terminator + whitespace) vs
    # flushed as the trailing remainder at end-of-stream. 1.0 = LLM
    # always ended with punctuation, every char overlaps with the
    # next sentence. <1.0 = some chars forced through as remainder
    # which can't overlap with anything (synth happens after stream
    # done). 0 = no chars submitted this turn. Metric 2.5 in the
    # perf-metrics taxonomy.
    sentence_split_coverage: float = 0.0
    # iter-046: bot speaking rate in words-per-minute, derived from
    # the worker's word_count_total / audio_seconds_total. UX-research
    # sweet spot is 150-180 WPM; outside that range is a tunable
    # knob (kokoro's `speed` parameter). 0 means no audio played
    # this turn (or no tokens produced — alignment failed).
    # Metric 1.13 in the perf-metrics taxonomy.
    bot_wpm: float = 0.0
    # iter-064: user speaking rate in words-per-minute, derived from
    # transcript word count / speech_duration. Symmetric to bot_wpm.
    # Useful for the mirroring effect: adapting bot WPM to match
    # user produces higher rapport and lower interruption rate.
    # Wide variance is normal — humans speak 100-200 WPM depending
    # on context (slow in monologue, fast in conversation). 0
    # means speech_duration was 0 or transcript empty.
    # Metric 1.14 in the perf-metrics taxonomy.
    user_wpm: float = 0.0
    # iter-040: count of sentences cut mid-stream by cancel_event
    # (vs completed naturally before barge-in fired). Only non-zero
    # on barge-in turns where the cancel landed during a sentence's
    # playback. 0 on barge-in turns means the cancel landed cleanly
    # in the silent gap between sentences (also a good outcome).
    # Metric 2.18 in the perf-metrics taxonomy. Validates the
    # iter-009 / iter-026 cancel plumbing.
    sentences_cancelled: int = 0
    # iter-014: surface filler + barge-in counters that were added
    # to the worker / coordinator in iter-011 / iter-012 but never
    # made it into the per-turn summary.
    fillers_played: int = 0
    # iter-081: identity (Python ``id()``) of the filler clip the
    # worker picked this turn, or 0 when none played. The id is
    # only meaningful within the running process — print_session_summary
    # aggregates these across turns into a session-wide diversity
    # count via set arithmetic. Persisted to perf snapshots as a
    # stable-per-process integer; useful for in-session aggregation,
    # NOT for cross-iteration time-series (process IDs change).
    # Metric 3.8 in the perf-metrics taxonomy ("Novel/speculative").
    last_filler_id: int = 0
    # iter-051: True if a filler played AND the LLM's first token
    # actually arrived faster than the configured idle_threshold —
    # i.e. the filler was unnecessary, the bot would have started
    # speaking soon enough on its own. False-positive fillers make
    # the bot sound disfluent for no reason. Tune idle_threshold
    # up if this rate is high. Metric 2.4 in the perf-metrics
    # taxonomy.
    filler_false_positive: bool = False
    barge_in: bool = False
    # iter-047: which phase the barge-in fired in. "" if no barge,
    # else "llm_stream" (interrupted while LLM was still streaming
    # tokens — user impatient with TTFS) or "playback" (interrupted
    # while bot was speaking — verbose / wrong response). Different
    # root causes, different fixes. Metric 2.11 in the perf-metrics
    # taxonomy.
    barge_in_phase: str = ""
    # iter-041: time from BargeInCoordinator.trigger() firing to
    # playback being fully stopped (worker thread joined). Metric
    # 2.10 in the perf-metrics taxonomy. The whole barge-in feature
    # lives or dies on this number — >200ms is when the user
    # thinks the bot is ignoring them. 0.0 means no barge-in this
    # turn (or the coordinator wasn't measured).
    barge_in_latency: float = 0.0
    # iter-037: count of mic frames flushed at start of turn (or
    # on the LLM-error path). Metric 2.19 from the perf-metrics
    # taxonomy. Many stale frames means the mic accumulated
    # unwanted audio between turns — bot voice leaking back via
    # OS loopback / Bluetooth duplex / acoustic echo. A reliable
    # signal that echo cancellation is needed in the user's setup.
    mic_stale_frames: int = 0
    # iter-061: time spent inside speaker_factory() inside the
    # SentenceWorker thread, opening the per-turn persistent output
    # device. iter-008's win was holding ONE speaker across all
    # sentences of a turn (vs reopening per sentence). A creep here
    # (driver change, Bluetooth pairing, SDL/PortAudio init) directly
    # delays TTFS for every turn. Yellow flag if >50ms. 0.0 on turns
    # where the worker exited before opening the speaker (shouldn't
    # happen on healthy turns). Metric 2.8 in the perf-metrics
    # taxonomy.
    speaker_open_seconds: float = 0.0
    # iter-062: peak SentenceWorker queue depth observed during this
    # turn. 1 = healthy (each sentence drained before the next
    # arrived). >1 = the LLM produced sentences faster than synth
    # could keep up; the bot will eventually catch up but if depth
    # grows large, synth is the bottleneck and streaming-overlap
    # can't fully mask it. Inverse of ``worker_idle_gap_total``
    # (worker starved). Metric 2.7 in the perf-metrics taxonomy.
    max_queue_depth: int = 0
    # iter-071: token-reveal lag — mean per-token wall-clock offset
    # between the moment a token is printed and the audio second
    # its ``start`` field claims. Positive = text falls behind audio
    # (UX feels broken — text is "subtitles late"); negative = text
    # leads audio (spoils the bot before it speaks). Aggregated from
    # per-sentence stats inside SentenceWorker; 0.0 when play_fn
    # doesn't support the lag_out kwarg (test stubs) or no tokens
    # were emitted. Metric 2.17 in the perf-metrics taxonomy.
    mean_token_reveal_lag: float = 0.0
    max_token_reveal_lag: float = 0.0
    # iter-063: time from the user's last in-speech frame to the
    # VAD declaring DONE_OK. Lower bound is roughly
    # ``silence_duration`` (the VAD has to wait that long before
    # deciding the user really stopped); the gap above that is
    # implementation overhead (chunk granularity, processing).
    # 0.0 on turns where the recorder didn't emit (DONE_TOO_SHORT
    # path, no transcription). Metric 1.2 in the perf-metrics
    # taxonomy — dominates "the agent feels slow" complaints.
    eot_latency: float = 0.0
    # iter-065: the part of eot_latency NOT explained by the
    # configured silence_duration. ``eot_overhead = max(0,
    # eot_latency - silence_duration_used)``. Decomposes the EoT
    # wait into "knob-budget" (the silence_duration we asked for)
    # vs implementation overhead (chunk granularity, processing).
    # If overhead is ~0, the way to reduce EoT latency is to tune
    # ``chat.vad.silence_duration`` lower. If overhead is >100ms,
    # there's something else slow in the recording loop and tuning
    # the knob won't help. Metric 1.3 in the perf-metrics taxonomy.
    eot_overhead: float = 0.0
    # iter-082: TTC (time-to-comprehension) proxy. Cross-turn gap
    # from the PREVIOUS turn's first bot audio to THIS turn's first
    # speech-detected frame. Captures how long the user listened
    # before responding. <500ms = user already knew the answer
    # (bot was telling them something they already knew). >5s =
    # user was confused / thinking. Both are signals; the bell-
    # curve target is 1-3s. 0 on turn 1 and on turns where the
    # prior turn produced no audio. Metric 3.14 in the perf-metrics
    # taxonomy ("Novel/speculative").
    time_to_comprehension: float = 0.0
    # iter-083: first-token-to-audio gap (FT-A). Time from when the
    # LLM's first token landed at the splitter to when the worker
    # played its first audio chunk: ``worker.first_audio_at -
    # first_token_at``. Complementary to ``llm_first_token`` —
    # together they decompose TTFS into "LLM-side" (tts_first_token)
    # and "post-LLM-side" (FT-A) halves. High FT-A = sentence-split
    # + TTS is the bottleneck (bot has tokens but can't speak yet).
    # High llm_first_token = LLM is the bottleneck. Tells you which
    # side to invest in. 0 on turns where either timestamp is
    # missing. Metric 3.18 in the perf-metrics taxonomy
    # ("Novel/speculative").
    first_token_to_audio: float = 0.0
    # iter-085: maximum inter-token gap observed during the LLM
    # stream this turn (excludes the first-token wait, which is
    # iter-052's llm_first_token). Catches mid-stream stalls —
    # currently invisible to operators because the user just
    # hears a long pause and no signal fires. >500ms is "the LLM
    # stalled noticeably mid-response"; >2s is "the user
    # definitely thought the bot was broken." 0 on single-token
    # turns. Metric 3.21 in the perf-metrics taxonomy
    # ("Novel/speculative") — the simpler "max gap" cousin of the
    # full stall-recoverability calculation.
    max_token_gap: float = 0.0
    # iter-154: organic-turn-taking naturalness metric (backlog #8 in
    # docs/research/organic-turn-taking.md). True when this turn's
    # end-of-utterance decision was a *false endpoint*: the agent
    # declared the user done and started responding, but the user
    # actually had more to say (the EOU model / silence VAD fired too
    # early and the user resumed). This is the headline quality metric
    # the LiveKit turn-detector / pipecat smart-turn literature tracks
    # ("false-endpoint rate"). Defaults False and is only populated by
    # the organic path — the proven half-duplex silence-VAD path leaves
    # it at its default, so the metric is purely additive. Pairs with
    # iter-056's barge_in_regret (a *latency*-based pre-emption signal)
    # as the *decision*-based pre-emption signal. Metric 3.22 in the
    # perf-metrics taxonomy ("Novel/speculative").
    false_endpoint: bool = False
    # iter-154: organic-turn-taking naturalness metric (backlog #8).
    # Count of user *continuer* utterances ("mhmm" / "yeah" / "right")
    # detected during this turn's agent speech and correctly NOT
    # treated as a turn-grab — recognized via iter-148's
    # classify_backchannel and acted on by iter-152's
    # decide_barge_action (CONTINUER ⇒ FINISH, hold the agent's floor).
    # 0 on the half-duplex path (continuers aren't classified there; a
    # barge always abandons). A non-zero count means continuer-aware
    # listening (backlog #5) recognized active-listening signals and
    # preserved the agent's turn instead of clipping it — the thing #5
    # buys, now measured rather than asserted. Metric 3.23 in the
    # perf-metrics taxonomy ("Novel/speculative").
    continuers_detected: int = 0
    # iter-163: organic-turn-taking diagnostic (backlog #9). True iff this
    # turn was force-emitted by the iter-157 max_merge_depth safety cap —
    # the organic UtteranceBuffer held a mid-thought utterance through
    # max_merge_depth consecutive merges without it ever looking complete,
    # then gave up holding it (to avoid starving the engine on a
    # pathological unfinished-forever stream) and cut the still-unfinished
    # running text loose as this turn. Always paired with
    # false_endpoint=True (a capped turn absorbed real merges on the way up),
    # but distinct: a clean repair lands on a complete-looking sentence,
    # whereas a capped turn is the backstop firing — a signal the operator
    # should see, since a recurring cap means the merge window / completeness
    # scorer is mistuned for this speaker (or the STT is fragmenting). False
    # on every natural release and on the half-duplex path (which never
    # holds, so the cap can never fire). Surfaced in the organic summary
    # block so iter-157's promised "bounded, OBSERVABLE delay" is actually
    # observable rather than silent.
    merge_capped: bool = False
    transcript: str = ""
    response: str = ""
    model: str = ""

    def print(self, turn: int) -> None:
        print()
        print(f"  {_DIM}{'─' * 56}{_RESET}")
        print(f"  {_BOLD}Turn {turn}{_RESET}")
        print(f"  {_DIM}You:{_RESET} \"{self.transcript}\"")
        print()
        print(f"  {_DIM}┌─ PIPELINE{_RESET}")
        # iter-082: TTC (time-to-comprehension) — cross-turn gap
        # from prev bot first audio to this turn's user speech
        # start. Skip when 0 (turn 1, or prior had no audio).
        # Bell-curve target is 1-3s; <500ms or >5s are both
        # interesting flags.
        if self.time_to_comprehension > 0:
            ms = self.time_to_comprehension * 1000
            if ms < 500 or ms > 5000:
                color = _YELLOW
            else:
                color = _DIM
            print(
                f"  {_DIM}│{_RESET}  TTC:           "
                f"{color}{ms:>7.0f}ms{_RESET}  "
                f"({_DIM}user listened before responding{_RESET})"
            )
        # iter-064: append user WPM to the Speech line when known.
        # Symmetric to iter-046's bot WPM display. No color coding —
        # humans speak across a wide range and there's no "correct"
        # rate for the user; only a "match the user" target for the
        # bot.
        if self.user_wpm > 0:
            print(
                f"  {_DIM}│{_RESET}  Speech:        "
                f"{self.speech_duration*1000:>7.0f}ms  "
                f"({_DIM}{self.user_wpm:.0f} WPM{_RESET})"
            )
        else:
            print(
                f"  {_DIM}│{_RESET}  Speech:        "
                f"{self.speech_duration*1000:>7.0f}ms"
            )
        # iter-063: EoT detection latency. Skip when 0 (recorder
        # didn't emit — DONE_TOO_SHORT path or test stub bypass).
        # Yellow when >1.0s — the user has stopped talking but the
        # agent is still waiting; the silence_duration knob is
        # tunable down to ~500ms in noisy rooms.
        if self.eot_latency > 0:
            ms = self.eot_latency * 1000
            color = _YELLOW if ms > 1000 else _DIM
            # iter-065: append the trailing-silence wall — how much
            # of the EoT wait is implementation overhead vs the
            # configured silence_duration. Skip the suffix when
            # overhead is trivial (<10ms — within chunk noise);
            # yellow when >100ms (knob-tuning won't help).
            suffix = f"  ({_DIM}silence wait{_RESET})"
            if self.eot_overhead > 0.010:
                ov_ms = self.eot_overhead * 1000
                ov_color = _YELLOW if ov_ms > 100 else _DIM
                suffix = (
                    f"  ({_DIM}silence wait{_RESET}, "
                    f"{ov_color}+{ov_ms:.0f}ms overhead{_RESET})"
                )
            print(
                f"  {_DIM}│{_RESET}  EoT detect:    "
                f"{color}{ms:>7.0f}ms{_RESET}{suffix}"
            )
        # iter-049: append STT RTF when measurable.
        # iter-072: also append preview divergence when populated.
        # Yellow when >0.3 (preview was misleading); dim otherwise.
        stt_extras: list[str] = []
        if self.stt_rtf > 0:
            rtf_color = _GREEN if self.stt_rtf < 1.0 else _YELLOW
            stt_extras.append(f"{rtf_color}RTF {self.stt_rtf:.2f}x{_RESET}")
        if self.stt_preview_divergence > 0:
            div_pct = self.stt_preview_divergence * 100
            div_color = _YELLOW if self.stt_preview_divergence > 0.3 else _DIM
            stt_extras.append(
                f"{div_color}preview Δ {div_pct:.0f}%{_RESET}"
            )
        if stt_extras:
            print(
                f"  {_DIM}│{_RESET}  STT:           "
                f"{self.stt_time*1000:>7.0f}ms  ({', '.join(stt_extras)})"
            )
        else:
            print(f"  {_DIM}│{_RESET}  STT:           {self.stt_time*1000:>7.0f}ms")
        # iter-083: append FT-A (first-token-to-audio) to the LLM
        # 1st-token line as a complementary "right side" diagnostic.
        # Together, llm_first_token and FT-A bracket where TTFS time
        # was spent. Skip the suffix when FT-A is 0 (rare error path).
        if self.first_token_to_audio > 0:
            fta_str = f"  ({_DIM}+{self.first_token_to_audio*1000:.0f}ms → audio{_RESET})"
        else:
            fta_str = ""
        print(
            f"  {_DIM}│{_RESET}  LLM 1st tok:   "
            f"{self.llm_first_token*1000:>7.0f}ms{fta_str}"
        )
        # iter-038: TTFsent — time-to-first-sentence. Show the gap
        # between first-token and first-sentence in parens so the
        # user sees how much "preamble lag" the splitter waited
        # through. Skip the line entirely on the rare turn where
        # no complete sentence emerged.
        if self.llm_first_sentence > 0:
            preamble_gap = self.llm_first_sentence - self.llm_first_token
            print(
                f"  {_DIM}│{_RESET}  LLM 1st sent:  "
                f"{self.llm_first_sentence*1000:>7.0f}ms  "
                f"({_DIM}+{preamble_gap*1000:.0f}ms preamble{_RESET})"
            )
        # iter-052: append TPS suffix when measurable.
        if self.llm_tps > 0:
            tps_str = f", {self.llm_tps:.0f} tps"
        else:
            tps_str = ""
        # iter-077: append approximate context-token count when
        # known. Helps explain a creeping llm_first_token even
        # when TPS looks healthy.
        if self.context_tokens > 0:
            ctx_str = f", {self.context_tokens} ctx"
        else:
            ctx_str = ""
        # iter-085: append max token gap when significant (>200ms).
        # Smaller gaps are normal token-streaming jitter; larger
        # ones reveal mid-stream stalls.
        if self.max_token_gap > 0.2:
            gap_color = _YELLOW if self.max_token_gap > 0.5 else _DIM
            gap_str = (
                f", {gap_color}max gap "
                f"{self.max_token_gap*1000:.0f}ms{_RESET}"
            )
        else:
            gap_str = ""
        print(
            f"  {_DIM}│{_RESET}  LLM total:     "
            f"{self.llm_total*1000:>7.0f}ms  "
            f"({self.model}{tps_str}{ctx_str}{gap_str})"
        )
        tts_suffix = f"({self.sentences_spoken} sentences"
        if self.fillers_played > 0:
            tts_suffix += f" + {self.fillers_played} filler"
            if self.fillers_played > 1:
                tts_suffix += "s"
            # iter-051: flag false positive. Marker is "*", with
            # an explanation appended by the session summary.
            if self.filler_false_positive:
                tts_suffix += "*"
        # iter-045: append mean sentence length as fragmentation
        # diagnostic. Yellow flag if <30 chars (over-fragmenting,
        # losing overlap) or >150 chars (under-fragmenting,
        # delaying TTFS).
        if self.mean_sentence_chars > 0:
            tts_suffix += f", avg {self.mean_sentence_chars:.0f} chars"
            # iter-070: append range when min/max actually diverge.
            # A wide range with a centered mean is the classic
            # bimodal-fragmentation signal (one short interjection
            # + one long sentence). Skip when min == max — adding
            # "[X..X]" is just noise.
            if (self.max_sentence_chars > self.min_sentence_chars
                    and self.min_sentence_chars > 0):
                tts_suffix += (
                    f" [{self.min_sentence_chars}..{self.max_sentence_chars}]"
                )
        # iter-059: split coverage as a percentage. Only emit on
        # turns where it's < 1.0 — a perfect 100% split is the
        # expected case and shouldn't clutter the line.
        if 0 < self.sentence_split_coverage < 1.0:
            tts_suffix += (
                f", {self.sentence_split_coverage*100:.0f}% complete"
            )
        tts_suffix += ")"
        # iter-050: append RTF to TTS suffix when measurable.
        if self.tts_rtf > 0:
            rtf_color = _GREEN if self.tts_rtf < 1.0 else _YELLOW
            tts_suffix += f"  ({rtf_color}RTF {self.tts_rtf:.2f}x{_RESET})"
        print(f"  {_DIM}│{_RESET}  TTS:           {self.tts_time*1000:>7.0f}ms  {tts_suffix}")
        print(f"  {_DIM}│{_RESET}  Playback:      {self.playback_time*1000:>7.0f}ms")
        # iter-061: speaker-open overhead. Skip on the common case
        # of 0 (subsequent turns reuse the persistent speaker — no
        # second open). Yellow flag when >50ms; the iter-008 win
        # was about avoiding per-sentence opens in the hot path.
        if self.speaker_open_seconds > 0:
            ms = self.speaker_open_seconds * 1000
            color = _YELLOW if ms > 50 else _DIM
            print(
                f"  {_DIM}│{_RESET}  Speaker open:  "
                f"{color}{ms:>6.0f}ms{_RESET}  "
                f"({_DIM}device init{_RESET})"
            )
        # iter-062: peak worker queue depth. Skip when ≤1 (healthy
        # case — producer/consumer kept pace). Yellow when ≥3:
        # synth is falling behind enough that mid-turn latency may
        # accumulate visibly.
        if self.max_queue_depth > 1:
            color = _YELLOW if self.max_queue_depth >= 3 else _DIM
            print(
                f"  {_DIM}│{_RESET}  Queue depth:   "
                f"{color}{self.max_queue_depth:>6d}{_RESET}  "
                f"({_DIM}synth backlog peak{_RESET})"
            )
        # iter-071: token-reveal lag. Skip when the metric wasn't
        # captured (mean and max both at 0). Yellow when |mean| >
        # 100ms — the user perceives the text as out of sync with
        # the bot's voice. Sign-aware: render with leading "+/-".
        if self.mean_token_reveal_lag != 0 or self.max_token_reveal_lag != 0:
            mean_ms = self.mean_token_reveal_lag * 1000
            max_ms = self.max_token_reveal_lag * 1000
            color = _YELLOW if abs(mean_ms) > 100 else _DIM
            print(
                f"  {_DIM}│{_RESET}  Token-reveal:  "
                f"{color}{mean_ms:>+6.0f}ms{_RESET} mean, "
                f"{_DIM}{max_ms:+.0f}ms peak{_RESET}"
            )
        # iter-046: bot WPM. Skip if 0 (no audio / no tokens). Color:
        # green if 130-200 (around the UX-research sweet spot 150-180),
        # yellow otherwise (too fast or too slow).
        if self.bot_wpm > 0:
            color = _GREEN if 130 <= self.bot_wpm <= 200 else _YELLOW
            print(
                f"  {_DIM}│{_RESET}  Bot WPM:       "
                f"{color}{self.bot_wpm:>6.0f}{_RESET}  "
                f"({_DIM}target 150-180{_RESET})"
            )
        # iter-043: streaming overlap. Skip the line on turns where
        # it's 0 (sequential — audio came after LLM finished, so
        # streaming bought us nothing this turn — common on very
        # short responses). Show as percentage. Color-code: green
        # if >50% (good overlap), yellow ≤50%.
        if self.streaming_overlap_ratio > 0:
            pct = self.streaming_overlap_ratio * 100
            color = _GREEN if pct >= 50 else _YELLOW
            print(
                f"  {_DIM}│{_RESET}  Overlap:       "
                f"{color}{pct:>6.0f}%{_RESET}  "
                f"({_DIM}LLM↔TTS concurrency{_RESET})"
            )
        # iter-074: bargeable-time fraction. Skip when 1.0 (the
        # healthy architectural default — clutter-free for clean
        # sessions). Anything < 1.0 surfaces as a yellow regression
        # alarm: the bot was uninterruptible for some fraction of
        # its speech.
        if 0 < self.bargeable_fraction < 0.99:
            pct = self.bargeable_fraction * 100
            print(
                f"  {_DIM}│{_RESET}  Bargeable:     "
                f"{_YELLOW}{pct:>6.0f}%{_RESET}  "
                f"({_DIM}watcher coverage of bot speech{_RESET})"
            )
        # iter-073: first-sentence overlap savings. Emit when >0 —
        # tells the operator how many ms were shaved off TTFS by
        # parallelizing first synth with the rest of LLM streaming.
        # Green when >100ms (meaningful TTFS win); dim otherwise.
        if self.first_synth_overlap_seconds > 0:
            ms = self.first_synth_overlap_seconds * 1000
            color = _GREEN if ms > 100 else _DIM
            print(
                f"  {_DIM}│{_RESET}  1st-synth save: "
                f"{color}{ms:>5.0f}ms{_RESET}  "
                f"({_DIM}TTFS shaved by streaming{_RESET})"
            )
        # iter-044: between-sentence worker idle gap. Skip when 0
        # (single-sentence responses or very fast LLM). >300ms is
        # "the worker is starving" — investigate.
        if self.worker_idle_gap_total > 0:
            color = _YELLOW if self.worker_idle_gap_total > 0.3 else _DIM
            print(
                f"  {_DIM}│{_RESET}  Idle gap:      "
                f"{color}{self.worker_idle_gap_total*1000:>6.0f}ms{_RESET}  "
                f"({_DIM}worker waited for sentences{_RESET})"
            )
        if self.barge_in:
            # iter-040: distinguish mid-stream cancel (cancel landed
            # during sentence playback — clean cut-off) vs between-
            # sentences cancel (cancel landed in the silent gap —
            # also clean, sentence ended naturally). Both are
            # success outcomes but they tell different stories
            # about how the user is timing their interruption.
            if self.sentences_cancelled > 0:
                cancel_note = (
                    f" ({self.sentences_cancelled} cut mid-stream)"
                )
            else:
                cancel_note = " (between sentences)"
            # iter-047: phase context. "llm_stream" = user interrupted
            # before bot started speaking (impatient with TTFS).
            # "playback" = user interrupted bot speech (verbose /
            # wrong response).
            if self.barge_in_phase == "llm_stream":
                phase_note = " (during LLM stream)"
            elif self.barge_in_phase == "playback":
                phase_note = " (during playback)"
            else:
                phase_note = ""
            # iter-056: regret marker. Bot pre-empted the user.
            regret_note = " — regret" if self.barge_in_regret else ""
            print(
                f"  {_DIM}│{_RESET}  {_YELLOW}Barge-in:      "
                f"yes (user interrupted){cancel_note}{phase_note}{regret_note}{_RESET}"
            )
            # iter-041: barge-in latency. Only meaningful when
            # >0 (some test paths leave it at 0). Color-code:
            # red if >300ms (user notices), yellow if 100-300ms,
            # green if <100ms.
            if self.barge_in_latency > 0:
                lat_ms = self.barge_in_latency * 1000
                if lat_ms > 300:
                    color = _YELLOW  # we don't have red, yellow is alarm
                elif lat_ms > 100:
                    color = _YELLOW
                else:
                    color = _GREEN
                print(
                    f"  {_DIM}│{_RESET}  {color}Barge latency: "
                    f"{lat_ms:>6.0f}ms{_RESET}  "
                    f"(detect → halt)"
                )
            # iter-057: primed-frames replay seconds. Only on barge
            # turns where the watcher captured frames. Bigger value
            # = more of the user's first words were preserved for
            # the next STT pass.
            if self.primed_frames_seconds > 0:
                print(
                    f"  {_DIM}│{_RESET}  {_DIM}Primed frames: "
                    f"{self.primed_frames_seconds*1000:>6.0f}ms{_RESET}  "
                    f"(carried into next turn)"
                )
            # iter-080: pre-empted words — content the LLM
            # generated but the user never heard. Only on barge
            # turns. Yellow when >10 words: that's >5 seconds of
            # spoken content lost, suggesting bot was being too
            # verbose (vs <5 words = a clean cut-off in the
            # middle of a normal sentence).
            if self.preempted_words > 0:
                color = _YELLOW if self.preempted_words > 10 else _DIM
                print(
                    f"  {_DIM}│{_RESET}  {color}Pre-empted:    "
                    f"{self.preempted_words:>6d} words{_RESET}  "
                    f"({_DIM}generated but not played{_RESET})"
                )
            # iter-060: LLM stream cancel-to-close. Only meaningful
            # on barge turns; >500ms is "the HTTP socket is hanging."
            if self.llm_cancel_to_close > 0:
                lat_ms = self.llm_cancel_to_close * 1000
                color = _YELLOW if lat_ms > 500 else _DIM
                print(
                    f"  {_DIM}│{_RESET}  {color}LLM cancel:    "
                    f"{lat_ms:>6.0f}ms{_RESET}  "
                    f"(trigger → stream close)"
                )
        # iter-037: only emit when non-zero — a clean turn shouldn't
        # spend pixels on a stale-frame counter that's almost always 0.
        # When >0 it's worth noticing — bot voice leaking back through
        # the OS mic is a real-world problem that points at acoustic
        # echo / Bluetooth duplex / loopback misconfiguration.
        if self.mic_stale_frames > 0:
            stale_seconds = self.mic_stale_frames / 16000  # RATE
            color = _YELLOW if stale_seconds > 0.5 else _DIM
            print(
                f"  {_DIM}│{_RESET}  {color}Mic stale:     "
                f"{self.mic_stale_frames:>5} frames ({stale_seconds:.1f}s){_RESET}"
            )
        # iter-154: organic-turn-taking signals (backlog #8). Only
        # emit when non-zero — the half-duplex default leaves both at
        # their defaults, so a half-duplex turn spends no pixels here.
        # A false endpoint is a yellow flag (the agent cut the user
        # off by mis-deciding end-of-turn); continuers are a positive
        # signal (the agent correctly held its floor), shown dim.
        if self.false_endpoint:
            # iter-163: a capped turn is the max_merge_depth backstop firing
            # (force-emit of still-unfinished text), not a clean repair —
            # annotate the line so the per-turn readout distinguishes the two.
            detail = (
                "merge-depth cap hit — force-emitted mid-thought (iter-157)"
                if self.merge_capped
                else "user wasn't done — EOU fired early"
            )
            print(
                f"  {_DIM}│{_RESET}  {_YELLOW}False endpoint:"
                f" yes{_RESET}  "
                f"({_DIM}{detail}{_RESET})"
            )
        if self.continuers_detected > 0:
            print(
                f"  {_DIM}│{_RESET}  {_DIM}Continuers:    "
                f"{self.continuers_detected:>5}{_RESET}  "
                f"({_DIM}backchannels held the floor{_RESET})"
            )
        print(f"  {_DIM}│{_RESET}")
        ttfs_color = _GREEN if self.ttfs < 3.0 else _YELLOW
        # iter-053: append naturalness bucket as a parenthetical
        # tag. "(natural)" is the sweet spot; "(rushed)" /
        # "(slow)" call out off-target turns.
        if self.naturalness_bucket:
            bucket_tag = f", {self.naturalness_bucket}"
        else:
            bucket_tag = ""
        print(
            f"  {_DIM}├─{_RESET} {_BOLD}TTFS:{_RESET}            "
            f"{ttfs_color}{self.ttfs*1000:>7.0f}ms{_RESET}  "
            f"(speech stop → speaker{bucket_tag})"
        )
        # iter-076: TTFS attribution breakdown. Decompose into
        # STT / LLM-to-first-sentence / synth+dispatch percentages.
        # Skip when ttfs == 0 (no audio played) or when the
        # underlying parts aren't measurable. The percentages can
        # be slightly off-100 due to small floating-point
        # residuals; render as floored integers so the visible
        # sum stays ≤ 100.
        if (
            self.ttfs > 0
            and self.stt_time > 0
            and self.llm_first_sentence > 0
            and self.synth_dispatch_seconds > 0
        ):
            stt_pct = int(self.stt_time / self.ttfs * 100)
            llm_pct = int(self.llm_first_sentence / self.ttfs * 100)
            synth_pct = int(self.synth_dispatch_seconds / self.ttfs * 100)
            print(
                f"  {_DIM}│{_RESET}  Attribution:   "
                f"{_DIM}STT {stt_pct}% + "
                f"LLM {llm_pct}% + "
                f"synth {synth_pct}%{_RESET}"
            )
        total_color = _GREEN if self.total_e2e < 6.0 else _YELLOW
        print(
            f"  {_DIM}└─{_RESET} {_BOLD}Total turn:{_RESET}      "
            f"{total_color}{self.total_e2e*1000:>7.0f}ms{_RESET}"
        )
        print(f"  {_DIM}{'─' * 56}{_RESET}")
        print()


def _median_ms(values: list[float]) -> float:
    """Return the median of `values` in milliseconds.

    Uses ``statistics.median`` so even-length lists return the
    average of the two middle elements (rather than the upper
    median that ``sorted[len//2]`` produces — see iter-017 for
    why that mattered).
    """
    if not values:
        return 0.0
    return statistics.median(values) * 1000


def _emit_ttfs_block(
    emit,
    ttfs_times: list[float],
    metrics_list: list,
    naturalness_counts: dict[str, int],
) -> None:
    """iter-089: extracted from print_session_summary's TTFS block.

    Renders all TTFS-related session-summary lines:
      - Median TTFS / Best TTFS (or n/a placeholders if empty).
      - Sub-second TTFS rate (iter-084).
      - Rhythm score + TTFS jitter (iter-055/068, ≥2 turns).
      - Cold-start penalty (iter-066, ≥2 turns + turn-1 has TTFS).
      - Naturalness distribution (iter-053).

    Behavior-preserving: output is byte-for-byte identical to the
    inline version that lived in print_session_summary before
    iter-089. ``emit`` is the same callable used by
    print_session_summary (writes to file or print()).
    """
    if not ttfs_times:
        # All turns ended without audio. Emit a placeholder rather
        # than a misleading "0ms" so the user knows it isn't a
        # win, it's an absence of data.
        emit(f"    {_BOLD}Median TTFS:      n/a{_RESET}")
        emit(f"    Best TTFS:        n/a")
        return

    emit(f"    {_BOLD}Median TTFS:      {_median_ms(ttfs_times):.0f}ms{_RESET}")
    emit(f"    Best TTFS:        {min(ttfs_times) * 1000:.0f}ms")
    # iter-084: sub-second turn rate. Single human-feel
    # threshold — what fraction of turns hit the snappy bar.
    sub_second = sum(1 for t in ttfs_times if t < 1.0)
    sub_pct = (sub_second / len(ttfs_times)) * 100
    emit(
        f"    Sub-second TTFS:  "
        f"{sub_second}/{len(ttfs_times)} ({sub_pct:.0f}%)"
    )
    # iter-055: conversation rhythm score. Needs ≥2 turns for
    # stdev. Clamp to [0, 1] since high-variance sessions can
    # produce stdev > median → negative raw score.
    if len(ttfs_times) >= 2:
        med = statistics.median(ttfs_times)
        sd = statistics.stdev(ttfs_times)
        raw = 1.0 - sd / max(med, 1e-6)
        rhythm = max(0.0, min(1.0, raw))
        emit(f"    Rhythm score:     {rhythm:.2f}")
        # iter-068: raw stdev as TTFS jitter alongside the
        # normalized rhythm score.
        emit(f"    TTFS jitter:      ±{sd * 1000:.0f}ms")
    # iter-066: cold-start latency penalty. Needs ≥2 turns AND
    # turn-1 must have measurable TTFS.
    first_turn_ttfs = (
        metrics_list[0].ttfs if metrics_list[0].ttfs > 0 else 0.0
    )
    steady_ttfs = [
        m.ttfs for m in metrics_list[1:] if m.ttfs > 0
    ]
    if first_turn_ttfs > 0 and len(steady_ttfs) >= 1:
        penalty = first_turn_ttfs - statistics.median(steady_ttfs)
        if abs(penalty) > 0.050:  # >50ms — above jitter floor
            ms = penalty * 1000
            emit(
                f"    Cold start:       {ms:+.0f}ms "
                f"vs steady state"
            )
    # iter-053: naturalness distribution. Show only when at
    # least one turn was bucketed.
    n_total = sum(naturalness_counts.values())
    if n_total > 0:
        emit(
            f"    Naturalness:      "
            f"{naturalness_counts['rushed']} rushed, "
            f"{naturalness_counts['natural']} natural, "
            f"{naturalness_counts['slow']} slow"
        )


@dataclass
class BargeStats:
    """iter-090: aggregated barge-in counters and latency lists
    consumed by ``_emit_barge_block``. Bundling them into a
    dataclass keeps the helper signature stable as more barge-
    side metrics arrive — each new metric extends the dataclass.

    Fields mirror the names in ``print_session_summary``'s body:
      ``barges_total`` (iter-019): count of barge_in turns.
      ``mid_cancels`` (iter-040): barges where a sentence was
        cut mid-stream.
      ``n`` (iter-019): total completed turns — denominator for
        the iter-069 interruption rate.
      ``barge_latencies`` (iter-041): per-turn detect→halt latencies.
      ``cancel_close_lats`` (iter-060): trigger→stream-close gaps.
      ``llm_phase_barges`` / ``playback_phase_barges`` (iter-047):
        phase-distribution counters.
      ``regret_barges`` (iter-056): barges firing within 200ms of
        first audio.
      ``preempted_total`` (iter-080): cumulative words pre-empted
        by mid-content barges.
      ``barge_turns_with_loss`` (iter-080): barges that lost
        content (vs clean cuts between sentences).
    """

    barges_total: int = 0
    mid_cancels: int = 0
    n: int = 0
    barge_latencies: list = field(default_factory=list)
    cancel_close_lats: list = field(default_factory=list)
    llm_phase_barges: int = 0
    playback_phase_barges: int = 0
    regret_barges: int = 0
    preempted_total: int = 0
    barge_turns_with_loss: int = 0


def _emit_barge_block(emit, stats: BargeStats) -> None:
    """iter-090: extracted from print_session_summary's barge block.

    Renders all barge-in session-summary lines:
      - Barge-ins: count + mid-stream % (iter-040).
      - Interruption rate: barges/turns (iter-069).
      - Median + worst barge latency (iter-041).
      - Median LLM cancel-to-close (iter-060).
      - Phase distribution (iter-047).
      - Regret rate (iter-056).
      - Pre-empted words total (iter-080).

    Behavior-preserving: byte-for-byte identical to the inline
    version. ``emit`` is the same callable used by
    ``print_session_summary``.
    """
    if stats.barges_total <= 0:
        return

    # iter-040: count + mid-stream % vs cleanly-between-sentences.
    if stats.mid_cancels:
        pct = (stats.mid_cancels / stats.barges_total) * 100
        emit(
            f"    Barge-ins:        {stats.barges_total} "
            f"({stats.mid_cancels} mid-stream, {pct:.0f}%)"
        )
    else:
        emit(
            f"    Barge-ins:        {stats.barges_total} "
            f"(all between sentences)"
        )
    # iter-069: interruption rate as fraction of total turns.
    if stats.n > 0:
        int_pct = (stats.barges_total / stats.n) * 100
        emit(
            f"    Interruption rate: "
            f"{stats.barges_total}/{stats.n} turns ({int_pct:.0f}%)"
        )
    # iter-041: median + worst barge-in latency.
    if stats.barge_latencies:
        emit(f"    Median barge:     {_median_ms(stats.barge_latencies):.0f}ms")
        emit(
            f"    Worst barge:      "
            f"{max(stats.barge_latencies) * 1000:.0f}ms"
        )
    # iter-060: median LLM cancel-to-close across barge turns.
    if stats.cancel_close_lats:
        emit(
            f"    Median LLM canc:  "
            f"{_median_ms(stats.cancel_close_lats):.0f}ms"
        )
    # iter-047: phase distribution.
    if stats.llm_phase_barges or stats.playback_phase_barges:
        emit(
            f"    Barge phases:     "
            f"{stats.llm_phase_barges} LLM-stream, "
            f"{stats.playback_phase_barges} playback"
        )
    # iter-056: regret rate.
    if stats.regret_barges:
        pct = (stats.regret_barges / stats.barges_total) * 100
        emit(
            f"    Regret rate:      "
            f"{stats.regret_barges}/{stats.barges_total} ({pct:.0f}%) "
            f"— bot may be pre-empting; raise silence_duration"
        )
    # iter-080: total words pre-empted across barge turns.
    if stats.preempted_total > 0:
        avg = stats.preempted_total / max(stats.barge_turns_with_loss, 1)
        emit(
            f"    Pre-empted words: "
            f"{stats.preempted_total} total "
            f"({stats.barge_turns_with_loss}/{stats.barges_total} barges, "
            f"{avg:.0f} avg/loss)"
        )


@dataclass
class OrganicStats:
    """iter-154: aggregated organic-turn-taking naturalness counters
    consumed by ``_emit_organic_block`` (backlog #8 in
    ``docs/research/organic-turn-taking.md``).

    The organic track (backlog #1/#5/#7) shipped its decision seams
    behind an off-by-default full-duplex gate; this is the measurement
    surface that lets the track be *measured, not asserted* once the
    seams are wired in. Each new organic-side metric extends the
    dataclass instead of growing ``_emit_organic_block``'s signature.

    Fields mirror the names in ``print_session_summary``'s body:
      ``false_endpoints`` (iter-154): turns where the EOU decision
        fired early and the user actually had more to say — the
        headline false-endpoint count the turn-detector literature
        tracks.
      ``continuers_total`` (iter-154): cumulative user continuers
        ("mhmm") recognized across the session and correctly NOT
        treated as turn-grabs (iter-148 classifier + iter-152
        decision). Validates continuer-aware listening (#5) is
        actually buying floor-holds.
      ``n`` (iter-154): total completed turns — denominator for the
        false-endpoint rate.
      ``utterances_held`` (iter-161): turns where the organic
        UtteranceAggregator held a mid-thought utterance for a merge
        (a successful capture buffered, NOT a VAD false trigger).
        Surfaced so the operator sees how often the merge buffer
        engaged — and so these holds are visibly distinct from the
        VAD false-trigger count they used to be miscounted as
        (the iter-159→iter-161 fix).
      ``merges_capped`` (iter-163): turns force-emitted by the
        iter-157 max_merge_depth safety cap — the buffer held a
        mid-thought utterance through max_merge_depth merges without
        it ever looking complete, then cut it loose still-unfinished
        to avoid starving the engine. A subset of ``false_endpoints``
        (every capped turn absorbed real merges), surfaced separately
        so iter-157's promised "bounded, OBSERVABLE delay" is visible
        rather than silent. A recurring count means the merge window /
        completeness scorer is mistuned for this speaker.
      ``backchannels_emitted`` (iter-175): how many mid-speech *agent*
        backchannels ("mhmm" / "i see" / "right" ...) the organic
        ``BackchannelMonitor`` / ``BackchannelDriver`` (#7, iter-170+)
        emitted during the session — the lifetime ``emit_count`` off the
        driver. The mirror of ``continuers_total``: that counts the
        *user's* continuers we held the floor through; this counts the
        *agent's* active-listening cues emitted *during* the user's
        speech. #7 shipped the whole pure stack (when/which/inputs/
        composition) but had no presence in the measurement surface, so
        the agent's backchannelling was invisible in the summary. 0 on
        the half-duplex / no-driver path (the driver never emits).
    """

    false_endpoints: int = 0
    continuers_total: int = 0
    n: int = 0
    utterances_held: int = 0
    merges_capped: int = 0
    backchannels_emitted: int = 0


def _emit_organic_block(emit, stats: OrganicStats) -> None:
    """iter-154: render the organic-turn-taking naturalness lines
    (backlog #8).

    Renders, when there is anything to show:
      - False-endpoint rate: false_endpoints/turns (the EOU
        mis-decision rate — the agent declared the user done early).
      - Continuers held: total continuers recognized and not treated
        as turn-grabs (the win continuer-aware listening buys).
      - Utterances merged-held (iter-161): mid-thought fragments the
        aggregator buffered for a merge rather than responding to
        immediately.
      - Merges capped (iter-163): turns the max_merge_depth backstop
        force-emitted while still mid-thought — iter-157's cap firing,
        now observable rather than silent.
      - Backchannels emitted (iter-175): mid-speech agent backchannels
        the organic driver emitted — the agent's active-listening
        signal, mirror of the user's "Continuers held" line.

    **Fully suppressed when all counters are zero** — which is the
    half-duplex default, so existing sessions print byte-for-byte the
    same summary they did before iter-154. The block only appears once
    the organic path is wired in and starts populating the per-turn
    ``false_endpoint`` / ``continuers_detected`` fields (or holds a
    mid-thought utterance, iter-161; or hits the merge cap, iter-163;
    or the agent emits a backchannel, iter-175).
    """
    if (
        stats.false_endpoints <= 0
        and stats.continuers_total <= 0
        and stats.utterances_held <= 0
        and stats.merges_capped <= 0
        and stats.backchannels_emitted <= 0
    ):
        return

    emit(f"    {_BOLD}Organic turn-taking:{_RESET}")
    # False-endpoint rate — the EOU mis-decision rate. Yellow framing
    # via the suggestion text; a high rate means the endpoint signal
    # is too eager (the agent keeps cutting the user off).
    if stats.false_endpoints > 0 and stats.n > 0:
        pct = (stats.false_endpoints / stats.n) * 100
        line = (
            f"    False endpoints:  "
            f"{stats.false_endpoints}/{stats.n} turns ({pct:.0f}%)"
        )
        # >20% of turns mis-ending is a real problem — the endpoint
        # heuristic is firing before the user is done.
        if pct > 20:
            line += " — EOU too eager; raise silence_duration"
        emit(line)
    elif stats.false_endpoints > 0:
        # n unknown (shouldn't happen from the live path, but keep the
        # count visible rather than dropping it silently).
        emit(f"    False endpoints:  {stats.false_endpoints}")
    # Continuers held — the positive signal: active-listening
    # backchannels that correctly did NOT abandon the agent's turn.
    if stats.continuers_total > 0:
        emit(
            f"    Continuers held:  {stats.continuers_total} "
            f"(backchannels kept the floor)"
        )
    # iter-161: mid-thought fragments the aggregator buffered for a
    # merge. Distinct from VAD false triggers — a successful capture
    # deliberately held to repair a false endpoint. Surfacing the count
    # makes the merge buffer's activity visible and documents that these
    # holds are NOT counted in the VAD false-trigger line.
    if stats.utterances_held > 0:
        emit(
            f"    Utterances held:  {stats.utterances_held} "
            f"(mid-thought, buffered for merge — not VAD false triggers)"
        )
    # iter-163: turns the iter-157 max_merge_depth backstop force-emitted
    # while still mid-thought. A subset of the false-endpoint count above
    # (every capped turn absorbed real merges), surfaced separately because
    # a capped turn is the cap firing — not a clean repair. A recurring
    # count is a tuning signal: the merge window / completeness scorer is
    # not matching this speaker (or the STT is fragmenting utterances), so
    # the buffer keeps hitting the depth limit instead of landing on a
    # complete-looking sentence.
    if stats.merges_capped > 0:
        emit(
            f"    Merges capped:    {stats.merges_capped} "
            f"(hit max_merge_depth still mid-thought — retune merge "
            f"window/EOU; iter-157 cap)"
        )
    # iter-175: mid-speech agent backchannels emitted by the organic
    # BackchannelMonitor/BackchannelDriver (#7). The agent's
    # active-listening signal — the mirror of "Continuers held" above
    # (the user's continuers). Surfaced so #7's emission is measured,
    # not asserted: a session running organic agent-backchannels can now
    # see how many "mhmm"s the agent actually placed during user speech.
    # 0 (no line) on the half-duplex / no-driver path.
    if stats.backchannels_emitted > 0:
        emit(
            f"    Backchannels:     {stats.backchannels_emitted} "
            f"(agent mid-speech cues — active listening)"
        )


@dataclass
class FillerStats:
    """iter-091: aggregated filler-side counters consumed by
    ``_emit_filler_block``. Each new filler-side metric extends
    the dataclass instead of growing the helper signature.

    Fields:
      ``fillers_total`` (iter-014): total filler clips played.
      ``filler_turns`` (iter-051): turns where ≥1 filler played
        (denominator for the FP rate).
      ``filler_false_positives`` (iter-051): filler turns where
        ``llm_first_token < idle_threshold`` — the filler wasn't
        actually needed.
      ``unique_filler_count`` (iter-081): distinct filler IDs
        played across the session.
    """

    fillers_total: int = 0
    filler_turns: int = 0
    filler_false_positives: int = 0
    unique_filler_count: int = 0
    # iter-096: when filler false positives fired AND we have a
    # current idle_threshold AND llm_first_token observations,
    # this carries a recommended new threshold. >0 = render the
    # value next to the "tune up" suggestion; 0.0 = omit.
    recommended_idle_threshold: float = 0.0


def _emit_filler_block(emit, stats: FillerStats) -> None:
    """iter-091: extracted from print_session_summary's filler
    section.

    Renders:
      - "Fillers played: N" (iter-014).
      - "Filler FP rate: M/K (X%)" (iter-051) when any FP fired,
        with iter-096 recommended threshold appended when set.
      - "Filler novelty: M unique / N (X%)" (iter-081) when
        ≥2 fillers played (single-play is trivially 100%).

    Behavior-preserving when ``recommended_idle_threshold == 0.0``:
    byte-for-byte identical to the iter-091 inline version. When
    set, appends "(try N.Ns)" to the FP-rate line.
    """
    if stats.fillers_total <= 0:
        return

    emit(f"    Fillers played:   {stats.fillers_total}")
    # iter-051: false-positive rate among filler turns. Only
    # emits when at least one false positive fired.
    if stats.filler_turns > 0 and stats.filler_false_positives > 0:
        fp_pct = (stats.filler_false_positives / stats.filler_turns) * 100
        # iter-096: append a concrete recommended value when the
        # caller computed one. Keeps the legacy text suffix when
        # not provided (regression-safe for existing tests).
        if stats.recommended_idle_threshold > 0:
            tail = (
                f" — tune idle_threshold up to "
                f"{stats.recommended_idle_threshold:.1f}s"
            )
        else:
            tail = " — tune idle_threshold up"
        emit(
            f"    Filler FP rate:   "
            f"{stats.filler_false_positives}/{stats.filler_turns} "
            f"({fp_pct:.0f}%){tail}"
        )
    # iter-081: filler novelty index — distinct clips / total
    # plays. Skip on single-play sessions (1/1 = 100% trivially).
    if stats.fillers_total >= 2:
        novelty_pct = (stats.unique_filler_count / stats.fillers_total) * 100
        emit(
            f"    Filler novelty:   "
            f"{stats.unique_filler_count} unique / {stats.fillers_total} "
            f"({novelty_pct:.0f}%)"
        )


@dataclass
class ErrorStats:
    """iter-092: aggregated error / failure-mode counters consumed
    by ``_emit_errors_block``. Bundles three related signals:

      ``llm_errors`` (iter-058): turn-fatal LLM exceptions (the
        whole turn was lost).
      ``worker_errors_total`` (iter-058): per-turn synth/play
        exceptions (turn may have produced partial audio).
      ``error_turns_with_audio`` (iter-067): worker-error turns
        where ttfs > 0 — silent partial degradation count.
      ``error_turns_total`` (iter-067): all turns where
        worker_errors > 0 — denominator for the recovery rate.
      ``n`` (iter-058): total completed turns.
      ``false_triggers`` (iter-058): VAD false-trigger count
        (also part of the "Errors:" attempts denominator).
      ``silent_turns`` (iter-079): turns where transcript was
        captured but ttfs == 0 — bot stayed silent despite
        successful STT.
    """

    llm_errors: int = 0
    worker_errors_total: int = 0
    error_turns_with_audio: int = 0
    error_turns_total: int = 0
    n: int = 0
    false_triggers: int = 0
    silent_turns: int = 0


def _emit_errors_block(emit, stats: ErrorStats) -> None:
    """iter-092: extracted from print_session_summary's errors and
    silent-turn block.

    Renders:
      - "Errors: N LLM, M worker (over X attempts)" (iter-058)
        when any error fired.
      - "Worker recovery: M/N turns produced audio (X%)"
        (iter-067) when any worker error fired.
      - "Silent turns: M/N (X%) — bot produced no audio"
        (iter-079) when any silent turn occurred.

    Behavior-preserving: byte-for-byte identical to the inline
    version.
    """
    # iter-058: error rate per stage. Show only when at least
    # one error happened.
    if stats.llm_errors > 0 or stats.worker_errors_total > 0:
        attempts = stats.n + stats.llm_errors + stats.false_triggers
        bits = []
        if stats.llm_errors > 0:
            bits.append(f"{stats.llm_errors} LLM")
        if stats.worker_errors_total > 0:
            bits.append(f"{stats.worker_errors_total} worker")
        emit(
            f"    Errors:           "
            f"{', '.join(bits)} "
            f"(over {attempts} attempt{'' if attempts == 1 else 's'})"
        )
        # iter-067: worker error-recovery success rate.
        if stats.error_turns_total > 0:
            pct = (
                stats.error_turns_with_audio / stats.error_turns_total
            ) * 100
            emit(
                f"    Worker recovery:  "
                f"{stats.error_turns_with_audio}/{stats.error_turns_total} "
                f"turns produced audio "
                f"({pct:.0f}%) — partial degradation"
            )
    # iter-079: silent-turn rate. Distinct from worker errors —
    # no exception fired, the worker just didn't manage to play
    # anything.
    if stats.silent_turns > 0 and stats.n > 0:
        pct = (stats.silent_turns / stats.n) * 100
        emit(
            f"    Silent turns:     "
            f"{stats.silent_turns}/{stats.n} ({pct:.0f}%) — bot produced no audio"
        )


@dataclass
class WpmStats:
    """iter-094: WPM medians consumed by ``_emit_wpm_block``.

    Bundles the iter-064 user-WPM list and iter-046 bot-WPM list
    (both filtered to nonzero values by the caller). The mirror
    gap (iter-064) is computed inline by the helper when both
    are present.

    Fields:
      ``user_wpms``: list of measurable per-turn user WPM values.
      ``bot_wpms``: list of measurable per-turn bot WPM values.
    """

    user_wpms: list = field(default_factory=list)
    bot_wpms: list = field(default_factory=list)


def _emit_wpm_block(emit, stats: WpmStats) -> None:
    """iter-094: extracted from print_session_summary's WPM block.

    Renders:
      - "Median user WPM: NNN" (iter-064) when measurable.
      - "Median bot WPM:  NNN" (iter-046) when measurable.
      - "Mirror gap: ±NN WPM (bot − user)" (iter-064) when both
        are measurable.

    Behavior-preserving: byte-for-byte identical to the inline
    version that lived in print_session_summary.
    """
    if stats.user_wpms:
        median_user_wpm = statistics.median(stats.user_wpms)
        emit(f"    Median user WPM:  {median_user_wpm:.0f}")
    if stats.bot_wpms:
        median_wpm = statistics.median(stats.bot_wpms)
        emit(f"    Median bot WPM:   {median_wpm:.0f}")
        if stats.user_wpms:
            gap = median_wpm - statistics.median(stats.user_wpms)
            emit(f"    Mirror gap:       {gap:+.0f} WPM (bot − user)")


def _emit_wpm_mirror_line(
    emit,
    *,
    active: bool,
    initial_speed: float,
    final_speed: float,
) -> None:
    """iter-215: surface the WPM mirror's per-session speed adaptation.

    Closes iter-214 backlog #1. The iter-213/214 WPM mirror adapts the Kokoro
    ``speed`` toward the user's measured rate turn-to-turn; this line *measures*
    that adaptation so it can be seen working (the iter-210 bot_wpm sentinel and
    the iter-064 "Mirror gap" WPM line should show the gap shrinking when it's
    on) rather than asserted.

    Emits one of two shapes when the mirror is active:
      - speed moved:  ``WPM mirror:       on, speed 1.00 → 1.12 (+0.12) …``
      - speed held:   ``WPM mirror:       on, speed held at 1.00 …``
    The ``held`` branch fires when start and end are within a hair (the mirror's
    deadband kept the rate put — e.g. a user already at the base rate), so the
    operator sees "it was on and chose not to move" rather than a misleading
    ``+0.00``.

    **Suppressed entirely when ``active`` is False** — the off-by-default path
    (no ``wpm_mirror`` config, ``mirror=None``). The summary is then byte-for-
    byte the pre-iter-215 output: a session that never enabled mirroring sees no
    new line. This is the whole point — measuring the adaptation can never
    regress the proven fixed-rate summary.
    """
    if not active:
        return
    delta = final_speed - initial_speed
    if abs(delta) < 0.005:
        # Within rounding of "no movement" — the mirror's deadband held the
        # rate. Report it as held rather than a misleading "+0.00".
        emit(
            f"    WPM mirror:       on, speed held at {initial_speed:.2f} "
            f"(adapts toward user rate — iter-213/214)"
        )
    else:
        emit(
            f"    WPM mirror:       on, speed {initial_speed:.2f} → "
            f"{final_speed:.2f} ({delta:+.2f}) "
            f"(adapted toward user rate — iter-213/214)"
        )


@dataclass
class SentenceStats:
    """iter-095: sentence-shape statistics consumed by
    ``_emit_sentence_block``. Bundles three iter-045/070/059
    signals tied to how the splitter chunked the LLM stream.

    Fields:
      ``sentence_lens`` (iter-045): per-turn mean sentence
        character lengths (filtered to >0).
      ``min_chars_seen`` (iter-070): shortest single sentence
        across the session, or 0 if no measurable turns.
      ``max_chars_seen`` (iter-070): longest single sentence
        across the session, or 0 if no measurable turns.
      ``coverage_values`` (iter-059): per-turn sentence-split
        coverage ratios (filtered to >0).
    """

    sentence_lens: list = field(default_factory=list)
    min_chars_seen: int = 0
    max_chars_seen: int = 0
    coverage_values: list = field(default_factory=list)


def _emit_sentence_block(emit, stats: SentenceStats) -> None:
    """iter-095: extracted from print_session_summary's sentence
    section.

    Renders:
      - "Mean sentence: NN chars" (iter-045) when measurable.
      - "Sentence range: [min..max] chars (session)" (iter-070)
        when min != max.
      - "Split coverage: NN%" (iter-059) when measurable.

    Behavior-preserving: byte-for-byte identical to the inline
    version that lived in print_session_summary.
    """
    if stats.sentence_lens:
        # iter-045: mean across the per-turn means.
        avg_chars = sum(stats.sentence_lens) / len(stats.sentence_lens)
        emit(f"    Mean sentence:    {avg_chars:.0f} chars")
        # iter-070: session-wide range. Skip when min == max
        # (single observation, range degenerate).
        if stats.max_chars_seen > 0 and stats.max_chars_seen != stats.min_chars_seen:
            emit(
                f"    Sentence range:   "
                f"[{stats.min_chars_seen}..{stats.max_chars_seen}] chars (session)"
            )
    if stats.coverage_values:
        # iter-059: median split coverage across turns. <90% is
        # a signal the LLM isn't ending with punctuation often
        # enough — system-prompt opportunity.
        median_cov = statistics.median(stats.coverage_values) * 100
        emit(f"    Split coverage:   {median_cov:.0f}%")


@dataclass
class HistoryStats:
    """iter-097: conversation-history management signals consumed
    by ``_emit_history_block``. Bundles iter-077 context size and
    iter-078 trim-event tracking — both about how the messages
    list grows and gets capped over a session.

    Fields:
      ``context_token_counts`` (iter-077): per-turn approximate
        context tokens sent to the LLM (filtered to >0).
      ``trim_events`` (iter-078): count of times trim_history
        actually evicted ≥1 message.
      ``trim_messages_evicted`` (iter-078): cumulative evicted
        message count across all trim_events.
    """

    context_token_counts: list = field(default_factory=list)
    trim_events: int = 0
    trim_messages_evicted: int = 0


def _emit_history_block(emit, stats: HistoryStats) -> None:
    """iter-097: extracted from print_session_summary's history
    block (iter-077 context tokens + iter-078 trim events).

    Renders:
      - "Context tokens: NN median, MM max" (iter-077) when any
        turn produced a count.
      - "Context growth: ±NN tokens (turn 1 → turn N)" (iter-077)
        when ≥3 turns have measurable context.
      - "Trim events: N (M evicted, X.X/event)" (iter-078) when
        any trim fired.

    Behavior-preserving: byte-for-byte identical to the inline
    version that lived in print_session_summary.
    """
    # iter-077: context size summary. Median = typical per-call
    # cost; max = worst case. If max ≫ median, late turns blew
    # up — likely a trim regression.
    if stats.context_token_counts:
        med_ctx = statistics.median(stats.context_token_counts)
        max_ctx = max(stats.context_token_counts)
        emit(f"    Context tokens:   {med_ctx:.0f} median, {max_ctx} max")
        if len(stats.context_token_counts) >= 3:
            growth = (
                stats.context_token_counts[-1]
                - stats.context_token_counts[0]
            )
            emit(
                f"    Context growth:   "
                f"{growth:+d} tokens (turn 1 → turn {len(stats.context_token_counts)})"
            )
    # iter-078: trim event rate. evicted/events ratio surfaces
    # severity (1.0 = steady-state one-eviction-per-trim;
    # higher = catching up after a longer interval).
    if stats.trim_events > 0:
        ratio = stats.trim_messages_evicted / stats.trim_events
        emit(
            f"    Trim events:      "
            f"{stats.trim_events} ({stats.trim_messages_evicted} evicted, "
            f"{ratio:.1f}/event)"
        )


@dataclass
class RecordingStats:
    """iter-103: bundle the contiguous mic-recording health
    signals into a single Stats object.

    iter-037 (mic stale): aggregate count of stale frames flushed
      across all turns. High count → constant echo or input
      driver hiccups.
    iter-048 (VAD false-trigger rate): turns where the recorder
      fired but no transcript came back, divided by total
      attempts (false_triggers + n).
    """

    stale_total: int = 0
    false_triggers: int = 0
    n: int = 0


def _emit_recording_block(emit, stats: RecordingStats) -> None:
    """iter-103: extracted from print_session_summary's recording
    block (iter-037 mic stale + iter-048 VAD false-trigger).

    Renders:
      - "Mic stale: N frames (X.Xs) — check echo cancellation"
        (iter-037) when any stale frames were flushed.
      - "VAD false-trig: F/A (P%) — tune silence_threshold or
        min_speech_duration" (iter-048) when ≥1 false trigger
        fired.

    Both lines use the leading 4-space indent (no tree pipe), so
    test assertions should match on substrings, not exact line
    equality. Behavior-preserving: byte-for-byte identical to the
    inline version that lived in print_session_summary.
    """
    if stats.stale_total:
        # iter-037: surface aggregate stale-frame total so a "session
        # had constant echo" pattern is visible at the end of the run.
        stale_seconds_total = stats.stale_total / 16000
        emit(
            f"    Mic stale:        {stats.stale_total} frames "
            f"({stale_seconds_total:.1f}s) — check echo cancellation"
        )
    # iter-048: VAD false-trigger rate. Only emit when at least one
    # false trigger happened — clean sessions don't need the line.
    if stats.false_triggers > 0:
        attempts = stats.false_triggers + stats.n
        pct = (stats.false_triggers / attempts) * 100
        emit(
            f"    VAD false-trig:   {stats.false_triggers}/{attempts} "
            f"({pct:.0f}%) — tune silence_threshold or min_speech_duration"
        )


def _emit_primed_audio_line(emit, primed_seconds_total: float) -> None:
    """iter-104: extracted from print_session_summary's iter-057
    primed-audio line. Reports cumulative seconds of audio carried
    into the next turn via the primed_frames mechanism.

    Suppressed when total is 0 (no priming happened — clean
    sessions don't need the line). Behavior-preserving: the
    formatting and "validates iter-025" rationale are unchanged
    from the inline version.
    """
    if primed_seconds_total > 0:
        emit(
            f"    Primed audio:     "
            f"{primed_seconds_total:.1f}s "
            f"(carried into next turn — validates iter-025)"
        )


def _emit_stranded_utterance_line(emit, stranded: Optional[str]) -> None:
    """iter-160: surface a mid-thought utterance the organic
    ``UtteranceAggregator`` was still holding when the session ended.

    backlog #9's hold-and-merge driver (iter-156) holds an utterance that
    looks unfinished, waiting for a quick continuation to merge on. Mid-
    session that pending always resolves: the NEXT utterance's measured
    silence gap forces a NEW release inside ``offer``. The one case
    ``offer`` can never reach is *shutdown* — the user trailed off after a
    fragment, never spoke again, then hit Ctrl+C. ``run_session`` flushes
    the aggregator on exit and records the released text on
    ``state.stranded_utterance``; this line makes that dropped fragment
    visible rather than silently lost.

    Suppressed (the overwhelmingly common case) when ``stranded`` is
    ``None`` or blank: no aggregator wired in, nothing was held, or
    half-duplex mode (which never holds). A clean session never sees it.
    """
    if stranded and stranded.strip():
        emit(
            f"    Stranded uttr.:   {stranded.strip()!r} "
            f"(held mid-thought at exit, never completed — iter-160)"
        )


def _emit_displaced_utterances_line(emit, displaced) -> None:
    """iter-162: surface mid-thought fragments the organic
    ``UtteranceAggregator`` released *alongside* a responded turn.

    When a held mid-thought fragment ("I was thinking about the") is NOT
    followed by a quick continuation but by a long silence and then a
    genuinely new utterance ("What time is it?"), the buffer releases the
    abandoned fragment as its own ``NEW`` turn *and* the new utterance in
    one ``offer`` — two distinct turns. iter-159's ``resolve_turn`` used to
    space-glue them into one garbled LLM input
    (``"I was thinking about the What time is it?"``). iter-162 responds to
    the new utterance only and routes the abandoned fragment(s) here — the
    mid-session analog of iter-160's shutdown ``stranded_utterance``.

    Suppressed (the overwhelmingly common case) when nothing was displaced:
    no aggregator wired in, every release was a single turn, or half-duplex
    mode (which never releases more than one turn at a time). A clean session
    never sees it.
    """
    frags = [f.strip() for f in (displaced or []) if f and f.strip()]
    if not frags:
        return
    if len(frags) == 1:
        emit(
            f"    Displaced uttr.:  {frags[0]!r} "
            f"(abandoned mid-thought, displaced by a new turn — iter-162)"
        )
    else:
        emit(
            f"    Displaced uttr.:  {len(frags)} fragments abandoned "
            f"mid-thought, displaced by new turns — iter-162:"
        )
        for frag in frags:
            emit(f"                      {frag!r}")


def _emit_flushed_utterances_line(emit, flushed) -> None:
    """iter-167: surface mid-thought fragments the organic
    ``UtteranceAggregator`` flushed *mid-session* on a long inter-turn idle
    silence.

    Backlog #9's hold-and-merge driver (iter-156) holds an utterance that looks
    unfinished, waiting for a quick continuation to merge on. Before iter-167
    that pending could only ever resolve when the NEXT utterance arrived
    (iter-162's displaced path) or at shutdown (iter-160's ``stranded_utterance``).
    The one case neither reached is the user who trails off mid-thought and then
    says *nothing* for a long beat. iter-165's recorder idle timeout + iter-166's
    ``idle_timed_out`` flag let ``run_session`` notice that silence; iter-164's
    ``decide_silence_flush`` (wired here via an injected decider) decides to give
    up waiting and flush the fragment to the engine as its own turn. The flushed
    text is recorded on ``state.flushed_utterances`` so this line surfaces it.

    Suppressed (the overwhelmingly common case) when nothing was flushed: no
    aggregator wired in, no ``flush_decider`` configured, no idle timeout fired,
    or half-duplex mode (which never holds). A clean session never sees it.
    """
    frags = [f.strip() for f in (flushed or []) if f and f.strip()]
    if not frags:
        return
    if len(frags) == 1:
        emit(
            f"    Flushed uttr.:    {frags[0]!r} "
            f"(trailed off, flushed on idle silence mid-session — iter-167)"
        )
    else:
        emit(
            f"    Flushed uttr.:    {len(frags)} fragments flushed "
            f"on idle silence mid-session — iter-167:"
        )
        for frag in frags:
            emit(f"                      {frag!r}")


def _emit_wer_line(emit, wer_values: list[float]) -> None:
    """iter-105: report median + max WER across turns where a
    reference transcript was supplied. Suppressed when no turn
    measured WER (the default — most sessions don't have
    ground-truth references).

    Format mirrors `_emit_history_block`'s context-tokens line:
    "WER: M.MM median, X.XX max (N turns measured)".

    Production interpretation guide (recorded inline so the
    operator sees a calibration anchor):
      < 0.10 — production-grade STT
      0.10-0.20 — acceptable for clean audio
      0.20-0.40 — degraded; tune mic / silence_threshold
      > 0.40 — STT is failing — check input quality
    """
    if not wer_values:
        return
    import statistics
    med = statistics.median(wer_values)
    worst = max(wer_values)
    emit(
        f"    WER:              "
        f"{med:.2f} median, {worst:.2f} max "
        f"({len(wer_values)} turns measured)"
    )


def _longest_consecutive_run(values: list) -> tuple[int, object]:
    """iter-116: find the longest consecutive-equal run in a list.

    Shared by iter-114 (`_emit_filler_diversity_line`) and iter-115
    (`_emit_naturalness_consistency_line`). Both helpers had the
    same single-pass scan duplicated verbatim; this consolidates.

    Returns ``(length, value)`` where ``length`` is the count of
    consecutive equal items in the longest run and ``value`` is
    that item. For empty input, returns ``(0, None)``. Ties on
    length resolve to the EARLIER run (first encountered) —
    matches both prior call-sites' behavior.

    No filtering happens here — callers pre-filter (zeros for
    iter-114, empty strings for iter-115). Keeps the helper a
    pure list-scanning primitive.
    """
    if not values:
        return (0, None)

    longest_run = 1
    longest_value = values[0]
    cur_run = 1
    cur = values[0]
    for v in values[1:]:
        if v == cur:
            cur_run += 1
            if cur_run > longest_run:
                longest_run = cur_run
                longest_value = cur
        else:
            cur = v
            cur_run = 1
    return (longest_run, longest_value)


def _emit_filler_diversity_line(
    emit, filler_ids: list[int], threshold: int = 3,
) -> None:
    """iter-114: defensive sentinel for iter-113's cross-turn
    filler variety fix.

    Scans the per-turn ``last_filler_id`` sequence for runs of
    the SAME id ≥ ``threshold`` (default 3). When found, emits a
    warning line so the operator notices when the cross-turn FIFO
    regression happened — without iter-113's `recent_filler_ids`
    deque, the picker can pick the same filler turn after turn.

    Filtering rules:
      - 0 (no filler fired this turn) is excluded — only consecutive
        non-zero ids count.
      - Runs are counted in the FILTERED sequence (zeroed turns
        don't break a run). Rationale: the user perception is
        "same filler 3 times in a row" regardless of whether
        intervening turns happened to skip the filler.

    Suppressed when:
      - No turn ever fired a filler (filler_ids is all zeros) —
        non-filler sessions don't need the line.
      - The longest run is below threshold — clean variety.

    Output format mirrors the iter-074 bargeable-warning style:

        Filler diversity:  filler X repeated 4 turns running
                           — iter-113 cross-turn FIFO may not be wired

    Threshold rationale: 3 consecutive same-fillers is the
    smallest pattern a user typically perceives as repetition. 2
    is too noisy (random can trivially produce 2 in a row); 4+
    raises the bar so far that real regressions wouldn't fire it
    until many turns later.
    """
    # Filter out zero (no-filler turns) but preserve the sequence
    # of fillers AS PLAYED — runs are counted on this filtered list.
    fired = [fid for fid in filler_ids if fid != 0]
    if not fired:
        return

    # iter-116: shared run-finder.
    longest_run, longest_id = _longest_consecutive_run(fired)
    if longest_run < threshold:
        return

    emit(
        f"    Filler diversity: filler {longest_id} repeated "
        f"{longest_run} turns running "
        f"— iter-113 cross-turn FIFO may not be wired"
    )


def _emit_naturalness_consistency_line(
    emit, buckets: list[str], threshold: int = 5,
) -> None:
    """iter-115: detect consecutive runs of the same non-"natural"
    naturalness bucket (iter-053). When 5+ turns in a row land in
    "rushed" or "slow", the speed setting needs adjustment.

    Built on the same shape as iter-114's
    ``_emit_filler_diversity_line`` — confirms the diversity-check
    pattern is reusable across different metrics. Same rationale
    for filtering empty values (no audio played that turn) before
    counting runs.

    Threshold = 5 is higher than iter-114's 3 because natural
    speech-rate variation is normal — a brief "rushed" or "slow"
    streak isn't a config problem. 5+ consecutive same-bucket
    turns is the smallest pattern where "the operator's speed
    config is wrong" is more likely than "noise."

    "natural" runs are NEVER flagged: the goal is to be in that
    bucket. The check fires only on rushed/slow.

    iter-126: "natural" is filtered out BEFORE the run scan, not
    just suppressed at the end. This fixes iter-115's documented
    limitation: a long "natural" run that overshadows a shorter
    "rushed" / "slow" run no longer hides the rushed/slow signal.
    The user-perception rationale matches iter-114's zero-filter
    rule: N rushed turns is N rushed turns regardless of intervening
    "natural" turns. Filtering before the scan also makes the
    helper consistent across iter-114/115/120 — all three filter
    "uninteresting" values up front.

    Output mirrors iter-114's "name the responsible iteration"
    convention so operators can find the fix path:

        Naturalness: 6 consecutive 'rushed' turns
                     — consider reducing speed (iter-053 bucket)
    """
    # iter-126: filter empty AND "natural" up front. Both are
    # "uninteresting" values that shouldn't break runs of the
    # bucket types we care about.
    filtered = [b for b in buckets if b and b != "natural"]
    if not filtered:
        return

    longest_run, longest_bucket = _longest_consecutive_run(filtered)
    if longest_run < threshold:
        return
    # No need for an explicit `if longest_bucket == "natural": return`
    # since "natural" was filtered out. Pre-iter-126 had this
    # guard as a backstop; iter-126 makes it dead code, removed.

    if longest_bucket == "rushed":
        suggestion = "consider reducing speed"
    elif longest_bucket == "slow":
        suggestion = "consider increasing speed"
    else:
        # Defensive: an unknown bucket name shouldn't break the
        # line — emit a generic suggestion.
        suggestion = "consider tuning speed"

    emit(
        f"    Naturalness:      {longest_run} consecutive "
        f"{longest_bucket!r} turns — {suggestion} "
        f"(iter-053 bucket)"
    )


def _emit_barge_phase_consistency_line(
    emit, phases: list[str], threshold: int = 4,
) -> None:
    """iter-120: detect consecutive runs of the same barge-in
    phase. Third instance of the diversity-check pattern after
    iter-114 (filler) and iter-115 (naturalness) — uses the
    shared ``_longest_consecutive_run`` helper from iter-116.

    Two phases are tracked (iter-047):
      - "llm_stream" — user barged BEFORE bot speech started.
        High recurrence suggests the user is impatient with LLM
        TTFB, or the bot is slow to start.
      - "playback" — user barged DURING bot speech. High
        recurrence suggests the bot speaks too long, or the user
        is interrupt-happy by habit.

    Both phases are flagged on consecutive runs (unlike
    iter-115's naturalness, where "natural" was the desired state
    and excluded). Threshold = 4 (lower than iter-115's 5)
    because barge events are already rarer + more semantically
    loaded than naturalness buckets.

    Suppression rules:
      - Empty strings (no barge that turn) are filtered before
        the run scan, mirroring iter-114/iter-115.
      - When the longest run is below threshold, no warning fires.
      - When no turn ever barged, the line stays silent — quiet
        sessions don't need it.

    Output mirrors iter-114/iter-115's "name the responsible
    iteration" convention so operators can find the fix path:

        Barge phase: 5 consecutive 'playback' barges
                     — user habit or bot speaks too long (iter-047)
    """
    non_empty = [p for p in phases if p]
    if not non_empty:
        return

    longest_run, longest_phase = _longest_consecutive_run(non_empty)
    if longest_run < threshold:
        return

    if longest_phase == "llm_stream":
        suggestion = (
            "user impatient with LLM TTFB, or bot slow to start"
        )
    elif longest_phase == "playback":
        suggestion = "user habit or bot speaks too long"
    else:
        # Defensive: an unrecognized phase string still emits a
        # generic warning instead of silently dropping the
        # signal. Future iterations may add more phases.
        suggestion = "consistent barge phase — investigate"

    emit(
        f"    Barge phase:      {longest_run} consecutive "
        f"{longest_phase!r} barges — {suggestion} "
        f"(iter-047 phase)"
    )


def _sentence_length_bucket(mean_chars: float) -> str:
    """iter-128: bucket a per-turn ``mean_sentence_chars`` into a
    coarse category. Used by ``_emit_sentence_length_consistency_line``
    to detect runs of unusually-short or unusually-long bot output.

    Buckets (chosen against iter-095 perf-data observations
    where typical mean_sentence_chars sits at 25-50):

        ``very_short`` — < 15 chars: choppy. Caused by an
            over-aggressive splitter (iter-088 with
            ``AGGRESSIVE_MIN_CHARS`` set too low) or an LLM that
            keeps emitting one-word answers.
        ``short``      — 15-30 chars: brief but not problematic.
        ``medium``     — 30-60 chars: the desired state.
        ``long``       — > 60 chars: wall-of-text. Either LLM
            rambles or the splitter is too lax.

    Returns ``""`` when ``mean_chars`` is non-positive (no
    sentences this turn) — empty-string filter applies in the
    consumer, mirroring iter-114/115/120/126.
    """
    if mean_chars <= 0:
        return ""
    if mean_chars < 15:
        return "very_short"
    if mean_chars < 30:
        return "short"
    if mean_chars < 60:
        return "medium"
    return "long"


def _emit_sentence_length_consistency_line(
    emit, mean_chars_list: list[float], threshold: int = 5,
) -> None:
    """iter-128: detect consecutive runs of unusually-short or
    unusually-long sentence output. Fourth instance of the
    diversity-check pattern after iter-114 (filler), iter-115
    (naturalness), iter-120 (barge-phase). First instance applied
    to a CONTINUOUS metric — buckets it via
    ``_sentence_length_bucket`` before running the scan.

    Filter rule (mirrors iter-126's "natural" exclusion): drop
    "medium" and "short" buckets before the scan. They're the
    "fine" states; only "very_short" and "long" warrant warning.
    Empty buckets (turns with no sentences) also drop.

    Threshold = 5: same as iter-115, since sentence-length is a
    similarly noisy signal where brief excursions are normal.

    Output:

        Sentence length: 5 consecutive 'very_short' turns
                         — splitter may be over-aggressive
                         (iter-095 mean_sentence_chars)
    """
    # Bucketize, then drop "uninteresting" buckets (empty,
    # medium, short).
    interesting = {"very_short", "long"}
    filtered = [
        b for b in (
            _sentence_length_bucket(mc) for mc in mean_chars_list
        )
        if b in interesting
    ]
    if not filtered:
        return

    longest_run, longest_bucket = _longest_consecutive_run(filtered)
    if longest_run < threshold:
        return

    if longest_bucket == "very_short":
        suggestion = "splitter may be over-aggressive"
    elif longest_bucket == "long":
        suggestion = "splitter may be too lax (or LLM rambles)"
    else:
        # Defensive: future buckets that pass the filter rule.
        suggestion = "consider tuning splitter"

    emit(
        f"    Sentence length:  {longest_run} consecutive "
        f"{longest_bucket!r} turns — {suggestion} "
        f"(iter-095 mean_sentence_chars)"
    )


def _stt_rtf_bucket(rtf: float) -> str:
    """iter-140: bucket a per-turn ``stt_rtf`` (stt_time /
    speech_duration) into a coarse category. Used by
    ``_emit_stt_rtf_consistency_line`` to detect runs of STT that
    consistently runs slower than realtime — the signal that the
    chosen engine/model is too heavy for the host hardware and
    end-of-turn STT is the latency bottleneck.

    Buckets (chosen against the iter-049 RTF semantics where
    mlx-whisper on Apple Silicon lands ~0.1-0.3):

        ``realtime`` — < 1.0: STT keeps up; the desired state.
            Can be invoked inline at end-of-turn with no stall.
        ``slow``     — 1.0-2.0: STT is the bottleneck. Streaming
            partial transcription would help, or a smaller model.
        ``very_slow``— > 2.0: STT takes 2x+ the speech duration.
            The engine/model is badly mismatched to the hardware.

    Returns ``""`` when ``rtf`` is non-positive (false-trigger
    turn or stt_time unmeasured) — empty-string filter applies in
    the consumer, mirroring iter-114/115/120/126/128.
    """
    if rtf <= 0:
        return ""
    if rtf < 1.0:
        return "realtime"
    if rtf <= 2.0:
        return "slow"
    return "very_slow"


def _emit_stt_rtf_consistency_line(
    emit, stt_rtf_list: list[float], threshold: int = 5,
) -> None:
    """iter-140: detect consecutive runs of STT running slower
    than realtime. FIFTH instance of the diversity-check pattern
    after iter-114 (filler), iter-115/126 (naturalness), iter-120
    (barge-phase), iter-128 (sentence-length). Second instance
    applied to a CONTINUOUS metric — buckets it via
    ``_stt_rtf_bucket`` before running the scan (same shape as
    iter-128).

    Filter rule (mirrors iter-128's "medium"/"short" exclusion):
    drop the "realtime" bucket before the scan. It's the "fine"
    state; only "slow" and "very_slow" warrant warning. Empty
    buckets (turns with no measurable STT) also drop.

    Threshold = 5: same as iter-115/128, since RTF varies turn to
    turn with utterance length and a brief slow excursion is
    normal; a sustained run is the real signal.

    Output:

        STT speed: 5 consecutive 'very_slow' turns — STT engine
                   is the bottleneck, try a smaller model or
                   streaming STT (iter-049 stt_rtf)
    """
    # Bucketize, then drop the "uninteresting" bucket (empty,
    # realtime).
    interesting = {"slow", "very_slow"}
    filtered = [
        b for b in (
            _stt_rtf_bucket(r) for r in stt_rtf_list
        )
        if b in interesting
    ]
    if not filtered:
        return

    longest_run, longest_bucket = _longest_consecutive_run(filtered)
    if longest_run < threshold:
        return

    if longest_bucket == "very_slow":
        suggestion = (
            "STT engine is badly mismatched to the hardware "
            "(>2x realtime)"
        )
    elif longest_bucket == "slow":
        suggestion = (
            "STT is the bottleneck, try a smaller model or "
            "streaming STT"
        )
    else:
        # Defensive: future buckets that pass the filter rule.
        suggestion = "consider a lighter STT engine"

    emit(
        f"    STT speed:        {longest_run} consecutive "
        f"{longest_bucket!r} turns — {suggestion} "
        f"(iter-049 stt_rtf)"
    )


def _tts_rtf_bucket(rtf: float) -> str:
    """iter-141: bucket a per-turn ``tts_rtf`` (tts_time /
    audio_seconds_total) into a coarse category. Used by
    ``_emit_tts_rtf_consistency_line`` to detect runs of TTS that
    consistently synthesizes slower than the audio it produces —
    the signal that synth, not playback, is the latency bottleneck
    and synth-overlap won't help.

    SIXTH instance of the diversity-check pattern, and the THIRD
    applied to a continuous metric (after iter-128 sentence-length
    and iter-140 stt-rtf). A near-mechanical clone of
    ``_stt_rtf_bucket`` (iter-140) — same boundaries, same
    semantics, different source metric (iter-050 tts_rtf).

    Buckets (chosen against the iter-050 RTF semantics where
    Kokoro on Apple Silicon lands ~0.1-0.3):

        ``realtime`` — < 1.0: synth keeps up; the desired state.
            Synth-overlap streams usefully ahead of playback.
        ``slow``     — 1.0-2.0: synth is the bottleneck. A lighter
            voice/engine or pre-rendered fillers would help.
        ``very_slow``— > 2.0: synth takes 2x+ the audio duration.
            The engine/voice is badly mismatched to the hardware.

    Returns ``""`` when ``rtf`` is non-positive (no audio produced
    this turn or tts_time unmeasured) — empty-string filter applies
    in the consumer, mirroring iter-114/115/120/126/128/140.
    """
    if rtf <= 0:
        return ""
    if rtf < 1.0:
        return "realtime"
    if rtf <= 2.0:
        return "slow"
    return "very_slow"


def _emit_tts_rtf_consistency_line(
    emit, tts_rtf_list: list[float], threshold: int = 5,
) -> None:
    """iter-141: detect consecutive runs of TTS synthesizing
    slower than realtime. SIXTH instance of the diversity-check
    pattern after iter-114 (filler), iter-115/126 (naturalness),
    iter-120 (barge-phase), iter-128 (sentence-length), iter-140
    (stt-rtf). THIRD instance applied to a CONTINUOUS metric —
    buckets it via ``_tts_rtf_bucket`` before running the scan
    (same shape as iter-128/140).

    Filter rule (mirrors iter-140's "realtime" exclusion): drop the
    "realtime" bucket before the scan. It's the "fine" state; only
    "slow" and "very_slow" warrant warning. Empty buckets (turns
    with no audio produced) also drop.

    Threshold = 5: same as iter-115/128/140, since RTF varies turn
    to turn with utterance length and a brief slow excursion is
    normal; a sustained run is the real signal.

    Output:

        TTS speed: 5 consecutive 'very_slow' turns — TTS engine
                   is the bottleneck, try a lighter voice or
                   pre-rendered fillers (iter-050 tts_rtf)
    """
    # Bucketize, then drop the "uninteresting" bucket (empty,
    # realtime).
    interesting = {"slow", "very_slow"}
    filtered = [
        b for b in (
            _tts_rtf_bucket(r) for r in tts_rtf_list
        )
        if b in interesting
    ]
    if not filtered:
        return

    longest_run, longest_bucket = _longest_consecutive_run(filtered)
    if longest_run < threshold:
        return

    if longest_bucket == "very_slow":
        suggestion = (
            "TTS engine is badly mismatched to the hardware "
            "(>2x realtime)"
        )
    elif longest_bucket == "slow":
        suggestion = (
            "TTS is the bottleneck, try a lighter voice or "
            "pre-rendered fillers"
        )
    else:
        # Defensive: future buckets that pass the filter rule.
        suggestion = "consider a lighter TTS engine"

    emit(
        f"    TTS speed:        {longest_run} consecutive "
        f"{longest_bucket!r} turns — {suggestion} "
        f"(iter-050 tts_rtf)"
    )


def _llm_tps_bucket(tps: float) -> str:
    """iter-142: bucket a per-turn ``llm_tps`` (LLM stream
    throughput in tokens/sec, measured after first token) into a
    coarse category. Used by ``_emit_llm_tps_consistency_line`` to
    detect runs of the LLM streaming slowly — the signal that the
    model can't feed complete sentences to the TTS worker fast
    enough, starving synth-overlap regardless of how fast STT/TTS
    run.

    SEVENTH instance of the diversity-check pattern, and the FOURTH
    applied to a continuous metric (after iter-128 sentence-length,
    iter-140 stt-rtf, iter-141 tts-rtf). UNLIKE the three RTF-style
    bucketers, ``llm_tps`` is "bigger is better" — the fine state is
    a HIGH value, so the boundaries invert: small tps is the
    problematic end. This is the first inverted-direction continuous
    bucketer in the family.

    Buckets (chosen against iter-052's TPS semantics — local 7B-13B
    models on Apple Silicon land 30-80 tps, cloud APIs 20-60 tps):

        ``fast``   — >= 25 tps: the LLM keeps up; the desired state.
            Complete sentences reach the worker fast enough that the
            iter-008 streaming-overlap design buys real TTFS savings.
        ``slow``   — 10-25 tps: the LLM lags. Sentences arrive in
            bursts; synth-overlap is partially starved.
        ``very_slow``— < 10 tps: the LLM is the dominant bottleneck.
            The worker idles waiting for tokens; a smaller/quantized
            model or a faster backend is needed.

    Returns ``""`` when ``tps`` is non-positive (no measurable LLM
    stream this turn — e.g. a single-token or empty response) —
    empty-string filter applies in the consumer, mirroring
    iter-114/115/120/126/128/140/141.
    """
    if tps <= 0:
        return ""
    if tps >= 25.0:
        return "fast"
    if tps >= 10.0:
        return "slow"
    return "very_slow"


def _emit_llm_tps_consistency_line(
    emit, llm_tps_list: list[float], threshold: int = 5,
) -> None:
    """iter-142: detect consecutive runs of the LLM streaming
    slower than its useful throughput. SEVENTH instance of the
    diversity-check pattern after iter-114 (filler), iter-115/126
    (naturalness), iter-120 (barge-phase), iter-128
    (sentence-length), iter-140 (stt-rtf), iter-141 (tts-rtf).
    FOURTH instance applied to a CONTINUOUS metric — buckets it via
    ``_llm_tps_bucket`` before running the scan (same shape as
    iter-128/140/141).

    Filter rule: drop the ``"fast"`` bucket before the scan. It's
    the "fine" state; only "slow" and "very_slow" warrant warning.
    Empty buckets (turns with no measurable LLM stream) also drop.
    NOTE the inversion versus iter-140/141: there the fine bucket is
    "realtime" (a LOW value); here it's "fast" (a HIGH value),
    because tps is bigger-is-better. The filter rule absorbs the
    inversion — the run-scan stays direction-agnostic.

    Threshold = 5: same as iter-115/128/140/141, since tps varies
    turn to turn with prompt size and backend warm-up and a brief
    slow excursion is normal; a sustained run is the real signal.

    Output:

        LLM speed: 5 consecutive 'very_slow' turns — LLM is the
                   dominant bottleneck, try a smaller/quantized
                   model or a faster backend (iter-052 llm_tps)
    """
    # Bucketize, then drop the "uninteresting" bucket (empty, fast).
    interesting = {"slow", "very_slow"}
    filtered = [
        b for b in (
            _llm_tps_bucket(t) for t in llm_tps_list
        )
        if b in interesting
    ]
    if not filtered:
        return

    longest_run, longest_bucket = _longest_consecutive_run(filtered)
    if longest_run < threshold:
        return

    if longest_bucket == "very_slow":
        suggestion = (
            "LLM is the dominant bottleneck, try a "
            "smaller/quantized model or a faster backend"
        )
    elif longest_bucket == "slow":
        suggestion = (
            "LLM stream lags synth, try a smaller model or "
            "fewer context tokens"
        )
    else:
        # Defensive: future buckets that pass the filter rule.
        suggestion = "consider a faster LLM backend"

    emit(
        f"    LLM speed:        {longest_run} consecutive "
        f"{longest_bucket!r} turns — {suggestion} "
        f"(iter-052 llm_tps)"
    )


def _streaming_overlap_bucket(ratio: float) -> str:
    """iter-143: bucket a per-turn ``streaming_overlap_ratio`` (the
    fraction of bot synth that ran concurrently with LLM streaming,
    iter-043) into a coarse category. Used by
    ``_emit_streaming_overlap_consistency_line`` to detect runs of
    turns where the iter-008 streaming-overlap design isn't paying
    off — the worker barely overlaps synth with the LLM stream, so
    TTFS savings evaporate.

    EIGHTH instance of the diversity-check pattern, and the FIFTH
    applied to a continuous metric (after iter-128 sentence-length,
    iter-140 stt-rtf, iter-141 tts-rtf, iter-142 llm-tps). Like
    iter-142 ``llm_tps`` — and UNLIKE the iter-140/141 RTF
    bucketers — ``streaming_overlap_ratio`` is "bigger is better":
    the fine state is a HIGH value (lots of overlap), so the
    boundaries invert and the problematic end is a small ratio. This
    is the SECOND inverted-direction continuous bucketer.

    Buckets (chosen against iter-043's overlap semantics — the
    session-summary "Median overlap" line already calls >50% healthy
    and <20% a sign overlap isn't happening):

        ``high``  — >= 0.50: the worker generally got audio out
            before the LLM finished; the iter-008 design is paying
            off. The desired state.
        ``low``   — 0.20-0.50: overlap is partial. The bot is
            responding fast or the LLM is chatty enough that synth
            and stream only partly overlap.
        ``very_low``— < 0.20 (but > 0): overlap is essentially not
            happening — synth runs sequentially after the stream.
            Investigate first-sentence latency (iter-038 TTFsent)
            and synth time.

    Returns ``""`` when ``ratio`` is non-positive (no measurable
    overlap this turn — e.g. a single-sentence response where the
    whole-stream ratio is undefined) — empty-string filter applies
    in the consumer, mirroring iter-114/115/120/126/128/140/141/142.
    """
    if ratio <= 0:
        return ""
    if ratio >= 0.50:
        return "high"
    if ratio >= 0.20:
        return "low"
    return "very_low"


def _emit_streaming_overlap_consistency_line(
    emit, overlap_list: list[float], threshold: int = 5,
) -> None:
    """iter-143: detect consecutive runs of turns where the
    streaming-overlap design barely overlapped synth with the LLM
    stream. EIGHTH instance of the diversity-check pattern after
    iter-114 (filler), iter-115/126 (naturalness), iter-120
    (barge-phase), iter-128 (sentence-length), iter-140 (stt-rtf),
    iter-141 (tts-rtf), iter-142 (llm-tps). FIFTH instance applied
    to a CONTINUOUS metric — buckets it via
    ``_streaming_overlap_bucket`` before running the scan (same
    shape as iter-128/140/141/142).

    Filter rule: drop the ``"high"`` bucket before the scan. It's
    the "fine" state; only "low" and "very_low" warrant warning.
    Empty buckets (turns with no measurable overlap) also drop.
    Like iter-142 and UNLIKE iter-140/141, the fine bucket is a HIGH
    value ("high") because overlap is bigger-is-better — the SECOND
    inverted-direction instance. The filter rule absorbs the
    inversion; the run-scan stays direction-agnostic.

    Threshold = 5: same as iter-115/128/140/141/142. Overlap varies
    turn to turn with response length and LLM speed, and a brief
    low-overlap excursion (e.g. a one-sentence reply) is normal; a
    sustained run is the real signal that the iter-008 design is
    failing to mask synth.

    Output:

        Synth overlap: 5 consecutive 'very_low' turns — synth runs
                       sequentially after the LLM stream; check
                       first-sentence latency and synth time
                       (iter-043 streaming_overlap_ratio)
    """
    # Bucketize, then drop the "uninteresting" bucket (empty, high).
    interesting = {"low", "very_low"}
    filtered = [
        b for b in (
            _streaming_overlap_bucket(r) for r in overlap_list
        )
        if b in interesting
    ]
    if not filtered:
        return

    longest_run, longest_bucket = _longest_consecutive_run(filtered)
    if longest_run < threshold:
        return

    if longest_bucket == "very_low":
        suggestion = (
            "synth runs sequentially after the LLM stream; "
            "check first-sentence latency and synth time"
        )
    elif longest_bucket == "low":
        suggestion = (
            "overlap is only partial; the bot may be replying "
            "too fast or the LLM stream lags synth"
        )
    else:
        # Defensive: future buckets that pass the filter rule.
        suggestion = "investigate first-sentence latency and synth time"

    emit(
        f"    Synth overlap:    {longest_run} consecutive "
        f"{longest_bucket!r} turns — {suggestion} "
        f"(iter-043 streaming_overlap_ratio)"
    )


def _synth_dispatch_bucket(seconds: float) -> str:
    """iter-208: bucket a per-turn ``synth_dispatch_seconds``
    (iter-076's TTFS attribution residual — the part of TTFS not
    explained by ``stt_time + llm_first_sentence``: synth time,
    speaker open, queue dispatch, audio-device buffering) into a
    coarse category. Used by ``_emit_synth_dispatch_consistency_line``
    to detect runs where the final pipeline leg consistently
    dominates time-to-first-sound.

    Buckets (chosen against VISION's sub-500ms TTFS target, where
    this leg should be a small slice of the budget — STT and LLM
    already have their own sentinels at iter-140 and iter-142):

        ``fast``     — < 0.15s: synth+dispatch is a thin slice of
            TTFS; the desired state.
        ``slow``     — 0.15-0.35s: the leg is eating a noticeable
            chunk of the opening latency. A lighter voice/engine,
            a warmer speaker, or smaller first-chunk dispatch
            would help.
        ``very_slow``— > 0.35s: synth/dispatch alone blows most of
            the sub-500ms budget; the bot feels laggy before the
            first word regardless of how fast STT and the LLM ran.

    This is the THIRD and final TTFS-leg sentinel, completing the
    iter-076 decomposition: STT (iter-140 ``stt_rtf``) + LLM
    (iter-142 ``llm_tps``) + synth/dispatch (here). Like
    iter-140/141 and UNLIKE iter-142/143, the fine bucket is a LOW
    value — smaller latency is better, so the boundaries are NOT
    inverted.

    Returns ``""`` when ``seconds`` is non-positive (no audio this
    turn or the residual was unmeasured) — empty-string filter
    applies in the consumer, mirroring iter-114/115/120/126/128/
    140/141.
    """
    if seconds <= 0:
        return ""
    if seconds < 0.15:
        return "fast"
    if seconds <= 0.35:
        return "slow"
    return "very_slow"


def _emit_synth_dispatch_consistency_line(
    emit, synth_dispatch_list: list[float], threshold: int = 5,
) -> None:
    """iter-208: detect consecutive runs of turns where the
    synth+dispatch leg of TTFS (iter-076 ``synth_dispatch_seconds``)
    consistently dominated the opening latency. NINTH instance of
    the diversity-check pattern after iter-114 (filler),
    iter-115/126 (naturalness), iter-120 (barge-phase), iter-128
    (sentence-length), iter-140 (stt-rtf), iter-141 (tts-rtf),
    iter-142 (llm-tps), iter-143 (streaming-overlap). SIXTH instance
    applied to a CONTINUOUS metric — buckets it via
    ``_synth_dispatch_bucket`` before running the scan (same shape
    as iter-128/140/141/142/143).

    This completes the iter-076 TTFS attribution decomposition with
    a sentinel on every leg: STT (iter-140), LLM (iter-142), and
    now the synth+dispatch residual. The session-summary already
    prints the median ``synth %`` of TTFS, but a high median can
    hide behind one bad turn; a *sustained run* in the slow bucket
    is the actionable signal that the leg — not STT or the LLM — is
    the recurring opening-latency bottleneck.

    Filter rule: drop the ``"fast"`` bucket before the scan. It's
    the "fine" state; only "slow" and "very_slow" warrant warning.
    Empty buckets (turns with no audio / unmeasured residual) also
    drop. Like iter-140/141 and UNLIKE iter-142/143, the fine
    bucket is a LOW value (small latency) — synth+dispatch is
    smaller-is-better.

    Threshold = 5: same as iter-115/128/140/141/142/143. The leg
    varies turn to turn with sentence length and engine warmth, and
    a single slow opening (e.g. first turn cold-start) is normal; a
    sustained run is the real signal.

    Output:

        Synth/dispatch:   5 consecutive 'slow' turns — the synth +
                          speaker-open + dispatch leg is eating the
                          opening latency; try a lighter voice or
                          warm the speaker (iter-076 synth_dispatch_seconds)
    """
    # Bucketize, then drop the "uninteresting" bucket (empty, fast).
    interesting = {"slow", "very_slow"}
    filtered = [
        b for b in (
            _synth_dispatch_bucket(s) for s in synth_dispatch_list
        )
        if b in interesting
    ]
    if not filtered:
        return

    longest_run, longest_bucket = _longest_consecutive_run(filtered)
    if longest_run < threshold:
        return

    if longest_bucket == "very_slow":
        suggestion = (
            "synth+dispatch alone blows most of the sub-500ms "
            "budget; switch to a lighter voice/engine or pre-warm "
            "the speaker"
        )
    elif longest_bucket == "slow":
        suggestion = (
            "the synth + speaker-open + dispatch leg is eating the "
            "opening latency; try a lighter voice or warm the speaker"
        )
    else:
        # Defensive: future buckets that pass the filter rule.
        suggestion = "investigate synth time and speaker-open latency"

    emit(
        f"    Synth/dispatch:   {longest_run} consecutive "
        f"{longest_bucket!r} turns — {suggestion} "
        f"(iter-076 synth_dispatch_seconds)"
    )


def _eot_overhead_bucket(seconds: float) -> str:
    """iter-209: bucket a per-turn ``eot_overhead`` (iter-065's
    "trailing-silence wall" — the part of end-of-turn latency NOT
    explained by the configured ``silence_duration``:
    ``max(0, eot_latency - silence_duration_used)``) into a coarse
    category. Used by ``_emit_eot_overhead_consistency_line`` to
    detect runs where the recording loop itself — not the
    silence_duration knob — is the recurring cause of "the agent
    feels slow" (metric 1.2 in the perf-metrics taxonomy, which
    dominates that complaint).

    The decomposition matters because it tells the operator WHICH
    lever to pull. If overhead is ~0, the whole EoT wait is the
    ``silence_duration`` we asked for, and the fix is to tune
    ``chat.vad.silence_duration`` lower. If overhead is sustained
    and large, there's something else slow in the recording loop
    (chunk granularity, processing) and tuning the knob WON'T help.

    Buckets (chosen against iter-065's own ">100ms = something
    else is slow" guidance):

        ``fast``     — < 0.10s: overhead is negligible; the EoT
            wait is essentially the configured silence_duration, so
            reduce latency by tuning the knob. The desired state.
        ``slow``     — 0.10-0.25s: recording-loop overhead is a
            noticeable slice of the EoT wait; tuning silence_duration
            alone won't recover it.
        ``very_slow``— > 0.25s: the recording loop dominates the EoT
            wait; the user is left hanging well past the knob budget
            regardless of how low silence_duration is set.

    Like iter-140/141/208 and UNLIKE the inverted iter-142/143, the
    fine bucket is a LOW value — smaller overhead is better, so the
    boundaries are NOT inverted.

    Returns ``""`` when ``seconds`` is non-positive (turn with no
    measurable EoT overhead — DONE_TOO_SHORT / no-transcription
    path, or the wait was fully explained by silence_duration) —
    empty-string filter applies in the consumer, mirroring
    iter-114/115/120/126/128/140/141/208.
    """
    if seconds <= 0:
        return ""
    if seconds < 0.10:
        return "fast"
    if seconds <= 0.25:
        return "slow"
    return "very_slow"


def _emit_eot_overhead_consistency_line(
    emit, eot_overhead_list: list[float], threshold: int = 5,
) -> None:
    """iter-209: detect consecutive runs of turns where the
    end-of-turn latency was dominated by recording-loop *overhead*
    (iter-065 ``eot_overhead``) rather than the configured
    ``silence_duration``. TENTH instance of the diversity-check
    pattern after iter-114 (filler), iter-115/126 (naturalness),
    iter-120 (barge-phase), iter-128 (sentence-length), iter-140
    (stt-rtf), iter-141 (tts-rtf), iter-142 (llm-tps), iter-143
    (streaming-overlap), iter-208 (synth-dispatch). SEVENTH instance
    applied to a CONTINUOUS metric — buckets it via
    ``_eot_overhead_bucket`` before running the scan (same shape as
    iter-128/140/141/142/143/208).

    The session-summary already prints the *median* EoT overhead
    (iter-065), but a high median can hide behind one cold-start
    turn; a *sustained run* in the slow bucket is the actionable
    signal that the recording loop — not the silence_duration knob
    — is the recurring cause of "the agent feels slow" (metric 1.2,
    which dominates that complaint). It tells the operator NOT to
    chase the knob: tuning ``chat.vad.silence_duration`` lower won't
    recover overhead the VAD wait never owned.

    Filter rule: drop the ``"fast"`` bucket before the scan. It's
    the "fine" state (overhead negligible; the knob is the right
    lever); only "slow" and "very_slow" warrant warning. Empty
    buckets (turns with no measurable EoT overhead) also drop. Like
    iter-140/141/208 and UNLIKE iter-142/143, the fine bucket is a
    LOW value (small overhead) — overhead is smaller-is-better.

    Threshold = 5: same as iter-115/128/140/141/142/143/208. The
    overhead varies turn to turn with chunk alignment and
    processing jitter, and a single slow turn (e.g. first turn
    cold-start) is normal; a sustained run is the real signal.

    Output:

        EoT overhead:     5 consecutive 'slow' turns — recording-loop
                          overhead is eating the end-of-turn wait;
                          tuning silence_duration won't recover it,
                          profile the recording loop (iter-065 eot_overhead)
    """
    # Bucketize, then drop the "uninteresting" bucket (empty, fast).
    interesting = {"slow", "very_slow"}
    filtered = [
        b for b in (
            _eot_overhead_bucket(s) for s in eot_overhead_list
        )
        if b in interesting
    ]
    if not filtered:
        return

    longest_run, longest_bucket = _longest_consecutive_run(filtered)
    if longest_run < threshold:
        return

    if longest_bucket == "very_slow":
        suggestion = (
            "the recording loop dominates the end-of-turn wait; "
            "tuning silence_duration won't help — profile chunk "
            "granularity and recording-loop processing"
        )
    elif longest_bucket == "slow":
        suggestion = (
            "recording-loop overhead is eating the end-of-turn "
            "wait; tuning silence_duration won't recover it, "
            "profile the recording loop"
        )
    else:
        # Defensive: future buckets that pass the filter rule.
        suggestion = (
            "investigate recording-loop overhead beyond silence_duration"
        )

    emit(
        f"    EoT overhead:     {longest_run} consecutive "
        f"{longest_bucket!r} turns — {suggestion} "
        f"(iter-065 eot_overhead)"
    )


def _bot_wpm_bucket(wpm: float) -> str:
    """iter-210: bucket a per-turn ``bot_wpm`` (iter-046 bot speaking
    rate in words-per-minute, ``word_count_total / audio_seconds_total``)
    into a coarse category. Used by ``_emit_bot_wpm_consistency_line``
    to detect runs where the synthesized speech rate sat outside the
    human-comprehension sweet spot — too fast to follow or so slow it
    drags. Metric 1.13 in the perf-metrics taxonomy.

    ELEVENTH instance of the diversity-check pattern, and the EIGHTH
    applied to a continuous metric. It is the FIRST with a TWO-SIDED
    (band) sweet spot: every prior continuous bucketer was monotonic —
    either smaller-is-better (iter-140/141/208/209) or
    bigger-is-better (iter-142/143). Here the fine state is a BAND in
    the MIDDLE, and the two problematic buckets sit at OPPOSITE ends
    and call for OPPOSITE fixes (lower vs raise the kokoro ``speed``
    knob). That makes the per-value suggestion branch carry real
    signal rather than a severity gradient.

    Buckets (boundaries aligned with the existing per-turn display,
    which colors 130-200 WPM green — see ``TurnMetrics`` bot-WPM row;
    UX research puts the conversational sweet spot ~150-180 WPM):

        ``natural``  — 130-200 WPM inclusive: the comprehension sweet
            spot; the desired state.
        ``rushed``   — > 200 WPM: the bot talks faster than listeners
            comfortably parse — lower the kokoro ``speed`` parameter.
        ``sluggish`` — 0 < wpm < 130: the pace drags and the bot feels
            unresponsive — raise the kokoro ``speed`` parameter.

    Returns ``""`` when ``wpm`` is non-positive (no audio played this
    turn, or alignment produced no word count) — empty-string filter
    applies in the consumer, mirroring
    iter-114/115/120/126/128/140/141/142/143/208/209.
    """
    if wpm <= 0:
        return ""
    if wpm < 130.0:
        return "sluggish"
    if wpm <= 200.0:
        return "natural"
    return "rushed"


def _emit_bot_wpm_consistency_line(
    emit, bot_wpm_list: list[float], threshold: int = 5,
) -> None:
    """iter-210: detect consecutive runs of turns where the bot's
    synthesized speaking rate sat outside the comprehension sweet
    spot. ELEVENTH instance of the diversity-check pattern after
    iter-114 (filler), iter-115/126 (naturalness), iter-120
    (barge-phase), iter-128 (sentence-length), iter-140 (stt-rtf),
    iter-141 (tts-rtf), iter-142 (llm-tps), iter-143
    (streaming-overlap), iter-208 (synth-dispatch), iter-209
    (eot-overhead). EIGHTH instance applied to a CONTINUOUS metric —
    buckets it via ``_bot_wpm_bucket`` before running the scan.

    FIRST instance with a TWO-SIDED (band) sweet spot. Prior
    continuous sentinels filtered out one extreme as the "fine" state
    (iter-140/141/208/209 keep low; iter-142/143 keep high). Here the
    fine state is the MIDDLE band (``natural``), and BOTH extremes are
    flagged — but they are NOT interchangeable: ``rushed`` and
    ``sluggish`` need OPPOSITE corrections to the same kokoro
    ``speed`` knob. The per-value suggestion branch is therefore
    load-bearing (not a severity gradient like iter-208/209), and the
    run scan still treats the two flagged buckets as distinct phases —
    a rushed run and a sluggish run never merge.

    Filter rule: drop the ``"natural"`` bucket (the sweet spot) and
    ``""`` (no audio that turn) before the scan. Only ``rushed`` and
    ``sluggish`` warrant warning.

    Threshold = 5: same as iter-115/128/140/141/142/143/208/209. WPM
    varies turn to turn with sentence content and synth jitter, and a
    single fast or slow turn is normal; a sustained run is the
    actionable signal that the kokoro ``speed`` knob is mis-set.

    Output:

        Bot speech rate:  5 consecutive 'rushed' turns — the bot
                          out-paces comfortable listening; lower the
                          kokoro speed knob (iter-046 bot_wpm)
    """
    # Bucketize, then drop the "uninteresting" buckets (empty, natural).
    interesting = {"rushed", "sluggish"}
    filtered = [
        b for b in (
            _bot_wpm_bucket(w) for w in bot_wpm_list
        )
        if b in interesting
    ]
    if not filtered:
        return

    longest_run, longest_bucket = _longest_consecutive_run(filtered)
    if longest_run < threshold:
        return

    if longest_bucket == "rushed":
        suggestion = (
            "the bot out-paces comfortable listening; lower the "
            "kokoro speed knob"
        )
    elif longest_bucket == "sluggish":
        suggestion = (
            "the bot's pace drags and feels unresponsive; raise "
            "the kokoro speed knob"
        )
    else:
        # Defensive: future buckets that pass the filter rule.
        suggestion = "tune the kokoro speed knob toward 150-180 WPM"

    emit(
        f"    Bot speech rate:  {longest_run} consecutive "
        f"{longest_bucket!r} turns — {suggestion} "
        f"(iter-046 bot_wpm)"
    )


def _max_token_gap_bucket(seconds: float) -> str:
    """iter-211: bucket a per-turn ``max_token_gap`` (iter-085's
    maximum inter-token gap during the LLM stream, excluding the
    first-token wait) into a coarse category. Used by
    ``_emit_max_token_gap_consistency_line`` to detect runs where the
    LLM stalled mid-response — a failure mode that, as iter-085's field
    comment notes, is "currently invisible to operators because the
    user just hears a long pause and no signal fires." Metric 3.21 in
    the perf-metrics taxonomy.

    TWELFTH instance of the diversity-check pattern, and the NINTH
    applied to a continuous metric. Like iter-140/141/208/209 and
    UNLIKE the inverted iter-142/143, the fine bucket is a LOW value —
    a smaller worst-case gap is better, so the boundaries are NOT
    inverted. (Note the contrast with iter-142's ``llm_tps``: that
    bucketer is bigger-is-better because *average* throughput is good
    when high; this one watches the *worst* single stall, which is bad
    when high — different statistic, opposite direction.)

    Buckets (chosen against iter-085's own guidance —
    ">500ms = the LLM stalled noticeably mid-response"; ">2s = the
    user definitely thought the bot was broken"):

        ``fast``     — < 0.50s: no perceptible mid-stream stall; the
            LLM kept feeding the splitter. The desired state.
        ``slow``     — 0.50-2.0s: a noticeable mid-stream stall; the
            user heard a gap and may have wondered if the bot was
            still working.
        ``very_slow``— > 2.0s: a severe stall; the user likely thought
            the bot was broken.

    Returns ``""`` when ``seconds`` is non-positive (single-token turn,
    or no measurable inter-token gap) — empty-string filter applies in
    the consumer, mirroring
    iter-114/115/120/126/128/140/141/208/209/210.
    """
    if seconds <= 0:
        return ""
    if seconds < 0.50:
        return "fast"
    if seconds <= 2.0:
        return "slow"
    return "very_slow"


def _emit_max_token_gap_consistency_line(
    emit, max_token_gap_list: list[float], threshold: int = 5,
) -> None:
    """iter-211: detect consecutive runs of turns where the LLM
    stalled mid-stream (iter-085 ``max_token_gap`` — the worst
    inter-token gap that turn, excluding the first-token wait).
    TWELFTH instance of the diversity-check pattern after iter-114
    (filler), iter-115/126 (naturalness), iter-120 (barge-phase),
    iter-128 (sentence-length), iter-140 (stt-rtf), iter-141
    (tts-rtf), iter-142 (llm-tps), iter-143 (streaming-overlap),
    iter-208 (synth-dispatch), iter-209 (eot-overhead), iter-210
    (bot-wpm). NINTH instance applied to a CONTINUOUS metric — buckets
    it via ``_max_token_gap_bucket`` before running the scan (same
    shape as iter-128/140/141/142/143/208/209).

    iter-085 added the per-turn metric precisely because mid-stream
    stalls are otherwise *silent*: the user hears a long pause and no
    signal fires. A single stalled turn is normal jitter (GC pause,
    a heavy token, a momentary load spike); a *sustained run* of
    stalled turns is the actionable signal that the LLM backend is
    consistently struggling to keep the stream fed — distinct from
    iter-142's ``llm_tps`` sentinel, which watches *average*
    throughput. A model can post a healthy mean TPS yet still stall
    badly once per turn; this sentinel catches that worst-case shape
    the average smooths over.

    Filter rule: drop the ``"fast"`` bucket before the scan. It's the
    "fine" state (no perceptible stall); only "slow" and "very_slow"
    warrant warning. Empty buckets (single-token turns, no measurable
    gap) also drop. Like iter-140/141/208/209 and UNLIKE iter-142/143,
    the fine bucket is a LOW value (small gap) — the worst-case gap is
    smaller-is-better.

    Threshold = 5: same as iter-115/128/140/141/142/143/208/209/210.
    Inter-token timing is jittery turn to turn, and a single stalled
    turn is normal; a sustained run is the real signal.

    Output:

        Max token gap:    5 consecutive 'slow' turns — the LLM keeps
                          stalling mid-stream; the user hears repeated
                          pauses — check the LLM backend
                          (iter-085 max_token_gap)
    """
    # Bucketize, then drop the "uninteresting" bucket (empty, fast).
    interesting = {"slow", "very_slow"}
    filtered = [
        b for b in (
            _max_token_gap_bucket(s) for s in max_token_gap_list
        )
        if b in interesting
    ]
    if not filtered:
        return

    longest_run, longest_bucket = _longest_consecutive_run(filtered)
    if longest_run < threshold:
        return

    if longest_bucket == "very_slow":
        suggestion = (
            "the LLM stalls severely mid-stream (>2s gaps); the user "
            "likely thinks the bot is broken — check the LLM backend "
            "load, model size, and host contention"
        )
    elif longest_bucket == "slow":
        suggestion = (
            "the LLM keeps stalling mid-stream; the user hears "
            "repeated pauses — check the LLM backend (a healthy mean "
            "TPS can still hide these worst-case stalls)"
        )
    else:
        # Defensive: future buckets that pass the filter rule.
        suggestion = (
            "investigate recurring mid-stream LLM stalls"
        )

    emit(
        f"    Max token gap:    {longest_run} consecutive "
        f"{longest_bucket!r} turns — {suggestion} "
        f"(iter-085 max_token_gap)"
    )


def _ttfs_bucket(seconds: float) -> str:
    """iter-212: bucket a per-turn ``ttfs`` (time-to-first-speech =
    speech-stop → first-audio-to-speaker, the headline latency metric
    behind the VISION's "sub-500ms / latency is the feature" goal) into
    a coarse category. Used by ``_emit_ttfs_consistency_line`` to detect
    runs where the bot was consistently slow to start talking back.
    Metric 2.x in the perf-metrics taxonomy.

    THIRTEENTH instance of the diversity-check pattern, and the TENTH
    applied to a continuous metric. Like
    iter-140/141/208/209/211 and UNLIKE the inverted iter-142/143, the
    fine bucket is a LOW value — a smaller TTFS is better, so the
    boundaries are NOT inverted.

    Buckets (the ``snappy`` boundary is aligned with the existing
    per-turn green display, which colors TTFS green below 3.0s — see
    ``TurnMetrics.print``'s ``ttfs_color``):

        ``snappy``   — < 3.0s: the bot starts talking back promptly;
            matches the per-turn green display. The desired state.
        ``slow``     — 3.0-6.0s: the bot takes a noticeable beat before
            speaking; the conversation feels laggy.
        ``very_slow``— > 6.0s: the bot takes so long the user wonders
            whether it heard them at all.

    Returns ``""`` when ``seconds`` is non-positive (no audio played
    that turn — error turn or pure barge) — empty-string filter applies
    in the consumer, mirroring
    iter-114/115/120/126/128/140/141/208/209/210/211.
    """
    if seconds <= 0:
        return ""
    if seconds < 3.0:
        return "snappy"
    if seconds <= 6.0:
        return "slow"
    return "very_slow"


def _emit_ttfs_consistency_line(
    emit, ttfs_list: list[float], threshold: int = 5,
) -> None:
    """iter-212: detect consecutive runs of turns where the bot was
    slow to start speaking (``ttfs`` — time-to-first-speech, the
    headline latency metric). THIRTEENTH instance of the
    diversity-check pattern after iter-114 (filler), iter-115/126
    (naturalness), iter-120 (barge-phase), iter-128 (sentence-length),
    iter-140 (stt-rtf), iter-141 (tts-rtf), iter-142 (llm-tps), iter-143
    (streaming-overlap), iter-208 (synth-dispatch), iter-209
    (eot-overhead), iter-210 (bot-wpm), iter-211 (max-token-gap). TENTH
    instance applied to a CONTINUOUS metric — buckets it via
    ``_ttfs_bucket`` before running the scan (same shape as
    iter-128/140/141/142/143/208/209/211).

    TTFS is *the* number the VISION optimizes for ("latency is the
    feature"; "sub-500ms on a MacBook"). The session summary already
    reports the TTFS median and a first-turn-warmup penalty
    (``_emit_ttfs_block``), but a healthy median can hide a sustained
    stretch of slow turns: a few snappy turns early pull the median
    down while the back half of the session crawls. This sentinel
    catches that run shape the median smooths over — distinct from the
    iter-211 max-token-gap sentinel (a mid-stream stall AFTER the bot
    started) and the iter-209 eot-overhead sentinel (end-of-turn
    detection wait BEFORE TTFS starts): this watches the whole
    speech-stop → speaker latency end to end.

    Filter rule: drop the ``"snappy"`` bucket before the scan. It's the
    "fine" state (prompt response, matches the green display); only
    "slow" and "very_slow" warrant warning. Empty buckets (no audio
    that turn — error/barge) also drop. Like
    iter-140/141/208/209/211 and UNLIKE iter-142/143, the fine bucket
    is a LOW value (small TTFS) — TTFS is smaller-is-better.

    Threshold = 5: same as
    iter-115/128/140/141/142/143/208/209/210/211. TTFS jitters turn to
    turn (model warmup, GC, context growth); a single slow turn is
    normal, a sustained run is the real signal.

    Output:

        TTFS:             6 consecutive 'slow' turns — the bot is
                          consistently slow to start speaking; the
                          conversation feels laggy — profile the STT /
                          LLM-first-sentence / synth-dispatch split
                          (iter-212 ttfs)
    """
    # Bucketize, then drop the "uninteresting" bucket (empty, snappy).
    interesting = {"slow", "very_slow"}
    filtered = [
        b for b in (_ttfs_bucket(s) for s in ttfs_list)
        if b in interesting
    ]
    if not filtered:
        return

    longest_run, longest_bucket = _longest_consecutive_run(filtered)
    if longest_run < threshold:
        return

    if longest_bucket == "very_slow":
        suggestion = (
            "the bot takes so long to start speaking (>6s) the user "
            "wonders whether it heard them — profile the STT / "
            "LLM-first-sentence / synth-dispatch split and check host "
            "contention"
        )
    elif longest_bucket == "slow":
        suggestion = (
            "the bot is consistently slow to start speaking; the "
            "conversation feels laggy — profile the STT / "
            "LLM-first-sentence / synth-dispatch split"
        )
    else:
        # Defensive: future buckets that pass the filter rule.
        suggestion = (
            "investigate recurring slow time-to-first-speech"
        )

    emit(
        f"    TTFS:             {longest_run} consecutive "
        f"{longest_bucket!r} turns — {suggestion} "
        f"(iter-212 ttfs)"
    )


def _stt_preview_divergence_bucket(divergence: float) -> str:
    """iter-224: bucket a per-turn ``stt_preview_divergence``
    (iter-072's live-preview-vs-final transcript distance in
    ``[0.0, 1.0]``, computed via ``difflib.SequenceMatcher`` in
    ``record_utterance_streaming``) into a coarse category. Used by
    ``_emit_stt_preview_divergence_consistency_line`` to detect runs
    where the live STT preview consistently diverged from the final
    transcript — the signal that incremental Whisper output is
    unreliable, so the live preview is misleading the user rather
    than helping them.

    Buckets (chosen against the iter-072 semantics, where 0 = the
    preview matched the final perfectly and ``>0.3`` is documented
    as "preview UX is broken"):

        ``good``   — < 0.15: the live preview tracked the final
            closely; the desired state. Incremental STT was useful.
        ``noisy``  — 0.15-0.30: the preview drifted from the final
            often enough to be mildly distracting, but the user
            could still mostly read along.
        ``broken`` — > 0.30: the preview was so different from the
            final that the user had to wait for the final transcript
            to know if they were understood — the live preview added
            no value (or actively misled).

    Like iter-140/141/208 and UNLIKE iter-142/143, the fine bucket
    is a LOW value — smaller divergence is better, so the boundaries
    are NOT inverted.

    Returns ``""`` when ``divergence`` is non-positive (no streaming
    STT this turn, or a perfect preview match — both the "fine"
    state) — empty-string filter applies in the consumer, mirroring
    iter-114/115/120/126/128/140/141/208.
    """
    if divergence <= 0:
        return ""
    if divergence < 0.15:
        return "good"
    if divergence <= 0.30:
        return "noisy"
    return "broken"


def _emit_stt_preview_divergence_consistency_line(
    emit, divergence_list: list[float], threshold: int = 5,
) -> None:
    """iter-224: detect consecutive runs of turns where the live STT
    preview diverged from the final transcript. Latest instance of
    the diversity-check pattern after iter-114 (filler),
    iter-115/126 (naturalness), iter-120 (barge-phase), iter-128
    (sentence-length), iter-140 (stt-rtf), iter-141 (tts-rtf),
    iter-142 (llm-tps), iter-143 (streaming-overlap), iter-208
    (synth-dispatch), iter-209 (eot-overhead), iter-210 (bot-wpm),
    iter-211 (max-token-gap), iter-212 (ttfs). Buckets the per-turn
    ``stt_preview_divergence`` via ``_stt_preview_divergence_bucket``
    before running the scan (same continuous-metric shape as
    iter-128/140/141/142/143/208).

    The iter-072 metric exists precisely because a bad preview is
    otherwise silent: the session-summary never surfaced it, so a
    persistently-wrong live transcript ("preview UX is broken,"
    per iter-072) goes unnoticed even though it directly degrades
    the hands-free read-along experience the VISION targets. A
    *sustained run* in the noisy/broken bucket is the actionable
    signal that incremental Whisper output is unreliable on this
    host — distinct from iter-140's ``stt_rtf`` sentinel, which
    watches STT *speed*: STT can be fast (good RTF) yet produce a
    preview that keeps changing under the user (bad divergence).

    Filter rule: drop the ``"good"`` bucket before the scan. It's
    the "fine" state; only "noisy" and "broken" warrant warning.
    Empty buckets (turns with no streaming STT / a perfect match)
    also drop. Like iter-140/141/208, the fine bucket is a LOW value
    (small divergence) — preview divergence is smaller-is-better.

    Threshold = 5: same as iter-115/128/140/141/142/143/208. The
    divergence varies turn to turn with utterance length and how
    incrementally Whisper settles, and a single noisy preview (e.g.
    a long rambling utterance) is normal; a sustained run is the
    real signal.

    Output:

        STT preview:      5 consecutive 'broken' turns — the live
                          preview diverges badly from the final;
                          users can't trust the read-along, check
                          incremental Whisper or hide the preview
                          (iter-072 stt_preview_divergence)
    """
    # Bucketize, then drop the "uninteresting" bucket (empty, good).
    interesting = {"noisy", "broken"}
    filtered = [
        b for b in (
            _stt_preview_divergence_bucket(d) for d in divergence_list
        )
        if b in interesting
    ]
    if not filtered:
        return

    longest_run, longest_bucket = _longest_consecutive_run(filtered)
    if longest_run < threshold:
        return

    if longest_bucket == "broken":
        suggestion = (
            "the live preview diverges badly from the final; users "
            "can't trust the read-along, check incremental Whisper "
            "or hide the preview"
        )
    elif longest_bucket == "noisy":
        suggestion = (
            "the live preview drifts from the final often; "
            "incremental Whisper output is unreliable on this host, "
            "consider a steadier preview window"
        )
    else:
        # Defensive: future buckets that pass the filter rule.
        suggestion = "investigate recurring live-preview divergence"

    emit(
        f"    STT preview:      {longest_run} consecutive "
        f"{longest_bucket!r} turns — {suggestion} "
        f"(iter-072 stt_preview_divergence)"
    )


def _split_coverage_bucket(coverage: float) -> str:
    """iter-225: bucket a per-turn ``sentence_split_coverage``
    (iter-059's fraction of submitted chars that landed inside a
    complete sentence — chars that overlapped synth with the LLM
    stream — versus the trailing remainder flushed at end-of-stream)
    into a coarse category. Used by
    ``_emit_split_coverage_consistency_line`` to detect runs of turns
    where the sentence splitter kept flushing a large remainder, so
    the iter-008 streaming-overlap design can't mask synth latency.

    A continuous-metric instance like iter-128/140/141/142/143/208.
    Like iter-142 ``llm_tps`` and iter-143 ``streaming_overlap_ratio``
    — and UNLIKE the iter-140/141 RTF bucketers —
    ``sentence_split_coverage`` is "bigger is better": the fine state
    is a HIGH value (close to 1.0 = every char overlapped), so the
    boundaries invert and the problematic end is a small coverage.
    This is the THIRD inverted-direction continuous bucketer.

    Buckets (chosen against iter-059's semantics, where 1.0 = the LLM
    always ended on punctuation so every char overlapped the next
    sentence, and <1.0 = some chars were forced through as the
    trailing remainder which synthesizes only AFTER the stream is
    done, defeating overlap):

        ``full``    — >= 0.90: the splitter caught essentially all
            chars inside complete sentences; the remainder flush was
            negligible. The desired state.
        ``partial`` — 0.70-0.90: a noticeable remainder flushed each
            turn — overlap is leaking, but most chars still overlap.
        ``poor``    — < 0.70 (but > 0): a large remainder flushes at
            end-of-stream, so a big share of synth runs sequentially
            after the LLM finishes. The iter-008 overlap design is
            being defeated — check the sentence splitter / punctuation
            handling.

    Returns ``""`` when ``coverage`` is non-positive (no chars
    submitted this turn — the "no measurement" state) — empty-string
    filter applies in the consumer, mirroring
    iter-114/115/120/126/128/140/141/142/143/208/224.
    """
    if coverage <= 0:
        return ""
    if coverage >= 0.90:
        return "full"
    if coverage >= 0.70:
        return "partial"
    return "poor"


def _emit_split_coverage_consistency_line(
    emit, coverage_list: list[float], threshold: int = 5,
) -> None:
    """iter-225: detect consecutive runs of turns where the sentence
    splitter flushed a large trailing remainder, so the iter-008
    streaming-overlap design couldn't mask synth latency. Latest
    instance of the diversity-check pattern after iter-114 (filler),
    iter-115/126 (naturalness), iter-120 (barge-phase), iter-128
    (sentence-length), iter-140 (stt-rtf), iter-141 (tts-rtf),
    iter-142 (llm-tps), iter-143 (streaming-overlap), iter-208
    (synth-dispatch), iter-209 (eot-overhead), iter-210 (bot-wpm),
    iter-211 (max-token-gap), iter-212 (ttfs), iter-224
    (stt-preview-divergence). Buckets the per-turn
    ``sentence_split_coverage`` via ``_split_coverage_bucket`` before
    running the scan (same continuous-metric shape as
    iter-128/140/141/142/143/208/224).

    The iter-059 metric exists because remainder flushing is
    otherwise silent at the session level: each turn's per-turn line
    shows the "% complete" suffix only when <1.0, but a *sustained*
    low-coverage run — the splitter systematically failing to close
    sentences, e.g. an LLM that rarely emits terminal punctuation —
    never surfaces in the summary. A sustained run is the actionable
    signal that overlap is being defeated turn after turn, directly
    eroding the TTFS the VISION optimizes for. Distinct from iter-143's
    ``streaming_overlap_ratio`` sentinel: that measures how much synth
    DID overlap; this measures whether the splitter even GAVE synth
    full sentences to overlap with. Low coverage is an upstream cause
    of low overlap.

    Filter rule: drop the ``"full"`` bucket before the scan. It's the
    "fine" state; only "partial" and "poor" warrant warning. Empty
    buckets (turns with no submitted chars) also drop. Like iter-142
    and iter-143 and UNLIKE iter-140/141, the fine bucket is a HIGH
    value ("full") because coverage is bigger-is-better — the THIRD
    inverted-direction instance. The filter rule absorbs the
    inversion; the run-scan stays direction-agnostic.

    Threshold = 5: same as iter-115/128/140/141/142/143/208/224.
    Coverage varies turn to turn with how the LLM happens to end a
    response (a one-word answer with no period legitimately flushes a
    remainder); a single low-coverage turn is normal, a sustained run
    is the real signal.

    Output:

        Split coverage:   5 consecutive 'poor' turns — a large
                          remainder flushes after the LLM stream so
                          synth runs sequentially; check the sentence
                          splitter / terminal punctuation
                          (iter-059 sentence_split_coverage)
    """
    # Bucketize, then drop the "uninteresting" bucket (empty, full).
    interesting = {"partial", "poor"}
    filtered = [
        b for b in (
            _split_coverage_bucket(c) for c in coverage_list
        )
        if b in interesting
    ]
    if not filtered:
        return

    longest_run, longest_bucket = _longest_consecutive_run(filtered)
    if longest_run < threshold:
        return

    if longest_bucket == "poor":
        suggestion = (
            "a large remainder flushes after the LLM stream so synth "
            "runs sequentially; check the sentence splitter / terminal "
            "punctuation"
        )
    elif longest_bucket == "partial":
        suggestion = (
            "a noticeable remainder flushes each turn; overlap is "
            "leaking, the LLM may rarely end on punctuation"
        )
    else:
        # Defensive: future buckets that pass the filter rule.
        suggestion = "investigate recurring sentence-split remainder flushing"

    emit(
        f"    Split coverage:   {longest_run} consecutive "
        f"{longest_bucket!r} turns — {suggestion} "
        f"(iter-059 sentence_split_coverage)"
    )


def _worker_idle_gap_bucket(idle_gap: float) -> str:
    """iter-226: bucket a per-turn ``worker_idle_gap_total``
    (iter-044's cumulative seconds the SentenceWorker spent blocked
    waiting for the next sentence AFTER the first one — i.e. the LLM
    didn't keep up with synth+playback) into a coarse category. Used by
    ``_emit_worker_idle_gap_consistency_line`` to detect runs of turns
    where the worker repeatedly starved between sentences, so the
    iter-008 streaming-overlap design produced gaps of silence the
    user can hear.

    A continuous-metric instance like
    iter-128/140/141/142/143/208/224/225. Like iter-140/141 (RTF) and
    iter-208 (synth-dispatch) and iter-224 (preview-divergence) — and
    UNLIKE iter-142/143/225 — ``worker_idle_gap_total`` is
    "smaller is better": the fine state is a LOW value (0 = the worker
    never waited), so the boundaries are NOT inverted; the problematic
    end is a LARGE idle gap.

    Buckets (chosen against iter-044's per-turn yellow-flag line, which
    colors the inline figure yellow above 0.3s — that's the operator-
    visible "the worker is starving" threshold):

        ``smooth``   — < 0.3s: cumulative idle gap is below the
            iter-044 yellow line; the LLM kept up with synth. The
            desired state.
        ``stalled``  — 0.3-1.0s: a noticeable cumulative gap — the
            user hears short silences between sentences; the LLM is
            lagging the synth+playback pipeline.
        ``starved``  — > 1.0s: a large cumulative gap — synth sat idle
            for over a second waiting on the LLM; streaming overlap is
            being defeated and the turn feels stop-start.

    Returns ``""`` when ``idle_gap`` is non-positive (the worker never
    blocked after the first sentence — a single-sentence turn, or a
    turn where the LLM always stayed ahead; both the "fine"/"no
    measurement" state) — empty-string filter applies in the consumer,
    mirroring iter-114/115/120/126/128/140/141/142/143/208/224/225.
    """
    if idle_gap <= 0:
        return ""
    if idle_gap < 0.3:
        return "smooth"
    if idle_gap <= 1.0:
        return "stalled"
    return "starved"


def _emit_worker_idle_gap_consistency_line(
    emit, idle_gap_list: list[float], threshold: int = 5,
) -> None:
    """iter-226: detect consecutive runs of turns where the
    SentenceWorker starved between sentences, so the iter-008
    streaming-overlap design produced audible gaps of silence. Latest
    instance of the diversity-check pattern after iter-114 (filler),
    iter-115/126 (naturalness), iter-120 (barge-phase), iter-128
    (sentence-length), iter-140 (stt-rtf), iter-141 (tts-rtf),
    iter-142 (llm-tps), iter-143 (streaming-overlap), iter-208
    (synth-dispatch), iter-209 (eot-overhead), iter-210 (bot-wpm),
    iter-211 (max-token-gap), iter-212 (ttfs), iter-224
    (stt-preview-divergence), iter-225 (sentence-split-coverage).
    Buckets the per-turn ``worker_idle_gap_total`` via
    ``_worker_idle_gap_bucket`` before running the scan (same
    continuous-metric shape as iter-128/140/141/142/143/208/224/225).

    The iter-044 metric exists to localize pipeline bottlenecks
    (combined with ``streaming_overlap_ratio``), but it's otherwise
    silent at the session level: each turn colors the inline figure
    yellow above 0.3s, but a *sustained* high-idle-gap run — the LLM
    systematically failing to keep up with synth+playback, e.g. a
    slow model, a long system prompt, or context creep — never
    surfaces in the summary. A sustained run is the actionable signal
    that the user is hearing stop-start audio turn after turn, eroding
    the conversational feel the VISION targets. Distinct from
    iter-143's ``streaming_overlap_ratio`` sentinel: that measures the
    FRACTION of synth that overlapped the stream; this measures the
    ABSOLUTE seconds the worker sat idle. A turn can show decent
    overlap ratio yet still leave the worker blocked for a long gap on
    one slow sentence — the inverse of iter-062's ``max_queue_depth``
    (worker backed up vs worker starved).

    Filter rule: drop the ``"smooth"`` bucket before the scan. It's
    the "fine" state; only "stalled" and "starved" warrant warning.
    Empty buckets (single-sentence turns / turns the worker never
    blocked) also drop. Like iter-140/141/208/224 and UNLIKE
    iter-142/143/225, the fine bucket is a LOW value ("smooth")
    because idle gap is smaller-is-better — the boundaries are NOT
    inverted.

    Threshold = 5: same as iter-115/128/140/141/142/143/208/224/225.
    Idle gap varies turn to turn with how long the LLM happens to
    pause mid-response; a single high-idle turn is normal, a sustained
    run is the real signal.

    Output:

        Worker idle gap:  5 consecutive 'starved' turns — synth sat
                          idle over a second waiting on the LLM;
                          streaming overlap is defeated, check LLM
                          throughput / context size
                          (iter-044 worker_idle_gap_total)
    """
    # Bucketize, then drop the "uninteresting" bucket (empty, smooth).
    interesting = {"stalled", "starved"}
    filtered = [
        b for b in (
            _worker_idle_gap_bucket(g) for g in idle_gap_list
        )
        if b in interesting
    ]
    if not filtered:
        return

    longest_run, longest_bucket = _longest_consecutive_run(filtered)
    if longest_run < threshold:
        return

    if longest_bucket == "starved":
        suggestion = (
            "synth sat idle over a second waiting on the LLM; "
            "streaming overlap is defeated, check LLM throughput / "
            "context size"
        )
    elif longest_bucket == "stalled":
        suggestion = (
            "the user hears short silences between sentences; the LLM "
            "is lagging synth+playback"
        )
    else:
        # Defensive: future buckets that pass the filter rule.
        suggestion = "investigate recurring SentenceWorker idle gaps"

    emit(
        f"    Worker idle gap:  {longest_run} consecutive "
        f"{longest_bucket!r} turns — {suggestion} "
        f"(iter-044 worker_idle_gap_total)"
    )


def _emit_bargeable_line(emit, bargeable_values: list[float]) -> None:
    """iter-104: extracted from print_session_summary's iter-074
    bargeable line. Reports the WORST bargeable fraction across
    turns when any turn dipped below 99% — that's the threshold
    where watcher-coverage regressions become operator-visible.

    Suppressed when bargeable_values is empty OR every turn was
    ≥99% bargeable (clean sessions don't need the line).
    Behavior-preserving: the "worst%" + "below_count/total turns
    < 99%" pattern is unchanged from the inline version.
    """
    if bargeable_values and min(bargeable_values) < 0.99:
        worst = min(bargeable_values) * 100
        below_count = sum(1 for v in bargeable_values if v < 0.99)
        emit(
            f"    Bargeable:        "
            f"{worst:.0f}% worst "
            f"({below_count}/{len(bargeable_values)} turns < 99%) — "
            f"watcher coverage regression"
        )


@dataclass
class SessionMeta:
    """iter-086: session-level signals collected by the driver
    (mic_chat) and passed into ``print_session_summary`` as a
    single object, rather than as a growing list of kwargs.

    Each field tracks something the per-turn ``TurnMetrics`` can't
    express because it spans turns or non-turn events:

      ``false_triggers`` (iter-048): turns where the recorder fired
        but no transcript came back — VAD noise.
      ``session_seconds`` (iter-054): wall-clock from session start
        to summary call. 0.0 when not provided.
      ``llm_errors`` (iter-058): turn-fatal LLM exceptions.
      ``trim_events`` (iter-078): how many times trim_history
        actually evicted ≥1 message.
      ``trim_messages_evicted`` (iter-078): cumulative evicted
        message count across all trim_events.

    All fields default to 0 — a `SessionMeta()` with no args is a
    "no extra context" object. Callers that don't care about any
    of these signals can continue to pass nothing (legacy kwargs
    on print_session_summary are still accepted, see that
    function's docstring).
    """

    false_triggers: int = 0
    session_seconds: float = 0.0
    llm_errors: int = 0
    trim_events: int = 0
    trim_messages_evicted: int = 0
    # iter-096: filler idle_threshold the chat loop ran with
    # (passed by mic_chat). When non-zero AND filler false
    # positives fired, print_session_summary computes a
    # recommended new value from the observed llm_first_token
    # distribution and surfaces it on the FP-rate line.
    idle_threshold: float = 0.0
    # iter-160: a mid-thought utterance the organic UtteranceAggregator
    # was still holding when the session ended (the user trailed off and
    # never landed a continuation, then hit Ctrl+C). ``None`` for the
    # overwhelmingly common case — no aggregator, nothing held, or
    # half-duplex (which never holds). When set, the summary surfaces it
    # so the dropped final fragment is visible rather than silently lost.
    stranded_utterance: Optional[str] = None
    # iter-161: count of turns where the organic aggregator HELD a
    # mid-thought utterance for a merge (buffered, not responded to).
    # ``run_session`` tracks these separately from VAD false triggers;
    # surfaced in the organic-turn-taking summary block. 0 on the
    # half-duplex / no-aggregator path (nothing is ever held).
    utterances_held: int = 0
    # iter-162: mid-thought fragments the organic aggregator released
    # alongside a responded turn (the user trailed off, a long silence
    # proved the fragment was NOT a false endpoint, then a genuinely new
    # thought displaced it). Each is captured-but-abandoned text — the
    # mid-session analog of ``stranded_utterance`` — surfaced rather than
    # silently glued onto the response. Empty for the common case.
    utterances_displaced: list[str] = field(default_factory=list)
    # iter-167: mid-thought fragments the organic aggregator flushed
    # mid-session on a long inter-turn idle silence (the recorder's iter-165
    # idle timeout fired, and iter-164's ``decide_silence_flush`` chose FLUSH).
    # The mid-session-idle analog of ``stranded_utterance`` (shutdown) and
    # ``utterances_displaced`` (new-thought displacement). Empty for the common
    # case — no idle timeout wired, no flush decider, or half-duplex.
    flushed_utterances: list[str] = field(default_factory=list)
    # iter-175: lifetime count of mid-speech agent backchannels the organic
    # BackchannelMonitor / BackchannelDriver (#7, iter-170+) emitted during
    # the session — the driver's ``emit_count``. The agent-side mirror of
    # ``utterances_held`` / the user-continuer count: surfaced in the
    # organic-turn-taking summary block so #7's emission is measured, not
    # asserted. 0 on the half-duplex / no-driver path (the driver never
    # emits). SessionMeta-only — no legacy kwarg path.
    backchannels_emitted: int = 0
    # iter-215: WPM-mirror adaptation surface (closes iter-214 backlog #1). The
    # iter-213/214 WPM mirror now *adapts* the Kokoro ``speed`` toward the
    # user's measured rate turn-to-turn; these three fields let the summary
    # *measure* that adaptation rather than asserting it.
    #   ``wpm_mirror_active``: whether a live WpmMirror was wired in (the
    #     operator turned ``chat.wpm_mirror.enabled`` on). False on the
    #     off-by-default path — the line is then suppressed entirely, so the
    #     summary is byte-for-byte the pre-iter-215 output.
    #   ``wpm_mirror_initial_speed``: the Kokoro speed the session started with
    #     (``SpeedController.initial``).
    #   ``wpm_mirror_final_speed``: the speed it ended with
    #     (``SpeedController.current()``) — the net of every per-turn nudge.
    # Pair them with the existing iter-064 "Mirror gap" WPM line: when mirroring
    # is on, the gap should shrink as the speed drifts toward the user's rate
    # (the iter-210 bot_wpm sentinel sees the same convergence). SessionMeta-only
    # — no legacy kwarg path.
    wpm_mirror_active: bool = False
    wpm_mirror_initial_speed: float = 0.0
    wpm_mirror_final_speed: float = 0.0


def print_session_summary(
    metrics_list: list[TurnMetrics],
    llm_config: dict,
    *,
    file=None,
    meta: Optional["SessionMeta"] = None,
    false_triggers: int = 0,
    session_seconds: float = 0.0,
    llm_errors: int = 0,
    trim_events: int = 0,
    trim_messages_evicted: int = 0,
) -> None:
    """Print a multi-line session summary on KeyboardInterrupt.

    Was inlined inside ``mic_chat.run_chat``'s KeyboardInterrupt
    handler with two issues iter-017 fixes:
      - ``sorted[len//2]`` reports the upper median for even-length
        lists, biasing 2-turn (and other small) sessions.
      - It was untestable without instantiating mic_chat.

    `file` defaults to ``sys.stdout`` (via ``print``); tests pass
    a ``StringIO`` to inspect the output.

    iter-048: ``false_triggers`` counts turns where
    ``ChatLoop.run_one_turn`` returned ``metrics=None`` WITHOUT
    ``had_error`` — i.e. VAD fired ACTIVE but the utterance was
    too short or transcription came back empty. Caller (mic_chat)
    tracks these and passes the total. Defaults to 0 for back-
    compat with callers that don't track yet. Metric 1.4 in the
    perf-metrics taxonomy.

    iter-078: ``trim_events`` counts how many times across the
    session ``trim_history`` actually evicted at least one
    message; ``trim_messages_evicted`` is the cumulative count
    of evicted messages. Validates the trim threshold is
    calibrated: if events == 0 across a long session, the cap
    is too loose (context-token growth from iter-077 will be
    showing the same story). If trim_messages_evicted/events is
    consistently 1, the cap is exactly right (each turn trims
    one). Metric 2.24 in the perf-metrics taxonomy.

    iter-086: ``meta`` (a ``SessionMeta``) is the preferred way to
    pass session-level signals — future additions extend the
    dataclass instead of growing this kwarg list. When ``meta`` is
    provided it takes precedence; the legacy kwargs are still
    accepted for backwards compatibility and merge in for ANY
    field not covered by ``meta``. Mixed-mode (some via meta, some
    via kwargs) is rare but works.
    """
    # iter-086: consolidate the session-level signals into a single
    # SessionMeta. ``meta`` wins for any field it provides; legacy
    # kwargs fill in the rest. This keeps the function body uniform
    # — every reference is ``meta_eff.field`` regardless of where
    # the value came from.
    if meta is not None:
        meta_eff = SessionMeta(
            false_triggers=meta.false_triggers or false_triggers,
            session_seconds=meta.session_seconds or session_seconds,
            llm_errors=meta.llm_errors or llm_errors,
            trim_events=meta.trim_events or trim_events,
            trim_messages_evicted=(
                meta.trim_messages_evicted or trim_messages_evicted
            ),
            # iter-096: idle_threshold has no legacy kwarg path —
            # only flows through SessionMeta.
            idle_threshold=meta.idle_threshold,
            # iter-160: stranded fragment — SessionMeta-only, no legacy
            # kwarg path.
            stranded_utterance=meta.stranded_utterance,
            # iter-161: held-utterance count — SessionMeta-only.
            utterances_held=meta.utterances_held,
            # iter-162: displaced mid-thought fragments — SessionMeta-only.
            utterances_displaced=meta.utterances_displaced,
            # iter-167: mid-session idle-flushed fragments — SessionMeta-only.
            flushed_utterances=meta.flushed_utterances,
            # iter-175: agent backchannel emit count — SessionMeta-only.
            backchannels_emitted=meta.backchannels_emitted,
            # iter-215: WPM-mirror adaptation surface — SessionMeta-only.
            wpm_mirror_active=meta.wpm_mirror_active,
            wpm_mirror_initial_speed=meta.wpm_mirror_initial_speed,
            wpm_mirror_final_speed=meta.wpm_mirror_final_speed,
        )
    else:
        meta_eff = SessionMeta(
            false_triggers=false_triggers,
            session_seconds=session_seconds,
            llm_errors=llm_errors,
            trim_events=trim_events,
            trim_messages_evicted=trim_messages_evicted,
        )
    # Local rebinds so the rest of the function reads naturally.
    false_triggers = meta_eff.false_triggers
    session_seconds = meta_eff.session_seconds
    llm_errors = meta_eff.llm_errors
    trim_events = meta_eff.trim_events
    trim_messages_evicted = meta_eff.trim_messages_evicted
    # iter-160: stranded mid-thought fragment held at shutdown.
    stranded_utterance = meta_eff.stranded_utterance
    # iter-161: count of mid-thought utterances buffered for a merge.
    utterances_held = meta_eff.utterances_held
    # iter-162: mid-thought fragments displaced by a genuinely-new thought.
    utterances_displaced = meta_eff.utterances_displaced
    # iter-167: mid-thought fragments flushed mid-session on idle silence.
    flushed_utterances = meta_eff.flushed_utterances
    # iter-175: lifetime agent-backchannel emit count from the organic driver.
    backchannels_emitted = meta_eff.backchannels_emitted
    # iter-215: WPM-mirror per-session speed adaptation surface.
    wpm_mirror_active = meta_eff.wpm_mirror_active
    wpm_mirror_initial_speed = meta_eff.wpm_mirror_initial_speed
    wpm_mirror_final_speed = meta_eff.wpm_mirror_final_speed

    def _emit(line: str = "") -> None:
        if file is None:
            print(line)
        else:
            file.write(line + "\n")

    _emit()
    _emit()
    _emit(f"{_DIM}{'─' * 56}{_RESET}")
    if not metrics_list:
        _emit(f"{_BOLD}  Session ended (no completed turns){_RESET}")
        # iter-161: a session can hold one or more mid-thought utterances
        # yet complete zero turns (every fragment was buffered, none ever
        # merged into a responded turn before the user quit). Surface the
        # held count here too so it isn't dropped on the early return —
        # false_endpoints/continuers are necessarily 0 with no turns, so
        # the block only shows the held line.
        # iter-175: a zero-turn session can still have emitted agent
        # backchannels (the user spoke a long monologue the agent nodded
        # along to, then quit before finishing a turn), so surface that
        # count on the early return too rather than dropping it.
        _emit_organic_block(
            _emit,
            OrganicStats(
                utterances_held=utterances_held,
                backchannels_emitted=backchannels_emitted,
            ),
        )
        # iter-160: even a session with zero completed turns can strand a
        # mid-thought fragment — the user spoke one unfinished utterance,
        # it was held, then they quit. Surface it before the early return.
        _emit_stranded_utterance_line(_emit, stranded_utterance)
        # iter-162: a session can also displace mid-thought fragments yet
        # complete zero turns (each abandoned fragment rode in on a turn
        # whose new utterance was itself then held). Surface them too.
        _emit_displaced_utterances_line(_emit, utterances_displaced)
        # iter-167: likewise a zero-turn session can flush a fragment on an
        # inter-turn idle silence — the user trailed off, stayed silent past
        # the idle window, the fragment was flushed, then they quit before
        # speaking a completed turn. Surface it before the early return.
        _emit_flushed_utterances_line(_emit, flushed_utterances)
        _emit()
        return

    n = len(metrics_list)
    stt_times = [m.stt_time for m in metrics_list]
    # iter-049: STT RTF over turns where it was measurable.
    stt_rtfs = [m.stt_rtf for m in metrics_list if m.stt_rtf > 0]
    # iter-072: STT preview-vs-final divergence over turns where
    # both preview and final actually emerged. Filter out 0
    # because that's the "no preview produced" signal as well as
    # the "preview matched perfectly" signal — they're collapsed
    # in this metric. (A "preview matched perfectly" turn is a
    # rare lucky case anyway; if the median is 0 we'd genuinely
    # have nothing to surface.)
    stt_div_values = [
        m.stt_preview_divergence
        for m in metrics_list
        if m.stt_preview_divergence > 0
    ]
    llm_ft = [m.llm_first_token for m in metrics_list]
    # iter-083: FT-A across turns where both timestamps existed
    # (filter zeros — turn errored before LLM or before audio).
    fta_values = [
        m.first_token_to_audio
        for m in metrics_list
        if m.first_token_to_audio > 0
    ]
    # iter-052: LLM TPS over turns where it was measurable.
    llm_tpses = [m.llm_tps for m in metrics_list if m.llm_tps > 0]
    # iter-085: max-token-gap values across turns where the metric
    # was meaningful. Filter zeros (single-token responses) AND
    # sub-200ms gaps (normal jitter). Emitting only meaningful
    # stalls keeps clean sessions clutter-free.
    stall_gaps = [
        m.max_token_gap for m in metrics_list if m.max_token_gap > 0.2
    ]
    # iter-038: median TTFsent over turns where a sentence actually
    # emerged. Filter out 0s (parallel to iter-031's TTFS-zero filter)
    # so a turn with no complete sentence doesn't bias the median.
    llm_fs = [m.llm_first_sentence for m in metrics_list if m.llm_first_sentence > 0]
    tts_times = [m.tts_time for m in metrics_list]
    # iter-050: TTS RTF over turns where it was measurable.
    tts_rtfs = [m.tts_rtf for m in metrics_list if m.tts_rtf > 0]
    # iter-031: a turn that ended without audio (worker error,
    # barge-in before first audio, LLM produced no tokens) leaves
    # ``metrics.ttfs`` at its 0.0 default. Including those zeros
    # in the aggregate biases the median down and makes "Best
    # TTFS: 0ms" appear, which is misleading — TTFS only has
    # meaning for turns that actually played audio. Filter.
    ttfs_times = [m.ttfs for m in metrics_list if m.ttfs > 0]
    # iter-053: naturalness distribution — count turns in each bucket.
    naturalness_counts = {"rushed": 0, "natural": 0, "slow": 0}
    for m in metrics_list:
        if m.naturalness_bucket in naturalness_counts:
            naturalness_counts[m.naturalness_bucket] += 1
    fillers_total = sum(m.fillers_played for m in metrics_list)
    # iter-051: filler false-positive count + denominator (turns
    # where any filler played).
    filler_turns = sum(1 for m in metrics_list if m.fillers_played > 0)
    filler_false_positives = sum(
        1 for m in metrics_list if m.filler_false_positive
    )
    barges_total = sum(1 for m in metrics_list if m.barge_in)
    # iter-040: barge-ins where the cancel actually cut a sentence
    # mid-stream. The "mid-stream rate" tells you "how aggressively
    # are users interrupting" — high = interrupt mid-sentence
    # (impatient or wrong response); low = wait for a natural pause.
    mid_cancels = sum(
        1 for m in metrics_list if m.barge_in and m.sentences_cancelled > 0
    )
    # iter-047: barge-in phase distribution.
    llm_phase_barges = sum(
        1 for m in metrics_list
        if m.barge_in and m.barge_in_phase == "llm_stream"
    )
    playback_phase_barges = sum(
        1 for m in metrics_list
        if m.barge_in and m.barge_in_phase == "playback"
    )
    # iter-056: regret count — barges firing within 200ms of bot
    # first audio. High count = end-of-turn detection misjudges.
    regret_barges = sum(1 for m in metrics_list if m.barge_in_regret)
    # iter-080: total words pre-empted across all barge turns.
    # ``barge_turns`` denominator distinct from preempted_total —
    # tells the operator how often the cut-off happened mid-content
    # vs cleanly between sentences (where preempted_words == 0).
    preempted_total = sum(m.preempted_words for m in metrics_list)
    barge_turns_with_loss = sum(
        1 for m in metrics_list if m.preempted_words > 0
    )
    # iter-154: organic-turn-taking naturalness aggregates (backlog
    # #8). Both stay 0 on the half-duplex path (the per-turn fields
    # are only populated once the organic seams are wired in), so the
    # organic block is fully suppressed for today's sessions.
    false_endpoints_total = sum(
        1 for m in metrics_list if m.false_endpoint
    )
    # iter-163: turns force-emitted by the max_merge_depth cap. Read
    # defensively (getattr) so a pre-iter-163 TurnMetrics shape without the
    # field contributes nothing. A subset of false_endpoints_total.
    merges_capped_total = sum(
        1 for m in metrics_list if getattr(m, "merge_capped", False)
    )
    continuers_total = sum(m.continuers_detected for m in metrics_list)
    # iter-057: total seconds of audio carried over via primed frames.
    # iter-058: total worker errors across the session (sum of
    # per-turn worker_errors counts).
    worker_errors_total = sum(m.worker_errors for m in metrics_list)
    primed_seconds_total = sum(
        m.primed_frames_seconds for m in metrics_list
    )
    # iter-041: barge-in latency over turns where it was measured
    # (>0 — both triggered_at and playback_stopped_at have to be
    # set for the metric to be meaningful).
    # iter-060: LLM cancel-to-close latencies across barge turns.
    cancel_close_lats = [
        m.llm_cancel_to_close
        for m in metrics_list
        if m.llm_cancel_to_close > 0
    ]
    barge_latencies = [
        m.barge_in_latency
        for m in metrics_list
        if m.barge_in and m.barge_in_latency > 0
    ]
    # iter-037: aggregate mic-stale-frame totals. Only surface when
    # something actually leaked — a clean session shouldn't be cluttered
    # with a "0 stale frames" line.
    stale_total = sum(m.mic_stale_frames for m in metrics_list)
    # iter-043: streaming overlap ratios across turns where they
    # could be computed (>0 = audio overlapped LLM stream).
    overlap_ratios = [
        m.streaming_overlap_ratio
        for m in metrics_list
        if m.streaming_overlap_ratio > 0
    ]
    # iter-073: first-sentence overlap savings (seconds shaved off
    # TTFS by parallelizing first synth with LLM streaming). Filter
    # zeros — sequential turns or no-audio turns.
    first_overlap_secs = [
        m.first_synth_overlap_seconds
        for m in metrics_list
        if m.first_synth_overlap_seconds > 0
    ]
    # iter-076: TTFS attribution percentages across turns where all
    # three legs are measurable. Stored as ratios in [0, 1]; we'll
    # render the medians as ints in the emit block.
    attribution_turns = [
        m for m in metrics_list
        if m.ttfs > 0
        and m.stt_time > 0
        and m.llm_first_sentence > 0
        and m.synth_dispatch_seconds > 0
    ]
    # iter-074: bargeable-time fractions across turns where audio
    # actually played (>0 in the field). The interesting statistic
    # is whether ALL turns hit 1.0 (healthy) or any dropped below.
    bargeable_values = [
        m.bargeable_fraction
        for m in metrics_list
        if m.bargeable_fraction > 0
    ]
    # iter-045: mean sentence-length over turns where any sentence
    # was actually submitted (>0). Operator can spot a fragmentation
    # regression at session-summary glance — e.g. a system-prompt
    # nudge that crashes the avg from 70 → 25 chars.
    # iter-059: split coverage values across turns where measurable.
    coverage_values = [
        m.sentence_split_coverage
        for m in metrics_list
        if m.sentence_split_coverage > 0
    ]
    sentence_lens = [
        m.mean_sentence_chars
        for m in metrics_list
        if m.mean_sentence_chars > 0
    ]
    # iter-046: bot WPM across measurable turns.
    bot_wpms = [m.bot_wpm for m in metrics_list if m.bot_wpm > 0]
    # iter-077: context-token counts across turns where the LLM
    # call actually happened (>0 = the messages list had non-empty
    # content). Filter zeros — turns that errored before LLM.
    context_token_counts = [
        m.context_tokens for m in metrics_list if m.context_tokens > 0
    ]
    # iter-071: token-reveal lag — both mean and max collected
    # across turns where the play_fn supplied lag stats. ``!= 0``
    # filter rather than ``> 0`` because lag can be legitimately
    # negative (text-leading-audio).
    token_lag_means = [
        m.mean_token_reveal_lag for m in metrics_list
        if m.mean_token_reveal_lag != 0
    ]
    token_lag_maxes = [
        m.max_token_reveal_lag for m in metrics_list
        if m.max_token_reveal_lag != 0
    ]
    # iter-064: user WPM across turns where transcript + speech_duration
    # were both non-zero. Filter zeros — empty transcript / zero
    # speech turns shouldn't bias the average.
    user_wpms = [m.user_wpm for m in metrics_list if m.user_wpm > 0]
    # iter-061: speaker-open seconds across turns where it was set.
    # Filter out 0s — those represent turns whose worker exited before
    # ever opening the speaker (early error path). Healthy turns
    # always set this >0.
    speaker_opens = [
        m.speaker_open_seconds
        for m in metrics_list
        if m.speaker_open_seconds > 0
    ]
    # iter-062: peak queue depth across all turns. The relevant
    # statistic is the WORST observation — a single turn that piled
    # up sentences is the bottleneck signal we care about. Filter
    # out ≤1 (healthy turns); the line itself only emits when at
    # least one turn backed up.
    queue_peaks = [
        m.max_queue_depth for m in metrics_list if m.max_queue_depth > 1
    ]
    # iter-063: EoT detection latencies across turns where the
    # recorder emitted. Filter zeros (DONE_TOO_SHORT / no-transcription
    # turns leave the field at default).
    eot_latencies = [m.eot_latency for m in metrics_list if m.eot_latency > 0]
    # iter-082: TTC values across turns where the cross-turn
    # measurement was possible. Filter zeros (turn 1 + post-silent-
    # prev turns).
    ttc_values = [
        m.time_to_comprehension
        for m in metrics_list
        if m.time_to_comprehension > 0
    ]
    # iter-065: trailing-silence wall — overhead beyond the configured
    # silence_duration. Filter zeros + sub-chunk noise; the line only
    # emits when at least one turn showed real overhead.
    eot_overheads = [m.eot_overhead for m in metrics_list if m.eot_overhead > 0.010]

    # iter-054: include session duration in the header when known.
    # Format the duration human-readably:
    #   <60s  → "Ns"
    #   <1h   → "Mm Ns" or "Mm"
    #   ≥1h   → "Hh Mm"
    if session_seconds > 0:
        if session_seconds < 60:
            duration_str = f" over {session_seconds:.0f}s"
        elif session_seconds < 3600:
            mins = int(session_seconds // 60)
            secs = int(session_seconds % 60)
            duration_str = (
                f" over {mins}m {secs}s" if secs else f" over {mins}m"
            )
        else:
            hours = int(session_seconds // 3600)
            mins = int((session_seconds % 3600) // 60)
            duration_str = f" over {hours}h {mins}m"
    else:
        duration_str = ""
    _emit(
        f"{_BOLD}  Session Summary "
        f"({n} turn{'' if n == 1 else 's'}{duration_str}){_RESET}"
    )
    # iter-054: turns/minute as a useful denominator for rate metrics.
    # Only emit when session_seconds was provided AND >= a reasonable
    # threshold (avoid divide-by-tiny on unit-test-shaped sessions).
    if session_seconds >= 1.0 and n > 0:
        tpm = (n / session_seconds) * 60.0
        _emit(f"    Turns/min:        {tpm:.1f}")
    # iter-063: EoT detection latency — emit before STT to mirror
    # the per-turn pipeline order (speech → EoT → STT → LLM → ...).
    # The relevant statistic is the median: it tells the operator
    # how long the user is left hanging on average after they stop
    # talking. Worst is also useful — a single 2s outlier feels
    # broken even if the median is fine.
    if eot_latencies:
        _emit(f"    Median EoT:       {_median_ms(eot_latencies):.0f}ms")
        if len(eot_latencies) >= 2 and max(eot_latencies) > min(eot_latencies):
            _emit(f"    Worst EoT:        {max(eot_latencies) * 1000:.0f}ms")
        # iter-065: trailing-silence wall median. Only meaningful
        # when at least one turn showed real overhead — otherwise
        # the EoT wait is fully explained by silence_duration.
        if eot_overheads:
            ov_med = statistics.median(eot_overheads) * 1000
            _emit(f"    EoT overhead:     {ov_med:.0f}ms (above silence_duration)")
    # iter-082: TTC median + outlier counts. Sub-500ms = "user
    # didn't need to listen" (bot was telling them what they
    # already knew); >5s = "user was confused / thinking." Both
    # are signals; the bell-curve target is 1-3s.
    if ttc_values:
        med_ttc_ms = statistics.median(ttc_values) * 1000
        rushed = sum(1 for v in ttc_values if v < 0.5)
        slow = sum(1 for v in ttc_values if v > 5.0)
        outlier_str = ""
        if rushed or slow:
            bits = []
            if rushed:
                bits.append(f"{rushed} rushed")
            if slow:
                bits.append(f"{slow} slow")
            outlier_str = f" ({', '.join(bits)})"
        _emit(f"    Median TTC:       {med_ttc_ms:.0f}ms{outlier_str}")
    _emit(f"    Median STT:       {_median_ms(stt_times):.0f}ms")
    if stt_rtfs:
        _emit(f"    Median STT RTF:   {statistics.median(stt_rtfs):.2f}x")
    # iter-072: median preview divergence as a percentage. >30%
    # is "the live preview was generally misleading"; <10% is
    # "the preview was reliable enough for users to trust."
    if stt_div_values:
        med_div_pct = statistics.median(stt_div_values) * 100
        _emit(f"    STT preview Δ:    {med_div_pct:.0f}% (median)")
    _emit(f"    Median LLM 1st:   {_median_ms(llm_ft):.0f}ms")
    # iter-083: median FT-A. Together with Median LLM 1st, the
    # operator can see at a glance which side of TTFS dominates
    # (LLM-bound vs synth/dispatch-bound).
    if fta_values:
        _emit(f"    Median FT-A:      {_median_ms(fta_values):.0f}ms")
    if llm_tpses:
        _emit(f"    Median LLM TPS:   {statistics.median(llm_tpses):.0f}")
    # iter-085: surface the WORST stall observed across the
    # session + how many turns showed any stall. The worst is
    # the operator-actionable signal (one bad turn ruins the
    # demo); the count tells you whether stalls are persistent
    # (fix the endpoint) or sporadic (noise).
    if stall_gaps:
        worst_ms = max(stall_gaps) * 1000
        _emit(
            f"    Worst LLM stall:  "
            f"{worst_ms:.0f}ms ({len(stall_gaps)}/{n} turns)"
        )
    if llm_fs:
        _emit(f"    Median LLM sent:  {_median_ms(llm_fs):.0f}ms")
    _emit(f"    Median TTS:       {_median_ms(tts_times):.0f}ms")
    if tts_rtfs:
        _emit(f"    Median TTS RTF:   {statistics.median(tts_rtfs):.2f}x")
    # iter-089: TTFS block extracted to _emit_ttfs_block helper —
    # 80 lines of co-emitted lines (median, best, sub-second,
    # rhythm, jitter, cold-start, naturalness) collapsed into one
    # call. Behavior-preserving: output is byte-for-byte identical
    # to the inline version.
    _emit_ttfs_block(_emit, ttfs_times, metrics_list, naturalness_counts)
    # iter-091: filler block extracted to _emit_filler_block.
    unique_filler_ids = {
        m.last_filler_id for m in metrics_list
        if m.last_filler_id != 0
    }
    # iter-096: compute a recommended idle_threshold when FPs are
    # firing AND the caller passed the current threshold via
    # SessionMeta AND we have observed llm_first_token data. The
    # recommendation is the larger of:
    #   (a) 1.2x the current threshold — modest bump.
    #   (b) 75th percentile of observed first-token times + 100ms
    #       safety margin — covers most real first-token waits
    #       while still allowing fillers on the slow tail.
    # 0.0 = "no recommendation" (don't render).
    recommended_idle = 0.0
    if (
        filler_false_positives > 0
        and meta_eff.idle_threshold > 0
        and len([m.llm_first_token for m in metrics_list
                 if m.llm_first_token > 0]) >= 2
    ):
        positives = sorted(
            m.llm_first_token for m in metrics_list
            if m.llm_first_token > 0
        )
        # 75th percentile via the standard inclusive method.
        idx = max(0, int(round(0.75 * (len(positives) - 1))))
        p75 = positives[idx]
        recommended_idle = max(meta_eff.idle_threshold * 1.2, p75 + 0.1)
        # Round to one decimal so the rendered value is clean.
        recommended_idle = round(recommended_idle, 1)
    _emit_filler_block(
        _emit,
        FillerStats(
            fillers_total=fillers_total,
            filler_turns=filler_turns,
            filler_false_positives=filler_false_positives,
            unique_filler_count=len(unique_filler_ids),
            recommended_idle_threshold=recommended_idle,
        ),
    )
    # iter-114: cross-turn filler-diversity sentinel. Only fires
    # when 3+ consecutive turns played the same filler id —
    # would surface if iter-113's FIFO regressed.
    _emit_filler_diversity_line(
        _emit,
        [m.last_filler_id for m in metrics_list],
    )
    # iter-115: naturalness-consistency check. Only fires when
    # 5+ consecutive turns landed in the same non-"natural"
    # bucket (rushed/slow) — surfaces a speed-config problem.
    _emit_naturalness_consistency_line(
        _emit,
        [m.naturalness_bucket for m in metrics_list],
    )
    # iter-120: barge-phase consistency check. Only fires when
    # 4+ consecutive turns barged in the same phase
    # (llm_stream / playback) — surfaces a UX issue.
    _emit_barge_phase_consistency_line(
        _emit,
        [m.barge_in_phase for m in metrics_list],
    )
    # iter-128: sentence-length-bucket consistency check. Only
    # fires when 5+ consecutive turns produced very_short
    # (< 15 chars) or long (> 60 chars) sentences — surfaces a
    # splitter-tuning issue.
    _emit_sentence_length_consistency_line(
        _emit,
        [m.mean_sentence_chars for m in metrics_list],
    )
    # iter-140: STT-RTF consistency check. Only fires when 5+
    # consecutive turns ran STT slower than realtime
    # (slow 1.0-2.0 / very_slow > 2.0) — surfaces an engine/model
    # that's too heavy for the host hardware.
    _emit_stt_rtf_consistency_line(
        _emit,
        [m.stt_rtf for m in metrics_list],
    )
    # iter-141: TTS-RTF consistency check. Only fires when 5+
    # consecutive turns synthesized TTS slower than realtime
    # (slow 1.0-2.0 / very_slow > 2.0) — surfaces an engine/voice
    # that's too heavy for the host hardware (synth-overlap won't
    # help when synth itself is the bottleneck).
    _emit_tts_rtf_consistency_line(
        _emit,
        [m.tts_rtf for m in metrics_list],
    )
    # iter-142: LLM-TPS consistency check. Only fires when 5+
    # consecutive turns streamed the LLM slower than its useful
    # throughput (slow 10-25 tps / very_slow < 10 tps) — surfaces an
    # LLM that's starving the TTS worker, the one continuous-metric
    # sentinel where the fine state is a HIGH value (fast tps).
    _emit_llm_tps_consistency_line(
        _emit,
        [m.llm_tps for m in metrics_list],
    )
    # iter-143: streaming-overlap consistency check. Only fires when
    # 5+ consecutive turns barely overlapped synth with the LLM
    # stream (low 0.20-0.50 / very_low < 0.20) — surfaces the iter-008
    # streaming-overlap design failing to mask synth, the SECOND
    # continuous-metric sentinel whose fine state is a HIGH value
    # (lots of overlap).
    _emit_streaming_overlap_consistency_line(
        _emit,
        [m.streaming_overlap_ratio for m in metrics_list],
    )
    # iter-208: synth+dispatch consistency check. Only fires when
    # 5+ consecutive turns spent a noticeable chunk of TTFS in the
    # synth + speaker-open + dispatch leg (slow 0.15-0.35s /
    # very_slow > 0.35s) — completes the iter-076 TTFS attribution
    # decomposition with a sentinel on every leg (STT iter-140,
    # LLM iter-142, synth/dispatch here). Surfaces the final leg
    # as the recurring opening-latency bottleneck even when its
    # median hides behind one bad turn.
    _emit_synth_dispatch_consistency_line(
        _emit,
        [m.synth_dispatch_seconds for m in metrics_list],
    )
    # iter-209: EoT-overhead consistency check. Only fires when 5+
    # consecutive turns left the end-of-turn wait dominated by
    # recording-loop overhead (slow 0.10-0.25s / very_slow > 0.25s)
    # rather than the configured silence_duration — surfaces the
    # case where tuning chat.vad.silence_duration WON'T recover the
    # latency (the recording loop itself is slow), so the operator
    # doesn't chase the wrong knob. EoT latency is metric 1.2, which
    # dominates "the agent feels slow" complaints.
    _emit_eot_overhead_consistency_line(
        _emit,
        [m.eot_overhead for m in metrics_list],
    )
    # iter-210: bot-speech-rate consistency check. Only fires when 5+
    # consecutive turns synthesized speech outside the comprehension
    # sweet spot (rushed > 200 WPM / sluggish < 130 WPM). FIRST
    # two-sided-band sentinel: the fine state is the MIDDLE (130-200
    # WPM, matching the per-turn green display), and the two flagged
    # ends need OPPOSITE kokoro `speed`-knob corrections. The median
    # bot-WPM line (iter-094) can read healthy while a sustained
    # rushed/sluggish run hides inside it.
    _emit_bot_wpm_consistency_line(
        _emit,
        [m.bot_wpm for m in metrics_list],
    )
    # iter-211: max-token-gap consistency check. Only fires when 5+
    # consecutive turns suffered a noticeable worst-case mid-stream LLM
    # stall (slow 0.50-2.0s / very_slow > 2.0s). iter-085 added the
    # per-turn metric because these stalls are otherwise silent — the
    # user just hears a pause and no signal fires. Distinct from the
    # iter-142 llm-tps sentinel: that watches AVERAGE throughput, this
    # watches the WORST single gap, which a healthy mean TPS can hide.
    _emit_max_token_gap_consistency_line(
        _emit,
        [m.max_token_gap for m in metrics_list],
    )
    # iter-212: TTFS consistency check. Only fires when 5+ consecutive
    # turns were slow to start speaking (slow 3.0-6.0s / very_slow >
    # 6.0s). TTFS is the headline latency metric the VISION optimizes
    # for; the median + warmup-penalty block (_emit_ttfs_block) can read
    # healthy while a sustained slow run hides in the back half of the
    # session. Distinct from iter-211 (mid-stream stall AFTER the bot
    # starts) and iter-209 (EoT wait BEFORE TTFS starts): this watches
    # the whole speech-stop → speaker latency end to end.
    _emit_ttfs_consistency_line(
        _emit,
        [m.ttfs for m in metrics_list],
    )
    # iter-224: STT-preview-divergence consistency check. Only fires
    # when 5+ consecutive turns had the live preview drift from the
    # final transcript (noisy 0.15-0.30 / broken > 0.30). iter-072
    # added the per-turn metric because a bad preview is otherwise
    # silent — the session-summary never surfaced it, so a
    # persistently-wrong read-along transcript ("preview UX is
    # broken") goes unnoticed. Distinct from iter-140's stt_rtf
    # sentinel: that watches STT SPEED; STT can be fast yet still
    # produce a preview that keeps changing under the user.
    _emit_stt_preview_divergence_consistency_line(
        _emit,
        [m.stt_preview_divergence for m in metrics_list],
    )
    # iter-225: sentence-split-coverage consistency check. Only fires
    # when 5+ consecutive turns flushed a large trailing remainder
    # (partial 0.70-0.90 / poor < 0.70). iter-059 added the per-turn
    # metric but it only shows a "% complete" suffix on a single
    # turn's line — a SUSTAINED low-coverage run (the splitter
    # systematically failing to close sentences) never surfaces in the
    # summary, yet it defeats the iter-008 streaming-overlap design
    # turn after turn. Distinct from iter-143's streaming_overlap_ratio
    # sentinel: that measures how much synth DID overlap; this measures
    # whether the splitter even gave synth full sentences to overlap
    # with. Low coverage is an upstream cause of low overlap.
    _emit_split_coverage_consistency_line(
        _emit,
        [m.sentence_split_coverage for m in metrics_list],
    )
    # iter-226: worker-idle-gap consistency check. Only fires when 5+
    # consecutive turns left the SentenceWorker blocked between
    # sentences (stalled 0.3-1.0s / starved > 1.0s). iter-044 added
    # the per-turn metric to localize pipeline bottlenecks, but it
    # only colors a single turn's inline figure yellow above 0.3s — a
    # SUSTAINED high-idle-gap run (the LLM systematically failing to
    # keep up with synth+playback) never surfaces in the summary, yet
    # the user hears stop-start audio turn after turn. Distinct from
    # iter-143's streaming_overlap_ratio sentinel: that measures the
    # FRACTION of synth that overlapped; this measures the ABSOLUTE
    # seconds the worker sat idle, which a decent overlap ratio can
    # still hide on one slow sentence.
    _emit_worker_idle_gap_consistency_line(
        _emit,
        [m.worker_idle_gap_total for m in metrics_list],
    )
    # iter-090: barge block extracted to _emit_barge_block helper.
    # ~76 lines of co-emitted lines (count, interruption rate,
    # latency, phase distribution, regret, pre-empted words)
    # collapsed into one call. Behavior-preserving: byte-for-byte
    # identical to the inline version.
    _emit_barge_block(
        _emit,
        BargeStats(
            barges_total=barges_total,
            mid_cancels=mid_cancels,
            n=n,
            barge_latencies=barge_latencies,
            cancel_close_lats=cancel_close_lats,
            llm_phase_barges=llm_phase_barges,
            playback_phase_barges=playback_phase_barges,
            regret_barges=regret_barges,
            preempted_total=preempted_total,
            barge_turns_with_loss=barge_turns_with_loss,
        ),
    )
    # iter-154: organic-turn-taking naturalness block (backlog #8).
    # Fully suppressed when both counters are zero — the half-duplex
    # default — so today's summaries are byte-for-byte unchanged.
    _emit_organic_block(
        _emit,
        OrganicStats(
            false_endpoints=false_endpoints_total,
            continuers_total=continuers_total,
            n=n,
            # iter-161: held-utterance count tracked by run_session and
            # threaded through SessionMeta.
            utterances_held=utterances_held,
            # iter-163: merge-cap force-emits, derived from the per-turn
            # TurnMetrics.merge_capped flags.
            merges_capped=merges_capped_total,
            # iter-175: lifetime agent-backchannel emit count from the
            # organic driver, threaded through SessionMeta.
            backchannels_emitted=backchannels_emitted,
        ),
    )
    # iter-057: total seconds of audio carried over by the watcher
    # via primed_frames. Report regardless of whether we computed
    # the barge-block above (the metric is technically only set on
    # barge turns, but its reporting is independent of barge-count
    # totals).
    # iter-104: extracted to _emit_primed_audio_line helper.
    _emit_primed_audio_line(_emit, primed_seconds_total)
    # iter-160: surface a mid-thought fragment the organic aggregator was
    # holding at shutdown (suppressed unless one was actually stranded).
    _emit_stranded_utterance_line(_emit, stranded_utterance)
    # iter-162: surface mid-thought fragments the aggregator released
    # alongside a responded turn (suppressed unless any were displaced).
    _emit_displaced_utterances_line(_emit, utterances_displaced)
    # iter-167: surface mid-thought fragments flushed mid-session on a long
    # inter-turn idle silence (suppressed unless any were flushed).
    _emit_flushed_utterances_line(_emit, flushed_utterances)
    # iter-058: error rate per stage. LLM errors are session-level
    # (kill the turn outright); worker errors are per-turn (partial
    # turn — some sentences synthed, others raised). Show only when
    # at least one error happened.
    # iter-092: errors block extracted to _emit_errors_block helper.
    error_turns = [m for m in metrics_list if m.worker_errors > 0]
    silent_turns = sum(
        1 for m in metrics_list if m.transcript and m.ttfs == 0
    )
    _emit_errors_block(
        _emit,
        ErrorStats(
            llm_errors=llm_errors,
            worker_errors_total=worker_errors_total,
            error_turns_with_audio=sum(1 for m in error_turns if m.ttfs > 0),
            error_turns_total=len(error_turns),
            n=n,
            false_triggers=false_triggers,
            silent_turns=silent_turns,
        ),
    )
    # iter-103: extracted to _emit_recording_block. Stats
    # bundle iter-037 stale_total + iter-048 false_triggers.
    _emit_recording_block(
        _emit,
        RecordingStats(
            stale_total=stale_total,
            false_triggers=false_triggers,
            n=n,
        ),
    )
    if overlap_ratios:
        # iter-043: median streaming overlap across measurable turns.
        # >50% means the worker generally got audio out before the
        # LLM finished — iter-008 streaming-overlap is paying off.
        # <20% means the bot responded so fast (or the LLM is so
        # chatty) that overlap isn't happening; investigate
        # first-sentence latency (iter-038's TTFsent) and synth time.
        median_pct = statistics.median(overlap_ratios) * 100
        _emit(f"    Median overlap:   {median_pct:.0f}%")
    # iter-073: median first-sentence savings. The interesting
    # statistic — total ms shaved off TTFS on average. >100ms is
    # meaningful (users notice the difference between 600ms and
    # 700ms TTFS).
    if first_overlap_secs:
        med_save_ms = statistics.median(first_overlap_secs) * 1000
        _emit(f"    1st-synth saved:  {med_save_ms:.0f}ms median")
    # iter-076: TTFS attribution medians. Tells the operator at
    # session-summary glance which pipeline leg dominates: high
    # STT% means transcription latency, high LLM% means first-
    # sentence wait (preamble or low TPS), high synth% means
    # TTS or dispatch overhead.
    if attribution_turns:
        stt_pcts = [
            m.stt_time / m.ttfs for m in attribution_turns
        ]
        llm_pcts = [
            m.llm_first_sentence / m.ttfs for m in attribution_turns
        ]
        synth_pcts = [
            m.synth_dispatch_seconds / m.ttfs for m in attribution_turns
        ]
        med_stt = int(statistics.median(stt_pcts) * 100)
        med_llm = int(statistics.median(llm_pcts) * 100)
        med_synth = int(statistics.median(synth_pcts) * 100)
        _emit(
            f"    TTFS breakdown:   "
            f"STT {med_stt}% + LLM {med_llm}% + synth {med_synth}%"
        )
    # iter-074: bargeable-time fraction summary. Emit ONLY when at
    # least one turn dropped below the healthy threshold — clean
    # sessions don't need the line, but a regression should be
    # impossible to miss.
    # iter-104: extracted to _emit_bargeable_line helper.
    _emit_bargeable_line(_emit, bargeable_values)
    # iter-105: WER line — only emits when at least one turn
    # carried a measured WER (i.e., a reference transcript was
    # supplied at the call site). Most sessions have no
    # references → silent.
    wer_values = [m.wer for m in metrics_list if m.wer_measured]
    _emit_wer_line(_emit, wer_values)
    # iter-095: sentence block extracted to _emit_sentence_block.
    longest = max(
        (m.max_sentence_chars for m in metrics_list
         if m.max_sentence_chars > 0),
        default=0,
    )
    shortest = min(
        (m.min_sentence_chars for m in metrics_list
         if m.min_sentence_chars > 0),
        default=0,
    )
    _emit_sentence_block(
        _emit,
        SentenceStats(
            sentence_lens=sentence_lens,
            min_chars_seen=shortest,
            max_chars_seen=longest,
            coverage_values=coverage_values,
        ),
    )
    # iter-094: WPM block extracted to _emit_wpm_block helper.
    _emit_wpm_block(_emit, WpmStats(user_wpms=user_wpms, bot_wpms=bot_wpms))
    # iter-215: surface the WPM mirror's per-session speed adaptation right
    # under the WPM medians / mirror gap it complements. Suppressed entirely on
    # the off-by-default path (no live mirror), so a session that never enabled
    # mirroring sees the pre-iter-215 summary byte-for-byte.
    _emit_wpm_mirror_line(
        _emit,
        active=wpm_mirror_active,
        initial_speed=wpm_mirror_initial_speed,
        final_speed=wpm_mirror_final_speed,
    )
    # iter-077: context size summary. The MEDIAN tells you the
    # typical per-call cost; the MAX tells you the worst case.
    # Pair them: if max ≫ median, late turns blew up — likely a
    # trim regression. The growth_per_turn slope is also useful
    # but needs paired indices, not just values; emit it only
    # when ≥3 turns have the metric (least-squares is overkill;
    # just first-vs-last delta gives a quick signal).
    # iter-097: history block extracted to _emit_history_block.
    _emit_history_block(
        _emit,
        HistoryStats(
            context_token_counts=context_token_counts,
            trim_events=trim_events,
            trim_messages_evicted=trim_messages_evicted,
        ),
    )
    # iter-071: median mean-lag + worst peak across the session.
    # Sign-preserved on output. >50ms median is "the user notices
    # the desync"; >300ms peak on any one token is a visible glitch
    # even if the average looks OK.
    if token_lag_means:
        med_lag_ms = statistics.median(token_lag_means) * 1000
        # Pick the lag with the largest absolute value as "worst".
        worst_peak = max(token_lag_maxes, key=abs) if token_lag_maxes else 0.0
        worst_ms = worst_peak * 1000
        _emit(
            f"    Token lag:        "
            f"{med_lag_ms:+.0f}ms median, "
            f"{worst_ms:+.0f}ms worst peak"
        )
    # iter-062: worst queue depth across the session. Skip when no
    # turn backed up (≤1) — clean sessions don't need the line.
    # When multiple turns backed up, show how many to differentiate
    # "one bad turn" from "synth is chronically behind."
    if queue_peaks:
        worst = max(queue_peaks)
        n_backed = len(queue_peaks)
        if n_backed == 1:
            _emit(f"    Worst queue:      {worst} (1 turn backed up)")
        else:
            _emit(
                f"    Worst queue:      {worst} "
                f"({n_backed}/{n} turns backed up)"
            )
    # iter-061: speaker-open overhead. Median + worst across measured
    # turns. >50ms median is "the persistent-speaker win is slipping" —
    # the iter-008 design assumes opens are cheap because they happen
    # once per turn rather than once per sentence; if they get expensive
    # we lose the headroom. Single-turn sessions show just the one
    # value.
    if speaker_opens:
        worst_ms = max(speaker_opens) * 1000
        if len(speaker_opens) > 1:
            med_ms = statistics.median(speaker_opens) * 1000
            _emit(
                f"    Speaker open:     "
                f"median {med_ms:.0f}ms / worst {worst_ms:.0f}ms"
            )
        else:
            _emit(f"    Speaker open:     {worst_ms:.0f}ms")
    _emit(f"    Model:            {llm_config.get('model', 'unknown')}")
    _emit()
