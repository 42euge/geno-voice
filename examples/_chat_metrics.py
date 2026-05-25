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
from dataclasses import dataclass

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
    # iter-053: TTFS bucketed against the human-conversation
    # sweet spot. "rushed" (<200ms): bot interrupted natural
    # turn-taking pause; "natural" (200-400ms): matches human
    # conversational rhythm; "slow" (>400ms): user notices
    # latency. Counter-intuitive: lower TTFS isn't always better.
    # "" when no audio played this turn. Metric 3.1 in the
    # perf-metrics taxonomy ("Novel/speculative").
    naturalness_bucket: str = ""
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
        print(f"  {_DIM}│{_RESET}  LLM 1st tok:   {self.llm_first_token*1000:>7.0f}ms")
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
        print(
            f"  {_DIM}│{_RESET}  LLM total:     "
            f"{self.llm_total*1000:>7.0f}ms  "
            f"({self.model}{tps_str}{ctx_str})"
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


def print_session_summary(
    metrics_list: list[TurnMetrics],
    llm_config: dict,
    *,
    file=None,
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
    """
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
    # iter-052: LLM TPS over turns where it was measurable.
    llm_tpses = [m.llm_tps for m in metrics_list if m.llm_tps > 0]
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
    if llm_tpses:
        _emit(f"    Median LLM TPS:   {statistics.median(llm_tpses):.0f}")
    if llm_fs:
        _emit(f"    Median LLM sent:  {_median_ms(llm_fs):.0f}ms")
    _emit(f"    Median TTS:       {_median_ms(tts_times):.0f}ms")
    if tts_rtfs:
        _emit(f"    Median TTS RTF:   {statistics.median(tts_rtfs):.2f}x")
    if ttfs_times:
        _emit(
            f"    {_BOLD}Median TTFS:      {_median_ms(ttfs_times):.0f}ms{_RESET}"
        )
        _emit(f"    Best TTFS:        {min(ttfs_times) * 1000:.0f}ms")
        # iter-055: conversation rhythm score. 1 - (stdev / median).
        # Higher = more consistent cadence (feels like a personality).
        # Lower = jittery (feels like a system). Needs ≥2 turns
        # for stdev to be defined; clamp to [0, 1] since high-variance
        # sessions can produce stdev > median → negative raw score.
        if len(ttfs_times) >= 2:
            med = statistics.median(ttfs_times)
            sd = statistics.stdev(ttfs_times)
            raw = 1.0 - sd / max(med, 1e-6)
            rhythm = max(0.0, min(1.0, raw))
            _emit(f"    Rhythm score:     {rhythm:.2f}")
            # iter-068: promote the raw stdev to its own line. The
            # rhythm score is a normalized [0,1] number useful for
            # at-a-glance comparison; the jitter in milliseconds is
            # more actionable when tuning. Humans tolerate consistent
            # slow turn-taking better than inconsistent fast turn-
            # taking — a 250ms jitter at 600ms median feels more
            # broken than a steady 750ms median.
            _emit(f"    TTFS jitter:      ±{sd * 1000:.0f}ms")
        # iter-066: cold-start latency penalty. The turn-1 TTFS minus
        # the steady-state median (turns 2:N). Captures lazy
        # initialization that hits turn 1 disproportionately — model
        # load, speaker open, TTS warmup, lazy imports — and gets
        # buried in the overall median. Needs ≥2 turns with measurable
        # TTFS, AND turn 1 must have measurable TTFS (otherwise we'd
        # be comparing an absent first turn). Skip emit when penalty
        # is below the chunk-noise floor (±50ms): turn-to-turn jitter
        # from playback timing alone can produce that gap on healthy
        # systems.
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
                # Sign matters: positive means turn 1 was slower
                # (the typical cold-start case); negative means
                # turn 1 was faster (rare — could be cache warming
                # in subsequent turns going wrong, or bot reaching
                # GC pauses post-turn-1).
                _emit(
                    f"    Cold start:       {ms:+.0f}ms "
                    f"vs steady state"
                )
        # iter-053: naturalness distribution. Total = sum of all
        # buckets. Show only when at least one turn was bucketed.
        n_total = sum(naturalness_counts.values())
        if n_total > 0:
            _emit(
                f"    Naturalness:      "
                f"{naturalness_counts['rushed']} rushed, "
                f"{naturalness_counts['natural']} natural, "
                f"{naturalness_counts['slow']} slow"
            )
    else:
        # All turns ended without audio. Emit a placeholder rather
        # than a misleading "0ms" so the user knows it isn't a
        # win, it's an absence of data.
        _emit(f"    {_BOLD}Median TTFS:      n/a{_RESET}")
        _emit(f"    Best TTFS:        n/a")
    if fillers_total:
        _emit(f"    Fillers played:   {fillers_total}")
        # iter-051: false-positive rate among the turns where a
        # filler played. Tune idle_threshold up if FP is high —
        # the bot's rendering disfluency for no benefit.
        if filler_turns > 0:
            fp_pct = (filler_false_positives / filler_turns) * 100
            if filler_false_positives > 0:
                _emit(
                    f"    Filler FP rate:   "
                    f"{filler_false_positives}/{filler_turns} "
                    f"({fp_pct:.0f}%) — tune idle_threshold up"
                )
    if barges_total:
        if mid_cancels:
            pct = (mid_cancels / barges_total) * 100
            _emit(
                f"    Barge-ins:        {barges_total} "
                f"({mid_cancels} mid-stream, {pct:.0f}%)"
            )
        else:
            _emit(
                f"    Barge-ins:        {barges_total} "
                f"(all between sentences)"
            )
        # iter-069: interruption rate as a fraction of total turns.
        # Distinct from the mid-stream % above (denominator is total
        # barges, not turns). Industry single-number UX KPI: "what
        # fraction of bot turns did the user feel they had to
        # interrupt." High = bot is too verbose, slow, or wrong.
        # Low = bot is well-tuned for this user. Metric 1.18 in the
        # perf-metrics taxonomy.
        if n > 0:
            int_pct = (barges_total / n) * 100
            _emit(
                f"    Interruption rate: "
                f"{barges_total}/{n} turns ({int_pct:.0f}%)"
            )
        # iter-041: median + worst barge-in latency. Useful for
        # tuning the watcher's poll interval and the worker cancel
        # path. >200ms median is a reliable "feels broken" signal.
        if barge_latencies:
            _emit(
                f"    Median barge:     {_median_ms(barge_latencies):.0f}ms"
            )
            _emit(
                f"    Worst barge:      "
                f"{max(barge_latencies) * 1000:.0f}ms"
            )
        # iter-060: median LLM cancel-to-close across barge turns.
        # >500ms median is a reliable "HTTP socket hangs."
        if cancel_close_lats:
            _emit(
                f"    Median LLM canc:  "
                f"{_median_ms(cancel_close_lats):.0f}ms"
            )
        # iter-047: phase distribution. Only show when at least one
        # phase value was set; gives root-cause hint:
        #   high LLM-phase = users impatient with TTFS — fix LLM TTFT.
        #   high playback-phase = bot output is verbose / wrong —
        #     fix system prompt or response quality.
        if llm_phase_barges or playback_phase_barges:
            _emit(
                f"    Barge phases:     "
                f"{llm_phase_barges} LLM-stream, "
                f"{playback_phase_barges} playback"
            )
        # iter-056: regret rate. High = end-of-turn detection
        # misjudges; consider raising silence_duration.
        if regret_barges:
            pct = (regret_barges / barges_total) * 100
            _emit(
                f"    Regret rate:      "
                f"{regret_barges}/{barges_total} ({pct:.0f}%) "
                f"— bot may be pre-empting; raise silence_duration"
            )
    # iter-057: total seconds of audio carried over by the watcher
    # via primed_frames. Report regardless of whether we computed
    # the barge-block above (the metric is technically only set on
    # barge turns, but its reporting is independent of barge-count
    # totals).
    if primed_seconds_total > 0:
        _emit(
            f"    Primed audio:     "
            f"{primed_seconds_total:.1f}s "
            f"(carried into next turn — validates iter-025)"
        )
    # iter-058: error rate per stage. LLM errors are session-level
    # (kill the turn outright); worker errors are per-turn (partial
    # turn — some sentences synthed, others raised). Show only when
    # at least one error happened.
    if llm_errors > 0 or worker_errors_total > 0:
        attempts = n + llm_errors + false_triggers
        bits = []
        if llm_errors > 0:
            bits.append(f"{llm_errors} LLM")
        if worker_errors_total > 0:
            bits.append(f"{worker_errors_total} worker")
        _emit(
            f"    Errors:           "
            f"{', '.join(bits)} "
            f"(over {attempts} attempt{'' if attempts == 1 else 's'})"
        )
        # iter-067: worker error-recovery success rate. Of the turns
        # where the SentenceWorker raised at least one synth/play
        # exception, what fraction still produced audio (ttfs > 0)?
        # 100% recovery is silent partial degradation — the user
        # heard a complete-sounding response but a sentence inside
        # was dropped. 0% recovery means every error knocked out
        # the whole turn (loud failure — user notices). Surface so
        # the operator can spot bugs that were swallowed by the
        # worker's per-sentence error isolation.
        error_turns = [m for m in metrics_list if m.worker_errors > 0]
        if error_turns:
            recovered = sum(1 for m in error_turns if m.ttfs > 0)
            pct = (recovered / len(error_turns)) * 100
            _emit(
                f"    Worker recovery:  "
                f"{recovered}/{len(error_turns)} turns produced audio "
                f"({pct:.0f}%) — partial degradation"
            )
    # iter-079: silent-turn rate. Turns where the user spoke
    # (transcript captured) but the bot produced no audio
    # (ttfs == 0). Distinct from worker errors — no exception
    # fired, the worker just didn't manage to play anything.
    # The user's experience: "I said something and got silence."
    # Common causes: worker.errors took out every sentence, LLM
    # returned an empty response, all sentences were pre-empted
    # by a barge that landed before first_audio_at, the LLM
    # produced no terminator-bearing tokens (no synth submissions).
    # Only emit when the rate is non-zero — clean sessions don't
    # need a "0 silent turns" line.
    silent_turns = sum(
        1 for m in metrics_list if m.transcript and m.ttfs == 0
    )
    if silent_turns > 0 and n > 0:
        pct = (silent_turns / n) * 100
        _emit(
            f"    Silent turns:     "
            f"{silent_turns}/{n} ({pct:.0f}%) — bot produced no audio"
        )
    if stale_total:
        # iter-037: surface aggregate stale-frame total so a "session
        # had constant echo" pattern is visible at the end of the run.
        stale_seconds_total = stale_total / 16000
        _emit(
            f"    Mic stale:        {stale_total} frames "
            f"({stale_seconds_total:.1f}s) — check echo cancellation"
        )
    # iter-048: VAD false-trigger rate. Only emit when at least one
    # false trigger happened — clean sessions don't need the line.
    if false_triggers > 0:
        attempts = false_triggers + n
        pct = (false_triggers / attempts) * 100
        _emit(
            f"    VAD false-trig:   {false_triggers}/{attempts} "
            f"({pct:.0f}%) — tune silence_threshold or min_speech_duration"
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
    if bargeable_values and min(bargeable_values) < 0.99:
        worst = min(bargeable_values) * 100
        below_count = sum(1 for v in bargeable_values if v < 0.99)
        _emit(
            f"    Bargeable:        "
            f"{worst:.0f}% worst "
            f"({below_count}/{len(bargeable_values)} turns < 99%) — "
            f"watcher coverage regression"
        )
    if sentence_lens:
        # iter-045: mean across the per-turn means.
        avg_chars = sum(sentence_lens) / len(sentence_lens)
        _emit(f"    Mean sentence:    {avg_chars:.0f} chars")
        # iter-070: longest sentence observed across the session.
        # >150 chars is "under-fragmented run-on" — delays TTFS
        # because synth waits on the full sentence. Operator can
        # spot a single bad turn that the mean alone hides.
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
        if longest > 0 and longest != shortest:
            _emit(
                f"    Sentence range:   "
                f"[{shortest}..{longest}] chars (session)"
            )
    if coverage_values:
        # iter-059: median split coverage across turns. <90% is a
        # signal the LLM isn't ending its responses with punctuation
        # often enough — system-prompt opportunity.
        median_cov = statistics.median(coverage_values) * 100
        _emit(f"    Split coverage:   {median_cov:.0f}%")
    # iter-064: user WPM (median across measurable turns) + the
    # mirror gap (bot - user WPM) when both are known. The mirror
    # gap predicts conversational "feel": ≈0 = mirroring (high
    # rapport); >40 = bot too fast for user (likely interruption
    # source); <-40 = bot too slow (user impatient).
    if user_wpms:
        median_user_wpm = statistics.median(user_wpms)
        _emit(f"    Median user WPM:  {median_user_wpm:.0f}")
    if bot_wpms:
        median_wpm = statistics.median(bot_wpms)
        _emit(f"    Median bot WPM:   {median_wpm:.0f}")
        if user_wpms:
            gap = median_wpm - statistics.median(user_wpms)
            _emit(f"    Mirror gap:       {gap:+.0f} WPM (bot − user)")
    # iter-077: context size summary. The MEDIAN tells you the
    # typical per-call cost; the MAX tells you the worst case.
    # Pair them: if max ≫ median, late turns blew up — likely a
    # trim regression. The growth_per_turn slope is also useful
    # but needs paired indices, not just values; emit it only
    # when ≥3 turns have the metric (least-squares is overkill;
    # just first-vs-last delta gives a quick signal).
    if context_token_counts:
        med_ctx = statistics.median(context_token_counts)
        max_ctx = max(context_token_counts)
        _emit(f"    Context tokens:   {med_ctx:.0f} median, {max_ctx} max")
        if len(context_token_counts) >= 3:
            growth = context_token_counts[-1] - context_token_counts[0]
            _emit(
                f"    Context growth:   "
                f"{growth:+d} tokens (turn 1 → turn {len(context_token_counts)})"
            )
    # iter-078: trim event rate. Skip the line on sessions where
    # trim never fired (early sessions, or trim threshold so high
    # it's never tripped — the latter is a calibration issue
    # surfaced by ``Context tokens: max`` keeping growing). The
    # ratio of evicted/events surfaces the per-trim severity:
    # evicted/events == 1 means each trim cap was a steady-state
    # one-message eviction; >1 means the trim is catching up
    # after a longer interval (might happen if max_user_assistant
    # was lowered mid-session).
    if trim_events > 0:
        ratio = trim_messages_evicted / trim_events
        _emit(
            f"    Trim events:      "
            f"{trim_events} ({trim_messages_evicted} evicted, "
            f"{ratio:.1f}/event)"
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
