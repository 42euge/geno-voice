# Performance Metrics Taxonomy for geno-voice

A research/brainstorm catalog of metrics worth instrumenting on a local-first
voice agent. The pipeline shape under consideration is:

```
mic → VAD → STT (whisper.cpp) → LLM stream → sentence splitter → TTS (kokoro/piper)
    → persistent speaker → playback (with token-aligned reveal)
```

with cross-cutting threads: `SentenceWorker` (synth + play in background),
`BargeInWatcher` + `BargeInCoordinator` (interrupt mid-stream), filler clips
(masking LLM first-token delay), and primed-frame replay (carrying the user's
first syllables across a turn boundary).

The current `TurnMetrics` (`examples/_chat_metrics.py`) covers gross-grain
latency: `speech_duration`, `stt_time`, `llm_first_token`, `llm_total`,
`tts_time`, `playback_time`, `ttfs`, `total_e2e`, `sentences_spoken`,
`fillers_played`, `barge_in`. That is the floor. Below are ~45 candidate
additions, grouped into three axes.

---

## Category 1 — Standard voice-agent metrics

Things that show up in Vapi, Deepgram, OpenAI Realtime, LiveKit Agents,
Pipecat dashboards. If we want to compare numbers with a hosted SaaS, we need
these.

### 1.1 Speech-to-speech latency (S2S)
- One-sentence: time from user's last audible sample to bot's first audible sample.
- Where: `_chat_loop.run_one_turn` already computes `ttfs = first_audio_at - speech_ended_at`. Promote and rename for industry-comparable terminology.
- Why: the canonical voice-agent number; everything else is a sub-budget.
- How: `worker.first_audio_at - speech_ended_at` (already done; just rename in summary printout).

### 1.2 End-of-turn (EoT) detection latency
- One-sentence: elapsed time between the user's true last speech sample and the VAD declaring `DONE_OK`.
- Where: `_chat_recording.record_utterance_streaming` — instrument inside the `VadEvent.DONE_OK` branch.
- Why: dominates "the agent feels slow" complaints; the user has stopped talking but VAD is still waiting out `silence_duration`.
- How: timestamp the last frame whose `level >= silence_threshold`, subtract from the frame at which `DONE_OK` fires.

### 1.3 VAD trailing-silence wall
- One-sentence: how much of EoT latency is the configured `silence_duration` vs anything else.
- Where: same as 1.2 — emit both EoT total and the budgeted `silence_duration`.
- Why: tells you whether to tune the knob or chase a bug. If wall == config, shorten the knob; if wall >> config, hunt latency in the loop.
- How: `eot_latency - silence_duration`.

### 1.4 VAD false-trigger rate
- One-sentence: fraction of `record_utterance_streaming` calls that produce `DONE_TOO_SHORT` or empty transcripts despite an `ACTIVE` event.
- Where: aggregate in `_chat_metrics.print_session_summary`; bump a counter when `record_utterance_streaming` returns `(b"", 0, 0)`.
- Why: noisy thresholds waste a turn (and may reset the conversation context).
- How: `false_triggers / (false_triggers + completed_turns)`.

### 1.5 VAD missed-speech rate
- One-sentence: fraction of human-judged user utterances the agent failed to start recording.
- Where: offline-only — needs an annotated audio log. Tag in `_chat_recording` debug stream.
- Why: complementary to false-trigger; tuning thresholds always trades these.
- How: scored against a ground-truth tagged session.

### 1.6 Word Error Rate (WER)
- One-sentence: STT transcription error rate against a labeled reference.
- Where: offline batch via `examples/build_dataset.py` style harness; per-turn `metrics.transcript` stored.
- Why: any pipeline metric is meaningless if STT is wrong — every other downstream cost is wasted on the wrong intent.
- How: `jiwer` or `editdistance` against a reference transcript, normalized.

### 1.7 STT real-time factor (RTF)
- One-sentence: `stt_time / speech_duration` — how much faster than realtime your transcriber runs.
- Where: trivially derived from existing `metrics.stt_time / metrics.speech_duration`.
- Why: tells you whether you can safely run STT inline (RTF < 1) or need streaming partial transcription.
- How: divide in the printer.

### 1.8 STT preview-vs-final divergence
- One-sentence: edit distance between the live preview transcript shown during recording and the final transcript.
- Where: `_chat_recording.record_utterance_streaming` already computes a `preview_text` and `final_text`.
- Why: high divergence makes the live preview useless and forces the user to wait for the final to know if they were understood.
- How: `Levenshtein(preview_text, final_text) / max(len(...), 1)`.

### 1.9 LLM tokens-per-second (TPS)
- One-sentence: stream throughput of the LLM in tokens/sec after first token.
- Where: count tokens in the `for token in llm_gen` loop in `_chat_loop`.
- Why: directly gates how fast complete sentences arrive at the worker, which gates when audio actually plays.
- How: `(token_count - 1) / (llm_stream_done_at - first_token_at)`.

### 1.10 LLM time-to-first-sentence (TTFsent)
- One-sentence: time from LLM stream open to the first complete sentence reaching `worker.submit`.
- Where: `_chat_loop` — sample `clock()` after the first `worker.submit(sentence)`.
- Why: TTS can't start until a complete sentence arrives; this is the real latency floor for first audio, distinct from `llm_first_token`.
- How: `first_submit_at - llm_start`.

### 1.11 TTS real-time factor
- One-sentence: synth time per output audio second.
- Where: `SentenceWorker._run` already tracks `tts_time` and audio length is `len(audio_np)/rate`.
- Why: TTS RTF > 1 means TTS becomes the bottleneck and streaming overlap can't save you.
- How: `tts_time / sum(len(audio_np)/rate)` per turn.

### 1.12 Turn-taking jitter
- One-sentence: stddev of the gap between user end-of-speech and bot first audio across a session.
- Where: `_chat_metrics.print_session_summary`, computed over the per-turn `ttfs` list.
- Why: humans tolerate slow turn-taking better than inconsistent turn-taking; jitter is what feels "uncanny".
- How: `statistics.stdev(ttfs_times)`.

### 1.13 Bot speaking rate (words-per-minute)
- One-sentence: actual delivered WPM during bot speech.
- Where: `_chat_playback.play_aligned` has tokens with `start` times; count words / total audio seconds.
- Why: too fast = user can't follow; too slow = user interrupts. Voice-agent UX research clusters around 150-180 WPM.
- How: `len([t for t in tokens if not _is_punct_only(t)]) / (audio_duration / 60)`.

### 1.14 User speaking rate
- One-sentence: WPM measured on the user side.
- Where: `_chat_loop` — derive from `len(metrics.transcript.split()) / metrics.speech_duration * 60`.
- Why: lets you adapt bot WPM to match user (mirroring effect → higher rapport, lower interruption rate).
- How: see above.

### 1.15 Turn count / session length
- One-sentence: number of completed turns and total session wall-clock.
- Where: already implicit in `print_session_summary`'s `len(metrics_list)`; expose duration too.
- Why: baseline denominator for any rate metric.
- How: `time.monotonic()` at session start, subtract at exit.

### 1.16 Error rate (per stage)
- One-sentence: fraction of turns where STT, LLM, TTS, or playback raised.
- Where: each subsystem already accumulates errors (`worker.errors`, `had_error` in `TurnResult`); aggregate per stage.
- Why: tells you which subsystem is your reliability bottleneck.
- How: counters per `errors` list; divide by turns.

### 1.17 Audio device underrun / overrun count
- One-sentence: number of times the speaker buffer underran or the mic overflowed.
- Where: `_chat_playback.play_aligned` + `_chat_recording` `exception_on_overflow` already swallowed today.
- Why: underruns produce audible pops; overruns drop user audio.
- How: surface PyAudio's overflow flag instead of suppressing, count.

### 1.18 Interruption rate
- One-sentence: fraction of bot turns where the user barges in.
- Where: `_chat_metrics` already tracks `barge_in: bool` per turn.
- Why: high interruption = bot is too verbose, slow, or wrong; it's the single most informative UX signal.
- How: `sum(m.barge_in) / len(metrics_list)`.

### 1.19 Bargeable-time fraction
- One-sentence: portion of bot speech during which a barge-in is even *possible* (watcher is running).
- Where: `BargeInWatcher` lifetime spans LLM-stream + playback; track its active wall-clock vs total turn time.
- Why: if the watcher is only up 60% of the turn, the bot is functionally uninterruptible the rest of the time.
- How: `(watcher.stop_at - watcher.start_at) / metrics.total_e2e`.

### 1.20 Cold-start (first turn) latency penalty
- One-sentence: extra latency on turn 1 vs steady-state median.
- Where: `print_session_summary` — separate `metrics_list[0]` from `metrics_list[1:]`.
- Why: first-turn TTFS includes model loading, speaker open, possibly TTS warmup; users judge the whole session by it.
- How: `metrics_list[0].ttfs - median(m.ttfs for m in metrics_list[1:])`.

---

## Category 2 — Architecture-specific metrics

These exploit knowledge of the actual `geno-voice` design (streaming overlap,
fillers, persistent speaker, primed-frame replay, sentence splitter). They
would be meaningless or trivially zero in a synchronous pipeline.

### 2.1 Streaming overlap ratio
- One-sentence: fraction of LLM-stream wall-clock during which TTS or playback was concurrently active.
- Where: `_chat_loop` knows `llm_start`, `llm_stream_done_at`, `worker.first_audio_at`; the worker knows when synth/play actually run.
- Why: the whole point of `SentenceWorker` is to run TTS in parallel with token receipt. If overlap is 0, the worker is just adding latency.
- How: instrument intervals in worker (`synth_at`, `play_at`); compute `union_overlap_with([llm_start, llm_stream_done_at])`.

### 2.2 First-sentence overlap savings
- One-sentence: how much of `tts_time` for the first sentence happened *before* `llm_stream_done_at` — i.e. masked by ongoing LLM streaming.
- Where: `SentenceWorker._run` per-sentence timestamps.
- Why: this is the user-perceived win of streaming sentence dispatch. If the first synth happens entirely after LLM completion, you bought nothing.
- How: `max(0, min(tts_done, llm_done) - max(tts_start, llm_start))`.

### 2.3 Filler-mask success rate
- One-sentence: of the turns that played a filler, fraction where the filler audio actually overlapped the LLM first-token wait.
- Where: `SentenceWorker._run` already plays the filler when `Empty` fires after `idle_threshold`; correlate with `first_token_at`.
- Why: a filler that finishes playing before the LLM responds was wasted; a filler that gets cut off the moment the first sentence arrives is ideal.
- How: success if `filler_play_end >= first_token_at` (we masked the gap) AND `filler_play_start < first_token_at` (we masked something real).

### 2.4 Filler false-positive rate
- One-sentence: fraction of fillers played even though the LLM responded faster than `idle_threshold` would have predicted.
- Where: `SentenceWorker` — compare `idle_threshold` configured vs actual `first_token_at - llm_start`.
- Why: false-positive fillers make the bot sound disfluent for no reason. Tune `idle_threshold` per LLM endpoint.
- How: count when `fillers_played > 0` and `metrics.llm_first_token < idle_threshold`.

### 2.5 Sentence-split coverage
- One-sentence: fraction of LLM tokens delivered to the worker as part of a complete sentence vs flushed as the trailing remainder.
- Where: `_chat_loop` — count tokens going through `split_complete_sentences` vs the final `remaining = token_buffer.strip(); worker.submit(remaining)`.
- Why: the trailing remainder forces a synth at the end of the turn, which can't overlap with anything; high remainder = LLM not producing punctuation = lost overlap.
- How: track `bytes_in_complete_sentences / total_tokens_chars`.

### 2.6 Sentence-split fragmentation
- One-sentence: average sentence length (in tokens) submitted to the worker.
- Where: `_chat_loop` — log `len(sentence)` per `worker.submit`.
- Why: very short sentences ("Yes.") synth fast but lose overlap because TTS finishes before the next sentence arrives. Very long sentences increase TTFS.
- How: histogram sentence lengths.

### 2.7 Worker queue depth (peak / average)
- One-sentence: how many sentences are queued waiting for synth at any moment.
- Where: `SentenceWorker._queue.qsize()` polled.
- Why: peak > 1 means the LLM is outpacing TTS — your TTS is the bottleneck. Peak ≈ 0 means TTS is starved (which is fine, but means streaming doesn't help).
- How: sample `_queue.qsize()` from a daemon at fixed interval.

### 2.8 Speaker-open overhead per turn
- One-sentence: time spent inside `speaker_factory()` during turn vs total turn time.
- Where: `SentenceWorker._run` opens speaker once at start; instrument `speaker_factory()` call.
- Why: persistent-speaker is one of the iter-008 wins; if the open cost is creeping back (driver / Bluetooth / SDL), TTFS regresses silently.
- How: `t_after_open - t_before_open`.

### 2.9 Persistent-speaker open count per session
- One-sentence: number of distinct speaker streams opened across the session.
- Where: counter in `_chat_loop`.
- Why: if the worker is being recreated per turn (it is, today) the per-turn open cost still hits N times. Tells you whether to lift the speaker to session scope.
- How: `+= 1` on each `speaker_factory()` call.

### 2.10 Barge-in latency
- One-sentence: time from `BargeInWatcher` ACTIVE event to `play_aligned` exiting via `cancel_event`.
- Where: `BargeInCoordinator.triggered_at` is already there; add a `playback_stopped_at` in the worker.
- Why: barge-in latency > ~200ms is the moment the user thinks the bot is ignoring them. The whole barge-in feature lives or dies on this.
- How: `playback_stopped_at - triggered_at`.

### 2.11 Barge-in phase distribution
- One-sentence: histogram of barge-ins by phase — LLM-stream phase vs playback phase.
- Where: already computed as the diagnostic string in `_chat_loop` (`"LLM-stream phase" or "playback phase"`); just emit as metric.
- Why: barge-ins during LLM stream mean the user is interrupting a *silent* bot — likely impatient with TTFS. Playback-phase = interrupting bot speech = different cause (verbose, wrong, etc.).
- How: counter per branch.

### 2.12 Primed-frames replay duration
- One-sentence: audio seconds carried over via `next_primed_frames` into the next turn.
- Where: `_chat_loop` already prints `len(next_primed) * chunk / rate`; promote to metric.
- Why: tells you how much of the user's first words would have been lost without the watcher's frame buffer. Validates iter-025 lead-in.
- How: `len(next_primed) * CHUNK / RATE`.

### 2.13 Primed-frames STT contribution
- One-sentence: WER on barged-in turns with primed frames vs without (A/B).
- Where: offline ablation comparing `primed_frames=None` vs current behavior.
- Why: directly validates whether the priming buys transcription quality or just feel-good seconds of audio.
- How: rerun a session twice, compare per-turn WER on `barge_in=True` turns.

### 2.14 LLM stream cancel-to-close
- One-sentence: time between `coord.trigger()` and the underlying HTTP stream actually closing.
- Where: `_chat_loop` `finally` block calls `llm_gen.close()`; instrument before/after.
- Why: hanging HTTP streams hold sockets and waste tokens we paid for. Also blocks the next turn if not actually closed.
- How: `t_after_close - coord.triggered_at`.

### 2.15 Worker error-recovery success
- One-sentence: fraction of turns where `worker.errors` is non-empty but the turn still produced audio.
- Where: `worker.errors` + `metrics.ttfs > 0` already available.
- Why: silent partial degradation — a sentence failed to synth but the rest played — masks bugs.
- How: `count(errors and ttfs > 0) / count(errors)`.

### 2.16 Sentence-worker idle gap
- One-sentence: time the worker spent blocked on `_queue.get()` between sentences.
- Where: `SentenceWorker._run` — instrument time before/after `self._queue.get(...)`.
- Why: gaps mean the LLM didn't produce a complete sentence in time; if this is large, sentence-splitter is too greedy or LLM is too slow. If 0, streaming is fully utilizing TTS.
- How: cumulative `t_after_get - t_before_get` excluding the first.

### 2.17 Token-reveal lag
- One-sentence: average wall-clock delay between an audio sample playing and its corresponding token being printed to terminal.
- Where: `play_aligned` already aligns by `tokens[idx]["start"] <= pos`; instrument that boundary.
- Why: if reveal lags audio, the on-screen text falls behind voice and the UX feels broken. If reveal *leads* audio, it spoils the bot.
- How: per token, `t_print - (t_audio_chunk_start + token_start)`.

### 2.18 Cancel-event correctness rate
- One-sentence: fraction of barge-ins where `cancel_event` actually halted playback mid-sentence (vs the play loop completing naturally first).
- Where: `play_aligned` — flag whether exit was via `cancel_event.is_set()` vs `samples_played >= total_samples`.
- Why: validates iter-009 / iter-026 cancel plumbing and iter-023 signature detection.
- How: counter per exit branch.

### 2.19 Mic flush stale-frame count
- One-sentence: number of frames `flush_pending_audio` discards at start of each turn.
- Where: `_chat_loop` calls it after `worker.start()`; the helper already returns the drained count.
- Why: many stale frames = bot audio leaked through OS mic / loopback / Bluetooth duplex; tells you when echo cancellation is needed.
- How: aggregate `drained` per turn.

### 2.20 Loopback / acoustic-echo barge-in rate
- One-sentence: barge-ins triggered by the bot's own audio (false positives from speaker → mic).
- Where: cross-correlate `BargeInWatcher.events` timestamps with `play_aligned` audio chunks playing on the same machine.
- Why: a recurring failure mode in headphone-less laptop use. If bot audio is detected as speech, the bot interrupts itself.
- How: barge-ins where `triggered_at` falls within bot-audio-active interval, classified offline by listening to mic capture.

### 2.21 VAD-config consistency
- One-sentence: whether the `BargeInWatcher` VAD config matches the `record_utterance_streaming` VAD config.
- Where: `_chat_loop` — assertion / metric on `chat.vad.silence_threshold` plumbed identically. iter-028 fixed this once; track it as a regression sentinel.
- Why: drift here causes noisy-room false barges or quiet-room missed barges only on the watcher side.
- How: hash the VAD config struct; alert on mismatch.

### 2.22 Sentence first-audio path length
- One-sentence: number of distinct timestamps between speech end and first audio (mic-end → VAD-eot → STT-done → LLM-1st-token → 1st-sentence-split → synth-done → playback-1st-chunk).
- Where: the chat loop has every one of these timestamps already; expose as a structured trace.
- Why: TTFS is one number, but the breakdown tells you which leg to optimize. Currently you only see `stt_time + llm_first_token + tts_time` summed, which double-counts overlap.
- How: emit a structured timeline event per turn with all 7 timestamps.

### 2.23 Conversation history grow rate
- One-sentence: tokens of context being sent to the LLM per turn.
- Where: `_chat_loop` calls `trim_history` with `max_user_assistant=20`; estimate `sum(len(m["content"].split()) for m in messages)`.
- Why: LLM TTFB scales with input context; if you don't trim aggressively, late-session turns get progressively slower.
- How: count tokens before each LLM call.

### 2.24 Trim event rate
- One-sentence: how often `trim_history` actually evicts messages (vs no-op).
- Where: `_chat_helpers.trim_history` — return a flag when it trimmed.
- Why: validates the trim threshold is calibrated to the typical session.
- How: counter.

---

## Category 3 — Novel / speculative metrics

Things I haven't seen elsewhere but seem worth measuring once. Cheap to
instrument; the question is whether they correlate with anything.

### 3.1 Naturalness gap
- One-sentence: gap between user speech-end and bot first audio, with the human-conversation sweet spot of 200-400ms.
- Where: same as `ttfs`, but bucket into `< 200ms` (rushed/preempt), `200-400ms` (natural), `> 400ms` (slow).
- Why: humans don't optimize for minimum latency — they optimize for natural pause. Sub-200ms feels robotic / interrupting; > 400ms feels laggy. Most voice agents report only "lower is better".
- How: histogram TTFS into the three buckets.

### 3.2 Conversation rhythm score
- One-sentence: 1 - (stddev of TTFS / median TTFS) — how consistent the bot's response cadence is across a session.
- Where: `_chat_metrics.print_session_summary`.
- Why: consistency feels like a "personality"; jitter feels like a "system". A slow-but-steady bot can outperform a fast-but-jittery one in user trust.
- How: `1 - stdev(ttfs) / max(median(ttfs), 1e-6)`.

### 3.3 Thinking-out-loud opportunity
- One-sentence: counterfactual — could TTS have started earlier if the sentence splitter was less greedy?
- Where: simulate alternate split strategies (split-on-comma, fixed-token-window) over recorded LLM token streams.
- Why: today the splitter waits for a sentence terminator. With a more aggressive policy, first audio could arrive earlier — at the cost of a more disfluent prosody.
- How: replay the token stream with different splitters offline, measure simulated TTFS delta.

### 3.4 Regret rate
- One-sentence: fraction of barge-ins where the user starts speaking within 200ms of bot first audio.
- Where: `_chat_loop` — compare `coord.triggered_at` against `worker.first_audio_at`.
- Why: a barge-in this fast usually means the user was already mid-utterance and the bot pre-empted them. It implies the bot misjudged end-of-turn (1.2 fired too early).
- How: count when `triggered_at - first_audio_at < 0.2`.

### 3.5 Bot-self-interrupt rate
- One-sentence: barge-ins triggered while bot audio amplitude is the dominant signal (likely echo).
- Where: same instrumentation as 2.20 but with a real-time correlation rather than offline.
- Why: distinguishes "user impatient" barge-ins from "bot heard itself" barge-ins. Drives the decision to ship AEC.
- How: classify each barge-in by RMS ratio of mic input vs known bot audio at `triggered_at`.

### 3.6 Interruption recovery quality
- One-sentence: did the bot's next-turn response actually address the interrupting user message?
- Where: offline LLM-judge evaluation over barge-in turn pairs.
- Why: a barge-in is only "handled" if the bot pivots. A bot that ignores the interruption and continues its previous line is failing silently.
- How: LLM-as-judge on `(barged_response, next_user_turn, next_bot_turn)`.

### 3.7 Pre-empted-content loss
- One-sentence: number of bot tokens generated by the LLM but never spoken because of a barge-in.
- Where: `_chat_loop` — `len(metrics.response.split())` vs `worker.sentences_spoken`'s aggregate token count.
- Why: signals the bot is being verbose: large pre-empted-content = bot generated a lot the user didn't want.
- How: `total_tokens_generated - tokens_actually_played`.

### 3.8 Filler novelty index
- One-sentence: distinct fillers used across the session vs total fillers played.
- Where: `SentenceWorker.fillers_played` + which filler was picked.
- Why: hearing the same "umm" three times in a row is worse than no filler. Measures whether the picker is actually distributing.
- How: `unique_fillers / total_fillers_played`.

### 3.9 LLM "verbosity vs latency" tradeoff
- One-sentence: per-turn correlation between response length and total turn latency.
- Where: `_chat_metrics` already has `len(response)` and `total_e2e`.
- Why: identifies when prompt engineering for terseness would help latency more than infra optimization. Sometimes "say less" is the cheapest perf fix.
- How: scatter `len(response)` vs `total_e2e`; report Pearson r.

### 3.10 First-syllable preservation rate
- One-sentence: fraction of barged turns where the next-turn STT contains the user's first word, validated against held-out audio.
- Where: combine `next_primed_frames` capture with full-session mic dump.
- Why: tests whether the lead-in buffer (iter-025) actually rescues the syllable it's supposed to.
- How: align dumped mic at `triggered_at - lead_in_chunks * dt` with eventual STT transcript; check first word coincidence.

### 3.11 Silent-turn rate
- One-sentence: turns where the user said something but the bot produced no audio (worker errors, LLM empty, all sentences pre-empted).
- Where: `metrics.ttfs == 0` after a successful STT.
- Why: invisible failure mode — the user thinks the bot is broken but no error fires.
- How: counter where `transcript and ttfs == 0`.

### 3.12 Pause-followed-by-speech (pause-recovery)
- One-sentence: fraction of `DONE_TOO_SHORT` events that are followed by a successful turn within 2s — i.e. user paused mid-thought.
- Where: `_chat_recording` already returns `(b"", 0, 0)` on too-short; track timestamps.
- Why: tells you whether `min_speech_duration` is set too aggressively: if many too-shorts are followed by real speech, the user is being clipped.
- How: window-pair `too_short` events with the next non-empty turn.

### 3.13 Adaptive-rate margin
- One-sentence: ratio of bot WPM to user WPM, per turn.
- Where: 1.13 / 1.14 as raw inputs.
- Why: hypothesis — ratio in [0.8, 1.2] correlates with lower interruption rate. If true, you can adapt TTS speed.
- How: `bot_wpm / user_wpm`.

### 3.14 Time-to-comprehension (TTC) proxy
- One-sentence: time from bot first audio to first user token of the next turn — i.e. how long the user listened before responding.
- Where: `_chat_loop` between turns.
- Why: very short TTC = user already knew the answer (bot underperformed). Very long TTC = user is confused. Both are signals.
- How: `next_turn_speech_start - current_turn_first_audio`.

### 3.15 Energy-per-turn
- One-sentence: rough Joules consumed per turn (CPU + GPU/Metal time) — for offline-only deployments where battery matters.
- Where: wrap each subsystem (`stt`, `llm`, `tts`) with `psutil.cpu_times()` deltas; on Apple Silicon, `powermetrics` sampling.
- Why: a local-first agent's selling point is privacy *and* offline capability; battery is the second axis.
- How: subsystem CPU-second × measured CPU-Joules constant.

### 3.16 Cross-modal consistency
- One-sentence: did the rendered terminal text match what the speaker actually played?
- Where: `play_aligned` — with `cancel_event`, text emission stops with audio (iter-026). Verify by capturing both.
- Why: regression sentinel for the "audio cuts but text keeps printing" bug.
- How: ratio of tokens emitted to terminal vs tokens whose `start` was below the actual `samples_played` at exit.

### 3.17 Conversation-end ergonomics
- One-sentence: did the user end with KeyboardInterrupt during a bot turn (ungraceful) or during silence (graceful)?
- Where: `mic_chat.run_chat`'s top-level handler.
- Why: high "ungraceful" rate means the bot doesn't yield naturally; users feel they have to fight it to leave.
- How: classify SIGINT timestamp against current pipeline phase.

### 3.18 First-token-to-audio gap (FT-A)
- One-sentence: time between LLM first token and first audio out — measures the "have tokens, can't speak yet" window.
- Where: `_chat_loop` — `worker.first_audio_at - first_token_at`.
- Why: complementary to TTFS. If FT-A is large, sentence-split + TTS is the bottleneck; if `llm_first_token` is large, LLM is the bottleneck. Lets you target investment.
- How: subtract.

### 3.19 Sub-second turn rate
- One-sentence: fraction of turns where TTFS < 1.0s.
- Where: `_chat_metrics.print_session_summary`.
- Why: a single human-feel threshold across the session, easier to track than median.
- How: `sum(1 for m in metrics_list if 0 < m.ttfs < 1.0) / len(metrics_list)`.

### 3.20 Barge-in regret-recovery half-life
- One-sentence: after a "regret" barge-in (3.4), how long until subsequent turns return to baseline TTFS.
- Where: per-turn TTFS series + barge-in markers.
- Why: tests whether the agent "learns" or panics after an interruption — a single regret event shouldn't cascade.
- How: fit exponential decay on `|ttfs[t] - baseline|` post barge-in.

### 3.21 LLM-stall recoverability
- One-sentence: fraction of LLM stalls (gap > N seconds between tokens) that resolve before barge-in fires.
- Where: instrument inter-token gaps in the `for token in llm_gen` loop.
- Why: tells you whether `idle_threshold` for fillers is well-tuned, *and* whether mid-stream LLM stalls are a thing for your endpoint.
- How: per-turn max `t_token[i+1] - t_token[i]`.

### 3.22 Phantom-sentence rate
- One-sentence: sentences submitted to the worker but containing only filler words / acknowledgments ("Sure.", "Okay.") that could have been a filler clip.
- Where: classify tokens of each `sentence` before `worker.submit`.
- Why: a real LLM sentence costs TTS time; a pre-rendered filler clip is free. Detecting these tells you whether prompt tuning could push them into filler territory.
- How: regex / classifier over sentence text vs filler corpus.

---

## Wiring notes

Most of these can be added without changing the public API of `TurnMetrics`:

1. Extend the dataclass with optional new fields (default `None` / `0`).
2. Instrument the relevant module — `_chat_loop`, `_chat_pipeline`,
   `_chat_recording`, `_chat_playback` — populating fields where the
   timestamps already exist.
3. Aggregate in `print_session_summary`, with statisitc choices spelled
   out (median for skewed latencies, mean for rates, stdev for jitter).
4. For offline-only metrics (WER, false-trigger, regret-recovery), add a
   structured-event JSON dump per turn so a separate analysis script can
   compute them later without slowing down the live loop.

The cheapest 10x value-per-line subset, if forced to pick: 1.1, 1.2, 1.3,
1.9, 1.10, 1.18, 2.1, 2.3, 2.10, 2.22, 3.1, 3.4. Those are tightly aligned
with both standard practice and this architecture's distinctive parts.
