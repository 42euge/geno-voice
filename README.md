# geno-voice

Local voice pipeline for offline, privacy-first AI voice interaction.

## What

A reusable STT/TTS stack that runs entirely on-device. No cloud APIs, no data leaving the machine.

Used by [geno-reflect](https://github.com/42euge/geno-reflect) and other geno-* projects that need voice interaction.

## Components

- **STT (Speech-to-Text):** Whisper.cpp — local transcription
- **TTS (Text-to-Speech):** Kokoro / Piper — local speech synthesis
- **VAD (Voice Activity Detection):** Silero VAD — detect when the user starts/stops speaking

## Installation

Install via geno-tools:

```bash
geno-tools install geno-voice
```

Or from within an agent session:

```
/geno-tools install geno-voice
```

## Evaluating a new STT engine

`scripts/run_stt_benchmark.py` runs any registered `STTEngine` against
the WER fixture corpus (`tests/fixtures/wer/`) and reports per-fixture
pass/fail against recorded WER bands. The corpus has 5 audio fixtures
covering common failure modes: clean speech, pangram, noise, heavy
noise, and multi-speaker cross-talk.

### Quick benchmark

```bash
python scripts/run_stt_benchmark.py --engine faster_whisper --model tiny
```

```
clean_audio               PASS  WER 0.20  band [0.00, 0.40]  elapsed 0.29s
quick_brown_fox_audio     PASS  WER 0.11  band [0.00, 0.30]  elapsed 0.24s
noisy_audio               PASS  WER 0.20  band [0.10, 0.50]  elapsed 0.26s
catastrophic_audio        PASS  WER 1.00  band [0.80, 1.30]  elapsed 0.26s
multispeaker_audio        PASS  WER 0.80  band [0.60, 1.10]  elapsed 0.26s

5/5 fixtures passed in 1.3s
```

### Saving a baseline + diffing changes

The benchmark supports machine-readable output (`--format json|csv`)
and a `--diff` mode for comparing against a saved baseline:

```bash
# Save baseline before changes
python scripts/run_stt_benchmark.py --engine faster_whisper \
    --format json > baseline.json

# ...iterate on engine code...

# Compare against baseline
python scripts/run_stt_benchmark.py --engine faster_whisper \
    --diff baseline.json
```

```
clean_audio               PASS   WER 0.20 -> 0.20  Δ +0.000
noisy_audio               PASS   WER 0.20 -> 0.25  Δ +0.050
multispeaker_audio        FAIL   WER 0.80 -> 1.20  Δ +0.400 (regressed)

4/5 → 5/5 fixtures passing (+1)
Improvements: noisy_audio
Regressions: multispeaker_audio
```

### CI integration

The cleanest gate is `--fail-on-regression`: the process exits
non-zero only when a fixture that **passed** in the baseline now
**fails**. Pre-existing failures don't block — a PR is allowed
through as long as it leaves the corpus no worse than it found it.

```bash
# Exits 1 iff something regressed; no jq/grep plumbing needed.
python scripts/run_stt_benchmark.py --engine faster_whisper \
    --diff baseline.json --fail-on-regression
```

The diff report still prints (text/json/csv per `--format`), so
CI logs show exactly what changed. `--fail-on-regression`
requires `--diff` (exit 2 otherwise — there's no baseline to
regress against).

`--fail-on-regression` only sees fixtures that are still in the
corpus, so deleting a failing fixture would slip past it as
"fewer failures". `--fail-on-removed` closes that gap: it exits
non-zero when a fixture present in the baseline is missing from
the current corpus. Combine the two for a strict gate that
blocks a PR which makes a fixture worse **or** drops one:

```bash
# Fail if anything regressed OR a baseline fixture disappeared.
python scripts/run_stt_benchmark.py --engine faster_whisper \
    --diff baseline.json --fail-on-regression --fail-on-removed
```

Like `--fail-on-regression`, `--fail-on-removed` requires
`--diff` (exit 2 otherwise).

For convenience, `scripts/ci-gate.sh` is the committed wrapper
that wires both gates together so a CI step is a single call:

```bash
# Fails (exit 1) if anything regressed OR a baseline fixture was
# dropped; exit 0 otherwise. Defaults: --engine faster_whisper,
# --baseline baseline.json.
scripts/ci-gate.sh --baseline baseline.json
```

Pass `--engine`/`--model` to pick the engine, and forward any
extra benchmark flags after a `--` separator (e.g.
`scripts/ci-gate.sh --baseline baseline.json -- --device cpu`).
A missing baseline exits 2 with the exact command to create one.

If you'd rather gate in a shell pipeline against the JSON dump,
`regression_count` carries the same signal:

```bash
python scripts/run_stt_benchmark.py --engine faster_whisper \
    --diff baseline.json --format json \
    | jq '.regression_count > 0' | grep -q true && exit 1
```

Output formats:
- `text` (default) — human-readable per-row report
- `json` — full result/diff dump (top-level aggregates +
  per-fixture records)
- `csv` — header + one row per fixture, RFC-4180 compliant

### Adding a new STT engine

1. Implement `stt.base.STTEngine` (transcribe a wav blob → text).
2. Register in `stt/__init__.py:ENGINES`.
3. Run `python scripts/run_stt_benchmark.py --engine <your_name>`.
4. If most fixtures pass, the engine is ready for production wiring
   in `mic_chat.py:run_chat`.

The corpus is forgiving on espeak-ng-generated audio. Production-
grade STT (whisper-large, etc.) typically lands at WER 0.0-0.10 on
the clean fixtures.

## Contributing patterns

The voice pipeline has accreted two reusable code patterns. Both are
documented in detail in [`GENO.md`](GENO.md) — read that section
before adding similar code, so a new instance matches the existing
shape (and the doc-sync tests in `tests/unit/` stay green).

- **mic_chat.py extraction pattern** — how to pull a subroutine out of
  `examples/mic_chat.py:run_chat` into its own testable module: inject
  callable dependencies (not engine classes), inject a `log` callable,
  keep ANSI styling at the caller, return a dataclass, and lazy-import
  platform deps inside closures. See `GENO.md` →
  *mic_chat.py extraction pattern*.

- **Session-summary diversity-check pattern** — how to add a
  session-summary warning that fires when N+ consecutive turns share
  the same problematic metric value (e.g. a run of rushed-sounding
  turns, or low LLM-stream/synth overlap). Filter uninteresting values
  first, reuse the `_longest_consecutive_run` primitive, pick a
  per-signal threshold, and name the responsible iteration in the
  warning text. See `GENO.md` → *Session-summary diversity-check
  pattern*.

Both sections are guarded by drift-sentinel tests
(`tests/unit/test_extraction_pattern_doc.py`,
`tests/unit/test_diversity_pattern_doc.py`) that fail if a new
instance lands without a matching doc update.

## Research

Longer-horizon design exploration lives under [`docs/research/`](docs/research/):

- **[Organic / full-duplex turn-taking](docs/research/organic-turn-taking.md)** —
  a living research doc on moving geno-voice beyond rigid half-duplex
  ("you speak, it waits, it replies") toward backchannels, semantic
  end-of-turn, utterance queueing, and barge-in. Surveys the SOTA
  (Moshi, pipecat `smart-turn`, LiveKit `turn-detector`, Krisp) with a
  fit assessment for our pipeline, and carries a prioritized backlog
  that subsequent laps work through. Shipped so far: the rule-based
  backchannel/continuer classifier (`session/backchannel.py`,
  `tests/unit/test_backchannel.py`) and the **turn-decider seam**
  (`session/turn_decider.py`, `tests/unit/test_turn_decider.py`) —
  a swappable silence→confidence mapper that un-hardcodes
  `pipecat_server.py`'s `smart_turn_confidence` (the old literal `0.5`
  sat below the engine's backchannel threshold, leaving the
  silence-driven turn tiers dead) and is the drop-in interface a future
  audio `smart-turn` model implements. Also shipped: the **rule-based
  text EOU precursor** (`session/text_eou.py`,
  `tests/unit/test_text_eou.py`) — `utterance_completeness(text)` returns
  a (0.0, 1.0] multiplier that *dampens* the silence confidence when the
  transcript trails off on a conjunction / dangling preposition / filler
  / ellipsis, so the engine stays silent through a mid-thought pause it
  would otherwise treat as a turn-end. `TextAwareTurnDecider` folds it
  into the silence seam behind the identical `confidence(...)` interface
  (a complete utterance multiplies by 1.0 — a conservative, monotone
  refinement), and `pipecat_server.py` now uses it. Also shipped: the
  **full-duplex config flag scaffolding** (`session/full_duplex.py`,
  `tests/unit/test_full_duplex.py`) — an off-by-default `FullDuplexConfig`
  gate (`GENO_FULL_DUPLEX` master flag + per-behavior overrides) so future
  organic behaviors (continuer-aware barge-in, agent backchannels) land
  behind a switch and the proven half-duplex path is never regressed; a
  default config is byte-for-byte today's behavior. Also shipped: the
  **continuer-aware barge-in decision** (`session/barge_decision.py`,
  `tests/unit/test_barge_decision.py`) — `decide_barge_action(transcript,
  energy, *, config)` composes the backchannel classifier and the
  full-duplex gate into an `ABANDON` (true interruption) / `FINISH` (user
  only backchanneled) verdict, so a "mhmm" during agent speech keeps the
  agent talking instead of clipping its own sentence. With a default
  (half-duplex) config it returns `ABANDON` for every transcript —
  byte-for-byte today's "any barge cancels" behavior; only with
  `continuer_aware_listening_active()` does a recognized continuer
  `FINISH`. Also shipped: the **agent backchannel emission timing**
  decision (`session/backchannel_timing.py`,
  `tests/unit/test_backchannel_timing.py`) — the *emit* half of
  backchanneling that complements the classifier's *recognize* half.
  `decide_backchannel_timing(*, user_speaking_secs, pause_secs,
  secs_since_last_backchannel, config, timing)` returns `EMIT` / `HOLD`,
  deciding when the agent should "mhmm" *during* a user monologue (a
  clause-boundary pause `[0.3, 2.0)s`, past a warm-up, not rate-limited),
  not only on trailing silence. Its `max_pause` of 2.0s is exactly
  `turn_decider`'s `silence_floor`, so the mid-speech window and the
  silence-driven `PLAY_CUE` window partition the silence axis with no
  overlap. Gated behind `agent_backchannels_active()`: with a default
  config it always `HOLD`s, byte-for-byte today's "agent stays silent
  during user speech." Its **stateful driver** is
  `session/backchannel_monitor.py` (`tests/unit/test_backchannel_monitor.py`):
  `BackchannelMonitor.observe(*, now, monologue_start_at, pause_secs)`
  derives `user_speaking_secs` and `secs_since_last_backchannel` and
  routes them through the seam, recording the emit timestamp *iff* the
  decision is `EMIT` so the `min_between_cues_secs` rate limit engages
  across calls — the one piece of cross-event state the pure seam can't
  carry (without it the agent would re-emit "mhmm" on every qualifying
  pause frame). It also owns the **second** piece of cross-event state the
  seam can't carry: its position in the shared cue rotation
  (`session/cue_rotation.py`, `tests/unit/test_cue_rotation.py`) — on an
  emit `observe` returns `BackchannelDecision.cue_type`, advancing through
  `CUE_ROTATION` ("mhmm" → "i see" → "right" → ...) so consecutive
  backchannels don't repeat one sound; a held frame never burns a rotation
  slot. `CUE_ROTATION` is now the single source of truth shared with
  `TurnTakingEngine`'s silence-driven `PLAY_CUE` path (it used to live in
  `turn_taking.py`), so the two cue paths can't drift apart. `reset()`
  clears the rate limit for a fresh session but **keeps** the rotation
  position (a new monologue continues the rotation rather than always
  replaying "mhmm"). Mirrors the `UtteranceBuffer` / `UtteranceAggregator`
  driver relationship to the merge seam; default (half-duplex) config ⇒
  `emit=False` always and state never mutates. The **silence-driven**
  `PLAY_CUE` path is now wired to the same rotation too: `plan_cue_broadcast(decision,
  *, trigger_fired)` (`session/turn_taking.py`,
  `tests/unit/test_plan_cue_broadcast.py`) is the pure gate
  `pipecat_server.py`'s `Broadcaster` consults after `TurnTakingEngine.decide`
  — it returns the engine's **rotated** `decision.cue.cue_type` to broadcast,
  or `None` when the action isn't `PLAY_CUE` / an NLP trigger fired / no cue is
  attached. This fixed two live-path bugs (iter-172): the branch read
  `Action.play_cue` (an `AttributeError` — the member is `PLAY_CUE`, so the cue
  branch crashed on every transcript), and `broadcast_cue` re-picked a *random*
  cue from a private `CUE_TYPES` list (dropping the rotation and risking cue
  keys outside it). Now the live broadcast is indexed to the single
  `CUE_ROTATION` source of truth. Also shipped: the **organic-path naturalness
  metrics** (`examples/_chat_metrics.py`,
  `tests/unit/test_emit_organic_block.py`) — two additive `TurnMetrics`
  fields, `false_endpoint` (the EOU decision fired early and the user
  had more to say) and `continuers_detected` (user backchannels that
  correctly held the agent's floor), surfaced per-turn and as a session
  summary "Organic turn-taking" block (false-endpoint rate + continuers
  held). Both default off and stay zero on the half-duplex path, so the
  block is fully suppressed and today's summaries are byte-for-byte
  unchanged; the track becomes *measured, not asserted* once the seams
  are wired in. Also shipped: the **utterance buffer-merge decision**
  (`session/utterance_merging.py`,
  `tests/unit/test_utterance_merging.py`) — the *user*-side half of
  utterance queueing that complements the agent-side abandon-vs-finish
  decision. `decide_utterance_continuation(prev_text, next_text,
  gap_secs, *, config)` returns `MERGE` / `NEW`: when a silence endpoint
  fires on *unfinished*-looking text (`text_eou` completeness ≤ 0.6) and
  a continuation arrives within the pause window (gap ≤ 2.0s, the
  `turn_decider` silence-floor), the prior endpoint was a **false
  positive** — the user paused mid-thought ("…about the [pause]
  …deadline") — so the two are glued into one turn. Both gates must hold;
  a quick gap after a complete sentence or an unfinished prior after a
  long gap both stay `NEW`. Gated behind `utterance_merging_active()`
  (the fourth `FullDuplexConfig` sub-flag,
  `GENO_FULL_DUPLEX_UTTERANCE_MERGING`): with a default config it returns
  `NEW` for every input, byte-for-byte today's "each endpoint is its own
  turn" behavior. This is the decision that *avoids* the false endpoints
  the iter-154 `false_endpoint` metric *counts*. Also shipped: the
  **stateful utterance buffer** (`session/utterance_buffer.py`,
  `tests/unit/test_utterance_buffer.py`) — the live-loop driver that wraps
  the stateless merge decision in the hold-and-merge *state* an STT loop
  needs. `UtteranceBuffer.offer(text, gap_secs)` holds an
  unfinished-looking utterance, merges a quick continuation onto it, and
  emits finished turns as `EmittedTurn(text, false_endpoint)`; `flush()`
  releases a held pending when no continuation arrives. Each merged turn
  carries `false_endpoint=True` so the caller sets `TurnMetrics.false_endpoint`
  and iter-154's metric finally populates from the live path. With a default
  config the buffer is a **transparent zero-latency passthrough** — every
  `offer` emits immediately, nothing is ever held, `flush` is always empty —
  so the proven half-duplex path is byte-for-byte unchanged. A
  `max_merge_depth` cap (iter-157, default 8) bounds how many continuations a
  held pending may absorb before it is force-emitted — a backstop so a
  pathological unfinished-forever STT stream can't starve the engine; it sits
  well above any realistic conversation, so it never fires in practice. Built
  on top: the **cross-turn aggregator** (`session/utterance_aggregator.py`,
  `tests/unit/test_utterance_aggregator.py`, iter-158) — the buffer's
  `offer(text, gap_secs)` needs the inter-utterance silence gap, which the
  buffer deliberately never measures (no clock reads). `UtteranceAggregator`
  owns the one scalar of state the buffer can't — the previous utterance's
  endpoint timestamp — and `offer(text, speech_start_at, speech_end_at)`
  derives `gap_secs` from the speech timestamps the recorder already surfaces,
  routes through the buffer, and returns an `AggregatedResult` (turns, held,
  and the measured `gap_secs`). This keeps the eventual live STT-loop wiring a
  thin driver: it hands the aggregator the two timestamps and feeds the
  returned turns to the engine. With a default config the aggregator is the
  same transparent passthrough as the buffer beneath it. **Wired into the live
  loop** (`examples/_chat_loop.py`, `tests/unit/test_chat_loop_aggregator.py`,
  iter-159) behind an off-by-default `ChatLoop(aggregator=...)` seam:
  `mic_chat.run_chat` builds one from `full_duplex_config_from_env()`, and per
  finalized utterance `run_one_turn` offers the transcript + the recorder's
  speech timestamps. `examples/_chat_aggregation.py::resolve_turn` folds the
  variable-length `AggregatedResult` into one decision for the single-turn loop
  — a *held* utterance re-listens (no LLM stream for half a thought); a
  *released* (possibly merged) turn is responded to and sets
  `TurnMetrics.false_endpoint`, populating iter-154's false-endpoint metric from
  the live path. With `aggregator=None` (default) or the `GENO_FULL_DUPLEX*`
  flags unset, the path is byte-for-byte unchanged. **Flushed on shutdown**
  (`examples/_chat_session.py`, `tests/unit/test_chat_session.py`, iter-160):
  `run_session` calls `aggregator.flush()` on exit so a mid-thought fragment the
  buffer was still holding when the user trailed off and hit Ctrl+C isn't
  silently lost — the released text rides out on `SessionState.stranded_utterance`
  and the session summary surfaces it (`_emit_stranded_utterance_line`). Mid-
  session a held pending always resolves via the next utterance's gap inside
  `offer`; shutdown is the one path `offer` can't reach. **Held utterances are
  counted separately from VAD false triggers** (`examples/_chat_loop.py`,
  `examples/_chat_session.py`, `tests/unit/test_chat_session.py`, iter-161): a
  held mid-thought utterance returns `TurnResult(held=True)` so `run_session`
  bumps `SessionState.utterances_held` instead of `false_triggers` — a buffered
  fragment is a *successful* capture, not a VAD misfire, and conflating the two
  (as iter-159's wiring did) silently inflated the false-trigger rate whenever
  merging was on. The count surfaces on its own "Utterances held" line in the
  organic-turn-taking summary block (`OrganicStats.utterances_held`).
  **Displaced fragments aren't glued onto the response** (`examples/_chat_aggregation.py`,
  `examples/_chat_loop.py`, `examples/_chat_session.py`,
  `tests/unit/test_chat_aggregation.py`, `tests/unit/test_displaced_utterances_line.py`,
  iter-162): when a held mid-thought fragment ("I was thinking about the") is
  followed not by a quick continuation but by a long silence and then a
  genuinely new utterance ("What time is it?"), the buffer releases *two*
  distinct turns in one `offer`. `resolve_turn` now responds to the **last**
  turn only (the new thought) and surfaces the abandoned earlier fragment(s) on
  `TurnResult.displaced` → `SessionState.utterances_displaced` → a "Displaced
  uttr." summary line (`_emit_displaced_utterances_line`) — the mid-session
  analog of iter-160's shutdown `stranded_utterance`. The pre-iter-162 code
  space-glued them into one garbled LLM input ("I was thinking about the What
  time is it?"); the responded turn's `false_endpoint` is now its own flag, not
  an OR across the abandoned fragments.
  **The merge-depth cap is a distinct signal** (`session/utterance_buffer.py`,
  `session/utterance_aggregator.py`, `examples/_chat_aggregation.py`,
  `examples/_chat_metrics.py`, `examples/_chat_loop.py`, iter-163): when the
  iter-157 `max_merge_depth` backstop *force-emits* a still-mid-thought utterance
  (a pathological "unfinished forever" stream), that turn now sets a distinct
  `merge_capped` flag — threaded `EmittedTurn` / `BufferResult.capped` →
  `AggregatedResult.capped` → `ResolvedTurn.merge_capped` →
  `TurnMetrics.merge_capped` → `OrganicStats.merges_capped` + a "Merges capped"
  summary line — instead of being silently counted as a clean merge. The cap
  firing is a tuning signal (retune the merge window/EOU), so surfacing it
  honors the "no silent caps" discipline.
  **Mid-session long-silence flush decision** (`session/silence_flush.py`,
  `tests/unit/test_silence_flush.py`, iter-164): the pure seam for the
  still-deferred half of the merge story. The `UtteranceBuffer` only releases a
  held fragment when the *next utterance* arrives — a user who trails off
  mid-thought and then says nothing leaves the fragment held until a new thought
  displaces it (iter-162) or shutdown flushes it (iter-160).
  `decide_silence_flush(held_text, silence_secs, …)` answers *should the loop
  give up waiting and `FLUSH` the held fragment to the engine now?* — `FLUSH`
  iff the inter-turn silence has **exceeded** the merge window
  (`silence_secs > max_gap_secs`), the same scalar `decide_utterance_continuation`
  uses, so the flush deadline and the merge window can't drift apart. With a
  default `FullDuplexConfig()` it returns `HOLD` for every input (and the buffer
  never holds anyway) — byte-for-byte today's behavior. Wiring it into
  `run_session`'s inter-turn clock read is the named follow-on, mirroring the
  decision-seam-first rhythm of iter-152/153.
  **Pre-speech idle timeout** (`record_utterance_streaming`'s `idle_timeout`
  arg, `examples/_chat_recording.py`, `tests/unit/test_chat_recording.py`,
  iter-165): the recorder *mechanism* the iter-164 flush decision needs.
  `record_utterance_streaming` blocks forever waiting for speech to start, so
  the live loop can never regain control during a long inter-turn pause to flush
  a held fragment — the blocker every lap since iter-160 named ("`run_session`
  reads no clock between turns"). With `idle_timeout=N` set, the recorder returns
  the empty-utterance `(b"", 0.0, 0.0)` tuple after `N` seconds of *pre-speech*
  silence (gated on `first_speech_at is None`, so a mid-utterance trailing pause
  can never trip it — `silence_duration` still owns end-of-turn), and flags
  `out_metrics["idle_timed_out"] = True` so the caller can tell a timeout from a
  VAD false trigger. `None` (default) preserves the wait-forever behavior
  byte-for-byte.
  **`ChatLoop` idle-timeout wiring** (`ChatLoop`'s `idle_timeout` ctor arg +
  `TurnResult.idle_timed_out`, `examples/_chat_loop.py`,
  `tests/unit/test_chat_loop.py`, iter-166): threads iter-165's recorder
  `idle_timeout` through `ChatLoop.__init__` to `record_utterance_streaming` and
  surfaces the recorder's `idle_timed_out` side-band flag on `TurnResult` (the
  empty-wav path now reads `rec_metrics["idle_timed_out"]`). A no-metrics turn is
  therefore one of three distinguishable causes — a held mid-thought fragment
  (`held`), a deliberate pre-speech idle timeout (`idle_timed_out`), or a VAD
  false trigger (neither flag). `None` (default) keeps the wait-forever path
  byte-for-byte. The last wiring hop — `run_session` reading `idle_timed_out`,
  measuring the inter-turn silence, and driving `should_flush_held_utterance`
  (iter-164) to flush a held fragment mid-session — is the named follow-on.
  **`run_session` mid-session flush wiring** (`run_session`'s `idle_timeout` +
  `flush_decider` params, `SessionState.idle_timeouts` + `.flushed_utterances`,
  `examples/_chat_session.py`, `tests/unit/test_chat_session.py` +
  `tests/unit/test_flushed_utterances_line.py`, iter-167): the second wiring hop
  consumes `TurnResult.idle_timed_out`. `run_session` grows an injected
  `flush_decider(held_text, silence_secs) -> bool` (production binds
  `should_flush_held_utterance` to the aggregator's own config) and an
  `idle_timeout` (the recorder window, used only as the `silence_secs` fed to the
  decider — `run_session` reads no clock itself). On an idle-timeout turn it bumps
  a separate `SessionState.idle_timeouts` counter (so enabling a timeout never
  inflates the false-trigger rate) and `_maybe_flush_on_idle` flushes a held
  mid-thought fragment when the decider says so, recording the released text on
  `SessionState.flushed_utterances` + a "Flushed uttr." summary line — the
  mid-session-idle analog of iter-160's shutdown strand and iter-162's displaced
  fragments. Half-duplex wires neither param so the wait-forever path is unchanged.
  Records but does not yet *respond* to the fragment — a `ChatLoop` text-only
  response entrypoint (`run_one_turn` always records from the mic first) is the
  named follow-on.
  **`ChatLoop.respond_to_text` — the spoken-response entrypoint**
  (`ChatLoop.respond_to_text(messages, text)` + the extracted
  `ChatLoop._stream_response` half, `examples/_chat_loop.py`,
  `tests/unit/test_chat_loop.py`, iter-168): the named follow-on. The LLM-stream
  → synth/play → barge-in-watcher half of `run_one_turn` is pulled into a shared
  `_stream_response`, leaving `run_one_turn` to own only Phase 1 (record + STT +
  organic aggregation). `respond_to_text` drives that shared half on a piece of
  text with **no mic recording** — building a `TurnMetrics` with synthetic
  anchors (`stt_time=0`, speech/turn clocks sampled now) and answering the text
  as its own turn (user + assistant appended, response spoken through the same
  worker). This is the path a flushed mid-thought fragment (iter-167) needs to be
  *answered* rather than only recorded; blank text is a no-op, and the
  `run_one_turn` path stays byte-for-byte unchanged (the Phase 2 body is moved,
  not modified). Wiring `run_session`'s `_maybe_flush_on_idle` to call it (so a
  flushed fragment is spoken, not just listed in `flushed_utterances`) is the
  remaining hop.
  **Speaking the flushed fragment — #9's mid-session flush, end-to-end**
  (`run_session`'s `respond_fn` param + `_speak_flushed_fragment` /
  `_record_completed_turn` helpers, `examples/_chat_session.py`,
  `tests/unit/test_chat_session.py`, iter-169): the last hop.
  `_maybe_flush_on_idle` now *returns* the flushed text (`None` whenever nothing
  was flushed), and `run_session` answers it through an injected `respond_fn`
  (production binds `ChatLoop.respond_to_text`, gated to organic mode like
  `flush_decider`). The spoken flush is counted exactly like a mic turn — metrics
  printed, `all_metrics` appended, turn counter advanced, trim run — via a shared
  `_record_completed_turn` extracted from the success block. A raising / errored
  / no-metrics `respond_fn` degrades to the iter-167 record-only behavior
  (`llm_errors` bumped, fragment still on `flushed_utterances`); `respond_fn=None`
  (default / half-duplex) keeps that path byte-for-byte. **Backlog #9's
  mid-session flush is now closed end-to-end: a trailed-off fragment, after a
  long idle silence, is flushed *and* spoken.**
- **[Performance metrics taxonomy](docs/perf-metrics-taxonomy.md)** — a
  catalog of metrics worth instrumenting on a local-first voice agent.

## Project Structure

```
geno-voice/
├── GENO.md           # agent instructions
├── SKILL.md          # umbrella skill manifest
├── genotools.yaml    # geno-tools manifest
├── skills/           # skill definitions
│   └── geno-voice/   #   umbrella
├── stt/              # speech-to-text pipeline
├── tts/              # text-to-speech pipeline
├── vad/              # voice activity detection
├── examples/         # usage examples and integration demos
├── docs/             # documentation site
└── mkdocs.yml        # MkDocs configuration
```

## License

MIT
