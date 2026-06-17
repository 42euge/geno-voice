# Organic / Full-Duplex Voice Interaction — Research Track

> **Living document.** Each lap on the "organic voice" track appends findings
> and decisions here. The goal: move geno-voice beyond rigid half-duplex
> ("you speak, it waits, it replies") toward conversation that feels organic —
> backchannels, overlap, smart end-of-turn (EOU), utterance queueing, barge-in.
>
> Bootstrapped iter-148 (2026-06-16). See the [backlog](#organic-voice-backlog)
> at the bottom; the top items are mirrored into `ITERATION_LOG.md` "next
> directions" so subsequent laps continue this track by default.

## Why this track

Today's pipeline is **strictly half-duplex**. Two entrypoints embody it:

- **`examples/mic_talk.py` / `mic_chat.py`** — record an utterance (VAD opens
  a window, trailing silence closes it), transcribe, respond, then listen
  again. The `vad_step` seam (iter-147) and the recording state machine make
  the turn boundary a pure function of *silence duration*.
- **`pipecat_server.py`** — the RestReflect sidecar. Silero VAD emits
  `VADUserStartedSpeakingFrame` / `VADUserStoppedSpeakingFrame`; the
  `STTProcessor` buffers between them and transcribes on stop; the
  `Broadcaster` runs `session/turn_taking.py`'s `TurnTakingEngine` to decide
  STAY_SILENT / PLAY_CUE / SPEAK_*. This is the most "organic-ready" surface
  we have: it already has a backchannel-cue concept (`broadcast_cue`,
  `CUE_TYPES`) and a multi-signal turn engine.

The turn boundary is **silence-only**. A pause mid-thought ("I was thinking…
[2s] …about the deadline") looks identical to a finished turn. Real
conversation uses *semantics, prosody, and syntax* to predict end-of-turn, and
overlaps freely (backchannels, barge-in). This track closes that gap
incrementally, **pipecat-native where possible** (the engine already runs
pipecat), each step measurable (false-endpoint rate, turn latency) rather than
asserted.

---

## SOTA landscape (seeded iter-148)

### 1. Full-duplex models

#### Moshi (Kyutai)
- **What:** A full-duplex speech-to-speech foundation model. Instead of a
  turn gate, Moshi models **two audio streams simultaneously** — its own
  output *and* the user's input — as parallel token streams, plus a text
  "inner monologue" stream that scaffolds the spoken output. There is no
  explicit "your turn / my turn" state: silence, overlap, and backchannels
  are all just what the model predicts next on each stream.
- **Architecture / signals:** A 7B temporal transformer (Helium LM backbone)
  + a smaller "depth" transformer over Mimi neural-codec tokens (~12.5 Hz
  frame rate, multiple codebooks). The inner-monologue text stream is
  predicted slightly ahead of audio, improving linguistic quality. Dual audio
  channels make backchanneling and interruption *native*.
- **Size / latency:** 7B params; theoretical ~160 ms, ~200 ms practical
  end-to-end latency on an L4-class GPU. Real-time only with a capable GPU.
- **License:** Code Apache-2.0; weights released under a permissive
  (CC-BY-style) license. Repo: `kyutai-labs/moshi`.
- **Fit for geno-voice:** **Aspirational, not adoptable now.** It replaces —
  not augments — our Silero-VAD + mlx-whisper STT + Kokoro TTS stack with a
  single GPU-bound model, contradicting the "local, modest-hardware, Apple
  Silicon" design. Value to us is *conceptual*: the dual-stream framing is the
  north star for "what organic feels like," and the inner-monologue idea
  (decide intent in text before committing audio) maps onto our existing
  text-first `TurnTakingEngine`. Track it; don't build on it yet.
  - Sources: <https://github.com/kyutai-labs/moshi>,
    <https://arxiv.org/abs/2410.00037> (Moshi paper).

### 2. Semantic end-of-turn / EOU (beyond VAD silence)

#### pipecat `smart-turn` — **most directly adoptable**
- **What:** An open-source, open-weights turn-detection model from the pipecat
  team (Daily). Audio-in, it predicts whether the user has *semantically*
  finished their turn — distinguishing "I think that's it." (complete) from
  "I think…" (incomplete, mid-thought) even when both are followed by the same
  silence. v2 expanded language coverage and trained on more natural data.
- **Signals / architecture:** Audio-only classifier (no transcript needed);
  consumes the recent speech buffer and outputs a completion probability.
  Designed to run **inside a pipecat pipeline** as a turn analyzer, replacing
  or augmenting pure-silence VAD endpointing.
- **Size / latency:** Small enough for CPU/edge inference; tens-of-ms
  inference on the captured buffer. Exact size varies by release.
- **License:** Open weights (BSD-style, per pipecat-ai). Repo:
  `pipecat-ai/smart-turn`.
- **Fit for geno-voice:** **Best near-term target.** `pipecat_server.py`
  already runs a pipecat `Pipeline` with a `SileroVADAnalyzer`. smart-turn is
  designed to slot in alongside VAD as the endpoint signal — its output is
  exactly the `smart_turn_confidence` parameter the `TurnTakingEngine.decide`
  already accepts (currently hardcoded to `0.5`!). The engine is *already*
  shaped for it; we just feed silence-only today. The clean path: define a
  `turn_decider` seam (backlog #2) that today maps silence→confidence, then
  later sources confidence from smart-turn without touching the engine.
  - Sources: <https://github.com/pipecat-ai/smart-turn>,
    <https://www.daily.co/blog/smart-turn/>.

#### LiveKit `turn-detector`
- **What:** An open-weights transformer LM that does **contextual EOU** — it
  reads the running transcript and predicts whether the latest utterance ends
  a turn, using linguistic context (a trailing "and…" or "because…" lowers
  end-probability). Ships in the LiveKit Agents framework.
- **Signals / architecture:** Text-based (consumes ASR transcript), so it's
  complementary to audio-only smart-turn. Quantized to run on CPU.
- **Size / latency:** ~hundreds of MB; few-tens-of-ms on CPU.
- **License:** Open weights, Apache-2.0 framework.
- **Fit for geno-voice:** **Adoptable as a text-side EOU signal**, but a worse
  fit than smart-turn for the pipecat path (LiveKit-framework-shaped). Most
  valuable as a *reference design* for a text EOU classifier we could run on
  the STT transcript (we already have the transcript in `Broadcaster`). A
  rule-based precursor — "transcript ends in a conjunction / filler →
  incomplete" — is cheap, testable, and a good backlog item (#4).
  - Sources: <https://docs.livekit.io/agents/build/turns/turn-detector/>,
    <https://github.com/livekit/agents>.

#### Krisp turn-taking model
- **What:** A ~6M-parameter **audio-only** turn-taking model doing EOU **plus
  backchannel prediction** — it predicts both "is the user done?" and "is now
  a good moment for the agent to emit a short backchannel?".
- **Signals / architecture:** Tiny audio classifier; the backchannel-timing
  head is the notable part — most models only do EOU.
- **Size / latency:** ~6M params → microcontroller-class; sub-10ms.
- **License:** Vendor (Krisp) — not open weights as of this writing; treat as
  a design reference, not a dependency.
- **Fit for geno-voice:** **Design reference for backchannel *timing*.** Our
  `TurnTakingEngine` already *picks* and rate-limits cues; what it lacks is a
  learned "good moment to backchannel" signal. The Krisp framing validates a
  dedicated backchannel-opportunity signal as its own seam — which our
  rule-based backchannel classifier (backlog #1, shipped iter-148) is the
  first, dependency-free step toward.
  - Source: <https://krisp.ai/blog/> (turn-taking model announcement).

#### arXiv 2603.13379 — hierarchical EOU with primary-speaker modeling (Mar 2026)
- **What:** A hierarchical end-of-utterance approach that explicitly models a
  **primary speaker** among background voices/noise, so EOU decisions aren't
  corrupted by cross-talk or a TV in the room.
- **Signals / architecture:** Hierarchical (frame-level + utterance-level)
  with a speaker-identity conditioning stream; targets robustness in noisy,
  multi-speaker rooms.
- **Fit for geno-voice:** **Relevant to robustness, not near-term.** Our
  `filter_noise` (`session/triggers.py`) already drops Whisper hallucinations
  and the test-audio path RMS-gates. Primary-speaker modeling is a future
  hardening direction for noisy environments; log it as a known frontier, not
  a backlog item yet.
  - Source: <https://arxiv.org/abs/2603.13379>.

### 3. Backchanneling

- **What:** Two halves. (a) **Recognizing** user backchannels — "mm-hmm",
  "yeah", "right", "uh-huh", "go on" — as **continuers** that mean *keep going,
  I'm listening*, NOT turn-ends. (b) **Emitting** agent backchannels *during*
  user speech to signal active listening (the human "mm-hmm" while someone
  talks).
- **Signals:** Continuers are reliably **short** (1–3 tokens), **low-energy /
  low-pitch-excursion**, and lexically closed-class. That makes a rule-based
  first pass (short-token + low-energy + closed lexicon) genuinely effective
  before any model — this is exactly what Krisp's tiny model learns.
- **Fit for geno-voice:** **Strong, immediate fit.** Two gaps exist today:
  1. `pipecat_server.py`'s `STTProcessor` transcribes the buffered utterance
     and `filter_noise` **discards** filler-only utterances ("yeah", "mhmm")
     entirely. A backchannel is thrown away as noise rather than recognized as
     a *continuer signal* (which should, e.g., suppress a premature
     SPEAK_FULL and reset the silence clock — the user hasn't yielded).
  2. The engine already *emits* cues (`PLAY_CUE` / `broadcast_cue`) but only
     on silence + confidence; it has no notion of "the user just backchanneled,
     so I should keep listening."
  A pure `classify_backchannel(text, energy=…)` seam (backlog #1, **shipped
  iter-148**) closes gap (1) as a dependency-free, fully-tested primitive that
  the turn engine and pipecat server can both consume.

### 4. Utterance queueing / interruption (barge-in)

- **What:** Handling user utterances that arrive **while the agent is still
  speaking**: buffer/merge partial utterances, and decide whether to **abandon**
  the current TTS (true interruption) or **finish** it (the user only
  backchanneled).
- **Signals:** The abandon-vs-finish decision hinges on *what* the user said —
  a backchannel ("mhmm") → finish; a substantive interruption ("wait, no") →
  abandon. This is precisely where the backchannel classifier (above) pays off
  a second time.
- **Fit for geno-voice:** **Partially built, ripe to deepen.** geno-voice
  already has substantial barge-in machinery on the `mic_chat` path:
  `BargeInWatcher` + `BargeInCoordinator` (iter-009/010), cancel-flush
  (iter-026), and a barge-phase consistency sentinel (iter-120). What's missing
  is the **abandon-vs-finish discrimination**: today any barge cancels. Wiring
  the backchannel classifier into the coordinator so a *continuer* doesn't
  abort the agent's turn is a high-value, well-scoped future item (backlog #5).

---

## How the pieces map onto geno-voice's pipeline

```
mic → Silero VAD ──► VADUserStarted/StoppedSpeakingFrame
                       │
            (silence-only endpoint today)
                       ▼
        STTProcessor (mlx-whisper, buffers between start/stop)
                       │ TranscriptionFrame(text)
                       ▼
        Broadcaster ──► detect_triggers(text)          [session/triggers.py]
                   └──► TurnTakingEngine.decide(        [session/turn_taking.py]
                            silence_duration,
                            smart_turn_confidence=0.5,  ◄── HARDCODED today
                            transcript_chunk=text)
                       │ Action: STAY_SILENT / PLAY_CUE / SPEAK_*
                       ▼
            broadcast_cue()  /  forward to LLM
```

The seams this track will exploit, in dependency order:

1. **`smart_turn_confidence` is hardcoded `0.5`.** A `turn_decider` seam that
   computes it (silence heuristic now, smart-turn model later) is a pure
   drop-in. The engine is *already* parameterized for it.
2. **Backchannels are discarded by `filter_noise`.** A `classify_backchannel`
   seam recovers them as continuer signals before they're dropped.
3. **Barge-in always abandons.** The classifier output lets the coordinator
   distinguish abandon (substantive) from finish (continuer).

---

## Organic-voice backlog

Prioritized **easiest-highest-leverage first**. Each item is independently
shippable with tests, keeping geno-voice's "every code path tested, every
output discoverable from README" discipline. Local only.

| # | Item | Leverage | Status |
|---|------|----------|--------|
| 1 | **Rule-based backchannel/continuer classifier** — pure `classify_backchannel(text, energy=…)` in `session/backchannel.py`: short-token + closed-lexicon + optional low-energy gate ⇒ `CONTINUER` / `SUBSTANTIVE` / `NOT_SPEECH`. Dependency-free, fully testable. Foundation for #5. | High | **DONE iter-148** |
| 2 | **`turn_decider` seam** — pure function wrapping today's silence→confidence heuristic behind the same interface a smart-turn model would use, so `smart_turn_confidence` stops being hardcoded `0.5` and the model swaps in later without touching `TurnTakingEngine`. | High | **DONE iter-149** |
| 3 | **Full-duplex config flag scaffolding** — a `TurnTakingConfig` / env flag (`GENO_FULL_DUPLEX`) that gates organic behaviors (continuer-aware listening, agent backchannels) off by default, so the half-duplex path is never regressed while the track matures. | Medium | **DONE iter-151** |
| 4 | **Rule-based text EOU precursor** — `is_utterance_complete(text)` that lowers end-of-turn likelihood when the transcript ends in a conjunction / filler / trailing-off marker (mirrors LiveKit turn-detector's linguistic signal; reuses `_TRAILING_PATTERNS`). Feeds #2's confidence. | Medium | **DONE iter-150** |
| 5 | **Continuer-aware barge-in** — wire #1 into `BargeInCoordinator` so a *continuer* utterance ("mhmm") during agent speech does NOT abandon the turn (finish), while a substantive interruption does (abandon). Measure: false-abandon rate. | High | **DONE iter-152** (decision seam `decide_barge_action`; coordinator wiring is the follow-on) |
| 6 | **Adopt pipecat `smart-turn`** — replace #2's heuristic body with the smart-turn model inside `pipecat_server.py`'s pipeline; same `turn_decider` interface. Measure false-endpoint rate vs silence-only baseline on recorded sessions. | High | TODO (blocked on model + Apple Silicon) |
| 7 | **Agent backchannel emission timing** — a learned/heuristic "good moment to backchannel" signal (Krisp-style) feeding the existing `PLAY_CUE` path, so the agent emits continuers *during* long user speech, not only on silence. | Medium | **DONE iter-153** (decision seam `decide_backchannel_timing`) + **iter-170** (`session/backchannel_monitor.py` — the stateful `BackchannelMonitor` driver the seam needs: `observe(*, now, monologue_start_at, pause_secs)` derives `user_speaking_secs` + `secs_since_last_backchannel` and routes them through the seam, recording the emit timestamp *iff* the decision is `EMIT` so the `min_between_cues_secs` rate limit actually engages across calls — the one piece of cross-event state the pure seam can't carry. Mirrors the iter-156 `UtteranceBuffer` / iter-158 `UtteranceAggregator` driver relationship to the iter-155 seam. Default (half-duplex) config ⇒ `emit=False` always, state never mutated. Live cue-path wiring — `pipecat_server.py`'s `Broadcaster`, blocked on the absent pipecat dep on the x86_64 runner — is the named follow-on) + **iter-171** (`session/cue_rotation.py` — the monitor now also owns the *second* piece of cross-event state the pure seam can't carry: *which* cue to play. `CUE_ROTATION` + a pure `cue_for_index(i)` are extracted from `turn_taking.py` into a shared dependency-free module so the silence-driven `PLAY_CUE` path and the mid-speech monitor index one list and can't drift; `BackchannelMonitor.observe` returns `BackchannelDecision.cue_type` on an emit, advancing the rotation ("mhmm" → "i see" → "right" → ...) so consecutive backchannels don't repeat — a held frame never burns a slot, and `reset()` keeps the rotation position (a fresh monologue continues rather than replaying "mhmm"). `turn_taking.py` re-exports `CUE_ROTATION` for back-compat and routes `_next_cue` through `cue_for_index`, byte-for-byte the same rotation. Half-duplex ⇒ `cue_type=None`, index never advances) + **iter-172** (`plan_cue_broadcast(decision, *, trigger_fired)` — the pure gate that finally wires the **silence-driven** `PLAY_CUE` path in `pipecat_server.py`'s `Broadcaster` to the shared rotation, fixing two live bugs: the branch read `Action.play_cue` (an `AttributeError` — member is `PLAY_CUE` — so the cue branch crashed on every transcript frame) and `broadcast_cue` re-picked a *random* cue from a private `CUE_TYPES` list, dropping the engine's rotation and risking cue keys outside it. Now the planner returns the engine's *rotated* `decision.cue.cue_type` (or `None` when not PLAY_CUE / trigger fired / no cue) and `broadcast_cue(cue_type)` plays exactly that. The mid-speech `BackchannelMonitor` → `Broadcaster` wiring stays the named follow-on, blocked on the absent pipecat dep) + **iter-173** (`session/monologue_clock.py` — the VAD-event driver that supplies the two derived quantities `BackchannelMonitor.observe` can't compute itself: `MonologueClock` consumes `on_speech_start(now)`/`on_speech_stop(now)` (the VAD started/stopped stream) and exposes `monologue_start_at` + `pause_secs(now)`. The key semantics: a "monologue" survives **brief clause-boundary pauses** and resets `monologue_start_at` only on a `stop → start` gap `>= reset_gap_secs` (default 2.0s — exactly `backchannel_timing.py`'s `max_pause_secs` and `turn_decider.py`'s `silence_floor_secs`, so the clause-pause/turn-end boundary is one shared scalar across the whole organic stack and can't drift), so `user_speaking_secs` accumulates across the pauses and the warm-up gate eventually clears; `pause_secs` is the live gap since the last stop, zero while speaking, clamped `>= 0`. Pure/dependency-free, file-path-loadable like its siblings. This is the **last pure piece** before the (still pipecat-blocked) live `Broadcaster` wiring becomes a thin adapter — `clock.on_speech_*` from the VAD frames, then `monitor.observe(now, clock.monologue_start_at, clock.pause_secs(now))` → `broadcast_cue(d.cue_type)`) |
| 8 | **Naturalness metrics for the organic path** — extend `TurnMetrics` / session-summary with false-endpoint rate and continuer-detection counts so the track is measured, not asserted. | Medium | **DONE iter-154** (`false_endpoint` + `continuers_detected` `TurnMetrics` fields; `_emit_organic_block` summary block; populating them from the live organic path is the follow-on) |
| 9 | **Utterance buffer-merge** (Section 4's second half) — a pure `decide_utterance_continuation(prev_text, next_text, gap_secs)` that, when a silence endpoint fires on *unfinished*-looking text and a continuation arrives within the pause window, returns `MERGE` (the endpoint was a false positive — user paused mid-thought) vs `NEW`. Composes #4's `utterance_completeness` + #3's gate. Repairs the false endpoints #8 measures. | Medium | **DONE iter-155** (decision seam `decide_utterance_continuation`) + **iter-156** (stateful `UtteranceBuffer` hold-and-merge driver) + **iter-157** (`max_merge_depth` starvation cap) + **iter-158** (`UtteranceAggregator` cross-turn gap-measuring driver) + **iter-159** (live `ChatLoop` wiring behind the off-by-default `aggregator` seam — held ⇒ re-listen, merged ⇒ respond to joined text + set `TurnMetrics.false_endpoint`) + **iter-160** (`run_session` flushes the aggregator on shutdown — a held mid-thought fragment the user never completed before Ctrl+C surfaces on `SessionState.stranded_utterance` + a session-summary line rather than vanishing inside the buffer) + **iter-161** (held utterances counted separately from VAD false triggers — `TurnResult.held` flag ⇒ `SessionState.utterances_held` + an "Utterances held" line in the organic summary block, fixing iter-159's silent inflation of the false-trigger rate) + **iter-162** (a multi-turn release — an abandoned mid-thought fragment + a genuinely-new utterance, split by a long silence — no longer space-glued into one garbled LLM input; `resolve_turn` responds to the last turn and surfaces the abandoned fragment(s) on `TurnResult.displaced` ⇒ `SessionState.utterances_displaced` + a "Displaced uttr." summary line, the mid-session analog of iter-160's shutdown `stranded_utterance`) + **iter-163** (the iter-157 `max_merge_depth` cap force-emit is now a distinct `merge_capped` signal — threaded `EmittedTurn`/`BufferResult.capped` → `AggregatedResult.capped` → `ResolvedTurn.merge_capped` → `TurnMetrics.merge_capped` → `OrganicStats.merges_capped` + a "Merges capped" summary line — instead of being silently counted as a clean merge, honoring the "no silent caps" discipline) + **iter-164** (`session/silence_flush.py` — the pure `decide_silence_flush(held_text, silence_secs, …)` decision for the still-deferred mid-session long-silence flush: `FLUSH` a held mid-thought fragment iff the inter-turn silence *exceeds* `max_gap_secs` (the same merge-window scalar, so the flush deadline and merge window can't drift apart), else `HOLD`. Closes the buffer's blind spot — a fragment held when the user trails off and then says nothing was only released by the next utterance (iter-162) or shutdown (iter-160). Decision-seam-first per the iter-152/153 rhythm; `run_session` inter-turn clock wiring is the follow-on) + **iter-165** (`record_utterance_streaming` grows an optional `idle_timeout` arg — the recorder *mechanism* the iter-164 flush decision needs. The recorder blocked forever waiting for speech to start, so the loop could never regain control during a long inter-turn pause to flush a held fragment; with `idle_timeout=N` it returns the empty-utterance tuple after `N` seconds of *pre-speech* silence — gated on `first_speech_at is None` so a mid-utterance trailing pause can't trip it — and flags `out_metrics["idle_timed_out"]=True` so a timeout is distinguishable from a VAD false trigger. `None` (default) preserves wait-forever behavior byte-for-byte; wiring it through `ChatLoop`→`run_session` to drive `should_flush_held_utterance` is the follow-on) + **iter-166** (first wiring hop: `ChatLoop.__init__` grows an `idle_timeout` arg threaded to `record_utterance_streaming`, and the recorder's `idle_timed_out` side-band flag is surfaced on `TurnResult.idle_timed_out` — so a no-metrics turn is now one of three distinguishable causes: held fragment / deliberate idle timeout / VAD false trigger. `None` default keeps the wait-forever path byte-for-byte; the last hop — `run_session` reading the flag, measuring the inter-turn silence, and driving `should_flush_held_utterance` for a real mid-session flush — is the named follow-on) + **iter-167** (second wiring hop: `run_session` grows injected `idle_timeout` + `flush_decider` params, consumes `TurnResult.idle_timed_out` into a separate `SessionState.idle_timeouts` counter, and a `_maybe_flush_on_idle` helper flushes a held mid-thought fragment via the injected decider — production binds `should_flush_held_utterance` to the aggregator's own config — recording the released text on `SessionState.flushed_utterances` + a "Flushed uttr." summary line. The mid-session-idle analog of iter-160's shutdown strand and iter-162's displaced fragments. Half-duplex wires neither param so the wait-forever path is byte-for-byte unchanged. Records but does not yet *respond* to the fragment — a `ChatLoop` text-only response entrypoint is the named follow-on) + **iter-168** (the spoken-response entrypoint: the LLM-stream → synth/play → barge-in-watcher half of `ChatLoop.run_one_turn` is extracted into a shared `_stream_response`, and a new `ChatLoop.respond_to_text(messages, text)` drives it on a piece of text with NO mic recording — building a `TurnMetrics` with synthetic anchors (`stt_time=0`, speech/turn clocks sampled now) and answering the text as its own turn. This is the path a flushed mid-thought fragment needs to be *answered* rather than only recorded; blank text is a no-op and the `run_one_turn` path is byte-for-byte unchanged (Phase 2 moved, not modified). Wiring `_maybe_flush_on_idle` to call `respond_to_text` so a flushed fragment is spoken is the remaining hop) + **iter-169** (the last hop — end-to-end: `run_session` grows an injected `respond_fn` param (production binds `ChatLoop.respond_to_text`), `_maybe_flush_on_idle` now *returns* the flushed text, and a new `_speak_flushed_fragment` helper answers it as its own turn — counting the spoken turn exactly like a mic turn (metrics printed, `all_metrics` appended, turn counter advanced, trim run) via a shared `_record_completed_turn` extracted from the success block. A failing/raising `respond_fn` degrades to record-only (`llm_errors` bumped, fragment still on `flushed_utterances`). `respond_fn=None` (default / half-duplex) keeps the iter-167 record-only path byte-for-byte. **Backlog #9's mid-session flush is now closed end-to-end: a trailed-off fragment, after a long idle silence, is flushed AND spoken.**) |

---

## Findings log (append per lap)

### iter-148 (2026-06-16) — bootstrap + backchannel classifier (#1)

- Seeded the SOTA landscape above (Moshi, smart-turn, LiveKit turn-detector,
  Krisp, arXiv 2603.13379) and the backlog.
- **Key realization:** the pipeline is *already shaped* for semantic
  turn-taking — `TurnTakingEngine.decide` takes a `smart_turn_confidence`
  parameter that `pipecat_server.py` hardcodes to `0.5`. The cheapest organic
  wins are seams that *feed* existing parameters, not rewrites.
- **Shipped backlog #1:** `session/backchannel.py` — a pure, dependency-free
  `classify_backchannel(text, energy=None)` returning `CONTINUER` /
  `SUBSTANTIVE` / `NOT_SPEECH`, with an optional low-energy gate (continuers
  are short + low-energy + closed-lexicon). Recognizes the backchannels that
  `filter_noise` currently discards as a distinct *continuer* signal, so a
  future lap can stop a continuer from being treated as a turn-end or a
  barge-abandon. 30 unit tests; see `tests/unit/test_backchannel.py`.
- **Next:** backlog #2 (`turn_decider` seam) — unhardcode
  `smart_turn_confidence`.

### iter-149 (2026-06-16) — turn-decider seam (#2)

- **Shipped backlog #2:** `session/turn_decider.py` — the swappable seam
  between *where turn-end confidence comes from* and *what the engine does
  with it*. `silence_confidence(silence_duration_secs, config)` is a pure,
  monotone, saturating map: 0.0 at/below `silence_floor_secs` (2.0s — a pause,
  not a turn-end), linear ramp, 1.0 at/above `silence_ceiling_secs` (5.0s).
  `SilenceTurnDecider.confidence(silence_duration_secs=…, transcript_chunk=…)`
  is the interface a future audio `smart-turn` decider (backlog #6) implements
  unchanged; `transcript_chunk` is accepted-and-ignored today for forward-compat
  with a text EOU signal (#4). 25 tests (`tests/unit/test_turn_decider.py`),
  including two that load the real `TurnTakingEngine` by file path.
- **Bug this surfaced and fixed:** the hardcoded `0.5` was *below* the engine's
  `smart_turn_backchannel_min` (0.6), so **every silence-driven backchannel /
  response tier in `TurnTakingEngine` was dead in production** — the engine
  could only ever fire on an NLP trigger or LLM assessment. The default ramp is
  tuned so `confidence(4.0)≈0.67 ≥ 0.6` (backchannel tier now reachable at the
  engine's `silence_backchannel_min`) and `confidence(6.0)=1.0 ≥ 0.85` (response
  tier reachable at `silence_response_min`). An integration test pins this:
  `decide(4.5, 0.5)` ⇒ `STAY_SILENT` (old) vs `decide(4.5, silence_confidence(4.5))`
  ⇒ `PLAY_CUE` (new).
- **Wired into `pipecat_server.py`:** `Broadcaster` now holds a
  `SilenceTurnDecider`; both `decide(...)` call sites source confidence from it
  instead of the literal `0.5` (live transcription path + offline replay path).
- **Next:** backlog #4 (rule-based text EOU precursor `is_utterance_complete`,
  reusing `_TRAILING_PATTERNS`) — its output feeds this seam's confidence via
  the already-present `transcript_chunk` argument; or backlog #3 (full-duplex
  config flag scaffolding) to gate organic behaviors off by default.

### iter-150 (2026-06-16) — rule-based text EOU precursor (#4)

- **Shipped backlog #4:** `session/text_eou.py` — a pure, dependency-free
  `utterance_completeness(text) -> float` (and boolean `is_utterance_complete`)
  that returns a completeness **multiplier** in (0.0, 1.0]: `1.0` when the
  transcript shows no sign of being unfinished, lower when it trails off on a
  **conjunction** ("…and/because/but" → 0.2, strongest), a **dangling
  preposition/article/possessive** ("…to/the/my" → 0.3), a **hesitation filler**
  ("…um/like" → 0.35), an **ellipsis** ("…/..." → 0.5), or a **comma** (0.6).
  This is the cheap precursor to LiveKit's learned text turn-detector; a model
  can later replace the body behind the same interface.
- **Design choice — multiplier, not a replacement signal.** The output
  *dampens* the iter-149 silence-derived confidence rather than overriding it:
  `confidence = silence_confidence(silence) * utterance_completeness(text)`. So
  the change is **monotone and conservative** — it can only *lower* confidence on
  textual evidence of incompleteness, never raise it. A complete utterance (or
  no transcript) multiplies by 1.0, leaving the silence-only behaviour exactly
  intact. The combined `TextAwareTurnDecider` implements the **identical**
  `confidence(*, silence_duration_secs, transcript_chunk=None)` interface as
  `SilenceTurnDecider`, so `pipecat_server.py` swaps it in with no call-site
  signature change (it already passed `transcript_chunk=text` since iter-149).
- **Why it matters:** silence-only endpointing can't tell "I was thinking…
  [pause] …about the deadline" from a finished turn. Feeding completeness in
  means a 4.5s pause after "…that and" (conjunction, ×0.2) drops below the
  engine's `smart_turn_backchannel_min`, so the engine STAYS_SILENT instead of
  barging into a mid-thought — while the same pause after "that's my whole
  point." still fires `PLAY_CUE`. An engine integration test (loaded by file
  path) pins exactly this contrast.
- **Subtle bug avoided in design:** demonstratives/quantifiers
  (this/that/these/those/some/any) were initially in the dangling set but are
  frequently *complete* sentence-final pronouns ("I did this", "I want some") —
  including them dampened finished turns. They're excluded; only true
  must-have-an-object function words (prepositions, articles, possessives)
  remain. ("that" stays in the conjunction set as a relative pronoun, a
  separate, stronger signal.)
- **Note on `_TRAILING_PATTERNS` reuse:** the backlog framed this as "reuse
  `session/triggers.py:_TRAILING_PATTERNS`", but importing `session.triggers`
  runs `session/__init__` which eagerly pulls pipecat (absent on the x86_64
  runner). `text_eou.py` stays pure stdlib (loads by file path in tests, like
  its siblings), so it re-expresses the trailing-off idea as an EOU-framed
  superset (incomplete ⇒ more coming) rather than importing the emotional
  PLAY_CUE patterns. Same concept, decoupled module.
- 49 unit tests (`tests/unit/test_text_eou.py`): complete utterances → 1.0,
  every marker class, precedence (conjunction > ellipsis-position > comma; "so"
  resolves as conjunction not filler), substring guard ("android" ≠ marker),
  config validation/frozen/custom values, the `TextAwareTurnDecider` (complete
  text == silence-only, no/empty transcript == silence-only, incomplete dampens
  & never exceeds silence, 0-silence stays 0, keyword-only, injected configs,
  interface-match with `SilenceTurnDecider`), and the engine integration
  contrast.
- **Next:** backlog #3 (full-duplex config flag scaffolding, `GENO_FULL_DUPLEX`)
  to gate organic behaviors off by default; or backlog #5 (continuer-aware
  barge-in — wire iter-148's `classify_backchannel` into `BargeInCoordinator`).

### iter-151 (2026-06-16) — full-duplex config flag scaffolding (#3)

- **Shipped backlog #3:** `session/full_duplex.py` — the off-by-default gate
  for the organic behaviors, so subsequent laps add behavior *behind* a switch
  rather than introducing both behavior and guard at once.
  - `FullDuplexConfig` (frozen dataclass): a master `enabled` switch plus
    three-state (`bool | None`) per-behavior sub-flags
    (`continuer_aware_listening`, `agent_backchannels`). A `None` sub-flag
    **inherits** the master; an explicit `True`/`False` overrides it (organic
    mode on, but one behavior held back). Effective state is read through
    `continuer_aware_listening_active()` / `agent_backchannels_active()` /
    `any_active()` so call sites never re-derive the inherit logic.
  - **The half-duplex invariant:** a default `FullDuplexConfig()` has
    `enabled=False` and every `*_active()` resolves `False` — byte-for-byte
    today's behavior. A test pins this.
  - `full_duplex_config_from_env(env=os.environ)` reads `GENO_FULL_DUPLEX`
    (master) + `GENO_FULL_DUPLEX_CONTINUER_AWARE` /
    `GENO_FULL_DUPLEX_AGENT_BACKCHANNELS` (overrides). `env` is injected so
    parsing is testable without touching the process environment.
  - `parse_bool_flag` uses **closed** TRUTHY/FALSY sets and raises on an
    unrecognized value (naming the offending var) — a misspelled enable flag
    that silently leaves organic mode off is the worst failure mode for a
    gate, so a typo surfaces loudly instead. `None` (unset) is distinct from
    `""` (set-but-empty ⇒ falsy).
- **No runtime behavior change, nothing wired yet.** The module is pure,
  dependency-free, and as-yet-unconsumed; backlog #5 (continuer-aware barge-in)
  and #7 (agent backchannels) read these flags in later laps. 38 unit tests
  (`tests/unit/test_full_duplex.py`): bool parsing (every truthy/falsy
  spelling, case/whitespace, unset-vs-empty, typo raises), the half-duplex
  default invariant, the inherit/override matrix, and the env builder
  (empty ⇒ half-duplex, master on/off, sub-flag overrides, bad value
  propagation, frozen result).
- **Next:** backlog #5 (continuer-aware barge-in — wire iter-148's
  `classify_backchannel` into `BargeInCoordinator`, gated behind
  `continuer_aware_listening_active()`); or backlog #8 (naturalness metrics:
  false-endpoint rate + continuer counts in `TurnMetrics`/session-summary).

### iter-152 (2026-06-16) — continuer-aware barge-in decision (#5)

- **Shipped backlog #5 (the decision seam):** `session/barge_decision.py` —
  `decide_barge_action(transcript, energy=None, *, config) -> BargeAction`
  (`ABANDON` / `FINISH`), the pure composition of two earlier seams: the
  backchannel classifier (#1, iter-148) and the full-duplex gate (#3,
  iter-151). A user "mhmm" / "yeah" / "right" during agent speech is a
  *continuer* — *keep going, I'm listening* — not a turn-grab; abandoning the
  agent's turn on it clips its own sentence for nothing. The seam decides
  abandon-vs-finish from the barge transcript so a later lap can gate
  `BargeInCoordinator.trigger()` behind it.
- **The half-duplex invariant is the whole point.** Rule 1 short-circuits on
  `config.continuer_aware_listening_active()`: with a default
  `FullDuplexConfig()` (the switch off), the function returns `ABANDON` for
  *every* transcript — byte-for-byte today's "any barge cancels" behavior, and
  the transcript isn't even classified. Only with continuer-aware listening
  explicitly on does Rule 2 run: a confirmed `CONTINUER` ⇒ `FINISH`, while
  `SUBSTANTIVE` real speech *and* `NOT_SPEECH` empty/noise both ⇒ `ABANDON`.
  Rule 2 is deliberately conservative toward `ABANDON` — only a *confirmed*
  continuer holds the floor, so a misclassification errs on responsiveness
  (the user who really interrupted is never left talking over a droning agent).
- **Why a decision seam, not coordinator wiring, this lap.** Same discipline as
  the rest of the track: ship the pure, fully-tested primitive behind the
  off-by-default gate first; wire it into the live `mic_chat` barge path
  (`should_abandon_turn(text, ...)` guarding `coord.trigger()`) as a separate,
  reviewable lap. `should_abandon_turn` is the call-site-shaped boolean
  convenience — with a default config it's always `True`, so the existing
  unconditional `coord.trigger()` is unchanged when wired.
- **`energy` / `max_words` / `energy_ceiling` thread through to the
  classifier** so the audio-aware emphatic-"YEAH!"-takes-the-floor gate
  (iter-148) works here too: a loud short continuer above the ceiling
  classifies SUBSTANTIVE ⇒ ABANDON even under organic mode.
- 40 unit tests (`tests/unit/test_barge_decision.py`): the half-duplex
  invariant (every transcript abandons under default / explicit-default /
  master-on-but-held-back configs; continuer isn't even classified when
  gated); organic mode (continuer FINISHes, substantive/empty ABANDON,
  sub-flag-true overrides master-off); the energy gate (loud continuer
  abandons, quiet finishes, custom ceiling); `max_words` threading;
  `should_abandon_turn` boolean + decide-match; and purity/interface
  (config keyword-only, no config mutation, distinct enum values). Loaded by
  file path under a stub `session` namespace — the same `session/__init__`
  pipecat-bypass trick the text_eou / turn_decider tests use.
- **Next:** wire `should_abandon_turn` into the `mic_chat` barge path (the
  follow-on to this lap — gate `coord.trigger()` behind it, threading the
  barge transcript + RMS energy in) and measure false-abandon rate; or
  backlog #8 (naturalness metrics: false-endpoint rate + continuer counts in
  `TurnMetrics` / session-summary), or #7 (agent backchannel emission timing).

### iter-153 (2026-06-16) — agent backchannel emission timing (#7)

- **Shipped backlog #7 (the decision seam):** `session/backchannel_timing.py` —
  `decide_backchannel_timing(*, user_speaking_secs, pause_secs,
  secs_since_last_backchannel, config, timing) -> BackchannelTiming`
  (`EMIT` / `HOLD`), the *emit* half of backchanneling that complements the
  *recognize* half in `session/backchannel.py` (#1, iter-148). Where the
  `TurnTakingEngine` already emits cues only on **trailing silence** ≥
  `silence_backchannel_min` (4.0s) — a turn-end-ish cue — this seam decides
  when the agent should "mhmm" *during* the user's monologue, the human
  nod-along. Krisp's tiny turn-taking model calls out exactly this
  backchannel-opportunity head as the signal most EOU models lack; this is the
  rule-based, dependency-free first step toward it.
- **The half-duplex invariant is the whole point.** Rule 1 short-circuits on
  `config.agent_backchannels_active()`: with a default `FullDuplexConfig()`
  (the switch off), the function returns `HOLD` for *every* input — the agent
  never speaks during user speech, byte-for-byte today's behavior, and no other
  signal is even consulted. Only with agent backchannels explicitly on do rules
  2–4 run.
- **The signal partitions the silence axis cleanly.** A good mid-speech moment
  is a brief **clause-boundary pause**: `min_pause_secs <= pause_secs <
  max_pause_secs`. The default `max_pause_secs` is **2.0s — exactly
  `turn_decider.py`'s `silence_floor_secs`** ("a pause, not a turn-end"), so
  the mid-speech backchannel window `[0.3, 2.0)` and the silence-driven
  `PLAY_CUE` window `≥ silence_backchannel_min` (4.0s) do not overlap: a short
  pause is a nod-along, a long one is a turn-end cue, and neither path claims
  the other's gap. A test pins `max_pause == 2.0` so a future edit that breaks
  the partition fails loudly.
- **Same rate limits as the engine.** `min_speaking_before_first_cue_secs`
  (15.0) and `min_between_cues_secs` (20.0) mirror `TurnTakingConfig`, so the
  agent doesn't backchannel over a one-word reply (rule 2 warm-up) or chatter
  "mhmm mhmm mhmm" (rule 3 rate limit). `secs_since_last_backchannel=None`
  (never backchanneled yet) passes the rate gate.
- **Why a decision seam, not cue-path wiring, this lap.** Same discipline as
  the rest of the track (iter-148/149/150/151/152): ship the pure, fully-tested
  primitive behind the off-by-default gate first; wire it into the live
  `PLAY_CUE` / `broadcast_cue` path (`if should_emit_backchannel(...):
  broadcast_cue(...)`) as a separate, reviewable lap. With a default config
  `should_emit_backchannel` is always False, so the live cue path is unchanged
  until agent backchannels are explicitly enabled.
- 44 unit tests (`tests/unit/test_backchannel_timing.py`): the half-duplex
  invariant (every input HOLDs under default / explicit-default / master-on-
  but-held-back configs); organic-mode EMIT (master-on and sub-flag-overrides-
  master-off); the warm-up gate (below/at/custom); the rate-limit gate
  (too-soon/at-boundary/`None`-passes/custom); the pause window (below-min,
  at-min inclusive, just-below-max, at-max exclusive, above-max, zero, custom
  window); rule precedence (gate > warm-up > rate-limit > pause); the
  `should_emit_backchannel` boolean (default-always-False, organic split,
  decide-match across a grid); and config validation/purity/interface
  (defaults, frozen, `max > min_pause` invariant, negative-value guards,
  keyword-only, no config/timing mutation, distinct enum values). Loaded by
  file path under a stub `session` namespace — the same `session/__init__`
  pipecat-bypass trick the barge_decision / text_eou tests use.
- **Next:** wire `should_emit_backchannel` into the live cue path (the follow-on
  to this lap — gate a mid-speech `broadcast_cue` behind it, feeding the
  user-speaking duration + within-speech pause from the VAD/STT loop, gated
  behind `agent_backchannels_active()`); or backlog #8 (naturalness metrics:
  false-endpoint rate + continuer counts in `TurnMetrics` / session-summary, so
  the track is measured not asserted).

### iter-154 (2026-06-16) — naturalness metrics for the organic path (#8)

- **Shipped backlog #8 (the measurement surface):** the organic track shipped
  its decision seams (#1/#5/#7) behind an off-by-default gate but had no way to
  *measure* whether they help. This lap adds that surface in
  `examples/_chat_metrics.py`:
  - Two additive `TurnMetrics` fields: `false_endpoint: bool` (the headline
    metric the LiveKit turn-detector / pipecat smart-turn literature tracks —
    the EOU decision fired early and the user actually had more to say) and
    `continuers_detected: int` (user backchannels recognized via iter-148's
    `classify_backchannel` and held the agent's floor via iter-152's
    `decide_barge_action`, instead of clipping the turn). Metrics 3.22 / 3.23
    in the perf-metrics taxonomy.
  - An `OrganicStats` dataclass + `_emit_organic_block` session-summary helper
    rendering a "False endpoints: N/M turns (X%)" rate (with an "EOU too eager;
    raise silence_duration" suggestion above a 20% threshold) and a
    "Continuers held: N" positive line.
  - Per-turn `TurnMetrics.print()` renders a yellow "False endpoint: yes" flag
    and a dim "Continuers: N" line.
- **Purely additive — byte-for-byte unchanged half-duplex output.** Both fields
  default off and stay zero on the proven half-duplex silence-VAD path (it
  neither mis-decides an organic endpoint nor classifies continuers), so the
  per-turn lines and the whole `_emit_organic_block` are suppressed and today's
  summaries are identical to pre-iter-154. The block only appears once the
  organic seams are wired into the live path and start populating the fields —
  exactly the "measured, not asserted" goal.
- **`false_endpoint` complements iter-056's `barge_in_regret`.** Regret is the
  *latency*-based pre-emption signal (user barged within 200ms of bot audio);
  `false_endpoint` is the *decision*-based one (the agent declared end-of-turn
  and the user resumed). Two angles on the same "the agent cut the user off"
  failure.
- **Why measurement now, live-population later.** Same discipline as the rest
  of the track: ship the metric surface (default-off, fully tested) before the
  code that feeds it. Wiring the live organic path to set `false_endpoint` (on
  a resumed-after-endpoint event) and `continuers_detected` (from
  `decide_barge_action` returning `FINISH`) is the follow-on lap.
- 20 unit tests (`tests/unit/test_emit_organic_block.py`): field defaults;
  `OrganicStats` defaults; both-zero suppression (the half-duplex guarantee);
  the false-endpoint rate + the 20%-exclusive suggestion boundary; the
  unknown-`n` count-only path; continuers-alone; both-signals-under-one-header;
  and `print_session_summary` wiring (half-duplex summary has no organic block,
  false-endpoint/continuer turns surface, per-turn `print()` emits/omits the
  lines).
- **Next:** populate `false_endpoint` / `continuers_detected` from the live
  organic path (the follow-on — set them when an EOU decision is contradicted
  by resumed speech, and when `decide_barge_action` returns `FINISH`); or wire
  the still-pending `should_abandon_turn` (iter-152) / `should_emit_backchannel`
  (iter-153) seams into `mic_chat`.

### iter-155 (2026-06-16) — utterance buffer-merge decision (#9, Section 4's 2nd half)

- **Shipped backlog #9** — `session/utterance_merging.py`, the pure decision
  seam for the *other* half of Section 4 ("Utterance queueing"). #5 (iter-152)
  covered abandon-vs-finish on the agent side; this covers **buffer/merge**
  on the user side: when a silence endpoint fires on unfinished-looking text
  and a continuation arrives within the pause window, the prior endpoint was a
  **false positive** (the user paused mid-thought, "I was thinking about the…
  [pause] …deadline"). `decide_utterance_continuation(prev_text, next_text,
  gap_secs)` returns `MERGE` (glue the two into one turn) or `NEW` (a genuine
  new turn). Convenience boolean `should_merge_utterance`.
- **Composes two shipped seams:** `text_eou.utterance_completeness` (#4,
  iter-150) scores the prior text's incompleteness; `FullDuplexConfig` (#3,
  iter-151) gates the behavior off by default. Added a fourth sub-flag
  `utterance_merging` (+ `GENO_FULL_DUPLEX_UTTERANCE_MERGING` env var,
  `utterance_merging_active()`, folded into `any_active()`).
- **Two gates, both required (organic mode only).** A merge needs (1) a *quick*
  gap (`gap_secs <= max_gap_secs`, default 2.0s — the `turn_decider`
  silence-floor "a pause, not a turn-end") **and** (2) an *unfinished* prior
  (`utterance_completeness(prev) <= incomplete_ceiling`, default 0.6 — the
  `text_eou` complete-threshold). Only the unfinished-AND-quick corner is a
  false endpoint to repair: a quick gap after a complete sentence is a new
  thought; an unfinished prior after a long gap is an abandoned one. Both stay
  `NEW`.
- **The half-duplex invariant is the whole point.** With a default
  `FullDuplexConfig()` (`utterance_merging` inactive), `decide_utterance_
  continuation` returns `NEW` for *every* input — byte-for-byte today's "each
  endpoint is its own turn" behavior; the prior text isn't even scored. So
  wiring this into the live STT loop (the follow-on) can never regress the
  proven half-duplex path.
- **Asymmetry vs `barge_decision` is deliberate.** There the conservative
  default is `ABANDON` (stay responsive — never trap a user talking over a
  droning agent); here it is `NEW` (never glue two genuinely separate turns
  together). Both err toward today's behavior on ambiguity, but the "safe"
  direction differs by which failure is worse for each path.
- **Directly repairs the `false_endpoint` metric #8 (iter-154) added.** That
  metric *counts* false endpoints; this seam is the decision that would *avoid*
  them once wired in — the measure and the repair were shipped one lap apart.
- 29 unit tests (`tests/unit/test_utterance_merging.py`): the half-duplex
  invariant (default/explicit-default config ⇒ `NEW` across a grid, boolean
  False); the two organic gates (unfinished+quick merges; conjunction/filler
  merges; complete-prior, long-gap, and complete+long all `NEW`); boundaries
  (gap at-max inclusive, just-above exclusive, zero; completeness at-ceiling
  inclusive via comma=0.6, above-ceiling `NEW`); empty/blank/empty-prior
  continuation; sub-flag resolution (master-off+sub-on, master-on+sub-off);
  custom `max_gap_secs` / `incomplete_ceiling` / `eou_config`; purity
  (no input/config mutation, keyword-only config, distinct enum values,
  defaults match sibling seams, boolean matches decide). Plus 8 new
  `test_full_duplex.py` cases for the fourth sub-flag.
- **Next:** wire `should_merge_utterance` into the live STT loop (hold the
  just-finalized text + its silence gap, merge the next chunk before feeding
  the turn engine, and set `TurnMetrics.false_endpoint` when a merge fires —
  closing the iter-154 metric's live-population loop); or wire the still-pending
  `should_abandon_turn` (iter-152) / `should_emit_backchannel` (iter-153) seams.

### iter-156 (2026-06-16) — stateful utterance buffer (the #9 live-loop driver)

- **Shipped the live-loop driver for backlog #9** — `session/utterance_buffer.py`,
  the stateful `UtteranceBuffer` that wraps iter-155's *stateless*
  `decide_utterance_continuation` in the hold-and-merge state a real STT loop
  needs. Every lap since iter-155 named the same follow-on ("wire it into the
  live STT loop: hold the just-finalized text + gap, merge the next chunk, set
  `TurnMetrics.false_endpoint` when a merge fires"); that follow-on needs *state*
  — a held pending, the running merged text, the accumulated false-endpoint flag
  — which a pure decision function deliberately doesn't carry. This module is
  that state, kept pure (no I/O, no clock reads, `gap_secs` injected) so the
  actual audio loop stays a thin driver.
- **The API.** `offer(text, gap_secs)` returns a `BufferResult` — the
  `EmittedTurn`s ready for the engine *now* (usually 0 or 1) plus the text still
  `held`. `flush()` releases a held pending when no continuation arrives (a
  silence longer than `max_gap_secs`, or shutdown). Each `EmittedTurn` carries
  `false_endpoint: bool` — `True` iff that turn absorbed a merged continuation,
  so the caller sets `TurnMetrics.false_endpoint = turn.false_endpoint` and
  iter-154's metric finally populates from the live organic path (the measure
  shipped iter-154, the decision iter-155, the live producer here — three laps
  to close the loop).
- **The half-duplex invariant is the whole point.** With a default
  `FullDuplexConfig()` (`utterance_merging` inactive) the buffer is a
  *transparent zero-latency passthrough*: every `offer` emits its text
  immediately with `false_endpoint=False`, nothing is ever held, `flush` is
  always empty. Byte-for-byte today's "each endpoint is its own turn, fed at
  once" behavior, **no added latency**. Only with merging explicitly on does the
  hold-and-merge machinery engage — and even then *only an unfinished-looking
  utterance is held*; a complete thought emits immediately, so complete turns
  never pay the latency of waiting for a continuation.
- **The merged-flag travels with the pending.** A merge that keeps the running
  text still-unfinished stays held across calls (e.g. "I was thinking about the"
  → merge "the upcoming and" → still unfinished → held again → merge "the launch
  date." → complete → emit). The `false_endpoint` flag accumulates so the
  eventually-released turn — whether released by a later `NEW` arrival or by
  `flush` — reports the repair correctly. Tests pin chained merges and the
  flush-preserves-flag case.
- **`held` is informational only.** The caller doesn't act on `held`; it exists
  so a live loop or test can observe what's being buffered. The actionable output
  is always `turns`.
- 32 unit tests (`tests/unit/test_utterance_buffer.py`): the half-duplex
  passthrough invariant (default / explicit-default / master-on-but-held-back
  configs all emit immediately, never hold, empty flush, even in the exact
  unfinished+quick corner); organic holding (unfinished held, complete emits
  immediately); the merge (glue + single-spaced join + `false_endpoint=True`,
  chained merges accumulate); `NEW` paths (long gap releases prior + emits new,
  complete-then-quick both separate, new-unfinished re-held); flush (releases
  held, preserves merged flag, empty when nothing pending); boundaries (gap
  at-max inclusive / just-above exclusive, completeness at-ceiling via comma,
  empty/None/blank inputs); custom `max_gap_secs` / `incomplete_ceiling` /
  `eou_config` threading; `BufferResult`/`EmittedTurn` contracts (defaults,
  frozen, `.merged`); and purity (independent buffers don't share state). Loaded
  by file path under a stub `session` namespace — the same pipecat-bypass trick
  the sibling seams use.
- **Next:** wire `UtteranceBuffer` into the live `mic_chat` / `pipecat_server`
  STT loop — instantiate from `full_duplex_config_from_env`, route finalized
  transcripts through `offer(text, measured_gap)`, feed the returned `turns` to
  the engine, call `flush()` on a long-silence / shutdown, and set each turn's
  `TurnMetrics.false_endpoint` from `EmittedTurn.false_endpoint`. That's the
  first lap that actually changes a live entrypoint's behavior (still behind the
  off-by-default flag). Or wire the still-pending `should_abandon_turn`
  (iter-152) / `should_emit_backchannel` (iter-153) seams.

### iter-157 (2026-06-16) — merge-depth safety cap on `UtteranceBuffer` (#9 hardening)

- **Hardened the iter-156 driver before live wiring** — added a bounded
  `max_merge_depth` (default `DEFAULT_MAX_MERGE_DEPTH = 8`) to `UtteranceBuffer`.
  iter-156's buffer holds an unfinished pending and lets *chained merges
  accumulate with no upper bound*: a held candidate that keeps looking
  unfinished is re-held on every continuation. In a live loop that's a
  starvation hole — a pathological STT stream that never finalizes a
  complete-looking sentence (or noise the completeness scorer reads as trailing
  off) would let the buffer hold-and-merge *forever*, and the turn engine would
  **never receive the utterance**. This is exactly the kind of unbounded-hold
  that must close *before* the buffer touches a live entrypoint, not after.
- **The cap converts starvation into a bounded, observable delay.** After a
  pending has absorbed `max_merge_depth` continuations, the next merge
  force-emits the running text as a turn (keeping its `false_endpoint=True` flag
  — it repaired real false endpoints on the way up) instead of holding again,
  and the buffer starts fresh. Same role iter-085's `max_token_gap` watch and
  iter-014's rms-empty guard play on their own paths: a backstop for the
  degenerate stream.
- **Zero behavior change in practice.** The default (8) sits well above any
  realistic conversation — a genuine mid-thought pause produces one, occasionally
  two false endpoints per turn, never eight — so the cap never fires and the
  merge behavior is byte-for-byte iter-156's. Pinned by
  `test_below_cap_behaves_like_iter156`. Half-duplex passthrough is entirely
  unaffected (it never holds), pinned by
  `test_cap_irrelevant_in_half_duplex_passthrough`.
- **New observability:** `merge_count` read-only property exposes how many
  continuations the held pending has absorbed (0 when idle), so a live loop /
  test can watch it approach the cap. Resets on every release (NEW arrival,
  flush, or force-emit). A `max_merge_depth < 1` is rejected with `ValueError`
  (a cap below 1 would defeat holding entirely — that's what half-duplex already
  does).
- **+12 unit tests** (44 total in `tests/unit/test_utterance_buffer.py`): default
  value; `merge_count` start/track/reset-on-new/reset-on-flush; force-emit at the
  cap (cap=2 and the cap=1 first-merge corner); force-emitted turn keeps
  `false_endpoint`; below-cap == iter-156; fresh budget after force-emit;
  `ValueError` on cap < 1; half-duplex no-op.
- **Next:** the live wiring is now safe to do — `UtteranceBuffer` self-limits.
  Wire it into the live `mic_chat` / `pipecat_server` STT loop
  (instantiate from `full_duplex_config_from_env`, route finalized transcripts
  through `offer`, feed `turns`, `flush()` on long-silence / shutdown, set
  `TurnMetrics.false_endpoint` from `EmittedTurn.false_endpoint`). Or wire the
  still-pending `should_abandon_turn` (iter-152) / `should_emit_backchannel`
  (iter-153) seams.

### iter-158 (2026-06-16) — cross-turn utterance aggregator (#9 live-loop, second half)

- **Shipped the last pure piece before the entrypoint wiring** —
  `session/utterance_aggregator.py`. Every lap since iter-155 named the same
  next direction (*wire `UtteranceBuffer` into the live STT loop*) and deferred
  it, because that wiring needs one value the buffer deliberately does not own:
  the **inter-utterance silence gap** the buffer's `offer(text, gap_secs)`
  consumes. The buffer (like the `decide_utterance_continuation` seam beneath
  it) reads no clock and holds no timestamp on purpose — so the gap must be
  measured and injected. Measuring it requires the *one* scalar of cross-turn
  state the buffer can't carry: the **previous utterance's endpoint timestamp**.
  This module is that state.
- **`UtteranceAggregator` owns a `UtteranceBuffer` + `prev_end_at`.**
  `offer(text, speech_start_at, speech_end_at)` computes
  `gap_secs = max(0.0, speech_start_at - prev_end_at)` (the silence since the
  last utterance *ended*, not started), routes `(text, gap_secs)` through the
  buffer, records `speech_end_at` as the new `prev_end_at`, and returns an
  `AggregatedResult` (turns, held, and the measured `gap_secs` so call sites /
  tests see exactly what drove the decision). `flush()` releases any held
  pending and clears `prev_end_at` so the next utterance starts a fresh
  conversation (gap `inf` ⇒ a genuine new turn).
- **The two timestamps are already on the recorder.**
  `record_utterance_streaming` surfaces `speech_start_at` via `out_metrics`
  (iter-082), and `_chat_loop` already computes `speech_ended_at` (the
  last-speech frame). So the eventual entrypoint wiring stays a thin driver: it
  hands the aggregator the two stamps the recorder produces and feeds the
  returned turns to the engine. The aggregator does the *subtraction* — the one
  stateful step the buffer can't, because it has to remember the prior endpoint.
- **First utterance / post-flush gap is `float('inf')`** (no prior endpoint) —
  nothing to merge with, so the buffer treats it as a fresh candidate. A
  negative raw gap (clock-skew across the recorder's frame clock — next start
  stamped before prior end) is **clamped to `0.0`**, mirroring `_chat_loop`'s
  defensive clamps (TTC, eot_overhead); zero/negative silence is the strongest
  mid-thought-pause signal, so reading it as "quick" is correct.
- **Half-duplex invariant flows through end-to-end.** With a default
  `FullDuplexConfig()` the underlying buffer is a transparent passthrough, so
  the aggregator emits every utterance immediately with `false_endpoint=False`
  and never holds — the gap is measured and reported but never changes the
  output. Byte-for-byte today's behavior, zero added latency.
- **Injected-buffer seam** for tests / advanced wiring (mutually exclusive with
  the construction tuning args — the ambiguous combination raises `ValueError`
  rather than silently ignoring one). Construction args (`config`, `eou_config`,
  `max_gap_secs`, `incomplete_ceiling`, `max_merge_depth`) otherwise thread
  through to a freshly-built buffer.
- **+27 unit tests** (`tests/unit/test_utterance_aggregator.py`): half-duplex
  passthrough (emit-immediately, never-hold-even-in-the-merge-corner, empty
  flush, explicit-default); gap computation (first-offer `inf`, start−prev_end,
  uses-prev-*end*-not-start, negative clamp, `prev_end_at` updates each offer);
  organic merge driven by the measured gap (quick-merges, long-gap-releases-both,
  complete-prior-emits, flush-releases-held, flush-resets-to-inf,
  flush-gap-inf); chained accumulation; tuning threading (`max_gap_secs`,
  `max_merge_depth`, `incomplete_ceiling`); injected buffer (used / rejected
  with config / rejected with eou_config); `AggregatedResult` contract
  (defaults, frozen, `.merged`); purity (independent aggregators isolated);
  empty-text handling.
- **No runtime behavior change.** The module is unwired; nothing in the live
  `mic_chat` / `mic_talk` / `pipecat_server` path imports it yet, and even when
  wired the default config makes it a transparent passthrough.
- **Next:** the live STT-loop wiring is now fully unblocked — both pure pieces
  (`UtteranceBuffer` + `UtteranceAggregator`) exist. Wire `UtteranceAggregator`
  into `_chat_loop` / `pipecat_server`: instantiate from
  `full_duplex_config_from_env`, call `offer(transcript, speech_start_at,
  speech_ended_at)` per finalized utterance, feed `result.turns` to the engine,
  call `flush()` on long-silence / shutdown, and set `TurnMetrics.false_endpoint`
  from each turn's flag — closing iter-154's metric live-population loop on the
  user side. Or wire the still-pending `should_abandon_turn` (iter-152) /
  `should_emit_backchannel` (iter-153) seams.

### iter-159 (2026-06-16) — live `ChatLoop` wiring of the aggregator (#9 entrypoint)

- **Wired the organic aggregator into the live turn loop** — the follow-on every
  lap since iter-155 named. Both pure pieces existed (`UtteranceBuffer` +
  `UtteranceAggregator`); this lap connects them to `ChatLoop.run_one_turn`
  behind a new **off-by-default `aggregator=None` seam**. With `aggregator=None`
  (the default, and what every existing call site passes) the loop is
  byte-for-byte the pre-iter-159 path. `mic_chat.run_chat` now constructs an
  `UtteranceAggregator(config=full_duplex_config_from_env())`, so the live chat
  uses it — but with the `GENO_FULL_DUPLEX*` flags unset the config is
  half-duplex and the buffer is a transparent passthrough, so behavior is
  unchanged until merging is explicitly enabled.
- **The impedance mismatch is isolated in a pure helper.** `run_one_turn` is
  single-utterance-in / single-response-out; the aggregator may hold (0 turns)
  or release several at once. `examples/_chat_aggregation.py::resolve_turn`
  folds an `AggregatedResult` into one `ResolvedTurn(respond, text,
  false_endpoint, held)`: no turns ⇒ `respond=False`; one+ turns ⇒ join their
  non-empty texts (a long-gap release glues the held fragment onto the new turn
  rather than dropping it), `false_endpoint` is the OR across released turns.
  Kept pure + dependency-free (duck-typed over `.turns`/`.held`) so it loads
  without `session/__init__`'s eager pipecat import — the sibling-seam trick.
  This is the track's rhythm: extract the policy-laden fold, keep the loop a
  thin consumer.
- **Two new live branches, after STT, before the LLM stream.**
  - **HELD** (`respond=False`): the utterance looks mid-thought and was buffered.
    `run_one_turn` returns no-metrics (same shape as a false trigger), so
    `run_session` re-listens without consuming the turn counter — and crucially
    without opening an LLM stream for half an utterance. A dim status line shows
    the held text + measured gap.
  - **RELEASED** (`respond=True`): respond to the (possibly merged) text and set
    `metrics.false_endpoint = resolved.false_endpoint`, **populating iter-154's
    metric from the live path** — the false-endpoint rate is now measured, not
    just asserted by the seam tests.
- **Timestamps come from the recorder, gap math from the aggregator.** The loop
  hands `offer(transcript, speech_start_at, speech_ended_at)` the two stamps the
  recorder already produces (`speech_start_at` via `out_metrics`/iter-082,
  `speech_ended_at` = `clock() - silence_duration`). Both share `self._clock`;
  the aggregator clamps any negative frame-clock-skew gap. `speech_start_at`
  falls back to `speech_ended_at` on the (unreachable-for-non-empty-transcript)
  path where the recorder didn't latch a speech frame.
- **+16 tests.** `tests/unit/test_chat_aggregation.py` (+11): held/no-held,
  single-turn (plain + merged false_endpoint + held-passthrough), multi-turn
  (join-with-space, false_endpoint OR, strip+drop-empty, all-empty-collapse),
  `ResolvedTurn` frozen + defaults. `tests/unit/test_chat_loop_aggregator.py`
  (+5): drives the *real* `ChatLoop.run_one_turn` with a real aggregator +
  virtual audio + a manual clock — aggregator=None unchanged, half-duplex
  passthrough responds-immediately, organic hold (no metrics, text buffered, no
  user message appended), organic merge (joined text fed to LLM +
  `false_endpoint=True`), complete-utterance emits immediately.
- **Verification:** full unit suite **2145 passed** (2129 prior + 11 + 5);
  integration **30 passed, 1 skipped**; `py_compile` clean.
- **Next:** call `aggregator.flush()` on a long inter-turn silence (the user
  trailed off and stopped after a held fragment) and at session shutdown, so a
  held pending isn't stranded — feed the flushed turn to the engine. This needs
  a hook in `run_session` (which owns the inter-turn boundary) rather than
  `run_one_turn` (which only sees one utterance). Then wire the still-pending
  `should_abandon_turn` (iter-152) / `should_emit_backchannel` (iter-153) seams
  behind their sub-flags.

### iter-160 (2026-06-16) — flush the aggregator on shutdown (#9 follow-on)

- **Closed the one path `offer` can never reach: shutdown with a held pending.**
  iter-159 wired the aggregator into `run_one_turn`, but a *held* mid-thought
  fragment only ever surfaces when the **next** utterance arrives — its measured
  gap forces a `NEW` release inside `offer`. If the user trails off after a
  fragment, never speaks again, and hits Ctrl+C, that text sits in the buffer's
  `_pending` and is silently lost. `run_session` now calls `aggregator.flush()`
  on exit (after the `KeyboardInterrupt`), resolves the released text through the
  same `resolve_turn` helper, and records it on the new
  `SessionState.stranded_utterance`.
- **The hook lives in `run_session`, not `run_one_turn`.** `run_one_turn` only
  sees one utterance; the inter-turn / shutdown boundary is owned by
  `run_session`. This matches iter-159's "Next" prediction exactly. The flush is
  purely additive — the session is ending, so the stranded text is *surfaced*
  (in the summary) rather than responded to.
- **Surfaced, not dropped.** `mic_chat.run_chat` passes the aggregator to
  `run_session` and threads `state.stranded_utterance` into
  `SessionMeta.stranded_utterance`. `print_session_summary` emits a
  `_emit_stranded_utterance_line` — suppressed (the common case) unless a
  fragment was actually held, and surfaced on **both** the normal path and the
  no-completed-turns early return (a user can strand a fragment without ever
  completing a turn). Mirrors the iter-114+ session-summary line conventions:
  names `iter-160` in the text for `grep`, defensive suppression on
  None/blank/half-duplex.
- **Defensive:** a misbehaving aggregator's `flush()` exception is swallowed so
  it can never mask the summary the operator is about to read.
- **Half-duplex unchanged.** A default `FullDuplexConfig` never holds, so `flush`
  releases nothing and `stranded_utterance` stays `None` — byte-for-byte today's
  behavior. `aggregator=None` (every existing call site) is likewise untouched.
- **+19 tests.** `tests/unit/test_chat_session.py` (+8): no-aggregator,
  flush-always-called, held-fragment-recorded, blank-collapse, exception-
  swallowed, recorded-after-completed-turns, plus two **real**-`UtteranceAggregator`
  integrations (organic strands a held fragment on exit; half-duplex never
  strands). `tests/unit/test_stranded_utterance_line.py` (+11): helper
  suppression (None/empty/whitespace), emission/formatting (quoted, stripped,
  names iter-160), and `print_session_summary` integration on both the normal
  and zero-turn paths.
- **Verification:** full unit suite **2164 passed** (2145 prior + 8 + 11);
  integration **30 passed, 1 skipped**; `py_compile` clean.
- **Next:** also flush on a long *mid-session* inter-turn silence (not just
  shutdown) so a trailed-off fragment is fed to the engine as its own turn before
  the user starts a genuinely new thought — needs `run_session` to measure the
  inter-turn gap (today it doesn't read a clock between turns). Then wire the
  still-pending `should_abandon_turn` (iter-152) / `should_emit_backchannel`
  (iter-153) seams behind their sub-flags.

### iter-161 (2026-06-16) — held utterances ≠ VAD false triggers (#9 metric fix)

- **Fixed a metric-correctness bug iter-159's wiring introduced.** When the
  organic aggregator *holds* a mid-thought utterance, `run_one_turn` returns a
  no-metrics `TurnResult` (so `run_session` re-listens for the continuation).
  But `run_session` counted *every* no-metrics, no-error turn as a **VAD false
  trigger** (`state.false_triggers += 1`). A held utterance is the *opposite* of
  a false trigger: the transcript was captured fine and is being deliberately
  buffered for a merge. So whenever utterance-merging was on, every mid-thought
  hold silently inflated the false-trigger rate the iter-048 summary reports —
  making the VAD look noisier than it is and burying the organic buffer's real
  activity inside an unrelated metric.
- **The fix — a `held` flag on the no-metrics path.**
  - `TurnResult` gains `held: bool = False` (`examples/_chat_loop.py`). The
    held-branch return in `run_one_turn` now sets `held=True`; every other
    no-metrics return (no-transcription, too-short) leaves it `False`.
  - `run_session` (`examples/_chat_session.py`) reads it defensively
    (`getattr(result, "held", False)` — a pre-iter-161 `TurnResult` shape with
    no field still counts as a false trigger) and routes a held turn to a new
    `SessionState.utterances_held` counter instead of `false_triggers`. Both
    paths still re-listen without consuming the turn counter — only the
    *attribution* changes.
  - The count threads `SessionState.utterances_held` → `SessionMeta.utterances_held`
    → `OrganicStats.utterances_held` → a new "Utterances held" line in
    `_emit_organic_block` (`examples/_chat_metrics.py`), surfaced on both the
    normal and the zero-completed-turns early-return paths (a session can hold
    a fragment yet complete no turns). The line names the count as
    "buffered for merge — not VAD false triggers" so the distinction is explicit
    in the summary.
- **Half-duplex / no-aggregator unchanged.** A default `FullDuplexConfig` never
  holds, so `held` is always `False`, `utterances_held` stays 0, and the organic
  block is suppressed — byte-for-byte today's summary. `aggregator=None` (every
  existing call site) is likewise untouched.
- **+12 tests.** `tests/unit/test_chat_session.py` (+4): held bumps
  `utterances_held` not `false_triggers` (same prompt cadence); held + a genuine
  false trigger counted separately; a result object lacking `held` defaults to
  false-trigger (back-compat); `utterances_held` defaults 0.
  `tests/unit/test_chat_loop_aggregator.py` (+3 assertions): held path sets
  `held=True`; no-aggregator and half-duplex paths leave `held=False`.
  `tests/unit/test_emit_organic_block.py` (+8): `OrganicStats.utterances_held`
  default; held-alone emits the block + line; held=0 omits; held doesn't emit
  the false-endpoint/continuer lines; held + other signals share one header;
  `print_session_summary`/`SessionMeta` wiring on the normal + zero-turn paths;
  and the headline guarantee — a held utterance does **not** appear in the VAD
  false-trigger line.
- **Verification:** full unit suite **2176 passed** (2164 prior + 12);
  integration **30 passed, 1 skipped**; `py_compile` of the four touched modules
  clean.
- **Next:** the mid-session long-silence flush iter-160 named (feed a trailed-off
  fragment to the engine as its own turn before a genuinely new thought; needs an
  inter-turn clock read in `run_session`). Then wire the still-pending
  `should_abandon_turn` (iter-152) / `should_emit_backchannel` (iter-153) seams
  behind their sub-flags.

### iter-162 (2026-06-17) — displaced fragments not glued onto the response (#9 correctness fix)

- **Fixed a live correctness bug iter-159's wiring exposed.** In organic mode,
  when the user trails off mid-thought ("I was thinking about the") and the
  aggregator holds it, then — after a long silence (> `max_gap_secs`) that
  proves the held text was *not* a false endpoint — speaks a genuinely new
  utterance ("What time is it?"), `UtteranceBuffer.offer` releases **two**
  distinct turns in a single call: the abandoned fragment (released as its own
  `NEW` turn) *and* the new utterance. iter-159's `resolve_turn` space-glued the
  whole `turns` list into one string and fed `"I was thinking about the What
  time is it?"` to the LLM — a garbled, semantically-wrong prompt.
- **Root cause:** `resolve_turn` treated *every* multi-turn release as a
  continuation to join. But the buffer only ever emits >1 turn when a measured
  silence forced a `NEW` boundary — i.e. precisely the case where the turns are
  **distinct**, not a continuation. A genuine merge is a *single* released turn
  (the running text glued internally, then emitted once it looks complete), so
  joining at the `resolve_turn` layer was never correct.
- **Fix (`examples/_chat_aggregation.py`):** respond to the **last** released
  turn only; surface the earlier ones as `ResolvedTurn.displaced` (a tuple, in
  order). `false_endpoint` now reflects the *responded* turn's own flag, not an
  OR across the abandoned fragments (an abandoned merged fragment must not
  falsely stamp a fresh, non-merged response). Single-turn releases (the common
  case, and every half-duplex release) leave `displaced` empty — byte-for-byte
  the prior behavior.
- **Surfacing — the mid-session analog of iter-160's shutdown `stranded_utterance`:**
  - `TurnResult.displaced: tuple[str, ...]` (`examples/_chat_loop.py`) carries
    the abandoned fragments out of `run_one_turn`, on both the responded and the
    LLM-error returns (the displaced text is real captured speech, independent
    of whether the response then succeeded).
  - `run_session` (`examples/_chat_session.py`) extends
    `SessionState.utterances_displaced` (a list, in order) from every turn,
    read defensively via `getattr` so a pre-iter-162 `TurnResult` shape
    contributes nothing.
  - Threads `SessionState.utterances_displaced` → `SessionMeta.utterances_displaced`
    → a new "Displaced uttr." line via `_emit_displaced_utterances_line`
    (`examples/_chat_metrics.py`), surfaced on both the normal and the
    zero-completed-turns early-return paths. One fragment ⇒ a single quoted
    line; multiple ⇒ a count header + one quoted line each. The line names
    `iter-162` for `grep`.
- **Half-duplex / no-aggregator unchanged.** The buffer never releases more than
  one turn at a time there, so `displaced` is always empty, the new counter
  stays empty, and the summary line is suppressed — byte-for-byte today's
  output. `aggregator=None` (every existing call site) is untouched.
- **+23 tests.** `tests/unit/test_chat_aggregation.py` (rewrote the multi-turn
  block + defaults): respond-to-last-not-joined (the headline garble repro);
  single-turn release has no displaced; `false_endpoint` is the responded turn's
  own flag (both directions); three-turn release displaces all but the last;
  blank-turn handling; all-blank collapse. `tests/unit/test_chat_loop_aggregator.py`
  (+1 test, +3 assertions): the live long-silence-displaces-fragment path
  (responds to the new utterance, not the glue; fragment rides out on
  `displaced`); the genuine-merge path asserts `displaced == ()`; no-aggregator
  and half-duplex assert empty. `tests/unit/test_chat_session.py` (+5):
  collection from a successful turn, accumulation in order across turns,
  collection even when the turn errored, back-compat default-empty, default-empty
  clean session. `tests/unit/test_displaced_utterances_line.py` (new, +14):
  helper suppression (None/empty/all-blank), single-fragment formatting (quoted,
  stripped, names iter-162, explains abandoned-mid-thought), multi-fragment
  (header + per-line, blanks dropped from count), and `print_session_summary`
  integration on the normal + zero-turn paths.
- **Verification:** full unit suite **2199 passed** (2176 prior + 23);
  integration **30 passed, 1 skipped**; `py_compile` of the five touched modules
  clean.
- **Next:** the mid-session long-silence *flush* iter-160/161 named still stands
  as the larger item (feed a trailed-off fragment to the engine as its own turn
  via an inter-turn clock read in `run_session`, rather than only surfacing it
  in the summary) — this lap makes the *displaced*-fragment half correct first.
  Then wire the still-pending `should_abandon_turn` (iter-152) /
  `should_emit_backchannel` (iter-153) seams behind their sub-flags.

### iter-163 (2026-06-17) — merge-depth cap force-emit is a distinct signal (#9 observability fix)

- **Made the iter-157 starvation backstop observable instead of silent.** The
  `max_merge_depth` cap (default 8) exists so a pathological "unfinished forever"
  stream can't hold an utterance indefinitely: once a held pending has absorbed
  N continuations and the running text *still* looks mid-thought, the buffer
  **force-emits** it rather than holding again. Before this lap that force-emit
  was indistinguishable from a clean merge — it set `false_endpoint=True` and
  was counted in the same summary line as genuine repairs. An operator watching
  the session summary had no way to see the backstop fire, which is exactly the
  "no silent caps" discipline this track keeps flagging.
- **Fix — a distinct `merge_capped` flag threaded the full data-flow chain:**
  - `EmittedTurn.merge_capped: bool` (`session/utterance_buffer.py`) — set
    `True` only at the cap force-emit site. Always paired with
    `false_endpoint=True` but semantically distinct (a backstop firing, not a
    clean repair). `BufferResult.capped` is the any-over-turns roll-up.
  - `AggregatedResult.capped` (`session/utterance_aggregator.py`) mirrors the
    buffer property end-to-end.
  - `ResolvedTurn.merge_capped` (`examples/_chat_aggregation.py`) carries the
    **responded** turn's own flag (mirroring iter-162's `false_endpoint` rule —
    a capped *displaced* fragment must not stamp a fresh, uncapped response).
    Read via `getattr` so a pre-iter-163 turn double defaults to `False`.
  - `TurnMetrics.merge_capped` (`examples/_chat_metrics.py`) stamped in
    `ChatLoop.run_one_turn` (`examples/_chat_loop.py`), which also prints a
    runtime `merge-depth cap hit` status line distinct from the natural-merge
    line.
- **Surfacing:** `OrganicStats.merges_capped` + a new "Merges capped" line in
  `_emit_organic_block` (suppressed at zero, like every sibling signal), naming
  `iter-157` so an operator who sees it can find the cap's context. The per-turn
  `TurnMetrics.print` "False endpoint: yes" line now annotates *merge-depth cap
  hit — force-emitted mid-thought* when capped, vs *user wasn't done* otherwise,
  so the distinction survives into per-turn replay too.
- **Half-duplex / no-aggregator unchanged.** A passthrough buffer never holds,
  so the cap can never fire: `merge_capped` is always `False`, the counter stays
  zero, the line is suppressed — byte-for-byte today's output. `aggregator=None`
  is untouched.
- **+24 tests.** `test_utterance_buffer.py` (+8): cap sets `merge_capped`,
  natural merge doesn't, `BufferResult.capped` roll-up, defaults.
  `test_utterance_aggregator.py` (+5, +1 fixed): `AggregatedResult.capped`
  default/reflect, cap path end-to-end, natural release not capped, half-duplex
  never capped; fixed `test_max_merge_depth_threads_through` to expect
  `merge_capped=True`. `test_chat_aggregation.py` (+6): responded-turn's-own-flag
  (both directions), displaced-capped-fragment doesn't stamp response,
  back-compat `getattr` default, defaults. `test_chat_loop_aggregator.py` (+1
  class, +3 assertions): the live cap force-emit stamps `TurnMetrics.merge_capped`
  (`max_merge_depth=1` aggregator); no-aggregator/half-duplex/natural-merge all
  assert `merge_capped is False`. `test_emit_organic_block.py` (+8): the
  "Merges capped" line (alone, zero-suppressed, with other signals), the
  `TurnMetrics`/`OrganicStats` defaults, and `print_session_summary` wiring
  (capped turn surfaces both the false-endpoint rate and the capped line; an
  uncapped false endpoint omits the capped line).
- **Verification:** full unit suite **2225 passed** (2199 prior + 26 net new
  test functions); integration **30 passed, 1 skipped**; `py_compile` of the
  five touched modules clean.
- **Next:** the mid-session long-silence *flush* iter-160/161 named still stands
  as the larger item (feed a trailed-off fragment to the engine as its own turn
  via an inter-turn clock read in `run_session`). Then wire the still-pending
  `should_abandon_turn` (iter-152) / `should_emit_backchannel` (iter-153) seams
  behind their sub-flags.

### iter-164 (2026-06-17) — mid-session long-silence flush decision (#9, the deferred half)

- **Shipped the pure decision seam** every lap since iter-160 named as the next
  direction: `session/silence_flush.py`,
  `decide_silence_flush(*, held_text, silence_secs, config, max_gap_secs)` →
  `FLUSH` / `HOLD`, plus the `should_flush_held_utterance(...)` boolean mirror.
- **The blind spot it closes.** The `UtteranceBuffer` only releases a held
  mid-thought fragment when the *next utterance arrives* — `offer` measures the
  gap that preceded that utterance and, if long, releases the held fragment as a
  displaced `NEW` turn (iter-162). The one case `offer` can never reach is the
  user who trails off mid-thought and then says **nothing** for a long beat:
  there is no next utterance to drive a release, so the fragment sits held until
  a genuinely-new thought finally displaces it or shutdown flushes it
  (iter-160). Both are too late — the user paused, waited, and the agent stayed
  mute on a fragment it could have answered.
- **The boundary mirrors the merge window exactly.** `FLUSH` iff
  `silence_secs > max_gap_secs` (default 2.0s) — the *same scalar*
  `decide_utterance_continuation` uses as its "quick gap" gate. At exactly
  `max_gap_secs` a continuation would still `MERGE` (rule 3 is `gap <=
  max_gap_secs`), so the flush must still `HOLD` at the boundary; only strictly
  beyond it is the window provably closed. A dedicated test
  (`test_flush_boundary_is_exactly_the_merge_window`) pins both seams to agree
  so a future edit to one can't silently desync the flush deadline from the
  merge window.
- **Half-duplex invariant.** With a default `FullDuplexConfig()`
  (`utterance_merging` inactive) the decision is `HOLD` for every input — and in
  that mode the buffer never holds a fragment anyway, so there is nothing to
  flush. Byte-for-byte today's behavior. Only with merging explicitly on can a
  held fragment exist and a long silence flush it.
- **Decision-seam-first rhythm.** Like iter-152 (`decide_barge_action` →
  coordinator wiring) and iter-153 (`decide_backchannel_timing` → cue-path
  wiring), this lap ships the pure, exhaustively-tested decision and leaves the
  live wiring as the explicit follow-on: `run_session` needs an inter-turn clock
  read (it reads no clock between turns today) to measure `silence_secs` and
  call `should_flush_held_utterance` before re-listening, then respond to the
  flushed fragment as its own turn.
- **+20 tests** (`tests/unit/test_silence_flush.py`): the half-duplex invariant
  (default config never flushes, grid + boolean + explicit-default-same-as-None);
  the organic window gate (long silence flushes, just-over flushes, at-boundary
  holds, just-under holds, zero holds, custom `max_gap_secs` tracks); the
  nothing-held gate (empty/whitespace/None held all hold); sub-flag resolution
  (merging sub-flag on with master off flushes; sub-flag explicitly off holds);
  the boolean mirror; and the merge-window-agreement cross-check.
- **Verification:** `test_silence_flush.py` **20 passed**; full unit suite
  **2245 passed** (2225 prior + 20 net new); integration **30 passed, 1
  skipped**; `py_compile` of `session/silence_flush.py` clean.
- **Next:** wire `should_flush_held_utterance` into `run_session`'s inter-turn
  path (the live half of this seam — measure the silence since the buffer last
  held, flush + respond to the fragment as its own turn before the next
  `[N] waiting...`). Then the still-pending `should_abandon_turn` (iter-152) /
  `should_emit_backchannel` (iter-153) coordinator/cue wirings.

### iter-165 (2026-06-17) — pre-speech idle timeout on the recorder (#9, the flush mechanism)

- **Shipped the recorder mechanism** the iter-164 flush decision needs:
  `record_utterance_streaming` grows an optional `idle_timeout` arg. The blocker
  every lap since iter-160 named was *"`run_session` reads no clock between
  turns"* — but the deeper cause is that `record_utterance_streaming` **blocks
  forever waiting for speech to start**. There is no point at which the loop
  regains control during a long inter-turn pause, so a held mid-thought fragment
  (iter-156) can only ever surface when the *next* utterance arrives (iter-162)
  or at shutdown (iter-160). iter-164 shipped the pure *decision*
  (`decide_silence_flush`); this lap ships the *mechanism* that lets the loop
  reach that decision mid-session.
- **What it does.** With `idle_timeout=N` set, the recorder returns the
  empty-utterance `(b"", 0.0, 0.0)` tuple — the same shape a VAD false trigger /
  too-short utterance returns — after `N` seconds of unbroken *pre-speech*
  silence, instead of blocking. The timeout is measured off the same
  frame-aligned virtual clock (`now - t_origin`) as everything else, so it is
  deterministic under the test `FrameClock` and tracks wall time in production.
- **Gated on `first_speech_at is None`.** The timeout only governs the
  *pre-speech* wait. Once a speech frame is seen, the normal `silence_duration`
  end-of-turn logic owns the rest of the utterance and the idle timeout no
  longer applies — so a long mid-utterance trailing pause can never be
  misread as an idle timeout (a dedicated test pins this: 2.0s trailing silence
  with `idle_timeout=0.5` still produces a real transcript via DONE_OK).
- **Distinct cause flag.** On a timeout the recorder sets
  `out_metrics["idle_timed_out"] = True` so the live loop can tell a deliberate
  idle timeout (the inter-turn-flush trigger) apart from a genuine VAD false
  trigger — both return the same empty tuple, but only one should drive a flush.
- **`None` (default) preserves wait-forever behavior byte-for-byte.** No call
  site passes `idle_timeout` yet, so the recorder waits for speech exactly as
  before. The new path is entirely opt-in.
- **+7 tests** (`tests/unit/test_chat_recording.py`,
  `TestRecordUtterancePreSpeechIdleTimeout`): default-None records normally (no
  flag); timeout fires on pure pre-speech silence (empty tuple + flag set);
  transcribe_fn never called on timeout; speech-before-timeout records normally;
  the mid-utterance-trailing-pause gate; the timeout fires near the configured
  window (`mic.reads` count); and `idle_timeout=0.0` fires on the first frame
  (distinct from `None`).
- **Verification:** `test_chat_recording.py` **21 passed** (14 prior + 7 net
  new); full unit suite **2252 passed** (2245 prior + 7); integration **30
  passed, 1 skipped**; `py_compile -W error::SyntaxWarning` of
  `examples/_chat_recording.py` clean.
- **Next:** wire `idle_timeout` through `ChatLoop` and `run_session`'s
  inter-turn path so a held fragment's silence is measured and
  `should_flush_held_utterance` (iter-164) drives a real mid-session flush —
  the live half of backlog #9's still-deferred flush, now unblocked on the
  recorder side. Then the still-pending `should_abandon_turn` (iter-152) /
  `should_emit_backchannel` (iter-153) coordinator/cue wirings.

### iter-166 (2026-06-17) — wire the idle timeout through ChatLoop (#9, the wiring's first hop)

- **Threaded iter-165's recorder `idle_timeout` through `ChatLoop`.** iter-165
  added the pre-speech `idle_timeout` mechanism to `record_utterance_streaming`
  but left it inert — no call site passed it. This lap takes the first of the
  two named wiring hops: `ChatLoop.__init__` grows an `idle_timeout` arg
  (default `None`), threads it into the `record_utterance_streaming` call, and
  the empty-wav return path now reads the recorder's
  `rec_metrics["idle_timed_out"]` side-band flag onto a new
  `TurnResult.idle_timed_out` field.
- **Three distinguishable no-metrics causes.** A turn that produces no metrics
  is now one of: a held mid-thought fragment (`held=True`, iter-161), a
  deliberate pre-speech idle timeout (`idle_timed_out=True`, this lap), or a
  genuine VAD false trigger (neither flag). The next hop — `run_session` — needs
  exactly this distinction: only the idle-timeout cause should drive a
  mid-session flush of a held fragment; a false trigger should just re-listen.
- **`None` (default) keeps the proven path byte-for-byte.** No production call
  site passes `idle_timeout`, so the recorder still waits forever for speech and
  `idle_timed_out` is always `False`. The new behavior is entirely opt-in and
  only meaningful alongside an injected aggregator with merging on.
- **+4 tests** (`tests/unit/test_chat_loop.py`, `TestIdleTimeoutWiring`):
  default-None waits for speech and never flags; `idle_timeout` set fires on
  pure pre-speech silence (no-metrics TurnResult flagged, LLM never reached,
  speaker never opened, no user message appended); speech-before-timeout records
  normally and stays unflagged; a real VAD false trigger (no `idle_timeout`)
  returns no metrics with `idle_timed_out=False`, pinning the
  false-trigger-vs-timeout distinction. The timeout fires off the recorder's
  frame-aligned virtual clock, so it is deterministic under the default
  `time.monotonic` (a silence-padding `VirtualMicStream` elapses the virtual
  window within a bounded read count).
- **Verification:** `test_chat_loop.py` **16 passed** (12 prior + 4 net new);
  full unit suite **2256 passed** (2252 prior + 4); integration **30 passed, 1
  skipped**; `py_compile -W error::SyntaxWarning` of `examples/_chat_loop.py`
  clean.
- **Next:** the final hop — `run_session` reads `TurnResult.idle_timed_out`,
  measures the inter-turn silence since the buffer last held, feeds it to
  `should_flush_held_utterance` (iter-164), and responds to the flushed fragment
  as its own turn before the next `[N] waiting...`. That closes backlog #9's
  mid-session flush end-to-end. Then the still-pending `should_abandon_turn`
  (iter-152) / `should_emit_backchannel` (iter-153) coordinator/cue wirings.

### iter-167 (2026-06-17) — flush a held fragment mid-session on idle silence (#9, the wiring's second hop)

- **`run_session` now consumes the `idle_timed_out` flag iter-166 surfaced.**
  iter-166 made a no-metrics turn distinguishable as held / idle-timeout / false
  trigger but left the idle-timeout case as just another re-listen. This lap
  takes the second wiring hop: `run_session` grows two injected params —
  `idle_timeout: Optional[float]` (the recorder's pre-speech window, used only
  as the `silence_secs` fed to the decider — `run_session` reads no clock
  itself) and `flush_decider: Optional[Callable[[str, float], bool]]`. On an
  `idle_timed_out` turn it increments a new `SessionState.idle_timeouts` counter
  (separate from `false_triggers`, so enabling an idle timeout never inflates the
  false-trigger rate) and calls a module-level `_maybe_flush_on_idle` helper.
- **`_maybe_flush_on_idle` is the mid-session flush.** Guards, first-to-bail
  wins: no aggregator → nothing held; no `flush_decider` → caller opted out;
  `aggregator.pending` blank → nothing to flush; `flush_decider(held, silence)`
  says HOLD (the merge window hasn't closed — e.g. an `idle_timeout` shorter than
  `max_gap_secs`) → return. Otherwise it calls `aggregator.flush()`, runs the
  released text through `resolve_turn`, and records it on a new
  `SessionState.flushed_utterances` list. Both the decider call and the flush are
  wrapped in `try/except` — a misbehaving injected callable must not break the
  live loop. When `idle_timeout` is `None` the silence fed to the decider falls
  back to `inf` (the timeout demonstrably fired, so let the decider's own
  `> max_gap_secs` gate decide).
- **Decoupled from the `session` package exactly like `trim_messages`.** The
  flush *decision* is the injected `flush_decider` callable; production
  (`mic_chat.py`, organic mode only) binds it to
  `session.silence_flush.should_flush_held_utterance` against the *same*
  `full_duplex_config` the aggregator runs with (so the half-duplex gate +
  `max_gap_secs` window agree with the buffer), and sources `idle_timeout` from
  `chat.idle_timeout` (default `DEFAULT_MAX_GAP_SECS`). Half-duplex wires neither
  (both `None`) → the proven wait-forever path is byte-for-byte unchanged.
- **`flushed_utterances` surfaces in the summary.**
  `_emit_flushed_utterances_line` (mirroring iter-162's displaced-utterances
  line) prints a `Flushed uttr.:` line — single fragment quoted inline, multiple
  under an `N fragments flushed on idle silence mid-session — iter-167:` header —
  on both the normal and the no-completed-turns early-return paths.
- **Records, doesn't yet respond.** This lap *records* the flushed fragment so it
  is visible and the buffer is unblocked; producing a *spoken* response to it
  needs a `ChatLoop` text-only response entrypoint (`run_one_turn` always records
  from the mic first), which is the explicit next hop.
- **+17 tests** (`tests/unit/test_chat_session.py`): three-way no-metrics
  counting; back-compat getattr default; defaults zero/empty; no-decider strands
  at shutdown instead of flushing; decider-says-flush records the fragment (and
  sees `(held, silence)`); decider-says-hold; nothing-held; blank-pending; `None`
  idle_timeout → `inf` silence; no-aggregator; flush-releasing-blank;
  decider-exception-swallowed; flush-exception-swallowed; multiple-flushes
  accumulate; and two end-to-end cases driving the real
  `session.silence_flush` (flushes mid-session when silence exceeds the window,
  strands at shutdown when under it). **+14 tests**
  (`tests/unit/test_flushed_utterances_line.py`): suppression / single / multi /
  summary-integration for the new summary line.
- **Verification:** full unit suite **2287 passed** (2256 prior + 31 net new);
  integration **30 passed, 1 skipped**; `py_compile -W error::SyntaxWarning` of
  `examples/_chat_session.py`, `examples/_chat_metrics.py`, `examples/mic_chat.py`
  clean.
- **Next:** the spoken-response hop — a `ChatLoop` text-only response entrypoint
  so a flushed fragment is *answered* as its own turn, not merely recorded. Then
  the still-pending `should_abandon_turn` (iter-152) / `should_emit_backchannel`
  (iter-153) coordinator/cue wirings.

### iter-168 (2026-06-17) — `ChatLoop.respond_to_text`, the spoken-response hop (#9, the wiring's last piece)

The named follow-on from iter-167: a flushed mid-thought fragment could be
*recorded* (`SessionState.flushed_utterances`) but never *answered*, because the
only path into the LLM→TTS machinery was `ChatLoop.run_one_turn`, which always
records from the mic first. This lap gives the flushed text a path straight into
the response half.

- **Extracted `_stream_response`.** The LLM-stream → synth/play → barge-in-watcher
  half of `run_one_turn` (everything after Phase 1's record + STT + organic
  aggregation) is pulled into a private `ChatLoop._stream_response(messages,
  metrics, *, speech_ended_at, stt_time, turn_start, displaced=())`. Phase 1
  locals that leak forward are exactly three (`speech_ended_at`, `stt_time`,
  `turn_start`) plus `metrics` and `displaced` — confirmed by grep before the
  cut — so the seam is narrow. The body is **moved, not modified**: the
  `run_one_turn` path is byte-for-byte the pre-iter-168 inline Phase 2, proven by
  the unchanged 16-test `test_chat_loop.py` suite still passing.
- **`respond_to_text(messages, text)`.** The public entrypoint. Strips `text`
  (blank ⇒ a no-op `TurnResult(metrics=None)` with no history mutation and no
  speaker opened — a caller that flushes an empty fragment is harmless), builds a
  `TurnMetrics(speech_duration=0.0, model=…)` with `transcript=stripped`, anchors
  the response-half clocks at `self._clock()` (no recording happened, so TTFS /
  total_e2e measure from the moment answering starts and `stt_time=0`), and
  delegates to `_stream_response`. User + assistant messages are appended exactly
  as a recorded turn would (popped on LLM error), so conversation history stays
  consistent and the response is spoken through the same worker. `displaced` is
  always empty — a flushed fragment IS the turn, nothing was displaced by it.
- **+7 tests** (`tests/unit/test_chat_loop.py`, `TestRespondToText`): answers text
  and updates history (transcript = the fragment, response spoken, `stt_time` /
  `speech_duration` zeroed, audio reached the speaker); strips whitespace; blank
  and empty text are no-ops (no history mutation, no speaker opened); `displaced`
  always empty; LLM error pops the user message leaving history clean; model
  stamped from config. These tests never push mic audio — `respond_to_text`
  bypasses Phase 1 entirely.
- **Records-vs-responds is now unblocked, but not yet wired.** This lap ships the
  *entrypoint*; `run_session._maybe_flush_on_idle` still only records the flushed
  fragment on `flushed_utterances`. Calling `respond_to_text` from there (so a
  flushed fragment is *spoken* as its own turn, with the turn-counter / metrics
  semantics that implies) is the remaining hop — same decision-seam/mechanism/
  wiring rhythm the whole #9 chain has followed (iter-164→165→166→167→168).
- **Verification:** `test_chat_loop.py` **23 passed** (16 prior + 7 net new);
  full unit suite **2294 passed** (2287 prior + 7); integration **30 passed, 1
  skipped**; `py_compile -W error::SyntaxWarning` of `examples/_chat_loop.py`
  clean.
- **Next:** wire `respond_to_text` into `_maybe_flush_on_idle` so a mid-session
  flushed fragment is answered, not just listed — closing #9's mid-session flush
  end-to-end. Then the still-pending `should_abandon_turn` (iter-152) /
  `should_emit_backchannel` (iter-153) coordinator/cue wirings.

### iter-169 (2026-06-17) — speak the flushed fragment: #9's mid-session flush, end-to-end

The named follow-on from iter-168, and the **last hop of backlog #9's
mid-session flush**. iter-167 wired `run_session` to *flush* a held mid-thought
fragment on a long inter-turn idle silence (recording it on
`SessionState.flushed_utterances`); iter-168 shipped `ChatLoop.respond_to_text`,
the text-only response entrypoint. This lap connects them: a flushed fragment is
now *spoken* as its own turn, not merely listed.

- **`_maybe_flush_on_idle` now returns the flushed text** (was `-> None`). Every
  no-flush path returns `None` (any guard bailed, the decider held, or the
  released turn was empty / no-respond), so a `None` return is the unambiguous
  "do not speak" signal; on a real flush it returns the released text it just
  appended to `flushed_utterances`.
- **`run_session` grows an injected `respond_fn`** — `(messages, text) -> result`
  matching `respond_to_text`. On a flush that returns text *and* a wired
  `respond_fn`, a new module-level `_speak_flushed_fragment` answers the fragment
  through it. `respond_fn=None` (default) keeps the iter-167 record-only path
  byte-for-byte; production (`mic_chat.py`) binds `chat_loop.respond_to_text`
  under the **same organic-mode gate** as `flush_decider` (half-duplex never
  holds, so it never speaks a flush).
- **Shared `_record_completed_turn`.** The success-block bookkeeping (newline →
  `metrics.print(turn+1)` → `all_metrics.append` → `turn += 1` → trim →
  eviction counters) is extracted so a spoken flush is counted *identically* to a
  mic turn. The body is moved, not modified — the mic-turn path is byte-for-byte
  unchanged, proven by the unchanged `test_chat_session.py` cases.
- **Failure degrades to record-only.** `_speak_flushed_fragment` wraps
  `respond_fn` in `try/except`: a raising callable, a `had_error` result, or a
  `metrics=None` result never crashes the live loop — the first two bump
  `llm_errors` (mirroring the mic-turn error path), and the fragment is already
  recorded on `flushed_utterances`. A failed *spoken* response thus falls back to
  exactly the pre-iter-169 record-only behavior.
- **+9 tests** (`tests/unit/test_chat_session.py`): fragment spoken when
  `respond_fn` wired (called with the flushed text, turn counted, counter
  advanced); recorded-not-spoken without `respond_fn`; `respond_fn` not called on
  a HOLD; conversation history threaded through; LLM-error / raised-exception /
  `metrics=None` all degrade to record-only; trim runs after the spoken turn; and
  an end-to-end case with the REAL aggregator + decider + `respond_fn`.
- **Verification:** `test_chat_session.py` **57 passed** (48 prior + 9 net new);
  full unit suite **2303 passed** (2294 prior + 9); integration **30 passed, 1
  skipped**; `py_compile -W error::SyntaxWarning` of `examples/_chat_session.py`,
  `examples/mic_chat.py`, `examples/_chat_loop.py` clean.
- **Next:** backlog #9 is closed end-to-end. The still-pending coordinator/cue
  wirings remain: `should_abandon_turn` (iter-152) into the `mic_chat` barge
  path and `should_emit_backchannel` (iter-153) into the live cue path, both
  behind their full-duplex sub-flags.

### iter-170 (2026-06-17) — `BackchannelMonitor`: the stateful driver for #7

**Branch:** `iter-170-backchannel-monitor` · backlog #7 driver layer.

iter-153 shipped `decide_backchannel_timing` — the *pure decision* for "should
the agent emit a mid-speech backchannel now?" — but it is stateless. It demands
the caller inject three quantities, and one of them, `secs_since_last_backchannel`,
is **not** something a recorder can measure: it depends on the monitor's *own
past EMIT decisions*. A live cue path calling the seam each frame would pass
`secs_since_last_backchannel=None` forever, so the rate limit
(`min_between_cues_secs`, 20s) would never engage and the agent would chatter
"mhmm mhmm mhmm" on every qualifying pause frame.

`session/backchannel_monitor.py` closes that gap — the same seam → stateful
driver step that took iter-155's `decide_utterance_continuation` to iter-156's
`UtteranceBuffer` / iter-158's `UtteranceAggregator`.

- **`BackchannelMonitor.observe(*, now, monologue_start_at, pause_secs)`** derives
  `user_speaking_secs = now - monologue_start_at` and
  `secs_since_last_backchannel = now - last_emit` (`None` until the first emit),
  both clamped `>= 0` against clock skew (mirroring the aggregator / `_chat_loop`
  defensive clamps), routes them through `decide_backchannel_timing`, and — *iff*
  the decision is `EMIT` — records `now` as the new last-emit timestamp and bumps
  `emit_count` before returning a frozen `BackchannelDecision` (the actionable
  `emit` bool plus the derived quantities, for observability).
- **The one stateful step the seam can't do:** remembering its own last emit.
  Recording on EMIT is what makes the rate limit bite across calls — proven by
  `test_repeated_qualifying_pauses_do_not_chatter` (a stream of perfect pauses 1s
  apart fires at 11/16/21/26s, every 5s, not every second).
- **`reset()`** clears the last-emit timestamp so a fresh session / long lull
  starts the rate limit over; the lifetime `emit_count` is intentionally kept.
- **Half-duplex invariant.** Default `FullDuplexConfig()` ⇒ the seam
  short-circuits to `HOLD`, so `emit` is always `False` and the last-emit
  timestamp is *never* set — the monitor is an inert observer. Byte-for-byte
  today's behavior; the emit machinery engages only behind the off-by-default
  `agent_backchannels` flag.
- **+16 tests** (`tests/unit/test_backchannel_monitor.py`): half-duplex inert
  (default / sub-flag-off-under-master-on); sub-flag-on-under-master-off active;
  every gate (warm-up / no-pause / turn-end-gap hold, clause-pause emits);
  rate-limit engages after first emit, clears after the interval, no-chatter
  stream; `reset` clears the limit but keeps the count; negative-duration and
  negative-since-last clamps; decision dataclass echoes inputs + is frozen; and
  the real default 15s warm-up composing through.
- **Verification:** `test_backchannel_monitor.py` **16 passed**; full unit suite
  **2319 passed** (2303 prior + 16 net new); integration **30 passed, 1 skipped**;
  `py_compile -W error::SyntaxWarning` of `session/backchannel_monitor.py` clean.
- **Next:** the live cue-path wiring — feed `BackchannelMonitor.observe` from
  `pipecat_server.py`'s `Broadcaster` (which already owns the monologue clock and
  the `broadcast_cue` path) so a well-timed mid-speech pause emits a cue. Blocked
  on the absent pipecat dep on the x86_64 runner, the same wall #6 hits. The
  sibling `should_abandon_turn` (iter-152) → `mic_chat` barge-path wiring remains
  too, blocked on no transcript being available at VAD-trigger time.

### iter-171 (2026-06-17) — shared cue rotation; the monitor picks *which* cue (#7)

- **Shipped the second piece of cross-event state** the `BackchannelMonitor`
  needs. iter-170 gave the monitor a *when* (record the emit timestamp so the
  rate limit engages); it still had no *which* — `observe` returned no cue at
  all, so a live cue path would know it's time to backchannel but not what to
  play. The cue rotation (`CUE_ROTATION` + a rotating index) was the other piece
  of state the pure `decide_backchannel_timing` seam can't carry, and it already
  existed — buried inside `TurnTakingEngine` as a module-level list and a
  `_cue_index`.
- **`session/cue_rotation.py`** — new dependency-free module, the single source
  of truth: `CUE_ROTATION = ["mhmm", "i_see", "right", "go_on", "mhmm",
  "tell_me_more"]` ("mhmm" twice on purpose — the neutral continuer recurs more
  than the pointed cues) plus a pure total `cue_for_index(i)` that wraps modulo
  length (negatives included, so a monotonic counter never bounds-checks).
- **`BackchannelMonitor`** grows a `_cue_index` advanced *only on a real emit*;
  `observe` returns `BackchannelDecision.cue_type` — the next cue on an emit,
  `None` on a hold (no cue to play). A held frame never burns a rotation slot,
  so the rotation tracks *backchannels emitted*, not *frames observed*. `reset()`
  clears the rate-limit clock but **keeps** the rotation position, so a fresh
  monologue continues the rotation rather than always replaying "mhmm" (a human
  doesn't open every lull with the identical sound). A new read-only `cue_index`
  property exposes the position for observability.
- **`turn_taking.py`** imports `CUE_ROTATION` + `cue_for_index` from the shared
  module (re-exporting `CUE_ROTATION` for back-compat with existing importers)
  and routes `_next_cue` through `cue_for_index` — byte-for-byte the same
  rotation it produced before, now from one source so the silence-driven
  `PLAY_CUE` path and the mid-speech monitor can't drift apart.
- **Half-duplex invariant intact:** a default (half-duplex) monitor returns
  `cue_type=None` for every input and never advances the index — the cue
  machinery, like the emit machinery, engages only behind the off-by-default
  `agent_backchannels` flag.
- **+10 tests:** `test_cue_rotation.py` (5 — non-empty/known-cues, in-order walk,
  modulo wrap, negative indices, mhmm-most-frequent) and 5 net-new in
  `test_backchannel_monitor.py` (emit carries first cue / hold carries None;
  consecutive emits rotate and wrap; held frames don't burn slots; reset keeps
  the rotation position; half-duplex never advances the cue).
- **Verification:** `test_backchannel_monitor.py` + `test_cue_rotation.py`
  **26 passed**; full unit suite **2329 passed** (2319 prior + 10 net new);
  integration **30 passed, 1 skipped**; `py_compile -W error::SyntaxWarning` of
  `session/cue_rotation.py session/backchannel_monitor.py session/turn_taking.py`
  clean.
- **Next:** the monitor now answers both *when* and *which* — the live cue-path
  wiring (feed `BackchannelMonitor.observe` from `pipecat_server.py`'s
  `Broadcaster` and play `decision.cue_type` via `broadcast_cue`) is unchanged as
  the named follow-on, still blocked on the absent pipecat dep on the x86_64
  runner. The sibling `should_abandon_turn` (iter-152) → `mic_chat` barge-path
  wiring also remains, blocked on no transcript at VAD-trigger time.

### iter-172 (2026-06-17) — wire the live `PLAY_CUE` path to the shared rotation; fix two cue bugs (#7)

- **Closed the rotation drift on the *silence-driven* side.** iter-171 unified
  the *pure* cue layer (`session/cue_rotation.py`) so `TurnTakingEngine` and the
  mid-speech `BackchannelMonitor` index one list — but the **live sidecar**
  (`pipecat_server.py`'s `Broadcaster`) never consumed it. It carried two real
  bugs:
  1. The decision branch read `decision.action == Action.play_cue` — but the
     enum member is `PLAY_CUE`. `Action.play_cue` is an **`AttributeError`**, so
     the cue branch *crashed on every transcript frame* (not merely dead — it
     raised inside `process_frame`).
  2. `broadcast_cue()` threw away the engine's rotated `decision.cue` and
     re-picked a **random** cue from a private `CUE_TYPES = ["mhmm", "hmm",
     "okay", "i_see", "right", "go_on"]` list — dropping the rotation *and*
     including cue keys ("hmm", "okay") not in the shared `CUE_ROTATION`.
- **`plan_cue_broadcast(decision, *, trigger_fired=False)`** — new pure,
  dependency-free, file-path-loadable gate in `session/turn_taking.py`. Returns
  the cue to play (the engine's **rotated** `decision.cue.cue_type`) or `None`
  when: the action isn't `PLAY_CUE`, an NLP trigger fired (a continuer would
  step on a warranted real response), or no cue is attached. The live path now
  reads `cue_type = plan_cue_broadcast(decision, trigger_fired=trigger.triggered)`
  and `broadcast_cue(cue_type)` fetches *that* cue (the `/cue/{cue_type}`
  endpoint still randomizes the *take* within the cue's bank, so it doesn't
  sound identical each play). The private `CUE_TYPES` list is gone; `Action` is
  no longer imported by `pipecat_server.py` (the gate moved into the seam).
- **+8 tests** (`tests/unit/test_plan_cue_broadcast.py`, loads `turn_taking` by
  file path into a synthetic `session` namespace like `test_turn_decider.py`):
  PLAY_CUE returns the rotated cue; the `Action.play_cue` typo raises
  (regression pin); trigger suppresses; every non-PLAY_CUE action returns None;
  PLAY_CUE without a cue returns None; default `trigger_fired` allows; the cue
  tracks the shared rotation not a private list; and an end-to-end engine test
  that consecutive backchannel-tier decisions rotate through `CUE_ROTATION` and
  wrap.
- **Verification:** `test_plan_cue_broadcast.py` **8 passed**; full unit suite
  **2337 passed** (2329 prior + 8 net new); integration **30 passed, 1
  skipped**; `py_compile -W error::SyntaxWarning` of `pipecat_server.py
  session/turn_taking.py` clean.
- **Next:** the *silence-driven* cue path is now rotation-correct and crash-free.
  The **mid-speech** `BackchannelMonitor.observe` → `Broadcaster` wiring (play a
  cue *during* long user speech, not only on trailing silence) is still the
  named #7 follow-on, blocked on the absent pipecat dep on the x86_64 runner —
  but `plan_cue_broadcast` is the template that wiring will reuse. The sibling
  `should_abandon_turn` (iter-152) → `mic_chat` barge-path wiring also remains,
  blocked on no transcript at VAD-trigger time.

### iter-173 (2026-06-17) — the monologue clock: the VAD-event driver for #7

**Branch:** `iter-173-monologue-clock` · backlog #7 driver layer (the input
side of `BackchannelMonitor`).

iter-170 shipped the stateful `BackchannelMonitor` and iter-171 gave it the cue
rotation, so it answers both *when* and *which*. But its
`observe(*, now, monologue_start_at, pause_secs)` consumes two quantities the
monitor cannot compute itself — `monologue_start_at` and `pause_secs` — and
**nothing on main derived them.** Neither is a raw clock read:

- A *monologue* is not one VAD speech segment. A person mid-thought pauses at
  clause boundaries ("I was thinking… [0.5s] …about the deadline") without
  yielding the floor — and those brief pauses are *exactly* the backchannel
  opportunities the monitor watches for. So the monologue must survive a short
  `stop → start` gap and reset only on a turn-end-sized one.
- `pause_secs` is the live gap since the last speech frame, zero while talking.

Deriving both from the VAD `started`/`stopped` event stream is cross-event
state the pure seam can't carry — the same gap iter-156's `UtteranceBuffer` /
iter-158's `UtteranceAggregator` filled for the merge seam and iter-170's
`BackchannelMonitor` filled for the emit timestamp.

**What changed:**

1. **`session/monologue_clock.py`** — new pure, dependency-free module.
   `MonologueClock` consumes `on_speech_start(now)` / `on_speech_stop(now)` and
   exposes `monologue_start_at`, `pause_secs(now)`, and read-only `speaking` /
   `last_stop_at` views.
   - **Monologue continuity.** `on_speech_start` begins a *new* monologue (sets
     `monologue_start_at = now`) iff there is no current monologue *or* the gap
     since the last stop is `>= reset_gap_secs`; otherwise the gap is a
     clause-boundary pause and the existing monologue continues untouched, so
     `user_speaking_secs` keeps accumulating across the pause and the warm-up
     gate eventually clears.
   - **The reset threshold is the shared scalar.** `reset_gap_secs` defaults to
     2.0s — *exactly* `backchannel_timing.py`'s `max_pause_secs` and
     `turn_decider.py`'s `silence_floor_secs`. The "is this gap a clause pause
     or a turn-end?" boundary is now one number across the monologue clock, the
     backchannel-timing window, and the turn decider; they cannot drift apart.
   - **Pause deferral.** `on_speech_stop` does *not* end the monologue (a stop
     could be a clause pause or a true turn-end — only the next start, or its
     absence, decides), so the reset decision lives entirely in
     `on_speech_start`. `pause_secs(now)` reports the growing gap meanwhile,
     clamped `>= 0` against clock skew.
   - `reset()` clears everything for a fresh conversation.

2. **+18 tests** (`tests/unit/test_monologue_clock.py`, file-path-loaded into a
   stub `session` namespace like `test_backchannel_monitor.py`): config
   validation (default == shared 2.0s; non-positive rejected); initial state;
   first-start begins monologue; pause grows after stop / clamps on skew /
   zeroes on resume; short gap continues the monologue (single + repeated clause
   pauses); long gap resets; the threshold is inclusive on the reset side
   (exactly 2.0s resets, 1.99s continues); custom threshold; `reset` clears all
   and a post-reset start is always fresh; **two end-to-end integration tests**
   driving a real `BackchannelMonitor` from the clock's outputs — organic mode
   emits "mhmm" exactly in the clause-pause window after warm-up, and the
   half-duplex invariant survives the wiring (default monitor never emits even
   with a perfect clock-reported pause).

**Verification:**
- `python -m pytest tests/unit/test_monologue_clock.py -q` → **18 passed**.
- Full unit suite (in worktree, pre-merge): **2355 passed** (2337 prior + 18).
- Integration (`tests/integration/`): **30 passed, 1 skipped**.
- `python -W error::SyntaxWarning -m py_compile session/monologue_clock.py`
  clean.

**Next:** all three pure pieces of the mid-speech cue path now exist — *when*
(`decide_backchannel_timing`, iter-153), *which* (`cue_rotation` + the monitor's
rotation index, iter-171), and the *inputs* (`MonologueClock`, this lap). The
remaining hop is the live `pipecat_server.py` `Broadcaster` wiring:
`clock.on_speech_start/stop` from the VAD frames, then
`monitor.observe(now, clock.monologue_start_at, clock.pause_secs(now))` →
`broadcast_cue(d.cue_type)` — a thin adapter, still blocked on the absent
pipecat dep on the x86_64 runner. The sibling `should_abandon_turn` (iter-152) →
`mic_chat` barge-path wiring also remains, blocked on no transcript at
VAD-trigger time.
