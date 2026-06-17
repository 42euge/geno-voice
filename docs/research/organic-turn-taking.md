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
| 3 | **Full-duplex config flag scaffolding** — a `TurnTakingConfig` / env flag (`GENO_FULL_DUPLEX`) that gates organic behaviors (continuer-aware listening, agent backchannels) off by default, so the half-duplex path is never regressed while the track matures. | Medium | TODO |
| 4 | **Rule-based text EOU precursor** — `is_utterance_complete(text)` that lowers end-of-turn likelihood when the transcript ends in a conjunction / filler / trailing-off marker (mirrors LiveKit turn-detector's linguistic signal; reuses `_TRAILING_PATTERNS`). Feeds #2's confidence. | Medium | TODO |
| 5 | **Continuer-aware barge-in** — wire #1 into `BargeInCoordinator` so a *continuer* utterance ("mhmm") during agent speech does NOT abandon the turn (finish), while a substantive interruption does (abandon). Measure: false-abandon rate. | High | TODO |
| 6 | **Adopt pipecat `smart-turn`** — replace #2's heuristic body with the smart-turn model inside `pipecat_server.py`'s pipeline; same `turn_decider` interface. Measure false-endpoint rate vs silence-only baseline on recorded sessions. | High | TODO (blocked on model + Apple Silicon) |
| 7 | **Agent backchannel emission timing** — a learned/heuristic "good moment to backchannel" signal (Krisp-style) feeding the existing `PLAY_CUE` path, so the agent emits continuers *during* long user speech, not only on silence. | Medium | TODO |
| 8 | **Naturalness metrics for the organic path** — extend `TurnMetrics` / session-summary with false-endpoint rate and continuer-detection counts so the track is measured, not asserted. | Medium | TODO |

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
