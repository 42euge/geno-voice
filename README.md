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
  during user speech." Also shipped: the **organic-path naturalness
  metrics** (`examples/_chat_metrics.py`,
  `tests/unit/test_emit_organic_block.py`) — two additive `TurnMetrics`
  fields, `false_endpoint` (the EOU decision fired early and the user
  had more to say) and `continuers_detected` (user backchannels that
  correctly held the agent's floor), surfaced per-turn and as a session
  summary "Organic turn-taking" block (false-endpoint rate + continuers
  held). Both default off and stay zero on the half-duplex path, so the
  block is fully suppressed and today's summaries are byte-for-byte
  unchanged; the track becomes *measured, not asserted* once the seams
  are wired in.
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
