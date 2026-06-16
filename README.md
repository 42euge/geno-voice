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
