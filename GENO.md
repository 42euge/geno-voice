# geno-voice — Local Voice Pipeline

Local voice pipeline for offline, privacy-first AI voice interaction. Provides
on-device STT, TTS, and VAD for geno-ecosystem projects that need voice
interaction without cloud APIs.

## Skills

| Skill | Sub-skillset | Slash command |
|-------|-------------|---------------|
| geno-voice | — | — (umbrella) |

## Repo structure

```
geno-voice/
├── GENO.md              # agent instructions (this file)
├── SKILL.md             # umbrella skill manifest
├── genotools.yaml       # geno-tools manifest
├── skills/              # skill definitions
│   └── geno-voice/      #   umbrella
├── stt/                 # speech-to-text pipeline
├── tts/                 # text-to-speech pipeline
├── vad/                 # voice activity detection
├── examples/            # usage examples and integration demos
├── docs/                # MkDocs Material site
└── README.md            # human-readable project overview
```

## Conventions

- Skill directories live under `skills/` and each contains a `SKILL.md`
- The umbrella skill at `skills/geno-voice/SKILL.md` describes the full skillset
- Component pipelines (STT, TTS, VAD) are organized in their own top-level directories

### mic_chat.py extraction pattern

When pulling a subroutine out of `examples/mic_chat.py:run_chat`,
default to this shape (four instances confirm it: iter-107
`prerender_fillers`, iter-108 `load_engines`, iter-109
`build_audio_io`, iter-110 `run_session`):

1. **Inject callable dependencies, not engines.** Caller wraps
   the engine + voice + speed in a closure / factory. The
   extracted module knows nothing about specific engine
   classes, so tests pass any callable matching the expected
   signature.
2. **Inject `log` (callable, default `print`).** Module emits
   plain text without ANSI codes. Tests capture into a list
   via `log=lines.append`. **Skip this seam when the extracted
   code doesn't emit user-facing strings** — `build_audio_io`
   is silent so it has no `log` arg.
3. **ANSI styling stays at the caller.** `mic_chat.py`
   re-applies `YELLOW` / `GREEN` / leading-space indent in a
   small log-adapter closure. Keeps presentation owned by the
   entrypoint.
4. **Return a dataclass, not a tuple.** Future fields can extend
   without breaking call sites. `LoadedEngines`, `AudioIO`,
   `SessionState`, `RecordingStats` all follow this shape.
5. **Lazy-import platform deps inside the closures, not at
   module scope.** `build_audio_io` defers `import pyaudio`
   into `speaker_factory` so the module is importable on x86_64
   Linux. Same trick lets tests stub pyaudio without
   monkey-patching the runtime.

### Session-summary diversity-check pattern

When adding a session-summary warning that fires on **N+
consecutive turns sharing the same problematic value**, follow
this template (five instances confirm it: iter-114
`_emit_filler_diversity_line`, iter-115 + iter-126
`_emit_naturalness_consistency_line`, iter-120
`_emit_barge_phase_consistency_line`, iter-128
`_emit_sentence_length_consistency_line`, iter-140
`_emit_stt_rtf_consistency_line`):

1. **Filter "uninteresting" values BEFORE the run scan.**
   Each instance has its own filter rule:
   - iter-114 drops `0` (no filler that turn).
   - iter-115 + iter-126 drops `""` (no audio) and `"natural"` (the
     desired state).
   - iter-120 drops `""` (no barge that turn).
   - iter-128 drops `""` (no sentences) and `"medium"`/`"short"`
     (the fine states).
   - iter-140 drops `""` (no measurable STT) and `"realtime"`
     (the fine state).

   The filter rule is per-instance policy — never bake it into
   the shared run-finder. Keeps `_longest_consecutive_run`
   (iter-116) a pure list-scanning primitive.

2. **Use `_longest_consecutive_run` from `_chat_metrics.py`.**
   Returns `(length, value)` of the longest consecutive-equal
   run. Earliest-tie rule: the first run wins on length.

3. **Apply per-instance threshold.** Conventions earned across
   instances:
   - 3 (iter-114 filler) — for high-noise random-pick signals.
   - 4 (iter-120 barge-phase) — for semantically-loaded events.
   - 5 (iter-115 naturalness, iter-128 sentence-length,
     iter-140 stt-rtf) — for general "natural variation is
     normal" signals.

   Higher threshold = lower false-positive rate at the cost
   of longer runs needed to fire. Pick based on how rare the
   underlying event is.

4. **For continuous metrics, bucket BEFORE filtering.**
   iter-128 is the first instance applied to a non-string
   signal: `_sentence_length_bucket` maps `mean_sentence_chars`
   to `"very_short"`/`"short"`/`"medium"`/`"long"`. iter-140 is
   the second: `_stt_rtf_bucket` maps `stt_rtf` to
   `"realtime"`/`"slow"`/`"very_slow"`. The bucketing function
   is testable in isolation; the run-scan then consumes the
   bucketed values like any other categorical signal.

5. **Per-value suggestion mapping inside the helper.** When
   multiple values warrant warnings (rushed/slow,
   llm_stream/playback, very_short/long, slow/very_slow), the
   suggestion text per value lives inside the helper. Don't
   externalize to the caller (overkill for two values), don't
   make it one-size-fits-all (loses signal). Per-value branches
   are the right shape.

6. **Defensive fallback for unknown values.** Each helper has
   an `else` branch that emits a generic suggestion when the
   bucket value doesn't match any expected case. Catches future
   additions without dropping the signal silently.

7. **Name the responsible iteration in the warning text.** So
   operators searching `ITERATION_LOG.md` for "iter-XYZ" find
   the full context. Compare to iter-074's bargeable warning
   ("watcher coverage regression") which forces the operator
   to grep — the iter-114+ pattern names the fix iter directly.

8. **Tests cover the matrix:** empty/no-value suppression,
   below-threshold, at-or-above-threshold (per value), filter
   semantics (intervening "uninteresting" values don't break
   runs; phase changes between flagged values do break runs),
   custom threshold, longest-of-multiple, output formatting,
   defensive unknown-value path.

## Architecture

### Components

- **STT (Speech-to-Text):** Whisper.cpp — local transcription on Apple Silicon
- **TTS (Text-to-Speech):** Kokoro / Piper — local speech synthesis
- **VAD (Voice Activity Detection):** Silero VAD — detect when the user starts/stops speaking

All components run entirely on-device. No cloud APIs, no data leaving the machine.

## Dependencies and runtime

Designed for Apple Silicon Macs. Component-specific dependencies:

- **Whisper.cpp:** C++ build with Metal acceleration
- **Kokoro / Piper:** Python-based TTS engines
- **Silero VAD:** PyTorch-based voice activity detection
