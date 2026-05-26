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
default to this shape (three instances confirm it: iter-107
`prerender_fillers`, iter-108 `load_engines`, iter-109
`build_audio_io`):

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
   `RecordingStats` all follow this shape.
5. **Lazy-import platform deps inside the closures, not at
   module scope.** `build_audio_io` defers `import pyaudio`
   into `speaker_factory` so the module is importable on x86_64
   Linux. Same trick lets tests stub pyaudio without
   monkey-patching the runtime.

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
