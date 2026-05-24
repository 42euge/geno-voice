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
