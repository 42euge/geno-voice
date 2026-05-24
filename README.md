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
