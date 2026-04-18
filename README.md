# geno-voice

Local voice pipeline for offline, privacy-first AI voice interaction.

## What

A reusable STT/TTS stack that runs entirely on-device. No cloud APIs, no data leaving the machine.

Used by [geno-reflect](https://github.com/42euge/geno-reflect) and other geno-* projects that need voice interaction.

## Components

- **STT (Speech-to-Text):** Whisper.cpp — local transcription
- **TTS (Text-to-Speech):** Kokoro / Piper — local speech synthesis
- **VAD (Voice Activity Detection):** Silero VAD — detect when the user starts/stops speaking

## Project Structure

```
geno-voice/
├── README.md
├── stt/              # Speech-to-text pipeline
├── tts/              # Text-to-speech pipeline
├── vad/              # Voice activity detection
└── examples/         # Usage examples and integration demos
```
