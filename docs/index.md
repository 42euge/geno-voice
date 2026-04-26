# geno-voice

Local voice pipeline for offline, privacy-first AI voice interaction.

geno-voice provides on-device speech-to-text, text-to-speech, and voice activity
detection for geno-ecosystem projects that need voice interaction. All processing
happens locally — no cloud APIs, no data leaving the machine.

## Components

| Component | Engine | Purpose |
|-----------|--------|---------|
| STT | Whisper.cpp | Local speech-to-text transcription |
| TTS | Kokoro / Piper | Local text-to-speech synthesis |
| VAD | Silero VAD | Voice activity detection |

## Navigation

- [Getting Started](getting-started.md) — installation, prerequisites, first use
