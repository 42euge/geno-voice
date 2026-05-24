---
name: geno-voice
description: >-
  Local voice pipeline for offline, privacy-first AI voice interaction.
  STT via Whisper.cpp, TTS via Kokoro/Piper, VAD via Silero — all on-device.
allowed-tools: "Bash(find *) Bash(ls *) Bash(cat *) Bash(grep *) Bash(python3 *) Read(*)"
license: MIT
metadata:
  author: 42euge
  version: "0.1.0"
---

# geno-voice — Local Voice Pipeline

Provides on-device speech-to-text, text-to-speech, and voice activity detection
for geno-ecosystem projects that need voice interaction. No cloud APIs, no data
leaving the machine.

## Components

- **STT (Speech-to-Text):** Whisper.cpp — local transcription
- **TTS (Text-to-Speech):** Kokoro / Piper — local speech synthesis
- **VAD (Voice Activity Detection):** Silero VAD — detect when the user starts/stops speaking

## Installation

```
geno-tools install geno-voice
```

Or from within an agent session:

```
/geno-tools install geno-voice
```
