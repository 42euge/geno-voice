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

# geno-voice — Umbrella Skill

Local voice pipeline for offline, privacy-first AI voice interaction. Provides
on-device STT, TTS, and VAD for geno-ecosystem projects.

## Components

- **STT (Speech-to-Text):** Whisper.cpp — local transcription
- **TTS (Text-to-Speech):** Kokoro / Piper — local speech synthesis
- **VAD (Voice Activity Detection):** Silero VAD — detect when the user starts/stops speaking
