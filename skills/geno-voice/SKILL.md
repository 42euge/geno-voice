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
observability:
  success_signal: "voice pipeline component (STT, TTS, or VAD) initialized or audio processed successfully"
  failure_signals:
    - "Whisper.cpp model not found or failed to load"
    - "Kokoro/Piper TTS engine unavailable"
    - "Silero VAD model loading failed"
    - "audio device not accessible"
  knowledge_reads:
    - "local Whisper.cpp models (STT)"
    - "Kokoro/Piper voice models (TTS)"
    - "Silero VAD model weights"
  knowledge_writes:
    - "transcription output (STT results)"
    - "synthesized audio files (TTS output)"
---

# geno-voice — Umbrella Skill

Local voice pipeline for offline, privacy-first AI voice interaction. Provides
on-device STT, TTS, and VAD for geno-ecosystem projects.

## Components

- **STT (Speech-to-Text):** Whisper.cpp — local transcription
- **TTS (Text-to-Speech):** Kokoro / Piper — local speech synthesis
- **VAD (Voice Activity Detection):** Silero VAD — detect when the user starts/stops speaking

## Completion

When this skill finishes, emit a trace:

```bash
geno-trace emit \
  --skill geno-voice \
  --status <success|failure|abandoned> \
  --tool-calls <approximate count> \
  --errors <count of tool/command errors>
```

- `success` = voice pipeline component initialized or audio processed successfully
- `failure` = model not found, engine unavailable, or audio device inaccessible
- `abandoned` = user stopped early
