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

## Streaming TTS endpoint

Install `.[endpoint]` to serve Kokoro, Breeze-TTS-2, or an installed model
adapter over WebSocket:

```bash
geno-voice-remote-server --protocol websocket --model kokoro
geno-voice start-endpoint --protocol websocket --model Breeze-TTS-2
```

The service streams mono PCM audio and accepts interruption, supersession, and
priority backchannel commands. It is TTS-only and intended for a trusted
private network.

## Navigation

- [Getting Started](getting-started.md) — installation, prerequisites, first use
- [Open-source TTS models](research/open-source-tts-models-for-quick-agent.md) —
  model and runtime options for expressive local synthesis
- [Full-duplex recent literature](research/full-duplex-recent-literature.md) —
  primary-source survey and staged voice-agent architecture
- [Full-duplex papers explained](research/full-duplex-paper-explainers.md) —
  plain-language explanations of the archived research papers
- [Engineering full-duplex systems survey](research/survey/full-duplex-systems/README.md) —
  LaTeX source, BibTeX bibliography, and compiled systems survey
