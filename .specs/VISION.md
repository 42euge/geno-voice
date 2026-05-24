# Vision

Offline voice SDK that makes any local AI agent conversational.

## Why

Voice AI today means shipping audio to someone else's cloud — latency, privacy loss, internet dependency. The intelligence already runs locally; the voice layer shouldn't be the weak link.

- **Offline-first** — STT/TTS/VAD on local hardware. No round-trips.
- **Voice-driven dev** — Talk to Claude Code hands-free.
- **Universal layer** — Pluggable into any agent or app. Not married to one LLM.
- **Beyond the keyboard** — Mobile, workshop, driving, accessibility.

## Success looks like

1. **Sub-500ms on a MacBook** — No discrete GPU. Feels instantaneous.
2. **Cloud-competitive quality** — Rivals Whisper API / ElevenLabs, running locally.
3. **Claude Code native** — Voice as first-class I/O for coding sessions.
4. **Daily driver** — Not a demo. The default way you talk to AI.
5. **Framework** — Others build voice agents on top without reinventing the pipeline.

## Principles

- **Local by default** — Cloud is an optional accelerator, never a requirement.
- **Latency is the feature** — Every decision optimizes for time-to-first-byte.
- **Composable** — STT, TTS, VAD, wake word are independent swappable modules.
- **Real hardware** — Targets Apple Silicon. No "works on an A100" asterisks.
