# geno-voice

Local audio pipeline for privacy-conscious AI voice interaction.

geno-voice provides on-device speech-to-text, text-to-speech, and voice activity
detection for geno-ecosystem projects that need voice interaction. Raw audio is
processed locally. Agent mode can send the resulting transcript and conversation
text to a configured OpenAI-compatible LLM endpoint such as Blue LiteLLM.

## Components

| Component | Engine | Purpose |
|-----------|--------|---------|
| STT | MLX Whisper / faster-whisper | Local speech-to-text transcription |
| TTS | Kokoro / Piper | Local text-to-speech synthesis |
| VAD | Energy VAD / Silero VAD | Live and offline voice activity detection |

## Navigation

- [Getting Started](getting-started.md) — installation, prerequisites, first use
- [Human voice turn explainer](human-voice-turn-explainer.md) — every code path
  from live microphone input to the response heard, including the Blue endpoint,
  streaming TTS, and full-duplex echo behavior
- [Full-duplex recent literature](research/full-duplex-recent-literature.md) —
  primary-source survey and a staged architecture that keeps Blue LiteLLM as the
  agent brain
- [Full-duplex papers explained](research/full-duplex-paper-explainers.md) —
  plain-language explanations of every archived paper, with equations where
  they make the mechanism clearer
- [Engineering full-duplex systems survey](research/survey/full-duplex-systems/README.md) —
  LaTeX source, BibTeX bibliography, and compiled systems-survey PDF
