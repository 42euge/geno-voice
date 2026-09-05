# geno-voice

Library for adding privacy-conscious voice interaction to any project.

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

## Model-agnostic TTS serving

Install the endpoint extra and launch one local model over WebSocket, gRPC,
WebRTC, or RTP:

```bash
python -m pip install -e '.[endpoint]'
geno-voice-remote-server --protocol websocket --model kokoro
geno-voice-remote-server --protocol webrtc --model Breeze-TTS-2 \
  --host 0.0.0.0 --model-path /models/Breeze-TTS-2 \
  --runtime-path /opt/breeze-tts --device cuda:0
```

The endpoint accepts incremental text, immediate speech, cancellation,
supersession, and priority backchannel requests while streaming audio. It does
not ingest a microphone or make turn-taking decisions. All transports share
the same session semantics and canonical mono 24 kHz PCM; WebRTC converts the
media track to 48 kHz and RTP uses `L16/24000/1`.

The compatibility command `geno-voice start-endpoint` accepts the same options.

This release is for a trusted internal LAN and has no authentication or TLS.
Breeze-TTS-2 also carries a research/non-commercial model license, which is
reported at launch. See the repository README's **Streaming TTS endpoint**
section for command/event contracts, ports, model plugins, and the Z2 smoke
procedure.

## Navigation

- [Getting Started](getting-started.md) — installation, prerequisites, first use
- [Human voice turn explainer](human-voice-turn-explainer.md) — every code path
  from live microphone input to the response heard, including the Blue endpoint,
  streaming TTS, and full-duplex echo behavior
