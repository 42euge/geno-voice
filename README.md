# geno-voice

A reusable library for adding private, local voice interaction to any project.

## What

An STT/TTS/VAD stack that runs entirely on-device. No cloud APIs, no data
leaving the machine.

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

For development, install the Python package and streaming endpoint extras:

```bash
python -m pip install -e '.[endpoint]'
```

## Streaming TTS endpoint

`geno-voice-remote-server` loads one TTS model and exposes interruptible,
low-latency synthesis over WebSocket, bidirectional gRPC, or WebRTC. The
service is TTS-only; callers retain VAD, STT, turn-taking, backchannel timing,
and dialogue policy.

```bash
geno-voice-remote-server --list-models
geno-voice-remote-server --protocol websocket --model kokoro \
  --host 0.0.0.0 --voice af_heart
geno-voice start-endpoint --protocol websocket --model Breeze-TTS-2 \
  --host 0.0.0.0 \
  --model-path /models/Breeze-TTS-2 \
  --runtime-path /opt/breeze-tts \
  --device cuda:0
```

The supported transports are:

| Protocol | Command value | Connection surface | Port |
|----------|---------------|--------------------|------|
| WebSocket | `websocket` or `ws` | `WS /v1/tts/stream` | 8765 |
| gRPC | `grpc` | `geno.voice.endpoint.v1.TTS/Stream` | 50051 |
| WebRTC | `webrtc` | HTTP offer plus audio/data channels | 8787 |

The connection supports append, commit, speak, cancel, supersede, backchannel,
and close commands. Each server process loads one model. The endpoint is
intended for a trusted private network and does not provide authentication or
TLS.

Additional open-source models plug in through the `geno_voice.tts_models`
Python entry-point group. Model-specific downloads, runtimes, threading, and
licenses remain inside each adapter. Breeze-TTS-2 uses its official CUDA
runtime and is governed by the BreezeBlue Research and Non-Commercial License;
the server prints that warning at launch.

For a manual LAN smoke test, bind to `0.0.0.0`, connect using the host's private
address, verify the ready event, stream a short `speak` request, cancel a longer
request, then submit another request to confirm the session remains usable.

## Project Structure

```
geno-voice/
├── GENO.md           # agent instructions
├── SKILL.md          # umbrella skill manifest
├── genotools.yaml    # geno-tools manifest
├── skills/           # skill definitions
│   └── geno-voice/   #   umbrella
├── geno_voice/       # Python package and remote TTS server
├── stt/              # speech-to-text pipeline
├── tts/              # text-to-speech pipeline
├── vad/              # voice activity detection
├── examples/         # usage examples and integration demos
├── docs/             # documentation site
└── mkdocs.yml        # MkDocs configuration
```

## License

MIT
