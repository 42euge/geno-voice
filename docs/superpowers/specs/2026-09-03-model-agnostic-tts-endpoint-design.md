# Model-Agnostic Streaming TTS Endpoint Design

## Goal

Add a `geno-voice-remote-server` executable, with `geno-voice start-endpoint`
as a compatibility command, that loads one local TTS model and serves
low-latency, interruptible speech over WebSocket, bidirectional gRPC, WebRTC,
or RTP. Breeze-TTS-2 is the first remote-GPU target, Kokoro is the
first existing geno-voice target, and additional models plug in without
changing session or transport code.

The endpoint is a TTS renderer, not a voice-agent brain. It accepts text and
playback-control commands while streaming audio and timing/state events back.
VAD, STT, end-of-turn prediction, backchannel timing policy, and dialogue
policy stay in the client-side voice-agent controller.

## Constraints

- The service runs on a trusted internal LAN. It has no authentication or TLS
  in this release.
- It must remain importable and testable on a Mac without CUDA, Breeze,
  WebRTC, or gRPC packages installed.
- Heavy/model/protocol imports are lazy and selected by the CLI.
- One process loads one model instance. Breeze inference is serialized because
  its official runtime is single-request.
- Canonical streamed audio is mono 24 kHz signed 16-bit PCM. WebRTC resamples
  to the transport-required 48 kHz audio clock; RTP uses L16/24000/1.
- Breeze-TTS-2's research/non-commercial license is surfaced at launch. The
  endpoint does not weaken or reinterpret upstream model licenses.
- The command starts exactly one transport. Operators may start separate
  processes when they need several transports simultaneously.

## Approaches considered

### 1. One independent server implementation per protocol

Each server would load and call the model directly. This looks quick but
duplicates cancellation, text buffering, backchannel priority, capability
reporting, and error handling four times. Model quirks would leak into every
protocol. Rejected because the resulting modules would be shallow and drift.

### 2. Shared session host with model and transport adapters — selected

A transport-neutral host owns session state, queues, cancellation, event
ordering, and model scheduling. Model adapters satisfy one synthesis
interface; transport adapters translate one command/event vocabulary. This
keeps both seams small, lets core tests run without network/model packages,
and loads the selected model once.

### 3. Gateway in front of each model's official server

Geno-voice could proxy the official Breeze HTTP server, a Kokoro process, and
future servers. This isolates dependencies but adds another network hop,
requires a different process manager per model, and cannot guarantee common
cancellation/timing semantics. It remains a possible deployment adapter, but
is not the primary architecture.

## Deep modules and seams

### Model module

`TTSModelAdapter` is the model seam. Its interface exposes:

- immutable `ModelCapabilities`;
- asynchronous `load()` and `close()` lifecycle;
- `synthesize(SynthesisRequest, cancellation)` yielding canonical
  `AudioChunk` objects.

The interface hides model downloads, device setup, prompts, tensor formats,
threading, and conversion to canonical PCM. Adapters may expose only
capabilities they truly implement. A capability record includes streaming,
alignment, voice cloning, voice design, rate control, and supported sample
rate.

Built-in adapters:

- `BreezeTTS2Adapter` uses the official Breeze runtime from an importable
  installation or `--runtime-path`, and weights from `--model-path`. It maps
  instruction/reference fields into Breeze templates and emits codec chunks.
  It declares no word alignment or numeric rate control.
- `KokoroAdapter` wraps the existing geno-voice engine, converts WAV chunks to
  PCM, and preserves its ordinary voice/speed controls. Alignment can be added
  later through the repository's existing `synthesize_with_alignment` path;
  the first server adapter declares only what it emits now.

Third-party adapters are discovered through the
`geno_voice.tts_models` Python entry-point group. An entry point returns an
adapter factory. Model names are normalized case-insensitively by replacing
underscores and spaces with hyphens. Built-ins reserve `breeze-tts-2`,
`breeze`, and `kokoro`.

### Session module

`SynthesisSession` is the central deep module. Transports only call
`handle(command)` and consume `events()`. The implementation owns:

- incremental text buffering;
- commit/immediate-speak commands;
- bounded normal and priority queues;
- monotonically increasing request, audio-sequence, and PTS values;
- cancellation and supersession while synthesis is active;
- priority backchannel jobs;
- conversion of model chunks into ordered state/audio/alignment events; and
- deterministic cleanup on disconnect.

Receive and synthesis run independently. A `cancel` or `supersede` command
therefore sets the active cancellation token immediately even while a model
worker is producing audio.

The endpoint does not decide when to backchannel. A voice-agent controller
sends a `speak` command with `priority: "backchannel"`. Priority jobs move
ahead of queued normal speech. `interrupt: true` additionally cancels the
active normal request. This supports pre-rendered or short cues without
embedding conversational policy in TTS.

### Transport modules

Each transport is an adapter around `SynthesisSession`. It must preserve the
same command meanings, event order, request IDs, PCM content, and cancellation
semantics.

## Command and event contract

All JSON transports use these commands:

```json
{"type":"append","text":"The answer arrives ","request_id":"turn-7"}
{"type":"append","text":"incrementally.","request_id":"turn-7"}
{"type":"commit","request_id":"turn-7"}
{"type":"speak","text":"Mm-hmm.","request_id":"cue-2","priority":"backchannel"}
{"type":"cancel","request_id":"turn-7"}
{"type":"supersede","text":"Corrected answer.","request_id":"turn-8"}
{"type":"close"}
```

Rules:

- `append` requires non-empty text and one stable request ID. Text from
  different request IDs cannot share a buffer.
- `commit` queues the buffered text. An empty commit is rejected.
- `speak` queues complete text immediately. `priority` is `normal` or
  `backchannel`; `interrupt` defaults to false.
- `cancel` cancels the matching active request and removes matching queued
  work. Missing/already-finished IDs produce an idempotent `cancelled` event.
- `supersede` cancels all active/queued normal speech, clears buffered text,
  and immediately queues its replacement.
- `close` cancels the session and releases transport resources.
- Text is limited to 64 KiB per request, and pending synthesis is limited to
  32 jobs. Exceeding a limit produces an error event without killing the
  connection.

Events are JSON unless a transport has a typed representation:

- `ready`: session ID, model name, canonical audio format, capabilities;
- `accepted`: request ID and queue state;
- `started`: request ID and priority;
- `audio`: request ID, sequence, PTS samples, sample count, sample rate,
  encoding, and `final`; the PCM payload is carried separately where useful;
- `alignment`: request ID and source/token timing when the model provides it;
- `completed`: request ID and total samples;
- `cancelled`: request ID and whether active work was interrupted;
- `error`: stable code, message, and optional request ID; and
- `closed`.

Audio PTS is session-relative and monotonic. Cancellation never emits
`completed` for the cancelled request. Audio already delivered cannot be
recalled; transports stop sending new chunks as soon as cancellation reaches
the model/session boundary.

## Protocol mappings

### WebSocket

An HTTP host exposes:

- `GET /health`;
- `GET /v1/capabilities`; and
- `WS /v1/tts/stream`.

Commands and non-audio events are JSON text frames. Audio uses a binary
`GVA1` envelope: four magic bytes, a two-byte network-order JSON-header
length, the UTF-8 audio-event header, then PCM bytes. This keeps each audio
chunk self-describing and atomic.

### Bidirectional gRPC

`geno.voice.endpoint.v1.TTS/Stream` is a bidirectional streaming RPC.
`ClientMessage` carries one typed command. `ServerMessage` carries one typed
event and optional `bytes audio`. The generated protobuf modules are checked
in so endpoint clients do not require a compiler at runtime.

### WebRTC

The signaling host exposes `POST /v1/webrtc/offer` and returns an SDP answer.
The client creates an ordered data channel named `geno-voice-control`; JSON
commands and events use that channel. The server adds one audio media track.
Audio is resampled from canonical 24 kHz PCM to 48 kHz frames for WebRTC/Opus.
Closing the peer connection closes its synthesis session. ICE/STUN/TURN is not
configured for this LAN-only release.

### RTP

RTP needs a separate control plane. The HTTP control host exposes:

- `POST /v1/rtp/sessions` with destination host/port;
- `POST /v1/rtp/sessions/{id}/commands`;
- `GET /v1/rtp/sessions/{id}/events` as server-sent events; and
- `DELETE /v1/rtp/sessions/{id}`.

Audio is RFC 3550 RTP over UDP with dynamic payload type 96 described as
`L16/24000/1`. L16 samples are network-byte-order. Sequence and timestamp
wrap follow RTP rules. The sender periodically emits RTCP sender reports to
the configured RTCP port (default RTP port + 1). The session-create response
includes a small SDP description clients can hand to an RTP receiver.

## CLI

The installed console command adds:

```text
geno-voice-remote-server \
  --protocol webrtc \
  --model Breeze-TTS-2 \
  --host 0.0.0.0 \
  --port 8787 \
  --model-path /models/Breeze-TTS-2 \
  --runtime-path /opt/breeze-tts
```

Arguments:

- `--protocol`: case-insensitive `websocket`/`ws`, `grpc`, `webrtc`, or `rtp`;
- `--model`: built-in alias or installed model entry point;
- `--host`: default `127.0.0.1`; use `0.0.0.0` for the trusted LAN;
- `--port`: protocol default when omitted;
- `--model-path`, `--runtime-path`, and `--device`: passed through the model
  configuration;
- `--voice`: optional default voice;
- `--log-level`; and
- `--list-models`: list built-ins/plugins without loading a model.

Parser construction imports no endpoint dependencies. Dispatch lazily imports
the endpoint launcher. Startup prints the selected model, capabilities,
address, audio format, upstream license warning when applicable, and protocol
client instructions.

## Errors and lifecycle

- Missing optional protocol packages fail before model loading with an exact
  install hint such as `pip install 'geno-voice[endpoint]'`.
- Model import/configuration failures name the selected adapter and preserve
  the original cause.
- Model loading completes before the server reports healthy.
- SIGINT/SIGTERM stops accepting sessions, cancels active work, closes model
  resources, and then stops the protocol host.
- One session failure emits an error and closes only that session.
- Unexpected model-worker exceptions become `MODEL_ERROR` events; the host
  remains available when the adapter can continue.
- LAN mode emits a warning when bound to a non-loopback address because this
  release has no authentication or encryption.

## Packaging

Core session/model-registry code uses the standard library. Optional
dependencies live under a `project.optional-dependencies.endpoint` extra:

- FastAPI/Uvicorn for WebSocket, WebRTC signaling, and RTP control;
- `grpcio` and `protobuf` for gRPC;
- `aiortc` and `av` for WebRTC; and
- `numpy` for model/audio conversion.

Breeze's official CUDA runtime and weights are installed separately on the Z2
because they have strict Torch/CUDA pins and a distinct model license. Kokoro
continues to use the repository's existing dependency.

## Testing

Tests cross the same interfaces as callers:

1. Pure session tests use a deterministic fake model to prove append/commit,
   event ordering, monotonic PTS, cancel, supersede, priority, queue limits,
   and cleanup.
2. Registry tests prove normalization, built-ins, plugin discovery, duplicate
   rejection, and lazy imports.
3. CLI tests prove case-insensitive parsing, defaults, lazy dispatch, list
   behavior, and useful dependency/model errors.
4. Wire tests prove WebSocket envelopes and protobuf round trips.
5. Transport tests run loopback WebSocket/gRPC/RTP flows with the fake model.
   WebRTC tests exercise signaling, data-channel routing, resampling, and
   cleanup through injected peer/track fakes; an optional marked test uses
   real `aiortc` when installed.
6. Adapter tests inject fake official runtimes. No test downloads or loads a
   real speech model.
7. A Z2 smoke procedure loads Breeze, connects through each protocol, checks
   first audio, sends cancel during synthesis, and verifies silence/no further
   packets after cancellation.

## Non-goals

- microphone ingestion, VAD, STT, echo cancellation, or dialogue management;
- deciding whether/when an agent should backchannel;
- public-internet security, accounts, quotas, or multi-tenant isolation;
- claiming every Hugging Face TTS checkpoint works without an adapter;
- changing upstream model licenses; or
- guaranteeing word alignment for models that do not emit timing data.
