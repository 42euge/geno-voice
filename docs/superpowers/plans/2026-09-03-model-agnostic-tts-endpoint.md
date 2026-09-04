# Model-Agnostic Streaming TTS Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `geno-voice start-endpoint` with a model-agnostic, interruptible TTS session served over WebSocket, bidirectional gRPC, WebRTC, or RTP.

**Architecture:** One loaded `TTSModelAdapter` feeds independent `SynthesisSession` instances. Sessions own incremental text, priority, cancellation, sequencing, and events; protocol adapters only translate their wire format. Breeze-TTS-2 and Kokoro are built-ins, while Python entry points add future models.

**Tech Stack:** Python 3.11+, asyncio, FastAPI/Uvicorn, grpcio/protobuf, aiortc/PyAV, UDP RTP/RTCP, pytest.

**Spec:** `docs/superpowers/specs/2026-09-03-model-agnostic-tts-endpoint-design.md`

## Global Constraints

- Trusted internal LAN only; no authentication or TLS in this release.
- Imports and tests must work without CUDA, Breeze, aiortc, or grpc installed.
- Heavy dependencies are imported lazily after CLI selection.
- One process loads one model; adapters return mono 24 kHz signed 16-bit PCM.
- Breeze inference is serialized and its research/non-commercial license warning is printed.
- The endpoint is TTS-only; VAD, STT, turn policy, and backchannel policy remain upstream.
- Every production behavior follows red-green-refactor and receives a focused test.

---

### Task 1: Core command, event, and session module

**Files:**
- Create: `geno_voice/endpoint/__init__.py`
- Create: `geno_voice/endpoint/types.py`
- Create: `geno_voice/endpoint/session.py`
- Create: `tests/unit/test_endpoint_session.py`

**Interfaces:**
- Produces: `ModelCapabilities`, `SynthesisRequest`, `AudioChunk`, `EndpointCommand`, `EndpointEvent`, `CancellationToken`, and `SynthesisSession`.
- `SynthesisSession.handle(command)` accepts append/commit/speak/cancel/supersede/close; `events()` asynchronously yields ordered events.

- [x] **Step 1: Write failing session tests**

```python
async def test_append_commit_streams_ordered_audio():
    model = FakeModel([b"one", b"two"])
    session = SynthesisSession(model, session_id="s1")
    await session.start()
    await session.handle(EndpointCommand.append("r1", "Hello "))
    await session.handle(EndpointCommand.append("r1", "world."))
    await session.handle(EndpointCommand.commit("r1"))
    events = await collect_through(session, "completed")
    assert [event.type for event in events] == [
        "ready", "accepted", "started", "audio", "audio", "completed"
    ]
    assert [e.pts_samples for e in events if e.type == "audio"] == [0, 3]

async def test_cancel_interrupts_active_request():
    model = BlockingFakeModel()
    session = SynthesisSession(model, session_id="s1")
    await session.start()
    await session.handle(EndpointCommand.speak("r1", "Long answer"))
    await model.started.wait()
    await session.handle(EndpointCommand.cancel("r1"))
    assert (await next_type(session, "cancelled")).request_id == "r1"
    assert model.cancelled.is_set()
```

- [x] **Step 2: Run tests and verify RED**

Run: `../../.venv/bin/python -m pytest tests/unit/test_endpoint_session.py -q`

Expected: collection fails because `geno_voice.endpoint` does not exist.

- [x] **Step 3: Implement immutable wire/domain types and the session worker**

```python
@dataclass(frozen=True)
class AudioChunk:
    pcm: bytes
    sample_rate: int = 24_000
    alignment: tuple[AlignmentSpan, ...] = ()

class SynthesisSession:
    async def start(self) -> None: ...
    async def handle(self, command: EndpointCommand) -> None: ...
    async def events(self) -> AsyncIterator[EndpointEvent]: ...
    async def close(self) -> None: ...
```

Implement the 64 KiB text limit, 32-job limit, normal/backchannel priority queue, monotonic PTS/audio sequence, matching-request cancellation, normal-lane supersession, and deterministic close. The model worker receives a `CancellationToken` checked between chunks.

- [x] **Step 4: Run focused tests and verify GREEN**

Run: `../../.venv/bin/python -m pytest tests/unit/test_endpoint_session.py -q`

Expected: all core session tests pass without network/model dependencies.

- [x] **Step 5: Commit**

```bash
git add geno_voice/endpoint tests/unit/test_endpoint_session.py
git commit -m "feat: add interruptible TTS endpoint sessions"
```

### Task 2: Model registry, Breeze adapter, and Kokoro adapter

**Files:**
- Create: `geno_voice/endpoint/registry.py`
- Create: `geno_voice/endpoint/models/__init__.py`
- Create: `geno_voice/endpoint/models/breeze.py`
- Create: `geno_voice/endpoint/models/kokoro.py`
- Create: `tests/unit/test_endpoint_registry.py`
- Create: `tests/unit/test_endpoint_models.py`

**Interfaces:**
- Consumes: `AudioChunk`, `CancellationToken`, `ModelCapabilities`, `SynthesisRequest`.
- Produces: `ModelRegistry`, `BreezeTTS2Adapter`, and `KokoroAdapter`, each satisfying `TTSModelAdapter` structurally.

- [x] **Step 1: Write failing registry and adapter tests**

```python
def test_registry_normalizes_alias_and_loads_factory_lazily():
    registry = ModelRegistry(load_plugins=False)
    assert registry.resolve("Breeze_TTS_2").canonical_name == "breeze-tts-2"

async def test_breeze_adapter_converts_float_chunks_to_pcm16(fake_breeze_modules):
    adapter = BreezeTTS2Adapter(model_path="/models/breeze", runtime_path="/opt/breeze")
    await adapter.load()
    chunks = [chunk async for chunk in adapter.synthesize(request(), CancellationToken())]
    assert chunks[0].pcm == b"\xff\x7f\x01\x80"

async def test_kokoro_adapter_unwraps_wav_chunks(fake_kokoro_engine):
    adapter = KokoroAdapter(engine=fake_kokoro_engine)
    chunks = [chunk async for chunk in adapter.synthesize(request(), CancellationToken())]
    assert chunks[0].pcm == b"\x01\x00\x02\x00"
```

- [x] **Step 2: Run tests and verify RED**

Run: `../../.venv/bin/python -m pytest tests/unit/test_endpoint_registry.py tests/unit/test_endpoint_models.py -q`

Expected: imports fail for missing registry/adapters.

- [x] **Step 3: Implement built-ins and plugin discovery**

```python
class ModelRegistry:
    ENTRY_POINT_GROUP = "geno_voice.tts_models"

    def names(self) -> tuple[str, ...]: ...
    def resolve(self, name: str) -> ModelDescriptor: ...
    def create(self, name: str, config: ModelConfig) -> TTSModelAdapter: ...
```

Register aliases without importing model modules until `create`. Discover
`importlib.metadata.entry_points(group=...)`, reject duplicate normalized
names, and surface the originating entry point in errors. Breeze adds
`runtime_path` to `sys.path` only for its import attempt, uses the official
`load_runtime`, templates, and `FastBreezeStreamingRuntime`, serializes calls,
and converts float numpy chunks to PCM16. Kokoro calls the existing engine in a
worker thread and extracts PCM frames from its WAV chunks.

- [x] **Step 4: Run focused tests and verify GREEN**

Run: `../../.venv/bin/python -m pytest tests/unit/test_endpoint_registry.py tests/unit/test_endpoint_models.py -q`

Expected: all registry/adapter tests pass with injected fakes and no model downloads.

- [x] **Step 5: Commit**

```bash
git add geno_voice/endpoint/registry.py geno_voice/endpoint/models tests/unit/test_endpoint_registry.py tests/unit/test_endpoint_models.py
git commit -m "feat: add pluggable Breeze and Kokoro endpoint models"
```

### Task 3: CLI, launcher, and optional dependencies

**Files:**
- Create: `geno_voice/endpoint/cli.py`
- Create: `geno_voice/endpoint/host.py`
- Modify: `examples/gv.py`
- Modify: `pyproject.toml`
- Create: `tests/unit/test_gv_start_endpoint.py`
- Create: `tests/unit/test_endpoint_host.py`

**Interfaces:**
- Consumes: `ModelRegistry` and transport launchers by protocol name.
- Produces: `EndpointConfig`, `run_endpoint(config)`, CLI command `start-endpoint`.

- [x] **Step 1: Write failing parser and lazy-dispatch tests**

```python
def test_start_endpoint_parser_is_case_insensitive():
    args = gv.build_parser().parse_args([
        "start-endpoint", "--protocol=WebRTC", "--model=Breeze-TTS-2"
    ])
    assert args.protocol == "webrtc"
    assert args.model == "Breeze-TTS-2"

def test_dispatch_passes_endpoint_config_without_importing_models(monkeypatch):
    seen = []
    monkeypatch.setattr("geno_voice.endpoint.cli.run_endpoint", seen.append)
    assert gv.main(["start-endpoint", "--protocol", "ws", "--model", "kokoro"]) == 0
    assert seen[0].protocol == "websocket"
```

- [x] **Step 2: Run tests and verify RED**

Run: `../../.venv/bin/python -m pytest tests/unit/test_gv_start_endpoint.py tests/unit/test_endpoint_host.py -q`

Expected: argparse rejects `start-endpoint`.

- [x] **Step 3: Implement CLI and host lifecycle**

```python
@dataclass(frozen=True)
class EndpointConfig:
    protocol: str
    model: str
    host: str = "127.0.0.1"
    port: int | None = None
    model_path: Path | None = None
    runtime_path: Path | None = None
    device: str | None = None
    voice: str | None = None
```

Add protocol aliases/default ports, `--list-models`, model/runtime/device/voice
flags, and lazy handler dispatch. `run_endpoint` loads the adapter before
starting its selected transport, warns for non-loopback LAN binding and Breeze
licensing, and always closes the model on exit. Add an `endpoint` optional
dependency group containing FastAPI/Uvicorn, grpcio/protobuf, aiortc/av, and
numpy.

- [x] **Step 4: Run focused and existing CLI tests**

Run: `../../.venv/bin/python -m pytest tests/unit/test_gv_start_endpoint.py tests/unit/test_endpoint_host.py tests/unit/test_gv_agent_cli.py -q`

Expected: endpoint and existing agent CLI tests pass.

- [x] **Step 5: Commit**

```bash
git add geno_voice/endpoint/cli.py geno_voice/endpoint/host.py examples/gv.py pyproject.toml tests/unit/test_gv_start_endpoint.py tests/unit/test_endpoint_host.py
git commit -m "feat: launch streaming TTS endpoints from geno-voice"
```

### Task 4: WebSocket transport and atomic audio envelope

**Files:**
- Create: `geno_voice/endpoint/transports/__init__.py`
- Create: `geno_voice/endpoint/transports/wire.py`
- Create: `geno_voice/endpoint/transports/websocket.py`
- Create: `tests/unit/test_endpoint_wire.py`
- Create: `tests/integration/test_endpoint_websocket.py`

**Interfaces:**
- Consumes: `SynthesisSession`, JSON commands, `EndpointEvent`.
- Produces: `encode_audio_envelope(event)`, `decode_audio_envelope(data)`, `create_websocket_app(host)` and `serve_websocket(host, bind, port)`.

- [x] **Step 1: Write failing wire and loopback tests**

```python
def test_audio_envelope_round_trip():
    packet = encode_audio_envelope(audio_event(sequence=7), b"\x01\x00")
    header, pcm = decode_audio_envelope(packet)
    assert packet[:4] == b"GVA1"
    assert header["sequence"] == 7
    assert pcm == b"\x01\x00"

def test_websocket_stream_accepts_speak_and_returns_binary_audio(fake_host):
    with TestClient(create_websocket_app(fake_host)) as client:
        with client.websocket_connect("/v1/tts/stream") as socket:
            assert socket.receive_json()["type"] == "ready"
            socket.send_json({"type": "speak", "request_id": "r1", "text": "Hi"})
            assert receive_until_binary(socket).startswith(b"GVA1")
```

- [x] **Step 2: Run tests and verify RED**

Run: `../../.venv/bin/python -m pytest tests/unit/test_endpoint_wire.py tests/integration/test_endpoint_websocket.py -q`

Expected: transport modules are missing.

- [x] **Step 3: Implement wire validation and concurrent WebSocket pumps**

```python
async def websocket_endpoint(ws: WebSocket) -> None:
    session = await host.open_session()
    await ws.accept()
    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(receive_commands(ws, session))
        tasks.create_task(send_events(ws, session))
```

Reject malformed JSON/commands with stable error events. Keep receive and send
independent so cancel arrives during synthesis. Emit health/capabilities routes
and close only the affected session on disconnect.

- [x] **Step 4: Run focused tests and verify GREEN**

Run: `../../.venv/bin/python -m pytest tests/unit/test_endpoint_wire.py tests/integration/test_endpoint_websocket.py -q`

Expected: envelope and loopback flows pass.

- [x] **Step 5: Commit**

```bash
git add geno_voice/endpoint/transports tests/unit/test_endpoint_wire.py tests/integration/test_endpoint_websocket.py
git commit -m "feat: serve TTS sessions over WebSocket"
```

### Task 5: Typed bidirectional gRPC transport

**Files:**
- Create: `geno_voice/endpoint/proto/tts_endpoint.proto`
- Create: `geno_voice/endpoint/proto/tts_endpoint_pb2.py`
- Create: `geno_voice/endpoint/proto/tts_endpoint_pb2_grpc.py`
- Create: `geno_voice/endpoint/proto/__init__.py`
- Create: `geno_voice/endpoint/transports/grpc.py`
- Create: `tests/integration/test_endpoint_grpc.py`

**Interfaces:**
- Consumes: typed protobuf `ClientMessage`, `SynthesisSession`.
- Produces: `TTS.Stream(stream ClientMessage) returns (stream ServerMessage)` and `serve_grpc`.

- [ ] **Step 1: Add the proto contract and failing round-trip test**

```protobuf
service TTS { rpc Stream(stream ClientMessage) returns (stream ServerMessage); }
message ClientMessage {
  string type = 1;
  string request_id = 2;
  string text = 3;
  string priority = 4;
  bool interrupt = 5;
}
message ServerMessage {
  string type = 1;
  string request_id = 2;
  bytes audio = 3;
  string json = 4;
}
```

Test a real `grpc.aio` loopback stream with the fake host and assert ready,
audio bytes, completion, and cancellation.

- [ ] **Step 2: Generate stubs and verify RED**

Run: `../../.venv/bin/python -m grpc_tools.protoc -I geno_voice/endpoint/proto --python_out=geno_voice/endpoint/proto --grpc_python_out=geno_voice/endpoint/proto geno_voice/endpoint/proto/tts_endpoint.proto && ../../.venv/bin/python -m pytest tests/integration/test_endpoint_grpc.py -q`

Expected: loopback test fails because the servicer/launcher is absent.

- [ ] **Step 3: Implement concurrent gRPC receive/event pumps**

```python
class TTSServicer(tts_endpoint_pb2_grpc.TTSServicer):
    async def Stream(self, request_iterator, context):
        session = await self._host.open_session()
        receiver = asyncio.create_task(receive_requests(request_iterator, session))
        try:
            async for event in session.events():
                yield event_to_proto(event)
        finally:
            receiver.cancel()
            await session.close()
```

Translate validation/model errors into typed error events rather than gRPC
process failures. Import grpc lazily and provide the endpoint-extra install
hint when missing.

- [ ] **Step 4: Run gRPC tests and verify GREEN**

Run: `../../.venv/bin/python -m pytest tests/integration/test_endpoint_grpc.py -q`

Expected: real loopback bidi stream passes.

- [ ] **Step 5: Commit**

```bash
git add geno_voice/endpoint/proto geno_voice/endpoint/transports/grpc.py tests/integration/test_endpoint_grpc.py
git commit -m "feat: serve TTS sessions over bidirectional gRPC"
```

### Task 6: WebRTC transport

**Files:**
- Create: `geno_voice/endpoint/transports/webrtc.py`
- Create: `tests/unit/test_endpoint_webrtc.py`

**Interfaces:**
- Consumes: `SynthesisSession`, SDP offer, ordered `geno-voice-control` data channel.
- Produces: signaling FastAPI app, `SessionAudioTrack`, and `serve_webrtc`.

- [ ] **Step 1: Write failing signaling/data/audio tests with peer fakes**

```python
async def test_offer_returns_answer_and_binds_control_channel(fake_peer_factory, fake_host):
    app = create_webrtc_app(fake_host, peer_factory=fake_peer_factory)
    response = await post_offer(app, sdp="offer", type="offer")
    assert response.json()["type"] == "answer"
    peer = fake_peer_factory.created[0]
    await peer.control.emit({"type": "speak", "request_id": "r1", "text": "Hi"})
    assert await peer.audio_track.next_pcm() == b"\x01\x00"

async def test_peer_disconnect_closes_session(fake_peer_factory, fake_host):
    await establish_peer(fake_peer_factory, fake_host)
    await fake_peer_factory.created[0].set_state("closed")
    assert fake_host.sessions[0].is_closed
```

- [ ] **Step 2: Run tests and verify RED**

Run: `../../.venv/bin/python -m pytest tests/unit/test_endpoint_webrtc.py -q`

Expected: WebRTC module is missing.

- [ ] **Step 3: Implement signaling, data-channel commands, and audio track**

```python
class SessionAudioTrack(MediaStreamTrack):
    kind = "audio"

    async def recv(self) -> AudioFrame:
        pcm = await self._audio_queue.get()
        return resample_pcm24k_to_audio_frame48k(pcm, pts=self._next_pts)
```

Use `RTCPeerConnection`, `RTCSessionDescription`, and PyAV resampling through
lazy imports. Send non-audio events on the data channel. Close peer/session on
either side's disconnect. Do not configure STUN/TURN.

- [ ] **Step 4: Run fake and optional real-aiortc tests**

Run: `../../.venv/bin/python -m pytest tests/unit/test_endpoint_webrtc.py -q`

Expected: signaling, routing, audio timing, and cleanup pass without requiring
a real browser.

- [ ] **Step 5: Commit**

```bash
git add geno_voice/endpoint/transports/webrtc.py tests/unit/test_endpoint_webrtc.py
git commit -m "feat: serve TTS sessions over WebRTC"
```

### Task 7: RTP/RTCP transport, operator docs, and full verification

**Files:**
- Create: `geno_voice/endpoint/transports/rtp.py`
- Create: `tests/unit/test_endpoint_rtp.py`
- Create: `tests/integration/test_endpoint_rtp.py`
- Modify: `README.md`
- Modify: `docs/index.md`

**Interfaces:**
- Consumes: RTP session-create/control HTTP calls and session audio events.
- Produces: RTP packetizer, RTCP sender reports, SSE events, SDP description, and `serve_rtp`.

- [ ] **Step 1: Write failing packet and loopback tests**

```python
def test_rtp_packet_has_sequence_timestamp_ssrc_and_big_endian_l16():
    packet = packetize_l16(b"\x01\x00\x02\x00", sequence=9, timestamp=480, ssrc=7)
    assert parse_header(packet) == {"payload_type": 96, "sequence": 9, "timestamp": 480, "ssrc": 7}
    assert packet[12:] == b"\x00\x01\x00\x02"

async def test_rtp_http_control_sends_udp_and_cancel_stops_packets(fake_host, udp_receiver):
    session = await create_rtp_session(target=udp_receiver.address)
    await send_command(session, {"type": "speak", "request_id": "r1", "text": "Hi"})
    assert (await udp_receiver.packet()).payload_type == 96
    await send_command(session, {"type": "cancel", "request_id": "r1"})
    assert await udp_receiver.no_packet_for(0.1)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `../../.venv/bin/python -m pytest tests/unit/test_endpoint_rtp.py tests/integration/test_endpoint_rtp.py -q`

Expected: RTP transport is missing.

- [ ] **Step 3: Implement HTTP control, RTP L16, RTCP reports, SSE, and SDP**

```python
def packetize_l16(pcm_le: bytes, *, sequence: int, timestamp: int, ssrc: int) -> bytes:
    header = struct.pack("!BBHII", 0x80, 96, sequence & 0xFFFF, timestamp & 0xFFFFFFFF, ssrc)
    samples = array("h", pcm_le)
    samples.byteswap()
    return header + samples.tobytes()
```

Split PCM into 20 ms/480-sample RTP packets, pace delivery against the event
loop clock, wrap sequence/timestamp, send periodic RTCP sender reports, emit
session events over SSE, and delete/cancel session state idempotently.

- [ ] **Step 4: Document install, commands, contracts, and Z2 smoke procedure**

```markdown
pip install -e '.[endpoint]'
geno-voice start-endpoint --protocol websocket --model kokoro --host 0.0.0.0
geno-voice start-endpoint --protocol webrtc --model Breeze-TTS-2 \
  --host 0.0.0.0 --model-path /models/Breeze-TTS-2 --runtime-path /opt/breeze-tts
```

Document each protocol's connection surface, model plugin entry-point group,
Breeze license/runtime prerequisites, cancellation check, and LAN-only warning.

- [ ] **Step 5: Run focused protocol suite and full unit suite**

Run: `../../.venv/bin/python -m pytest tests/unit/test_endpoint_*.py tests/integration/test_endpoint_*.py -q`

Run: `../../.venv/bin/python -m pytest tests/unit -q`

Expected: endpoint protocol suite and all existing unit tests pass.

- [ ] **Step 6: Exercise installed CLI help without endpoint extras**

Run: `PYTHONPATH=. ../../.venv/bin/python -m geno_voice start-endpoint --help`

Expected: help lists all protocols/model options without importing CUDA,
Breeze, grpc, or aiortc.

- [ ] **Step 7: Commit**

```bash
git add geno_voice/endpoint/transports/rtp.py tests/unit/test_endpoint_rtp.py tests/integration/test_endpoint_rtp.py README.md docs/index.md
git commit -m "feat: complete multi-protocol TTS endpoint"
```
