"""Real FastAPI/TestClient loopback tests for the WebSocket endpoint."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from geno_voice.endpoint.host import EndpointHost
from geno_voice.endpoint.transports.websocket import create_websocket_app
from geno_voice.endpoint.transports.wire import decode_audio_envelope
from geno_voice.endpoint.types import (
    AudioChunk,
    CancellationToken,
    ModelCapabilities,
    SynthesisRequest,
)


class FakeStreamingModel:
    name = "fake-streaming"
    capabilities = ModelCapabilities(streaming=True, rate_control=True)

    async def synthesize(
        self, request: SynthesisRequest, cancellation: CancellationToken
    ) -> AsyncIterator[AudioChunk]:
        if not cancellation.cancelled:
            yield AudioChunk(pcm=b"\x01\x00\x02\x00", final=True)


def receive_until_binary(socket):
    json_events = []
    while True:
        message = socket.receive()
        if message.get("bytes") is not None:
            return json_events, message["bytes"]
        json_events.append(message["text"])


def receive_json_type(socket, event_type: str):
    while True:
        event = socket.receive_json()
        if event["type"] == event_type:
            return event


def test_websocket_speak_returns_binary_audio_and_completion() -> None:
    host = EndpointHost(FakeStreamingModel())
    with TestClient(create_websocket_app(host)) as client:
        with client.websocket_connect("/v1/tts/stream") as socket:
            ready = socket.receive_json()
            assert ready["type"] == "ready"
            assert ready["model"] == "fake-streaming"

            socket.send_json(
                {"type": "speak", "request_id": "r1", "text": "Hello"}
            )
            _, packet = receive_until_binary(socket)
            header, pcm = decode_audio_envelope(packet)
            assert header["request_id"] == "r1"
            assert header["sample_count"] == 2
            assert pcm == b"\x01\x00\x02\x00"
            assert receive_json_type(socket, "completed")["request_id"] == "r1"

    deadline = time.monotonic() + 1.0
    while host.session_count and time.monotonic() < deadline:
        time.sleep(0.01)
    assert host.session_count == 0


def test_websocket_bad_json_emits_error_and_connection_stays_usable() -> None:
    host = EndpointHost(FakeStreamingModel())
    with TestClient(create_websocket_app(host)) as client:
        with client.websocket_connect("/v1/tts/stream") as socket:
            socket.receive_json()
            socket.send_text("{not-json")
            error = socket.receive_json()
            assert error["type"] == "error"
            assert error["code"] == "INVALID_JSON"

            socket.send_json(
                {"type": "speak", "request_id": "r2", "text": "Still alive"}
            )
            _, packet = receive_until_binary(socket)
            assert decode_audio_envelope(packet)[0]["request_id"] == "r2"


def test_websocket_health_and_capabilities_describe_loaded_host() -> None:
    host = EndpointHost(FakeStreamingModel())
    with TestClient(create_websocket_app(host)) as client:
        health = client.get("/health")
        capabilities = client.get("/v1/capabilities")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "model": "fake-streaming",
        "sessions": 0,
    }
    assert capabilities.json()["audio"] == {
        "encoding": "pcm_s16le",
        "sample_rate": 24_000,
        "channels": 1,
    }
    assert capabilities.json()["capabilities"]["rate_control"] is True
