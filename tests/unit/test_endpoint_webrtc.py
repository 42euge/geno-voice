"""Signaling, control-channel, audio, and cleanup tests for WebRTC."""

from __future__ import annotations

import asyncio
import inspect
import json
from array import array
from collections.abc import AsyncIterator
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from geno_voice.endpoint.host import EndpointHost
from geno_voice.endpoint.transports.webrtc import create_webrtc_app
from geno_voice.endpoint.types import (
    AudioChunk,
    CancellationToken,
    ModelCapabilities,
    SynthesisRequest,
)


class FakeWebRTCModel:
    name = "webrtc-fake"
    capabilities = ModelCapabilities(streaming=True)

    async def synthesize(
        self, request: SynthesisRequest, cancellation: CancellationToken
    ) -> AsyncIterator[AudioChunk]:
        pcm = array("h", [1_000] * 480).tobytes()
        yield AudioChunk(pcm=pcm, final=True)


class FakeChannel:
    label = "geno-voice-control"
    ordered = True
    readyState = "open"

    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}
        self.sent: list[str] = []
        self.closed = False

    def on(self, event):
        def register(callback):
            self.handlers.setdefault(event, []).append(callback)
            return callback

        return register

    def send(self, value):
        self.sent.append(value)

    def close(self):
        self.closed = True

    async def emit(self, event, *args):
        for callback in self.handlers.get(event, []):
            result = callback(*args)
            if inspect.isawaitable(result):
                await result
        await asyncio.sleep(0)


class FakePeer:
    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}
        self.connectionState = "new"
        self.audio_track = None
        self.remote = None
        self.localDescription = None
        self.closed = False

    def on(self, event):
        def register(callback):
            self.handlers.setdefault(event, []).append(callback)
            return callback

        return register

    def addTrack(self, track):
        self.audio_track = track

    async def setRemoteDescription(self, description):
        self.remote = description

    async def createAnswer(self):
        return SimpleNamespace(sdp="fake-answer", type="answer")

    async def setLocalDescription(self, description):
        self.localDescription = description

    async def close(self):
        self.closed = True
        self.connectionState = "closed"

    async def emit(self, event, *args):
        for callback in self.handlers.get(event, []):
            result = callback(*args)
            if inspect.isawaitable(result):
                await result
        await asyncio.sleep(0)


class FakePeerFactory:
    def __init__(self) -> None:
        self.created: list[FakePeer] = []

    def __call__(self):
        peer = FakePeer()
        self.created.append(peer)
        return peer


def test_offer_binds_control_channel_and_resamples_session_audio() -> None:
    async def scenario() -> None:
        host = EndpointHost(FakeWebRTCModel())
        peers = FakePeerFactory()
        app = create_webrtc_app(host, peer_factory=peers)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/webrtc/offer", json={"sdp": "fake-offer", "type": "offer"}
            )

        assert response.status_code == 200
        assert response.json() == {"sdp": "fake-answer", "type": "answer"}
        peer = peers.created[0]
        assert peer.remote.sdp == "fake-offer"
        assert peer.audio_track.kind == "audio"

        control = FakeChannel()
        await peer.emit("datachannel", control)
        await control.emit(
            "message",
            json.dumps({"type": "speak", "request_id": "r1", "text": "Hi"}),
        )
        frame = await asyncio.wait_for(peer.audio_track.recv(), 1.0)

        assert frame.sample_rate == 48_000
        assert frame.samples == 960
        assert frame.pts == 0
        event_types = [json.loads(value)["type"] for value in control.sent]
        assert "ready" in event_types
        assert "accepted" in event_types
        assert "started" in event_types
        assert "completed" in event_types

        peer.connectionState = "closed"
        await peer.emit("connectionstatechange")
        assert host.session_count == 0

    asyncio.run(scenario())


def test_peer_disconnect_closes_only_its_session() -> None:
    async def scenario() -> None:
        host = EndpointHost(FakeWebRTCModel())
        peers = FakePeerFactory()
        app = create_webrtc_app(host, peer_factory=peers)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/v1/webrtc/offer", json={"sdp": "one", "type": "offer"}
            )
            await client.post(
                "/v1/webrtc/offer", json={"sdp": "two", "type": "offer"}
            )

        assert host.session_count == 2
        peers.created[0].connectionState = "failed"
        await peers.created[0].emit("connectionstatechange")

        assert host.session_count == 1
        assert peers.created[0].closed is True
        assert peers.created[1].closed is False
        await host.close()

    asyncio.run(scenario())


def test_offer_rejects_malformed_signaling_without_opening_session() -> None:
    async def scenario() -> None:
        host = EndpointHost(FakeWebRTCModel())
        app = create_webrtc_app(host, peer_factory=FakePeerFactory())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/webrtc/offer", json={"type": "answer"}
            )

        assert response.status_code == 400
        assert host.session_count == 0

    asyncio.run(scenario())


def test_app_shutdown_closes_active_peers_and_sessions() -> None:
    host = EndpointHost(FakeWebRTCModel())
    peers = FakePeerFactory()
    app = create_webrtc_app(host, peer_factory=peers)

    with TestClient(app) as client:
        response = client.post(
            "/v1/webrtc/offer", json={"sdp": "fake-offer", "type": "offer"}
        )
        assert response.status_code == 200
        assert host.session_count == 1

    assert peers.created[0].closed is True
    assert host.session_count == 0
