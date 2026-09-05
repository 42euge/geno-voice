"""Lifecycle tests for a loaded model shared by endpoint sessions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from geno_voice.endpoint.cli import EndpointConfig
from geno_voice.endpoint.host import (
    EndpointHost,
    ensure_protocol_dependencies,
    run_endpoint_async,
)
from geno_voice.endpoint.registry import ModelConfig
from geno_voice.endpoint.types import (
    AudioChunk,
    CancellationToken,
    ModelCapabilities,
    SynthesisRequest,
)


class FakeAdapter:
    name = "fake"
    capabilities = ModelCapabilities(streaming=True)

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def load(self) -> None:
        self.calls.append("model.load")

    async def synthesize(
        self, request: SynthesisRequest, cancellation: CancellationToken
    ) -> AsyncIterator[AudioChunk]:
        yield AudioChunk(pcm=b"\x01\x00")

    async def close(self) -> None:
        self.calls.append("model.close")


class FakeRegistry:
    def __init__(self, adapter: FakeAdapter, calls: list[str]) -> None:
        self.adapter = adapter
        self.calls = calls

    def create(self, name: str, config: ModelConfig):
        self.calls.append(f"registry.create:{name}")
        assert config == ModelConfig(device="cuda:1", voice="voice-a")
        return self.adapter


def test_endpoint_runner_loads_model_before_serving_and_always_closes() -> None:
    async def scenario() -> None:
        calls: list[str] = []
        adapter = FakeAdapter(calls)
        registry = FakeRegistry(adapter, calls)

        async def serve(host, *, bind, port, log_level):
            calls.append(f"serve:{bind}:{port}:{log_level}")
            assert host.model is adapter
            session = await host.open_session(session_id="s1")
            assert session.session_id == "s1"

        await run_endpoint_async(
            EndpointConfig(
                protocol="websocket",
                model="fake",
                host="0.0.0.0",
                port=9000,
                device="cuda:1",
                voice="voice-a",
                log_level="debug",
            ),
            registry=registry,
            transport_loader=lambda protocol: serve,
            dependency_checker=lambda protocol: None,
            log=lambda line: calls.append(f"log:{line}"),
        )

        assert calls.index("model.load") < calls.index("serve:0.0.0.0:9000:debug")
        assert calls[-1] == "model.close"
        assert any("no authentication or TLS" in call for call in calls)

    asyncio.run(scenario())


def test_endpoint_host_creates_independent_sessions_on_one_model() -> None:
    async def scenario() -> None:
        adapter = FakeAdapter([])
        host = EndpointHost(adapter, default_voice="voice-a")

        first = await host.open_session(session_id="one")
        second = await host.open_session(session_id="two")

        assert first is not second
        assert host.session_count == 2
        await host.close_session("one")
        assert host.session_count == 1
        await host.close()
        assert host.session_count == 0

    asyncio.run(scenario())


def test_websocket_preflight_requires_a_websocket_protocol_driver(monkeypatch) -> None:
    available = {"fastapi", "uvicorn"}
    monkeypatch.setattr(
        "geno_voice.endpoint.host.importlib.util.find_spec",
        lambda name: object() if name in available else None,
    )

    with pytest.raises(RuntimeError, match=r"websockets.*geno-voice\[endpoint\]"):
        ensure_protocol_dependencies("websocket")


def test_breeze_license_warning_is_printed_at_launch() -> None:
    async def scenario() -> None:
        calls: list[str] = []
        adapter = FakeAdapter(calls)
        adapter.license_warning = "research/non-commercial test warning"

        await run_endpoint_async(
            EndpointConfig(protocol="websocket", model="breeze-tts-2"),
            registry=FakeRegistryForWarning(adapter),
            transport_loader=lambda protocol: no_op_server,
            dependency_checker=lambda protocol: None,
            log=calls.append,
        )

        assert "WARNING: research/non-commercial test warning" in calls

    asyncio.run(scenario())


class FakeRegistryForWarning:
    def __init__(self, adapter) -> None:
        self.adapter = adapter

    def create(self, name, config):
        return self.adapter


async def no_op_server(host, *, bind, port, log_level):
    return None
