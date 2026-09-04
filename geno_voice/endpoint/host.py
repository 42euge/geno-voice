"""Loaded-model lifecycle and session hosting for endpoint transports."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import uuid
from dataclasses import asdict
from typing import Any, Callable

from .registry import ModelConfig, ModelRegistry
from .session import SynthesisSession
from .types import TTSModelAdapter


TRANSPORT_TARGETS = {
    "websocket": "geno_voice.endpoint.transports.websocket:serve_websocket",
    "grpc": "geno_voice.endpoint.transports.grpc:serve_grpc",
    "webrtc": "geno_voice.endpoint.transports.webrtc:serve_webrtc",
    "rtp": "geno_voice.endpoint.transports.rtp:serve_rtp",
}

PROTOCOL_DEPENDENCIES = {
    "websocket": (("fastapi", "fastapi"), ("uvicorn", "uvicorn")),
    "grpc": (("grpc", "grpcio"),),
    "webrtc": (
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("aiortc", "aiortc"),
        ("av", "av"),
    ),
    "rtp": (("fastapi", "fastapi"), ("uvicorn", "uvicorn")),
}


class EndpointHost:
    """Share one loaded adapter across independent synthesis sessions."""

    def __init__(
        self, model: TTSModelAdapter, *, default_voice: str | None = None
    ) -> None:
        self.model = model
        self.default_voice = default_voice
        self._sessions: dict[str, SynthesisSession] = {}

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    async def open_session(self, *, session_id: str | None = None) -> SynthesisSession:
        identifier = session_id or uuid.uuid4().hex
        if identifier in self._sessions:
            raise ValueError(f"session {identifier!r} already exists")
        session = SynthesisSession(
            self.model,
            session_id=identifier,
            default_voice=self.default_voice,
        )
        self._sessions[identifier] = session
        try:
            await session.start()
        except Exception:
            self._sessions.pop(identifier, None)
            raise
        return session

    async def close_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            await session.close()

    async def close(self) -> None:
        sessions = tuple(self._sessions.values())
        self._sessions.clear()
        if sessions:
            await asyncio.gather(
                *(session.close() for session in sessions),
                return_exceptions=True,
            )


def ensure_protocol_dependencies(protocol: str) -> None:
    try:
        dependencies = PROTOCOL_DEPENDENCIES[protocol]
    except KeyError as exc:
        raise ValueError(f"unsupported endpoint protocol: {protocol}") from exc
    missing = [
        distribution
        for module, distribution in dependencies
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"{protocol} endpoint requires {names}; install with "
            "pip install 'geno-voice[endpoint]'"
        )


def load_transport(protocol: str):
    try:
        target = TRANSPORT_TARGETS[protocol]
    except KeyError as exc:
        raise ValueError(f"unsupported endpoint protocol: {protocol}") from exc
    module_name, attribute = target.split(":", 1)
    return getattr(importlib.import_module(module_name), attribute)


async def run_endpoint_async(
    config,
    *,
    registry: ModelRegistry | None = None,
    transport_loader: Callable[[str], Any] = load_transport,
    dependency_checker: Callable[[str], None] = ensure_protocol_dependencies,
    log: Callable[[str], None] = print,
) -> None:
    """Load one model, serve one protocol, and release both on shutdown."""

    dependency_checker(config.protocol)
    serve = transport_loader(config.protocol)
    registry = registry or ModelRegistry()
    adapter = registry.create(
        config.model,
        ModelConfig(
            model_path=config.model_path,
            runtime_path=config.runtime_path,
            device=config.device,
            voice=config.voice,
        ),
    )

    if config.host not in {"127.0.0.1", "localhost", "::1"}:
        log(
            "WARNING: endpoint is bound to the LAN with no authentication or TLS; "
            "use only on a trusted internal network."
        )
    license_warning = getattr(adapter, "license_warning", None)
    if license_warning:
        log(f"WARNING: {license_warning}")

    host = EndpointHost(adapter, default_voice=config.voice)
    try:
        await adapter.load()
        log(
            f"Loaded {adapter.name}: canonical mono 24000 Hz PCM16; "
            f"capabilities={asdict(adapter.capabilities)}"
        )
        log(
            f"Serving {config.protocol} on {config.host}:{config.resolved_port}"
        )
        await serve(
            host,
            bind=config.host,
            port=config.resolved_port,
            log_level=config.log_level,
        )
    finally:
        await host.close()
        await adapter.close()


def run_endpoint(config) -> None:
    asyncio.run(run_endpoint_async(config))
