"""Lightweight CLI configuration for launching a streaming TTS endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .host import run_endpoint
from .registry import ModelRegistry


PROTOCOL_ALIASES = {
    "websocket": "websocket",
    "ws": "websocket",
    "grpc": "grpc",
    "webrtc": "webrtc",
    "rtp": "rtp",
}

DEFAULT_PORTS = {
    "websocket": 8_765,
    "grpc": 50_051,
    "webrtc": 8_787,
    "rtp": 8_790,
}


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
    log_level: str = "info"

    @property
    def resolved_port(self) -> int:
        return self.port if self.port is not None else DEFAULT_PORTS[self.protocol]


def endpoint_config_from_args(args: Any) -> EndpointConfig:
    return EndpointConfig(
        protocol=args.protocol,
        model=args.model,
        host=args.host,
        port=args.port if args.port is not None else DEFAULT_PORTS[args.protocol],
        model_path=args.model_path,
        runtime_path=args.runtime_path,
        device=args.device,
        voice=args.voice,
        log_level=args.log_level,
    )


def print_models(*, log=print) -> None:
    registry = ModelRegistry()
    for name in registry.names():
        descriptor = registry.resolve(name)
        log(f"{name}\t{descriptor.origin}")
