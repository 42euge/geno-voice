"""Shared argument definitions for both TTS server command forms."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable


def model_type(raw: str) -> str:
    """Accept model aliases, repository IDs, or paths without whitespace."""
    if not raw or raw != raw.strip() or any(character.isspace() for character in raw):
        raise argparse.ArgumentTypeError(
            f"model must be a non-empty id without whitespace, got {raw!r}"
        )
    return raw


def endpoint_protocol_type(raw: str) -> str:
    """Normalize supported endpoint transport names."""
    aliases = {
        "websocket": "websocket",
        "ws": "websocket",
        "grpc": "grpc",
        "webrtc": "webrtc",
    }
    normalized = str(raw).strip().lower()
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(aliases)
        raise argparse.ArgumentTypeError(
            f"protocol must be one of {supported}, got {raw!r}"
        ) from exc


def port_type(raw: str) -> int:
    """Parse an IANA-valid TCP or UDP port."""
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"port must be an integer, got {raw!r}"
        ) from exc
    if not 1 <= value <= 65_535:
        raise argparse.ArgumentTypeError(f"port must be in [1, 65535], got {value}")
    return value


def add_endpoint_arguments(
    parser: argparse.ArgumentParser,
    *,
    protocol_type: Callable[[str], str],
    model_type: Callable[[str], str],
    port_type: Callable[[str], int],
) -> None:
    """Attach the complete streaming-endpoint option set to ``parser``."""
    parser.add_argument(
        "--protocol",
        type=protocol_type,
        default="websocket",
        help="websocket/ws, grpc, or webrtc (default: websocket)",
    )
    parser.add_argument(
        "--model",
        type=model_type,
        default="kokoro",
        help="Built-in or installed geno_voice.tts_models adapter (default: kokoro)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=port_type)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--runtime-path", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--voice")
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List built-in and installed model adapters, then exit",
    )
