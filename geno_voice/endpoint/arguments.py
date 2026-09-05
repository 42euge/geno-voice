"""Shared argument definitions for both TTS server command forms."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable


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
        help="websocket/ws, grpc, webrtc, or rtp (default: websocket)",
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
