"""Installed ``geno-voice`` console entrypoint."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build the lightweight top-level command parser."""
    parser = argparse.ArgumentParser(prog="geno-voice")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "start-endpoint",
        add_help=False,
        help="Serve a TTS model over WebSocket, gRPC, WebRTC, or RTP",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch a geno-voice command without importing optional runtimes."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parsed, remaining = build_parser().parse_known_args(arguments)

    if parsed.command == "start-endpoint":
        from geno_voice.endpoint import cli as endpoint_cli

        return endpoint_cli.main(remaining)
    raise AssertionError(f"unhandled command: {parsed.command}")
