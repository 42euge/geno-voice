"""Regression coverage for the expressive-sample WebSocket client."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from geno_voice.endpoint.transports.wire import encode_audio_envelope
from geno_voice.endpoint.types import EndpointEvent


def load_generator() -> ModuleType:
    path = (
        Path(__file__).parents[2]
        / "samples"
        / "breeze-tts-2-expressive"
        / "generate.py"
    )
    spec = importlib.util.spec_from_file_location("expressive_sample_generator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSocket:
    def __init__(self, messages: list[str | bytes]) -> None:
        self.messages = iter(messages)
        self.sent: list[str] = []

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str | bytes:
        return next(self.messages)


def test_synthesis_continues_session_wide_chunk_sequence() -> None:
    generator = load_generator()
    pcm = b"\x01\x00\x02\x00"
    packet = encode_audio_envelope(
        EndpointEvent(
            type="audio",
            session_id="session-1",
            request_id="sample-2",
            sequence=48,
            pts_samples=96_000,
            sample_count=2,
            sample_rate=24_000,
            encoding="pcm_s16le",
        ),
        pcm,
    )
    socket = FakeSocket(
        [
            packet,
            json.dumps(
                {
                    "type": "completed",
                    "session_id": "session-1",
                    "request_id": "sample-2",
                    "total_samples": 2,
                }
            ),
        ]
    )

    audio, next_sequence = asyncio.run(
        generator.synthesize_sample(
            socket,
            {
                "id": "sample-2",
                "text": "Hello again.",
                "instruction": "Speak naturally.",
            },
            1.0,
            first_sequence=48,
        )
    )

    assert audio == pcm
    assert next_sequence == 49
