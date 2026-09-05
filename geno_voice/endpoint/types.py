"""Transport-neutral command, event, and model types for TTS endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class AlignmentSpan:
    """Timing for a source-text span in canonical audio samples."""

    text: str
    start_sample: int
    end_sample: int
    source_start: int | None = None
    source_end: int | None = None


@dataclass(frozen=True)
class ModelCapabilities:
    """Features an adapter can truthfully provide."""

    streaming: bool = True
    alignment: bool = False
    voice_cloning: bool = False
    voice_design: bool = False
    rate_control: bool = False
    sample_rate: int = 24_000


@dataclass(frozen=True)
class SynthesisRequest:
    request_id: str
    text: str
    priority: str = "normal"
    voice: str | None = None
    speed: float | None = None
    instruction: str | None = None
    reference_audio: bytes | None = None
    reference_text: str | None = None


@dataclass(frozen=True)
class AudioChunk:
    pcm: bytes
    sample_rate: int = 24_000
    alignment: tuple[AlignmentSpan, ...] = ()
    final: bool = False


@dataclass(frozen=True)
class EndpointCommand:
    type: str
    request_id: str | None = None
    text: str | None = None
    priority: str = "normal"
    interrupt: bool = False
    voice: str | None = None
    speed: float | None = None
    instruction: str | None = None
    reference_audio: bytes | None = None
    reference_text: str | None = None

    @classmethod
    def append(cls, request_id: str, text: str) -> EndpointCommand:
        return cls(type="append", request_id=request_id, text=text)

    @classmethod
    def commit(cls, request_id: str) -> EndpointCommand:
        return cls(type="commit", request_id=request_id)

    @classmethod
    def speak(
        cls,
        request_id: str,
        text: str,
        *,
        priority: str = "normal",
        interrupt: bool = False,
        voice: str | None = None,
        speed: float | None = None,
        instruction: str | None = None,
        reference_audio: bytes | None = None,
        reference_text: str | None = None,
    ) -> EndpointCommand:
        return cls(
            type="speak",
            request_id=request_id,
            text=text,
            priority=priority,
            interrupt=interrupt,
            voice=voice,
            speed=speed,
            instruction=instruction,
            reference_audio=reference_audio,
            reference_text=reference_text,
        )

    @classmethod
    def cancel(cls, request_id: str) -> EndpointCommand:
        return cls(type="cancel", request_id=request_id)

    @classmethod
    def supersede(
        cls,
        request_id: str,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
    ) -> EndpointCommand:
        return cls(
            type="supersede",
            request_id=request_id,
            text=text,
            voice=voice,
            speed=speed,
        )

    @classmethod
    def close(cls) -> EndpointCommand:
        return cls(type="close")


@dataclass(frozen=True)
class EndpointEvent:
    type: str
    session_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    capabilities: ModelCapabilities | None = None
    priority: str | None = None
    queue_depth: int | None = None
    audio: bytes | None = None
    sequence: int | None = None
    pts_samples: int | None = None
    sample_count: int | None = None
    sample_rate: int | None = None
    encoding: str | None = None
    final: bool | None = None
    alignment: tuple[AlignmentSpan, ...] = ()
    total_samples: int | None = None
    interrupted: bool | None = None
    code: str | None = None
    message: str | None = None

    def to_dict(self, *, include_audio: bool = False) -> dict[str, Any]:
        """Return the stable JSON-compatible event representation."""

        result: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if value is None or (key == "alignment" and not value):
                continue
            if key == "audio":
                if include_audio:
                    result[key] = value
                continue
            result[key] = value
        return result


class CancellationToken:
    """Cooperative cancellation shared with model adapters."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()


class TTSModelAdapter(Protocol):
    name: str
    capabilities: ModelCapabilities

    async def load(self) -> None: ...

    async def close(self) -> None: ...

    def synthesize(
        self, request: SynthesisRequest, cancellation: CancellationToken
    ) -> AsyncIterator[AudioChunk]: ...
