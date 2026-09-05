"""Adapter for geno-voice's existing Kokoro WAV-streaming engine."""

from __future__ import annotations

import asyncio
import io
import wave
from typing import Any

from ..types import (
    AudioChunk,
    CancellationToken,
    ModelCapabilities,
    SynthesisRequest,
)
from . import stream_sync_iterator


class KokoroAdapter:
    name = "kokoro"
    capabilities = ModelCapabilities(streaming=True, rate_control=True)

    def __init__(
        self, *, engine: Any = None, default_voice: str = "af_heart"
    ) -> None:
        self._engine = engine
        self.default_voice = default_voice

    async def load(self) -> None:
        if self._engine is None:
            from tts import get_engine

            self._engine = get_engine("kokoro")
        loader = getattr(self._engine, "_load", None)
        if loader is not None:
            await asyncio.to_thread(loader)

    async def synthesize(
        self, request: SynthesisRequest, cancellation: CancellationToken
    ):
        if self._engine is None:
            raise RuntimeError("Kokoro adapter has not been loaded")
        voice = request.voice or self.default_voice
        speed = request.speed if request.speed is not None else 1.0

        def chunks():
            wav_chunks = self._engine.stream(request.text, voice, speed)
            return (self._unwrap_wav(wav_bytes) for wav_bytes in wav_chunks)

        async for pcm in stream_sync_iterator(chunks, cancellation):
            if pcm:
                yield AudioChunk(pcm=pcm)

    @staticmethod
    def _unwrap_wav(wav_bytes: bytes) -> bytes:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError("Kokoro must emit mono signed 16-bit WAV chunks")
            if wav.getframerate() != 24_000:
                raise ValueError(
                    f"Kokoro emitted {wav.getframerate()} Hz WAV; expected 24000 Hz"
                )
            return wav.readframes(wav.getnframes())

    async def close(self) -> None:
        close = getattr(self._engine, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result
