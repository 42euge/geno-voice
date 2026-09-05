"""Model-agnostic streaming TTS endpoint primitives."""

from .session import SynthesisSession
from .types import (
    AlignmentSpan,
    AudioChunk,
    CancellationToken,
    EndpointCommand,
    EndpointEvent,
    ModelCapabilities,
    SynthesisRequest,
    TTSModelAdapter,
)

__all__ = [
    "AlignmentSpan",
    "AudioChunk",
    "CancellationToken",
    "EndpointCommand",
    "EndpointEvent",
    "ModelCapabilities",
    "SynthesisRequest",
    "SynthesisSession",
    "TTSModelAdapter",
]
