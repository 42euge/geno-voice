"""Shared JSON command parsing and atomic binary audio envelopes."""

from __future__ import annotations

import base64
import binascii
import json
import struct
from collections.abc import Mapping
from typing import Any

from ..types import EndpointCommand, EndpointEvent


AUDIO_MAGIC = b"GVA1"
_PREFIX = struct.Struct("!4sH")


class WireCommandError(ValueError):
    def __init__(
        self, code: str, message: str, *, request_id: str | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.request_id = request_id


def encode_audio_envelope(event: EndpointEvent, pcm: bytes | None = None) -> bytes:
    if event.type != "audio":
        raise ValueError("GVA1 envelopes require an audio event")
    payload = event.audio if pcm is None else pcm
    if payload is None:
        raise ValueError("audio event has no PCM payload")
    header = json.dumps(
        event.to_dict(), separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    if len(header) > 65_535:
        raise ValueError("audio event header exceeds 65535 bytes")
    return _PREFIX.pack(AUDIO_MAGIC, len(header)) + header + payload


def decode_audio_envelope(packet: bytes) -> tuple[dict[str, Any], bytes]:
    if len(packet) < _PREFIX.size:
        raise ValueError("truncated GVA1 envelope")
    magic, header_length = _PREFIX.unpack_from(packet)
    if magic != AUDIO_MAGIC:
        raise ValueError("invalid GVA1 envelope magic")
    header_end = _PREFIX.size + header_length
    if len(packet) < header_end:
        raise ValueError("truncated GVA1 JSON header")
    try:
        header = json.loads(packet[_PREFIX.size:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid GVA1 JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError("GVA1 JSON header must be an object")
    return header, packet[header_end:]


def command_from_mapping(value: Mapping[str, Any]) -> EndpointCommand:
    if not isinstance(value, Mapping):
        raise WireCommandError("INVALID_COMMAND", "command must be a JSON object")
    command_type = value.get("type")
    if not isinstance(command_type, str):
        raise WireCommandError("INVALID_COMMAND", "command type must be a string")
    request_id_value = value.get("request_id")
    request_id = request_id_value if isinstance(request_id_value, str) else None

    if command_type == "close":
        return EndpointCommand.close()
    if command_type not in {"append", "commit", "speak", "cancel", "supersede"}:
        raise WireCommandError(
            "INVALID_COMMAND",
            f"unknown command: {command_type}",
            request_id=request_id,
        )
    if not isinstance(request_id_value, str):
        raise WireCommandError(
            "INVALID_REQUEST_ID",
            "request_id must be a string",
            request_id=request_id,
        )

    if command_type == "commit":
        return EndpointCommand.commit(request_id_value)
    if command_type == "cancel":
        return EndpointCommand.cancel(request_id_value)

    text = value.get("text")
    if not isinstance(text, str):
        raise WireCommandError(
            "INVALID_TEXT", "text must be a string", request_id=request_id
        )
    if command_type == "append":
        return EndpointCommand.append(request_id_value, text)

    speed = value.get("speed")
    if speed is not None and (isinstance(speed, bool) or not isinstance(speed, (int, float))):
        raise WireCommandError(
            "INVALID_SPEED", "speed must be a number", request_id=request_id
        )
    interrupt = value.get("interrupt", False)
    if not isinstance(interrupt, bool):
        raise WireCommandError(
            "INVALID_INTERRUPT", "interrupt must be a boolean", request_id=request_id
        )
    priority = value.get("priority", "normal")
    if not isinstance(priority, str):
        raise WireCommandError(
            "INVALID_PRIORITY", "priority must be a string", request_id=request_id
        )
    voice = value.get("voice")
    if voice is not None and not isinstance(voice, str):
        raise WireCommandError(
            "INVALID_VOICE", "voice must be a string", request_id=request_id
        )
    instruction = value.get("instruction")
    if instruction is not None and not isinstance(instruction, str):
        raise WireCommandError(
            "INVALID_INSTRUCTION",
            "instruction must be a string",
            request_id=request_id,
        )
    reference_text = value.get("reference_text")
    if reference_text is not None and not isinstance(reference_text, str):
        raise WireCommandError(
            "INVALID_REFERENCE",
            "reference_text must be a string",
            request_id=request_id,
        )
    reference_audio = _decode_reference_audio(value.get("reference_audio"), request_id)

    if command_type == "supersede":
        return EndpointCommand.supersede(
            request_id_value,
            text,
            voice=voice,
            speed=float(speed) if speed is not None else None,
        )
    return EndpointCommand.speak(
        request_id_value,
        text,
        priority=priority,
        interrupt=interrupt,
        voice=voice,
        speed=float(speed) if speed is not None else None,
        instruction=instruction,
        reference_audio=reference_audio,
        reference_text=reference_text,
    )


def _decode_reference_audio(value: Any, request_id: str | None) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise WireCommandError(
            "INVALID_REFERENCE",
            "reference_audio must be base64 text",
            request_id=request_id,
        )
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise WireCommandError(
            "INVALID_REFERENCE",
            "reference_audio is not valid base64",
            request_id=request_id,
        ) from exc
