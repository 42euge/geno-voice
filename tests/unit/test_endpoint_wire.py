"""Tests for transport wire encoding and JSON command validation."""

from __future__ import annotations

import pytest

from geno_voice.endpoint.transports.wire import (
    WireCommandError,
    command_from_mapping,
    decode_audio_envelope,
    encode_audio_envelope,
)
from geno_voice.endpoint.types import EndpointEvent


def audio_event() -> EndpointEvent:
    return EndpointEvent(
        type="audio",
        session_id="s1",
        request_id="r1",
        sequence=7,
        pts_samples=480,
        sample_count=1,
        sample_rate=24_000,
        encoding="pcm_s16le",
        final=False,
        audio=b"\x01\x00",
    )


def test_audio_envelope_round_trip_is_atomic_and_self_describing() -> None:
    packet = encode_audio_envelope(audio_event())

    header, pcm = decode_audio_envelope(packet)

    assert packet[:4] == b"GVA1"
    assert header == {
        "type": "audio",
        "session_id": "s1",
        "request_id": "r1",
        "sequence": 7,
        "pts_samples": 480,
        "sample_count": 1,
        "sample_rate": 24_000,
        "encoding": "pcm_s16le",
        "final": False,
    }
    assert pcm == b"\x01\x00"


@pytest.mark.parametrize(
    "packet",
    [
        b"BAD!\x00\x02{}",
        b"GVA1\x00",
        b"GVA1\x00\x08{}",
        b"GVA1\x00\x02[]",
    ],
)
def test_audio_envelope_rejects_corrupt_or_non_object_headers(packet) -> None:
    with pytest.raises(ValueError):
        decode_audio_envelope(packet)


def test_json_mapping_preserves_interrupt_and_model_options() -> None:
    command = command_from_mapping(
        {
            "type": "speak",
            "request_id": "cue-1",
            "text": "Mm-hmm.",
            "priority": "backchannel",
            "interrupt": True,
            "voice": "S0",
            "speed": 1.1,
            "instruction": "Quiet acknowledgement",
        }
    )

    assert command.request_id == "cue-1"
    assert command.priority == "backchannel"
    assert command.interrupt is True
    assert command.voice == "S0"
    assert command.speed == 1.1


def test_json_mapping_rejects_unknown_command_with_stable_code() -> None:
    with pytest.raises(WireCommandError) as exc:
        command_from_mapping({"type": "dance", "request_id": "r1"})

    assert exc.value.code == "INVALID_COMMAND"
    assert exc.value.request_id == "r1"
