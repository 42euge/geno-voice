"""RTP/RTCP packet and SSE framing tests."""

from __future__ import annotations

import json
import struct

from geno_voice.endpoint.transports.rtp import (
    encode_sse,
    packetize_l16,
    packetize_rtcp_sender_report,
)


def test_rtp_packet_has_sequence_timestamp_ssrc_and_big_endian_l16() -> None:
    packet = packetize_l16(
        b"\x01\x00\x02\x00", sequence=9, timestamp=480, ssrc=7
    )

    first, payload_type, sequence, timestamp, ssrc = struct.unpack(
        "!BBHII", packet[:12]
    )
    assert {
        "version": first >> 6,
        "payload_type": payload_type & 0x7F,
        "sequence": sequence,
        "timestamp": timestamp,
        "ssrc": ssrc,
    } == {
        "version": 2,
        "payload_type": 96,
        "sequence": 9,
        "timestamp": 480,
        "ssrc": 7,
    }
    assert packet[12:] == b"\x00\x01\x00\x02"


def test_rtp_packet_wraps_sequence_and_timestamp() -> None:
    packet = packetize_l16(
        b"\x00\x00", sequence=65_537, timestamp=0x1_0000_0002, ssrc=3
    )

    _, _, sequence, timestamp, _ = struct.unpack("!BBHII", packet[:12])
    assert sequence == 1
    assert timestamp == 2


def test_rtcp_sender_report_contains_counts_and_rtp_clock() -> None:
    packet = packetize_rtcp_sender_report(
        ssrc=7,
        rtp_timestamp=960,
        packet_count=2,
        octet_count=1_920,
        now=1_700_000_000.5,
    )

    first, packet_type, length, ssrc = struct.unpack("!BBHI", packet[:8])
    rtp_timestamp, packet_count, octet_count = struct.unpack("!III", packet[16:28])
    assert first >> 6 == 2
    assert packet_type == 200
    assert length == 6
    assert ssrc == 7
    assert rtp_timestamp == 960
    assert packet_count == 2
    assert octet_count == 1_920


def test_sse_event_is_one_json_data_record() -> None:
    encoded = encode_sse({"type": "cancelled", "request_id": "r1"})

    assert encoded.startswith("data: ")
    assert encoded.endswith("\n\n")
    assert json.loads(encoded[6:-2]) == {
        "type": "cancelled",
        "request_id": "r1",
    }
