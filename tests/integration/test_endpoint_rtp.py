"""HTTP-control plus real UDP loopback tests for the RTP endpoint."""

from __future__ import annotations

import asyncio
import socket
import struct
from array import array
from collections.abc import AsyncIterator

import httpx

from geno_voice.endpoint.host import EndpointHost
from geno_voice.endpoint.transports.rtp import create_rtp_app
from geno_voice.endpoint.types import (
    AudioChunk,
    CancellationToken,
    ModelCapabilities,
    SynthesisRequest,
)


class LargeChunkModel:
    name = "rtp-fake"
    capabilities = ModelCapabilities(streaming=True)

    async def synthesize(
        self, request: SynthesisRequest, cancellation: CancellationToken
    ) -> AsyncIterator[AudioChunk]:
        yield AudioChunk(pcm=array("h", range(480 * 20)).tobytes(), final=True)


def udp_receiver() -> socket.socket:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.setblocking(False)
    return receiver


def parse_rtp(packet: bytes) -> dict[str, int | bytes]:
    _, payload_type, sequence, timestamp, ssrc = struct.unpack("!BBHII", packet[:12])
    return {
        "payload_type": payload_type & 0x7F,
        "sequence": sequence,
        "timestamp": timestamp,
        "ssrc": ssrc,
        "payload": packet[12:],
    }


def test_rtp_http_control_sends_udp_and_cancel_stops_pacing() -> None:
    async def scenario() -> None:
        receiver = udp_receiver()
        host = EndpointHost(LargeChunkModel())
        app = create_rtp_app(host)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                created = await client.post(
                    "/v1/rtp/sessions",
                    json={
                        "destination_host": "127.0.0.1",
                        "destination_port": receiver.getsockname()[1],
                    },
                )
                assert created.status_code == 201
                body = created.json()
                assert "a=rtpmap:96 L16/24000/1" in body["sdp"]
                session_id = body["session_id"]

                command = await client.post(
                    f"/v1/rtp/sessions/{session_id}/commands",
                    json={"type": "speak", "request_id": "r1", "text": "Hi"},
                )
                assert command.status_code == 202

                first_packet, _ = await asyncio.wait_for(
                    asyncio.get_running_loop().sock_recvfrom(receiver, 2_048), 1.0
                )
                parsed = parse_rtp(first_packet)
                assert parsed["payload_type"] == 96
                assert len(parsed["payload"]) == 960

                cancelled = await client.post(
                    f"/v1/rtp/sessions/{session_id}/commands",
                    json={"type": "cancel", "request_id": "r1"},
                )
                assert cancelled.status_code == 202

                await asyncio.sleep(0.06)
                while True:
                    try:
                        receiver.recvfrom(2_048)
                    except BlockingIOError:
                        break
                try:
                    await asyncio.wait_for(
                        asyncio.get_running_loop().sock_recvfrom(receiver, 2_048),
                        0.08,
                    )
                    raise AssertionError("RTP continued after cancellation")
                except TimeoutError:
                    pass

                deleted = await client.delete(f"/v1/rtp/sessions/{session_id}")
                assert deleted.status_code == 204
                deleted_again = await client.delete(
                    f"/v1/rtp/sessions/{session_id}"
                )
                assert deleted_again.status_code == 204
            assert host.session_count == 0
        finally:
            receiver.close()
            await host.close()

    asyncio.run(scenario())


def test_rtp_control_rejects_bad_command_without_deleting_session() -> None:
    async def scenario() -> None:
        receiver = udp_receiver()
        host = EndpointHost(LargeChunkModel())
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_rtp_app(host)),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/v1/rtp/sessions",
                    json={
                        "destination_host": "127.0.0.1",
                        "destination_port": receiver.getsockname()[1],
                    },
                )
                session_id = response.json()["session_id"]
                bad = await client.post(
                    f"/v1/rtp/sessions/{session_id}/commands",
                    json={"type": "dance", "request_id": "r1"},
                )
                assert bad.status_code == 400
                assert bad.json()["code"] == "INVALID_COMMAND"
                assert host.session_count == 1
                await client.delete(f"/v1/rtp/sessions/{session_id}")
        finally:
            receiver.close()
            await host.close()

    asyncio.run(scenario())
