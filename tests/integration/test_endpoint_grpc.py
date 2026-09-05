"""Real grpc.aio loopback coverage for bidirectional TTS streaming."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import grpc

from geno_voice.endpoint.host import EndpointHost
from geno_voice.endpoint.proto import tts_endpoint_pb2, tts_endpoint_pb2_grpc
from geno_voice.endpoint.transports.grpc import TTSServicer
from geno_voice.endpoint.types import (
    AudioChunk,
    CancellationToken,
    ModelCapabilities,
    SynthesisRequest,
)


class FakeGrpcModel:
    name = "grpc-fake"
    capabilities = ModelCapabilities(streaming=True)

    async def synthesize(
        self, request: SynthesisRequest, cancellation: CancellationToken
    ) -> AsyncIterator[AudioChunk]:
        yield AudioChunk(pcm=b"\x01\x00\x02\x00", final=True)


async def start_server(host):
    server = grpc.aio.server()
    tts_endpoint_pb2_grpc.add_TTSServicer_to_server(TTSServicer(host), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    return server, port


def test_grpc_bidi_stream_returns_ready_audio_and_completion() -> None:
    async def scenario() -> None:
        host = EndpointHost(FakeGrpcModel())
        server, port = await start_server(host)
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        try:
            stub = tts_endpoint_pb2_grpc.TTSStub(channel)
            call = stub.Stream()
            ready = await call.read()
            assert ready.type == "ready"
            assert json.loads(ready.json)["model"] == "grpc-fake"

            await call.write(
                tts_endpoint_pb2.ClientMessage(
                    type="speak", request_id="r1", text="Hello"
                )
            )
            received = []
            while True:
                event = await call.read()
                received.append(event)
                if event.type == "completed":
                    break

            audio = next(event for event in received if event.type == "audio")
            assert audio.request_id == "r1"
            assert audio.audio == b"\x01\x00\x02\x00"
            assert audio.sample_count == 2

            await call.write(tts_endpoint_pb2.ClientMessage(type="close"))
            closed = await call.read()
            assert closed.type == "closed"
            await call.done_writing()
        finally:
            await channel.close()
            await server.stop(0)
            await host.close()

    asyncio.run(scenario())


def test_grpc_validation_error_does_not_abort_stream() -> None:
    async def scenario() -> None:
        host = EndpointHost(FakeGrpcModel())
        server, port = await start_server(host)
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        try:
            call = tts_endpoint_pb2_grpc.TTSStub(channel).Stream()
            await call.read()
            await call.write(
                tts_endpoint_pb2.ClientMessage(type="dance", request_id="r1")
            )
            error = await call.read()
            assert error.type == "error"
            assert error.code == "INVALID_COMMAND"

            await call.write(
                tts_endpoint_pb2.ClientMessage(
                    type="speak", request_id="r2", text="Still alive"
                )
            )
            while (await call.read()).type != "completed":
                pass
            await call.write(tts_endpoint_pb2.ClientMessage(type="close"))
            await call.read()
            await call.done_writing()
        finally:
            await channel.close()
            await server.stop(0)
            await host.close()

    asyncio.run(scenario())
