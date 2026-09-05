"""Typed bidirectional gRPC adapter for synthesis sessions."""

from __future__ import annotations

import asyncio
import base64
import json

try:
    import grpc
except ImportError as exc:  # pragma: no cover - protected by host preflight
    raise RuntimeError(
        "gRPC endpoint requires grpcio; install pip install 'geno-voice[endpoint]'"
    ) from exc

from ..proto import tts_endpoint_pb2, tts_endpoint_pb2_grpc
from .wire import WireCommandError, command_from_mapping


class TTSServicer(tts_endpoint_pb2_grpc.TTSServicer):
    def __init__(self, host) -> None:
        self._host = host

    async def Stream(self, request_iterator, context):
        session = await self._host.open_session()
        receiver = asyncio.create_task(
            _receive_requests(request_iterator, session),
            name=f"grpc-receive-{session.session_id}",
        )
        try:
            async for event in session.events():
                yield event_to_proto(event)
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
            await self._host.close_session(session.session_id)


async def _receive_requests(request_iterator, session) -> None:
    try:
        async for request in request_iterator:
            try:
                command = command_from_proto(request)
            except WireCommandError as exc:
                await session.report_error(
                    exc.code, str(exc), request_id=exc.request_id
                )
                continue
            await session.handle(command)
            if command.type == "close":
                return
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await session.report_error("TRANSPORT_ERROR", str(exc))
    finally:
        await session.close()


def command_from_proto(message):
    value = {"type": message.type}
    if message.request_id:
        value["request_id"] = message.request_id
    if message.text:
        value["text"] = message.text
    if message.priority:
        value["priority"] = message.priority
    if message.interrupt:
        value["interrupt"] = True
    if message.voice:
        value["voice"] = message.voice
    if message.speed:
        value["speed"] = message.speed
    if message.instruction:
        value["instruction"] = message.instruction
    if message.reference_audio:
        value["reference_audio"] = base64.b64encode(
            message.reference_audio
        ).decode("ascii")
    if message.reference_text:
        value["reference_text"] = message.reference_text
    return command_from_mapping(value)


def event_to_proto(event):
    payload = event.to_dict()
    values = {
        "type": event.type,
        "json": json.dumps(payload, separators=(",", ":"), sort_keys=True),
    }
    string_fields = (
        "session_id",
        "request_id",
        "encoding",
        "priority",
        "code",
        "message",
    )
    integer_fields = (
        "sequence",
        "pts_samples",
        "sample_count",
        "sample_rate",
        "queue_depth",
        "total_samples",
    )
    for field in string_fields:
        value = getattr(event, field)
        if value is not None:
            values[field] = value
    for field in integer_fields:
        value = getattr(event, field)
        if value is not None:
            values[field] = value
    if event.audio is not None:
        values["audio"] = event.audio
    if event.final is not None:
        values["final"] = event.final
    if event.interrupted is not None:
        values["interrupted"] = event.interrupted
    return tts_endpoint_pb2.ServerMessage(**values)


async def serve_grpc(
    host, *, bind: str, port: int, log_level: str = "info"
) -> None:
    del log_level  # gRPC uses the process logging configuration.
    server = grpc.aio.server()
    tts_endpoint_pb2_grpc.add_TTSServicer_to_server(TTSServicer(host), server)
    address = f"{bind}:{port}"
    if server.add_insecure_port(address) == 0:
        raise RuntimeError(f"gRPC could not bind to {address}")
    await server.start()
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=1.0)
