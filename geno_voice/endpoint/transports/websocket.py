"""Full-duplex WebSocket adapter for synthesis sessions."""

import asyncio
import json
from dataclasses import asdict

from .wire import WireCommandError, command_from_mapping, encode_audio_envelope


def create_websocket_app(host):
    try:
        from fastapi import FastAPI, WebSocket
    except ImportError as exc:  # pragma: no cover - protected by host preflight
        raise RuntimeError(
            "WebSocket endpoint requires fastapi; install "
            "pip install 'geno-voice[endpoint]'"
        ) from exc

    app = FastAPI(title="geno-voice TTS WebSocket endpoint")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": host.model.name,
            "sessions": host.session_count,
        }

    @app.get("/v1/capabilities")
    async def capabilities():
        return {
            "model": host.model.name,
            "capabilities": asdict(host.model.capabilities),
            "audio": {
                "encoding": "pcm_s16le",
                "sample_rate": 24_000,
                "channels": 1,
            },
        }

    @app.websocket("/v1/tts/stream")
    async def stream(socket: WebSocket):
        await socket.accept()
        session = await host.open_session()
        receiver = asyncio.create_task(_receive_commands(socket, session))
        sender = asyncio.create_task(_send_events(socket, session))
        try:
            done, pending = await asyncio.wait(
                {receiver, sender}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*done, *pending, return_exceptions=True)
        finally:
            await host.close_session(session.session_id)

    return app


async def _receive_commands(socket, session) -> None:
    from starlette.websockets import WebSocketDisconnect

    while True:
        try:
            message = await socket.receive()
        except WebSocketDisconnect:
            return
        if message.get("type") == "websocket.disconnect":
            return
        text = message.get("text")
        if text is None:
            await session.report_error(
                "INVALID_COMMAND", "commands must use JSON text frames"
            )
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            await session.report_error("INVALID_JSON", "frame is not valid JSON")
            continue
        try:
            command = command_from_mapping(value)
        except WireCommandError as exc:
            await session.report_error(
                exc.code, str(exc), request_id=exc.request_id
            )
            continue
        await session.handle(command)
        if command.type == "close":
            return


async def _send_events(socket, session) -> None:
    from starlette.websockets import WebSocketDisconnect

    try:
        async for event in session.events():
            if event.type == "audio":
                await socket.send_bytes(encode_audio_envelope(event))
            else:
                await socket.send_json(event.to_dict())
    except (WebSocketDisconnect, RuntimeError):
        return


async def serve_websocket(
    host, *, bind: str, port: int, log_level: str = "info"
) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - protected by host preflight
        raise RuntimeError(
            "WebSocket endpoint requires uvicorn; install "
            "pip install 'geno-voice[endpoint]'"
        ) from exc

    server = uvicorn.Server(
        uvicorn.Config(
            create_websocket_app(host),
            host=bind,
            port=port,
            log_level=log_level,
        )
    )
    await server.serve()
