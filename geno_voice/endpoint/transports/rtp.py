"""RTP L16 audio with HTTP control, SSE events, and RTCP reports."""

import asyncio
import json
import secrets
import socket
import struct
import time
from array import array
from contextlib import asynccontextmanager

from .wire import WireCommandError, command_from_mapping


RTP_PAYLOAD_TYPE = 96
RTP_SAMPLES_PER_PACKET = 480
RTP_BYTES_PER_PACKET = RTP_SAMPLES_PER_PACKET * 2
RTCP_INTERVAL_SECONDS = 5.0
NTP_UNIX_EPOCH_OFFSET = 2_208_988_800


def packetize_l16(
    pcm_le: bytes, *, sequence: int, timestamp: int, ssrc: int
) -> bytes:
    if len(pcm_le) % 2:
        raise ValueError("L16 packet payload requires complete PCM16 samples")
    header = struct.pack(
        "!BBHII",
        0x80,
        RTP_PAYLOAD_TYPE,
        sequence & 0xFFFF,
        timestamp & 0xFFFFFFFF,
        ssrc & 0xFFFFFFFF,
    )
    samples = array("h")
    samples.frombytes(pcm_le)
    samples.byteswap()
    return header + samples.tobytes()


def packetize_rtcp_sender_report(
    *,
    ssrc: int,
    rtp_timestamp: int,
    packet_count: int,
    octet_count: int,
    now: float | None = None,
) -> bytes:
    unix_time = time.time() if now is None else now
    whole_seconds = int(unix_time)
    ntp_seconds = (whole_seconds + NTP_UNIX_EPOCH_OFFSET) & 0xFFFFFFFF
    ntp_fraction = int((unix_time - whole_seconds) * (1 << 32)) & 0xFFFFFFFF
    return struct.pack(
        "!BBHIIIIII",
        0x80,
        200,
        6,
        ssrc & 0xFFFFFFFF,
        ntp_seconds,
        ntp_fraction,
        rtp_timestamp & 0xFFFFFFFF,
        packet_count & 0xFFFFFFFF,
        octet_count & 0xFFFFFFFF,
    )


def encode_sse(event: dict) -> str:
    return f"data: {json.dumps(event, separators=(',', ':'), sort_keys=True)}\n\n"


class RtpOutputSession:
    def __init__(
        self,
        host,
        session,
        *,
        destination_host: str,
        destination_port: int,
        rtcp_port: int,
    ) -> None:
        self.host = host
        self.session = session
        self.session_id = session.session_id
        self.destination = (destination_host, destination_port)
        self.rtcp_destination = (destination_host, rtcp_port)
        self.ssrc = secrets.randbits(32)
        self.sequence = secrets.randbits(16)
        self.timestamp = secrets.randbits(32)
        self.packet_count = 0
        self.octet_count = 0
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cancelled_requests: set[str] = set()
        self._history: list[dict] = []
        self._subscribers: set[asyncio.Queue] = set()
        self._pump_task = None
        self._rtcp_task = None
        self._next_send_time: float | None = None
        self._closed = False
        self._close_lock = asyncio.Lock()

    def start(self) -> None:
        self._pump_task = asyncio.create_task(
            self._pump_events(), name=f"rtp-events-{self.session_id}"
        )
        self._rtcp_task = asyncio.create_task(
            self._rtcp_loop(), name=f"rtcp-reports-{self.session_id}"
        )

    async def handle(self, command) -> None:
        if command.type == "cancel" and command.request_id:
            self._cancelled_requests.add(command.request_id)
        elif command.type in {"speak", "supersede"} and command.request_id:
            self._cancelled_requests.discard(command.request_id)
        await self.session.handle(command)

    async def events(self):
        queue = asyncio.Queue()
        for event in self._history:
            yield event
        if self._closed:
            return
        self._subscribers.add(queue)
        try:
            while True:
                event = await queue.get()
                yield event
                if event.get("type") == "closed":
                    return
        finally:
            self._subscribers.discard(queue)

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            await self.host.close_session(self.session_id)
            tasks = [task for task in (self._pump_task, self._rtcp_task) if task]
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._socket.close()
            if not self._history or self._history[-1].get("type") != "closed":
                self._publish({"type": "closed", "session_id": self.session_id})

    async def _pump_events(self) -> None:
        try:
            async for event in self.session.events():
                self._publish(event.to_dict())
                if event.type == "audio" and event.audio:
                    await self._send_audio(event.request_id or "", event.audio)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._publish(
                {
                    "type": "error",
                    "session_id": self.session_id,
                    "code": "RTP_SEND_ERROR",
                    "message": str(exc),
                }
            )

    async def _send_audio(self, request_id: str, pcm: bytes) -> None:
        loop = asyncio.get_running_loop()
        if self._next_send_time is None:
            self._next_send_time = loop.time()
        for offset in range(0, len(pcm), RTP_BYTES_PER_PACKET):
            if self._closed or request_id in self._cancelled_requests:
                return
            payload = pcm[offset : offset + RTP_BYTES_PER_PACKET]
            delay = self._next_send_time - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            if self._closed or request_id in self._cancelled_requests:
                return
            packet = packetize_l16(
                payload,
                sequence=self.sequence,
                timestamp=self.timestamp,
                ssrc=self.ssrc,
            )
            self._socket.sendto(packet, self.destination)
            sample_count = len(payload) // 2
            self.sequence = (self.sequence + 1) & 0xFFFF
            self.timestamp = (self.timestamp + sample_count) & 0xFFFFFFFF
            self.packet_count = (self.packet_count + 1) & 0xFFFFFFFF
            self.octet_count = (self.octet_count + len(payload)) & 0xFFFFFFFF
            self._next_send_time += sample_count / 24_000
            if self.packet_count == 1:
                self._send_rtcp_report()

    async def _rtcp_loop(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(RTCP_INTERVAL_SECONDS)
                if self.packet_count:
                    self._send_rtcp_report()
        except asyncio.CancelledError:
            raise

    def _send_rtcp_report(self) -> None:
        report = packetize_rtcp_sender_report(
            ssrc=self.ssrc,
            rtp_timestamp=self.timestamp,
            packet_count=self.packet_count,
            octet_count=self.octet_count,
        )
        self._socket.sendto(report, self.rtcp_destination)

    def _publish(self, event: dict) -> None:
        self._history.append(event)
        for queue in tuple(self._subscribers):
            queue.put_nowait(event)


class RtpSessionManager:
    def __init__(self, host) -> None:
        self.host = host
        self.sessions: dict[str, RtpOutputSession] = {}

    async def create(
        self,
        *,
        destination_host: str,
        destination_port: int,
        rtcp_port: int,
    ) -> RtpOutputSession:
        resolved_host = socket.gethostbyname(destination_host)
        session = await self.host.open_session()
        output = RtpOutputSession(
            self.host,
            session,
            destination_host=resolved_host,
            destination_port=destination_port,
            rtcp_port=rtcp_port,
        )
        self.sessions[output.session_id] = output
        output.start()
        return output

    def get(self, session_id: str) -> RtpOutputSession | None:
        return self.sessions.get(session_id)

    async def close(self, session_id: str) -> None:
        output = self.sessions.pop(session_id, None)
        if output is not None:
            await output.close()

    async def close_all(self) -> None:
        outputs = tuple(self.sessions.values())
        self.sessions.clear()
        if outputs:
            await asyncio.gather(*(output.close() for output in outputs))


def create_rtp_app(host):
    try:
        from fastapi import FastAPI, HTTPException, Request, Response
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - protected by host preflight
        raise RuntimeError(
            "RTP control requires fastapi; install pip install 'geno-voice[endpoint]'"
        ) from exc

    manager = RtpSessionManager(host)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await manager.close_all()

    app = FastAPI(title="geno-voice TTS RTP endpoint", lifespan=lifespan)
    app.state.rtp_sessions = manager

    @app.post("/v1/rtp/sessions")
    async def create_session(request: Request):
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        destination_host = body.get("destination_host") if isinstance(body, dict) else None
        destination_port = body.get("destination_port") if isinstance(body, dict) else None
        rtcp_port = body.get("rtcp_port") if isinstance(body, dict) else None
        if (
            not isinstance(destination_host, str)
            or not destination_host
            or any(character.isspace() for character in destination_host)
            or isinstance(destination_port, bool)
            or not isinstance(destination_port, int)
            or not 1 <= destination_port <= 65_535
        ):
            raise HTTPException(
                status_code=400,
                detail="destination_host and destination_port (1..65535) are required",
            )
        if rtcp_port is None:
            if destination_port == 65_535:
                raise HTTPException(
                    status_code=400,
                    detail="rtcp_port is required when destination_port is 65535",
                )
            rtcp_port = destination_port + 1
        if (
            isinstance(rtcp_port, bool)
            or not isinstance(rtcp_port, int)
            or not 1 <= rtcp_port <= 65_535
        ):
            raise HTTPException(status_code=400, detail="rtcp_port must be in 1..65535")
        try:
            output = await manager.create(
                destination_host=destination_host,
                destination_port=destination_port,
                rtcp_port=rtcp_port,
            )
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"invalid destination: {exc}") from exc
        session_id = output.session_id
        sdp = _build_sdp(output.destination[0], destination_port, rtcp_port)
        return JSONResponse(
            {
                "session_id": session_id,
                "sdp": sdp,
                "commands_url": f"/v1/rtp/sessions/{session_id}/commands",
                "events_url": f"/v1/rtp/sessions/{session_id}/events",
            },
            status_code=201,
        )

    @app.post("/v1/rtp/sessions/{session_id}/commands")
    async def command(session_id: str, request: Request):
        output = manager.get(session_id)
        if output is None:
            raise HTTPException(status_code=404, detail="RTP session not found")
        try:
            body = await request.json()
            parsed = command_from_mapping(body)
        except WireCommandError as exc:
            return JSONResponse(
                {
                    "type": "error",
                    "code": exc.code,
                    "message": str(exc),
                    "request_id": exc.request_id,
                },
                status_code=400,
            )
        except Exception:
            return JSONResponse(
                {
                    "type": "error",
                    "code": "INVALID_JSON",
                    "message": "request body is not valid JSON",
                },
                status_code=400,
            )
        if parsed.type == "close":
            await manager.close(session_id)
        else:
            await output.handle(parsed)
        return JSONResponse({"status": "accepted"}, status_code=202)

    @app.get("/v1/rtp/sessions/{session_id}/events")
    async def events(session_id: str, request: Request):
        output = manager.get(session_id)
        if output is None:
            raise HTTPException(status_code=404, detail="RTP session not found")

        async def stream():
            async for event in output.events():
                if await request.is_disconnected():
                    return
                yield encode_sse(event)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete("/v1/rtp/sessions/{session_id}")
    async def delete_session(session_id: str):
        await manager.close(session_id)
        return Response(status_code=204)

    return app


def _build_sdp(destination_host: str, rtp_port: int, rtcp_port: int) -> str:
    return "\r\n".join(
        (
            "v=0",
            "o=geno-voice 0 0 IN IP4 127.0.0.1",
            "s=geno-voice streaming TTS",
            f"c=IN IP4 {destination_host}",
            "t=0 0",
            f"m=audio {rtp_port} RTP/AVP {RTP_PAYLOAD_TYPE}",
            f"a=rtpmap:{RTP_PAYLOAD_TYPE} L16/24000/1",
            f"a=rtcp:{rtcp_port}",
            "",
        )
    )


async def serve_rtp(
    host, *, bind: str, port: int, log_level: str = "info"
) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - protected by host preflight
        raise RuntimeError(
            "RTP endpoint requires uvicorn; install pip install 'geno-voice[endpoint]'"
        ) from exc
    server = uvicorn.Server(
        uvicorn.Config(
            create_rtp_app(host),
            host=bind,
            port=port,
            log_level=log_level,
        )
    )
    await server.serve()
