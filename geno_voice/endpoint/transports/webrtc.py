"""LAN WebRTC signaling, data-channel control, and 48 kHz audio."""

import asyncio
import json
from collections import deque
from fractions import Fraction

try:
    import numpy as np
    from aiortc import (
        MediaStreamTrack,
        RTCPeerConnection,
        RTCSessionDescription,
    )
    from aiortc.mediastreams import MediaStreamError
    from av import AudioFrame
    from av.audio.fifo import AudioFifo
    from av.audio.resampler import AudioResampler
except ImportError as exc:  # pragma: no cover - protected by host preflight
    raise RuntimeError(
        "WebRTC endpoint requires aiortc, av, and numpy; install "
        "pip install 'geno-voice[endpoint]'"
    ) from exc

from .wire import WireCommandError, command_from_mapping


_FLUSH = object()
_END = object()


class SessionAudioTrack(MediaStreamTrack):
    """Route session events to one control channel and one WebRTC audio track."""

    kind = "audio"
    FRAME_SAMPLES = 960

    def __init__(self, session) -> None:
        super().__init__()
        self._session = session
        self._audio_queue = asyncio.Queue()
        self._control = None
        self._pending_control: deque[str] = deque()
        self._output_frames: deque[AudioFrame] = deque()
        self._resampler = self._new_resampler()
        self._fifo = AudioFifo()
        self._next_pts = 0
        self._pump_task = asyncio.create_task(
            self._pump_events(), name=f"webrtc-events-{session.session_id}"
        )

    def bind_control(self, channel) -> None:
        if channel.label != "geno-voice-control" or not channel.ordered:
            channel.close()
            return
        self._control = channel

        @channel.on("message")
        def on_message(message):
            asyncio.create_task(self._handle_control(message))

        @channel.on("open")
        def on_open():
            self._flush_control()

        self._flush_control()

    async def recv(self) -> AudioFrame:
        while not self._output_frames:
            item = await self._audio_queue.get()
            if item is _END:
                self._flush_resampler()
                self._drain_fifo(partial=True)
                if not self._output_frames:
                    raise MediaStreamError
                break
            if item is _FLUSH:
                self._flush_resampler()
                self._drain_fifo(partial=True)
                self._resampler = self._new_resampler()
                continue
            self._write_audio_event(item)
            self._drain_fifo(partial=False)

        frame = self._output_frames.popleft()
        frame.pts = self._next_pts
        frame.sample_rate = 48_000
        frame.time_base = Fraction(1, 48_000)
        self._next_pts += frame.samples
        return frame

    def stop(self) -> None:
        if self.readyState == "ended":
            return
        if not self._pump_task.done():
            self._pump_task.cancel()
        self._audio_queue.put_nowait(_END)
        super().stop()

    async def _pump_events(self) -> None:
        try:
            async for event in self._session.events():
                if event.type == "audio":
                    await self._audio_queue.put(event)
                else:
                    self._send_control(event.to_dict())
                    if event.type in {"completed", "cancelled"}:
                        await self._audio_queue.put(_FLUSH)
                    if event.type == "closed":
                        await self._audio_queue.put(_END)
        except asyncio.CancelledError:
            raise

    async def _handle_control(self, raw_message) -> None:
        if not isinstance(raw_message, str):
            await self._session.report_error(
                "INVALID_COMMAND", "commands must be JSON text messages"
            )
            return
        try:
            value = json.loads(raw_message)
        except json.JSONDecodeError:
            await self._session.report_error("INVALID_JSON", "message is not valid JSON")
            return
        try:
            command = command_from_mapping(value)
        except WireCommandError as exc:
            await self._session.report_error(
                exc.code, str(exc), request_id=exc.request_id
            )
            return
        await self._session.handle(command)

    def _send_control(self, value) -> None:
        encoded = json.dumps(value, separators=(",", ":"), sort_keys=True)
        if self._control is None or self._control.readyState != "open":
            self._pending_control.append(encoded)
            return
        self._control.send(encoded)

    def _flush_control(self) -> None:
        if self._control is None or self._control.readyState != "open":
            return
        while self._pending_control:
            self._control.send(self._pending_control.popleft())

    @staticmethod
    def _new_resampler() -> AudioResampler:
        return AudioResampler(format="s16", layout="mono", rate=48_000)

    def _write_audio_event(self, event) -> None:
        samples = np.frombuffer(event.audio, dtype="<i2").reshape(1, -1)
        frame = AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = 24_000
        frame.pts = event.pts_samples
        frame.time_base = Fraction(1, 24_000)
        for output in self._resampler.resample(frame):
            self._fifo.write(output)

    def _flush_resampler(self) -> None:
        for output in self._resampler.resample(None):
            self._fifo.write(output)

    def _drain_fifo(self, *, partial: bool) -> None:
        while self._fifo.samples >= self.FRAME_SAMPLES:
            frame = self._fifo.read(self.FRAME_SAMPLES)
            if frame is not None:
                self._output_frames.append(frame)
        if partial and self._fifo.samples:
            frame = self._fifo.read(self._fifo.samples)
            if frame is not None:
                self._output_frames.append(frame)


def create_webrtc_app(host, *, peer_factory=None):
    try:
        from fastapi import FastAPI, HTTPException, Request
    except ImportError as exc:  # pragma: no cover - protected by host preflight
        raise RuntimeError(
            "WebRTC signaling requires fastapi; install "
            "pip install 'geno-voice[endpoint]'"
        ) from exc

    peer_factory = peer_factory or RTCPeerConnection
    app = FastAPI(title="geno-voice TTS WebRTC endpoint")
    app.state.peers = set()

    @app.post("/v1/webrtc/offer")
    async def offer(request: Request):
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid JSON offer") from exc
        if (
            not isinstance(body, dict)
            or body.get("type") != "offer"
            or not isinstance(body.get("sdp"), str)
            or not body["sdp"]
        ):
            raise HTTPException(
                status_code=400,
                detail="offer requires non-empty sdp and type='offer'",
            )

        session = await host.open_session()
        peer = peer_factory()
        track = SessionAudioTrack(session)
        peer.addTrack(track)
        app.state.peers.add(peer)
        cleaned = False

        async def cleanup() -> None:
            nonlocal cleaned
            if cleaned:
                return
            cleaned = True
            app.state.peers.discard(peer)
            await host.close_session(session.session_id)
            track.stop()
            if peer.connectionState != "closed":
                await peer.close()

        @peer.on("datachannel")
        def on_datachannel(channel):
            track.bind_control(channel)

        @peer.on("connectionstatechange")
        async def on_connectionstatechange():
            if peer.connectionState in {"failed", "closed", "disconnected"}:
                await cleanup()

        try:
            await peer.setRemoteDescription(
                RTCSessionDescription(sdp=body["sdp"], type="offer")
            )
            answer = await peer.createAnswer()
            await peer.setLocalDescription(answer)
        except Exception as exc:
            await cleanup()
            raise HTTPException(status_code=400, detail=f"invalid WebRTC offer: {exc}") from exc

        return {
            "sdp": peer.localDescription.sdp,
            "type": peer.localDescription.type,
        }

    return app


async def serve_webrtc(
    host, *, bind: str, port: int, log_level: str = "info"
) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - protected by host preflight
        raise RuntimeError(
            "WebRTC endpoint requires uvicorn; install "
            "pip install 'geno-voice[endpoint]'"
        ) from exc
    server = uvicorn.Server(
        uvicorn.Config(
            create_webrtc_app(host),
            host=bind,
            port=port,
            log_level=log_level,
        )
    )
    await server.serve()
