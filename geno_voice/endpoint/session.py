"""Interruptible synthesis sessions shared by every endpoint transport."""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from .types import (
    CancellationToken,
    EndpointCommand,
    EndpointEvent,
    SynthesisRequest,
    TTSModelAdapter,
)


@dataclass(frozen=True)
class _Job:
    request: SynthesisRequest


class SynthesisSession:
    """Own command state, scheduling, cancellation, and ordered TTS events."""

    MAX_TEXT_BYTES = 64 * 1024
    MAX_PENDING_JOBS = 32

    def __init__(
        self,
        model: TTSModelAdapter,
        *,
        session_id: str,
        default_voice: str | None = None,
    ) -> None:
        self._model = model
        self.session_id = session_id
        self._default_voice = default_voice
        self._buffers: dict[str, str] = {}
        self._normal: deque[_Job] = deque()
        self._priority: deque[_Job] = deque()
        self._events: asyncio.Queue[EndpointEvent] = asyncio.Queue()
        self._jobs_available = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._active: _Job | None = None
        self._active_cancellation: CancellationToken | None = None
        self._cancel_events_emitted: set[str] = set()
        self._sequence = 0
        self._pts_samples = 0
        self._started = False
        self._closed = False
        self._close_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._started:
            return
        if self._closed:
            raise RuntimeError("cannot start a closed synthesis session")
        self._started = True
        await self._emit(
            EndpointEvent(
                type="ready",
                session_id=self.session_id,
                model=self._model.name,
                capabilities=self._model.capabilities,
                sample_rate=24_000,
                encoding="pcm_s16le",
            )
        )
        self._worker = asyncio.create_task(
            self._run_worker(), name=f"tts-session-{self.session_id}"
        )

    async def handle(self, command: EndpointCommand) -> None:
        if command.type == "close":
            await self.close()
            return
        if not self._started:
            raise RuntimeError("synthesis session has not been started")
        if self._closed:
            return

        handlers: dict[str, Any] = {
            "append": self._handle_append,
            "commit": self._handle_commit,
            "speak": self._handle_speak,
            "cancel": self._handle_cancel,
            "supersede": self._handle_supersede,
        }
        handler = handlers.get(command.type)
        if handler is None:
            await self._error("INVALID_COMMAND", f"unknown command: {command.type}")
            return
        await handler(command)

    async def events(self):
        """Yield events in the exact order produced by the session."""

        while True:
            event = await self._events.get()
            yield event
            if event.type == "closed":
                return

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            if self._active_cancellation is not None:
                self._active_cancellation.cancel()
            self._buffers.clear()
            self._normal.clear()
            self._priority.clear()
            self._jobs_available.set()

            worker = self._worker
            if worker is not None:
                await asyncio.sleep(0)
                if not worker.done():
                    worker.cancel()
                with suppress(asyncio.CancelledError):
                    await worker
            await self._emit(EndpointEvent(type="closed", session_id=self.session_id))

    async def _handle_append(self, command: EndpointCommand) -> None:
        request_id = self._required_request_id(command)
        if request_id is None:
            return
        text = command.text or ""
        if not text:
            await self._error("INVALID_TEXT", "append text must not be empty", request_id)
            return
        combined = self._buffers.get(request_id, "") + text
        if len(combined.encode("utf-8")) > self.MAX_TEXT_BYTES:
            await self._error(
                "TEXT_TOO_LARGE",
                f"request text exceeds {self.MAX_TEXT_BYTES} UTF-8 bytes",
                request_id,
            )
            return
        self._buffers[request_id] = combined

    async def _handle_commit(self, command: EndpointCommand) -> None:
        request_id = self._required_request_id(command)
        if request_id is None:
            return
        text = self._buffers.get(request_id, "")
        if not text:
            await self._error("EMPTY_COMMIT", "no buffered text to commit", request_id)
            return
        if await self._queue_request(
            SynthesisRequest(
                request_id=request_id,
                text=text,
                voice=command.voice or self._default_voice,
            )
        ):
            self._buffers.pop(request_id, None)

    async def _handle_speak(self, command: EndpointCommand) -> None:
        request = await self._request_from_command(command)
        if request is None:
            return
        if command.interrupt and self._active is not None:
            await self._cancel_active_if(
                lambda job: job.request.priority == "normal"
            )
        await self._queue_request(request)

    async def _handle_cancel(self, command: EndpointCommand) -> None:
        request_id = self._required_request_id(command)
        if request_id is None:
            return

        interrupted = False
        if (
            self._active is not None
            and self._active.request.request_id == request_id
            and self._active_cancellation is not None
        ):
            self._active_cancellation.cancel()
            interrupted = True

        removed = self._remove_queued(lambda job: job.request.request_id == request_id)
        self._buffers.pop(request_id, None)
        await self._emit_cancelled(request_id, interrupted=interrupted)
        if removed:
            self._jobs_available.set()

    async def _handle_supersede(self, command: EndpointCommand) -> None:
        request = await self._request_from_command(command, priority="normal")
        if request is None:
            return

        self._buffers.clear()
        await self._cancel_active_if(lambda job: job.request.priority == "normal")
        removed = self._remove_queued(lambda job: job.request.priority == "normal")
        for job in removed:
            await self._emit_cancelled(job.request.request_id, interrupted=False)
        await self._queue_request(request)

    async def _request_from_command(
        self, command: EndpointCommand, *, priority: str | None = None
    ) -> SynthesisRequest | None:
        request_id = self._required_request_id(command)
        if request_id is None:
            return None
        text = command.text or ""
        if not text:
            await self._error("INVALID_TEXT", "speech text must not be empty", request_id)
            return None
        if len(text.encode("utf-8")) > self.MAX_TEXT_BYTES:
            await self._error(
                "TEXT_TOO_LARGE",
                f"request text exceeds {self.MAX_TEXT_BYTES} UTF-8 bytes",
                request_id,
            )
            return None
        chosen_priority = priority or command.priority
        if chosen_priority not in {"normal", "backchannel"}:
            await self._error(
                "INVALID_PRIORITY",
                "priority must be normal or backchannel",
                request_id,
            )
            return None
        return SynthesisRequest(
            request_id=request_id,
            text=text,
            priority=chosen_priority,
            voice=command.voice or self._default_voice,
            speed=command.speed,
            instruction=command.instruction,
            reference_audio=command.reference_audio,
            reference_text=command.reference_text,
        )

    async def _queue_request(self, request: SynthesisRequest) -> bool:
        if len(self._normal) + len(self._priority) >= self.MAX_PENDING_JOBS:
            await self._error(
                "QUEUE_FULL",
                f"session already has {self.MAX_PENDING_JOBS} pending jobs",
                request.request_id,
            )
            return False
        job = _Job(request=request)
        queue = self._priority if request.priority == "backchannel" else self._normal
        queue.append(job)
        self._cancel_events_emitted.discard(request.request_id)
        await self._emit(
            EndpointEvent(
                type="accepted",
                session_id=self.session_id,
                request_id=request.request_id,
                priority=request.priority,
                queue_depth=len(self._normal) + len(self._priority),
            )
        )
        self._jobs_available.set()
        return True

    async def _run_worker(self) -> None:
        while not self._closed:
            job = self._take_next_job()
            if job is None:
                self._jobs_available.clear()
                if self._normal or self._priority:
                    self._jobs_available.set()
                    continue
                await self._jobs_available.wait()
                continue
            await self._synthesize(job)

    def _take_next_job(self) -> _Job | None:
        if self._priority:
            return self._priority.popleft()
        if self._normal:
            return self._normal.popleft()
        return None

    async def _synthesize(self, job: _Job) -> None:
        request = job.request
        cancellation = CancellationToken()
        self._active = job
        self._active_cancellation = cancellation
        request_samples = 0
        await self._emit(
            EndpointEvent(
                type="started",
                session_id=self.session_id,
                request_id=request.request_id,
                priority=request.priority,
            )
        )
        try:
            async for chunk in self._model.synthesize(request, cancellation):
                if cancellation.cancelled or self._closed:
                    break
                if chunk.sample_rate != 24_000:
                    raise ValueError(
                        f"model emitted {chunk.sample_rate} Hz audio; expected 24000 Hz"
                    )
                if len(chunk.pcm) % 2:
                    raise ValueError("model emitted an odd number of PCM16 bytes")
                sample_count = len(chunk.pcm) // 2
                await self._emit(
                    EndpointEvent(
                        type="audio",
                        session_id=self.session_id,
                        request_id=request.request_id,
                        audio=chunk.pcm,
                        sequence=self._sequence,
                        pts_samples=self._pts_samples,
                        sample_count=sample_count,
                        sample_rate=24_000,
                        encoding="pcm_s16le",
                        final=chunk.final,
                        alignment=chunk.alignment,
                    )
                )
                self._sequence += 1
                self._pts_samples += sample_count
                request_samples += sample_count
                if chunk.alignment:
                    await self._emit(
                        EndpointEvent(
                            type="alignment",
                            session_id=self.session_id,
                            request_id=request.request_id,
                            alignment=chunk.alignment,
                        )
                    )
            if not cancellation.cancelled and not self._closed:
                await self._emit(
                    EndpointEvent(
                        type="completed",
                        session_id=self.session_id,
                        request_id=request.request_id,
                        total_samples=request_samples,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed:
                await self._error("MODEL_ERROR", str(exc), request.request_id)
        finally:
            self._active = None
            self._active_cancellation = None

    async def _cancel_active_if(self, predicate) -> None:
        if (
            self._active is None
            or self._active_cancellation is None
            or not predicate(self._active)
        ):
            return
        request_id = self._active.request.request_id
        self._active_cancellation.cancel()
        await self._emit_cancelled(request_id, interrupted=True)

    def _remove_queued(self, predicate) -> list[_Job]:
        removed: list[_Job] = []
        for queue in (self._priority, self._normal):
            retained: deque[_Job] = deque()
            while queue:
                job = queue.popleft()
                if predicate(job):
                    removed.append(job)
                else:
                    retained.append(job)
            queue.extend(retained)
        return removed

    async def _emit_cancelled(self, request_id: str, *, interrupted: bool) -> None:
        if request_id in self._cancel_events_emitted:
            return
        self._cancel_events_emitted.add(request_id)
        await self._emit(
            EndpointEvent(
                type="cancelled",
                session_id=self.session_id,
                request_id=request_id,
                interrupted=interrupted,
            )
        )

    def _required_request_id(self, command: EndpointCommand) -> str | None:
        request_id = (command.request_id or "").strip()
        if request_id:
            return request_id
        self._events.put_nowait(
            EndpointEvent(
                type="error",
                session_id=self.session_id,
                code="INVALID_REQUEST_ID",
                message="request_id must not be empty",
            )
        )
        return None

    async def _error(
        self, code: str, message: str, request_id: str | None = None
    ) -> None:
        await self._emit(
            EndpointEvent(
                type="error",
                session_id=self.session_id,
                request_id=request_id,
                code=code,
                message=message,
            )
        )

    async def _emit(self, event: EndpointEvent) -> None:
        await self._events.put(event)
