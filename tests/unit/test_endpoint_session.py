"""Behavioral tests for transport-neutral streaming TTS sessions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from geno_voice.endpoint.session import SynthesisSession
from geno_voice.endpoint.types import (
    AudioChunk,
    CancellationToken,
    EndpointCommand,
    EndpointEvent,
    ModelCapabilities,
    SynthesisRequest,
)


class FakeModel:
    name = "fake"
    capabilities = ModelCapabilities(streaming=True)

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.requests: list[SynthesisRequest] = []

    async def synthesize(
        self, request: SynthesisRequest, cancellation: CancellationToken
    ) -> AsyncIterator[AudioChunk]:
        self.requests.append(request)
        for pcm in self._chunks:
            if cancellation.cancelled:
                return
            yield AudioChunk(pcm=pcm)


class BlockingFakeModel:
    name = "blocking-fake"
    capabilities = ModelCapabilities(streaming=True)

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.requests: list[SynthesisRequest] = []

    async def synthesize(
        self, request: SynthesisRequest, cancellation: CancellationToken
    ) -> AsyncIterator[AudioChunk]:
        self.requests.append(request)
        self.started.set()
        await cancellation.wait()
        self.cancelled.set()
        if False:  # Make this an async generator without producing audio.
            yield AudioChunk(pcm=b"")


class GateModel:
    """Blocks the first request so queue ordering can be observed."""

    name = "gate-fake"
    capabilities = ModelCapabilities(streaming=True)

    def __init__(self) -> None:
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self.requests: list[SynthesisRequest] = []

    async def synthesize(
        self, request: SynthesisRequest, cancellation: CancellationToken
    ) -> AsyncIterator[AudioChunk]:
        self.requests.append(request)
        if len(self.requests) == 1:
            self.first_started.set()
            await self.release_first.wait()
        if not cancellation.cancelled:
            yield AudioChunk(pcm=b"\x01\x00")


async def collect_through(
    session: SynthesisSession, event_type: str, *, timeout: float = 1.0
) -> list[EndpointEvent]:
    events: list[EndpointEvent] = []
    stream = session.events()
    while True:
        event = await asyncio.wait_for(anext(stream), timeout)
        events.append(event)
        if event.type == event_type:
            return events


async def next_type(
    session: SynthesisSession, event_type: str, *, timeout: float = 1.0
) -> EndpointEvent:
    return (await collect_through(session, event_type, timeout=timeout))[-1]


def test_append_commit_streams_ordered_audio() -> None:
    async def scenario() -> None:
        first_pcm = b"\x01\x00\x02\x00\x03\x00"
        second_pcm = b"\x04\x00\x05\x00"
        model = FakeModel([first_pcm, second_pcm])
        session = SynthesisSession(model, session_id="s1")
        await session.start()
        await session.handle(EndpointCommand.append("r1", "Hello "))
        await session.handle(EndpointCommand.append("r1", "world."))
        await session.handle(EndpointCommand.commit("r1"))

        events = await collect_through(session, "completed")

        assert [event.type for event in events] == [
            "ready",
            "accepted",
            "started",
            "audio",
            "audio",
            "completed",
        ]
        assert model.requests == [
            SynthesisRequest(request_id="r1", text="Hello world.")
        ]
        assert [event.sequence for event in events if event.type == "audio"] == [0, 1]
        assert [event.pts_samples for event in events if event.type == "audio"] == [0, 3]
        assert [event.sample_count for event in events if event.type == "audio"] == [3, 2]
        assert [event.audio for event in events if event.type == "audio"] == [
            first_pcm,
            second_pcm,
        ]
        assert events[-1].total_samples == 5
        await session.close()

    asyncio.run(scenario())


def test_cancel_interrupts_only_the_matching_active_request() -> None:
    async def scenario() -> None:
        model = BlockingFakeModel()
        session = SynthesisSession(model, session_id="s1")
        await session.start()
        await session.handle(EndpointCommand.speak("r1", "Long answer"))
        await model.started.wait()

        await session.handle(EndpointCommand.cancel("someone-else"))
        other = await next_type(session, "cancelled")
        assert other.request_id == "someone-else"
        assert other.interrupted is False
        assert model.cancelled.is_set() is False

        await session.handle(EndpointCommand.cancel("r1"))
        cancelled = await next_type(session, "cancelled")
        assert cancelled.request_id == "r1"
        assert cancelled.interrupted is True
        await asyncio.wait_for(model.cancelled.wait(), 1.0)
        await session.close()

    asyncio.run(scenario())


def test_backchannel_priority_moves_ahead_of_queued_normal_speech() -> None:
    async def scenario() -> None:
        model = GateModel()
        session = SynthesisSession(model, session_id="s1")
        await session.start()
        await session.handle(EndpointCommand.speak("r1", "First"))
        await model.first_started.wait()
        await session.handle(EndpointCommand.speak("r2", "Second"))
        await session.handle(
            EndpointCommand.speak("cue", "Mm-hmm.", priority="backchannel")
        )
        model.release_first.set()

        completed: list[str] = []
        stream = session.events()
        while len(completed) < 3:
            event = await asyncio.wait_for(anext(stream), 1.0)
            if event.type == "completed":
                completed.append(event.request_id or "")

        assert completed == ["r1", "cue", "r2"]
        assert [request.request_id for request in model.requests] == ["r1", "cue", "r2"]
        await session.close()

    asyncio.run(scenario())


def test_supersede_cancels_normal_work_and_queues_replacement() -> None:
    async def scenario() -> None:
        model = GateModel()
        session = SynthesisSession(model, session_id="s1")
        await session.start()
        await session.handle(EndpointCommand.speak("old-active", "Old"))
        await model.first_started.wait()
        await session.handle(EndpointCommand.speak("old-queued", "Also old"))

        await session.handle(EndpointCommand.supersede("new", "Corrected"))
        model.release_first.set()
        events = await collect_through(session, "completed")

        cancelled_ids = {
            event.request_id for event in events if event.type == "cancelled"
        }
        assert cancelled_ids == {"old-active", "old-queued"}
        assert events[-1].request_id == "new"
        assert [request.request_id for request in model.requests] == [
            "old-active",
            "new",
        ]
        await session.close()

    asyncio.run(scenario())


def test_text_limit_counts_utf8_bytes_without_killing_session() -> None:
    async def scenario() -> None:
        model = FakeModel([b"\x00\x00"])
        session = SynthesisSession(model, session_id="s1")
        await session.start()
        await session.handle(EndpointCommand.append("too-big", "é" * 32_768))
        await session.handle(EndpointCommand.append("too-big", "x"))

        error = await next_type(session, "error")
        assert error.code == "TEXT_TOO_LARGE"
        assert error.request_id == "too-big"

        await session.handle(EndpointCommand.speak("valid", "Still usable"))
        completed = await next_type(session, "completed")
        assert completed.request_id == "valid"
        await session.close()

    asyncio.run(scenario())


def test_queue_limit_rejects_only_the_excess_job() -> None:
    async def scenario() -> None:
        model = GateModel()
        session = SynthesisSession(model, session_id="s1")
        await session.start()
        await session.handle(EndpointCommand.speak("active", "Active"))
        await model.first_started.wait()
        for index in range(32):
            await session.handle(EndpointCommand.speak(f"q{index}", "Queued"))
        await session.handle(EndpointCommand.speak("excess", "No room"))

        error = await next_type(session, "error")
        assert error.code == "QUEUE_FULL"
        assert error.request_id == "excess"

        await session.handle(EndpointCommand.cancel("q0"))
        await session.handle(EndpointCommand.speak("replacement", "Now fits"))
        accepted = await next_type(session, "accepted")
        assert accepted.request_id == "replacement"
        model.release_first.set()
        await session.close()

    asyncio.run(scenario())


def test_close_cancels_work_and_emits_one_terminal_event() -> None:
    async def scenario() -> None:
        model = BlockingFakeModel()
        session = SynthesisSession(model, session_id="s1")
        await session.start()
        await session.handle(EndpointCommand.speak("r1", "Long answer"))
        await model.started.wait()

        await session.close()
        await session.close()

        events = await collect_through(session, "closed")
        assert [event.type for event in events].count("closed") == 1
        await asyncio.wait_for(model.cancelled.wait(), 1.0)

    asyncio.run(scenario())
