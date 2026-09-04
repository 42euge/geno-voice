"""Built-in endpoint model adapters and shared streaming helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

from ..types import CancellationToken

T = TypeVar("T")


@dataclass(frozen=True)
class _ThreadFailure:
    error: BaseException


_END = object()


async def stream_sync_iterator(
    iterator_factory: Callable[[], Iterator[T]], cancellation: CancellationToken
):
    """Bridge a blocking model iterator without blocking endpoint control traffic."""

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[T | _ThreadFailure | object] = asyncio.Queue()

    def produce() -> None:
        iterator: Iterator[T] | None = None
        try:
            iterator = iterator_factory()
            for item in iterator:
                if cancellation.cancelled:
                    break
                asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()
        except BaseException as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put(_ThreadFailure(exc)), loop
            ).result()
        finally:
            if iterator is not None:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            asyncio.run_coroutine_threadsafe(queue.put(_END), loop).result()

    producer = asyncio.create_task(asyncio.to_thread(produce))
    try:
        while True:
            item = await queue.get()
            if item is _END:
                break
            if isinstance(item, _ThreadFailure):
                raise item.error
            yield item
    finally:
        cancellation.cancel()
        await producer
