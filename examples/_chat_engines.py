"""STT + TTS engine loading — extracted from mic_chat.py:run_chat.

iter-108: Original code lived inline in `run_chat`, hardcoded the
`WhisperEngine` + `get_tts_engine("kokoro")` imports, and mixed
loading with timing and stdout logging. Untestable on x86_64
because `WhisperEngine` imports `mlx-whisper` at module level
(Mac-only).

This module pulls the loading sequence behind a factory-pair
shape:

    load_engines(stt_factory, tts_factory) -> LoadedEngines

The caller in `mic_chat.py` passes lambdas that re-create the
existing behavior:

    load_engines(
        stt_factory=lambda: WhisperEngine(model_repo=model_repo),
        tts_factory=lambda: get_tts_engine("kokoro"),
    )

Tests pass any callable returning an object with `_load()`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class LoadedEngines:
    """Bundle returned by `load_engines`. Each engine field is the
    instance returned by its factory; the corresponding `*_load_seconds`
    field is the wall-clock the factory + `_load()` took.
    """

    stt: Any
    tts: Any
    stt_load_seconds: float
    tts_load_seconds: float


def load_engines(
    stt_factory: Callable[[], Any],
    tts_factory: Callable[[], Any],
    *,
    log: Callable[[str], None] = print,
) -> LoadedEngines:
    """Construct + warm both engines, time each, log per-engine load
    duration in ms.

    Each factory must return an instance exposing a callable
    ``_load()`` (the contract WhisperEngine + the TTS engine
    families satisfy today). The caller is responsible for
    wiring model_repo / voice / etc. into the factory closure.

    The wall-clock includes both factory construction AND the
    `_load()` call. That's intentional — operators reading the
    "STT loaded in 250ms" line care about end-to-end startup
    cost, not about which sub-step dominates. If a finer
    breakdown is ever needed, split into two factories or
    return both timings.

    Logs two single-line summaries via `log` (default `print`):

        STT loaded in 250ms
        TTS loaded in 80ms

    Tests can capture into a list with `log=lines.append`.
    """
    t0 = time.monotonic()
    stt = stt_factory()
    stt._load()
    stt_load = time.monotonic() - t0

    t1 = time.monotonic()
    tts = tts_factory()
    tts._load()
    tts_load = time.monotonic() - t1

    log(f"STT loaded in {stt_load * 1000:.0f}ms")
    log(f"TTS loaded in {tts_load * 1000:.0f}ms")

    return LoadedEngines(
        stt=stt,
        tts=tts,
        stt_load_seconds=stt_load,
        tts_load_seconds=tts_load,
    )
