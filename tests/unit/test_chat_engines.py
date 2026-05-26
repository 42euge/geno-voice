"""Tests for iter-108 — load_engines helper.

The function takes two factory callables, each returning an
object with `_load()`. It must:
  - Call each factory exactly once
  - Call _load() exactly once on each result
  - Time both load sequences (factory + _load combined)
  - Log one line per engine via the `log` callable
  - Return a LoadedEngines bundle with both instances + timings
  - Propagate factory errors (no swallowing) — load failures
    are unrecoverable for the chat CLI
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_engines import LoadedEngines, load_engines  # noqa: E402


# ---- Fakes ----------------------------------------------------------------


class _FakeEngine:
    """Stand-in for WhisperEngine / TTS engine. Tracks _load() calls."""

    def __init__(self, name: str = "fake", load_delay: float = 0.0):
        self.name = name
        self.load_delay = load_delay
        self.load_count = 0

    def _load(self):
        self.load_count += 1
        if self.load_delay > 0:
            time.sleep(self.load_delay)


def _capture():
    lines: list[str] = []

    def log(line: str) -> None:
        lines.append(line)

    return log, lines


# ---- Happy path -----------------------------------------------------------


def test_returns_loaded_engines_bundle():
    """Both factories called, both _load() invoked, both
    instances bundled into the dataclass."""
    stt = _FakeEngine("stt")
    tts = _FakeEngine("tts")

    log, _ = _capture()
    result = load_engines(
        stt_factory=lambda: stt,
        tts_factory=lambda: tts,
        log=log,
    )

    assert isinstance(result, LoadedEngines)
    assert result.stt is stt
    assert result.tts is tts
    assert stt.load_count == 1
    assert tts.load_count == 1


def test_factories_called_exactly_once():
    """Factories shouldn't be called more than once — this is a
    one-shot startup operation."""
    stt_calls = 0
    tts_calls = 0

    def stt_factory():
        nonlocal stt_calls
        stt_calls += 1
        return _FakeEngine("stt")

    def tts_factory():
        nonlocal tts_calls
        tts_calls += 1
        return _FakeEngine("tts")

    log, _ = _capture()
    load_engines(stt_factory, tts_factory, log=log)
    assert stt_calls == 1
    assert tts_calls == 1


def test_load_seconds_are_positive():
    """Both timings must be ≥ 0. Even instant fakes register
    something measurable in monotonic time, but be permissive
    on the lower bound (a noop _load might be sub-microsecond
    on fast hardware)."""
    log, _ = _capture()
    result = load_engines(
        stt_factory=lambda: _FakeEngine("stt"),
        tts_factory=lambda: _FakeEngine("tts"),
        log=log,
    )
    assert result.stt_load_seconds >= 0.0
    assert result.tts_load_seconds >= 0.0


def test_load_seconds_reflect_simulated_delay():
    """A factory whose _load() sleeps 50ms must report ≥ 50ms."""
    log, _ = _capture()
    result = load_engines(
        stt_factory=lambda: _FakeEngine("stt", load_delay=0.05),
        tts_factory=lambda: _FakeEngine("tts"),
        log=log,
    )
    # Permissive: sleep(0.05) sometimes lands at 0.049 on Linux.
    assert result.stt_load_seconds >= 0.04


# ---- Logging ---------------------------------------------------------------


def test_log_emits_two_lines_in_order():
    """One log line per engine, STT first then TTS."""
    log, lines = _capture()
    load_engines(
        stt_factory=lambda: _FakeEngine("stt"),
        tts_factory=lambda: _FakeEngine("tts"),
        log=log,
    )
    assert len(lines) == 2
    assert "STT loaded in" in lines[0]
    assert "TTS loaded in" in lines[1]
    assert "ms" in lines[0]
    assert "ms" in lines[1]


def test_log_lines_are_plain_no_ansi():
    """Module emits plain text. ANSI styling is the caller's
    responsibility — same convention as iter-107 prerender_fillers."""
    log, lines = _capture()
    load_engines(
        stt_factory=lambda: _FakeEngine("stt"),
        tts_factory=lambda: _FakeEngine("tts"),
        log=log,
    )
    for ln in lines:
        assert "\x1b[" not in ln, f"ANSI code leaked into {ln!r}"
        # Also no leading whitespace / indentation.
        assert not ln.startswith(" "), f"leading space in {ln!r}"


def test_default_log_is_print(capsys):
    """When `log` is not passed, output flows through `print`."""
    load_engines(
        stt_factory=lambda: _FakeEngine("stt"),
        tts_factory=lambda: _FakeEngine("tts"),
    )
    captured = capsys.readouterr()
    assert "STT loaded in" in captured.out
    assert "TTS loaded in" in captured.out


# ---- Error propagation ----------------------------------------------------


def test_stt_factory_failure_propagates():
    """A factory exception bubbles up. The chat CLI wants the
    process to exit on engine load failure, not continue with a
    half-loaded state."""

    def bad_stt():
        raise RuntimeError("model not found")

    log, _ = _capture()
    with pytest.raises(RuntimeError, match="model not found"):
        load_engines(
            stt_factory=bad_stt,
            tts_factory=lambda: _FakeEngine("tts"),
            log=log,
        )


def test_stt_factory_failure_skips_tts():
    """When STT factory fails, TTS factory is NOT called.
    Defensive: a half-loaded state would leak resources."""
    tts_called = False

    def stt_factory():
        raise RuntimeError("nope")

    def tts_factory():
        nonlocal tts_called
        tts_called = True
        return _FakeEngine("tts")

    log, _ = _capture()
    with pytest.raises(RuntimeError):
        load_engines(stt_factory, tts_factory, log=log)
    assert tts_called is False


def test_tts_load_failure_propagates():
    """TTS engine raising in `_load()` (not factory) also
    propagates."""

    class BadTTS:
        def _load(self):
            raise RuntimeError("tts kaboom")

    log, _ = _capture()
    with pytest.raises(RuntimeError, match="tts kaboom"):
        load_engines(
            stt_factory=lambda: _FakeEngine("stt"),
            tts_factory=lambda: BadTTS(),
            log=log,
        )


# ---- Factory shape parity --------------------------------------------------


def test_factories_receive_no_arguments():
    """The factory signature is `() -> engine`. Real callers
    bake model_repo / voice into a closure; the load_engines
    function MUST NOT pass kwargs of its own."""
    received_args: list[tuple] = []

    def stt_factory(*args, **kwargs):
        received_args.append((args, kwargs))
        return _FakeEngine("stt")

    log, _ = _capture()
    load_engines(
        stt_factory=stt_factory,
        tts_factory=lambda: _FakeEngine("tts"),
        log=log,
    )
    assert received_args == [((), {})]


def test_returns_dataclass_not_dict():
    """Return type is a LoadedEngines dataclass, not a tuple or
    dict. Locks in the API for future callers."""
    log, _ = _capture()
    result = load_engines(
        stt_factory=lambda: _FakeEngine("stt"),
        tts_factory=lambda: _FakeEngine("tts"),
        log=log,
    )
    assert hasattr(result, "stt")
    assert hasattr(result, "tts")
    assert hasattr(result, "stt_load_seconds")
    assert hasattr(result, "tts_load_seconds")
