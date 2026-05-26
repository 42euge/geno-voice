"""Tests for iter-110 — run_session helper.

The function drives a chat loop until KeyboardInterrupt, tracking
session-level state in a SessionState bundle. Tests pass:
  - A stub `chat_loop` whose `.run_one_turn` returns a queued
    sequence of results (success / error / false-trigger).
  - A stub `trim_messages` so the test doesn't import ChatLoop.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_session import SessionState, run_session  # noqa: E402


# ---- Fakes ----------------------------------------------------------------


@dataclass
class _StubResult:
    """Mimics the ChatLoop.run_one_turn return shape."""
    metrics: Optional[Any] = None
    had_error: bool = False
    next_primed_frames: Optional[list] = None


class _StubMetrics:
    """Mimics TurnMetrics. Only `print(turn)` is called by the loop."""

    def __init__(self):
        self.printed_turns: list[int] = []

    def print(self, turn: int) -> None:
        self.printed_turns.append(turn)


class _StubChatLoop:
    """Hand-fed result queue + raise on N-th turn (default
    indefinite). Each `.run_one_turn` pops the next queued
    result; if the queue runs out, raise KeyboardInterrupt to
    end the session."""

    def __init__(self, queue: list[_StubResult]):
        self.queue = list(queue)
        self.calls: list[tuple] = []

    def run_one_turn(self, messages, primed_frames=None):
        self.calls.append((list(messages), primed_frames))
        if not self.queue:
            raise KeyboardInterrupt
        return self.queue.pop(0)


def _stub_trim(messages, max_user_assistant: int):
    """No-op trim — preserves messages exactly. Tests that
    exercise trim use a different stub."""
    return messages


def _silent(*_args, **_kwargs):
    pass


# ---- Empty / immediate exit cases -----------------------------------------


def test_immediate_keyboard_interrupt_returns_state_with_system_prompt():
    """Operator hits Ctrl+C before the first turn completes.
    SessionState should still be populated with the system prompt
    and all counters at zero."""
    loop = _StubChatLoop(queue=[])  # immediate KeyboardInterrupt
    state = run_session(
        loop, "be concise",
        log=_silent, prompt_log=_silent,
        trim_messages=_stub_trim,
    )
    assert isinstance(state, SessionState)
    assert state.messages == [{"role": "system", "content": "be concise"}]
    assert state.all_metrics == []
    assert state.false_triggers == 0
    assert state.llm_errors == 0
    assert state.trim_events == 0
    assert state.trim_messages_evicted == 0


def test_session_start_reflects_clock_callable():
    """The session_start field should come from the injected
    clock at function entry, before any turns run."""
    loop = _StubChatLoop(queue=[])
    state = run_session(
        loop, "p",
        log=_silent, prompt_log=_silent,
        trim_messages=_stub_trim,
        clock=lambda: 12345.6,
    )
    assert state.session_start == 12345.6


# ---- Successful turn flow -------------------------------------------------


def test_one_successful_turn_records_metrics_and_continues():
    """A turn with metrics gets appended to all_metrics + the
    metric.print(turn) is called."""
    m = _StubMetrics()
    loop = _StubChatLoop(queue=[_StubResult(metrics=m)])
    state = run_session(
        loop, "p",
        log=_silent, prompt_log=_silent,
        trim_messages=_stub_trim,
    )
    assert state.all_metrics == [m]
    assert m.printed_turns == [1]


def test_multiple_successful_turns_increment_turn_index():
    """Turn counter passed to .print() increments correctly."""
    metrics = [_StubMetrics() for _ in range(3)]
    loop = _StubChatLoop(queue=[_StubResult(metrics=m) for m in metrics])
    state = run_session(
        loop, "p",
        log=_silent, prompt_log=_silent,
        trim_messages=_stub_trim,
    )
    assert len(state.all_metrics) == 3
    assert metrics[0].printed_turns == [1]
    assert metrics[1].printed_turns == [2]
    assert metrics[2].printed_turns == [3]


# ---- iter-058 LLM-error handling ------------------------------------------


def test_had_error_increments_llm_errors_and_continues():
    """An LLM-error turn doesn't add to all_metrics, but does
    bump llm_errors. The same turn counter is reused next time
    (the operator sees [N] waiting again)."""
    loop = _StubChatLoop(queue=[
        _StubResult(had_error=True),
        _StubResult(metrics=_StubMetrics()),
    ])
    captured_turns: list[int] = []
    state = run_session(
        loop, "p",
        log=_silent,
        prompt_log=lambda t: captured_turns.append(t),
        trim_messages=_stub_trim,
    )
    assert state.llm_errors == 1
    assert len(state.all_metrics) == 1
    # Turn 1: error, prompt fired with [1]
    # Turn 1 again: success, prompt fired with [1]; then turn += 1
    # Turn 2: prompt fires for the queue-empty KI, [2]
    assert captured_turns == [1, 1, 2]


# ---- iter-048 false-trigger handling --------------------------------------


def test_no_metrics_no_error_increments_false_triggers():
    """A turn that produced neither metrics nor error is a VAD
    false trigger. Bumps false_triggers, reuses turn counter."""
    loop = _StubChatLoop(queue=[
        _StubResult(metrics=None, had_error=False),
        _StubResult(metrics=_StubMetrics()),
    ])
    captured_turns: list[int] = []
    state = run_session(
        loop, "p",
        log=_silent,
        prompt_log=lambda t: captured_turns.append(t),
        trim_messages=_stub_trim,
    )
    assert state.false_triggers == 1
    assert len(state.all_metrics) == 1
    # Same shape as the LLM-error test: false-trig keeps the
    # counter, success advances it, then the queue-empty KI
    # gets one more prompt.
    assert captured_turns == [1, 1, 2]


# ---- primed_frames threading ---------------------------------------------


def test_primed_frames_threaded_between_turns():
    """next_primed_frames from turn N becomes primed_frames input
    to turn N+1. iter-025 / iter-057 contract."""
    primed_a = [b"frame_a"]
    primed_b = [b"frame_b"]
    loop = _StubChatLoop(queue=[
        _StubResult(metrics=_StubMetrics(), next_primed_frames=primed_a),
        _StubResult(metrics=_StubMetrics(), next_primed_frames=primed_b),
    ])
    run_session(
        loop, "p",
        log=_silent, prompt_log=_silent,
        trim_messages=_stub_trim,
    )
    # First call had no primed_frames; second saw primed_a; third
    # (KeyboardInterrupt) saw primed_b.
    assert loop.calls[0][1] is None
    assert loop.calls[1][1] is primed_a
    assert loop.calls[2][1] is primed_b


# ---- iter-078 trim handling -----------------------------------------------


def test_trim_event_recorded_when_eviction_happens():
    """When trim_messages returns a shorter list, trim_events
    bumps and trim_messages_evicted reflects the diff."""
    def shrinking_trim(messages, max_user_assistant: int):
        # Drop one message every call, but only when there's
        # something to drop (avoids messages going negative).
        return messages[1:] if len(messages) > 0 else messages

    # Append history each turn so the shrinker has something to
    # evict — production run_one_turn does the same.
    class _AppendingChatLoop(_StubChatLoop):
        def run_one_turn(self, messages, primed_frames=None):
            messages.append({"role": "user", "content": "u"})
            messages.append({"role": "assistant", "content": "a"})
            return super().run_one_turn(messages, primed_frames)

    loop = _AppendingChatLoop(queue=[
        _StubResult(metrics=_StubMetrics()),
        _StubResult(metrics=_StubMetrics()),
    ])
    state = run_session(
        loop, "p",
        log=_silent, prompt_log=_silent,
        trim_messages=shrinking_trim,
    )
    # 2 successful turns, each evicts 1 → 2 events, 2 evicted.
    assert state.trim_events == 2
    assert state.trim_messages_evicted == 2


def test_no_trim_event_when_no_eviction():
    """No-op trim (returns messages unchanged) → trim_events stays 0."""
    loop = _StubChatLoop(queue=[_StubResult(metrics=_StubMetrics())])
    state = run_session(
        loop, "p",
        log=_silent, prompt_log=_silent,
        trim_messages=_stub_trim,
    )
    assert state.trim_events == 0
    assert state.trim_messages_evicted == 0


def test_trim_only_runs_after_successful_turn():
    """An LLM-error or false-trigger turn must NOT call
    trim_messages — invariant from the original inline code."""
    trim_calls: list[int] = []

    def counting_trim(messages, max_user_assistant: int):
        trim_calls.append(len(messages))
        return messages

    loop = _StubChatLoop(queue=[
        _StubResult(had_error=True),     # no trim
        _StubResult(metrics=None),        # no trim (false trigger)
        _StubResult(metrics=_StubMetrics()),  # trim once
    ])
    run_session(
        loop, "p",
        log=_silent, prompt_log=_silent,
        trim_messages=counting_trim,
    )
    assert len(trim_calls) == 1


# ---- prompt_log + log discipline ------------------------------------------


def test_prompt_log_fires_each_turn():
    """[N] waiting prompt fires before every run_one_turn call,
    including the eventual one that raises KeyboardInterrupt."""
    captured_turns: list[int] = []
    loop = _StubChatLoop(queue=[
        _StubResult(metrics=_StubMetrics()),
        _StubResult(metrics=_StubMetrics()),
    ])
    run_session(
        loop, "p",
        log=_silent,
        prompt_log=lambda t: captured_turns.append(t),
        trim_messages=_stub_trim,
    )
    # 2 successful turns, then 1 final prompt that triggered KI.
    assert captured_turns == [1, 2, 3]


def test_log_called_once_per_successful_turn():
    """log emits the post-stream newline; called once per
    successful turn, never on error/false-trigger."""
    log_calls: list[str] = []
    loop = _StubChatLoop(queue=[
        _StubResult(metrics=_StubMetrics()),
        _StubResult(had_error=True),
        _StubResult(metrics=_StubMetrics()),
    ])
    run_session(
        loop, "p",
        log=lambda s: log_calls.append(s),
        prompt_log=_silent,
        trim_messages=_stub_trim,
    )
    assert len(log_calls) == 2


# ---- max_user_assistant kwarg --------------------------------------------


def test_max_user_assistant_passed_to_trim():
    """The cap kwarg threads through to the injected
    trim_messages callable."""
    received_caps: list[int] = []

    def recording_trim(messages, max_user_assistant: int):
        received_caps.append(max_user_assistant)
        return messages

    loop = _StubChatLoop(queue=[_StubResult(metrics=_StubMetrics())])
    run_session(
        loop, "p",
        log=_silent, prompt_log=_silent,
        trim_messages=recording_trim,
        max_user_assistant=5,
    )
    assert received_caps == [5]


# ---- system_prompt seeded -----------------------------------------------


def test_system_prompt_is_first_message():
    """First message is always the system prompt — seed used by
    LLM context conditioning."""
    loop = _StubChatLoop(queue=[])
    state = run_session(
        loop, "you are a helpful assistant",
        log=_silent, prompt_log=_silent,
        trim_messages=_stub_trim,
    )
    assert state.messages[0] == {
        "role": "system", "content": "you are a helpful assistant",
    }
