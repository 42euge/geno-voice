"""Tests for iter-110 — run_session helper.

The function drives a chat loop until KeyboardInterrupt, tracking
session-level state in a SessionState bundle. Tests pass:
  - A stub `chat_loop` whose `.run_one_turn` returns a queued
    sequence of results (success / error / false-trigger).
  - A stub `trim_messages` so the test doesn't import ChatLoop.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
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
    # iter-161: the organic aggregator held this utterance mid-thought
    # (a successful capture being buffered for merge, NOT a VAD false
    # trigger). Defaults False so the legacy no-metrics path is unchanged.
    held: bool = False
    # iter-162: mid-thought fragments displaced by a genuinely-new turn
    # in this offer (abandoned, surfaced separately, not glued onto the
    # response). Defaults empty so the legacy path is unchanged.
    displaced: tuple = ()
    # iter-166: the recorder's pre-speech idle timeout fired on this turn
    # (the user stayed silent for the whole inter-turn window). Defaults
    # False so the legacy no-metrics path is unchanged.
    idle_timed_out: bool = False


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


# ---- iter-161 held-utterance handling -------------------------------------


def test_held_utterance_increments_utterances_held_not_false_triggers():
    """A no-metrics turn flagged ``held`` is the organic aggregator
    buffering a mid-thought fragment for merge — a successful capture,
    NOT a VAD false trigger. It bumps ``utterances_held`` and leaves
    ``false_triggers`` alone, while reusing the turn counter exactly
    like a false trigger (the operator re-listens for the continuation)."""
    loop = _StubChatLoop(queue=[
        _StubResult(metrics=None, had_error=False, held=True),
        _StubResult(metrics=_StubMetrics()),
    ])
    captured_turns: list[int] = []
    state = run_session(
        loop, "p",
        log=_silent,
        prompt_log=lambda t: captured_turns.append(t),
        trim_messages=_stub_trim,
    )
    assert state.utterances_held == 1
    assert state.false_triggers == 0
    assert len(state.all_metrics) == 1
    # Same prompt cadence as a false trigger: held reuses [1], success
    # advances to [1]→turn 2, queue-empty KI prompts [2].
    assert captured_turns == [1, 1, 2]


def test_held_and_false_trigger_counted_separately():
    """A held utterance and a genuine false trigger in the same session
    land in different counters — held does not pollute false_triggers."""
    loop = _StubChatLoop(queue=[
        _StubResult(metrics=None, held=True),   # held mid-thought
        _StubResult(metrics=None, held=False),  # genuine VAD false trigger
        _StubResult(metrics=_StubMetrics()),
    ])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
    )
    assert state.utterances_held == 1
    assert state.false_triggers == 1


def test_no_metrics_without_held_attr_defaults_to_false_trigger():
    """Back-compat: a result object that lacks a ``held`` attribute (the
    pre-iter-161 TurnResult shape) is still treated as a false trigger.
    ``run_session`` reads ``held`` defensively via getattr."""
    # SimpleNamespace WITHOUT a `held` field — mimics an old TurnResult.
    old_shape = SimpleNamespace(
        metrics=None, had_error=False, next_primed_frames=None,
    )
    loop = _StubChatLoop(queue=[old_shape, _StubResult(metrics=_StubMetrics())])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
    )
    assert state.false_triggers == 1
    assert state.utterances_held == 0


def test_utterances_held_defaults_zero():
    """A clean session (no held utterances) leaves the counter at 0."""
    loop = _StubChatLoop(queue=[_StubResult(metrics=_StubMetrics())])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
    )
    assert state.utterances_held == 0


# ---- iter-162: displaced-fragment collection ------------------------------


def test_displaced_collected_from_successful_turn():
    """A responded turn carrying ``displaced`` records the abandoned
    fragment(s) on ``state.utterances_displaced`` while still counting the
    turn as a normal success."""
    loop = _StubChatLoop(queue=[
        _StubResult(metrics=_StubMetrics(), displaced=("I was thinking about the",)),
    ])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
    )
    assert state.utterances_displaced == ["I was thinking about the"]
    assert len(state.all_metrics) == 1  # the turn still counts as a success
    assert state.false_triggers == 0


def test_displaced_accumulate_in_order_across_turns():
    """Fragments displaced across multiple turns accumulate in order."""
    loop = _StubChatLoop(queue=[
        _StubResult(metrics=_StubMetrics(), displaced=("first frag",)),
        _StubResult(metrics=_StubMetrics(), displaced=("second frag", "third frag")),
        _StubResult(metrics=_StubMetrics()),  # no displacement
    ])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
    )
    assert state.utterances_displaced == ["first frag", "second frag", "third frag"]


def test_displaced_collected_even_when_turn_errored():
    """A fragment the aggregator released rides out even if the turn it
    arrived on then hit an LLM error — the displaced text is real captured
    speech, independent of whether the response succeeded."""
    loop = _StubChatLoop(queue=[
        _StubResult(metrics=None, had_error=True, displaced=("stranded bit",)),
        _StubResult(metrics=_StubMetrics()),
    ])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
    )
    assert state.utterances_displaced == ["stranded bit"]
    assert state.llm_errors == 1


def test_displaced_without_attr_defaults_empty():
    """Back-compat: a result lacking ``displaced`` (pre-iter-162 shape)
    contributes nothing — read defensively via getattr."""
    old_shape = SimpleNamespace(
        metrics=_StubMetrics(), had_error=False, next_primed_frames=None,
    )
    loop = _StubChatLoop(queue=[old_shape])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
    )
    assert state.utterances_displaced == []


def test_utterances_displaced_defaults_empty():
    """A clean session leaves the list empty."""
    loop = _StubChatLoop(queue=[_StubResult(metrics=_StubMetrics())])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
    )
    assert state.utterances_displaced == []


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


# ---- iter-160: aggregator flush on exit -----------------------------------


@dataclass
class _StubEmittedTurn:
    """Mimics session.utterance_buffer.EmittedTurn — resolve_turn only
    reads .text / .false_endpoint."""
    text: str
    false_endpoint: bool = False


@dataclass
class _StubFlushResult:
    """Mimics AggregatedResult / BufferResult — resolve_turn duck-types
    over .turns / .held."""
    turns: list = field(default_factory=list)
    held: Optional[str] = None


class _StubAggregator:
    """Records whether flush() was called and returns a canned result."""

    def __init__(self, flush_result=None, raise_on_flush=False):
        self._flush_result = flush_result
        self._raise = raise_on_flush
        self.flush_calls = 0

    def flush(self):
        self.flush_calls += 1
        if self._raise:
            raise RuntimeError("boom")
        return self._flush_result


def test_no_aggregator_leaves_stranded_none():
    """Default path (no aggregator) never sets stranded_utterance."""
    loop = _StubChatLoop(queue=[])
    state = run_session(
        loop, "p",
        log=_silent, prompt_log=_silent,
        trim_messages=_stub_trim,
    )
    assert state.stranded_utterance is None


def test_aggregator_flushed_on_exit_even_with_nothing_held():
    """flush() is always called on exit so the buffer's cross-turn state
    is reset; an empty flush (nothing held) leaves stranded_utterance None."""
    agg = _StubAggregator(flush_result=_StubFlushResult(turns=[], held=None))
    loop = _StubChatLoop(queue=[])
    state = run_session(
        loop, "p",
        log=_silent, prompt_log=_silent,
        trim_messages=_stub_trim,
        aggregator=agg,
    )
    assert agg.flush_calls == 1
    assert state.stranded_utterance is None


def test_held_fragment_at_shutdown_recorded_as_stranded():
    """A flush that releases a held mid-thought fragment records the text
    on state.stranded_utterance."""
    agg = _StubAggregator(
        flush_result=_StubFlushResult(
            turns=[_StubEmittedTurn("I was going to say", False)],
            held=None,
        )
    )
    loop = _StubChatLoop(queue=[])
    state = run_session(
        loop, "p",
        log=_silent, prompt_log=_silent,
        trim_messages=_stub_trim,
        aggregator=agg,
    )
    assert agg.flush_calls == 1
    assert state.stranded_utterance == "I was going to say"


def test_flushed_blank_turn_leaves_stranded_none():
    """A flush releasing only blank text collapses (resolve_turn returns
    respond=False) — nothing stranded."""
    agg = _StubAggregator(
        flush_result=_StubFlushResult(
            turns=[_StubEmittedTurn("   ", False)], held=None,
        )
    )
    loop = _StubChatLoop(queue=[])
    state = run_session(
        loop, "p",
        log=_silent, prompt_log=_silent,
        trim_messages=_stub_trim,
        aggregator=agg,
    )
    assert state.stranded_utterance is None


def test_flush_exception_swallowed_state_still_returned():
    """A misbehaving aggregator must not crash the summary path — the
    exception is swallowed and stranded_utterance stays None."""
    agg = _StubAggregator(raise_on_flush=True)
    loop = _StubChatLoop(queue=[_StubResult(metrics=_StubMetrics())])
    state = run_session(
        loop, "p",
        log=_silent, prompt_log=_silent,
        trim_messages=_stub_trim,
        aggregator=agg,
    )
    assert agg.flush_calls == 1
    assert state.stranded_utterance is None
    # The completed turn still landed — flush is purely additive.
    assert len(state.all_metrics) == 1


def test_stranded_recorded_after_completed_turns():
    """A held fragment after one or more completed turns is still
    surfaced — the flush runs regardless of turn count."""
    agg = _StubAggregator(
        flush_result=_StubFlushResult(
            turns=[_StubEmittedTurn("and another thing", True)], held=None,
        )
    )
    loop = _StubChatLoop(queue=[_StubResult(metrics=_StubMetrics())])
    state = run_session(
        loop, "p",
        log=_silent, prompt_log=_silent,
        trim_messages=_stub_trim,
        aggregator=agg,
    )
    assert len(state.all_metrics) == 1
    assert state.stranded_utterance == "and another thing"


# ---- iter-160: real UtteranceAggregator integration -----------------------


def _load_real_aggregator():
    """Load session.utterance_aggregator by path (dodge the eager-pipecat
    import in session/__init__, absent on the x86_64 runner)."""
    import importlib.util
    import types

    session_dir = ROOT / "session"
    if "session" not in sys.modules:
        pkg = types.ModuleType("session")
        pkg.__path__ = [str(session_dir)]
        sys.modules["session"] = pkg
    for name in (
        "full_duplex",
        "text_eou",
        "utterance_merging",
        "utterance_buffer",
        "utterance_aggregator",
    ):
        full = f"session.{name}"
        if full in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(
            full, session_dir / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "session"
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    from session.full_duplex import FullDuplexConfig
    from session.utterance_aggregator import UtteranceAggregator

    return UtteranceAggregator, FullDuplexConfig


def test_real_organic_aggregator_strands_held_fragment_on_exit():
    """End-to-end with a REAL organic UtteranceAggregator: offer an
    unfinished-looking utterance (held, no continuation arrives), then let
    run_session flush on KeyboardInterrupt — the held text surfaces as
    stranded_utterance."""
    UtteranceAggregator, FullDuplexConfig = _load_real_aggregator()
    agg = UtteranceAggregator(
        config=FullDuplexConfig(enabled=True, utterance_merging=True)
    )
    # Offer a mid-thought fragment; with no prior endpoint the gap is inf
    # but the unfinished completeness score makes the buffer hold it.
    res = agg.offer("I was about to", speech_start_at=10.0, speech_end_at=11.0)
    assert res.turns == []  # held, nothing released yet
    assert agg.pending == "I was about to"

    loop = _StubChatLoop(queue=[])  # immediate KeyboardInterrupt — shutdown
    state = run_session(
        loop, "p",
        log=_silent, prompt_log=_silent,
        trim_messages=_stub_trim,
        aggregator=agg,
    )
    assert state.stranded_utterance == "I was about to"
    # Flush reset the buffer's cross-turn state.
    assert agg.pending is None


# ---- iter-167: idle-timeout handling + mid-session flush ------------------


class _StubAggregatorWithPending:
    """Aggregator stub exposing ``pending`` and a canned ``flush()``.

    Mirrors the real ``UtteranceAggregator`` surface ``_maybe_flush_on_idle``
    touches: a ``pending`` property (the held text) and ``flush()`` returning a
    duck-typed result with ``.turns`` / ``.held``."""

    def __init__(self, pending=None, flush_result=None, raise_on_flush=False):
        self._pending = pending
        self._flush_result = flush_result
        self._raise = raise_on_flush
        self.flush_calls = 0

    @property
    def pending(self):
        return self._pending

    def flush(self):
        # Idempotent like the real aggregator: a flush clears the pending, so a
        # SECOND flush (run_session always flushes again on shutdown — the
        # iter-160 path) releases nothing. This lets a test distinguish a
        # MID-SESSION flush (records on flushed_utterances) from the shutdown
        # flush (records on stranded_utterance).
        self.flush_calls += 1
        if self._raise:
            raise RuntimeError("boom")
        if self._pending is None or not self._pending.strip():
            return _StubFlushResult(turns=[], held=None)
        self._pending = None
        return self._flush_result


def test_idle_timeout_increments_idle_timeouts_not_false_triggers():
    """A no-metrics turn flagged ``idle_timed_out`` is the recorder's
    pre-speech idle timeout firing — a deliberate inter-turn-silence signal,
    NOT a VAD misfire. It bumps ``idle_timeouts`` and leaves ``false_triggers``
    and ``utterances_held`` alone, reusing the turn counter like a false
    trigger."""
    loop = _StubChatLoop(queue=[
        _StubResult(metrics=None, idle_timed_out=True),
        _StubResult(metrics=_StubMetrics()),
    ])
    captured_turns: list[int] = []
    state = run_session(
        loop, "p",
        log=_silent,
        prompt_log=lambda t: captured_turns.append(t),
        trim_messages=_stub_trim,
    )
    assert state.idle_timeouts == 1
    assert state.false_triggers == 0
    assert state.utterances_held == 0
    assert len(state.all_metrics) == 1
    assert captured_turns == [1, 1, 2]


def test_idle_timeout_held_and_false_trigger_counted_separately():
    """idle-timeout, held, and genuine false-trigger turns land in three
    distinct counters — none pollutes another."""
    loop = _StubChatLoop(queue=[
        _StubResult(metrics=None, idle_timed_out=True),
        _StubResult(metrics=None, held=True),
        _StubResult(metrics=None),  # genuine false trigger
        _StubResult(metrics=_StubMetrics()),
    ])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
    )
    assert state.idle_timeouts == 1
    assert state.utterances_held == 1
    assert state.false_triggers == 1


def test_idle_timeout_without_attr_defaults_to_false_trigger():
    """Back-compat: a result lacking ``idle_timed_out`` (pre-iter-166 shape)
    is still a false trigger — read defensively via getattr."""
    ns = SimpleNamespace(
        metrics=None, had_error=False, next_primed_frames=None,
    )
    loop = _StubChatLoop(queue=[ns])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
    )
    assert state.idle_timeouts == 0
    assert state.false_triggers == 1


def test_idle_timeouts_and_flushed_default_zero_and_empty():
    """Defaults: no idle timeouts seen, nothing flushed."""
    loop = _StubChatLoop(queue=[_StubResult(metrics=_StubMetrics())])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
    )
    assert state.idle_timeouts == 0
    assert state.flushed_utterances == []


def test_idle_timeout_no_flush_decider_does_not_flush():
    """An idle timeout with an aggregator holding a fragment but NO
    flush_decider wired never flushes MID-SESSION — byte-for-byte the
    pre-iter-167 path. (The shutdown flush still runs — iter-160 — so the held
    fragment strands rather than being flushed mid-session.)"""
    agg = _StubAggregatorWithPending(
        pending="I was thinking about the",
        flush_result=_StubFlushResult(
            turns=[_StubEmittedTurn("I was thinking about the")], held=None,
        ),
    )
    loop = _StubChatLoop(queue=[_StubResult(metrics=None, idle_timed_out=True)])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg,
        idle_timeout=5.0,
        # flush_decider omitted (None)
    )
    assert state.idle_timeouts == 1
    assert state.flushed_utterances == []  # nothing flushed mid-session
    # The shutdown flush (iter-160) released it instead — stranded, not flushed.
    assert agg.flush_calls == 1
    assert state.stranded_utterance == "I was thinking about the"


def test_idle_timeout_decider_says_flush_records_fragment():
    """An idle timeout + a held fragment + a decider that says FLUSH
    releases the fragment and records it on flushed_utterances."""
    agg = _StubAggregatorWithPending(
        pending="I was thinking about the",
        flush_result=_StubFlushResult(
            turns=[_StubEmittedTurn("I was thinking about the")], held=None,
        ),
    )
    seen: list[tuple] = []

    def decider(held, silence):
        seen.append((held, silence))
        return True

    loop = _StubChatLoop(queue=[_StubResult(metrics=None, idle_timed_out=True)])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=5.0, flush_decider=decider,
    )
    assert state.idle_timeouts == 1
    assert state.flushed_utterances == ["I was thinking about the"]
    # Mid-session flush (1) + shutdown flush (2, releases nothing now).
    assert agg.flush_calls == 2
    # The decider saw the held text and the idle_timeout as the silence.
    assert seen == [("I was thinking about the", 5.0)]
    # Mid-session flush already released it, so nothing strands at shutdown.
    assert state.stranded_utterance is None


def test_idle_timeout_decider_says_hold_does_not_flush():
    """A decider that says HOLD (e.g. idle_timeout shorter than the merge
    window) leaves the fragment held — nothing flushed."""
    agg = _StubAggregatorWithPending(
        pending="I was thinking about the",
        flush_result=_StubFlushResult(turns=[], held="I was thinking about the"),
    )
    loop = _StubChatLoop(queue=[_StubResult(metrics=None, idle_timed_out=True)])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=1.0, flush_decider=lambda h, s: False,
    )
    assert state.idle_timeouts == 1
    assert state.flushed_utterances == []  # HOLD ⇒ no mid-session flush
    # Only the shutdown flush ran (HOLD short-circuited the mid-session one).
    assert agg.flush_calls == 1


def test_idle_timeout_nothing_held_does_not_flush():
    """An idle timeout with the aggregator holding nothing never calls the
    decider or flush — there is nothing to flush."""
    agg = _StubAggregatorWithPending(pending=None)
    called = []
    loop = _StubChatLoop(queue=[_StubResult(metrics=None, idle_timed_out=True)])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=5.0,
        flush_decider=lambda h, s: called.append((h, s)) or True,
    )
    assert state.idle_timeouts == 1
    assert state.flushed_utterances == []
    # Only the shutdown flush ran; the mid-session path bailed (nothing held).
    assert agg.flush_calls == 1
    assert called == []  # decider never consulted — nothing held


def test_idle_timeout_blank_pending_does_not_flush():
    """A whitespace-only pending is treated as nothing held."""
    agg = _StubAggregatorWithPending(pending="   ")
    loop = _StubChatLoop(queue=[_StubResult(metrics=None, idle_timed_out=True)])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=5.0, flush_decider=lambda h, s: True,
    )
    assert state.flushed_utterances == []
    # Mid-session path bailed (blank pending); only the shutdown flush ran.
    assert agg.flush_calls == 1


def test_idle_timeout_none_idle_timeout_feeds_inf_silence():
    """When idle_timeout is None (caller didn't tell run_session the
    recorder's window) the decider is fed inf silence — the timeout
    demonstrably fired, so let the decider's own gate decide."""
    agg = _StubAggregatorWithPending(
        pending="held bit",
        flush_result=_StubFlushResult(
            turns=[_StubEmittedTurn("held bit")], held=None,
        ),
    )
    seen: list[tuple] = []
    loop = _StubChatLoop(queue=[_StubResult(metrics=None, idle_timed_out=True)])
    run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=None,
        flush_decider=lambda h, s: seen.append((h, s)) or True,
    )
    assert seen == [("held bit", float("inf"))]


def test_idle_timeout_no_aggregator_does_not_flush():
    """An idle timeout with no aggregator wired just counts — no flush."""
    loop = _StubChatLoop(queue=[_StubResult(metrics=None, idle_timed_out=True)])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        idle_timeout=5.0, flush_decider=lambda h, s: True,
    )
    assert state.idle_timeouts == 1
    assert state.flushed_utterances == []


def test_idle_timeout_flush_releasing_blank_records_nothing():
    """A flush that releases only blank text collapses (resolve_turn
    respond=False) — nothing recorded even though flush was called."""
    agg = _StubAggregatorWithPending(
        pending="held bit",
        flush_result=_StubFlushResult(turns=[_StubEmittedTurn("   ")], held=None),
    )
    loop = _StubChatLoop(queue=[_StubResult(metrics=None, idle_timed_out=True)])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=5.0, flush_decider=lambda h, s: True,
    )
    # Mid-session flush (1, blank → resolve_turn respond=False) + shutdown (2).
    assert agg.flush_calls == 2
    assert state.flushed_utterances == []


def test_idle_timeout_decider_exception_swallowed():
    """A misbehaving decider must not crash the live loop — treated as HOLD,
    nothing flushed, the session continues."""
    agg = _StubAggregatorWithPending(
        pending="held bit",
        flush_result=_StubFlushResult(
            turns=[_StubEmittedTurn("held bit")], held=None,
        ),
    )

    def boom(held, silence):
        raise RuntimeError("decider blew up")

    loop = _StubChatLoop(queue=[
        _StubResult(metrics=None, idle_timed_out=True),
        _StubResult(metrics=_StubMetrics()),
    ])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=5.0, flush_decider=boom,
    )
    assert state.idle_timeouts == 1
    assert state.flushed_utterances == []  # decider raised ⇒ no mid-session flush
    # The decider raised before any mid-session flush; only the shutdown flush
    # ran (releasing the still-held fragment as stranded).
    assert agg.flush_calls == 1
    assert state.stranded_utterance == "held bit"
    # The completed turn still landed.
    assert len(state.all_metrics) == 1


def test_idle_timeout_flush_exception_swallowed():
    """A misbehaving aggregator.flush() must not crash the loop — swallowed,
    nothing recorded, the session continues."""
    agg = _StubAggregatorWithPending(pending="held bit", raise_on_flush=True)
    loop = _StubChatLoop(queue=[
        _StubResult(metrics=None, idle_timed_out=True),
        _StubResult(metrics=_StubMetrics()),
    ])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=5.0, flush_decider=lambda h, s: True,
    )
    assert state.idle_timeouts == 1
    assert state.flushed_utterances == []
    # Mid-session flush raised (1, caught); shutdown flush raised too (2, caught).
    assert agg.flush_calls == 2
    assert state.stranded_utterance is None
    assert len(state.all_metrics) == 1


def test_multiple_idle_flushes_accumulate_in_order():
    """Two idle-timeout turns each flushing a fragment accumulate on
    flushed_utterances in order."""
    class _MultiAgg:
        def __init__(self):
            self._pendings = ["first frag", "second frag"]
            self._results = [
                _StubFlushResult(turns=[_StubEmittedTurn("first frag")]),
                _StubFlushResult(turns=[_StubEmittedTurn("second frag")]),
            ]
            self.flush_calls = 0

        @property
        def pending(self):
            return self._pendings[self.flush_calls] if (
                self.flush_calls < len(self._pendings)
            ) else None

        def flush(self):
            # The shutdown flush (iter-160) calls once more after the two
            # mid-session flushes; return an empty result then.
            if self.flush_calls >= len(self._results):
                self.flush_calls += 1
                return _StubFlushResult(turns=[], held=None)
            r = self._results[self.flush_calls]
            self.flush_calls += 1
            return r

    agg = _MultiAgg()
    loop = _StubChatLoop(queue=[
        _StubResult(metrics=None, idle_timed_out=True),
        _StubResult(metrics=None, idle_timed_out=True),
    ])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=5.0, flush_decider=lambda h, s: True,
    )
    assert state.idle_timeouts == 2
    assert state.flushed_utterances == ["first frag", "second frag"]


def _load_real_silence_flush():
    """Load session.silence_flush by path (same eager-pipecat dodge as
    _load_real_aggregator)."""
    import importlib.util
    session_dir = ROOT / "session"
    spec = importlib.util.spec_from_file_location(
        "session.silence_flush", session_dir / "silence_flush.py",
    )
    sf = importlib.util.module_from_spec(spec)
    sf.__package__ = "session"
    spec.loader.exec_module(sf)
    return sf.should_flush_held_utterance


def test_real_organic_aggregator_flushes_held_fragment_mid_session():
    """End-to-end with a REAL organic UtteranceAggregator + the REAL
    should_flush_held_utterance decider: offer an unfinished fragment (held),
    then an idle-timeout turn with a long silence flushes it mid-session and
    records it on flushed_utterances — the held pending is released, not left
    until shutdown."""
    UtteranceAggregator, FullDuplexConfig = _load_real_aggregator()
    should_flush = _load_real_silence_flush()

    config = FullDuplexConfig(enabled=True, utterance_merging=True)
    agg = UtteranceAggregator(config=config)
    res = agg.offer("I was about to", speech_start_at=10.0, speech_end_at=11.0)
    assert res.turns == []  # held
    assert agg.pending == "I was about to"

    decider = lambda held, silence: should_flush(
        held_text=held, silence_secs=silence, config=config,
    )
    # idle_timeout=5.0 > the 2.0s merge window so the decider flushes.
    loop = _StubChatLoop(queue=[_StubResult(metrics=None, idle_timed_out=True)])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=5.0, flush_decider=decider,
    )
    assert state.idle_timeouts == 1
    assert state.flushed_utterances == ["I was about to"]
    assert agg.pending is None  # the flush released it mid-session
    # And nothing was left to strand at shutdown.
    assert state.stranded_utterance is None


def test_real_organic_aggregator_holds_when_idle_timeout_under_window():
    """An idle_timeout SHORTER than the merge window leaves the real decider
    saying HOLD — the fragment stays pending, nothing flushed mid-session
    (it would still strand at shutdown)."""
    UtteranceAggregator, FullDuplexConfig = _load_real_aggregator()
    should_flush = _load_real_silence_flush()

    config = FullDuplexConfig(enabled=True, utterance_merging=True)
    agg = UtteranceAggregator(config=config)
    agg.offer("I was about to", speech_start_at=10.0, speech_end_at=11.0)
    assert agg.pending == "I was about to"

    decider = lambda held, silence: should_flush(
        held_text=held, silence_secs=silence, config=config,
    )
    # idle_timeout=1.0 < 2.0s merge window ⇒ HOLD.
    loop = _StubChatLoop(queue=[_StubResult(metrics=None, idle_timed_out=True)])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=1.0, flush_decider=decider,
    )
    assert state.idle_timeouts == 1
    assert state.flushed_utterances == []  # HOLD ⇒ no mid-session flush
    # The fragment was NOT flushed mid-session; the iter-160 shutdown flush
    # then released it as stranded (so pending is cleared by that final flush).
    assert state.stranded_utterance == "I was about to"
    assert agg.pending is None


def test_real_half_duplex_aggregator_never_strands():
    """A half-duplex aggregator (default config) never holds, so flush on
    exit releases nothing — stranded_utterance stays None."""
    UtteranceAggregator, FullDuplexConfig = _load_real_aggregator()
    agg = UtteranceAggregator(config=FullDuplexConfig())  # half-duplex
    # Even an unfinished-looking utterance is emitted immediately.
    res = agg.offer("I was about to", speech_start_at=10.0, speech_end_at=11.0)
    assert len(res.turns) == 1
    assert agg.pending is None

    loop = _StubChatLoop(queue=[])
    state = run_session(
        loop, "p",
        log=_silent, prompt_log=_silent,
        trim_messages=_stub_trim,
        aggregator=agg,
    )
    assert state.stranded_utterance is None


# ---- iter-169: speak a flushed mid-session fragment as its own turn --------


class _RespondFn:
    """Stub for the injected respond_fn (ChatLoop.respond_to_text shape).

    Records each (messages-snapshot, text) call. By default returns a result
    carrying a fresh _StubMetrics and appends a user+assistant message pair to
    `messages` (mirroring respond_to_text's history mutation), so the session
    bookkeeping has something real to fold in.
    """

    def __init__(self, *, had_error=False, metrics_none=False, raise_exc=False):
        self.calls: list[tuple] = []
        self._had_error = had_error
        self._metrics_none = metrics_none
        self._raise = raise_exc
        self.metrics_made: list = []

    def __call__(self, messages, text):
        self.calls.append((list(messages), text))
        if self._raise:
            raise RuntimeError("respond boom")
        if self._had_error:
            return SimpleNamespace(metrics=None, had_error=True,
                                   next_primed_frames=None)
        if self._metrics_none:
            return SimpleNamespace(metrics=None, had_error=False,
                                   next_primed_frames=None)
        m = _StubMetrics()
        self.metrics_made.append(m)
        # respond_to_text appends user + assistant to history on success.
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": "ok"})
        return SimpleNamespace(metrics=m, had_error=False,
                               next_primed_frames=None)


def _flushing_agg(text="I was thinking about the"):
    return _StubAggregatorWithPending(
        pending=text,
        flush_result=_StubFlushResult(turns=[_StubEmittedTurn(text)], held=None),
    )


def test_flushed_fragment_spoken_when_respond_fn_wired():
    """A mid-session flush + a wired respond_fn SPEAKS the fragment: respond_fn
    is called with the flushed text, the spoken turn is counted (metrics
    printed, all_metrics appended, turn counter advanced)."""
    agg = _flushing_agg()
    respond = _RespondFn()
    captured_turns: list[int] = []
    loop = _StubChatLoop(queue=[
        _StubResult(metrics=None, idle_timed_out=True),
        _StubResult(metrics=_StubMetrics()),  # a normal mic turn after
    ])
    state = run_session(
        loop, "p", log=_silent,
        prompt_log=lambda t: captured_turns.append(t),
        trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=5.0, flush_decider=lambda h, s: True,
        respond_fn=respond,
    )
    # The fragment was both recorded AND spoken.
    assert state.flushed_utterances == ["I was thinking about the"]
    assert len(respond.calls) == 1
    assert respond.calls[0][1] == "I was thinking about the"
    # The spoken flush counts as a real turn: its metrics printed under turn 1,
    # appended to all_metrics alongside the later mic turn (2 total).
    assert len(state.all_metrics) == 2
    assert respond.metrics_made[0].printed_turns == [1]
    # idle_timeout turn did NOT consume the prompt counter, but the spoken turn
    # advanced it: prompt [1] (idle), then mic turn sees [2], queue-empty KI [3].
    assert captured_turns == [1, 2, 3]
    assert state.idle_timeouts == 1


def test_flushed_fragment_recorded_not_spoken_without_respond_fn():
    """No respond_fn (default) ⇒ the fragment is recorded on flushed_utterances
    but NOT spoken — byte-for-byte the pre-iter-169 path."""
    agg = _flushing_agg()
    loop = _StubChatLoop(queue=[_StubResult(metrics=None, idle_timed_out=True)])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=5.0, flush_decider=lambda h, s: True,
        # respond_fn omitted (None)
    )
    assert state.flushed_utterances == ["I was thinking about the"]
    assert state.all_metrics == []  # nothing spoken


def test_respond_fn_not_called_when_nothing_flushed():
    """A HOLD decision (nothing flushed) never calls respond_fn, even when one
    is wired."""
    agg = _flushing_agg()
    respond = _RespondFn()
    loop = _StubChatLoop(queue=[_StubResult(metrics=None, idle_timed_out=True)])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=5.0, flush_decider=lambda h, s: False,
        respond_fn=respond,
    )
    assert state.flushed_utterances == []
    assert respond.calls == []


def test_spoken_flush_appends_to_conversation_history():
    """The spoken flush threads the SAME messages list through respond_fn, so
    the user+assistant pair it appends persists into state.messages."""
    agg = _flushing_agg("the meeting is at three")
    respond = _RespondFn()
    loop = _StubChatLoop(queue=[_StubResult(metrics=None, idle_timed_out=True)])
    state = run_session(
        loop, "sys", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=5.0, flush_decider=lambda h, s: True,
        respond_fn=respond,
    )
    assert state.messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "the meeting is at three"},
        {"role": "assistant", "content": "ok"},
    ]


def test_spoken_flush_llm_error_counts_as_llm_error():
    """respond_fn returning had_error counts the spoken flush as an LLM error
    and does NOT advance the turn counter — mirrors the mic-turn error path.
    The fragment is still recorded on flushed_utterances."""
    agg = _flushing_agg()
    respond = _RespondFn(had_error=True)
    loop = _StubChatLoop(queue=[_StubResult(metrics=None, idle_timed_out=True)])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=5.0, flush_decider=lambda h, s: True,
        respond_fn=respond,
    )
    assert state.flushed_utterances == ["I was thinking about the"]
    assert state.llm_errors == 1
    assert state.all_metrics == []


def test_spoken_flush_respond_fn_exception_swallowed():
    """A respond_fn that RAISES must not break the live loop: counted as an
    llm_error, the fragment stays recorded, the session returns normally."""
    agg = _flushing_agg()
    respond = _RespondFn(raise_exc=True)
    loop = _StubChatLoop(queue=[_StubResult(metrics=None, idle_timed_out=True)])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=5.0, flush_decider=lambda h, s: True,
        respond_fn=respond,
    )
    assert state.flushed_utterances == ["I was thinking about the"]
    assert state.llm_errors == 1
    assert state.all_metrics == []


def test_spoken_flush_metrics_none_records_nothing():
    """respond_fn returning metrics=None (declined to produce a turn) records
    no metrics and leaves the turn counter alone — the fragment stays only on
    flushed_utterances."""
    agg = _flushing_agg()
    respond = _RespondFn(metrics_none=True)
    loop = _StubChatLoop(queue=[_StubResult(metrics=None, idle_timed_out=True)])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=5.0, flush_decider=lambda h, s: True,
        respond_fn=respond,
    )
    assert state.flushed_utterances == ["I was thinking about the"]
    assert state.llm_errors == 0
    assert state.all_metrics == []


def test_spoken_flush_runs_trim_after_turn():
    """The spoken flush runs trim_messages like any completed turn — an
    eviction increments the trim counters."""
    agg = _flushing_agg()
    respond = _RespondFn()

    def _evicting_trim(messages, max_user_assistant):
        # Drop the oldest non-system message to force one eviction.
        if len(messages) > 2:
            return messages[:1] + messages[2:]
        return messages

    loop = _StubChatLoop(queue=[_StubResult(metrics=None, idle_timed_out=True)])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent,
        trim_messages=_evicting_trim,
        aggregator=agg, idle_timeout=5.0, flush_decider=lambda h, s: True,
        respond_fn=respond,
    )
    assert state.trim_events == 1
    assert state.trim_messages_evicted == 1


def test_real_organic_flush_spoken_end_to_end():
    """End-to-end with the REAL aggregator + decider AND a respond_fn: the held
    fragment is flushed mid-session and spoken as its own turn."""
    UtteranceAggregator, FullDuplexConfig = _load_real_aggregator()
    should_flush = _load_real_silence_flush()

    config = FullDuplexConfig(enabled=True, utterance_merging=True)
    agg = UtteranceAggregator(config=config)
    agg.offer("I was about to", speech_start_at=10.0, speech_end_at=11.0)
    assert agg.pending == "I was about to"

    decider = lambda held, silence: should_flush(
        held_text=held, silence_secs=silence, config=config,
    )
    respond = _RespondFn()
    loop = _StubChatLoop(queue=[_StubResult(metrics=None, idle_timed_out=True)])
    state = run_session(
        loop, "p", log=_silent, prompt_log=_silent, trim_messages=_stub_trim,
        aggregator=agg, idle_timeout=5.0, flush_decider=decider,
        respond_fn=respond,
    )
    assert state.flushed_utterances == ["I was about to"]
    assert respond.calls and respond.calls[0][1] == "I was about to"
    assert len(state.all_metrics) == 1  # the spoken flush turn
    assert state.stranded_utterance is None
