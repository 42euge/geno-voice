"""Main turn loop — extracted from mic_chat.py:run_chat.

iter-110: completes the run_chat decomposition started in
iter-107 (filler pre-render), iter-108 (engine load), iter-109
(audio I/O closures). The remaining big block was the turn loop
itself — ~50 lines mixing per-turn flow, session-level counters,
KeyboardInterrupt handling, and trim plumbing.

Pulled into `run_session(...)` returning a `SessionState` so
the caller can pass it straight to `print_session_summary`.
The state bundle replaces the 6 mutable locals that previously
lived in `run_chat`'s scope.

Follows the GENO.md "mic_chat.py extraction pattern":
- chat_loop is a callable dependency (the .run_one_turn shape
  is what we actually use, not the class)
- log injected (default print)
- ANSI styling stays at the caller — the prompt line "[N]
  waiting..." gets re-styled via a `prompt_log` callable
- Returns SessionState dataclass
- No platform deps to lazy-import here (this module is pure
  orchestration over already-extracted pieces)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class SessionState:
    """Bundle of session-level state collected across turns.

    Each field corresponds to one of the 6 mutable locals that
    previously lived in `run_chat`'s scope. After `run_session`
    returns (KeyboardInterrupt or other exit), the caller hands
    this bundle to `print_session_summary` — the field names
    line up with `SessionMeta` 1:1 so the conversion is mechanical.

    The `messages` list is mutated in place across turns and held
    here so a future eval/replay tool can inspect the conversation
    history post-mortem.
    """

    messages: list[dict] = field(default_factory=list)
    all_metrics: list = field(default_factory=list)
    false_triggers: int = 0
    # iter-161: turns where the organic UtteranceAggregator HELD the
    # utterance mid-thought (a successful capture buffered for a merge),
    # which iter-159 surfaces as a no-metrics TurnResult flagged
    # ``held``. Counted separately from ``false_triggers`` so a held
    # fragment — the opposite of a VAD misfire — does not inflate the
    # false-trigger rate. 0 on the half-duplex / no-aggregator path
    # (nothing is ever held).
    utterances_held: int = 0
    # iter-162: mid-thought fragments the organic UtteranceAggregator
    # released *alongside* a responded turn — the user trailed off, a long
    # silence proved the fragment was NOT a false endpoint, then a genuinely
    # new thought arrived and displaced it. Each is captured-but-abandoned
    # text (the mid-session analog of ``stranded_utterance``, which is the
    # shutdown case). Collected in order across the session so the summary
    # can surface them rather than the pre-iter-162 behavior of silently
    # gluing them onto the response. Empty on the half-duplex / no-aggregator
    # path (the buffer never releases more than one turn at a time there).
    utterances_displaced: list[str] = field(default_factory=list)
    llm_errors: int = 0
    trim_events: int = 0
    trim_messages_evicted: int = 0
    primed_frames: Optional[list[bytes]] = None
    session_start: float = 0.0
    # iter-160: text the organic UtteranceAggregator was still holding
    # back when the session ended (the user trailed off mid-thought and
    # never landed a continuation, then hit Ctrl+C). ``None`` when no
    # aggregator was wired in, when nothing was held, or when merging is
    # off (half-duplex never holds). Surfaced in the session summary so a
    # dropped final fragment is visible rather than silently lost.
    stranded_utterance: Optional[str] = None


def run_session(
    chat_loop: Any,
    system_prompt: str,
    *,
    max_user_assistant: int = 20,
    log: Callable[[str], None] = print,
    prompt_log: Callable[[int], None] = lambda turn: print(
        f"  [{turn}] waiting...", end="", flush=True
    ),
    clock: Callable[[], float] = time.monotonic,
    trim_messages: Optional[Callable[[list[dict], int], list[dict]]] = None,
    aggregator: Any = None,
) -> SessionState:
    """Run the chat loop until KeyboardInterrupt.

    Args:
        chat_loop: object with `.run_one_turn(messages, primed_frames=None)`
            returning a result with `.metrics`, `.had_error`,
            `.next_primed_frames`. The production wiring is a
            `ChatLoop` instance; tests pass any object matching
            the contract.
        system_prompt: first message added to the conversation.
        max_user_assistant: cap passed to `trim_messages` after
            each successful turn (iter-024).
        log: emit callable for the post-turn newline that flushes
            the streamed bot text. Default `print`. Tests can
            silence with `log=lambda s: None`.
        prompt_log: emit callable for the "[N] waiting..." prompt
            shown before each turn. Default re-creates the
            mic_chat.py styling. Tests can capture turn count.
        clock: monotonic-time source for `session_start`.
        trim_messages: callable for context-cap enforcement. When
            None (default), defers to the `ChatLoop.trim_messages`
            staticmethod. Tests pass a stub to avoid the import.
        aggregator: the same `UtteranceAggregator` instance wired into
            `chat_loop` (iter-159), or None (default). When present, on
            exit `run_session` calls `aggregator.flush()` to release any
            mid-thought utterance the buffer was still holding — the
            user trailed off and never landed a continuation, then hit
            Ctrl+C. The flushed text is recorded on
            `state.stranded_utterance` so the session summary can surface
            it rather than dropping it silently. None / nothing-held /
            half-duplex all leave `stranded_utterance` at None.

    Returns:
        Populated SessionState. The caller is responsible for
        passing it to print_session_summary (or any other
        terminal aggregator).

    Side effects:
        Mutates `state.messages` in place across turns.
        Catches `KeyboardInterrupt` and returns normally — that's
        the expected exit path. Other exceptions propagate.

    iter-048 false-trigger semantics: a turn that produced no
    metrics and no error is counted as a false trigger and the
    loop continues without consuming the turn counter (the
    operator sees the same `[N] waiting...` prompt next time).

    iter-058 LLM-error semantics: a turn flagged `had_error` is
    counted in `llm_errors` and the loop continues; the same turn
    counter is reused.

    iter-078 trim semantics: after every SUCCESSFUL turn (metrics
    populated, no error), `trim_messages` is called and any
    eviction increments both counters.
    """
    if trim_messages is None:
        # Lazy import — keeps run_session importable without
        # pulling in ChatLoop (and its transitive deps). Same
        # rule 5 of the GENO.md pattern.
        from examples._chat_loop import ChatLoop
        trim_messages = ChatLoop.trim_messages

    state = SessionState(
        messages=[{"role": "system", "content": system_prompt}],
        session_start=clock(),
    )

    try:
        turn = 0
        while True:
            prompt_log(turn + 1)
            result = chat_loop.run_one_turn(
                state.messages, primed_frames=state.primed_frames,
            )
            state.primed_frames = result.next_primed_frames
            # iter-162: collect any displaced mid-thought fragments the
            # aggregator released alongside this turn. Read defensively
            # (getattr) so a pre-iter-162 TurnResult shape without the field
            # is treated as "none displaced". Captured here — before the
            # error / no-metrics branches — so a fragment surfaces even when
            # the turn it rode in on then errored or was a held re-listen.
            state.utterances_displaced.extend(getattr(result, "displaced", ()) or ())
            if result.had_error:
                state.llm_errors += 1
                continue
            if result.metrics is None:
                # iter-161: a no-metrics turn is EITHER the organic
                # aggregator holding a mid-thought utterance for a merge
                # (a successful capture — ``held``) OR a genuine VAD
                # false trigger (recorder fired but no transcript). Read
                # ``held`` defensively (getattr) so a pre-iter-161
                # TurnResult shape without the field still counts as a
                # false trigger. Both paths re-listen without consuming
                # the turn counter.
                if getattr(result, "held", False):
                    state.utterances_held += 1
                else:
                    state.false_triggers += 1
                continue
            log("")  # newline after streamed bot text
            result.metrics.print(turn + 1)
            state.all_metrics.append(result.metrics)
            turn += 1
            len_before = len(state.messages)
            state.messages = trim_messages(
                state.messages, max_user_assistant=max_user_assistant,
            )
            evicted = len_before - len(state.messages)
            if evicted > 0:
                state.trim_events += 1
                state.trim_messages_evicted += evicted
    except KeyboardInterrupt:
        # Expected exit path — the caller knows to dump the summary.
        pass

    # iter-160: flush the organic aggregator on the way out. A held
    # mid-thought utterance only ever surfaces mid-session when the NEXT
    # utterance arrives (its measured gap forces a NEW release inside
    # ``offer``). The one case ``offer`` can never reach is *shutdown*:
    # the user trailed off after a fragment, never spoke again, then hit
    # Ctrl+C. That text is held inside the buffer and would be silently
    # lost. Flushing here releases it; we record (not respond to — the
    # session is ending) the joined text on ``state.stranded_utterance``
    # so the summary can surface the dropped fragment. None / nothing-held
    # / half-duplex (never holds) all leave it at None.
    if aggregator is not None:
        try:
            flushed = aggregator.flush()
        except Exception:
            # A misbehaving aggregator must not mask the summary the
            # caller is about to print — swallow and leave it at None.
            flushed = None
        if flushed is not None:
            from examples._chat_aggregation import resolve_turn
            resolved = resolve_turn(flushed)
            if resolved.respond and resolved.text:
                state.stranded_utterance = resolved.text

    return state
