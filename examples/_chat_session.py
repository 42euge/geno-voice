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
    # iter-167: turns where the recorder's pre-speech idle timeout (iter-165)
    # fired — the user stayed silent for the whole inter-turn window without
    # starting to speak, so the loop regained control instead of blocking. The
    # recorder returns a no-metrics TurnResult flagged ``idle_timed_out``
    # (iter-166). Counted SEPARATELY from ``false_triggers`` because an idle
    # timeout is a deliberate inter-turn-silence signal, NOT a VAD misfire —
    # conflating the two would inflate the false-trigger rate every time an
    # operator enabled an ``idle_timeout``. 0 on the default wait-forever path
    # (no ``idle_timeout`` wired into the recorder).
    idle_timeouts: int = 0
    # iter-167: mid-thought fragments the organic UtteranceAggregator was
    # holding when a long inter-turn idle silence (the iter-165 recorder
    # timeout) proved no continuation was coming, so ``run_session`` flushed
    # them mid-session via the injected ``flush_decider`` (the iter-164
    # ``decide_silence_flush`` seam). The mid-session analog of
    # ``stranded_utterance`` (the shutdown case) and of iter-162's
    # ``utterances_displaced`` (the new-thought-displaces case): a trailed-off
    # fragment that the buffer would otherwise have held until a genuinely-new
    # thought displaced it or shutdown flushed it. Collected in order so the
    # summary can surface them. Empty on the half-duplex / no-aggregator /
    # no-idle_timeout path. NOTE: this lap *records* the flushed fragment so it
    # is visible and the buffer is unblocked; producing a spoken response to it
    # is the explicit next hop (needs a ChatLoop text-only response entrypoint —
    # ``run_one_turn`` always records from the mic first).
    flushed_utterances: list[str] = field(default_factory=list)
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


def _record_completed_turn(
    state: SessionState,
    metrics: Any,
    turn: int,
    *,
    log: Callable[[str], None],
    trim_messages: Callable[[list[dict], int], list[dict]],
    max_user_assistant: int,
) -> int:
    """Fold one completed turn's metrics into the session state.

    iter-169: extracted from ``run_session``'s inline success block so the same
    bookkeeping drives both a mic turn and a *spoken flushed fragment* (the
    iter-168 ``respond_to_text`` path wired in below). The body is moved, not
    modified — the mic-turn path is byte-for-byte the pre-iter-169 inline block,
    proven by the unchanged ``test_chat_session.py`` suite.

    Prints the metrics under ``turn + 1``, appends them to ``all_metrics``,
    advances the turn counter, runs the context-cap trim and records any
    eviction. Returns the new turn count (the caller threads it back into the
    loop local).
    """
    log("")  # newline after streamed bot text
    metrics.print(turn + 1)
    state.all_metrics.append(metrics)
    turn += 1
    len_before = len(state.messages)
    state.messages = trim_messages(
        state.messages, max_user_assistant=max_user_assistant,
    )
    evicted = len_before - len(state.messages)
    if evicted > 0:
        state.trim_events += 1
        state.trim_messages_evicted += evicted
    return turn


def _speak_flushed_fragment(
    state: SessionState,
    respond_fn: Callable[[list[dict], str], Any],
    text: str,
    turn: int,
    *,
    log: Callable[[str], None],
    trim_messages: Callable[[list[dict], int], list[dict]],
    max_user_assistant: int,
) -> int:
    """iter-169: speak a mid-session flushed fragment as its own turn.

    The last hop of backlog #9's mid-session flush. ``_maybe_flush_on_idle``
    released a held mid-thought fragment (and recorded it on
    ``flushed_utterances``); this answers it through the injected ``respond_fn``
    (production: ``ChatLoop.respond_to_text``, iter-168 — LLM stream → synth/play
    over the text, no mic recording). On a successful response the turn is folded
    into the session exactly like a mic turn (``_record_completed_turn``); on an
    LLM error it is counted in ``llm_errors`` and the turn counter is left alone,
    mirroring the mic-turn error path.

    Wrapped in ``try/except``: a misbehaving ``respond_fn`` must not break the
    live loop — the flushed text is already recorded on ``flushed_utterances``,
    so a failed *spoken* response degrades to the pre-iter-169 record-only
    behavior rather than crashing the session. Returns the (possibly advanced)
    turn count.
    """
    try:
        res = respond_fn(state.messages, text)
    except Exception:
        # respond_fn raised — the fragment is still recorded; count as an LLM
        # error (the spoken response failed) and keep the loop alive.
        state.llm_errors += 1
        return turn
    if getattr(res, "had_error", False):
        state.llm_errors += 1
        return turn
    metrics = getattr(res, "metrics", None)
    if metrics is None:
        # respond_fn declined to produce a turn (e.g. blank text after its own
        # strip). Nothing to record; the fragment stays on flushed_utterances.
        return turn
    return _record_completed_turn(
        state, metrics, turn,
        log=log,
        trim_messages=trim_messages,
        max_user_assistant=max_user_assistant,
    )


def _maybe_flush_on_idle(
    state: SessionState,
    aggregator: Any,
    flush_decider: Optional[Callable[[str, float], bool]],
    idle_timeout: Optional[float],
) -> Optional[str]:
    """iter-167: on a recorder idle-timeout turn, maybe flush a held fragment.

    The mid-session half of backlog #9. When the recorder's pre-speech idle
    timeout (iter-165) fires, the user has been silent for the whole inter-turn
    window without starting a new utterance. If the organic aggregator is
    holding a mid-thought fragment, this is exactly the moment the iter-164
    ``decide_silence_flush`` seam was built to answer: give up waiting for a
    continuation and flush the fragment now, rather than leaving it held until a
    genuinely-new thought displaces it (iter-162) or shutdown flushes it
    (iter-160).

    Stays decoupled from the ``session`` package (whose eager pipecat import is
    absent on the x86_64 test runner) the same way ``trim_messages`` does: the
    flush *decision* is an injected callable ``flush_decider(held_text,
    silence_secs) -> bool`` — production wires it to
    ``session.silence_flush.should_flush_held_utterance`` bound to the
    aggregator's config; tests pass a stub. When ``flush_decider`` is ``None``
    (the default) nothing flushes — byte-for-byte the pre-iter-167 path, even
    with an aggregator wired in.

    Guards, first to bail wins:
      - No aggregator ⇒ nothing is ever held; return.
      - No ``flush_decider`` ⇒ caller opted out of mid-session flushing; return.
      - Nothing held (``aggregator.pending`` blank/absent) ⇒ return.
      - ``flush_decider`` says HOLD ⇒ the merge window has not closed yet (e.g.
        an ``idle_timeout`` shorter than ``max_gap_secs`` — a continuation could
        still arrive); return without flushing.

    The ``silence_secs`` fed to the decider is ``idle_timeout`` — the recorder
    timed out after exactly that much pre-speech silence, so it is the inter-turn
    silence elapsed. When ``idle_timeout`` is ``None`` (the caller did not tell
    ``run_session`` the recorder's window) we fall back to ``inf``: the timeout
    flag demonstrably fired, so *some* long silence elapsed, and ``inf`` lets the
    decider's own ``> max_gap_secs`` gate make the call.

    On a flush we record the released text on ``state.flushed_utterances`` (the
    mid-session analog of ``stranded_utterance``) so the summary surfaces it, and
    return that text so the caller can *speak* it as its own turn (iter-169 wires
    ``ChatLoop.respond_to_text`` — the iter-168 text-only response entrypoint — to
    the returned fragment). Returns ``None`` whenever nothing was flushed (any
    guard bailed, the decider held, or the released turn was empty / no-respond),
    so a ``None`` return is the unambiguous "do not speak" signal.
    """
    if aggregator is None or flush_decider is None:
        return None
    held = getattr(aggregator, "pending", None)
    if not held or not held.strip():
        return None
    silence_secs = idle_timeout if idle_timeout is not None else float("inf")
    try:
        do_flush = flush_decider(held, silence_secs)
    except Exception:
        # A misbehaving decider must not break the live loop — treat as HOLD.
        return None
    if not do_flush:
        return None
    try:
        flushed = aggregator.flush()
    except Exception:
        # A misbehaving aggregator must not break the live loop.
        return None
    if flushed is None:
        return None
    from examples._chat_aggregation import resolve_turn
    resolved = resolve_turn(flushed)
    if resolved.respond and resolved.text:
        state.flushed_utterances.append(resolved.text)
        return resolved.text
    return None


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
    idle_timeout: Optional[float] = None,
    flush_decider: Optional[Callable[[str, float], bool]] = None,
    respond_fn: Optional[Callable[[list[dict], str], Any]] = None,
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
        idle_timeout: the pre-speech idle-timeout window (seconds) wired into
            the recorder via `ChatLoop` (iter-166), or None (default). Used
            ONLY as the `silence_secs` fed to `flush_decider` on an
            idle-timeout turn — `run_session` reads no clock itself. When None,
            an idle-timeout turn falls back to `inf` silence (the timeout
            demonstrably fired, so let the decider's own window gate decide).
        flush_decider: injected mid-session flush decision (iter-167), or None
            (default). A callable `(held_text, silence_secs) -> bool` —
            production binds `session.silence_flush.should_flush_held_utterance`
            to the aggregator's config; tests pass a stub. On a recorder
            idle-timeout turn (`TurnResult.idle_timed_out`), if the aggregator
            is holding a mid-thought fragment and this returns True, the
            fragment is flushed mid-session and recorded on
            `state.flushed_utterances`. When None (default), no mid-session
            flush ever happens — byte-for-byte the pre-iter-167 path. Kept
            injected (not imported) so `run_session` stays decoupled from the
            `session` package's eager pipecat import, the same rule
            `trim_messages` follows.
        respond_fn: injected spoken-response entrypoint (iter-169), or None
            (default). A callable `(messages, text) -> result` matching
            `ChatLoop.respond_to_text` (iter-168): it answers `text` as its own
            turn — LLM stream → synth/play, with no mic recording — appending the
            user + assistant messages to `messages` and returning a result with
            `.metrics`/`.had_error`. When a mid-session idle flush releases a
            held fragment AND `respond_fn` is wired, `run_session` *speaks* that
            fragment (counting it as a real turn: metrics printed, `all_metrics`
            appended, turn counter advanced, trim run) instead of only recording
            it on `flushed_utterances`. When None (default), a flushed fragment
            is recorded but not spoken — byte-for-byte the pre-iter-169 path.
            Kept injected (not imported) for the same decoupling reason as
            `flush_decider`.

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

    iter-167 idle-timeout semantics: a no-metrics turn flagged
    `idle_timed_out` (the recorder's pre-speech idle timeout fired)
    increments `idle_timeouts` — counted separately from
    `false_triggers` — and, when a `flush_decider` is wired and the
    aggregator is holding a mid-thought fragment, may flush that
    fragment mid-session (recorded on `flushed_utterances`). Like a
    false trigger / held turn, it re-listens without consuming the
    turn counter.
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
                # A no-metrics turn is now one of THREE distinguishable causes
                # (iter-166's framing):
                #   - iter-166 ``idle_timed_out``: the recorder's pre-speech
                #     idle timeout fired — the user stayed silent for the whole
                #     inter-turn window. A deliberate inter-turn-silence signal,
                #     NOT a misfire. iter-167 consumes it: this is exactly the
                #     moment the iter-164 ``decide_silence_flush`` seam was built
                #     for — flush a held mid-thought fragment now rather than
                #     leaving it held until a new thought displaces it or
                #     shutdown flushes it.
                #   - iter-161 ``held``: the organic aggregator is buffering a
                #     mid-thought utterance for a merge (a successful capture).
                #   - genuine VAD false trigger (recorder fired but no
                #     transcript): neither flag.
                # All three re-listen without consuming the turn counter. The
                # flags are read defensively (getattr) so a pre-iter-166 /
                # pre-iter-161 TurnResult shape still resolves to the false
                # trigger fallback. ``idle_timed_out`` is checked first because
                # it is the only one that can drive a mid-session flush; it and
                # ``held`` are mutually exclusive in ChatLoop (the timeout fires
                # on the empty-wav / no-speech path, ``held`` only after a
                # transcript was captured and offered to the aggregator).
                if getattr(result, "idle_timed_out", False):
                    state.idle_timeouts += 1
                    flushed_text = _maybe_flush_on_idle(
                        state, aggregator, flush_decider, idle_timeout,
                    )
                    # iter-169: if a fragment was flushed AND a respond_fn is
                    # wired, *speak* it as its own turn (the iter-168
                    # ChatLoop.respond_to_text path) — LLM→TTS over the flushed
                    # text, no mic recording. This closes backlog #9's
                    # mid-session flush end-to-end: the trailed-off fragment is
                    # answered, not just listed in flushed_utterances. The spoken
                    # turn is counted exactly like a mic turn (metrics printed,
                    # turn counter advanced, trim run) via the shared
                    # _record_completed_turn helper. When respond_fn is None
                    # (default) the fragment is recorded but not spoken —
                    # byte-for-byte the pre-iter-169 path.
                    if flushed_text and respond_fn is not None:
                        turn = _speak_flushed_fragment(
                            state, respond_fn, flushed_text, turn,
                            log=log,
                            trim_messages=trim_messages,
                            max_user_assistant=max_user_assistant,
                        )
                elif getattr(result, "held", False):
                    state.utterances_held += 1
                else:
                    state.false_triggers += 1
                continue
            turn = _record_completed_turn(
                state, result.metrics, turn,
                log=log,
                trim_messages=trim_messages,
                max_user_assistant=max_user_assistant,
            )
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
