"""Single-turn collapse of an ``UtteranceAggregator`` result — the pure seam
that lets the synchronous ``ChatLoop`` consume the organic aggregator (iter-158).

``ChatLoop.run_one_turn`` records *one* utterance and responds to it: a single
in / single out shape. ``UtteranceAggregator.offer`` (the backlog #9 live-loop
driver) does not match that shape — in organic mode it may

  - **hold** the utterance (return zero turns, waiting for a mid-thought
    continuation to merge on), or
  - release **several** turns at once (a held pending plus a fresh complete
    turn that displaced it).

``resolve_turn`` folds that variable-length ``AggregatedResult`` into the one
decision the single-turn loop needs: respond to nothing this cycle (everything
is held — the loop should re-listen, treating it like a false trigger), or
respond to this text now, carrying the false-endpoint flag forward to
``TurnMetrics.false_endpoint`` (iter-154's metric, finally populated from the
live path).

Why a separate pure helper rather than inlining the fold in ``run_one_turn``?
The track's rhythm (iter-155→158): keep the genuinely policy-laden decision
pure and exhaustively testable, so the entrypoint wiring stays a thin consumer.
The multi-turn-join + all-empty-collapse + held-passthrough rules are exactly
the kind of corner logic that earns isolated tests. And keeping this module
dependency-free (it duck-types over ``.turns`` / ``.held`` rather than importing
the aggregator) means it loads without dragging in ``session/__init__``'s eager
pipecat import — the same trap the sibling seams dodge.

**Half-duplex is unaffected.** With merging off the aggregator always returns
exactly one turn and never holds, so ``resolve_turn`` returns
``respond=True, text=<that turn>, false_endpoint=False`` every time — byte-for-
byte the pre-aggregator single-utterance behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ResolvedTurn", "resolve_turn"]


@dataclass(frozen=True)
class ResolvedTurn:
    """The single-turn decision distilled from an ``AggregatedResult``.

    ``respond`` is ``False`` when the aggregator held everything this cycle
    (nothing ready to feed the engine) — the loop should re-listen rather than
    open an LLM stream. When ``True``, ``text`` is the turn to respond to *now*
    (the most-recently-released one) and ``false_endpoint`` is ``True`` iff that
    responded turn repaired a false endpoint. ``held`` mirrors the aggregator's
    still-pending text (informational — the loop does not act on it, but a test
    / live observer can see what is being buffered).

    ``displaced`` (iter-162) holds any *earlier* released turns from a
    multi-turn release, in order — the mid-thought fragments the user abandoned
    when a long silence proved they were *not* a false endpoint and a genuinely
    new thought began. They are the **mid-session analog of iter-160's
    shutdown** ``stranded_utterance``: captured fine, never completed, and *not*
    part of the text the loop responds to. A single-turn release (the common
    case) leaves it empty.

    ``merge_capped`` (iter-163) is ``True`` iff the *responded* turn was
    force-emitted by the ``max_merge_depth`` safety cap (iter-157) rather than
    released naturally — the held pending stayed unfinished-looking through
    ``max_merge_depth`` merges and the buffer cut it loose to avoid starving the
    engine. Always paired with ``false_endpoint=True`` (a capped turn absorbed
    real merges), but distinct: the text still looked mid-thought when emitted,
    so the loop surfaces it as a separate signal rather than a clean repair. The
    flag mirrors the *responded* (last) turn only — an earlier displaced
    fragment that happened to be capped does not stamp the fresh response.
    """

    respond: bool
    text: str
    false_endpoint: bool = False
    held: str | None = None
    displaced: tuple[str, ...] = ()
    merge_capped: bool = False


def resolve_turn(result) -> ResolvedTurn:
    """Collapse an aggregator ``result`` into one ``ResolvedTurn``.

    ``result`` is duck-typed: any object with a ``turns`` iterable (each item
    exposing ``.text`` and ``.false_endpoint``) and a ``held`` attribute —
    :class:`session.utterance_aggregator.AggregatedResult` and
    :class:`session.utterance_buffer.BufferResult` both match.

    Rules:
      - No turns ⇒ ``respond=False`` (everything held; re-listen).
      - **Respond to the LAST released turn.** A multi-turn release is *never* a
        continuation to glue together: :class:`UtteranceBuffer` only ever emits
        more than one turn when a measured silence forced a ``NEW`` boundary
        (the held pending released as its own turn) *and then* a fresh utterance
        was emitted. Those are semantically distinct turns — the earlier ones
        are abandoned mid-thought fragments, the last is the new thing to answer
        now. iter-162 fixes the pre-existing bug where these were space-glued
        into one garbled LLM input (``"I was thinking about the What time is
        it?"``). ``false_endpoint`` and ``merge_capped`` are the responded
        (last) turn's own flags.
      - Earlier released turns become ``displaced`` — surfaced like iter-160's
        ``stranded_utterance`` rather than fed to the engine.
      - A release of only blank turns collapses to ``respond=False`` — there is
        nothing to say.
    """
    turns = list(getattr(result, "turns", []) or [])
    held = getattr(result, "held", None)

    if not turns:
        return ResolvedTurn(respond=False, text="", false_endpoint=False, held=held)

    # Keep each turn paired with its own false-endpoint + merge-capped flags so
    # the responded turn carries *its* flags (not an OR across abandoned
    # fragments). ``merge_capped`` is read defensively (getattr) so a
    # pre-iter-163 EmittedTurn shape without the field resolves to False.
    nonblank = [
        (t.text.strip(), bool(t.false_endpoint), bool(getattr(t, "merge_capped", False)))
        for t in turns
        if t.text and t.text.strip()
    ]

    if not nonblank:
        # Defensive: every released turn was blank — nothing to respond to.
        return ResolvedTurn(respond=False, text="", false_endpoint=False, held=held)

    *earlier, (text, false_endpoint, merge_capped) = nonblank
    return ResolvedTurn(
        respond=True,
        text=text,
        false_endpoint=false_endpoint,
        held=held,
        displaced=tuple(t for t, _, _ in earlier),
        merge_capped=merge_capped,
    )
