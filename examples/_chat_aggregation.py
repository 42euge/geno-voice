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
    open an LLM stream. When ``True``, ``text`` is the (possibly merged) turn to
    respond to and ``false_endpoint`` is ``True`` iff any released turn repaired
    a false endpoint. ``held`` mirrors the aggregator's still-pending text
    (informational — the loop does not act on it, but a test / live observer
    can see what is being buffered).
    """

    respond: bool
    text: str
    false_endpoint: bool = False
    held: str | None = None


def resolve_turn(result) -> ResolvedTurn:
    """Collapse an aggregator ``result`` into one ``ResolvedTurn``.

    ``result`` is duck-typed: any object with a ``turns`` iterable (each item
    exposing ``.text`` and ``.false_endpoint``) and a ``held`` attribute —
    :class:`session.utterance_aggregator.AggregatedResult` and
    :class:`session.utterance_buffer.BufferResult` both match.

    Rules:
      - No turns ⇒ ``respond=False`` (everything held; re-listen).
      - One or more turns ⇒ join their non-empty texts with single spaces (a
        multi-turn release glues a held pending onto the turn that displaced
        it), ``false_endpoint`` is the OR across all released turns.
      - A release of only blank turns collapses to ``respond=False`` — there is
        nothing to say.
    """
    turns = list(getattr(result, "turns", []) or [])
    held = getattr(result, "held", None)

    if not turns:
        return ResolvedTurn(respond=False, text="", false_endpoint=False, held=held)

    parts = [t.text.strip() for t in turns if t.text and t.text.strip()]
    false_endpoint = any(t.false_endpoint for t in turns)

    if not parts:
        # Defensive: every released turn was blank — nothing to respond to.
        return ResolvedTurn(respond=False, text="", false_endpoint=False, held=held)

    return ResolvedTurn(
        respond=True,
        text=" ".join(parts),
        false_endpoint=false_endpoint,
        held=held,
    )
