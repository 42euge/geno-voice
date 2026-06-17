"""
Stateful utterance buffer-merge coordinator for the organic turn-taking track
(the live-loop driver for backlog item #9 in
``docs/research/organic-turn-taking.md``).

Backlog #9 shipped the *pure decision* in ``utterance_merging.py`` (iter-155):
``decide_utterance_continuation(prev_text, next_text, gap_secs)`` answers
"should these two endpointed utterances ``MERGE`` into one turn, or stay a
``NEW`` turn?" — but it is stateless. Every lap since named the same follow-on:
**wire it into the live STT loop** — "hold the just-finalized text + its
silence gap, merge the next chunk before feeding the turn engine, and set
``TurnMetrics.false_endpoint`` when a merge fires."

That follow-on needs *state* the decision seam deliberately does not carry: a
held "pending" utterance, the running merged text, and whether the eventual
turn repaired a false endpoint. This module is that state, kept pure and
testable so the actual audio loop stays a thin driver:

  - ``UtteranceBuffer`` accumulates finalized STT text. The caller injects the
    measured silence ``gap_secs`` (no clock reads here, same as the seam), and
    the buffer composes ``decide_utterance_continuation`` to decide whether each
    new chunk merges with what it is holding.
  - ``offer(text, gap_secs)`` returns the turns now ready for the engine (0+)
    plus the text still being held. ``flush()`` releases a held pending when no
    continuation arrives (the user trailed off and stopped) or on shutdown.
  - Each emitted turn carries a ``false_endpoint`` flag: ``True`` iff it
    absorbed a merged continuation, so the caller sets
    ``TurnMetrics.false_endpoint`` and iter-154's metric finally populates from
    the live path.

**The half-duplex invariant is the whole point.** With a default
``FullDuplexConfig()`` (``utterance_merging`` inactive) the buffer is a
*transparent passthrough*: every ``offer`` emits its text immediately with
``false_endpoint=False``, nothing is ever held, ``flush`` is always empty.
Byte-for-byte today's "each endpoint is its own turn, fed at once" behavior,
zero added latency. Only when merging is explicitly on does the hold-and-merge
machinery engage — and even then only an *unfinished-looking* utterance is
held (a complete thought emits immediately), so complete turns never pay the
latency of waiting for a continuation.

Design follows the GENO.md conventions and the sibling seams: pure (no I/O, no
clock reads — ``gap_secs`` injected), an injected optional config, a small
dataclass return so call sites read clearly. Like its siblings it loads by file
path in tests to dodge ``session/__init__``'s eager pipecat import (absent on
the x86_64 runner).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from session.full_duplex import FullDuplexConfig
from session.text_eou import TextEOUConfig, utterance_completeness
from session.utterance_merging import (
    DEFAULT_INCOMPLETE_CEILING,
    DEFAULT_MAX_GAP_SECS,
    UtteranceAction,
    decide_utterance_continuation,
)

__all__ = [
    "EmittedTurn",
    "BufferResult",
    "UtteranceBuffer",
]


@dataclass(frozen=True)
class EmittedTurn:
    """One finished turn the buffer has released to the engine.

    ``false_endpoint`` is ``True`` iff this turn absorbed at least one merged
    continuation — i.e. an earlier silence endpoint was a false positive that
    the buffer repaired by gluing the next chunk on. The caller sets
    ``TurnMetrics.false_endpoint = turn.false_endpoint`` so iter-154's metric
    populates from the live organic path.
    """

    text: str
    false_endpoint: bool = False


@dataclass(frozen=True)
class BufferResult:
    """Outcome of one ``offer`` / ``flush`` call.

    ``turns`` are ready to feed the engine *now*, in order (usually 0 or 1).
    ``held`` is the text the buffer is still holding back, waiting for a
    possible continuation (``None`` when nothing is pending). ``held`` is
    purely informational — the caller does not act on it; it exists so a live
    loop / test can observe what is being buffered.
    """

    turns: list[EmittedTurn] = field(default_factory=list)
    held: str | None = None

    @property
    def merged(self) -> bool:
        """True iff any turn released by this call repaired a false endpoint."""
        return any(t.false_endpoint for t in self.turns)


def _join(prev: str, nxt: str) -> str:
    """Glue a merged continuation onto the prior text with a single space.

    Both sides are stripped of surrounding whitespace so the join never
    produces double spaces or a leading/trailing gap, regardless of how the
    STT chunks were spaced.
    """
    return f"{prev.strip()} {nxt.strip()}".strip()


class UtteranceBuffer:
    """Hold-and-merge accumulator over finalized STT utterances.

    The live STT loop calls :meth:`offer` with each finalized transcript and
    the silence ``gap_secs`` that preceded it, then feeds the returned
    :class:`EmittedTurn` texts to the turn engine. When the loop decides no
    continuation is coming (a long silence, or shutdown) it calls
    :meth:`flush` to release whatever is held.

    Pure and deterministic: no I/O, no clock reads. The caller owns timing and
    injects ``gap_secs``. ``config`` gates the behavior (default: a fresh
    half-duplex ``FullDuplexConfig()`` ⇒ transparent passthrough); ``eou_config``
    tunes the completeness scorer; ``max_gap_secs`` / ``incomplete_ceiling``
    thread through to :func:`decide_utterance_continuation` unchanged.
    """

    def __init__(
        self,
        *,
        config: FullDuplexConfig | None = None,
        eou_config: TextEOUConfig | None = None,
        max_gap_secs: float = DEFAULT_MAX_GAP_SECS,
        incomplete_ceiling: float = DEFAULT_INCOMPLETE_CEILING,
    ) -> None:
        self._config = config if config is not None else FullDuplexConfig()
        self._eou_config = eou_config
        self._max_gap_secs = max_gap_secs
        self._incomplete_ceiling = incomplete_ceiling
        #: The utterance currently held back (None when nothing pending).
        self._pending: str | None = None
        #: Did the current pending result from at least one merge? Travels with
        #: the pending so the eventual EmittedTurn carries the right flag even
        #: when the turn is released laps later (via flush or a NEW arrival).
        self._pending_merged: bool = False

    @property
    def pending(self) -> str | None:
        """The text currently held back, or ``None``. Read-only view."""
        return self._pending

    @property
    def active(self) -> bool:
        """True iff merging is on for this buffer (organic mode)."""
        return self._config.utterance_merging_active()

    def offer(self, text: str, gap_secs: float) -> BufferResult:
        """Offer a finalized utterance + the silence gap that preceded it.

        Returns a :class:`BufferResult` with the turns ready for the engine now
        and the text (if any) still held. ``gap_secs`` is the measured silence
        between the *previous* endpoint and this one.

        Half-duplex (the default): emit ``text`` immediately, never hold —
        byte-for-byte today's behavior. Organic mode: decide ``MERGE`` / ``NEW``
        against any pending, then hold the result iff it still looks unfinished
        (a complete thought emits immediately).
        """
        text = text or ""

        # Half-duplex invariant — transparent passthrough, zero latency.
        if not self._config.utterance_merging_active():
            return BufferResult(turns=[EmittedTurn(text, False)], held=None)

        turns: list[EmittedTurn] = []

        if self._pending is None:
            candidate = text
            candidate_merged = False
        else:
            action = decide_utterance_continuation(
                self._pending,
                text,
                gap_secs,
                config=self._config,
                eou_config=self._eou_config,
                max_gap_secs=self._max_gap_secs,
                incomplete_ceiling=self._incomplete_ceiling,
            )
            if action is UtteranceAction.MERGE:
                # The prior endpoint was a false positive — glue on the
                # continuation and keep the running text as the new candidate.
                candidate = _join(self._pending, text)
                candidate_merged = True
            else:
                # A genuine new turn — release the pending, start fresh.
                turns.append(EmittedTurn(self._pending, self._pending_merged))
                candidate = text
                candidate_merged = False
            self._pending = None
            self._pending_merged = False

        # Decide whether to hold the candidate (might still get a continuation)
        # or emit it now (a complete thought has nothing to wait for).
        if not candidate.strip():
            # Nothing meaningful to hold or emit (e.g. an empty/noise chunk
            # that flushed a pending NEW above).
            return BufferResult(turns=turns, held=None)

        completeness = utterance_completeness(candidate, self._eou_config)
        if completeness <= self._incomplete_ceiling:
            self._pending = candidate
            self._pending_merged = candidate_merged
            return BufferResult(turns=turns, held=self._pending)

        turns.append(EmittedTurn(candidate, candidate_merged))
        return BufferResult(turns=turns, held=None)

    def flush(self) -> BufferResult:
        """Release any held pending as a finished turn.

        Called when the loop concludes no continuation is coming (a silence
        longer than ``max_gap_secs``, or shutdown). In half-duplex mode nothing
        is ever held, so this is always an empty result.
        """
        if self._pending is None:
            return BufferResult(turns=[], held=None)
        turn = EmittedTurn(self._pending, self._pending_merged)
        self._pending = None
        self._pending_merged = False
        return BufferResult(turns=[turn], held=None)
