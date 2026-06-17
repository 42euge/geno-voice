"""
Cross-turn utterance aggregator for the organic turn-taking track (the second
half of backlog item #9's live-loop driver in
``docs/research/organic-turn-taking.md``).

iter-156's ``UtteranceBuffer`` is the hold-and-merge *state* a live STT loop
needs, but its ``offer(text, gap_secs)`` requires a value the buffer
deliberately does not own: the **inter-utterance silence gap** — the seconds of
silence between the *previous* utterance's endpoint and *this* one's start. The
buffer (like the ``decide_utterance_continuation`` seam beneath it) keeps that
out of scope on purpose: it reads no clock and holds no timestamp, so the gap
must be measured and injected by the caller.

Every lap since iter-155 named the same next direction — *wire the buffer into
the live STT loop* — and every lap deferred it, because that wiring needs one
more pure piece the buffer doesn't carry: the **cross-turn timestamp state**
that turns "the recorder told me this utterance started at T and the previous
one ended at T0" into the ``gap_secs`` the buffer consumes. This module is that
piece, kept pure and testable so the actual audio loop stays a thin driver:

  - ``UtteranceAggregator`` owns a ``UtteranceBuffer`` plus a single scalar of
    cross-turn state: the previous utterance's *endpoint* timestamp.
  - ``offer(text, speech_start_at, speech_end_at)`` computes
    ``gap_secs = speech_start_at - prev_end_at`` (the silence since the last
    utterance ended), routes ``(text, gap_secs)`` through the buffer, records
    this utterance's ``speech_end_at`` as the new ``prev_end_at``, and returns
    the buffer's turns plus the gap it measured.
  - ``flush()`` releases any held pending (long silence / shutdown) and clears
    the cross-turn state so the next utterance starts a fresh conversation.

The caller still reads no clock *inside* the decision path — it hands the
aggregator the two timestamps the recorder already surfaces
(``record_utterance_streaming`` emits ``speech_start_at`` via ``out_metrics``,
and ``speech_ended_at`` is the recorder's last-speech frame; see
``examples/_chat_loop.py``). The aggregator does the subtraction, which is the
*one* stateful step the buffer can't: it has to remember the prior endpoint.

**The half-duplex invariant is preserved end-to-end.** With a default
``FullDuplexConfig()`` the underlying buffer is a transparent passthrough, so
the aggregator emits every utterance immediately with ``false_endpoint=False``
and never holds — the measured gap is computed and reported but never changes
the output. Byte-for-byte today's "each endpoint is its own turn" behavior,
zero added latency. Only with utterance merging explicitly on does the gap
actually gate a hold-and-merge.

Design follows the GENO.md conventions and the sibling seams: pure (no I/O, no
clock reads — both timestamps injected), an injected optional config (or an
injected buffer for tests), and a small frozen dataclass return so call sites
read clearly. Like its siblings it loads by file path in tests to dodge
``session/__init__``'s eager pipecat import (absent on the x86_64 runner).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from session.full_duplex import FullDuplexConfig
from session.text_eou import TextEOUConfig
from session.utterance_buffer import (
    DEFAULT_MAX_MERGE_DEPTH,
    EmittedTurn,
    UtteranceBuffer,
)
from session.utterance_merging import (
    DEFAULT_INCOMPLETE_CEILING,
    DEFAULT_MAX_GAP_SECS,
)

__all__ = [
    "AggregatedResult",
    "UtteranceAggregator",
]


@dataclass(frozen=True)
class AggregatedResult:
    """Outcome of one ``offer`` / ``flush`` call on the aggregator.

    ``turns`` and ``held`` mirror :class:`session.utterance_buffer.BufferResult`
    — the turns ready to feed the engine now (in order, usually 0 or 1) and the
    text still being held back (``None`` when nothing is pending). ``gap_secs``
    is the inter-utterance silence the aggregator measured for *this* offer:
    ``float('inf')`` for the first utterance after construction or a ``flush``
    (no prior endpoint to measure against), so call sites and tests can see
    exactly what gap drove the buffer's decision. ``flush`` reports
    ``gap_secs=float('inf')`` since it consumes no new utterance.
    """

    turns: list[EmittedTurn] = field(default_factory=list)
    held: str | None = None
    gap_secs: float = float("inf")

    @property
    def merged(self) -> bool:
        """True iff any turn released by this call repaired a false endpoint."""
        return any(t.false_endpoint for t in self.turns)

    @property
    def capped(self) -> bool:
        """True iff any turn released by this call was force-emitted by the
        ``max_merge_depth`` safety cap (iter-163). Mirrors
        :attr:`session.utterance_buffer.BufferResult.capped`.
        """
        return any(t.merge_capped for t in self.turns)


class UtteranceAggregator:
    """Cross-turn driver around :class:`UtteranceBuffer`.

    Owns the buffer and the *one* scalar of state the buffer can't carry: the
    previous utterance's endpoint timestamp. The live STT loop calls
    :meth:`offer` with each finalized transcript and the recorder's measured
    ``speech_start_at`` / ``speech_end_at`` for that utterance; the aggregator
    computes the silence gap, drives the buffer, and returns the turns ready
    for the engine. On a long silence or shutdown the loop calls :meth:`flush`.

    Pure and deterministic: no I/O, no clock reads. The caller injects the two
    timestamps (already measured by the recorder). ``config`` gates the
    behavior (default: a fresh half-duplex ``FullDuplexConfig()`` ⇒ transparent
    passthrough); ``eou_config`` / ``max_gap_secs`` / ``incomplete_ceiling`` /
    ``max_merge_depth`` thread through to the buffer unchanged. A ready-built
    ``buffer`` may be injected instead (tests / advanced wiring); it is mutually
    exclusive with the buffer-construction tuning args.
    """

    def __init__(
        self,
        *,
        config: FullDuplexConfig | None = None,
        eou_config: TextEOUConfig | None = None,
        max_gap_secs: float = DEFAULT_MAX_GAP_SECS,
        incomplete_ceiling: float = DEFAULT_INCOMPLETE_CEILING,
        max_merge_depth: int = DEFAULT_MAX_MERGE_DEPTH,
        buffer: UtteranceBuffer | None = None,
    ) -> None:
        if buffer is not None:
            # An injected buffer brings its own config + tuning; accepting
            # construction args too would silently ignore them, which reads as
            # a bug at the call site. Reject the ambiguous combination loudly.
            if config is not None or eou_config is not None:
                raise ValueError(
                    "pass either a pre-built `buffer` or construction args "
                    "(config / eou_config / tuning), not both"
                )
            self._buffer = buffer
        else:
            self._buffer = UtteranceBuffer(
                config=config,
                eou_config=eou_config,
                max_gap_secs=max_gap_secs,
                incomplete_ceiling=incomplete_ceiling,
                max_merge_depth=max_merge_depth,
            )
        #: Endpoint (speech-end) timestamp of the most recent utterance, or
        #: ``None`` before the first utterance / after a flush. The gap for the
        #: next utterance is ``next.speech_start_at - prev_end_at``.
        self._prev_end_at: float | None = None

    @property
    def buffer(self) -> UtteranceBuffer:
        """The underlying hold-and-merge buffer. Read-only view."""
        return self._buffer

    @property
    def pending(self) -> str | None:
        """The text the buffer is currently holding back, or ``None``."""
        return self._buffer.pending

    @property
    def prev_end_at(self) -> float | None:
        """Endpoint timestamp of the most recent utterance (``None`` when the
        aggregator is idle — before the first utterance or after a flush).
        """
        return self._prev_end_at

    @property
    def active(self) -> bool:
        """True iff merging is on for the underlying buffer (organic mode)."""
        return self._buffer.active

    def offer(
        self,
        text: str,
        speech_start_at: float,
        speech_end_at: float,
    ) -> AggregatedResult:
        """Offer a finalized utterance with its measured speech endpoints.

        ``speech_start_at`` / ``speech_end_at`` are clock timestamps (same
        clock the recorder uses) for when this utterance's speech began and
        ended. The aggregator computes the inter-utterance gap
        ``speech_start_at - prev_end_at`` (the silence since the previous
        utterance ended), routes ``(text, gap)`` through the buffer, then
        records ``speech_end_at`` as the new previous endpoint.

        Returns an :class:`AggregatedResult` with the turns ready for the
        engine now, the text still held, and the gap that drove the decision.
        For the first utterance (no prior endpoint) the gap is
        ``float('inf')`` — there is nothing to merge with, so the buffer treats
        it as a fresh candidate regardless.

        A negative raw gap (the next utterance's start stamped slightly before
        the prior end — clock-skew across the recorder's frame clock) is
        clamped to ``0.0``, mirroring the defensive clamps in
        ``_chat_loop`` (TTC, eot_overhead). A clamped-to-zero gap is "quick",
        which is the correct reading: zero/negative silence is the strongest
        possible mid-thought-pause signal.
        """
        if self._prev_end_at is None:
            gap_secs = float("inf")
        else:
            gap_secs = max(0.0, speech_start_at - self._prev_end_at)

        result = self._buffer.offer(text, gap_secs)
        self._prev_end_at = speech_end_at
        return AggregatedResult(
            turns=result.turns,
            held=result.held,
            gap_secs=gap_secs,
        )

    def flush(self) -> AggregatedResult:
        """Release any held pending and reset the cross-turn state.

        Called when the loop concludes no continuation is coming (a silence
        longer than ``max_gap_secs``, or shutdown). Delegates to the buffer's
        ``flush`` and clears ``prev_end_at`` so the next utterance starts a
        fresh conversation (its gap is ``float('inf')`` ⇒ a genuine new turn).
        In half-duplex mode nothing is ever held, so the turns are empty — but
        the state reset still happens, keeping the two modes structurally
        identical.
        """
        result = self._buffer.flush()
        self._prev_end_at = None
        return AggregatedResult(turns=result.turns, held=result.held)
