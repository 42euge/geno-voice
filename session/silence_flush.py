"""
Mid-session long-silence flush decision for the organic turn-taking track
(the still-deferred half of backlog item #9 in
``docs/research/organic-turn-taking.md``).

iter-156's ``UtteranceBuffer`` holds a mid-thought fragment ("I was thinking
about the") waiting for a quick continuation to merge on. But the buffer has a
blind spot the live loop keeps tripping over: it only ever *releases* a held
pending when the **next utterance arrives** — its ``offer`` measures the gap
that preceded that next utterance and, if the gap was long, releases the held
fragment as a ``NEW`` (displaced) turn (iter-162). The one case ``offer`` can
never reach is the user who trails off mid-thought and then says **nothing at
all** for a long beat. There is no next utterance to drive a release, so the
fragment sits held — fed to the engine only when a genuinely-new thought finally
arrives and displaces it (iter-162), or at shutdown via ``flush`` (iter-160).
Both are too late: the user paused, waited, and the agent stayed mute on a
fragment it could have answered.

Every lap since iter-160 has named the same next direction:

  > **Mid-session long-silence flush** — also flush on a long *inter-turn*
  > silence so a trailed-off fragment is fed to the engine as its own turn
  > before the user starts a genuinely new thought. Needs ``run_session`` to
  > measure the inter-turn gap (today it reads no clock between turns).

This module is the **pure decision** that wiring needs — shipped first, the way
this track always ships a behavior (the decision seam, then the live wiring as a
separate lap: iter-152 ``decide_barge_action`` → coordinator wiring,
iter-153 ``decide_backchannel_timing`` → cue-path wiring). It answers exactly
one question: *the buffer is holding a mid-thought fragment and this much
silence has elapsed with no continuation — should the loop give up waiting and*
**FLUSH** *the fragment to the engine now, or keep* **HOLD**-ing?

**The gap axis is partitioned to agree with the merge window.** A continuation
arriving within ``max_gap_secs`` (default 2.0s) of the prior endpoint still
``MERGE`` (``decide_utterance_continuation``'s rule 3). So a flush must not
fire until that window has *elapsed* — otherwise it would emit a fragment a beat
before the continuation that would have completed it. The boundary therefore
mirrors the merge gate exactly: ``silence_secs <= max_gap_secs`` is still
"within the window" (``HOLD``), and only ``silence_secs > max_gap_secs`` is
"the window closed, no continuation came" (``FLUSH``). The two seams read the
same scalar (``max_gap_secs``) so the merge window and the flush deadline can
never drift apart.

**The half-duplex invariant is the whole point.** With a default
``FullDuplexConfig()`` (``utterance_merging`` inactive), this function returns
``HOLD`` for *every* input — and in that mode the buffer never holds anything
anyway, so there is nothing to flush. Byte-for-byte today's behavior. Only when
utterance merging is explicitly on can a held fragment exist, and only then can a
long silence flush it. So wiring this into ``run_session``'s inter-turn path (a
later lap) can never regress the proven half-duplex path; the new behavior lives
entirely behind the off-by-default switch.

Design follows the GENO.md conventions and its sibling seams
(#1/#3/#4/#5/#7/#9): a pure function (no I/O, no clock reads — the caller
injects the already-measured ``silence_secs``), an injected optional config, and
a small enum return so call sites read clearly. Like its siblings it loads by
file path in tests to dodge ``session/__init__``'s eager pipecat import (absent
on the x86_64 runner).
"""

from __future__ import annotations

from enum import Enum

from session.full_duplex import FullDuplexConfig
from session.utterance_merging import DEFAULT_MAX_GAP_SECS

__all__ = [
    "SilenceFlushAction",
    "decide_silence_flush",
    "should_flush_held_utterance",
]


class SilenceFlushAction(Enum):
    """What to do with a held mid-thought fragment after a beat of silence."""

    #: The merge window elapsed with no continuation — give up waiting and emit
    #: the held fragment to the engine now as its own (false-endpoint) turn,
    #: rather than letting it sit held until a new thought displaces it or
    #: shutdown flushes it.
    FLUSH = "flush"
    #: Keep holding — either the silence is still within the merge window (a
    #: continuation could yet arrive), or merging is off / nothing is held.
    HOLD = "hold"


def decide_silence_flush(
    *,
    held_text: str,
    silence_secs: float,
    config: FullDuplexConfig | None = None,
    max_gap_secs: float = DEFAULT_MAX_GAP_SECS,
) -> SilenceFlushAction:
    """Decide whether to ``FLUSH`` a held mid-thought fragment after silence.

    Pure function — no I/O, no clock reads. The caller injects the
    already-measured ``silence_secs`` (the inter-turn silence elapsed since the
    held fragment was buffered, with no new utterance yet). ``config`` gates the
    behavior (default: a fresh half-duplex ``FullDuplexConfig()``);
    ``max_gap_secs`` is the merge window (default ``DEFAULT_MAX_GAP_SECS``,
    shared with ``decide_utterance_continuation`` so the flush deadline and the
    merge window are the same scalar).

    Rules, in order (first match wins):

      1. **Gate first.** If ``config.utterance_merging_active()`` is False
         (the default), return ``HOLD`` unconditionally — half-duplex never
         holds a fragment, so there is nothing to flush. This is the
         half-duplex invariant; no other input is consulted.
      2. **Nothing held ⇒ HOLD.** A blank ``held_text`` means the buffer is
         idle (no pending). There is nothing to flush.
      3. **Window gate.** ``FLUSH`` iff the silence has *exceeded* the merge
         window (``silence_secs > max_gap_secs``) — the continuation window
         closed and none arrived, so the fragment is an abandoned-but-complete
         thought to answer now. At or below the window (``silence_secs <=
         max_gap_secs``) a continuation could still merge, so ``HOLD``.

    Rule 3's boundary deliberately matches ``decide_utterance_continuation``'s
    rule 3 (``gap_secs <= max_gap_secs`` is "quick" ⇒ eligible to ``MERGE``):
    at exactly ``max_gap_secs`` a continuation would still merge, so we must not
    have flushed yet. Only strictly beyond it is the window provably closed.
    """
    if config is None:
        config = FullDuplexConfig()

    # Rule 1 — the half-duplex invariant. Off by default ⇒ nothing is ever
    # held, so there is nothing to flush. No other signal is consulted.
    if not config.utterance_merging_active():
        return SilenceFlushAction.HOLD

    # Rule 2 — nothing held to flush.
    if not held_text or not held_text.strip():
        return SilenceFlushAction.HOLD

    # Rule 3 — the merge window has elapsed with no continuation.
    if silence_secs > max_gap_secs:
        return SilenceFlushAction.FLUSH
    return SilenceFlushAction.HOLD


def should_flush_held_utterance(
    *,
    held_text: str,
    silence_secs: float,
    config: FullDuplexConfig | None = None,
    max_gap_secs: float = DEFAULT_MAX_GAP_SECS,
) -> bool:
    """Convenience boolean: True iff ``decide_silence_flush`` ⇒ ``FLUSH``.

    The natural call-site shape — a live ``run_session`` measuring the
    inter-turn silence would read ``if should_flush_held_utterance(...):
    flushed = aggregator.flush(); respond_to(flushed)`` before re-listening.
    With a default (half-duplex) config this is always False, so the existing
    flow is unchanged until utterance merging is explicitly enabled.
    """
    return (
        decide_silence_flush(
            held_text=held_text,
            silence_secs=silence_secs,
            config=config,
            max_gap_secs=max_gap_secs,
        )
        is SilenceFlushAction.FLUSH
    )
