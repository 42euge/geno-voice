"""
Utterance buffer-merge decision for the organic turn-taking track
(backlog item #4's second half in ``docs/research/organic-turn-taking.md``).

Section 4 of the research ("Utterance queueing / interruption") names two
halves. The *abandon-vs-finish* discrimination — does a barge during agent
speech interrupt or merely backchannel? — shipped as ``barge_decision.py``
(#5, iter-152). This module is the **other** half: **buffer/merge partial
utterances** that arrive on the *user* side.

The problem it solves is the classic silence-only false endpoint. The VAD
closes a window on trailing silence, the STT finalizes "I was thinking about
the", and the engine treats that as a finished turn — but the user only paused
mid-thought ("…the deadline" lands a beat later). Today those become two
separate turns; the first one fires a premature response. A human listener
hears "about the" and *knows* more is coming, because the syntax is obviously
unfinished. That linguistic signal is exactly what ``text_eou.py`` (#4,
iter-150) already computes.

This module is the pure, dependency-free seam that decides — given the just-
endpointed text, the next utterance's text, the silence gap between them, and
the full-duplex config — whether the two should **MERGE** into one turn (the
first endpoint was a false positive) or stay as a **NEW** turn (a genuine new
utterance). It composes two earlier seams:

  - ``session/text_eou.py`` (#4, iter-150): ``utterance_completeness`` →
    a (0.0, 1.0] multiplier; low ⇒ the prior text trailed off unfinished.
  - ``session/full_duplex.py`` (#3, iter-151): ``FullDuplexConfig`` gates the
    behavior off by default.

**The half-duplex invariant is the whole point.** With a default
``FullDuplexConfig()`` (``utterance_merging`` inactive), this function returns
``NEW`` for *every* input — byte-for-byte today's "each endpoint is its own
turn" behavior. The prior text is not even scored in that path. Only when
utterance merging is explicitly turned on does an unfinished-looking prior
utterance followed by a quick continuation yield ``MERGE``. So wiring this into
the live STT/turn path (a later lap) can never regress the proven half-duplex
path; the new behavior lives entirely behind the off-by-default switch.

**The two gates (organic mode only).** A merge requires *both*:

  1. **A short gap.** The continuation must arrive within ``max_gap_secs``
     (default 2.0s — the ``turn_decider`` ``silence_floor_secs``, "a pause,
     not a turn-end"). A long gap means the user really did stop and the next
     utterance is a fresh turn, regardless of how unfinished the first looked.
  2. **An unfinished prior.** ``utterance_completeness(prev_text)`` must be at
     or below ``incomplete_ceiling`` (default the ``text_eou``
     ``complete_threshold``, 0.6). A prior that already looked complete ("I'm
     done.") is a real turn-end even if the user spoke again quickly — that's
     a new thought, not a continuation.

Both gates must hold: a quick gap after a *complete* sentence is a new turn,
and an unfinished prior after a *long* gap is an abandoned thought. Only the
unfinished-AND-quick corner is a false endpoint to repair. Note the asymmetry
versus ``barge_decision``: there the conservative default is ABANDON (stay
responsive); here it is NEW (don't glue unrelated turns together). Both err
toward today's behavior on ambiguity.

Design follows the GENO.md conventions and its sibling seams (#1/#3/#4/#5/#7):
a pure function (no I/O, no clock reads — the caller injects the already-
measured ``gap_secs``), an injected optional config, an injected ``text_eou``
config, and a small enum return so call sites read clearly. Like its siblings
it loads by file path in tests to dodge ``session/__init__``'s eager pipecat
import (absent on the x86_64 runner).
"""

from __future__ import annotations

from enum import Enum

from session.full_duplex import FullDuplexConfig
from session.text_eou import (
    DEFAULT_COMPLETE_THRESHOLD,
    TextEOUConfig,
    utterance_completeness,
)

__all__ = [
    "UtteranceAction",
    "DEFAULT_MAX_GAP_SECS",
    "DEFAULT_INCOMPLETE_CEILING",
    "decide_utterance_continuation",
    "should_merge_utterance",
]

#: A continuation arriving within this many seconds of the prior endpoint is
#: "quick" — short enough to be a mid-thought pause rather than a real
#: turn-end. Mirrors ``turn_decider.silence_floor_secs`` (2.0s, "a pause, not
#: a turn-end") so the merge window and the confidence ramp agree on what a
#: pause is.
DEFAULT_MAX_GAP_SECS: float = 2.0

#: ``utterance_completeness(prev_text)`` at or below this is "unfinished
#: enough" to be a merge candidate. Mirrors ``text_eou``'s
#: ``DEFAULT_COMPLETE_THRESHOLD`` (0.6) so the boolean ``is_utterance_complete``
#: and this gate partition completeness at the same point: a prior the boolean
#: calls *incomplete* is exactly a merge candidate.
DEFAULT_INCOMPLETE_CEILING: float = DEFAULT_COMPLETE_THRESHOLD


class UtteranceAction(Enum):
    """What to do with a freshly-endpointed utterance and its follow-on."""

    #: The prior endpoint was a false positive — the user paused mid-thought
    #: and resumed. Glue the continuation onto the prior text as one turn.
    MERGE = "merge"
    #: A genuine new turn — the prior utterance was complete, or the gap was
    #: long enough to be a real turn-end. Keep them separate (today's
    #: behavior).
    NEW = "new"


def decide_utterance_continuation(
    prev_text: str,
    next_text: str,
    gap_secs: float,
    *,
    config: FullDuplexConfig | None = None,
    eou_config: TextEOUConfig | None = None,
    max_gap_secs: float = DEFAULT_MAX_GAP_SECS,
    incomplete_ceiling: float = DEFAULT_INCOMPLETE_CEILING,
) -> UtteranceAction:
    """Decide whether ``next_text`` ``MERGE``s with ``prev_text`` or is a ``NEW`` turn.

    Pure function — no I/O, no clock reads. ``gap_secs`` (the measured silence
    between the prior endpoint and this continuation) is injected by the
    caller. ``config`` is injected (default: a fresh half-duplex
    ``FullDuplexConfig()``); ``eou_config`` tunes the completeness scorer.

    Rules, in order:

      1. **Gate first.** If ``config.utterance_merging_active()`` is False
         (the default), return ``NEW`` unconditionally — byte-for-byte today's
         "each endpoint is its own turn" behavior. The prior text is not even
         scored; this is the half-duplex invariant.
      2. **Empty continuation ⇒ NEW.** A blank ``next_text`` is nothing to
         merge (and would be dropped as noise upstream); there is no
         continuation, so the prior stands as its own turn.
      3. **Both gates.** With merging on and a real continuation, ``MERGE``
         iff the gap is quick (``gap_secs <= max_gap_secs``) *and* the prior
         text looks unfinished (``utterance_completeness(prev_text) <=
         incomplete_ceiling``). Otherwise ``NEW``.

    Rule 3 is deliberately conservative toward ``NEW``: a merge only fires in
    the unfinished-AND-quick corner. A quick gap after a complete sentence, or
    an unfinished prior after a long gap, both stay ``NEW`` — so a
    misjudgment errs on the side of *not* gluing two real turns together (the
    user who started a genuinely new thought is never trapped behind a stale
    fragment).
    """
    if config is None:
        config = FullDuplexConfig()

    # Rule 1 — the half-duplex invariant. Off by default ⇒ every endpoint is
    # its own turn, exactly as today. We don't even score the prior text here.
    if not config.utterance_merging_active():
        return UtteranceAction.NEW

    # Rule 2 — no continuation to merge.
    if not next_text or not next_text.strip():
        return UtteranceAction.NEW

    # Rule 3 — organic mode: merge only the unfinished-AND-quick corner.
    if gap_secs > max_gap_secs:
        return UtteranceAction.NEW

    completeness = utterance_completeness(prev_text, eou_config)
    if completeness <= incomplete_ceiling:
        return UtteranceAction.MERGE
    return UtteranceAction.NEW


def should_merge_utterance(
    prev_text: str,
    next_text: str,
    gap_secs: float,
    *,
    config: FullDuplexConfig | None = None,
    eou_config: TextEOUConfig | None = None,
    max_gap_secs: float = DEFAULT_MAX_GAP_SECS,
    incomplete_ceiling: float = DEFAULT_INCOMPLETE_CEILING,
) -> bool:
    """Convenience boolean: True iff ``decide_utterance_continuation`` ⇒ ``MERGE``.

    The natural call-site shape — a live STT loop holding the just-finalized
    ``prev_text`` would read ``if should_merge_utterance(prev, nxt, gap, ...):
    prev = prev + " " + nxt`` before feeding the turn engine. With a default
    (half-duplex) config this is always False, so the existing "each utterance
    is a turn" flow is unchanged.
    """
    return (
        decide_utterance_continuation(
            prev_text,
            next_text,
            gap_secs,
            config=config,
            eou_config=eou_config,
            max_gap_secs=max_gap_secs,
            incomplete_ceiling=incomplete_ceiling,
        )
        is UtteranceAction.MERGE
    )
