"""
Continuer-aware barge-in decision for the organic turn-taking track
(backlog item #5 in ``docs/research/organic-turn-taking.md``).

geno-voice already has substantial barge-in machinery on the ``mic_chat``
path (``BargeInWatcher`` + ``BargeInCoordinator``, iter-009/010; cancel-flush,
iter-026). What it lacks is the **abandon-vs-finish discrimination**: today
*any* barge cancels the agent's turn. But a user "mhmm" / "yeah" / "right"
during agent speech is a **continuer** — it means *keep going, I'm listening*,
not *stop, it's my turn*. Abandoning the turn on a continuer is the wrong call;
it makes the agent feel skittish and clips its own sentences for nothing.

This module is the pure, dependency-free seam that decides — given the barge
transcript, an optional energy signal, and the full-duplex config — whether a
barge should **ABANDON** the agent's turn (a true interruption) or **FINISH**
it (the user only backchanneled). It composes two earlier seams:

  - ``session/backchannel.py`` (#1, iter-148): ``classify_backchannel`` →
    CONTINUER / SUBSTANTIVE / NOT_SPEECH.
  - ``session/full_duplex.py`` (#3, iter-151): ``FullDuplexConfig`` gates the
    behavior off by default.

**The half-duplex invariant is the whole point.** With a default
``FullDuplexConfig()`` (``continuer_aware_listening`` inactive), this function
returns ``ABANDON`` for *every* transcript — byte-for-byte today's
"any barge cancels" behavior. Only when continuer-aware listening is
explicitly turned on does a recognized continuer yield ``FINISH``. So wiring
this into ``BargeInCoordinator`` (a later lap) can never regress the proven
half-duplex path; the new behavior lives entirely behind the off-by-default
switch.

Design follows the GENO.md conventions: a pure function (no I/O, no clock
reads), an injected optional ``energy`` signal kept at the boundary, an
injected config, and a small enum return so call sites read clearly. Like its
sibling seams it loads by file path in tests to dodge ``session/__init__``'s
eager pipecat import (absent on the x86_64 runner).
"""

from __future__ import annotations

from enum import Enum

from session.backchannel import (
    Backchannel,
    DEFAULT_ENERGY_CEILING,
    DEFAULT_MAX_CONTINUER_WORDS,
    classify_backchannel,
)
from session.full_duplex import FullDuplexConfig

__all__ = [
    "BargeAction",
    "decide_barge_action",
    "should_abandon_turn",
]


class BargeAction(Enum):
    """What a barge-in should do to the agent's in-progress turn."""

    #: True interruption — stop speaking, the user is taking the floor.
    ABANDON = "abandon"
    #: The user only backchanneled ("mhmm") — keep speaking, finish the turn.
    FINISH = "finish"


def decide_barge_action(
    transcript: str,
    energy: float | None = None,
    *,
    config: FullDuplexConfig | None = None,
    max_words: int = DEFAULT_MAX_CONTINUER_WORDS,
    energy_ceiling: float = DEFAULT_ENERGY_CEILING,
) -> BargeAction:
    """Decide whether a barge ``ABANDON``s or ``FINISH``es the agent's turn.

    Pure function — no I/O, no clock reads. The ``energy`` signal (a
    normalized RMS in 0.0–1.0) is injected by the caller and kept optional so
    the decision is usable from text-only contexts (tests) and audio-aware
    ones (``pipecat_server`` / the ``mic_chat`` watcher) alike. ``config`` is
    injected (default: a fresh half-duplex ``FullDuplexConfig()``).

    Rules, in order:

      1. **Gate first.** If ``config.continuer_aware_listening_active()`` is
         False (the default), return ``ABANDON`` unconditionally — byte-for-byte
         today's "any barge cancels" behavior. The transcript is not even
         classified; this is the half-duplex invariant.
      2. With continuer-aware listening on, classify the barge transcript:
         a recognized ``CONTINUER`` ⇒ ``FINISH`` (keep speaking). Anything
         else (``SUBSTANTIVE`` real speech, or ``NOT_SPEECH`` empty/noise) ⇒
         ``ABANDON``.

    Rule 2 is deliberately conservative toward ``ABANDON``: only a *confirmed*
    continuer holds the floor. An empty / unrecognized transcript abandons, so
    a misclassification errs on the side of responsiveness (the user who really
    interrupted is never left talking over a droning agent).
    """
    if config is None:
        config = FullDuplexConfig()

    # Rule 1 — the half-duplex invariant. Off by default ⇒ any barge abandons,
    # exactly as today. We don't even classify the transcript in this path.
    if not config.continuer_aware_listening_active():
        return BargeAction.ABANDON

    # Rule 2 — organic mode: only a confirmed continuer keeps the floor.
    kind = classify_backchannel(
        transcript,
        energy,
        max_words=max_words,
        energy_ceiling=energy_ceiling,
    )
    if kind is Backchannel.CONTINUER:
        return BargeAction.FINISH
    return BargeAction.ABANDON


def should_abandon_turn(
    transcript: str,
    energy: float | None = None,
    *,
    config: FullDuplexConfig | None = None,
    max_words: int = DEFAULT_MAX_CONTINUER_WORDS,
    energy_ceiling: float = DEFAULT_ENERGY_CEILING,
) -> bool:
    """Convenience boolean: True iff ``decide_barge_action`` ⇒ ``ABANDON``.

    The natural call-site shape — ``BargeInCoordinator.trigger`` already does
    the abandon work, so a gate reads ``if should_abandon_turn(text, ...):
    coord.trigger()``. With a default (half-duplex) config this is always
    True, so the existing unconditional ``coord.trigger()`` is unchanged.
    """
    return (
        decide_barge_action(
            transcript,
            energy,
            config=config,
            max_words=max_words,
            energy_ceiling=energy_ceiling,
        )
        is BargeAction.ABANDON
    )
