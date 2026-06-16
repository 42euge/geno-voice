"""
Backchannel / continuer classifier for the organic turn-taking track.

A *continuer* (a.k.a. backchannel) is a short, low-energy, closed-class
utterance — "mm-hmm", "yeah", "right", "uh-huh", "go on" — that means
*keep going, I'm listening*, NOT *I'm done, your turn*. Conversation-analytic
work and recent turn-taking models (Krisp's audio-only model, pipecat
smart-turn) treat continuers as a first-class signal distinct from a true
turn-end; geno-voice's pipeline currently throws them away (``filter_noise``
in ``session/triggers.py`` discards filler-only utterances as noise).

This module is the dependency-free, fully-testable first step of backlog
item #1 in ``docs/research/organic-turn-taking.md``. It recognizes the
backchannels that ``filter_noise`` would discard as a distinct CONTINUER
class so a later lap can:

  - stop a continuer from being treated as a turn-end (it should reset the
    silence clock, not trigger SPEAK_FULL), and
  - stop a continuer from *abandoning* the agent's turn during barge-in
    (a continuer means "finish"; a substantive interruption means "abandon").

Design follows the GENO.md conventions: a pure function (no I/O, no clock
reads), an injected optional ``energy`` signal kept at the boundary, and a
small enum return so call sites read clearly.
"""

from __future__ import annotations

import re
from enum import Enum

__all__ = [
    "Backchannel",
    "classify_backchannel",
    "is_continuer",
    "CONTINUER_LEXICON",
    "DEFAULT_MAX_CONTINUER_WORDS",
    "DEFAULT_ENERGY_CEILING",
]


class Backchannel(Enum):
    """Result of classifying a transcript chunk for backchannel intent."""

    #: Short, closed-class "keep going" signal — NOT a turn-end.
    CONTINUER = "continuer"
    #: Real, content-bearing speech — may end a turn / warrant a response.
    SUBSTANTIVE = "substantive"
    #: Empty / whitespace — nothing was said.
    NOT_SPEECH = "not_speech"


#: Closed-class continuer lexicon. Each entry is a single "word" after
#: stripping punctuation; multi-word continuers ("go on", "uh huh") are
#: matched as phrases below. Kept lowercase; matching is case-insensitive.
CONTINUER_LEXICON: frozenset[str] = frozenset(
    {
        "mhmm", "mhm", "mmhmm", "mmhm", "mm", "mmm",
        "uhhuh", "uhuh",
        "hmm", "hm", "huh",
        "yeah", "yep", "yup", "yes", "ya", "yah",
        "ok", "okay", "kay",
        "right", "sure", "true",
        "oh", "ah", "aha", "ahh", "ohh",
        "wow", "nice", "cool",
        "i see", "go on", "uh huh", "mm hmm", "for sure", "makes sense",
        "got it", "gotcha", "totally", "exactly", "indeed",
    }
)

#: Continuers are short. More words than this ⇒ it's carrying content.
DEFAULT_MAX_CONTINUER_WORDS: int = 2

#: Optional low-energy gate. When an ``energy`` (normalized RMS, 0.0–1.0) is
#: supplied and exceeds this ceiling, the utterance is loud/emphatic enough
#: that we treat it as substantive even if the words look like a continuer
#: (e.g. an emphatic "YEAH!" that is actually taking the floor). When no
#: energy is supplied the gate is skipped — lexicon + length decide.
DEFAULT_ENERGY_CEILING: float = 0.35

# Strip leading/trailing punctuation and collapse internal whitespace so
# "Mm-hmm." and "mm hmm" and "uh-huh!" all normalize to a lexicon key.
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation (hyphens included), collapse whitespace."""
    text = _PUNCT.sub(" ", text.lower())
    return _WS.sub(" ", text).strip()


def classify_backchannel(
    text: str,
    energy: float | None = None,
    *,
    max_words: int = DEFAULT_MAX_CONTINUER_WORDS,
    energy_ceiling: float = DEFAULT_ENERGY_CEILING,
) -> Backchannel:
    """Classify a transcript chunk as CONTINUER / SUBSTANTIVE / NOT_SPEECH.

    Pure function — no I/O, no clock reads. The ``energy`` signal (a
    normalized RMS in 0.0–1.0, the same scale ``session.compute`` /
    ``mic_*`` use) is injected by the caller and kept optional so the
    classifier is usable from text-only contexts (tests, the
    ``TurnTakingEngine``) and audio-aware ones (``pipecat_server``) alike.

    Rules, in order:
      1. Empty / whitespace ⇒ NOT_SPEECH.
      2. More than ``max_words`` words ⇒ SUBSTANTIVE (continuers are short).
      3. A normalized phrase or every token in the closed continuer lexicon,
         AND (if ``energy`` is given) energy at/under ``energy_ceiling`` ⇒
         CONTINUER.
      4. Otherwise ⇒ SUBSTANTIVE.
    """
    norm = _normalize(text)
    if not norm:
        return Backchannel.NOT_SPEECH

    words = norm.split()
    if len(words) > max_words:
        return Backchannel.SUBSTANTIVE

    # Whole-phrase match (e.g. "i see", "go on") or every token a continuer.
    is_lexical_continuer = norm in CONTINUER_LEXICON or all(
        w in CONTINUER_LEXICON for w in words
    )
    if not is_lexical_continuer:
        return Backchannel.SUBSTANTIVE

    # Loud/emphatic short utterance is taking the floor, not backchanneling.
    if energy is not None and energy > energy_ceiling:
        return Backchannel.SUBSTANTIVE

    return Backchannel.CONTINUER


def is_continuer(
    text: str,
    energy: float | None = None,
    *,
    max_words: int = DEFAULT_MAX_CONTINUER_WORDS,
    energy_ceiling: float = DEFAULT_ENERGY_CEILING,
) -> bool:
    """Convenience boolean: True iff ``classify_backchannel`` ⇒ CONTINUER."""
    return (
        classify_backchannel(
            text,
            energy,
            max_words=max_words,
            energy_ceiling=energy_ceiling,
        )
        is Backchannel.CONTINUER
    )
