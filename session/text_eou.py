"""
Rule-based text end-of-turn (EOU) precursor for the organic turn-taking track
(backlog item #4 in ``docs/research/organic-turn-taking.md``).

Pure-silence endpointing can't tell a finished turn from a mid-thought pause:
"I was thinking… [2s] …about the deadline" and "That's it. [2s]" look identical
to a VAD. *Linguistic* context disambiguates them — an utterance that trails off
on a conjunction ("…and"), a dangling preposition ("…to"), a bare article
("…the"), or a filler ("…um") is almost certainly **incomplete**, no matter how
long the silence after it. This is the signal LiveKit's ``turn-detector`` learns
from a transcript LM; here it's a cheap, dependency-free, fully-testable rule
set — a precursor that a learned model (backlog #6) can later replace behind the
same interface.

The output *feeds* the turn-decider seam (backlog #2): ``utterance_completeness``
returns a multiplier in [0.0, 1.0] that **dampens** the silence-derived
confidence when the transcript looks unfinished, so the engine stays silent
through a trailing-off pause instead of barging in. A syntactically complete
utterance returns 1.0 — no dampening — so wiring this in is a conservative,
monotone change: it can only *lower* confidence on evidence of incompleteness,
never raise it.

This mirrors (and extends) ``session/triggers.py``'s ``_TRAILING_PATTERNS``,
which already encodes a few "trailing off" markers ("…", "but yeah", "so") for
the emotional PLAY_CUE path. We don't import it: ``import session.triggers``
runs ``session/__init__`` which eagerly pulls pipecat (absent on the x86_64
runner), and this module — like ``backchannel.py`` / ``turn_decider.py`` — stays
pure stdlib so it loads by file path in the unit suite. The trailing-off regexes
below are the EOU-framed superset of that idea (incomplete ⇒ *more* coming,
rather than emotionally spent).

Design follows the GENO.md conventions: a pure function (no I/O, no clock
reads — the transcript is injected), module-level marker sets, a small frozen
config for the tunables, and a thin decider class implementing the identical
``confidence(...)`` interface as ``SilenceTurnDecider`` so call sites swap with
no signature change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from session.turn_decider import (
    TurnDeciderConfig,
    silence_confidence,
)

__all__ = [
    "TextEOUConfig",
    "utterance_completeness",
    "is_utterance_complete",
    "TextAwareTurnDecider",
    "CONJUNCTION_MARKERS",
    "DANGLING_MARKERS",
    "FILLER_MARKERS",
    "DEFAULT_COMPLETE_THRESHOLD",
]


#: Coordinating / subordinating conjunctions. Ending on one of these is the
#: strongest text signal of "more is coming" — "…and", "…because", "…but".
CONJUNCTION_MARKERS: frozenset[str] = frozenset(
    {
        "and", "but", "or", "nor", "so", "yet",
        "because", "cause", "since", "as",
        "while", "whilst", "although", "though", "if", "unless",
        "until", "till", "whether", "that", "which", "who", "whom",
        "whose", "where", "when", "before", "after",
    }
)

#: Dangling function words — prepositions, articles, possessives. A noun (the
#: object) is expected to follow: "…to", "…the", "…with my". Demonstratives and
#: quantifiers (this/that/these/those/some/any) are deliberately excluded — they
#: are frequently *complete* sentence-final pronouns ("I did this", "I want
#: some"), so flagging them would dampen finished turns. ("that" is a relative
#: pronoun in CONJUNCTION_MARKERS, a separate, stronger signal.)
DANGLING_MARKERS: frozenset[str] = frozenset(
    {
        # prepositions
        "to", "for", "with", "of", "at", "in", "on", "from", "by",
        "about", "into", "onto", "over", "under", "between", "through",
        "without", "within", "toward", "towards", "upon",
        # articles / possessives (a noun must follow)
        "the", "a", "an",
        "my", "your", "his", "her", "their", "our", "its",
    }
)

#: Hesitation fillers. Ending on one means the speaker is still composing —
#: "…um", "…like", "…you know".
FILLER_MARKERS: frozenset[str] = frozenset(
    {
        "um", "umm", "uh", "uhh", "er", "erm", "hmm", "hm",
        "like", "well", "so", "basically", "literally", "actually",
    }
)

#: Completeness at/above this is treated as "complete enough" by
#: ``is_utterance_complete``. The dampening in ``utterance_completeness`` is
#: continuous; this threshold only matters for the boolean convenience.
DEFAULT_COMPLETE_THRESHOLD: float = 0.6


@dataclass(frozen=True)
class TextEOUConfig:
    """Tunable completeness multipliers for each incompleteness class.

    Each value is the completeness (in (0.0, 1.0]) assigned when the utterance
    ends on a marker of that class. Lower ⇒ stronger dampening of the
    silence-derived confidence. A complete utterance always scores 1.0.
    Ordering of the constants reflects strength: a dangling conjunction is a
    harder signal of incompleteness than a closing comma.
    """

    #: Ends on a conjunction ("…and", "…because") — strongest "more coming".
    conjunction_completeness: float = 0.2
    #: Ends on a dangling preposition / article ("…to", "…the").
    dangling_completeness: float = 0.3
    #: Ends on a hesitation filler ("…um", "…like").
    filler_completeness: float = 0.35
    #: Ends on an ellipsis ("…") — trailing off, weaker than a word marker.
    ellipsis_completeness: float = 0.5
    #: Ends on a comma — a clause boundary, mildly incomplete.
    comma_completeness: float = 0.6
    #: Boolean threshold for ``is_utterance_complete``.
    complete_threshold: float = DEFAULT_COMPLETE_THRESHOLD

    def __post_init__(self) -> None:
        for name in (
            "conjunction_completeness",
            "dangling_completeness",
            "filler_completeness",
            "ellipsis_completeness",
            "comma_completeness",
        ):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0] (got {v})")


# Trailing ellipsis: "...", "…", or two-or-more dots, optionally trailing space.
_ELLIPSIS_END = re.compile(r"(\.\s*\.[\s.]*|…)\s*$")
# Strip trailing terminal/quoting punctuation to find the last *word* token.
_TRAILING_PUNCT = re.compile(r"[^\w]+$")
# Strip leading punctuation/quotes from the isolated last token.
_LEADING_PUNCT = re.compile(r"^[^\w]+")


def _last_word(text: str) -> str:
    """Lowercased final word token, with surrounding punctuation stripped."""
    stripped = _TRAILING_PUNCT.sub("", text)
    if not stripped:
        return ""
    tail = stripped.split()[-1]
    tail = _LEADING_PUNCT.sub("", tail)
    return tail.lower()


def utterance_completeness(
    text: str,
    config: TextEOUConfig | None = None,
) -> float:
    """Return a completeness multiplier in (0.0, 1.0] for a transcript chunk.

    Pure function — no I/O, no clock reads. ``1.0`` means the text shows no
    sign of being unfinished (a complete sentence, or simply no incompleteness
    marker); lower values flag a trailing-off utterance that should dampen the
    silence-derived turn-end confidence.

    Rules, checked in strength order (strongest dampening wins):
      1. Empty / whitespace ⇒ 1.0 (no text evidence; don't dampen — silence
         alone decides).
      2. Ends on an ellipsis ("…", "...") ⇒ ``ellipsis_completeness``.
      3. Last word is a conjunction ⇒ ``conjunction_completeness``.
      4. Last word is a dangling preposition / article ⇒ ``dangling_completeness``.
      5. Last word is a hesitation filler ⇒ ``filler_completeness``.
      6. Ends on a comma ⇒ ``comma_completeness``.
      7. Otherwise ⇒ 1.0 (looks complete).

    Note the closed classes overlap intentionally — "so" is both a conjunction
    and a filler. Conjunction is checked first (the stronger signal), so "so"
    scores ``conjunction_completeness``.
    """
    cfg = config or TextEOUConfig()
    stripped = text.strip()
    if not stripped:
        return 1.0

    if _ELLIPSIS_END.search(stripped):
        return cfg.ellipsis_completeness

    word = _last_word(stripped)
    if word:
        if word in CONJUNCTION_MARKERS:
            return cfg.conjunction_completeness
        if word in DANGLING_MARKERS:
            return cfg.dangling_completeness
        if word in FILLER_MARKERS:
            return cfg.filler_completeness

    if stripped.endswith(","):
        return cfg.comma_completeness

    return 1.0


def is_utterance_complete(
    text: str,
    config: TextEOUConfig | None = None,
) -> bool:
    """Convenience boolean: True iff completeness >= ``complete_threshold``.

    An empty utterance is "complete" (1.0 >= threshold) — there's no evidence
    of an unfinished thought, so the boolean defers to silence like the
    multiplier does.
    """
    cfg = config or TextEOUConfig()
    return utterance_completeness(text, cfg) >= cfg.complete_threshold


class TextAwareTurnDecider:
    """Turn-decider that folds text EOU completeness into silence confidence.

    Implements the *identical* ``confidence(*, silence_duration_secs,
    transcript_chunk=None)`` interface as ``SilenceTurnDecider`` so call sites
    (``pipecat_server.py``) swap in this decider with no signature change. The
    combined confidence is::

        silence_confidence(silence) * utterance_completeness(transcript_chunk)

    A complete utterance (or no transcript) multiplies by 1.0 — behaviour is
    identical to the silence-only decider. An incomplete one (trailing
    conjunction / preposition / filler / ellipsis) dampens the confidence, so
    the engine stays silent through a mid-thought pause it would otherwise
    treat as a turn-end. The product is monotone in both inputs and never
    exceeds the silence-only value, making this a conservative refinement.
    """

    def __init__(
        self,
        silence_config: TurnDeciderConfig | None = None,
        text_config: TextEOUConfig | None = None,
    ):
        self.silence_config = silence_config or TurnDeciderConfig()
        self.text_config = text_config or TextEOUConfig()

    def confidence(
        self,
        *,
        silence_duration_secs: float,
        transcript_chunk: str | None = None,
    ) -> float:
        base = silence_confidence(silence_duration_secs, self.silence_config)
        if not transcript_chunk:
            return base
        completeness = utterance_completeness(transcript_chunk, self.text_config)
        return base * completeness
