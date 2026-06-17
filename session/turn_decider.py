"""
Turn-decider seam for the organic turn-taking track (backlog item #2).

``TurnTakingEngine.decide`` already accepts a ``smart_turn_confidence``
parameter (0.0–1.0): how confident we are that the user has *semantically*
finished their turn, beyond mere silence. The engine is fully parameterized
for it — but ``pipecat_server.py`` hardcodes it to ``0.5`` at both call
sites. Worse, ``0.5`` is *below* the engine's ``smart_turn_backchannel_min``
(``0.6``), so every silence-driven backchannel / response tier is **dead in
production today**: the engine can only ever fire on an NLP trigger or an LLM
assessment, never on the silence + confidence path it was built around.

This module is the swappable seam between "where the confidence comes from"
and "what the engine does with it". Today the body is a pure
silence → confidence heuristic (longer silence ⇒ more likely the turn ended).
A later lap replaces the *body* with pipecat's audio-only ``smart-turn``
model (backlog #6) — same ``confidence(...)`` interface, so
``TurnTakingEngine`` never changes. See
``docs/research/organic-turn-taking.md``.

Design follows the GENO.md conventions: a pure function (no I/O, no clock
reads — ``silence_duration_secs`` is injected by the caller), a small config
dataclass so call sites read clearly, and a thin class wrapping the function
so a model-backed decider can later implement the identical interface.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "TurnDeciderConfig",
    "silence_confidence",
    "SilenceTurnDecider",
    "DEFAULT_SILENCE_FLOOR_SECS",
    "DEFAULT_SILENCE_CEILING_SECS",
]

#: Below this much trailing silence, we assume the user is still mid-thought
#: (a pause, not a turn-end): confidence floors at 0.0. A natural mid-utterance
#: pause ("I was thinking… about the deadline") sits under this.
DEFAULT_SILENCE_FLOOR_SECS: float = 2.0

#: At/above this much trailing silence, we're as confident as a silence-only
#: signal can be that the turn ended: confidence saturates at 1.0. Chosen so
#: the linear ramp clears the engine's ``smart_turn_backchannel_min`` (0.6) by
#: its ``silence_backchannel_min`` window (4.0s) — ``confidence(4.0) ≈ 0.67`` —
#: and saturates to ``>= smart_turn_response_min`` (0.85) before its
#: ``silence_response_min`` window (6.0s), so the previously-dead silence tiers
#: become reachable.
DEFAULT_SILENCE_CEILING_SECS: float = 5.0


@dataclass(frozen=True)
class TurnDeciderConfig:
    """Tunables for the silence → confidence heuristic.

    The ramp is linear between ``silence_floor_secs`` (→ 0.0) and
    ``silence_ceiling_secs`` (→ 1.0). Chosen so the curve crosses the
    engine's ``smart_turn_backchannel_min`` (0.6) and ``smart_turn_response_min``
    (0.85) within the engine's own silence windows, making the previously
    dead silence-driven tiers reachable. A smart-turn model implementation
    ignores these and sources confidence from audio instead.
    """

    silence_floor_secs: float = DEFAULT_SILENCE_FLOOR_SECS
    silence_ceiling_secs: float = DEFAULT_SILENCE_CEILING_SECS

    def __post_init__(self) -> None:
        if self.silence_ceiling_secs <= self.silence_floor_secs:
            raise ValueError(
                "silence_ceiling_secs must be greater than silence_floor_secs "
                f"(got floor={self.silence_floor_secs}, "
                f"ceiling={self.silence_ceiling_secs})"
            )


def silence_confidence(
    silence_duration_secs: float,
    config: TurnDeciderConfig | None = None,
) -> float:
    """Map trailing-silence duration to a turn-end confidence in [0.0, 1.0].

    Pure function — no I/O, no clock reads. The confidence ramps linearly
    from 0.0 at ``silence_floor_secs`` to 1.0 at ``silence_ceiling_secs`` and
    is clamped outside that band:

      - ``silence <= floor``   ⇒ 0.0  (still mid-thought; a pause, not a turn-end)
      - ``floor < silence < ceiling`` ⇒ linear ramp
      - ``silence >= ceiling`` ⇒ 1.0  (as sure as silence alone can make us)

    Negative durations clamp to 0.0. This is the heuristic stand-in for an
    audio-based smart-turn model; the *shape* (monotone, saturating) is what a
    model would also produce, so swapping the body later is behaviour-stable.
    """
    cfg = config or TurnDeciderConfig()
    floor = cfg.silence_floor_secs
    ceiling = cfg.silence_ceiling_secs

    if silence_duration_secs <= floor:
        return 0.0
    if silence_duration_secs >= ceiling:
        return 1.0
    return (silence_duration_secs - floor) / (ceiling - floor)


class SilenceTurnDecider:
    """Silence-only ``TurnDecider`` — the heuristic the engine feeds today.

    The swappable seam: ``confidence(silence_duration_secs=…)`` is the
    interface a future pipecat ``smart-turn`` decider implements too (sourcing
    confidence from the recent audio buffer instead of silence). Call sites
    (``pipecat_server.py``) hold a ``TurnDecider`` and ask it for confidence
    rather than passing a literal ``0.5``.

    ``transcript_chunk`` is accepted (and ignored here) so the interface is
    forward-compatible with a text-aware EOU signal (backlog #4) without a
    later call-site change.
    """

    def __init__(self, config: TurnDeciderConfig | None = None):
        self.config = config or TurnDeciderConfig()

    def confidence(
        self,
        *,
        silence_duration_secs: float,
        transcript_chunk: str | None = None,
    ) -> float:
        return silence_confidence(silence_duration_secs, self.config)
