"""
Agent backchannel emission timing for the organic turn-taking track
(backlog item #7 in ``docs/research/organic-turn-taking.md``).

The other half of backchanneling. ``session/backchannel.py`` (#1, iter-148)
*recognizes* a user continuer; this module decides when the **agent** should
*emit* one — a short "mhmm" / "right" *during* the user's speech to signal
active listening, the way a human nods along while someone talks.

geno-voice's ``TurnTakingEngine`` already emits cues (``PLAY_CUE`` →
``broadcast_cue``), but only on **trailing silence** ≥ ``silence_backchannel_min``
(4.0s) — i.e. when the user has basically stopped. That is a turn-end-ish cue,
not the human "mm-hmm while you keep talking." Krisp's tiny turn-taking model
calls out a dedicated *backchannel-opportunity* head — "is now a good moment for
the agent to backchannel?" — as the notable signal most EOU models lack. This
module is the rule-based, dependency-free first step toward that head.

**The signal.** A good moment to backchannel mid-speech is a brief *clause
boundary* pause — long enough to be a real gap (not mid-word), but clearly
**below** the turn-end silence floor (above that, it is the silence-driven
``PLAY_CUE`` path's job, not a mid-speech backchannel). That floor is exactly
``turn_decider.py``'s ``silence_floor_secs`` (2.0s, "a pause, not a turn-end"),
so the two paths partition the silence axis cleanly: ``[min_pause, max_pause)``
is the mid-speech backchannel window; ``≥ silence_backchannel_min`` is the
turn-end cue window. That no-overlap partition requires ``max_pause_secs <=
silence_backchannel_min`` (2.0 ≤ 4.0), pinned by
``tests/unit/test_cue_window_partition_invariant.py`` (iter-180) against
``turn_taking.py`` itself, not a literal. Plus the same rate limits the engine
already uses (a minimum monologue length before the first cue, a minimum gap
between cues), so
the agent doesn't backchannel over a one-word reply or chatter "mhmm mhmm mhmm."

**The half-duplex invariant is the whole point.** With a default
``FullDuplexConfig()`` (``agent_backchannels`` inactive), this function returns
``HOLD`` for *every* input — byte-for-byte today's behavior, where the agent
never speaks during user speech. Only when agent backchannels are explicitly
turned on does a well-timed pause yield ``EMIT``. So wiring this into the live
cue path (a later lap) can never regress the proven half-duplex path; the new
behavior lives entirely behind the off-by-default switch.

Design follows the GENO.md conventions and its sibling seams (#1/#3/#5): a pure
function (no I/O, no clock reads — the caller injects the already-measured
durations), an injected config, a frozen timing-config dataclass, and a small
enum return so call sites read clearly. Like its siblings it loads by file path
in tests to dodge ``session/__init__``'s eager pipecat import (absent on the
x86_64 runner).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from session.full_duplex import FullDuplexConfig

__all__ = [
    "BackchannelTiming",
    "BackchannelTimingConfig",
    "decide_backchannel_timing",
    "should_emit_backchannel",
]


class BackchannelTiming(Enum):
    """Whether *now* is a good moment for the agent to emit a backchannel."""

    #: Emit a short continuer cue ("mhmm") — the user paused at a clause
    #: boundary and is mid-monologue; nodding along signals active listening.
    EMIT = "emit"
    #: Stay quiet — too soon, too frequent, no pause, or a turn-end-sized gap
    #: that the silence-driven ``PLAY_CUE`` path owns instead.
    HOLD = "hold"


@dataclass(frozen=True)
class BackchannelTimingConfig:
    """Thresholds for the mid-speech backchannel-opportunity heuristic.

    Defaults mirror the analogous knobs the ``TurnTakingEngine`` already uses
    for its trailing-silence cues so the two paths feel consistent:

      - ``min_speaking_before_first_cue_secs`` / ``min_between_cues_secs`` —
        the same rate limits as ``TurnTakingConfig`` (15.0 / 20.0), so the
        agent doesn't backchannel over a short reply or chatter repeatedly.
      - ``min_pause_secs`` — a pause must be at least this long to count as a
        real clause boundary (not a mid-word stumble). Default 0.3s.
      - ``max_pause_secs`` — at/above this the gap is heading toward turn-end
        territory; hand it to the silence-driven ``PLAY_CUE`` path instead of
        emitting a *mid-speech* backchannel. Default 2.0s — exactly
        ``turn_decider.py``'s ``silence_floor_secs`` ("a pause, not a
        turn-end") and ``monologue_clock.py``'s ``reset_gap_secs``, so the two
        backchannel paths partition the silence axis with no overlap. That
        cross-module equality is pinned by
        ``tests/unit/test_shared_silence_floor_invariant.py`` (iter-179) against
        the sibling modules themselves, not a hardcoded ``2.0``. And it must
        stay ``<=`` ``turn_taking.py``'s ``silence_backchannel_min`` (4.0s,
        where the turn-end cue window opens) so this mid-speech window and that
        turn-end window don't overlap — pinned by
        ``tests/unit/test_cue_window_partition_invariant.py`` (iter-180).
    """

    min_speaking_before_first_cue_secs: float = 15.0
    min_between_cues_secs: float = 20.0
    min_pause_secs: float = 0.3
    max_pause_secs: float = 2.0

    def __post_init__(self) -> None:
        if self.min_speaking_before_first_cue_secs < 0:
            raise ValueError("min_speaking_before_first_cue_secs must be >= 0")
        if self.min_between_cues_secs < 0:
            raise ValueError("min_between_cues_secs must be >= 0")
        if self.min_pause_secs < 0:
            raise ValueError("min_pause_secs must be >= 0")
        if self.max_pause_secs <= self.min_pause_secs:
            raise ValueError(
                "max_pause_secs must be > min_pause_secs "
                f"(got min={self.min_pause_secs}, max={self.max_pause_secs})"
            )


def decide_backchannel_timing(
    *,
    user_speaking_secs: float,
    pause_secs: float,
    secs_since_last_backchannel: float | None = None,
    config: FullDuplexConfig | None = None,
    timing: BackchannelTimingConfig | None = None,
) -> BackchannelTiming:
    """Decide whether the agent should ``EMIT`` a mid-speech backchannel now.

    Pure function — no I/O, no clock reads. The caller injects the
    already-measured durations (how long the user has been talking this
    monologue, the current within-speech pause, and how long since the agent
    last backchanneled). ``config`` gates the behavior (default: a fresh
    half-duplex ``FullDuplexConfig()``); ``timing`` carries the thresholds
    (default: ``BackchannelTimingConfig()``).

    Rules, in order (first ``HOLD`` wins):

      1. **Gate first.** If ``config.agent_backchannels_active()`` is False
         (the default), return ``HOLD`` unconditionally — the agent never
         speaks during user speech. This is the half-duplex invariant; no
         other input is even consulted.
      2. **Warm-up.** ``user_speaking_secs`` below
         ``min_speaking_before_first_cue_secs`` ⇒ ``HOLD`` (don't backchannel
         over a brief reply).
      3. **Rate limit.** A known ``secs_since_last_backchannel`` below
         ``min_between_cues_secs`` ⇒ ``HOLD`` (no "mhmm mhmm mhmm"). ``None``
         (never backchanneled yet) passes this gate.
      4. **Pause window.** ``EMIT`` iff
         ``min_pause_secs <= pause_secs < max_pause_secs`` — a real clause-
         boundary gap that is still clearly below the turn-end floor.
         Otherwise ``HOLD`` (continuous speech below ``min_pause``, or a
         turn-end-sized gap at/above ``max_pause`` that the silence path owns).
    """
    if config is None:
        config = FullDuplexConfig()
    if timing is None:
        timing = BackchannelTimingConfig()

    # Rule 1 — the half-duplex invariant. Off by default ⇒ never emit during
    # user speech, exactly as today. No other signal is consulted.
    if not config.agent_backchannels_active():
        return BackchannelTiming.HOLD

    # Rule 2 — warm-up: don't backchannel over a short reply.
    if user_speaking_secs < timing.min_speaking_before_first_cue_secs:
        return BackchannelTiming.HOLD

    # Rule 3 — rate limit: a known recent cue too close behind us holds.
    if (
        secs_since_last_backchannel is not None
        and secs_since_last_backchannel < timing.min_between_cues_secs
    ):
        return BackchannelTiming.HOLD

    # Rule 4 — the opportunity: a clause-boundary pause below the turn-end
    # floor. (Continuous speech and turn-end-sized gaps both HOLD.)
    if timing.min_pause_secs <= pause_secs < timing.max_pause_secs:
        return BackchannelTiming.EMIT
    return BackchannelTiming.HOLD


def should_emit_backchannel(
    *,
    user_speaking_secs: float,
    pause_secs: float,
    secs_since_last_backchannel: float | None = None,
    config: FullDuplexConfig | None = None,
    timing: BackchannelTimingConfig | None = None,
) -> bool:
    """Convenience boolean: True iff ``decide_backchannel_timing`` ⇒ ``EMIT``.

    The natural call-site shape — ``if should_emit_backchannel(...):
    broadcast_cue(...)``. With a default (half-duplex) config this is always
    False, so the live cue path is unchanged until agent backchannels are
    explicitly enabled.
    """
    return (
        decide_backchannel_timing(
            user_speaking_secs=user_speaking_secs,
            pause_secs=pause_secs,
            secs_since_last_backchannel=secs_since_last_backchannel,
            config=config,
            timing=timing,
        )
        is BackchannelTiming.EMIT
    )
