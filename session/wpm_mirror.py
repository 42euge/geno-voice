"""
WPM-mirroring seam for the conversational-rhythm track (backlog item #3 in
the ``ITERATION_LOG.md`` "next directions").

UX research on conversational rapport (and iter-064's own note when it added
``user_wpm``) says it plainly: a bot that **matches the user's speaking rate**
feels more natural and is interrupted less. A user speaking quickly wants
quick answers; a user speaking slowly is jarred by a bot that races. iter-046
shipped ``bot_wpm`` (the bot's measured rate), iter-064 shipped ``user_wpm``
(the user's) and the session-summary "Mirror gap: ±NN WPM (bot − user)" line,
and iter-210 shipped a sentinel that fires when the bot's rate is *consistently*
mis-set. Those three **measure** the gap; this seam is the **decision** that
would close it — nudge the bot's Kokoro ``speed`` knob toward the rate that
would make ``bot_wpm`` track ``user_wpm``.

This module is the swappable seam between *the measured user rate* and *the TTS
speed the next turn should use*. Today the body is a pure, damped, clamped
proportional map. A later lap could replace the body with a learned
rate-matcher behind the identical ``speed(...)`` interface without touching the
call site.

Design follows the GENO.md / ``turn_decider.py`` conventions exactly:

- **Pure function** (no I/O, no clock reads — ``user_wpm`` and ``current_speed``
  are injected by the caller).
- **Off-by-default gate.** A default ``WpmMirrorConfig`` has ``enabled=False``
  and ``mirrored_speed`` returns ``current_speed`` *unchanged* for every input —
  byte-for-byte today's fixed-rate behavior. The mirroring engages only behind
  the explicit switch, so wiring this into the live TTS path (the named
  follow-on) can never regress the proven constant-speed path. This mirrors the
  half-duplex invariant the organic-track seams keep.
- **A small frozen config** so call sites read clearly, with ``__post_init__``
  validation (a misconfigured mirror that silently races or drones is the worst
  failure mode, so bad tunables raise loudly).
- **A thin class** (``WpmMirror``) wrapping the function so a model-backed
  rate-matcher can later implement the identical interface.

See ``docs/research/organic-turn-taking.md`` and the iter-064 / iter-210 log
entries.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "WpmMirrorConfig",
    "mirrored_speed",
    "WpmMirror",
    "DEFAULT_BASE_WPM",
    "DEFAULT_STRENGTH",
    "DEFAULT_MIN_SPEED",
    "DEFAULT_MAX_SPEED",
    "DEFAULT_MIN_DELTA",
]

#: The bot's measured ``bot_wpm`` (iter-046) when the Kokoro ``speed`` knob is
#: ``1.0`` — i.e. the calibration that converts a *target WPM* into a *speed*.
#: ``speed ≈ target_wpm / base_wpm``. 165 is the midpoint of the
#: ``TurnMetrics.print`` green band (130–200 WPM, iter-046) and a reasonable
#: Kokoro nominal; a deployment can recalibrate it without touching the seam.
DEFAULT_BASE_WPM: float = 165.0

#: Fraction of the gap between the current speed and the speed that would
#: *exactly* match the user that a single turn closes. ``0.0`` ⇒ never move
#: (equivalent to disabled), ``1.0`` ⇒ jump straight to the user's rate.
#: The default ``0.5`` is deliberately damped: the user's per-turn WPM is noisy
#: (a one-word "yeah" reads as a very low rate), so a partial nudge converges
#: over a few turns without lurching on a single outlier — the same
#: "natural variation is normal, react gradually" discipline the iter-115+
#: consistency sentinels embody.
DEFAULT_STRENGTH: float = 0.5

#: Intelligibility clamp on the resulting speed multiplier. Below ``min_speed``
#: the bot drones; above ``max_speed`` Kokoro slurs and clips. The mirror never
#: produces a speed outside ``[min_speed, max_speed]`` no matter how extreme the
#: measured ``user_wpm`` (a 40-WPM near-silence or a 400-WPM burst).
DEFAULT_MIN_SPEED: float = 0.8
DEFAULT_MAX_SPEED: float = 1.3

#: Deadband on the *change*: if the proposed new speed differs from the current
#: one by less than this, keep the current speed. Stops imperceptible
#: sub-``min_delta`` nudges from churning the rate turn-to-turn (which the
#: iter-210 ``bot_wpm`` sentinel would otherwise see as needless variation).
#: ``0.0`` disables the deadband.
DEFAULT_MIN_DELTA: float = 0.05


@dataclass(frozen=True)
class WpmMirrorConfig:
    """Tunables for the ``user_wpm → bot speed`` mirroring map.

    A default instance is the **off switch**: ``enabled=False`` ⇒
    ``mirrored_speed`` is the identity on ``current_speed``. Only an explicit
    ``enabled=True`` engages the proportional nudge. A model-backed rate-matcher
    ignores ``base_wpm`` / ``strength`` and sources the target from elsewhere,
    but still honours the ``enabled`` gate and the ``[min_speed, max_speed]``
    clamp.
    """

    enabled: bool = False
    base_wpm: float = DEFAULT_BASE_WPM
    strength: float = DEFAULT_STRENGTH
    min_speed: float = DEFAULT_MIN_SPEED
    max_speed: float = DEFAULT_MAX_SPEED
    min_delta: float = DEFAULT_MIN_DELTA

    def __post_init__(self) -> None:
        if self.base_wpm <= 0:
            raise ValueError(f"base_wpm must be positive (got {self.base_wpm})")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(
                f"strength must be in [0.0, 1.0] (got {self.strength})"
            )
        if self.min_speed <= 0:
            raise ValueError(f"min_speed must be positive (got {self.min_speed})")
        if self.max_speed < self.min_speed:
            raise ValueError(
                "max_speed must be >= min_speed "
                f"(got min={self.min_speed}, max={self.max_speed})"
            )
        if self.min_delta < 0:
            raise ValueError(f"min_delta must be >= 0 (got {self.min_delta})")


def mirrored_speed(
    user_wpm: float,
    current_speed: float,
    config: WpmMirrorConfig | None = None,
) -> float:
    """Map a measured user speaking rate to the next turn's TTS speed.

    Pure function — no I/O, no clock reads. The shape:

      1. **Gate.** If the config is disabled, return ``current_speed`` unchanged
         (the whole point of the off-by-default invariant). No other input is
         consulted.
      2. **No measurement.** ``user_wpm <= 0`` (an empty / zero-duration turn,
         the iter-064 guard) carries no signal ⇒ return ``current_speed``.
      3. **Ideal speed.** The speed that would make ``bot_wpm`` exactly match
         the user is ``ideal = user_wpm / base_wpm``.
      4. **Damped nudge.** Move ``strength`` of the way from ``current_speed``
         toward ``ideal``: ``target = current + strength * (ideal - current)``.
         ``strength < 1`` converges over a few turns rather than lurching on a
         single noisy per-turn WPM.
      5. **Clamp.** Clamp ``target`` to ``[min_speed, max_speed]`` so the bot is
         never driven to an unintelligible rate by an extreme measurement.
      6. **Deadband.** If the clamped target differs from ``current_speed`` by
         less than ``min_delta``, keep ``current_speed`` (no imperceptible
         churn).

    The map is monotone in ``user_wpm`` (a faster user never yields a slower
    bot) — the property a learned replacement would also hold.
    """
    cfg = config or WpmMirrorConfig()

    if not cfg.enabled:
        return current_speed
    if user_wpm <= 0:
        return current_speed

    ideal = user_wpm / cfg.base_wpm
    target = current_speed + cfg.strength * (ideal - current_speed)

    # Intelligibility clamp.
    if target < cfg.min_speed:
        target = cfg.min_speed
    elif target > cfg.max_speed:
        target = cfg.max_speed

    # Deadband: ignore imperceptible changes.
    if abs(target - current_speed) < cfg.min_delta:
        return current_speed

    return target


class WpmMirror:
    """Stateless ``user_wpm → speed`` mirror — the heuristic the TTS path feeds.

    The swappable seam: ``speed(user_wpm=…, current_speed=…)`` is the interface
    a future learned rate-matcher implements too (sourcing the target rate from
    a model instead of the proportional map). A call site holds a ``WpmMirror``
    and asks it for the next turn's speed rather than hardcoding a constant.

    Unlike ``turn_decider``'s ``SilenceTurnDecider`` the mirror carries no
    cross-turn state: each call is a pure function of the just-measured
    ``user_wpm`` and the current speed. The ``strength`` damping already smooths
    across turns, so no accumulator is needed.
    """

    def __init__(self, config: WpmMirrorConfig | None = None):
        self.config = config or WpmMirrorConfig()

    def speed(self, *, user_wpm: float, current_speed: float) -> float:
        return mirrored_speed(user_wpm, current_speed, self.config)
