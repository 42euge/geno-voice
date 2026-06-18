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

import statistics
from dataclasses import dataclass, field

__all__ = [
    "WpmMirrorConfig",
    "mirrored_speed",
    "WpmMirror",
    "SpeedTrajectory",
    "simulate_speed_trajectory",
    "MirrorGridPoint",
    "sweep_mirror_grid",
    "pick_best_mirror_config",
    "TUNING_CORPUS_WPMS",
    "TUNING_STRENGTH_AXIS",
    "tune_strength",
    "CalibrationSample",
    "BaseWpmCalibration",
    "calibrate_base_wpm",
    "DEFAULT_LURCH_WEIGHT",
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


# --------------------------------------------------------------------------
# iter-216 — offline trajectory simulator for tuning base_wpm / strength.
#
# iter-213 shipped the pure mirror, iter-214 wired it into the live TTS path,
# and iter-215 surfaced the per-session start→end drift in the summary. The
# loop now *adapts* and *measures* the rate — but the tunables (``base_wpm``
# 165, ``strength`` 0.5) are still the seed defaults, never validated against
# a sequence of varied user pacing.
#
# ``simulate_speed_trajectory`` is the offline tool that closes that gap. Given
# a config and a sequence of per-turn ``user_wpm`` values (e.g. a slow → fast
# → slow arc, the case the live ``SpeedController`` would see across a real
# conversation), it replays the *exact same* turn-by-turn fold the live
# ``SpeedController.observe`` does — feeding each turn's clamped/deadbanded
# output speed as the next turn's ``current_speed`` — and reports the resulting
# speed trajectory plus convergence diagnostics. That lets a later lap pick
# ``base_wpm`` / ``strength`` from data (does the speed track the user's pacing
# without lurching? how fast does a sustained rate converge?) rather than from
# the seed defaults, with no live session and no audio.
#
# Pure: no I/O, no clock, no state — a thin deterministic loop over
# ``mirrored_speed``. The whole point is that it is the *same* map the live
# path runs, so its verdict transfers.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeedTrajectory:
    """Result of replaying a ``user_wpm`` sequence through the mirror.

    Every field is derived purely from the input sequence and config so the
    same inputs always yield the same trajectory (the simulator is the offline
    twin of the live ``SpeedController`` fold).

    Attributes:
      speeds: the speed *after* each turn's observation, in turn order. Length
        equals the number of input ``user_wpm`` values. ``speeds[i]`` is the
        speed the synth path would use for turn ``i+1``'s sentences.
      initial_speed: the speed the trajectory started from (before any turn).
      final_speed: the speed after the last turn (``speeds[-1]``, or
        ``initial_speed`` for an empty sequence).
      ideal_final_speed: the speed that would make ``bot_wpm`` exactly match
        the *last measurable* ``user_wpm`` (``user_wpm / base_wpm``, clamped to
        the intelligibility band), i.e. the rate the trajectory is converging
        toward at the end. ``None`` when no turn carried a measurable rate or
        mirroring is disabled (nothing to converge to).
      final_gap: ``final_speed - ideal_final_speed`` — the residual distance to
        the converged target after the whole sequence. ``None`` when
        ``ideal_final_speed`` is ``None``. A small ``|final_gap|`` means the
        chosen ``strength`` converged within the sequence length.
      max_step: the largest single-turn absolute speed change across the
        sequence (``0.0`` for a held / empty / disabled trajectory) — the
        "lurch" diagnostic. A large ``max_step`` on noisy input means
        ``strength`` is too aggressive.
      moves: how many turns actually changed the speed (a deadband hold or a
        no-measurement turn does not count) — the churn diagnostic.
    """

    speeds: list = field(default_factory=list)
    initial_speed: float = 1.0
    final_speed: float = 1.0
    ideal_final_speed: float | None = None
    final_gap: float | None = None
    max_step: float = 0.0
    moves: int = 0


def simulate_speed_trajectory(
    user_wpms,
    initial_speed: float = 1.0,
    config: WpmMirrorConfig | None = None,
) -> SpeedTrajectory:
    """Replay a per-turn ``user_wpm`` sequence through the mirror.

    Deterministically folds each ``user_wpm`` through :func:`mirrored_speed`,
    threading each turn's output speed in as the next turn's ``current_speed``
    — the exact loop the live ``SpeedController.observe`` runs, but offline over
    a whole sequence. Returns a :class:`SpeedTrajectory` carrying the per-turn
    speeds and the convergence / lurch / churn diagnostics a tuning lap reads.

    Args:
      user_wpms: iterable of per-turn measured user speaking rates (WPM). A
        ``<= 0`` value (the iter-064 "no measurement" guard) is replayed
        faithfully — it carries no signal so the speed holds for that turn.
      initial_speed: the speed before any turn (the ``mic_chat`` CLI / config
        value, historically ``1.0``).
      config: the :class:`WpmMirrorConfig` under test. With a disabled config
        (the default) the speed never moves — the trajectory is flat at
        ``initial_speed`` and ``ideal_final_speed`` is ``None``.

    Pure — no I/O, no clock, no mutation of ``config`` or the input.
    """
    cfg = config or WpmMirrorConfig()
    wpms = [float(w) for w in user_wpms]

    speeds: list = []
    speed = float(initial_speed)
    max_step = 0.0
    moves = 0
    for w in wpms:
        new_speed = mirrored_speed(w, speed, cfg)
        step = abs(new_speed - speed)
        if step > max_step:
            max_step = step
        if new_speed != speed:
            moves += 1
        speed = new_speed
        speeds.append(speed)

    final_speed = speed if speeds else float(initial_speed)

    # The target the trajectory is converging toward is set by the last
    # *measurable* user rate (the same one the live path would be tracking at
    # the end). With mirroring disabled or no measurable turn, there is nothing
    # to converge to.
    ideal_final_speed: float | None = None
    final_gap: float | None = None
    if cfg.enabled:
        last_measurable = next((w for w in reversed(wpms) if w > 0), None)
        if last_measurable is not None:
            ideal = last_measurable / cfg.base_wpm
            if ideal < cfg.min_speed:
                ideal = cfg.min_speed
            elif ideal > cfg.max_speed:
                ideal = cfg.max_speed
            ideal_final_speed = ideal
            final_gap = final_speed - ideal

    return SpeedTrajectory(
        speeds=speeds,
        initial_speed=float(initial_speed),
        final_speed=final_speed,
        ideal_final_speed=ideal_final_speed,
        final_gap=final_gap,
        max_step=max_step,
        moves=moves,
    )


# --------------------------------------------------------------------------
# iter-217 — offline base_wpm × strength grid sweep + data-driven picker.
#
# iter-216 shipped ``simulate_speed_trajectory`` — the single-config offline
# twin of the live ``SpeedController`` fold. Its backlog item #1 is the natural
# next hop: *run* it over a realistic varied-pacing arc across a grid of
# ``base_wpm`` × ``strength`` candidates, read each cell's convergence
# (``final_gap``) / lurch (``max_step``) / churn (``moves``) diagnostics, and
# pick the pair that tracks the user without lurching — so the seed defaults
# (``base_wpm`` 165, ``strength`` 0.5) can be replaced from data (or kept with
# a documented reason) rather than by assertion.
#
# This is the same shape as the VAD ``sweep_param`` harness
# (``fixtures/replay_vad.py``): fold a parameter grid over a fixed corpus into
# one machine-readable comparison row per cell, then aggregate into a verdict.
# Here the "corpus" is the per-turn ``user_wpm`` arc and the "replay" is the
# pure trajectory fold — no audio, no clock, no live session.
#
# Pure: each cell is one ``simulate_speed_trajectory`` call; the sweep and the
# picker mutate nothing and read no I/O. The picker's verdict transfers to the
# live path because it scores the *same* fold the live ``SpeedController`` runs.
# --------------------------------------------------------------------------


#: Default weight on the lurch term (``max_step``) relative to the convergence
#: term (``|final_gap|``) in :meth:`MirrorGridPoint.score`. Both terms are in
#: speed-multiplier units, so a weight of ``1.0`` would treat a 0.1 residual gap
#: and a 0.1 single-turn jump as equally bad. The default ``0.5`` says a smooth
#: approach matters, but converging to the user's rate matters twice as much —
#: a bot that lags the user slightly is less jarring than one that lurches every
#: turn, but a bot that never reaches the user's pace defeats the whole mirror.
DEFAULT_LURCH_WEIGHT: float = 0.5


@dataclass(frozen=True)
class MirrorGridPoint:
    """One ``(base_wpm, strength)`` cell of a tuning grid sweep.

    Carries the cell's tunables alongside the convergence / lurch / churn
    diagnostics of replaying the shared arc through it. ``score`` folds the two
    that matter for tuning — residual gap and lurch — into a single
    lower-is-better number so a grid can be ranked.

    Attributes mirror the :class:`SpeedTrajectory` fields the cell produced:
      base_wpm, strength: the tunables under test (the grid axes).
      final_speed: the speed after the last turn of the arc.
      ideal_final_speed: the band-clamped target the arc converges toward
        (``None`` when no measurable turn — the cell is unscorable).
      final_gap: ``final_speed - ideal_final_speed`` (``None`` when unscorable).
      max_step: the largest single-turn speed change (the lurch diagnostic).
      moves: how many turns changed the speed (the churn diagnostic).
    """

    base_wpm: float
    strength: float
    final_speed: float
    ideal_final_speed: float | None
    final_gap: float | None
    max_step: float
    moves: int

    @property
    def abs_final_gap(self) -> float | None:
        """``|final_gap|`` — the unsigned residual distance to the target."""
        return None if self.final_gap is None else abs(self.final_gap)

    def score(self, lurch_weight: float = DEFAULT_LURCH_WEIGHT) -> float | None:
        """Lower-is-better tuning score: ``|final_gap| + lurch_weight*max_step``.

        ``None`` when the cell is unscorable (no measurable turn / disabled), so
        a picker can skip it. A cell that both converges (small ``|final_gap|``)
        and stays smooth (small ``max_step``) scores low; one that lags the user
        (large gap) or jumps every turn (large step) scores high.
        """
        if self.abs_final_gap is None:
            return None
        return self.abs_final_gap + lurch_weight * self.max_step


def sweep_mirror_grid(
    user_wpms,
    base_wpms,
    strengths,
    initial_speed: float = 1.0,
    template: WpmMirrorConfig | None = None,
) -> list:
    """Replay one ``user_wpm`` arc through every ``base_wpm`` × ``strength`` cell.

    The grid analogue of :func:`simulate_speed_trajectory`: for each
    ``(base_wpm, strength)`` pair it builds an **enabled** config (cloning the
    non-grid tunables — ``min_speed`` / ``max_speed`` / ``min_delta`` — from
    ``template``, defaulting to the seed values) and folds the shared arc
    through it, collecting one :class:`MirrorGridPoint` per cell.

    Args:
      user_wpms: the per-turn arc replayed identically through every cell (e.g.
        a slow → fast → slow conversation). The fixed "corpus".
      base_wpms: candidate ``base_wpm`` calibrations (the first grid axis).
      strengths: candidate ``strength`` damping values (the second axis).
      initial_speed: the speed before any turn (the ``mic_chat`` start speed).
      template: a :class:`WpmMirrorConfig` whose ``min_speed`` / ``max_speed`` /
        ``min_delta`` (and ``enabled``-overridden) are reused for every cell, so
        the sweep varies only the two grid axes against a fixed band/deadband.
        ``None`` uses the seed defaults.

    Returns one ``MirrorGridPoint`` per cell in row-major order (outer loop
    ``base_wpms``, inner loop ``strengths``) — the machine-readable comparison
    table a tuning lap reads. Pure: no I/O, no mutation of inputs.
    """
    tmpl = template or WpmMirrorConfig()
    points: list = []
    for base_wpm in base_wpms:
        for strength in strengths:
            cfg = WpmMirrorConfig(
                enabled=True,
                base_wpm=float(base_wpm),
                strength=float(strength),
                min_speed=tmpl.min_speed,
                max_speed=tmpl.max_speed,
                min_delta=tmpl.min_delta,
            )
            traj = simulate_speed_trajectory(user_wpms, initial_speed, cfg)
            points.append(
                MirrorGridPoint(
                    base_wpm=cfg.base_wpm,
                    strength=cfg.strength,
                    final_speed=traj.final_speed,
                    ideal_final_speed=traj.ideal_final_speed,
                    final_gap=traj.final_gap,
                    max_step=traj.max_step,
                    moves=traj.moves,
                )
            )
    return points


def pick_best_mirror_config(
    points,
    lurch_weight: float = DEFAULT_LURCH_WEIGHT,
) -> MirrorGridPoint | None:
    """Pick the lowest-:meth:`~MirrorGridPoint.score` cell from a grid sweep.

    Scores every cell with the shared ``lurch_weight`` and returns the one with
    the smallest (best) score — the ``(base_wpm, strength)`` pair that converges
    to the user's pace without lurching across the arc. Unscorable cells (no
    measurable turn) are skipped. ``None`` when the grid is empty or every cell
    is unscorable.

    Earliest-tie rule (matching ``_longest_consecutive_run`` and the VAD sweep):
    on an exact score tie the earlier cell in row-major order wins, so a stable
    grid ordering yields a stable pick. Pure — reads nothing, mutates nothing.
    """
    best: MirrorGridPoint | None = None
    best_score: float | None = None
    for p in points:
        s = p.score(lurch_weight)
        if s is None:
            continue
        if best_score is None or s < best_score:
            best = p
            best_score = s
    return best


# --------------------------------------------------------------------------
# iter-219 — the canonical in-band tuning corpus + the data-driven strength
# verdict (the backlog item repeated across iter-216/217/218).
#
# iter-216 shipped the trajectory simulator, iter-217 the grid sweep + picker,
# and iter-218 the `gv simulate-mirror` CLI. The named follow-on in all three
# backlogs was the same: *run* the sweep on a corpus whose per-turn rates stay
# inside the intelligibility band so ``final_gap`` measures real pacing-tracking
# rather than the ``min_speed``/``max_speed`` clamp — then either change the seed
# defaults from the verdict or document why they stand. The iter-217 demo arc
# ``[120,140,200,230,200,140,120]`` could not do that: its slow tail (120 WPM)
# clamps below the 0.8 ``min_speed`` floor at every base, so its verdict was an
# artifact of *which base pushed the clamped final speed lowest*, not of tracking.
#
# Two findings from running the real sweep close the item:
#
# 1. **``base_wpm`` is NOT tunable offline — only ``strength`` is.** ``base_wpm``
#    is the hardware calibration ``bot_wpm at speed 1.0``; the simulator's own
#    ``ideal = user_wpm / base_wpm`` *uses* it to define the convergence target,
#    so a sweep that varies ``base_wpm`` is scoring each cell against its own
#    moving target — circular. The right ``base_wpm`` is whatever a deployment's
#    Kokoro voice actually clocks at speed 1.0 (an on-device measurement, not a
#    replay). ``strength``, by contrast, is pure convergence dynamics at a fixed
#    base and *is* answerable offline: does the speed track a varied-pacing arc
#    without lurching or churning?
#
# 2. **The seed ``strength=0.5`` wins the fair test.** On ``TUNING_CORPUS_WPMS``
#    (a slow→fast→slow arc, every rate inside the band at base 150/165/180, with
#    a sustained tail so every candidate has converged) the lowest-``score`` cell
#    at the seed base 165 is ``strength=0.5``: ``0.3``/``0.4`` lag (the deadband
#    blocks the small early nudges so the speed never catches the user), and
#    ``0.6``/``0.7`` lurch (a single noisy turn jumps the rate further than is
#    comfortable). 0.5 is the knee. So the seed default STANDS, now from data.
#
# ``tune_strength`` is the in-tree, audio-free reproduction of that verdict; the
# unit tests pin both the corpus's in-band invariant and the 0.5 winner so a
# later change to the mirror map that would shift the knee fails loudly.
# --------------------------------------------------------------------------


#: Canonical varied-pacing tuning corpus: a slow→fast→slow per-turn ``user_wpm``
#: arc whose every rate sits inside ``[144, 195]`` — the intersection of the
#: intelligibility band (``[0.8·base, 1.3·base]``) across base_wpm 150/165/180 —
#: so ``simulate_speed_trajectory``'s ``final_gap`` measures pacing-tracking, not
#: the ``min_speed``/``max_speed`` clamp. The sustained slow tail (three turns at
#: 150) lets every ``strength`` candidate converge, so the ranking turns on lurch
#: and churn rather than residual gap. Unlike the iter-217 demo arc (whose 120-WPM
#: tail clamps below the floor at every base), this corpus produces a *fair*
#: ``strength`` verdict. See the iter-219 log entry.
TUNING_CORPUS_WPMS: tuple = (165.0, 150.0, 190.0, 195.0, 170.0, 150.0, 150.0, 150.0)

#: The ``strength`` candidates swept by :func:`tune_strength`. Spans the seed
#: default (0.5) with one step either side at 0.1 resolution, so the verdict
#: shows the knee, not just a binary.
TUNING_STRENGTH_AXIS: tuple = (0.3, 0.4, 0.5, 0.6, 0.7)


def tune_strength(
    user_wpms=TUNING_CORPUS_WPMS,
    base_wpm: float = DEFAULT_BASE_WPM,
    strengths=TUNING_STRENGTH_AXIS,
    initial_speed: float = 1.0,
    lurch_weight: float = DEFAULT_LURCH_WEIGHT,
) -> MirrorGridPoint | None:
    """Pick the best ``strength`` at a *fixed* ``base_wpm`` over a tuning arc.

    The offline ``strength`` tuner. Unlike :func:`sweep_mirror_grid` (which
    varies ``base_wpm`` too), this holds ``base_wpm`` fixed because ``base_wpm``
    is a hardware calibration that cannot be tuned by replay — the simulator
    *uses* it to define the convergence target, so sweeping it scores each cell
    against its own moving target. ``strength`` is pure convergence dynamics and
    *is* answerable offline, which is what this returns.

    Folds ``user_wpms`` (defaulting to :data:`TUNING_CORPUS_WPMS`) through a
    one-row grid of ``strengths`` at the single ``base_wpm`` and returns the
    lowest-:meth:`~MirrorGridPoint.score` cell — the damping that converges to
    the user's pace without lurching. ``None`` if no candidate is scorable.

    Pure — a thin wrapper over :func:`sweep_mirror_grid` /
    :func:`pick_best_mirror_config`; reads nothing, mutates nothing.
    """
    points = sweep_mirror_grid(
        user_wpms,
        [float(base_wpm)],
        strengths,
        initial_speed=initial_speed,
    )
    return pick_best_mirror_config(points, lurch_weight=lurch_weight)


# --------------------------------------------------------------------------
# iter-220 — on-device ``base_wpm`` calibration from rendered samples.
#
# iter-219 closed the offline tuning question with a hard finding: ``base_wpm``
# is NOT tunable by replay. It is the bot's actual ``bot_wpm`` (iter-046) at
# Kokoro ``speed=1.0`` — a hardware/voice calibration — and the simulator's own
# ``ideal = user_wpm / base_wpm`` *uses* it to define the convergence target, so
# sweeping it scores each cell against its own moving target (circular). The
# right ``base_wpm`` for a deployment is therefore a *measurement*: synthesize a
# known-length script, time the audio, and back out the rate the voice clocks at
# speed 1.0.
#
# This is the audio-free arithmetic core of that calibration. A
# ``CalibrationSample`` is one render — ``words`` synthesized into
# ``audio_seconds`` of audio at a known Kokoro ``speed`` — and it derives both
# the measured ``bot_wpm`` (the iter-046 ``words·60/audio_seconds``) and the
# ``implied_base_wpm`` (that rate normalized back to speed 1.0, i.e.
# ``bot_wpm / speed``, since ``bot_wpm ≈ speed · base_wpm``).
# ``calibrate_base_wpm`` folds one-or-more samples into a robust *median*
# ``implied_base_wpm`` plus spread (min↔max) and drift-vs-nominal diagnostics, so
# an operator can set ``base_wpm`` from their own voice instead of the 165
# nominal seed (``DEFAULT_BASE_WPM``).
#
# The real Kokoro render that *produces* the samples is the on-device follow-on
# (gate on a real synth); this lap ships the pure measurement that turns a
# rendered duration into a ``base_wpm`` verdict — no I/O, no clock, no audio.
# Unit-tested in isolation exactly like the iter-216/217/219 simulator engine it
# complements.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationSample:
    """One TTS render used to calibrate ``base_wpm``.

    A known-length script (``words``) synthesized into ``audio_seconds`` of
    audio at a known Kokoro ``speed`` knob. Derives the measured rate and the
    rate the voice would clock at the calibration point (``speed=1.0``).

    Attributes:
      words: number of words in the rendered script (must be ``> 0``).
      audio_seconds: measured duration of the rendered audio (must be ``> 0``).
      speed: the Kokoro ``speed`` knob the render used (must be ``> 0``;
        defaults to the ``1.0`` calibration point).

    All three are validated in ``__post_init__`` because a non-positive value is
    a measurement bug (an empty script, a zero-length clip, or a missing speed)
    that would otherwise divide by zero or produce a nonsensical negative rate.
    """

    words: int
    audio_seconds: float
    speed: float = 1.0

    def __post_init__(self) -> None:
        if self.words <= 0:
            raise ValueError(f"words must be positive (got {self.words})")
        if self.audio_seconds <= 0:
            raise ValueError(
                f"audio_seconds must be positive (got {self.audio_seconds})"
            )
        if self.speed <= 0:
            raise ValueError(f"speed must be positive (got {self.speed})")

    @property
    def bot_wpm(self) -> float:
        """Measured speaking rate of this render (iter-046 convention)."""
        return self.words / (self.audio_seconds / 60.0)

    @property
    def implied_base_wpm(self) -> float:
        """The rate the voice would clock at ``speed=1.0`` (the calibration).

        ``bot_wpm ≈ speed · base_wpm`` (the same relation the live
        ``SpeedController`` inverts when it picks ``speed ≈ target_wpm /
        base_wpm``), so dividing the measured rate by the render's ``speed``
        normalizes it back to the ``speed=1.0`` calibration point.
        """
        return self.bot_wpm / self.speed


@dataclass(frozen=True)
class BaseWpmCalibration:
    """Verdict of folding one-or-more :class:`CalibrationSample` renders.

    Attributes:
      implied_base_wpm: the robust **median** of the samples'
        ``implied_base_wpm`` — the value to set ``DEFAULT_BASE_WPM`` to for this
        voice. Median (not mean) so a single mis-timed render does not skew it.
      n_samples: how many samples were folded.
      min_base_wpm / max_base_wpm: the extremes of the per-sample
        ``implied_base_wpm`` — the calibration's range.
      spread: ``max_base_wpm - min_base_wpm`` — a large spread means the renders
        disagree (inconsistent synth or a bad sample), so the median is less
        trustworthy.
      default_base_wpm: the nominal seed the calibration is compared against.
      drift: ``implied_base_wpm - default_base_wpm`` — how far this voice clocks
        from the 165 nominal. Positive ⇒ the voice is faster than nominal at
        speed 1.0 (so the seed would make the mirror under-shoot the target
        speed); negative ⇒ slower.
    """

    implied_base_wpm: float
    n_samples: int
    min_base_wpm: float
    max_base_wpm: float
    spread: float
    default_base_wpm: float
    drift: float


def calibrate_base_wpm(
    samples,
    default_base_wpm: float = DEFAULT_BASE_WPM,
) -> BaseWpmCalibration | None:
    """Fold calibration renders into a robust ``base_wpm`` verdict.

    Each :class:`CalibrationSample`'s ``implied_base_wpm`` normalizes its render
    back to the ``speed=1.0`` calibration point, so samples taken at *different*
    speeds are directly comparable. Returns their **median** as the calibrated
    ``base_wpm`` (robust to a single mis-timed render) plus spread and
    drift-vs-nominal diagnostics.

    Args:
      samples: iterable of :class:`CalibrationSample`. Empty ⇒ ``None`` (nothing
        to calibrate from).
      default_base_wpm: the nominal seed to report ``drift`` against
        (defaults to :data:`DEFAULT_BASE_WPM`).

    Pure — no I/O, no clock, no mutation of the input.
    """
    bases = [s.implied_base_wpm for s in samples]
    if not bases:
        return None
    median = statistics.median(bases)
    lo = min(bases)
    hi = max(bases)
    return BaseWpmCalibration(
        implied_base_wpm=median,
        n_samples=len(bases),
        min_base_wpm=lo,
        max_base_wpm=hi,
        spread=hi - lo,
        default_base_wpm=float(default_base_wpm),
        drift=median - float(default_base_wpm),
    )
