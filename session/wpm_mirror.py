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
    "dispersion_grade",
    "dispersion_margin",
    "CALIB_AGREE_REL_SPREAD",
    "CALIB_LOOSE_REL_SPREAD",
    "CalibrationVerdict",
    "calibration_verdict",
    "BaseWpmCalibrationBatch",
    "calibrate_base_wpm_batch",
    "CALIB_BATCH_GRADE_ORDER",
    "DEFAULT_CALIB_SPREAD_MAX",
    "DEFAULT_CALIB_DRIFT_MIN",
    "DEFAULT_CALIB_MIN_SAMPLES",
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
      relative_spread: ``spread / implied_base_wpm`` — the spread normalized by
        the median (iter-393), a dimensionless coefficient of dispersion. The
        absolute ``spread`` is base-dependent (a 10 WPM range is tight at a
        300-WPM voice but wide at a 100-WPM voice), so this companion lets an
        operator judge whether the renders AGREE *independent* of the voice's
        nominal rate. ``0.0`` when the renders agree exactly; larger means more
        disagreement relative to the rate.
      dispersion_grade: a categorical trust grade (iter-394) bucketing
        ``relative_spread`` into ``"agree"`` / ``"loose"`` / ``"scattered"`` — the
        calibration analogue of iter-348's ``vad-gap-confidence`` grade, turning
        the raw dimensionless dispersion into a one-glance read of how
        trustworthy the median is. ``"agree"`` (small relative spread ⇒ the
        renders cluster tightly, the median is solid), ``"loose"`` (moderate ⇒
        usable but sanity-check), ``"scattered"`` (large ⇒ the renders disagree,
        re-render more consistently). Computed from ``relative_spread`` alone, so
        it is voice-comparable the same way ``relative_spread`` is. A
        single-sample calibration grades ``"agree"`` (zero spread) — see
        :func:`dispersion_grade` for the boundaries.
      dispersion_margin: how much ``relative_spread`` HEADROOM the calibration
        has before its ``dispersion_grade`` would degrade to the next-worse grade
        (iter-396). The grade says *which* trust band the renders fall in;
        ``dispersion_margin`` says *how comfortably* — an ``"agree"`` at
        ``relative_spread`` 0.049 (margin 0.001) is one noisy render from
        ``"loose"``, while one at 0.005 (margin 0.045) is rock-solid. It is the
        calibration analogue of iter-348's ``separation_ratio`` (how robustly the
        VAD valley is earned, not just which grade it gets). ``0.0`` means the
        calibration sits exactly on a grade knee; ``"scattered"`` (the worst
        grade) has no worse grade to degrade to, so its margin is ``None``.
        Computed from ``relative_spread`` alone, so it inherits that field's
        voice-independence — see :func:`dispersion_margin`.
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
    relative_spread: float
    dispersion_grade: str
    dispersion_margin: float | None
    default_base_wpm: float
    drift: float


#: ``relative_spread`` at or below which the calibration renders are deemed to
#: AGREE — the median is trustworthy. 0.05 means the per-sample implied_base_wpm
#: extremes span at most 5% of the median rate, tight enough that a re-seed rests
#: on solid ground. Chosen to sit just above the iter-222 ``spread_max`` gate at
#: a nominal voice (10 WPM / 165 ≈ 0.061), so a calibration that PASSES the
#: adopt-gate's absolute-spread test typically also reads "agree" here.
CALIB_AGREE_REL_SPREAD: float = 0.05

#: ``relative_spread`` at or below which the calibration is ``"loose"`` (above it
#: is ``"scattered"``). 0.15 means the extremes span up to 15% of the median —
#: usable as a starting point but worth a sanity-check; beyond it the renders
#: disagree enough that the median is not a reliable base.
CALIB_LOOSE_REL_SPREAD: float = 0.15


def dispersion_grade(relative_spread: float) -> str:
    """Bucket a calibration's ``relative_spread`` into a categorical trust grade.

    The calibration analogue of iter-348's :func:`vad_gap_confidence` dominance
    grade: where that grades how dominant a VAD gap valley is, this grades how
    tightly the calibration renders cluster — turning the iter-393 dimensionless
    ``relative_spread`` (spread / median) into a one-glance ``"agree"`` /
    ``"loose"`` / ``"scattered"`` read of how trustworthy the median base is.

    Boundaries (inclusive lower band first, so a render set on the knee grades
    the more favourable side):

    - ``relative_spread <= CALIB_AGREE_REL_SPREAD`` (0.05) ⇒ ``"agree"`` — the
      renders cluster within 5% of the median; the median is solid.
    - ``<= CALIB_LOOSE_REL_SPREAD`` (0.15) ⇒ ``"loose"`` — usable but worth a
      sanity-check against the per-sample CSV/JSON before re-seeding.
    - otherwise ⇒ ``"scattered"`` — the renders disagree; re-render more
      consistently before trusting the median.

    A single-sample calibration has ``relative_spread == 0.0`` and so grades
    ``"agree"`` — one timing is internally consistent (it cannot disagree with
    itself); the iter-222 verdict's ``min_samples`` gate, not this grade, is what
    flags "one render is not a calibration". Pure: a function of one float.
    """
    rs = float(relative_spread)
    if rs <= CALIB_AGREE_REL_SPREAD:
        return "agree"
    if rs <= CALIB_LOOSE_REL_SPREAD:
        return "loose"
    return "scattered"


def dispersion_margin(relative_spread: float) -> float | None:
    """How much ``relative_spread`` headroom is left before the grade degrades.

    The iter-394 :func:`dispersion_grade` answers *which* trust band a
    calibration falls in (``"agree"`` / ``"loose"`` / ``"scattered"``); this
    answers *how comfortably* it holds that band — the distance from
    ``relative_spread`` up to the knee where it would tip into the next-WORSE
    grade. It is the calibration analogue of iter-348's ``separation_ratio``,
    which grades how robustly the VAD valley is earned rather than just which
    grade it gets: an ``"agree"`` at ``relative_spread`` 0.049 (margin 0.001) is
    one noisy render away from ``"loose"``, while one at 0.005 (margin 0.045)
    sits firmly inside the band.

    Returns, by grade:

    - ``"agree"`` ⇒ ``CALIB_AGREE_REL_SPREAD - relative_spread`` — headroom to
      the agree/loose knee (0.05).
    - ``"loose"`` ⇒ ``CALIB_LOOSE_REL_SPREAD - relative_spread`` — headroom to
      the loose/scattered knee (0.15).
    - ``"scattered"`` ⇒ ``None``. The worst grade has no worse grade to degrade
      into, so "headroom to the next-worse grade" is undefined — spelled
      ``None`` the same way the rest of the calibration/gap family spells "not
      measurable" (cf. :func:`vad_gap_confidence`'s ``separation_ratio``).

    A value sitting exactly on a knee grades the more favourable side (the
    iter-394 inclusive-lower-band convention) with a ``0.0`` margin — it is in
    the better band, but one hair from leaving it. Because the result is purely
    a function of ``relative_spread`` (itself voice-independent, iter-393), the
    margin is voice-comparable: the SAME relative spread at a 100-WPM and a
    300-WPM voice yields the same margin. Pure: a function of one float.
    """
    rs = float(relative_spread)
    grade = dispersion_grade(rs)
    if grade == "agree":
        return CALIB_AGREE_REL_SPREAD - rs
    if grade == "loose":
        return CALIB_LOOSE_REL_SPREAD - rs
    return None


def calibrate_base_wpm(
    samples,
    default_base_wpm: float = DEFAULT_BASE_WPM,
) -> BaseWpmCalibration | None:
    """Fold calibration renders into a robust ``base_wpm`` verdict.

    Each :class:`CalibrationSample`'s ``implied_base_wpm`` normalizes its render
    back to the ``speed=1.0`` calibration point, so samples taken at *different*
    speeds are directly comparable. Returns their **median** as the calibrated
    ``base_wpm`` (robust to a single mis-timed render) plus spread and
    drift-vs-nominal diagnostics (including the iter-393 ``relative_spread``, the
    spread normalized by the median so it can be compared across voices, and the
    iter-394 ``dispersion_grade`` that buckets it into ``"agree"`` / ``"loose"``
    / ``"scattered"``, and the iter-396 ``dispersion_margin`` saying how much
    relative-spread headroom is left before that grade would degrade).

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
    spread = hi - lo
    # median is a positive rate (each implied_base_wpm > 0 since bot_wpm and
    # speed are both positive), so the division is always well-defined.
    relative_spread = spread / median
    return BaseWpmCalibration(
        implied_base_wpm=median,
        n_samples=len(bases),
        min_base_wpm=lo,
        max_base_wpm=hi,
        spread=spread,
        relative_spread=relative_spread,
        dispersion_grade=dispersion_grade(relative_spread),
        dispersion_margin=dispersion_margin(relative_spread),
        default_base_wpm=float(default_base_wpm),
        drift=median - float(default_base_wpm),
    )


# --------------------------------------------------------------------------
# iter-222 — data-driven verdict over a base_wpm calibration.
#
# iter-220 measured ``implied_base_wpm`` and iter-221 surfaced it on the CLI,
# but both stop at raw numbers (median, spread, drift) and leave the operator to
# eyeball whether to actually re-seed ``DEFAULT_BASE_WPM``. That is the same gap
# iter-219 closed for ``strength`` with a data-driven *verdict* rather than a
# bare grid. This is the calibration's verdict: a re-seed is worth doing only
# when the renders AGREE (small spread ⇒ the median is trustworthy), there are
# ENOUGH of them (a single render is one timing, not a calibration), AND the
# drift from nominal is large enough to MATTER (a ±2 WPM drift is noise the
# damped mirror absorbs; re-seeding for it just churns config). All three gates
# must pass to recommend adoption — otherwise keep the current nominal seed.
# Pure arithmetic over an existing ``BaseWpmCalibration``; no I/O, no clock.
# --------------------------------------------------------------------------

#: Max per-sample range (``max_base_wpm - min_base_wpm``, in WPM) for the
#: calibration to be trusted. Above this the renders disagree (inconsistent
#: synth or a bad sample), so the median is not a reliable base.
DEFAULT_CALIB_SPREAD_MAX: float = 10.0

#: Minimum absolute ``drift`` (measured median − nominal, in WPM) worth
#: re-seeding for. Below this the measured base is close enough to the nominal
#: that the damped mirror absorbs the difference, so re-seeding only churns the
#: config without changing behavior.
DEFAULT_CALIB_DRIFT_MIN: float = 5.0

#: Minimum number of samples for the median to be robust. A single render is
#: one timing, not a calibration; a couple of renders can still be a fluke.
DEFAULT_CALIB_MIN_SAMPLES: int = 3


@dataclass(frozen=True)
class CalibrationVerdict:
    """Data-driven recommendation over a :class:`BaseWpmCalibration`.

    Attributes:
      recommend: ``True`` iff all three gates pass — the renders agree
        (``spread <= spread_max``), there are enough of them
        (``n_samples >= min_samples``), and the drift is large enough to matter
        (``abs(drift) >= drift_min``). When ``True`` the operator should re-seed
        ``DEFAULT_BASE_WPM`` to ``implied_base_wpm``; when ``False`` keep the
        current nominal.
      reason: a short human-readable explanation of the decision — which gate
        failed, or that all passed.
      implied_base_wpm: the calibration's measured median (echoed for the
        caller's convenience; the value to adopt when ``recommend`` is ``True``).
      drift: the calibration's drift from nominal (echoed).
      spread: the calibration's per-sample range (echoed).
      n_samples: how many samples backed the calibration (echoed).
      dispersion_grade: the calibration's iter-394 voice-comparable trust grade
        (``"agree"`` / ``"loose"`` / ``"scattered"``), echoed from the underlying
        :class:`BaseWpmCalibration`. iter-395 also folds it into ``reason``: the
        adopt/keep call cites the grade so the decision's trust footing is
        spelled out in the same line. The grade is a *reading aid*, not a fourth
        gate — the trust gate is still the absolute ``spread <= spread_max``
        test (the grade and the gate agree by construction, see
        :func:`dispersion_grade`); naming it in ``reason`` just makes the
        voice-independent view of that same trust visible at a glance.
      spread_max / drift_min / min_samples: the thresholds the verdict was
        computed against (echoed so the decision is self-describing).
    """

    recommend: bool
    reason: str
    implied_base_wpm: float
    drift: float
    spread: float
    n_samples: int
    dispersion_grade: str
    spread_max: float
    drift_min: float
    min_samples: int


def calibration_verdict(
    calibration: "BaseWpmCalibration | None",
    *,
    spread_max: float = DEFAULT_CALIB_SPREAD_MAX,
    drift_min: float = DEFAULT_CALIB_DRIFT_MIN,
    min_samples: int = DEFAULT_CALIB_MIN_SAMPLES,
) -> CalibrationVerdict | None:
    """Decide whether to re-seed ``DEFAULT_BASE_WPM`` from a calibration.

    Folds the three trust/significance gates over an existing
    :class:`BaseWpmCalibration` (the iter-220 measurement). A re-seed is
    recommended only when **all** of:

    - **enough samples** — ``n_samples >= min_samples`` (a single render is one
      timing, not a calibration);
    - **renders agree** — ``spread <= spread_max`` (a wide spread means the
      median is not trustworthy);
    - **drift matters** — ``abs(drift) >= drift_min`` (a tiny drift is noise the
      damped mirror absorbs, so re-seeding only churns config).

    The gates are checked in that order so ``reason`` names the *first* failure
    (sample count is the most fundamental, then trust, then significance).

    iter-395 echoes the calibration's iter-394 ``dispersion_grade`` on the
    verdict and cites it in the two trust-themed ``reason`` branches (the
    spread-pass recommend and the spread-fail rejection), so the adopt/keep call
    spells out its voice-comparable trust footing. The grade is a reading aid,
    not a fourth gate — the trust gate remains the absolute ``spread`` test, and
    the grade agrees with it by construction (see :func:`dispersion_grade`).

    Args:
      calibration: a :class:`BaseWpmCalibration`, or ``None`` (no samples ⇒
        nothing to decide ⇒ this function returns ``None``, mirroring
        :func:`calibrate_base_wpm`'s empty contract).
      spread_max: max trusted per-sample range (defaults to
        :data:`DEFAULT_CALIB_SPREAD_MAX`).
      drift_min: min absolute drift worth re-seeding for (defaults to
        :data:`DEFAULT_CALIB_DRIFT_MIN`).
      min_samples: min sample count for a robust median (defaults to
        :data:`DEFAULT_CALIB_MIN_SAMPLES`).

    Pure — reads only the calibration's fields, mutates nothing.
    """
    if calibration is None:
        return None

    spread_max = float(spread_max)
    drift_min = float(drift_min)
    min_samples = int(min_samples)

    n = calibration.n_samples
    spread = calibration.spread
    drift = calibration.drift
    grade = calibration.dispersion_grade

    if n < min_samples:
        recommend = False
        reason = (
            f"only {n} sample(s) — need {min_samples}+ for a robust median; "
            "keep the current nominal"
        )
    elif spread > spread_max:
        recommend = False
        reason = (
            f"renders disagree (spread {spread:.1f} > {spread_max:.1f} WPM, "
            f"dispersion {grade}) — the median is not trustworthy; re-render "
            "more consistently"
        )
    elif abs(drift) < drift_min:
        recommend = False
        reason = (
            f"drift {drift:+.1f} WPM is below the {drift_min:.1f} WPM threshold "
            "— the damped mirror absorbs it; keep the current nominal"
        )
    else:
        recommend = True
        reason = (
            f"renders agree (spread {spread:.1f} <= {spread_max:.1f}, "
            f"dispersion {grade}) over {n} samples and drift {drift:+.1f} WPM "
            f"is significant — re-seed base_wpm to "
            f"{calibration.implied_base_wpm:.1f}"
        )

    return CalibrationVerdict(
        recommend=recommend,
        reason=reason,
        implied_base_wpm=calibration.implied_base_wpm,
        drift=drift,
        spread=spread,
        n_samples=n,
        dispersion_grade=grade,
        spread_max=spread_max,
        drift_min=drift_min,
        min_samples=min_samples,
    )


# --------------------------------------------------------------------------
# iter-397 — batch calibration over a CORPUS of voices.
#
# Where :func:`calibrate_base_wpm` folds the renders of ONE voice into a single
# median, this generalises to N voices: calibrate each voice's renders
# independently and tabulate their implied_base_wpm / dispersion grade / margin /
# drift in one structure, plus an outlier-robust corpus median so an operator can
# see at a glance which voices AGREE on a base rate and which are OUTLIERS. It is
# the calibration analogue of how ``gv vad-gap-recommend-batch`` (iter-385)
# generalises ``gv vad-gap-recommend``: the single voice is the one-row special
# case, the batch is the full corpus. Each row keeps its own voice-comparable
# dispersion grade (a per-voice property), so a voice whose implied_base_wpm is an
# outlier but reads ``"scattered"`` is plausibly just a noisy render set while one
# that reads ``"agree"`` is a real disagreement worth chasing.
# --------------------------------------------------------------------------

#: Canonical descending-trust order for the batch dispersion-grade histogram. The
#: three real grades (:func:`dispersion_grade`) plus ``uncalibrated`` — the bucket
#: for a voice with NO samples (its calibration is ``None``, so it has no grade),
#: kept distinct from any real grade so an empty voice never silently merges into
#: ``"scattered"``. Mirrors :data:`GAP_RECOMMEND_BATCH_GRADE_ORDER` in ``gv.py``.
CALIB_BATCH_GRADE_ORDER = ("agree", "loose", "scattered", "uncalibrated")


@dataclass(frozen=True)
class BaseWpmCalibrationBatch:
    """Verdict of calibrating a CORPUS of voices (iter-397).

    The batch analogue of :class:`BaseWpmCalibration`: one
    :class:`BaseWpmCalibration` per voice (``rows``), summarised by the
    outlier-robust corpus median of the per-voice ``implied_base_wpm``.

    Attributes:
      rows: one entry per voice, in input order, each a dict with keys
        ``voice`` (the label), ``calibration`` (the voice's
        :class:`BaseWpmCalibration`, or ``None`` when the voice had no samples),
        and ``delta_from_median_wpm`` (the voice's ``implied_base_wpm`` minus the
        corpus median, ``None`` for an uncalibrated voice). The per-voice grade /
        margin / drift live on the embedded ``calibration`` so each row agrees
        EXACTLY with ``gv calibrate-base-wpm`` on that voice.
      num_voices: how many voices were submitted (``len(rows)``).
      num_calibrated: how many carried at least one sample (fed the corpus
        aggregates); an uncalibrated voice contributes a row but not a number.
      implied_base_wpm_median: the outlier-robust **median** of the calibrated
        voices' ``implied_base_wpm`` — the rate a fleet-wide default would sit at
        (``None`` when no voice calibrated). Median (not mean) so one outlier
        voice cannot drag the corpus centre.
      implied_base_wpm_min / implied_base_wpm_max: the extremes of the
        per-voice ``implied_base_wpm`` (``None`` when no voice calibrated).
      implied_base_wpm_spread: ``max - min`` of the per-voice medians — how far
        apart the voices clock (``None`` when no voice calibrated). A large
        spread means the corpus's voices genuinely differ in rate, so a single
        fleet-wide ``DEFAULT_BASE_WPM`` would mis-serve the extremes.
      grade_counts: how many voices sit at each dispersion grade, keyed by
        :data:`CALIB_BATCH_GRADE_ORDER` (always all four buckets, summing to
        ``num_voices``) — the corpus's trust profile at a glance.
      default_base_wpm: the nominal seed each voice's drift was measured against
        (echoed; shared by every voice so the drifts are comparable).

    Pure / frozen; built from :func:`calibrate_base_wpm` per voice.
    """

    rows: tuple
    num_voices: int
    num_calibrated: int
    implied_base_wpm_median: float | None
    implied_base_wpm_min: float | None
    implied_base_wpm_max: float | None
    implied_base_wpm_spread: float | None
    grade_counts: dict
    default_base_wpm: float


def calibrate_base_wpm_batch(
    voices,
    default_base_wpm: float = DEFAULT_BASE_WPM,
) -> BaseWpmCalibrationBatch:
    """Calibrate a corpus of voices and tabulate their base rates (iter-397).

    ``voices`` is an iterable of ``(label, samples)`` pairs — one per voice,
    where ``samples`` is that voice's iterable of :class:`CalibrationSample`
    renders (the same input :func:`calibrate_base_wpm` takes). Each voice is
    calibrated independently against the shared ``default_base_wpm`` so the
    per-voice drifts are apples-to-apples, and the per-voice ``implied_base_wpm``
    values are summarised by their outlier-robust median.

    A voice with no samples calibrates to ``None`` (the
    :func:`calibrate_base_wpm` empty contract): it contributes a row tagged
    ``uncalibrated`` and is excluded from the corpus median / extremes / spread,
    but is still counted in ``num_voices``. This mirrors how
    ``vad_gap_recommend_batch`` keeps a <2-segment recording in the table with a
    ``None`` recommendation rather than dropping it silently.

    Returns a :class:`BaseWpmCalibrationBatch`. Pure — no I/O, no clock, no
    mutation of the inputs.
    """
    rows = []
    calibrated = []
    for label, samples in voices:
        calib = calibrate_base_wpm(samples, default_base_wpm=default_base_wpm)
        rows.append(
            {
                "voice": label,
                "calibration": calib,
                # delta filled in once the corpus median is known.
                "delta_from_median_wpm": None,
            }
        )
        if calib is not None:
            calibrated.append(calib.implied_base_wpm)

    if calibrated:
        median = statistics.median(calibrated)
        lo = min(calibrated)
        hi = max(calibrated)
        spread = hi - lo
        for r in rows:
            calib = r["calibration"]
            if calib is not None:
                r["delta_from_median_wpm"] = calib.implied_base_wpm - median
    else:
        median = lo = hi = spread = None

    counts = {g: 0 for g in CALIB_BATCH_GRADE_ORDER}
    for r in rows:
        calib = r["calibration"]
        key = "uncalibrated" if calib is None else calib.dispersion_grade
        if key in counts:
            counts[key] += 1
        else:
            # Defensive: an unrecognised future grade lands in uncalibrated rather
            # than vanishing — the counts must still sum to num_voices.
            counts["uncalibrated"] += 1

    return BaseWpmCalibrationBatch(
        rows=tuple(rows),
        num_voices=len(rows),
        num_calibrated=len(calibrated),
        implied_base_wpm_median=median,
        implied_base_wpm_min=lo,
        implied_base_wpm_max=hi,
        implied_base_wpm_spread=spread,
        grade_counts=counts,
        default_base_wpm=float(default_base_wpm),
    )
