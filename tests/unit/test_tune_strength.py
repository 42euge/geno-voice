"""Tests for iter-219 — the canonical in-band tuning corpus + strength verdict.

This closes the backlog item repeated across iter-216/217/218: *run* the offline
grid sweep on a corpus whose per-turn rates stay inside the intelligibility band
so ``final_gap`` measures real pacing-tracking rather than the
``min_speed``/``max_speed`` clamp, then either change the seed defaults from the
verdict or document why they stand.

Two findings, both pinned here:

1. ``base_wpm`` is NOT tunable offline — the simulator *uses* it to define the
   convergence target (``ideal = user_wpm / base_wpm``), so a base_wpm sweep
   scores each cell against its own moving target. Only ``strength`` (pure
   convergence dynamics at a fixed base) is answerable by replay, which is what
   :func:`tune_strength` returns.

2. The seed ``strength=0.5`` wins the fair test on ``TUNING_CORPUS_WPMS`` — the
   knee between lagging (0.3/0.4, deadband blocks tracking) and lurching
   (0.6/0.7). So the seed default stands, now from data.

Like the rest of the wpm_mirror suite, the module is loaded by file path to
bypass ``session/__init__``'s eager pipecat import (not installable on this
x86_64 Linux runner).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_WM_PATH = Path(__file__).resolve().parents[2] / "session" / "wpm_mirror.py"
_spec = importlib.util.spec_from_file_location("_wm_tune_under_test", _WM_PATH)
_wm = importlib.util.module_from_spec(_spec)
sys.modules["_wm_tune_under_test"] = _wm
_spec.loader.exec_module(_wm)

WpmMirrorConfig = _wm.WpmMirrorConfig
simulate_speed_trajectory = _wm.simulate_speed_trajectory
sweep_mirror_grid = _wm.sweep_mirror_grid
pick_best_mirror_config = _wm.pick_best_mirror_config
tune_strength = _wm.tune_strength
TUNING_CORPUS_WPMS = _wm.TUNING_CORPUS_WPMS
TUNING_STRENGTH_AXIS = _wm.TUNING_STRENGTH_AXIS
DEFAULT_BASE_WPM = _wm.DEFAULT_BASE_WPM
DEFAULT_STRENGTH = _wm.DEFAULT_STRENGTH
DEFAULT_MIN_SPEED = _wm.DEFAULT_MIN_SPEED
DEFAULT_MAX_SPEED = _wm.DEFAULT_MAX_SPEED


# The three grid bases the iter-218 CLI defaults to; the corpus is engineered to
# stay in band for all of them at once.
GRID_BASES = (150.0, 165.0, 180.0)


# --------------------------------------------------------------------------
# The corpus itself: every rate inside the band at every grid base.
# --------------------------------------------------------------------------


def test_corpus_is_in_band_for_every_grid_base():
    """Every WPM is inside ``[0.8·base, 1.3·base]`` at base 150/165/180.

    This is the whole point of the corpus: ``final_gap`` measures tracking, not
    the intelligibility clamp. If a future edit pushes a rate out of band at any
    grid base, the verdict would become a clamp artifact (the iter-217 bug) — so
    pin the invariant.
    """
    for base in GRID_BASES:
        lo = DEFAULT_MIN_SPEED * base
        hi = DEFAULT_MAX_SPEED * base
        for w in TUNING_CORPUS_WPMS:
            assert lo <= w <= hi, (
                f"WPM {w} outside band [{lo:.0f},{hi:.0f}] at base {base:.0f}"
            )


def test_corpus_actually_varies_and_ends_sustained():
    """The corpus is a real slow→fast→slow arc with a sustained tail.

    A flat or monotone arc would not exercise lurch (a held speed never lurches)
    and a tail that is not sustained would not let candidates converge — both
    would make the strength ranking meaningless. Pin the shape: multiple
    distinct rates, and the last three turns equal (the sustained tail).
    """
    assert len(set(TUNING_CORPUS_WPMS)) >= 3
    assert TUNING_CORPUS_WPMS[-1] == TUNING_CORPUS_WPMS[-2] == TUNING_CORPUS_WPMS[-3]


def test_no_corpus_turn_clamps_in_a_converged_trajectory():
    """Folding the corpus at the seed config never hits the band clamp.

    The trajectory speeds must all stay strictly inside the open band — if any
    turn's speed pegged at ``min_speed``/``max_speed`` the convergence target
    would be a clamp artifact rather than a tracked rate.
    """
    cfg = WpmMirrorConfig(enabled=True, base_wpm=DEFAULT_BASE_WPM, strength=DEFAULT_STRENGTH)
    traj = simulate_speed_trajectory(TUNING_CORPUS_WPMS, config=cfg)
    for s in traj.speeds:
        assert DEFAULT_MIN_SPEED < s < DEFAULT_MAX_SPEED


# --------------------------------------------------------------------------
# tune_strength: the data-driven verdict.
# --------------------------------------------------------------------------


def test_seed_strength_wins_the_fair_test():
    """The data-driven verdict at the seed base is the seed ``strength``.

    This is the deliverable: running the real sweep on the in-band corpus picks
    ``strength=0.5``, so the seed default STANDS from data, not assertion.
    """
    best = tune_strength()
    assert best is not None
    assert best.strength == DEFAULT_STRENGTH  # 0.5
    assert best.base_wpm == DEFAULT_BASE_WPM  # 165


def test_tune_strength_equals_an_explicit_fixed_base_sweep():
    """``tune_strength`` is exactly a one-row sweep + pick at the fixed base."""
    points = sweep_mirror_grid(
        TUNING_CORPUS_WPMS, [DEFAULT_BASE_WPM], TUNING_STRENGTH_AXIS
    )
    expected = pick_best_mirror_config(points)
    best = tune_strength()
    assert best == expected


def test_tune_strength_only_sweeps_the_one_base():
    """The verdict cell carries the fixed base, never another — base is held."""
    best = tune_strength(base_wpm=150.0)
    assert best is not None
    assert best.base_wpm == 150.0


def test_low_strength_lags_high_strength_lurches():
    """The knee is real: 0.3 leaves a bigger gap, 0.7 takes a bigger step.

    Pins *why* 0.5 wins — it is between a laggy small-strength config (large
    residual ``|final_gap|``) and a lurchy large-strength one (large
    ``max_step``) — so a future map change that flattens the knee fails.
    """
    pts = {
        p.strength: p
        for p in sweep_mirror_grid(
            TUNING_CORPUS_WPMS, [DEFAULT_BASE_WPM], [0.3, 0.5, 0.7]
        )
    }
    # 0.3 lags worse than 0.5.
    assert pts[0.3].abs_final_gap > pts[0.5].abs_final_gap
    # 0.7 lurches worse than 0.5.
    assert pts[0.7].max_step > pts[0.5].max_step


def test_tune_strength_custom_strengths_and_arc():
    """Caller can override the corpus and the strength axis."""
    arc = [180.0, 180.0, 180.0, 180.0]
    best = tune_strength(user_wpms=arc, strengths=[0.5, 1.0])
    assert best is not None
    assert best.strength in (0.5, 1.0)


def test_tune_strength_none_when_no_measurable_turn():
    """An all-silent arc has no scorable cell ⇒ ``None`` (no verdict)."""
    assert tune_strength(user_wpms=[0.0, -1.0, 0.0]) is None


def test_tune_strength_does_not_mutate_module_constants():
    """The tuner is pure — calling it leaves the shared constants untouched."""
    corpus_before = tuple(TUNING_CORPUS_WPMS)
    axis_before = tuple(TUNING_STRENGTH_AXIS)
    tune_strength()
    assert tuple(TUNING_CORPUS_WPMS) == corpus_before
    assert tuple(TUNING_STRENGTH_AXIS) == axis_before


def test_tune_strength_is_deterministic():
    """Same inputs ⇒ same verdict cell."""
    a = tune_strength()
    b = tune_strength()
    assert a == b


def test_lurch_weight_threads_through():
    """A heavy lurch penalty shifts the verdict toward a smoother (lower) strength.

    With ``lurch_weight`` cranked up, ``max_step`` dominates ``|final_gap|``, so
    the verdict should favour a lower (smoother) strength than the balanced pick.
    """
    balanced = tune_strength(lurch_weight=0.5)
    lurchy_penalized = tune_strength(lurch_weight=5.0)
    assert balanced is not None and lurchy_penalized is not None
    assert lurchy_penalized.strength <= balanced.strength
