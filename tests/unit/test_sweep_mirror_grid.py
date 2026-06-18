"""Tests for iter-217 — the offline base_wpm × strength grid sweep + picker.

iter-216 shipped ``simulate_speed_trajectory`` (the single-config offline twin
of the live ``SpeedController`` fold). Its backlog item #1 is to *run* it over a
realistic varied-pacing arc across a grid of ``base_wpm`` × ``strength``
candidates, read each cell's convergence / lurch / churn diagnostics, and pick
the pair that tracks the user without lurching — so the seed defaults can be
chosen from data rather than asserted.

``session.wpm_mirror.sweep_mirror_grid`` folds the shared arc through every
grid cell, and ``pick_best_mirror_config`` ranks the cells by a lower-is-better
score. These tests pin that the sweep is the faithful grid analogue of the
single-config simulator, that the score balances convergence against lurch, and
that the picker honours the earliest-tie rule.

Like the rest of the wpm_mirror suite, the module is loaded by file path to
bypass ``session/__init__``'s eager pipecat import (not installable on this
x86_64 Linux runner).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_WM_PATH = Path(__file__).resolve().parents[2] / "session" / "wpm_mirror.py"
_spec = importlib.util.spec_from_file_location("_wm_grid_under_test", _WM_PATH)
_wm = importlib.util.module_from_spec(_spec)
sys.modules["_wm_grid_under_test"] = _wm
_spec.loader.exec_module(_wm)

WpmMirrorConfig = _wm.WpmMirrorConfig
simulate_speed_trajectory = _wm.simulate_speed_trajectory
sweep_mirror_grid = _wm.sweep_mirror_grid
pick_best_mirror_config = _wm.pick_best_mirror_config
MirrorGridPoint = _wm.MirrorGridPoint
DEFAULT_LURCH_WEIGHT = _wm.DEFAULT_LURCH_WEIGHT


# A realistic slow → fast → slow conversational arc (WPM per turn).
ARC = [120.0, 140.0, 200.0, 230.0, 200.0, 140.0, 120.0]


# --------------------------------------------------------------------------
# Grid shape: one cell per (base_wpm, strength), row-major.
# --------------------------------------------------------------------------

class TestGridShape:
    def test_one_point_per_cell_row_major(self):
        pts = sweep_mirror_grid(ARC, [150.0, 165.0], [0.3, 0.5, 0.7])
        assert len(pts) == 2 * 3
        # row-major: outer base_wpm, inner strength
        assert (pts[0].base_wpm, pts[0].strength) == (150.0, 0.3)
        assert (pts[1].base_wpm, pts[1].strength) == (150.0, 0.5)
        assert (pts[2].base_wpm, pts[2].strength) == (150.0, 0.7)
        assert (pts[3].base_wpm, pts[3].strength) == (165.0, 0.3)
        assert (pts[5].base_wpm, pts[5].strength) == (165.0, 0.7)

    def test_empty_grid_axes_yield_no_points(self):
        assert sweep_mirror_grid(ARC, [], [0.5]) == []
        assert sweep_mirror_grid(ARC, [165.0], []) == []

    def test_accepts_int_axes(self):
        pts = sweep_mirror_grid([198, 198], [165], [1])
        assert pts[0].base_wpm == 165.0
        assert pts[0].strength == 1.0


# --------------------------------------------------------------------------
# Each cell matches the single-config simulator exactly.
# --------------------------------------------------------------------------

class TestMatchesSimulator:
    def test_cell_equals_direct_simulation(self):
        base_wpm, strength = 165.0, 0.5
        pts = sweep_mirror_grid(ARC, [base_wpm], [strength])
        p = pts[0]

        cfg = WpmMirrorConfig(enabled=True, base_wpm=base_wpm, strength=strength)
        traj = simulate_speed_trajectory(ARC, 1.0, cfg)

        assert p.final_speed == traj.final_speed
        assert p.ideal_final_speed == traj.ideal_final_speed
        assert p.final_gap == traj.final_gap
        assert p.max_step == traj.max_step
        assert p.moves == traj.moves

    def test_initial_speed_threaded_through(self):
        pts_a = sweep_mirror_grid(ARC, [165.0], [0.5], initial_speed=1.0)
        pts_b = sweep_mirror_grid(ARC, [165.0], [0.5], initial_speed=1.2)
        # different start ⇒ different trajectory (band/deadband aside)
        assert pts_a[0].final_speed != pts_b[0].final_speed or \
            pts_a[0].max_step != pts_b[0].max_step

    def test_template_band_and_deadband_reused(self):
        tmpl = WpmMirrorConfig(min_speed=0.5, max_speed=2.0, min_delta=0.0)
        # 660 WPM / 165 = 4.0 ideal; with the wide template it clamps to 2.0,
        # with the seed template it would clamp to 1.3.
        wide = sweep_mirror_grid([660.0], [165.0], [1.0], template=tmpl)
        seed = sweep_mirror_grid([660.0], [165.0], [1.0])
        assert wide[0].ideal_final_speed == pytest.approx(2.0)
        assert seed[0].ideal_final_speed == pytest.approx(1.3)


# --------------------------------------------------------------------------
# Score: |final_gap| + lurch_weight * max_step (lower is better).
# --------------------------------------------------------------------------

class TestScore:
    def test_score_formula(self):
        p = MirrorGridPoint(
            base_wpm=165.0, strength=0.5, final_speed=1.1,
            ideal_final_speed=1.2, final_gap=-0.1, max_step=0.2, moves=3,
        )
        assert p.abs_final_gap == pytest.approx(0.1)
        assert p.score(lurch_weight=0.5) == pytest.approx(0.1 + 0.5 * 0.2)
        assert p.score(lurch_weight=0.0) == pytest.approx(0.1)

    def test_unscorable_cell_returns_none(self):
        p = MirrorGridPoint(
            base_wpm=165.0, strength=0.5, final_speed=1.0,
            ideal_final_speed=None, final_gap=None, max_step=0.0, moves=0,
        )
        assert p.abs_final_gap is None
        assert p.score() is None

    def test_default_lurch_weight_is_half(self):
        assert DEFAULT_LURCH_WEIGHT == 0.5


# --------------------------------------------------------------------------
# Picker: lowest score wins, earliest-tie, skip unscorable.
# --------------------------------------------------------------------------

class TestPicker:
    def test_picks_lowest_score(self):
        # A sustained fast rate: a higher strength converges with a smaller
        # residual gap over a short sequence, so it should win on score.
        arc = [198.0] * 4
        pts = sweep_mirror_grid(arc, [165.0], [0.3, 0.5, 0.9],
                                template=WpmMirrorConfig(min_delta=0.0))
        best = pick_best_mirror_config(pts, lurch_weight=0.0)  # gap-only
        # gap-only score ⇒ the strongest damping converges closest.
        assert best.strength == 0.9

    def test_lurch_weight_penalizes_jumpy_cells(self):
        arc = [198.0] * 4
        pts = sweep_mirror_grid(arc, [165.0], [0.3, 0.5, 0.9],
                                template=WpmMirrorConfig(min_delta=0.0))
        # With a heavy lurch penalty the gentle cell can beat the jumpy one.
        gap_only = pick_best_mirror_config(pts, lurch_weight=0.0)
        heavy = pick_best_mirror_config(pts, lurch_weight=5.0)
        assert heavy.max_step <= gap_only.max_step

    def test_empty_grid_returns_none(self):
        assert pick_best_mirror_config([]) is None

    def test_all_unscorable_returns_none(self):
        # no measurable turn ⇒ every cell unscorable
        pts = sweep_mirror_grid([0.0, -1.0], [150.0, 165.0], [0.3, 0.5])
        assert all(p.score() is None for p in pts)
        assert pick_best_mirror_config(pts) is None

    def test_earliest_tie_wins(self):
        # Two hand-built cells with identical scores: the earlier one wins.
        a = MirrorGridPoint(base_wpm=150.0, strength=0.5, final_speed=1.1,
                            ideal_final_speed=1.2, final_gap=-0.1,
                            max_step=0.2, moves=1)
        b = MirrorGridPoint(base_wpm=165.0, strength=0.5, final_speed=1.1,
                            ideal_final_speed=1.2, final_gap=-0.1,
                            max_step=0.2, moves=1)
        assert pick_best_mirror_config([a, b]) is a
        assert pick_best_mirror_config([b, a]) is b


# --------------------------------------------------------------------------
# A data-driven verdict over the realistic arc — the backlog #1 motivation.
# --------------------------------------------------------------------------

class TestDataDrivenPick:
    def test_seed_defaults_are_in_the_grid_and_pickable(self):
        # The grid the tuning lap would run: base 150/165/180 × strength
        # 0.3/0.5/0.7, over the slow→fast→slow arc.
        pts = sweep_mirror_grid(ARC, [150.0, 165.0, 180.0], [0.3, 0.5, 0.7])
        best = pick_best_mirror_config(pts)
        assert best is not None
        # the winner is one of the grid cells
        assert (best.base_wpm, best.strength) in {
            (b, s) for b in [150.0, 165.0, 180.0] for s in [0.3, 0.5, 0.7]
        }
        # and it is genuinely the minimum-score cell
        scored = [(p.score(), p) for p in pts if p.score() is not None]
        assert best.score() == min(s for s, _ in scored)

    def test_best_converges_within_band(self):
        pts = sweep_mirror_grid(ARC, [150.0, 165.0, 180.0], [0.3, 0.5, 0.7])
        best = pick_best_mirror_config(pts)
        # the picked cell's final speed stays intelligible
        assert 0.8 <= best.final_speed <= 1.3


# --------------------------------------------------------------------------
# Purity.
# --------------------------------------------------------------------------

class TestPurity:
    def test_does_not_mutate_inputs(self):
        arc = list(ARC)
        bases = [150.0, 165.0]
        strengths = [0.3, 0.5]
        sweep_mirror_grid(arc, bases, strengths)
        assert arc == ARC
        assert bases == [150.0, 165.0]
        assert strengths == [0.3, 0.5]

    def test_deterministic(self):
        a = sweep_mirror_grid(ARC, [150.0, 165.0], [0.3, 0.5])
        b = sweep_mirror_grid(ARC, [150.0, 165.0], [0.3, 0.5])
        assert [p.score() for p in a] == [p.score() for p in b]
