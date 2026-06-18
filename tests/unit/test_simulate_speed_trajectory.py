"""Tests for iter-216 — the offline WPM-mirror speed-trajectory simulator.

``session.wpm_mirror.simulate_speed_trajectory`` is the tool backlog item #1
(from iter-215's "next planned items") needs to *validate* the mirror's
``base_wpm`` / ``strength`` tunables. iter-213 shipped the pure mirror,
iter-214 wired it live, iter-215 surfaced the per-session drift in the summary
— the loop adapts and measures, but the tunables are still the seed defaults.

The simulator replays a sequence of per-turn ``user_wpm`` values through the
*exact same* fold the live ``SpeedController.observe`` runs (each turn's output
speed becomes the next turn's ``current_speed``), so its convergence / lurch /
churn verdict on a slow→fast→slow arc transfers to the live path. These tests
pin that it is the faithful offline twin of the live loop and reports the right
diagnostics.

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
_spec = importlib.util.spec_from_file_location("_wm_sim_under_test", _WM_PATH)
_wm = importlib.util.module_from_spec(_spec)
sys.modules["_wm_sim_under_test"] = _wm
_spec.loader.exec_module(_wm)

WpmMirrorConfig = _wm.WpmMirrorConfig
mirrored_speed = _wm.mirrored_speed
simulate_speed_trajectory = _wm.simulate_speed_trajectory
SpeedTrajectory = _wm.SpeedTrajectory
DEFAULT_BASE_WPM = _wm.DEFAULT_BASE_WPM


def _enabled(**kw):
    """A mirroring-enabled config with overridable tunables."""
    base = dict(enabled=True, base_wpm=165.0, strength=0.5,
                min_speed=0.8, max_speed=1.3, min_delta=0.05)
    base.update(kw)
    return WpmMirrorConfig(**base)


# --------------------------------------------------------------------------
# The off-by-default invariant carries into the simulator.
# --------------------------------------------------------------------------

class TestDisabled:
    def test_disabled_config_holds_speed_flat(self):
        traj = simulate_speed_trajectory(
            [120.0, 240.0, 60.0], initial_speed=1.0,
        )  # default config is disabled
        assert traj.speeds == [1.0, 1.0, 1.0]
        assert traj.final_speed == 1.0
        assert traj.initial_speed == 1.0
        assert traj.max_step == 0.0
        assert traj.moves == 0

    def test_disabled_has_no_convergence_target(self):
        traj = simulate_speed_trajectory([120.0, 240.0])
        assert traj.ideal_final_speed is None
        assert traj.final_gap is None

    def test_disabled_from_nonunity_initial(self):
        traj = simulate_speed_trajectory([200.0], initial_speed=1.1)
        assert traj.speeds == [1.1]
        assert traj.final_speed == 1.1


# --------------------------------------------------------------------------
# Empty / degenerate sequences.
# --------------------------------------------------------------------------

class TestEmpty:
    def test_empty_sequence_returns_initial(self):
        traj = simulate_speed_trajectory([], initial_speed=0.95, config=_enabled())
        assert traj.speeds == []
        assert traj.final_speed == 0.95
        assert traj.initial_speed == 0.95
        assert traj.max_step == 0.0
        assert traj.moves == 0
        assert traj.ideal_final_speed is None
        assert traj.final_gap is None

    def test_all_nonmeasurable_holds_and_no_target(self):
        # iter-064 "no measurement" guard: user_wpm <= 0 carries no signal.
        traj = simulate_speed_trajectory([0.0, -3.0, 0.0], config=_enabled())
        assert traj.speeds == [1.0, 1.0, 1.0]
        assert traj.moves == 0
        assert traj.ideal_final_speed is None
        assert traj.final_gap is None


# --------------------------------------------------------------------------
# The fold matches the live SpeedController loop, turn-by-turn.
# --------------------------------------------------------------------------

class TestMatchesLiveFold:
    def test_threads_output_speed_into_next_turn(self):
        cfg = _enabled(min_delta=0.0)  # no deadband so every step shows
        wpms = [220.0, 220.0, 220.0]
        traj = simulate_speed_trajectory(wpms, initial_speed=1.0, config=cfg)

        # Reproduce the fold by hand using mirrored_speed directly.
        s = 1.0
        expected = []
        for w in wpms:
            s = mirrored_speed(w, s, cfg)
            expected.append(s)
        assert traj.speeds == expected

    def test_sustained_rate_converges_toward_ideal(self):
        # A user holding 198 WPM: ideal = 198/165 = 1.2. With strength 0.5 the
        # speed should walk monotonically up toward 1.2 and the final gap small.
        cfg = _enabled(min_delta=0.0)
        traj = simulate_speed_trajectory([198.0] * 8, initial_speed=1.0, config=cfg)
        assert traj.ideal_final_speed == pytest.approx(1.2)
        # Monotone non-decreasing toward the target.
        for a, b in zip(traj.speeds, traj.speeds[1:]):
            assert b >= a
        assert traj.final_speed <= 1.2 + 1e-9
        assert abs(traj.final_gap) < 0.01

    def test_nonmeasurable_turn_holds_within_sequence(self):
        cfg = _enabled(min_delta=0.0)
        # measurable, then a no-signal turn, then measurable again
        traj = simulate_speed_trajectory([198.0, 0.0, 198.0], config=cfg)
        # Turn 2 (index 1) holds the turn-1 speed.
        assert traj.speeds[1] == traj.speeds[0]
        # Turn 3 resumes moving toward the ideal.
        assert traj.speeds[2] >= traj.speeds[1]


# --------------------------------------------------------------------------
# Convergence target is set by the LAST measurable rate.
# --------------------------------------------------------------------------

class TestConvergenceTarget:
    def test_ideal_tracks_last_measurable_rate(self):
        cfg = _enabled()
        # ends slow (99 WPM -> ideal 0.6 -> clamped to min_speed 0.8)
        traj = simulate_speed_trajectory([231.0, 99.0], config=cfg)
        assert traj.ideal_final_speed == pytest.approx(0.8)  # clamped to min

    def test_ideal_skips_trailing_nonmeasurable(self):
        cfg = _enabled()
        traj = simulate_speed_trajectory([198.0, 0.0, -1.0], config=cfg)
        # last measurable is 198 -> ideal 1.2
        assert traj.ideal_final_speed == pytest.approx(1.2)

    def test_ideal_clamped_to_band(self):
        cfg = _enabled()
        # 660 WPM -> ideal 4.0 -> clamped to max_speed 1.3
        traj = simulate_speed_trajectory([660.0], config=cfg)
        assert traj.ideal_final_speed == pytest.approx(1.3)


# --------------------------------------------------------------------------
# Lurch / churn diagnostics.
# --------------------------------------------------------------------------

class TestDiagnostics:
    def test_max_step_records_largest_single_jump(self):
        cfg = _enabled(min_delta=0.0, strength=1.0)  # jump straight to ideal
        # 1.0 -> 1.2 (step 0.2) -> back to 0.8 (clamped, step 0.4)
        traj = simulate_speed_trajectory([198.0, 60.0], initial_speed=1.0, config=cfg)
        assert traj.max_step == pytest.approx(0.4)

    def test_moves_counts_changed_turns_only(self):
        cfg = _enabled(min_delta=0.0)
        # second 165 is exactly at ideal once speed reaches 1.0 — but starting
        # at 1.0 with user at base_wpm holds, so 0 moves.
        traj = simulate_speed_trajectory([165.0, 165.0], initial_speed=1.0, config=cfg)
        assert traj.moves == 0
        assert traj.max_step == 0.0

    def test_deadband_suppresses_small_moves(self):
        # A user only slightly off base (170 vs 165): ideal 1.03, half-step to
        # ~1.015 — under the 0.05 deadband, so it holds and counts no move.
        cfg = _enabled()  # min_delta 0.05
        traj = simulate_speed_trajectory([170.0], initial_speed=1.0, config=cfg)
        assert traj.moves == 0
        assert traj.speeds == [1.0]

    def test_strength_one_lurches_more_than_half(self):
        fast = _enabled(min_delta=0.0, strength=1.0)
        slow = _enabled(min_delta=0.0, strength=0.5)
        seq = [198.0]
        tf = simulate_speed_trajectory(seq, config=fast)
        ts = simulate_speed_trajectory(seq, config=slow)
        assert tf.max_step > ts.max_step


# --------------------------------------------------------------------------
# A realistic slow -> fast -> slow conversational arc.
# --------------------------------------------------------------------------

class TestVariedPacingArc:
    def test_arc_tracks_pacing_without_exceeding_band(self):
        cfg = _enabled(min_delta=0.0)
        arc = [120.0, 140.0, 200.0, 230.0, 200.0, 140.0, 120.0]
        traj = simulate_speed_trajectory(arc, initial_speed=1.0, config=cfg)
        # never leaves the intelligibility band
        for s in traj.speeds:
            assert cfg.min_speed <= s <= cfg.max_speed
        # ends converging back toward the slow tail (120 -> ideal ~0.8 clamp)
        assert traj.ideal_final_speed == pytest.approx(0.8)
        # the speed at the fast peak is higher than at the slow tail
        assert max(traj.speeds) > traj.speeds[-1]

    def test_higher_base_wpm_yields_slower_speeds(self):
        # base_wpm is the calibration: a higher base means a given user rate
        # maps to a lower speed. Pin the monotone relationship a tuning lap relies on.
        arc = [200.0] * 6
        low = simulate_speed_trajectory(arc, config=_enabled(min_delta=0.0, base_wpm=150.0))
        high = simulate_speed_trajectory(arc, config=_enabled(min_delta=0.0, base_wpm=200.0))
        assert low.final_speed > high.final_speed


# --------------------------------------------------------------------------
# Purity.
# --------------------------------------------------------------------------

class TestPurity:
    def test_does_not_mutate_config_or_input(self):
        cfg = _enabled()
        before_cfg = (cfg.enabled, cfg.base_wpm, cfg.strength)
        seq = [198.0, 0.0, 120.0]
        seq_copy = list(seq)
        simulate_speed_trajectory(seq, config=cfg)
        assert (cfg.enabled, cfg.base_wpm, cfg.strength) == before_cfg
        assert seq == seq_copy

    def test_deterministic(self):
        cfg = _enabled()
        seq = [120.0, 240.0, 60.0, 200.0]
        a = simulate_speed_trajectory(seq, config=cfg)
        b = simulate_speed_trajectory(seq, config=cfg)
        assert a.speeds == b.speeds
        assert a.final_gap == b.final_gap

    def test_accepts_int_and_float_wpm(self):
        cfg = _enabled(min_delta=0.0)
        a = simulate_speed_trajectory([198, 198], config=cfg)
        b = simulate_speed_trajectory([198.0, 198.0], config=cfg)
        assert a.speeds == b.speeds
