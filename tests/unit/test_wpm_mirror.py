"""Tests for iter-213 — the WPM-mirroring seam.

``session/wpm_mirror.py`` is backlog item #3 of the conversational-rhythm /
organic track (mirrored into ``ITERATION_LOG.md`` "next directions"). It is the
swappable seam between *the measured user speaking rate* (``user_wpm``,
iter-064) and *the TTS ``speed`` the next turn should use*, so the bot can
adapt its rate toward the user's for higher rapport / lower interruption.

The seam is pure (no I/O, no clock, no state), so these tests drive it directly
with injected ``user_wpm`` / ``current_speed`` values.

The headline contract these tests pin is the **off-by-default invariant**: a
default ``WpmMirrorConfig`` (``enabled=False``) leaves ``current_speed``
untouched for *every* input — byte-for-byte today's fixed-rate behavior — so
wiring the seam into the live TTS path (the named follow-on) can never regress
the proven constant-speed path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ``session/__init__.py`` eagerly imports pipecat-dependent modules
# (session.compute), which aren't installable on this x86_64 Linux runner.
# ``wpm_mirror`` is pure stdlib, so load it directly by file path to bypass the
# package ``__init__`` — mirrors how the turn_decider / backchannel tests keep
# platform deps out of the unit path.
_WM_PATH = Path(__file__).resolve().parents[2] / "session" / "wpm_mirror.py"
_spec = importlib.util.spec_from_file_location("_wm_under_test", _WM_PATH)
_wm = importlib.util.module_from_spec(_spec)
sys.modules["_wm_under_test"] = _wm
_spec.loader.exec_module(_wm)

WpmMirrorConfig = _wm.WpmMirrorConfig
mirrored_speed = _wm.mirrored_speed
WpmMirror = _wm.WpmMirror
DEFAULT_BASE_WPM = _wm.DEFAULT_BASE_WPM
DEFAULT_STRENGTH = _wm.DEFAULT_STRENGTH
DEFAULT_MIN_SPEED = _wm.DEFAULT_MIN_SPEED
DEFAULT_MAX_SPEED = _wm.DEFAULT_MAX_SPEED
DEFAULT_MIN_DELTA = _wm.DEFAULT_MIN_DELTA


# --------------------------------------------------------------------------
# The off-by-default invariant (the whole point of the seam).
# --------------------------------------------------------------------------

class TestDisabledInvariant:
    def test_default_config_is_disabled(self):
        assert WpmMirrorConfig().enabled is False

    @pytest.mark.parametrize(
        "user_wpm,current_speed",
        [
            (0.0, 1.0),
            (120.0, 1.0),
            (200.0, 1.0),
            (40.0, 1.2),
            (400.0, 0.9),
            (165.0, 1.0),  # exactly base — even here, disabled means no move
        ],
    )
    def test_disabled_returns_current_speed_unchanged(self, user_wpm, current_speed):
        # The default config is the off switch: identity on current_speed
        # for every input, and the user_wpm isn't even consulted.
        assert mirrored_speed(user_wpm, current_speed) == current_speed

    def test_disabled_explicitly(self):
        cfg = WpmMirrorConfig(enabled=False, base_wpm=165.0, strength=1.0)
        # Even with full strength, disabled wins.
        assert mirrored_speed(150.0, 1.0, cfg) == 1.0

    def test_none_config_matches_default_disabled(self):
        assert mirrored_speed(150.0, 1.0, None) == 1.0
        assert mirrored_speed(150.0, 1.0) == 1.0


# --------------------------------------------------------------------------
# No-measurement guard (iter-064's zero-WPM guard).
# --------------------------------------------------------------------------

class TestNoMeasurement:
    @pytest.mark.parametrize("user_wpm", [0.0, -10.0, -0.001])
    def test_nonpositive_user_wpm_returns_current_speed(self, user_wpm):
        cfg = WpmMirrorConfig(enabled=True)
        assert mirrored_speed(user_wpm, 1.0, cfg) == 1.0

    def test_zero_user_wpm_no_move_even_at_full_strength(self):
        cfg = WpmMirrorConfig(enabled=True, strength=1.0)
        assert mirrored_speed(0.0, 1.15, cfg) == 1.15


# --------------------------------------------------------------------------
# The proportional nudge (enabled, mid-range — no clamp, no deadband).
# --------------------------------------------------------------------------

class TestProportionalNudge:
    def test_faster_user_speeds_bot_up(self):
        # base 165, user 198 ⇒ ideal 1.2; from 1.0 at strength 0.5 ⇒ 1.1.
        cfg = WpmMirrorConfig(enabled=True, base_wpm=165.0, strength=0.5)
        assert mirrored_speed(198.0, 1.0, cfg) == pytest.approx(1.1)

    def test_slower_user_slows_bot_down(self):
        # user 132 ⇒ ideal 0.8; from 1.0 at strength 0.5 ⇒ 0.9.
        cfg = WpmMirrorConfig(enabled=True, base_wpm=165.0, strength=0.5)
        assert mirrored_speed(132.0, 1.0, cfg) == pytest.approx(0.9)

    def test_user_at_base_wpm_holds_speed(self):
        # ideal == current == 1.0 ⇒ no movement (and deadband would catch it
        # anyway). user_wpm == base_wpm is the fixed point.
        cfg = WpmMirrorConfig(enabled=True, base_wpm=165.0, strength=0.5)
        assert mirrored_speed(165.0, 1.0, cfg) == 1.0

    def test_full_strength_jumps_to_ideal(self):
        # strength 1.0 ⇒ target == ideal exactly (within clamp).
        cfg = WpmMirrorConfig(
            enabled=True, base_wpm=165.0, strength=1.0,
            min_speed=0.5, max_speed=2.0, min_delta=0.0,
        )
        assert mirrored_speed(165.0 * 1.25, 1.0, cfg) == pytest.approx(1.25)

    def test_partial_strength_moves_fraction_of_gap(self):
        # current 1.0, ideal 1.4, strength 0.25 ⇒ 1.0 + 0.25*0.4 = 1.1.
        cfg = WpmMirrorConfig(
            enabled=True, base_wpm=100.0, strength=0.25,
            min_speed=0.5, max_speed=2.0, min_delta=0.0,
        )
        assert mirrored_speed(140.0, 1.0, cfg) == pytest.approx(1.1)

    def test_nudge_from_nonunity_current_speed(self):
        # current 1.2, ideal 0.8, strength 0.5 ⇒ 1.2 + 0.5*(-0.4) = 1.0.
        cfg = WpmMirrorConfig(
            enabled=True, base_wpm=165.0, strength=0.5,
            min_speed=0.5, max_speed=2.0, min_delta=0.0,
        )
        assert mirrored_speed(132.0, 1.2, cfg) == pytest.approx(1.0)

    def test_monotone_in_user_wpm(self):
        # A faster user never yields a slower bot speed.
        cfg = WpmMirrorConfig(
            enabled=True, base_wpm=165.0, strength=0.5,
            min_speed=0.5, max_speed=2.0, min_delta=0.0,
        )
        prev = None
        for wpm in range(90, 280, 10):
            s = mirrored_speed(float(wpm), 1.0, cfg)
            if prev is not None:
                assert s >= prev
            prev = s


# --------------------------------------------------------------------------
# The intelligibility clamp.
# --------------------------------------------------------------------------

class TestClamp:
    def test_extreme_fast_user_clamped_to_max(self):
        # 400 WPM burst ⇒ ideal ~2.4; clamp to max_speed.
        cfg = WpmMirrorConfig(
            enabled=True, base_wpm=165.0, strength=1.0,
            min_speed=0.8, max_speed=1.3, min_delta=0.0,
        )
        assert mirrored_speed(400.0, 1.0, cfg) == 1.3

    def test_extreme_slow_user_clamped_to_min(self):
        # 40 WPM near-silence ⇒ ideal ~0.24; clamp to min_speed.
        cfg = WpmMirrorConfig(
            enabled=True, base_wpm=165.0, strength=1.0,
            min_speed=0.8, max_speed=1.3, min_delta=0.0,
        )
        assert mirrored_speed(40.0, 1.0, cfg) == 0.8

    def test_clamp_at_exact_max_boundary(self):
        # ideal exactly == max_speed ⇒ returned as-is.
        cfg = WpmMirrorConfig(
            enabled=True, base_wpm=100.0, strength=1.0,
            min_speed=0.8, max_speed=1.3, min_delta=0.0,
        )
        assert mirrored_speed(130.0, 1.0, cfg) == pytest.approx(1.3)

    def test_clamp_never_exceeds_band_across_grid(self):
        cfg = WpmMirrorConfig(
            enabled=True, base_wpm=165.0, strength=1.0,
            min_speed=0.85, max_speed=1.25, min_delta=0.0,
        )
        for wpm in range(20, 500, 7):
            s = mirrored_speed(float(wpm), 1.0, cfg)
            assert 0.85 <= s <= 1.25


# --------------------------------------------------------------------------
# The deadband on the change.
# --------------------------------------------------------------------------

class TestDeadband:
    def test_sub_delta_change_keeps_current_speed(self):
        # ideal 1.03, strength 1.0 ⇒ target 1.03; |1.03-1.0|=0.03 < 0.05 ⇒ hold.
        cfg = WpmMirrorConfig(
            enabled=True, base_wpm=100.0, strength=1.0,
            min_speed=0.5, max_speed=2.0, min_delta=0.05,
        )
        assert mirrored_speed(103.0, 1.0, cfg) == 1.0

    def test_change_at_or_above_delta_applies(self):
        # target 1.05; |0.05| not < 0.05 ⇒ applies.
        cfg = WpmMirrorConfig(
            enabled=True, base_wpm=100.0, strength=1.0,
            min_speed=0.5, max_speed=2.0, min_delta=0.05,
        )
        assert mirrored_speed(105.0, 1.0, cfg) == pytest.approx(1.05)

    def test_zero_deadband_applies_any_change(self):
        cfg = WpmMirrorConfig(
            enabled=True, base_wpm=100.0, strength=1.0,
            min_speed=0.5, max_speed=2.0, min_delta=0.0,
        )
        assert mirrored_speed(101.0, 1.0, cfg) == pytest.approx(1.01)

    def test_deadband_measured_against_current_not_ideal(self):
        # current 1.0, ideal 1.5, strength 0.05 ⇒ target 1.025;
        # |0.025| < 0.05 ⇒ hold at 1.0 (the small *step* is what's gated,
        # not the large gap to ideal).
        cfg = WpmMirrorConfig(
            enabled=True, base_wpm=100.0, strength=0.05,
            min_speed=0.5, max_speed=2.0, min_delta=0.05,
        )
        assert mirrored_speed(150.0, 1.0, cfg) == 1.0


# --------------------------------------------------------------------------
# Convergence: repeated application walks toward (but never past) ideal.
# --------------------------------------------------------------------------

class TestConvergence:
    def test_iterating_converges_toward_ideal(self):
        cfg = WpmMirrorConfig(
            enabled=True, base_wpm=165.0, strength=0.5,
            min_speed=0.5, max_speed=2.0, min_delta=0.0,
        )
        ideal = 200.0 / 165.0
        speed = 1.0
        for _ in range(20):
            speed = mirrored_speed(200.0, speed, cfg)
        assert speed == pytest.approx(ideal, abs=1e-3)

    def test_never_overshoots_ideal(self):
        # With strength in [0,1] a single step never crosses ideal.
        cfg = WpmMirrorConfig(
            enabled=True, base_wpm=165.0, strength=0.9,
            min_speed=0.5, max_speed=2.0, min_delta=0.0,
        )
        ideal = 132.0 / 165.0  # 0.8, below current 1.0
        speed = mirrored_speed(132.0, 1.0, cfg)
        assert speed >= ideal


# --------------------------------------------------------------------------
# Config validation.
# --------------------------------------------------------------------------

class TestConfigValidation:
    def test_defaults(self):
        cfg = WpmMirrorConfig()
        assert cfg.enabled is False
        assert cfg.base_wpm == DEFAULT_BASE_WPM
        assert cfg.strength == DEFAULT_STRENGTH
        assert cfg.min_speed == DEFAULT_MIN_SPEED
        assert cfg.max_speed == DEFAULT_MAX_SPEED
        assert cfg.min_delta == DEFAULT_MIN_DELTA

    def test_frozen(self):
        cfg = WpmMirrorConfig()
        with pytest.raises(Exception):
            cfg.enabled = True  # type: ignore[misc]

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_nonpositive_base_wpm_raises(self, bad):
        with pytest.raises(ValueError, match="base_wpm"):
            WpmMirrorConfig(base_wpm=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
    def test_strength_out_of_range_raises(self, bad):
        with pytest.raises(ValueError, match="strength"):
            WpmMirrorConfig(strength=bad)

    @pytest.mark.parametrize("good", [0.0, 0.5, 1.0])
    def test_strength_in_range_ok(self, good):
        WpmMirrorConfig(strength=good)

    @pytest.mark.parametrize("bad", [0.0, -0.5])
    def test_nonpositive_min_speed_raises(self, bad):
        with pytest.raises(ValueError, match="min_speed"):
            WpmMirrorConfig(min_speed=bad)

    def test_max_below_min_raises(self):
        with pytest.raises(ValueError, match="max_speed"):
            WpmMirrorConfig(min_speed=1.2, max_speed=1.0)

    def test_max_equal_min_ok(self):
        WpmMirrorConfig(min_speed=1.0, max_speed=1.0)

    def test_negative_min_delta_raises(self):
        with pytest.raises(ValueError, match="min_delta"):
            WpmMirrorConfig(min_delta=-0.01)


# --------------------------------------------------------------------------
# The WpmMirror class wrapper (the swappable-interface seam).
# --------------------------------------------------------------------------

class TestWpmMirrorClass:
    def test_default_mirror_is_disabled_passthrough(self):
        m = WpmMirror()
        assert m.speed(user_wpm=200.0, current_speed=1.0) == 1.0

    def test_enabled_mirror_matches_function(self):
        cfg = WpmMirrorConfig(
            enabled=True, base_wpm=165.0, strength=0.5, min_delta=0.0,
        )
        m = WpmMirror(cfg)
        assert m.speed(user_wpm=198.0, current_speed=1.0) == mirrored_speed(
            198.0, 1.0, cfg
        )

    def test_speed_is_keyword_only(self):
        m = WpmMirror(WpmMirrorConfig(enabled=True))
        with pytest.raises(TypeError):
            m.speed(200.0, 1.0)  # type: ignore[misc]

    def test_mirror_holds_its_config(self):
        cfg = WpmMirrorConfig(enabled=True, base_wpm=200.0)
        m = WpmMirror(cfg)
        assert m.config is cfg

    def test_mirror_is_stateless_across_calls(self):
        # Two independent mirrors / repeated calls don't share or accumulate
        # hidden state — each call is a pure function of its arguments.
        cfg = WpmMirrorConfig(
            enabled=True, base_wpm=165.0, strength=0.5, min_delta=0.0,
        )
        m = WpmMirror(cfg)
        a = m.speed(user_wpm=198.0, current_speed=1.0)
        b = m.speed(user_wpm=198.0, current_speed=1.0)
        assert a == b


# --------------------------------------------------------------------------
# Purity.
# --------------------------------------------------------------------------

class TestPurity:
    def test_does_not_mutate_config(self):
        cfg = WpmMirrorConfig(enabled=True, base_wpm=165.0, strength=0.5)
        before = (cfg.enabled, cfg.base_wpm, cfg.strength, cfg.min_speed,
                  cfg.max_speed, cfg.min_delta)
        mirrored_speed(198.0, 1.0, cfg)
        after = (cfg.enabled, cfg.base_wpm, cfg.strength, cfg.min_speed,
                 cfg.max_speed, cfg.min_delta)
        assert before == after
