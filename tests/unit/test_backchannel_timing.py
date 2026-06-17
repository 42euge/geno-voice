"""Tests for iter-153 — agent backchannel emission timing (backlog #7).

``session/backchannel_timing.py`` decides when the *agent* should emit a short
backchannel ("mhmm") *during* user speech — the emit half of backchanneling,
complementing the recognize half in ``session/backchannel.py`` (#1, iter-148).
``decide_backchannel_timing(*, user_speaking_secs, pause_secs,
secs_since_last_backchannel, config, timing)`` returns ``EMIT`` (good moment)
or ``HOLD`` (too soon / too frequent / no pause / turn-end-sized gap).

The whole point is the **half-duplex invariant**: with a default
``FullDuplexConfig()`` (agent backchannels off) the decision is ``HOLD`` for
every input — the agent never speaks during user speech, byte-for-byte today's
behavior. Only with ``agent_backchannels`` explicitly on does a well-timed
clause-boundary pause yield ``EMIT``.

``backchannel_timing`` does ``from session.full_duplex import ...`` at module
scope, but ``session/__init__.py`` eagerly imports pipecat-dependent modules
(absent on the x86_64 Linux runner). So we stand up a stub ``session``
namespace package and load ``full_duplex`` / ``backchannel_timing`` into it by
file path — the same trick test_barge_decision.py / test_text_eou.py use.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SESSION_DIR = Path(__file__).resolve().parents[2] / "session"


def _load_by_path(name, filename, package=None):
    spec = importlib.util.spec_from_file_location(name, _SESSION_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    if package is not None:
        mod.__package__ = package
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if "session" not in sys.modules:
    _pkg = types.ModuleType("session")
    _pkg.__path__ = [str(_SESSION_DIR)]
    sys.modules["session"] = _pkg
if "session.full_duplex" not in sys.modules:
    _load_by_path("session.full_duplex", "full_duplex.py", package="session")

_bt = _load_by_path(
    "session.backchannel_timing", "backchannel_timing.py", package="session"
)

BackchannelTiming = _bt.BackchannelTiming
BackchannelTimingConfig = _bt.BackchannelTimingConfig
decide_backchannel_timing = _bt.decide_backchannel_timing
should_emit_backchannel = _bt.should_emit_backchannel

FullDuplexConfig = sys.modules["session.full_duplex"].FullDuplexConfig


# Shorthands for an organic-mode config that turns agent backchannels on.
def _organic() -> "FullDuplexConfig":
    return FullDuplexConfig(enabled=True)


def _organic_bc_on() -> "FullDuplexConfig":
    # Master off, but the agent-backchannels sub-flag forced on.
    return FullDuplexConfig(enabled=False, agent_backchannels=True)


# A set of inputs that, under an organic config, sit squarely in the EMIT
# window — long monologue, clause-boundary pause, no recent cue.
_GOOD = dict(
    user_speaking_secs=30.0,
    pause_secs=0.8,
    secs_since_last_backchannel=None,
)


# --------------------------------------------------------------------------
# Half-duplex invariant: default config never emits, regardless of timing.
# --------------------------------------------------------------------------

class TestHalfDuplexInvariant:
    def test_default_config_holds_on_good_timing(self):
        assert (
            decide_backchannel_timing(**_GOOD)
            is BackchannelTiming.HOLD
        )

    def test_explicit_default_config_holds(self):
        assert (
            decide_backchannel_timing(config=FullDuplexConfig(), **_GOOD)
            is BackchannelTiming.HOLD
        )

    def test_master_on_but_backchannels_held_off_holds(self):
        # Organic master on, but agent_backchannels explicitly False.
        cfg = FullDuplexConfig(enabled=True, agent_backchannels=False)
        assert (
            decide_backchannel_timing(config=cfg, **_GOOD)
            is BackchannelTiming.HOLD
        )

    @pytest.mark.parametrize(
        "user_speaking_secs,pause_secs,since",
        [
            (30.0, 0.8, None),     # otherwise-perfect EMIT input
            (5.0, 0.8, None),      # too-soon
            (30.0, 0.05, None),    # no real pause
            (30.0, 5.0, None),     # turn-end-sized gap
            (30.0, 0.8, 1.0),      # rate-limited
        ],
    )
    def test_every_input_holds_under_default(
        self, user_speaking_secs, pause_secs, since
    ):
        assert (
            decide_backchannel_timing(
                user_speaking_secs=user_speaking_secs,
                pause_secs=pause_secs,
                secs_since_last_backchannel=since,
            )
            is BackchannelTiming.HOLD
        )


# --------------------------------------------------------------------------
# Organic mode: the EMIT window.
# --------------------------------------------------------------------------

class TestOrganicModeEmit:
    def test_good_timing_emits_master_on(self):
        assert (
            decide_backchannel_timing(config=_organic(), **_GOOD)
            is BackchannelTiming.EMIT
        )

    def test_good_timing_emits_subflag_on_overrides_master_off(self):
        # Master off but sub-flag forced on ⇒ still organic for this behavior.
        assert (
            decide_backchannel_timing(config=_organic_bc_on(), **_GOOD)
            is BackchannelTiming.EMIT
        )

    def test_never_backchanneled_passes_rate_limit(self):
        assert (
            decide_backchannel_timing(
                config=_organic(),
                user_speaking_secs=30.0,
                pause_secs=0.8,
                secs_since_last_backchannel=None,
            )
            is BackchannelTiming.EMIT
        )

    def test_long_gap_since_last_cue_emits(self):
        assert (
            decide_backchannel_timing(
                config=_organic(),
                user_speaking_secs=30.0,
                pause_secs=0.8,
                secs_since_last_backchannel=25.0,
            )
            is BackchannelTiming.EMIT
        )


# --------------------------------------------------------------------------
# Organic mode: the warm-up gate (rule 2).
# --------------------------------------------------------------------------

class TestWarmUpGate:
    def test_below_warmup_holds(self):
        assert (
            decide_backchannel_timing(
                config=_organic(),
                user_speaking_secs=10.0,  # < 15.0 default
                pause_secs=0.8,
            )
            is BackchannelTiming.HOLD
        )

    def test_exactly_at_warmup_emits(self):
        # Boundary: >= min_speaking passes.
        assert (
            decide_backchannel_timing(
                config=_organic(),
                user_speaking_secs=15.0,
                pause_secs=0.8,
            )
            is BackchannelTiming.EMIT
        )

    def test_custom_warmup_threshold(self):
        timing = BackchannelTimingConfig(
            min_speaking_before_first_cue_secs=5.0
        )
        assert (
            decide_backchannel_timing(
                config=_organic(),
                timing=timing,
                user_speaking_secs=6.0,
                pause_secs=0.8,
            )
            is BackchannelTiming.EMIT
        )


# --------------------------------------------------------------------------
# Organic mode: the rate-limit gate (rule 3).
# --------------------------------------------------------------------------

class TestRateLimitGate:
    def test_too_soon_after_last_cue_holds(self):
        assert (
            decide_backchannel_timing(
                config=_organic(),
                user_speaking_secs=30.0,
                pause_secs=0.8,
                secs_since_last_backchannel=5.0,  # < 20.0 default
            )
            is BackchannelTiming.HOLD
        )

    def test_exactly_at_rate_limit_emits(self):
        # Boundary: >= min_between_cues passes.
        assert (
            decide_backchannel_timing(
                config=_organic(),
                user_speaking_secs=30.0,
                pause_secs=0.8,
                secs_since_last_backchannel=20.0,
            )
            is BackchannelTiming.EMIT
        )

    def test_custom_rate_limit(self):
        timing = BackchannelTimingConfig(min_between_cues_secs=3.0)
        assert (
            decide_backchannel_timing(
                config=_organic(),
                timing=timing,
                user_speaking_secs=30.0,
                pause_secs=0.8,
                secs_since_last_backchannel=4.0,
            )
            is BackchannelTiming.EMIT
        )


# --------------------------------------------------------------------------
# Organic mode: the pause window (rule 4).
# --------------------------------------------------------------------------

class TestPauseWindow:
    def test_below_min_pause_holds(self):
        # Continuous speech — no real clause boundary.
        assert (
            decide_backchannel_timing(
                config=_organic(),
                user_speaking_secs=30.0,
                pause_secs=0.1,  # < 0.3 default
            )
            is BackchannelTiming.HOLD
        )

    def test_exactly_at_min_pause_emits(self):
        assert (
            decide_backchannel_timing(
                config=_organic(),
                user_speaking_secs=30.0,
                pause_secs=0.3,  # inclusive lower bound
            )
            is BackchannelTiming.EMIT
        )

    def test_just_below_max_pause_emits(self):
        assert (
            decide_backchannel_timing(
                config=_organic(),
                user_speaking_secs=30.0,
                pause_secs=1.99,
            )
            is BackchannelTiming.EMIT
        )

    def test_exactly_at_max_pause_holds(self):
        # Exclusive upper bound: at the turn-end floor, the silence path owns it.
        assert (
            decide_backchannel_timing(
                config=_organic(),
                user_speaking_secs=30.0,
                pause_secs=2.0,
            )
            is BackchannelTiming.HOLD
        )

    def test_above_max_pause_holds(self):
        assert (
            decide_backchannel_timing(
                config=_organic(),
                user_speaking_secs=30.0,
                pause_secs=4.0,
            )
            is BackchannelTiming.HOLD
        )

    def test_zero_pause_holds(self):
        assert (
            decide_backchannel_timing(
                config=_organic(),
                user_speaking_secs=30.0,
                pause_secs=0.0,
            )
            is BackchannelTiming.HOLD
        )

    def test_custom_pause_window(self):
        timing = BackchannelTimingConfig(
            min_pause_secs=0.5, max_pause_secs=1.0
        )
        # 0.4 is below the custom min ⇒ HOLD
        assert (
            decide_backchannel_timing(
                config=_organic(),
                timing=timing,
                user_speaking_secs=30.0,
                pause_secs=0.4,
            )
            is BackchannelTiming.HOLD
        )
        # 0.7 is inside the custom window ⇒ EMIT
        assert (
            decide_backchannel_timing(
                config=_organic(),
                timing=timing,
                user_speaking_secs=30.0,
                pause_secs=0.7,
            )
            is BackchannelTiming.EMIT
        )


# --------------------------------------------------------------------------
# Rule precedence: an earlier HOLD beats a later EMIT condition.
# --------------------------------------------------------------------------

class TestRulePrecedence:
    def test_gate_beats_perfect_timing(self):
        # Gate off (default) wins even with otherwise-perfect timing.
        assert (
            decide_backchannel_timing(
                user_speaking_secs=30.0, pause_secs=0.8
            )
            is BackchannelTiming.HOLD
        )

    def test_warmup_beats_good_pause(self):
        # Good pause, but below warm-up ⇒ HOLD (rule 2 before rule 4).
        assert (
            decide_backchannel_timing(
                config=_organic(),
                user_speaking_secs=2.0,
                pause_secs=0.8,
            )
            is BackchannelTiming.HOLD
        )

    def test_rate_limit_beats_good_pause(self):
        # Good pause + warmed up, but rate-limited ⇒ HOLD (rule 3 before 4).
        assert (
            decide_backchannel_timing(
                config=_organic(),
                user_speaking_secs=30.0,
                pause_secs=0.8,
                secs_since_last_backchannel=1.0,
            )
            is BackchannelTiming.HOLD
        )


# --------------------------------------------------------------------------
# should_emit_backchannel boolean convenience.
# --------------------------------------------------------------------------

class TestShouldEmitBoolean:
    def test_default_always_false(self):
        assert should_emit_backchannel(**_GOOD) is False

    def test_organic_good_timing_true(self):
        assert (
            should_emit_backchannel(config=_organic(), **_GOOD) is True
        )

    def test_organic_bad_timing_false(self):
        assert (
            should_emit_backchannel(
                config=_organic(),
                user_speaking_secs=30.0,
                pause_secs=5.0,  # turn-end gap
            )
            is False
        )

    def test_matches_decide(self):
        # The boolean is exactly (decide == EMIT) across a grid.
        grid = [
            dict(user_speaking_secs=30.0, pause_secs=0.8),
            dict(user_speaking_secs=5.0, pause_secs=0.8),
            dict(user_speaking_secs=30.0, pause_secs=0.05),
            dict(user_speaking_secs=30.0, pause_secs=3.0),
            dict(
                user_speaking_secs=30.0,
                pause_secs=0.8,
                secs_since_last_backchannel=2.0,
            ),
        ]
        for kwargs in grid:
            decided = decide_backchannel_timing(config=_organic(), **kwargs)
            boolean = should_emit_backchannel(config=_organic(), **kwargs)
            assert boolean is (decided is BackchannelTiming.EMIT)


# --------------------------------------------------------------------------
# Config validation / purity / interface.
# --------------------------------------------------------------------------

class TestConfigValidation:
    def test_default_config_values(self):
        c = BackchannelTimingConfig()
        assert c.min_speaking_before_first_cue_secs == 15.0
        assert c.min_between_cues_secs == 20.0
        assert c.min_pause_secs == 0.3
        assert c.max_pause_secs == 2.0

    def test_frozen(self):
        c = BackchannelTimingConfig()
        with pytest.raises(Exception):
            c.min_pause_secs = 0.5  # type: ignore[misc]

    def test_max_must_exceed_min_pause(self):
        with pytest.raises(ValueError, match="max_pause_secs must be >"):
            BackchannelTimingConfig(min_pause_secs=1.0, max_pause_secs=1.0)

    def test_max_below_min_pause_raises(self):
        with pytest.raises(ValueError, match="max_pause_secs must be >"):
            BackchannelTimingConfig(min_pause_secs=2.0, max_pause_secs=1.0)

    def test_negative_warmup_raises(self):
        with pytest.raises(ValueError, match="min_speaking_before_first_cue"):
            BackchannelTimingConfig(min_speaking_before_first_cue_secs=-1.0)

    def test_negative_between_cues_raises(self):
        with pytest.raises(ValueError, match="min_between_cues_secs"):
            BackchannelTimingConfig(min_between_cues_secs=-1.0)

    def test_negative_min_pause_raises(self):
        with pytest.raises(ValueError, match="min_pause_secs"):
            BackchannelTimingConfig(min_pause_secs=-0.1)

    def test_max_pause_equals_turn_decider_floor(self):
        # The default max_pause is exactly turn_decider's silence_floor (2.0s)
        # so the mid-speech and turn-end backchannel paths don't overlap.
        assert BackchannelTimingConfig().max_pause_secs == 2.0


class TestPurityAndInterface:
    def test_config_keyword_only(self):
        # decide_backchannel_timing takes only keyword args (leading *).
        with pytest.raises(TypeError):
            decide_backchannel_timing(30.0, 0.8)  # type: ignore[misc]

    def test_does_not_mutate_config(self):
        cfg = _organic()
        before = (
            cfg.enabled,
            cfg.continuer_aware_listening,
            cfg.agent_backchannels,
        )
        decide_backchannel_timing(config=cfg, **_GOOD)
        after = (
            cfg.enabled,
            cfg.continuer_aware_listening,
            cfg.agent_backchannels,
        )
        assert before == after

    def test_does_not_mutate_timing(self):
        timing = BackchannelTimingConfig()
        before = (
            timing.min_speaking_before_first_cue_secs,
            timing.min_between_cues_secs,
            timing.min_pause_secs,
            timing.max_pause_secs,
        )
        decide_backchannel_timing(config=_organic(), timing=timing, **_GOOD)
        after = (
            timing.min_speaking_before_first_cue_secs,
            timing.min_between_cues_secs,
            timing.min_pause_secs,
            timing.max_pause_secs,
        )
        assert before == after

    def test_distinct_enum_values(self):
        assert BackchannelTiming.EMIT is not BackchannelTiming.HOLD
        assert BackchannelTiming.EMIT.value == "emit"
        assert BackchannelTiming.HOLD.value == "hold"
