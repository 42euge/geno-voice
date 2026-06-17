"""Tests for iter-149 — the turn-decider seam.

``session/turn_decider.py`` is backlog item #2 of the organic turn-taking
track (``docs/research/organic-turn-taking.md``). It is the swappable seam
between *where the turn-end confidence comes from* and *what
``TurnTakingEngine`` does with it*. Today the body is a pure
silence → confidence heuristic; a later lap swaps in pipecat's audio-only
``smart-turn`` model behind the identical interface.

``silence_confidence`` / ``SilenceTurnDecider`` are pure (no I/O, no clock),
so these tests drive them directly with injected silence durations.

These tests also pin the *contract* the seam exists to fix: with the default
config, the heuristic clears the engine's ``smart_turn_backchannel_min`` and
``smart_turn_response_min`` thresholds within the engine's own silence
windows — i.e. it un-deadens the silence-driven tiers that the hardcoded
``0.5`` left unreachable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ``session/__init__.py`` eagerly imports pipecat-dependent modules
# (session.compute), which aren't installable on this x86_64 Linux runner.
# ``turn_decider`` is pure stdlib, so load it directly by file path to bypass
# the package ``__init__`` — mirrors how the mic_* / backchannel tests keep
# platform deps out of the unit path. The module is registered in
# ``sys.modules`` so its frozen dataclass annotations resolve.
_TD_PATH = Path(__file__).resolve().parents[2] / "session" / "turn_decider.py"
_spec = importlib.util.spec_from_file_location("_td_under_test", _TD_PATH)
_td = importlib.util.module_from_spec(_spec)
sys.modules["_td_under_test"] = _td
_spec.loader.exec_module(_td)

TurnDeciderConfig = _td.TurnDeciderConfig
silence_confidence = _td.silence_confidence
SilenceTurnDecider = _td.SilenceTurnDecider
FLOOR = _td.DEFAULT_SILENCE_FLOOR_SECS
CEILING = _td.DEFAULT_SILENCE_CEILING_SECS


# ---------------------------------------------------------------------------
# silence_confidence — clamping
# ---------------------------------------------------------------------------

class TestSilenceConfidenceClamping:
    def test_at_floor_is_zero(self):
        assert silence_confidence(FLOOR) == 0.0

    def test_below_floor_is_zero(self):
        assert silence_confidence(FLOOR - 0.5) == 0.0

    def test_zero_silence_is_zero(self):
        assert silence_confidence(0.0) == 0.0

    def test_negative_clamps_to_zero(self):
        assert silence_confidence(-5.0) == 0.0

    def test_at_ceiling_is_one(self):
        assert silence_confidence(CEILING) == 1.0

    def test_above_ceiling_is_one(self):
        assert silence_confidence(CEILING + 100.0) == 1.0


# ---------------------------------------------------------------------------
# silence_confidence — linear ramp between floor and ceiling
# ---------------------------------------------------------------------------

class TestSilenceConfidenceRamp:
    def test_midpoint_is_half(self):
        mid = (FLOOR + CEILING) / 2
        assert silence_confidence(mid) == pytest.approx(0.5)

    def test_monotonic_increasing_in_band(self):
        prev = -1.0
        # step through the open band (floor, ceiling)
        steps = [FLOOR + (CEILING - FLOOR) * f for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
        for s in steps:
            c = silence_confidence(s)
            assert c > prev
            assert 0.0 < c < 1.0
            prev = c

    def test_quarter_point(self):
        q = FLOOR + (CEILING - FLOOR) * 0.25
        assert silence_confidence(q) == pytest.approx(0.25)

    def test_just_above_floor_is_positive(self):
        c = silence_confidence(FLOOR + 0.001)
        assert 0.0 < c < 0.01


# ---------------------------------------------------------------------------
# The contract this seam exists to fix: default curve reaches the engine's
# smart-turn thresholds inside the engine's own silence windows.
# ---------------------------------------------------------------------------

class TestEngineThresholdContract:
    # Engine defaults (session/turn_taking.py TurnTakingConfig) the heuristic
    # is tuned against. Duplicated here as literals so a change to either side
    # trips a test rather than silently re-deadening a tier.
    SILENCE_BACKCHANNEL_MIN = 4.0
    SILENCE_RESPONSE_MIN = 6.0
    SMART_TURN_BACKCHANNEL_MIN = 0.6
    SMART_TURN_RESPONSE_MIN = 0.85

    def test_backchannel_window_clears_backchannel_threshold(self):
        # At the engine's backchannel silence window, confidence must clear
        # smart_turn_backchannel_min, else the backchannel tier stays dead.
        c = silence_confidence(self.SILENCE_BACKCHANNEL_MIN)
        assert c >= self.SMART_TURN_BACKCHANNEL_MIN

    def test_response_window_clears_response_threshold(self):
        c = silence_confidence(self.SILENCE_RESPONSE_MIN)
        assert c >= self.SMART_TURN_RESPONSE_MIN

    def test_hardcoded_half_was_below_backchannel_threshold(self):
        # Documents WHY the seam matters: the old hardcoded 0.5 sat below the
        # engine's backchannel threshold, so silence-driven tiers never fired.
        assert 0.5 < self.SMART_TURN_BACKCHANNEL_MIN

    def test_short_pause_yields_low_confidence(self):
        # A mid-thought pause under the floor must not look like a turn-end.
        assert silence_confidence(1.5) == 0.0


# ---------------------------------------------------------------------------
# Custom config
# ---------------------------------------------------------------------------

class TestCustomConfig:
    def test_custom_band_remaps_midpoint(self):
        cfg = TurnDeciderConfig(silence_floor_secs=0.0, silence_ceiling_secs=10.0)
        assert silence_confidence(5.0, cfg) == pytest.approx(0.5)
        assert silence_confidence(0.0, cfg) == 0.0
        assert silence_confidence(10.0, cfg) == 1.0

    def test_ceiling_not_above_floor_raises(self):
        with pytest.raises(ValueError):
            TurnDeciderConfig(silence_floor_secs=5.0, silence_ceiling_secs=5.0)
        with pytest.raises(ValueError):
            TurnDeciderConfig(silence_floor_secs=5.0, silence_ceiling_secs=4.0)

    def test_config_is_frozen(self):
        cfg = TurnDeciderConfig()
        with pytest.raises(Exception):
            cfg.silence_floor_secs = 1.0  # type: ignore[misc]

    def test_narrow_band_is_steep(self):
        cfg = TurnDeciderConfig(silence_floor_secs=3.0, silence_ceiling_secs=3.5)
        assert silence_confidence(3.0, cfg) == 0.0
        assert silence_confidence(3.25, cfg) == pytest.approx(0.5)
        assert silence_confidence(3.5, cfg) == 1.0


# ---------------------------------------------------------------------------
# SilenceTurnDecider — the swappable interface
# ---------------------------------------------------------------------------

class TestSilenceTurnDecider:
    def test_confidence_matches_function(self):
        d = SilenceTurnDecider()
        for s in (0.0, 2.0, 3.5, 4.0, 6.0, 12.0):
            assert d.confidence(silence_duration_secs=s) == silence_confidence(s)

    def test_transcript_chunk_ignored_today(self):
        # Accepted for forward-compat with a text-aware EOU (backlog #4); the
        # silence-only decider ignores it, so result is identical with/without.
        d = SilenceTurnDecider()
        with_text = d.confidence(silence_duration_secs=4.0, transcript_chunk="and so")
        without = d.confidence(silence_duration_secs=4.0)
        assert with_text == without

    def test_uses_injected_config(self):
        cfg = TurnDeciderConfig(silence_floor_secs=0.0, silence_ceiling_secs=2.0)
        d = SilenceTurnDecider(cfg)
        assert d.confidence(silence_duration_secs=1.0) == pytest.approx(0.5)

    def test_default_config_when_none(self):
        d = SilenceTurnDecider()
        assert d.config.silence_floor_secs == FLOOR
        assert d.config.silence_ceiling_secs == CEILING

    def test_confidence_is_keyword_only(self):
        d = SilenceTurnDecider()
        with pytest.raises(TypeError):
            d.confidence(4.0)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Integration with the real engine: feeding the seam's output un-deadens the
# silence-driven tiers that hardcoded 0.5 left unreachable.
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    """Loads the real TurnTakingEngine. Skipped if pipecat (pulled in by
    ``session/__init__``) isn't installed on the runner — but turn_taking
    itself is pure, so we load it by file path like the seam under test."""

    @staticmethod
    def _load_by_path(name, filename, package=None):
        path = Path(__file__).resolve().parents[2] / "session" / filename
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        if package is not None:
            mod.__package__ = package
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    @classmethod
    def _load_engine(cls):
        # ``turn_taking`` does ``from session.triggers import ...``. Importing
        # ``session`` normally runs ``session/__init__`` which pulls pipecat
        # (absent here). Build a minimal ``session`` namespace package and load
        # ``triggers`` + ``turn_taking`` into it by file path — both are pure.
        import types
        try:
            if "session" not in sys.modules:
                pkg = types.ModuleType("session")
                pkg.__path__ = []  # mark as a package
                sys.modules["session"] = pkg
            if "session.triggers" not in sys.modules:
                cls._load_by_path("session.triggers", "triggers.py", package="session")
            return cls._load_by_path("session.turn_taking", "turn_taking.py", package="session")
        except Exception:
            return None

    def test_silence_seam_reaches_backchannel_where_hardcoded_half_did_not(self):
        tt = self._load_engine()
        if tt is None:
            pytest.skip("turn_taking unavailable on this runner")

        # Build an engine in a state where a backchannel WOULD be appropriate
        # (enough speaking, no recent cue, mature session).
        engine = tt.TurnTakingEngine()
        engine.state.session_start = tt.time.time() - 300  # past early-session window
        engine.update_state(user_spoke_secs=30)  # > min_speaking_before_first_cue

        silence = 4.5  # inside [backchannel_min=4, response_min=6)

        # Old behaviour: hardcoded 0.5 < smart_turn_backchannel_min ⇒ STAY_SILENT.
        old = engine.decide(silence, 0.5)
        assert old.action == tt.Action.STAY_SILENT

        # New behaviour: seam-derived confidence clears the threshold ⇒ PLAY_CUE.
        conf = silence_confidence(silence)
        new = engine.decide(silence, conf)
        assert new.action == tt.Action.PLAY_CUE

    def test_silence_seam_reaches_response_tier(self):
        tt = self._load_engine()
        if tt is None:
            pytest.skip("turn_taking unavailable on this runner")

        engine = tt.TurnTakingEngine()
        engine.state.session_start = tt.time.time() - 300
        engine.update_state(user_spoke_secs=90)  # > long_monologue_secs

        silence = 7.0  # >= response_min
        conf = silence_confidence(silence)
        assert conf >= 0.85
        decision = engine.decide(silence, conf)
        # Long monologue + extended silence + high confidence ⇒ offer reflection.
        assert decision.action == tt.Action.SPEAK_BRIEF
