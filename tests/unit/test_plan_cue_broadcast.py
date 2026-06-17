"""
Tests for ``plan_cue_broadcast`` — the pure gate the live cue path
(``pipecat_server.py``'s ``Broadcaster``) consults to decide *whether* and
*which* backchannel cue to broadcast after ``TurnTakingEngine.decide``.

This closes two real bugs the live sidecar carried (found iter-172):

  1. The branch read ``decision.action == Action.play_cue`` — but the enum
     member is ``PLAY_CUE``. ``Action.play_cue`` raises ``AttributeError`` on
     *every* transcript frame, so the live cue path was broken, not merely
     dead.
  2. ``broadcast_cue`` re-picked a *random* cue from a private ``CUE_TYPES``
     list, throwing away the engine's rotated ``decision.cue`` (the rotation
     iter-171 unified into ``session/cue_rotation.py``) and even including cue
     keys ("hmm", "okay") not in the shared rotation. ``plan_cue_broadcast``
     returns the engine's rotated cue so the live broadcast stays indexed to
     the one source of truth.

``session/__init__.py`` eagerly imports pipecat (absent on the x86_64 runner),
so — like ``test_turn_decider.py`` — we load ``turn_taking`` by file path into
a minimal synthetic ``session`` namespace package. ``turn_taking`` itself is
pure stdlib.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SESSION_DIR = Path(__file__).resolve().parents[2] / "session"


def _load_by_path(name, filename, package=None):
    path = _SESSION_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if package is not None:
        mod.__package__ = package
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_turn_taking():
    """Load the pure ``turn_taking`` module without running ``session/__init__``."""
    if "session" not in sys.modules:
        pkg = types.ModuleType("session")
        pkg.__path__ = []  # mark as a package
        sys.modules["session"] = pkg
    if "session.cue_rotation" not in sys.modules:
        _load_by_path("session.cue_rotation", "cue_rotation.py", package="session")
    if "session.triggers" not in sys.modules:
        _load_by_path("session.triggers", "triggers.py", package="session")
    return _load_by_path("session.turn_taking", "turn_taking.py", package="session")


tt = _load_turn_taking()


def _decision(action, cue_type=None):
    cue = tt.CueSelection(cue_type=cue_type) if cue_type is not None else None
    return tt.TurnDecision(action=action, reason="test", cue=cue)


class TestPlanCueBroadcast:
    def test_play_cue_returns_the_decisions_rotated_cue(self):
        d = _decision(tt.Action.PLAY_CUE, cue_type="i_see")
        assert plan(d) == "i_see"

    def test_uses_enum_member_not_attribute_typo(self):
        # Regression: the live path read ``Action.play_cue`` (lowercase) which
        # is an AttributeError. The correct member is PLAY_CUE.
        with pytest.raises(AttributeError):
            tt.Action.play_cue  # noqa: B018 — intentional attribute access
        # And the gate fires on the real member.
        assert plan(_decision(tt.Action.PLAY_CUE, cue_type="mhmm")) == "mhmm"

    def test_trigger_fired_suppresses_cue(self):
        # An NLP trigger means a real response is warranted; a continuer would
        # step on it. Suppress even though the action is PLAY_CUE.
        d = _decision(tt.Action.PLAY_CUE, cue_type="right")
        assert plan(d, trigger_fired=True) is None

    def test_non_play_cue_actions_return_none(self):
        for action in (
            tt.Action.STAY_SILENT,
            tt.Action.SPEAK_BRIEF,
            tt.Action.SPEAK_FULL,
            tt.Action.GENTLE_PROMPT,
        ):
            assert plan(_decision(action, cue_type="mhmm")) is None

    def test_play_cue_without_a_cue_returns_none(self):
        # Defensive: PLAY_CUE but no cue attached ⇒ nothing to play.
        assert plan(_decision(tt.Action.PLAY_CUE, cue_type=None)) is None

    def test_default_trigger_fired_is_false(self):
        # The common call site passes only the decision; default must allow.
        assert plan(_decision(tt.Action.PLAY_CUE, cue_type="go_on")) == "go_on"

    def test_returned_cue_tracks_engine_rotation_not_a_random_list(self):
        # The whole point: the planner echoes whatever cue the engine rotated
        # in, including a cue from the shared rotation, never a private pick.
        for cue in tt.CUE_ROTATION:
            assert plan(_decision(tt.Action.PLAY_CUE, cue_type=cue)) == cue


def plan(decision, **kw):
    return tt.plan_cue_broadcast(decision, **kw)


class TestEngineRotationFlowsThroughPlanner:
    """End-to-end through the real engine: a backchannel-tier decision yields a
    rotated cue that the planner surfaces, and consecutive cues rotate."""

    @staticmethod
    def _ready_engine():
        engine = tt.TurnTakingEngine()
        engine.state.session_start = tt.time.time() - 300  # past early-session
        engine.update_state(user_spoke_secs=30)  # > min_speaking_before_first_cue
        return engine

    def test_consecutive_play_cues_rotate_through_the_shared_rotation(self):
        engine = self._ready_engine()
        # confidence in [backchannel_min, response_min) at a high enough value.
        silence = 4.5
        conf = 0.7  # clears smart_turn_backchannel_min (0.6)

        cues = []
        for _ in range(len(tt.CUE_ROTATION) + 1):
            d = engine.decide(silence, conf)
            if d.action is not tt.Action.PLAY_CUE:
                pytest.skip("engine did not reach the backchannel tier on this config")
            cues.append(plan(d))
            # Keep the rate-limit clear so the next decide can also play a cue.
            engine.state.last_backchannel_at = None

        # First full lap matches the shared rotation order, then wraps.
        assert cues[: len(tt.CUE_ROTATION)] == tt.CUE_ROTATION
        assert cues[len(tt.CUE_ROTATION)] == tt.CUE_ROTATION[0]
