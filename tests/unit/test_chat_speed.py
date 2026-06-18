"""Tests for iter-214 — SpeedController (the live-path holder for the iter-213
WPM-mirroring seam).

The controller is the mutable per-session speed value the synth path reads via
``current()`` and the turn loop updates via ``observe(user_wpm)``. Its headline
contract is the **off-by-default invariant**: ``mirror=None`` ⇒ ``observe`` is a
no-op and the speed never moves from its initial value — byte-for-byte today's
fixed-rate path.

The mirror is duck-typed (anything exposing ``speed(*, user_wpm,
current_speed)``), so these tests drive the controller with lightweight stubs
rather than importing ``session.wpm_mirror`` (which pulls the pipecat-dependent
``session`` package).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_speed import SpeedController  # noqa: E402


# ---- Stubs ----------------------------------------------------------------


class _RecordingMirror:
    """Returns ``return_value`` from ``speed`` and records every call's
    kwargs so tests can assert on what the controller fed it."""

    def __init__(self, return_value: float):
        self.return_value = return_value
        self.calls: list[dict] = []

    def speed(self, *, user_wpm: float, current_speed: float) -> float:
        self.calls.append({"user_wpm": user_wpm, "current_speed": current_speed})
        return self.return_value


class _SequenceMirror:
    """Returns each value from a queue in turn — exercises convergence over
    several ``observe`` calls."""

    def __init__(self, values):
        self.values = list(values)
        self.calls: list[dict] = []

    def speed(self, *, user_wpm: float, current_speed: float) -> float:
        self.calls.append({"user_wpm": user_wpm, "current_speed": current_speed})
        return self.values.pop(0)


class _RaisingMirror:
    def speed(self, *, user_wpm: float, current_speed: float) -> float:
        raise RuntimeError("boom")


class _NonNumericMirror:
    def speed(self, *, user_wpm: float, current_speed: float):
        return "not a number"


# ---- Off-by-default (no mirror) -------------------------------------------


def test_initial_speed_is_current():
    c = SpeedController(1.0)
    assert c.current() == 1.0
    assert c.speed == 1.0


def test_initial_speed_coerced_to_float():
    c = SpeedController(1)  # int in, float out
    assert isinstance(c.current(), float)
    assert c.current() == 1.0


def test_no_mirror_observe_is_noop():
    """The whole off-by-default invariant: observe never moves the speed."""
    c = SpeedController(1.0)
    for wpm in (0.0, 50.0, 165.0, 300.0, -10.0):
        assert c.observe(wpm) == 1.0
    assert c.current() == 1.0


def test_no_mirror_active_is_false():
    assert SpeedController(1.0).active is False


def test_no_mirror_preserves_non_unity_initial_speed():
    c = SpeedController(0.85)
    c.observe(250.0)
    assert c.current() == 0.85


# ---- With a mirror ---------------------------------------------------------


def test_active_true_with_mirror():
    assert SpeedController(1.0, mirror=_RecordingMirror(1.0)).active is True


def test_observe_delegates_to_mirror_with_current_speed():
    mirror = _RecordingMirror(1.2)
    c = SpeedController(1.0, mirror=mirror)
    out = c.observe(200.0)
    assert out == 1.2
    assert c.current() == 1.2
    assert mirror.calls == [{"user_wpm": 200.0, "current_speed": 1.0}]


def test_observe_feeds_updated_speed_on_next_call():
    """The second observe sees the speed the first one set as current_speed."""
    mirror = _SequenceMirror([1.1, 1.2])
    c = SpeedController(1.0, mirror=mirror)
    c.observe(220.0)
    c.observe(220.0)
    assert [call["current_speed"] for call in mirror.calls] == [1.0, 1.1]
    assert c.current() == 1.2


def test_observe_returns_new_speed():
    mirror = _RecordingMirror(0.9)
    c = SpeedController(1.0, mirror=mirror)
    assert c.observe(140.0) == 0.9


def test_observe_result_coerced_to_float():
    mirror = _RecordingMirror(1)  # int return
    c = SpeedController(1.0, mirror=mirror)
    assert isinstance(c.observe(165.0), float)
    assert c.current() == 1.0


def test_current_returns_live_value_across_turns():
    """current() always reflects the latest observe — the synth path reads it
    fresh per sentence."""
    mirror = _SequenceMirror([1.05, 1.1, 1.15])
    c = SpeedController(1.0, mirror=mirror)
    seen = [c.current()]
    for _ in range(3):
        c.observe(210.0)
        seen.append(c.current())
    assert seen == [1.0, 1.05, 1.1, 1.15]


# ---- Robustness: a misbehaving mirror must not break the loop -------------


def test_raising_mirror_keeps_prior_speed():
    c = SpeedController(1.0, mirror=_RaisingMirror())
    assert c.observe(200.0) == 1.0
    assert c.current() == 1.0


def test_nonnumeric_mirror_return_keeps_prior_speed():
    c = SpeedController(1.0, mirror=_NonNumericMirror())
    assert c.observe(200.0) == 1.0
    assert c.current() == 1.0


def test_raising_mirror_does_not_disable_future_observes():
    """A transient mirror failure shouldn't poison later good calls."""
    class _FlakyMirror:
        def __init__(self):
            self.n = 0

        def speed(self, *, user_wpm, current_speed):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("transient")
            return 1.3

    c = SpeedController(1.0, mirror=_FlakyMirror())
    assert c.observe(200.0) == 1.0   # first call raised → unchanged
    assert c.observe(200.0) == 1.3   # second call succeeds


# ---- Property accessor -----------------------------------------------------


def test_property_speed_matches_current():
    mirror = _RecordingMirror(1.25)
    c = SpeedController(1.0, mirror=mirror)
    c.observe(206.0)
    assert c.speed == c.current() == 1.25
