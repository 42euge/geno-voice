"""Tests for iter-173 — the monologue clock (backlog #7 VAD-event driver).

``session/monologue_clock.py`` is the VAD-event driver that feeds
``BackchannelMonitor.observe`` (iter-170) the two derived quantities it cannot
compute itself: ``monologue_start_at`` (when the current monologue began — a
speech run that survives brief clause pauses but resets on a turn-end gap) and
``pause_secs`` (the live within-speech gap). It consumes
``on_speech_start(now)`` / ``on_speech_stop(now)`` events — the same
``VADUserStartedSpeakingFrame`` / ``VADUserStoppedSpeakingFrame`` stream the
live ``Broadcaster`` sees.

The module is dependency-free, but it lives under ``session/`` whose
``__init__`` eagerly imports pipecat-dependent modules (absent on the x86_64
Linux runner). So we load it by file path into a stub ``session`` namespace —
the same trick test_backchannel_monitor.py / test_backchannel_timing.py use.
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
if "session.monologue_clock" not in sys.modules:
    _load_by_path("session.monologue_clock", "monologue_clock.py", package="session")

from session.monologue_clock import MonologueClock, MonologueClockConfig  # noqa: E402


# --- config validation -----------------------------------------------------

def test_config_defaults_match_shared_silence_floor():
    # 2.0s is the shared clause-pause/turn-end scalar across backchannel_timing
    # (max_pause_secs) and turn_decider (silence_floor_secs).
    assert MonologueClockConfig().reset_gap_secs == 2.0


@pytest.mark.parametrize("bad", [0.0, -0.5])
def test_config_rejects_non_positive_reset_gap(bad):
    with pytest.raises(ValueError):
        MonologueClockConfig(reset_gap_secs=bad)


# --- initial state ----------------------------------------------------------

def test_fresh_clock_has_no_monologue_and_zero_pause():
    clock = MonologueClock()
    assert clock.monologue_start_at is None
    assert clock.last_stop_at is None
    assert clock.speaking is False
    assert clock.pause_secs(100.0) == 0.0


# --- monologue start --------------------------------------------------------

def test_first_speech_start_begins_monologue():
    clock = MonologueClock()
    clock.on_speech_start(10.0)
    assert clock.monologue_start_at == 10.0
    assert clock.speaking is True
    # No pause while speaking.
    assert clock.pause_secs(12.0) == 0.0


# --- pause tracking ---------------------------------------------------------

def test_pause_grows_after_stop():
    clock = MonologueClock()
    clock.on_speech_start(10.0)
    clock.on_speech_stop(12.0)
    assert clock.speaking is False
    assert clock.last_stop_at == 12.0
    assert clock.pause_secs(12.5) == pytest.approx(0.5)
    assert clock.pause_secs(13.0) == pytest.approx(1.0)


def test_pause_clamps_against_clock_skew():
    clock = MonologueClock()
    clock.on_speech_start(10.0)
    clock.on_speech_stop(12.0)
    # now < last_stop (skew) ⇒ clamped to 0, never negative.
    assert clock.pause_secs(11.0) == 0.0


def test_pause_zero_again_after_resuming_speech():
    clock = MonologueClock()
    clock.on_speech_start(10.0)
    clock.on_speech_stop(12.0)
    assert clock.pause_secs(12.5) == pytest.approx(0.5)
    clock.on_speech_start(12.6)  # short gap ⇒ resume
    assert clock.pause_secs(13.0) == 0.0


# --- user_speaking_secs: the third derived accessor -------------------------

def test_user_speaking_secs_zero_before_any_speech():
    # No monologue yet ⇒ 0.0, never a TypeError from ``now - None``.
    clock = MonologueClock()
    assert clock.user_speaking_secs(100.0) == 0.0


def test_user_speaking_secs_grows_with_monologue():
    clock = MonologueClock()
    clock.on_speech_start(10.0)
    assert clock.user_speaking_secs(16.0) == pytest.approx(6.0)


def test_user_speaking_secs_counts_across_a_clause_pause():
    # The whole point of the monologue clock: speaking time accumulates across
    # a brief clause-boundary pause (monologue_start_at unchanged).
    clock = MonologueClock()
    clock.on_speech_start(10.0)
    clock.on_speech_stop(11.0)
    clock.on_speech_start(11.5)  # 0.5s gap < 2.0s ⇒ same monologue
    assert clock.user_speaking_secs(14.0) == pytest.approx(4.0)


def test_user_speaking_secs_resets_with_a_new_monologue():
    clock = MonologueClock()
    clock.on_speech_start(10.0)
    clock.on_speech_stop(12.0)
    clock.on_speech_start(15.0)  # 3.0s gap >= 2.0s ⇒ new monologue at 15.0
    assert clock.user_speaking_secs(16.0) == pytest.approx(1.0)


def test_user_speaking_secs_clamps_against_clock_skew():
    # now < monologue_start_at (skew) ⇒ clamped to 0, never negative.
    clock = MonologueClock()
    clock.on_speech_start(10.0)
    assert clock.user_speaking_secs(9.0) == 0.0


def test_user_speaking_secs_zero_after_reset():
    clock = MonologueClock()
    clock.on_speech_start(10.0)
    clock.reset()
    assert clock.user_speaking_secs(20.0) == 0.0


def test_user_speaking_secs_keeps_counting_during_a_pause():
    # The accessor measures monologue length, not active-speech length, so it
    # keeps growing while the user is paused mid-monologue (the pause hasn't
    # yet been resolved as a turn-end).
    clock = MonologueClock()
    clock.on_speech_start(10.0)
    clock.on_speech_stop(12.0)
    assert clock.user_speaking_secs(13.0) == pytest.approx(3.0)


# --- clause-boundary pause: monologue CONTINUES -----------------------------

def test_short_gap_continues_monologue():
    clock = MonologueClock()
    clock.on_speech_start(10.0)
    clock.on_speech_stop(11.0)
    # 0.5s gap < 2.0s reset ⇒ clause pause, same monologue.
    clock.on_speech_start(11.5)
    assert clock.monologue_start_at == 10.0  # unchanged
    # user_speaking_secs keeps accumulating across the clause pause.
    assert clock.monologue_start_at is not None
    assert 14.0 - clock.monologue_start_at == pytest.approx(4.0)


def test_multiple_clause_pauses_keep_one_monologue():
    clock = MonologueClock()
    clock.on_speech_start(10.0)
    for stop, start in [(11.0, 11.4), (13.0, 13.5), (15.0, 15.2)]:
        clock.on_speech_stop(stop)
        clock.on_speech_start(start)
    assert clock.monologue_start_at == 10.0


# --- turn-end gap: monologue RESETS -----------------------------------------

def test_long_gap_resets_monologue():
    clock = MonologueClock()
    clock.on_speech_start(10.0)
    clock.on_speech_stop(12.0)
    # 3.0s gap >= 2.0s reset ⇒ user yielded the floor, new monologue.
    clock.on_speech_start(15.0)
    assert clock.monologue_start_at == 15.0


def test_gap_exactly_at_threshold_resets():
    # >= reset_gap_secs resets (boundary is inclusive on the reset side).
    clock = MonologueClock()
    clock.on_speech_start(10.0)
    clock.on_speech_stop(12.0)
    clock.on_speech_start(14.0)  # exactly 2.0s gap
    assert clock.monologue_start_at == 14.0


def test_gap_just_below_threshold_continues():
    clock = MonologueClock()
    clock.on_speech_start(10.0)
    clock.on_speech_stop(12.0)
    clock.on_speech_start(13.99)  # 1.99s gap < 2.0s
    assert clock.monologue_start_at == 10.0


def test_custom_reset_gap_threshold():
    clock = MonologueClock(config=MonologueClockConfig(reset_gap_secs=0.5))
    clock.on_speech_start(10.0)
    clock.on_speech_stop(11.0)
    clock.on_speech_start(11.6)  # 0.6s gap >= 0.5 ⇒ reset
    assert clock.monologue_start_at == 11.6


# --- reset ------------------------------------------------------------------

def test_reset_clears_all_state():
    clock = MonologueClock()
    clock.on_speech_start(10.0)
    clock.on_speech_stop(12.0)
    clock.reset()
    assert clock.monologue_start_at is None
    assert clock.last_stop_at is None
    assert clock.speaking is False
    assert clock.pause_secs(20.0) == 0.0


def test_start_after_reset_begins_fresh_monologue_regardless_of_gap():
    clock = MonologueClock()
    clock.on_speech_start(10.0)
    clock.on_speech_stop(12.0)
    clock.reset()
    # Even a tiny gap after reset begins a brand-new monologue (no last_stop).
    clock.on_speech_start(12.1)
    assert clock.monologue_start_at == 12.1


# --- integration with BackchannelMonitor ------------------------------------

def test_feeds_backchannel_monitor_through_a_clause_pause():
    """End-to-end: the clock's outputs drive a real BackchannelMonitor emit.

    Across a long monologue with a clause-boundary pause, the clock keeps one
    ``monologue_start_at`` (so warm-up clears) and reports the live pause, and
    an organic-mode monitor emits exactly in the clause-pause window.
    """
    # Load the monitor + its deps by path (same stub-session trick).
    for name, fn in [
        ("session.full_duplex", "full_duplex.py"),
        ("session.cue_rotation", "cue_rotation.py"),
        ("session.backchannel_timing", "backchannel_timing.py"),
        ("session.backchannel_monitor", "backchannel_monitor.py"),
    ]:
        if name not in sys.modules:
            _load_by_path(name, fn, package="session")
    from session.backchannel_monitor import BackchannelMonitor
    from session.full_duplex import FullDuplexConfig

    organic = FullDuplexConfig(enabled=True, agent_backchannels=True)
    monitor = BackchannelMonitor(config=organic)
    clock = MonologueClock()

    clock.on_speech_start(0.0)
    # Still speaking at t=16 (past 15s warm-up), no pause ⇒ HOLD (pause 0).
    d = monitor.observe(
        now=16.0,
        monologue_start_at=clock.monologue_start_at,
        pause_secs=clock.pause_secs(16.0),
    )
    assert d.emit is False  # no pause yet

    # User pauses at a clause boundary at t=16.
    clock.on_speech_stop(16.0)
    d = monitor.observe(
        now=16.5,  # 0.5s into the pause — in [0.3, 2.0) window
        monologue_start_at=clock.monologue_start_at,
        pause_secs=clock.pause_secs(16.5),
    )
    assert d.emit is True
    assert d.cue_type == "mhmm"  # first cue in the rotation

    # The pause was a clause boundary, not a turn-end: monologue continues.
    clock.on_speech_start(16.6)
    assert clock.monologue_start_at == 0.0


def test_half_duplex_monitor_never_emits_with_clock_inputs():
    """The half-duplex invariant survives the clock wiring: default config
    monitor never emits even when the clock reports a perfect pause window."""
    for name, fn in [
        ("session.full_duplex", "full_duplex.py"),
        ("session.cue_rotation", "cue_rotation.py"),
        ("session.backchannel_timing", "backchannel_timing.py"),
        ("session.backchannel_monitor", "backchannel_monitor.py"),
    ]:
        if name not in sys.modules:
            _load_by_path(name, fn, package="session")
    from session.backchannel_monitor import BackchannelMonitor

    monitor = BackchannelMonitor()  # default ⇒ half-duplex
    clock = MonologueClock()
    clock.on_speech_start(0.0)
    clock.on_speech_stop(20.0)
    d = monitor.observe(
        now=20.5,
        monologue_start_at=clock.monologue_start_at,
        pause_secs=clock.pause_secs(20.5),
    )
    assert d.emit is False
    assert d.cue_type is None
