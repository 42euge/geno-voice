"""Tests for iter-174 — the backchannel driver (backlog #7 composition).

``session/backchannel_driver.py`` composes ``MonologueClock`` (iter-173, the
VAD-event clock) and ``BackchannelMonitor`` (iter-170, the stateful emit/cue
driver) into one object, so the live ``pipecat_server.py`` ``Broadcaster``
wiring is a truly thin shim: ``on_speech_start`` / ``on_speech_stop`` off the
VAD frames, ``observe(now)`` on a tick, ``broadcast_cue(d.cue_type)`` on emit.
It also closes the latent ``TypeError`` in the hand-spelled composition: an
``observe`` tick before any speech start (``monologue_start_at is None``) used
to hit ``BackchannelMonitor.observe``'s unguarded ``now - None`` subtraction;
the driver short-circuits that to a no-emit decision.

The module is dependency-free beyond its two organic-track siblings, but it
lives under ``session/`` whose ``__init__`` eagerly imports pipecat-dependent
modules (absent on the x86_64 Linux runner). So we load it (and its dep chain)
by file path into a stub ``session`` namespace — the same trick
test_monologue_clock.py / test_backchannel_monitor.py use.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

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
for _name, _fn in [
    ("session.full_duplex", "full_duplex.py"),
    ("session.cue_rotation", "cue_rotation.py"),
    ("session.backchannel_timing", "backchannel_timing.py"),
    ("session.backchannel_monitor", "backchannel_monitor.py"),
    ("session.monologue_clock", "monologue_clock.py"),
    ("session.backchannel_driver", "backchannel_driver.py"),
]:
    if _name not in sys.modules:
        _load_by_path(_name, _fn, package="session")

from session.backchannel_driver import BackchannelDriver  # noqa: E402
from session.backchannel_monitor import BackchannelMonitor  # noqa: E402
from session.cue_rotation import CUE_ROTATION  # noqa: E402
from session.full_duplex import FullDuplexConfig  # noqa: E402
from session.monologue_clock import MonologueClock, MonologueClockConfig  # noqa: E402


def _organic():
    return FullDuplexConfig(enabled=True, agent_backchannels=True)


# --- construction / wiring --------------------------------------------------

def test_default_construction_builds_both_collaborators():
    d = BackchannelDriver()
    assert isinstance(d.clock, MonologueClock)
    assert isinstance(d.monitor, BackchannelMonitor)


def test_default_is_half_duplex_inactive():
    assert BackchannelDriver().active is False


def test_organic_config_is_active():
    assert BackchannelDriver(config=_organic()).active is True


def test_injected_collaborators_take_precedence():
    clock = MonologueClock()
    monitor = BackchannelMonitor(config=_organic())
    d = BackchannelDriver(clock=clock, monitor=monitor)
    assert d.clock is clock
    assert d.monitor is monitor
    assert d.active is True


def test_clock_config_forwarded():
    d = BackchannelDriver(clock_config=MonologueClockConfig(reset_gap_secs=0.5))
    # A 0.6s gap (>= 0.5) resets the monologue under the custom config.
    d.on_speech_start(0.0)
    d.on_speech_stop(1.0)
    d.on_speech_start(1.6)
    assert d.monologue_start_at == 1.6


# --- the None-tick crash guard (the bug this lap closes) --------------------

def test_observe_before_any_speech_does_not_crash():
    """The hand-spelled composition passed monologue_start_at=None straight to
    BackchannelMonitor.observe, which does ``now - None`` ⇒ TypeError. The
    driver short-circuits instead."""
    d = BackchannelDriver(config=_organic())
    decision = d.observe(now=5.0)  # no on_speech_start yet
    assert decision.emit is False
    assert decision.cue_type is None
    assert decision.user_speaking_secs == 0.0


def test_observe_after_reset_does_not_crash():
    d = BackchannelDriver(config=_organic())
    d.on_speech_start(0.0)
    d.reset()
    decision = d.observe(now=3.0)
    assert decision.emit is False
    assert decision.cue_type is None


def test_none_tick_reports_secs_since_last_backchannel():
    """If a backchannel was emitted, then the monologue resets, a None-start
    tick still reports the real gap since that emit (not a crash, not None)."""
    d = BackchannelDriver(config=_organic())
    d.on_speech_start(0.0)
    d.on_speech_stop(16.0)
    emit = d.observe(now=16.5)
    assert emit.emit is True
    # Wipe the clock's monologue but keep the monitor's last-emit timestamp.
    d.clock.reset()
    tick = d.observe(now=20.0)
    assert tick.emit is False
    assert tick.secs_since_last_backchannel == 20.0 - 16.5


def test_none_tick_secs_since_is_none_when_never_emitted():
    d = BackchannelDriver(config=_organic())
    tick = d.observe(now=5.0)
    assert tick.secs_since_last_backchannel is None


def test_none_tick_secs_since_sources_monitor_accessor():
    """The no-monologue short-circuit reports exactly the monitor's own
    secs_since_last_backchannel(now) — not a driver-local recompute — so the
    None guard and skew clamp can't drift from the in-decision derivation."""
    d = BackchannelDriver(config=_organic())
    d.on_speech_start(0.0)
    d.on_speech_stop(16.0)
    assert d.observe(now=16.5).emit is True  # arm the monitor's last-emit clock
    d.clock.reset()  # wipe the monologue, keep the monitor's emit timestamp
    tick = d.observe(now=20.0)
    assert tick.emit is False
    assert tick.secs_since_last_backchannel == d.monitor.secs_since_last_backchannel(
        now=20.0
    )


def test_none_tick_echoes_pause_secs_zero_before_speech():
    d = BackchannelDriver(config=_organic())
    tick = d.observe(now=5.0)
    assert tick.pause_secs == 0.0


# --- end-to-end emit through a clause pause ---------------------------------

def test_emits_in_clause_pause_window():
    """The whole point: drive the driver off VAD-shaped events and get an emit
    exactly in the clause-pause window after warm-up — no hand composition."""
    d = BackchannelDriver(config=_organic())
    d.on_speech_start(0.0)
    # Still speaking past warm-up, no pause ⇒ HOLD.
    assert d.observe(now=16.0).emit is False
    # Clause-boundary pause.
    d.on_speech_stop(16.0)
    decision = d.observe(now=16.5)  # 0.5s into the pause — in [0.3, 2.0)
    assert decision.emit is True
    assert decision.cue_type == CUE_ROTATION[0]  # "mhmm"
    assert d.emit_count == 1


def test_monologue_continues_across_clause_pause():
    d = BackchannelDriver(config=_organic())
    d.on_speech_start(0.0)
    d.on_speech_stop(16.0)
    d.observe(now=16.5)
    d.on_speech_start(16.6)  # short gap ⇒ same monologue
    assert d.monologue_start_at == 0.0


def test_consecutive_emits_rotate_cues():
    """Two emits separated by enough time rotate through the cue bank."""
    d = BackchannelDriver(config=_organic())
    d.on_speech_start(0.0)
    d.on_speech_stop(16.0)
    first = d.observe(now=16.5)
    assert first.cue_type == CUE_ROTATION[0]
    # Resume, run long again, pause again well past the rate limit.
    d.on_speech_start(16.6)
    d.on_speech_stop(40.0)
    second = d.observe(now=40.5)
    assert second.emit is True
    assert second.cue_type == CUE_ROTATION[1]
    assert d.cue_index == 2


def test_rate_limit_holds_second_emit_too_soon():
    """A second clause pause too soon after an emit is rate-limited (HOLD)."""
    d = BackchannelDriver(config=_organic())
    d.on_speech_start(0.0)
    d.on_speech_stop(16.0)
    assert d.observe(now=16.5).emit is True
    # Another pause only ~1s later — within the 20s min-between-cues window.
    d.on_speech_start(16.6)
    d.on_speech_stop(17.6)
    assert d.observe(now=18.1).emit is False
    assert d.emit_count == 1


# --- half-duplex invariant --------------------------------------------------

def test_half_duplex_never_emits():
    d = BackchannelDriver()  # default ⇒ half-duplex
    d.on_speech_start(0.0)
    d.on_speech_stop(20.0)
    decision = d.observe(now=20.5)  # a perfect pause window
    assert decision.emit is False
    assert decision.cue_type is None
    assert d.emit_count == 0


def test_half_duplex_never_mutates_monitor_state():
    d = BackchannelDriver()
    d.on_speech_start(0.0)
    d.on_speech_stop(20.0)
    d.observe(now=20.5)
    assert d.last_backchannel_at is None
    assert d.cue_index == 0


# --- passthrough views ------------------------------------------------------

def test_speaking_passthrough():
    d = BackchannelDriver()
    assert d.speaking is False
    d.on_speech_start(0.0)
    assert d.speaking is True
    d.on_speech_stop(1.0)
    assert d.speaking is False


def test_reset_clears_clock_but_keeps_rotation():
    d = BackchannelDriver(config=_organic())
    d.on_speech_start(0.0)
    d.on_speech_stop(16.0)
    d.observe(now=16.5)  # emit ⇒ cue_index advances to 1
    assert d.cue_index == 1
    d.reset()
    assert d.monologue_start_at is None
    assert d.last_backchannel_at is None  # rate limit cleared
    assert d.cue_index == 1  # rotation position survives reset
    assert d.emit_count == 1  # lifetime tally survives reset


def test_reset_rate_limit_allows_immediate_reemit():
    """After reset the rate limit is cleared, so a fresh monologue can emit
    again even though it would otherwise be inside the min-between window."""
    d = BackchannelDriver(config=_organic())
    d.on_speech_start(0.0)
    d.on_speech_stop(16.0)
    assert d.observe(now=16.5).emit is True
    d.reset()
    d.on_speech_start(17.0)
    d.on_speech_stop(33.0)
    # Only ~16.5s after the prior emit — would be rate-limited without reset.
    assert d.observe(now=33.5).emit is True
    assert d.cue_index == 2  # continued the rotation, not replayed
