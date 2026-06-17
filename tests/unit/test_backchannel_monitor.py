"""Tests for iter-170 — stateful agent-backchannel monitor (backlog #7 driver).

``session/backchannel_monitor.py`` is the live-loop driver for the iter-153
``decide_backchannel_timing`` seam (backlog #7). The seam is stateless — it
demands the caller inject ``secs_since_last_backchannel``, which depends on the
monitor's *own past EMIT decisions*, not on anything a recorder measures. The
monitor owns that one scalar of cross-event state: it records the emit
timestamp the moment it decides ``EMIT``, so the *next* ``observe`` sees a real
``secs_since_last_backchannel`` and the rate limit engages.

The whole point is the **half-duplex invariant**: with a default
``FullDuplexConfig()`` (agent backchannels off) ``observe`` returns
``emit=False`` for every input and never mutates state — byte-for-byte today's
behavior. Only with ``agent_backchannels`` explicitly on does a well-timed
clause-boundary pause yield ``emit=True`` and arm the rate limit.

``backchannel_monitor`` does ``from session.* import ...`` at module scope, but
``session/__init__.py`` eagerly imports pipecat-dependent modules (absent on
the x86_64 Linux runner). So we stand up a stub ``session`` namespace package
and load the dependency chain by file path — the same trick
test_backchannel_timing.py / test_barge_decision.py / test_text_eou.py use.
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
if "session.backchannel_timing" not in sys.modules:
    _load_by_path(
        "session.backchannel_timing",
        "backchannel_timing.py",
        package="session",
    )

_bm = _load_by_path(
    "session.backchannel_monitor", "backchannel_monitor.py", package="session"
)

BackchannelMonitor = _bm.BackchannelMonitor
BackchannelDecision = _bm.BackchannelDecision

FullDuplexConfig = sys.modules["session.full_duplex"].FullDuplexConfig
BackchannelTimingConfig = sys.modules[
    "session.backchannel_timing"
].BackchannelTimingConfig


# ---- config shorthands ------------------------------------------------------

def _organic() -> "FullDuplexConfig":
    """Master full-duplex on ⇒ agent_backchannels resolves True."""
    return FullDuplexConfig(enabled=True)


def _organic_bc_on() -> "FullDuplexConfig":
    """Master off but the agent-backchannels sub-flag forced on."""
    return FullDuplexConfig(enabled=False, agent_backchannels=True)


def _organic_bc_off() -> "FullDuplexConfig":
    """Master on but the agent-backchannels sub-flag forced OFF."""
    return FullDuplexConfig(enabled=True, agent_backchannels=False)


# A timing config with tiny thresholds so the arithmetic in tests is obvious.
#   warm-up    : 10s of monologue before the first cue
#   rate limit : 5s between cues
#   pause window: [0.3, 2.0)
def _timing() -> "BackchannelTimingConfig":
    return BackchannelTimingConfig(
        min_speaking_before_first_cue_secs=10.0,
        min_between_cues_secs=5.0,
        min_pause_secs=0.3,
        max_pause_secs=2.0,
    )


# A monologue start anchored at 0 so ``now`` *is* ``user_speaking_secs``.
def _mon(config=None, timing=None) -> "BackchannelMonitor":
    return BackchannelMonitor(
        config=config if config is not None else _organic(),
        timing=timing if timing is not None else _timing(),
    )


# ---- half-duplex invariant --------------------------------------------------

def test_default_config_is_half_duplex_inert():
    """Default config ⇒ never emits, never records, regardless of timing."""
    m = BackchannelMonitor()  # default half-duplex
    assert m.active is False
    # A textbook-perfect opportunity: long monologue, ideal pause.
    d = m.observe(now=100.0, monologue_start_at=0.0, pause_secs=0.5)
    assert d.emit is False
    assert m.last_backchannel_at is None
    assert m.emit_count == 0


def test_sub_flag_off_under_master_on_is_inert():
    """Master on but agent_backchannels explicitly off ⇒ inert."""
    m = _mon(config=_organic_bc_off())
    assert m.active is False
    d = m.observe(now=100.0, monologue_start_at=0.0, pause_secs=0.5)
    assert d.emit is False
    assert m.last_backchannel_at is None


def test_sub_flag_on_under_master_off_is_active():
    """Master off but agent_backchannels forced on ⇒ active."""
    m = _mon(config=_organic_bc_on())
    assert m.active is True


# ---- the gates (organic mode) ----------------------------------------------

def test_warm_up_holds_short_monologue():
    """Below the warm-up floor (10s), a perfect pause still holds."""
    m = _mon()
    d = m.observe(now=5.0, monologue_start_at=0.0, pause_secs=0.5)
    assert d.emit is False
    assert d.user_speaking_secs == 5.0
    assert m.last_backchannel_at is None


def test_no_pause_holds():
    """Continuous speech (pause below the min) holds even past warm-up."""
    m = _mon()
    d = m.observe(now=30.0, monologue_start_at=0.0, pause_secs=0.05)
    assert d.emit is False


def test_turn_end_sized_gap_holds():
    """A gap at/above max_pause is the silence-path's job ⇒ hold."""
    m = _mon()
    d = m.observe(now=30.0, monologue_start_at=0.0, pause_secs=2.5)
    assert d.emit is False


def test_clause_pause_past_warmup_emits():
    """Long monologue + clause-boundary pause ⇒ emit, and state is recorded."""
    m = _mon()
    d = m.observe(now=12.0, monologue_start_at=0.0, pause_secs=0.5)
    assert d.emit is True
    assert d.user_speaking_secs == 12.0
    assert d.secs_since_last_backchannel is None  # first emit
    assert m.last_backchannel_at == 12.0
    assert m.emit_count == 1


# ---- the stateful piece: rate limit across calls ---------------------------

def test_rate_limit_engages_after_first_emit():
    """The second qualifying pause too soon after the first is rate-limited."""
    m = _mon()
    first = m.observe(now=12.0, monologue_start_at=0.0, pause_secs=0.5)
    assert first.emit is True
    # 3s later — below the 5s rate limit — another perfect pause: HOLD.
    second = m.observe(now=15.0, monologue_start_at=0.0, pause_secs=0.5)
    assert second.emit is False
    assert second.secs_since_last_backchannel == 3.0
    # last-emit unchanged (a HOLD never updates it), count unchanged.
    assert m.last_backchannel_at == 12.0
    assert m.emit_count == 1


def test_rate_limit_clears_after_interval():
    """Once min_between_cues has elapsed, a qualifying pause emits again."""
    m = _mon()
    m.observe(now=12.0, monologue_start_at=0.0, pause_secs=0.5)
    # 6s later (> 5s rate limit): emits, and the timestamp advances.
    third = m.observe(now=18.0, monologue_start_at=0.0, pause_secs=0.5)
    assert third.emit is True
    assert third.secs_since_last_backchannel == 6.0
    assert m.last_backchannel_at == 18.0
    assert m.emit_count == 2


def test_repeated_qualifying_pauses_do_not_chatter():
    """Without the monitor's state, the pure seam would emit every frame.

    Drive a stream of qualifying pauses 1s apart; only the ones spaced beyond
    the rate limit fire. This is the regression the driver exists to prevent.
    """
    m = _mon()
    emits = []
    for t in range(11, 30):  # 11s..29s, every 1s, all perfect pauses
        d = m.observe(now=float(t), monologue_start_at=0.0, pause_secs=0.5)
        if d.emit:
            emits.append(t)
    # First fires at 11s (past 10s warm-up); next not until 11+5=16, then 21,
    # 26 — every 5s, not every second.
    assert emits == [11, 16, 21, 26]
    assert m.emit_count == 4


# ---- reset ------------------------------------------------------------------

def test_reset_clears_rate_limit_but_keeps_count():
    """reset() lets the next observe emit immediately; emit_count persists."""
    m = _mon()
    m.observe(now=12.0, monologue_start_at=0.0, pause_secs=0.5)
    assert m.emit_count == 1
    m.reset()
    assert m.last_backchannel_at is None
    # Immediately after reset, a qualifying pause emits (no rate limit).
    d = m.observe(now=13.0, monologue_start_at=12.0, pause_secs=0.5)
    # user_speaking_secs measured from the NEW monologue start.
    assert d.user_speaking_secs == pytest.approx(1.0)
    # Still below warm-up (1s < 10s) ⇒ holds. Use a fresh long monologue:
    d2 = m.observe(now=25.0, monologue_start_at=12.0, pause_secs=0.5)
    assert d2.emit is True
    assert d2.secs_since_last_backchannel is None  # reset cleared it
    assert m.emit_count == 2  # lifetime tally preserved across reset


# ---- defensive clamps -------------------------------------------------------

def test_negative_monologue_duration_clamped():
    """now before monologue_start (clock skew) ⇒ user_speaking_secs clamps 0."""
    m = _mon()
    d = m.observe(now=5.0, monologue_start_at=8.0, pause_secs=0.5)
    assert d.user_speaking_secs == 0.0
    assert d.emit is False  # 0s monologue is below warm-up


def test_negative_since_last_clamped():
    """A 'now' before the recorded emit (skew) clamps secs_since to >= 0."""
    m = _mon()
    m.observe(now=12.0, monologue_start_at=0.0, pause_secs=0.5)  # emit at 12
    # now=11 (before the emit) — skew. secs_since clamps to 0, not negative.
    d = m.observe(now=11.0, monologue_start_at=0.0, pause_secs=0.5)
    assert d.secs_since_last_backchannel == 0.0
    assert d.emit is False  # 0 < 5s rate limit ⇒ held


# ---- decision dataclass / observability ------------------------------------

def test_decision_echoes_inputs():
    """The returned dataclass exposes the derived quantities for observability."""
    m = _mon()
    d = m.observe(now=14.0, monologue_start_at=2.0, pause_secs=0.7)
    assert isinstance(d, BackchannelDecision)
    assert d.user_speaking_secs == 12.0
    assert d.pause_secs == 0.7
    assert d.secs_since_last_backchannel is None


def test_decision_is_frozen():
    m = _mon()
    d = m.observe(now=12.0, monologue_start_at=0.0, pause_secs=0.5)
    with pytest.raises(Exception):
        d.emit = False  # frozen dataclass


# ---- default timing config (sanity that real defaults compose) -------------

def test_default_timing_warm_up_is_15s():
    """With the real default timing (15s warm-up), a 12s monologue holds."""
    m = BackchannelMonitor(config=_organic())  # default timing
    d = m.observe(now=12.0, monologue_start_at=0.0, pause_secs=0.5)
    assert d.emit is False  # 12 < 15 default warm-up
    d2 = m.observe(now=16.0, monologue_start_at=0.0, pause_secs=0.5)
    assert d2.emit is True  # 16 >= 15
