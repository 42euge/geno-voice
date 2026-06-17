"""Tests for iter-179 — the shared clause-pause/turn-end scalar invariant (#7).

Three modules across the organic turn-taking stack each carry a default that
their own docstrings repeatedly describe as *the same scalar*, the single
boundary between "a clause-boundary pause (the monologue / agent keeps going)"
and "a turn-end-sized gap (the user yielded the floor)":

  - ``turn_decider.py``        — ``TurnDeciderConfig.silence_floor_secs``
    (confidence is 0.0 at/below this: "a pause, not a turn-end").
  - ``backchannel_timing.py``  — ``BackchannelTimingConfig.max_pause_secs``
    (at/above this a gap is turn-end territory, handed to the silence path
    instead of a mid-speech backchannel).
  - ``monologue_clock.py``     — ``MonologueClockConfig.reset_gap_secs``
    (a ``stop → start`` gap at/above this starts a NEW monologue).

The three docstrings all claim the same thing — e.g. ``monologue_clock``:
"default 2.0s — exactly ``backchannel_timing.py``'s ``max_pause_secs`` and
``turn_decider.py``'s ``silence_floor_secs``, so the clause-pause/turn-end
boundary is one shared scalar across the whole organic stack and the three
paths can't drift."

But until this lap **nothing tested that cross-module equality**. Each module's
own suite only asserted its default ``== 2.0`` against a *hardcoded literal*
(``test_config_defaults_match_shared_silence_floor`` in test_monologue_clock,
``test_max_pause_equals_turn_decider_floor`` in test_backchannel_timing) — the
same hand-maintained-mirror trap iter-178 removed from the cue-rotation guard.
If someone retuned ``silence_floor_secs`` to 2.5 (a plausible "endpoint a touch
later" tweak), the monologue clock would still reset at 2.0 and the
backchannel-timing window would still close at 2.0, **silently** reopening the
[2.0, 2.5) overlap the shared scalar exists to prevent — the mid-speech
backchannel path and the silence-driven ``PLAY_CUE`` path would both fire in
that band — and every per-module ``== 2.0`` test would stay green because none
of them looks at a sibling. This module makes that drift a red test: it asserts
the three defaults are equal *to each other*, derived from the modules
themselves, not from a literal.

All three modules are pure stdlib (no pipecat at module scope), but they live
under ``session/`` whose ``__init__`` eagerly imports pipecat-dependent modules
(absent on the x86_64 Linux runner). So we load each by file path into a stub
``session`` namespace — the same trick the sibling tests use.
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
    ("session.turn_decider", "turn_decider.py"),
    ("session.backchannel_timing", "backchannel_timing.py"),
    ("session.monologue_clock", "monologue_clock.py"),
]:
    if _name not in sys.modules:
        _load_by_path(_name, _fn, package="session")

from session.backchannel_timing import BackchannelTimingConfig  # noqa: E402
from session.monologue_clock import MonologueClockConfig  # noqa: E402
from session.turn_decider import (  # noqa: E402
    DEFAULT_SILENCE_FLOOR_SECS,
    TurnDeciderConfig,
)


# The three defaults that the docstrings claim are one shared scalar.
def _silence_floor():
    return TurnDeciderConfig().silence_floor_secs


def _max_pause():
    return BackchannelTimingConfig().max_pause_secs


def _reset_gap():
    return MonologueClockConfig().reset_gap_secs


# --- the cross-module invariant (the gap this lap closes) -------------------


def test_clock_reset_gap_equals_turn_decider_silence_floor():
    """``MonologueClockConfig.reset_gap_secs`` must equal
    ``TurnDeciderConfig.silence_floor_secs`` — both are the clause-pause /
    turn-end boundary. Derived from the modules, not a literal, so retuning one
    without the other turns this red."""
    assert _reset_gap() == _silence_floor()


def test_backchannel_max_pause_equals_turn_decider_silence_floor():
    """``BackchannelTimingConfig.max_pause_secs`` must equal
    ``TurnDeciderConfig.silence_floor_secs`` — at/above the floor a gap is
    turn-end territory the silence-driven path owns, not a mid-speech
    backchannel."""
    assert _max_pause() == _silence_floor()


def test_backchannel_max_pause_equals_clock_reset_gap():
    """The two organic-stack consumers of the floor must also agree with each
    other (transitivity guard — if both equal the floor they equal each other,
    but pinning it directly localizes a failure to the right pair)."""
    assert _max_pause() == _reset_gap()


def test_all_three_defaults_are_one_shared_scalar():
    """The headline invariant: all three defaults are a single value. A
    three-way equality so a drift in *any* one of them fails here regardless of
    which module changed."""
    floor = _silence_floor()
    assert _reset_gap() == floor
    assert _max_pause() == floor
    # And the module-level constant the turn_decider config defaults from.
    assert DEFAULT_SILENCE_FLOOR_SECS == floor


def test_shared_scalar_is_currently_two_seconds():
    """Pins the present value (2.0s) so an *intentional* retune is a deliberate
    edit to this test, not a silent change. Distinct from the equality tests:
    those guarantee the three move together; this records where they sit today.
    """
    assert _silence_floor() == 2.0
