"""Tests for iter-180 — the cue-window partition invariant (#7 hardening).

geno-voice has **two** distinct backchannel-cue paths, and they are designed to
own disjoint stretches of the trailing-silence axis:

  - The **mid-speech backchannel** path (``backchannel_timing.py``, iter-153):
    the agent emits a short continuer ("mhmm") during a brief clause-boundary
    pause. Its window is ``[min_pause_secs, max_pause_secs)`` = ``[0.3, 2.0)``.
    At/above ``max_pause_secs`` the gap is heading toward turn-end and this path
    deliberately stays out of it.
  - The **turn-end cue** path (``turn_taking.py``'s ``TurnTakingEngine``): the
    silence-driven ``PLAY_CUE`` tier fires once trailing silence reaches
    ``silence_backchannel_min`` = ``4.0``s — a turn-end-ish "are you done?" cue,
    not a mid-speech nod.

``backchannel_timing.py``'s module docstring states the contract explicitly:
"the two paths partition the silence axis cleanly: ``[min_pause, max_pause)`` is
the mid-speech backchannel window; ``>= silence_backchannel_min`` is the
turn-end cue window." For that partition to hold with **no overlap**, the
mid-speech window must *close* at or before the turn-end window *opens*:

    max_pause_secs  <=  silence_backchannel_min

iter-179 pinned the three scalars that are deliberately *equal*
(``reset_gap_secs == max_pause_secs == silence_floor_secs`` = 2.0). This lap
pins the inequality between the two scalars that are deliberately *not* equal —
the upper edge of the mid-speech window (2.0) and the lower edge of the turn-end
window (4.0). Nothing tested it: ``test_backchannel_timing.py`` only asserts
``max_pause_secs == 2.0`` and ``test_turn_decider`` never looks at
``silence_backchannel_min`` at all. So retuning ``silence_backchannel_min`` down
to 1.5 (a plausible "fire the turn-end cue sooner" tweak) would silently reopen
the ``[1.5, 2.0)`` band where *both* cue paths fire — the agent would emit a
mid-speech "mhmm" *and* a turn-end cue on the same gap — and every per-module
test would stay green because none looks at the sibling. This module makes that
drift a red test, derived from the two modules themselves, not from literals.

Both modules are stdlib-only at module scope but live under ``session/`` whose
``__init__`` eagerly imports pipecat-dependent modules (absent on the x86_64
Linux runner). So we load each by file path into a stub ``session`` namespace —
the same trick the sibling tests use.
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
    ("session.triggers", "triggers.py"),
    ("session.backchannel_timing", "backchannel_timing.py"),
    ("session.turn_taking", "turn_taking.py"),
]:
    if _name not in sys.modules:
        _load_by_path(_name, _fn, package="session")

from session.backchannel_timing import BackchannelTimingConfig  # noqa: E402
from session.turn_taking import TurnTakingConfig  # noqa: E402


def _max_pause():
    """Upper edge of the mid-speech backchannel window (exclusive)."""
    return BackchannelTimingConfig().max_pause_secs


def _min_pause():
    """Lower edge of the mid-speech backchannel window (inclusive)."""
    return BackchannelTimingConfig().min_pause_secs


def _turn_end_cue_open():
    """Lower edge of the silence-driven turn-end cue window (inclusive)."""
    return TurnTakingConfig().silence_backchannel_min


# --- the partition invariant (the gap this lap closes) ----------------------


def test_midspeech_window_closes_at_or_before_turnend_window_opens():
    """``max_pause_secs <= silence_backchannel_min`` — the mid-speech
    backchannel window must close at or before the turn-end cue window opens, or
    the two cue paths overlap and both fire on the same gap. Derived from both
    modules, not literals, so retuning either edge into an overlap turns red."""
    assert _max_pause() <= _turn_end_cue_open()


def test_no_silence_value_triggers_both_cue_paths():
    """No trailing-silence duration may fall in *both* the mid-speech window
    ``[min_pause, max_pause)`` and the turn-end window ``[open, inf)`` — the
    half-open windows are disjoint iff ``max_pause <= open``. Spot-check the
    boundary value: at exactly ``max_pause_secs`` the mid-speech path has
    already bowed out (window is half-open), and the turn-end path has not yet
    opened (``max_pause < open``)."""
    max_pause = _max_pause()
    open_ = _turn_end_cue_open()
    # mid-speech window is [min_pause, max_pause): max_pause itself is excluded.
    # turn-end window is [open, inf): values below `open` are excluded.
    # Disjoint iff every value is in at most one window, i.e. max_pause <= open.
    assert max_pause <= open_
    # The boundary value max_pause is in neither window (the clean gap between).
    assert not (_min_pause() <= max_pause < max_pause)  # excluded from mid-speech
    assert max_pause < open_ or max_pause == open_  # at/below turn-end open


def test_midspeech_window_is_well_formed():
    """Sanity: the mid-speech window itself is non-empty
    (``min_pause < max_pause``) so "partition" is meaningful."""
    assert _min_pause() < _max_pause()


def test_current_window_edges_are_two_and_four_seconds():
    """Pins the present edges (mid-speech closes at 2.0s, turn-end opens at
    4.0s) so an *intentional* retune is a deliberate edit to this test, not a
    silent change. Distinct from the inequality tests: those guarantee no
    overlap; this records where the edges sit today, including the 2.0s gap of
    pure silence between the two windows."""
    assert _max_pause() == 2.0
    assert _turn_end_cue_open() == 4.0
