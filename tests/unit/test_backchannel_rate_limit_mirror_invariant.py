"""Tests for iter-181 — the backchannel rate-limit mirror invariant (#7 hardening).

The two backchannel-cue paths share more than the silence-axis partition that
iter-179/180 pinned. They also share two **rate limits** — by deliberate
design, so the agent's mid-speech "mhmm" and its silence-driven turn-end cue
feel like the same speaker obeying one set of manners, not two subsystems with
divergent tempos:

  - ``min_speaking_before_first_cue_secs`` — the user must have been talking at
    least this long before *either* path emits its first cue (don't backchannel
    over a one-word reply). Default 15.0s in both.
  - ``min_between_cues_secs`` — at least this long must pass between cues from
    *either* path (no "mhmm mhmm mhmm" chatter). Default 20.0s in both.

``backchannel_timing.py``'s ``BackchannelTimingConfig`` docstring states the
contract explicitly: "Defaults mirror the analogous knobs the ``TurnTakingEngine``
already uses for its trailing-silence cues so the two paths feel consistent ...
the same rate limits as ``TurnTakingConfig`` (15.0 / 20.0)." So the invariant is
a cross-module **equality**:

    BackchannelTimingConfig().min_speaking_before_first_cue_secs
        == TurnTakingConfig().min_speaking_before_first_cue_secs
    BackchannelTimingConfig().min_between_cues_secs
        == TurnTakingConfig().min_between_cues_secs

The direct sibling of iter-179 (which pinned the *silence-floor* scalar shared
across three modules) and iter-180 (the *partition inequality* between the two
windows). This lap pins the *rate-limit* scalars shared between the two cue
paths' configs. Nothing tested it: ``test_backchannel_timing.py`` only asserts
``min_speaking_before_first_cue_secs == 15.0`` / ``min_between_cues_secs == 20.0``
against *hardcoded literals* (``test_config_defaults`` at L435-436), and
``test_turn_decider`` / ``test_turn_taking`` never read ``TurnTakingConfig``'s
copies for comparison. So retuning ``TurnTakingConfig.min_between_cues_secs``
down to 10.0 (a plausible "let the turn-end cue fire a touch more often" tweak)
would silently leave the mid-speech path still rate-limited at 20.0 — the two
paths would now backchannel at *different* cadences, the exact divergence the
"so the two paths feel consistent" docstring promises can't happen — and every
per-module test would stay green because none looks at the sibling. This module
makes that drift a red test, derived from the two configs themselves, not from
literals.

Both modules are stdlib-only at module scope but live under ``session/`` whose
``__init__`` eagerly imports pipecat-dependent modules (absent on the x86_64
Linux runner). So we load each by file path into a stub ``session`` namespace —
the same trick the sibling invariant tests use.
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


def _bc_warmup():
    """Mid-speech path: min user-speech before the first backchannel."""
    return BackchannelTimingConfig().min_speaking_before_first_cue_secs


def _bc_gap():
    """Mid-speech path: min gap between backchannels."""
    return BackchannelTimingConfig().min_between_cues_secs


def _tt_warmup():
    """Turn-end path: min user-speech before the first cue."""
    return TurnTakingConfig().min_speaking_before_first_cue_secs


def _tt_gap():
    """Turn-end path: min gap between cues."""
    return TurnTakingConfig().min_between_cues_secs


# --- the rate-limit mirror invariant (the gap this lap closes) --------------


def test_warmup_floor_is_shared_across_both_cue_paths():
    """``min_speaking_before_first_cue_secs`` is one shared scalar across the
    mid-speech (``BackchannelTimingConfig``) and turn-end (``TurnTakingConfig``)
    paths. Derived from both configs, not literals, so retuning one warm-up
    floor without the other turns red."""
    assert _bc_warmup() == _tt_warmup()


def test_between_cues_gap_is_shared_across_both_cue_paths():
    """``min_between_cues_secs`` is one shared scalar across both cue paths, so
    the agent backchannels at one cadence regardless of which path emits.
    Derived from both configs, not literals."""
    assert _bc_gap() == _tt_gap()


def test_both_rate_limits_mirror_in_one_assertion():
    """Headline: both rate-limit knobs match across the two configs at once, so
    the "the two paths feel consistent" docstring contract holds as a unit."""
    bc = BackchannelTimingConfig()
    tt = TurnTakingConfig()
    assert (
        bc.min_speaking_before_first_cue_secs,
        bc.min_between_cues_secs,
    ) == (
        tt.min_speaking_before_first_cue_secs,
        tt.min_between_cues_secs,
    )


def test_warmup_is_not_longer_than_between_cues_gap():
    """Sanity on the shared values' relative ordering: the warm-up floor (don't
    cue until the user has talked a while) is <= the between-cues gap, matching
    both configs' 15.0 <= 20.0. Keeps the pair well-formed if either is
    retuned."""
    assert _bc_warmup() <= _bc_gap()
    assert _tt_warmup() <= _tt_gap()


def test_current_rate_limits_are_fifteen_and_twenty_seconds():
    """Pins the present shared values (warm-up 15.0s, between-cues 20.0s) so an
    *intentional* retune is a deliberate edit to this test, not a silent change.
    Distinct from the equality tests: those guarantee the two paths move
    together; this records where the shared values sit today."""
    assert _bc_warmup() == 15.0
    assert _bc_gap() == 20.0
    assert _tt_warmup() == 15.0
    assert _tt_gap() == 20.0
