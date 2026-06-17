"""Tests for iter-171 — shared backchannel-cue rotation (backlog #7).

``session/cue_rotation.py`` is the single source of truth for the backchannel
cue rotation, shared by the silence-driven ``TurnTakingEngine`` path and the
mid-speech ``BackchannelMonitor`` (iter-170). Before this lap ``CUE_ROTATION``
lived inside ``turn_taking.py`` and the monitor returned no cue at all; now both
index the same list so they can't drift apart.

Loaded by file path to dodge ``session/__init__``'s eager pipecat import (absent
on the x86_64 Linux runner), the same trick the sibling organic-track tests use.
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
if "session.cue_rotation" not in sys.modules:
    _load_by_path("session.cue_rotation", "cue_rotation.py", package="session")

_cr = sys.modules["session.cue_rotation"]
CUE_ROTATION = _cr.CUE_ROTATION
cue_for_index = _cr.cue_for_index


def _load_cue_bank():
    """Load ``generate_cues.py``'s ``CUE_BANK`` — the *real* source of truth for
    which cue types get audio synthesized into ``session/cues/<type>/``.

    ``generate_cues.py`` imports ``requests`` at module scope (it POSTs to the
    voice server), so skip if it's absent rather than failing on an env quirk;
    the cue-rotation/bank drift this test guards is unrelated to whether the
    HTTP client happens to be installed on the runner.
    """
    import pytest

    pytest.importorskip("requests")
    mod = _load_by_path("session._generate_cues_under_test", "generate_cues.py")
    return mod.CUE_BANK


def test_rotation_cues_all_exist_in_the_synthesis_bank():
    """Every rotation cue must be a key in ``generate_cues.py``'s ``CUE_BANK``.

    This is the invariant that keeps the live cue path from silently 404ing: the
    rotation only plays cue types that ``generate_cues.py`` actually synthesizes
    into ``session/cues/<type>/`` and that ``server.py``'s ``/cue/{cue_type}``
    can therefore serve. The earlier ``<= known`` check mirrored the bank's keys
    by hand — a hardcoded set that could itself drift; this derives the allowed
    set straight from the bank so a cue renamed/removed from the bank fails here
    instead of at runtime.
    """
    bank = _load_cue_bank()
    missing = sorted(set(CUE_ROTATION) - set(bank))
    assert not missing, (
        f"CUE_ROTATION references cue types absent from generate_cues.py's "
        f"CUE_BANK (no audio would be synthesized → /cue/ 404s): {missing}"
    )


def test_rotation_cues_each_have_at_least_one_variant():
    """A bank key with an empty variant list synthesizes nothing — ``/cue/``
    would then 404 with 'no cues available'. Guard that every rotation cue has
    at least one ``(text, speed)`` variant to render."""
    bank = _load_cue_bank()
    empty = sorted(c for c in set(CUE_ROTATION) if not bank.get(c))
    assert not empty, (
        f"CUE_ROTATION cues with no synthesis variants in CUE_BANK: {empty}"
    )


def test_rotation_is_non_empty():
    """The rotation must list at least one cue (else nothing ever plays)."""
    assert CUE_ROTATION


def test_cue_for_index_walks_the_rotation_in_order():
    for i, expected in enumerate(CUE_ROTATION):
        assert cue_for_index(i) == expected


def test_cue_for_index_wraps_modulo_length():
    n = len(CUE_ROTATION)
    assert cue_for_index(n) == CUE_ROTATION[0]
    assert cue_for_index(n + 1) == CUE_ROTATION[1]
    assert cue_for_index(2 * n) == CUE_ROTATION[0]


def test_cue_for_index_handles_negative_indices():
    """Python modulo keeps negatives in range — never raises, always valid."""
    assert cue_for_index(-1) == CUE_ROTATION[-1]
    assert cue_for_index(-len(CUE_ROTATION)) == CUE_ROTATION[0]


def test_mhmm_is_the_most_frequent_cue():
    """The neutral 'mhmm' recurs more than the pointed cues (design note)."""
    counts = {c: CUE_ROTATION.count(c) for c in set(CUE_ROTATION)}
    assert counts["mhmm"] == max(counts.values())
    assert counts["mhmm"] >= 2
