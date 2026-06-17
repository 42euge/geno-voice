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


def test_rotation_is_non_empty_and_known_cues():
    """The rotation lists real cue-bank keys (see generate_cues.py CUE_BANK)."""
    assert CUE_ROTATION  # non-empty
    known = {"mhmm", "i_see", "right", "go_on", "tell_me_more", "okay", "hmm"}
    assert set(CUE_ROTATION) <= known


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
