"""Tests for iter-164 — mid-session long-silence flush decision (backlog #9).

``session/silence_flush.py`` decides whether a held mid-thought fragment should
be ``FLUSH``ed to the engine after a beat of inter-turn silence, or kept on
``HOLD``. It is the still-deferred half of backlog #9: the ``UtteranceBuffer``
only releases a held pending when the *next utterance* arrives, so a user who
trails off mid-thought and then says nothing leaves the fragment held until a
new thought displaces it (iter-162) or shutdown flushes it (iter-160). This
seam is the pure decision a future ``run_session`` inter-turn clock read will
consume to flush that fragment *as its own turn* once the merge window elapses.

``decide_silence_flush(*, held_text, silence_secs, config, max_gap_secs)``
returns ``FLUSH`` (window elapsed, no continuation came) or ``HOLD`` (still
within the window / nothing held / merging off).

The whole point is the **half-duplex invariant**: with a default
``FullDuplexConfig()`` (utterance merging off) the decision is ``HOLD`` for
every input — and in that mode the buffer never holds anything anyway, so there
is nothing to flush. Only with ``utterance_merging`` explicitly on can a held
fragment exist, and only then can a long silence flush it.

``silence_flush`` does ``from session.full_duplex import ...`` and
``from session.utterance_merging import ...`` at module scope, but
``session/__init__.py`` eagerly imports pipecat-dependent modules (absent on the
x86_64 Linux runner). So we stand up a stub ``session`` namespace package and
load ``text_eou`` / ``full_duplex`` / ``utterance_merging`` / ``silence_flush``
into it by file path — the same trick test_utterance_merging.py uses.
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
if "session.text_eou" not in sys.modules:
    _load_by_path("session.text_eou", "text_eou.py", package="session")
if "session.full_duplex" not in sys.modules:
    _load_by_path("session.full_duplex", "full_duplex.py", package="session")
if "session.utterance_merging" not in sys.modules:
    _load_by_path(
        "session.utterance_merging", "utterance_merging.py", package="session"
    )

_sf = _load_by_path(
    "session.silence_flush", "silence_flush.py", package="session"
)

SilenceFlushAction = _sf.SilenceFlushAction
decide_silence_flush = _sf.decide_silence_flush
should_flush_held_utterance = _sf.should_flush_held_utterance

FullDuplexConfig = sys.modules["session.full_duplex"].FullDuplexConfig
DEFAULT_MAX_GAP_SECS = sys.modules["session.utterance_merging"].DEFAULT_MAX_GAP_SECS


# An organic-mode config (master on) and a sub-flag-only variant, mirroring the
# sibling seams' fixtures.
def _organic() -> "FullDuplexConfig":
    return FullDuplexConfig(enabled=True)


def _organic_merging_on() -> "FullDuplexConfig":
    # Master off, but the utterance-merging sub-flag forced on.
    return FullDuplexConfig(enabled=False, utterance_merging=True)


_HELD = "I was thinking about the"


# ---- the half-duplex invariant (default config) ------------------------------


class TestHalfDuplexInvariant:
    """A default FullDuplexConfig() ⇒ HOLD for every input, no exceptions."""

    def test_default_config_long_silence_held_is_hold(self):
        # The textbook flush candidate — held fragment, long silence — but the
        # gate is off by default.
        action = decide_silence_flush(held_text=_HELD, silence_secs=5.0)
        assert action is SilenceFlushAction.HOLD

    def test_default_config_never_flushes_grid(self):
        cases = [
            (_HELD, 2.01),
            ("and then", 10.0),
            ("because", DEFAULT_MAX_GAP_SECS + 0.5),
            ("um what was", 100.0),
        ]
        for held, silence in cases:
            assert (
                decide_silence_flush(held_text=held, silence_secs=silence)
                is SilenceFlushAction.HOLD
            ), (held, silence)

    def test_should_flush_false_by_default(self):
        assert (
            should_flush_held_utterance(held_text=_HELD, silence_secs=5.0) is False
        )

    def test_explicit_default_config_same_as_none(self):
        cfg = FullDuplexConfig()
        assert decide_silence_flush(
            held_text=_HELD, silence_secs=5.0, config=cfg
        ) is SilenceFlushAction.HOLD


# ---- organic mode: the window gate -------------------------------------------


class TestOrganicWindowGate:
    """With merging on, FLUSH iff silence strictly exceeds the merge window."""

    def test_long_silence_flushes(self):
        action = decide_silence_flush(
            held_text=_HELD, silence_secs=5.0, config=_organic()
        )
        assert action is SilenceFlushAction.FLUSH

    def test_silence_just_over_window_flushes(self):
        action = decide_silence_flush(
            held_text=_HELD,
            silence_secs=DEFAULT_MAX_GAP_SECS + 0.01,
            config=_organic(),
        )
        assert action is SilenceFlushAction.FLUSH

    def test_silence_at_window_boundary_holds(self):
        # Exactly max_gap_secs is still "within the window" — a continuation
        # arriving now would MERGE (decide_utterance_continuation's rule 3 is
        # gap <= max_gap_secs), so we must not have flushed.
        action = decide_silence_flush(
            held_text=_HELD,
            silence_secs=DEFAULT_MAX_GAP_SECS,
            config=_organic(),
        )
        assert action is SilenceFlushAction.HOLD

    def test_silence_just_under_window_holds(self):
        action = decide_silence_flush(
            held_text=_HELD,
            silence_secs=DEFAULT_MAX_GAP_SECS - 0.01,
            config=_organic(),
        )
        assert action is SilenceFlushAction.HOLD

    def test_zero_silence_holds(self):
        action = decide_silence_flush(
            held_text=_HELD, silence_secs=0.0, config=_organic()
        )
        assert action is SilenceFlushAction.HOLD

    def test_boundary_matches_merge_window_via_custom_max_gap(self):
        # The flush deadline must track max_gap_secs, not a hardcoded 2.0.
        action_hold = decide_silence_flush(
            held_text=_HELD, silence_secs=3.0, config=_organic(), max_gap_secs=4.0
        )
        assert action_hold is SilenceFlushAction.HOLD
        action_flush = decide_silence_flush(
            held_text=_HELD, silence_secs=4.5, config=_organic(), max_gap_secs=4.0
        )
        assert action_flush is SilenceFlushAction.FLUSH


# ---- organic mode: nothing-held gate -----------------------------------------


class TestNothingHeld:
    """No held text ⇒ HOLD even under a long silence in organic mode."""

    def test_empty_held_holds(self):
        action = decide_silence_flush(
            held_text="", silence_secs=99.0, config=_organic()
        )
        assert action is SilenceFlushAction.HOLD

    def test_whitespace_held_holds(self):
        action = decide_silence_flush(
            held_text="   \t  ", silence_secs=99.0, config=_organic()
        )
        assert action is SilenceFlushAction.HOLD

    def test_none_held_holds(self):
        # Defensive: a None pending (the buffer's idle state) must not crash.
        action = decide_silence_flush(
            held_text=None, silence_secs=99.0, config=_organic()
        )
        assert action is SilenceFlushAction.HOLD


# ---- sub-flag resolution -----------------------------------------------------


class TestSubFlagResolution:
    """The utterance_merging sub-flag forced on (master off) still flushes."""

    def test_subflag_on_master_off_flushes(self):
        action = decide_silence_flush(
            held_text=_HELD, silence_secs=5.0, config=_organic_merging_on()
        )
        assert action is SilenceFlushAction.FLUSH

    def test_subflag_off_master_on_via_explicit_false(self):
        # Master on but the merging sub-flag explicitly disabled ⇒ HOLD.
        cfg = FullDuplexConfig(enabled=True, utterance_merging=False)
        action = decide_silence_flush(
            held_text=_HELD, silence_secs=5.0, config=cfg
        )
        assert action is SilenceFlushAction.HOLD


# ---- should_flush_held_utterance convenience boolean -------------------------


class TestShouldFlushBoolean:
    """The boolean mirror tracks the enum decision exactly."""

    def test_true_on_flush(self):
        assert (
            should_flush_held_utterance(
                held_text=_HELD, silence_secs=5.0, config=_organic()
            )
            is True
        )

    def test_false_on_hold_within_window(self):
        assert (
            should_flush_held_utterance(
                held_text=_HELD, silence_secs=1.0, config=_organic()
            )
            is False
        )

    def test_false_on_nothing_held(self):
        assert (
            should_flush_held_utterance(
                held_text="", silence_secs=99.0, config=_organic()
            )
            is False
        )

    def test_custom_max_gap_tracks(self):
        assert (
            should_flush_held_utterance(
                held_text=_HELD,
                silence_secs=3.0,
                config=_organic(),
                max_gap_secs=4.0,
            )
            is False
        )
        assert (
            should_flush_held_utterance(
                held_text=_HELD,
                silence_secs=5.0,
                config=_organic(),
                max_gap_secs=4.0,
            )
            is True
        )


# ---- consistency with the merge window ---------------------------------------


def test_flush_boundary_is_exactly_the_merge_window():
    """The flush deadline and the merge window are the same scalar.

    A continuation at exactly ``max_gap_secs`` still MERGEs
    (``decide_utterance_continuation``'s rule 3 is ``gap <= max_gap_secs``), so
    at that boundary we must still be HOLDing. Only strictly beyond it is the
    window provably closed. This pins the two seams to agree.
    """
    decide_merge = sys.modules[
        "session.utterance_merging"
    ].decide_utterance_continuation
    UtteranceAction = sys.modules["session.utterance_merging"].UtteranceAction

    boundary = DEFAULT_MAX_GAP_SECS
    # At the boundary: merge still possible, flush still holds.
    assert (
        decide_merge(_HELD, "deadline", boundary, config=_organic())
        is UtteranceAction.MERGE
    )
    assert (
        decide_silence_flush(
            held_text=_HELD, silence_secs=boundary, config=_organic()
        )
        is SilenceFlushAction.HOLD
    )
    # Just beyond: merge no longer fires, flush takes over.
    beyond = boundary + 0.01
    assert (
        decide_merge(_HELD, "deadline", beyond, config=_organic())
        is UtteranceAction.NEW
    )
    assert (
        decide_silence_flush(
            held_text=_HELD, silence_secs=beyond, config=_organic()
        )
        is SilenceFlushAction.FLUSH
    )
