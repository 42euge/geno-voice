"""Tests for iter-155 — utterance buffer-merge decision (backlog #4, 2nd half).

``session/utterance_merging.py`` composes two earlier seams — the text EOU
completeness scorer (#4, iter-150) and the full-duplex gate (#3, iter-151) —
into a pure ``decide_utterance_continuation(prev_text, next_text, gap_secs, *,
config)`` that returns ``MERGE`` (the prior endpoint was a false positive — the
user paused mid-thought) or ``NEW`` (a genuine new turn).

The whole point is the **half-duplex invariant**: with a default
``FullDuplexConfig()`` the decision is ``NEW`` for every input — byte-for-byte
today's "each endpoint is its own turn" behavior. Only with utterance merging
explicitly on does an unfinished-looking prior + a quick continuation yield
``MERGE``, and only in that unfinished-AND-quick corner.

``utterance_merging`` does ``from session.text_eou import ...`` and
``from session.full_duplex import ...`` at module scope, but
``session/__init__.py`` eagerly imports pipecat-dependent modules (absent on
the x86_64 Linux runner). So we stand up a stub ``session`` namespace package
and load ``text_eou`` / ``full_duplex`` / ``utterance_merging`` into it by file
path — the same trick test_barge_decision.py uses for its sibling import.
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

_um = _load_by_path(
    "session.utterance_merging", "utterance_merging.py", package="session"
)
_fd = sys.modules["session.full_duplex"]
_eou = sys.modules["session.text_eou"]

UtteranceAction = _um.UtteranceAction
decide_utterance_continuation = _um.decide_utterance_continuation
should_merge_utterance = _um.should_merge_utterance
DEFAULT_MAX_GAP_SECS = _um.DEFAULT_MAX_GAP_SECS
DEFAULT_INCOMPLETE_CEILING = _um.DEFAULT_INCOMPLETE_CEILING
FullDuplexConfig = _fd.FullDuplexConfig
TextEOUConfig = _eou.TextEOUConfig


# A config with utterance merging on (master enabled). Half-duplex is the
# default-constructed config used in the invariant tests below.
ORGANIC = FullDuplexConfig(enabled=True)


# ---- the half-duplex invariant (default config) ------------------------------


class TestHalfDuplexInvariant:
    """A default FullDuplexConfig() ⇒ NEW for every input, no exceptions."""

    def test_default_config_unfinished_quick_is_new(self):
        # The textbook merge candidate — but the gate is off by default.
        action = decide_utterance_continuation(
            "I was thinking about the", "deadline", 0.5
        )
        assert action is UtteranceAction.NEW

    def test_default_config_never_merges_grid(self):
        cases = [
            ("and", "then we left", 0.1),
            ("because", "it rained", 1.9),
            ("I was going to the", "store", 0.0),
            ("um", "what was it", 0.3),
        ]
        for prev, nxt, gap in cases:
            assert (
                decide_utterance_continuation(prev, nxt, gap)
                is UtteranceAction.NEW
            ), (prev, nxt, gap)

    def test_should_merge_false_by_default(self):
        assert should_merge_utterance("I was thinking about the", "it", 0.2) is False

    def test_explicit_default_config_same_as_none(self):
        cfg = FullDuplexConfig()
        assert (
            decide_utterance_continuation(
                "I went to the", "store", 0.4, config=cfg
            )
            is UtteranceAction.NEW
        )


# ---- organic mode: the two gates ---------------------------------------------


class TestOrganicGates:
    def test_unfinished_and_quick_merges(self):
        # "…the" is a dangling article (completeness 0.3 <= 0.6) and the gap is
        # quick (0.5 <= 2.0): the canonical false endpoint to repair.
        action = decide_utterance_continuation(
            "I was thinking about the", "deadline", 0.5, config=ORGANIC
        )
        assert action is UtteranceAction.MERGE
        assert should_merge_utterance(
            "I was thinking about the", "deadline", 0.5, config=ORGANIC
        ) is True

    def test_conjunction_ending_merges(self):
        # "…and" is the strongest incompleteness signal.
        assert (
            decide_utterance_continuation("we packed up and", "left", 1.0, config=ORGANIC)
            is UtteranceAction.MERGE
        )

    def test_filler_ending_merges(self):
        assert (
            decide_utterance_continuation("it was kind of um", "weird", 0.8, config=ORGANIC)
            is UtteranceAction.MERGE
        )

    def test_complete_prior_is_new_even_when_quick(self):
        # A finished sentence + a quick follow-on is a NEW thought, not a merge.
        assert (
            decide_utterance_continuation("I'm done.", "Actually wait", 0.3, config=ORGANIC)
            is UtteranceAction.NEW
        )

    def test_unfinished_prior_after_long_gap_is_new(self):
        # Unfinished text but the gap exceeds the window ⇒ abandoned thought.
        assert (
            decide_utterance_continuation(
                "I was thinking about the", "deadline", 3.0, config=ORGANIC
            )
            is UtteranceAction.NEW
        )

    def test_complete_prior_after_long_gap_is_new(self):
        assert (
            decide_utterance_continuation("That's all.", "Next topic", 5.0, config=ORGANIC)
            is UtteranceAction.NEW
        )


# ---- boundary conditions -----------------------------------------------------


class TestBoundaries:
    def test_gap_at_max_is_inclusive_merge(self):
        # gap == max_gap_secs is still "quick" (<=).
        assert (
            decide_utterance_continuation(
                "going to the", "store", DEFAULT_MAX_GAP_SECS, config=ORGANIC
            )
            is UtteranceAction.MERGE
        )

    def test_gap_just_above_max_is_new(self):
        assert (
            decide_utterance_continuation(
                "going to the", "store", DEFAULT_MAX_GAP_SECS + 0.01, config=ORGANIC
            )
            is UtteranceAction.NEW
        )

    def test_zero_gap_merges(self):
        assert (
            decide_utterance_continuation("going to the", "store", 0.0, config=ORGANIC)
            is UtteranceAction.MERGE
        )

    def test_completeness_at_ceiling_is_inclusive_merge(self):
        # A comma ending scores exactly DEFAULT_INCOMPLETE_CEILING (0.6); the
        # gate is <=, so it merges.
        assert TextEOUConfig().comma_completeness == DEFAULT_INCOMPLETE_CEILING
        assert (
            decide_utterance_continuation("first,", "then second", 0.5, config=ORGANIC)
            is UtteranceAction.MERGE
        )

    def test_completeness_above_ceiling_is_new(self):
        # A plainly complete clause scores 1.0 > 0.6 ⇒ NEW.
        assert (
            decide_utterance_continuation("the cat sat", "on the mat", 0.5, config=ORGANIC)
            is UtteranceAction.NEW
        )


# ---- empty / blank continuation ----------------------------------------------


class TestEmptyContinuation:
    def test_empty_next_is_new(self):
        assert (
            decide_utterance_continuation("going to the", "", 0.5, config=ORGANIC)
            is UtteranceAction.NEW
        )

    def test_whitespace_next_is_new(self):
        assert (
            decide_utterance_continuation("going to the", "   ", 0.5, config=ORGANIC)
            is UtteranceAction.NEW
        )

    def test_empty_prior_is_new(self):
        # Empty prior scores 1.0 (no incompleteness evidence) ⇒ NEW.
        assert (
            decide_utterance_continuation("", "hello there", 0.3, config=ORGANIC)
            is UtteranceAction.NEW
        )


# ---- subflag resolution ------------------------------------------------------


class TestSubflagResolution:
    def test_subflag_on_with_master_off_enables_merge(self):
        cfg = FullDuplexConfig(enabled=False, utterance_merging=True)
        assert (
            decide_utterance_continuation("going to the", "store", 0.4, config=cfg)
            is UtteranceAction.MERGE
        )

    def test_subflag_off_with_master_on_disables_merge(self):
        cfg = FullDuplexConfig(enabled=True, utterance_merging=False)
        assert (
            decide_utterance_continuation("going to the", "store", 0.4, config=cfg)
            is UtteranceAction.NEW
        )


# ---- custom thresholds -------------------------------------------------------


class TestCustomThresholds:
    def test_custom_max_gap_widens_window(self):
        # Default 2.0s would reject a 2.5s gap; a custom 3.0s window accepts it.
        assert (
            decide_utterance_continuation(
                "going to the", "store", 2.5, config=ORGANIC
            )
            is UtteranceAction.NEW
        )
        assert (
            decide_utterance_continuation(
                "going to the", "store", 2.5, config=ORGANIC, max_gap_secs=3.0
            )
            is UtteranceAction.MERGE
        )

    def test_custom_incomplete_ceiling_tightens_text_gate(self):
        # A comma (0.6) merges by default; a stricter 0.5 ceiling rejects it.
        assert (
            decide_utterance_continuation("first,", "then", 0.5, config=ORGANIC)
            is UtteranceAction.MERGE
        )
        assert (
            decide_utterance_continuation(
                "first,", "then", 0.5, config=ORGANIC, incomplete_ceiling=0.5
            )
            is UtteranceAction.NEW
        )

    def test_custom_eou_config_changes_scoring(self):
        # Raise the dangling-completeness above the ceiling so "…the" no longer
        # qualifies as unfinished.
        lenient = TextEOUConfig(dangling_completeness=0.9)
        assert (
            decide_utterance_continuation(
                "going to the", "store", 0.4, config=ORGANIC, eou_config=lenient
            )
            is UtteranceAction.NEW
        )


# ---- purity / interface ------------------------------------------------------


class TestPurityAndInterface:
    def test_inputs_not_mutated(self):
        prev, nxt = "going to the", "store"
        decide_utterance_continuation(prev, nxt, 0.4, config=ORGANIC)
        assert prev == "going to the"
        assert nxt == "store"

    def test_config_not_mutated(self):
        before = (ORGANIC.enabled, ORGANIC.utterance_merging)
        decide_utterance_continuation("going to the", "store", 0.4, config=ORGANIC)
        assert (ORGANIC.enabled, ORGANIC.utterance_merging) == before

    def test_keyword_only_config(self):
        with pytest.raises(TypeError):
            decide_utterance_continuation("a", "b", 0.5, ORGANIC)  # type: ignore[misc]

    def test_action_values_distinct(self):
        assert UtteranceAction.MERGE.value != UtteranceAction.NEW.value

    def test_defaults_match_sibling_seams(self):
        # Window mirrors turn_decider's silence floor; ceiling mirrors
        # text_eou's complete threshold.
        assert DEFAULT_MAX_GAP_SECS == 2.0
        assert DEFAULT_INCOMPLETE_CEILING == TextEOUConfig().complete_threshold

    def test_should_merge_matches_decide(self):
        cases = [
            ("going to the", "store", 0.4),
            ("I'm done.", "next", 0.4),
            ("going to the", "store", 5.0),
            ("going to the", "", 0.4),
        ]
        for prev, nxt, gap in cases:
            expected = (
                decide_utterance_continuation(prev, nxt, gap, config=ORGANIC)
                is UtteranceAction.MERGE
            )
            assert (
                should_merge_utterance(prev, nxt, gap, config=ORGANIC) is expected
            ), (prev, nxt, gap)
