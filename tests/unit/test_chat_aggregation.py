"""Tests for iter-159 — ``resolve_turn``, the pure single-turn collapse of an
``AggregatedResult`` from ``UtteranceAggregator``.

``ChatLoop.run_one_turn`` is a *synchronous, single-turn* function: it records
one utterance and responds to it. The aggregator (iter-158), by contrast, may
**hold** an utterance (emit zero turns, waiting for a mid-thought continuation)
or release **several** turns at once. ``resolve_turn`` is the pure seam that
bridges that impedance mismatch: it folds the aggregator's variable-length
``turns`` list into the one decision a single-turn loop needs —

  - respond to nothing this turn (everything is being held), or
  - respond to this (possibly merged) text, carrying the false-endpoint flag.

Kept pure and dependency-free (duck-typed over ``.turns`` / ``.held``) so it
loads with no ``session`` import — the eager-pipecat trap the sibling seams
dodge — and is exhaustively testable in isolation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_aggregation import ResolvedTurn, resolve_turn  # noqa: E402


# ---- Lightweight doubles mirroring the aggregator's return shape ------------


@dataclass(frozen=True)
class _Turn:
    text: str
    false_endpoint: bool = False
    merge_capped: bool = False


@dataclass(frozen=True)
class _Result:
    turns: list
    held: str | None = None


# ---- No-turn (held) path ----------------------------------------------------


class TestHeld:
    def test_no_turns_does_not_respond(self):
        r = resolve_turn(_Result(turns=[], held="I think that"))
        assert r.respond is False
        assert r.text == ""
        assert r.false_endpoint is False
        assert r.held == "I think that"

    def test_no_turns_no_held(self):
        r = resolve_turn(_Result(turns=[], held=None))
        assert r.respond is False
        assert r.text == ""
        assert r.held is None


# ---- Single-turn path (the common case) -------------------------------------


class TestSingleTurn:
    def test_one_turn_responds_with_its_text(self):
        r = resolve_turn(_Result(turns=[_Turn("Hello there.")], held=None))
        assert r.respond is True
        assert r.text == "Hello there."
        assert r.false_endpoint is False

    def test_one_merged_turn_carries_false_endpoint(self):
        r = resolve_turn(
            _Result(turns=[_Turn("I think that it is fine.", True)])
        )
        assert r.respond is True
        assert r.text == "I think that it is fine."
        assert r.false_endpoint is True

    def test_held_is_passed_through_alongside_a_released_turn(self):
        # A released turn AND a freshly-held pending can coexist (the
        # aggregator released a complete prior and is now holding a new
        # fragment). resolve_turn reports both.
        r = resolve_turn(
            _Result(turns=[_Turn("Done.")], held="and then")
        )
        assert r.respond is True
        assert r.text == "Done."
        assert r.held == "and then"


# ---- Multi-turn release (the corner) ----------------------------------------
#
# iter-162: a multi-turn release is NEVER a continuation to join. The buffer
# only emits >1 turn when a measured silence forced a NEW boundary (the held
# pending released as its own turn) *and then* a fresh utterance was emitted.
# Those are semantically distinct turns — the earlier ones are abandoned
# mid-thought fragments, the LAST is the new thing to answer now. The pre-
# iter-162 behavior space-glued them into one garbled LLM input; the fix
# responds to the last turn only and surfaces the earlier ones as ``displaced``.


class TestMultiTurn:
    def test_two_turns_respond_to_last_not_joined(self):
        # The headline fix: "I was thinking about the" was abandoned (a long
        # silence proved it wasn't a false endpoint), then "What time is it?"
        # is the genuinely-new turn. We must answer the new one, NOT the glued
        # "I was thinking about the What time is it?".
        r = resolve_turn(
            _Result(turns=[_Turn("I was thinking about the"),
                           _Turn("What time is it?")])
        )
        assert r.respond is True
        assert r.text == "What time is it?"
        assert r.displaced == ("I was thinking about the",)

    def test_single_turn_release_has_no_displaced(self):
        r = resolve_turn(_Result(turns=[_Turn("the sky is blue.")]))
        assert r.respond is True
        assert r.text == "the sky is blue."
        assert r.displaced == ()

    def test_false_endpoint_is_the_responded_turns_own_flag(self):
        # iter-162: false_endpoint reflects the RESPONDED (last) turn, not an
        # OR across the abandoned fragments. An abandoned merged fragment must
        # not falsely stamp a fresh, non-merged response.
        r = resolve_turn(
            _Result(turns=[_Turn("a", True), _Turn("b", False)])
        )
        assert r.respond is True
        assert r.text == "b"
        assert r.false_endpoint is False
        assert r.displaced == ("a",)

    def test_responded_turn_keeps_its_own_false_endpoint(self):
        r = resolve_turn(
            _Result(turns=[_Turn("a", False), _Turn("b", True)])
        )
        assert r.text == "b"
        assert r.false_endpoint is True
        assert r.displaced == ("a",)

    def test_three_turns_displace_all_but_last(self):
        r = resolve_turn(
            _Result(turns=[_Turn("one"), _Turn("two"), _Turn("three.")])
        )
        assert r.text == "three."
        assert r.displaced == ("one", "two")

    def test_displaced_strips_and_drops_empty_turns(self):
        r = resolve_turn(
            _Result(turns=[_Turn("  one  "), _Turn(""), _Turn(" two ")])
        )
        # Blank middle turn dropped; last non-blank answered, rest displaced.
        assert r.text == "two"
        assert r.displaced == ("one",)

    def test_all_empty_turns_collapse_to_no_response(self):
        # Defensive: a release of only blank turns has nothing to say.
        r = resolve_turn(_Result(turns=[_Turn("   "), _Turn("")]))
        assert r.respond is False
        assert r.text == ""
        assert r.displaced == ()


# ---- iter-163: merge_capped flows to the responded turn ---------------------
#
# A capped turn is a force-emit of still-unfinished text by the iter-157
# backstop, not a clean repair. resolve_turn carries the RESPONDED turn's own
# merge_capped flag (mirroring the false_endpoint rule), so the loop can stamp
# TurnMetrics.merge_capped and the operator sees the cap fire.


class TestMergeCapped:
    def test_single_capped_turn_carries_flag(self):
        r = resolve_turn(
            _Result(turns=[_Turn("I was thinking about the", True,
                                  merge_capped=True)])
        )
        assert r.respond is True
        assert r.merge_capped is True
        assert r.false_endpoint is True       # always paired

    def test_natural_turn_is_not_capped(self):
        r = resolve_turn(_Result(turns=[_Turn("All done.", False)]))
        assert r.merge_capped is False

    def test_merge_capped_is_responded_turns_own_flag(self):
        # A capped DISPLACED fragment must not stamp a fresh, uncapped
        # response — only the last (responded) turn's flag counts.
        r = resolve_turn(
            _Result(turns=[_Turn("a", True, merge_capped=True),
                           _Turn("b", False, merge_capped=False)])
        )
        assert r.text == "b"
        assert r.merge_capped is False
        assert r.displaced == ("a",)

    def test_responded_turn_keeps_its_own_capped_flag(self):
        r = resolve_turn(
            _Result(turns=[_Turn("a", False, merge_capped=False),
                           _Turn("b", True, merge_capped=True)])
        )
        assert r.text == "b"
        assert r.merge_capped is True
        assert r.displaced == ("a",)

    def test_missing_merge_capped_attr_defaults_false(self):
        # Back-compat: a pre-iter-163 turn double without the attribute
        # resolves to merge_capped=False (read via getattr).
        @dataclass(frozen=True)
        class _OldTurn:
            text: str
            false_endpoint: bool = False

        r = resolve_turn(_Result(turns=[_OldTurn("hi.", True)]))
        assert r.merge_capped is False
        assert r.false_endpoint is True


# ---- ResolvedTurn contract --------------------------------------------------


class TestResolvedTurnContract:
    def test_is_frozen(self):
        r = ResolvedTurn(respond=True, text="x")
        try:
            r.respond = False  # type: ignore[misc]
        except Exception as e:
            assert "frozen" in str(type(e)).lower() or "attribute" in str(e).lower()
        else:
            raise AssertionError("ResolvedTurn should be frozen")

    def test_defaults(self):
        r = ResolvedTurn(respond=False, text="")
        assert r.false_endpoint is False
        assert r.held is None
        # iter-162: displaced defaults to the empty tuple (single-turn case).
        assert r.displaced == ()
        # iter-163: merge_capped defaults False (natural release).
        assert r.merge_capped is False
