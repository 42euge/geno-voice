"""Tests for iter-033 — sentence splitter recognizes closing parens
and brackets between terminator and whitespace.

iter-022 added support for closing quotes (.") so US-style quoted
speech split correctly. iter-033 generalizes the same lookbehind
to closing parens (.), brackets (.] / .}) and any combination
because LLMs frequently produce parenthetical asides:

    He left (long ago.) Today he returned.
    Per spec [see ref.] We can continue.

Both should split into two sentences. Pre-iter-033 the splitter
saw `.)` and `.]` as just `.` followed by a non-quote char and
skipped the split.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_helpers import (  # noqa: E402
    _CLOSING_AFTER_TERMINATOR,
    _CLOSING_QUOTES,
    SENTENCE_END,
    split_complete_sentences,
)


class TestClosingParens:
    def test_simple_paren_close_after_period(self):
        result = split_complete_sentences(
            "He left (long ago.) Today returned."
        )
        assert result == (["He left (long ago.)"], "Today returned.")

    def test_paren_close_after_exclamation(self):
        result = split_complete_sentences(
            "Watch out (it's hot!) Sit down please."
        )
        assert result == (["Watch out (it's hot!)"], "Sit down please.")

    def test_paren_close_after_question(self):
        result = split_complete_sentences(
            "She wondered (was it true?) Then asked again."
        )
        assert result == (["She wondered (was it true?)"], "Then asked again.")

    def test_paren_at_very_end_no_split(self):
        # No whitespace after the close paren, no split.
        result = split_complete_sentences("He left (long ago.)")
        assert result == ([], "He left (long ago.)")

    def test_multiple_paren_sentences(self):
        result = split_complete_sentences(
            "First sentence. (An aside.) Second sentence."
        )
        # First period splits "First sentence." Aside is one piece,
        # then "Second sentence." is the trailing in-progress.
        assert result[0] == ["First sentence.", "(An aside.)"]
        assert result[1] == "Second sentence."


class TestClosingBrackets:
    def test_bracket_close_after_period(self):
        result = split_complete_sentences("Per spec [see ref.] We continue.")
        assert result == (["Per spec [see ref.]"], "We continue.")

    def test_curly_close_after_period(self):
        # Curly braces also count as closing chars.
        result = split_complete_sentences("Set { x = 1.} Move on.")
        assert result == (["Set { x = 1.}"], "Move on.")


class TestRegressionsFromIter022:
    """Make sure the iter-022 closing-quote behavior still works
    after generalizing the constant.
    """

    def test_straight_double_quote(self):
        result = split_complete_sentences('He said "hi." Then left.')
        assert result == (['He said "hi."'], "Then left.")

    def test_straight_single_quote(self):
        result = split_complete_sentences("She replied 'ok.' We left.")
        assert result == (["She replied 'ok.'"], "We left.")

    def test_smart_double_quote(self):
        result = split_complete_sentences('He said “hi.” Then left.')
        assert result == (['He said “hi.”'], "Then left.")

    def test_smart_single_quote(self):
        result = split_complete_sentences("She said 'ok.’ We left.")
        # Smart right single is U+2019.
        assert result == (["She said 'ok.’"], "We left.")


class TestPlainSentencesUnaffected:
    """Regression: non-quoted, non-parenthesized sentences still
    split exactly as before.
    """

    def test_simple_pair(self):
        assert split_complete_sentences("Hello. World.") == (
            ["Hello."],
            "World.",
        )

    def test_three_sentences(self):
        assert split_complete_sentences("One. Two! Three?") == (
            ["One.", "Two!"],
            "Three?",
        )

    def test_no_terminator(self):
        assert split_complete_sentences("no terminator yet") == (
            [],
            "no terminator yet",
        )


class TestAbbreviationInsideParens:
    """The abbreviation walk-back should still see the period inside
    a closing paren as a non-terminating abbreviation when applicable.
    """

    def test_etc_inside_paren_no_split(self):
        # "etc." is non-terminating. Followed by ")", it should still
        # NOT split — the user is saying "(etc.) and more" all in one.
        result = split_complete_sentences("See note (etc.) and more.")
        assert result == ([], "See note (etc.) and more.")

    def test_real_sentence_in_paren_does_split(self):
        # "Smith." is a real sentence-ender (Mr. Smith is one
        # sentence ending with the period after "Smith"). Splits.
        result = split_complete_sentences("Done (Mr. Smith.) End.")
        assert result == (["Done (Mr. Smith.)"], "End.")

    def test_ie_inside_paren_no_split(self):
        # "i.e." inside a paren followed by ")", followed by space.
        result = split_complete_sentences("Asked (i.e.) responded.")
        assert result == ([], "Asked (i.e.) responded.")


class TestConstantAndRegex:
    def test_closing_after_terminator_includes_parens_and_brackets(self):
        for ch in ")]}":
            assert ch in _CLOSING_AFTER_TERMINATOR

    def test_closing_after_terminator_includes_quotes(self):
        for ch in '"\'”’':
            assert ch in _CLOSING_AFTER_TERMINATOR

    def test_backwards_compat_alias_kept(self):
        # iter-022 callers used `_CLOSING_QUOTES`. Alias preserved.
        assert _CLOSING_QUOTES == _CLOSING_AFTER_TERMINATOR

    def test_regex_matches_all_closing_variants(self):
        for ch in _CLOSING_AFTER_TERMINATOR:
            buffer = f"end.{ch} next"
            assert SENTENCE_END.search(buffer) is not None, (
                f"SENTENCE_END regex did not match terminator+{ch!r}"
            )
