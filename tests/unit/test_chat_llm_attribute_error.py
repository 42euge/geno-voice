"""Tests for iter-032 — SSE parser must skip chunks where
``choices[0]`` is a non-dict (None / string / number).

Pre-iter-032 the except clause caught
``(JSONDecodeError, KeyError, IndexError, TypeError)``. A malformed
chunk like ``{"choices": [null]}`` decodes successfully (well-formed
JSON), the index lookup ``choices[0]`` succeeds (returns None), and
then ``.get("delta", {})`` raises ``AttributeError`` — which was
NOT caught. The result: every token after the bad chunk was lost
because the generator aborted.

We've seen this empirically with some local proxy setups that
inject keep-alive heartbeats as ``{"choices": [null]}``. The fix
adds ``AttributeError`` to the except clause; these tests prove
the bad chunk is now skipped and trailing tokens reach the consumer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_llm import parse_sse_token_stream  # noqa: E402


def _data_line(payload):
    """Helper: format a Python dict as an SSE 'data: ...' line."""
    return "data: " + json.dumps(payload)


def _good(content):
    return _data_line({"choices": [{"delta": {"content": content}}]})


class TestNonDictChoicesElement:
    """Each subtest: insert a malformed chunk between two good ones,
    confirm the malformed chunk is skipped and the trailing token
    still reaches the consumer.
    """

    def test_choices_element_is_none(self):
        lines = [
            _good("hello "),
            _data_line({"choices": [None]}),
            _good("world"),
            "data: [DONE]",
        ]
        tokens = list(parse_sse_token_stream(lines))
        assert tokens == ["hello ", "world"]

    def test_choices_element_is_string(self):
        lines = [
            _good("a "),
            _data_line({"choices": ["unexpected string"]}),
            _good("b"),
            "data: [DONE]",
        ]
        tokens = list(parse_sse_token_stream(lines))
        assert tokens == ["a ", "b"]

    def test_choices_element_is_int(self):
        lines = [
            _good("a "),
            _data_line({"choices": [42]}),
            _good("b"),
            "data: [DONE]",
        ]
        tokens = list(parse_sse_token_stream(lines))
        assert tokens == ["a ", "b"]

    def test_choices_element_is_list(self):
        # A list .get is also missing — same fault path.
        lines = [
            _good("a "),
            _data_line({"choices": [[1, 2, 3]]}),
            _good("b"),
            "data: [DONE]",
        ]
        tokens = list(parse_sse_token_stream(lines))
        assert tokens == ["a ", "b"]


class TestStreamFullyAbortsWithoutFix:
    """Regression-style guard: if someone removed AttributeError
    from the except clause, the entire stream would abort on the
    first bad chunk and only the pre-bad-chunk tokens would survive.
    These tests assert we get ALL good tokens, including ones AFTER
    the bad chunk.
    """

    def test_multiple_bad_chunks_interspersed(self):
        lines = [
            _good("one "),
            _data_line({"choices": [None]}),
            _good("two "),
            _data_line({"choices": ["bad"]}),
            _good("three "),
            _data_line({"choices": [42]}),
            _good("four"),
            "data: [DONE]",
        ]
        tokens = list(parse_sse_token_stream(lines))
        # All four good tokens make it through.
        assert tokens == ["one ", "two ", "three ", "four"]

    def test_bad_chunk_at_start_does_not_lose_good_tokens(self):
        lines = [
            _data_line({"choices": [None]}),
            _good("hello"),
            "data: [DONE]",
        ]
        tokens = list(parse_sse_token_stream(lines))
        assert tokens == ["hello"]

    def test_bad_chunk_at_end_does_not_break_done(self):
        lines = [
            _good("hello"),
            _data_line({"choices": [None]}),
            "data: [DONE]",
        ]
        tokens = list(parse_sse_token_stream(lines))
        assert tokens == ["hello"]


class TestExistingErrorPathsStillCaught:
    """Sanity: iter-032 added AttributeError to the except clause.
    Previously-handled errors (JSONDecodeError, KeyError, IndexError,
    TypeError) must still be silently skipped.
    """

    def test_json_decode_error(self):
        lines = [
            _good("a "),
            "data: this is not json{",
            _good("b"),
            "data: [DONE]",
        ]
        tokens = list(parse_sse_token_stream(lines))
        assert tokens == ["a ", "b"]

    def test_missing_choices_key(self):
        lines = [
            _good("a "),
            _data_line({"foo": "bar"}),  # no 'choices' → KeyError
            _good("b"),
            "data: [DONE]",
        ]
        tokens = list(parse_sse_token_stream(lines))
        assert tokens == ["a ", "b"]

    def test_empty_choices_array(self):
        lines = [
            _good("a "),
            _data_line({"choices": []}),  # IndexError on [0]
            _good("b"),
            "data: [DONE]",
        ]
        tokens = list(parse_sse_token_stream(lines))
        assert tokens == ["a ", "b"]

    def test_choices_is_not_indexable(self):
        # ``choices`` is an int — chunk["choices"][0] raises TypeError
        # ("'int' object is not subscriptable").
        lines = [
            _good("a "),
            _data_line({"choices": 5}),
            _good("b"),
            "data: [DONE]",
        ]
        tokens = list(parse_sse_token_stream(lines))
        assert tokens == ["a ", "b"]
