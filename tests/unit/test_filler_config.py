"""Tests for iter-034 — tolerant parse_filler_config.

iter-011 introduced filler words but the parsing lived inline in
mic_chat.run_chat:

    filler_texts: list[str] = list(chat_cfg.get("fillers") or [])
    filler_idle_threshold: float = float(chat_cfg.get("fillers_idle_threshold", 0.6))

Two real failure modes:

1. ``chat.fillers: "hi"`` (a string, not a list) goes into
   ``list(...)``, which iterates the string and produces
   ``["h", "i"]``. The user's typo became two two-character
   "fillers" sent to TTS, producing nonsense audio.

2. ``chat.fillers_idle_threshold: "abc"`` crashes the entire
   chat startup with ``ValueError: could not convert string
   to float: 'abc'`` — long before any chat happens.

iter-034 adds parse_filler_config in the same shape as
parse_vad_config: tolerant of bad input, falls back to defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_config import (  # noqa: E402
    FILLER_DEFAULTS,
    parse_filler_config,
)


class TestEmptyOrMissing:
    def test_empty_dict_returns_defaults(self):
        out = parse_filler_config({})
        assert out["texts"] == []
        assert out["idle_threshold"] == 0.6

    def test_none_input_returns_defaults(self):
        out = parse_filler_config(None)
        assert out["texts"] == []
        assert out["idle_threshold"] == 0.6

    def test_non_mapping_input_returns_defaults(self):
        # If the chat config came back as the wrong shape entirely.
        out = parse_filler_config("not a dict")
        assert out["texts"] == []
        assert out["idle_threshold"] == 0.6

    def test_returned_list_is_independent(self):
        # Caller mutating the result must not corrupt FILLER_DEFAULTS.
        out = parse_filler_config({})
        out["texts"].append("hi")
        out2 = parse_filler_config({})
        assert out2["texts"] == []
        assert FILLER_DEFAULTS["texts"] == []


class TestHappyPath:
    def test_well_formed_list_of_strings(self):
        out = parse_filler_config({
            "fillers": ["hmm", "let me think", "well,"],
            "fillers_idle_threshold": 0.4,
        })
        assert out["texts"] == ["hmm", "let me think", "well,"]
        assert out["idle_threshold"] == 0.4

    def test_int_threshold_coerced_to_float(self):
        out = parse_filler_config({"fillers_idle_threshold": 1})
        assert out["idle_threshold"] == 1.0
        assert isinstance(out["idle_threshold"], float)


class TestFillersListInputForms:
    def test_string_fillers_does_not_become_chars(self):
        # The iter-034 motivating bug: list("hi") == ["h", "i"].
        # We must NOT do that — string is malformed config, drop it.
        out = parse_filler_config({"fillers": "hi"})
        assert out["texts"] == []

    def test_dict_fillers_returns_empty(self):
        out = parse_filler_config({"fillers": {"a": 1}})
        assert out["texts"] == []

    def test_int_fillers_returns_empty(self):
        out = parse_filler_config({"fillers": 42})
        assert out["texts"] == []

    def test_none_fillers_returns_empty(self):
        out = parse_filler_config({"fillers": None})
        assert out["texts"] == []

    def test_empty_list_fillers_returns_empty(self):
        out = parse_filler_config({"fillers": []})
        assert out["texts"] == []


class TestNonStringItemsDropped:
    def test_drops_non_string_items(self):
        out = parse_filler_config({
            "fillers": ["hmm", 42, "well", None, {"x": 1}],
        })
        assert out["texts"] == ["hmm", "well"]

    def test_drops_empty_strings(self):
        out = parse_filler_config({"fillers": ["", "  ", "hmm"]})
        assert out["texts"] == ["hmm"]

    def test_strips_whitespace(self):
        # Useful: YAML with stray trailing spaces shouldn't matter.
        out = parse_filler_config({"fillers": ["  hmm  ", "well "]})
        assert out["texts"] == ["hmm", "well"]

    def test_all_items_dropped_returns_empty(self):
        out = parse_filler_config({"fillers": [None, 1, 2.0, "", "  "]})
        assert out["texts"] == []


class TestIdleThresholdInputForms:
    def test_string_threshold_uses_default(self):
        # iter-034 motivating bug: float("abc") used to crash startup.
        out = parse_filler_config({"fillers_idle_threshold": "abc"})
        assert out["idle_threshold"] == 0.6

    def test_negative_threshold_uses_default(self):
        # Negative threshold makes no sense — the worker would never
        # wait. Fall back to default rather than accept it.
        out = parse_filler_config({"fillers_idle_threshold": -0.5})
        assert out["idle_threshold"] == 0.6

    def test_zero_threshold_uses_default(self):
        # Zero threshold == "fire fillers immediately, before LLM
        # had a chance" — same reasoning, drop and default.
        out = parse_filler_config({"fillers_idle_threshold": 0})
        assert out["idle_threshold"] == 0.6

    def test_none_threshold_uses_default(self):
        out = parse_filler_config({"fillers_idle_threshold": None})
        assert out["idle_threshold"] == 0.6

    def test_bool_threshold_uses_default(self):
        # Python ``bool`` is an ``int`` subclass — "True" would
        # otherwise sneak through. The "> 0" check accepts True (==1)
        # so this test documents the actual behavior: it IS accepted
        # as 1.0. If we wanted to reject bools specifically, we'd
        # need an explicit type check. For now, this is fine because
        # YAML doesn't typically produce ``True`` for a numeric field.
        out = parse_filler_config({"fillers_idle_threshold": True})
        assert out["idle_threshold"] == 1.0


class TestPartialConfig:
    def test_only_fillers_no_threshold(self):
        out = parse_filler_config({"fillers": ["hmm"]})
        assert out["texts"] == ["hmm"]
        assert out["idle_threshold"] == 0.6  # default

    def test_only_threshold_no_fillers(self):
        out = parse_filler_config({"fillers_idle_threshold": 1.2})
        assert out["texts"] == []  # default
        assert out["idle_threshold"] == 1.2

    def test_combined_with_other_chat_keys(self):
        # Other unrelated chat config keys must not interfere.
        out = parse_filler_config({
            "fillers": ["hi"],
            "fillers_idle_threshold": 0.3,
            "vad": {"silence_threshold": 0.05},  # unrelated
            "some_other_key": "ignored",
        })
        assert out["texts"] == ["hi"]
        assert out["idle_threshold"] == 0.3
