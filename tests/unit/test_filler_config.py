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
        # Python ``bool`` is an ``int`` subclass — without the guard
        # "True" would sneak through the "> 0" check and become 1.0.
        # iter-188 added an explicit ``not isinstance(_, bool)`` guard
        # (matching the iter-186/187 reject side), so a typo'd
        # ``True``/``False`` now falls back to the default instead.
        assert parse_filler_config(
            {"fillers_idle_threshold": True}
        )["idle_threshold"] == 0.6
        assert parse_filler_config(
            {"fillers_idle_threshold": False}
        )["idle_threshold"] == 0.6


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


class TestParseFillerConfigWarn:
    """iter-188: the optional ``warn`` callable surfaces dropped
    filler config so a typo isn't silently swallowed by the default.
    Mirrors the iter-187 ``parse_vad_config`` warn seam."""

    def test_no_warn_by_default(self):
        # Backwards compatible: omitting warn never raises and the
        # return value is unchanged from the silent behavior.
        out = parse_filler_config({"fillers": "hmm"})
        assert out["texts"] == []

    def test_valid_config_emits_no_warnings(self):
        warnings: list[str] = []
        parse_filler_config(
            {"fillers": ["hmm", "well"], "fillers_idle_threshold": 0.5},
            warn=warnings.append,
        )
        assert warnings == []

    def test_missing_keys_are_silent(self):
        warnings: list[str] = []
        parse_filler_config({}, warn=warnings.append)
        assert warnings == []

    def test_non_mapping_chat_is_silent(self):
        warnings: list[str] = []
        parse_filler_config(None, warn=warnings.append)
        assert warnings == []

    def test_fillers_not_a_list_warns(self):
        warnings: list[str] = []
        out = parse_filler_config({"fillers": "hmm"}, warn=warnings.append)
        assert out["texts"] == []
        assert len(warnings) == 1
        assert "fillers" in warnings[0]
        assert "list" in warnings[0]

    def test_non_string_item_warns(self):
        warnings: list[str] = []
        out = parse_filler_config(
            {"fillers": ["ok", 5, {"x": 1}]}, warn=warnings.append
        )
        assert out["texts"] == ["ok"]
        # 5 and the dict each warn once.
        assert len(warnings) == 2
        joined = " ".join(warnings)
        assert "int" in joined
        assert "dict" in joined

    def test_empty_string_item_warns(self):
        warnings: list[str] = []
        out = parse_filler_config(
            {"fillers": ["ok", "   ", ""]}, warn=warnings.append
        )
        assert out["texts"] == ["ok"]
        assert len(warnings) == 2  # whitespace-only + empty
        assert all("empty" in w or "whitespace" in w for w in warnings)

    def test_bad_threshold_warns(self):
        warnings: list[str] = []
        out = parse_filler_config(
            {"fillers_idle_threshold": "abc"}, warn=warnings.append
        )
        assert out["idle_threshold"] == 0.6
        assert len(warnings) == 1
        assert "fillers_idle_threshold" in warnings[0]
        assert "abc" in warnings[0]
        assert "0.6" in warnings[0]

    def test_non_positive_threshold_warns(self):
        warnings: list[str] = []
        parse_filler_config(
            {"fillers_idle_threshold": -1.0}, warn=warnings.append
        )
        assert len(warnings) == 1
        assert "-1.0" in warnings[0]

    def test_bool_threshold_warns(self):
        warnings: list[str] = []
        parse_filler_config(
            {"fillers_idle_threshold": True}, warn=warnings.append
        )
        assert len(warnings) == 1
        assert "fillers_idle_threshold" in warnings[0]

    def test_valid_fillers_with_bad_threshold_warns_once(self):
        # Only the rejected key warns; valid keys stay silent.
        warnings: list[str] = []
        out = parse_filler_config(
            {"fillers": ["hmm"], "fillers_idle_threshold": 0},
            warn=warnings.append,
        )
        assert out["texts"] == ["hmm"]
        assert len(warnings) == 1
        assert "fillers_idle_threshold" in warnings[0]
