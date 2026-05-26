"""Tests for iter-123 — parse_max_user_assistant helper.

Same convention as iter-020 / iter-034 / iter-119 parse-family
helpers: tolerant of malformed input, defaults backfilled,
never raises.

The cap is an int (not a float), with a special "0 = no cap"
sentinel for operators bypassing the trim during eval/replay.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_config import (  # noqa: E402
    MAX_USER_ASSISTANT_DEFAULT,
    parse_max_user_assistant,
)
from examples._chat_helpers import trim_history  # noqa: E402


# ---- Default ----------------------------------------------------------


def test_returns_int():
    """The contract is `dict | None → int`. Locks the type so
    callers don't need to coerce."""
    out = parse_max_user_assistant({})
    assert isinstance(out, int)


def test_default_is_20():
    """The iter-024 historical default. Pre-iter-123 callers
    relied on it being 20; iter-123 preserves that for any
    operator who hasn't set the new key."""
    assert MAX_USER_ASSISTANT_DEFAULT == 20
    assert parse_max_user_assistant({}) == 20


def test_missing_key_returns_default():
    """An empty chat dict (no max_user_assistant) → default."""
    out = parse_max_user_assistant({"other_key": "ignore"})
    assert out == 20


# ---- Malformed input -------------------------------------------------


def test_non_mapping_input_returns_default():
    """Defensive: None, list, string, int, etc → default."""
    for bad in [None, [], "20", 42, object()]:
        # Note 42 here is a bare int (not under a key) — the
        # function expects a mapping at the top level.
        assert parse_max_user_assistant(bad) == 20


def test_string_value_falls_back():
    """String "10" → default. We don't auto-coerce."""
    assert parse_max_user_assistant({"max_user_assistant": "10"}) == 20


def test_float_value_falls_back():
    """Float 10.0 → default. The cap is int-typed."""
    assert parse_max_user_assistant({"max_user_assistant": 10.0}) == 20


def test_negative_value_falls_back():
    """Negative cap is nonsensical → default."""
    assert parse_max_user_assistant({"max_user_assistant": -5}) == 20


def test_bool_value_falls_back():
    """Defensive: bool is a subclass of int. Without the guard,
    True → cap of 1 and False → cap of 0 ("no cap"). Both are
    surprising for a typo'd yaml value like `true`."""
    assert parse_max_user_assistant({"max_user_assistant": True}) == 20
    assert parse_max_user_assistant({"max_user_assistant": False}) == 20


# ---- Valid values ----------------------------------------------------


def test_zero_is_returned_as_zero():
    """Special "no cap" sentinel — explicitly returned, not
    coerced to default."""
    assert parse_max_user_assistant({"max_user_assistant": 0}) == 0


def test_small_positive_value():
    """Common operator-tuned values."""
    assert parse_max_user_assistant({"max_user_assistant": 5}) == 5


def test_large_positive_value():
    assert parse_max_user_assistant({"max_user_assistant": 100}) == 100


def test_default_value_passes_through():
    """Setting it explicitly to 20 (the default) is honored —
    not silently elided. This matters for operators who want
    explicit config audit trails."""
    assert parse_max_user_assistant({"max_user_assistant": 20}) == 20


# ---- Cap=0 round-trip semantics ------------------------------------


def test_cap_zero_passes_full_history_through_trim():
    """When the cap is 0, trim_history must return the
    unmodified list. Validates the "Python -0 == 0 → tail[-0:]
    is full tail" invariant the iter-123 wiring relies on.

    Without this invariant holding, cap=0 would evict EVERYTHING,
    which is a horrifying surprise for an operator setting the
    sentinel for eval/replay scenarios.
    """
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    out = trim_history(messages, max_user_assistant=0)
    assert out == messages
    # Defensive: must return a NEW list (or be safe to treat as
    # such — trim_history's contract is "does not mutate input").
    out.append({"role": "user", "content": "u3"})
    assert len(messages) == 5  # input unchanged


def test_cap_zero_with_empty_tail_returns_just_head():
    """Edge case: only system message + cap=0 → just the system
    message back."""
    messages = [{"role": "system", "content": "sys"}]
    out = trim_history(messages, max_user_assistant=0)
    assert out == messages


def test_cap_zero_with_no_system_returns_full_input():
    """Edge case: no system prompt + cap=0 → still full input
    (no trim)."""
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    out = trim_history(messages, max_user_assistant=0)
    assert out == messages


# ---- Independence -----------------------------------------------------


def test_does_not_mutate_input_dict():
    """The function reads the dict, doesn't write."""
    cfg = {"max_user_assistant": 7, "other": "value"}
    snapshot = dict(cfg)
    parse_max_user_assistant(cfg)
    assert cfg == snapshot


def test_does_not_mutate_default_constant():
    """No side effects on MAX_USER_ASSISTANT_DEFAULT."""
    snapshot = MAX_USER_ASSISTANT_DEFAULT
    parse_max_user_assistant({"max_user_assistant": 5})
    parse_max_user_assistant({"max_user_assistant": 999})
    assert MAX_USER_ASSISTANT_DEFAULT == snapshot
