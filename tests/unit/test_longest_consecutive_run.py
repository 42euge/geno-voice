"""Tests for iter-116 — _longest_consecutive_run primitive.

Pure list-scanning helper extracted from iter-114 + iter-115.
Returns (length, value) for the longest consecutive-equal run
in the input list. Empty input → (0, None). Ties resolve to the
EARLIER (first-encountered) run.

Tested as a stand-alone function so its behavior is locked
independently of either consuming `_emit_*_line` helper.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import _longest_consecutive_run  # noqa: E402


# ---- Empty / singleton --------------------------------------------------


def test_empty_returns_zero_none():
    assert _longest_consecutive_run([]) == (0, None)


def test_single_value_returns_one():
    assert _longest_consecutive_run([42]) == (1, 42)


def test_single_string_value():
    assert _longest_consecutive_run(["solo"]) == (1, "solo")


# ---- Single run --------------------------------------------------------


def test_all_same_returns_full_length():
    assert _longest_consecutive_run([7, 7, 7, 7, 7]) == (5, 7)


def test_two_in_a_row():
    assert _longest_consecutive_run([7, 7]) == (2, 7)


# ---- Multiple runs -----------------------------------------------------


def test_one_run_in_otherwise_distinct():
    """[1, 2, 2, 2, 3] — longest run is 3 of value 2."""
    assert _longest_consecutive_run([1, 2, 2, 2, 3]) == (3, 2)


def test_alternating_returns_one():
    """No run of length > 1."""
    assert _longest_consecutive_run([1, 2, 1, 2, 1]) == (1, 1)


def test_run_at_start():
    """Earliest-tie rule: first run wins on length."""
    assert _longest_consecutive_run([5, 5, 5, 1, 2, 3]) == (3, 5)


def test_run_at_end():
    assert _longest_consecutive_run([1, 2, 3, 9, 9, 9]) == (3, 9)


def test_two_equal_length_runs_first_wins():
    """[1, 1, 1, 2, 2, 2] — both runs are 3 long. Tie → first wins."""
    assert _longest_consecutive_run([1, 1, 1, 2, 2, 2]) == (3, 1)


def test_longer_later_run_wins():
    """[1, 1, 2, 2, 2, 2] — second run (length 4) beats first (length 2)."""
    assert _longest_consecutive_run([1, 1, 2, 2, 2, 2]) == (4, 2)


# ---- String values ---------------------------------------------------


def test_string_run():
    """Strings work the same as numbers."""
    assert _longest_consecutive_run(
        ["rushed", "rushed", "natural", "rushed"],
    ) == (2, "rushed")


def test_mixed_strings_with_run():
    assert _longest_consecutive_run(
        ["a", "b", "b", "b", "c"],
    ) == (3, "b")


# ---- Mixed types -----------------------------------------------------


def test_zero_runs_count_when_not_filtered():
    """The helper does NOT filter — callers do that. Verifies that
    if zeros are PASSED IN, they count."""
    assert _longest_consecutive_run([0, 0, 0, 1, 0, 0]) == (3, 0)


def test_none_values_treated_normally():
    """None compares equal to None — runs of None work."""
    assert _longest_consecutive_run([None, None, None]) == (3, None)


# ---- Long inputs -----------------------------------------------------


def test_large_list_with_central_run():
    """100 distinct values, run of 7 in the middle."""
    values = list(range(50)) + [99] * 7 + list(range(50, 100))
    assert _longest_consecutive_run(values) == (7, 99)


# ---- Iter-114 / Iter-115 round-trip parity ---------------------------


def test_iter_114_filtered_input_shape():
    """The shape iter-114 passes (after zero-filtering)."""
    fired = [101, 0, 101, 0, 101]
    fired_filtered = [f for f in fired if f != 0]
    assert _longest_consecutive_run(fired_filtered) == (3, 101)


def test_iter_115_filtered_input_shape():
    """The shape iter-115 passes (after empty-string filtering)."""
    buckets = ["", "rushed", "rushed", "rushed", "rushed", "rushed", ""]
    non_empty = [b for b in buckets if b]
    assert _longest_consecutive_run(non_empty) == (5, "rushed")


def test_returns_tuple_not_list():
    """Lock the return shape — callers tuple-unpack it."""
    result = _longest_consecutive_run([1, 1])
    assert isinstance(result, tuple)
    assert len(result) == 2
