"""Tests for iter-114 — _emit_filler_diversity_line helper.

Defensive sentinel for iter-113's cross-turn filler variety
fix. Scans the per-turn last_filler_id sequence for runs of the
same id ≥ threshold; emits a warning line when found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_filler_diversity_line,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line=""):
        lines.append(_strip_ansi(line))

    return emit, lines


# ---- Empty / no-filler sessions -----------------------------------------


def test_empty_list_emits_nothing():
    """No turns at all → no warning."""
    emit, lines = _capture()
    _emit_filler_diversity_line(emit, [])
    assert lines == []


def test_all_zero_emits_nothing():
    """No turn fired a filler — non-filler sessions don't need
    the line."""
    emit, lines = _capture()
    _emit_filler_diversity_line(emit, [0, 0, 0, 0])
    assert lines == []


# ---- Below threshold (clean variety) ------------------------------------


def test_no_repetition_emits_nothing():
    """Distinct filler each turn — no warning."""
    emit, lines = _capture()
    _emit_filler_diversity_line(emit, [101, 102, 103, 101, 104])
    assert lines == []


def test_two_in_a_row_below_default_threshold():
    """Default threshold is 3; runs of 2 are not flagged
    (random.choice can trivially produce 2 in a row)."""
    emit, lines = _capture()
    _emit_filler_diversity_line(emit, [101, 101, 102, 103])
    assert lines == []


def test_alternating_pattern_no_runs():
    """101, 102, 101, 102 — no consecutive run, no warning."""
    emit, lines = _capture()
    _emit_filler_diversity_line(emit, [101, 102, 101, 102, 101])
    assert lines == []


# ---- At-or-above threshold (repetition detected) ------------------------


def test_three_in_a_row_emits_warning():
    """Default threshold = 3."""
    emit, lines = _capture()
    _emit_filler_diversity_line(emit, [101, 101, 101])
    assert len(lines) == 1
    assert "Filler diversity" in lines[0]
    assert "filler 101" in lines[0]
    assert "3 turns running" in lines[0]
    assert "iter-113 cross-turn FIFO" in lines[0]


def test_four_in_a_row_reports_run_length():
    """The warning mentions the actual run length, not just ≥
    threshold."""
    emit, lines = _capture()
    _emit_filler_diversity_line(emit, [101, 101, 101, 101])
    assert "4 turns running" in lines[0]


def test_run_at_end_of_sequence():
    """Run that extends to the last turn must be detected."""
    emit, lines = _capture()
    _emit_filler_diversity_line(emit, [101, 102, 103, 103, 103])
    assert "filler 103" in lines[0]
    assert "3 turns running" in lines[0]


def test_run_at_start_of_sequence():
    """Run at the very beginning."""
    emit, lines = _capture()
    _emit_filler_diversity_line(emit, [101, 101, 101, 102])
    assert "filler 101" in lines[0]
    assert "3 turns running" in lines[0]


def test_longest_of_multiple_runs_is_reported():
    """If two distinct ids each have a run, report the longer.
    Tie-breaks: first one encountered."""
    emit, lines = _capture()
    _emit_filler_diversity_line(emit, [101, 101, 101, 102, 102, 102, 102])
    # 102 has a longer run (4 vs 3).
    assert "filler 102" in lines[0]
    assert "4 turns running" in lines[0]


# ---- Zero-filtering (zero turns don't break runs) -----------------------


def test_zeros_between_same_filler_dont_break_run():
    """Per the docstring rule: zero-turns are filtered out
    BEFORE counting runs. The filtered sequence [101, 101, 101]
    still triggers the warning even though there were
    intervening no-filler turns."""
    emit, lines = _capture()
    _emit_filler_diversity_line(emit, [101, 0, 101, 0, 101])
    assert len(lines) == 1
    assert "filler 101" in lines[0]
    assert "3 turns running" in lines[0]


def test_zeros_at_start_dont_count():
    """Leading zeros filter out cleanly."""
    emit, lines = _capture()
    _emit_filler_diversity_line(emit, [0, 0, 101, 101, 101])
    assert "filler 101" in lines[0]


def test_zeros_at_end_dont_count():
    emit, lines = _capture()
    _emit_filler_diversity_line(emit, [101, 101, 101, 0, 0])
    assert "filler 101" in lines[0]


# ---- Custom threshold ---------------------------------------------------


def test_threshold_5_does_not_fire_on_4_run():
    """Operator can raise the threshold to suppress short-run
    warnings."""
    emit, lines = _capture()
    _emit_filler_diversity_line(emit, [101, 101, 101, 101], threshold=5)
    assert lines == []


def test_threshold_5_fires_on_5_run():
    emit, lines = _capture()
    _emit_filler_diversity_line(
        emit, [101, 101, 101, 101, 101], threshold=5,
    )
    assert "5 turns running" in lines[0]


def test_threshold_2_fires_on_2_run():
    """Stricter threshold catches shorter repetitions."""
    emit, lines = _capture()
    _emit_filler_diversity_line(emit, [101, 101, 102], threshold=2)
    assert "2 turns running" in lines[0]


# ---- Output formatting ---------------------------------------------------


def test_line_has_leading_4_space_indent():
    """Match the _emit_*_block family's indent convention."""
    emit, lines = _capture()
    _emit_filler_diversity_line(emit, [101, 101, 101])
    assert lines[0].startswith("    ")


def test_warning_includes_iter_113_attribution():
    """The line names iter-113 explicitly so the operator
    knows which fix to check when investigating."""
    emit, lines = _capture()
    _emit_filler_diversity_line(emit, [101, 101, 101])
    assert "iter-113" in lines[0]
