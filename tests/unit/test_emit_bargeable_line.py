"""Tests for iter-104 — _emit_bargeable_line helper.

Single-line emit: reports the WORST bargeable fraction across
turns (iter-074), but only when at least one turn dipped below
99%. Clean sessions (every turn ≥99% bargeable) suppress the
line.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import _emit_bargeable_line  # noqa: E402


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line=""):
        lines.append(_strip_ansi(line))

    return emit, lines


def test_empty_list_emits_nothing():
    """No turns reported a bargeable fraction → no line."""
    emit, lines = _capture()
    _emit_bargeable_line(emit, [])
    assert lines == []


def test_all_above_threshold_emits_nothing():
    """Every turn ≥99% — operator doesn't need to see the line.
    1.0 = 100%, 0.99 = exactly threshold (excluded)."""
    emit, lines = _capture()
    _emit_bargeable_line(emit, [1.0, 1.0, 0.99, 1.0])
    assert lines == []


def test_one_turn_below_threshold_emits_line():
    """A single sub-99% turn surfaces the regression."""
    emit, lines = _capture()
    _emit_bargeable_line(emit, [1.0, 0.85, 1.0])
    assert len(lines) == 1
    assert "Bargeable:" in lines[0]
    # min is 0.85 → 85% worst.
    assert "85%" in lines[0]
    # 1 of 3 turns < 99%.
    assert "1/3" in lines[0]
    assert "watcher coverage regression" in lines[0]


def test_multiple_turns_below_threshold():
    """Below-count counts ALL sub-99% turns, not just the worst."""
    emit, lines = _capture()
    _emit_bargeable_line(emit, [0.5, 0.7, 0.9, 1.0, 0.95])
    # min 0.5 → 50%, 4 of 5 < 99%.
    assert "50%" in lines[0]
    assert "4/5" in lines[0]


def test_worst_rounds_to_zero_decimals():
    """50.4% should render as 50%, not 50.4%."""
    emit, lines = _capture()
    _emit_bargeable_line(emit, [0.504])
    assert "50%" in lines[0]


def test_threshold_boundary_exactly_99():
    """0.99 exactly is NOT below threshold (the guard is < 0.99,
    not <= 0.99). A list with exactly-99% values stays silent."""
    emit, lines = _capture()
    _emit_bargeable_line(emit, [0.99, 0.99])
    assert lines == []


def test_just_below_threshold_emits():
    """0.989 IS below threshold → 99% worst, but watcher
    regression flagged."""
    emit, lines = _capture()
    _emit_bargeable_line(emit, [0.989])
    assert len(lines) == 1
    # 0.989 * 100 = 98.9 → rounds to 99% with .0f.
    assert "99%" in lines[0]


def test_line_has_leading_4_space_indent():
    """Match the _emit_*_block family's indent convention."""
    emit, lines = _capture()
    _emit_bargeable_line(emit, [0.5])
    assert lines[0].startswith("    ")
