"""Tests for iter-105 — _emit_wer_line helper.

Single-line emit: reports median + max WER across turns where a
reference transcript was supplied. Suppressed when no turn
measured WER (default for sessions without ground-truth refs).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import _emit_wer_line  # noqa: E402


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line=""):
        lines.append(_strip_ansi(line))

    return emit, lines


def test_empty_list_emits_nothing():
    """No turn measured WER → no line. The default for sessions
    without ground-truth references."""
    emit, lines = _capture()
    _emit_wer_line(emit, [])
    assert lines == []


def test_single_perfect_turn():
    """One turn, WER 0.0 → line shows '0.00 median, 0.00 max'."""
    emit, lines = _capture()
    _emit_wer_line(emit, [0.0])
    assert len(lines) == 1
    assert "WER:" in lines[0]
    assert "0.00 median" in lines[0]
    assert "0.00 max" in lines[0]
    assert "1 turns measured" in lines[0]


def test_single_imperfect_turn():
    """One turn, non-zero WER. The line still shows it — useful
    even with N=1 because the operator sees there WAS a
    reference."""
    emit, lines = _capture()
    _emit_wer_line(emit, [0.15])
    assert "0.15 median" in lines[0]
    assert "0.15 max" in lines[0]


def test_multi_turn_median_and_max():
    """Median and max computed correctly across multiple turns."""
    emit, lines = _capture()
    _emit_wer_line(emit, [0.1, 0.05, 0.2, 0.15, 0.1])
    # Median of [0.05, 0.1, 0.1, 0.15, 0.2] = 0.1, max = 0.2.
    assert "0.10 median" in lines[0]
    assert "0.20 max" in lines[0]
    assert "5 turns measured" in lines[0]


def test_two_turn_median_is_average():
    """Even-length lists: median is the mean of the middle two
    (statistics.median behavior, matches _emit_ttfs_block
    convention)."""
    emit, lines = _capture()
    _emit_wer_line(emit, [0.1, 0.3])
    # Median of [0.1, 0.3] = 0.2.
    assert "0.20 median" in lines[0]
    assert "0.30 max" in lines[0]


def test_format_two_decimals():
    """0.123 should render as 0.12, not 0.123 or 0.1."""
    emit, lines = _capture()
    _emit_wer_line(emit, [0.123])
    assert "0.12" in lines[0]


def test_high_wer_above_one():
    """WER > 1.0 is possible (insertion-heavy hypothesis).
    Format still works."""
    emit, lines = _capture()
    _emit_wer_line(emit, [1.5])
    assert "1.50" in lines[0]


def test_line_has_leading_4_space_indent():
    """Match the _emit_*_block family's indent convention."""
    emit, lines = _capture()
    _emit_wer_line(emit, [0.0])
    assert lines[0].startswith("    ")


def test_turn_count_uses_actual_list_length():
    """The "(N turns measured)" suffix counts list length —
    NOT the median of any input. Edge case: a list with one
    duplicate value, length 3."""
    emit, lines = _capture()
    _emit_wer_line(emit, [0.1, 0.1, 0.1])
    assert "3 turns measured" in lines[0]
