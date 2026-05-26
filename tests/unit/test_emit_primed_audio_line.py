"""Tests for iter-104 — _emit_primed_audio_line helper.

Single-line emit: reports cumulative primed-audio seconds carried
across turns (iter-057). Suppressed when total is 0.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import _emit_primed_audio_line  # noqa: E402


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line=""):
        lines.append(_strip_ansi(line))

    return emit, lines


def test_zero_total_emits_nothing():
    """Clean sessions (no primed frames) don't bloat the summary."""
    emit, lines = _capture()
    _emit_primed_audio_line(emit, 0.0)
    assert lines == []


def test_negative_or_zero_treated_as_clean():
    """Defensive: a 0.0 (or negative, though impossible in
    practice) total emits nothing — the > 0 guard isolates clean
    sessions from any caller-side accounting glitch."""
    emit, lines = _capture()
    _emit_primed_audio_line(emit, -0.5)
    assert lines == []


def test_positive_total_emits_one_line():
    """Any total > 0 surfaces the primed-audio line with one
    decimal place and the iter-025 validation note."""
    emit, lines = _capture()
    _emit_primed_audio_line(emit, 0.7)
    assert len(lines) == 1
    assert "Primed audio:" in lines[0]
    assert "0.7s" in lines[0]
    assert "validates iter-025" in lines[0]


def test_total_rounds_to_one_decimal():
    """0.349 should render as 0.3, not 0.349 or 0."""
    emit, lines = _capture()
    _emit_primed_audio_line(emit, 0.349)
    assert "0.3s" in lines[0]


def test_large_total_uses_default_format():
    """No special-casing for big numbers — 12.34s renders as 12.3s."""
    emit, lines = _capture()
    _emit_primed_audio_line(emit, 12.34)
    assert "12.3s" in lines[0]


def test_line_has_leading_4_space_indent():
    """Match the _emit_*_block family's indent convention."""
    emit, lines = _capture()
    _emit_primed_audio_line(emit, 1.0)
    assert lines[0].startswith("    ")
