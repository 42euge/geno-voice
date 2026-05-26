"""Tests for iter-120 — _emit_barge_phase_consistency_line helper.

Third instance of the diversity-check pattern (after iter-114 +
iter-115). Detects 4+ consecutive turns barging in the same
phase ("llm_stream" or "playback") and emits a UX warning.

Both phases are flagged on consecutive runs (unlike iter-115
where "natural" was excluded — for barges, both phases are
informative).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_barge_phase_consistency_line,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line=""):
        lines.append(_strip_ansi(line))

    return emit, lines


# ---- Empty / no-barge sessions -----------------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_barge_phase_consistency_line(emit, [])
    assert lines == []


def test_all_empty_strings_emit_nothing():
    """No turn ever barged. Quiet session — no line."""
    emit, lines = _capture()
    _emit_barge_phase_consistency_line(emit, ["", "", "", "", "", ""])
    assert lines == []


# ---- Below threshold (clean) -------------------------------------------


def test_three_consecutive_below_default_threshold():
    """Default threshold = 4. 3-in-a-row doesn't fire."""
    emit, lines = _capture()
    _emit_barge_phase_consistency_line(
        emit, ["playback", "playback", "playback"],
    )
    assert lines == []


def test_alternating_phases_no_run():
    emit, lines = _capture()
    _emit_barge_phase_consistency_line(
        emit, ["playback", "llm_stream", "playback", "llm_stream"],
    )
    assert lines == []


# ---- Both phases flagged --------------------------------------------


def test_four_playback_in_a_row_fires_with_long_speech_suggestion():
    emit, lines = _capture()
    _emit_barge_phase_consistency_line(
        emit, ["playback"] * 4,
    )
    assert len(lines) == 1
    assert "Barge phase" in lines[0]
    assert "4 consecutive" in lines[0]
    assert "'playback'" in lines[0]
    assert "user habit or bot speaks too long" in lines[0]
    assert "iter-047" in lines[0]


def test_four_llm_stream_in_a_row_fires_with_ttft_suggestion():
    emit, lines = _capture()
    _emit_barge_phase_consistency_line(
        emit, ["llm_stream"] * 5,
    )
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]
    assert "'llm_stream'" in lines[0]
    assert "user impatient with LLM TTFB" in lines[0]


def test_neither_phase_excluded():
    """Unlike iter-115's 'natural' which was excluded, both
    barge phases warrant warnings — each carries a different
    UX signal."""
    emit, lines = _capture()
    _emit_barge_phase_consistency_line(emit, ["playback"] * 4)
    assert lines  # fires
    emit2, lines2 = _capture()
    _emit_barge_phase_consistency_line(emit2, ["llm_stream"] * 4)
    assert lines2  # also fires


# ---- Empty-string filtering (between barges) ---------------------------


def test_empty_strings_dont_break_run():
    """Turns with no barge (empty string) filter out before the
    scan. User perception: 4 consecutive playback barges
    regardless of intervening quiet turns."""
    emit, lines = _capture()
    _emit_barge_phase_consistency_line(
        emit,
        ["playback", "", "playback", "", "playback", "", "playback"],
    )
    assert "4 consecutive" in lines[0]


def test_leading_and_trailing_empty_strings():
    emit, lines = _capture()
    _emit_barge_phase_consistency_line(
        emit, ["", "", "playback", "playback", "playback", "playback", ""],
    )
    assert "4 consecutive" in lines[0]


def test_phase_change_breaks_run():
    """Mixing playback + llm_stream across non-empty turns DOES
    break the run. Filtering only drops empties."""
    emit, lines = _capture()
    _emit_barge_phase_consistency_line(
        emit,
        ["playback", "playback", "llm_stream", "playback", "playback"],
    )
    # Longest run is 2 (either side of the break).
    assert lines == []


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_barge_phase_consistency_line(
        emit, ["playback"] * 3, threshold=3,
    )
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_4_run():
    emit, lines = _capture()
    _emit_barge_phase_consistency_line(
        emit, ["playback"] * 4, threshold=10,
    )
    assert lines == []


# ---- Longest-of-multiple --------------------------------------------


def test_longest_of_multiple_runs_is_reported():
    """5 playback + 4 llm_stream → playback wins (longer)."""
    emit, lines = _capture()
    _emit_barge_phase_consistency_line(
        emit,
        ["playback"] * 5 + ["llm_stream"] * 4,
    )
    assert "5 consecutive" in lines[0]
    assert "'playback'" in lines[0]


def test_run_at_end_detected():
    emit, lines = _capture()
    _emit_barge_phase_consistency_line(
        emit,
        ["playback", "llm_stream"] + ["playback"] * 4,
    )
    assert "4 consecutive" in lines[0]
    assert "'playback'" in lines[0]


# ---- Output formatting ---------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_barge_phase_consistency_line(emit, ["playback"] * 4)
    assert lines[0].startswith("    ")


def test_warning_includes_phase_attribution():
    """The line names iter-047 (the phase-classification iter)
    so operators can find the fix path quickly."""
    emit, lines = _capture()
    _emit_barge_phase_consistency_line(emit, ["llm_stream"] * 4)
    assert "iter-047" in lines[0]


def test_unknown_phase_falls_back_to_generic_suggestion():
    """Defensive: an unrecognized phase string still emits a
    warning (instead of dropping the signal silently)."""
    emit, lines = _capture()
    _emit_barge_phase_consistency_line(
        emit, ["future_phase"] * 4,
    )
    assert len(lines) == 1
    assert "consistent barge phase" in lines[0]


# ---- Diversity pattern parity (iter-114 + iter-115) ----------------


def test_iter_116_helper_is_used():
    """Sanity that this iteration is in the same family. Same
    O(N) scan semantics — large lists work."""
    # 1000 turns, all playback — should still detect.
    emit, lines = _capture()
    _emit_barge_phase_consistency_line(
        emit, ["playback"] * 1000,
    )
    assert "1000 consecutive" in lines[0]
