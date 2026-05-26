"""Tests for iter-115 — _emit_naturalness_consistency_line helper.

Mirrors iter-114's diversity-check shape on a different metric.
Detects runs of the same non-"natural" bucket ≥ threshold (5
default) and emits a speed-tuning suggestion. "natural" runs
are never flagged.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_naturalness_consistency_line,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line=""):
        lines.append(_strip_ansi(line))

    return emit, lines


# ---- Empty / no-bucket sessions -----------------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_naturalness_consistency_line(emit, [])
    assert lines == []


def test_all_empty_strings_emit_nothing():
    """Bucket "" means no audio played that turn (or naturalness
    couldn't be computed). All empties → no warning."""
    emit, lines = _capture()
    _emit_naturalness_consistency_line(emit, ["", "", "", "", "", ""])
    assert lines == []


# ---- "natural" never fires ---------------------------------------------


def test_natural_run_does_not_fire():
    """A run of "natural" buckets is the desired state — never
    flagged."""
    emit, lines = _capture()
    _emit_naturalness_consistency_line(
        emit,
        ["natural"] * 10,
    )
    assert lines == []


def test_long_natural_run_does_not_fire_even_with_brief_outliers():
    """natural-natural-natural-rushed-natural-rushed → after
    iter-126 filters "natural" out, only [rushed, rushed] remains.
    Longest run is 2, below default threshold of 5 → silent."""
    emit, lines = _capture()
    _emit_naturalness_consistency_line(
        emit,
        ["natural"] * 6 + ["rushed", "natural", "rushed"],
    )
    # Filtered: [rushed, rushed] → run of 2, below threshold.
    assert lines == []


# ---- Below threshold (clean) -------------------------------------------


def test_four_rushed_below_default_threshold():
    """Default threshold = 5; 4 in a row doesn't fire."""
    emit, lines = _capture()
    _emit_naturalness_consistency_line(
        emit,
        ["rushed"] * 4 + ["natural"],
    )
    assert lines == []


def test_alternating_rushed_natural_no_run():
    emit, lines = _capture()
    _emit_naturalness_consistency_line(
        emit,
        ["rushed", "natural", "rushed", "natural", "rushed", "natural"],
    )
    assert lines == []


# ---- At/above threshold (warning fires) --------------------------------


def test_five_rushed_in_a_row_fires():
    """Default threshold = 5."""
    emit, lines = _capture()
    _emit_naturalness_consistency_line(emit, ["rushed"] * 5)
    assert len(lines) == 1
    assert "Naturalness" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'rushed'" in lines[0]
    assert "consider reducing speed" in lines[0]
    assert "iter-053" in lines[0]


def test_slow_run_emits_increasing_speed_suggestion():
    """Slow runs trigger the OPPOSITE recommendation."""
    emit, lines = _capture()
    _emit_naturalness_consistency_line(emit, ["slow"] * 6)
    assert "consider increasing speed" in lines[0]
    assert "'slow'" in lines[0]
    assert "6 consecutive" in lines[0]


def test_run_at_end():
    emit, lines = _capture()
    _emit_naturalness_consistency_line(
        emit,
        ["natural", "natural", "rushed", "rushed", "rushed", "rushed", "rushed"],
    )
    assert "5 consecutive" in lines[0]
    assert "rushed" in lines[0]


def test_longest_of_multiple_runs_is_reported():
    """If two distinct buckets each have a run, report the longer."""
    emit, lines = _capture()
    _emit_naturalness_consistency_line(
        emit,
        # 5 rushed, 6 slow → slow wins.
        ["rushed"] * 5 + ["slow"] * 6,
    )
    assert "6 consecutive" in lines[0]
    assert "'slow'" in lines[0]


# ---- Empty-bucket filtering ---------------------------------------------


def test_empty_buckets_dont_break_run():
    """A turn with bucket="" (no audio) doesn't break a run —
    it's filtered before the scan. User perception is "5 rushed
    turns" regardless of intervening no-audio turns."""
    emit, lines = _capture()
    _emit_naturalness_consistency_line(
        emit,
        ["rushed", "", "rushed", "", "rushed", "", "rushed", "rushed"],
    )
    # Filtered: ["rushed"] × 5 → triggers default threshold.
    assert "5 consecutive" in lines[0]


def test_leading_empty_buckets():
    emit, lines = _capture()
    _emit_naturalness_consistency_line(
        emit,
        ["", "", "rushed", "rushed", "rushed", "rushed", "rushed"],
    )
    assert "5 consecutive" in lines[0]


# ---- Custom threshold ----------------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    """Operator can lower the threshold for stricter checking."""
    emit, lines = _capture()
    _emit_naturalness_consistency_line(
        emit, ["rushed"] * 3, threshold=3,
    )
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_5_run():
    """Raising the threshold suppresses noisy patterns."""
    emit, lines = _capture()
    _emit_naturalness_consistency_line(
        emit, ["rushed"] * 5, threshold=10,
    )
    assert lines == []


# ---- Output formatting -------------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_naturalness_consistency_line(emit, ["rushed"] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_bucket_attribution():
    """The line names iter-053 (the bucket-classification iter)
    so operators can find the fix path quickly."""
    emit, lines = _capture()
    _emit_naturalness_consistency_line(emit, ["rushed"] * 5)
    assert "iter-053" in lines[0]


def test_unknown_bucket_falls_back_to_generic_suggestion():
    """Defensive: an unrecognized bucket name (e.g., a future
    addition) doesn't break the line — emits a generic message."""
    emit, lines = _capture()
    _emit_naturalness_consistency_line(
        emit, ["weird_bucket"] * 5,
    )
    # Should not crash. With "weird_bucket" not matching any known
    # case, suggestion is generic.
    assert len(lines) == 1
    assert "consider tuning speed" in lines[0]


# ---- Mixed natural + rushed sequences ---------------------------------


def test_iter_126_natural_filter_preserves_run_in_split_pattern():
    """iter-126 filter rule: removing "natural" makes
    non-consecutive same-bucket turns appear consecutive in the
    filtered list. Same precedent as iter-114's zero-filter for
    fillers — user perception is "N rushed turns" regardless of
    intervening natural turns.

    [rushed, natural, rushed, natural, rushed, natural, rushed,
     rushed, natural] → filtered = [rushed]*5 → fires.
    """
    emit, lines = _capture()
    _emit_naturalness_consistency_line(
        emit,
        [
            "rushed", "natural", "rushed", "natural",
            "rushed", "natural", "rushed", "rushed", "natural",
        ],
    )
    # Filtered: [rushed]*5 → run of 5, default threshold met.
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]
    assert "'rushed'" in lines[0]


def test_iter_126_filter_does_not_affect_pure_rushed_runs():
    """When there's no "natural" to filter, behavior is
    unchanged — the filter is a no-op."""
    emit, lines = _capture()
    _emit_naturalness_consistency_line(
        emit, ["rushed"] * 5,
    )
    assert "5 consecutive" in lines[0]


def test_iter_126_mixed_rushed_and_slow_picks_longest():
    """When the filtered list has runs of multiple non-natural
    buckets, the LONGEST among them wins. Same tie-breaking as
    pre-iter-126."""
    emit, lines = _capture()
    _emit_naturalness_consistency_line(
        emit,
        ["natural"] * 3 + ["rushed"] * 3 + ["natural"] + ["slow"] * 6,
    )
    # Filtered: [rushed]*3 + [slow]*6 → longest is slow (6).
    assert "6 consecutive" in lines[0]
    assert "'slow'" in lines[0]
    assert "consider increasing speed" in lines[0]


def test_natural_run_longer_than_rushed_run_fires_after_iter_126():
    """iter-126 fix: when "natural" is the dominant bucket but a
    threshold-crossing "rushed" or "slow" run exists, the helper
    fires on the non-natural run.

    Pre-iter-126, the helper reported only the LONGEST run
    overall. With 7 natural + 5 rushed, "natural" was longer,
    suppressed by the "is longest_bucket natural?" check, and
    nothing fired. The rushed signal was lost.

    Post-iter-126, "natural" is filtered out before the scan
    (mirroring iter-114's zero-filter precedent), so the 5-run
    of "rushed" surfaces directly.
    """
    emit, lines = _capture()
    _emit_naturalness_consistency_line(
        emit,
        ["natural"] * 7 + ["rushed"] * 5,
    )
    # iter-126 fix: rushed run now surfaces.
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]
    assert "'rushed'" in lines[0]
    assert "consider reducing speed" in lines[0]
