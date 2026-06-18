"""Tests for iter-225 — _emit_split_coverage_consistency_line.

Latest instance of the diversity-check pattern, applied to a
CONTINUOUS metric: buckets the per-turn ``sentence_split_coverage``
(iter-059's fraction of submitted chars that landed inside a complete
sentence, in [0.0, 1.0]) via ``_split_coverage_bucket`` before
scanning. Detects 5+ consecutive turns that landed in the "partial"
or "poor" bucket — the signal that the sentence splitter keeps
flushing a large trailing remainder, defeating the iter-008
streaming-overlap design turn after turn.

Like iter-142 (llm-tps) and iter-143 (streaming-overlap) and UNLIKE
iter-140/141/224, the fine bucket is a HIGH value (coverage close to
1.0 is better), so the boundaries ARE inverted — this is the THIRD
inverted-direction continuous bucketer.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_split_coverage_consistency_line,
    _split_coverage_bucket,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line=""):
        lines.append(_strip_ansi(line))

    return emit, lines


# ---- Bucket boundaries -----------------------------------------------


def test_bucket_zero_returns_empty():
    """0 coverage = no chars submitted this turn → empty bucket (the
    no-measurement state, filtered by the consumer)."""
    assert _split_coverage_bucket(0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty. Shouldn't happen (metric is
    a fraction in [0,1]) but a defensive fallback is cheap."""
    assert _split_coverage_bucket(-1.0) == ""


def test_bucket_full_boundary():
    """>= 0.90 → full (the desired state). 1.0 and 0.90 are full."""
    assert _split_coverage_bucket(1.0) == "full"
    assert _split_coverage_bucket(0.90) == "full"


def test_bucket_partial_boundary():
    """0.70-0.90 → partial. 0.70 inclusive, just under 0.90."""
    assert _split_coverage_bucket(0.70) == "partial"
    assert _split_coverage_bucket(0.899) == "partial"


def test_bucket_poor_boundary():
    """> 0 but < 0.70 → poor."""
    assert _split_coverage_bucket(0.699) == "poor"
    assert _split_coverage_bucket(0.01) == "poor"


def test_bucket_handles_floats():
    """sentence_split_coverage is a float — bucket must handle
    fine-grained values around the inverted boundaries."""
    assert _split_coverage_bucket(0.8999) == "partial"
    assert _split_coverage_bucket(0.9001) == "full"
    assert _split_coverage_bucket(0.7001) == "partial"
    assert _split_coverage_bucket(0.6999) == "poor"


# ---- Empty / no-measurement sessions ---------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_split_coverage_consistency_line(emit, [])
    assert lines == []


def test_all_zero_coverage_emit_nothing():
    """All turns had no submitted chars (0 coverage) → no warning;
    nothing measurable."""
    emit, lines = _capture()
    _emit_split_coverage_consistency_line(emit, [0.0] * 10)
    assert lines == []


# ---- "full" excluded -------------------------------------------------


def test_long_full_run_does_not_fire():
    """A 10-turn run of full-coverage (>=0.90) splits is the desired
    state — never flagged."""
    emit, lines = _capture()
    _emit_split_coverage_consistency_line(emit, [0.98] * 10)
    assert lines == []


def test_perfect_coverage_does_not_fire():
    """1.0 every turn = perfect overlap; never flagged."""
    emit, lines = _capture()
    _emit_split_coverage_consistency_line(emit, [1.0] * 8)
    assert lines == []


def test_alternating_full_and_partial_only_partial_counts():
    """[0.98, 0.8, 0.98, 0.8, ...] → after filtering, [partial] runs
    of 1. Below threshold → silent."""
    emit, lines = _capture()
    _emit_split_coverage_consistency_line(
        emit, [0.98, 0.8, 0.98, 0.8, 0.98, 0.8],
    )
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_partial_in_a_row_fires():
    """Default threshold = 5."""
    emit, lines = _capture()
    _emit_split_coverage_consistency_line(emit, [0.8] * 5)
    assert len(lines) == 1
    assert "Split coverage" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'partial'" in lines[0]
    assert "overlap is" in lines[0]
    assert "iter-059" in lines[0]


def test_six_poor_in_a_row_fires():
    emit, lines = _capture()
    _emit_split_coverage_consistency_line(emit, [0.5] * 6)
    assert len(lines) == 1
    assert "6 consecutive" in lines[0]
    assert "'poor'" in lines[0]
    assert "synth runs sequentially" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold not met."""
    emit, lines = _capture()
    _emit_split_coverage_consistency_line(emit, [0.8] * 4)
    assert lines == []


# ---- Filter behavior (full interleavings) ---------------------------


def test_full_between_partial_doesnt_break_run():
    """Same precedent as iter-126/128/140/143: filter the
    uninteresting bucket out before scanning. A 'full' interleaving
    doesn't break a partial run."""
    emit, lines = _capture()
    # partial, full, partial, full, partial, partial, partial
    _emit_split_coverage_consistency_line(
        emit, [0.8, 0.98, 0.8, 0.98, 0.8, 0.8, 0.8],
    )
    # Filtered: [partial]*5 → fires.
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]


def test_poor_breaks_partial_run():
    """Phase change between flagged buckets DOES break the run.
    partial followed by poor are both noteworthy but not the same
    run."""
    emit, lines = _capture()
    # 3 partial, 1 poor, 3 partial → longest run is 3 of partial.
    # Below threshold.
    _emit_split_coverage_consistency_line(
        emit, [0.8, 0.8, 0.8, 0.5, 0.8, 0.8, 0.8],
    )
    assert lines == []


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_split_coverage_consistency_line(emit, [0.5] * 3, threshold=3)
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_5_run():
    emit, lines = _capture()
    _emit_split_coverage_consistency_line(emit, [0.5] * 5, threshold=10)
    assert lines == []


# ---- Longest of multiple ------------------------------------------


def test_longer_poor_run_beats_shorter_partial_run():
    """[partial]*4 + [poor]*7 → only poor passes threshold; warning
    fires for poor."""
    emit, lines = _capture()
    _emit_split_coverage_consistency_line(
        emit, [0.8] * 4 + [0.5] * 7,
    )
    assert "7 consecutive" in lines[0]
    assert "'poor'" in lines[0]


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_split_coverage_consistency_line(emit, [0.8] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_059_attribution():
    emit, lines = _capture()
    _emit_split_coverage_consistency_line(emit, [0.8] * 5)
    assert "iter-059" in lines[0]


# ---- Pattern parity with prior instances --------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_split_coverage_consistency_line(emit, [0.8] * 1000)
    assert "1000 consecutive" in lines[0]
