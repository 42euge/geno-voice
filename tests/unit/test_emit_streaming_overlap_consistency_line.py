"""Tests for iter-143 — _emit_streaming_overlap_consistency_line.

Eighth instance of the diversity-check pattern. Fifth applied to a
CONTINUOUS metric (after iter-128 sentence-length, iter-140 stt-rtf,
iter-141 tts-rtf, iter-142 llm-tps) — buckets the per-turn
``streaming_overlap_ratio`` via ``_streaming_overlap_bucket`` before
scanning. Detects 5+ consecutive turns that landed in the "low" or
"very_low" bucket, suggesting the iter-008 streaming-overlap design
isn't masking synth.

Like iter-142 ``llm_tps`` and UNLIKE the RTF instances,
``streaming_overlap_ratio`` is bigger-is-better: the fine bucket
("high") is a HIGH value, so the bucket boundaries invert. This is
the SECOND inverted-direction continuous bucketer, so the boundary
tests below probe the high end as the "good" state.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_streaming_overlap_consistency_line,
    _streaming_overlap_bucket,
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
    """0 overlap = no measurable overlap this turn → empty bucket
    (filtered by the consumer)."""
    assert _streaming_overlap_bucket(0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty. Shouldn't happen in
    practice but a defensive fallback is cheap."""
    assert _streaming_overlap_bucket(-0.1) == ""


def test_bucket_high_boundary():
    """>= 0.50 → high (the desired, well-overlapped state). 0.50 is
    the lower edge of high."""
    assert _streaming_overlap_bucket(0.50) == "high"
    assert _streaming_overlap_bucket(1.0) == "high"


def test_bucket_low_boundary():
    """0.20-0.50 → low. Inclusive lower edge, exclusive upper."""
    assert _streaming_overlap_bucket(0.20) == "low"
    assert _streaming_overlap_bucket(0.4999) == "low"


def test_bucket_very_low_boundary():
    """< 0.20 (but > 0) → very_low."""
    assert _streaming_overlap_bucket(0.1999) == "very_low"
    assert _streaming_overlap_bucket(0.01) == "very_low"


def test_bucket_handles_floats():
    """streaming_overlap_ratio is a float — bucket must handle
    fine-grained values around the (inverted) boundaries."""
    assert _streaming_overlap_bucket(0.4999) == "low"
    assert _streaming_overlap_bucket(0.5001) == "high"
    assert _streaming_overlap_bucket(0.20) == "low"
    assert _streaming_overlap_bucket(0.1999) == "very_low"


# ---- Empty / no-measurement sessions ---------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_streaming_overlap_consistency_line(emit, [])
    assert lines == []


def test_all_zero_overlap_emit_nothing():
    """All turns had no measurable overlap (0) → no warning."""
    emit, lines = _capture()
    _emit_streaming_overlap_consistency_line(emit, [0.0] * 10)
    assert lines == []


# ---- "high" excluded (the inverted fine state) -----------------------


def test_long_high_run_does_not_fire():
    """A 10-turn run of high overlap is the desired state — never
    flagged. The fine bucket here is HIGH."""
    emit, lines = _capture()
    _emit_streaming_overlap_consistency_line(emit, [0.75] * 10)
    assert lines == []


def test_alternating_high_and_low_only_low_counts():
    """[0.75, 0.3, 0.75, 0.3, ...] → after filtering, [low] runs of
    1. Below threshold → silent."""
    emit, lines = _capture()
    _emit_streaming_overlap_consistency_line(
        emit, [0.75, 0.3, 0.75, 0.3, 0.75, 0.3],
    )
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_low_in_a_row_fires():
    """Default threshold = 5."""
    emit, lines = _capture()
    _emit_streaming_overlap_consistency_line(emit, [0.3] * 5)
    assert len(lines) == 1
    assert "Synth overlap" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'low'" in lines[0]
    assert "overlap is only partial" in lines[0]
    assert "iter-043" in lines[0]


def test_six_very_low_in_a_row_fires():
    emit, lines = _capture()
    _emit_streaming_overlap_consistency_line(emit, [0.1] * 6)
    assert len(lines) == 1
    assert "6 consecutive" in lines[0]
    assert "'very_low'" in lines[0]
    assert "sequentially after the LLM stream" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold not met."""
    emit, lines = _capture()
    _emit_streaming_overlap_consistency_line(emit, [0.3] * 4)
    assert lines == []


# ---- Filter behavior (high interleavings) ----------------------


def test_high_between_low_doesnt_break_run():
    """Same precedent as iter-126/128/140/141/142: filter the
    uninteresting bucket out before scanning. A 'high' interleaving
    doesn't break a low run."""
    emit, lines = _capture()
    # low, high, low, high, low, low, low
    _emit_streaming_overlap_consistency_line(
        emit, [0.3, 0.75, 0.3, 0.75, 0.3, 0.3, 0.3],
    )
    # Filtered: [low]*5 → fires.
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]


def test_very_low_breaks_low_run():
    """Phase change between flagged buckets DOES break the run.
    low followed by very_low are both noteworthy but not the same
    run."""
    emit, lines = _capture()
    # 3 low, 1 very_low, 3 low → longest run is 3 of low. Below
    # threshold.
    _emit_streaming_overlap_consistency_line(
        emit, [0.3, 0.3, 0.3, 0.1, 0.3, 0.3, 0.3],
    )
    assert lines == []


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_streaming_overlap_consistency_line(emit, [0.1] * 3, threshold=3)
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_5_run():
    emit, lines = _capture()
    _emit_streaming_overlap_consistency_line(emit, [0.1] * 5, threshold=10)
    assert lines == []


# ---- Longest of multiple ------------------------------------------


def test_longer_very_low_run_beats_shorter_low_run():
    """[low]*4 + [very_low]*7 → only very_low passes threshold;
    warning fires for very_low."""
    emit, lines = _capture()
    _emit_streaming_overlap_consistency_line(emit, [0.3] * 4 + [0.1] * 7)
    assert "7 consecutive" in lines[0]
    assert "'very_low'" in lines[0]


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_streaming_overlap_consistency_line(emit, [0.3] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_043_attribution():
    emit, lines = _capture()
    _emit_streaming_overlap_consistency_line(emit, [0.3] * 5)
    assert "iter-043" in lines[0]


# ---- Pattern parity with iter-114/115/120/126/128/140/141/142 -------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_streaming_overlap_consistency_line(emit, [0.3] * 1000)
    assert "1000 consecutive" in lines[0]
