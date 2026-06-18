"""Tests for iter-211 — _emit_max_token_gap_consistency_line.

Twelfth instance of the diversity-check pattern. Ninth applied to a
CONTINUOUS metric (after iter-128 sentence-length, iter-140 stt-rtf,
iter-141 tts-rtf, iter-142 llm-tps, iter-143 streaming-overlap,
iter-208 synth-dispatch, iter-209 eot-overhead) — buckets the per-turn
``max_token_gap`` via ``_max_token_gap_bucket`` before scanning.
Detects 5+ consecutive turns that landed in the "slow" or "very_slow"
bucket, surfacing the case where the LLM stalls mid-stream — a failure
mode iter-085's field comment calls "currently invisible to operators
because the user just hears a long pause and no signal fires"
(metric 3.21).

Like iter-140/141/208/209 and UNLIKE iter-142/143, the fine bucket is a
LOW value (small worst-case gap) — the gap is smaller-is-better, so the
boundaries are NOT inverted. Note the contrast with iter-142's
``llm_tps`` sentinel: that watches AVERAGE throughput (bigger-is-better),
this watches the WORST single stall (smaller-is-better) — different
statistic, opposite direction.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_max_token_gap_consistency_line,
    _max_token_gap_bucket,
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
    """0s = no measurable inter-token gap (single-token turn) → empty
    bucket (filtered by the consumer)."""
    assert _max_token_gap_bucket(0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty. Shouldn't happen but the
    fallback is cheap."""
    assert _max_token_gap_bucket(-0.5) == ""


def test_bucket_fast_boundary():
    """< 0.50s → fast (the desired state — no perceptible stall).
    0.4999 is the upper edge."""
    assert _max_token_gap_bucket(0.1) == "fast"
    assert _max_token_gap_bucket(0.4999) == "fast"


def test_bucket_slow_boundary():
    """0.50-2.0s inclusive → slow."""
    assert _max_token_gap_bucket(0.50) == "slow"
    assert _max_token_gap_bucket(2.0) == "slow"


def test_bucket_very_slow_boundary():
    """> 2.0s → very_slow."""
    assert _max_token_gap_bucket(2.001) == "very_slow"
    assert _max_token_gap_bucket(5.0) == "very_slow"


def test_bucket_handles_floats():
    """max_token_gap is a float — bucket must handle fine-grained
    values around the boundaries."""
    assert _max_token_gap_bucket(0.4999) == "fast"
    assert _max_token_gap_bucket(0.5001) == "slow"
    assert _max_token_gap_bucket(2.0) == "slow"
    assert _max_token_gap_bucket(2.0001) == "very_slow"


# ---- Empty / no-measurement sessions ---------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_max_token_gap_consistency_line(emit, [])
    assert lines == []


def test_all_zero_emit_nothing():
    """All turns were single-token / no measurable gap (0s) → no
    warning."""
    emit, lines = _capture()
    _emit_max_token_gap_consistency_line(emit, [0.0] * 10)
    assert lines == []


# ---- "fast" excluded -------------------------------------------------


def test_long_fast_run_does_not_fire():
    """A 10-turn run of negligible gaps is the desired state — never
    flagged."""
    emit, lines = _capture()
    _emit_max_token_gap_consistency_line(emit, [0.1] * 10)
    assert lines == []


def test_alternating_fast_and_slow_only_slow_counts():
    """[0.1, 1.0, 0.1, 1.0, ...] → after filtering, [slow] runs of 1.
    Below threshold → silent."""
    emit, lines = _capture()
    _emit_max_token_gap_consistency_line(
        emit, [0.1, 1.0, 0.1, 1.0, 0.1, 1.0],
    )
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_slow_in_a_row_fires():
    """Default threshold = 5."""
    emit, lines = _capture()
    _emit_max_token_gap_consistency_line(emit, [1.0] * 5)
    assert len(lines) == 1
    assert "Max token gap" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'slow'" in lines[0]
    assert "the LLM keeps stalling mid-stream" in lines[0]
    assert "iter-085" in lines[0]


def test_six_very_slow_in_a_row_fires():
    emit, lines = _capture()
    _emit_max_token_gap_consistency_line(emit, [3.0] * 6)
    assert len(lines) == 1
    assert "6 consecutive" in lines[0]
    assert "'very_slow'" in lines[0]
    assert "the LLM stalls severely mid-stream" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold not met."""
    emit, lines = _capture()
    _emit_max_token_gap_consistency_line(emit, [1.0] * 4)
    assert lines == []


# ---- Filter behavior (fast interleavings) ----------------------


def test_fast_between_slow_doesnt_break_run():
    """Same precedent as iter-126/128/140/208/209: filter the
    uninteresting bucket out before scanning. A 'fast' interleaving
    doesn't break a slow run."""
    emit, lines = _capture()
    # slow, fast, slow, fast, slow, slow, slow
    _emit_max_token_gap_consistency_line(
        emit, [1.0, 0.1, 1.0, 0.1, 1.0, 1.0, 1.0],
    )
    # Filtered: [slow]*5 → fires.
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]


def test_very_slow_breaks_slow_run():
    """Phase change between flagged buckets DOES break the run.
    slow followed by very_slow are both noteworthy but not the
    same run."""
    emit, lines = _capture()
    # 3 slow, 1 very_slow, 3 slow → longest run is 3 of slow.
    # Below threshold.
    _emit_max_token_gap_consistency_line(
        emit, [1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0],
    )
    assert lines == []


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_max_token_gap_consistency_line(emit, [3.0] * 3, threshold=3)
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_5_run():
    emit, lines = _capture()
    _emit_max_token_gap_consistency_line(emit, [3.0] * 5, threshold=10)
    assert lines == []


# ---- Longest of multiple ------------------------------------------


def test_longer_very_slow_run_beats_shorter_slow_run():
    """[slow]*4 + [very_slow]*7 → only very_slow passes threshold;
    warning fires for very_slow."""
    emit, lines = _capture()
    _emit_max_token_gap_consistency_line(emit, [1.0] * 4 + [3.0] * 7)
    assert "7 consecutive" in lines[0]
    assert "'very_slow'" in lines[0]


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_max_token_gap_consistency_line(emit, [1.0] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_085_attribution():
    emit, lines = _capture()
    _emit_max_token_gap_consistency_line(emit, [1.0] * 5)
    assert "iter-085" in lines[0]


# ---- Pattern parity with iter-114/.../210 -------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_max_token_gap_consistency_line(emit, [1.0] * 1000)
    assert "1000 consecutive" in lines[0]
