"""Tests for iter-307 — _emit_queue_depth_consistency_line.

NINETEENTH instance of the diversity-check pattern, applied to a
CONTINUOUS (integer-valued) metric: buckets the per-turn
``max_queue_depth`` (iter-062's peak SentenceWorker queue depth — synth
jobs waiting behind the one in flight) via ``_queue_depth_bucket`` before
scanning. Detects 5+ consecutive turns that landed in the "backlog"
(depth 2) or "swamped" (depth >=3) bucket — the signal that the LLM
systematically outran synth, so the iter-008 streaming-overlap design
let mid-turn latency accumulate.

EXACT INVERSE of the iter-226 worker-idle-gap sentinel (synth starved
waiting on the LLM vs synth swamped by the LLM). Like iter-140/141 (RTF),
iter-208 (synth-dispatch), iter-224 (preview-divergence) and iter-226
(worker-idle-gap) — and UNLIKE iter-142/143/225 — the fine bucket is a
LOW value (depth 1 is best), so the boundaries are NOT inverted: the
problematic end is a LARGE backlog.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_queue_depth_consistency_line,
    _queue_depth_bucket,
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
    """depth 0 = the metric wasn't captured (no synth jobs this turn) →
    empty bucket (the no-measurement state, filtered by the consumer)."""
    assert _queue_depth_bucket(0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty. Shouldn't happen (a peak depth
    is non-negative) but a defensive fallback is cheap."""
    assert _queue_depth_bucket(-1) == ""


def test_bucket_smooth_boundary():
    """depth 1 → smooth (each sentence drained before the next arrived;
    the desired state, and iter-062's skip-when-<=1 case)."""
    assert _queue_depth_bucket(1) == "smooth"


def test_bucket_backlog_boundary():
    """depth 2 → backlog (iter-062's dim case — one sentence waited)."""
    assert _queue_depth_bucket(2) == "backlog"


def test_bucket_swamped_boundary():
    """depth >=3 → swamped (iter-062's yellow case — synth fell behind
    by three or more sentences)."""
    assert _queue_depth_bucket(3) == "swamped"
    assert _queue_depth_bucket(10) == "swamped"


# ---- Empty / no-measurement sessions ---------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_queue_depth_consistency_line(emit, [])
    assert lines == []


def test_all_zero_depth_emit_nothing():
    """All turns uncaptured (depth 0) → no warning; nothing
    measurable."""
    emit, lines = _capture()
    _emit_queue_depth_consistency_line(emit, [0] * 10)
    assert lines == []


# ---- "smooth" excluded -----------------------------------------------


def test_long_smooth_run_does_not_fire():
    """A 10-turn run of healthy depth-1 turns is the desired state —
    never flagged."""
    emit, lines = _capture()
    _emit_queue_depth_consistency_line(emit, [1] * 10)
    assert lines == []


def test_alternating_smooth_and_backlog_only_backlog_counts():
    """[1, 2, 1, 2, ...] → after filtering, [backlog] runs of 1. Below
    threshold → silent."""
    emit, lines = _capture()
    _emit_queue_depth_consistency_line(emit, [1, 2, 1, 2, 1, 2])
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_backlog_in_a_row_fires():
    """Default threshold = 5."""
    emit, lines = _capture()
    _emit_queue_depth_consistency_line(emit, [2] * 5)
    assert len(lines) == 1
    assert "Synth backlog" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'backlog'" in lines[0]
    assert "mildly outrunning" in lines[0]
    assert "iter-062" in lines[0]


def test_six_swamped_in_a_row_fires():
    emit, lines = _capture()
    _emit_queue_depth_consistency_line(emit, [4] * 6)
    assert len(lines) == 1
    assert "6 consecutive" in lines[0]
    assert "'swamped'" in lines[0]
    assert "bottleneck" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold not met."""
    emit, lines = _capture()
    _emit_queue_depth_consistency_line(emit, [2] * 4)
    assert lines == []


# ---- Filter behavior (smooth interleavings) -------------------------


def test_smooth_between_backlog_doesnt_break_run():
    """Same precedent as iter-126/128/140/143/225/226: filter the
    uninteresting bucket out before scanning. A 'smooth' (depth-1)
    interleaving doesn't break a backlog run."""
    emit, lines = _capture()
    # backlog, smooth, backlog, smooth, backlog, backlog, backlog
    _emit_queue_depth_consistency_line(emit, [2, 1, 2, 1, 2, 2, 2])
    # Filtered: [backlog]*5 → fires.
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]


def test_swamped_breaks_backlog_run():
    """Phase change between flagged buckets DOES break the run.
    backlog followed by swamped are both noteworthy but not the same
    run."""
    emit, lines = _capture()
    # 3 backlog, 1 swamped, 3 backlog → longest run is 3 of backlog.
    # Below threshold.
    _emit_queue_depth_consistency_line(emit, [2, 2, 2, 4, 2, 2, 2])
    assert lines == []


def test_uncaptured_zero_between_swamped_doesnt_break_run():
    """A depth-0 (uncaptured) turn filters out and doesn't break a
    swamped run, same as the smooth filter."""
    emit, lines = _capture()
    _emit_queue_depth_consistency_line(emit, [4, 4, 0, 4, 4, 4])
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]
    assert "'swamped'" in lines[0]


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_queue_depth_consistency_line(emit, [4] * 3, threshold=3)
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_5_run():
    emit, lines = _capture()
    _emit_queue_depth_consistency_line(emit, [4] * 5, threshold=10)
    assert lines == []


# ---- Longest of multiple ------------------------------------------


def test_longer_swamped_run_beats_shorter_backlog_run():
    """[backlog]*4 + [swamped]*7 → only swamped passes threshold;
    warning fires for swamped."""
    emit, lines = _capture()
    _emit_queue_depth_consistency_line(emit, [2] * 4 + [4] * 7)
    assert "7 consecutive" in lines[0]
    assert "'swamped'" in lines[0]


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_queue_depth_consistency_line(emit, [2] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_062_attribution():
    emit, lines = _capture()
    _emit_queue_depth_consistency_line(emit, [2] * 5)
    assert "iter-062" in lines[0]


# ---- Pattern parity with prior instances --------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_queue_depth_consistency_line(emit, [2] * 1000)
    assert len(lines) == 1
    assert "1000 consecutive" in lines[0]
