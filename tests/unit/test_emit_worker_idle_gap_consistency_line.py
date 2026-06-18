"""Tests for iter-226 — _emit_worker_idle_gap_consistency_line.

Latest instance of the diversity-check pattern, applied to a
CONTINUOUS metric: buckets the per-turn ``worker_idle_gap_total``
(iter-044's cumulative seconds the SentenceWorker spent blocked
waiting for the next sentence after the first, in seconds) via
``_worker_idle_gap_bucket`` before scanning. Detects 5+ consecutive
turns that landed in the "stalled" or "starved" bucket — the signal
that the LLM systematically failed to keep up with synth+playback, so
the iter-008 streaming-overlap design left audible gaps of silence.

Like iter-140/141 (RTF), iter-208 (synth-dispatch) and iter-224
(preview-divergence) — and UNLIKE iter-142/143/225 — the fine bucket
is a LOW value (idle gap near 0 is better), so the boundaries are NOT
inverted: the problematic end is a LARGE idle gap.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_worker_idle_gap_consistency_line,
    _worker_idle_gap_bucket,
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
    """0 idle gap = the worker never blocked after the first sentence
    → empty bucket (the no-measurement / fine state, filtered by the
    consumer)."""
    assert _worker_idle_gap_bucket(0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty. Shouldn't happen (metric is
    a cumulative non-negative duration) but a defensive fallback is
    cheap."""
    assert _worker_idle_gap_bucket(-1.0) == ""


def test_bucket_smooth_boundary():
    """> 0 but < 0.3s → smooth (below the iter-044 yellow line; the
    desired state)."""
    assert _worker_idle_gap_bucket(0.01) == "smooth"
    assert _worker_idle_gap_bucket(0.299) == "smooth"


def test_bucket_stalled_boundary():
    """0.3-1.0s → stalled. 0.3 inclusive, 1.0 inclusive."""
    assert _worker_idle_gap_bucket(0.3) == "stalled"
    assert _worker_idle_gap_bucket(1.0) == "stalled"


def test_bucket_starved_boundary():
    """> 1.0s → starved."""
    assert _worker_idle_gap_bucket(1.001) == "starved"
    assert _worker_idle_gap_bucket(5.0) == "starved"


def test_bucket_handles_floats():
    """worker_idle_gap_total is a float (seconds) — bucket must handle
    fine-grained values around the (non-inverted) boundaries."""
    assert _worker_idle_gap_bucket(0.2999) == "smooth"
    assert _worker_idle_gap_bucket(0.3001) == "stalled"
    assert _worker_idle_gap_bucket(0.9999) == "stalled"
    assert _worker_idle_gap_bucket(1.0001) == "starved"


# ---- Empty / no-measurement sessions ---------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_worker_idle_gap_consistency_line(emit, [])
    assert lines == []


def test_all_zero_idle_gap_emit_nothing():
    """All turns the worker never blocked (0 idle gap) → no warning;
    nothing measurable."""
    emit, lines = _capture()
    _emit_worker_idle_gap_consistency_line(emit, [0.0] * 10)
    assert lines == []


# ---- "smooth" excluded -----------------------------------------------


def test_long_smooth_run_does_not_fire():
    """A 10-turn run of smooth idle gaps (<0.3s) is the desired
    state — never flagged."""
    emit, lines = _capture()
    _emit_worker_idle_gap_consistency_line(emit, [0.05] * 10)
    assert lines == []


def test_just_below_yellow_line_does_not_fire():
    """0.29s every turn = under the iter-044 yellow line; never
    flagged."""
    emit, lines = _capture()
    _emit_worker_idle_gap_consistency_line(emit, [0.29] * 8)
    assert lines == []


def test_alternating_smooth_and_stalled_only_stalled_counts():
    """[0.05, 0.5, 0.05, 0.5, ...] → after filtering, [stalled] runs
    of 1. Below threshold → silent."""
    emit, lines = _capture()
    _emit_worker_idle_gap_consistency_line(
        emit, [0.05, 0.5, 0.05, 0.5, 0.05, 0.5],
    )
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_stalled_in_a_row_fires():
    """Default threshold = 5."""
    emit, lines = _capture()
    _emit_worker_idle_gap_consistency_line(emit, [0.5] * 5)
    assert len(lines) == 1
    assert "Worker idle gap" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'stalled'" in lines[0]
    assert "short silences" in lines[0]
    assert "iter-044" in lines[0]


def test_six_starved_in_a_row_fires():
    emit, lines = _capture()
    _emit_worker_idle_gap_consistency_line(emit, [2.0] * 6)
    assert len(lines) == 1
    assert "6 consecutive" in lines[0]
    assert "'starved'" in lines[0]
    assert "over a second" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold not met."""
    emit, lines = _capture()
    _emit_worker_idle_gap_consistency_line(emit, [0.5] * 4)
    assert lines == []


# ---- Filter behavior (smooth interleavings) -------------------------


def test_smooth_between_stalled_doesnt_break_run():
    """Same precedent as iter-126/128/140/143/225: filter the
    uninteresting bucket out before scanning. A 'smooth' interleaving
    doesn't break a stalled run."""
    emit, lines = _capture()
    # stalled, smooth, stalled, smooth, stalled, stalled, stalled
    _emit_worker_idle_gap_consistency_line(
        emit, [0.5, 0.05, 0.5, 0.05, 0.5, 0.5, 0.5],
    )
    # Filtered: [stalled]*5 → fires.
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]


def test_starved_breaks_stalled_run():
    """Phase change between flagged buckets DOES break the run.
    stalled followed by starved are both noteworthy but not the same
    run."""
    emit, lines = _capture()
    # 3 stalled, 1 starved, 3 stalled → longest run is 3 of stalled.
    # Below threshold.
    _emit_worker_idle_gap_consistency_line(
        emit, [0.5, 0.5, 0.5, 2.0, 0.5, 0.5, 0.5],
    )
    assert lines == []


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_worker_idle_gap_consistency_line(emit, [2.0] * 3, threshold=3)
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_5_run():
    emit, lines = _capture()
    _emit_worker_idle_gap_consistency_line(emit, [2.0] * 5, threshold=10)
    assert lines == []


# ---- Longest of multiple ------------------------------------------


def test_longer_starved_run_beats_shorter_stalled_run():
    """[stalled]*4 + [starved]*7 → only starved passes threshold;
    warning fires for starved."""
    emit, lines = _capture()
    _emit_worker_idle_gap_consistency_line(
        emit, [0.5] * 4 + [2.0] * 7,
    )
    assert "7 consecutive" in lines[0]
    assert "'starved'" in lines[0]


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_worker_idle_gap_consistency_line(emit, [0.5] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_044_attribution():
    emit, lines = _capture()
    _emit_worker_idle_gap_consistency_line(emit, [0.5] * 5)
    assert "iter-044" in lines[0]


# ---- Pattern parity with prior instances --------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_worker_idle_gap_consistency_line(emit, [0.5] * 1000)
    assert len(lines) == 1
    assert "1000 consecutive" in lines[0]
