"""Tests for iter-209 — _emit_eot_overhead_consistency_line.

Tenth instance of the diversity-check pattern. Seventh applied to a
CONTINUOUS metric (after iter-128 sentence-length, iter-140 stt-rtf,
iter-141 tts-rtf, iter-142 llm-tps, iter-143 streaming-overlap,
iter-208 synth-dispatch) — buckets the per-turn ``eot_overhead`` via
``_eot_overhead_bucket`` before scanning. Detects 5+ consecutive
turns that landed in the "slow" or "very_slow" bucket, surfacing the
case where the recording loop itself — not the silence_duration knob
— is the recurring cause of "the agent feels slow" (iter-065's
"trailing-silence wall", metric 1.2).

Like iter-140/141/208 and UNLIKE iter-142/143, the fine bucket is a
LOW value (small overhead) — overhead is smaller-is-better, so the
boundaries are NOT inverted.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_eot_overhead_consistency_line,
    _eot_overhead_bucket,
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
    """0s = no measurable overhead (wait fully explained by
    silence_duration / no-transcription turn) → empty bucket
    (filtered by the consumer)."""
    assert _eot_overhead_bucket(0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty. Shouldn't happen
    (eot_overhead is clamped to max(0, ...)) but the fallback is
    cheap."""
    assert _eot_overhead_bucket(-0.5) == ""


def test_bucket_fast_boundary():
    """< 0.10s → fast (the desired state — the knob is the right
    lever). 0.0999 is the upper edge."""
    assert _eot_overhead_bucket(0.02) == "fast"
    assert _eot_overhead_bucket(0.0999) == "fast"


def test_bucket_slow_boundary():
    """0.10-0.25s inclusive → slow."""
    assert _eot_overhead_bucket(0.10) == "slow"
    assert _eot_overhead_bucket(0.25) == "slow"


def test_bucket_very_slow_boundary():
    """> 0.25s → very_slow."""
    assert _eot_overhead_bucket(0.251) == "very_slow"
    assert _eot_overhead_bucket(2.0) == "very_slow"


def test_bucket_handles_floats():
    """eot_overhead is a float — bucket must handle fine-grained
    values around the boundaries."""
    assert _eot_overhead_bucket(0.0999) == "fast"
    assert _eot_overhead_bucket(0.1001) == "slow"
    assert _eot_overhead_bucket(0.25) == "slow"
    assert _eot_overhead_bucket(0.2501) == "very_slow"


# ---- Empty / no-measurement sessions ---------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_eot_overhead_consistency_line(emit, [])
    assert lines == []


def test_all_zero_emit_nothing():
    """All turns had no measurable overhead (0s) → no warning."""
    emit, lines = _capture()
    _emit_eot_overhead_consistency_line(emit, [0.0] * 10)
    assert lines == []


# ---- "fast" excluded -------------------------------------------------


def test_long_fast_run_does_not_fire():
    """A 10-turn run of negligible overhead is the desired state
    (knob is the right lever) — never flagged."""
    emit, lines = _capture()
    _emit_eot_overhead_consistency_line(emit, [0.02] * 10)
    assert lines == []


def test_alternating_fast_and_slow_only_slow_counts():
    """[0.02, 0.15, 0.02, 0.15, ...] → after filtering, [slow]
    runs of 1. Below threshold → silent."""
    emit, lines = _capture()
    _emit_eot_overhead_consistency_line(
        emit, [0.02, 0.15, 0.02, 0.15, 0.02, 0.15],
    )
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_slow_in_a_row_fires():
    """Default threshold = 5."""
    emit, lines = _capture()
    _emit_eot_overhead_consistency_line(emit, [0.15] * 5)
    assert len(lines) == 1
    assert "EoT overhead" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'slow'" in lines[0]
    assert "tuning silence_duration won't recover it" in lines[0]
    assert "iter-065" in lines[0]


def test_six_very_slow_in_a_row_fires():
    emit, lines = _capture()
    _emit_eot_overhead_consistency_line(emit, [0.4] * 6)
    assert len(lines) == 1
    assert "6 consecutive" in lines[0]
    assert "'very_slow'" in lines[0]
    assert "the recording loop dominates the end-of-turn wait" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold not met."""
    emit, lines = _capture()
    _emit_eot_overhead_consistency_line(emit, [0.15] * 4)
    assert lines == []


# ---- Filter behavior (fast interleavings) ----------------------


def test_fast_between_slow_doesnt_break_run():
    """Same precedent as iter-126/128/140/208: filter the
    uninteresting bucket out before scanning. A 'fast' interleaving
    doesn't break a slow run."""
    emit, lines = _capture()
    # slow, fast, slow, fast, slow, slow, slow
    _emit_eot_overhead_consistency_line(
        emit, [0.15, 0.02, 0.15, 0.02, 0.15, 0.15, 0.15],
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
    _emit_eot_overhead_consistency_line(
        emit, [0.15, 0.15, 0.15, 0.4, 0.15, 0.15, 0.15],
    )
    assert lines == []


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_eot_overhead_consistency_line(emit, [0.4] * 3, threshold=3)
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_5_run():
    emit, lines = _capture()
    _emit_eot_overhead_consistency_line(emit, [0.4] * 5, threshold=10)
    assert lines == []


# ---- Longest of multiple ------------------------------------------


def test_longer_very_slow_run_beats_shorter_slow_run():
    """[slow]*4 + [very_slow]*7 → only very_slow passes threshold;
    warning fires for very_slow."""
    emit, lines = _capture()
    _emit_eot_overhead_consistency_line(emit, [0.15] * 4 + [0.4] * 7)
    assert "7 consecutive" in lines[0]
    assert "'very_slow'" in lines[0]


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_eot_overhead_consistency_line(emit, [0.15] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_065_attribution():
    emit, lines = _capture()
    _emit_eot_overhead_consistency_line(emit, [0.15] * 5)
    assert "iter-065" in lines[0]


# ---- Pattern parity with iter-114/.../208 -------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_eot_overhead_consistency_line(emit, [0.15] * 1000)
    assert "1000 consecutive" in lines[0]
