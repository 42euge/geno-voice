"""Tests for iter-140 — _emit_stt_rtf_consistency_line.

Fifth instance of the diversity-check pattern. Second applied
to a CONTINUOUS metric (after iter-128 sentence-length) — buckets
the per-turn ``stt_rtf`` via ``_stt_rtf_bucket`` before scanning.
Detects 5+ consecutive turns that landed in the "slow" or
"very_slow" bucket, suggesting the STT engine/model is too heavy
for the host hardware.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_stt_rtf_consistency_line,
    _stt_rtf_bucket,
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
    """0 RTF = unmeasured / false-trigger turn → empty bucket
    (filtered by the consumer)."""
    assert _stt_rtf_bucket(0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty. Shouldn't happen in
    practice but a defensive fallback is cheap."""
    assert _stt_rtf_bucket(-1.0) == ""


def test_bucket_realtime_boundary():
    """< 1.0 → realtime (the desired state). 0.999 is the upper
    edge."""
    assert _stt_rtf_bucket(0.1) == "realtime"
    assert _stt_rtf_bucket(0.999) == "realtime"


def test_bucket_slow_boundary():
    """1.0-2.0 inclusive → slow."""
    assert _stt_rtf_bucket(1.0) == "slow"
    assert _stt_rtf_bucket(2.0) == "slow"


def test_bucket_very_slow_boundary():
    """> 2.0 → very_slow."""
    assert _stt_rtf_bucket(2.001) == "very_slow"
    assert _stt_rtf_bucket(10.0) == "very_slow"


def test_bucket_handles_floats():
    """stt_rtf is a float — bucket must handle fine-grained
    values around the boundaries."""
    assert _stt_rtf_bucket(0.9999) == "realtime"
    assert _stt_rtf_bucket(1.0001) == "slow"
    assert _stt_rtf_bucket(2.0) == "slow"
    assert _stt_rtf_bucket(2.0001) == "very_slow"


# ---- Empty / no-measurement sessions ---------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_stt_rtf_consistency_line(emit, [])
    assert lines == []


def test_all_zero_rtf_emit_nothing():
    """All turns had no measurable STT (0 RTF) → no warning."""
    emit, lines = _capture()
    _emit_stt_rtf_consistency_line(emit, [0.0] * 10)
    assert lines == []


# ---- "realtime" excluded ---------------------------------------------


def test_long_realtime_run_does_not_fire():
    """A 10-turn run of fast (sub-realtime) STT is the desired
    state — never flagged."""
    emit, lines = _capture()
    _emit_stt_rtf_consistency_line(emit, [0.2] * 10)
    assert lines == []


def test_alternating_realtime_and_slow_only_slow_counts():
    """[0.2, 1.5, 0.2, 1.5, ...] → after filtering, [slow] runs
    of 1. Below threshold → silent."""
    emit, lines = _capture()
    _emit_stt_rtf_consistency_line(
        emit, [0.2, 1.5, 0.2, 1.5, 0.2, 1.5],
    )
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_slow_in_a_row_fires():
    """Default threshold = 5."""
    emit, lines = _capture()
    _emit_stt_rtf_consistency_line(emit, [1.5] * 5)
    assert len(lines) == 1
    assert "STT speed" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'slow'" in lines[0]
    assert "smaller model or streaming STT" in lines[0]
    assert "iter-049" in lines[0]


def test_six_very_slow_in_a_row_fires():
    emit, lines = _capture()
    _emit_stt_rtf_consistency_line(emit, [3.0] * 6)
    assert len(lines) == 1
    assert "6 consecutive" in lines[0]
    assert "'very_slow'" in lines[0]
    assert "badly mismatched to the hardware" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold not met."""
    emit, lines = _capture()
    _emit_stt_rtf_consistency_line(emit, [1.5] * 4)
    assert lines == []


# ---- Filter behavior (realtime interleavings) ----------------------


def test_realtime_between_slow_doesnt_break_run():
    """Same precedent as iter-126/128: filter the uninteresting
    bucket out before scanning. A 'realtime' interleaving doesn't
    break a slow run."""
    emit, lines = _capture()
    # slow, realtime, slow, realtime, slow, slow, slow
    _emit_stt_rtf_consistency_line(
        emit, [1.5, 0.2, 1.5, 0.2, 1.5, 1.5, 1.5],
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
    _emit_stt_rtf_consistency_line(
        emit, [1.5, 1.5, 1.5, 3.0, 1.5, 1.5, 1.5],
    )
    assert lines == []


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_stt_rtf_consistency_line(emit, [3.0] * 3, threshold=3)
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_5_run():
    emit, lines = _capture()
    _emit_stt_rtf_consistency_line(emit, [3.0] * 5, threshold=10)
    assert lines == []


# ---- Longest of multiple ------------------------------------------


def test_longer_very_slow_run_beats_shorter_slow_run():
    """[slow]*4 + [very_slow]*7 → only very_slow passes threshold;
    warning fires for very_slow."""
    emit, lines = _capture()
    _emit_stt_rtf_consistency_line(emit, [1.5] * 4 + [3.0] * 7)
    assert "7 consecutive" in lines[0]
    assert "'very_slow'" in lines[0]


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_stt_rtf_consistency_line(emit, [1.5] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_049_attribution():
    emit, lines = _capture()
    _emit_stt_rtf_consistency_line(emit, [1.5] * 5)
    assert "iter-049" in lines[0]


# ---- Pattern parity with iter-114/115/120/126/128 ---------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_stt_rtf_consistency_line(emit, [1.5] * 1000)
    assert "1000 consecutive" in lines[0]
