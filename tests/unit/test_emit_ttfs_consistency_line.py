"""Tests for iter-212 — _emit_ttfs_consistency_line.

Thirteenth instance of the diversity-check pattern. Tenth applied to a
CONTINUOUS metric (after iter-128 sentence-length, iter-140 stt-rtf,
iter-141 tts-rtf, iter-142 llm-tps, iter-143 streaming-overlap,
iter-208 synth-dispatch, iter-209 eot-overhead, iter-210 bot-wpm,
iter-211 max-token-gap) — buckets the per-turn ``ttfs`` via
``_ttfs_bucket`` before scanning. Detects 5+ consecutive turns that
landed in the "slow" or "very_slow" bucket, surfacing the case where
the bot was consistently slow to start speaking — the headline latency
metric the VISION optimizes for ("latency is the feature").

Like iter-140/141/208/209/211 and UNLIKE the inverted iter-142/143, the
fine bucket is a LOW value (small TTFS) — TTFS is smaller-is-better, so
the boundaries are NOT inverted. The ``snappy`` boundary (3.0s) is
aligned with the existing per-turn green TTFS display.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_ttfs_consistency_line,
    _ttfs_bucket,
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
    """0s = no audio played that turn (error/barge) → empty bucket
    (filtered by the consumer)."""
    assert _ttfs_bucket(0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty. Shouldn't happen but the
    fallback is cheap."""
    assert _ttfs_bucket(-1.0) == ""


def test_bucket_snappy_boundary():
    """< 3.0s → snappy (the desired state — matches the per-turn green
    display). 2.9999 is the upper edge."""
    assert _ttfs_bucket(0.4) == "snappy"
    assert _ttfs_bucket(2.9999) == "snappy"


def test_bucket_slow_boundary():
    """3.0-6.0s inclusive → slow."""
    assert _ttfs_bucket(3.0) == "slow"
    assert _ttfs_bucket(6.0) == "slow"


def test_bucket_very_slow_boundary():
    """> 6.0s → very_slow."""
    assert _ttfs_bucket(6.001) == "very_slow"
    assert _ttfs_bucket(12.0) == "very_slow"


def test_bucket_handles_floats():
    """ttfs is a float — bucket must handle fine-grained values around
    the boundaries."""
    assert _ttfs_bucket(2.9999) == "snappy"
    assert _ttfs_bucket(3.0001) == "slow"
    assert _ttfs_bucket(6.0) == "slow"
    assert _ttfs_bucket(6.0001) == "very_slow"


# ---- Empty / no-measurement sessions ---------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_ttfs_consistency_line(emit, [])
    assert lines == []


def test_all_zero_emit_nothing():
    """All turns played no audio (0s — error/barge) → no warning."""
    emit, lines = _capture()
    _emit_ttfs_consistency_line(emit, [0.0] * 10)
    assert lines == []


# ---- "snappy" excluded -----------------------------------------------


def test_long_snappy_run_does_not_fire():
    """A 10-turn run of prompt responses is the desired state — never
    flagged."""
    emit, lines = _capture()
    _emit_ttfs_consistency_line(emit, [0.5] * 10)
    assert lines == []


def test_alternating_snappy_and_slow_only_slow_counts():
    """[0.5, 4.0, 0.5, 4.0, ...] → after filtering, [slow] runs of 1.
    Below threshold → silent."""
    emit, lines = _capture()
    _emit_ttfs_consistency_line(
        emit, [0.5, 4.0, 0.5, 4.0, 0.5, 4.0],
    )
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_slow_in_a_row_fires():
    """Default threshold = 5."""
    emit, lines = _capture()
    _emit_ttfs_consistency_line(emit, [4.0] * 5)
    assert len(lines) == 1
    assert "TTFS:" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'slow'" in lines[0]
    assert "consistently slow to start speaking" in lines[0]
    assert "iter-212" in lines[0]


def test_six_very_slow_in_a_row_fires():
    emit, lines = _capture()
    _emit_ttfs_consistency_line(emit, [8.0] * 6)
    assert len(lines) == 1
    assert "6 consecutive" in lines[0]
    assert "'very_slow'" in lines[0]
    assert "wonders whether it heard them" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold not met."""
    emit, lines = _capture()
    _emit_ttfs_consistency_line(emit, [4.0] * 4)
    assert lines == []


# ---- Filter behavior (snappy interleavings) ----------------------


def test_snappy_between_slow_doesnt_break_run():
    """Same precedent as iter-126/128/140/208/209/211: filter the
    uninteresting bucket out before scanning. A 'snappy' interleaving
    doesn't break a slow run."""
    emit, lines = _capture()
    # slow, snappy, slow, snappy, slow, slow, slow
    _emit_ttfs_consistency_line(
        emit, [4.0, 0.5, 4.0, 0.5, 4.0, 4.0, 4.0],
    )
    # Filtered: [slow]*5 → fires.
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]


def test_very_slow_breaks_slow_run():
    """Phase change between flagged buckets DOES break the run.
    slow followed by very_slow are both noteworthy but not the same
    run."""
    emit, lines = _capture()
    # 3 slow, 1 very_slow, 3 slow → longest run is 3 of slow.
    # Below threshold.
    _emit_ttfs_consistency_line(
        emit, [4.0, 4.0, 4.0, 8.0, 4.0, 4.0, 4.0],
    )
    assert lines == []


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_ttfs_consistency_line(emit, [8.0] * 3, threshold=3)
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_5_run():
    emit, lines = _capture()
    _emit_ttfs_consistency_line(emit, [8.0] * 5, threshold=10)
    assert lines == []


# ---- Longest of multiple ------------------------------------------


def test_longer_very_slow_run_beats_shorter_slow_run():
    """[slow]*4 + [very_slow]*7 → only very_slow passes threshold;
    warning fires for very_slow."""
    emit, lines = _capture()
    _emit_ttfs_consistency_line(emit, [4.0] * 4 + [8.0] * 7)
    assert "7 consecutive" in lines[0]
    assert "'very_slow'" in lines[0]


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_ttfs_consistency_line(emit, [4.0] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_212_attribution():
    emit, lines = _capture()
    _emit_ttfs_consistency_line(emit, [4.0] * 5)
    assert "iter-212" in lines[0]


# ---- Pattern parity with iter-114/.../211 -------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_ttfs_consistency_line(emit, [4.0] * 1000)
    assert "1000 consecutive" in lines[0]
