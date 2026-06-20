"""Tests for iter-325 — _emit_preempted_words_consistency_line.

Fifteenth instance of the session-summary diversity-check pattern, and
the SECOND barge-CONDITIONAL one (after iter-120 barge-phase) — buckets
the per-turn ``preempted_words`` (iter-080 — words the LLM generated but
the user never heard because a barge cut the stream mid-content) via
``_preempted_words_bucket`` before scanning. Detects 4+ consecutive barge
turns that landed in the "minor" (1-10 words) or "heavy" (> 10 words)
bucket, surfacing that barges are systematically throwing away generated
speech — a verbose bot the user keeps cutting off (heavy) or a habitual
mid-sentence interrupter (minor).

Unlike the continuous bucketers, this is a COUNT metric and 0 is the fine
state (no barge, or a clean inter-sentence cut), filtered before the scan
exactly like iter-114's filler-count — NOT the empty-string filter of the
continuous instances. The threshold is 4, matching iter-120's
barge-phase sentinel, because pre-emption events are barge-conditional and
already rare. It is NOT inverted: more pre-empted words is strictly worse.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_preempted_words_consistency_line,
    _preempted_words_bucket,
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
    """0 = no barge, or a clean inter-sentence cut (no words lost) →
    empty bucket (filtered by the consumer)."""
    assert _preempted_words_bucket(0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty. iter-080 clamps with
    max(0, ...) so this shouldn't happen, but the fallback is cheap."""
    assert _preempted_words_bucket(-3) == ""


def test_bucket_minor_boundary():
    """1-10 words → minor. Both edges inclusive; 10 is the per-turn
    yellow threshold (iter-080 colors > 10 yellow)."""
    assert _preempted_words_bucket(1) == "minor"
    assert _preempted_words_bucket(10) == "minor"


def test_bucket_heavy_boundary():
    """> 10 words → heavy (the per-turn yellow state — > ~5s lost)."""
    assert _preempted_words_bucket(11) == "heavy"
    assert _preempted_words_bucket(100) == "heavy"


# ---- Not inverted: 0 is the fine state, high is the problem ----------


def test_direction_zero_is_fine_high_is_bad():
    """Sanity on direction: this is NOT inverted (contrast iter-324).
    0 filters out (good — no loss); larger counts bucket to the
    problematic ends."""
    assert _preempted_words_bucket(0) == ""
    assert _preempted_words_bucket(5) == "minor"
    assert _preempted_words_bucket(50) == "heavy"


# ---- Empty / no-loss sessions ----------------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_preempted_words_consistency_line(emit, [])
    assert lines == []


def test_all_zero_emits_nothing():
    """No barge ever lost words (every turn 0) → no warning. The
    common no-barge / clean-cut session."""
    emit, lines = _capture()
    _emit_preempted_words_consistency_line(emit, [0] * 12)
    assert lines == []


# ---- At/above threshold (warning fires) ------------------------------


def test_four_heavy_in_a_row_fires():
    """Default threshold = 4 (shared with iter-120 barge-phase)."""
    emit, lines = _capture()
    _emit_preempted_words_consistency_line(emit, [25] * 4)
    assert len(lines) == 1
    assert "Pre-empted run" in lines[0]
    assert "4 consecutive" in lines[0]
    assert "'heavy'" in lines[0]
    assert "barge turns" in lines[0]
    assert "verbose" in lines[0]
    assert "iter-080" in lines[0]


def test_four_minor_in_a_row_fires():
    emit, lines = _capture()
    _emit_preempted_words_consistency_line(emit, [3] * 4)
    assert len(lines) == 1
    assert "4 consecutive" in lines[0]
    assert "'minor'" in lines[0]
    assert "mid-sentence" in lines[0]


def test_three_in_a_row_below_default_threshold():
    """3 in a row → default threshold (4) not met."""
    emit, lines = _capture()
    _emit_preempted_words_consistency_line(emit, [25] * 3)
    assert lines == []


# ---- Filter behavior (zero interleavings) ----------------------------


def test_zero_between_heavy_doesnt_break_run():
    """A 0 (no-barge / clean-cut turn) is filtered out before the
    scan, so it does NOT break a run of flagged barge turns on either
    side — mirrors iter-114's filler-count filter."""
    emit, lines = _capture()
    # heavy, 0, heavy, 0, heavy, heavy → filtered to [heavy]*4 → fires.
    _emit_preempted_words_consistency_line(emit, [25, 0, 25, 0, 25, 25])
    assert len(lines) == 1
    assert "4 consecutive" in lines[0]
    assert "'heavy'" in lines[0]


def test_minor_breaks_heavy_run():
    """Phase change between the two flagged buckets DOES break the run.
    They never merge — minor and heavy localize different causes."""
    emit, lines = _capture()
    # 3 heavy, 1 minor, 3 heavy → longest run is 3 of heavy. Below
    # threshold of 4.
    _emit_preempted_words_consistency_line(
        emit, [25, 25, 25, 5, 25, 25, 25],
    )
    assert lines == []


def test_few_scattered_heavy_barges_below_threshold():
    """Clean turns (0) are filtered, so they don't separate runs — but
    a SMALL number of heavy barges still can't reach the threshold of 4.
    3 heavy barges scattered among clean turns filter to [heavy]*3,
    below threshold → silent. (Once 4+ word-losing barges occur, even
    with clean turns between, the run DOES fire — see
    test_zero_between_heavy_doesnt_break_run; that is the intended
    "sustained loss" signal, not noise.)"""
    emit, lines = _capture()
    _emit_preempted_words_consistency_line(
        emit, [25, 0, 0, 30, 0, 0, 0, 40],
    )
    assert lines == []


# ---- Custom threshold ------------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_preempted_words_consistency_line(emit, [25] * 3, threshold=3)
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_run():
    emit, lines = _capture()
    _emit_preempted_words_consistency_line(emit, [25] * 4, threshold=10)
    assert lines == []


# ---- Longest of multiple ---------------------------------------------


def test_longer_minor_run_beats_shorter_heavy_run():
    """[heavy]*3 + [minor]*5 → only the minor run passes threshold;
    warning fires for minor."""
    emit, lines = _capture()
    _emit_preempted_words_consistency_line(emit, [25] * 3 + [5] * 5)
    assert "5 consecutive" in lines[0]
    assert "'minor'" in lines[0]


# ---- Output formatting -----------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_preempted_words_consistency_line(emit, [25] * 4)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_080_attribution():
    emit, lines = _capture()
    _emit_preempted_words_consistency_line(emit, [25] * 4)
    assert "iter-080" in lines[0]
    assert "preempted_words" in lines[0]


def test_heavy_warning_points_at_verbosity_fix():
    """The heavy warning must name the actionable knob (shorten
    replies), not just describe the symptom."""
    emit, lines = _capture()
    _emit_preempted_words_consistency_line(emit, [50] * 4)
    assert "shorten replies" in lines[0]


def test_says_barge_turns_not_just_turns():
    """The unit is barge turns, not all turns — the word 'barge' must
    appear so the operator knows these are interruption events, not a
    per-turn count over the whole session."""
    emit, lines = _capture()
    _emit_preempted_words_consistency_line(emit, [25] * 4)
    assert "barge turns" in lines[0]


# ---- Distinctness from iter-120 (barge-PHASE) ------------------------


def test_distinct_from_barge_phase_sentinel():
    """iter-120 counts barge PHASES (categorical: when the user barged);
    iter-325 counts pre-empted WORDS (quantitative: how much speech was
    lost). They share the threshold of 4 and the barge-conditional
    framing, but answer different questions. This pins that iter-325's
    line is unambiguously about lost-word COUNT, not phase."""
    emit, lines = _capture()
    _emit_preempted_words_consistency_line(emit, [25] * 4)
    assert "preempted_words" in lines[0]
    # Must NOT claim to be about barge phase (llm_stream / playback).
    assert "llm_stream" not in lines[0]
    assert "playback" not in lines[0]


# ---- Pattern parity with the prior instances -------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_preempted_words_consistency_line(emit, [25] * 1000)
    assert "1000 consecutive" in lines[0]
