"""Tests for iter-128 — _emit_sentence_length_consistency_line.

Fourth instance of the diversity-check pattern. First applied
to a CONTINUOUS metric — buckets it via
``_sentence_length_bucket`` before scanning. Detects 5+
consecutive turns that landed in the "very_short" or "long"
bucket, suggesting a splitter-tuning issue.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_sentence_length_consistency_line,
    _sentence_length_bucket,
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
    """0 chars = no sentences this turn → empty bucket
    (filtered by the consumer)."""
    assert _sentence_length_bucket(0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty (treated as 'no
    sentences'). Shouldn't happen in practice but a defensive
    fallback is cheap."""
    assert _sentence_length_bucket(-5) == ""


def test_bucket_very_short_boundary():
    """< 15 → very_short. Exactly 14 is the upper edge."""
    assert _sentence_length_bucket(1) == "very_short"
    assert _sentence_length_bucket(14) == "very_short"


def test_bucket_short_boundary():
    """15-29 → short."""
    assert _sentence_length_bucket(15) == "short"
    assert _sentence_length_bucket(29) == "short"


def test_bucket_medium_boundary():
    """30-59 → medium (the desired state)."""
    assert _sentence_length_bucket(30) == "medium"
    assert _sentence_length_bucket(59) == "medium"


def test_bucket_long_boundary():
    """≥ 60 → long."""
    assert _sentence_length_bucket(60) == "long"
    assert _sentence_length_bucket(150) == "long"


def test_bucket_handles_floats():
    """mean_sentence_chars is a float — bucket must handle that."""
    assert _sentence_length_bucket(14.9) == "very_short"
    assert _sentence_length_bucket(15.0) == "short"
    assert _sentence_length_bucket(29.999) == "short"
    assert _sentence_length_bucket(30.001) == "medium"


# ---- Empty / no-sentences sessions -----------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_sentence_length_consistency_line(emit, [])
    assert lines == []


def test_all_zero_chars_emit_nothing():
    """All turns had no sentences (0 chars) → no warning."""
    emit, lines = _capture()
    _emit_sentence_length_consistency_line(emit, [0.0] * 10)
    assert lines == []


# ---- "Medium" and "short" excluded ----------------------------------


def test_long_medium_run_does_not_fire():
    """A 10-turn run of medium sentences is the desired state —
    never flagged."""
    emit, lines = _capture()
    _emit_sentence_length_consistency_line(emit, [40.0] * 10)
    assert lines == []


def test_long_short_run_does_not_fire():
    """Short sentences (15-30 chars) aren't problematic; only
    very_short and long warrant warnings."""
    emit, lines = _capture()
    _emit_sentence_length_consistency_line(emit, [25.0] * 10)
    assert lines == []


def test_alternating_medium_and_long_only_long_counts():
    """[40, 80, 40, 80, ...] → after filtering, [long] runs of
    1. Below threshold → silent."""
    emit, lines = _capture()
    _emit_sentence_length_consistency_line(
        emit, [40.0, 80.0, 40.0, 80.0, 40.0, 80.0],
    )
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_very_short_in_a_row_fires():
    """Default threshold = 5."""
    emit, lines = _capture()
    _emit_sentence_length_consistency_line(emit, [10.0] * 5)
    assert len(lines) == 1
    assert "Sentence length" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'very_short'" in lines[0]
    assert "splitter may be over-aggressive" in lines[0]
    assert "iter-095" in lines[0]


def test_six_long_in_a_row_fires():
    emit, lines = _capture()
    _emit_sentence_length_consistency_line(emit, [80.0] * 6)
    assert len(lines) == 1
    assert "6 consecutive" in lines[0]
    assert "'long'" in lines[0]
    assert "splitter may be too lax" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold not met."""
    emit, lines = _capture()
    _emit_sentence_length_consistency_line(emit, [10.0] * 4)
    assert lines == []


# ---- Filter behavior (medium/short interleavings) ------------------


def test_medium_between_very_short_doesnt_break_run():
    """Same precedent as iter-126: filter the uninteresting
    bucket out before scanning. A 'medium' interleaving
    doesn't break a very_short run."""
    emit, lines = _capture()
    # very_short, medium, very_short, medium, very_short, very_short, very_short
    _emit_sentence_length_consistency_line(
        emit,
        [10.0, 40.0, 10.0, 40.0, 10.0, 10.0, 10.0],
    )
    # Filtered: [very_short]*5 → fires.
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]


def test_short_between_long_doesnt_break_run():
    """'short' is also filtered out — a short between two long
    runs doesn't break the long run."""
    emit, lines = _capture()
    _emit_sentence_length_consistency_line(
        emit, [80.0, 25.0, 80.0, 80.0, 80.0, 80.0],
    )
    # Filtered: [long]*5 → fires.
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]
    assert "'long'" in lines[0]


def test_long_breaks_very_short_run():
    """Phase change between flagged buckets DOES break the
    run. very_short followed by long means the splitter went
    from over-aggressive to too-lax — both noteworthy but
    not the same run."""
    emit, lines = _capture()
    # 3 very_short, 1 long, 3 very_short → longest run is 3 of
    # very_short. Below threshold.
    _emit_sentence_length_consistency_line(
        emit, [10.0, 10.0, 10.0, 80.0, 10.0, 10.0, 10.0],
    )
    assert lines == []


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_sentence_length_consistency_line(
        emit, [80.0] * 3, threshold=3,
    )
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_5_run():
    emit, lines = _capture()
    _emit_sentence_length_consistency_line(
        emit, [80.0] * 5, threshold=10,
    )
    assert lines == []


# ---- Longest of multiple ------------------------------------------


def test_longer_long_run_beats_shorter_very_short_run():
    """[v_s]*4 + [long]*7 → only long passes threshold; warning
    fires for long."""
    emit, lines = _capture()
    _emit_sentence_length_consistency_line(
        emit, [10.0] * 4 + [80.0] * 7,
    )
    assert "7 consecutive" in lines[0]
    assert "'long'" in lines[0]


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_sentence_length_consistency_line(emit, [10.0] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_095_attribution():
    emit, lines = _capture()
    _emit_sentence_length_consistency_line(emit, [10.0] * 5)
    assert "iter-095" in lines[0]


# ---- Pattern parity with iter-114/115/120/126 -------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_sentence_length_consistency_line(emit, [10.0] * 1000)
    assert "1000 consecutive" in lines[0]
