"""Tests for iter-324 — _emit_first_synth_overlap_consistency_line.

Fourteenth instance of the session-summary diversity-check pattern, and
the TENTH applied to a CONTINUOUS metric (after iter-128 sentence-length,
iter-140 stt-rtf, iter-141 tts-rtf, iter-142 llm-tps, iter-143
streaming-overlap, iter-208 synth-dispatch, iter-209 eot-overhead,
iter-210 bot-wpm, iter-323 user-wpm) — buckets the per-turn
``first_synth_overlap_seconds`` (iter-073) via
``_first_synth_overlap_bucket`` before scanning. Detects 5+ consecutive
turns that landed in the "low" or "very_low" bucket, surfacing that the
iter-008 streaming-overlap design shaved little or nothing off TTFS
specifically (the first sentence shipped near-sequentially).

Like iter-142 ``llm_tps``, iter-143 ``streaming_overlap_ratio`` and
iter-225 ``sentence_split_coverage`` — and UNLIKE the RTF instances —
first-synth save is bigger-is-better: the fine bucket ("high") is a HIGH
value, so the bucket boundaries invert. This is the FOURTH
inverted-direction continuous bucketer, so the boundary tests below probe
the high end as the "good" state.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_first_synth_overlap_consistency_line,
    _first_synth_overlap_bucket,
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
    """0 = no measurable first-synth overlap this turn (ambiguous:
    sequential or no-audio) → empty bucket (filtered by the consumer)."""
    assert _first_synth_overlap_bucket(0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty. Shouldn't happen (iter-073
    clamps with max(0.0, ...)) but a defensive fallback is cheap."""
    assert _first_synth_overlap_bucket(-0.05) == ""


def test_bucket_high_boundary():
    """>= 0.10s (100ms) → high (the desired, well-masked state).
    0.10 is the lower edge of high and matches the per-turn green
    display threshold."""
    assert _first_synth_overlap_bucket(0.10) == "high"
    assert _first_synth_overlap_bucket(1.0) == "high"


def test_bucket_low_boundary():
    """0.02-0.10s → low. Inclusive lower edge, exclusive upper."""
    assert _first_synth_overlap_bucket(0.02) == "low"
    assert _first_synth_overlap_bucket(0.0999) == "low"


def test_bucket_very_low_boundary():
    """< 0.02s (but > 0) → very_low."""
    assert _first_synth_overlap_bucket(0.0199) == "very_low"
    assert _first_synth_overlap_bucket(0.001) == "very_low"


def test_bucket_handles_floats():
    """first_synth_overlap_seconds is a float — bucket must handle
    fine-grained values around the (inverted) boundaries."""
    assert _first_synth_overlap_bucket(0.0999) == "low"
    assert _first_synth_overlap_bucket(0.1001) == "high"
    assert _first_synth_overlap_bucket(0.02) == "low"
    assert _first_synth_overlap_bucket(0.0199) == "very_low"


# ---- Distinctness from iter-143 (whole-stream ratio) -----------------


def test_distinct_from_streaming_overlap_bucket():
    """iter-324 buckets ABSOLUTE seconds; iter-143 buckets a [0,1]
    RATIO. The two must not be confused — feeding the same numeric
    value can land in different buckets. e.g. 0.30 is a 'low'
    whole-stream ratio for iter-143 but a 'high' (>=0.10s) first-synth
    save here. Pinning this guards against someone collapsing the two
    sentinels into one."""
    assert _streaming_overlap_bucket(0.30) == "low"
    assert _first_synth_overlap_bucket(0.30) == "high"


# ---- Empty / no-measurement sessions ---------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_first_synth_overlap_consistency_line(emit, [])
    assert lines == []


def test_all_zero_overlap_emit_nothing():
    """All turns had no measurable first-synth overlap (0) → no
    warning."""
    emit, lines = _capture()
    _emit_first_synth_overlap_consistency_line(emit, [0.0] * 10)
    assert lines == []


# ---- "high" excluded (the inverted fine state) -----------------------


def test_long_high_run_does_not_fire():
    """A 10-turn run of well-masked first synth is the desired state —
    never flagged. The fine bucket here is HIGH."""
    emit, lines = _capture()
    _emit_first_synth_overlap_consistency_line(emit, [0.25] * 10)
    assert lines == []


def test_alternating_high_and_low_only_low_counts():
    """[0.25, 0.05, 0.25, 0.05, ...] → after filtering, [low] runs of
    1. Below threshold → silent."""
    emit, lines = _capture()
    _emit_first_synth_overlap_consistency_line(
        emit, [0.25, 0.05, 0.25, 0.05, 0.25, 0.05],
    )
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_low_in_a_row_fires():
    """Default threshold = 5."""
    emit, lines = _capture()
    _emit_first_synth_overlap_consistency_line(emit, [0.05] * 5)
    assert len(lines) == 1
    assert "1st-synth save" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'low'" in lines[0]
    assert "tail of the first synth" in lines[0]
    assert "iter-073" in lines[0]


def test_six_very_low_in_a_row_fires():
    emit, lines = _capture()
    _emit_first_synth_overlap_consistency_line(emit, [0.005] * 6)
    assert len(lines) == 1
    assert "6 consecutive" in lines[0]
    assert "'very_low'" in lines[0]
    assert "sequentially after the LLM stream" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold not met."""
    emit, lines = _capture()
    _emit_first_synth_overlap_consistency_line(emit, [0.05] * 4)
    assert lines == []


# ---- Filter behavior (high interleavings) ----------------------------


def test_high_between_low_doesnt_break_run():
    """Same precedent as iter-126/128/140/141/142/143: filter the
    uninteresting bucket out before scanning. A 'high' interleaving
    doesn't break a low run."""
    emit, lines = _capture()
    # low, high, low, high, low, low, low
    _emit_first_synth_overlap_consistency_line(
        emit, [0.05, 0.25, 0.05, 0.25, 0.05, 0.05, 0.05],
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
    _emit_first_synth_overlap_consistency_line(
        emit, [0.05, 0.05, 0.05, 0.005, 0.05, 0.05, 0.05],
    )
    assert lines == []


def test_zero_breaks_run():
    """A zero (no-measurement turn) is filtered out, so it does NOT
    break a run of flagged turns on either side — mirrors the
    empty-bucket semantics of every prior continuous instance."""
    emit, lines = _capture()
    # 3 low, 1 zero (dropped), 2 low → filtered to [low]*5 → fires.
    _emit_first_synth_overlap_consistency_line(
        emit, [0.05, 0.05, 0.05, 0.0, 0.05, 0.05],
    )
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]


# ---- Custom threshold ------------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_first_synth_overlap_consistency_line(
        emit, [0.005] * 3, threshold=3,
    )
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_5_run():
    emit, lines = _capture()
    _emit_first_synth_overlap_consistency_line(
        emit, [0.005] * 5, threshold=10,
    )
    assert lines == []


# ---- Longest of multiple ---------------------------------------------


def test_longer_very_low_run_beats_shorter_low_run():
    """[low]*4 + [very_low]*7 → only very_low passes threshold;
    warning fires for very_low."""
    emit, lines = _capture()
    _emit_first_synth_overlap_consistency_line(
        emit, [0.05] * 4 + [0.005] * 7,
    )
    assert "7 consecutive" in lines[0]
    assert "'very_low'" in lines[0]


# ---- Output formatting -----------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_first_synth_overlap_consistency_line(emit, [0.05] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_073_attribution():
    emit, lines = _capture()
    _emit_first_synth_overlap_consistency_line(emit, [0.05] * 5)
    assert "iter-073" in lines[0]
    assert "first_synth_overlap_seconds" in lines[0]


def test_warning_mentions_ttfs_actionability():
    """The warning must point at the actionable knob (first-sentence
    latency / synth dispatch), not just describe the symptom."""
    emit, lines = _capture()
    _emit_first_synth_overlap_consistency_line(emit, [0.005] * 5)
    assert "first-sentence latency" in lines[0]


# ---- Pattern parity with the prior continuous instances --------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_first_synth_overlap_consistency_line(emit, [0.05] * 1000)
    assert "1000 consecutive" in lines[0]
