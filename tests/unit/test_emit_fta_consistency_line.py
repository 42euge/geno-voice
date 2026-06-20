"""Tests for iter-311 — _emit_fta_consistency_line.

TWENTY-THIRD instance of the diversity-check pattern, applied to a
CONTINUOUS (float-valued) metric: buckets the per-turn
``first_token_to_audio`` (iter-083's FT-A — the gap from the LLM's first
token landing at the splitter to the worker playing its first audio
chunk) via ``_fta_bucket`` before scanning. Detects 5+ consecutive turns
that landed in the "slow" (0.15-0.35s) or "very_slow" (>0.35s) bucket —
the signal that sentence-split + TTS is the persistent post-LLM
bottleneck (the bot has tokens but can't speak yet), eroding the
sub-500ms TTFS budget.

Like iter-140/141 (RTF), iter-208 (synth-dispatch), iter-209
(eot-overhead), iter-212 (ttfs), iter-224 (preview-divergence), iter-226
(worker-idle-gap), iter-307 (synth-backlog), iter-308 (cancel-close),
iter-309 (speaker-open), iter-310 (mic-stale) — and UNLIKE iter-142/143/225
— the problematic end is a LARGE value (small FT-A is best), so the
boundaries are NOT inverted: the "snappy" fine bucket is a LOW value and
gets filtered alongside the empty (no-measurement) bucket.

Threshold is 5 (the general-signal default, NOT iter-120/308/310's
event-gated 4): FT-A is measured on essentially every audio-producing
turn, a high-frequency signal where natural per-turn variation is normal.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_fta_consistency_line,
    _fta_bucket,
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
    """0 = no measured FT-A this turn (errored before the LLM produced a
    token or before any audio played) → empty bucket (the no-event
    state, filtered by the consumer)."""
    assert _fta_bucket(0.0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty (clock-skew guard)."""
    assert _fta_bucket(-0.5) == ""


def test_bucket_snappy_boundary():
    """Just above 0 up to (not incl.) 0.15s → snappy (the bot speaks
    almost as soon as it has tokens — the desired state, filtered)."""
    assert _fta_bucket(0.001) == "snappy"
    assert _fta_bucket(0.10) == "snappy"
    assert _fta_bucket(0.1499) == "snappy"


def test_bucket_slow_boundary():
    """0.15s up to 0.35s inclusive → slow."""
    assert _fta_bucket(0.15) == "slow"
    assert _fta_bucket(0.25) == "slow"
    assert _fta_bucket(0.35) == "slow"


def test_bucket_very_slow_boundary():
    """> 0.35s → very_slow."""
    assert _fta_bucket(0.3501) == "very_slow"
    assert _fta_bucket(1.0) == "very_slow"


# ---- Empty / no-event sessions ---------------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_fta_consistency_line(emit, [])
    assert lines == []


def test_all_zero_emit_nothing():
    """A session where FT-A was never measured → no warning."""
    emit, lines = _capture()
    _emit_fta_consistency_line(emit, [0.0] * 10)
    assert lines == []


def test_all_snappy_emit_nothing():
    """A healthy session (every turn speaks promptly after first token)
    → the snappy bucket is filtered, nothing fires."""
    emit, lines = _capture()
    _emit_fta_consistency_line(emit, [0.05] * 10)
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_slow_in_a_row_fires():
    """Default threshold = 5 (the high-frequency general default)."""
    emit, lines = _capture()
    _emit_fta_consistency_line(emit, [0.25] * 5)
    assert len(lines) == 1
    assert "FT-A" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'slow'" in lines[0]
    assert "iter-083" in lines[0]


def test_five_very_slow_in_a_row_fires():
    emit, lines = _capture()
    _emit_fta_consistency_line(emit, [0.5] * 5)
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]
    assert "'very_slow'" in lines[0]
    assert ">350ms" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold (5) not met."""
    emit, lines = _capture()
    _emit_fta_consistency_line(emit, [0.5] * 4)
    assert lines == []


# ---- Filter behavior (snappy/empty interleavings) -------------------


def test_snappy_between_slow_doesnt_break_run():
    """The snappy fine-bucket is filtered before scanning, so a fast turn
    interleaved between slow turns doesn't break the slow run."""
    emit, lines = _capture()
    # slow x5 with snappy interleaved → filtered: [slow]*5 → fires.
    _emit_fta_consistency_line(
        emit, [0.25, 0.05, 0.25, 0.25, 0.05, 0.25, 0.25]
    )
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]
    assert "'slow'" in lines[0]


def test_empty_between_slow_doesnt_break_run():
    """A no-measurement (0.0) turn is also filtered and doesn't break a
    slow run."""
    emit, lines = _capture()
    _emit_fta_consistency_line(
        emit, [0.25, 0.0, 0.25, 0.25, 0.0, 0.25, 0.25]
    )
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]


def test_very_slow_breaks_slow_run():
    """Phase change between flagged buckets DOES break the run. slow then
    very_slow are both noteworthy but not the same run."""
    emit, lines = _capture()
    # 4 slow, 1 very_slow, 4 slow → longest run is 4 of slow. Below 5.
    _emit_fta_consistency_line(
        emit, [0.25, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25]
    )
    assert lines == []


def test_slow_and_very_slow_dont_merge_into_one_run():
    """4 slow + 4 very_slow → no single bucket reaches 5. Distinct
    buckets never merge into a longer combined run."""
    emit, lines = _capture()
    _emit_fta_consistency_line(
        emit, [0.25] * 4 + [0.5] * 4
    )
    assert lines == []


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_fta_consistency_line(emit, [0.5] * 3, threshold=3)
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_run():
    emit, lines = _capture()
    _emit_fta_consistency_line(emit, [0.5] * 5, threshold=10)
    assert lines == []


# ---- Longest of multiple ------------------------------------------


def test_longer_very_slow_run_beats_shorter_slow_run():
    """[slow]*4 + [very_slow]*6 → only very_slow passes threshold;
    warning fires for very_slow."""
    emit, lines = _capture()
    _emit_fta_consistency_line(emit, [0.25] * 4 + [0.5] * 6)
    assert "6 consecutive" in lines[0]
    assert "'very_slow'" in lines[0]


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_fta_consistency_line(emit, [0.25] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_083_attribution():
    emit, lines = _capture()
    _emit_fta_consistency_line(emit, [0.25] * 5)
    assert "iter-083" in lines[0]
    assert "first_token_to_audio" in lines[0]


def test_slow_suggestion_distinct_from_very_slow():
    """Per-value suggestion mapping: the slow and very_slow branches
    produce different operator guidance (not one-size-fits-all)."""
    emit_slow, slow_lines = _capture()
    _emit_fta_consistency_line(emit_slow, [0.25] * 5)
    emit_vslow, vslow_lines = _capture()
    _emit_fta_consistency_line(emit_vslow, [0.5] * 5)
    assert slow_lines[0] != vslow_lines[0]
    assert ">350ms" in vslow_lines[0]
    assert ">350ms" not in slow_lines[0]


# ---- Pattern parity with prior instances --------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_fta_consistency_line(emit, [0.25] * 1000)
    assert len(lines) == 1
    assert "1000 consecutive" in lines[0]
