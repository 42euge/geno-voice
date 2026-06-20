"""Tests for iter-308 — _emit_cancel_close_consistency_line.

TWENTIETH instance of the diversity-check pattern, applied to a
CONTINUOUS (float-valued) metric: buckets the per-turn
``llm_cancel_to_close`` (iter-060's barge-teardown latency — the gap
from the barge trigger firing to the LLM HTTP stream actually closing)
via ``_cancel_close_bucket`` before scanning. Detects 4+ consecutive
BARGE turns that landed in the "slow" (0.5-1.0s) or "hung" (>1.0s)
bucket — the signal that tearing down the LLM stream on barge-in was
reliably laggy, so the old reply kept generating server-side past every
interruption.

Like iter-140/141 (RTF), iter-208 (synth-dispatch), iter-209
(eot-overhead), iter-224 (preview-divergence), iter-226 (worker-idle-gap)
and iter-307 (synth-backlog) — and UNLIKE iter-142/143/225 — the fine
bucket is a LOW value (prompt teardown is best), so the boundaries are
NOT inverted: the problematic end is a LARGE latency.

Threshold is 4 (NOT the usual 5), mirroring iter-120's barge-phase
sentinel: cancel-to-close is only measured on barge turns, which are
rarer, so a shorter run is already a strong signal.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _cancel_close_bucket,
    _emit_cancel_close_consistency_line,
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
    """0.0 = no barge teardown measured this turn (the no-barge default)
    → empty bucket (the no-measurement state, filtered by the
    consumer)."""
    assert _cancel_close_bucket(0.0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty. Shouldn't happen (a latency is
    non-negative) but a defensive fallback is cheap."""
    assert _cancel_close_bucket(-0.5) == ""


def test_bucket_prompt_boundary():
    """Just above 0 and up to 500ms → prompt (the socket closed promptly
    — the desired state, and iter-060's dim case)."""
    assert _cancel_close_bucket(0.001) == "prompt"
    assert _cancel_close_bucket(0.25) == "prompt"
    assert _cancel_close_bucket(0.50) == "prompt"


def test_bucket_slow_boundary():
    """Just above 500ms up to 1.0s → slow (iter-060's yellow case starts
    here — teardown noticeably laggy)."""
    assert _cancel_close_bucket(0.5001) == "slow"
    assert _cancel_close_bucket(0.75) == "slow"
    assert _cancel_close_bucket(1.0) == "slow"


def test_bucket_hung_boundary():
    """> 1.0s → hung (the socket is effectively hanging)."""
    assert _cancel_close_bucket(1.0001) == "hung"
    assert _cancel_close_bucket(5.0) == "hung"


# ---- Empty / no-measurement sessions ---------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_cancel_close_consistency_line(emit, [])
    assert lines == []


def test_all_zero_emit_nothing():
    """All turns no-barge (0.0) → no warning; nothing measurable."""
    emit, lines = _capture()
    _emit_cancel_close_consistency_line(emit, [0.0] * 10)
    assert lines == []


# ---- "prompt" excluded -----------------------------------------------


def test_long_prompt_run_does_not_fire():
    """A 10-turn run of prompt (<=500ms) teardowns is the desired state —
    never flagged."""
    emit, lines = _capture()
    _emit_cancel_close_consistency_line(emit, [0.2] * 10)
    assert lines == []


def test_alternating_prompt_and_slow_only_slow_counts():
    """[0.2, 0.7, 0.2, 0.7, ...] → after filtering, [slow] runs of 1.
    Below threshold → silent."""
    emit, lines = _capture()
    _emit_cancel_close_consistency_line(emit, [0.2, 0.7, 0.2, 0.7, 0.2, 0.7])
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_four_slow_in_a_row_fires():
    """Default threshold = 4 (barge-gated, lower bar than the usual 5)."""
    emit, lines = _capture()
    _emit_cancel_close_consistency_line(emit, [0.7] * 4)
    assert len(lines) == 1
    assert "LLM cancel teardown" in lines[0]
    assert "4 consecutive" in lines[0]
    assert "'slow'" in lines[0]
    assert "up to a" in lines[0]
    assert "iter-060" in lines[0]


def test_five_hung_in_a_row_fires():
    emit, lines = _capture()
    _emit_cancel_close_consistency_line(emit, [2.0] * 5)
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]
    assert "'hung'" in lines[0]
    assert "hangs >1s" in lines[0]


def test_below_threshold_does_not_fire():
    """3 in a row → default threshold (4) not met."""
    emit, lines = _capture()
    _emit_cancel_close_consistency_line(emit, [0.7] * 3)
    assert lines == []


# ---- Filter behavior (prompt interleavings) -------------------------


def test_prompt_between_slow_doesnt_break_run():
    """Same precedent as iter-126/128/140/143/225/226/307: filter the
    uninteresting bucket out before scanning. A 'prompt' (<=500ms)
    interleaving doesn't break a slow run."""
    emit, lines = _capture()
    # slow, prompt, slow, prompt, slow, slow → filtered: [slow]*4 → fires.
    _emit_cancel_close_consistency_line(emit, [0.7, 0.2, 0.7, 0.2, 0.7, 0.7])
    assert len(lines) == 1
    assert "4 consecutive" in lines[0]


def test_hung_breaks_slow_run():
    """Phase change between flagged buckets DOES break the run. slow
    followed by hung are both noteworthy but not the same run."""
    emit, lines = _capture()
    # 3 slow, 1 hung, 3 slow → longest run is 3 of slow. Below threshold.
    _emit_cancel_close_consistency_line(
        emit, [0.7, 0.7, 0.7, 2.0, 0.7, 0.7, 0.7]
    )
    assert lines == []


def test_no_barge_zero_between_hung_doesnt_break_run():
    """A 0.0 (no-barge / uncaptured) turn filters out and doesn't break a
    hung run, same as the prompt filter."""
    emit, lines = _capture()
    _emit_cancel_close_consistency_line(emit, [2.0, 2.0, 0.0, 2.0, 2.0])
    assert len(lines) == 1
    assert "4 consecutive" in lines[0]
    assert "'hung'" in lines[0]


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_cancel_close_consistency_line(emit, [2.0] * 3, threshold=3)
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_run():
    emit, lines = _capture()
    _emit_cancel_close_consistency_line(emit, [2.0] * 4, threshold=10)
    assert lines == []


# ---- Longest of multiple ------------------------------------------


def test_longer_hung_run_beats_shorter_slow_run():
    """[slow]*3 + [hung]*6 → only hung passes threshold; warning fires
    for hung."""
    emit, lines = _capture()
    _emit_cancel_close_consistency_line(emit, [0.7] * 3 + [2.0] * 6)
    assert "6 consecutive" in lines[0]
    assert "'hung'" in lines[0]


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_cancel_close_consistency_line(emit, [0.7] * 4)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_060_attribution():
    emit, lines = _capture()
    _emit_cancel_close_consistency_line(emit, [0.7] * 4)
    assert "iter-060" in lines[0]


# ---- Pattern parity with prior instances --------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_cancel_close_consistency_line(emit, [0.7] * 1000)
    assert len(lines) == 1
    assert "1000 consecutive" in lines[0]
