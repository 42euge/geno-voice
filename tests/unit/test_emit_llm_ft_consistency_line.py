"""Tests for iter-312 — _emit_llm_ft_consistency_line.

TWENTY-FOURTH instance of the diversity-check pattern, applied to a
CONTINUOUS (float-valued) metric: buckets the per-turn ``llm_first_token``
(iter-052 — the time from the LLM stream starting to its FIRST token
arriving) via ``_llm_ft_bucket`` before scanning. Detects 5+ consecutive
turns that landed in the "slow" (0.3-0.6s) or "very_slow" (>0.6s) bucket —
the signal that the LLM's time-to-first-byte is the persistent bottleneck
(model warmup / overloaded backend / context creep), eroding the sub-500ms
TTFS budget.

This is the SIBLING of iter-311's FT-A sentinel: iter-083 decomposes TTFS
into the LLM-side half (``llm_first_token``, this) and the post-LLM-side
half (``first_token_to_audio``, FT-A — iter-311). Together they watch both
halves.

Like iter-140/141 (RTF), iter-208 (synth-dispatch), iter-209
(eot-overhead), iter-212 (ttfs), iter-224 (preview-divergence), iter-226
(worker-idle-gap), iter-307 (synth-backlog), iter-308 (cancel-close),
iter-309 (speaker-open), iter-310 (mic-stale), iter-311 (fta) — and UNLIKE
iter-142/143/225 — the problematic end is a LARGE value (small
first-token latency is best), so the boundaries are NOT inverted: the
"snappy" fine bucket is a LOW value and gets filtered alongside the empty
(no-measurement) bucket.

Threshold is 5 (the general-signal default, NOT iter-120/308/310's
event-gated 4): llm_first_token is measured on essentially every responding
turn, a high-frequency signal where natural per-turn variation is normal.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_llm_ft_consistency_line,
    _llm_ft_bucket,
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
    """0 = no first token this turn (LLM errored before emitting) →
    empty bucket (the no-event state, filtered by the consumer)."""
    assert _llm_ft_bucket(0.0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty (clock-skew guard)."""
    assert _llm_ft_bucket(-0.5) == ""


def test_bucket_snappy_boundary():
    """Just above 0 up to (not incl.) 0.3s → snappy (the model emits its
    first token promptly — the desired state, filtered)."""
    assert _llm_ft_bucket(0.001) == "snappy"
    assert _llm_ft_bucket(0.2) == "snappy"
    assert _llm_ft_bucket(0.2999) == "snappy"


def test_bucket_slow_boundary():
    """0.3s up to 0.6s inclusive → slow."""
    assert _llm_ft_bucket(0.3) == "slow"
    assert _llm_ft_bucket(0.45) == "slow"
    assert _llm_ft_bucket(0.6) == "slow"


def test_bucket_very_slow_boundary():
    """> 0.6s → very_slow (exceeds the filler idle_threshold default)."""
    assert _llm_ft_bucket(0.6001) == "very_slow"
    assert _llm_ft_bucket(1.5) == "very_slow"


# ---- Empty / no-event sessions ---------------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_llm_ft_consistency_line(emit, [])
    assert lines == []


def test_all_zero_emit_nothing():
    """A session where the LLM never produced a token → no warning."""
    emit, lines = _capture()
    _emit_llm_ft_consistency_line(emit, [0.0] * 10)
    assert lines == []


def test_all_snappy_emit_nothing():
    """A healthy session (every turn's first token arrives promptly) →
    the snappy bucket is filtered, nothing fires."""
    emit, lines = _capture()
    _emit_llm_ft_consistency_line(emit, [0.1] * 10)
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_slow_in_a_row_fires():
    """Default threshold = 5 (the high-frequency general default)."""
    emit, lines = _capture()
    _emit_llm_ft_consistency_line(emit, [0.45] * 5)
    assert len(lines) == 1
    assert "LLM 1st tok" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'slow'" in lines[0]
    assert "iter-052" in lines[0]


def test_five_very_slow_in_a_row_fires():
    emit, lines = _capture()
    _emit_llm_ft_consistency_line(emit, [0.9] * 5)
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]
    assert "'very_slow'" in lines[0]
    assert ">600ms" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold (5) not met."""
    emit, lines = _capture()
    _emit_llm_ft_consistency_line(emit, [0.9] * 4)
    assert lines == []


# ---- Filter behavior (snappy/empty interleavings) -------------------


def test_snappy_between_slow_doesnt_break_run():
    """The snappy fine-bucket is filtered before scanning, so a fast turn
    interleaved between slow turns doesn't break the slow run."""
    emit, lines = _capture()
    _emit_llm_ft_consistency_line(
        emit, [0.45, 0.1, 0.45, 0.45, 0.1, 0.45, 0.45]
    )
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]
    assert "'slow'" in lines[0]


def test_empty_between_slow_doesnt_break_run():
    """A no-token (0.0) turn is also filtered and doesn't break a slow
    run."""
    emit, lines = _capture()
    _emit_llm_ft_consistency_line(
        emit, [0.45, 0.0, 0.45, 0.45, 0.0, 0.45, 0.45]
    )
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]


def test_very_slow_breaks_slow_run():
    """Phase change between flagged buckets DOES break the run. slow then
    very_slow are both noteworthy but not the same run."""
    emit, lines = _capture()
    # 4 slow, 1 very_slow, 4 slow → longest run is 4 of slow. Below 5.
    _emit_llm_ft_consistency_line(
        emit, [0.45, 0.45, 0.45, 0.45, 0.9, 0.45, 0.45, 0.45, 0.45]
    )
    assert lines == []


def test_slow_and_very_slow_dont_merge_into_one_run():
    """4 slow + 4 very_slow → no single bucket reaches 5. Distinct
    buckets never merge into a longer combined run."""
    emit, lines = _capture()
    _emit_llm_ft_consistency_line(emit, [0.45] * 4 + [0.9] * 4)
    assert lines == []


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_llm_ft_consistency_line(emit, [0.9] * 3, threshold=3)
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_run():
    emit, lines = _capture()
    _emit_llm_ft_consistency_line(emit, [0.9] * 5, threshold=10)
    assert lines == []


# ---- Longest of multiple ------------------------------------------


def test_longer_very_slow_run_beats_shorter_slow_run():
    """[slow]*4 + [very_slow]*6 → only very_slow passes threshold;
    warning fires for very_slow."""
    emit, lines = _capture()
    _emit_llm_ft_consistency_line(emit, [0.45] * 4 + [0.9] * 6)
    assert "6 consecutive" in lines[0]
    assert "'very_slow'" in lines[0]


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_llm_ft_consistency_line(emit, [0.45] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_052_attribution():
    emit, lines = _capture()
    _emit_llm_ft_consistency_line(emit, [0.45] * 5)
    assert "iter-052" in lines[0]
    assert "llm_first_token" in lines[0]


def test_slow_suggestion_distinct_from_very_slow():
    """Per-value suggestion mapping: the slow and very_slow branches
    produce different operator guidance (not one-size-fits-all)."""
    emit_slow, slow_lines = _capture()
    _emit_llm_ft_consistency_line(emit_slow, [0.45] * 5)
    emit_vslow, vslow_lines = _capture()
    _emit_llm_ft_consistency_line(emit_vslow, [0.9] * 5)
    assert slow_lines[0] != vslow_lines[0]
    assert ">600ms" in vslow_lines[0]
    assert ">600ms" not in slow_lines[0]


# ---- Pattern parity with prior instances --------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_llm_ft_consistency_line(emit, [0.45] * 1000)
    assert len(lines) == 1
    assert "1000 consecutive" in lines[0]
