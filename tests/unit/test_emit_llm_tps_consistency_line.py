"""Tests for iter-142 — _emit_llm_tps_consistency_line.

Seventh instance of the diversity-check pattern. Fourth applied
to a CONTINUOUS metric (after iter-128 sentence-length, iter-140
stt-rtf, iter-141 tts-rtf) — buckets the per-turn ``llm_tps`` via
``_llm_tps_bucket`` before scanning. Detects 5+ consecutive turns
that landed in the "slow" or "very_slow" bucket, suggesting the LLM
is streaming too slowly to feed the TTS worker.

UNLIKE the RTF instances, ``llm_tps`` is bigger-is-better: the fine
bucket ("fast") is a HIGH value, so the bucket boundaries invert.
This is the first inverted-direction continuous bucketer, so the
boundary tests below probe the high end as the "good" state.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_llm_tps_consistency_line,
    _llm_tps_bucket,
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
    """0 tps = no measurable LLM stream this turn → empty bucket
    (filtered by the consumer)."""
    assert _llm_tps_bucket(0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty. Shouldn't happen in
    practice but a defensive fallback is cheap."""
    assert _llm_tps_bucket(-1.0) == ""


def test_bucket_fast_boundary():
    """>= 25.0 → fast (the desired, high-throughput state). 25.0 is
    the lower edge of fast."""
    assert _llm_tps_bucket(25.0) == "fast"
    assert _llm_tps_bucket(80.0) == "fast"


def test_bucket_slow_boundary():
    """10.0-25.0 → slow. Inclusive lower edge, exclusive upper."""
    assert _llm_tps_bucket(10.0) == "slow"
    assert _llm_tps_bucket(24.999) == "slow"


def test_bucket_very_slow_boundary():
    """< 10.0 → very_slow."""
    assert _llm_tps_bucket(9.999) == "very_slow"
    assert _llm_tps_bucket(1.0) == "very_slow"


def test_bucket_handles_floats():
    """llm_tps is a float — bucket must handle fine-grained values
    around the (inverted) boundaries."""
    assert _llm_tps_bucket(24.9999) == "slow"
    assert _llm_tps_bucket(25.0001) == "fast"
    assert _llm_tps_bucket(10.0) == "slow"
    assert _llm_tps_bucket(9.9999) == "very_slow"


# ---- Empty / no-measurement sessions ---------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_llm_tps_consistency_line(emit, [])
    assert lines == []


def test_all_zero_tps_emit_nothing():
    """All turns had no measurable LLM stream (0 tps) → no
    warning."""
    emit, lines = _capture()
    _emit_llm_tps_consistency_line(emit, [0.0] * 10)
    assert lines == []


# ---- "fast" excluded (the inverted fine state) -----------------------


def test_long_fast_run_does_not_fire():
    """A 10-turn run of fast (high-throughput) LLM streaming is the
    desired state — never flagged. The fine bucket here is HIGH."""
    emit, lines = _capture()
    _emit_llm_tps_consistency_line(emit, [60.0] * 10)
    assert lines == []


def test_alternating_fast_and_slow_only_slow_counts():
    """[60, 15, 60, 15, ...] → after filtering, [slow] runs of 1.
    Below threshold → silent."""
    emit, lines = _capture()
    _emit_llm_tps_consistency_line(
        emit, [60.0, 15.0, 60.0, 15.0, 60.0, 15.0],
    )
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_slow_in_a_row_fires():
    """Default threshold = 5."""
    emit, lines = _capture()
    _emit_llm_tps_consistency_line(emit, [15.0] * 5)
    assert len(lines) == 1
    assert "LLM speed" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'slow'" in lines[0]
    assert "LLM stream lags synth" in lines[0]
    assert "iter-052" in lines[0]


def test_six_very_slow_in_a_row_fires():
    emit, lines = _capture()
    _emit_llm_tps_consistency_line(emit, [5.0] * 6)
    assert len(lines) == 1
    assert "6 consecutive" in lines[0]
    assert "'very_slow'" in lines[0]
    assert "dominant bottleneck" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold not met."""
    emit, lines = _capture()
    _emit_llm_tps_consistency_line(emit, [15.0] * 4)
    assert lines == []


# ---- Filter behavior (fast interleavings) ----------------------


def test_fast_between_slow_doesnt_break_run():
    """Same precedent as iter-126/128/140/141: filter the
    uninteresting bucket out before scanning. A 'fast' interleaving
    doesn't break a slow run."""
    emit, lines = _capture()
    # slow, fast, slow, fast, slow, slow, slow
    _emit_llm_tps_consistency_line(
        emit, [15.0, 60.0, 15.0, 60.0, 15.0, 15.0, 15.0],
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
    _emit_llm_tps_consistency_line(
        emit, [15.0, 15.0, 15.0, 5.0, 15.0, 15.0, 15.0],
    )
    assert lines == []


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_llm_tps_consistency_line(emit, [5.0] * 3, threshold=3)
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_5_run():
    emit, lines = _capture()
    _emit_llm_tps_consistency_line(emit, [5.0] * 5, threshold=10)
    assert lines == []


# ---- Longest of multiple ------------------------------------------


def test_longer_very_slow_run_beats_shorter_slow_run():
    """[slow]*4 + [very_slow]*7 → only very_slow passes threshold;
    warning fires for very_slow."""
    emit, lines = _capture()
    _emit_llm_tps_consistency_line(emit, [15.0] * 4 + [5.0] * 7)
    assert "7 consecutive" in lines[0]
    assert "'very_slow'" in lines[0]


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_llm_tps_consistency_line(emit, [15.0] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_052_attribution():
    emit, lines = _capture()
    _emit_llm_tps_consistency_line(emit, [15.0] * 5)
    assert "iter-052" in lines[0]


# ---- Pattern parity with iter-114/115/120/126/128/140/141 -----------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_llm_tps_consistency_line(emit, [15.0] * 1000)
    assert "1000 consecutive" in lines[0]
