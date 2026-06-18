"""Tests for iter-224 — _emit_stt_preview_divergence_consistency_line.

Latest instance of the diversity-check pattern, applied to a
CONTINUOUS metric: buckets the per-turn ``stt_preview_divergence``
(iter-072 live-preview-vs-final transcript distance in [0.0, 1.0])
via ``_stt_preview_divergence_bucket`` before scanning. Detects 5+
consecutive turns that landed in the "noisy" or "broken" bucket —
the signal that the live STT preview is unreliable and the
read-along is misleading the user rather than helping them.

Like iter-140/141/208 and UNLIKE iter-142/143, the fine bucket is a
LOW value (small divergence is better), so the boundaries are not
inverted.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_stt_preview_divergence_consistency_line,
    _stt_preview_divergence_bucket,
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
    """0 divergence = perfect preview match / no streaming STT →
    empty bucket (the fine state, filtered by the consumer)."""
    assert _stt_preview_divergence_bucket(0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty. Shouldn't happen (metric
    is clamped to [0,1]) but a defensive fallback is cheap."""
    assert _stt_preview_divergence_bucket(-1.0) == ""


def test_bucket_good_boundary():
    """< 0.15 → good (the desired state). 0.149 is the upper
    edge."""
    assert _stt_preview_divergence_bucket(0.01) == "good"
    assert _stt_preview_divergence_bucket(0.149) == "good"


def test_bucket_noisy_boundary():
    """0.15-0.30 inclusive → noisy."""
    assert _stt_preview_divergence_bucket(0.15) == "noisy"
    assert _stt_preview_divergence_bucket(0.30) == "noisy"


def test_bucket_broken_boundary():
    """> 0.30 → broken (the documented iter-072 'preview UX is
    broken' threshold)."""
    assert _stt_preview_divergence_bucket(0.301) == "broken"
    assert _stt_preview_divergence_bucket(1.0) == "broken"


def test_bucket_handles_floats():
    """stt_preview_divergence is a float — bucket must handle
    fine-grained values around the boundaries."""
    assert _stt_preview_divergence_bucket(0.1499) == "good"
    assert _stt_preview_divergence_bucket(0.1501) == "noisy"
    assert _stt_preview_divergence_bucket(0.30) == "noisy"
    assert _stt_preview_divergence_bucket(0.3001) == "broken"


# ---- Empty / no-measurement sessions ---------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_stt_preview_divergence_consistency_line(emit, [])
    assert lines == []


def test_all_zero_divergence_emit_nothing():
    """All turns had a perfect preview match (0 divergence) → no
    warning. This is the desired state."""
    emit, lines = _capture()
    _emit_stt_preview_divergence_consistency_line(emit, [0.0] * 10)
    assert lines == []


# ---- "good" excluded -------------------------------------------------


def test_long_good_run_does_not_fire():
    """A 10-turn run of low-divergence previews is the desired
    state — never flagged."""
    emit, lines = _capture()
    _emit_stt_preview_divergence_consistency_line(emit, [0.05] * 10)
    assert lines == []


def test_alternating_good_and_noisy_only_noisy_counts():
    """[0.05, 0.2, 0.05, 0.2, ...] → after filtering, [noisy] runs
    of 1. Below threshold → silent."""
    emit, lines = _capture()
    _emit_stt_preview_divergence_consistency_line(
        emit, [0.05, 0.2, 0.05, 0.2, 0.05, 0.2],
    )
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_noisy_in_a_row_fires():
    """Default threshold = 5."""
    emit, lines = _capture()
    _emit_stt_preview_divergence_consistency_line(emit, [0.2] * 5)
    assert len(lines) == 1
    assert "STT preview" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'noisy'" in lines[0]
    assert "incremental Whisper output is unreliable" in lines[0]
    assert "iter-072" in lines[0]


def test_six_broken_in_a_row_fires():
    emit, lines = _capture()
    _emit_stt_preview_divergence_consistency_line(emit, [0.5] * 6)
    assert len(lines) == 1
    assert "6 consecutive" in lines[0]
    assert "'broken'" in lines[0]
    assert "can't trust the read-along" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold not met."""
    emit, lines = _capture()
    _emit_stt_preview_divergence_consistency_line(emit, [0.2] * 4)
    assert lines == []


# ---- Filter behavior (good interleavings) ---------------------------


def test_good_between_noisy_doesnt_break_run():
    """Same precedent as iter-126/128/140: filter the uninteresting
    bucket out before scanning. A 'good' interleaving doesn't break
    a noisy run."""
    emit, lines = _capture()
    # noisy, good, noisy, good, noisy, noisy, noisy
    _emit_stt_preview_divergence_consistency_line(
        emit, [0.2, 0.05, 0.2, 0.05, 0.2, 0.2, 0.2],
    )
    # Filtered: [noisy]*5 → fires.
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]


def test_broken_breaks_noisy_run():
    """Phase change between flagged buckets DOES break the run.
    noisy followed by broken are both noteworthy but not the same
    run."""
    emit, lines = _capture()
    # 3 noisy, 1 broken, 3 noisy → longest run is 3 of noisy.
    # Below threshold.
    _emit_stt_preview_divergence_consistency_line(
        emit, [0.2, 0.2, 0.2, 0.5, 0.2, 0.2, 0.2],
    )
    assert lines == []


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_stt_preview_divergence_consistency_line(
        emit, [0.5] * 3, threshold=3,
    )
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_5_run():
    emit, lines = _capture()
    _emit_stt_preview_divergence_consistency_line(
        emit, [0.5] * 5, threshold=10,
    )
    assert lines == []


# ---- Longest of multiple ------------------------------------------


def test_longer_broken_run_beats_shorter_noisy_run():
    """[noisy]*4 + [broken]*7 → only broken passes threshold;
    warning fires for broken."""
    emit, lines = _capture()
    _emit_stt_preview_divergence_consistency_line(
        emit, [0.2] * 4 + [0.5] * 7,
    )
    assert "7 consecutive" in lines[0]
    assert "'broken'" in lines[0]


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_stt_preview_divergence_consistency_line(emit, [0.2] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_072_attribution():
    emit, lines = _capture()
    _emit_stt_preview_divergence_consistency_line(emit, [0.2] * 5)
    assert "iter-072" in lines[0]


# ---- Pattern parity with prior instances --------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_stt_preview_divergence_consistency_line(emit, [0.2] * 1000)
    assert "1000 consecutive" in lines[0]
