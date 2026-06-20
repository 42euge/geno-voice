"""Tests for iter-309 — _emit_speaker_open_consistency_line.

TWENTY-FIRST instance of the diversity-check pattern, applied to a
CONTINUOUS (float-valued) metric: buckets the per-turn
``speaker_open_seconds`` (iter-061's speaker-open overhead — the time
spent opening the audio output device before a turn's first sentence
plays) via ``_speaker_open_bucket`` before scanning. Detects 5+
consecutive turns that landed in the "slow" (50-150ms) or "stalled"
(>150ms) bucket — the signal that the persistent-speaker reuse the
iter-008 streaming design depends on has broken, so every turn re-pays
device-init latency before its first audio.

Like iter-140/141 (RTF), iter-208 (synth-dispatch), iter-209
(eot-overhead), iter-224 (preview-divergence), iter-226 (worker-idle-gap),
iter-307 (synth-backlog) and iter-308 (cancel-close) — and UNLIKE
iter-142/143/225 — the fine bucket is a LOW value (an instant open is
best), so the boundaries are NOT inverted: the problematic end is a
LARGE latency.

Threshold is 5 (the general-signal default, NOT iter-120/308's
barge-gated 4): speaker-open is measured on every device-opening turn,
a high-frequency signal where natural variation is normal.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_speaker_open_consistency_line,
    _speaker_open_bucket,
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
    """0.0 = no speaker open measured this turn (persistent-speaker
    reuse or the early error path) → empty bucket (the no-measurement
    state, filtered by the consumer)."""
    assert _speaker_open_bucket(0.0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty. Shouldn't happen (a latency is
    non-negative) but a defensive fallback is cheap."""
    assert _speaker_open_bucket(-0.5) == ""


def test_bucket_instant_boundary():
    """Just above 0 and up to 50ms → instant (the device opened promptly
    — the desired state, and iter-061's dim case)."""
    assert _speaker_open_bucket(0.001) == "instant"
    assert _speaker_open_bucket(0.025) == "instant"
    assert _speaker_open_bucket(0.050) == "instant"


def test_bucket_slow_boundary():
    """Just above 50ms up to 150ms → slow (iter-061's yellow case starts
    here — opening noticeably laggy)."""
    assert _speaker_open_bucket(0.0501) == "slow"
    assert _speaker_open_bucket(0.10) == "slow"
    assert _speaker_open_bucket(0.150) == "slow"


def test_bucket_stalled_boundary():
    """> 150ms → stalled (device init dominates the turn's startup)."""
    assert _speaker_open_bucket(0.1501) == "stalled"
    assert _speaker_open_bucket(1.0) == "stalled"


# ---- Empty / no-measurement sessions ---------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_speaker_open_consistency_line(emit, [])
    assert lines == []


def test_all_zero_emit_nothing():
    """All turns reused the persistent speaker (0.0) → no warning;
    nothing measurable."""
    emit, lines = _capture()
    _emit_speaker_open_consistency_line(emit, [0.0] * 10)
    assert lines == []


# ---- "instant" excluded ----------------------------------------------


def test_long_instant_run_does_not_fire():
    """A 10-turn run of instant (<=50ms) opens is the desired state —
    never flagged."""
    emit, lines = _capture()
    _emit_speaker_open_consistency_line(emit, [0.02] * 10)
    assert lines == []


def test_alternating_instant_and_slow_only_slow_counts():
    """[0.02, 0.10, 0.02, 0.10, ...] → after filtering, [slow] runs of 1.
    Below threshold → silent."""
    emit, lines = _capture()
    _emit_speaker_open_consistency_line(
        emit, [0.02, 0.10, 0.02, 0.10, 0.02, 0.10]
    )
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_slow_in_a_row_fires():
    """Default threshold = 5 (general-signal default, higher bar than
    the barge-gated 4)."""
    emit, lines = _capture()
    _emit_speaker_open_consistency_line(emit, [0.10] * 5)
    assert len(lines) == 1
    assert "Speaker open" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'slow'" in lines[0]
    assert "50-150ms" in lines[0]
    assert "iter-061" in lines[0]


def test_five_stalled_in_a_row_fires():
    emit, lines = _capture()
    _emit_speaker_open_consistency_line(emit, [0.30] * 5)
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]
    assert "'stalled'" in lines[0]
    assert ">150ms" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold (5) not met."""
    emit, lines = _capture()
    _emit_speaker_open_consistency_line(emit, [0.10] * 4)
    assert lines == []


# ---- Filter behavior (instant interleavings) ------------------------


def test_instant_between_slow_doesnt_break_run():
    """Same precedent as iter-126/128/140/143/225/226/307/308: filter the
    uninteresting bucket out before scanning. An 'instant' (<=50ms)
    interleaving doesn't break a slow run."""
    emit, lines = _capture()
    # slow x5 with an instant interleaved → filtered: [slow]*5 → fires.
    _emit_speaker_open_consistency_line(
        emit, [0.10, 0.02, 0.10, 0.10, 0.02, 0.10, 0.10]
    )
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]


def test_stalled_breaks_slow_run():
    """Phase change between flagged buckets DOES break the run. slow
    followed by stalled are both noteworthy but not the same run."""
    emit, lines = _capture()
    # 4 slow, 1 stalled, 4 slow → longest run is 4 of slow. Below 5.
    _emit_speaker_open_consistency_line(
        emit, [0.10, 0.10, 0.10, 0.10, 0.30, 0.10, 0.10, 0.10, 0.10]
    )
    assert lines == []


def test_reused_zero_between_stalled_doesnt_break_run():
    """A 0.0 (persistent-speaker reuse / uncaptured) turn filters out and
    doesn't break a stalled run, same as the instant filter."""
    emit, lines = _capture()
    _emit_speaker_open_consistency_line(
        emit, [0.30, 0.30, 0.0, 0.30, 0.30, 0.30]
    )
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]
    assert "'stalled'" in lines[0]


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_speaker_open_consistency_line(emit, [0.30] * 3, threshold=3)
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_run():
    emit, lines = _capture()
    _emit_speaker_open_consistency_line(emit, [0.30] * 5, threshold=10)
    assert lines == []


# ---- Longest of multiple ------------------------------------------


def test_longer_stalled_run_beats_shorter_slow_run():
    """[slow]*4 + [stalled]*6 → only stalled passes threshold; warning
    fires for stalled."""
    emit, lines = _capture()
    _emit_speaker_open_consistency_line(emit, [0.10] * 4 + [0.30] * 6)
    assert "6 consecutive" in lines[0]
    assert "'stalled'" in lines[0]


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_speaker_open_consistency_line(emit, [0.10] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_061_attribution():
    emit, lines = _capture()
    _emit_speaker_open_consistency_line(emit, [0.10] * 5)
    assert "iter-061" in lines[0]


# ---- Pattern parity with prior instances --------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_speaker_open_consistency_line(emit, [0.10] * 1000)
    assert len(lines) == 1
    assert "1000 consecutive" in lines[0]
