"""Tests for iter-310 — _emit_mic_stale_consistency_line.

TWENTY-SECOND instance of the diversity-check pattern, applied to a
CONTINUOUS (integer-valued) metric: buckets the per-turn
``mic_stale_frames`` count (iter-037's echo signal — the number of mic
frames flushed at the start of a turn because the mic accumulated
unwanted audio between turns) via ``_mic_stale_bucket`` before scanning.
Detects 4+ consecutive turns that landed in the "minor" (1-8000 frames,
≤0.5s) or "echo" (>8000 frames, >0.5s) bucket — the signal that the
bot's voice is reliably leaking back through the OS mic between turns
(acoustic echo / Bluetooth duplex / loopback) and the setup needs echo
cancellation.

Like iter-140/141 (RTF), iter-208 (synth-dispatch), iter-209
(eot-overhead), iter-224 (preview-divergence), iter-226 (worker-idle-gap),
iter-307 (synth-backlog), iter-308 (cancel-close) and iter-309
(speaker-open) — and UNLIKE iter-142/143/225 — the problematic end is a
LARGE value (no stale audio is best), so the boundaries are NOT inverted.

UNLIKE iter-309 (speaker-open), there is no "measured but fine"
intermediate bucket: opening a device is an expected cost (instant is
fine), but ANY stale frame is a symptom (the desired state is literally
0). So this mirrors iter-114's filler count — filter only the no-event 0,
flag EVERY measured value.

Threshold is 4 (NOT the threshold-5 general default): stale frames are
near-always 0 on a clean session, so a non-zero flush is itself an
event — like iter-120's barge-phase and iter-308's cancel-close,
event-gated rare signals earn the lower bar.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_mic_stale_consistency_line,
    _mic_stale_bucket,
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
    """0 = no stale frames this turn (the mic was silent between turns —
    the common clean case) → empty bucket (the no-event state, filtered
    by the consumer)."""
    assert _mic_stale_bucket(0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty. Shouldn't happen (a frame count
    is non-negative) but a defensive fallback is cheap."""
    assert _mic_stale_bucket(-100) == ""


def test_bucket_minor_boundary():
    """1 frame up to 8000 (≤0.5s at 16 kHz) → minor (some audio leaking
    but below iter-037's >0.5s yellow flag)."""
    assert _mic_stale_bucket(1) == "minor"
    assert _mic_stale_bucket(4000) == "minor"
    assert _mic_stale_bucket(8000) == "minor"


def test_bucket_echo_boundary():
    """> 8000 frames (>0.5s at 16 kHz) → echo (iter-037's yellow flag —
    the bot's voice reliably leaking back through the OS mic)."""
    assert _mic_stale_bucket(8001) == "echo"
    assert _mic_stale_bucket(32000) == "echo"


# ---- Empty / no-event sessions ---------------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_mic_stale_consistency_line(emit, [])
    assert lines == []


def test_all_zero_emit_nothing():
    """A clean session (no stale frames ever flushed) → no warning;
    nothing measurable."""
    emit, lines = _capture()
    _emit_mic_stale_consistency_line(emit, [0] * 10)
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_four_minor_in_a_row_fires():
    """Default threshold = 4 (event-gated default, lower bar than the
    high-frequency threshold-5 family)."""
    emit, lines = _capture()
    _emit_mic_stale_consistency_line(emit, [4000] * 4)
    assert len(lines) == 1
    assert "Mic stale" in lines[0]
    assert "4 consecutive" in lines[0]
    assert "'minor'" in lines[0]
    assert "iter-037" in lines[0]


def test_four_echo_in_a_row_fires():
    emit, lines = _capture()
    _emit_mic_stale_consistency_line(emit, [16000] * 4)
    assert len(lines) == 1
    assert "4 consecutive" in lines[0]
    assert "'echo'" in lines[0]
    assert "echo cancellation" in lines[0]


def test_below_threshold_does_not_fire():
    """3 in a row → default threshold (4) not met."""
    emit, lines = _capture()
    _emit_mic_stale_consistency_line(emit, [16000] * 3)
    assert lines == []


# ---- Filter behavior (clean-turn interleavings) ---------------------


def test_clean_zero_between_echo_doesnt_break_run():
    """Same precedent as iter-114/126/.../309: filter the no-event 0 out
    before scanning. A clean (0-frame) turn doesn't break an echo run."""
    emit, lines = _capture()
    # echo x4 with a clean 0 interleaved → filtered: [echo]*4 → fires.
    _emit_mic_stale_consistency_line(
        emit, [16000, 0, 16000, 16000, 0, 16000]
    )
    assert len(lines) == 1
    assert "4 consecutive" in lines[0]
    assert "'echo'" in lines[0]


def test_echo_breaks_minor_run():
    """Phase change between flagged buckets DOES break the run. minor
    followed by echo are both noteworthy but not the same run."""
    emit, lines = _capture()
    # 3 minor, 1 echo, 3 minor → longest run is 3 of minor. Below 4.
    _emit_mic_stale_consistency_line(
        emit, [4000, 4000, 4000, 16000, 4000, 4000, 4000]
    )
    assert lines == []


def test_minor_and_echo_dont_merge_into_one_run():
    """3 minor + 3 echo → no single bucket reaches 4. Distinct buckets
    never merge into a longer combined run."""
    emit, lines = _capture()
    _emit_mic_stale_consistency_line(
        emit, [4000, 4000, 4000, 16000, 16000, 16000]
    )
    assert lines == []


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_mic_stale_consistency_line(emit, [16000] * 3, threshold=3)
    assert "3 consecutive" in lines[0]


def test_threshold_10_suppresses_default_run():
    emit, lines = _capture()
    _emit_mic_stale_consistency_line(emit, [16000] * 4, threshold=10)
    assert lines == []


# ---- Longest of multiple ------------------------------------------


def test_longer_echo_run_beats_shorter_minor_run():
    """[minor]*3 + [echo]*5 → only echo passes threshold; warning fires
    for echo."""
    emit, lines = _capture()
    _emit_mic_stale_consistency_line(emit, [4000] * 3 + [16000] * 5)
    assert "5 consecutive" in lines[0]
    assert "'echo'" in lines[0]


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_mic_stale_consistency_line(emit, [16000] * 4)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_037_attribution():
    emit, lines = _capture()
    _emit_mic_stale_consistency_line(emit, [16000] * 4)
    assert "iter-037" in lines[0]


def test_minor_suggestion_distinct_from_echo():
    """Per-value suggestion mapping: the minor and echo branches produce
    different operator guidance (not one-size-fits-all)."""
    emit_minor, minor_lines = _capture()
    _emit_mic_stale_consistency_line(emit_minor, [4000] * 4)
    emit_echo, echo_lines = _capture()
    _emit_mic_stale_consistency_line(emit_echo, [16000] * 4)
    assert minor_lines[0] != echo_lines[0]
    assert "under 0.5s" in minor_lines[0]
    assert ">0.5s" in echo_lines[0]


# ---- Pattern parity with prior instances --------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_mic_stale_consistency_line(emit, [16000] * 1000)
    assert len(lines) == 1
    assert "1000 consecutive" in lines[0]
