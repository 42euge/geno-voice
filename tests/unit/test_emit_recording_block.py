"""Tests for iter-103 — _emit_recording_block helper.

The helper bundles iter-037 (mic stale frames) and iter-048 (VAD
false-trigger rate) into a single output block. Both lines are
suppressed when the underlying signal is zero, so the helper has
two boolean knobs and a 2x2 matrix of outputs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    RecordingStats,
    _emit_recording_block,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line=""):
        lines.append(_strip_ansi(line))

    return emit, lines


def test_silent_when_clean_session():
    """No stale frames + no false triggers → no output. Clean
    sessions don't bloat the summary."""
    emit, lines = _capture()
    _emit_recording_block(emit, RecordingStats())
    assert lines == []


def test_stale_only_emits_one_line():
    """Stale frames present but no false triggers → only the mic
    stale line."""
    emit, lines = _capture()
    _emit_recording_block(
        emit, RecordingStats(stale_total=16000, false_triggers=0, n=10),
    )
    assert len(lines) == 1
    assert "Mic stale:" in lines[0]
    # 16000 frames at 16kHz = 1.0s.
    assert "16000 frames" in lines[0]
    assert "1.0s" in lines[0]
    assert "echo cancellation" in lines[0]


def test_false_triggers_only_emits_one_line():
    """False triggers present but no stale frames → only the VAD
    false-trig line."""
    emit, lines = _capture()
    _emit_recording_block(
        emit, RecordingStats(stale_total=0, false_triggers=2, n=8),
    )
    assert len(lines) == 1
    assert "VAD false-trig:" in lines[0]
    # 2 false out of 10 attempts = 20%.
    assert "2/10" in lines[0]
    assert "20%" in lines[0]
    assert "silence_threshold" in lines[0]


def test_both_signals_emit_two_lines_in_order():
    """When both fire, mic stale comes first, false-trig second.
    Order matters — it matches the legacy inline output."""
    emit, lines = _capture()
    _emit_recording_block(
        emit,
        RecordingStats(stale_total=8000, false_triggers=1, n=4),
    )
    assert len(lines) == 2
    assert "Mic stale:" in lines[0]
    assert "VAD false-trig:" in lines[1]
    # 8000 / 16000 = 0.5s.
    assert "0.5s" in lines[0]
    # 1 false out of 5 attempts = 20%.
    assert "1/5" in lines[1]
    assert "20%" in lines[1]


def test_stale_seconds_uses_16khz_divisor():
    """iter-037 normalizes stale frames to seconds at the mic
    sample rate (16kHz hardcoded). 32000 frames = 2.0s."""
    emit, lines = _capture()
    _emit_recording_block(
        emit, RecordingStats(stale_total=32000),
    )
    assert "2.0s" in lines[0]


def test_false_trigger_pct_rounds_to_zero_decimals():
    """The percentage is formatted as integer (no decimals).
    1 / 3 attempts ≈ 33% (1 false + 2 succeeded)."""
    emit, lines = _capture()
    _emit_recording_block(
        emit, RecordingStats(false_triggers=1, n=2),
    )
    assert "1/3" in lines[0]
    assert "33%" in lines[0]


def test_lines_have_leading_4_space_indent():
    """All session-summary block helpers use a leading 4-space
    indent (no tree pipe). Match the existing _emit_*_block
    family."""
    emit, lines = _capture()
    _emit_recording_block(
        emit,
        RecordingStats(stale_total=1000, false_triggers=1, n=1),
    )
    assert all(ln.startswith("    ") for ln in lines)


def test_n_zero_with_one_false_trigger():
    """Edge case: n=0 (no successful turns) but a false trigger
    fired. attempts=1, pct=100%. The helper must not divide by
    zero — it doesn't, because attempts is false_triggers + n,
    never zero when this branch fires."""
    emit, lines = _capture()
    _emit_recording_block(
        emit, RecordingStats(false_triggers=1, n=0),
    )
    assert "1/1" in lines[0]
    assert "100%" in lines[0]
