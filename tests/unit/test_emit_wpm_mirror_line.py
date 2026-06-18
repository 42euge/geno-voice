"""Tests for iter-215 — _emit_wpm_mirror_line helper.

Closes iter-214 backlog #1: surface the WPM mirror's per-session speed
adaptation in the session summary so the iter-213/214 mirroring effect is
*measured*, not asserted.

Headline contract is the **off-by-default suppression**: with ``active=False``
(no live mirror — the overwhelmingly common path) the helper emits nothing, so
the summary is byte-for-byte the pre-iter-215 output. Mirrors the
iter-094/104/160 single-line-emitter test style.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import _emit_wpm_mirror_line  # noqa: E402


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line: str = "") -> None:
        lines.append(_strip_ansi(line))

    return emit, lines


# ---- Off-by-default suppression -------------------------------------------


class TestInactiveSuppressed:
    def test_inactive_emits_nothing(self):
        emit, lines = _capture()
        _emit_wpm_mirror_line(
            emit, active=False, initial_speed=1.0, final_speed=1.2,
        )
        assert lines == []

    def test_inactive_suppressed_even_when_speeds_differ(self):
        """A large drift is irrelevant when the mirror was never active."""
        emit, lines = _capture()
        _emit_wpm_mirror_line(
            emit, active=False, initial_speed=0.8, final_speed=1.3,
        )
        assert lines == []


# ---- Active, speed moved ---------------------------------------------------


class TestActiveMoved:
    def test_speed_increased(self):
        emit, lines = _capture()
        _emit_wpm_mirror_line(
            emit, active=True, initial_speed=1.0, final_speed=1.12,
        )
        assert len(lines) == 1
        assert "WPM mirror:" in lines[0]
        assert "on," in lines[0]
        assert "1.00 → 1.12" in lines[0]
        assert "(+0.12)" in lines[0]

    def test_speed_decreased_shows_negative_delta(self):
        emit, lines = _capture()
        _emit_wpm_mirror_line(
            emit, active=True, initial_speed=1.0, final_speed=0.85,
        )
        assert "1.00 → 0.85" in lines[0]
        assert "(-0.15)" in lines[0]

    def test_non_unity_initial(self):
        emit, lines = _capture()
        _emit_wpm_mirror_line(
            emit, active=True, initial_speed=0.90, final_speed=1.05,
        )
        assert "0.90 → 1.05" in lines[0]
        assert "(+0.15)" in lines[0]

    def test_moved_branch_not_held_text(self):
        emit, lines = _capture()
        _emit_wpm_mirror_line(
            emit, active=True, initial_speed=1.0, final_speed=1.2,
        )
        assert "held at" not in lines[0]


# ---- Active, speed held (deadband) ----------------------------------------


class TestActiveHeld:
    def test_exact_no_movement_is_held(self):
        emit, lines = _capture()
        _emit_wpm_mirror_line(
            emit, active=True, initial_speed=1.0, final_speed=1.0,
        )
        assert len(lines) == 1
        assert "held at 1.00" in lines[0]
        assert "→" not in lines[0]

    def test_sub_threshold_drift_is_held(self):
        """A drift under the 0.005 rounding floor reports as held, not +0.00."""
        emit, lines = _capture()
        _emit_wpm_mirror_line(
            emit, active=True, initial_speed=1.0, final_speed=1.004,
        )
        assert "held at" in lines[0]
        assert "+0.00" not in lines[0]

    def test_held_reports_initial_speed_value(self):
        emit, lines = _capture()
        _emit_wpm_mirror_line(
            emit, active=True, initial_speed=0.95, final_speed=0.95,
        )
        assert "held at 0.95" in lines[0]


# ---- Boundary between held and moved --------------------------------------


class TestBoundary:
    def test_at_threshold_counts_as_moved(self):
        """0.005 and above is a real move (abs(delta) < 0.005 is the held gate)."""
        emit, lines = _capture()
        _emit_wpm_mirror_line(
            emit, active=True, initial_speed=1.000, final_speed=1.006,
        )
        # 0.006 >= 0.005 → moved branch.
        assert "→" in lines[0]
        assert "held at" not in lines[0]


# ---- Single line, no spurious output --------------------------------------


class TestSingleLine:
    @pytest.mark.parametrize(
        "init,final",
        [(1.0, 1.0), (1.0, 1.2), (1.0, 0.8), (0.9, 1.1)],
    )
    def test_active_emits_exactly_one_line(self, init, final):
        emit, lines = _capture()
        _emit_wpm_mirror_line(
            emit, active=True, initial_speed=init, final_speed=final,
        )
        assert len(lines) == 1
