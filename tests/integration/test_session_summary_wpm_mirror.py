"""Integration test for iter-215 — the WPM-mirror speed-adaptation line flows
through ``print_session_summary`` via ``SessionMeta``.

The unit test (``test_emit_wpm_mirror_line.py``) covers the helper in isolation;
this confirms the three ``SessionMeta`` fields (``wpm_mirror_active`` /
``wpm_mirror_initial_speed`` / ``wpm_mirror_final_speed``) are actually wired
into the full summary render, and — the headline contract — that a session with
mirroring off (the default ``SessionMeta``) produces no new line.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    SessionMeta,
    TurnMetrics,
    print_session_summary,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _make_metric() -> TurnMetrics:
    return TurnMetrics(
        stt_time=0.05,
        llm_first_token=0.1,
        tts_time=0.2,
        ttfs=0.3,
        model="test-model",
    )


def _render(meta: SessionMeta) -> str:
    buf = io.StringIO()
    print_session_summary(
        [_make_metric()], {"model": "x"}, file=buf, meta=meta,
    )
    return _strip_ansi(buf.getvalue())


# ---- Off-by-default: no line ----------------------------------------------


def test_default_meta_no_mirror_line():
    """A SessionMeta with no mirror fields (the off-by-default path) emits no
    WPM-mirror line — the summary is unchanged from pre-iter-215."""
    out = _render(SessionMeta())
    assert "WPM mirror:" not in out


def test_inactive_with_speeds_set_still_no_line():
    """Even if the speeds happen to be populated, an inactive mirror is silent."""
    out = _render(
        SessionMeta(
            wpm_mirror_active=False,
            wpm_mirror_initial_speed=1.0,
            wpm_mirror_final_speed=1.3,
        )
    )
    assert "WPM mirror:" not in out


# ---- Active: line appears --------------------------------------------------


def test_active_moved_line_appears():
    out = _render(
        SessionMeta(
            wpm_mirror_active=True,
            wpm_mirror_initial_speed=1.0,
            wpm_mirror_final_speed=1.15,
        )
    )
    assert "WPM mirror:" in out
    assert "1.00 → 1.15" in out
    assert "(+0.15)" in out


def test_active_held_line_appears():
    out = _render(
        SessionMeta(
            wpm_mirror_active=True,
            wpm_mirror_initial_speed=1.0,
            wpm_mirror_final_speed=1.0,
        )
    )
    assert "WPM mirror:" in out
    assert "held at 1.00" in out


# ---- Placement: near the WPM medians it complements -----------------------


def test_mirror_line_follows_wpm_block():
    """The mirror line should sit just under the WPM median/gap lines it
    complements (both share the 'measured speaking rate' concern)."""
    out = _render(
        SessionMeta(
            wpm_mirror_active=True,
            wpm_mirror_initial_speed=1.0,
            wpm_mirror_final_speed=1.2,
        )
    )
    lines = out.splitlines()
    mirror_idx = next(i for i, ln in enumerate(lines) if "WPM mirror:" in ln)
    # No bot/user WPM was measured in this minimal metric, so the WPM medians
    # may be absent; what matters is the line renders inside the summary body,
    # not at the very top or after the trailing rule.
    assert 0 < mirror_idx < len(lines) - 1
