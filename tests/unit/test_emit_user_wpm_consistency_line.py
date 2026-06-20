"""Tests for iter-323 — _emit_user_wpm_consistency_line.

Symmetric twin of iter-210's bot-WPM sentinel. The SECOND instance of
the diversity-check pattern with a TWO-SIDED (band) sweet spot, and the
NINTH applied to a CONTINUOUS metric (after iter-128 sentence-length,
iter-140 stt-rtf, iter-141 tts-rtf, iter-142 llm-tps, iter-143
streaming-overlap, iter-208 synth-dispatch, iter-209 eot-overhead,
iter-210 bot-wpm) — buckets the per-turn ``user_wpm`` (iter-064 user
speaking rate) via ``_user_wpm_bucket`` before scanning. Detects 5+
consecutive turns landing in the "fast" or "slow" bucket, surfacing a
WPM-mirror ``base_wpm`` target mistuned for this speaker.

The fine state is the MIDDLE band ("natural", 130-200 WPM), and BOTH
extremes are flagged but need OPPOSITE recalibrations (fast → raise
base_wpm; slow → lower base_wpm). The run scan keeps the two flagged
buckets distinct — a fast run and a slow run never merge into one.

Note the bucket NAMES differ from iter-210's ("fast"/"slow" vs
"rushed"/"sluggish"): a user speaking fast isn't a defect to fix the way
a rushed bot is — it's a property of the speaker the mirror adapts TO.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_user_wpm_consistency_line,
    _user_wpm_bucket,
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
    """0 WPM = no measurable speech (speech_duration 0 / empty
    transcript) → empty bucket (filtered by the consumer)."""
    assert _user_wpm_bucket(0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty. Shouldn't happen (WPM is a
    non-negative rate) but the fallback is cheap."""
    assert _user_wpm_bucket(-50.0) == ""


def test_bucket_natural_band():
    """130-200 WPM inclusive → natural (the conversational sweet spot,
    matching the per-turn green display and the mirror's 165 default)."""
    assert _user_wpm_bucket(130.0) == "natural"
    assert _user_wpm_bucket(165.0) == "natural"
    assert _user_wpm_bucket(200.0) == "natural"


def test_bucket_slow_below_band():
    """0 < wpm < 130 → slow."""
    assert _user_wpm_bucket(1.0) == "slow"
    assert _user_wpm_bucket(129.99) == "slow"


def test_bucket_fast_above_band():
    """> 200 WPM → fast."""
    assert _user_wpm_bucket(200.01) == "fast"
    assert _user_wpm_bucket(400.0) == "fast"


def test_bucket_handles_float_edges():
    """user_wpm is a float — bucket must handle values straddling the
    band boundaries."""
    assert _user_wpm_bucket(129.999) == "slow"
    assert _user_wpm_bucket(130.001) == "natural"
    assert _user_wpm_bucket(200.0) == "natural"
    assert _user_wpm_bucket(200.001) == "fast"


def test_bucket_boundaries_match_bot_wpm_band():
    """The user band must match iter-210's bot band exactly (same
    130-200 sweet spot) so the mirror's symmetric reasoning holds — only
    the bucket NAMES differ."""
    from examples._chat_metrics import _bot_wpm_bucket

    for wpm in (50.0, 129.0, 130.0, 165.0, 200.0, 201.0, 300.0):
        user_b = _user_wpm_bucket(wpm)
        bot_b = _bot_wpm_bucket(wpm)
        # Same band membership: natural↔natural, the low end and the
        # high end line up even though names differ.
        low = {"slow", "sluggish"}
        high = {"fast", "rushed"}
        if bot_b == "natural":
            assert user_b == "natural"
        elif bot_b in low:
            assert user_b in low
        elif bot_b in high:
            assert user_b in high


# ---- Empty / no-measurement sessions ---------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_user_wpm_consistency_line(emit, [])
    assert lines == []


def test_all_zero_emits_nothing():
    """All turns had no measurable speech (0 WPM) → no warning."""
    emit, lines = _capture()
    _emit_user_wpm_consistency_line(emit, [0.0] * 10)
    assert lines == []


# ---- "natural" excluded (the sweet spot is never flagged) -----------


def test_long_natural_run_does_not_fire():
    """A 10-turn run in the conversational sweet spot is the desired
    state — never flagged."""
    emit, lines = _capture()
    _emit_user_wpm_consistency_line(emit, [165.0] * 10)
    assert lines == []


def test_alternating_natural_and_fast_only_fast_counts():
    """[165, 250, 165, 250, ...] → after filtering, [fast] runs of 1.
    Below threshold → silent."""
    emit, lines = _capture()
    _emit_user_wpm_consistency_line(
        emit, [165.0, 250.0, 165.0, 250.0, 165.0, 250.0],
    )
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_fast_in_a_row_fires():
    """Default threshold = 5. Fast → raise base_wpm."""
    emit, lines = _capture()
    _emit_user_wpm_consistency_line(emit, [250.0] * 5)
    assert len(lines) == 1
    assert "User speech rate" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'fast'" in lines[0]
    assert "raise base_wpm" in lines[0]
    assert "calibrate-base-wpm" in lines[0]
    assert "iter-064" in lines[0]


def test_six_slow_in_a_row_fires():
    """Slow → lower base_wpm (OPPOSITE direction from fast — the
    per-value branch carries the real signal)."""
    emit, lines = _capture()
    _emit_user_wpm_consistency_line(emit, [100.0] * 6)
    assert len(lines) == 1
    assert "6 consecutive" in lines[0]
    assert "'slow'" in lines[0]
    assert "lower base_wpm" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold not met."""
    emit, lines = _capture()
    _emit_user_wpm_consistency_line(emit, [250.0] * 4)
    assert lines == []


# ---- Filter behavior (natural interleavings) ------------------------


def test_natural_between_fast_doesnt_break_run():
    """Same precedent as iter-126/128/140/208/209/210: filter the
    uninteresting bucket out before scanning. A 'natural' interleaving
    doesn't break a fast run."""
    emit, lines = _capture()
    # fast, natural, fast, natural, fast, fast, fast
    _emit_user_wpm_consistency_line(
        emit, [250.0, 165.0, 250.0, 165.0, 250.0, 250.0, 250.0],
    )
    # Filtered: [fast]*5 → fires.
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]
    assert "'fast'" in lines[0]


# ---- Two-sided band: opposite extremes are distinct phases ----------


def test_fast_and_slow_runs_do_not_merge():
    """THE two-sided-band invariant (shared with iter-210): fast and
    slow are BOTH flagged but sit at OPPOSITE ends, so they must NOT
    merge into one run. 3 fast + 3 slow → longest run is 3 (below
    threshold), even though 6 turns total were "outside the band"."""
    emit, lines = _capture()
    _emit_user_wpm_consistency_line(
        emit, [250.0, 250.0, 250.0, 100.0, 100.0, 100.0],
    )
    assert lines == []


def test_slow_breaks_fast_run():
    """A slow turn between fast turns breaks the fast run (phase change
    between two flagged buckets, like iter-208/209/210's
    very_slow-breaks-slow case)."""
    emit, lines = _capture()
    # 3 fast, 1 slow, 3 fast → longest fast run is 3.
    _emit_user_wpm_consistency_line(
        emit, [250.0, 250.0, 250.0, 100.0, 250.0, 250.0, 250.0],
    )
    assert lines == []


def test_longer_slow_run_wins_over_shorter_fast_run():
    """[fast]*4 + [slow]*7 → only the slow run passes the threshold;
    warning fires for slow with the lower-base_wpm fix."""
    emit, lines = _capture()
    _emit_user_wpm_consistency_line(emit, [250.0] * 4 + [100.0] * 7)
    assert len(lines) == 1
    assert "7 consecutive" in lines[0]
    assert "'slow'" in lines[0]
    assert "lower base_wpm" in lines[0]


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_user_wpm_consistency_line(emit, [250.0] * 3, threshold=3)
    assert "3 consecutive" in lines[0]
    assert "'fast'" in lines[0]


def test_threshold_10_suppresses_default_5_run():
    emit, lines = _capture()
    _emit_user_wpm_consistency_line(emit, [100.0] * 5, threshold=10)
    assert lines == []


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_user_wpm_consistency_line(emit, [250.0] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_064_attribution():
    emit, lines = _capture()
    _emit_user_wpm_consistency_line(emit, [100.0] * 5)
    assert "iter-064" in lines[0]


def test_distinguishes_from_bot_wpm_line():
    """The user line must be unambiguously about the USER, not the bot —
    a 'User speech rate' label and a mirror/base_wpm fix (not a kokoro
    speed-knob fix, which is the bot's iter-210 remedy)."""
    emit, lines = _capture()
    _emit_user_wpm_consistency_line(emit, [250.0] * 5)
    assert "User speech rate" in lines[0]
    assert "Bot speech rate" not in lines[0]
    assert "base_wpm" in lines[0]
    assert "kokoro speed" not in lines[0]


# ---- Pattern parity with iter-114/.../210 -------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_user_wpm_consistency_line(emit, [250.0] * 1000)
    assert "1000 consecutive" in lines[0]
