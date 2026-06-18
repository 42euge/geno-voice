"""Tests for iter-210 — _emit_bot_wpm_consistency_line.

Eleventh instance of the diversity-check pattern. Eighth applied to a
CONTINUOUS metric (after iter-128 sentence-length, iter-140 stt-rtf,
iter-141 tts-rtf, iter-142 llm-tps, iter-143 streaming-overlap,
iter-208 synth-dispatch, iter-209 eot-overhead) — buckets the per-turn
``bot_wpm`` (iter-046 bot speaking rate) via ``_bot_wpm_bucket`` before
scanning. Detects 5+ consecutive turns landing in the "rushed" or
"sluggish" bucket, surfacing a mis-set kokoro ``speed`` knob.

FIRST instance with a TWO-SIDED (band) sweet spot: the fine state is
the MIDDLE band ("natural", 130-200 WPM), and BOTH extremes are
flagged but need OPPOSITE corrections (rushed → lower speed; sluggish
→ raise speed). The run scan keeps the two flagged buckets distinct —
a rushed run and a sluggish run never merge into one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _bot_wpm_bucket,
    _emit_bot_wpm_consistency_line,
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
    """0 WPM = no audio played / no word count → empty bucket
    (filtered by the consumer)."""
    assert _bot_wpm_bucket(0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty. Shouldn't happen (WPM is a
    non-negative rate) but the fallback is cheap."""
    assert _bot_wpm_bucket(-50.0) == ""


def test_bucket_natural_band():
    """130-200 WPM inclusive → natural (the sweet spot, matching the
    per-turn green display)."""
    assert _bot_wpm_bucket(130.0) == "natural"
    assert _bot_wpm_bucket(165.0) == "natural"
    assert _bot_wpm_bucket(200.0) == "natural"


def test_bucket_sluggish_below_band():
    """0 < wpm < 130 → sluggish."""
    assert _bot_wpm_bucket(1.0) == "sluggish"
    assert _bot_wpm_bucket(129.99) == "sluggish"


def test_bucket_rushed_above_band():
    """> 200 WPM → rushed."""
    assert _bot_wpm_bucket(200.01) == "rushed"
    assert _bot_wpm_bucket(400.0) == "rushed"


def test_bucket_handles_float_edges():
    """bot_wpm is a float — bucket must handle values straddling the
    band boundaries."""
    assert _bot_wpm_bucket(129.999) == "sluggish"
    assert _bot_wpm_bucket(130.001) == "natural"
    assert _bot_wpm_bucket(200.0) == "natural"
    assert _bot_wpm_bucket(200.001) == "rushed"


# ---- Empty / no-measurement sessions ---------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_bot_wpm_consistency_line(emit, [])
    assert lines == []


def test_all_zero_emits_nothing():
    """All turns had no audio (0 WPM) → no warning."""
    emit, lines = _capture()
    _emit_bot_wpm_consistency_line(emit, [0.0] * 10)
    assert lines == []


# ---- "natural" excluded (the sweet spot is never flagged) -----------


def test_long_natural_run_does_not_fire():
    """A 10-turn run in the comprehension sweet spot is the desired
    state — never flagged."""
    emit, lines = _capture()
    _emit_bot_wpm_consistency_line(emit, [165.0] * 10)
    assert lines == []


def test_alternating_natural_and_rushed_only_rushed_counts():
    """[165, 250, 165, 250, ...] → after filtering, [rushed] runs of
    1. Below threshold → silent."""
    emit, lines = _capture()
    _emit_bot_wpm_consistency_line(
        emit, [165.0, 250.0, 165.0, 250.0, 165.0, 250.0],
    )
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_rushed_in_a_row_fires():
    """Default threshold = 5. Rushed → lower the speed knob."""
    emit, lines = _capture()
    _emit_bot_wpm_consistency_line(emit, [250.0] * 5)
    assert len(lines) == 1
    assert "Bot speech rate" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'rushed'" in lines[0]
    assert "lower the kokoro speed knob" in lines[0]
    assert "iter-046" in lines[0]


def test_six_sluggish_in_a_row_fires():
    """Sluggish → raise the speed knob (OPPOSITE direction from
    rushed — the per-value branch carries the real signal)."""
    emit, lines = _capture()
    _emit_bot_wpm_consistency_line(emit, [100.0] * 6)
    assert len(lines) == 1
    assert "6 consecutive" in lines[0]
    assert "'sluggish'" in lines[0]
    assert "raise the kokoro speed knob" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold not met."""
    emit, lines = _capture()
    _emit_bot_wpm_consistency_line(emit, [250.0] * 4)
    assert lines == []


# ---- Filter behavior (natural interleavings) ------------------------


def test_natural_between_rushed_doesnt_break_run():
    """Same precedent as iter-126/128/140/208/209: filter the
    uninteresting bucket out before scanning. A 'natural' interleaving
    doesn't break a rushed run."""
    emit, lines = _capture()
    # rushed, natural, rushed, natural, rushed, rushed, rushed
    _emit_bot_wpm_consistency_line(
        emit, [250.0, 165.0, 250.0, 165.0, 250.0, 250.0, 250.0],
    )
    # Filtered: [rushed]*5 → fires.
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]
    assert "'rushed'" in lines[0]


# ---- Two-sided band: opposite extremes are distinct phases ----------


def test_rushed_and_sluggish_runs_do_not_merge():
    """THE two-sided-band invariant: rushed and sluggish are BOTH
    flagged but sit at OPPOSITE ends, so they must NOT merge into one
    run. 3 rushed + 3 sluggish → longest run is 3 (below threshold),
    even though 6 turns total were "outside the band"."""
    emit, lines = _capture()
    _emit_bot_wpm_consistency_line(
        emit, [250.0, 250.0, 250.0, 100.0, 100.0, 100.0],
    )
    assert lines == []


def test_sluggish_breaks_rushed_run():
    """A sluggish turn between rushed turns breaks the rushed run
    (phase change between two flagged buckets, like iter-208/209's
    very_slow-breaks-slow case)."""
    emit, lines = _capture()
    # 3 rushed, 1 sluggish, 3 rushed → longest rushed run is 3.
    _emit_bot_wpm_consistency_line(
        emit, [250.0, 250.0, 250.0, 100.0, 250.0, 250.0, 250.0],
    )
    assert lines == []


def test_longer_sluggish_run_wins_over_shorter_rushed_run():
    """[rushed]*4 + [sluggish]*7 → only the sluggish run passes the
    threshold; warning fires for sluggish with the raise-speed fix."""
    emit, lines = _capture()
    _emit_bot_wpm_consistency_line(emit, [250.0] * 4 + [100.0] * 7)
    assert len(lines) == 1
    assert "7 consecutive" in lines[0]
    assert "'sluggish'" in lines[0]
    assert "raise the kokoro speed knob" in lines[0]


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_bot_wpm_consistency_line(emit, [250.0] * 3, threshold=3)
    assert "3 consecutive" in lines[0]
    assert "'rushed'" in lines[0]


def test_threshold_10_suppresses_default_5_run():
    emit, lines = _capture()
    _emit_bot_wpm_consistency_line(emit, [100.0] * 5, threshold=10)
    assert lines == []


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_bot_wpm_consistency_line(emit, [250.0] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_046_attribution():
    emit, lines = _capture()
    _emit_bot_wpm_consistency_line(emit, [100.0] * 5)
    assert "iter-046" in lines[0]


# ---- Pattern parity with iter-114/.../209 -------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_bot_wpm_consistency_line(emit, [250.0] * 1000)
    assert "1000 consecutive" in lines[0]
