"""Tests for iter-306 — _emit_token_reveal_lag_consistency_line.

Eighteenth instance of the diversity-check pattern. Fifteenth applied to
a CONTINUOUS metric (after iter-128 sentence-length, iter-140 stt-rtf,
iter-141 tts-rtf, iter-142 llm-tps, iter-143 streaming-overlap, iter-208
synth-dispatch, iter-209 eot-overhead, iter-210 bot-wpm, iter-211
max-token-gap, iter-212 ttfs, iter-224 stt-preview-divergence, iter-225
sentence-split-coverage, iter-226 worker-idle-gap, iter-305 ttc) —
buckets the per-turn ``mean_token_reveal_lag`` (iter-071 token-reveal
lag) via ``_token_reveal_lag_bucket`` before scanning. Detects 5+
consecutive turns landing in the "lagging" or "spoiling" bucket,
surfacing a sustained run of on-screen text out of sync with the audio.

THIRD instance with a TWO-SIDED (band) sweet spot — after iter-210
(bot-wpm) and iter-305 (ttc) — and the FIRST whose band straddles ZERO
on a SIGNED metric. The fine state is the MIDDLE band ("synced", |lag| ≤
100ms), and BOTH extremes are flagged but carry OPPOSITE-SIGN diagnoses
(lagging → text behind audio, subtitles late; spoiling → text ahead of
audio, spoils the bot). The run scan keeps the two flagged buckets
distinct — a lagging run and a spoiling run never merge into one.

Note the SIGNED-zero subtlety unique to this sentinel: 0.0 means "metric
not captured" (the play_fn supplied no lag stats), matching the session
summary's own ``!= 0`` collection filter — so an exact 0.0 returns "" and
is dropped, NOT treated as a perfectly-synced turn.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_token_reveal_lag_consistency_line,
    _token_reveal_lag_bucket,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line=""):
        lines.append(_strip_ansi(line))

    return emit, lines


# ---- Bucket boundaries -----------------------------------------------


def test_bucket_exact_zero_returns_empty():
    """0.0 = metric not captured this turn (play_fn supplied no lag
    stats) → empty bucket, filtered by the consumer. This is the
    SIGNED-zero subtlety: 0.0 is the uninstrumented marker, NOT a
    perfectly-synced turn — matching the session summary's ``!= 0``
    collection filter."""
    assert _token_reveal_lag_bucket(0.0) == ""


def test_bucket_synced_band_around_zero():
    """|lag| ≤ 100ms → synced (the sweet spot; the band STRADDLES zero,
    unlike iter-210/305's positive-only bands). A tiny non-zero lag in
    either direction is still synced."""
    assert _token_reveal_lag_bucket(0.05) == "synced"
    assert _token_reveal_lag_bucket(-0.05) == "synced"
    assert _token_reveal_lag_bucket(0.1) == "synced"
    assert _token_reveal_lag_bucket(-0.1) == "synced"


def test_bucket_lagging_above_band():
    """lag > +100ms → lagging (text falls behind audio, subtitles
    late)."""
    assert _token_reveal_lag_bucket(0.1001) == "lagging"
    assert _token_reveal_lag_bucket(0.5) == "lagging"
    assert _token_reveal_lag_bucket(2.0) == "lagging"


def test_bucket_spoiling_below_band():
    """lag < -100ms → spoiling (text leads audio, spoils the bot)."""
    assert _token_reveal_lag_bucket(-0.1001) == "spoiling"
    assert _token_reveal_lag_bucket(-0.5) == "spoiling"
    assert _token_reveal_lag_bucket(-2.0) == "spoiling"


def test_bucket_handles_float_edges():
    """Lag is a signed float — bucket must handle values straddling both
    band boundaries; boundaries align with iter-071's |mean| > 100ms
    yellow flag."""
    assert _token_reveal_lag_bucket(0.0999) == "synced"
    assert _token_reveal_lag_bucket(0.1) == "synced"
    assert _token_reveal_lag_bucket(0.1001) == "lagging"
    assert _token_reveal_lag_bucket(-0.0999) == "synced"
    assert _token_reveal_lag_bucket(-0.1) == "synced"
    assert _token_reveal_lag_bucket(-0.1001) == "spoiling"


# ---- Empty / no-measurement sessions ---------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_token_reveal_lag_consistency_line(emit, [])
    assert lines == []


def test_all_zero_emits_nothing():
    """All turns had no lag measurement (0.0) → no warning."""
    emit, lines = _capture()
    _emit_token_reveal_lag_consistency_line(emit, [0.0] * 10)
    assert lines == []


# ---- "synced" excluded (the sweet spot is never flagged) ------------


def test_long_synced_run_does_not_fire():
    """A 10-turn run inside the sync band is the desired state — never
    flagged. Includes both signs of small lag to prove the band
    straddles zero."""
    emit, lines = _capture()
    _emit_token_reveal_lag_consistency_line(
        emit, [0.05, -0.05, 0.02, -0.08, 0.0, 0.09, -0.01, 0.03, -0.04, 0.0],
    )
    assert lines == []


def test_alternating_synced_and_lagging_only_lagging_counts():
    """[0.05, 0.2, 0.05, 0.2, ...] → after filtering, [lagging] runs of
    1. Below threshold → silent."""
    emit, lines = _capture()
    _emit_token_reveal_lag_consistency_line(
        emit, [0.05, 0.2, 0.05, 0.2, 0.05, 0.2],
    )
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_lagging_in_a_row_fires():
    """Default threshold = 5. Lagging → subtitles late, reveal too
    slow."""
    emit, lines = _capture()
    _emit_token_reveal_lag_consistency_line(emit, [0.2] * 5)
    assert len(lines) == 1
    assert "Token sync drift" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'lagging'" in lines[0]
    assert "behind" in lines[0]
    assert "sooner" in lines[0]
    assert "iter-071" in lines[0]


def test_six_spoiling_in_a_row_fires():
    """Spoiling → text leads audio, spoils the bot (OPPOSITE-SIGN
    diagnosis from lagging — the per-value branch carries the real
    signal)."""
    emit, lines = _capture()
    _emit_token_reveal_lag_consistency_line(emit, [-0.2] * 6)
    assert len(lines) == 1
    assert "6 consecutive" in lines[0]
    assert "'spoiling'" in lines[0]
    assert "leads" in lines[0]
    assert "later" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold not met."""
    emit, lines = _capture()
    _emit_token_reveal_lag_consistency_line(emit, [0.2] * 4)
    assert lines == []


# ---- Filter behavior (synced interleavings) -------------------------


def test_synced_between_lagging_doesnt_break_run():
    """Same precedent as iter-126/128/140/208/209/210/305: filter the
    uninteresting bucket out before scanning. A 'synced' interleaving
    doesn't break a lagging run."""
    emit, lines = _capture()
    # lagging, synced, lagging, synced, lagging, lagging, lagging
    _emit_token_reveal_lag_consistency_line(
        emit, [0.2, 0.05, 0.2, 0.05, 0.2, 0.2, 0.2],
    )
    # Filtered: [lagging]*5 → fires.
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]
    assert "'lagging'" in lines[0]


def test_zero_uncaptured_between_spoiling_doesnt_break_run():
    """An uncaptured turn (0.0 → "") is dropped by the filter, so it
    doesn't break a spoiling run either — same as a synced interleaving.
    Pins the SIGNED-zero-as-uncaptured semantics through the consumer."""
    emit, lines = _capture()
    # spoiling, 0.0(uncaptured), spoiling x4
    _emit_token_reveal_lag_consistency_line(
        emit, [-0.2, 0.0, -0.2, -0.2, -0.2, -0.2],
    )
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]
    assert "'spoiling'" in lines[0]


# ---- Two-sided band: opposite extremes are distinct phases ----------


def test_lagging_and_spoiling_runs_do_not_merge():
    """THE two-sided-band invariant: lagging and spoiling are BOTH
    flagged but sit at OPPOSITE (signed) ends, so they must NOT merge
    into one run. 3 lagging + 3 spoiling → longest run is 3 (below
    threshold), even though 6 turns total were "outside the band"."""
    emit, lines = _capture()
    _emit_token_reveal_lag_consistency_line(
        emit, [0.2, 0.2, 0.2, -0.2, -0.2, -0.2],
    )
    assert lines == []


def test_spoiling_breaks_lagging_run():
    """A spoiling turn between lagging turns breaks the lagging run
    (phase change between two flagged buckets, like iter-210's
    sluggish-breaks-rushed and iter-305's pensive-breaks-rushed)."""
    emit, lines = _capture()
    # 3 lagging, 1 spoiling, 3 lagging → longest lagging run is 3.
    _emit_token_reveal_lag_consistency_line(
        emit, [0.2, 0.2, 0.2, -0.2, 0.2, 0.2, 0.2],
    )
    assert lines == []


def test_longer_spoiling_run_wins_over_shorter_lagging_run():
    """[lagging]*4 + [spoiling]*7 → only the spoiling run passes the
    threshold; warning fires for spoiling with the leads-audio
    diagnosis."""
    emit, lines = _capture()
    _emit_token_reveal_lag_consistency_line(emit, [0.2] * 4 + [-0.2] * 7)
    assert len(lines) == 1
    assert "7 consecutive" in lines[0]
    assert "'spoiling'" in lines[0]
    assert "leads" in lines[0]


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_token_reveal_lag_consistency_line(emit, [0.2] * 3, threshold=3)
    assert "3 consecutive" in lines[0]
    assert "'lagging'" in lines[0]


def test_threshold_10_suppresses_default_5_run():
    emit, lines = _capture()
    _emit_token_reveal_lag_consistency_line(emit, [-0.2] * 5, threshold=10)
    assert lines == []


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_token_reveal_lag_consistency_line(emit, [0.2] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_071_attribution():
    emit, lines = _capture()
    _emit_token_reveal_lag_consistency_line(emit, [-0.2] * 5)
    assert "iter-071" in lines[0]
    assert "mean_token_reveal_lag" in lines[0]


# ---- Pattern parity with iter-114/.../305 -------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_token_reveal_lag_consistency_line(emit, [0.2] * 1000)
    assert "1000 consecutive" in lines[0]
