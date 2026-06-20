"""Tests for iter-305 — _emit_ttc_consistency_line.

Seventeenth instance of the diversity-check pattern. Fourteenth applied
to a CONTINUOUS metric (after iter-128 sentence-length, iter-140
stt-rtf, iter-141 tts-rtf, iter-142 llm-tps, iter-143 streaming-overlap,
iter-208 synth-dispatch, iter-209 eot-overhead, iter-210 bot-wpm,
iter-211 max-token-gap, iter-212 ttfs, iter-224 stt-preview-divergence,
iter-225 sentence-split-coverage, iter-226 worker-idle-gap) — buckets
the per-turn ``time_to_comprehension`` (iter-082 TTC proxy) via
``_ttc_bucket`` before scanning. Detects 5+ consecutive turns landing in
the "rushed" or "pensive" bucket, surfacing a sustained run of the user
responding outside the natural 0.5-5.0s listening window.

SECOND instance with a TWO-SIDED (band) sweet spot — after iter-210
(bot-wpm). The fine state is the MIDDLE band ("natural", 0.5-5.0s), and
BOTH extremes are flagged but carry OPPOSITE diagnoses (rushed → the bot
may be over-explaining; pensive → the bot may be unclear). The run scan
keeps the two flagged buckets distinct — a rushed run and a pensive run
never merge into one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_ttc_consistency_line,
    _ttc_bucket,
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
    """0s = turn 1 / prior turn produced no audio → empty bucket
    (filtered by the consumer)."""
    assert _ttc_bucket(0) == ""


def test_bucket_negative_returns_empty():
    """Defensive: negative input → empty. Shouldn't happen (TTC is a
    non-negative cross-turn gap) but the fallback is cheap."""
    assert _ttc_bucket(-1.0) == ""


def test_bucket_natural_band():
    """0.5-5.0s inclusive → natural (the sweet spot, matching the
    per-turn non-yellow display; bell-curve target 1-3s)."""
    assert _ttc_bucket(0.5) == "natural"
    assert _ttc_bucket(2.0) == "natural"
    assert _ttc_bucket(5.0) == "natural"


def test_bucket_rushed_below_band():
    """0 < ttc < 0.5s → rushed (the user answered almost instantly)."""
    assert _ttc_bucket(0.01) == "rushed"
    assert _ttc_bucket(0.499) == "rushed"


def test_bucket_pensive_above_band():
    """> 5.0s → pensive (the user took a long beat before answering)."""
    assert _ttc_bucket(5.001) == "pensive"
    assert _ttc_bucket(30.0) == "pensive"


def test_bucket_handles_float_edges():
    """TTC is a float — bucket must handle values straddling the band
    boundaries; boundaries align with iter-082's 500ms / 5000ms yellow
    flags."""
    assert _ttc_bucket(0.4999) == "rushed"
    assert _ttc_bucket(0.5) == "natural"
    assert _ttc_bucket(5.0) == "natural"
    assert _ttc_bucket(5.0001) == "pensive"


# ---- Empty / no-measurement sessions ---------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_ttc_consistency_line(emit, [])
    assert lines == []


def test_all_zero_emits_nothing():
    """All turns had no cross-turn measurement (0s) → no warning."""
    emit, lines = _capture()
    _emit_ttc_consistency_line(emit, [0.0] * 10)
    assert lines == []


# ---- "natural" excluded (the sweet spot is never flagged) -----------


def test_long_natural_run_does_not_fire():
    """A 10-turn run in the listening sweet spot is the desired state —
    never flagged."""
    emit, lines = _capture()
    _emit_ttc_consistency_line(emit, [2.0] * 10)
    assert lines == []


def test_alternating_natural_and_rushed_only_rushed_counts():
    """[2.0, 0.2, 2.0, 0.2, ...] → after filtering, [rushed] runs of 1.
    Below threshold → silent."""
    emit, lines = _capture()
    _emit_ttc_consistency_line(
        emit, [2.0, 0.2, 2.0, 0.2, 2.0, 0.2],
    )
    assert lines == []


# ---- At/above threshold (warning fires) -----------------------------


def test_five_rushed_in_a_row_fires():
    """Default threshold = 5. Rushed → the bot may be over-explaining."""
    emit, lines = _capture()
    _emit_ttc_consistency_line(emit, [0.2] * 5)
    assert len(lines) == 1
    assert "User response gap" in lines[0]
    assert "5 consecutive" in lines[0]
    assert "'rushed'" in lines[0]
    assert "over-explaining" in lines[0]
    assert "iter-082" in lines[0]


def test_six_pensive_in_a_row_fires():
    """Pensive → the bot's replies may be unclear (OPPOSITE diagnosis
    from rushed — the per-value branch carries the real signal)."""
    emit, lines = _capture()
    _emit_ttc_consistency_line(emit, [8.0] * 6)
    assert len(lines) == 1
    assert "6 consecutive" in lines[0]
    assert "'pensive'" in lines[0]
    assert "unclear" in lines[0]


def test_below_threshold_does_not_fire():
    """4 in a row → default threshold not met."""
    emit, lines = _capture()
    _emit_ttc_consistency_line(emit, [0.2] * 4)
    assert lines == []


# ---- Filter behavior (natural interleavings) ------------------------


def test_natural_between_pensive_doesnt_break_run():
    """Same precedent as iter-126/128/140/208/209/210: filter the
    uninteresting bucket out before scanning. A 'natural' interleaving
    doesn't break a pensive run."""
    emit, lines = _capture()
    # pensive, natural, pensive, natural, pensive, pensive, pensive
    _emit_ttc_consistency_line(
        emit, [8.0, 2.0, 8.0, 2.0, 8.0, 8.0, 8.0],
    )
    # Filtered: [pensive]*5 → fires.
    assert len(lines) == 1
    assert "5 consecutive" in lines[0]
    assert "'pensive'" in lines[0]


# ---- Two-sided band: opposite extremes are distinct phases ----------


def test_rushed_and_pensive_runs_do_not_merge():
    """THE two-sided-band invariant: rushed and pensive are BOTH flagged
    but sit at OPPOSITE ends, so they must NOT merge into one run. 3
    rushed + 3 pensive → longest run is 3 (below threshold), even though
    6 turns total were "outside the band"."""
    emit, lines = _capture()
    _emit_ttc_consistency_line(
        emit, [0.2, 0.2, 0.2, 8.0, 8.0, 8.0],
    )
    assert lines == []


def test_pensive_breaks_rushed_run():
    """A pensive turn between rushed turns breaks the rushed run (phase
    change between two flagged buckets, like iter-210's
    sluggish-breaks-rushed case)."""
    emit, lines = _capture()
    # 3 rushed, 1 pensive, 3 rushed → longest rushed run is 3.
    _emit_ttc_consistency_line(
        emit, [0.2, 0.2, 0.2, 8.0, 0.2, 0.2, 0.2],
    )
    assert lines == []


def test_longer_pensive_run_wins_over_shorter_rushed_run():
    """[rushed]*4 + [pensive]*7 → only the pensive run passes the
    threshold; warning fires for pensive with the unclear-replies
    diagnosis."""
    emit, lines = _capture()
    _emit_ttc_consistency_line(emit, [0.2] * 4 + [8.0] * 7)
    assert len(lines) == 1
    assert "7 consecutive" in lines[0]
    assert "'pensive'" in lines[0]
    assert "unclear" in lines[0]


# ---- Custom threshold ----------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_ttc_consistency_line(emit, [0.2] * 3, threshold=3)
    assert "3 consecutive" in lines[0]
    assert "'rushed'" in lines[0]


def test_threshold_10_suppresses_default_5_run():
    emit, lines = _capture()
    _emit_ttc_consistency_line(emit, [8.0] * 5, threshold=10)
    assert lines == []


# ---- Output formatting --------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_ttc_consistency_line(emit, [0.2] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_082_attribution():
    emit, lines = _capture()
    _emit_ttc_consistency_line(emit, [8.0] * 5)
    assert "iter-082" in lines[0]
    assert "time_to_comprehension" in lines[0]


# ---- Pattern parity with iter-114/.../226 -------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_ttc_consistency_line(emit, [0.2] * 1000)
    assert "1000 consecutive" in lines[0]
