"""Tests for iter-326 — _emit_regret_barge_consistency_line.

Sixteenth instance of the session-summary diversity-check pattern, and the
FIRST applied to a BOOLEAN per-turn metric — scans the per-turn
``barge_in_regret`` flag (iter-056 — a barge firing within 200ms of bot
first audio, meaning the bot started talking while the user was still going,
so end-of-turn detection fired too early). Detects 4+ consecutive regret
barges, surfacing that the EOU detector is systematically pre-empting the
user (the fix iter-056 names: raise chat.vad.silence_duration).

Unlike the count/continuous instances, the value is already categorical:
True is the single interesting value and False (no barge, or a well-timed
barge) is the fine state, filtered before the scan exactly like iter-114's
zero-filter. The threshold is 4, matching iter-120/iter-325, because regret
barges are a rare subset of barges. It is NOT inverted: a regret barge is
strictly worse than its absence.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_regret_barge_consistency_line,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line=""):
        lines.append(_strip_ansi(line))

    return emit, lines


# ---- Empty / no-regret sessions --------------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_regret_barge_consistency_line(emit, [])
    assert lines == []


def test_all_false_emits_nothing():
    """No regret barge ever (every turn False) → no warning. The common
    no-barge / well-timed-barge session."""
    emit, lines = _capture()
    _emit_regret_barge_consistency_line(emit, [False] * 12)
    assert lines == []


# ---- At/above threshold (warning fires) ------------------------------


def test_four_regret_in_a_row_fires():
    """Default threshold = 4 (shared with iter-120 / iter-325)."""
    emit, lines = _capture()
    _emit_regret_barge_consistency_line(emit, [True] * 4)
    assert len(lines) == 1
    assert "Regret run" in lines[0]
    assert "4 consecutive regret barges" in lines[0]
    assert "iter-056" in lines[0]


def test_three_in_a_row_below_default_threshold():
    """3 in a row → default threshold (4) not met."""
    emit, lines = _capture()
    _emit_regret_barge_consistency_line(emit, [True] * 3)
    assert lines == []


# ---- Filter behavior (False interleavings) ---------------------------


def test_false_between_regret_doesnt_break_run():
    """A False (no-barge / well-timed-barge turn) is filtered out before
    the scan, so it does NOT break a run of regret barges on either side —
    mirrors iter-114's filler-count filter and iter-325's zero-filter."""
    emit, lines = _capture()
    # T, F, T, F, T, T → filtered to [regret]*4 → fires.
    _emit_regret_barge_consistency_line(
        emit, [True, False, True, False, True, True],
    )
    assert len(lines) == 1
    assert "4 consecutive regret barges" in lines[0]


def test_few_scattered_regret_barges_below_threshold():
    """Clean turns (False) are filtered, so they don't separate runs — but
    a SMALL number of regret barges still can't reach the threshold of 4.
    3 regret barges scattered among clean turns filter to [regret]*3,
    below threshold → silent. (Once 4+ regret barges occur, even with clean
    turns between, the run DOES fire — see
    test_false_between_regret_doesnt_break_run.)"""
    emit, lines = _capture()
    _emit_regret_barge_consistency_line(
        emit, [True, False, False, True, False, False, False, True],
    )
    assert lines == []


# ---- Custom threshold ------------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_regret_barge_consistency_line(emit, [True] * 3, threshold=3)
    assert "3 consecutive regret barges" in lines[0]


def test_threshold_10_suppresses_default_run():
    emit, lines = _capture()
    _emit_regret_barge_consistency_line(emit, [True] * 4, threshold=10)
    assert lines == []


# ---- Longest of multiple ---------------------------------------------


def test_longest_run_reported():
    """Two separate runs ([T]*2, gap, [T]*5) → the longer (5) is reported.
    False filters out, but here the runs are genuinely separate only
    because... they're not: filtering merges them. So a single False gap is
    NOT a separator — confirm the merged run length."""
    emit, lines = _capture()
    # [T,T, F,F, T,T,T,T,T] → filtered to [regret]*7 → fires with 7.
    _emit_regret_barge_consistency_line(
        emit, [True, True, False, False] + [True] * 5,
    )
    assert "7 consecutive regret barges" in lines[0]


# ---- Output formatting -----------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_regret_barge_consistency_line(emit, [True] * 4)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_056_attribution():
    emit, lines = _capture()
    _emit_regret_barge_consistency_line(emit, [True] * 4)
    assert "iter-056" in lines[0]
    assert "barge_in_regret" in lines[0]


def test_warning_points_at_silence_duration_fix():
    """The warning must name the actionable knob (raise
    chat.vad.silence_duration), not just describe the symptom — matching
    iter-056's per-session regret-rate line."""
    emit, lines = _capture()
    _emit_regret_barge_consistency_line(emit, [True] * 4)
    assert "silence_duration" in lines[0]
    assert "too early" in lines[0]


def test_says_regret_barges():
    """The unit is regret barges, not all turns — both 'regret' and
    'barge' must appear so the operator knows these are too-early
    interruption events, not a per-turn count over the whole session."""
    emit, lines = _capture()
    _emit_regret_barge_consistency_line(emit, [True] * 4)
    assert "regret barges" in lines[0]


# ---- Distinctness from iter-120 / iter-325 ---------------------------


def test_distinct_from_barge_phase_and_preempted_sentinels():
    """All three (iter-120 phase, iter-325 words, iter-326 regret) are run
    scans over barge turns, but answer different questions. This pins that
    iter-326's line is unambiguously about the regret flag (EOU misfire),
    not phase or word count."""
    emit, lines = _capture()
    _emit_regret_barge_consistency_line(emit, [True] * 4)
    assert "barge_in_regret" in lines[0]
    # Must NOT claim to be about barge phase or pre-empted word count.
    assert "llm_stream" not in lines[0]
    assert "playback" not in lines[0]
    assert "preempted_words" not in lines[0]


# ---- Boolean-specific: single interesting value ----------------------


def test_only_true_is_interesting():
    """The metric is already categorical with one interesting value (True).
    A list of all False (the fine state) never fires regardless of length —
    there is no 'opposite' bucket that could trip the scan."""
    emit, lines = _capture()
    _emit_regret_barge_consistency_line(emit, [False] * 100)
    assert lines == []


# ---- Pattern parity with the prior instances -------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_regret_barge_consistency_line(emit, [True] * 1000)
    assert "1000 consecutive regret barges" in lines[0]
