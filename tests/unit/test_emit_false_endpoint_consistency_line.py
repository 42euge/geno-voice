"""Tests for iter-327 — _emit_false_endpoint_consistency_line.

Seventeenth instance of the session-summary diversity-check pattern, and the
SECOND applied to a BOOLEAN per-turn metric — scans the per-turn
``false_endpoint`` flag (iter-154 — the EOU decision declared the user done
and the agent started responding, but the user actually had more to say, so
the endpoint model / silence VAD fired too early and the user resumed).
Detects 5+ consecutive false endpoints, surfacing that the EOU detector is
systematically pre-empting the user (the fix iter-154 names: raise
chat.vad.silence_duration).

This is the symmetric twin of iter-326's regret-barge scan: both watch the
same failure (EOU pre-empting the user) from the two halves of iter-154's
pre-emption literature — regret = FIRED too early at the audio layer
(barge-conditional, threshold 4); false_endpoint = DECIDED too early at the
turn layer (every turn, threshold 5). The value is already categorical: True
is the single interesting value and False (a clean endpoint, or the
half-duplex default) is the fine state, filtered before the scan exactly like
iter-114's zero-filter. It is NOT inverted: a false endpoint is strictly worse
than its absence.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    _emit_false_endpoint_consistency_line,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line=""):
        lines.append(_strip_ansi(line))

    return emit, lines


# ---- Empty / no-false-endpoint sessions ------------------------------


def test_empty_list_emits_nothing():
    emit, lines = _capture()
    _emit_false_endpoint_consistency_line(emit, [])
    assert lines == []


def test_all_false_emits_nothing():
    """No false endpoint ever (every turn cleanly endpointed, or the
    half-duplex default) → no warning. The common case."""
    emit, lines = _capture()
    _emit_false_endpoint_consistency_line(emit, [False] * 12)
    assert lines == []


# ---- At/above threshold (warning fires) ------------------------------


def test_five_false_endpoints_in_a_row_fires():
    """Default threshold = 5 (the continuous-metric default; false_endpoint
    is NOT barge-conditional, so it does not use iter-326's 4)."""
    emit, lines = _capture()
    _emit_false_endpoint_consistency_line(emit, [True] * 5)
    assert len(lines) == 1
    assert "False-EP run" in lines[0]
    assert "5 consecutive false endpoints" in lines[0]
    assert "iter-154" in lines[0]


def test_four_in_a_row_below_default_threshold():
    """4 in a row → default threshold (5) not met. This is the key
    distinction from iter-326's barge-conditional threshold of 4."""
    emit, lines = _capture()
    _emit_false_endpoint_consistency_line(emit, [True] * 4)
    assert lines == []


# ---- Filter behavior (False interleavings) ---------------------------


def test_false_between_true_doesnt_break_run():
    """A False (clean-endpoint turn) is filtered out before the scan, so it
    does NOT break a run of false endpoints on either side — mirrors
    iter-114's filler-count filter and iter-326's False-filter."""
    emit, lines = _capture()
    # T, F, T, F, T, F, T, T → filtered to [false_ep]*5 → fires.
    _emit_false_endpoint_consistency_line(
        emit, [True, False, True, False, True, False, True, True],
    )
    assert len(lines) == 1
    assert "5 consecutive false endpoints" in lines[0]


def test_few_scattered_false_endpoints_below_threshold():
    """Clean turns (False) are filtered, so they don't separate runs — but
    a SMALL number of false endpoints still can't reach the threshold of 5.
    4 false endpoints scattered among clean turns filter to [false_ep]*4,
    below threshold → silent."""
    emit, lines = _capture()
    _emit_false_endpoint_consistency_line(
        emit,
        [True, False, False, True, False, True, False, False, True],
    )
    assert lines == []


# ---- Custom threshold ------------------------------------------------


def test_threshold_3_catches_smaller_pattern():
    emit, lines = _capture()
    _emit_false_endpoint_consistency_line(emit, [True] * 3, threshold=3)
    assert "3 consecutive false endpoints" in lines[0]


def test_threshold_10_suppresses_default_run():
    emit, lines = _capture()
    _emit_false_endpoint_consistency_line(emit, [True] * 5, threshold=10)
    assert lines == []


# ---- Longest of multiple ---------------------------------------------


def test_longest_run_reported():
    """A single False gap is NOT a separator — filtering merges the runs.
    [T,T, F,F, T,T,T,T,T] → filtered to [false_ep]*7 → fires with 7."""
    emit, lines = _capture()
    _emit_false_endpoint_consistency_line(
        emit, [True, True, False, False] + [True] * 5,
    )
    assert "7 consecutive false endpoints" in lines[0]


# ---- Output formatting -----------------------------------------------


def test_line_has_leading_4_space_indent():
    emit, lines = _capture()
    _emit_false_endpoint_consistency_line(emit, [True] * 5)
    assert lines[0].startswith("    ")


def test_warning_includes_iter_154_attribution():
    emit, lines = _capture()
    _emit_false_endpoint_consistency_line(emit, [True] * 5)
    assert "iter-154" in lines[0]
    assert "false_endpoint" in lines[0]


def test_warning_points_at_silence_duration_fix():
    """The warning must name the actionable knob (raise
    chat.vad.silence_duration), not just describe the symptom — matching
    iter-154's organic-block false-endpoint rate line."""
    emit, lines = _capture()
    _emit_false_endpoint_consistency_line(emit, [True] * 5)
    assert "silence_duration" in lines[0]
    assert "too early" in lines[0]


def test_says_false_endpoints():
    """The unit is false endpoints — both 'false' and 'endpoint' must appear
    so the operator knows these are too-early turn decisions."""
    emit, lines = _capture()
    _emit_false_endpoint_consistency_line(emit, [True] * 5)
    assert "false endpoints" in lines[0]


# ---- Distinctness from iter-326 (the regret-barge twin) --------------


def test_distinct_from_regret_barge_sentinel():
    """iter-326 (regret barge — audio-layer pre-emption) and iter-327
    (false endpoint — decision-layer pre-emption) are near-mirror EOU
    misfires, but distinct: a false endpoint can occur with NO barge at all.
    This pins that iter-327's line is unambiguously about the false_endpoint
    flag, not the regret flag or barge phase."""
    emit, lines = _capture()
    _emit_false_endpoint_consistency_line(emit, [True] * 5)
    assert "false_endpoint" in lines[0]
    # Must NOT claim to be about the regret-barge twin or barge phase.
    assert "barge_in_regret" not in lines[0]
    assert "llm_stream" not in lines[0]
    assert "playback" not in lines[0]


def test_different_default_threshold_from_iter_326():
    """iter-326 (barge-conditional) fires at 4; iter-327 (every-turn) fires
    at 5. A 4-run that WOULD fire under iter-326's threshold must stay
    silent here under the continuous-metric default."""
    emit, lines = _capture()
    _emit_false_endpoint_consistency_line(emit, [True] * 4)
    assert lines == []
    emit, lines = _capture()
    _emit_false_endpoint_consistency_line(emit, [True] * 5)
    assert len(lines) == 1


# ---- Boolean-specific: single interesting value ----------------------


def test_only_true_is_interesting():
    """The metric is already categorical with one interesting value (True).
    A list of all False (the fine state) never fires regardless of length —
    there is no 'opposite' bucket that could trip the scan."""
    emit, lines = _capture()
    _emit_false_endpoint_consistency_line(emit, [False] * 100)
    assert lines == []


# ---- Pattern parity with the prior instances -------------------------


def test_iter_116_helper_handles_large_input():
    """Sanity that the iter-116 _longest_consecutive_run scales —
    1000-element list works."""
    emit, lines = _capture()
    _emit_false_endpoint_consistency_line(emit, [True] * 1000)
    assert "1000 consecutive false endpoints" in lines[0]
