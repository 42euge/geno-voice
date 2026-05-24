"""Tests for iter-017 ``print_session_summary`` and the time-of-day
abbreviation additions to the iter-016 splitter.

Two small features bundled into one iteration:

1. ``print_session_summary(metrics_list, llm_config, file=...)``
   replaces the inline KeyboardInterrupt summary block in
   ``mic_chat.run_chat``. Now testable via injected file. Also
   uses ``statistics.median`` instead of ``sorted[len//2]``, which
   was the upper median for even-length lists.

2. ``a.m.`` / ``p.m.`` added to ``NON_TERMINATING_ABBREVIATIONS``
   so common time-of-day formats like "9:30 a.m. Time to wake."
   don't split at the abbreviation period.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_helpers import split_complete_sentences  # noqa: E402
from examples._chat_metrics import (  # noqa: E402
    TurnMetrics,
    _median_ms,
    print_session_summary,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _make_metric(
    *,
    stt_time: float = 0.05,
    llm_first_token: float = 0.1,
    tts_time: float = 0.2,
    ttfs: float = 0.3,
    fillers_played: int = 0,
    barge_in: bool = False,
) -> TurnMetrics:
    return TurnMetrics(
        stt_time=stt_time,
        llm_first_token=llm_first_token,
        tts_time=tts_time,
        ttfs=ttfs,
        fillers_played=fillers_played,
        barge_in=barge_in,
        model="test-model",
    )


# ---- _median_ms --------------------------------------------------------------


class TestMedianMs:
    def test_empty_returns_zero(self):
        assert _median_ms([]) == 0.0

    def test_single_value(self):
        assert _median_ms([0.123]) == pytest.approx(123.0)

    def test_odd_length(self):
        assert _median_ms([0.1, 0.2, 0.3]) == pytest.approx(200.0)

    def test_even_length_averages_middle_two(self):
        # iter-017 fix: sorted[len//2] would have returned 0.3
        # (the upper median). statistics.median returns 0.25.
        assert _median_ms([0.1, 0.2, 0.3, 0.4]) == pytest.approx(250.0)

    def test_unsorted_input(self):
        assert _median_ms([0.4, 0.1, 0.3, 0.2]) == pytest.approx(250.0)


# ---- print_session_summary ---------------------------------------------------


class TestSessionSummary:
    def test_empty_list_emits_no_completed_turns_message(self):
        buf = io.StringIO()
        print_session_summary([], {"model": "test-model"}, file=buf)
        out = _strip_ansi(buf.getvalue())
        assert "no completed turns" in out
        # No median lines.
        assert "Median" not in out

    def test_single_turn_includes_all_lines(self):
        buf = io.StringIO()
        m = _make_metric(stt_time=0.05, llm_first_token=0.1, tts_time=0.2, ttfs=0.3)
        print_session_summary([m], {"model": "claude-test"}, file=buf)
        out = _strip_ansi(buf.getvalue())
        assert "Session Summary (1 turn)" in out
        assert "Median STT:       50ms" in out
        assert "Median LLM 1st:   100ms" in out
        assert "Median TTS:       200ms" in out
        assert "Median TTFS:      300ms" in out
        assert "Best TTFS:        300ms" in out
        assert "Model:            claude-test" in out

    def test_three_turns_pluralized_correctly(self):
        buf = io.StringIO()
        ms = [
            _make_metric(stt_time=0.05),
            _make_metric(stt_time=0.10),
            _make_metric(stt_time=0.15),
        ]
        print_session_summary(ms, {"model": "x"}, file=buf)
        out = _strip_ansi(buf.getvalue())
        assert "Session Summary (3 turns)" in out

    def test_even_length_uses_proper_median(self):
        """Regression: with [50, 100, 150, 200] ms STTs, the proper
        median is 125 ms; the buggy ``sorted[len//2]`` gave 150 ms.
        """
        buf = io.StringIO()
        ms = [
            _make_metric(stt_time=0.050),
            _make_metric(stt_time=0.100),
            _make_metric(stt_time=0.150),
            _make_metric(stt_time=0.200),
        ]
        print_session_summary(ms, {"model": "x"}, file=buf)
        out = _strip_ansi(buf.getvalue())
        assert "Median STT:       125ms" in out
        # The upper-median value should NOT appear as the median.
        assert "Median STT:       150ms" not in out

    def test_best_ttfs_is_minimum(self):
        buf = io.StringIO()
        ms = [
            _make_metric(ttfs=0.5),
            _make_metric(ttfs=0.2),
            _make_metric(ttfs=0.7),
        ]
        print_session_summary(ms, {"model": "x"}, file=buf)
        out = _strip_ansi(buf.getvalue())
        # min = 0.2s = 200ms.
        assert "Best TTFS:        200ms" in out

    def test_fillers_line_only_when_any_played(self):
        buf = io.StringIO()
        # No fillers across all turns — line should be absent.
        ms = [_make_metric(fillers_played=0), _make_metric(fillers_played=0)]
        print_session_summary(ms, {"model": "x"}, file=buf)
        assert "Fillers played" not in _strip_ansi(buf.getvalue())

        buf = io.StringIO()
        ms = [_make_metric(fillers_played=0), _make_metric(fillers_played=2)]
        print_session_summary(ms, {"model": "x"}, file=buf)
        out = _strip_ansi(buf.getvalue())
        assert "Fillers played:   2" in out

    def test_barge_ins_line_only_when_any_fired(self):
        buf = io.StringIO()
        ms = [_make_metric(barge_in=False), _make_metric(barge_in=False)]
        print_session_summary(ms, {"model": "x"}, file=buf)
        assert "Barge-ins" not in _strip_ansi(buf.getvalue())

        buf = io.StringIO()
        ms = [_make_metric(barge_in=True), _make_metric(barge_in=False),
              _make_metric(barge_in=True)]
        print_session_summary(ms, {"model": "x"}, file=buf)
        out = _strip_ansi(buf.getvalue())
        # 2 of 3 turns had barge-in.
        assert "Barge-ins:        2" in out

    def test_missing_model_key_falls_back_to_unknown(self):
        buf = io.StringIO()
        print_session_summary([_make_metric()], {}, file=buf)
        out = _strip_ansi(buf.getvalue())
        assert "Model:            unknown" in out

    def test_default_file_writes_to_stdout(self, capsys):
        # Verify the default-file branch reaches stdout via print().
        print_session_summary([_make_metric()], {"model": "x"})
        captured = capsys.readouterr()
        assert "Session Summary" in captured.out


# ---- a.m. / p.m. abbreviation handling --------------------------------------


class TestTimeOfDayAbbreviations:
    def test_am_does_not_split(self):
        complete, rest = split_complete_sentences(
            "It is 9:30 a.m. Time to wake. Now."
        )
        assert complete == ["It is 9:30 a.m. Time to wake."]
        assert rest == "Now."

    def test_pm_does_not_split(self):
        complete, rest = split_complete_sentences("Show is at 8 p.m. Be there.")
        assert complete == []
        assert rest == "Show is at 8 p.m. Be there."

    def test_capitalized_pm_does_not_split(self):
        complete, rest = split_complete_sentences("At 7 P.M. exactly. Then go.")
        assert complete == ["At 7 P.M. exactly."]
        assert rest == "Then go."

    def test_am_pm_dont_break_real_terminators_after(self):
        complete, rest = split_complete_sentences(
            "I left at 5 p.m. We stayed. Until 9 a.m. Yes."
        )
        # "5 p.m." stays glued to "We stayed." — that becomes the
        # one complete sentence (split fires at "stayed.\s+Until").
        # "Until 9 a.m. Yes." stays in the remainder because:
        #   - the a.m. period is an abbreviation (no split)
        #   - the final "Yes." has no trailing whitespace (no match)
        assert complete == ["I left at 5 p.m. We stayed."]
        assert rest == "Until 9 a.m. Yes."

    def test_two_full_sentences_with_am_pm_in_each(self):
        # Trailing space ensures the final "Yes." gets matched too.
        complete, rest = split_complete_sentences(
            "I left at 5 p.m. We stayed. Until 9 a.m. Yes. "
        )
        # Now both abbreviations stay glued to their following text,
        # and both real terminators ("stayed." and "Yes.") split.
        assert complete == [
            "I left at 5 p.m. We stayed.",
            "Until 9 a.m. Yes.",
        ]
        assert rest == ""
