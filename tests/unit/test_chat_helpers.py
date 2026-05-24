"""Unit tests for examples/_chat_helpers.py.

These tests are pure-Python and never touch pyaudio, mlx-whisper, or kokoro.
They run anywhere pytest runs, including x86_64 Linux CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `examples/` importable as a top-level package.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_helpers import (  # noqa: E402
    SENTENCE_END,
    TurnTimings,
    format_preview_line,
    split_complete_sentences,
    trim_history,
)


class TestSplitCompleteSentences:
    def test_empty_input(self):
        assert split_complete_sentences("") == ([], "")

    def test_no_terminator(self):
        assert split_complete_sentences("hello there") == ([], "hello there")

    def test_single_complete_sentence_in_progress(self):
        # period + space + new content → first is complete, second is in-progress
        complete, rest = split_complete_sentences("Hi there. How are")
        assert complete == ["Hi there."]
        assert rest == "How are"

    def test_multiple_complete_sentences(self):
        complete, rest = split_complete_sentences("One. Two! Three? still")
        assert complete == ["One.", "Two!", "Three?"]
        assert rest == "still"

    def test_only_terminated_sentence_returns_remainder_empty(self):
        # SENTENCE_END is `(?<=[.!?])\s+` — without trailing whitespace+content,
        # the regex doesn't split. This documents the behavior.
        complete, rest = split_complete_sentences("Done.")
        assert complete == []
        assert rest == "Done."

    def test_drops_whitespace_only_sentences(self):
        complete, rest = split_complete_sentences(".  .  hello")
        # `.` followed by whitespace is split; the empty pieces should be dropped.
        # First two splits are "" and "" (after stripping the leading dots).
        # Make sure at least we don't return empty strings as complete sentences.
        for s in complete:
            assert s.strip() != ""

    def test_does_not_split_mid_sentence_with_no_space(self):
        # "abc.def" should not split — there's no whitespace after the period.
        complete, rest = split_complete_sentences("abc.def ghi")
        assert complete == []
        assert rest == "abc.def ghi"

    def test_sentence_end_regex_matches_expected_terminators(self):
        for term in [".", "!", "?"]:
            assert SENTENCE_END.search(f"a{term} b") is not None
        assert SENTENCE_END.search("a, b") is None


class TestTrimHistory:
    def test_empty(self):
        assert trim_history([]) == []

    def test_no_system_short_list(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
        assert trim_history(msgs, max_user_assistant=20) == msgs

    def test_with_system_short_list(self):
        msgs = [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ]
        assert trim_history(msgs, max_user_assistant=20) == msgs

    def test_truncates_old_user_assistant_keeps_system(self):
        system = {"role": "system", "content": "sys"}
        body = [{"role": "user", "content": str(i)} for i in range(50)]
        out = trim_history([system] + body, max_user_assistant=10)
        assert out[0] == system
        assert len(out) == 11  # system + 10
        assert out[-1] == body[-1]
        assert out[1] == body[-10]

    def test_does_not_mutate_input(self):
        msgs = [{"role": "user", "content": str(i)} for i in range(30)]
        original = list(msgs)
        trim_history(msgs, max_user_assistant=5)
        assert msgs == original


class TestTurnTimings:
    def test_defaults(self):
        t = TurnTimings()
        assert t.llm_first_token == 0.0
        assert t.llm_total == 0.0
        assert t.ttfs == 0.0

    def test_first_token_recorded_once(self):
        t = TurnTimings(llm_start=10.0)
        t.record_first_token(11.0)
        t.record_first_token(99.0)  # ignored
        assert t.llm_first_token == pytest.approx(1.0)

    def test_llm_total_uses_stream_done_not_now(self):
        # The whole point of this struct: llm_total reflects only the
        # streaming window, not whatever happened after.
        t = TurnTimings(llm_start=100.0)
        t.record_stream_done(102.5)
        # Even if "now" is much later, llm_total stays put.
        assert t.llm_total == pytest.approx(2.5)

    def test_llm_total_zero_until_stream_done(self):
        t = TurnTimings(llm_start=100.0)
        t.record_first_token(101.0)
        assert t.llm_total == 0.0

    def test_ttfs_recorded_once(self):
        t = TurnTimings(speech_ended_at=10.0)
        assert t.record_ttfs(11.5) is True
        assert t.ttfs == pytest.approx(1.5)
        assert t.record_ttfs(99.0) is False
        assert t.ttfs == pytest.approx(1.5)

    def test_ttfs_no_op_without_speech_ended(self):
        t = TurnTimings()
        assert t.record_ttfs(10.0) is False
        assert t.ttfs == 0.0


class TestFormatPreviewLine:
    def test_short_text_unchanged(self):
        assert format_preview_line("hello", max_width=80) == "hello"

    def test_long_text_truncated_with_ellipsis(self):
        text = "x" * 200
        out = format_preview_line(text, max_width=40, prefix_len=7)
        assert out.endswith("…")
        # Available = max(10, 40 - 7) = 33; truncate to 32 + "…" = 33 chars total.
        assert len(out) == 33

    def test_minimum_width_floor(self):
        # Even with absurdly small terminal, we keep at least ~10 chars usable.
        out = format_preview_line("x" * 50, max_width=5, prefix_len=7)
        assert out.endswith("…")
        assert len(out) == 10
