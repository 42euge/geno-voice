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
    VadEvent,
    VadState,
    flush_pending_audio,
    format_preview_line,
    render_preview,
    split_complete_sentences,
    trim_history,
)


class FakeStream:
    """Minimal pyaudio-like stream for testing flush_pending_audio.

    Mirrors the two methods we depend on:
        - get_read_available() -> int
        - read(n_frames, exception_on_overflow=False) -> bytes

    Tracks every read for assertion in tests.
    """

    def __init__(self, available_frames: int = 0):
        self._available = available_frames
        self.reads: list[int] = []
        self.get_calls = 0
        self.raise_on_get = False
        self.raise_on_read = False

    def get_read_available(self) -> int:
        self.get_calls += 1
        if self.raise_on_get:
            raise OSError("input overflowed")
        return self._available

    def read(self, n_frames: int, exception_on_overflow: bool = False) -> bytes:
        if self.raise_on_read:
            raise OSError("input overflowed")
        actual = min(n_frames, self._available)
        self._available -= actual
        self.reads.append(actual)
        return b"\x00\x00" * actual


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


class TestFlushPendingAudio:
    def test_empty_stream_no_reads(self):
        s = FakeStream(available_frames=0)
        drained = flush_pending_audio(s, chunk_size=1024)
        assert drained == 0
        assert s.reads == []

    def test_drains_exactly_full_chunks(self):
        # 3 full chunks + 200 leftover < chunk_size → only 3 chunks consumed.
        s = FakeStream(available_frames=3 * 1024 + 200)
        drained = flush_pending_audio(s, chunk_size=1024)
        assert drained == 3 * 1024
        assert s.reads == [1024, 1024, 1024]
        # 200 frames remain — by design, we don't read partial chunks.
        assert s._available == 200

    def test_stops_when_below_chunk_size(self):
        s = FakeStream(available_frames=500)  # less than one chunk
        drained = flush_pending_audio(s, chunk_size=1024)
        assert drained == 0
        assert s.reads == []
        assert s._available == 500

    def test_max_iterations_safety_cap(self):
        # If the stream lies and always reports plenty available,
        # we must not loop forever.
        class LyingStream(FakeStream):
            def read(self, n_frames, exception_on_overflow=False):
                self.reads.append(n_frames)
                # never decrement _available
                return b"\x00" * (n_frames * 2)

        s = LyingStream(available_frames=10**9)
        drained = flush_pending_audio(s, chunk_size=1024, max_iterations=5)
        assert drained == 5 * 1024
        assert len(s.reads) == 5

    def test_handles_get_read_available_exception(self):
        s = FakeStream(available_frames=10 * 1024)
        s.raise_on_get = True
        drained = flush_pending_audio(s, chunk_size=1024)
        assert drained == 0
        assert s.reads == []

    def test_handles_read_exception_mid_drain(self):
        s = FakeStream(available_frames=10 * 1024)
        s.raise_on_read = True
        drained = flush_pending_audio(s, chunk_size=1024)
        # First get_read_available succeeds, first read raises → bail out clean.
        assert drained == 0
        assert s.reads == []

    def test_chunk_size_parameter_respected(self):
        s = FakeStream(available_frames=8000)
        drained = flush_pending_audio(s, chunk_size=2000)
        assert drained == 8000
        assert s.reads == [2000, 2000, 2000, 2000]


class TestRenderPreview:
    """Bug #4 regression tests — preview must never exceed terminal width.

    The visible width is what matters: ANSI escapes don't take up cells, so
    we strip them when measuring. The line must:
      - start with \\r and the clear-line escape so it overwrites cleanly
      - have visible_len <= max_width (no wraparound)
      - include the prefix verbatim
    """

    @staticmethod
    def _strip_ansi(s: str) -> str:
        import re
        return re.sub(r"\x1b\[[0-9;]*m|\x1b\[2K", "", s)

    def test_short_text_visible_under_width(self):
        from io import StringIO
        buf = StringIO()
        line = render_preview("hello world", max_width=80, file=buf)
        visible = self._strip_ansi(line).lstrip("\r")
        assert visible == "  You: hello world"
        assert len(visible) <= 80

    def test_long_text_truncated_to_fit(self):
        from io import StringIO
        buf = StringIO()
        long = "x" * 500
        line = render_preview(long, max_width=40, file=buf)
        visible = self._strip_ansi(line).lstrip("\r")
        # Must never exceed the requested width.
        assert len(visible) <= 40
        # Must end with the ellipsis indicating truncation.
        assert visible.endswith("…")
        # Must still start with the prefix.
        assert visible.startswith("  You: ")

    def test_starts_with_carriage_return_and_clear(self):
        from io import StringIO
        buf = StringIO()
        line = render_preview("hi", max_width=80, file=buf)
        assert line.startswith("\r\x1b[2K")  # \r + CLEAR_LINE

    def test_writes_to_provided_file_and_flushes(self):
        class CountingBuf:
            def __init__(self):
                self.text = ""
                self.flushes = 0

            def write(self, s):
                self.text += s

            def flush(self):
                self.flushes += 1

        buf = CountingBuf()
        render_preview("test", max_width=80, file=buf)
        assert buf.text != ""
        assert buf.flushes == 1

    def test_dim_off_omits_dim_codes(self):
        from io import StringIO
        buf = StringIO()
        line = render_preview("hi", max_width=80, file=buf, dim=False)
        # When dim=False the body should not be wrapped in dim ANSI.
        assert "\x1b[2m" not in line

    def test_custom_prefix_used_in_width_calculation(self):
        from io import StringIO
        buf = StringIO()
        # Prefix is 12 chars visible. Max width 30 → 18 cells for body.
        line = render_preview("y" * 100, max_width=30, file=buf, prefix=" Listening: ")
        visible = self._strip_ansi(line).lstrip("\r")
        assert visible.startswith(" Listening: ")
        assert len(visible) <= 30
        assert visible.endswith("…")

    def test_repeated_calls_can_overwrite_previous(self):
        # Simulate the live-preview rendering loop: each call should produce
        # output that begins with \r so prior contents on the row are erased.
        from io import StringIO
        buf = StringIO()
        render_preview("hello", max_width=80, file=buf)
        render_preview("hello world", max_width=80, file=buf)
        # Second \r in the buffer means we performed a rewrite-on-line.
        assert buf.getvalue().count("\r") == 2


class TestVadState:
    """Cover the VAD state machine without ever touching audio.

    The "audio" is a sequence of (rms, timestamp) pairs that we feed in.
    Test parameters use the same defaults as mic_chat.py so the tests
    document real behavior, not a synthetic config.
    """

    def _vad(self, **kw):
        return VadState(
            silence_threshold=0.02,
            silence_duration=0.8,
            min_speech_duration=0.3,
            **kw,
        )

    def test_initial_state_idle(self):
        v = self._vad()
        assert v.feed(0.001, 0.0) is VadEvent.IDLE
        assert v.feed(0.0, 0.05) is VadEvent.IDLE
        assert not v.speaking

    def test_first_loud_frame_flips_to_active(self):
        v = self._vad()
        assert v.feed(0.1, 1.0) is VadEvent.ACTIVE
        assert v.speaking
        assert v.speech_start == 1.0

    def test_continuous_speech_keeps_active(self):
        v = self._vad()
        for i in range(20):
            level = 0.1
            t = i * 0.05
            assert v.feed(level, t) is VadEvent.ACTIVE

    def test_quiet_frame_after_speech_starts_silence_timer(self):
        v = self._vad()
        v.feed(0.1, 0.0)
        assert v.feed(0.001, 0.1) is VadEvent.ACTIVE
        assert v.silence_start == 0.1

    def test_brief_silence_does_not_end_utterance(self):
        v = self._vad()
        v.feed(0.1, 0.0)
        v.feed(0.001, 0.1)
        # Below silence_duration; should still be ACTIVE.
        assert v.feed(0.001, 0.5) is VadEvent.ACTIVE
        # And then a loud frame resumes — silence_start clears.
        assert v.feed(0.1, 0.6) is VadEvent.ACTIVE
        assert v.silence_start is None

    def test_full_silence_window_ends_utterance_done_ok(self):
        v = self._vad()
        # 1 second of speech → well above 0.3s min.
        v.feed(0.1, 0.0)
        v.feed(0.1, 1.0)
        # Silence begins at t=1.05.
        v.feed(0.001, 1.05)
        # At t=1.85 we've had >= 0.8s of silence → DONE_OK
        assert v.feed(0.001, 1.85) is VadEvent.DONE_OK
        # State resets after a DONE event.
        assert not v.speaking

    def test_too_short_speech_yields_done_too_short(self):
        v = self._vad()
        v.feed(0.1, 0.0)
        # Only 0.1s of speech (well below 0.3 min), then 0.8s silence.
        v.feed(0.001, 0.1)
        assert v.feed(0.001, 0.9) is VadEvent.DONE_TOO_SHORT
        # Even a TOO_SHORT outcome resets so listening resumes.
        assert not v.speaking
        assert v.last_speech_duration < 0.3

    def test_state_resets_so_next_utterance_works(self):
        v = self._vad()
        # First utterance — too short.
        v.feed(0.1, 0.0)
        v.feed(0.001, 0.1)
        assert v.feed(0.001, 0.9) is VadEvent.DONE_TOO_SHORT

        # Second utterance — long enough.
        v.feed(0.1, 1.0)
        v.feed(0.1, 1.5)
        v.feed(0.001, 1.55)
        assert v.feed(0.001, 2.4) is VadEvent.DONE_OK

    def test_threshold_boundary_is_strictly_greater(self):
        # `level > threshold` so exact equality should NOT count as speech.
        v = self._vad()
        assert v.feed(0.02, 0.0) is VadEvent.IDLE
        assert v.feed(0.0201, 0.05) is VadEvent.ACTIVE

    def test_speech_duration_recorded_on_done(self):
        v = self._vad()
        v.feed(0.1, 0.0)
        v.feed(0.1, 0.5)
        v.feed(0.001, 0.6)
        # 1.5 - 0.6 = 0.9 ≥ 0.8 silence_duration → DONE_OK (avoids
        # 1.4 - 0.6 = 0.7999... float-precision pitfall).
        ev = v.feed(0.001, 1.5)
        assert ev is VadEvent.DONE_OK
        # speech_duration = (now - speech_start - silence_duration)
        #                 = (1.5 - 0.0 - 0.8) = 0.7
        assert abs(v.last_speech_duration - 0.7) < 0.01

    def test_idle_after_brief_blip_below_threshold(self):
        v = self._vad()
        # A single quiet frame and we never enter speaking.
        for i in range(5):
            assert v.feed(0.005, i * 0.1) is VadEvent.IDLE
        assert not v.speaking
        assert v.silence_start is None
