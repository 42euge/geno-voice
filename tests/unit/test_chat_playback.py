"""Tests for examples/_chat_playback.py.

The playback loop writes audio bytes into a speaker stream and reveals
tokens in sync with playback position. We drive it with a
VirtualSpeakerStream and a deterministic clock so the tests measure
the contract (what bytes were written, in what order text appeared)
rather than wall-clock timing.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_playback import (  # noqa: E402
    DEFAULT_PLAY_CHUNK,
    TTS_RATE,
    _is_punct_only,
    play_aligned,
)
from examples.virtual_audio import (  # noqa: E402
    VirtualSpeakerStream,
)


def _tone_audio(seconds: float = 0.1, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    """Float32 audio in [-1, 1] at TTS_RATE, suitable for play_aligned."""
    n = int(seconds * TTS_RATE)
    t = np.arange(n) / TTS_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class StepClock:
    """Returns 0.0, 0.001, 0.002, ... so playback duration > 0 in tests."""

    def __init__(self, step: float = 0.001):
        self._step = step
        self._t = 0.0

    def __call__(self) -> float:
        t = self._t
        self._t += self._step
        return t


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m|\x1b\[2K", "", s)


class TestIsPunctOnly:
    def test_period_is_punct_only(self):
        assert _is_punct_only(".") is True
        assert _is_punct_only("!") is True
        assert _is_punct_only("?") is True
        assert _is_punct_only(",") is True

    def test_word_is_not_punct_only(self):
        assert _is_punct_only("hello") is False
        assert _is_punct_only("hi.") is False  # mixed

    def test_empty_returns_false(self):
        assert _is_punct_only("") is False
        assert _is_punct_only("   ") is False


class TestPlayAlignedAudioWrites:
    def test_all_audio_bytes_written_to_speaker(self):
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        audio = _tone_audio(seconds=0.1)
        elapsed = play_aligned(
            spk,
            audio,
            tokens=[],
            output=io.StringIO(),
            clock=StepClock(),
        )
        # int16 PCM at TTS_RATE → 2 bytes per sample
        expected_bytes = len(audio) * 2
        assert len(spk.captured) == expected_bytes
        assert elapsed > 0

    def test_writes_happen_in_play_chunk_sized_blocks(self):
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        audio = _tone_audio(seconds=0.1)  # ~2400 samples
        play_aligned(
            spk,
            audio,
            tokens=[],
            output=io.StringIO(),
            clock=StepClock(),
            play_chunk=512,
        )
        # All writes except possibly the last should be 512 samples * 2 bytes.
        for w in spk.writes[:-1]:
            assert w == 512 * 2
        # Last write covers the tail (≤ 512 samples).
        assert spk.writes[-1] <= 512 * 2

    def test_empty_audio_writes_nothing_returns_zero_elapsed(self):
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        clock = StepClock()
        elapsed = play_aligned(
            spk,
            np.array([], dtype=np.float32),
            tokens=[],
            output=io.StringIO(),
            clock=clock,
        )
        assert spk.captured == bytearray()
        # clock is sampled twice (start, end) → elapsed is one step.
        assert elapsed == pytest.approx(0.001, abs=0.0001)

    def test_float_audio_scaled_to_int16_correctly(self):
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        # Saturate at +1.0 / -1.0 — should clamp to int16 max/min after scale.
        audio = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        play_aligned(
            spk,
            audio,
            tokens=[],
            output=io.StringIO(),
            clock=StepClock(),
            play_chunk=4,
        )
        decoded = spk.captured_int16
        # 1.0 * 32767 = 32767, -1.0 * 32767 = -32767
        assert decoded[0] == 32767
        assert decoded[1] == -32767
        assert decoded[2] == pytest.approx(16383, abs=2)
        assert decoded[3] == pytest.approx(-16383, abs=2)


class TestPlayAlignedFirstSentencePrefix:
    def test_first_sentence_emits_clear_and_bot_prefix(self):
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        out = io.StringIO()
        play_aligned(
            spk,
            _tone_audio(seconds=0.05),
            tokens=[],
            is_first_sentence=True,
            output=out,
            clock=StepClock(),
        )
        rendered = out.getvalue()
        # Must start with \r + CLEAR_LINE so the prior row is wiped.
        assert rendered.startswith("\r\x1b[2K")
        # Must contain the Bot: prefix (with two spaces of indent).
        visible = _strip_ansi(rendered).lstrip("\r")
        assert visible.startswith("  Bot: ")

    def test_subsequent_sentences_skip_bot_prefix(self):
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        out = io.StringIO()
        play_aligned(
            spk,
            _tone_audio(seconds=0.05),
            tokens=[],
            is_first_sentence=False,
            output=out,
            clock=StepClock(),
        )
        rendered = out.getvalue()
        assert "Bot:" not in rendered


class TestPlayAlignedTokenReveal:
    def test_tokens_emitted_in_order_during_playback(self):
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        out = io.StringIO()
        # 0.1s of audio, three tokens spread across it.
        play_aligned(
            spk,
            _tone_audio(seconds=0.1),
            tokens=[
                {"text": "hello", "start": 0.0},
                {"text": "world", "start": 0.04},
                {"text": "again", "start": 0.08},
            ],
            output=out,
            clock=StepClock(),
        )
        visible = _strip_ansi(out.getvalue())
        # All three tokens appear in order.
        i_hello = visible.find("hello")
        i_world = visible.find("world")
        i_again = visible.find("again")
        assert 0 <= i_hello < i_world < i_again

    def test_punctuation_tokens_use_backspace(self):
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        out = io.StringIO()
        play_aligned(
            spk,
            _tone_audio(seconds=0.1),
            tokens=[
                {"text": "hi", "start": 0.0},
                {"text": ".", "start": 0.02},
            ],
            output=out,
            clock=StepClock(),
        )
        rendered = out.getvalue()
        # Backspace must precede the period to attach it to "hi".
        idx_dot = rendered.find(". ")
        assert idx_dot > 0
        assert rendered[idx_dot - 1] == "\b"

    def test_empty_tokens_dropped(self):
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        out = io.StringIO()
        play_aligned(
            spk,
            _tone_audio(seconds=0.1),
            tokens=[
                {"text": "", "start": 0.0},
                {"text": "   ", "start": 0.02},
                {"text": "real", "start": 0.04},
            ],
            output=out,
            clock=StepClock(),
        )
        visible = _strip_ansi(out.getvalue())
        assert "real" in visible
        # No empty tokens leak through; we shouldn't see two consecutive
        # spaces from the empty/whitespace tokens.
        # (We compare to a stripped baseline — `real ` is the only word.)
        assert visible.strip() == "real"

    def test_words_are_bold_during_loop(self):
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        out = io.StringIO()
        play_aligned(
            spk,
            _tone_audio(seconds=0.1),
            tokens=[{"text": "hello", "start": 0.0}],
            output=out,
            clock=StepClock(),
        )
        rendered = out.getvalue()
        # Word should be wrapped in bold + reset.
        assert "\x1b[1mhello\x1b[0m" in rendered

    def test_tokens_past_audio_duration_flushed_after_loop(self):
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        out = io.StringIO()
        # 0.1s audio, but tokens claim to start at 1.0s — should still emit
        # in the post-loop flush.
        play_aligned(
            spk,
            _tone_audio(seconds=0.1),
            tokens=[{"text": "trailing", "start": 1.0}],
            output=out,
            clock=StepClock(),
        )
        visible = _strip_ansi(out.getvalue())
        assert "trailing" in visible

    def test_post_loop_tokens_emitted_without_bold(self):
        """Quirk-preservation regression test: the original code in
        mic_chat.py only bolds tokens emitted DURING the play loop;
        tokens emitted after the loop (when their start exceeds audio
        duration) are written in plain text. Document that here.
        """
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        out = io.StringIO()
        play_aligned(
            spk,
            _tone_audio(seconds=0.1),
            tokens=[{"text": "trailing", "start": 1.0}],
            output=out,
            clock=StepClock(),
        )
        rendered = out.getvalue()
        # Trailing token is NOT bolded.
        assert "\x1b[1mtrailing\x1b[0m" not in rendered
        # But it IS emitted in plain text, with a trailing space.
        assert "trailing " in rendered


class TestPlayAlignedClockInjection:
    def test_clock_used_for_elapsed_measurement(self):
        spk = VirtualSpeakerStream(rate=TTS_RATE)

        # Two-shot clock: start=10, end=12.5 → elapsed should be 2.5s.
        ticks = iter([10.0, 12.5])

        def clock():
            return next(ticks)

        elapsed = play_aligned(
            spk,
            _tone_audio(seconds=0.05),
            tokens=[],
            output=io.StringIO(),
            clock=clock,
        )
        assert elapsed == pytest.approx(2.5)

    def test_default_output_falls_back_to_stdout(self, capsys):
        spk = VirtualSpeakerStream(rate=TTS_RATE)
        play_aligned(
            spk,
            _tone_audio(seconds=0.05),
            tokens=[{"text": "stdout", "start": 0.0}],
            clock=StepClock(),
        )
        captured = capsys.readouterr()
        # Word lands in real stdout (captured by pytest's capsys).
        assert "stdout" in captured.out


class TestPlayAlignedInteractionWithSpeakerLoopback:
    """Sanity check that the iter-005 loopback wiring still works:
    play_aligned writes to a speaker that's connected to a mic, the
    audio shows up on the mic side, and a downstream consumer can read
    it. This is the building block for iter-009 barge-in tests.
    """

    def test_loopback_speaker_pushes_audio_into_paired_mic(self):
        from examples.virtual_audio import VirtualMicStream
        mic = VirtualMicStream(rate=TTS_RATE, chunk_size=DEFAULT_PLAY_CHUNK)
        spk = VirtualSpeakerStream(rate=TTS_RATE, loopback_to=mic)

        audio = _tone_audio(seconds=0.1)
        play_aligned(
            spk,
            audio,
            tokens=[],
            output=io.StringIO(),
            clock=StepClock(),
        )
        # All bytes the speaker received also showed up on the mic.
        assert mic.frames_buffered == len(audio)
        # And those frames decode back to the same int16 values.
        bytes_back = mic.read(len(audio))
        decoded_back = np.frombuffer(bytes_back, dtype=np.int16)
        decoded_orig = spk.captured_int16
        assert np.array_equal(decoded_back, decoded_orig)
