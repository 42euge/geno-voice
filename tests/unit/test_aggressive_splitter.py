"""Tests for iter-088 — aggressive first-sentence splitter.

When ``aggressive_first=True``, ``split_complete_sentences`` also
accepts comma+whitespace as a terminator, but only:
- For the FIRST matching position in the buffer.
- When the buffer before the comma is at least
  ``AGGRESSIVE_MIN_CHARS`` (20) characters.

ChatLoop tracks per-turn state, flipping the flag off as soon as
the first sentence emerges so subsequent splits revert to strict
``.!?`` matching.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_helpers import (  # noqa: E402
    AGGRESSIVE_MIN_CHARS,
    split_complete_sentences,
)
from examples._chat_loop import ChatLoop  # noqa: E402
from examples._chat_recording import CHUNK, RATE  # noqa: E402
from examples.virtual_audio import (  # noqa: E402
    VirtualMicStream,
    VirtualSpeakerStream,
    concat,
    make_silence,
    make_tone_burst,
)


# ---- split_complete_sentences contract --------------------------


class TestSplitterAggressiveFlag:
    def test_default_off_no_comma_split(self):
        # Backwards compat: default behavior unchanged.
        complete, rem = split_complete_sentences(
            "Well, let me think about that for a moment, more "
        )
        assert complete == []
        assert rem.startswith("Well")

    def test_aggressive_on_long_preamble_splits(self):
        # Buffer mid-stream — no trailing whitespace after the
        # final period, so only the comma split fires.
        complete, rem = split_complete_sentences(
            "Well, let me think about that for a moment, "
            "and then I will respond.",
            aggressive_first=True,
        )
        # First comma is right after "Well" (4 chars) — too short
        # to trip. Next comma is after "moment" (full preamble) —
        # that's >20 chars and should split there.
        assert len(complete) == 1
        assert "moment" in complete[0]
        assert complete[0].endswith(",")
        assert "and then" in rem

    def test_aggressive_short_pre_comma_no_split(self):
        # "Sure," is 5 chars before the comma — below the 20-char
        # threshold. No comma split. The trailing period+whitespace
        # is the only sentence terminator, so the strict splitter
        # closes it.
        complete, rem = split_complete_sentences(
            "Sure, let me think more. ",
            aggressive_first=True,
        )
        assert len(complete) == 1
        assert complete[0] == "Sure, let me think more."

    def test_aggressive_period_wins_over_comma(self):
        # If a period appears BEFORE any qualifying comma, the
        # period is preferred.
        complete, rem = split_complete_sentences(
            "Hello there. After this long preamble, more text",
            aggressive_first=True,
        )
        assert len(complete) == 1
        assert complete[0] == "Hello there."
        assert rem.startswith("After this long preamble")

    def test_aggressive_only_first_comma_considered(self):
        # Multiple commas in the buffer. Only the FIRST one (above
        # threshold) is a candidate. After splitting on it, the
        # remainder may have more commas — those don't split (we
        # added at most ONE comma match).
        complete, rem = split_complete_sentences(
            "This is a long preamble that exceeds twenty chars, "
            "and now there is a second part, and a third. ",
            aggressive_first=True,
        )
        # First comma after "twenty chars" splits. Then the "third."
        # period+space creates a second match. So we get 2 complete:
        # one ending in "," and one ending in "."
        assert len(complete) == 2
        assert complete[0].endswith("twenty chars,")
        assert complete[1].endswith("third.")
        assert rem == ""

    def test_aggressive_min_chars_constant_exposed(self):
        # The constant should be importable and equal to 20.
        assert AGGRESSIVE_MIN_CHARS == 20

    def test_empty_buffer(self):
        complete, rem = split_complete_sentences(
            "", aggressive_first=True,
        )
        assert complete == []
        assert rem == ""


# ---- ChatLoop wiring --------------------------------------------


def _const_synth(samples=2048):
    def synth(s):
        return np.full(samples, 0.5, dtype=np.float32), []
    return synth


def _fast_play(speaker, audio, tokens, *, is_first_sentence=False, cancel_event=None):
    audio_int16 = (audio * 32767).astype(np.int16)
    speaker.write(audio_int16.tobytes())
    return 0.0


def _yield_tokens(text):
    import re as _re
    def factory(messages, config):
        for p in _re.findall(r"\S+|\.|!|\?", text):
            yield p + " "
    return factory


def _push_one(mic):
    mic.push(concat(
        make_silence(0.3, rate=RATE),
        make_tone_burst(0.6, rate=RATE, amp=0.3),
        make_silence(1.5, rate=RATE),
    ))


def _build_loop(*, mic, response, aggressive=False):
    engine = SimpleNamespace(_last_text=None, model_repo="stub")
    return ChatLoop(
        mic=mic,
        speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
        stt_engine=engine,
        transcribe_fn=lambda w: "hi" if w else None,
        llm_stream_fn=_yield_tokens(response),
        llm_config={"model": "stub"},
        synth_fn=_const_synth(),
        play_fn=_fast_play,
        aggressive_first_sentence=aggressive,
    )


class TestChatLoopWiring:
    def test_default_off_strict_splits(self):
        # Aggressive disabled — long preamble produces ONE sentence
        # at the period, sentence_min/max reflect that.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        long_preamble = (
            "Well let me think about that for a moment "
            "and then I will respond."
        )
        loop = _build_loop(mic=mic, response=long_preamble, aggressive=False)
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Single sentence — no comma-aware split.
        assert result.metrics.sentences_spoken == 1

    def test_aggressive_on_splits_at_comma(self):
        # Aggressive enabled — the long preamble's comma triggers
        # an early split. Result: more than one sentence submitted.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        long_preamble = (
            "Well let me think about that for a moment, "
            "and then I will respond."
        )
        loop = _build_loop(mic=mic, response=long_preamble, aggressive=True)
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Two units submitted: the comma-split prefix + the
        # period-split tail.
        assert result.metrics.sentences_spoken == 2

    def test_aggressive_only_first_split(self):
        # Two long-preamble commas. Only the FIRST should aggressive-
        # split; the SECOND comma falls under strict (no split). The
        # final period closes the second sentence.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        response = (
            "Here is the first long preamble before splitting, "
            "and here is some more content that should not split, "
            "and here is the conclusion."
        )
        loop = _build_loop(mic=mic, response=response, aggressive=True)
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # 2 sentences: the comma-split first, then the period-split
        # rest. The MIDDLE comma must NOT have split — that would
        # produce 3 sentences.
        assert result.metrics.sentences_spoken == 2

    def test_default_chat_loop_off(self):
        # No explicit kwarg → default off.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine = SimpleNamespace(_last_text=None, model_repo="stub")
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=lambda w: "hi" if w else None,
            llm_stream_fn=_yield_tokens(
                "Well let me think about that for a moment, more text."
            ),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_fast_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Default behavior — single sentence (period-only split).
        assert result.metrics.sentences_spoken == 1
