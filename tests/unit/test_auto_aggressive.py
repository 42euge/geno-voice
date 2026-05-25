"""Tests for iter-093 — auto-aggressive splitter on stall.

Combines iter-085 (max_token_gap) with iter-088 (aggressive
splitter). When ChatLoop's auto_aggressive_threshold is >0 and
an inter-token gap exceeds it BEFORE the first sentence emerges,
the splitter flips to aggressive mode mid-turn so audio recovers
faster from the stall.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_loop import ChatLoop  # noqa: E402
from examples._chat_recording import CHUNK, RATE  # noqa: E402
from examples.virtual_audio import (  # noqa: E402
    VirtualMicStream,
    VirtualSpeakerStream,
    concat,
    make_silence,
    make_tone_burst,
)


def _const_synth(samples=2048):
    def synth(s):
        return np.full(samples, 0.5, dtype=np.float32), []
    return synth


def _fast_play(speaker, audio, tokens, *, is_first_sentence=False, cancel_event=None):
    audio_int16 = (audio * 32767).astype(np.int16)
    speaker.write(audio_int16.tobytes())
    return 0.0


def _yield_with_stall(text, *, stall_after, stall_duration):
    """Yield tokens with a stall after the Nth token. The stall
    appears as a long gap between consecutive tokens, which the
    iter-093 logic detects.
    """
    import re as _re

    def factory(messages, config):
        tokens = list(_re.findall(r"\S+|\.|!|\?", text))
        for i, p in enumerate(tokens):
            yield p + " "
            if stall_after is not None and i == stall_after - 1:
                time.sleep(stall_duration)

    return factory


def _push_one(mic):
    mic.push(concat(
        make_silence(0.3, rate=RATE),
        make_tone_burst(0.6, rate=RATE, amp=0.3),
        make_silence(1.5, rate=RATE),
    ))


def _build_loop(*, mic, response, threshold=0.0, stall_after=None,
                stall_duration=0.0, aggressive_first=False):
    engine = SimpleNamespace(_last_text=None, model_repo="stub")
    return ChatLoop(
        mic=mic,
        speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
        stt_engine=engine,
        transcribe_fn=lambda w: "hi" if w else None,
        llm_stream_fn=_yield_with_stall(
            response,
            stall_after=stall_after,
            stall_duration=stall_duration,
        ),
        llm_config={"model": "stub"},
        synth_fn=_const_synth(),
        play_fn=_fast_play,
        aggressive_first_sentence=aggressive_first,
        auto_aggressive_threshold=threshold,
    )


# ---- Threshold disabled (default) ------------------------------


class TestThresholdDisabled:
    def test_disabled_default_no_flip(self):
        # threshold=0 → never auto-flips. A long preamble with a
        # stall produces the same single sentence as without stall.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        # Long preamble that would split if aggressive were on,
        # but we have neither static aggressive nor stall threshold.
        loop = _build_loop(
            mic=mic,
            response="Well let me think about that for a moment, "
                     "and then I will respond.",
            threshold=0.0,
            stall_after=2,
            stall_duration=0.30,  # 300ms stall
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Strict splitter — single sentence (period-only split).
        assert result.metrics.sentences_spoken == 1


# ---- Threshold enabled but no stall --------------------------


class TestThresholdEnabledNoStall:
    def test_no_stall_no_flip(self):
        # threshold=0.5 enabled, but token stream is fast — no
        # gap exceeds 500ms, so no flip happens.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        loop = _build_loop(
            mic=mic,
            response="Well let me think about that for a moment, "
                     "and then I will respond.",
            threshold=0.5,
            # No stall.
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # No flip → strict splitter → 1 sentence.
        assert result.metrics.sentences_spoken == 1


# ---- Stall above threshold flips ------------------------------


class TestStallFlipsAggressive:
    def test_stall_after_long_preamble_flips(self):
        # Stall after token 8 — by then we have enough preamble
        # content to satisfy the comma-aware aggressive splitter.
        # Threshold = 200ms, stall = 350ms → flips.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        loop = _build_loop(
            mic=mic,
            response="Well let me think about that for a moment, "
                     "and then I will respond.",
            threshold=0.2,
            stall_after=8,  # ~"moment" position
            stall_duration=0.35,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Aggressive flipped on → comma-split + period-split = 2.
        assert result.metrics.sentences_spoken == 2

    def test_max_token_gap_reflects_stall(self):
        # Sanity: the iter-085 metric still records the gap.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        loop = _build_loop(
            mic=mic,
            response="Well let me think about that for a moment, "
                     "and then I will respond.",
            threshold=0.2,
            stall_after=8,
            stall_duration=0.35,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Gap should be at least the stall duration.
        assert result.metrics.max_token_gap >= 0.30


# ---- Stall below threshold no flip ----------------------------


class TestStallBelowThreshold:
    def test_small_stall_no_flip(self):
        # Threshold = 500ms, stall = 200ms → below, no flip.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        loop = _build_loop(
            mic=mic,
            response="Well let me think about that for a moment, "
                     "and then I will respond.",
            threshold=0.5,
            stall_after=8,
            stall_duration=0.20,  # below 500ms threshold
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Stays strict — single sentence.
        assert result.metrics.sentences_spoken == 1


# ---- Stall after first sentence is ignored -------------------


class TestStallAfterFirstSentenceIgnored:
    def test_stall_after_period_no_flip(self):
        # The flip is gated on first_sentence_at being None. Once
        # a sentence has emerged, a later stall doesn't re-arm.
        # This is desired: by then we have audio playing already.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        # First period emerges at "Hello." (token 1). Stall after
        # the period means first_sentence_at is set; the stall
        # doesn't cause additional comma-splits.
        loop = _build_loop(
            mic=mic,
            response="Hello. Then a long preamble, and more.",
            threshold=0.2,
            stall_after=2,  # After "Hello."
            stall_duration=0.35,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Strict splitter on the SECOND sentence (no aggressive
        # comma-split). Result: "Hello." + "Then a long preamble,
        # and more." = 2 sentences (the period-only split).
        assert result.metrics.sentences_spoken == 2


# ---- Static + auto-aggressive coexist -------------------------


class TestStaticPlusAuto:
    def test_static_aggressive_already_on_no_double_flip(self):
        # If aggressive_first_sentence=True is statically set, the
        # auto-flip gate ``not aggressive_active`` short-circuits
        # — no harm, no extra work, no double-flipping.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        loop = _build_loop(
            mic=mic,
            response="Well let me think about that for a moment, "
                     "and then I will respond.",
            threshold=0.2,
            stall_after=8,
            stall_duration=0.35,
            aggressive_first=True,  # static on
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Static aggressive already produces the comma split → 2
        # sentences; auto flip doesn't change anything.
        assert result.metrics.sentences_spoken == 2
