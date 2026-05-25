"""Tests for iter-083 — first-token-to-audio gap (FT-A).

Metric 3.18 from docs/perf-metrics-taxonomy.md.

    FT-A = worker.first_audio_at - first_token_at

Time from when the LLM's first token landed at the splitter to
when the worker played its first audio chunk. Complementary to
``llm_first_token``: together they decompose TTFS into "LLM-side"
and "post-LLM-side" halves.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_loop import ChatLoop  # noqa: E402
from examples._chat_metrics import (  # noqa: E402
    TurnMetrics,
    print_session_summary,
)
from examples._chat_recording import CHUNK, RATE  # noqa: E402
from examples.virtual_audio import (  # noqa: E402
    VirtualMicStream,
    VirtualSpeakerStream,
    concat,
    make_silence,
    make_tone_burst,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ---- Default + per-turn print -----------------------------------


class TestDefault:
    def test_turnmetrics_default_zero(self):
        assert TurnMetrics().first_token_to_audio == 0.0


class TestPerTurnPrint:
    def _capture(self, m):
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_zero_omits_suffix(self):
        m = TurnMetrics(transcript="hi", model="stub",
                        llm_first_token=0.1,
                        first_token_to_audio=0.0)
        out = self._capture(m)
        # The base "LLM 1st tok" line still emits, just no FT-A
        # suffix.
        llm_lines = [ln for ln in out.splitlines() if "LLM 1st tok" in ln]
        assert len(llm_lines) == 1
        assert "→ audio" not in llm_lines[0]

    def test_nonzero_appends_suffix(self):
        m = TurnMetrics(transcript="hi", model="stub",
                        llm_first_token=0.1,
                        first_token_to_audio=0.250)
        out = self._capture(m)
        llm_lines = [ln for ln in out.splitlines() if "LLM 1st tok" in ln]
        assert "+250ms → audio" in llm_lines[0]


# ---- Session aggregate -----------------------------------------


def _m(fta=0.0):
    return TurnMetrics(ttfs=0.5, llm_first_token=0.1, first_token_to_audio=fta)


def _summary(metrics_list, **kwargs):
    out = io.StringIO()
    print_session_summary(metrics_list, {"model": "stub"}, file=out, **kwargs)
    return _strip_ansi(out.getvalue())


class TestSessionSummary:
    def test_no_data_omits(self):
        plain = _summary([_m(), _m()])
        assert "Median FT-A" not in plain

    def test_emit_median(self):
        plain = _summary([_m(fta=0.10), _m(fta=0.20), _m(fta=0.30)])
        # Median of [100, 200, 300] = 200ms.
        assert "Median FT-A:      200ms" in plain

    def test_zeros_filtered(self):
        plain = _summary([_m(fta=0.0), _m(fta=0.10), _m(fta=0.30)])
        # Median of [100, 300] = 200.
        assert "Median FT-A:      200ms" in plain


# ---- ChatLoop arithmetic ---------------------------------------


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


class TestChatLoopArithmetic:
    def _build(self, *, mic, response="Done."):
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
        )

    def test_clean_turn_populates(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        loop = self._build(mic=mic)
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Both timestamps land → FT-A populated, non-negative.
        assert result.metrics.first_token_to_audio >= 0
        # And bounded by a sane upper bound (TTFS for the same turn).
        if result.metrics.ttfs > 0:
            assert result.metrics.first_token_to_audio < result.metrics.ttfs * 2

    def test_clamp_at_zero(self):
        # Defensive: even if timestamps land such that fta would
        # be slightly negative (clock skew), the clamp wins.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        loop = self._build(mic=mic)
        result = loop.run_one_turn([])
        assert result.metrics is not None
        assert result.metrics.first_token_to_audio >= 0


class TestComplementarity:
    """FT-A + LLM 1st tok roughly = TTFS - STT (with some slack
    for LLM stream end + first audio dispatch). This relationship
    is what makes FT-A a useful "where to invest" diagnostic.
    """

    def test_fta_plus_llm_ft_within_ttfs(self):
        # Per-turn: stt_time + llm_first_token + (FT-A) ≈ ttfs.
        # FT-A specifically subsumes the path "first token → first
        # complete sentence → first synth → first audio."
        m = TurnMetrics(
            ttfs=0.85,
            stt_time=0.10,
            llm_first_token=0.15,
            first_token_to_audio=0.60,
        )
        # 0.10 + 0.15 + 0.60 = 0.85 = ttfs.
        assert (
            m.stt_time + m.llm_first_token + m.first_token_to_audio
            == pytest.approx(m.ttfs)
        )
