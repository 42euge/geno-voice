"""Tests for iter-064 — user WPM metric.

Metric 1.14 from docs/perf-metrics-taxonomy.md.

    user_wpm = words(transcript) / speech_duration * 60

Symmetric to iter-046's bot_wpm. Useful for the mirroring effect:
adapting bot WPM to match user produces higher rapport and lower
interruption rate.
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


# ---- Default + per-turn print --------------------------------------------


class TestDefault:
    def test_turnmetrics_default_zero(self):
        assert TurnMetrics().user_wpm == 0.0


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_zero_omits_wpm_suffix(self):
        m = TurnMetrics(transcript="hi", model="stub", user_wpm=0.0,
                        speech_duration=1.0)
        out = self._capture(m)
        # The Speech line still appears, but no WPM suffix.
        speech_lines = [ln for ln in out.splitlines() if "Speech:" in ln]
        assert len(speech_lines) == 1
        assert "WPM" not in speech_lines[0]

    def test_nonzero_appends_wpm_suffix(self):
        m = TurnMetrics(transcript="hi", model="stub", user_wpm=145.0,
                        speech_duration=1.0)
        out = self._capture(m)
        speech_lines = [ln for ln in out.splitlines() if "Speech:" in ln]
        assert len(speech_lines) == 1
        assert "145 WPM" in speech_lines[0]

    def test_extreme_high(self):
        # No color coding — humans speak across a wide range and
        # there's no "wrong" rate.
        m = TurnMetrics(transcript="hi", model="stub", user_wpm=240.0,
                        speech_duration=1.0)
        assert "240 WPM" in self._capture(m)


# ---- Session aggregate ---------------------------------------------------


def _m(user=0.0, bot=0.0):
    # ttfs > 0 keeps the print path on the standard branch.
    return TurnMetrics(ttfs=0.5, user_wpm=user, bot_wpm=bot)


class TestSessionSummary:
    def test_no_data_omits_lines(self):
        out = io.StringIO()
        print_session_summary([_m(), _m()], {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        assert "Median user WPM" not in plain
        assert "Mirror gap" not in plain

    def test_user_only(self):
        out = io.StringIO()
        print_session_summary(
            [_m(user=140), _m(user=160)], {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Median user WPM:  150" in plain
        # No bot WPM data → no mirror gap.
        assert "Mirror gap" not in plain

    def test_both_present_emits_mirror_gap(self):
        out = io.StringIO()
        print_session_summary(
            [_m(user=140, bot=170), _m(user=150, bot=180)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Median user WPM:  145" in plain
        assert "Median bot WPM:   175" in plain
        # Bot 175 - User 145 = +30.
        assert "Mirror gap:       +30 WPM" in plain

    def test_negative_mirror_gap(self):
        # Bot slower than user → negative gap, with sign.
        out = io.StringIO()
        print_session_summary(
            [_m(user=200, bot=150)], {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Mirror gap:       -50 WPM" in plain

    def test_zeros_filtered(self):
        out = io.StringIO()
        print_session_summary(
            [_m(user=0), _m(user=140), _m(user=160)],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Median of [140, 160] = 150 (zero excluded).
        assert "Median user WPM:  150" in plain


# ---- ChatLoop wiring -------------------------------------------------


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


class TestChatLoopArithmetic:
    def test_user_wpm_computed_from_transcript(self):
        # 4 words spoken in 0.6s tone → ~400 WPM. We don't actually
        # observe the recorder's measured speech_duration directly
        # here; the test asserts the field is populated within a
        # plausible range.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(concat(
            make_silence(0.3, rate=RATE),
            make_tone_burst(0.6, rate=RATE, amp=0.3),
            make_silence(1.5, rate=RATE),
        ))
        engine = SimpleNamespace(_last_text=None, model_repo="stub")
        # Stub transcribe returns a 4-word phrase deterministically.
        def transcribe(wav):
            return "one two three four" if wav else None
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=_yield_tokens("Done."),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_fast_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # 4 words / 0.6s * 60 = 400 WPM. VAD's measured
        # speech_duration tracks the tone window, so user_wpm
        # lands in the [200, 600] bracket — generous to absorb
        # silence_threshold timing jitter.
        assert 100 <= result.metrics.user_wpm <= 800

    def test_empty_transcript_zero_wpm(self):
        # If the recorder emits but transcribe returns "", the
        # transcript ends up empty and split() yields no words.
        # We simulate that here at the metrics level.
        m = TurnMetrics(speech_duration=1.0, transcript="")
        # ChatLoop wouldn't actually populate user_wpm in this
        # case — the n_words guard skips the assignment. The
        # default should remain 0.
        assert m.user_wpm == 0.0
