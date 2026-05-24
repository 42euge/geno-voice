"""Tests for iter-046 — bot WPM metric.

Metric 1.13 from docs/perf-metrics-taxonomy.md. Words-per-minute
delivered during bot speech. UX-research sweet spot: 150-180 WPM.

Computed in the worker:
    word_count_total: non-punctuation tokens across sentences
    audio_seconds_total: cumulative len(audio_np) / 24000
ChatLoop:
    bot_wpm = word_count_total / (audio_seconds_total / 60)
"""

from __future__ import annotations

import io
import re
import sys
import time
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
from examples._chat_pipeline import SentenceWorker  # noqa: E402
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


# ---- Default + per-turn print ---------------------------------------------


class TestDefault:
    def test_metrics_default_zero(self):
        assert TurnMetrics().bot_wpm == 0.0

    def test_worker_default_zero(self):
        w = SentenceWorker(
            speaker_factory=lambda: _Speaker(),
            synth_fn=lambda s: (np.full(256, 0.5, dtype=np.float32), []),
            play_fn=_noop_play,
        )
        assert w.word_count_total == 0
        assert w.audio_seconds_total == 0.0


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_zero_omits_line(self):
        m = TurnMetrics(transcript="hi", model="stub", bot_wpm=0.0)
        out = self._capture(m)
        assert "Bot WPM" not in out

    def test_in_target_range_emits_with_target_label(self):
        m = TurnMetrics(transcript="hi", model="stub", bot_wpm=160.0)
        out = self._capture(m)
        assert "Bot WPM" in out
        assert "160" in out
        assert "target 150-180" in out

    def test_too_fast_still_shown(self):
        # >200 = too fast; line still emitted but yellow.
        m = TurnMetrics(transcript="hi", model="stub", bot_wpm=240.0)
        out = self._capture(m)
        assert "240" in out


# ---- Session aggregate ---------------------------------------------------


def _m(wpm):
    return TurnMetrics(ttfs=0.5, bot_wpm=wpm)


class TestSessionSummary:
    def test_no_data_omits(self):
        out = io.StringIO()
        print_session_summary([_m(0.0), _m(0.0)], {"model": "stub"}, file=out)
        assert "Median bot WPM" not in _strip_ansi(out.getvalue())

    def test_some_data_emits_median(self):
        out = io.StringIO()
        print_session_summary(
            [_m(150), _m(170), _m(160)],
            {"model": "stub"}, file=out,
        )
        assert "Median bot WPM:   160" in _strip_ansi(out.getvalue())

    def test_zero_turns_filtered(self):
        out = io.StringIO()
        print_session_summary(
            [_m(0.0), _m(150), _m(170)],
            {"model": "stub"}, file=out,
        )
        # Median of [150, 170] = 160. Without filter would be 150.
        assert "Median bot WPM:   160" in _strip_ansi(out.getvalue())


# ---- Worker token counting ----------------------------------------------


class _Speaker:
    def __init__(self):
        self.captured: list[bytes] = []
    def write(self, data): self.captured.append(data)
    def stop_stream(self): pass
    def close(self): pass


def _noop_play(speaker, audio, tokens, *, is_first_sentence=False, cancel_event=None):
    speaker.write((audio * 32767).astype(np.int16).tobytes())
    return 0.001


class TestWorkerWordCounting:
    def test_token_alignment_counted(self):
        # synth returns audio + token list with non-punct tokens.
        def synth(text):
            audio = np.full(24000, 0.5, dtype=np.float32)  # 1 second
            tokens = [
                {"text": "hello", "start": 0.0, "end": 0.4},
                {"text": ",", "start": 0.4, "end": 0.45},
                {"text": "world", "start": 0.5, "end": 0.9},
                {"text": ".", "start": 0.9, "end": 1.0},
            ]
            return audio, tokens

        w = SentenceWorker(
            speaker_factory=lambda: _Speaker(),
            synth_fn=synth,
            play_fn=_noop_play,
        )
        w.start()
        w.submit("hello, world.")
        w.submit_done()
        w.wait_done(timeout=2.0)
        # 2 non-punct tokens, 1 second of audio.
        assert w.word_count_total == 2
        assert w.audio_seconds_total == pytest.approx(1.0, abs=0.01)

    def test_empty_tokens_falls_back_to_whitespace_split(self):
        # When alignment is missing (kokoro misconfig), use the
        # sentence text's whitespace split as a fallback.
        def synth(text):
            audio = np.full(24000, 0.5, dtype=np.float32)
            return audio, []  # no alignment

        w = SentenceWorker(
            speaker_factory=lambda: _Speaker(),
            synth_fn=synth,
            play_fn=_noop_play,
        )
        w.start()
        w.submit("five words in this sentence")
        w.submit_done()
        w.wait_done(timeout=2.0)
        assert w.word_count_total == 5

    def test_multiple_sentences_accumulate(self):
        def synth(text):
            audio = np.full(12000, 0.5, dtype=np.float32)  # 0.5 sec
            tokens = [
                {"text": w, "start": 0, "end": 0} for w in text.split()
            ]
            return audio, tokens

        w = SentenceWorker(
            speaker_factory=lambda: _Speaker(),
            synth_fn=synth,
            play_fn=_noop_play,
        )
        w.start()
        w.submit("two words")
        w.submit("three more words")
        w.submit_done()
        w.wait_done(timeout=2.0)
        assert w.word_count_total == 5  # 2 + 3
        assert w.audio_seconds_total == pytest.approx(1.0, abs=0.01)


# ---- ChatLoop wires -----------------------------------------------------


def _stt_engine(transcript="hi"):
    engine = SimpleNamespace(_last_text=None, model_repo="stub")
    def transcribe(wav):
        return transcript if wav else None
    return engine, transcribe


def _slow_play(speaker, audio, tokens, *, is_first_sentence=False, cancel_event=None):
    audio_int16 = (audio * 32767).astype(np.int16)
    chunk = 256
    written = 0
    while written < len(audio_int16):
        if cancel_event is not None and cancel_event.is_set():
            break
        end = min(written + chunk, len(audio_int16))
        speaker.write(audio_int16[written:end].tobytes())
        written = end
        time.sleep(0.005)
    return 0.0


def _push_one(mic):
    mic.push(concat(
        make_silence(0.3, rate=RATE),
        make_tone_burst(1.0, rate=RATE, amp=0.3),
        make_silence(1.5, rate=RATE),
    ))


class TestChatLoopWires:
    def test_wpm_computed_when_audio_played(self):
        # Synth: 1 second of audio per sentence, no token alignment
        # → fallback to whitespace split. With one sentence "hi
        # there friend" (3 words) at 1 sec, WPM = 180.
        def synth(s):
            return np.full(24000, 0.5, dtype=np.float32), []

        def llm(messages, config):
            yield "hi there friend. "

        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine, transcribe = _stt_engine()
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=llm,
            llm_config={"model": "stub"},
            synth_fn=synth,
            play_fn=_slow_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # 3 words (split of "hi there friend.") in 1 second = 180 WPM.
        # The whitespace-split treats "friend." as 1 word (with
        # period). That's still 3 words. 3 / (1/60) = 180.
        assert 170 <= result.metrics.bot_wpm <= 190

    def test_zero_when_no_audio(self):
        # Synth returns empty audio → no play, no words counted.
        def synth(s):
            return np.array([], dtype=np.float32), []

        def llm(messages, config):
            yield "hi. "

        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine, transcribe = _stt_engine()
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=llm,
            llm_config={"model": "stub"},
            synth_fn=synth,
            play_fn=_slow_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        assert result.metrics.bot_wpm == 0.0
