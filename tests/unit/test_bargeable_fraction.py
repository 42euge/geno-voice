"""Tests for iter-074 — bargeable-time fraction.

Metric 1.19 from docs/perf-metrics-taxonomy.md.

    fraction = intersection(watcher_window, bot_speech_window) / bot_speech_duration

1.0 = barge possible throughout bot speech (architectural default
in current code). <1.0 = bot was uninterruptible for some fraction
of its speech — a regression that would happen if a future change
paused the watcher mid-turn (e.g. during fillers).
"""

from __future__ import annotations

import io
import re
import sys
import threading
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
from examples._chat_pipeline import BargeInWatcher  # noqa: E402
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


# ---- Watcher start_at / stopped_at instrumentation -----------------


class TestWatcherTimestamps:
    def test_defaults_none(self):
        # Construct without starting — both timestamps stay None.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        w = BargeInWatcher(mic=mic, on_speech_detected=lambda: None)
        assert w.started_at is None
        assert w.stopped_at is None

    def test_start_stamps_started_at(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        w = BargeInWatcher(mic=mic, on_speech_detected=lambda: None)
        w.start()
        try:
            assert w.started_at is not None
            assert w.stopped_at is None
        finally:
            w.stop(timeout=1.0)

    def test_stop_stamps_stopped_at(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        w = BargeInWatcher(mic=mic, on_speech_detected=lambda: None)
        w.start()
        w.stop(timeout=1.0)
        assert w.stopped_at is not None
        assert w.started_at is not None
        assert w.stopped_at >= w.started_at

    def test_stop_without_start_no_stamp(self):
        # stop() short-circuits when not started — both stay None.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        w = BargeInWatcher(mic=mic, on_speech_detected=lambda: None)
        w.stop(timeout=1.0)
        assert w.started_at is None
        assert w.stopped_at is None


# ---- Default + per-turn print --------------------------------------


class TestDefault:
    def test_turnmetrics_default_zero(self):
        assert TurnMetrics().bargeable_fraction == 0.0


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
        # 0 = "no audio played" or "watcher lifecycle didn't fire."
        m = TurnMetrics(transcript="hi", model="stub",
                        bargeable_fraction=0.0)
        assert "Bargeable" not in self._capture(m)

    def test_perfect_omits_line(self):
        # 1.0 = healthy default — clutter-free.
        m = TurnMetrics(transcript="hi", model="stub",
                        bargeable_fraction=1.0)
        assert "Bargeable" not in self._capture(m)

    def test_just_under_perfect_emits(self):
        # 99% threshold — just below should emit (regression alarm).
        m = TurnMetrics(transcript="hi", model="stub",
                        bargeable_fraction=0.85)
        out = self._capture(m)
        assert "Bargeable:" in out
        assert "85%" in out
        assert "watcher coverage of bot speech" in out

    def test_low_fraction_emits(self):
        m = TurnMetrics(transcript="hi", model="stub",
                        bargeable_fraction=0.40)
        out = self._capture(m)
        assert "Bargeable:" in out
        assert "40%" in out


# ---- Session aggregate ---------------------------------------------


def _m(frac=0.0):
    return TurnMetrics(ttfs=0.5, bargeable_fraction=frac)


def _summary(metrics_list, **kwargs):
    out = io.StringIO()
    print_session_summary(metrics_list, {"model": "stub"}, file=out, **kwargs)
    return _strip_ansi(out.getvalue())


class TestSessionSummary:
    def test_no_data_omits(self):
        plain = _summary([_m(), _m()])
        assert "Bargeable" not in plain

    def test_all_perfect_omits(self):
        # All 1.0 → no regression → no line.
        plain = _summary([_m(frac=1.0), _m(frac=1.0), _m(frac=1.0)])
        assert "Bargeable" not in plain

    def test_one_below_threshold_emits(self):
        plain = _summary([
            _m(frac=1.0),
            _m(frac=0.50),  # below 99%
            _m(frac=1.0),
        ])
        # Worst = 50%, 1/3 turns below 99%.
        assert "Bargeable:        50% worst (1/3 turns < 99%)" in plain

    def test_multiple_below_threshold(self):
        plain = _summary([
            _m(frac=0.40),
            _m(frac=0.60),
            _m(frac=0.80),
            _m(frac=1.0),
        ])
        # Worst = 40%, 3/4 turns below.
        assert "Bargeable:        40% worst (3/4 turns < 99%)" in plain


# ---- ChatLoop wiring ------------------------------------------------


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
        make_tone_burst(1.0, rate=RATE, amp=0.3),
        make_silence(1.5, rate=RATE),
    ))


class TestChatLoopWiring:
    def test_clean_turn_yields_full_coverage(self):
        # In the standard architecture, watcher.start precedes
        # worker.first_audio_at and watcher.stop is called right
        # after worker.wait_done — fraction should land at 1.0.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine = SimpleNamespace(_last_text=None, model_repo="stub")
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=lambda w: "hi" if w else None,
            llm_stream_fn=_yield_tokens("Done."),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_fast_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Architectural default — watcher covers all bot speech.
        assert result.metrics.bargeable_fraction == pytest.approx(1.0, abs=0.05)

    def test_bounded_in_unit_interval(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        engine = SimpleNamespace(_last_text=None, model_repo="stub")
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=lambda w: "hi" if w else None,
            llm_stream_fn=_yield_tokens("Hi there."),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(),
            play_fn=_fast_play,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        assert 0.0 <= result.metrics.bargeable_fraction <= 1.0
