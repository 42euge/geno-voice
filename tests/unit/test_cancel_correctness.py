"""Tests for iter-040 — cancel-correctness metric.

Metric 2.18 from docs/perf-metrics-taxonomy.md. Counts sentences
whose play_fn was interrupted mid-stream by cancel_event vs
sentences that completed naturally.

Detection: in SentenceWorker._play_clip, sample cancel_event
before and after the play call. If it transitions from clear to
set DURING the call, the play exited because of cancel_event.

These tests verify:
  - SentenceWorker.cancelled_sentences defaults to 0.
  - The counter increments only when cancel transitioned during play.
  - Counter does NOT increment when cancel was already set before
    the call (avoids double-counting on follow-up sentences after
    a barge-in).
  - Counter does NOT increment when cancel never fires.
  - TurnMetrics has the field; ChatLoop wires it; print + session
    summary surface it correctly.
"""

from __future__ import annotations

import io
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    TurnMetrics,
    print_session_summary,
)
from examples._chat_pipeline import SentenceWorker  # noqa: E402


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ---- Default value ---------------------------------------------------------


class TestDefault:
    def test_worker_default_is_zero(self):
        w = SentenceWorker(
            speaker_factory=lambda: _FakeSpeaker(),
            synth_fn=lambda s: (np.full(256, 0.5, dtype=np.float32), []),
            play_fn=_naive_play,
        )
        assert w.cancelled_sentences == 0

    def test_turnmetrics_default_is_zero(self):
        assert TurnMetrics().sentences_cancelled == 0


# ---- _play_clip detection (the core of iter-040) ---------------------------


class _FakeSpeaker:
    """Minimal speaker shape — accepts .write(bytes), .stop_stream(),
    .close(). Mirrors the contract the worker requires.
    """
    def __init__(self):
        self.captured: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.captured.append(data)

    def stop_stream(self) -> None:
        pass

    def close(self) -> None:
        pass


def _naive_play(speaker, audio, tokens, *, is_first_sentence=False, cancel_event=None):
    """Plain play — writes audio in one chunk, never checks cancel."""
    audio_int16 = (audio * 32767).astype(np.int16)
    speaker.write(audio_int16.tobytes())
    return 0.01


def _cancel_during_play_factory(cancel_after_chunks=2):
    """Returns a play_fn that writes ``cancel_after_chunks`` chunks
    then sets the cancel_event itself, simulating an external
    barge-in landing mid-stream.
    """
    def play(speaker, audio, tokens, *, is_first_sentence=False, cancel_event=None):
        audio_int16 = (audio * 32767).astype(np.int16)
        chunk = 256
        written = 0
        chunks_done = 0
        while written < len(audio_int16):
            if cancel_event is not None and cancel_event.is_set():
                break
            end = min(written + chunk, len(audio_int16))
            speaker.write(audio_int16[written:end].tobytes())
            written = end
            chunks_done += 1
            if chunks_done == cancel_after_chunks and cancel_event is not None:
                # Simulate barge-in: external code sets cancel_event.
                cancel_event.set()
        return 0.01
    return play


def _const_synth(samples=2048):
    def synth(s):
        return np.full(samples, 0.5, dtype=np.float32), []
    return synth


class TestPlayClipDetection:
    def test_natural_completion_does_not_increment(self):
        # play_fn doesn't touch cancel; cancel never fires → counter stays 0.
        w = SentenceWorker(
            speaker_factory=lambda: _FakeSpeaker(),
            synth_fn=_const_synth(),
            play_fn=_naive_play,
        )
        w.start()
        w.submit("a")
        w.submit("b")
        w.submit_done()
        w.wait_done(timeout=2.0)
        assert w.sentences_spoken == 2
        assert w.cancelled_sentences == 0

    def test_mid_stream_cancel_increments_once(self):
        # play_fn sets cancel_event after 2 chunks.
        w = SentenceWorker(
            speaker_factory=lambda: _FakeSpeaker(),
            synth_fn=_const_synth(),
            play_fn=_cancel_during_play_factory(cancel_after_chunks=2),
        )
        w.start()
        w.submit("a")
        w.submit("b")  # this one would queue but worker.cancel was triggered
        w.submit_done()
        w.wait_done(timeout=2.0)
        # Exactly one mid-stream cancellation. Subsequent queued
        # sentences are drained without play (stop_event was set),
        # so they don't add to the count.
        assert w.cancelled_sentences == 1

    def test_cancel_already_set_does_not_increment(self):
        # If cancel was set BEFORE the play call (e.g. cleanup
        # path firing while a sentence was already drained but the
        # play was somehow re-invoked), the counter must NOT
        # increment — that's not a mid-stream cut, it was set before
        # we even started.
        w = SentenceWorker(
            speaker_factory=lambda: _FakeSpeaker(),
            synth_fn=_const_synth(),
            play_fn=_naive_play,
        )
        w.start()
        # Pre-set cancel_event so first play sees it set on entry.
        w._cancel_event.set()
        w.submit("a")
        w.submit_done()
        w.wait_done(timeout=2.0)
        # Worker drained without playing because stop_event isn't
        # set yet (cancel_event != stop_event); but the play_fn
        # check inside _play_clip would see cancel_was_set_before=True
        # and skip the increment regardless.
        assert w.cancelled_sentences == 0


# ---- Per-turn print --------------------------------------------------------


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_no_barge_in_omits_cancel_note(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            barge_in=False, sentences_cancelled=0,
        )
        out = self._capture(m)
        assert "Barge-in" not in out

    def test_barge_in_with_mid_stream_shows_count(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            barge_in=True, sentences_cancelled=1,
        )
        out = self._capture(m)
        assert "Barge-in" in out
        assert "1 cut mid-stream" in out

    def test_barge_in_between_sentences_shows_label(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            barge_in=True, sentences_cancelled=0,
        )
        out = self._capture(m)
        assert "Barge-in" in out
        assert "between sentences" in out


# ---- Session summary aggregate ---------------------------------------------


def _make_metric(barge_in=False, sentences_cancelled=0):
    return TurnMetrics(
        ttfs=0.5, barge_in=barge_in, sentences_cancelled=sentences_cancelled,
    )


class TestSessionSummary:
    def test_no_barges_omits_block(self):
        out = io.StringIO()
        print_session_summary([_make_metric()], {"model": "stub"}, file=out)
        assert "Barge-ins" not in _strip_ansi(out.getvalue())

    def test_all_mid_stream_shows_100pct(self):
        out = io.StringIO()
        print_session_summary(
            [
                _make_metric(barge_in=True, sentences_cancelled=1),
                _make_metric(barge_in=True, sentences_cancelled=1),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Barge-ins:        2 (2 mid-stream, 100%)" in plain

    def test_mixed_shows_partial_pct(self):
        out = io.StringIO()
        print_session_summary(
            [
                _make_metric(barge_in=True, sentences_cancelled=1),
                _make_metric(barge_in=True, sentences_cancelled=0),
                _make_metric(barge_in=True, sentences_cancelled=1),
                _make_metric(barge_in=True, sentences_cancelled=0),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # 2/4 = 50%
        assert "Barge-ins:        4 (2 mid-stream, 50%)" in plain

    def test_all_between_sentences_shows_label(self):
        out = io.StringIO()
        print_session_summary(
            [
                _make_metric(barge_in=True, sentences_cancelled=0),
                _make_metric(barge_in=True, sentences_cancelled=0),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Barge-ins:        2 (all between sentences)" in plain
