"""Tests for iter-075 — play_fn kwargs assembly.

iter-071 noted that SentenceWorker._play_clip's play_fn invocation
had grown to a 4-branch if/elif chain (cancel_event × lag_out).
iter-075 collapsed that into a single dynamic kwargs dict.

These tests pin the contract — what kwargs the play_fn actually
receives across each signature shape — so future refactors can't
silently drop a kwarg.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_pipeline import SentenceWorker  # noqa: E402


def _speaker_factory():
    return SimpleNamespace(write=lambda b: None, close=lambda: None)


def _synth(s):
    return np.zeros(8, dtype=np.float32), []


def _capture_kwargs() -> tuple:
    """Returns (play_fn, captured_kwargs_list)."""
    captured: list[dict] = []

    def play(*args, **kwargs):
        captured.append(dict(kwargs))
        return 0.0

    return play, captured


class TestKwargsAssembly:
    """Exercise each play_fn signature shape and check the kwargs."""

    def _run_one(self, play_fn):
        w = SentenceWorker(
            speaker_factory=_speaker_factory,
            synth_fn=_synth,
            play_fn=play_fn,
        )
        w.start()
        w.submit("hi")
        w.submit_done()
        w.wait_done(timeout=2.0)

    def test_minimal_signature(self):
        # Only is_first_sentence — play_fn supports neither
        # cancel_event nor lag_out. Worker should not pass either.
        captured: list[dict] = []

        def play(speaker, audio, tokens, *, is_first_sentence=False):
            captured.append({"is_first_sentence": is_first_sentence})
            return 0.0

        self._run_one(play)
        assert len(captured) == 1
        assert "cancel_event" not in captured[0]
        assert "lag_out" not in captured[0]
        assert "is_first_sentence" in captured[0]

    def test_cancel_only(self):
        # play_fn supports cancel_event but not lag_out — worker
        # passes cancel_event, omits lag_out.
        seen: list[dict] = []

        def play(speaker, audio, tokens, *, is_first_sentence=False,
                 cancel_event=None):
            seen.append({
                "is_first_sentence": is_first_sentence,
                "has_cancel": cancel_event is not None,
            })
            return 0.0

        self._run_one(play)
        assert len(seen) == 1
        assert seen[0]["has_cancel"] is True

    def test_lag_only(self):
        # play_fn supports lag_out but not cancel_event — worker
        # passes lag_out (a dict), omits cancel_event.
        seen: list[dict] = []

        def play(speaker, audio, tokens, *, is_first_sentence=False,
                 lag_out=None):
            seen.append({
                "is_first_sentence": is_first_sentence,
                "lag_out_type": type(lag_out).__name__,
            })
            return 0.0

        self._run_one(play)
        assert len(seen) == 1
        # lag_out should be a fresh dict (the default the worker
        # passes), not None.
        assert seen[0]["lag_out_type"] == "dict"

    def test_both_kwargs(self):
        # Full signature — both cancel_event and lag_out land on
        # the play_fn.
        seen: list[dict] = []

        def play(speaker, audio, tokens, *, is_first_sentence=False,
                 cancel_event=None, lag_out=None):
            seen.append({
                "has_cancel": cancel_event is not None,
                "has_lag_out": lag_out is not None,
            })
            return 0.0

        self._run_one(play)
        assert len(seen) == 1
        assert seen[0]["has_cancel"] is True
        assert seen[0]["has_lag_out"] is True

    def test_var_keyword_signature(self):
        # play_fn with **kwargs accepts everything — worker passes
        # both kwargs.
        seen: list[dict] = []

        def play(speaker, audio, tokens, *, is_first_sentence=False, **kwargs):
            seen.append(dict(kwargs))
            return 0.0

        self._run_one(play)
        assert len(seen) == 1
        # Both extras should land in **kwargs.
        assert "cancel_event" in seen[0]
        assert "lag_out" in seen[0]


class TestRegressionNoTypeError:
    """The 4-branch chain existed because the old code raised
    TypeError when passing unsupported kwargs. The kwargs-dict
    refactor must maintain that property.
    """

    def test_minimal_signature_no_typeerror(self):
        # If the worker tried to pass cancel_event/lag_out to a
        # signature that doesn't accept them, this test would
        # raise. The errors list captures any exception inside
        # _play_clip.
        def play(speaker, audio, tokens, *, is_first_sentence=False):
            return 0.0

        w = SentenceWorker(
            speaker_factory=_speaker_factory,
            synth_fn=_synth,
            play_fn=play,
        )
        w.start()
        w.submit("hi")
        w.submit_done()
        w.wait_done(timeout=2.0)
        # No TypeError — empty errors list.
        assert w.errors == []
