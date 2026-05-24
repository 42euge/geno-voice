"""Tests for iter-023 play_fn signature introspection.

The original SentenceWorker code wrapped each play_fn call in a
``try/except TypeError`` to fall back to a no-cancel-event signature
when the play_fn was iter-008-style (no cancel_event support). That
swallow had a real bug: a play_fn whose BODY raised TypeError would
be retried without cancel_event — calling the function TWICE for
the same sentence, writing partial audio twice.

iter-023 replaces the per-call try/except with one-time
``inspect.signature`` introspection at worker construction. These
tests verify:
  - play_fn with cancel_event in signature → gets it
  - play_fn with **kwargs → gets it (variadic catch)
  - play_fn without cancel_event → doesn't get it (no error)
  - non-inspectable callables (some C extensions / builtins) →
    fall back to no-cancel-event (conservative)
  - play_fn whose body raises TypeError → surfaces ONCE in
    worker.errors, not twice (regression cover for the original bug)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_pipeline import (  # noqa: E402
    SentenceWorker,
    _play_fn_accepts_cancel_event,
)
from examples.virtual_audio import VirtualSpeakerStream  # noqa: E402


# ---- _play_fn_accepts_cancel_event unit tests --------------------------------


class TestSignatureIntrospection:
    def test_explicit_cancel_event_kwarg_detected(self):
        def f(speaker, audio, tokens, *, is_first_sentence=False, cancel_event=None):
            pass
        assert _play_fn_accepts_cancel_event(f) is True

    def test_no_cancel_event_kwarg_not_detected(self):
        def f(speaker, audio, tokens, *, is_first_sentence=False):
            pass
        assert _play_fn_accepts_cancel_event(f) is False

    def test_kwargs_variadic_detected(self):
        # **kwargs accepts any keyword including cancel_event.
        def f(speaker, audio, tokens, **kwargs):
            pass
        assert _play_fn_accepts_cancel_event(f) is True

    def test_lambda_with_cancel_event(self):
        f = lambda speaker, audio, tokens, *, is_first_sentence=False, cancel_event=None: 0.0
        assert _play_fn_accepts_cancel_event(f) is True

    def test_lambda_without_cancel_event(self):
        f = lambda speaker, audio, tokens, *, is_first_sentence=False: 0.0
        assert _play_fn_accepts_cancel_event(f) is False

    def test_uninspectable_callable_falls_back_to_false(self):
        # Some callables (a few C-extension builtins) raise on
        # inspect.signature. Make sure we don't crash — fall back
        # to the conservative no-cancel-event assumption.
        class Uninspectable:
            def __call__(self, speaker, audio, tokens, **kwargs):
                pass
        # Make the signature inspection raise.
        u = Uninspectable()
        # Even instances are inspectable, so force the failure
        # explicitly via a custom __signature__ that raises.
        class Raising:
            __slots__ = ()
            def __call__(self, *a, **kw):
                pass
            @property
            def __signature__(self):
                raise ValueError("can't inspect")
        r = Raising()
        # This should return False without raising.
        assert _play_fn_accepts_cancel_event(r) is False


# ---- SentenceWorker uses introspection result --------------------------------


def _const_synth(samples: int = 1024):
    def synth(s):
        return np.full(samples, 0.5, dtype=np.float32), []
    return synth


def _make_worker(play_fn):
    spk_holder = {"spk": None}

    def factory():
        spk_holder["spk"] = VirtualSpeakerStream(rate=24000)
        return spk_holder["spk"]

    w = SentenceWorker(
        speaker_factory=factory,
        synth_fn=_const_synth(samples=1024),
        play_fn=play_fn,
    )
    return w, spk_holder


class TestWorkerWithIntrospectedPlayFn:
    def test_play_fn_with_cancel_event_receives_it(self):
        received = {"cancel_event": None}

        def play(speaker, audio, tokens, *, is_first_sentence=False, cancel_event=None):
            received["cancel_event"] = cancel_event
            speaker.write((audio * 32767).astype(np.int16).tobytes())
            return 0.0

        w, _ = _make_worker(play)
        w.start()
        w.submit("hello")
        w.submit_done()
        w.wait_done(timeout=5.0)

        # The worker should have passed its internal _cancel_event.
        assert received["cancel_event"] is not None
        # Specifically, it should be a threading.Event-shaped object
        # — same underlying object the watcher would set on barge-in.
        assert hasattr(received["cancel_event"], "is_set")

    def test_play_fn_without_cancel_event_doesnt_receive_it(self):
        # No cancel_event in signature; iter-023 should NOT pass
        # cancel_event, and the call should succeed without error.
        called_with = {"args": None, "kwargs": None}

        def play(speaker, audio, tokens, *, is_first_sentence=False):
            called_with["args"] = (speaker, audio, tokens)
            called_with["kwargs"] = {"is_first_sentence": is_first_sentence}
            speaker.write((audio * 32767).astype(np.int16).tobytes())
            return 0.0

        w, _ = _make_worker(play)
        w.start()
        w.submit("hello")
        w.submit_done()
        w.wait_done(timeout=5.0)

        assert w.errors == []
        assert w.sentences_spoken == 1
        # cancel_event was not in the kwargs.
        assert "cancel_event" not in called_with["kwargs"]

    def test_kwargs_play_fn_receives_cancel_event(self):
        # **kwargs accepts everything, including cancel_event.
        received_kwargs = {}

        def play(speaker, audio, tokens, **kwargs):
            received_kwargs.update(kwargs)
            speaker.write((audio * 32767).astype(np.int16).tobytes())
            return 0.0

        w, _ = _make_worker(play)
        w.start()
        w.submit("hello")
        w.submit_done()
        w.wait_done(timeout=5.0)

        assert "cancel_event" in received_kwargs


# ---- Regression: TypeError in play_fn body surfaces once --------------------


class TestTypeErrorBugRegression:
    def test_buggy_play_fn_raises_typeerror_called_only_once(self):
        """The original bug: a play_fn whose BODY raised TypeError
        triggered the per-call ``try/except TypeError`` fallback,
        causing the function to be called twice — first with
        cancel_event, then without — both raising. Speaker
        received audio twice.

        With iter-023's introspection-once design, the TypeError
        propagates to the outer ``except Exception``, gets recorded
        in ``errors`` exactly once, and the speaker only receives
        the partial audio from one call.
        """
        call_count = {"n": 0}

        def buggy_play(speaker, audio, tokens, *, is_first_sentence=False, cancel_event=None):
            call_count["n"] += 1
            speaker.write((audio * 32767).astype(np.int16).tobytes())
            raise TypeError("real bug — not a signature mismatch")

        w, spk_holder = _make_worker(buggy_play)
        w.start()
        w.submit("hello")
        w.submit_done()
        w.wait_done(timeout=5.0)

        # Called exactly once — the bug used to call it twice.
        assert call_count["n"] == 1
        # Speaker received exactly one chunk's worth of audio
        # (1024 samples × 2 bytes/sample = 2048 bytes), not double.
        assert len(spk_holder["spk"].captured) == 1024 * 2
        # Error captured exactly once.
        assert len(w.errors) == 1
        assert isinstance(w.errors[0], TypeError)
        assert "real bug" in str(w.errors[0])
        # No sentence counted as spoken (play returned False via except).
        assert w.sentences_spoken == 0

    def test_buggy_play_fn_does_not_break_subsequent_sentences(self):
        # Verify the loop continues after a play_fn TypeError,
        # so a working subsequent sentence still plays.
        call_count = {"n": 0}

        def play(speaker, audio, tokens, *, is_first_sentence=False, cancel_event=None):
            call_count["n"] += 1
            speaker.write((audio * 32767).astype(np.int16).tobytes())
            if call_count["n"] == 1:
                raise TypeError("first call boom")
            return 0.0

        w, spk_holder = _make_worker(play)
        w.start()
        w.submit("first")
        w.submit("second")
        w.submit_done()
        w.wait_done(timeout=5.0)

        # First call raised once and was recorded; second call ran cleanly.
        assert call_count["n"] == 2
        assert len(w.errors) == 1
        assert w.sentences_spoken == 1
