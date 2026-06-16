"""Tests for iter-147 — the talk-mode VAD seam (examples/mic_talk.py).

``mic_talk.py`` is the canned-response voice loop (``gv talk``). Its
``record_utterance`` embeds a silence-gated voice-activity state machine:
above-threshold audio opens a speaking window, below-threshold audio while
speaking starts a trailing-silence timer, and once that timer holds for
``SILENCE_DURATION`` the utterance ends. That logic used to live inline in
the mic loop, was unreachable from a test (the module imported ``pyaudio``
at module scope), and had zero coverage.

iter-147 extracts the per-chunk step into the pure ``vad_step`` function
(no I/O, no clock reads — ``now`` is injected and wall-clock side-effects
stay at the caller) and makes ``pyaudio`` a lazy import. These tests drive
a full utterance lifecycle without pyaudio or mlx_whisper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples import mic_talk as mt  # noqa: E402


# ---- single-step shape -------------------------------------------------


def test_loud_chunk_before_speech_opens_window():
    """Above-threshold audio when not yet speaking starts the utterance."""
    step = mt.vad_step(0.5, speaking=False, silence_start=None, now=10.0)
    assert step.speaking is True
    assert step.started is True
    assert step.append is True
    assert step.done is False
    assert step.silence_start is None


def test_loud_chunk_while_speaking_does_not_restart():
    """Continued speech keeps the window open without re-signaling start."""
    step = mt.vad_step(0.5, speaking=True, silence_start=None, now=10.0)
    assert step.speaking is True
    assert step.started is False
    assert step.append is True
    assert step.done is False


def test_loud_chunk_clears_pending_silence_timer():
    """A loud chunk during trailing silence cancels the silence timer."""
    step = mt.vad_step(0.5, speaking=True, silence_start=5.0, now=10.0)
    assert step.silence_start is None
    assert step.append is True
    assert step.done is False


def test_quiet_chunk_before_speech_is_ignored():
    """Below-threshold audio before any speech is dropped (no append)."""
    step = mt.vad_step(0.0, speaking=False, silence_start=None, now=10.0)
    assert step.speaking is False
    assert step.started is False
    assert step.append is False
    assert step.done is False
    assert step.silence_start is None


def test_quiet_chunk_while_speaking_starts_silence_timer():
    """First quiet chunk after speech stamps the silence start (caller's now)."""
    step = mt.vad_step(0.0, speaking=True, silence_start=None, now=10.0)
    assert step.speaking is True
    assert step.append is True          # trailing silence is kept in the frames
    assert step.silence_start == 10.0
    assert step.done is False


def test_quiet_chunk_continues_silence_timer_below_duration():
    """Silence shorter than SILENCE_DURATION keeps recording, doesn't finish."""
    start = 10.0
    now = start + mt.SILENCE_DURATION - 0.01
    step = mt.vad_step(0.0, speaking=True, silence_start=start, now=now)
    assert step.silence_start == start  # timer is preserved, not reset
    assert step.append is True
    assert step.done is False


def test_quiet_chunk_finishes_after_silence_duration():
    """Silence held >= SILENCE_DURATION signals done."""
    start = 10.0
    now = start + mt.SILENCE_DURATION
    step = mt.vad_step(0.0, speaking=True, silence_start=start, now=now)
    assert step.done is True
    assert step.append is True


def test_threshold_boundary_is_strict():
    """Level exactly at the threshold counts as silence (strict >)."""
    step = mt.vad_step(mt.SILENCE_THRESHOLD, speaking=False, silence_start=None, now=10.0)
    assert step.speaking is False
    assert step.append is False
    # Just above the threshold opens the window.
    step2 = mt.vad_step(mt.SILENCE_THRESHOLD + 1e-6, speaking=False, silence_start=None, now=10.0)
    assert step2.speaking is True
    assert step2.started is True


def test_custom_thresholds_are_honored():
    """Caller-supplied threshold/duration override the module defaults."""
    # A level below the module default but above a low custom threshold opens.
    step = mt.vad_step(
        0.005, speaking=False, silence_start=None, now=10.0,
        silence_threshold=0.001,
    )
    assert step.speaking is True
    # A short custom silence_duration finishes sooner.
    step2 = mt.vad_step(
        0.0, speaking=True, silence_start=10.0, now=10.3,
        silence_duration=0.2,
    )
    assert step2.done is True


# ---- full-utterance sequence (driven like record_utterance's loop) -----


def _run_sequence(levels, *, dt=1.0, threshold=None, duration=3.0):
    """Replay a level sequence through vad_step like record_utterance does.

    Returns (appended_indices, done_at_index, started_at_index) so a test
    can assert which chunks were kept and when the utterance ended.
    """
    kwargs = {}
    if threshold is not None:
        kwargs["silence_threshold"] = threshold
    if duration is not None:
        kwargs["silence_duration"] = duration

    speaking = False
    silence_start = None
    appended = []
    started_at = None
    done_at = None
    for i, level in enumerate(levels):
        now = i * dt
        step = mt.vad_step(level, speaking, silence_start, now, **kwargs)
        speaking = step.speaking
        silence_start = step.silence_start
        if step.started:
            started_at = i
        if step.append:
            appended.append(i)
        if step.done:
            done_at = i
            break
    return appended, done_at, started_at


def test_full_utterance_starts_and_ends_on_silence():
    """Leading silence dropped, speech + trailing silence kept, ends on time."""
    # dt=1.0s, duration=3.0s -> silence must hold 3 chunks to finish.
    levels = [0.0, 0.0] + [0.5] * 5 + [0.0] * 9
    appended, done_at, started_at = _run_sequence(levels)
    # First two silent chunks ignored; speech opens at index 2.
    assert started_at == 2
    assert appended[0] == 2
    # Loud chunks 2..6 plus trailing silence kept until done.
    assert done_at is not None
    # Silence starts at index 7 (now=7.0); fires when now-7 >= 3 -> index 10.
    assert done_at == 10


def test_never_speaking_never_finishes():
    """All-silence input is fully ignored and never signals done."""
    appended, done_at, started_at = _run_sequence([0.0] * 20)
    assert appended == []
    assert done_at is None
    assert started_at is None


def test_speech_resumes_resets_silence_timer():
    """A loud chunk mid-silence resets the timer; utterance keeps going."""
    # Speak, go quiet briefly, speak again, then trail off to silence.
    levels = [0.5] * 3 + [0.0] * 2 + [0.5] * 2 + [0.0] * 9
    appended, done_at, started_at = _run_sequence(levels)
    assert started_at == 0
    # Mid-utterance quiet chunks (3,4) are kept (still speaking) and start a
    # timer, but loud chunks 5,6 reset it before duration (3) elapses, so
    # done fires only after the final silence run. Final silence starts at
    # index 7 (now=7.0); done when now-7 >= 3 -> index 10.
    assert done_at == 10
    # Every chunk up to done is appended (no chunk dropped once speaking).
    assert appended == list(range(0, 11))
