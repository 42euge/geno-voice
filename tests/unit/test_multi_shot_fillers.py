"""Tests for iter-087 — multi-shot fillers.

The original iter-011 design fired ONE filler per turn after
``idle_threshold`` elapsed. iter-087 enables up to MAX_FILLERS=2
fillers per turn, picking from clips not yet played to avoid
"umm umm" repetition.

Key invariants:
- Single-filler-config sessions stay single-shot (no clip to swap to).
- Multi-filler-config sessions can fire 2 if LLM stays slow.
- Distinct clips are picked across the 2 fires.
- Hard cap at 2 — even with 5 clips and a slow LLM, no third fire.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_pipeline import SentenceWorker  # noqa: E402


def _speaker_factory():
    return SimpleNamespace(write=lambda b: None, close=lambda: None)


def _zero_synth(s):
    return np.zeros(8, dtype=np.float32), []


def _no_op_play(*args, **kwargs):
    return 0.0


def _make_worker(*, fillers, picker, idle_threshold=0.02):
    return SentenceWorker(
        speaker_factory=_speaker_factory,
        synth_fn=_zero_synth,
        play_fn=_no_op_play,
        fillers=fillers,
        idle_threshold=idle_threshold,
        filler_picker=picker,
    )


def _run_for(w, *, wait_ms=120):
    """Start the worker, wait for the timeout window to fire, then
    submit_done. Wait long enough for multi-shot fires to settle.
    """
    w.start()
    def _later():
        time.sleep(wait_ms / 1000)
        w.submit_done()
    threading.Thread(target=_later, daemon=True).start()
    w.wait_done(timeout=2.0)


# ---- Single-clip config preserves single-shot behavior --------


class TestSingleClipConfig:
    def test_only_one_fire_with_one_clip(self):
        # Single clip available — multi-shot has nothing to pick
        # for the second fire, so it stops at one.
        clip = (np.zeros(8, dtype=np.float32), [])
        w = _make_worker(fillers=[clip], picker=lambda lst: lst[0])
        _run_for(w, wait_ms=120)
        assert w.fillers_played == 1


# ---- Multi-clip config + slow LLM = multi-shot ----------------


class TestMultiShot:
    def test_two_fires_when_lots_of_time(self):
        # 3 clips, generous wait window — both filler windows
        # should fit, producing 2 fires.
        clips = [
            (np.full(8, 0.1, dtype=np.float32), []),
            (np.full(8, 0.2, dtype=np.float32), []),
            (np.full(8, 0.3, dtype=np.float32), []),
        ]
        # Picker always picks the first available — observable
        # ordering: first fire = clips[0], second fire = clips[1].
        w = _make_worker(fillers=clips, picker=lambda lst: lst[0])
        _run_for(w, wait_ms=120)
        # 120ms / 20ms = 6 windows. Cap is 2 — exactly 2 fires.
        assert w.fillers_played == 2
        # Both clips should have been picked (different ids).
        # Final last_filler_id is whichever fired last.
        assert w.last_filler_id == id(clips[1])

    def test_distinct_clips(self):
        # 3 clips. Picker picks the first available. After the
        # first fire (clips[0]), the available list is [clips[1],
        # clips[2]] → picker picks clips[1]. Verify the two fires
        # were genuinely different.
        clips = [
            (np.full(8, 0.1, dtype=np.float32), []),
            (np.full(8, 0.2, dtype=np.float32), []),
            (np.full(8, 0.3, dtype=np.float32), []),
        ]
        w = _make_worker(fillers=clips, picker=lambda lst: lst[0])
        _run_for(w, wait_ms=120)
        assert w.fillers_played == 2
        # Distinct clips: last_filler_id != first picked id.
        # We know first was clips[0], last is clips[1].
        assert w.last_filler_id == id(clips[1])
        assert w.last_filler_id != id(clips[0])


# ---- Cap honored ------------------------------------------------


class TestMaxFillersCap:
    def test_no_third_fire(self):
        # 5 clips + long wait window. Even with plenty of time,
        # MAX_FILLERS = 2 means exactly 2 fires.
        clips = [
            (np.full(8, 0.1 * (i + 1), dtype=np.float32), [])
            for i in range(5)
        ]
        w = _make_worker(fillers=clips, picker=lambda lst: lst[0])
        # Long wait so 5+ filler windows would fit if the cap
        # weren't enforced.
        _run_for(w, wait_ms=200)
        # Cap enforced regardless of available time.
        assert w.fillers_played == 2


# ---- Distinct-clip enforcement: same id never picked twice ---


class TestDistinctIdEnforcement:
    def test_same_clip_id_never_picked_twice(self):
        # If the picker accidentally returns the same clip again
        # (e.g., a buggy custom picker), the available-list filter
        # at the worker level prevents the duplicate from firing.
        # Verify by injecting a picker that always returns the
        # FIRST item (which would be the same clip every time
        # without the filter).
        clips = [
            (np.full(8, 0.1, dtype=np.float32), []),
            (np.full(8, 0.2, dtype=np.float32), []),
        ]
        seen_clips: list[int] = []

        def picker(lst):
            chosen = lst[0]
            seen_clips.append(id(chosen))
            return chosen

        w = _make_worker(fillers=clips, picker=picker)
        _run_for(w, wait_ms=120)
        # Picker called twice (one per filler fire), with
        # different inputs (the worker filters out already-played).
        assert w.fillers_played == 2
        # The two picks were distinct clips.
        assert seen_clips[0] != seen_clips[1]


# ---- Sentence arrival cancels further fillers --------------------
#
# The "sentence-spoken disables further filler timeouts" invariant
# is implicit in the ``use_filler_timeout`` condition (which gates
# on ``sentences_spoken == 0``). This was already true pre-iter-087
# and isn't changed by the refactor. A direct test would race the
# inter-filler timeout against the test thread's sentence submit;
# instead we cover that behavior structurally via the existing
# tests in test_fillers.py and test_filler_false_positive.py.
