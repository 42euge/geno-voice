"""
Shared backchannel-cue rotation for the organic turn-taking track.

A backchannel cue ("mhmm", "i see", "right", ...) is the *content* the agent
plays when it decides to backchannel. Two code paths now need that content:

  - the silence-driven ``PLAY_CUE`` path in ``turn_taking.py`` (the original,
    trailing-silence cue), and
  - the mid-speech ``BackchannelMonitor`` (iter-170, backlog #7) — once it
    decides *when* to emit, it still has to pick *which* cue.

Both want the *same* rotation so the agent's continuers don't feel canned, and
both need to remember where they are in the rotation between emits — that index
is a second piece of cross-event state, exactly the kind the pure
``decide_backchannel_timing`` seam can't carry. This module is the single
source of truth for the rotation list and a tiny pure helper to index it, so
``turn_taking.py`` and ``backchannel_monitor.py`` can't drift apart (before this
lap ``CUE_ROTATION`` lived in ``turn_taking.py`` and the monitor returned no cue
at all).

Dependency-free by design (no I/O, no clock reads), like its organic-track
siblings, so tests can load it by file path without dragging in
``session/__init__``'s eager pipecat import (absent on the x86_64 runner).
"""

from __future__ import annotations

__all__ = [
    "CUE_ROTATION",
    "cue_for_index",
]

#: The backchannel cue rotation. Each entry is a ``cue_type`` key into the
#: pre-rendered cue bank (``session/cues/<cue_type>/``, see
#: ``generate_cues.py``). ``mhmm`` appears twice on purpose — it is the most
#: neutral continuer, so it should recur more often than the more pointed
#: cues ("tell me more") that would feel pushy if repeated.
CUE_ROTATION = ["mhmm", "i_see", "right", "go_on", "mhmm", "tell_me_more"]


def cue_for_index(index: int) -> str:
    """Return the cue for rotation position ``index`` (wraps modulo length).

    Pure and total: any integer (including negatives, which Python's modulo
    maps back into range) yields a valid cue, so a caller threading a
    monotonically-increasing counter never has to bounds-check. The caller owns
    the counter; this function just maps it onto the rotation.
    """
    return CUE_ROTATION[index % len(CUE_ROTATION)]
