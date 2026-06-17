"""
Monologue clock for the organic turn-taking track — the VAD-event driver that
feeds ``BackchannelMonitor.observe`` (backlog item #7 in
``docs/research/organic-turn-taking.md``).

iter-170 shipped the stateful ``BackchannelMonitor`` and iter-171 gave it the
cue rotation, so the monitor answers both *when* and *which*. But its
``observe(*, now, monologue_start_at, pause_secs)`` consumes two derived
quantities the monitor cannot compute itself:

  - ``monologue_start_at`` — when the user's *current monologue* began, and
  - ``pause_secs`` — the within-speech pause right now.

Neither is a raw clock read. A "monologue" is **not** the same as one VAD
speech segment: a person mid-thought pauses at clause boundaries ("I was
thinking… [0.5s] …about the deadline") without yielding the floor, and those
brief pauses are *exactly* the backchannel opportunities the monitor watches
for. So the monologue must survive a short stop→start gap and reset only when
the gap is turn-end-sized (the user actually handed the floor back). And
``pause_secs`` is the live gap since the last speech frame, zero while the user
is talking. Deriving both from the VAD ``started``/``stopped`` event stream is
cross-event state the pure seam can't carry — the same gap iter-156's
``UtteranceBuffer`` / iter-158's ``UtteranceAggregator`` filled for the
utterance-merge seam, and iter-170's ``BackchannelMonitor`` filled for the
emit-timestamp.

This module is that driver. ``MonologueClock`` consumes two events —
``on_speech_start(now)`` and ``on_speech_stop(now)`` — and exposes
``monologue_start_at`` and ``pause_secs(now)`` so a live cue path can wire:

    clock.on_speech_start(t)        # ← VADUserStartedSpeakingFrame
    ...
    clock.on_speech_stop(t)         # ← VADUserStoppedSpeakingFrame
    d = monitor.observe(
        now=now,
        monologue_start_at=clock.monologue_start_at,
        pause_secs=clock.pause_secs(now),
    )
    if d.emit:
        broadcast_cue(d.cue_type)

The monologue resets — a fresh ``monologue_start_at`` — only when a
``stop → start`` gap is **at or above** ``reset_gap_secs`` (default 2.0s,
exactly ``backchannel_timing.py``'s ``max_pause_secs`` and
``turn_decider.py``'s ``silence_floor_secs``, so the "is this gap a clause
pause or a turn-end?" boundary is the *same scalar* across the monologue clock,
the backchannel-timing window, and the turn decider — they cannot drift apart).
A shorter gap is a clause boundary: the same monologue continues, its start
unchanged, so ``user_speaking_secs`` keeps accumulating across the pause and
the warm-up gate (``min_speaking_before_first_cue_secs``) eventually clears.

Dependency-free by design (no I/O, no clock reads — every timestamp injected),
like its organic-track siblings, so tests load it by file path without dragging
in ``session/__init__``'s eager pipecat import (absent on the x86_64 runner).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MonologueClockConfig",
    "MonologueClock",
]


@dataclass(frozen=True)
class MonologueClockConfig:
    """Threshold for when a ``stop → start`` gap ends the current monologue.

    ``reset_gap_secs`` — a pause of at least this many seconds between the user
    stopping and starting again is treated as the user having *yielded the
    floor*: the next speech start begins a **new** monologue (a fresh
    ``monologue_start_at``). A shorter gap is a clause-boundary pause and the
    same monologue continues. Default 2.0s — exactly
    ``backchannel_timing.py``'s ``max_pause_secs`` and ``turn_decider.py``'s
    ``silence_floor_secs``, so the clause-pause/turn-end boundary is one shared
    scalar across the whole organic stack and the three paths can't drift. That
    cross-module equality is pinned by
    ``tests/unit/test_shared_silence_floor_invariant.py`` (iter-179) against the
    sibling modules themselves, not a hardcoded ``2.0``, so retuning one default
    without the others fails the test instead of silently reopening the overlap.
    """

    reset_gap_secs: float = 2.0

    def __post_init__(self) -> None:
        if self.reset_gap_secs <= 0:
            raise ValueError(
                f"reset_gap_secs must be > 0 (got {self.reset_gap_secs})"
            )


class MonologueClock:
    """Track the current monologue's start and the live within-speech pause.

    Cross-event state derived from the VAD ``started``/``stopped`` stream so a
    live cue path can feed ``BackchannelMonitor.observe`` the
    ``monologue_start_at`` and ``pause_secs`` it needs. Pure and deterministic:
    no I/O, no clock reads — the caller injects every timestamp (the same clock
    the audio path uses).

    A monologue spans multiple speech segments separated by *brief* pauses; it
    resets only on a ``stop → start`` gap ``>= config.reset_gap_secs`` (the user
    yielded the floor). ``config`` defaults to ``MonologueClockConfig()``.
    """

    def __init__(self, *, config: MonologueClockConfig | None = None) -> None:
        self._config = config if config is not None else MonologueClockConfig()
        #: Start of the current monologue, or ``None`` before any speech / after
        #: ``reset``. Unchanged across a clause-boundary pause; replaced on a
        #: turn-end-sized gap.
        self._monologue_start_at: float | None = None
        #: Timestamp of the most recent ``on_speech_stop``, or ``None`` while
        #: speaking / before the first stop. Drives ``pause_secs``.
        self._last_stop_at: float | None = None
        #: Whether the user is currently speaking (between a start and its stop).
        self._speaking: bool = False

    @property
    def speaking(self) -> bool:
        """True iff currently between an ``on_speech_start`` and its stop."""
        return self._speaking

    @property
    def monologue_start_at(self) -> float | None:
        """When the current monologue began, or ``None`` if no monologue yet."""
        return self._monologue_start_at

    @property
    def last_stop_at(self) -> float | None:
        """Timestamp of the most recent stop, or ``None``. Read-only view."""
        return self._last_stop_at

    def on_speech_start(self, now: float) -> None:
        """Record that the user started (or resumed) speaking at ``now``.

        Starts a **new** monologue (sets ``monologue_start_at = now``) iff there
        is no current monologue *or* the gap since the last stop is at/above
        ``reset_gap_secs`` (the user yielded the floor). Otherwise the gap is a
        clause-boundary pause and the existing monologue continues — its
        ``monologue_start_at`` is left untouched so ``user_speaking_secs`` keeps
        accumulating across the pause. Clears the pause clock either way.
        """
        gap = None if self._last_stop_at is None else now - self._last_stop_at
        if self._monologue_start_at is None or (
            gap is not None and gap >= self._config.reset_gap_secs
        ):
            self._monologue_start_at = now
        self._speaking = True
        self._last_stop_at = None

    def on_speech_stop(self, now: float) -> None:
        """Record that the user stopped speaking at ``now`` — a pause begins.

        Does **not** end the monologue: whether this stop is a clause-boundary
        pause or a true turn-end is only known once the *next* start arrives (or
        doesn't), so the decision is deferred to ``on_speech_start``. Until then
        ``pause_secs(now)`` reports the growing gap.
        """
        self._speaking = False
        self._last_stop_at = now

    def pause_secs(self, now: float) -> float:
        """The current within-speech pause: seconds since the last stop.

        Zero while the user is speaking (no pause) or before the first stop.
        Otherwise ``now - last_stop_at``, clamped ``>= 0`` against clock skew
        (mirroring the defensive clamps in the aggregator and the monitor).
        """
        if self._speaking or self._last_stop_at is None:
            return 0.0
        return max(0.0, now - self._last_stop_at)

    def user_speaking_secs(self, now: float) -> float:
        """How long the current monologue has run: ``now - monologue_start_at``.

        The third quantity ``BackchannelMonitor.observe`` consumes, alongside
        ``monologue_start_at`` and ``pause_secs`` — and the one every consumer
        used to recompute by hand. Folding it onto the clock keeps the None
        guard and the skew clamp in *one* place: it returns ``0.0`` before any
        speech / after ``reset`` (``monologue_start_at`` is ``None``) rather
        than raising on ``now - None`` — exactly the ``TypeError`` the iter-174
        ``BackchannelDriver`` had to patch around at its own call site — and
        clamps ``>= 0`` against clock skew, mirroring the monitor's own
        ``max(0.0, now - monologue_start_at)``.

        Note this measures *monologue length*, not active-speech length: it
        keeps growing through a clause-boundary pause (the monologue continues
        until a turn-end-sized gap resets ``monologue_start_at``), which is the
        same quantity the warm-up gate (``min_speaking_before_first_cue_secs``)
        is checked against.
        """
        start = self._monologue_start_at
        if start is None:
            return 0.0
        return max(0.0, now - start)

    def reset(self) -> None:
        """Clear all state — no monologue, not speaking, no pending pause.

        Called when a fresh conversation starts: the next ``on_speech_start``
        begins a brand-new monologue regardless of how long ago the last stop
        was.
        """
        self._monologue_start_at = None
        self._last_stop_at = None
        self._speaking = False
