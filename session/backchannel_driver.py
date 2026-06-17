"""
Backchannel driver for the organic turn-taking track — the single object that
composes the VAD-event clock with the stateful monitor so the live cue path is
a *thin* adapter (the last open piece of backlog item #7 in
``docs/research/organic-turn-taking.md``).

By iter-173 all three pure pieces existed: *when* to backchannel
(``decide_backchannel_timing``, iter-153), *which* cue to play
(``cue_rotation`` + the monitor's rotation index, iter-171), and the two
derived *inputs* the monitor needs — ``monologue_start_at`` and ``pause_secs``
(``MonologueClock``, iter-173). The iteration notes repeatedly described the
remaining live ``pipecat_server.py`` ``Broadcaster`` wiring as "a thin adapter":

    clock.on_speech_start(t)        # ← VADUserStartedSpeakingFrame
    clock.on_speech_stop(t)         # ← VADUserStoppedSpeakingFrame
    d = monitor.observe(
        now=now,
        monologue_start_at=clock.monologue_start_at,
        pause_secs=clock.pause_secs(now),
    )
    if d.emit:
        broadcast_cue(d.cue_type)

But that "thin adapter" is itself **two objects with a four-line composition
ritual repeated by hand at every call site** (both integration tests in
``test_monologue_clock.py`` spell it out), and the ``monitor.observe`` line has
a **latent crash**: it passes ``clock.monologue_start_at`` straight in, but that
is ``None`` until the first speech start — and ``observe`` computes
``now - monologue_start_at`` with no None guard, so an ``observe`` tick that
lands *before any speech* (a timer that fires on a freshly-reset session, a
warm-up poll) raises ``TypeError``. The composition is pure and testable; the
only genuinely pipecat-bound parts are feeding the VAD frames and calling
``broadcast_cue``. So this module extracts the composition — leaving the live
wiring a *truly* thin shim — and closes the None-tick crash in one place rather
than at every (future) call site.

``BackchannelDriver`` owns a ``MonologueClock`` and a ``BackchannelMonitor`` and
exposes three methods the live loop calls directly off the frame stream:

    driver = BackchannelDriver(config=organic_full_duplex_config)
    driver.on_speech_start(t)       # ← VADUserStartedSpeakingFrame
    driver.on_speech_stop(t)        # ← VADUserStoppedSpeakingFrame
    d = driver.observe(now)         # ← any tick; returns BackchannelDecision
    if d.emit:
        broadcast_cue(d.cue_type)

``observe(now)`` reads ``monologue_start_at`` / ``pause_secs`` off its own clock
and routes them through its own monitor — but **short-circuits to a no-emit
decision when there is no monologue yet** (``monologue_start_at is None``),
which is the correct organic answer (you can't backchannel a speaker who hasn't
started) *and* the fix for the bare-composition crash. Everything else is pure
delegation, so the half-duplex invariant is inherited verbatim from the monitor:
a default (half-duplex) ``FullDuplexConfig`` ⇒ ``observe`` returns
``emit=False`` for every tick and the monitor never mutates its state.

Dependency-free by design beyond its two organic-track siblings (no I/O, no
clock reads — every timestamp injected), so tests load it by file path without
dragging in ``session/__init__``'s eager pipecat import (absent on the x86_64
runner).
"""

from __future__ import annotations

from session.backchannel_monitor import BackchannelDecision, BackchannelMonitor
from session.backchannel_timing import BackchannelTimingConfig
from session.full_duplex import FullDuplexConfig
from session.monologue_clock import MonologueClock, MonologueClockConfig

__all__ = [
    "BackchannelDriver",
]


class BackchannelDriver:
    """Compose the monologue clock and the backchannel monitor into one object.

    The live cue path drives this directly off the VAD frame stream:
    ``on_speech_start`` / ``on_speech_stop`` for the
    ``VADUserStartedSpeakingFrame`` / ``VADUserStoppedSpeakingFrame`` events,
    and ``observe(now)`` on any tick to ask "emit a backchannel now?". The
    driver owns both pieces of cross-event state internally (the monologue
    boundary in the clock, the last-emit timestamp + cue-rotation index in the
    monitor), so the call site holds no state of its own beyond the driver
    reference.

    Construction mirrors the siblings: an optional ``config``
    (``FullDuplexConfig``, default half-duplex ⇒ never emits), an optional
    ``timing`` (``BackchannelTimingConfig``), and an optional ``clock_config``
    (``MonologueClockConfig``) — all forwarded to the clock/monitor. For tests
    that want to inject pre-built collaborators (e.g. a monitor primed with a
    last-emit timestamp), pass ``clock=`` and/or ``monitor=`` directly; an
    injected collaborator takes precedence over the matching ``*_config``.
    """

    def __init__(
        self,
        *,
        config: FullDuplexConfig | None = None,
        timing: BackchannelTimingConfig | None = None,
        clock_config: MonologueClockConfig | None = None,
        clock: MonologueClock | None = None,
        monitor: BackchannelMonitor | None = None,
    ) -> None:
        self._clock = (
            clock if clock is not None else MonologueClock(config=clock_config)
        )
        self._monitor = (
            monitor
            if monitor is not None
            else BackchannelMonitor(config=config, timing=timing)
        )

    # --- collaborators (read-only views) ------------------------------------

    @property
    def clock(self) -> MonologueClock:
        """The underlying :class:`MonologueClock` (read-only view)."""
        return self._clock

    @property
    def monitor(self) -> BackchannelMonitor:
        """The underlying :class:`BackchannelMonitor` (read-only view)."""
        return self._monitor

    # --- monitor passthroughs (observability) -------------------------------

    @property
    def active(self) -> bool:
        """True iff agent backchannels are on (organic mode) — from the monitor."""
        return self._monitor.active

    @property
    def speaking(self) -> bool:
        """True iff the user is currently speaking — from the clock."""
        return self._clock.speaking

    @property
    def monologue_start_at(self) -> float | None:
        """When the current monologue began, or ``None`` — from the clock."""
        return self._clock.monologue_start_at

    @property
    def last_backchannel_at(self) -> float | None:
        """Timestamp of the most recent emit, or ``None`` — from the monitor."""
        return self._monitor.last_backchannel_at

    @property
    def emit_count(self) -> int:
        """How many backchannels have been emitted — from the monitor."""
        return self._monitor.emit_count

    @property
    def cue_index(self) -> int:
        """Current position in the shared cue rotation — from the monitor."""
        return self._monitor.cue_index

    # --- VAD events ---------------------------------------------------------

    def on_speech_start(self, now: float) -> None:
        """Feed a ``VADUserStartedSpeakingFrame`` at ``now`` to the clock."""
        self._clock.on_speech_start(now)

    def on_speech_stop(self, now: float) -> None:
        """Feed a ``VADUserStoppedSpeakingFrame`` at ``now`` to the clock."""
        self._clock.on_speech_stop(now)

    # --- the decision -------------------------------------------------------

    def observe(self, now: float) -> BackchannelDecision:
        """Decide whether to emit a backchannel at time ``now``.

        Reads ``monologue_start_at`` and ``pause_secs(now)`` off the internal
        clock and routes them through the internal monitor — the four-line
        ritual the call sites used to spell out by hand, now in one place.

        **Short-circuits when there is no monologue yet** (the clock's
        ``monologue_start_at`` is ``None``: before the first speech start or
        right after a ``reset``). That returns a no-emit
        :class:`BackchannelDecision` with ``user_speaking_secs=0.0`` and no cue,
        which is both the correct organic answer (there is no speaker to back-
        channel) and the guard against ``BackchannelMonitor.observe``'s
        unguarded ``now - monologue_start_at`` subtraction, which would raise
        ``TypeError`` on a ``None`` start. The monitor's state is never touched
        on this path, so the half-duplex invariant and the rate limit are both
        preserved. The reported ``secs_since_last_backchannel`` comes from the
        monitor's own ``secs_since_last_backchannel(now)`` accessor (the owner of
        ``_last_backchannel_at``), not a hand-recompute, so the None guard and
        the skew clamp can't drift from the in-decision derivation.
        """
        start = self._clock.monologue_start_at
        if start is None:
            return BackchannelDecision(
                emit=False,
                user_speaking_secs=self._clock.user_speaking_secs(now),
                pause_secs=self._clock.pause_secs(now),
                secs_since_last_backchannel=self._monitor.secs_since_last_backchannel(
                    now
                ),
                cue_type=None,
            )
        return self._monitor.observe(
            now=now,
            monologue_start_at=start,
            pause_secs=self._clock.pause_secs(now),
        )

    def reset(self) -> None:
        """Reset both collaborators for a fresh conversation.

        Clears the clock entirely (no monologue, not speaking, no pending pause)
        and clears the monitor's last-emit timestamp so the rate limit starts
        over. As with the monitor's own ``reset``, the lifetime ``emit_count``
        and the cue-rotation position are intentionally *not* reset — a fresh
        monologue continues the rotation rather than replaying the first cue.
        """
        self._clock.reset()
        self._monitor.reset()
