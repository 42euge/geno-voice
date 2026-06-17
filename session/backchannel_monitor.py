"""
Stateful agent-backchannel monitor for the organic turn-taking track (the
live-loop driver for backlog item #7 in
``docs/research/organic-turn-taking.md``).

Backlog #7 shipped the *pure decision* in ``backchannel_timing.py`` (iter-153):
``decide_backchannel_timing(*, user_speaking_secs, pause_secs,
secs_since_last_backchannel, ...)`` answers "is *now* a good moment for the
agent to emit a mid-speech backchannel ('mhmm')?" — but it is stateless. It
demands the caller inject three already-measured quantities, and one of them is
**not** something the caller can read off a clock: ``secs_since_last_backchannel``
depends on the monitor's *own past EMIT decisions*, not on anything the
recorder measures.

That is the gap this module closes — exactly mirroring how iter-156's
``UtteranceBuffer`` / iter-158's ``UtteranceAggregator`` became the stateful
drivers for iter-155's ``decide_utterance_continuation`` seam. A live cue path
calling the pure seam each frame would pass ``secs_since_last_backchannel=None``
forever (it has no way to remember when it last emitted), so the rate limit
(``min_between_cues_secs``, default 20s) would never engage and the agent would
chatter "mhmm mhmm mhmm" on every qualifying pause frame. The monitor records
the emit timestamp the moment it decides ``EMIT``, so the *next* ``observe``
sees a real ``secs_since_last_backchannel`` and the rate limit finally bites.

  - ``BackchannelMonitor`` owns two scalars of cross-event state the pure seam
    can't carry: the timestamp of the most recent backchannel it emitted
    (``None`` until the first emit), and its position in the shared
    ``CUE_ROTATION`` (``session/cue_rotation.py``) so consecutive cues rotate
    ("mhmm" → "i see" → "right" → ...) instead of repeating one sound.
  - ``observe(*, now, monologue_start_at, pause_secs)`` computes
    ``user_speaking_secs = now - monologue_start_at`` and
    ``secs_since_last_backchannel = now - last_emit`` (``None`` if never),
    routes them through ``decide_backchannel_timing``, and — *iff* the decision
    is ``EMIT`` — records ``now`` as the new last-emit timestamp and picks the
    next cue from the rotation before returning. Returns a small frozen
    :class:`BackchannelDecision` (carrying the actionable ``cue_type`` on an
    emit) so call sites read clearly and tests can assert on the derived
    quantities.
  - ``reset()`` clears the last-emit timestamp so a fresh session (or a long
    lull) starts the rate limit over.

The caller still reads no clock *inside* the decision path — it hands the
monitor the two timestamps the audio path already tracks (when the user's
current monologue started, and "now") plus the within-speech ``pause_secs`` it
is already measuring for VAD. The monitor does the subtraction and the one
stateful step the seam can't: remembering its own last emit.

**The half-duplex invariant is the whole point.** With a default
``FullDuplexConfig()`` (``agent_backchannels`` inactive) ``observe`` returns
``emit=False`` for *every* input — ``decide_backchannel_timing`` short-circuits
to ``HOLD`` before consulting any signal, so the last-emit timestamp is never
set and the monitor is an inert observer. Byte-for-byte today's behavior, where
the agent never speaks during user speech. Only when agent backchannels are
explicitly turned on does a well-timed clause-boundary pause yield ``emit=True``
and arm the rate limit. So wiring this into the live cue path (a later lap) can
never regress the proven half-duplex path; the new behavior lives entirely
behind the off-by-default switch.

Design follows the GENO.md conventions and the sibling drivers: pure (no I/O,
no clock reads — every timestamp injected), an injected optional config, an
injected optional timing-config, and a small frozen dataclass return. Like its
siblings it loads by file path in tests to dodge ``session/__init__``'s eager
pipecat import (absent on the x86_64 runner).
"""

from __future__ import annotations

from dataclasses import dataclass

from session.backchannel_timing import (
    BackchannelTiming,
    BackchannelTimingConfig,
    decide_backchannel_timing,
)
from session.cue_rotation import cue_for_index
from session.full_duplex import FullDuplexConfig

__all__ = [
    "BackchannelDecision",
    "BackchannelMonitor",
]


@dataclass(frozen=True)
class BackchannelDecision:
    """Outcome of one :meth:`BackchannelMonitor.observe` call.

    ``emit`` is the actionable boolean — ``True`` iff the agent should emit a
    mid-speech backchannel now (the live cue path calls ``broadcast_cue`` on
    ``True``). The remaining fields are the quantities the monitor derived and
    fed to ``decide_backchannel_timing``, exposed for observability so a test
    or a session summary can see *why* the decision came out the way it did:

      - ``user_speaking_secs`` — how long the current monologue has run
        (``now - monologue_start_at``, clamped ``>= 0``).
      - ``pause_secs`` — the within-speech pause the caller measured (echoed
        back unchanged).
      - ``secs_since_last_backchannel`` — seconds since the monitor last
        emitted, or ``None`` if it never has this session (the value that the
        stateless seam could not compute on its own).
      - ``cue_type`` — *which* cue to play, advanced through the shared
        ``CUE_ROTATION`` on each emit so consecutive backchannels don't repeat
        ("mhmm" then "i see" then "right" ...). ``None`` when ``emit`` is False
        (no cue to play). This is the second piece of cross-event state the
        pure seam can't carry — where we are in the rotation — so the monitor
        owns it alongside the last-emit timestamp.
    """

    emit: bool
    user_speaking_secs: float
    pause_secs: float
    secs_since_last_backchannel: float | None
    cue_type: str | None = None


class BackchannelMonitor:
    """Cross-event state for the mid-speech agent-backchannel decision.

    The live cue path calls :meth:`observe` as the user speaks — passing the
    monologue start, the current time, and the within-speech pause it is
    already measuring — and emits a backchannel cue whenever the returned
    :class:`BackchannelDecision` has ``emit=True``. The monitor remembers when
    it last emitted so the rate limit (``min_between_cues_secs``) engages
    across calls, which the pure :func:`decide_backchannel_timing` seam cannot
    do on its own.

    Pure and deterministic: no I/O, no clock reads. The caller injects every
    timestamp. ``config`` gates the behavior (default: a fresh half-duplex
    ``FullDuplexConfig()`` ⇒ ``emit`` is always ``False``); ``timing`` carries
    the thresholds (default: ``BackchannelTimingConfig()``).
    """

    def __init__(
        self,
        *,
        config: FullDuplexConfig | None = None,
        timing: BackchannelTimingConfig | None = None,
    ) -> None:
        self._config = config if config is not None else FullDuplexConfig()
        self._timing = timing if timing is not None else BackchannelTimingConfig()
        #: Timestamp of the most recent emit (``None`` until the first). Set
        #: only when ``observe`` decides ``EMIT``; persists across monologues
        #: so the rate limit is session-wide, the way a human doesn't reset
        #: their "I just said mhmm" clock just because the speaker took a
        #: breath. Cleared by ``reset``.
        self._last_backchannel_at: float | None = None
        #: How many backchannels the monitor has emitted (observability).
        self._emit_count: int = 0
        #: Position in the shared ``CUE_ROTATION``. Advanced only on an actual
        #: emit, so consecutive backchannels rotate through the cue bank
        #: ("mhmm" → "i see" → "right" → ...) instead of repeating one cue. The
        #: second piece of cross-event state the pure seam can't carry (the
        #: first being ``_last_backchannel_at``). Mirrors ``_cue_index`` in
        #: ``TurnTakingEngine``; unlike the rate-limit clock it survives
        #: ``reset`` so a new monologue continues the rotation rather than
        #: replaying "mhmm" every time.
        self._cue_index: int = 0

    @property
    def active(self) -> bool:
        """True iff agent backchannels are on for this monitor (organic mode)."""
        return self._config.agent_backchannels_active()

    @property
    def last_backchannel_at(self) -> float | None:
        """Timestamp of the most recent emit, or ``None``. Read-only view."""
        return self._last_backchannel_at

    @property
    def emit_count(self) -> int:
        """How many backchannels this monitor has emitted (0 until the first)."""
        return self._emit_count

    @property
    def cue_index(self) -> int:
        """Current position in the shared ``CUE_ROTATION`` (advances per emit)."""
        return self._cue_index

    def secs_since_last_backchannel(self, now: float) -> float | None:
        """Seconds since the monitor last emitted, or ``None`` if it never has.

        The third quantity the decision consumes (alongside
        ``user_speaking_secs`` and ``pause_secs``) and the one *only the monitor
        can answer* — it depends on the monitor's own past EMIT decisions, not
        on anything a clock reads. ``observe`` derives it inline each call; this
        accessor folds that derivation onto one method so consumers that need it
        *outside* the decision path (e.g. the driver's no-monologue short-circuit
        in iter-174) source it from the single owner of ``_last_backchannel_at``
        rather than re-spelling the ``None`` guard and ``>= 0`` skew clamp — the
        monitor-side mirror of iter-176's ``MonologueClock.user_speaking_secs``.

        Returns ``None`` before the first emit / after ``reset`` (the value the
        stateless seam treats as "never backchanneled, rate limit passes"), else
        ``now - last_emit`` clamped ``>= 0`` against clock skew.
        """
        last_emit = self._last_backchannel_at
        if last_emit is None:
            return None
        return max(0.0, now - last_emit)

    def observe(
        self,
        *,
        now: float,
        monologue_start_at: float,
        pause_secs: float,
    ) -> BackchannelDecision:
        """Decide whether to emit a backchannel at time ``now``.

        ``now`` and ``monologue_start_at`` are clock timestamps (same clock the
        audio path uses): respectively the current moment and when the user's
        current monologue began. ``pause_secs`` is the within-speech pause the
        caller has already measured (the gap since the last speech frame).

        Computes ``user_speaking_secs = now - monologue_start_at`` (clamped to
        ``>= 0`` against clock skew, mirroring the defensive clamps in the
        aggregator and ``_chat_loop``) and
        ``secs_since_last_backchannel = now - last_emit`` (``None`` if the
        monitor has never emitted, likewise clamped ``>= 0``), then routes them
        through :func:`decide_backchannel_timing`. **Iff** the decision is
        ``EMIT`` it records ``now`` as the new last-emit timestamp (arming the
        rate limit for subsequent calls) and bumps the emit counter, then
        returns a :class:`BackchannelDecision`.

        In half-duplex mode the underlying seam always returns ``HOLD``, so
        ``emit`` is always ``False`` and the last-emit timestamp is never set —
        the monitor never mutates its state. Byte-for-byte today's behavior.
        """
        user_speaking_secs = max(0.0, now - monologue_start_at)
        secs_since = self.secs_since_last_backchannel(now)

        timing = decide_backchannel_timing(
            user_speaking_secs=user_speaking_secs,
            pause_secs=pause_secs,
            secs_since_last_backchannel=secs_since,
            config=self._config,
            timing=self._timing,
        )
        emit = timing is BackchannelTiming.EMIT
        cue_type: str | None = None
        if emit:
            # The one stateful step the seam can't do: remember this emit so the
            # next observe sees a real ``secs_since_last_backchannel`` and the
            # rate limit engages. Without this the agent would re-emit on every
            # qualifying pause frame.
            self._last_backchannel_at = now
            self._emit_count += 1
            # Pick *which* cue and advance the rotation — the other piece of
            # cross-event state. Only on a real emit, so a held frame never
            # burns a rotation slot.
            cue_type = cue_for_index(self._cue_index)
            self._cue_index += 1

        return BackchannelDecision(
            emit=emit,
            user_speaking_secs=user_speaking_secs,
            pause_secs=pause_secs,
            secs_since_last_backchannel=secs_since,
            cue_type=cue_type,
        )

    def reset(self) -> None:
        """Clear the last-emit timestamp (but keep the lifetime emit count).

        Called when a fresh conversation starts (or the caller wants the rate
        limit to start over after a long lull): the next ``observe`` sees
        ``secs_since_last_backchannel=None`` and the warm-up / rate-limit gates
        evaluate as if the monitor had never emitted. The ``emit_count`` is a
        lifetime tally and is intentionally *not* reset, so a session summary
        can still report the total. The cue-rotation position is likewise *not*
        reset, so a fresh monologue continues the rotation rather than always
        replaying the first cue ("mhmm") — the same way a person wouldn't open
        every new lull with the identical sound.
        """
        self._last_backchannel_at = None
