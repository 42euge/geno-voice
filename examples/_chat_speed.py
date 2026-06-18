"""Per-session mutable TTS speed — the live-path holder for the iter-213
WPM-mirroring seam.

iter-214 wires ``session.wpm_mirror.WpmMirror`` (iter-213, the
``user_wpm → bot speed`` decision) into ``mic_chat``. The obstacle the seam
left open: the Kokoro ``speed`` knob was a **constant float** captured once at
closure-build time (``build_audio_io(..., speed)`` baked it into ``synth_fn``).
To adapt the rate turn-to-turn, the speed has to become a *mutable per-session
value* that the synth path reads fresh on every sentence and the turn loop
updates after each measured ``user_wpm``.

``SpeedController`` is that holder. It is the smallest object that

  1. exposes the **current** speed as a zero-arg callable
     (``controller.current``) that ``build_audio_io``'s ``synth_fn`` calls per
     sentence (``build_audio_io`` now accepts a float *or* a callable), and
  2. **observes** each completed turn's ``user_wpm`` and folds it through the
     injected mirror to compute the next turn's speed (``observe(user_wpm)``).

Design follows the GENO.md ``mic_chat.py`` extraction conventions:

- **Inject the mirror as a duck-typed callable dependency**, not a concrete
  class. The controller only needs an object exposing
  ``speed(user_wpm=…, current_speed=…) -> float`` — exactly the
  ``WpmMirror`` interface (iter-213). Tests pass any stub of that shape; a
  future learned rate-matcher drops in without a call-site change.
- **Off-by-default safety.** ``mirror=None`` ⇒ ``observe`` is a no-op and the
  speed never moves from its initial value — byte-for-byte today's fixed-rate
  behavior. ``mic_chat`` only constructs a controller with a live mirror when
  the operator turns the ``wpm_mirror`` config on; otherwise it passes the
  plain float straight through, so the proven constant-speed path is untouched.
- **No I/O, no clock, no platform deps** — pure orchestration over the injected
  mirror, importable on the x86_64 test runner without pyaudio / kokoro /
  pipecat.

The controller is deliberately thin: the cross-turn damping that keeps the rate
from lurching on one noisy WPM already lives inside the mirror (iter-213's
``strength``), so the controller carries no state beyond the current speed.
"""

from __future__ import annotations

from typing import Any, Optional


class SpeedController:
    """Mutable holder for the per-session Kokoro ``speed`` multiplier.

    Wraps an initial speed and an optional WPM mirror. ``current()`` is the
    zero-arg accessor the synth path reads on every sentence; ``observe()``
    folds a just-measured ``user_wpm`` through the mirror to set the speed the
    *next* turn's sentences will use.

    With ``mirror=None`` (the default) ``observe`` is inert and the speed is the
    constant it was built with — the off-by-default path ``mic_chat`` uses when
    the operator has not enabled mirroring.
    """

    def __init__(self, initial_speed: float, mirror: Optional[Any] = None):
        """Args:
        initial_speed: the starting ``speed`` multiplier (the ``mic_chat``
            CLI / config value — historically a constant ``1.0``).
        mirror: an object exposing ``speed(*, user_wpm, current_speed) ->
            float`` (the iter-213 ``WpmMirror`` interface), or ``None`` to
            disable adaptation (``observe`` becomes a no-op). Injected, not
            imported, so this module stays free of the ``session`` package's
            eager pipecat import.
        """
        self._speed = float(initial_speed)
        self._mirror = mirror

    @property
    def speed(self) -> float:
        """The current speed multiplier (read-only property)."""
        return self._speed

    def current(self) -> float:
        """Zero-arg accessor for the current speed.

        This is the callable handed to ``build_audio_io(speed=…)`` so the
        ``synth_fn`` closure resolves the *live* speed on every sentence rather
        than a value baked in at build time.
        """
        return self._speed

    @property
    def active(self) -> bool:
        """Whether a mirror is wired in (adaptation can move the speed)."""
        return self._mirror is not None

    def observe(self, user_wpm: float) -> float:
        """Fold a completed turn's ``user_wpm`` into the next turn's speed.

        With no mirror this is a no-op returning the unchanged speed. With a
        mirror, the new speed is ``mirror.speed(user_wpm=…,
        current_speed=current)``; the mirror itself owns the gate / clamp /
        deadband / damping (iter-213), so the controller blindly trusts and
        stores whatever it returns. Returns the (possibly unchanged) speed.

        A misbehaving mirror must not break the live turn loop: any exception is
        swallowed and the current speed is kept (degrading to the fixed-rate
        path for that turn rather than crashing the session).
        """
        if self._mirror is None:
            return self._speed
        try:
            self._speed = float(
                self._mirror.speed(
                    user_wpm=user_wpm, current_speed=self._speed,
                )
            )
        except Exception:
            # Mirror raised / returned a non-number — keep the prior speed.
            pass
        return self._speed
