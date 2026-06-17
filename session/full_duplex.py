"""
Full-duplex config flag scaffolding for the organic turn-taking track
(backlog item #3 in ``docs/research/organic-turn-taking.md``).

The organic-voice track moves geno-voice beyond rigid half-duplex toward
conversation that overlaps freely: continuer-aware listening (a "mhmm" during
agent speech doesn't abandon the turn), agent backchannels emitted *during*
user speech, smart end-of-turn. Those behaviors change when the system speaks
and listens, so they must be **opt-in** — the proven half-duplex path
("you speak, it waits, it replies") is never regressed while the track matures.

This module is the gate. It is pure, dependency-free, and GENO.md-conventional:

  - ``FullDuplexConfig`` — a frozen dataclass. Every organic behavior is a
    flag that defaults **off**, so a default-constructed config is exactly
    today's half-duplex behavior.
  - A master ``enabled`` switch plus per-behavior sub-flags. A sub-flag left
    unset (``None``) **inherits** the master switch, so ``GENO_FULL_DUPLEX=1``
    turns the whole organic surface on; an explicit sub-flag overrides it
    (e.g. organic mode on but agent backchannels held off).
  - ``*_active`` resolution methods so call sites read one effective boolean
    rather than re-deriving the inherit logic.
  - ``full_duplex_config_from_env(env=os.environ)`` — the only I/O-touching
    edge; it reads env vars and delegates to ``parse_bool_flag`` so the
    parsing is unit-testable with an injected mapping.

Nothing consumes this yet — wiring continuer-aware barge-in (backlog #5) and
agent backchannel emission (#7) behind these flags is a later lap. Shipping
the gate first means those laps add behavior *behind an off-by-default switch*
rather than introducing both the behavior and its guard at once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

__all__ = [
    "FullDuplexConfig",
    "full_duplex_config_from_env",
    "parse_bool_flag",
    "TRUTHY",
    "FALSY",
    "ENV_FULL_DUPLEX",
    "ENV_CONTINUER_AWARE",
    "ENV_AGENT_BACKCHANNELS",
]

#: Recognized truthy / falsy env-var spellings (case-insensitive, trimmed).
#: Closed sets so a typo (``GENO_FULL_DUPLEX=ture``) raises instead of being
#: silently read as off — a misspelled enable flag that quietly leaves organic
#: mode disabled is the worst failure mode for a feature gate.
TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on", "y", "t"})
FALSY: frozenset[str] = frozenset({"0", "false", "no", "off", "n", "f", ""})

#: Env-var names. Master switch plus per-behavior overrides.
ENV_FULL_DUPLEX = "GENO_FULL_DUPLEX"
ENV_CONTINUER_AWARE = "GENO_FULL_DUPLEX_CONTINUER_AWARE"
ENV_AGENT_BACKCHANNELS = "GENO_FULL_DUPLEX_AGENT_BACKCHANNELS"


def parse_bool_flag(value: str | None, *, name: str = "flag") -> bool | None:
    """Parse an env-style boolean. Returns ``None`` when ``value`` is ``None``
    (the var is *unset* — distinct from set-but-empty, which is falsy).

    Matching is case-insensitive and whitespace-trimmed against the closed
    ``TRUTHY`` / ``FALSY`` sets. An unrecognized value raises ``ValueError``
    naming the offending variable, so a typo surfaces loudly rather than
    defaulting a feature gate to off.
    """
    if value is None:
        return None
    norm = value.strip().lower()
    if norm in TRUTHY:
        return True
    if norm in FALSY:
        return False
    raise ValueError(
        f"{name}={value!r} is not a recognized boolean "
        f"(use one of {sorted(TRUTHY)} / {sorted(FALSY - {''})})"
    )


@dataclass(frozen=True)
class FullDuplexConfig:
    """Gate for the organic turn-taking behaviors. All-off by default.

    ``enabled`` is the master switch for the whole organic surface. The
    per-behavior flags use a three-state ``bool | None``:

      - ``None`` (default) ⇒ *inherit* ``enabled``.
      - ``True`` / ``False`` ⇒ force that behavior on/off regardless of the
        master, so you can run organic mode with one behavior held back.

    Read the effective state through the ``*_active`` methods, never the raw
    sub-flag — they fold the inherit logic so call sites stay one-liners.

    A default ``FullDuplexConfig()`` is byte-for-byte today's half-duplex
    behavior: ``enabled`` is ``False`` and every ``*_active`` resolves ``False``.
    """

    enabled: bool = False
    continuer_aware_listening: bool | None = None
    agent_backchannels: bool | None = None

    def _resolve(self, sub_flag: bool | None) -> bool:
        """Effective value of a sub-flag: itself if set, else the master."""
        if sub_flag is None:
            return self.enabled
        return sub_flag

    def continuer_aware_listening_active(self) -> bool:
        """True iff a user continuer ("mhmm") during agent speech should be
        treated as *keep going* (finish the turn) rather than a barge-in
        abandon. Backlog #5 consumes this.
        """
        return self._resolve(self.continuer_aware_listening)

    def agent_backchannels_active(self) -> bool:
        """True iff the agent may emit backchannel cues *during* user speech
        (not only on trailing silence). Backlog #7 consumes this.
        """
        return self._resolve(self.agent_backchannels)

    def any_active(self) -> bool:
        """True iff *any* organic behavior is effectively on. A cheap guard
        for call sites that only need 'are we doing anything organic at all?'
        before paying for the finer-grained checks.
        """
        return (
            self.continuer_aware_listening_active()
            or self.agent_backchannels_active()
        )


def full_duplex_config_from_env(
    env: Mapping[str, str] | None = None,
) -> FullDuplexConfig:
    """Build a ``FullDuplexConfig`` from environment variables.

    Reads ``GENO_FULL_DUPLEX`` (master) and the per-behavior overrides
    ``GENO_FULL_DUPLEX_CONTINUER_AWARE`` / ``GENO_FULL_DUPLEX_AGENT_BACKCHANNELS``.
    An unset master defaults to ``False`` (half-duplex); unset sub-flags stay
    ``None`` so they inherit the master. ``env`` is injected (default
    ``os.environ``) so the parsing is testable without mutating the process
    environment. Propagates ``ValueError`` from ``parse_bool_flag`` on an
    unrecognized value.
    """
    if env is None:
        env = os.environ

    master = parse_bool_flag(env.get(ENV_FULL_DUPLEX), name=ENV_FULL_DUPLEX)
    continuer = parse_bool_flag(
        env.get(ENV_CONTINUER_AWARE), name=ENV_CONTINUER_AWARE
    )
    backchannels = parse_bool_flag(
        env.get(ENV_AGENT_BACKCHANNELS), name=ENV_AGENT_BACKCHANNELS
    )

    return FullDuplexConfig(
        enabled=bool(master),  # unset master ⇒ off
        continuer_aware_listening=continuer,
        agent_backchannels=backchannels,
    )
