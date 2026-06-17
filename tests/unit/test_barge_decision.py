"""Tests for iter-152 — continuer-aware barge-in decision (backlog #5).

``session/barge_decision.py`` composes two earlier seams — the backchannel
classifier (#1, iter-148) and the full-duplex gate (#3, iter-151) — into a
pure ``decide_barge_action(transcript, energy, *, config)`` that returns
``ABANDON`` (true interruption) or ``FINISH`` (user only backchanneled).

The whole point is the **half-duplex invariant**: with a default
``FullDuplexConfig()`` the decision is ``ABANDON`` for every transcript —
byte-for-byte today's "any barge cancels" behavior. Only with continuer-aware
listening explicitly on does a recognized continuer yield ``FINISH``.

``barge_decision`` does ``from session.backchannel import ...`` and
``from session.full_duplex import ...`` at module scope, but
``session/__init__.py`` eagerly imports pipecat-dependent modules (absent on
the x86_64 Linux runner). So we stand up a stub ``session`` namespace package
and load ``backchannel`` / ``full_duplex`` / ``barge_decision`` into it by file
path — the same trick test_text_eou.py uses for its sibling import.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SESSION_DIR = Path(__file__).resolve().parents[2] / "session"


def _load_by_path(name, filename, package=None):
    spec = importlib.util.spec_from_file_location(name, _SESSION_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    if package is not None:
        mod.__package__ = package
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if "session" not in sys.modules:
    _pkg = types.ModuleType("session")
    _pkg.__path__ = [str(_SESSION_DIR)]
    sys.modules["session"] = _pkg
if "session.backchannel" not in sys.modules:
    _load_by_path("session.backchannel", "backchannel.py", package="session")
if "session.full_duplex" not in sys.modules:
    _load_by_path("session.full_duplex", "full_duplex.py", package="session")

_bd = _load_by_path("session.barge_decision", "barge_decision.py", package="session")

BargeAction = _bd.BargeAction
decide_barge_action = _bd.decide_barge_action
should_abandon_turn = _bd.should_abandon_turn

FullDuplexConfig = sys.modules["session.full_duplex"].FullDuplexConfig
Backchannel = sys.modules["session.backchannel"].Backchannel
DEFAULT_ENERGY_CEILING = sys.modules["session.backchannel"].DEFAULT_ENERGY_CEILING


# A config with continuer-aware listening on (master switch is enough since
# the sub-flag inherits it). Used wherever we want organic behavior.
def _organic():
    return FullDuplexConfig(enabled=True)


# ---- The half-duplex invariant (default config) ------------------------------


class TestHalfDuplexInvariant:
    """A default config must ABANDON on every transcript — today's behavior."""

    @pytest.mark.parametrize(
        "transcript",
        [
            "mhmm",          # would be a CONTINUER under organic mode
            "yeah",
            "right",
            "wait no stop",  # SUBSTANTIVE
            "",              # NOT_SPEECH
            "   ",
            "go on",
        ],
    )
    def test_default_config_always_abandons(self, transcript):
        assert decide_barge_action(transcript) is BargeAction.ABANDON

    def test_default_config_passed_explicitly_abandons(self):
        assert decide_barge_action("mhmm", config=FullDuplexConfig()) is (
            BargeAction.ABANDON
        )

    def test_master_on_but_continuer_held_back_abandons(self):
        # Organic mode globally on, but this specific behavior forced off.
        cfg = FullDuplexConfig(enabled=True, continuer_aware_listening=False)
        assert decide_barge_action("mhmm", config=cfg) is BargeAction.ABANDON

    def test_continuer_does_not_get_classified_when_gated(self):
        # Even with energy that would matter, the gate short-circuits before
        # any classification — a continuer abandons under half-duplex.
        assert decide_barge_action("mhmm", 0.05) is BargeAction.ABANDON


# ---- Organic mode: continuer FINISHes, substantive ABANDONs ------------------


class TestOrganicMode:
    @pytest.mark.parametrize(
        "transcript",
        ["mhmm", "yeah", "right", "uh huh", "go on", "i see", "okay", "sure"],
    )
    def test_continuer_finishes(self, transcript):
        assert decide_barge_action(transcript, config=_organic()) is (
            BargeAction.FINISH
        )

    @pytest.mark.parametrize(
        "transcript",
        [
            "wait no that's wrong",
            "stop I changed my mind",
            "actually let me ask something else",
            "can you repeat the second part",
        ],
    )
    def test_substantive_abandons(self, transcript):
        assert decide_barge_action(transcript, config=_organic()) is (
            BargeAction.ABANDON
        )

    @pytest.mark.parametrize("transcript", ["", "   ", "\t\n"])
    def test_not_speech_abandons(self, transcript):
        # Conservative: empty/noise abandons rather than finishing, so a
        # misfire never leaves the user talking over a droning agent.
        assert decide_barge_action(transcript, config=_organic()) is (
            BargeAction.ABANDON
        )

    def test_sub_flag_true_overrides_master_off(self):
        # Master off, but continuer-aware explicitly on ⇒ organic behavior.
        cfg = FullDuplexConfig(enabled=False, continuer_aware_listening=True)
        assert decide_barge_action("mhmm", config=cfg) is BargeAction.FINISH
        assert decide_barge_action("wait no", config=cfg) is BargeAction.ABANDON


# ---- Energy gate flows through to the classifier -----------------------------


class TestEnergyGate:
    def test_loud_continuer_abandons_under_organic(self):
        # An emphatic "YEAH!" above the energy ceiling is taking the floor;
        # classify_backchannel returns SUBSTANTIVE, so we ABANDON.
        loud = DEFAULT_ENERGY_CEILING + 0.1
        assert decide_barge_action("yeah", loud, config=_organic()) is (
            BargeAction.ABANDON
        )

    def test_quiet_continuer_finishes_under_organic(self):
        quiet = DEFAULT_ENERGY_CEILING - 0.1
        assert decide_barge_action("yeah", quiet, config=_organic()) is (
            BargeAction.FINISH
        )

    def test_no_energy_uses_lexicon_only(self):
        assert decide_barge_action("yeah", None, config=_organic()) is (
            BargeAction.FINISH
        )

    def test_custom_energy_ceiling_threads_through(self):
        # Raise the ceiling so a previously-"loud" continuer counts as quiet.
        loud = DEFAULT_ENERGY_CEILING + 0.1
        assert decide_barge_action(
            "yeah", loud, config=_organic(), energy_ceiling=loud + 0.1
        ) is BargeAction.FINISH


# ---- max_words threads through -----------------------------------------------


class TestMaxWords:
    def test_default_two_word_continuer_finishes(self):
        assert decide_barge_action("go on", config=_organic()) is (
            BargeAction.FINISH
        )

    def test_three_continuer_words_abandons_at_default(self):
        # > DEFAULT_MAX_CONTINUER_WORDS (2) ⇒ classifier returns SUBSTANTIVE.
        assert decide_barge_action("yeah yeah yeah", config=_organic()) is (
            BargeAction.ABANDON
        )

    def test_raised_max_words_lets_longer_continuer_finish(self):
        assert decide_barge_action(
            "yeah yeah yeah", config=_organic(), max_words=3
        ) is BargeAction.FINISH


# ---- should_abandon_turn boolean ---------------------------------------------


class TestShouldAbandonTurn:
    def test_default_config_always_true(self):
        assert should_abandon_turn("mhmm") is True
        assert should_abandon_turn("wait no") is True

    def test_organic_continuer_false(self):
        assert should_abandon_turn("mhmm", config=_organic()) is False

    def test_organic_substantive_true(self):
        assert should_abandon_turn("wait no", config=_organic()) is True

    def test_matches_decide_barge_action(self):
        cfg = _organic()
        for t in ["mhmm", "wait no stop", "", "go on", "yeah yeah yeah"]:
            expected = decide_barge_action(t, config=cfg) is BargeAction.ABANDON
            assert should_abandon_turn(t, config=cfg) is expected


# ---- Interface / purity sanity -----------------------------------------------


class TestInterface:
    def test_energy_is_positional_config_keyword_only(self):
        # energy is the 2nd positional; config must be keyword-only.
        assert decide_barge_action("yeah", 0.1, config=_organic()) is (
            BargeAction.FINISH
        )
        with pytest.raises(TypeError):
            decide_barge_action("yeah", 0.1, _organic())  # type: ignore[misc]

    def test_pure_no_mutation_of_config(self):
        cfg = _organic()
        before = (
            cfg.enabled,
            cfg.continuer_aware_listening,
            cfg.agent_backchannels,
        )
        decide_barge_action("mhmm", config=cfg)
        after = (
            cfg.enabled,
            cfg.continuer_aware_listening,
            cfg.agent_backchannels,
        )
        assert before == after

    def test_two_actions_are_distinct(self):
        assert BargeAction.ABANDON is not BargeAction.FINISH
        assert BargeAction.ABANDON.value == "abandon"
        assert BargeAction.FINISH.value == "finish"
