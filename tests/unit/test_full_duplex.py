"""Tests for iter-151 — the full-duplex config flag scaffolding (backlog #3).

``session/full_duplex.py`` is the off-by-default gate for the organic
turn-taking behaviors (continuer-aware listening, agent backchannels). A
default ``FullDuplexConfig()`` must be byte-for-byte today's half-duplex
behavior; the env builder turns the surface on via ``GENO_FULL_DUPLEX`` while
per-behavior overrides can hold individual behaviors back.

``FullDuplexConfig`` / ``parse_bool_flag`` / ``full_duplex_config_from_env`` are
pure (env is injected, never read implicitly here), so these tests drive them
directly with injected mappings — no process-environment mutation.

``session/__init__.py`` eagerly imports pipecat-dependent modules (absent on
the x86_64 Linux runner). ``full_duplex`` is pure stdlib, so load it directly by
file path to bypass the package ``__init__`` — the same trick the
turn_decider / text_eou / backchannel tests use.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

_FD_PATH = Path(__file__).resolve().parents[2] / "session" / "full_duplex.py"
_spec = importlib.util.spec_from_file_location("_fd_under_test", _FD_PATH)
_fd = importlib.util.module_from_spec(_spec)
sys.modules["_fd_under_test"] = _fd
_spec.loader.exec_module(_fd)

FullDuplexConfig = _fd.FullDuplexConfig
full_duplex_config_from_env = _fd.full_duplex_config_from_env
parse_bool_flag = _fd.parse_bool_flag
TRUTHY = _fd.TRUTHY
FALSY = _fd.FALSY
ENV_FULL_DUPLEX = _fd.ENV_FULL_DUPLEX
ENV_CONTINUER_AWARE = _fd.ENV_CONTINUER_AWARE
ENV_AGENT_BACKCHANNELS = _fd.ENV_AGENT_BACKCHANNELS


# ---- parse_bool_flag ---------------------------------------------------------


class TestParseBoolFlag:
    def test_none_returns_none(self):
        # Unset var (None) is distinct from set-but-empty.
        assert parse_bool_flag(None) is None

    @pytest.mark.parametrize("value", sorted(TRUTHY))
    def test_truthy_spellings(self, value):
        assert parse_bool_flag(value) is True

    @pytest.mark.parametrize("value", sorted(FALSY))
    def test_falsy_spellings(self, value):
        assert parse_bool_flag(value) is False

    def test_empty_string_is_falsy_not_none(self):
        # Set-but-empty (GENO_FULL_DUPLEX=) is an explicit off, not unset.
        assert parse_bool_flag("") is False

    def test_case_insensitive(self):
        assert parse_bool_flag("TRUE") is True
        assert parse_bool_flag("True") is True
        assert parse_bool_flag("YES") is True
        assert parse_bool_flag("Off") is False

    def test_whitespace_trimmed(self):
        assert parse_bool_flag("  1  ") is True
        assert parse_bool_flag("\tfalse\n") is False

    def test_unrecognized_raises(self):
        with pytest.raises(ValueError):
            parse_bool_flag("ture")

    def test_unrecognized_raises_naming_var(self):
        with pytest.raises(ValueError) as exc:
            parse_bool_flag("maybe", name="GENO_FULL_DUPLEX")
        assert "GENO_FULL_DUPLEX" in str(exc.value)
        assert "maybe" in str(exc.value)

    def test_numeric_string_two_raises(self):
        # Only 0/1 are recognized numerics; 2 is a typo, not "on".
        with pytest.raises(ValueError):
            parse_bool_flag("2")


# ---- FullDuplexConfig defaults (the half-duplex invariant) -------------------


class TestDefaultIsHalfDuplex:
    def test_default_master_off(self):
        cfg = FullDuplexConfig()
        assert cfg.enabled is False

    def test_default_subflags_none(self):
        cfg = FullDuplexConfig()
        assert cfg.continuer_aware_listening is None
        assert cfg.agent_backchannels is None

    def test_default_all_active_resolve_false(self):
        cfg = FullDuplexConfig()
        assert cfg.continuer_aware_listening_active() is False
        assert cfg.agent_backchannels_active() is False
        assert cfg.any_active() is False

    def test_is_frozen(self):
        cfg = FullDuplexConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.enabled = True  # type: ignore[misc]


# ---- inherit logic -----------------------------------------------------------


class TestInheritLogic:
    def test_master_on_inherits_to_all_subflags(self):
        cfg = FullDuplexConfig(enabled=True)
        assert cfg.continuer_aware_listening_active() is True
        assert cfg.agent_backchannels_active() is True
        assert cfg.any_active() is True

    def test_subflag_true_overrides_master_off(self):
        # Master off but one behavior explicitly forced on.
        cfg = FullDuplexConfig(enabled=False, continuer_aware_listening=True)
        assert cfg.continuer_aware_listening_active() is True
        assert cfg.agent_backchannels_active() is False
        assert cfg.any_active() is True

    def test_subflag_false_overrides_master_on(self):
        # Organic mode on, but agent backchannels held back.
        cfg = FullDuplexConfig(enabled=True, agent_backchannels=False)
        assert cfg.continuer_aware_listening_active() is True
        assert cfg.agent_backchannels_active() is False
        # any_active still true — continuer-aware is on.
        assert cfg.any_active() is True

    def test_all_subflags_false_with_master_on_means_nothing_active(self):
        cfg = FullDuplexConfig(
            enabled=True,
            continuer_aware_listening=False,
            agent_backchannels=False,
        )
        assert cfg.continuer_aware_listening_active() is False
        assert cfg.agent_backchannels_active() is False
        assert cfg.any_active() is False


# ---- full_duplex_config_from_env ---------------------------------------------


class TestFromEnv:
    def test_empty_env_is_half_duplex(self):
        cfg = full_duplex_config_from_env({})
        assert cfg.enabled is False
        assert cfg.any_active() is False

    def test_master_flag_on(self):
        cfg = full_duplex_config_from_env({ENV_FULL_DUPLEX: "1"})
        assert cfg.enabled is True
        assert cfg.continuer_aware_listening_active() is True
        assert cfg.agent_backchannels_active() is True

    def test_master_flag_explicit_off(self):
        cfg = full_duplex_config_from_env({ENV_FULL_DUPLEX: "0"})
        assert cfg.enabled is False
        assert cfg.any_active() is False

    def test_subflag_override_on_with_master_unset(self):
        cfg = full_duplex_config_from_env({ENV_CONTINUER_AWARE: "yes"})
        assert cfg.enabled is False
        assert cfg.continuer_aware_listening_active() is True
        assert cfg.agent_backchannels_active() is False

    def test_subflag_override_off_with_master_on(self):
        cfg = full_duplex_config_from_env(
            {ENV_FULL_DUPLEX: "true", ENV_AGENT_BACKCHANNELS: "off"}
        )
        assert cfg.enabled is True
        assert cfg.continuer_aware_listening_active() is True
        assert cfg.agent_backchannels_active() is False

    def test_unset_subflag_stays_none_to_inherit(self):
        cfg = full_duplex_config_from_env({ENV_FULL_DUPLEX: "1"})
        # The raw sub-flag is None (inherits) — not coerced to True.
        assert cfg.continuer_aware_listening is None
        assert cfg.agent_backchannels is None

    def test_bad_value_propagates_valueerror(self):
        with pytest.raises(ValueError) as exc:
            full_duplex_config_from_env({ENV_FULL_DUPLEX: "enabled-please"})
        assert ENV_FULL_DUPLEX in str(exc.value)

    def test_bad_subflag_value_names_subflag(self):
        with pytest.raises(ValueError) as exc:
            full_duplex_config_from_env({ENV_AGENT_BACKCHANNELS: "ja"})
        assert ENV_AGENT_BACKCHANNELS in str(exc.value)

    def test_all_three_flags_independent(self):
        cfg = full_duplex_config_from_env(
            {
                ENV_FULL_DUPLEX: "0",
                ENV_CONTINUER_AWARE: "1",
                ENV_AGENT_BACKCHANNELS: "0",
            }
        )
        assert cfg.enabled is False
        assert cfg.continuer_aware_listening_active() is True
        assert cfg.agent_backchannels_active() is False

    def test_returns_frozen_config(self):
        cfg = full_duplex_config_from_env({ENV_FULL_DUPLEX: "1"})
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.enabled = False  # type: ignore[misc]
