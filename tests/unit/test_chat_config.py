"""Tests for examples/_chat_config.py — config parsing + validation.

Previously load_llm_config did file I/O, parsing, validation, and
sys.exit all in one function — couldn't be tested. Now the pure
logic is isolated in parse_llm_config / parse_chat_config and
these tests exercise it directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_config import (  # noqa: E402
    ConfigError,
    DEFAULT_MAX_TOKENS,
    parse_chat_config,
    parse_llm_config,
)


def _valid_cfg(**overrides) -> dict:
    """Minimal valid config; overrides merged into ``llm``."""
    llm = {
        "model": "claude-test",
        "base_url": "https://api.example.com",
        "api_key": "secret",
    }
    llm.update(overrides)
    return {"llm": llm}


# ---- happy path -------------------------------------------------------------


class TestParseLlmHappyPath:
    def test_minimum_valid_config(self):
        out = parse_llm_config(_valid_cfg())
        assert out["model"] == "claude-test"
        assert out["base_url"] == "https://api.example.com"
        assert out["api_key"] == "secret"
        assert out["max_tokens"] == DEFAULT_MAX_TOKENS

    def test_max_tokens_passed_through(self):
        out = parse_llm_config(_valid_cfg(max_tokens=512))
        assert out["max_tokens"] == 512

    def test_extra_fields_passed_through(self):
        out = parse_llm_config(_valid_cfg(system_prompt="be nice", temperature=0.7))
        assert out["system_prompt"] == "be nice"
        assert out["temperature"] == 0.7

    def test_base_url_trailing_slash_stripped(self):
        out = parse_llm_config(_valid_cfg(base_url="https://api.example.com/"))
        assert out["base_url"] == "https://api.example.com"

    def test_base_url_multiple_trailing_slashes_stripped(self):
        out = parse_llm_config(_valid_cfg(base_url="https://api.example.com////"))
        assert out["base_url"] == "https://api.example.com"

    def test_returns_a_new_dict_does_not_mutate_input(self):
        cfg = _valid_cfg()
        original_llm = dict(cfg["llm"])
        out = parse_llm_config(cfg)
        out["api_key"] = "MUTATED"
        # Input dict's llm section not affected.
        assert cfg["llm"] == original_llm


# ---- env-var resolution -----------------------------------------------------


class TestEnvVarResolution:
    def test_set_env_var_resolves(self):
        out = parse_llm_config(
            _valid_cfg(api_key="${MY_KEY}"),
            env={"MY_KEY": "actual-value"},
        )
        assert out["api_key"] == "actual-value"

    def test_unset_env_var_raises(self):
        with pytest.raises(ConfigError) as exc:
            parse_llm_config(_valid_cfg(api_key="${MISSING}"), env={})
        assert "MISSING" in str(exc.value)
        assert "not set" in str(exc.value)

    def test_empty_env_var_value_raises(self):
        with pytest.raises(ConfigError):
            parse_llm_config(_valid_cfg(api_key="${EMPTY}"), env={"EMPTY": ""})

    def test_empty_placeholder_raises(self):
        with pytest.raises(ConfigError) as exc:
            parse_llm_config(_valid_cfg(api_key="${}"), env={})
        assert "empty" in str(exc.value).lower()

    def test_literal_dollarsign_string_not_treated_as_placeholder(self):
        # "${X" without closing brace shouldn't trigger resolution.
        out = parse_llm_config(_valid_cfg(api_key="${X"), env={})
        assert out["api_key"] == "${X"

    def test_dollarsign_not_at_start_not_resolved(self):
        # "abc${X}" doesn't start with "${" so isn't a placeholder.
        out = parse_llm_config(
            _valid_cfg(api_key="abc${X}"),
            env={"X": "ignored"},
        )
        assert out["api_key"] == "abc${X}"

    def test_resolution_only_happens_when_placeholder_full_value(self):
        # Per the original load_llm_config, only api_key that is
        # exactly ``${VAR}`` is resolved. Partial templates aren't
        # supported. Document the contract.
        out = parse_llm_config(
            _valid_cfg(api_key="prefix-${KEY}"),
            env={"KEY": "value"},
        )
        assert out["api_key"] == "prefix-${KEY}"


# ---- structural failures ----------------------------------------------------


class TestStructuralFailures:
    def test_none_input_raises(self):
        with pytest.raises(ConfigError) as exc:
            parse_llm_config(None)
        assert "empty" in str(exc.value).lower()

    def test_non_mapping_top_level_raises(self):
        with pytest.raises(ConfigError) as exc:
            parse_llm_config(["not", "a", "dict"])
        assert "mapping" in str(exc.value).lower()

    def test_missing_llm_section_raises(self):
        with pytest.raises(ConfigError) as exc:
            parse_llm_config({"chat": {}})
        assert "'llm' section" in str(exc.value)

    def test_llm_is_none_raises(self):
        with pytest.raises(ConfigError):
            parse_llm_config({"llm": None})

    def test_llm_is_not_a_mapping_raises(self):
        with pytest.raises(ConfigError) as exc:
            parse_llm_config({"llm": "wrong-type"})
        assert "mapping" in str(exc.value).lower()


# ---- required field validation ---------------------------------------------


class TestRequiredFields:
    @pytest.mark.parametrize("field", ["model", "base_url", "api_key"])
    def test_missing_required_field_raises(self, field):
        cfg = _valid_cfg()
        del cfg["llm"][field]
        with pytest.raises(ConfigError) as exc:
            parse_llm_config(cfg)
        assert field in str(exc.value)

    @pytest.mark.parametrize("field", ["model", "base_url", "api_key"])
    def test_empty_string_required_field_raises(self, field):
        cfg = _valid_cfg(**{field: ""})
        with pytest.raises(ConfigError) as exc:
            parse_llm_config(cfg)
        assert field in str(exc.value)

    @pytest.mark.parametrize("field", ["model", "base_url", "api_key"])
    def test_non_string_required_field_raises(self, field):
        cfg = _valid_cfg(**{field: 123})
        with pytest.raises(ConfigError):
            parse_llm_config(cfg)

    def test_error_message_names_the_missing_field(self):
        cfg = _valid_cfg()
        del cfg["llm"]["model"]
        with pytest.raises(ConfigError) as exc:
            parse_llm_config(cfg)
        assert "'llm.model'" in str(exc.value)


# ---- parse_chat_config ------------------------------------------------------


class TestParseChatConfig:
    def test_no_chat_section_returns_empty_dict(self):
        assert parse_chat_config({"llm": {}}) == {}

    def test_chat_is_none_returns_empty_dict(self):
        assert parse_chat_config({"chat": None}) == {}

    def test_chat_is_not_dict_returns_empty_dict(self):
        # Tolerant — caller doesn't want to crash on weird YAML.
        assert parse_chat_config({"chat": ["not", "dict"]}) == {}

    def test_top_level_none_returns_empty_dict(self):
        assert parse_chat_config(None) == {}

    def test_top_level_non_dict_returns_empty_dict(self):
        assert parse_chat_config("not a dict") == {}

    def test_chat_section_passed_through(self):
        out = parse_chat_config(
            {"chat": {"fillers": ["hmm", "well"], "fillers_idle_threshold": 0.5}}
        )
        assert out == {"fillers": ["hmm", "well"], "fillers_idle_threshold": 0.5}

    def test_returns_a_new_dict_does_not_mutate_input(self):
        cfg = {"chat": {"fillers": ["a"]}}
        out = parse_chat_config(cfg)
        out["fillers"].append("MUTATED")
        # The fillers list IS the same object (Python copy semantics
        # — dict() does a shallow copy). Document this contract.
        # If you want a deep-immutable view, the caller wraps.
        # The important invariant is: mutating the OUTER dict
        # doesn't alter cfg["chat"].
        out["new_key"] = "x"
        assert "new_key" not in cfg["chat"]
