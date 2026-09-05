"""Tests for lazy, model-agnostic endpoint adapter discovery."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import pytest

from geno_voice.endpoint.registry import ModelConfig, ModelRegistry


def test_registry_normalizes_alias_without_importing_adapter() -> None:
    sys.modules.pop("geno_voice.endpoint.models.breeze", None)
    registry = ModelRegistry(load_plugins=False)

    descriptor = registry.resolve(" Breeze_TTS 2 ")

    assert descriptor.canonical_name == "breeze-tts-2"
    assert "geno_voice.endpoint.models.breeze" not in sys.modules
    assert registry.names() == ("breeze-tts-2", "kokoro")


def test_registry_creates_builtin_only_when_selected() -> None:
    registry = ModelRegistry(load_plugins=False)

    adapter = registry.create(
        "kokoro", ModelConfig(model_path=None, voice="af_heart")
    )

    assert adapter.name == "kokoro"
    assert adapter.default_voice == "af_heart"


@dataclass
class FakeEntryPoint:
    name: str
    value: str
    factory: object
    loads: int = 0

    def load(self):
        self.loads += 1
        return self.factory


def test_plugin_factory_is_discovered_but_loaded_only_on_create(monkeypatch) -> None:
    configs: list[ModelConfig] = []

    class Adapter:
        name = "robot"

    def factory(config: ModelConfig):
        configs.append(config)
        return Adapter()

    entry_point = FakeEntryPoint("Robot Voice", "robot_package:create", factory)
    monkeypatch.setattr(
        "geno_voice.endpoint.registry.metadata.entry_points",
        lambda **kwargs: [entry_point],
    )
    registry = ModelRegistry()

    assert registry.resolve("robot_voice").origin == "robot_package:create"
    assert entry_point.loads == 0

    config = ModelConfig(device="cuda:1")
    assert registry.create("robot-voice", config).name == "robot"
    assert entry_point.loads == 1
    assert configs == [config]


def test_plugin_cannot_shadow_a_builtin_alias(monkeypatch) -> None:
    entry_point = FakeEntryPoint("BREEZE", "shadow:create", lambda config: object())
    monkeypatch.setattr(
        "geno_voice.endpoint.registry.metadata.entry_points",
        lambda **kwargs: [entry_point],
    )

    with pytest.raises(ValueError, match="shadow:create.*reserved"):
        ModelRegistry()


def test_unknown_model_error_lists_canonical_models() -> None:
    registry = ModelRegistry(load_plugins=False)

    with pytest.raises(ValueError, match="breeze-tts-2, kokoro"):
        registry.resolve("missing")
