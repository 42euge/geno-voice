"""Lazy built-in and entry-point discovery for endpoint TTS models."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

from .types import TTSModelAdapter


def normalize_model_name(name: str) -> str:
    normalized = re.sub(r"[\s_]+", "-", name.strip().lower())
    return re.sub(r"-+", "-", normalized)


@dataclass(frozen=True)
class ModelConfig:
    model_path: Path | None = None
    runtime_path: Path | None = None
    device: str | None = None
    voice: str | None = None


@dataclass(frozen=True)
class ModelDescriptor:
    canonical_name: str
    aliases: tuple[str, ...]
    origin: str
    _factory: Callable[[ModelConfig], TTSModelAdapter]

    def create(self, config: ModelConfig) -> TTSModelAdapter:
        return self._factory(config)


def _import_symbol(target: str) -> Any:
    module_name, separator, attribute = target.partition(":")
    if not separator:
        raise ValueError(f"invalid adapter target {target!r}; expected module:attribute")
    return getattr(importlib.import_module(module_name), attribute)


def _builtin_factory(target: str, kind: str) -> Callable[[ModelConfig], TTSModelAdapter]:
    def create(config: ModelConfig) -> TTSModelAdapter:
        adapter_class = _import_symbol(target)
        if kind == "breeze":
            return adapter_class(
                model_path=config.model_path,
                runtime_path=config.runtime_path,
                device=config.device,
            )
        return adapter_class(default_voice=config.voice or "af_heart")

    return create


class ModelRegistry:
    ENTRY_POINT_GROUP = "geno_voice.tts_models"

    def __init__(self, *, load_plugins: bool = True) -> None:
        self._descriptors: dict[str, ModelDescriptor] = {}
        self._canonical: dict[str, ModelDescriptor] = {}
        self._register(
            canonical_name="breeze-tts-2",
            aliases=("breeze",),
            origin="geno_voice.endpoint.models.breeze:BreezeTTS2Adapter",
            factory=_builtin_factory(
                "geno_voice.endpoint.models.breeze:BreezeTTS2Adapter", "breeze"
            ),
        )
        self._register(
            canonical_name="kokoro",
            aliases=(),
            origin="geno_voice.endpoint.models.kokoro:KokoroAdapter",
            factory=_builtin_factory(
                "geno_voice.endpoint.models.kokoro:KokoroAdapter", "kokoro"
            ),
        )
        if load_plugins:
            self._discover_plugins()

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._canonical))

    def resolve(self, name: str) -> ModelDescriptor:
        normalized = normalize_model_name(name)
        try:
            return self._descriptors[normalized]
        except KeyError as exc:
            available = ", ".join(self.names())
            raise ValueError(
                f"unknown TTS model {name!r}; available models: {available}"
            ) from exc

    def create(self, name: str, config: ModelConfig) -> TTSModelAdapter:
        descriptor = self.resolve(name)
        try:
            return descriptor.create(config)
        except Exception as exc:
            raise RuntimeError(
                f"failed to create TTS adapter {descriptor.canonical_name!r} "
                f"from {descriptor.origin}: {exc}"
            ) from exc

    def _discover_plugins(self) -> None:
        entry_points = metadata.entry_points(group=self.ENTRY_POINT_GROUP)
        for entry_point in entry_points:
            origin = getattr(entry_point, "value", repr(entry_point))

            def create(config: ModelConfig, entry_point=entry_point):
                factory = entry_point.load()
                return factory(config)

            self._register(
                canonical_name=entry_point.name,
                aliases=(),
                origin=origin,
                factory=create,
            )

    def _register(
        self,
        *,
        canonical_name: str,
        aliases: tuple[str, ...],
        origin: str,
        factory: Callable[[ModelConfig], TTSModelAdapter],
    ) -> None:
        canonical = normalize_model_name(canonical_name)
        normalized_aliases = tuple(normalize_model_name(alias) for alias in aliases)
        descriptor = ModelDescriptor(
            canonical_name=canonical,
            aliases=normalized_aliases,
            origin=origin,
            _factory=factory,
        )
        for name in (canonical, *normalized_aliases):
            existing = self._descriptors.get(name)
            if existing is not None:
                raise ValueError(
                    f"model adapter {origin} uses reserved name {name!r} "
                    f"owned by {existing.origin}"
                )
        self._canonical[canonical] = descriptor
        for name in (canonical, *normalized_aliases):
            self._descriptors[name] = descriptor
