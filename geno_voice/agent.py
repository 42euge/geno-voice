"""Reusable entrypoint for the geno-voice agent conversation loop.

The command-line interface and future GUI integrations share this module.
Platform/audio dependencies stay behind ``run_agent`` so importing the public
configuration types never imports PyAudio, Whisper, or Kokoro.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Protocol


class AgentMode(str, Enum):
    """Supported voice-agent turn-taking modes."""

    FULL_DUPLEX = "full-duplex"
    HALF_DUPLEX = "half-duplex"


@dataclass(frozen=True)
class AgentModeConfig:
    """Runtime switches selected by an :class:`AgentMode`."""

    full_duplex: bool
    barge_in_enabled: bool


@dataclass(frozen=True)
class AgentConfig:
    """Configuration shared by the CLI and embedding applications."""

    mode: AgentMode
    stt_model: str
    voice: str = "af_heart"
    speed: float = 1.0
    llm_config: Mapping[str, Any] | None = None
    chat_config: Mapping[str, Any] | None = None


class ChatRunner(Protocol):
    def __call__(
        self,
        model_repo: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        *,
        full_duplex: bool | None = None,
        barge_in_enabled: bool = True,
        llm_config: Mapping[str, Any] | None = None,
        chat_config: Mapping[str, Any] | None = None,
    ) -> object: ...


def mode_config(mode: AgentMode | str) -> AgentModeConfig:
    """Resolve a public mode name to the concrete runtime switches."""

    resolved = mode if isinstance(mode, AgentMode) else AgentMode(mode)
    if resolved is AgentMode.FULL_DUPLEX:
        return AgentModeConfig(full_duplex=True, barge_in_enabled=True)
    return AgentModeConfig(full_duplex=False, barge_in_enabled=False)


def run_agent(
    config: AgentConfig,
    *,
    chat_runner: Callable[..., object] | None = None,
) -> object:
    """Run the shared voice-agent pipeline in the selected duplex mode.

    ``chat_runner`` is injectable for tests and alternate hosts. The production
    import is deliberately lazy because importing ``examples.mic_chat`` loads
    platform audio modules.
    """

    if chat_runner is None:
        from examples.mic_chat import run_chat

        chat_runner = run_chat

    runtime = mode_config(config.mode)
    return chat_runner(
        model_repo=config.stt_model,
        voice=config.voice,
        speed=config.speed,
        full_duplex=runtime.full_duplex,
        barge_in_enabled=runtime.barge_in_enabled,
        llm_config=config.llm_config,
        chat_config=config.chat_config,
    )
