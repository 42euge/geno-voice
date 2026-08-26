"""Tests for the reusable geno-voice agent-mode seam."""

from __future__ import annotations

import subprocess
import sys

import pytest

from geno_voice.agent import (
    AgentConfig,
    AgentMode,
    mode_config,
    run_agent,
)


def test_mode_config_selects_real_runtime_switches():
    full = mode_config(AgentMode.FULL_DUPLEX)
    half = mode_config(AgentMode.HALF_DUPLEX)

    assert full.full_duplex is True
    assert full.barge_in_enabled is True
    assert half.full_duplex is False
    assert half.barge_in_enabled is False


def test_mode_config_rejects_unknown_mode():
    with pytest.raises(ValueError):
        mode_config("quarter-duplex")


@pytest.mark.parametrize(
    ("mode", "expected_full", "expected_barge"),
    [
        (AgentMode.FULL_DUPLEX, True, True),
        (AgentMode.HALF_DUPLEX, False, False),
    ],
)
def test_run_agent_injects_mode_into_shared_runner(
    mode, expected_full, expected_barge
):
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        return "finished"

    result = run_agent(
        AgentConfig(
            mode=mode,
            stt_model="faster-whisper/large-v3",
            voice="bf_emma",
            speed=1.25,
            llm_config={
                "model": "blue-model",
                "base_url": "https://litellm.example/v1",
                "api_key": "test-only",
            },
            chat_config={"stt_engine": "faster_whisper"},
        ),
        chat_runner=runner,
    )

    assert result == "finished"
    assert calls == [
        {
            "model_repo": "faster-whisper/large-v3",
            "voice": "bf_emma",
            "speed": 1.25,
            "full_duplex": expected_full,
            "barge_in_enabled": expected_barge,
            "llm_config": {
                "model": "blue-model",
                "base_url": "https://litellm.example/v1",
                "api_key": "test-only",
            },
            "chat_config": {"stt_engine": "faster_whisper"},
        }
    ]


def test_importing_agent_api_does_not_import_platform_audio():
    # Check in a clean interpreter so other collected tests cannot affect the
    # module cache. OpenCode must be able to import config types without audio.
    code = (
        "import sys; import geno_voice.agent; "
        "assert 'examples.mic_chat' not in sys.modules; "
        "assert 'pyaudio' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0
