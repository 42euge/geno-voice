"""CLI coverage for ``geno-voice agent <mode>``."""

from __future__ import annotations

import pytest

from examples import gv
from geno_voice import agent as agent_api


@pytest.mark.parametrize("mode", ["full-duplex", "half-duplex"])
def test_agent_mode_parses_with_defaults(mode):
    args = gv.build_parser().parse_args(["agent", mode])

    assert args.command == "agent"
    assert args.agent_mode == mode
    assert args.model == gv.DEFAULT_MODEL
    assert args.voice == "af_heart"
    assert args.speed == 1.0


def test_agent_options_parse_and_stt_alias_is_clear():
    args = gv.build_parser().parse_args(
        [
            "agent",
            "full-duplex",
            "--stt-model",
            "large-v3",
            "--voice",
            "bf_emma",
            "--speed",
            "1.25",
        ]
    )

    assert args.model == "large-v3"
    assert args.voice == "bf_emma"
    assert args.speed == 1.25


def test_agent_requires_a_mode():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(["agent"])
    assert exc.value.code == 2


def test_agent_rejects_unknown_mode():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(["agent", "quarter-duplex"])
    assert exc.value.code == 2


def test_agent_dispatch_invokes_shared_runner(monkeypatch):
    calls = []
    monkeypatch.setattr(agent_api, "run_agent", calls.append)
    args = gv.build_parser().parse_args(["agent", "half-duplex"])

    assert gv.dispatch(args, gv.build_parser()) == 0
    assert len(calls) == 1
    assert calls[0].mode is agent_api.AgentMode.HALF_DUPLEX
