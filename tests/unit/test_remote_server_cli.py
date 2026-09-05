"""CLI coverage for the dedicated ``geno-voice-remote-server`` executable."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from geno_voice.endpoint import cli as endpoint_cli


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("WS", "websocket"),
        ("websocket", "websocket"),
        ("GRPC", "grpc"),
        ("WebRTC", "webrtc"),
    ],
)
def test_remote_server_parser_accepts_protocol_without_subcommand(
    raw,
    expected,
) -> None:
    args = endpoint_cli.build_parser().parse_args(
        [f"--protocol={raw}", "--model=Breeze-TTS-2"]
    )

    assert args.protocol == expected
    assert args.model == "Breeze-TTS-2"


@pytest.mark.parametrize("protocol", ["rtp"])
def test_remote_server_rejects_protocols_not_in_this_release(protocol) -> None:
    with pytest.raises(SystemExit):
        endpoint_cli.build_parser().parse_args(["--protocol", protocol])


def test_remote_server_main_launches_endpoint_from_direct_options(
    monkeypatch,
) -> None:
    seen = []
    monkeypatch.setattr(endpoint_cli, "run_endpoint", seen.append)

    result = endpoint_cli.main(
        [
            "--protocol",
            "ws",
            "--model",
            "kokoro",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--model-path",
            "/models/kokoro",
            "--runtime-path",
            "/opt/kokoro",
            "--device",
            "cuda:0",
            "--voice",
            "af_heart",
            "--log-level",
            "debug",
        ]
    )

    assert result == 0
    assert seen == [
        endpoint_cli.EndpointConfig(
            protocol="websocket",
            model="kokoro",
            host="0.0.0.0",
            port=9000,
            model_path=Path("/models/kokoro"),
            runtime_path=Path("/opt/kokoro"),
            device="cuda:0",
            voice="af_heart",
            log_level="debug",
        )
    ]


def test_remote_server_lists_models_without_starting_one(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(endpoint_cli, "print_models", lambda: calls.append("listed"))
    monkeypatch.setattr(
        endpoint_cli,
        "run_endpoint",
        lambda config: calls.append("started"),
    )

    assert endpoint_cli.main(["--list-models"]) == 0
    assert calls == ["listed"]


def test_remote_server_help_does_not_import_optional_protocol_packages() -> None:
    for module in ("fastapi", "uvicorn", "grpc", "aiortc", "av"):
        sys.modules.pop(module, None)

    with pytest.raises(SystemExit) as exc:
        endpoint_cli.main(["--help"])

    assert exc.value.code == 0
    assert not ({"fastapi", "uvicorn", "grpc", "aiortc", "av"} & sys.modules.keys())
