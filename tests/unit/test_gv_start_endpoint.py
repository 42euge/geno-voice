"""CLI coverage for ``geno-voice start-endpoint``."""

from __future__ import annotations

import sys

import pytest

from examples import gv
from geno_voice.endpoint import cli as endpoint_cli


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("WebRTC", "webrtc"),
        ("WS", "websocket"),
        ("websocket", "websocket"),
        ("GRPC", "grpc"),
        ("Rtp", "rtp"),
    ],
)
def test_start_endpoint_parser_normalizes_protocol_aliases(raw, expected) -> None:
    args = gv.build_parser().parse_args(
        ["start-endpoint", f"--protocol={raw}", "--model=Breeze-TTS-2"]
    )

    assert args.protocol == expected
    assert args.model == "Breeze-TTS-2"


def test_start_endpoint_parser_rejects_unknown_protocol() -> None:
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(
            ["start-endpoint", "--protocol", "http", "--model", "kokoro"]
        )

    assert exc.value.code == 2


def test_dispatch_passes_endpoint_config_without_importing_models(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(endpoint_cli, "run_endpoint", seen.append)
    sys.modules.pop("geno_voice.endpoint.models.kokoro", None)
    sys.modules.pop("geno_voice.endpoint.models.breeze", None)

    result = gv.main(
        [
            "start-endpoint",
            "--protocol",
            "ws",
            "--model",
            "kokoro",
            "--host",
            "0.0.0.0",
            "--voice",
            "af_heart",
        ]
    )

    assert result == 0
    assert seen == [
        endpoint_cli.EndpointConfig(
            protocol="websocket",
            model="kokoro",
            host="0.0.0.0",
            port=8_765,
            voice="af_heart",
        )
    ]
    assert "geno_voice.endpoint.models.kokoro" not in sys.modules
    assert "geno_voice.endpoint.models.breeze" not in sys.modules


def test_protocol_default_port_is_applied_by_endpoint_config() -> None:
    args = gv.build_parser().parse_args(
        ["start-endpoint", "--protocol", "grpc", "--model", "kokoro"]
    )

    config = endpoint_cli.endpoint_config_from_args(args)

    assert config.port == 50_051


def test_list_models_exits_without_starting_a_model(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        endpoint_cli,
        "print_models",
        lambda: calls.append("listed"),
        raising=False,
    )
    monkeypatch.setattr(
        endpoint_cli,
        "run_endpoint",
        lambda config: calls.append("started"),
    )

    assert gv.main(["start-endpoint", "--list-models"]) == 0
    assert calls == ["listed"]


def test_start_endpoint_help_does_not_import_optional_protocol_packages() -> None:
    for module in ("fastapi", "uvicorn", "grpc", "aiortc", "av"):
        sys.modules.pop(module, None)

    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(["start-endpoint", "--help"])

    assert exc.value.code == 0
    assert not ({"fastapi", "uvicorn", "grpc", "aiortc", "av"} & sys.modules.keys())
