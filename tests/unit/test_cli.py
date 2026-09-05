"""Tests for the installed ``geno-voice`` command."""

from __future__ import annotations

from geno_voice import cli
from geno_voice.endpoint import cli as endpoint_cli


def test_start_endpoint_delegates_to_remote_server(monkeypatch) -> None:
    seen: list[list[str]] = []

    def fake_endpoint_main(argv: list[str]) -> int:
        seen.append(argv)
        return 0

    monkeypatch.setattr(endpoint_cli, "main", fake_endpoint_main)

    result = cli.main(
        ["start-endpoint", "--protocol", "ws", "--model", "kokoro"]
    )

    assert result == 0
    assert seen == [["--protocol", "ws", "--model", "kokoro"]]
