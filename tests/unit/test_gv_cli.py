"""Tests for iter-145 — the ``gv`` CLI entrypoint (examples/gv.py).

The CLI is the primary user-facing surface but had zero test coverage.
iter-145 extracts a testable seam: ``build_parser`` (pure parser
construction, no audio imports) and ``dispatch`` (routing with
injectable handlers). These tests exercise both without ever importing
the real STT/TTS/LLM modules — handlers are stubs that record the args
they receive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples import gv  # noqa: E402


# ---- build_parser: defaults --------------------------------------------


def test_bench_defaults():
    args = gv.build_parser().parse_args(["bench"])
    assert args.command == "bench"
    assert args.model == gv.DEFAULT_MODEL
    # bench is STT-only — no voice/speed.
    assert not hasattr(args, "voice")
    assert not hasattr(args, "speed")


def test_stream_defaults():
    args = gv.build_parser().parse_args(["stream"])
    assert args.command == "stream"
    assert args.model == gv.DEFAULT_MODEL
    assert not hasattr(args, "voice")


def test_talk_defaults():
    args = gv.build_parser().parse_args(["talk"])
    assert args.command == "talk"
    assert args.model == gv.DEFAULT_MODEL
    assert args.voice == "af_heart"
    assert args.speed == 1.0


def test_chat_defaults():
    args = gv.build_parser().parse_args(["chat"])
    assert args.command == "chat"
    assert args.model == gv.DEFAULT_MODEL
    assert args.voice == "af_heart"
    assert args.speed == 1.0


def test_no_command_is_none():
    args = gv.build_parser().parse_args([])
    assert args.command is None


# ---- build_parser: overrides -------------------------------------------


def test_model_override_all_commands():
    for cmd in ("bench", "stream", "talk", "chat"):
        args = gv.build_parser().parse_args([cmd, "--model", "tiny"])
        assert args.model == "tiny"


def test_voice_and_speed_override():
    args = gv.build_parser().parse_args(
        ["chat", "--voice", "bf_emma", "--speed", "1.25"]
    )
    assert args.voice == "bf_emma"
    assert args.speed == 1.25


def test_speed_is_float():
    args = gv.build_parser().parse_args(["talk", "--speed", "2"])
    assert isinstance(args.speed, float)
    assert args.speed == 2.0


def test_unknown_command_exits_2():
    # argparse rejects unknown subcommands with SystemExit(2).
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(["nope"])
    assert exc.value.code == 2


def test_bad_speed_exits_2():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(["chat", "--speed", "fast"])
    assert exc.value.code == 2


# ---- speed_type: bounded validator (iter-182) --------------------------


def test_speed_type_accepts_bounds_and_midrange():
    # Inclusive endpoints and an in-range value all pass through as floats.
    assert gv.speed_type(str(gv.SPEED_MIN)) == gv.SPEED_MIN
    assert gv.speed_type(str(gv.SPEED_MAX)) == gv.SPEED_MAX
    mid = gv.speed_type("1.25")
    assert isinstance(mid, float)
    assert mid == 1.25


def test_speed_type_integer_string_becomes_float():
    value = gv.speed_type("2")
    assert isinstance(value, float)
    assert value == 2.0


@pytest.mark.parametrize("raw", ["fast", "", "1.0x", "abc"])
def test_speed_type_rejects_non_numbers(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.speed_type(raw)


@pytest.mark.parametrize("raw", ["0", "-1", "-0.5"])
def test_speed_type_rejects_zero_and_negative(raw):
    # The pre-iter-182 `type=float` happily forwarded these to the TTS engine.
    with pytest.raises(argparse.ArgumentTypeError):
        gv.speed_type(raw)


@pytest.mark.parametrize("raw", ["0.49", "2.01", "10", "100"])
def test_speed_type_rejects_out_of_range(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.speed_type(raw)


def test_speed_type_rejects_nan():
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        gv.speed_type("nan")
    # The message names nan, not the bounds (NaN is unordered).
    assert "nan" in str(exc.value)


@pytest.mark.parametrize("cmd", ["talk", "chat"])
def test_parser_rejects_out_of_range_speed_via_systemexit(cmd):
    # End-to-end through argparse: an out-of-range --speed exits 2.
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args([cmd, "--speed", "0"])
    assert exc.value.code == 2


@pytest.mark.parametrize("cmd", ["talk", "chat"])
def test_parser_accepts_in_range_speed(cmd):
    args = gv.build_parser().parse_args([cmd, "--speed", "1.5"])
    assert args.speed == 1.5


# ---- voice_type: format validator (iter-183) ---------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "af_heart",   # default / curated American female
        "am_adam",    # curated American male
        "bf_emma",    # British female — valid kokoro, NOT in curated VOICES
        "bm_george",  # British male
        "af_bella",
    ],
)
def test_voice_type_accepts_well_formed_ids(raw):
    # Membership in the American-only VOICES list is NOT required — only the
    # <lang><gender>_<name> format. bf_emma is the canonical "valid but not
    # curated" case the strict-whitelist approach would have wrongly rejected.
    assert gv.voice_type(raw) == raw


def test_voice_type_accepts_every_curated_voice():
    # Every id the engine actually ships must pass the format gate.
    from tts.kokoro_engine import VOICES

    for v in VOICES:
        assert gv.voice_type(v["id"]) == v["id"]


@pytest.mark.parametrize(
    "raw",
    [
        "",            # empty
        " ",           # whitespace
        "af_heart ",   # trailing whitespace
        " af_heart",   # leading whitespace
        "heart",       # no lang/gender prefix
        "af-heart",    # wrong separator
        "afheart",     # missing underscore
        "AF_HEART",    # uppercase
        "af_",         # empty name
        "a_heart",     # missing gender letter
        "ax_heart",    # gender not f/m
        "af_heart2",   # digit in name
        "af heart",    # space in id
    ],
)
def test_voice_type_rejects_malformed_ids(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.voice_type(raw)


def test_voice_type_message_names_the_value():
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        gv.voice_type("bogus voice")
    msg = str(exc.value)
    assert "bogus voice" in msg
    assert "lang" in msg  # the grammar hint is surfaced


@pytest.mark.parametrize("cmd", ["talk", "chat"])
def test_parser_rejects_malformed_voice_via_systemexit(cmd):
    # End-to-end through argparse: a malformed --voice exits 2.
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args([cmd, "--voice", "nope!"])
    assert exc.value.code == 2


@pytest.mark.parametrize("cmd", ["talk", "chat"])
def test_parser_accepts_noncurated_voice(cmd):
    # A valid-format voice outside the curated American set still parses.
    args = gv.build_parser().parse_args([cmd, "--voice", "bf_emma"])
    assert args.voice == "bf_emma"


# ---- model_type: non-empty / no-whitespace validator (iter-184) --------


@pytest.mark.parametrize(
    "raw",
    [
        "tiny",                                    # short alias
        "base",
        "large-v3",                                # alias with version suffix
        "large-v3-turbo",
        "mlx-community/whisper-large-v3-turbo",    # full HF repo id (the default)
        "openai/whisper-tiny",                     # third-party HF repo
        "/models/whisper.gguf",                    # absolute local path
        "./local-model",                           # relative local path
        gv.DEFAULT_MODEL,                          # the actual default
    ],
)
def test_model_type_accepts_well_formed_ids(raw):
    # No single grammar — aliases, repo ids, and paths all pass. The gate
    # only rejects empty / whitespace-bearing values, so every legitimate
    # form is preserved verbatim.
    assert gv.model_type(raw) == raw


@pytest.mark.parametrize(
    "raw",
    [
        "",      # empty
        "   ",   # whitespace only
        "\t",    # tab only
        "\n",    # newline only
    ],
)
def test_model_type_rejects_empty_and_whitespace_only(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.model_type(raw)


@pytest.mark.parametrize(
    "raw",
    [
        " tiny",          # leading space
        "tiny ",          # trailing space
        "  tiny  ",       # surrounding space
        "large v3",       # embedded space
        "mlx\tcommunity", # embedded tab
        "model\nname",    # embedded newline
    ],
)
def test_model_type_rejects_whitespace_in_id(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.model_type(raw)


def test_model_type_empty_message_names_the_value():
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        gv.model_type("")
    msg = str(exc.value)
    assert "non-empty" in msg


def test_model_type_whitespace_message_names_the_value():
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        gv.model_type("large v3")
    msg = str(exc.value)
    assert "whitespace" in msg
    assert "large v3" in msg


@pytest.mark.parametrize("cmd", ["bench", "stream", "talk", "chat"])
def test_parser_rejects_whitespace_model_via_systemexit(cmd):
    # End-to-end through argparse: a whitespace-bearing --model exits 2.
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args([cmd, "--model", "large v3"])
    assert exc.value.code == 2


@pytest.mark.parametrize("cmd", ["bench", "stream", "talk", "chat"])
def test_parser_accepts_well_formed_model(cmd):
    # A well-formed alias parses through on every command.
    args = gv.build_parser().parse_args([cmd, "--model", "tiny"])
    assert args.model == "tiny"


# ---- dispatch: routing with stub handlers ------------------------------


def _recording_handlers():
    """Build a handler map whose handlers record the args they get."""
    calls: list = []

    def make(name):
        def handler(args):
            calls.append((name, args))
        return handler

    handlers = {name: make(name) for name in ("bench", "stream", "talk", "chat")}
    return handlers, calls


@pytest.mark.parametrize("cmd", ["bench", "stream", "talk", "chat"])
def test_dispatch_routes_to_handler(cmd):
    parser = gv.build_parser()
    args = parser.parse_args([cmd])
    handlers, calls = _recording_handlers()

    rc = gv.dispatch(args, parser, handlers=handlers)

    assert rc == 0
    assert len(calls) == 1
    assert calls[0][0] == cmd
    assert calls[0][1] is args


def test_dispatch_passes_parsed_overrides_through():
    parser = gv.build_parser()
    args = parser.parse_args(["chat", "--voice", "bf_emma", "--speed", "1.5"])
    handlers, calls = _recording_handlers()

    gv.dispatch(args, parser, handlers=handlers)

    _, got = calls[0]
    assert got.voice == "bf_emma"
    assert got.speed == 1.5
    assert got.command == "chat"


def test_dispatch_no_command_prints_help_returns_1(capsys):
    parser = gv.build_parser()
    args = parser.parse_args([])
    handlers, calls = _recording_handlers()

    rc = gv.dispatch(args, parser, handlers=handlers)

    assert rc == 1
    assert calls == []  # no handler invoked
    out = capsys.readouterr().out
    assert "usage" in out.lower()


def test_dispatch_unknown_command_prints_help_returns_1(capsys):
    # A command not in the handler map (defensive — argparse normally
    # blocks unknown commands, but a custom handler map might omit one).
    parser = gv.build_parser()
    args = parser.parse_args(["chat"])
    handlers, calls = _recording_handlers()
    del handlers["chat"]

    rc = gv.dispatch(args, parser, handlers=handlers)

    assert rc == 1
    assert calls == []
    assert "usage" in capsys.readouterr().out.lower()


def test_dispatch_default_handlers_are_the_real_cmds():
    # The default handler map wires the real command functions.
    assert gv.DEFAULT_HANDLERS == {
        "bench": gv.cmd_bench,
        "stream": gv.cmd_stream,
        "talk": gv.cmd_talk,
        "chat": gv.cmd_chat,
        "simulate-mirror": gv.cmd_simulate_mirror,
        "calibrate-base-wpm": gv.cmd_calibrate_base_wpm,
        "vad": gv.cmd_vad,
        "vad-gaps": gv.cmd_vad_gaps,
        "vad-gap-percentiles": gv.cmd_vad_gap_percentiles,
        "vad-gap-cdf": gv.cmd_vad_gap_cdf,
        "vad-gap-recommend": gv.cmd_vad_gap_recommend,
        "vad-gap-recommend-diff": gv.cmd_vad_gap_recommend_diff,
        "vad-gap-recommend-sweep": gv.cmd_vad_gap_recommend_sweep,
        "vad-gap-recommend-knob-sweep": gv.cmd_vad_gap_recommend_knob_sweep,
        "vad-gap-recommend-knob-grid": gv.cmd_vad_gap_recommend_knob_grid,
        "vad-gap-confidence": gv.cmd_vad_gap_confidence,
        "vad-gap-cost": gv.cmd_vad_gap_cost,
        "vad-gap-peak": gv.cmd_vad_gap_peak,
        "vad-gap-hist": gv.cmd_vad_gap_histogram,
        "vad-gap-sweep": gv.cmd_vad_gap_sweep,
        "vad-gap-peak-sweep": gv.cmd_vad_gap_peak_sweep,
        "vad-diff": gv.cmd_vad_diff,
        "vad-gap-diff": gv.cmd_vad_gap_diff,
        "vad-sweep": gv.cmd_vad_sweep,
        "vad-grid": gv.cmd_vad_grid,
        "vad-gap-grid": gv.cmd_vad_gap_grid,
        "vad-gap-peak-grid": gv.cmd_vad_gap_peak_grid,
    }


# ---- main: end-to-end with injected handlers ---------------------------


def test_main_dispatches_via_default_path(monkeypatch):
    # Patch the handler map so main() routes a real argv without touching
    # audio modules.
    calls: list = []
    monkeypatch.setitem(gv.DEFAULT_HANDLERS, "stream", lambda a: calls.append(a))

    rc = gv.main(["stream", "--model", "tiny"])

    assert rc == 0
    assert len(calls) == 1
    assert calls[0].model == "tiny"


def test_main_no_args_returns_1(capsys):
    rc = gv.main([])
    assert rc == 1
    assert "usage" in capsys.readouterr().out.lower()


# ---- cmd_bench: argv rebuild logic -------------------------------------


def test_cmd_bench_default_model_omits_model_arg(monkeypatch):
    captured = {}

    def fake_bench_main():
        captured["argv"] = list(sys.argv)

    import types
    fake_mod = types.ModuleType("mic_bench")
    fake_mod.main = fake_bench_main
    monkeypatch.setitem(sys.modules, "mic_bench", fake_mod)

    args = gv.build_parser().parse_args(["bench"])
    gv.cmd_bench(args)

    assert captured["argv"] == ["gv bench"]


def test_cmd_bench_custom_model_forwarded(monkeypatch):
    captured = {}

    def fake_bench_main():
        captured["argv"] = list(sys.argv)

    import types
    fake_mod = types.ModuleType("mic_bench")
    fake_mod.main = fake_bench_main
    monkeypatch.setitem(sys.modules, "mic_bench", fake_mod)

    args = gv.build_parser().parse_args(["bench", "--model", "tiny"])
    gv.cmd_bench(args)

    assert captured["argv"] == ["gv bench", "--model", "tiny"]
