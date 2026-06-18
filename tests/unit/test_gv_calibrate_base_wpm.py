"""Tests for iter-221 — the ``gv calibrate-base-wpm`` subcommand (examples/gv.py).

iter-220 shipped the audio-free calibration core (``CalibrationSample`` /
``calibrate_base_wpm``): the pure arithmetic that turns a rendered TTS sample
(``words`` synthesized into ``audio_seconds`` of audio at a known Kokoro
``speed``) into a measured ``base_wpm``. iter-221 exposes that core on the
``gv`` CLI — the iter-218-style CLI-later twin of the iter-216 engine — so an
operator can fold rendered samples offline and read the implied ``base_wpm`` to
set ``DEFAULT_BASE_WPM`` from their own voice.

These tests exercise the ``--samples`` arg type, the pure ``render_calibration``
helper, and the handler (driven with an injected ``log`` so no real I/O
happens). The engine is pure stdlib loaded by file path, so the handler runs on
this x86_64 Linux runner without pipecat.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples import gv  # noqa: E402


# ---- parser: defaults & wiring -----------------------------------------


def test_calibrate_in_handler_map():
    assert gv.DEFAULT_HANDLERS["calibrate-base-wpm"] is gv.cmd_calibrate_base_wpm


def test_calibrate_defaults():
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm", "--samples", "50:18.2"]
    )
    assert args.command == "calibrate-base-wpm"
    assert args.samples == [(50.0, 18.2, 1.0)]
    # The nominal default is sourced from the engine seed.
    assert args.nominal == 165.0


def test_calibrate_requires_samples():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(["calibrate-base-wpm"])
    assert exc.value.code == 2


def test_calibrate_multiple_samples_and_nominal():
    args = gv.build_parser().parse_args(
        [
            "calibrate-base-wpm",
            "--samples", "50:18.2", "50:9.1:2.0", "30:12.0:1.0",
            "--nominal", "170",
        ]
    )
    assert args.samples == [
        (50.0, 18.2, 1.0),
        (50.0, 9.1, 2.0),
        (30.0, 12.0, 1.0),
    ]
    assert args.nominal == 170.0


# ---- calibration_sample_type: pure arg parsing -------------------------


def test_sample_type_two_fields_defaults_speed():
    assert gv.calibration_sample_type("50:18.2") == (50.0, 18.2, 1.0)


def test_sample_type_three_fields():
    assert gv.calibration_sample_type("50:9.1:2.0") == (50.0, 9.1, 2.0)


def test_sample_type_strips_whitespace():
    assert gv.calibration_sample_type(" 50 : 18.2 : 1.5 ") == (50.0, 18.2, 1.5)


@pytest.mark.parametrize(
    "raw",
    [
        "50",            # too few fields
        "50:18:1:9",     # too many fields
        "50::1",         # empty middle field
        ":18:1",         # empty leading field
        "",              # empty
    ],
)
def test_sample_type_rejects_bad_shape(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.calibration_sample_type(raw)


@pytest.mark.parametrize("raw", ["abc:18", "50:xx", "50:18:zz"])
def test_sample_type_rejects_non_numeric(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.calibration_sample_type(raw)


@pytest.mark.parametrize("raw", ["nan:18", "50:nan", "50:18:nan"])
def test_sample_type_rejects_nan(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.calibration_sample_type(raw)


@pytest.mark.parametrize("raw", ["0:18", "50:0", "50:18:0", "-5:18", "50:-1"])
def test_sample_type_rejects_non_positive(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.calibration_sample_type(raw)


def test_sample_type_rejects_non_string():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.calibration_sample_type(50)


def test_sample_type_end_to_end_systemexit():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(["calibrate-base-wpm", "--samples", "bad"])
    assert exc.value.code == 2


# ---- render_calibration: pure formatting -------------------------------


def test_render_calibration_none_no_samples():
    lines = gv.render_calibration(None)
    assert lines == [
        "base_wpm calibration: no samples (nothing to calibrate from)"
    ]


def test_render_calibration_all_fields():
    wm = gv._load_wpm_mirror()
    # 50 words / 18.2s at speed 1.0 → ~164.8 wpm.
    samples = [wm.CalibrationSample(words=50, audio_seconds=18.2)]
    calib = wm.calibrate_base_wpm(samples)
    lines = gv.render_calibration(calib)
    text = "\n".join(lines)
    assert "base_wpm calibration from rendered samples" in text
    assert "samples:" in text
    assert "implied base_wpm:" in text
    assert "range:" in text
    assert "spread:" in text
    assert "nominal:" in text
    assert "drift:" in text
    assert f"{calib.implied_base_wpm:.1f}" in text


# ---- cmd_calibrate_base_wpm: handler with injected log -----------------


def _run(argv):
    """Parse argv and run the handler, capturing the emitted log lines."""
    args = gv.build_parser().parse_args(argv)
    lines: list = []
    gv.cmd_calibrate_base_wpm(args, log=lines.append)
    return lines


def test_handler_emits_report():
    lines = _run(["calibrate-base-wpm", "--samples", "50:18.2"])
    text = "\n".join(lines)
    assert "calibration from rendered samples" in text
    assert "implied base_wpm" in text


def test_handler_matches_engine_directly():
    # The handler's report reflects the same fold the engine runs directly.
    wm = gv._load_wpm_mirror()
    samples = [
        wm.CalibrationSample(words=50, audio_seconds=18.2),
        wm.CalibrationSample(words=50, audio_seconds=9.1, speed=2.0),
    ]
    calib = wm.calibrate_base_wpm(samples, default_base_wpm=165.0)
    expected = gv.render_calibration(calib)
    lines = _run(["calibrate-base-wpm", "--samples", "50:18.2", "50:9.1:2.0"])
    assert lines == expected


def test_handler_nominal_threads_to_drift():
    # The --nominal override is reported as the drift baseline.
    lines = _run(["calibrate-base-wpm", "--samples", "50:18.2", "--nominal", "100"])
    text = "\n".join(lines)
    assert "100.0" in text  # nominal line


def test_handler_default_log_is_print(capsys):
    args = gv.build_parser().parse_args(["calibrate-base-wpm", "--samples", "50:18.2"])
    gv.cmd_calibrate_base_wpm(args)
    out = capsys.readouterr().out
    assert "calibration from rendered samples" in out


def test_handler_dispatch_routes(capsys):
    # End-to-end through main(): the subcommand dispatches and prints.
    rc = gv.main(["calibrate-base-wpm", "--samples", "50:18.2"])
    assert rc == 0
    assert "calibration from rendered samples" in capsys.readouterr().out
