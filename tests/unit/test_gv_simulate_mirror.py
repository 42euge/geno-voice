"""Tests for iter-218 — the ``gv simulate-mirror`` subcommand (examples/gv.py).

iter-216 shipped ``simulate_speed_trajectory`` and iter-217 shipped
``sweep_mirror_grid`` / ``pick_best_mirror_config`` — the offline twins of the
live ``SpeedController`` fold for validating the WPM-mirror tunables. iter-218
exposes them on the ``gv`` CLI so an operator can replay a user-WPM arc through
the mirror offline (no audio, no live session) — the named follow-on in the
iter-216/217 backlog.

These tests exercise the parser arg types, the pure render helpers, and the
handler (driven with an injected ``log`` so no real I/O happens). The engine is
pure stdlib loaded by file path, so the handler runs on this x86_64 Linux runner
without pipecat.
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


def test_simulate_mirror_in_handler_map():
    assert gv.DEFAULT_HANDLERS["simulate-mirror"] is gv.cmd_simulate_mirror


def test_simulate_mirror_defaults():
    args = gv.build_parser().parse_args(["simulate-mirror", "--wpms", "120,200,120"])
    assert args.command == "simulate-mirror"
    assert args.wpms == [120.0, 200.0, 120.0]
    assert args.initial_speed == 1.0
    assert args.grid is False
    # Seed defaults are sourced from the engine so they match the live config.
    assert args.base_wpm == 165.0
    assert args.strength == 0.5
    assert args.base_wpms == [150.0, 165.0, 180.0]
    assert args.strengths == [0.3, 0.5, 0.7]


def test_simulate_mirror_requires_wpms():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(["simulate-mirror"])
    assert exc.value.code == 2


def test_simulate_mirror_overrides():
    args = gv.build_parser().parse_args(
        [
            "simulate-mirror",
            "--wpms", "100,180",
            "--initial-speed", "0.9",
            "--base-wpm", "150",
            "--strength", "0.7",
        ]
    )
    assert args.wpms == [100.0, 180.0]
    assert args.initial_speed == 0.9
    assert args.base_wpm == 150.0
    assert args.strength == 0.7


def test_simulate_mirror_grid_flag_and_axes():
    args = gv.build_parser().parse_args(
        [
            "simulate-mirror",
            "--wpms", "120,200,120",
            "--grid",
            "--base-wpms", "160,170",
            "--strengths", "0.4,0.6",
        ]
    )
    assert args.grid is True
    assert args.base_wpms == [160.0, 170.0]
    assert args.strengths == [0.4, 0.6]


# ---- wpm_list_type: comma-separated arc parser -------------------------


def test_wpm_list_type_parses_floats():
    assert gv.wpm_list_type("120,140.5,200") == [120.0, 140.5, 200.0]


def test_wpm_list_type_strips_whitespace_and_blanks():
    # Surrounding spaces and a stray trailing comma are tolerated.
    assert gv.wpm_list_type(" 120 , 140 ,") == [120.0, 140.0]


def test_wpm_list_type_allows_nonpositive_marker():
    # A <=0 value is the iter-064 "no measurement that turn" marker — allowed so
    # an operator can model a silent / one-word turn in the arc.
    assert gv.wpm_list_type("120,0,-5,140") == [120.0, 0.0, -5.0, 140.0]


@pytest.mark.parametrize("raw", ["", "  ", ",", " , "])
def test_wpm_list_type_rejects_empty(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.wpm_list_type(raw)


@pytest.mark.parametrize("raw", ["120,fast", "abc", "120,,x"])
def test_wpm_list_type_rejects_non_numbers(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.wpm_list_type(raw)


def test_wpm_list_type_rejects_nan():
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        gv.wpm_list_type("120,nan")
    assert "nan" in str(exc.value)


def test_parser_rejects_bad_wpms_via_systemexit():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(["simulate-mirror", "--wpms", "fast"])
    assert exc.value.code == 2


# ---- positive_floats_type: grid axes -----------------------------------


def test_positive_floats_type_parses():
    assert gv.positive_floats_type("150,165,180") == [150.0, 165.0, 180.0]


@pytest.mark.parametrize("raw", ["0", "-1", "150,-2", "150,0"])
def test_positive_floats_type_rejects_nonpositive(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.positive_floats_type(raw)


@pytest.mark.parametrize("raw", ["", " ", ","])
def test_positive_floats_type_rejects_empty(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.positive_floats_type(raw)


@pytest.mark.parametrize("raw", ["abc", "150,x"])
def test_positive_floats_type_rejects_non_numbers(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.positive_floats_type(raw)


def test_positive_floats_type_rejects_nan():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.positive_floats_type("150,nan")


# ---- render_trajectory: pure formatting --------------------------------


def _traj(**kw):
    """Build a SpeedTrajectory via the engine loaded by file path."""
    wm = gv._load_wpm_mirror()
    return wm.SpeedTrajectory(**kw)


def test_render_trajectory_basic_fields():
    traj = _traj(
        speeds=[0.9, 0.85],
        initial_speed=1.0,
        final_speed=0.85,
        ideal_final_speed=0.8,
        final_gap=0.05,
        max_step=0.1,
        moves=2,
    )
    lines = gv.render_trajectory(traj)
    text = "\n".join(lines)
    assert "trajectory simulation" in text.lower()
    assert "1.000" in text  # initial speed
    assert "0.850" in text  # final speed
    assert "0.800" in text  # ideal
    assert "+0.050" in text  # signed gap
    assert "0.100" in text  # max step
    assert "2 of 2 turns" in text
    assert "0.900, 0.850" in text  # per-turn speeds


def test_render_trajectory_disabled_shows_na():
    # No measurable turn / disabled — ideal and gap render as n/a, no per-turn
    # line when speeds is empty.
    traj = _traj(speeds=[], initial_speed=1.0, final_speed=1.0)
    lines = gv.render_trajectory(traj)
    text = "\n".join(lines)
    assert "n/a" in text
    assert "per-turn speeds" not in text


# ---- render_grid: pure formatting --------------------------------------


def test_render_grid_rows_and_best():
    wm = gv._load_wpm_mirror()
    points = wm.sweep_mirror_grid(
        [120, 200, 120], [165.0, 180.0], [0.5], initial_speed=1.0
    )
    best = wm.pick_best_mirror_config(points)
    lines = gv.render_grid(points, best)
    text = "\n".join(lines)
    assert "grid sweep" in text.lower()
    # Title + column-header + one row per cell + best line.
    assert len(lines) == len(points) + 3
    assert "best: base_wpm=" in text
    assert f"{best.base_wpm:.1f}" in text


def test_render_grid_no_scorable_cell():
    wm = gv._load_wpm_mirror()
    # An all-non-measurable arc produces no scorable cell.
    points = wm.sweep_mirror_grid([0, -1], [165.0], [0.5])
    best = wm.pick_best_mirror_config(points)
    lines = gv.render_grid(points, best)
    text = "\n".join(lines)
    assert best is None
    assert "no scorable cell" in text
    assert "n/a" in text


# ---- cmd_simulate_mirror: handler with injected log --------------------


def _run(argv):
    """Parse argv and run the handler, capturing the emitted log lines."""
    args = gv.build_parser().parse_args(argv)
    lines: list = []
    gv.cmd_simulate_mirror(args, log=lines.append)
    return lines


def test_handler_trajectory_mode_emits_report():
    lines = _run(["simulate-mirror", "--wpms", "120,140,200,140,120"])
    text = "\n".join(lines)
    assert "trajectory simulation" in text.lower()
    assert "initial speed" in text
    assert "final speed" in text
    assert "per-turn speeds" in text


def test_handler_grid_mode_emits_table_and_pick():
    lines = _run(
        [
            "simulate-mirror",
            "--wpms", "120,140,200,140,120",
            "--grid",
            "--base-wpms", "165,180",
            "--strengths", "0.5",
        ]
    )
    text = "\n".join(lines)
    assert "grid sweep" in text.lower()
    assert "best: base_wpm=" in text


def test_handler_trajectory_matches_engine_directly():
    # The handler's report reflects the same fold the engine runs directly.
    wm = gv._load_wpm_mirror()
    cfg = wm.WpmMirrorConfig(enabled=True, base_wpm=165.0, strength=0.5)
    traj = wm.simulate_speed_trajectory(
        [120.0, 200.0, 120.0], initial_speed=1.0, config=cfg
    )
    expected = gv.render_trajectory(traj)
    lines = _run(["simulate-mirror", "--wpms", "120,200,120"])
    assert lines == expected


def test_handler_default_log_is_print(capsys):
    # With no injected log the handler prints to stdout (default print).
    args = gv.build_parser().parse_args(["simulate-mirror", "--wpms", "120,200"])
    gv.cmd_simulate_mirror(args)
    out = capsys.readouterr().out
    assert "trajectory simulation" in out.lower()


def test_handler_dispatch_routes(capsys):
    # End-to-end through main(): the subcommand dispatches and prints.
    rc = gv.main(["simulate-mirror", "--wpms", "120,200,120"])
    assert rc == 0
    assert "trajectory" in capsys.readouterr().out.lower()
