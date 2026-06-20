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


# ---- iter-315: --csv parser wiring -------------------------------------


def test_simulate_mirror_csv_defaults_false():
    args = gv.build_parser().parse_args(["simulate-mirror", "--wpms", "120,200"])
    assert args.csv is False


def test_simulate_mirror_csv_flag_sets_true():
    args = gv.build_parser().parse_args(
        ["simulate-mirror", "--wpms", "120,200", "--csv"]
    )
    assert args.csv is True


# ---- iter-315: render_grid_csv -----------------------------------------


import csv as _csv  # noqa: E402
import io as _io  # noqa: E402


def _parse_csv(text):
    """Parse a CSV string (the renderers strip the trailing terminator)."""
    return list(_csv.reader(_io.StringIO(text)))


def test_render_grid_csv_header_and_one_row_per_cell():
    wm = gv._load_wpm_mirror()
    points = wm.sweep_mirror_grid(
        [120, 200, 120], [165.0, 180.0], [0.5], initial_speed=1.0
    )
    best = wm.pick_best_mirror_config(points)
    rows = _parse_csv(gv.render_grid_csv(points, best))
    assert rows[0] == [
        "base_wpm",
        "strength",
        "final_speed",
        "final_gap",
        "max_step",
        "moves",
        "score",
        "is_best",
    ]
    # One row per cell, no prose footer (contrast render_grid's +3 lines).
    assert len(rows) == len(points) + 1


def test_render_grid_csv_marks_exactly_the_best_cell():
    wm = gv._load_wpm_mirror()
    points = wm.sweep_mirror_grid(
        [120, 200, 120], [165.0, 180.0], [0.4, 0.6], initial_speed=1.0
    )
    best = wm.pick_best_mirror_config(points)
    rows = _parse_csv(gv.render_grid_csv(points, best))
    body = rows[1:]
    best_flags = [r[-1] for r in body]
    # Exactly one cell flagged, and it is the picked (base_wpm, strength) pair.
    assert best_flags.count("1") == 1
    flagged = body[best_flags.index("1")]
    assert float(flagged[0]) == best.base_wpm
    assert float(flagged[1]) == best.strength


def test_render_grid_csv_no_scorable_cell_leaves_is_best_zero():
    wm = gv._load_wpm_mirror()
    # An all-non-measurable arc produces no scorable cell → best is None.
    points = wm.sweep_mirror_grid([0, -1], [165.0, 180.0], [0.5])
    best = wm.pick_best_mirror_config(points)
    assert best is None
    rows = _parse_csv(gv.render_grid_csv(points, best))
    body = rows[1:]
    # Every is_best is 0, and unscorable cells carry empty gap/score fields.
    assert all(r[-1] == "0" for r in body)
    for r in body:
        assert r[3] == ""  # final_gap
        assert r[6] == ""  # score


def test_render_grid_csv_no_trailing_newline():
    wm = gv._load_wpm_mirror()
    points = wm.sweep_mirror_grid([120, 200], [165.0], [0.5])
    best = wm.pick_best_mirror_config(points)
    out = gv.render_grid_csv(points, best)
    assert not out.endswith("\n")
    assert not out.endswith("\r")


def test_render_grid_csv_rounds_to_three_places():
    wm = gv._load_wpm_mirror()
    points = wm.sweep_mirror_grid([120, 200, 140], [165.0], [0.5])
    best = wm.pick_best_mirror_config(points)
    rows = _parse_csv(gv.render_grid_csv(points, best))
    # Numeric fields with decimals carry at most 3 fractional digits.
    for r in rows[1:]:
        for field in (r[2], r[4]):  # final_speed, max_step
            if "." in field:
                assert len(field.split(".")[1]) <= 3


# ---- iter-315: render_trajectory_csv -----------------------------------


def test_render_trajectory_csv_one_row_per_turn_with_wpm():
    wm = gv._load_wpm_mirror()
    cfg = wm.WpmMirrorConfig(enabled=True, base_wpm=165.0, strength=0.5)
    wpms = [120.0, 200.0, 120.0]
    traj = wm.simulate_speed_trajectory(wpms, initial_speed=1.0, config=cfg)
    rows = _parse_csv(gv.render_trajectory_csv(traj, wpms=wpms))
    assert rows[0] == ["turn", "user_wpm", "speed"]
    assert len(rows) == len(traj.speeds) + 1
    # 1-based turn index, paired user_wpm, and the engine's speed per row.
    for i, (row, speed) in enumerate(zip(rows[1:], traj.speeds), start=1):
        assert int(row[0]) == i
        assert float(row[1]) == wpms[i - 1]
        assert float(row[2]) == round(speed, 3)


def test_render_trajectory_csv_empty_arc_header_only():
    traj = _traj(speeds=[], initial_speed=1.0, final_speed=1.0)
    rows = _parse_csv(gv.render_trajectory_csv(traj, wpms=[]))
    assert rows == [["turn", "user_wpm", "speed"]]


def test_render_trajectory_csv_unpaired_wpms_leaves_user_wpm_empty():
    # When wpms length mismatches the speed count, user_wpm is left blank but the
    # speeds still emit (the speed column is the load-bearing one).
    traj = _traj(speeds=[0.9, 0.85], initial_speed=1.0, final_speed=0.85)
    rows = _parse_csv(gv.render_trajectory_csv(traj, wpms=[120.0]))
    body = rows[1:]
    assert [r[1] for r in body] == ["", ""]
    assert [float(r[2]) for r in body] == [0.9, 0.85]


def test_render_trajectory_csv_no_wpms_arg_leaves_user_wpm_empty():
    traj = _traj(speeds=[0.95], initial_speed=1.0, final_speed=0.95)
    rows = _parse_csv(gv.render_trajectory_csv(traj))
    assert rows[1][1] == ""
    assert float(rows[1][2]) == 0.95


def test_render_trajectory_csv_no_trailing_newline():
    traj = _traj(speeds=[0.9], initial_speed=1.0, final_speed=0.9)
    out = gv.render_trajectory_csv(traj, wpms=[120.0])
    assert not out.endswith("\n")
    assert not out.endswith("\r")


# ---- iter-315: handler --csv routing -----------------------------------


def test_handler_trajectory_csv_matches_renderer():
    wm = gv._load_wpm_mirror()
    cfg = wm.WpmMirrorConfig(enabled=True, base_wpm=165.0, strength=0.5)
    wpms = [120.0, 200.0, 120.0]
    traj = wm.simulate_speed_trajectory(wpms, initial_speed=1.0, config=cfg)
    expected = gv.render_trajectory_csv(traj, wpms=wpms)
    lines = _run(["simulate-mirror", "--wpms", "120,200,120", "--csv"])
    # The handler logs the CSV as one block (one log() call).
    assert lines == [expected]


def test_handler_grid_csv_matches_renderer():
    wm = gv._load_wpm_mirror()
    points = wm.sweep_mirror_grid(
        [120, 140, 200, 140, 120], [165.0, 180.0], [0.5], initial_speed=1.0
    )
    best = wm.pick_best_mirror_config(points)
    expected = gv.render_grid_csv(points, best)
    lines = _run(
        [
            "simulate-mirror",
            "--wpms", "120,140,200,140,120",
            "--grid",
            "--base-wpms", "165,180",
            "--strengths", "0.5",
            "--csv",
        ]
    )
    assert lines == [expected]


def test_handler_csv_trajectory_is_parseable_per_turn():
    lines = _run(["simulate-mirror", "--wpms", "120,200,120", "--csv"])
    rows = _parse_csv("\n".join(lines))
    assert rows[0] == ["turn", "user_wpm", "speed"]
    # Three input WPMs → three turn rows.
    assert len(rows) == 4
    assert [r[1] for r in rows[1:]] == ["120.0", "200.0", "120.0"]


def test_handler_csv_default_log_is_print(capsys):
    args = gv.build_parser().parse_args(
        ["simulate-mirror", "--wpms", "120,200", "--csv"]
    )
    gv.cmd_simulate_mirror(args)
    out = capsys.readouterr().out
    assert out.startswith("turn,user_wpm,speed")


# ---- iter-317: --json parser wiring ------------------------------------


import json as _json  # noqa: E402


def test_simulate_mirror_json_defaults_false():
    args = gv.build_parser().parse_args(["simulate-mirror", "--wpms", "120,200"])
    assert args.json is False


def test_simulate_mirror_json_flag_sets_true():
    args = gv.build_parser().parse_args(
        ["simulate-mirror", "--wpms", "120,200", "--json"]
    )
    assert args.json is True


def test_simulate_mirror_json_and_csv_mutually_exclusive():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(
            ["simulate-mirror", "--wpms", "120,200", "--json", "--csv"]
        )
    assert exc.value.code == 2


# ---- iter-317: render_grid_json ----------------------------------------


def test_render_grid_json_cells_and_best():
    wm = gv._load_wpm_mirror()
    points = wm.sweep_mirror_grid(
        [120, 200, 120], [165.0, 180.0], [0.5], initial_speed=1.0
    )
    best = wm.pick_best_mirror_config(points)
    payload = _json.loads(gv.render_grid_json(points, best))
    assert payload["mode"] == "grid"
    # One cell object per swept point.
    assert len(payload["cells"]) == len(points)
    cell = payload["cells"][0]
    assert set(cell) == {
        "base_wpm", "strength", "final_speed", "final_gap",
        "max_step", "moves", "score",
    }
    # best is the picked cell as the same shape.
    assert payload["best"]["base_wpm"] == best.base_wpm
    assert payload["best"]["strength"] == best.strength


def test_render_grid_json_values_match_engine():
    wm = gv._load_wpm_mirror()
    points = wm.sweep_mirror_grid([120, 200, 120], [165.0], [0.5])
    best = wm.pick_best_mirror_config(points)
    payload = _json.loads(gv.render_grid_json(points, best))
    for cell, p in zip(payload["cells"], points):
        assert cell["base_wpm"] == p.base_wpm
        assert cell["strength"] == p.strength
        assert cell["final_speed"] == round(p.final_speed, 3)
        assert cell["max_step"] == round(p.max_step, 3)
        assert cell["moves"] == p.moves
        assert cell["score"] == round(p.score(), 3)


def test_render_grid_json_unscorable_cell_nulls():
    wm = gv._load_wpm_mirror()
    # All-non-measurable arc → no scorable cell → best None, gap/score null.
    points = wm.sweep_mirror_grid([0, -1], [165.0, 180.0], [0.5])
    best = wm.pick_best_mirror_config(points)
    assert best is None
    payload = _json.loads(gv.render_grid_json(points, best))
    assert payload["best"] is None
    for cell in payload["cells"]:
        assert cell["final_gap"] is None
        assert cell["score"] is None


def test_render_grid_json_rounds_to_three_places():
    wm = gv._load_wpm_mirror()
    points = wm.sweep_mirror_grid([120, 200, 140], [165.0], [0.5])
    best = wm.pick_best_mirror_config(points)
    payload = _json.loads(gv.render_grid_json(points, best))
    for cell in payload["cells"]:
        for key in ("final_speed", "max_step", "score"):
            val = cell[key]
            if isinstance(val, float):
                assert round(val, 3) == val


# ---- iter-317: render_trajectory_json ----------------------------------


def test_render_trajectory_json_turns_and_scalars():
    wm = gv._load_wpm_mirror()
    cfg = wm.WpmMirrorConfig(enabled=True, base_wpm=165.0, strength=0.5)
    wpms = [120.0, 200.0, 120.0]
    traj = wm.simulate_speed_trajectory(wpms, initial_speed=1.0, config=cfg)
    payload = _json.loads(gv.render_trajectory_json(traj, wpms=wpms))
    assert payload["mode"] == "trajectory"
    # Arc-level scalars carried at top level (unlike the per-turn CSV).
    assert payload["initial_speed"] == round(traj.initial_speed, 3)
    assert payload["final_speed"] == round(traj.final_speed, 3)
    assert payload["moves"] == traj.moves
    # One turn object per speed, 1-based, paired user_wpm.
    assert len(payload["turns"]) == len(traj.speeds)
    for i, (t, speed) in enumerate(zip(payload["turns"], traj.speeds), start=1):
        assert t["turn"] == i
        assert t["user_wpm"] == wpms[i - 1]
        assert t["speed"] == round(speed, 3)


def test_render_trajectory_json_empty_arc_keeps_scalars():
    traj = _traj(speeds=[], initial_speed=1.0, final_speed=1.0)
    payload = _json.loads(gv.render_trajectory_json(traj, wpms=[]))
    assert payload["turns"] == []
    # The scalars are still present even with no turns.
    assert payload["initial_speed"] == 1.0
    assert payload["final_speed"] == 1.0
    assert payload["ideal_final_speed"] is None
    assert payload["final_gap"] is None


def test_render_trajectory_json_unpaired_wpms_user_wpm_null():
    traj = _traj(speeds=[0.9, 0.85], initial_speed=1.0, final_speed=0.85)
    payload = _json.loads(gv.render_trajectory_json(traj, wpms=[120.0]))
    assert [t["user_wpm"] for t in payload["turns"]] == [None, None]
    assert [t["speed"] for t in payload["turns"]] == [0.9, 0.85]


def test_render_trajectory_json_no_wpms_arg_user_wpm_null():
    traj = _traj(speeds=[0.95], initial_speed=1.0, final_speed=0.95)
    payload = _json.loads(gv.render_trajectory_json(traj))
    assert payload["turns"][0]["user_wpm"] is None
    assert payload["turns"][0]["speed"] == 0.95


# ---- iter-317: handler --json routing ----------------------------------


def test_handler_trajectory_json_matches_renderer():
    wm = gv._load_wpm_mirror()
    cfg = wm.WpmMirrorConfig(enabled=True, base_wpm=165.0, strength=0.5)
    wpms = [120.0, 200.0, 120.0]
    traj = wm.simulate_speed_trajectory(wpms, initial_speed=1.0, config=cfg)
    expected = gv.render_trajectory_json(traj, wpms=wpms)
    lines = _run(["simulate-mirror", "--wpms", "120,200,120", "--json"])
    assert lines == [expected]


def test_handler_grid_json_matches_renderer():
    wm = gv._load_wpm_mirror()
    points = wm.sweep_mirror_grid(
        [120, 140, 200, 140, 120], [165.0, 180.0], [0.5], initial_speed=1.0
    )
    best = wm.pick_best_mirror_config(points)
    expected = gv.render_grid_json(points, best)
    lines = _run(
        [
            "simulate-mirror",
            "--wpms", "120,140,200,140,120",
            "--grid",
            "--base-wpms", "165,180",
            "--strengths", "0.5",
            "--json",
        ]
    )
    assert lines == [expected]


def test_handler_json_trajectory_is_parseable():
    lines = _run(["simulate-mirror", "--wpms", "120,200,120", "--json"])
    payload = _json.loads("\n".join(lines))
    assert payload["mode"] == "trajectory"
    assert len(payload["turns"]) == 3
    assert [t["user_wpm"] for t in payload["turns"]] == [120.0, 200.0, 120.0]


def test_handler_json_default_log_is_print(capsys):
    args = gv.build_parser().parse_args(
        ["simulate-mirror", "--wpms", "120,200", "--json"]
    )
    gv.cmd_simulate_mirror(args)
    out = capsys.readouterr().out
    payload = _json.loads(out)
    assert payload["mode"] == "trajectory"
