"""Tests for iter-332 — the ``gv vad-gap-grid`` subcommand (examples/gv.py).

iter-330 shipped ``gv vad-gap-sweep`` — the inter-segment silence-gap
distribution across ONE swept knob; iter-331 pinned it against the real corpus.
This lap adds its 2-D analogue: ``gv vad-gap-grid`` is the gap-side twin of
``gv vad-grid`` (and the 2-D analogue of ``gv vad-gap-sweep``). Where
``vad-grid`` tabulates segment-count / speech-seconds per cell of a gate ×
column-knob grid, ``vad-gap-grid`` tabulates the min/mean/max gap per cell so an
operator can watch the shortest-pause floor MOVE across two knobs at once — the
cell that lifts the min gap clear of a target end-of-turn hangover
(``--min-silence-ms`` / the live ``chat.vad.silence_duration``) is the one that
buys merge headroom.

Like the rest of the VAD-analysis family, the handler takes injected
``segmenter`` / ``availability`` / ``log`` dependencies so every test runs
WITHOUT importing torch / silero-vad and without touching real audio — fast and
deterministic on the x86_64 Linux runner. The pure core (``vad_gap_grid``) and
the three renderers are exercised directly against lightweight stand-ins
mirroring just the ``SileroResult`` / ``SpeechSegment`` attributes they read.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples import gv  # noqa: E402


# ---- lightweight stand-ins for the SileroResult / SpeechSegment shapes ----


@dataclass
class _Seg:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class _Result:
    name: str
    segments: List[_Seg] = field(default_factory=list)

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    @property
    def speech_s(self) -> float:
        return sum(s.duration_s for s in self.segments)


def _result(*pairs, name="rec.wav"):
    return _Result(name=name, segments=[_Seg(a, b) for a, b in pairs])


# Two recurring stand-ins: a 3-segment result (2 gaps: 1.0s and 2.0s) and a
# single-segment result (no inter-segment pause).
def _three():
    return _result((0.0, 1.0), (2.0, 3.0), (5.0, 6.0))


def _single():
    return _result((0.0, 6.0))


# A 2-segment result with a single 3.0s inter-segment gap (used by the 2×2
# golden so each grid cell exercises a distinct gap-count shape).
def _two():
    return _result((0.0, 1.0), (4.0, 5.0))


# ---- parser: registration & defaults -----------------------------------


def test_vad_gap_grid_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-gap-grid"] is gv.cmd_vad_gap_grid


def test_parser_default_axes():
    args = gv.build_parser().parse_args(["vad-gap-grid", "rec.wav"])
    assert args.command == "vad-gap-grid"
    # Rows are always the gate; the default column axis is the hangover.
    assert args.thresholds == [0.3, 0.5, 0.7, 0.9]
    assert args.min_silences == [400.0, 600.0, 800.0, 1000.0]
    # The other column axes default to None (not provided).
    assert args.min_speeches is None
    assert args.speech_pads is None
    assert args.max_speeches is None


def test_parser_held_scalars_mirror_silero_params():
    """The held-fixed scalar knobs default to the same values as ``gv vad``."""
    args = gv.build_parser().parse_args(["vad-gap-grid", "rec.wav"])
    vad = gv.build_parser().parse_args(["vad", "rec.wav"])
    assert args.min_speech_ms == vad.min_speech_ms
    assert args.min_silence_ms == vad.min_silence_ms
    assert args.speech_pad_ms == vad.speech_pad_ms
    assert args.max_speech_s == vad.max_speech_s


def test_parser_column_axes_are_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["vad-gap-grid", "rec.wav", "--min-silences", "400,800",
             "--min-speeches", "50,100"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["vad-gap-grid", "rec.wav", "--speech-pads", "0,20",
             "--max-speeches", "5,inf"]
        )


def test_parser_json_csv_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-grid", "rec.wav", "--json", "--csv"])


def test_parser_has_no_target_pick_args():
    """Unlike vad-grid, the gap grid has no --target / --top / --tie-break."""
    args = gv.build_parser().parse_args(["vad-gap-grid", "rec.wav"])
    assert not hasattr(args, "target")
    assert not hasattr(args, "top")
    assert not hasattr(args, "tie_break")


# ---- pure core: vad_gap_grid --------------------------------------------


def test_core_basic_two_by_one_grid():
    cells = gv.vad_gap_grid([0.3, 0.9], [400.0], [_three(), _single()])
    assert cells[0] == {
        "threshold": 0.3,
        "min_silence_ms": 400.0,
        "num_segments": 3,
        "num_gaps": 2,
        "min_gap_s": 1.0,
        "mean_gap_s": 1.5,
        "max_gap_s": 2.0,
        "total_silence_s": 3.0,
    }
    # The single-segment cell has no pause: aggregates are None, NOT 0.0.
    assert cells[1] == {
        "threshold": 0.9,
        "min_silence_ms": 400.0,
        "num_segments": 1,
        "num_gaps": 0,
        "min_gap_s": None,
        "mean_gap_s": None,
        "max_gap_s": None,
        "total_silence_s": 0.0,
    }


def test_core_row_major_order():
    """Cells flatten row 0's columns first, then row 1's — the same order
    vad_segmentation_grid uses."""
    cells = gv.vad_gap_grid(
        [0.3, 0.9], [400.0, 800.0],
        [_three(), _single(), _single(), _three()],
    )
    coords = [(c["threshold"], c["min_silence_ms"]) for c in cells]
    assert coords == [(0.3, 400.0), (0.3, 800.0), (0.9, 400.0), (0.9, 800.0)]


def test_core_axis_keys_follow_axis_args():
    cells = gv.vad_gap_grid(
        [0.5], [50.0, 200.0], [_three(), _three()],
        col_axis="min_speech_ms",
    )
    assert "min_speech_ms" in cells[0]
    assert "min_silence_ms" not in cells[0]
    assert cells[0]["min_speech_ms"] == 50.0


def test_core_aggregates_match_vad_silence_gaps():
    """Each cell's aggregates equal an independent vad_silence_gaps on its
    result — the grid differences the SAME segmentation gap core does."""
    r = _three()
    direct = gv.vad_silence_gaps(r)
    cell = gv.vad_gap_grid([0.5], [400.0], [r])[0]
    for key in ("num_segments", "num_gaps", "min_gap_s", "mean_gap_s",
                "max_gap_s", "total_silence_s"):
        assert cell[key] == direct[key]


def test_core_min_gap_tracks_shortest_pause():
    cell = gv.vad_gap_grid([0.5], [400.0], [_three()])[0]
    assert cell["min_gap_s"] == 1.0  # the 1.0s pause, not the 2.0s one


def test_core_length_mismatch_raises():
    with pytest.raises(ValueError):
        gv.vad_gap_grid([0.3, 0.9], [400.0, 800.0], [_three()])


def test_core_empty_grid_is_empty():
    assert gv.vad_gap_grid([], [], []) == []


# ---- renderer: render_vad_gap_grid (human) ------------------------------


def test_render_human_header_and_rows():
    lines = gv.render_vad_gap_grid(
        [0.3, 0.9], [400.0], [_three(), _single()], name="rec.wav"
    )
    text = "\n".join(lines)
    assert lines[0] == "silero VAD gap grid — rec.wav (threshold × min_silence)"
    # The header names the two axes and the gap columns.
    assert "threshold" in lines[1]
    assert "min_silence" in lines[1]
    assert "min_gap" in lines[1]
    assert "mean_gap" in lines[1]
    assert "max_gap" in lines[1]
    # The 3-segment cell shows numeric gaps; the single-segment cell shows -.
    assert "1.000" in text
    assert "-" in lines[-1]


def test_render_human_single_segment_cell_dashes_aggregates():
    lines = gv.render_vad_gap_grid([0.9], [400.0], [_single()], name="rec.wav")
    data_row = lines[-1]
    assert "0.000" not in data_row
    assert "-" in data_row


def test_render_human_col_axis_label_for_min_speech():
    lines = gv.render_vad_gap_grid(
        [0.5], [50.0, 200.0], [_three(), _three()],
        name="rec.wav", col_axis="min_speech_ms",
    )
    assert "min_speech" in lines[0]
    assert "min_speech" in lines[1]


def test_render_human_unavailable():
    lines = gv.render_vad_gap_grid([], [], [None], name="rec.wav")
    assert len(lines) == 1
    assert lines[0].startswith("silero VAD unavailable")


def test_render_human_golden_full_matrix():
    """Byte-for-byte golden of a 2×2 threshold × min_silence grid.

    iter-340/341/342 pinned percentiles / histogram / diff verbatim; the grid
    is the next-most-alignment-sensitive surface because it is a 2-D table —
    column drift compounds across BOTH rows and columns. The other gap human
    reports (percentiles, histogram, diff) are golden-pinned, but the grid's
    `test_render_human_header_and_rows` only asserts structure + substrings
    ("threshold" appears, "1.000" appears *somewhere*, the last line has a
    "-"), so a silent regression in the two `{:>11}` swept-value columns, the
    `{:>8}`/`{:>4}` count fields, the `{:>7.3f}`/`{:>8.3f}` gap columns, the
    two-space gutters, or the row-major cell ordering would slip through.

    The four cells are deliberately distinct gap shapes: 2 gaps (_three), 1 gap
    (_two), 2 gaps again (row-major proves order — the second row repeats the
    first column's segmentations under the higher gate), and the <2-segment
    dash cell (_single). Results are supplied in row-major order: (0.30, 200),
    (0.30, 400), (0.90, 200), (0.90, 400).
    """
    lines = gv.render_vad_gap_grid(
        [0.3, 0.9],
        [200.0, 400.0],
        [_three(), _two(), _three(), _single()],
        name="rec.wav",
    )
    assert lines == [
        "silero VAD gap grid — rec.wav (threshold × min_silence)",
        "    threshold  min_silence  segments  gaps  "
        "min_gap  mean_gap   max_gap",
        "         0.30          200         3     2    "
        "1.000     1.500     2.000",
        "         0.30          400         2     1    "
        "3.000     3.000     3.000",
        "         0.90          200         3     2    "
        "1.000     1.500     2.000",
        "         0.90          400         1     0        "
        "-         -         -",
    ]
    # The dash cell's min_gap placeholder lines up with the numeric cells'
    # min_gap column above: both the right-justified `1.000` value and the `-`
    # placeholder end at the same character offset as the `min_gap` header
    # label (the `{:>7.3f}` / `{:>7}` fields share a width).
    min_gap_end = lines[1].index("min_gap") + len("min_gap")
    assert lines[2].index("1.000") + len("1.000") == min_gap_end
    assert lines[5].index("-") + len("-") == min_gap_end


def test_render_human_golden_seconds_column_axis():
    """Byte-for-byte golden exercising the `%g` seconds column-axis format.

    The default `min_silence` column is a millisecond knob (`{:.0f}` → bare
    `200`); switching `col_axis` to a seconds axis (`max_speech_s`) routes the
    column through `_format_sweep_axis_value`'s `%g` branch, so `5.0` renders as
    a compact `5` (not `5.000`) and `12.5` keeps its fractional part. This pins
    that the header label switches to `max_speech` AND the `%g` value spelling,
    which `test_render_human_col_axis_label_for_min_speech` only checks for the
    label substring, never the rendered value column.
    """
    lines = gv.render_vad_gap_grid(
        [0.5],
        [5.0, 12.5],
        [_three(), _single()],
        name="rec.wav",
        col_axis="max_speech_s",
    )
    assert lines == [
        "silero VAD gap grid — rec.wav (threshold × max_speech)",
        "    threshold   max_speech  segments  gaps  "
        "min_gap  mean_gap   max_gap",
        "         0.50            5         3     2    "
        "1.000     1.500     2.000",
        "         0.50         12.5         1     0        "
        "-         -         -",
    ]


# ---- renderer: render_vad_gap_grid_json ---------------------------------


def test_render_json_shape():
    text = gv.render_vad_gap_grid_json(
        [0.3, 0.9], [400.0], [_three(), _single()], name="rec.wav"
    )
    payload = json.loads(text)
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["row_axis"] == "threshold"
    assert payload["col_axis"] == "min_silence_ms"
    assert len(payload["grid"]) == 2
    assert payload["grid"][0]["num_gaps"] == 2
    assert payload["grid"][0]["min_gap_s"] == 1.0


def test_render_json_single_segment_aggregates_null():
    text = gv.render_vad_gap_grid_json([0.9], [400.0], [_single()], name="rec.wav")
    cell = json.loads(text)["grid"][0]
    assert cell["num_gaps"] == 0
    assert cell["min_gap_s"] is None
    assert cell["mean_gap_s"] is None
    assert cell["max_gap_s"] is None
    assert cell["total_silence_s"] == 0.0


def test_render_json_axis_names_carried():
    text = gv.render_vad_gap_grid_json(
        [0.5], [5.0], [_three()], name="rec.wav", col_axis="max_speech_s"
    )
    payload = json.loads(text)
    assert payload["col_axis"] == "max_speech_s"
    assert "max_speech_s" in payload["grid"][0]


def test_render_json_no_target_keys():
    """The gap grid never emits a target/best/top pick block."""
    payload = json.loads(
        gv.render_vad_gap_grid_json([0.5], [400.0], [_three()], name="rec.wav")
    )
    assert "target" not in payload
    assert "best" not in payload
    assert "top" not in payload


def test_render_json_unavailable():
    payload = json.loads(
        gv.render_vad_gap_grid_json([], [], [None], name="rec.wav")
    )
    assert payload["available"] is False
    assert "hint" in payload


# ---- renderer: render_vad_gap_grid_csv ----------------------------------


def test_render_csv_header_and_rows():
    text = gv.render_vad_gap_grid_csv(
        [0.3, 0.9], [400.0], [_three(), _single()], name="rec.wav"
    )
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == [
        "threshold", "min_silence_ms", "num_segments", "num_gaps",
        "min_gap_s", "mean_gap_s", "max_gap_s", "total_silence_s",
    ]
    assert rows[1] == ["0.3", "400.0", "3", "2", "1.0", "1.5", "2.0", "3.0"]


def test_render_csv_single_segment_empty_aggregate_cells():
    text = gv.render_vad_gap_grid_csv([0.9], [400.0], [_single()], name="rec.wav")
    rows = list(csv.reader(io.StringIO(text)))
    # The aggregate columns are empty (the CSV spelling of JSON null), not 0.0.
    assert rows[1] == ["0.9", "400.0", "1", "0", "", "", "", "0"]


def test_render_csv_axis_headers():
    text = gv.render_vad_gap_grid_csv(
        [0.5], [50.0], [_three()], name="rec.wav", col_axis="min_speech_ms"
    )
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0][0] == "threshold"
    assert rows[0][1] == "min_speech_ms"


def test_render_csv_unavailable():
    text = gv.render_vad_gap_grid_csv([], [], [None], name="rec.wav")
    assert text.startswith("# silero VAD unavailable")


# ---- handler: cmd_vad_gap_grid ------------------------------------------


def _run_handler(results, argv_extra=None, segmenter=None):
    """Drive cmd_vad_gap_grid with an injected segmenter returning ``results``
    in row-major order (one per grid cell)."""
    lines: List[str] = []
    argv = ["vad-gap-grid", "rec.wav", *(argv_extra or [])]
    args = gv.build_parser().parse_args(argv)
    it = iter(results)
    if segmenter is None:
        segmenter = lambda wav, params=None: next(it)  # noqa: E731
    gv.cmd_vad_gap_grid(
        args, log=lines.append, segmenter=segmenter, availability=lambda: True
    )
    return lines


def test_cmd_human_path():
    # 2 thresholds × 2 hangovers = 4 cells.
    lines = _run_handler(
        [_three(), _three(), _three(), _single()],
        argv_extra=["--thresholds", "0.3,0.9", "--min-silences", "400,800"],
    )
    text = "\n".join(lines)
    assert "silero VAD gap grid" in text
    assert "min_gap" in text


def test_cmd_json_path():
    lines = _run_handler(
        [_three(), _single()],
        argv_extra=["--thresholds", "0.3,0.9", "--min-silences", "400", "--json"],
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is True
    assert len(payload["grid"]) == 2
    assert payload["grid"][0]["num_gaps"] == 2


def test_cmd_csv_path():
    lines = _run_handler(
        [_three(), _single()],
        argv_extra=["--thresholds", "0.3,0.9", "--min-silences", "400", "--csv"],
    )
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    assert rows[0][0] == "threshold"
    assert rows[0][1] == "min_silence_ms"
    assert rows[0][3] == "num_gaps"


def test_cmd_min_speeches_axis_switches_column():
    seen = []

    def _seg(wav, params=None):
        seen.append((params.threshold, params.min_speech_ms))
        return _three()

    lines = _run_handler(
        [None],
        argv_extra=["--thresholds", "0.5", "--min-speeches", "50,200", "--json"],
        segmenter=_seg,
    )
    # The column axis is the floor (50 then 200); the gate is the row (0.5).
    assert seen == [(0.5, 50.0), (0.5, 200.0)]
    payload = json.loads("\n".join(lines))
    assert payload["col_axis"] == "min_speech_ms"
    assert [c["min_speech_ms"] for c in payload["grid"]] == [50.0, 200.0]


def test_cmd_holds_non_column_knobs_fixed():
    captured = []

    def _seg(wav, params=None):
        captured.append(
            (params.threshold, params.min_silence_ms, params.min_speech_ms)
        )
        return _three()

    _run_handler(
        [None],
        argv_extra=["--thresholds", "0.3,0.9", "--min-silences", "400",
                    "--min-speech-ms", "75"],
        segmenter=_seg,
    )
    # Gate × hangover grid; the min-speech floor is held at its scalar (75).
    assert captured == [(0.3, 400.0, 75.0), (0.9, 400.0, 75.0)]


def test_cmd_max_speeches_seconds_column():
    seen = []

    def _seg(wav, params=None):
        seen.append(params.max_speech_s)
        return _three()

    lines = _run_handler(
        [None],
        argv_extra=["--thresholds", "0.5", "--max-speeches", "5,inf", "--json"],
        segmenter=_seg,
    )
    assert seen[0] == 5.0
    assert seen[1] == float("inf")
    payload = json.loads("\n".join(lines))
    assert payload["col_axis"] == "max_speech_s"


def test_cmd_row_major_segmentation_order():
    """The handler segments cells in row-major order (each row's whole row of
    columns first), matching how vad_gap_grid flattens."""
    seen = []

    def _seg(wav, params=None):
        seen.append((params.threshold, params.min_silence_ms))
        return _three()

    _run_handler(
        [None],
        argv_extra=["--thresholds", "0.3,0.9", "--min-silences", "400,800"],
        segmenter=_seg,
    )
    assert seen == [(0.3, 400.0), (0.3, 800.0), (0.9, 400.0), (0.9, 800.0)]


def test_cmd_unavailable_human():
    lines: List[str] = []
    args = gv.build_parser().parse_args(["vad-gap-grid", "rec.wav"])
    gv.cmd_vad_gap_grid(
        args,
        log=lines.append,
        segmenter=lambda wav, params=None: pytest.fail("must not segment"),
        availability=lambda: False,
    )
    assert any("silero VAD unavailable" in ln for ln in lines)


def test_cmd_unavailable_json():
    lines: List[str] = []
    args = gv.build_parser().parse_args(["vad-gap-grid", "rec.wav", "--json"])
    gv.cmd_vad_gap_grid(
        args,
        log=lines.append,
        segmenter=lambda wav, params=None: pytest.fail("must not segment"),
        availability=lambda: False,
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is False


def test_cmd_unavailable_csv():
    lines: List[str] = []
    args = gv.build_parser().parse_args(["vad-gap-grid", "rec.wav", "--csv"])
    gv.cmd_vad_gap_grid(
        args,
        log=lines.append,
        segmenter=lambda wav, params=None: pytest.fail("must not segment"),
        availability=lambda: False,
    )
    assert any("silero VAD unavailable" in ln for ln in lines)


def test_cmd_uses_result_name_not_raw_path():
    """The report names the segmenter's basename (matching `gv vad`), not the
    raw CLI path argument."""
    lines = _run_handler(
        [_result((0.0, 1.0), (2.0, 3.0), name="clean.wav")],
        argv_extra=["--thresholds", "0.5", "--min-silences", "400", "--json"],
    )
    payload = json.loads("\n".join(lines))
    assert payload["name"] == "clean.wav"
