"""Tests for iter-373 — the ``gv vad-gap-recommend-knob-grid`` subcommand (examples/gv.py).

iter-372's ``gv vad-gap-recommend-knob-sweep`` sweeps ONE segmenter knob and
reports, at each value, the whole short/balanced/long recommended
``--min-silence-ms`` spread plus the iter-348 confidence grade. This surface is its
2-D analogue: it tabulates the same spread + grade across the cartesian product of
TWO knobs (the gate × a column knob), so an operator reads how the recommendation —
and, more tellingly, the confidence GRADE — moves in two dimensions at once instead
of running N separate 1-D knob sweeps. It is to ``vad-gap-recommend-knob-sweep``
what ``vad-gap-grid`` is to ``vad-gap-sweep``.

Each cell is anchored to :func:`vad_gap_recommend_sweep` over that cell's
segmentation, so the per-cell bias spread, valley accounting, and grade agree
EXACTLY with ``gv vad-gap-recommend-knob-sweep`` (and ``gv vad-gap-confidence``) at
the matching cell.

Like the rest of the VAD-analysis family, the handler takes injected ``segmenter``
/ ``availability`` / ``log`` dependencies so every test runs WITHOUT importing
torch / silero-vad and without touching real audio.
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
    sample_rate: int
    duration_s: float
    segments: List[_Seg] = field(default_factory=list)

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    @property
    def speech_s(self) -> float:
        return sum(s.duration_s for s in self.segments)


def _result(*pairs, name="rec.wav", sample_rate=16000, duration_s=30.0):
    return _Result(
        name=name,
        sample_rate=sample_rate,
        duration_s=duration_s,
        segments=[_Seg(a, b) for a, b in pairs],
    )


# A clean bimodal recording: short within-turn pauses (~0.3s) and one long
# between-turn pause (~2.0s) → a dominant valley, grades strong.
def _bimodal(name="rec.wav"):
    return _result((0, 1), (1.3, 2), (2.3, 3), (3.3, 4), (6, 7), name=name)


# Three segments, both gaps the same 1.0s → no valley (grades "none").
def _no_valley(name="rec.wav"):
    return _result((0, 1), (2, 3), (4, 5), name=name)


# A single segment → no gaps, nothing to recommend over.
def _single(name="rec.wav"):
    return _result((0, 1), name=name)


# ---- core: vad_gap_recommend_knob_grid -----------------------------------


def test_core_row_major_cells_keyed_by_both_axes():
    # 2 rows × 2 cols, row-major. Cells carry both axis keys.
    res = [_bimodal(), _bimodal(), _bimodal(), _bimodal()]
    cells = gv.vad_gap_recommend_knob_grid([0.3, 0.7], [400.0, 800.0], res)
    assert len(cells) == 4
    assert [(c["threshold"], c["min_silence_ms"]) for c in cells] == [
        (0.3, 400.0),
        (0.3, 800.0),
        (0.7, 400.0),
        (0.7, 800.0),
    ]


def test_core_each_cell_matches_recommend_sweep_exactly():
    res = _bimodal()
    cells = gv.vad_gap_recommend_knob_grid([0.5], [600.0], [res])
    s = gv.vad_gap_recommend_sweep(res)
    cell = cells[0]
    assert cell["biases"] == s["biases"]
    assert cell["spread_ms"] == s["spread_ms"]
    assert cell["spread_s"] == s["spread_s"]
    assert cell["split_found"] == s["split_found"]
    assert cell["num_segments"] == s["num_segments"]
    assert cell["num_gaps"] == s["num_gaps"]


def test_core_grade_matches_confidence_surface():
    # The genuinely-new signal: the confidence grade per cell agrees exactly with
    # the single confidence surface over the same segmentation.
    res = _bimodal()
    cells = gv.vad_gap_recommend_knob_grid([0.5], [600.0], [res])
    c = gv.vad_gap_confidence(res)
    assert cells[0]["grade"] == c["grade"] == "strong"
    assert cells[0]["dominance"] == c["dominance"]
    assert cells[0]["separation_ratio"] == c["separation_ratio"]


def test_core_carries_all_three_biases_in_order():
    cells = gv.vad_gap_recommend_knob_grid([0.5], [600.0], [_bimodal()])
    assert [b["bias"] for b in cells[0]["biases"]] == ["short", "balanced", "long"]


def test_core_no_valley_cell_grades_none():
    cells = gv.vad_gap_recommend_knob_grid([0.5], [600.0], [_no_valley()])
    cell = cells[0]
    assert cell["split_found"] is False
    assert cell["grade"] == "none"
    assert cell["dominance"] is None
    assert cell["separation_ratio"] is None


def test_core_no_gaps_cell_has_none_recommendations():
    cells = gv.vad_gap_recommend_knob_grid([0.5], [600.0], [_single()])
    cell = cells[0]
    assert cell["num_gaps"] == 0
    assert cell["spread_ms"] is None
    assert cell["spread_s"] is None
    for b in cell["biases"]:
        assert b["recommended_ms"] is None
        assert b["recommended_s"] is None


def test_core_grade_can_shift_across_cells():
    # The whole point of the grid: different segmentations at different cells can
    # yield different grades. A bimodal cell grades strong; a uniform cell grades
    # none — they sit in the same grid.
    cells = gv.vad_gap_recommend_knob_grid(
        [0.3], [400.0, 800.0], [_bimodal(), _no_valley()]
    )
    assert cells[0]["grade"] == "strong"
    assert cells[1]["grade"] == "none"


def test_core_axis_keys_follow_axis_arguments():
    cells = gv.vad_gap_recommend_knob_grid(
        [200.0], [40.0], [_bimodal()],
        row_axis="min_silence_ms", col_axis="speech_pad_ms",
    )
    assert "min_silence_ms" in cells[0]
    assert "speech_pad_ms" in cells[0]
    assert "threshold" not in cells[0]


def test_core_length_mismatch_raises():
    with pytest.raises(ValueError):
        # 2 × 2 = 4 expected, only 3 given.
        gv.vad_gap_recommend_knob_grid(
            [0.3, 0.7], [400.0, 800.0], [_bimodal(), _bimodal(), _bimodal()]
        )


# ---- human renderer ------------------------------------------------------


def test_render_human_header_and_columns():
    lines = gv.render_vad_gap_recommend_knob_grid(
        [0.3, 0.7], [400.0, 800.0],
        [_bimodal(), _bimodal(), _bimodal(), _bimodal()],
        name="rec.wav",
    )
    text = "\n".join(lines)
    assert "recommended-hangover knob grid" in text
    assert "threshold × min_silence" in text
    assert "short" in text
    assert "balanced" in text
    assert "confidence" in text


def test_render_human_shows_grade_per_cell():
    lines = gv.render_vad_gap_recommend_knob_grid(
        [0.3], [400.0, 800.0], [_bimodal(), _no_valley()], name="rec.wav"
    )
    text = "\n".join(lines)
    assert "strong" in text
    assert "none" in text


def test_render_human_no_gaps_cell_dashes():
    lines = gv.render_vad_gap_recommend_knob_grid(
        [0.5], [600.0], [_single()], name="rec.wav"
    )
    # The recommendation columns collapse to dashes for a <2-segment cell.
    assert any("-" in ln for ln in lines[2:])


def test_render_human_two_axis_labels_in_header():
    lines = gv.render_vad_gap_recommend_knob_grid(
        [0.5], [40.0], [_bimodal()],
        name="rec.wav", col_axis="speech_pad_ms",
    )
    text = "\n".join(lines)
    assert "threshold × speech_pad" in text


def test_render_human_unavailable_hint():
    lines = gv.render_vad_gap_recommend_knob_grid(
        [], [], [None], name="rec.wav"
    )
    assert any("silero VAD unavailable" in ln for ln in lines)


# ---- JSON renderer -------------------------------------------------------


def test_render_json_shape():
    text = gv.render_vad_gap_recommend_knob_grid_json(
        [0.3], [400.0, 800.0], [_bimodal(), _no_valley()],
        name="rec.wav", row_axis="threshold", col_axis="min_silence_ms",
    )
    payload = json.loads(text)
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["row_axis"] == "threshold"
    assert payload["col_axis"] == "min_silence_ms"
    assert len(payload["grid"]) == 2
    assert payload["grid"][0]["grade"] == "strong"
    assert payload["grid"][1]["grade"] == "none"
    # The per-bias spread is carried per cell.
    assert [b["bias"] for b in payload["grid"][0]["biases"]] == [
        "short",
        "balanced",
        "long",
    ]


def test_render_json_no_gaps_nulls():
    text = gv.render_vad_gap_recommend_knob_grid_json(
        [0.5], [600.0], [_single()], name="rec.wav"
    )
    cell = json.loads(text)["grid"][0]
    assert cell["spread_ms"] is None
    assert cell["dominance"] is None


def test_render_json_unavailable():
    text = gv.render_vad_gap_recommend_knob_grid_json(
        [], [], [None], name="rec.wav"
    )
    payload = json.loads(text)
    assert payload["available"] is False
    assert "hint" in payload


# ---- CSV renderer --------------------------------------------------------


def test_render_csv_header_and_one_row_per_cell():
    text = gv.render_vad_gap_recommend_knob_grid_csv(
        [0.3], [400.0, 800.0], [_bimodal(), _no_valley()],
        name="rec.wav", row_axis="threshold", col_axis="min_silence_ms",
    )
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == [
        "threshold",
        "min_silence_ms",
        "num_segments",
        "num_gaps",
        "short_ms",
        "balanced_ms",
        "long_ms",
        "spread_ms",
        "grade",
        "dominance",
        "separation_ratio",
    ]
    # One row per cell.
    assert len(rows) == 3
    assert rows[1][0] == "0.3"
    assert rows[1][1] == "400.0"
    assert rows[1][8] == "strong"
    assert rows[2][8] == "none"


def test_render_csv_axis_headers_follow_axes():
    text = gv.render_vad_gap_recommend_knob_grid_csv(
        [200.0], [40.0], [_bimodal()],
        name="rec.wav", row_axis="min_silence_ms", col_axis="speech_pad_ms",
    )
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0][0] == "min_silence_ms"
    assert rows[0][1] == "speech_pad_ms"


def test_render_csv_no_gaps_empty_cells():
    text = gv.render_vad_gap_recommend_knob_grid_csv(
        [0.5], [600.0], [_single()], name="rec.wav"
    )
    rows = list(csv.reader(io.StringIO(text)))
    # short_ms..separation_ratio all empty for a <2-segment cell.
    assert rows[1][4:] == ["", "", "", "", "", "", ""]


def test_render_csv_unavailable():
    text = gv.render_vad_gap_recommend_knob_grid_csv(
        [], [], [None], name="rec.wav"
    )
    assert text.startswith("# silero VAD unavailable")


# ---- handler: cmd_vad_gap_recommend_knob_grid ----------------------------


def _run_handler(results, argv_extra=None, segmenter=None):
    """Drive cmd_vad_gap_recommend_knob_grid with an injected segmenter returning
    ``results`` in row-major order (one per grid cell)."""
    lines: List[str] = []
    argv = ["vad-gap-recommend-knob-grid", "rec.wav", *(argv_extra or [])]
    args = gv.build_parser().parse_args(argv)
    it = iter(results)
    if segmenter is None:
        segmenter = lambda wav, params=None: next(it)  # noqa: E731
    gv.cmd_vad_gap_recommend_knob_grid(
        args, log=lines.append, segmenter=segmenter, availability=lambda: True
    )
    return lines


def test_cmd_human_path():
    # 4 default thresholds × 4 default min-silences = 16 cells.
    lines = _run_handler([_bimodal()] * 16)
    text = "\n".join(lines)
    assert "recommended-hangover knob grid" in text
    assert "confidence" in text


def test_cmd_json_path():
    lines = _run_handler(
        [_bimodal(), _no_valley()],
        argv_extra=["--thresholds", "0.3", "--min-silences", "400,800", "--json"],
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is True
    assert len(payload["grid"]) == 2
    assert payload["grid"][0]["grade"] == "strong"
    assert payload["grid"][1]["grade"] == "none"


def test_cmd_csv_path():
    lines = _run_handler(
        [_bimodal(), _no_valley()],
        argv_extra=["--thresholds", "0.3", "--min-silences", "400,800", "--csv"],
    )
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    assert rows[0][0] == "threshold"
    assert rows[0][1] == "min_silence_ms"
    assert rows[0][8] == "grade"


def test_cmd_default_grid_is_gate_by_hangover():
    seen = []

    def _seg(wav, params=None):
        seen.append((params.threshold, params.min_silence_ms))
        return _bimodal()

    lines = _run_handler(
        [None] * 4,
        argv_extra=["--thresholds", "0.3,0.7", "--min-silences", "400,800", "--json"],
        segmenter=_seg,
    )
    # Row-major: row 0 (gate 0.3) × both columns, then row 1 (gate 0.7).
    assert seen == [
        (0.3, 400.0),
        (0.3, 800.0),
        (0.7, 400.0),
        (0.7, 800.0),
    ]
    payload = json.loads("\n".join(lines))
    assert payload["col_axis"] == "min_silence_ms"


def test_cmd_min_speeches_column_axis_switches_dimension():
    seen = []

    def _seg(wav, params=None):
        seen.append(params.min_speech_ms)
        return _bimodal()

    lines = _run_handler(
        [None] * 2,
        argv_extra=["--thresholds", "0.5", "--min-speeches", "50,200", "--json"],
        segmenter=_seg,
    )
    assert seen == [50.0, 200.0]
    payload = json.loads("\n".join(lines))
    assert payload["col_axis"] == "min_speech_ms"
    assert [c["min_speech_ms"] for c in payload["grid"]] == [50.0, 200.0]


def test_cmd_speech_pads_column_axis_holds_other_knobs():
    seen = []

    def _seg(wav, params=None):
        seen.append((params.speech_pad_ms, params.min_silence_ms))
        return _bimodal()

    _run_handler(
        [None] * 2,
        argv_extra=["--thresholds", "0.5", "--speech-pads", "20,40",
                    "--min-silence-ms", "700"],
        segmenter=_seg,
    )
    assert seen == [(20.0, 700.0), (40.0, 700.0)]


def test_cmd_max_speeches_seconds_column_axis():
    seen = []

    def _seg(wav, params=None):
        seen.append(params.max_speech_s)
        return _bimodal()

    lines = _run_handler(
        [None] * 2,
        argv_extra=["--thresholds", "0.5", "--max-speeches", "5,inf", "--json"],
        segmenter=_seg,
    )
    assert seen[0] == 5.0
    assert seen[1] == float("inf")
    payload = json.loads("\n".join(lines))
    assert payload["col_axis"] == "max_speech_s"


def test_cmd_uses_result_name_not_raw_path():
    lines = _run_handler(
        [_bimodal(name="clean.wav")],
        argv_extra=["--thresholds", "0.5", "--min-silences", "600", "--json"],
    )
    payload = json.loads("\n".join(lines))
    assert payload["name"] == "clean.wav"


def test_cmd_unavailable_human():
    lines: List[str] = []
    args = gv.build_parser().parse_args(["vad-gap-recommend-knob-grid", "rec.wav"])
    gv.cmd_vad_gap_recommend_knob_grid(
        args,
        log=lines.append,
        segmenter=lambda wav, params=None: pytest.fail("must not segment"),
        availability=lambda: False,
    )
    assert any("silero VAD unavailable" in ln for ln in lines)


def test_cmd_unavailable_json():
    lines: List[str] = []
    args = gv.build_parser().parse_args(
        ["vad-gap-recommend-knob-grid", "rec.wav", "--json"]
    )
    gv.cmd_vad_gap_recommend_knob_grid(
        args,
        log=lines.append,
        segmenter=lambda wav, params=None: pytest.fail("must not segment"),
        availability=lambda: False,
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is False


def test_cmd_unavailable_csv():
    lines: List[str] = []
    args = gv.build_parser().parse_args(
        ["vad-gap-recommend-knob-grid", "rec.wav", "--csv"]
    )
    gv.cmd_vad_gap_recommend_knob_grid(
        args,
        log=lines.append,
        segmenter=lambda wav, params=None: pytest.fail("must not segment"),
        availability=lambda: False,
    )
    assert any("silero VAD unavailable" in ln for ln in lines)


def test_cmd_json_and_csv_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-gap-recommend-knob-grid", "rec.wav", "--json", "--csv"]
        )


def test_cmd_column_axes_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            [
                "vad-gap-recommend-knob-grid",
                "rec.wav",
                "--min-silences",
                "400,800",
                "--min-speeches",
                "50,100",
            ]
        )


def test_cmd_registered_in_dispatch_table():
    assert gv.DEFAULT_HANDLERS["vad-gap-recommend-knob-grid"] is (
        gv.cmd_vad_gap_recommend_knob_grid
    )
