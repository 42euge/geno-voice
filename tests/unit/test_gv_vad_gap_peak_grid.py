"""Tests for iter-365 — the ``gv vad-gap-peak-grid`` subcommand (examples/gv.py).

iter-350 shipped ``gv vad-gap-peak`` — the COSTLIEST cost-curve band (the
densest pause cluster, the most expensive place to raise the end-of-turn
hangover) at ONE knob setting; iter-364 shipped ``gv vad-gap-peak-sweep``, the
peak-side sweep tabulating that band across ONE swept knob; iter-332 shipped
``gv vad-gap-grid``, the gap-side 2-D grid. This lap adds the peak-side GRID:
``gv vad-gap-peak-grid`` is to ``gv vad-gap-peak-sweep`` what ``gv vad-gap-grid``
is to ``gv vad-gap-sweep`` — for each (row, col) cell of a gate × column-knob
grid it names the steepest band so an operator can watch how the cost of pushing
the hangover through the densest pause cluster MOVES across two knobs at once.

Like the rest of the VAD-analysis family, the handler takes injected
``segmenter`` / ``availability`` / ``log`` dependencies so every test runs
WITHOUT importing torch / silero-vad and without touching real audio — fast and
deterministic on the x86_64 Linux runner. The pure core (``vad_gap_peak_grid``)
and the three renderers are exercised directly against lightweight stand-ins
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


# Recurring stand-ins (same shapes the peak-sweep tests use):
#   _three: 3 segments / 2 gaps (1.0s, 2.0s). The 2.0s pause falls in the
#           800-1600ms band, so the cost peak is 800-1600ms, +1 merged, 0.125.
#   _single: one segment, no inter-segment pause (no peak).
#   _all_valley: 2 segments / one 9.0s gap beyond every cut, so every band is an
#           empty valley (bands exist but peak_found is False).
def _three():
    return _result((0.0, 1.0), (2.0, 3.0), (5.0, 6.0))


def _single():
    return _result((0.0, 6.0))


def _all_valley():
    return _result((0.0, 1.0), (10.0, 11.0))


def _multi_band():
    """4 segments / 3 gaps (0.3s, 0.5s, 1.0s) landing in three distinct cost
    bands, so band_rate_dist has count 3 — a meaningful spread to summarise
    (same stand-in the peak-sweep tests use for the iter-366 distribution)."""
    return _result((0.0, 1.0), (1.3, 2.0), (2.5, 3.0), (4.0, 5.0))


# ---- parser: registration & defaults -----------------------------------


def test_vad_gap_peak_grid_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-gap-peak-grid"] is gv.cmd_vad_gap_peak_grid


def test_parser_default_axes():
    args = gv.build_parser().parse_args(["vad-gap-peak-grid", "rec.wav"])
    assert args.command == "vad-gap-peak-grid"
    assert args.thresholds == [0.3, 0.5, 0.7, 0.9]
    # Default column axis lives on --min-silences (the hangover).
    assert args.min_silences == [400.0, 600.0, 800.0, 1000.0]
    assert args.min_speeches is None
    assert args.speech_pads is None
    assert args.max_speeches is None


def test_parser_cuts_ms_default():
    args = gv.build_parser().parse_args(["vad-gap-peak-grid", "rec.wav"])
    assert args.cuts_ms == list(gv.DEFAULT_GAP_CDF_CUTS_MS)


def test_parser_cuts_ms_custom_parsed():
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--cuts-ms", "200,400,800"]
    )
    assert args.cuts_ms == [200.0, 400.0, 800.0]


def test_parser_defaults_mirror_silero_params():
    """The held-fixed scalar knobs default to the same values as ``gv vad``."""
    args = gv.build_parser().parse_args(["vad-gap-peak-grid", "rec.wav"])
    vad = gv.build_parser().parse_args(["vad", "rec.wav"])
    assert args.min_speech_ms == vad.min_speech_ms
    assert args.min_silence_ms == vad.min_silence_ms
    assert args.speech_pad_ms == vad.speech_pad_ms
    assert args.max_speech_s == vad.max_speech_s


def test_parser_column_axes_are_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["vad-gap-peak-grid", "rec.wav", "--min-silences", "200,400",
             "--min-speeches", "50,100"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["vad-gap-peak-grid", "rec.wav", "--speech-pads", "0,20",
             "--max-speeches", "5,10"]
        )


def test_parser_json_csv_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-peak-grid", "rec.wav", "--json", "--csv"])


def test_parser_has_no_target_pick_args():
    """Like the gap grid, the peak grid has no --target / --top / --tie-break."""
    args = gv.build_parser().parse_args(["vad-gap-peak-grid", "rec.wav"])
    assert not hasattr(args, "target")
    assert not hasattr(args, "top")
    assert not hasattr(args, "tie_break")


# ---- pure core: vad_gap_peak_grid ---------------------------------------


def test_core_row_major_order_and_keys():
    rows = [0.3, 0.7]
    cols = [400.0, 800.0]
    results = [_three(), _single(), _all_valley(), _three()]
    cells = gv.vad_gap_peak_grid(rows, cols, results)
    assert len(cells) == 4
    # Row-major: (0.3,400), (0.3,800), (0.7,400), (0.7,800).
    assert [(c["threshold"], c["min_silence_ms"]) for c in cells] == [
        (0.3, 400.0), (0.3, 800.0), (0.7, 400.0), (0.7, 800.0)
    ]
    # First cell (_three): names the 800-1600 band.
    assert cells[0]["peak_found"] is True
    assert cells[0]["peak_from_ms"] == 800.0
    assert cells[0]["peak_to_ms"] == 1600.0
    assert cells[0]["peak_merged_added"] == 1
    assert cells[0]["peak_rate_per_100ms"] == 0.125
    # Second cell (_single): no gap, no peak.
    assert cells[1]["peak_found"] is False
    assert cells[1]["peak_from_ms"] is None
    assert cells[1]["peak_rate_per_100ms"] is None
    # Third cell (_all_valley): a gap exists but no cost peak.
    assert cells[2]["num_gaps"] == 1
    assert cells[2]["peak_found"] is False
    assert cells[2]["peak_from_ms"] is None


def test_core_peak_fields_match_vad_gap_peak():
    """Each cell's peak fields equal an independent vad_gap_peak (top_n=1) on its
    result — the grid names the SAME steepest band the verdict surface does."""
    r = _three()
    direct = gv.vad_gap_peak(r)
    cell = gv.vad_gap_peak_grid([0.5], [800.0], [r])[0]
    for key in ("num_segments", "num_gaps", "peak_found", "peak_from_ms",
                "peak_to_ms", "peak_width_ms", "peak_merged_added",
                "peak_rate_per_100ms"):
        assert cell[key] == direct[key]


def test_core_axis_keys_follow_axis_args():
    cells = gv.vad_gap_peak_grid(
        [0.5], [50.0], [_three()], col_axis="min_speech_ms"
    )
    assert "min_speech_ms" in cells[0]
    assert "min_silence_ms" not in cells[0]
    assert cells[0]["threshold"] == 0.5
    assert cells[0]["min_speech_ms"] == 50.0


def test_core_custom_cuts_ms_changes_band():
    """A coarser cut axis lands the 2.0s pause in a different band."""
    cells = gv.vad_gap_peak_grid(
        [0.5], [800.0], [_three()], cuts_ms=[1000.0, 3000.0]
    )
    assert cells[0]["peak_found"] is True
    assert cells[0]["peak_from_ms"] == 1000.0
    assert cells[0]["peak_to_ms"] == 3000.0


def test_core_length_mismatch_raises():
    with pytest.raises(ValueError):
        gv.vad_gap_peak_grid([0.3, 0.5], [400.0, 800.0], [_three()])


def test_core_empty_grid_is_empty():
    assert gv.vad_gap_peak_grid([], [], []) == []


# ---- renderer: render_vad_gap_peak_grid (human) -------------------------


def test_render_human_header_and_rows():
    lines = gv.render_vad_gap_peak_grid(
        [0.3], [400.0, 800.0], [_three(), _single()], name="rec.wav"
    )
    assert lines[0] == "silero VAD gap cost-peak grid — rec.wav (threshold × min_silence)"
    assert "threshold" in lines[1]
    assert "min_silence" in lines[1]
    assert "peak_band_ms" in lines[1]
    assert "rate/100ms" in lines[1]
    # First cell names the 800-1600 band; second cell (no peak) shows dashes.
    assert "800-1600" in lines[2]
    assert "0.125" in lines[2]
    assert "-" in lines[3]
    assert "800-1600" not in lines[3]


def test_render_human_column_axis_label_follows_axis():
    lines = gv.render_vad_gap_peak_grid(
        [0.5], [50.0], [_three()], name="rec.wav", col_axis="min_speech_ms"
    )
    assert "min_speech" in lines[0]
    assert "min_speech" in lines[1]


def test_render_human_unavailable_hint():
    lines = gv.render_vad_gap_peak_grid([], [], [None], name="rec.wav")
    assert len(lines) == 1
    assert "silero VAD unavailable" in lines[0]


# ---- renderer: render_vad_gap_peak_grid_json ----------------------------


def test_render_json_shape():
    out = gv.render_vad_gap_peak_grid_json(
        [0.3], [400.0, 800.0], [_three(), _single()], name="rec.wav"
    )
    payload = json.loads(out)
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["row_axis"] == "threshold"
    assert payload["col_axis"] == "min_silence_ms"
    assert payload["cuts_ms"] == list(gv.DEFAULT_GAP_CDF_CUTS_MS)
    assert len(payload["grid"]) == 2
    first = payload["grid"][0]
    assert first["peak_found"] is True
    assert first["peak_from_ms"] == 800.0
    assert first["peak_rate_per_100ms"] == 0.125
    # No-peak cell carries JSON null for the peak measures.
    second = payload["grid"][1]
    assert second["peak_found"] is False
    assert second["peak_from_ms"] is None
    assert second["peak_rate_per_100ms"] is None


def test_render_json_carries_custom_cuts_ms():
    out = gv.render_vad_gap_peak_grid_json(
        [0.5], [800.0], [_three()], name="rec.wav", cuts_ms=[1000.0, 3000.0]
    )
    payload = json.loads(out)
    assert payload["cuts_ms"] == [1000.0, 3000.0]
    assert payload["grid"][0]["peak_from_ms"] == 1000.0


def test_render_json_axis_keys_follow_axes():
    out = gv.render_vad_gap_peak_grid_json(
        [0.5], [50.0], [_three()], name="rec.wav", col_axis="min_speech_ms"
    )
    payload = json.loads(out)
    assert payload["col_axis"] == "min_speech_ms"
    assert "min_speech_ms" in payload["grid"][0]


def test_render_json_unavailable():
    out = gv.render_vad_gap_peak_grid_json([], [], [None], name="rec.wav")
    payload = json.loads(out)
    assert payload["available"] is False
    assert "install 'silero-vad'" in payload["hint"]


# ---- renderer: render_vad_gap_peak_grid_csv -----------------------------


def test_render_csv_header_and_rows():
    out = gv.render_vad_gap_peak_grid_csv(
        [0.3], [400.0, 800.0], [_three(), _single()], name="rec.wav"
    )
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0] == [
        "threshold", "min_silence_ms", "num_segments", "num_gaps", "peak_found",
        "peak_from_ms", "peak_to_ms", "peak_width_ms", "peak_merged_added",
        "peak_rate_per_100ms",
    ]
    # Peak cell.
    assert rows[1] == [
        "0.3", "400.0", "3", "2", "True", "800", "1600", "800", "1", "0.125"
    ]
    # No-peak cell: peak_found False, blank peak-measure cells.
    assert rows[2] == ["0.3", "800.0", "1", "0", "False", "", "", "", "", ""]


def test_render_csv_axis_headers_follow_axes():
    out = gv.render_vad_gap_peak_grid_csv(
        [0.5], [50.0], [_three()], name="rec.wav", col_axis="min_speech_ms"
    )
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0][0] == "threshold"
    assert rows[0][1] == "min_speech_ms"
    assert rows[1][0] == "0.5"
    assert rows[1][1] == "50.0"


def test_render_csv_custom_cuts_ms():
    out = gv.render_vad_gap_peak_grid_csv(
        [0.5], [800.0], [_three()], name="rec.wav", cuts_ms=[1000.0, 3000.0]
    )
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[1][5] == "1000"  # peak_from_ms
    assert rows[1][6] == "3000"  # peak_to_ms


def test_render_csv_unavailable_comment():
    out = gv.render_vad_gap_peak_grid_csv([], [], [None], name="rec.wav")
    assert out.startswith("# silero VAD unavailable")
    assert "\n" not in out


def test_render_csv_no_trailing_newline():
    out = gv.render_vad_gap_peak_grid_csv([0.5], [800.0], [_three()], name="rec.wav")
    assert not out.endswith("\n")


# ---- handler: cmd_vad_gap_peak_grid -------------------------------------


def _avail_true():
    return True


def _avail_false():
    return False


def _make_segmenter(result):
    """A segmenter stub ignoring params and always returning ``result``."""
    def _seg(wav, params=None):
        return result

    return _seg


def _stub_silero(monkeypatch):
    import types
    fake = types.ModuleType("vad.silero")
    fake.SileroParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vad.silero", fake)


def test_handler_default_grid_human(monkeypatch):
    _stub_silero(monkeypatch)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--thresholds", "0.3,0.7",
         "--min-silences", "400,800"]
    )
    lines = []
    gv.cmd_vad_gap_peak_grid(args, log=lines.append,
                             segmenter=_make_segmenter(_three()),
                             availability=_avail_true)
    assert lines[0].startswith("silero VAD gap cost-peak grid — rec.wav")
    # 2×2 grid → 4 data rows, each naming the 800-1600 band.
    assert len([ln for ln in lines if "800-1600" in ln]) == 4


def test_handler_min_speeches_column_axis(monkeypatch):
    _stub_silero(monkeypatch)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--thresholds", "0.5",
         "--min-speeches", "50,100,200", "--json"]
    )
    out = []
    gv.cmd_vad_gap_peak_grid(args, log=out.append,
                             segmenter=_make_segmenter(_three()),
                             availability=_avail_true)
    payload = json.loads(out[0])
    assert payload["col_axis"] == "min_speech_ms"
    # 1×3 grid.
    assert len(payload["grid"]) == 3
    assert all("min_speech_ms" in cell for cell in payload["grid"])


def test_handler_threads_cuts_ms(monkeypatch):
    _stub_silero(monkeypatch)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--thresholds", "0.5",
         "--min-silences", "800", "--cuts-ms", "1000,3000", "--json"]
    )
    out = []
    gv.cmd_vad_gap_peak_grid(args, log=out.append,
                             segmenter=_make_segmenter(_three()),
                             availability=_avail_true)
    payload = json.loads(out[0])
    assert payload["cuts_ms"] == [1000.0, 3000.0]
    assert payload["grid"][0]["peak_from_ms"] == 1000.0


def test_handler_csv(monkeypatch):
    _stub_silero(monkeypatch)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--thresholds", "0.5",
         "--min-silences", "800", "--csv"]
    )
    out = []
    gv.cmd_vad_gap_peak_grid(args, log=out.append,
                             segmenter=_make_segmenter(_three()),
                             availability=_avail_true)
    rows = list(csv.reader(io.StringIO(out[0])))
    assert rows[0][0] == "threshold"
    assert rows[0][1] == "min_silence_ms"
    assert rows[1] == [
        "0.5", "800.0", "3", "2", "True", "800", "1600", "800", "1", "0.125"
    ]


def test_handler_max_speeches_column_axis(monkeypatch):
    _stub_silero(monkeypatch)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--thresholds", "0.5",
         "--max-speeches", "5,inf", "--json"]
    )
    out = []
    gv.cmd_vad_gap_peak_grid(args, log=out.append,
                             segmenter=_make_segmenter(_three()),
                             availability=_avail_true)
    payload = json.loads(out[0])
    assert payload["col_axis"] == "max_speech_s"
    assert len(payload["grid"]) == 2


def test_handler_unavailable_human():
    args = gv.build_parser().parse_args(["vad-gap-peak-grid", "rec.wav"])
    lines = []
    gv.cmd_vad_gap_peak_grid(args, log=lines.append,
                             segmenter=_make_segmenter(_three()),
                             availability=_avail_false)
    assert len(lines) == 1
    assert "silero VAD unavailable" in lines[0]


def test_handler_unavailable_json():
    args = gv.build_parser().parse_args(["vad-gap-peak-grid", "rec.wav",
                                         "--json"])
    out = []
    gv.cmd_vad_gap_peak_grid(args, log=out.append,
                             segmenter=_make_segmenter(_three()),
                             availability=_avail_false)
    payload = json.loads(out[0])
    assert payload["available"] is False


def test_handler_unavailable_csv():
    args = gv.build_parser().parse_args(["vad-gap-peak-grid", "rec.wav",
                                         "--csv"])
    out = []
    gv.cmd_vad_gap_peak_grid(args, log=out.append,
                             segmenter=_make_segmenter(_three()),
                             availability=_avail_false)
    assert out[0].startswith("# silero VAD unavailable")


# ---- iter-367: per-cell band_rate_dist on the cost-peak grid ------------
#
# iter-358 added the observed non-empty band-rate distribution (the
# --min-rate-pct sample) to the SINGLE-shot `gv vad-gap-peak`; iter-366 carried
# it into the 1-D SWEEP. iter-367 completes the trio by carrying that same view
# into the 2-D GRID, one distribution per (row, col) cell, so an operator
# watches not just how the steepest band moves but how the whole cost-rate
# spread reshapes across two knobs at once. The core + JSON always carry it
# (machine consumers); --show-rate-dist gates the human face; the CSV
# verdict-row schema is unchanged (the iter-358/366 stance).


def test_core_cell_carries_band_rate_dist():
    cells = gv.vad_gap_peak_grid([0.5], [800.0], [_multi_band()])
    dist = cells[0]["band_rate_dist"]
    assert dist["count"] == 3
    assert dist["min"] == 0.125
    assert dist["max"] == 0.5
    # Default percentiles p50/p75/p90/p99.
    assert [e["p"] for e in dist["percentiles"]] == [50.0, 75.0, 90.0, 99.0]


def test_core_band_rate_dist_matches_single_shot():
    """Each cell's distribution equals an independent vad_gap_peak on the same
    result — the grid names the SAME spread the single-shot verdict does."""
    r = _multi_band()
    direct = gv.vad_gap_peak(r)["band_rate_dist"]
    cell = gv.vad_gap_peak_grid([0.5], [800.0], [r])[0]
    assert cell["band_rate_dist"] == direct


def test_core_band_rate_dist_honors_rate_pcts():
    cells = gv.vad_gap_peak_grid([0.5], [800.0], [_multi_band()], rate_pcts=[90.0])
    assert [e["p"] for e in cells[0]["band_rate_dist"]["percentiles"]] == [90.0]


def test_core_no_peak_cell_has_empty_dist():
    """An all-valley cell has bands but none non-empty → the empty distribution
    (count 0, aggregates None, percentiles [])."""
    cells = gv.vad_gap_peak_grid([0.5], [800.0], [_all_valley()])
    dist = cells[0]["band_rate_dist"]
    assert dist["count"] == 0
    assert dist["min"] is None
    assert dist["percentiles"] == []


def test_core_single_segment_cell_has_empty_dist():
    cells = gv.vad_gap_peak_grid([0.9], [800.0], [_single()])
    assert cells[0]["band_rate_dist"]["count"] == 0


def test_core_band_rate_dist_per_cell_row_major():
    """One distribution per (row, col) cell, in row-major order; a no-peak cell
    carries the empty distribution between two populated ones."""
    cells = gv.vad_gap_peak_grid(
        [0.3, 0.7], [400.0, 800.0],
        [_multi_band(), _all_valley(), _single(), _multi_band()],
    )
    counts = [c["band_rate_dist"]["count"] for c in cells]
    assert counts == [3, 0, 0, 3]


def test_render_human_default_omits_dist():
    """Without --show-rate-dist the human table is byte-for-byte the old shape:
    no 'band-rate dist' sub-block."""
    lines = gv.render_vad_gap_peak_grid(
        [0.3], [800.0], [_multi_band()], name="rec.wav"
    )
    assert not any("band-rate dist" in ln for ln in lines)


def test_render_human_show_rate_dist_appends_block():
    lines = gv.render_vad_gap_peak_grid(
        [0.3], [800.0], [_multi_band()], name="rec.wav", show_rate_dist=True
    )
    dist_lines = [ln for ln in lines if "band-rate dist" in ln]
    assert len(dist_lines) == 1
    assert "3 non-empty bands" in dist_lines[0]
    assert "(iter-367)" in dist_lines[0]
    # The default four percentile rows follow.
    assert any(ln.strip().startswith("p50:") for ln in lines)
    assert any(ln.strip().startswith("p99:") for ln in lines)


def test_render_human_show_rate_dist_per_cell():
    """One distribution block per grid cell (here a 1×2 grid)."""
    lines = gv.render_vad_gap_peak_grid(
        [0.3], [400.0, 800.0], [_multi_band(), _multi_band()],
        name="rec.wav", show_rate_dist=True,
    )
    assert len([ln for ln in lines if "band-rate dist" in ln]) == 2


def test_render_human_show_rate_dist_no_peak_note():
    """A no-peak cell under --show-rate-dist prints the 'no non-empty bands' note
    rather than percentile rows."""
    lines = gv.render_vad_gap_peak_grid(
        [0.5], [800.0], [_all_valley()], name="rec.wav", show_rate_dist=True
    )
    note = [ln for ln in lines if "band-rate dist" in ln]
    assert len(note) == 1
    assert "no non-empty bands" in note[0]
    assert not any(ln.strip().startswith("p50:") for ln in lines)


def test_render_human_show_rate_dist_honors_rate_pcts():
    lines = gv.render_vad_gap_peak_grid(
        [0.3], [800.0], [_multi_band()], name="rec.wav",
        show_rate_dist=True, rate_pcts=[90.0],
    )
    assert any(ln.strip().startswith("p90:") for ln in lines)
    assert not any(ln.strip().startswith("p50:") for ln in lines)


def test_render_json_cells_carry_band_rate_dist():
    """The JSON face ALWAYS carries band_rate_dist per cell (no flag), like the
    single-shot render_vad_gap_peak_json and the sweep's _sweep_json."""
    out = gv.render_vad_gap_peak_grid_json(
        [0.3], [800.0], [_multi_band()], name="rec.wav"
    )
    payload = json.loads(out)
    assert payload["rate_pcts"] == list(gv.DEFAULT_BAND_RATE_PCTS)
    dist = payload["grid"][0]["band_rate_dist"]
    assert dist["count"] == 3
    assert [e["p"] for e in dist["percentiles"]] == [50.0, 75.0, 90.0, 99.0]


def test_render_json_echoes_custom_rate_pcts():
    out = gv.render_vad_gap_peak_grid_json(
        [0.3], [800.0], [_multi_band()], name="rec.wav", rate_pcts=[50.0, 99.0]
    )
    payload = json.loads(out)
    assert payload["rate_pcts"] == [50.0, 99.0]
    assert [e["p"] for e in payload["grid"][0]["band_rate_dist"]["percentiles"]] \
        == [50.0, 99.0]


def test_render_csv_schema_unchanged_no_dist_column():
    """The CSV body is the iter-365 ten-column schema — band_rate_dist is NOT a
    column (the iter-358/366 verdict-row stance)."""
    out = gv.render_vad_gap_peak_grid_csv(
        [0.3], [800.0], [_multi_band()], name="rec.wav"
    )
    header = list(csv.reader(io.StringIO(out)))[0]
    assert header == [
        "threshold", "min_silence_ms", "num_segments", "num_gaps", "peak_found",
        "peak_from_ms", "peak_to_ms", "peak_width_ms", "peak_merged_added",
        "peak_rate_per_100ms",
    ]


# ---- parser & handler: --show-rate-dist / --rate-pcts -------------------


def test_parser_show_rate_dist_defaults():
    args = gv.build_parser().parse_args(["vad-gap-peak-grid", "rec.wav"])
    assert args.show_rate_dist is False
    assert args.rate_pcts == list(gv.DEFAULT_BAND_RATE_PCTS)


def test_parser_custom_rate_pcts():
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--show-rate-dist", "--rate-pcts", "50,99"]
    )
    assert args.show_rate_dist is True
    assert args.rate_pcts == [50.0, 99.0]


def test_handler_show_rate_dist_human(monkeypatch):
    _stub_silero(monkeypatch)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--thresholds", "0.3,0.7",
         "--min-silences", "800", "--show-rate-dist"]
    )
    lines = []
    gv.cmd_vad_gap_peak_grid(args, log=lines.append,
                             segmenter=_make_segmenter(_multi_band()),
                             availability=_avail_true)
    # 2×1 grid → one distribution block per cell.
    assert len([ln for ln in lines if "band-rate dist" in ln]) == 2
    assert any("(iter-367)" in ln for ln in lines)


def test_handler_default_no_dist_block(monkeypatch):
    _stub_silero(monkeypatch)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--thresholds", "0.5",
         "--min-silences", "800"]
    )
    lines = []
    gv.cmd_vad_gap_peak_grid(args, log=lines.append,
                             segmenter=_make_segmenter(_multi_band()),
                             availability=_avail_true)
    assert not any("band-rate dist" in ln for ln in lines)


def test_handler_rate_pcts_to_json(monkeypatch):
    _stub_silero(monkeypatch)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--thresholds", "0.5",
         "--min-silences", "800", "--rate-pcts", "50,99", "--json"]
    )
    out = []
    gv.cmd_vad_gap_peak_grid(args, log=out.append,
                             segmenter=_make_segmenter(_multi_band()),
                             availability=_avail_true)
    payload = json.loads(out[0])
    assert payload["rate_pcts"] == [50.0, 99.0]
    assert [e["p"] for e in payload["grid"][0]["band_rate_dist"]["percentiles"]] \
        == [50.0, 99.0]


def test_handler_unavailable_json_carries_no_grid(monkeypatch):
    """The unavailable JSON degrade is unchanged by --rate-pcts (no grid key)."""
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--rate-pcts", "50,99", "--json"]
    )
    out = []
    gv.cmd_vad_gap_peak_grid(args, log=out.append,
                             segmenter=_make_segmenter(_multi_band()),
                             availability=_avail_false)
    payload = json.loads(out[0])
    assert payload["available"] is False
    assert "grid" not in payload


# ---- iter-369: per-cell top-N ranking on the cost-peak grid -------------
#
# iter-354 added the top-N ranking (the N steepest bands) to the SINGLE-shot
# `gv vad-gap-peak`; iter-368 carried it into the 1-D SWEEP. iter-369 completes
# the trio by carrying that same view into the 2-D GRID, one ranking per
# (row, col) cell, so an operator watches not just how the single steepest band
# moves but how the WHOLE ranking reorders across two knobs at once. The core +
# JSON always carry the `peaks` list (machine consumers); --top-n > 1 gates the
# human numbered sub-block; the CSV verdict-row schema is unchanged (the
# iter-368 stance).


def test_core_cell_carries_peaks_default_top_n_1():
    """At the default top_n=1 each cell's `peaks` holds exactly the single
    steepest band, and its rank-1 entry mirrors the scalar peak_* fields."""
    cells = gv.vad_gap_peak_grid([0.5], [800.0], [_multi_band()])
    peaks = cells[0]["peaks"]
    assert len(peaks) == 1
    assert peaks[0]["rank"] == 1
    assert peaks[0]["from_ms"] == cells[0]["peak_from_ms"]
    assert peaks[0]["rate_per_100ms"] == cells[0]["peak_rate_per_100ms"]


def test_core_top_n_ranks_steepest_first():
    """top_n=3 names the three distinct bands ranked by descending rate."""
    cells = gv.vad_gap_peak_grid([0.5], [800.0], [_multi_band()], top_n=3)
    peaks = cells[0]["peaks"]
    assert [p["rank"] for p in peaks] == [1, 2, 3]
    rates = [p["rate_per_100ms"] for p in peaks]
    assert rates == sorted(rates, reverse=True)
    assert rates == [0.5, 0.25, 0.125]


def test_core_top_n_caps_at_available_bands():
    """Fewer than top_n entries appear when the range holds fewer non-empty
    bands (here three distinct bands, top_n=5 → three entries)."""
    cells = gv.vad_gap_peak_grid([0.5], [800.0], [_multi_band()], top_n=5)
    assert len(cells[0]["peaks"]) == 3


def test_core_peaks_match_single_shot():
    """The per-cell ranking equals an independent vad_gap_peak --top-n on the
    same result — the grid names the SAME ranking the single-shot verdict does."""
    r = _multi_band()
    direct = gv.vad_gap_peak(r, top_n=3)["peaks"]
    cell = gv.vad_gap_peak_grid([0.5], [800.0], [r], top_n=3)[0]
    assert cell["peaks"] == direct


def test_core_no_peak_cell_has_empty_peaks():
    """An all-valley cell (bands but none non-empty) carries an empty ranking."""
    cells = gv.vad_gap_peak_grid([0.5], [800.0], [_all_valley()], top_n=3)
    assert cells[0]["peaks"] == []


def test_core_single_segment_cell_has_empty_peaks():
    cells = gv.vad_gap_peak_grid([0.9], [800.0], [_single()], top_n=3)
    assert cells[0]["peaks"] == []


def test_core_peaks_per_cell_row_major():
    """A no-peak cell between two populated cells carries an empty ranking while
    its neighbours carry full rankings — preserving row-major order."""
    cells = gv.vad_gap_peak_grid(
        [0.3, 0.7], [400.0, 800.0],
        [_multi_band(), _all_valley(), _single(), _multi_band()],
        top_n=3,
    )
    lengths = [len(c["peaks"]) for c in cells]
    assert lengths == [3, 0, 0, 3]


def test_render_human_default_top_n_1_omits_ranking():
    """Without --top-n > 1 the human table is byte-for-byte the old shape: no
    'top ... costliest bands' sub-block."""
    lines = gv.render_vad_gap_peak_grid(
        [0.5], [800.0], [_multi_band()], name="rec.wav"
    )
    assert not any("costliest bands" in ln for ln in lines)


def test_render_human_top_n_appends_ranking_block():
    lines = gv.render_vad_gap_peak_grid(
        [0.5], [800.0], [_multi_band()], name="rec.wav", top_n=3
    )
    header = [ln for ln in lines if "costliest bands" in ln]
    assert len(header) == 1
    assert "top 3 costliest bands" in header[0]
    # Three numbered ranked lines follow (one per distinct band).
    numbered = [ln for ln in lines if ln.strip().startswith("#")]
    assert len(numbered) == 3
    assert numbered[0].strip().startswith("#1:")
    assert numbered[2].strip().startswith("#3:")


def test_render_human_top_n_per_cell():
    """One ranking block per cell (here a 1×2 grid → two cells)."""
    lines = gv.render_vad_gap_peak_grid(
        [0.5], [400.0, 800.0], [_multi_band(), _multi_band()],
        name="rec.wav", top_n=2,
    )
    assert len([ln for ln in lines if "costliest bands" in ln]) == 2


def test_render_human_top_n_no_peak_note():
    """A no-peak cell under --top-n prints the '(no cost peak)' note rather than
    numbered ranked lines."""
    lines = gv.render_vad_gap_peak_grid(
        [0.5], [800.0], [_all_valley()], name="rec.wav", top_n=3
    )
    note = [ln for ln in lines if "no cost peak" in ln]
    assert len(note) == 1
    assert not any(ln.strip().startswith("#") for ln in lines)


def test_render_human_top_n_and_rate_dist_both_appear():
    """The ranking and the band-rate-distribution blocks are independent; both
    appear under a cell when both are requested, the ranking first."""
    lines = gv.render_vad_gap_peak_grid(
        [0.5], [800.0], [_multi_band()], name="rec.wav",
        top_n=2, show_rate_dist=True,
    )
    rank_i = next(i for i, ln in enumerate(lines) if "costliest bands" in ln)
    dist_i = next(i for i, ln in enumerate(lines) if "band-rate dist" in ln)
    assert rank_i < dist_i


def test_render_json_cells_carry_peaks_and_top_n():
    """The JSON face ALWAYS carries `peaks` per cell + a top-level `top_n`, like
    the single-shot render_vad_gap_peak_json and the sweep's _sweep_json."""
    out = gv.render_vad_gap_peak_grid_json(
        [0.5], [800.0], [_multi_band()], name="rec.wav", top_n=3
    )
    payload = json.loads(out)
    assert payload["top_n"] == 3
    peaks = payload["grid"][0]["peaks"]
    assert [p["rank"] for p in peaks] == [1, 2, 3]


def test_render_json_default_top_n_1_carries_single_peak():
    """At the default top_n=1 the JSON still carries the `peaks` list (one entry)
    and top_n=1 — a strict superset of the iter-365/367 shape."""
    out = gv.render_vad_gap_peak_grid_json(
        [0.5], [800.0], [_multi_band()], name="rec.wav"
    )
    payload = json.loads(out)
    assert payload["top_n"] == 1
    assert len(payload["grid"][0]["peaks"]) == 1


def test_render_csv_schema_unchanged_no_peaks_column():
    """The CSV body is the iter-365 ten-column scalar-steepest-band schema —
    `peaks` is NOT a column even with a multi-band result (iter-368 stance)."""
    out = gv.render_vad_gap_peak_grid_csv(
        [0.5], [800.0], [_multi_band()], name="rec.wav"
    )
    header = list(csv.reader(io.StringIO(out)))[0]
    assert header == [
        "threshold", "min_silence_ms", "num_segments", "num_gaps",
        "peak_found", "peak_from_ms", "peak_to_ms", "peak_width_ms",
        "peak_merged_added", "peak_rate_per_100ms",
    ]
    assert "peaks" not in out
    assert "rank" not in out


def test_parser_top_n_default():
    args = gv.build_parser().parse_args(["vad-gap-peak-grid", "rec.wav"])
    assert args.top_n == 1


def test_parser_top_n_custom_parsed():
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--top-n", "4"]
    )
    assert args.top_n == 4


def test_parser_top_n_rejects_zero():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-gap-peak-grid", "rec.wav", "--top-n", "0"]
        )


def test_handler_top_n_human(monkeypatch):
    _stub_silero(monkeypatch)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--thresholds", "0.5",
         "--min-silences", "800", "--top-n", "3"]
    )
    lines = []
    gv.cmd_vad_gap_peak_grid(args, log=lines.append,
                             segmenter=_make_segmenter(_multi_band()),
                             availability=_avail_true)
    assert any("top 3 costliest bands" in ln for ln in lines)


def test_handler_default_top_n_no_ranking(monkeypatch):
    _stub_silero(monkeypatch)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--thresholds", "0.5",
         "--min-silences", "800"]
    )
    lines = []
    gv.cmd_vad_gap_peak_grid(args, log=lines.append,
                             segmenter=_make_segmenter(_multi_band()),
                             availability=_avail_true)
    assert not any("costliest bands" in ln for ln in lines)


def test_handler_threads_top_n_to_json(monkeypatch):
    _stub_silero(monkeypatch)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--thresholds", "0.5",
         "--min-silences", "800", "--top-n", "3", "--json"]
    )
    out = []
    gv.cmd_vad_gap_peak_grid(args, log=out.append,
                             segmenter=_make_segmenter(_multi_band()),
                             availability=_avail_true)
    payload = json.loads(out[0])
    assert payload["top_n"] == 3
    assert [p["rank"] for p in payload["grid"][0]["peaks"]] == [1, 2, 3]


def test_handler_top_n_csv_schema_unchanged(monkeypatch):
    """--top-n does not change the CSV (scalar steepest band only)."""
    _stub_silero(monkeypatch)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--thresholds", "0.5",
         "--min-silences", "800", "--top-n", "3", "--csv"]
    )
    out = []
    gv.cmd_vad_gap_peak_grid(args, log=out.append,
                             segmenter=_make_segmenter(_multi_band()),
                             availability=_avail_true)
    header = list(csv.reader(io.StringIO(out[0])))[0]
    assert "peaks" not in out[0]
    assert header[0] == "threshold"


# ==== iter-371: per-cell rate floor (--min-rate / --min-rate-pct) ==========
#
# The 2-D analogue of the iter-370 sweep floor. Where band_rate_dist (iter-367)
# and top_n (iter-369) are purely additive, the floor FILTERS bands before
# ranking, so the scalar peak columns + the ranking reflect it on ALL three
# faces. The genuinely-new per-cell datum is `effective_min_rate`: the adaptive
# floor resolves to a DIFFERENT absolute cut at each cell. band_rate_dist stays
# the full pre-floor distribution (the sample the percentile floor reads
# against).
#
# _multi_band has 3 distinct gaps → band rates 0.5 / 0.25 / 0.125. So:
#   p75 of [0.125, 0.25, 0.5] = 0.375 → only the 0.5 band survives.
#   --min-rate 0.3 (absolute) → only the 0.5 band survives.
#   --min-rate 1.0 → no band survives → no peak.


def test_core_cell_carries_effective_min_rate_default():
    """With no floor every cell carries effective_min_rate == 0.0 (a strict
    superset of the iter-369 shape)."""
    cells = gv.vad_gap_peak_grid([0.5], [800.0], [_multi_band()])
    assert cells[0]["effective_min_rate"] == 0.0


def test_core_absolute_floor_filters_bands():
    """--min-rate drops bands below the absolute threshold; the steepest survivor
    becomes the scalar peak, and effective_min_rate echoes the floor."""
    cells = gv.vad_gap_peak_grid([0.5], [800.0], [_multi_band()], min_rate=0.3)
    assert cells[0]["effective_min_rate"] == 0.3
    assert cells[0]["peak_found"] is True
    # Only the 0.5 band clears 0.3 → it is the (single) survivor.
    assert cells[0]["peak_rate_per_100ms"] == 0.5
    assert len(cells[0]["peaks"]) == 1


def test_core_absolute_floor_can_eliminate_peak():
    """A floor above every band's rate leaves no surviving peak — the no-peak
    verdict (peak_found False, empty peaks) exactly as an all-valley cell."""
    cells = gv.vad_gap_peak_grid([0.5], [800.0], [_multi_band()], min_rate=1.0)
    assert cells[0]["peak_found"] is False
    assert cells[0]["peak_rate_per_100ms"] is None
    assert cells[0]["peaks"] == []
    # The floor still rode the cell (it describes the cutoff, not the peak).
    assert cells[0]["effective_min_rate"] == 1.0


def test_core_pct_floor_resolves_per_cell_cut():
    """--min-rate-pct resolves to the Pth percentile of THIS cell's own band
    rates — p75 of [0.125,0.25,0.5] = 0.375."""
    cells = gv.vad_gap_peak_grid([0.5], [800.0], [_multi_band()], min_rate_pct=75.0)
    assert cells[0]["effective_min_rate"] == 0.375
    # Only the 0.5 band clears 0.375.
    assert cells[0]["peak_rate_per_100ms"] == 0.5
    assert len(cells[0]["peaks"]) == 1


def test_core_floor_matches_single_shot():
    """Each cell's floored verdict equals an independent vad_gap_peak with the
    same floor — the grid names the SAME survivors the single-shot does."""
    r = _multi_band()
    direct = gv.vad_gap_peak(r, min_rate_pct=75.0)
    cell = gv.vad_gap_peak_grid([0.5], [800.0], [r], min_rate_pct=75.0)[0]
    assert cell["effective_min_rate"] == direct["effective_min_rate"]
    assert cell["peak_rate_per_100ms"] == direct["peak_rate_per_100ms"]
    assert cell["peaks"] == direct["peaks"]


def test_core_band_rate_dist_is_full_pre_floor():
    """band_rate_dist is UNCHANGED by the floor — always the full pre-floor
    distribution (the sample --min-rate-pct reads against)."""
    floored = gv.vad_gap_peak_grid([0.5], [800.0], [_multi_band()],
                                   min_rate_pct=75.0)[0]
    unfloored = gv.vad_gap_peak_grid([0.5], [800.0], [_multi_band()])[0]
    assert floored["band_rate_dist"] == unfloored["band_rate_dist"]
    assert floored["band_rate_dist"]["count"] == 3


def test_core_pct_floor_reshapes_across_cells():
    """The adaptive cut differs cell to cell when the cost distribution differs:
    _multi_band (rates 0.5/0.25/0.125) vs _three (single 0.125 band)."""
    cells = gv.vad_gap_peak_grid([0.3], [400.0, 800.0],
                                 [_multi_band(), _three()], min_rate_pct=75.0)
    assert cells[0]["effective_min_rate"] == 0.375  # p75 of three rates
    assert cells[1]["effective_min_rate"] == 0.125  # p75 of one rate == that rate


def test_core_mutually_exclusive_floors_raise():
    """Setting both floors is rejected (delegated to vad_gap_peak)."""
    with pytest.raises(ValueError):
        gv.vad_gap_peak_grid([0.5], [800.0], [_multi_band()], min_rate=0.2,
                             min_rate_pct=75.0)


# ---- renderer: human floor header + per-cell adaptive note ---------------


def test_render_human_default_omits_floor_header():
    """No floor → no 'rate floor' header line (byte-for-byte unchanged)."""
    lines = gv.render_vad_gap_peak_grid([0.5], [800.0], [_multi_band()],
                                        name="rec.wav")
    assert not any("rate floor" in ln for ln in lines)
    assert not any("floor: p" in ln for ln in lines)


def test_render_human_absolute_floor_header_no_per_cell_note():
    """--min-rate prints the header once; no per-cell note (the cut is the same
    at every cell)."""
    lines = gv.render_vad_gap_peak_grid([0.3], [400.0, 800.0],
                                        [_multi_band(), _multi_band()],
                                        name="rec.wav", min_rate=0.3)
    header = [ln for ln in lines if "rate floor" in ln]
    assert len(header) == 1
    assert "0.300 per +100ms" in header[0]
    assert not any("floor: p" in ln for ln in lines)


def test_render_human_pct_floor_per_cell_note():
    """--min-rate-pct prints the header plus one per-cell note naming the resolved
    cut (which can differ cell to cell)."""
    lines = gv.render_vad_gap_peak_grid([0.3], [400.0, 800.0],
                                        [_multi_band(), _three()],
                                        name="rec.wav", min_rate_pct=75.0)
    assert any("rate floor:   p75" in ln for ln in lines)
    notes = [ln for ln in lines if "floor: p75 =" in ln]
    assert len(notes) == 2
    assert "0.375 per +100ms" in notes[0]
    assert "0.125 per +100ms" in notes[1]


def test_render_human_floor_unavailable_hint():
    """A None result still yields the shared install hint regardless of floor."""
    lines = gv.render_vad_gap_peak_grid([], [], [None], name="rec.wav",
                                        min_rate_pct=75.0)
    assert lines == [
        "silero VAD unavailable: install 'silero-vad' (pulls torch + "
        "torchaudio) to enable offline neural segmentation"
    ]


def test_render_human_floor_and_top_n_both_appear():
    """The per-cell floor note (iter-371) and the top-N ranking (iter-369) are
    independent — both appear under a cell, the floor note first."""
    lines = gv.render_vad_gap_peak_grid([0.3], [800.0], [_multi_band()],
                                        name="rec.wav", min_rate_pct=75.0,
                                        top_n=3)
    floor_idx = next(i for i, ln in enumerate(lines) if "floor: p75 =" in ln)
    rank_idx = next(i for i, ln in enumerate(lines)
                    if "costliest bands" in ln)
    assert floor_idx < rank_idx


# ---- renderer: JSON echoes floor + per-cell effective_min_rate -----------


def test_render_json_echoes_floor_and_per_cell_cut():
    out = gv.render_vad_gap_peak_grid_json([0.5], [800.0], [_multi_band()],
                                           name="rec.wav", min_rate_pct=75.0)
    payload = json.loads(out)
    assert payload["min_rate"] == 0.0
    assert payload["min_rate_pct"] == 75.0
    assert payload["grid"][0]["effective_min_rate"] == 0.375


def test_render_json_default_floor_shape():
    """No floor → min_rate 0.0 / min_rate_pct null / effective_min_rate 0.0 (a
    strict superset of the iter-369 JSON)."""
    out = gv.render_vad_gap_peak_grid_json([0.5], [800.0], [_multi_band()],
                                           name="rec.wav")
    payload = json.loads(out)
    assert payload["min_rate"] == 0.0
    assert payload["min_rate_pct"] is None
    assert payload["grid"][0]["effective_min_rate"] == 0.0


def test_render_json_absolute_floor_echo():
    out = gv.render_vad_gap_peak_grid_json([0.5], [800.0], [_multi_band()],
                                           name="rec.wav", min_rate=0.3)
    payload = json.loads(out)
    assert payload["min_rate"] == 0.3
    assert payload["min_rate_pct"] is None
    assert payload["grid"][0]["effective_min_rate"] == 0.3


# ---- renderer: CSV appends effective_min_rate column only under a floor ---


def test_render_csv_default_schema_unchanged():
    """No floor → the iter-365 ten-column schema, no effective_min_rate column."""
    out = gv.render_vad_gap_peak_grid_csv([0.5], [800.0], [_multi_band()],
                                          name="rec.wav")
    header = list(csv.reader(io.StringIO(out)))[0]
    assert header == [
        "threshold", "min_silence_ms", "num_segments", "num_gaps", "peak_found",
        "peak_from_ms", "peak_to_ms", "peak_width_ms", "peak_merged_added",
        "peak_rate_per_100ms",
    ]
    assert "effective_min_rate" not in out


def test_render_csv_floor_appends_column():
    """A floor adds the trailing effective_min_rate column with the resolved cut."""
    out = gv.render_vad_gap_peak_grid_csv([0.5], [800.0], [_multi_band()],
                                          name="rec.wav", min_rate_pct=75.0)
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0][-1] == "effective_min_rate"
    assert rows[1][-1] == "0.375"
    # The scalar peak columns reflect the floor (only the 0.5 band survives).
    assert rows[1][rows[0].index("peak_rate_per_100ms")] == "0.5"


def test_render_csv_floor_column_on_no_peak_cell():
    """Even a cell the floor eliminates carries the cut in the column (it
    describes the cutoff, not the peak)."""
    out = gv.render_vad_gap_peak_grid_csv([0.5], [800.0], [_multi_band()],
                                          name="rec.wav", min_rate=1.0)
    rows = list(csv.reader(io.StringIO(out)))
    pk_idx = rows[0].index("peak_found")
    assert rows[1][pk_idx] == "False"
    assert rows[1][-1] == "1.0"


# ---- parser: --min-rate / --min-rate-pct --------------------------------


def test_parser_floor_defaults():
    args = gv.build_parser().parse_args(["vad-gap-peak-grid", "rec.wav"])
    assert args.min_rate == 0.0
    assert args.min_rate_pct is None


def test_parser_min_rate_parsed():
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--min-rate", "0.3"]
    )
    assert args.min_rate == 0.3


def test_parser_min_rate_pct_parsed():
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--min-rate-pct", "75"]
    )
    assert args.min_rate_pct == 75.0


def test_parser_floors_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-gap-peak-grid", "rec.wav", "--min-rate", "0.2",
             "--min-rate-pct", "75"]
        )


def test_parser_min_rate_pct_rejects_out_of_range():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-gap-peak-grid", "rec.wav", "--min-rate-pct", "150"]
        )


# ---- handler: end-to-end floor threading --------------------------------


def test_handler_pct_floor_human(monkeypatch):
    _stub_silero(monkeypatch)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--thresholds", "0.5",
         "--min-silences", "800", "--min-rate-pct", "75"]
    )
    lines = []
    gv.cmd_vad_gap_peak_grid(args, log=lines.append,
                             segmenter=_make_segmenter(_multi_band()),
                             availability=_avail_true)
    assert any("rate floor:   p75" in ln for ln in lines)
    assert any("floor: p75 = 0.375" in ln for ln in lines)


def test_handler_default_no_floor_header(monkeypatch):
    _stub_silero(monkeypatch)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--thresholds", "0.5",
         "--min-silences", "800"]
    )
    lines = []
    gv.cmd_vad_gap_peak_grid(args, log=lines.append,
                             segmenter=_make_segmenter(_multi_band()),
                             availability=_avail_true)
    assert not any("rate floor" in ln for ln in lines)


def test_handler_threads_floor_to_json(monkeypatch):
    _stub_silero(monkeypatch)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--thresholds", "0.5",
         "--min-silences", "800", "--min-rate-pct", "75", "--json"]
    )
    out = []
    gv.cmd_vad_gap_peak_grid(args, log=out.append,
                             segmenter=_make_segmenter(_multi_band()),
                             availability=_avail_true)
    payload = json.loads(out[0])
    assert payload["min_rate_pct"] == 75.0
    assert payload["grid"][0]["effective_min_rate"] == 0.375


def test_handler_threads_floor_to_csv(monkeypatch):
    """--min-rate adds the effective_min_rate column to the CSV face."""
    _stub_silero(monkeypatch)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-grid", "rec.wav", "--thresholds", "0.5",
         "--min-silences", "800", "--min-rate", "0.3", "--csv"]
    )
    out = []
    gv.cmd_vad_gap_peak_grid(args, log=out.append,
                             segmenter=_make_segmenter(_multi_band()),
                             availability=_avail_true)
    header = list(csv.reader(io.StringIO(out[0])))[0]
    assert header[-1] == "effective_min_rate"
