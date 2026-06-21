"""Tests for iter-364 — the ``gv vad-gap-peak-sweep`` subcommand (examples/gv.py).

iter-350 shipped ``gv vad-gap-peak`` — the COSTLIEST cost-curve band (the
densest pause cluster, the most expensive place to raise the end-of-turn
hangover) at ONE knob setting; iter-330 shipped ``gv vad-gap-sweep``, the
gap-side sweep tabulating min/mean/max gap across a swept knob. This lap adds the
peak-side sweep: ``gv vad-gap-peak-sweep`` is to ``gv vad-gap-peak`` what
``gv vad-gap-sweep`` is to ``gv vad-gaps`` — for each swept value it names the
steepest band so an operator can watch how the cost of pushing the hangover
through the densest pause cluster MOVES as a segmenter knob (e.g. the
``--min-speech-ms`` floor) tightens.

Like the rest of the VAD-analysis family, the handler takes injected
``segmenter`` / ``availability`` / ``log`` dependencies so every test runs
WITHOUT importing torch / silero-vad and without touching real audio — fast and
deterministic on the x86_64 Linux runner. The pure core (``vad_gap_peak_sweep``)
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


# Recurring stand-ins:
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


# ---- parser: registration & defaults -----------------------------------


def test_vad_gap_peak_sweep_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-gap-peak-sweep"] is gv.cmd_vad_gap_peak_sweep


def test_parser_default_axis_is_thresholds():
    args = gv.build_parser().parse_args(["vad-gap-peak-sweep", "rec.wav"])
    assert args.command == "vad-gap-peak-sweep"
    assert args.thresholds == [0.3, 0.5, 0.7, 0.9]
    assert args.min_silences is None
    assert args.min_speeches is None
    assert args.speech_pads is None
    assert args.max_speeches is None


def test_parser_cuts_ms_default():
    args = gv.build_parser().parse_args(["vad-gap-peak-sweep", "rec.wav"])
    assert args.cuts_ms == list(gv.DEFAULT_GAP_CDF_CUTS_MS)


def test_parser_cuts_ms_custom_parsed():
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-sweep", "rec.wav", "--cuts-ms", "200,400,800"]
    )
    assert args.cuts_ms == [200.0, 400.0, 800.0]


def test_parser_defaults_mirror_silero_params():
    """The held-fixed scalar knobs default to the same values as ``gv vad``."""
    args = gv.build_parser().parse_args(["vad-gap-peak-sweep", "rec.wav"])
    vad = gv.build_parser().parse_args(["vad", "rec.wav"])
    assert args.threshold == vad.threshold
    assert args.min_speech_ms == vad.min_speech_ms
    assert args.min_silence_ms == vad.min_silence_ms
    assert args.speech_pad_ms == vad.speech_pad_ms
    assert args.max_speech_s == vad.max_speech_s


def test_parser_axes_are_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["vad-gap-peak-sweep", "rec.wav", "--thresholds", "0.3,0.5",
             "--min-silences", "200,400"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["vad-gap-peak-sweep", "rec.wav", "--min-speeches", "50,100",
             "--speech-pads", "0,20"]
        )


def test_parser_json_csv_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-peak-sweep", "rec.wav", "--json", "--csv"])


def test_parser_has_no_target_pick_args():
    """Like the gap sweep, the peak sweep has no --target / --top / --tie-break."""
    args = gv.build_parser().parse_args(["vad-gap-peak-sweep", "rec.wav"])
    assert not hasattr(args, "target")
    assert not hasattr(args, "top")
    assert not hasattr(args, "tie_break")


# ---- pure core: vad_gap_peak_sweep --------------------------------------


def test_core_basic_two_value_threshold_sweep():
    rows = gv.vad_gap_peak_sweep([0.3, 0.9], [_three(), _single()])
    assert rows[0] == {
        "threshold": 0.3,
        "num_segments": 3,
        "num_gaps": 2,
        "peak_found": True,
        "peak_from_ms": 800.0,
        "peak_to_ms": 1600.0,
        "peak_width_ms": 800.0,
        "peak_merged_added": 1,
        "peak_rate_per_100ms": 0.125,
        # iter-368: the top-N ranking at this value (top_n=1 → the single peak).
        "peaks": [
            {
                "rank": 1,
                "from_ms": 800.0,
                "to_ms": 1600.0,
                "from_s": 0.8,
                "to_s": 1.6,
                "width_ms": 800.0,
                "merged_added": 1,
                "rate_per_100ms": 0.125,
            },
        ],
        # iter-366: the observed non-empty band-rate distribution at this value.
        "band_rate_dist": {
            "count": 1,
            "min": 0.125,
            "mean": 0.125,
            "max": 0.125,
            "percentiles": [
                {"p": 50.0, "rate": 0.125},
                {"p": 75.0, "rate": 0.125},
                {"p": 90.0, "rate": 0.125},
                {"p": 99.0, "rate": 0.125},
            ],
        },
    }
    # The single-segment row has no pause: no peak, fields None / False, and the
    # empty band-rate distribution (iter-366).
    assert rows[1] == {
        "threshold": 0.9,
        "num_segments": 1,
        "num_gaps": 0,
        "peak_found": False,
        "peak_from_ms": None,
        "peak_to_ms": None,
        "peak_width_ms": None,
        "peak_merged_added": None,
        "peak_rate_per_100ms": None,
        # iter-368: no peak → empty ranking.
        "peaks": [],
        "band_rate_dist": {
            "count": 0,
            "min": None,
            "mean": None,
            "max": None,
            "percentiles": [],
        },
    }


def test_core_all_valley_row_has_no_peak():
    """A result whose only gap is beyond every cut has bands but no cost peak."""
    rows = gv.vad_gap_peak_sweep([0.5], [_all_valley()])
    assert rows[0]["num_gaps"] == 1
    assert rows[0]["peak_found"] is False
    assert rows[0]["peak_from_ms"] is None
    assert rows[0]["peak_rate_per_100ms"] is None


def test_core_axis_key_follows_axis_arg():
    rows = gv.vad_gap_peak_sweep([50.0, 200.0], [_three(), _three()],
                                 axis="min_speech_ms")
    assert "min_speech_ms" in rows[0]
    assert "threshold" not in rows[0]
    assert rows[0]["min_speech_ms"] == 50.0


def test_core_peak_fields_match_vad_gap_peak():
    """Each row's peak fields equal an independent vad_gap_peak (top_n=1) on its
    result — the sweep names the SAME steepest band the verdict surface does."""
    r = _three()
    direct = gv.vad_gap_peak(r)
    row = gv.vad_gap_peak_sweep([0.5], [r])[0]
    for key in ("num_segments", "num_gaps", "peak_found", "peak_from_ms",
                "peak_to_ms", "peak_width_ms", "peak_merged_added",
                "peak_rate_per_100ms"):
        assert row[key] == direct[key]


def test_core_custom_cuts_ms_changes_band():
    """A coarser cut axis lands the 2.0s pause in a different band."""
    rows = gv.vad_gap_peak_sweep([0.5], [_three()], cuts_ms=[1000.0, 3000.0])
    assert rows[0]["peak_found"] is True
    assert rows[0]["peak_from_ms"] == 1000.0
    assert rows[0]["peak_to_ms"] == 3000.0


def test_core_length_mismatch_raises():
    with pytest.raises(ValueError):
        gv.vad_gap_peak_sweep([0.3, 0.5], [_three()])


def test_core_empty_sweep_is_empty():
    assert gv.vad_gap_peak_sweep([], []) == []


# ---- renderer: render_vad_gap_peak_sweep (human) ------------------------


def test_render_human_header_and_rows():
    lines = gv.render_vad_gap_peak_sweep([0.3, 0.9], [_three(), _single()],
                                         name="rec.wav")
    assert lines[0] == "silero VAD gap cost-peak sweep — rec.wav"
    assert "threshold" in lines[1]
    assert "peak_band_ms" in lines[1]
    assert "rate/100ms" in lines[1]
    # First row names the 800-1600 band; second row (no peak) shows dashes.
    assert "800-1600" in lines[2]
    assert "0.125" in lines[2]
    assert "-" in lines[3]
    assert "800-1600" not in lines[3]


def test_render_human_axis_label_follows_axis():
    lines = gv.render_vad_gap_peak_sweep([50.0], [_three()], name="rec.wav",
                                         axis="min_speech_ms")
    assert "min_speech" in lines[1]


def test_render_human_unavailable_hint():
    lines = gv.render_vad_gap_peak_sweep([], [None], name="rec.wav")
    assert len(lines) == 1
    assert "silero VAD unavailable" in lines[0]


# ---- renderer: render_vad_gap_peak_sweep_json ---------------------------


def test_render_json_shape():
    out = gv.render_vad_gap_peak_sweep_json([0.3, 0.9], [_three(), _single()],
                                            name="rec.wav")
    payload = json.loads(out)
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["axis"] == "threshold"
    assert payload["cuts_ms"] == list(gv.DEFAULT_GAP_CDF_CUTS_MS)
    assert len(payload["sweep"]) == 2
    first = payload["sweep"][0]
    assert first["peak_found"] is True
    assert first["peak_from_ms"] == 800.0
    assert first["peak_rate_per_100ms"] == 0.125
    # No-peak row carries JSON null for the peak measures.
    second = payload["sweep"][1]
    assert second["peak_found"] is False
    assert second["peak_from_ms"] is None
    assert second["peak_rate_per_100ms"] is None


def test_render_json_carries_custom_cuts_ms():
    out = gv.render_vad_gap_peak_sweep_json([0.5], [_three()], name="rec.wav",
                                            cuts_ms=[1000.0, 3000.0])
    payload = json.loads(out)
    assert payload["cuts_ms"] == [1000.0, 3000.0]
    assert payload["sweep"][0]["peak_from_ms"] == 1000.0


def test_render_json_axis_key_follows_axis():
    out = gv.render_vad_gap_peak_sweep_json([200.0], [_three()], name="rec.wav",
                                            axis="min_silence_ms")
    payload = json.loads(out)
    assert payload["axis"] == "min_silence_ms"
    assert "min_silence_ms" in payload["sweep"][0]


def test_render_json_unavailable():
    out = gv.render_vad_gap_peak_sweep_json([], [None], name="rec.wav")
    payload = json.loads(out)
    assert payload["available"] is False
    assert "install 'silero-vad'" in payload["hint"]


# ---- renderer: render_vad_gap_peak_sweep_csv ----------------------------


def test_render_csv_header_and_rows():
    out = gv.render_vad_gap_peak_sweep_csv([0.3, 0.9], [_three(), _single()],
                                           name="rec.wav")
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0] == [
        "threshold", "num_segments", "num_gaps", "peak_found",
        "peak_from_ms", "peak_to_ms", "peak_width_ms", "peak_merged_added",
        "peak_rate_per_100ms",
    ]
    # Peak row.
    assert rows[1] == ["0.3", "3", "2", "True", "800", "1600", "800", "1", "0.125"]
    # No-peak row: peak_found False, blank peak-measure cells.
    assert rows[2] == ["0.9", "1", "0", "False", "", "", "", "", ""]


def test_render_csv_axis_header_follows_axis():
    out = gv.render_vad_gap_peak_sweep_csv([200.0], [_three()], name="rec.wav",
                                           axis="min_silence_ms")
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0][0] == "min_silence_ms"
    assert rows[1][0] == "200.0"


def test_render_csv_custom_cuts_ms():
    out = gv.render_vad_gap_peak_sweep_csv([0.5], [_three()], name="rec.wav",
                                           cuts_ms=[1000.0, 3000.0])
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[1][4] == "1000"  # peak_from_ms
    assert rows[1][5] == "3000"  # peak_to_ms


def test_render_csv_unavailable_comment():
    out = gv.render_vad_gap_peak_sweep_csv([], [None], name="rec.wav")
    assert out.startswith("# silero VAD unavailable")
    assert "\n" not in out


def test_render_csv_no_trailing_newline():
    out = gv.render_vad_gap_peak_sweep_csv([0.5], [_three()], name="rec.wav")
    assert not out.endswith("\n")


# ---- handler: cmd_vad_gap_peak_sweep ------------------------------------


def _avail_true():
    return True


def _avail_false():
    return False


def _make_segmenter(result):
    """A segmenter stub ignoring params and always returning ``result``."""
    def _seg(wav, params=None):
        return result

    return _seg


def test_handler_threshold_sweep_human(monkeypatch):
    # Stub SileroParams so the lazy `from vad.silero import SileroParams` inside
    # the handler succeeds without torch.
    import types
    fake = types.ModuleType("vad.silero")
    fake.SileroParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vad.silero", fake)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-sweep", "rec.wav", "--thresholds", "0.3,0.7"]
    )
    lines = []
    gv.cmd_vad_gap_peak_sweep(args, log=lines.append,
                              segmenter=_make_segmenter(_three()),
                              availability=_avail_true)
    assert lines[0] == "silero VAD gap cost-peak sweep — rec.wav"
    # Two swept values → two data rows after the two header lines.
    assert len([ln for ln in lines if "800-1600" in ln]) == 2


def test_handler_min_speeches_axis(monkeypatch):
    import types
    fake = types.ModuleType("vad.silero")
    fake.SileroParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vad.silero", fake)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-sweep", "rec.wav", "--min-speeches", "50,100,200",
         "--json"]
    )
    out = []
    gv.cmd_vad_gap_peak_sweep(args, log=out.append,
                              segmenter=_make_segmenter(_three()),
                              availability=_avail_true)
    payload = json.loads(out[0])
    assert payload["axis"] == "min_speech_ms"
    assert len(payload["sweep"]) == 3
    assert all("min_speech_ms" in row for row in payload["sweep"])


def test_handler_threads_cuts_ms(monkeypatch):
    import types
    fake = types.ModuleType("vad.silero")
    fake.SileroParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vad.silero", fake)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-sweep", "rec.wav", "--thresholds", "0.5",
         "--cuts-ms", "1000,3000", "--json"]
    )
    out = []
    gv.cmd_vad_gap_peak_sweep(args, log=out.append,
                              segmenter=_make_segmenter(_three()),
                              availability=_avail_true)
    payload = json.loads(out[0])
    assert payload["cuts_ms"] == [1000.0, 3000.0]
    assert payload["sweep"][0]["peak_from_ms"] == 1000.0


def test_handler_csv(monkeypatch):
    import types
    fake = types.ModuleType("vad.silero")
    fake.SileroParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vad.silero", fake)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-sweep", "rec.wav", "--thresholds", "0.5", "--csv"]
    )
    out = []
    gv.cmd_vad_gap_peak_sweep(args, log=out.append,
                              segmenter=_make_segmenter(_three()),
                              availability=_avail_true)
    rows = list(csv.reader(io.StringIO(out[0])))
    assert rows[0][0] == "threshold"
    assert rows[1] == ["0.5", "3", "2", "True", "800", "1600", "800", "1", "0.125"]


def test_handler_unavailable_human():
    args = gv.build_parser().parse_args(["vad-gap-peak-sweep", "rec.wav"])
    lines = []
    gv.cmd_vad_gap_peak_sweep(args, log=lines.append,
                              segmenter=_make_segmenter(_three()),
                              availability=_avail_false)
    assert len(lines) == 1
    assert "silero VAD unavailable" in lines[0]


def test_handler_unavailable_json():
    args = gv.build_parser().parse_args(["vad-gap-peak-sweep", "rec.wav",
                                         "--json"])
    out = []
    gv.cmd_vad_gap_peak_sweep(args, log=out.append,
                              segmenter=_make_segmenter(_three()),
                              availability=_avail_false)
    payload = json.loads(out[0])
    assert payload["available"] is False


def test_handler_unavailable_csv():
    args = gv.build_parser().parse_args(["vad-gap-peak-sweep", "rec.wav",
                                         "--csv"])
    out = []
    gv.cmd_vad_gap_peak_sweep(args, log=out.append,
                              segmenter=_make_segmenter(_three()),
                              availability=_avail_false)
    assert out[0].startswith("# silero VAD unavailable")


# ---- iter-366: per-row band_rate_dist on the cost-peak sweep ------------
#
# iter-358 added the observed non-empty band-rate distribution (the
# --min-rate-pct sample) to the SINGLE-shot `gv vad-gap-peak`. iter-366 carries
# that same view into the SWEEP, one distribution per swept value, so an
# operator watches not just how the steepest band moves but how the whole
# cost-rate spread shifts as a segmenter knob tightens. The core + JSON always
# carry it (machine consumers); --show-rate-dist gates the human face; the CSV
# verdict-row schema is unchanged (iter-358 stance).


def _multi_band():
    """4 segments / 3 gaps (0.3s, 0.5s, 1.0s) landing in three distinct cost
    bands, so band_rate_dist has count 3 — a meaningful spread to summarise."""
    return _result((0.0, 1.0), (1.3, 2.0), (2.5, 3.0), (4.0, 5.0))


def test_core_row_carries_band_rate_dist():
    rows = gv.vad_gap_peak_sweep([0.5], [_multi_band()])
    dist = rows[0]["band_rate_dist"]
    assert dist["count"] == 3
    assert dist["min"] == 0.125
    assert dist["max"] == 0.5
    # Default percentiles p50/p75/p90/p99.
    assert [e["p"] for e in dist["percentiles"]] == [50.0, 75.0, 90.0, 99.0]


def test_core_band_rate_dist_matches_single_shot():
    """The per-row distribution equals an independent vad_gap_peak on the same
    result — the sweep names the SAME spread the single-shot verdict does."""
    r = _multi_band()
    direct = gv.vad_gap_peak(r)["band_rate_dist"]
    row = gv.vad_gap_peak_sweep([0.5], [r])[0]
    assert row["band_rate_dist"] == direct


def test_core_band_rate_dist_honors_rate_pcts():
    rows = gv.vad_gap_peak_sweep([0.5], [_multi_band()], rate_pcts=[90.0])
    assert [e["p"] for e in rows[0]["band_rate_dist"]["percentiles"]] == [90.0]


def test_core_no_peak_row_has_empty_dist():
    """An all-valley row has bands but none non-empty → the empty distribution
    (count 0, aggregates None, percentiles [])."""
    rows = gv.vad_gap_peak_sweep([0.5], [_all_valley()])
    dist = rows[0]["band_rate_dist"]
    assert dist["count"] == 0
    assert dist["min"] is None
    assert dist["percentiles"] == []


def test_core_single_segment_row_has_empty_dist():
    rows = gv.vad_gap_peak_sweep([0.9], [_single()])
    assert rows[0]["band_rate_dist"]["count"] == 0


def test_render_human_default_omits_dist():
    """Without --show-rate-dist the human table is byte-for-byte the old shape:
    no 'band-rate dist' sub-block."""
    lines = gv.render_vad_gap_peak_sweep([0.3], [_multi_band()], name="rec.wav")
    assert not any("band-rate dist" in ln for ln in lines)


def test_render_human_show_rate_dist_appends_block():
    lines = gv.render_vad_gap_peak_sweep([0.3], [_multi_band()], name="rec.wav",
                                         show_rate_dist=True)
    dist_lines = [ln for ln in lines if "band-rate dist" in ln]
    assert len(dist_lines) == 1
    assert "3 non-empty bands" in dist_lines[0]
    assert "(iter-366)" in dist_lines[0]
    # The default four percentile rows follow.
    assert any(ln.strip().startswith("p50:") for ln in lines)
    assert any(ln.strip().startswith("p99:") for ln in lines)


def test_render_human_show_rate_dist_per_row():
    """One distribution block per swept value (here two values)."""
    lines = gv.render_vad_gap_peak_sweep([0.3, 0.7], [_multi_band(), _multi_band()],
                                         name="rec.wav", show_rate_dist=True)
    assert len([ln for ln in lines if "band-rate dist" in ln]) == 2


def test_render_human_show_rate_dist_no_peak_note():
    """A no-peak row under --show-rate-dist prints the 'no non-empty bands' note
    rather than percentile rows."""
    lines = gv.render_vad_gap_peak_sweep([0.5], [_all_valley()], name="rec.wav",
                                         show_rate_dist=True)
    note = [ln for ln in lines if "band-rate dist" in ln]
    assert len(note) == 1
    assert "no non-empty bands" in note[0]
    assert not any(ln.strip().startswith("p50:") for ln in lines)


def test_render_human_show_rate_dist_honors_rate_pcts():
    lines = gv.render_vad_gap_peak_sweep([0.3], [_multi_band()], name="rec.wav",
                                         show_rate_dist=True, rate_pcts=[90.0])
    assert any(ln.strip().startswith("p90:") for ln in lines)
    assert not any(ln.strip().startswith("p50:") for ln in lines)


def test_render_json_rows_carry_band_rate_dist():
    """The JSON face ALWAYS carries band_rate_dist per row (no flag), like the
    single-shot render_vad_gap_peak_json."""
    out = gv.render_vad_gap_peak_sweep_json([0.3], [_multi_band()], name="rec.wav")
    payload = json.loads(out)
    assert payload["rate_pcts"] == list(gv.DEFAULT_BAND_RATE_PCTS)
    dist = payload["sweep"][0]["band_rate_dist"]
    assert dist["count"] == 3
    assert [e["p"] for e in dist["percentiles"]] == [50.0, 75.0, 90.0, 99.0]


def test_render_json_echoes_custom_rate_pcts():
    out = gv.render_vad_gap_peak_sweep_json([0.3], [_multi_band()], name="rec.wav",
                                            rate_pcts=[50.0, 99.0])
    payload = json.loads(out)
    assert payload["rate_pcts"] == [50.0, 99.0]
    assert [e["p"] for e in payload["sweep"][0]["band_rate_dist"]["percentiles"]] \
        == [50.0, 99.0]


def test_render_csv_schema_unchanged_no_dist_column():
    """The CSV body is the iter-364 nine-column schema — band_rate_dist is NOT a
    column (the iter-358 verdict-row stance)."""
    out = gv.render_vad_gap_peak_sweep_csv([0.3], [_multi_band()], name="rec.wav")
    header = list(csv.reader(io.StringIO(out)))[0]
    assert header == [
        "threshold", "num_segments", "num_gaps", "peak_found",
        "peak_from_ms", "peak_to_ms", "peak_width_ms", "peak_merged_added",
        "peak_rate_per_100ms",
    ]
    assert "band_rate_dist" not in out


def test_parser_show_rate_dist_and_rate_pcts_defaults():
    args = gv.build_parser().parse_args(["vad-gap-peak-sweep", "rec.wav"])
    assert args.show_rate_dist is False
    assert args.rate_pcts == list(gv.DEFAULT_BAND_RATE_PCTS)


def test_parser_custom_rate_pcts_parsed():
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-sweep", "rec.wav", "--rate-pcts", "50,99"]
    )
    assert args.rate_pcts == [50.0, 99.0]


def test_handler_show_rate_dist_human(monkeypatch):
    import types
    fake = types.ModuleType("vad.silero")
    fake.SileroParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vad.silero", fake)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-sweep", "rec.wav", "--thresholds", "0.5",
         "--show-rate-dist"]
    )
    lines = []
    gv.cmd_vad_gap_peak_sweep(args, log=lines.append,
                              segmenter=_make_segmenter(_multi_band()),
                              availability=_avail_true)
    assert any("band-rate dist" in ln for ln in lines)


def test_handler_human_default_no_dist(monkeypatch):
    import types
    fake = types.ModuleType("vad.silero")
    fake.SileroParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vad.silero", fake)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-sweep", "rec.wav", "--thresholds", "0.5"]
    )
    lines = []
    gv.cmd_vad_gap_peak_sweep(args, log=lines.append,
                              segmenter=_make_segmenter(_multi_band()),
                              availability=_avail_true)
    assert not any("band-rate dist" in ln for ln in lines)


def test_handler_threads_rate_pcts_to_json(monkeypatch):
    import types
    fake = types.ModuleType("vad.silero")
    fake.SileroParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vad.silero", fake)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-sweep", "rec.wav", "--thresholds", "0.5",
         "--rate-pcts", "50,99", "--json"]
    )
    out = []
    gv.cmd_vad_gap_peak_sweep(args, log=out.append,
                              segmenter=_make_segmenter(_multi_band()),
                              availability=_avail_true)
    payload = json.loads(out[0])
    assert payload["rate_pcts"] == [50.0, 99.0]
    assert [e["p"] for e in payload["sweep"][0]["band_rate_dist"]["percentiles"]] \
        == [50.0, 99.0]


def test_handler_unavailable_json_carries_no_dist(monkeypatch):
    """The unavailable JSON degrade path is unchanged (no sweep rows to carry a
    distribution)."""
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-sweep", "rec.wav", "--show-rate-dist", "--json"]
    )
    out = []
    gv.cmd_vad_gap_peak_sweep(args, log=out.append,
                              segmenter=_make_segmenter(_multi_band()),
                              availability=_avail_false)
    payload = json.loads(out[0])
    assert payload["available"] is False
    assert "sweep" not in payload


# ---- iter-368: per-row top-N ranking on the cost-peak sweep -------------
#
# iter-354 added the top-N ranking (the N steepest bands) to the SINGLE-shot
# `gv vad-gap-peak`. iter-368 carries that same view into the SWEEP, one ranking
# per swept value, so an operator watches not just how the single steepest band
# moves but how the WHOLE ranking reorders as a segmenter knob tightens. The
# core + JSON always carry the `peaks` list (machine consumers); --top-n > 1
# gates the human numbered sub-block; the CSV verdict-row schema is unchanged
# (the iter-366 band_rate_dist stance).


def test_core_row_carries_peaks_default_top_n_1():
    """At the default top_n=1 each row's `peaks` holds exactly the single steepest
    band, and its rank-1 entry mirrors the scalar peak_* fields."""
    rows = gv.vad_gap_peak_sweep([0.5], [_multi_band()])
    peaks = rows[0]["peaks"]
    assert len(peaks) == 1
    assert peaks[0]["rank"] == 1
    assert peaks[0]["from_ms"] == rows[0]["peak_from_ms"]
    assert peaks[0]["rate_per_100ms"] == rows[0]["peak_rate_per_100ms"]


def test_core_top_n_ranks_steepest_first():
    """top_n=3 names the three distinct bands ranked by descending rate."""
    rows = gv.vad_gap_peak_sweep([0.5], [_multi_band()], top_n=3)
    peaks = rows[0]["peaks"]
    assert [p["rank"] for p in peaks] == [1, 2, 3]
    rates = [p["rate_per_100ms"] for p in peaks]
    assert rates == sorted(rates, reverse=True)
    assert rates == [0.5, 0.25, 0.125]


def test_core_top_n_caps_at_available_bands():
    """Fewer than top_n entries appear when the range holds fewer non-empty
    bands (here three distinct bands, top_n=5 → three entries)."""
    rows = gv.vad_gap_peak_sweep([0.5], [_multi_band()], top_n=5)
    assert len(rows[0]["peaks"]) == 3


def test_core_peaks_match_single_shot():
    """The per-row ranking equals an independent vad_gap_peak --top-n on the same
    result — the sweep names the SAME ranking the single-shot verdict does."""
    r = _multi_band()
    direct = gv.vad_gap_peak(r, top_n=3)["peaks"]
    row = gv.vad_gap_peak_sweep([0.5], [r], top_n=3)[0]
    assert row["peaks"] == direct


def test_core_no_peak_row_has_empty_peaks():
    """An all-valley row (bands but none non-empty) carries an empty ranking."""
    rows = gv.vad_gap_peak_sweep([0.5], [_all_valley()], top_n=3)
    assert rows[0]["peaks"] == []


def test_core_single_segment_row_has_empty_peaks():
    rows = gv.vad_gap_peak_sweep([0.9], [_single()], top_n=3)
    assert rows[0]["peaks"] == []


def test_render_human_default_top_n_1_omits_ranking():
    """Without --top-n > 1 the human table is byte-for-byte the old shape: no
    'top ... costliest bands' sub-block."""
    lines = gv.render_vad_gap_peak_sweep([0.5], [_multi_band()], name="rec.wav")
    assert not any("costliest bands" in ln for ln in lines)


def test_render_human_top_n_appends_ranking_block():
    lines = gv.render_vad_gap_peak_sweep([0.5], [_multi_band()], name="rec.wav",
                                         top_n=3)
    header = [ln for ln in lines if "costliest bands" in ln]
    assert len(header) == 1
    assert "top 3 costliest bands" in header[0]
    # Three numbered ranked lines follow (one per distinct band).
    numbered = [ln for ln in lines if ln.strip().startswith("#")]
    assert len(numbered) == 3
    assert numbered[0].strip().startswith("#1:")
    assert numbered[2].strip().startswith("#3:")


def test_render_human_top_n_per_row():
    """One ranking block per swept value (here two values)."""
    lines = gv.render_vad_gap_peak_sweep([0.3, 0.7], [_multi_band(), _multi_band()],
                                         name="rec.wav", top_n=2)
    assert len([ln for ln in lines if "costliest bands" in ln]) == 2


def test_render_human_top_n_no_peak_note():
    """A no-peak row under --top-n prints the '(no cost peak)' note rather than
    numbered ranked lines."""
    lines = gv.render_vad_gap_peak_sweep([0.5], [_all_valley()], name="rec.wav",
                                         top_n=3)
    note = [ln for ln in lines if "no cost peak" in ln]
    assert len(note) == 1
    assert not any(ln.strip().startswith("#") for ln in lines)


def test_render_human_top_n_and_rate_dist_both_appear():
    """The ranking and the band-rate-distribution blocks are independent; both
    appear under a row when both are requested, the ranking first."""
    lines = gv.render_vad_gap_peak_sweep([0.5], [_multi_band()], name="rec.wav",
                                         top_n=2, show_rate_dist=True)
    rank_i = next(i for i, ln in enumerate(lines) if "costliest bands" in ln)
    dist_i = next(i for i, ln in enumerate(lines) if "band-rate dist" in ln)
    assert rank_i < dist_i


def test_render_json_rows_carry_peaks_and_top_n():
    """The JSON face ALWAYS carries `peaks` per row + a top-level `top_n`, like
    the single-shot render_vad_gap_peak_json."""
    out = gv.render_vad_gap_peak_sweep_json([0.5], [_multi_band()], name="rec.wav",
                                            top_n=3)
    payload = json.loads(out)
    assert payload["top_n"] == 3
    peaks = payload["sweep"][0]["peaks"]
    assert [p["rank"] for p in peaks] == [1, 2, 3]


def test_render_json_default_top_n_1_carries_single_peak():
    """At the default top_n=1 the JSON still carries the `peaks` list (one entry)
    and top_n=1 — a strict superset of the iter-364/366 shape."""
    out = gv.render_vad_gap_peak_sweep_json([0.5], [_multi_band()], name="rec.wav")
    payload = json.loads(out)
    assert payload["top_n"] == 1
    assert len(payload["sweep"][0]["peaks"]) == 1


def test_render_csv_schema_unchanged_no_peaks_column():
    """The CSV body is the iter-364 nine-column scalar-steepest-band schema —
    `peaks` is NOT a column even with a multi-band result (iter-366 stance)."""
    out = gv.render_vad_gap_peak_sweep_csv([0.5], [_multi_band()], name="rec.wav")
    header = list(csv.reader(io.StringIO(out)))[0]
    assert header == [
        "threshold", "num_segments", "num_gaps", "peak_found",
        "peak_from_ms", "peak_to_ms", "peak_width_ms", "peak_merged_added",
        "peak_rate_per_100ms",
    ]
    assert "peaks" not in out
    assert "rank" not in out


def test_parser_top_n_default():
    args = gv.build_parser().parse_args(["vad-gap-peak-sweep", "rec.wav"])
    assert args.top_n == 1


def test_parser_top_n_custom_parsed():
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-sweep", "rec.wav", "--top-n", "4"]
    )
    assert args.top_n == 4


def test_parser_top_n_rejects_zero():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-gap-peak-sweep", "rec.wav", "--top-n", "0"]
        )


def test_handler_top_n_human(monkeypatch):
    import types
    fake = types.ModuleType("vad.silero")
    fake.SileroParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vad.silero", fake)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-sweep", "rec.wav", "--thresholds", "0.5", "--top-n", "3"]
    )
    lines = []
    gv.cmd_vad_gap_peak_sweep(args, log=lines.append,
                              segmenter=_make_segmenter(_multi_band()),
                              availability=_avail_true)
    assert any("top 3 costliest bands" in ln for ln in lines)


def test_handler_default_top_n_no_ranking(monkeypatch):
    import types
    fake = types.ModuleType("vad.silero")
    fake.SileroParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vad.silero", fake)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-sweep", "rec.wav", "--thresholds", "0.5"]
    )
    lines = []
    gv.cmd_vad_gap_peak_sweep(args, log=lines.append,
                              segmenter=_make_segmenter(_multi_band()),
                              availability=_avail_true)
    assert not any("costliest bands" in ln for ln in lines)


def test_handler_threads_top_n_to_json(monkeypatch):
    import types
    fake = types.ModuleType("vad.silero")
    fake.SileroParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vad.silero", fake)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-sweep", "rec.wav", "--thresholds", "0.5",
         "--top-n", "3", "--json"]
    )
    out = []
    gv.cmd_vad_gap_peak_sweep(args, log=out.append,
                              segmenter=_make_segmenter(_multi_band()),
                              availability=_avail_true)
    payload = json.loads(out[0])
    assert payload["top_n"] == 3
    assert [p["rank"] for p in payload["sweep"][0]["peaks"]] == [1, 2, 3]


def test_handler_top_n_csv_schema_unchanged(monkeypatch):
    """--top-n does not change the CSV (scalar steepest band only)."""
    import types
    fake = types.ModuleType("vad.silero")
    fake.SileroParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vad.silero", fake)
    args = gv.build_parser().parse_args(
        ["vad-gap-peak-sweep", "rec.wav", "--thresholds", "0.5",
         "--top-n", "3", "--csv"]
    )
    out = []
    gv.cmd_vad_gap_peak_sweep(args, log=out.append,
                              segmenter=_make_segmenter(_multi_band()),
                              availability=_avail_true)
    header = list(csv.reader(io.StringIO(out[0])))[0]
    assert "peaks" not in out[0]
    assert header[0] == "threshold"
