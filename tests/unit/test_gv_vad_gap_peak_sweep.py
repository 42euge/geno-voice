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
    }
    # The single-segment row has no pause: no peak, fields None / False.
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
