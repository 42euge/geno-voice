"""Tests for iter-330 — the ``gv vad-gap-sweep`` subcommand (examples/gv.py).

iter-328 shipped ``gv vad-gaps`` — the inter-segment silence-gap distribution
at ONE knob setting; iter-329 pinned it against the real corpus. This lap adds
its sweep: ``gv vad-gap-sweep`` is the gap-side analogue of ``gv vad-sweep``.
Where ``vad-sweep`` tabulates segment-count / speech-seconds across a swept
knob, ``vad-gap-sweep`` tabulates the min/mean/max gap so an operator can watch
the shortest-pause floor MOVE as the knob tightens — the value that lifts the
min gap clear of a target end-of-turn hangover (``--min-silence-ms`` / the live
``chat.vad.silence_duration``) is the one that buys merge headroom.

Like the rest of the VAD-analysis family, the handler takes injected
``segmenter`` / ``availability`` / ``log`` dependencies so every test runs
WITHOUT importing torch / silero-vad and without touching real audio — fast and
deterministic on the x86_64 Linux runner. The pure core (``vad_gap_sweep``) and
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


# ---- parser: registration & defaults -----------------------------------


def test_vad_gap_sweep_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-gap-sweep"] is gv.cmd_vad_gap_sweep


def test_parser_default_axis_is_thresholds():
    args = gv.build_parser().parse_args(["vad-gap-sweep", "rec.wav"])
    assert args.command == "vad-gap-sweep"
    assert args.thresholds == [0.3, 0.5, 0.7, 0.9]
    # The other axes default to None (not provided).
    assert args.min_silences is None
    assert args.min_speeches is None
    assert args.speech_pads is None
    assert args.max_speeches is None


def test_parser_defaults_mirror_silero_params():
    """The held-fixed scalar knobs default to the same values as ``gv vad``."""
    args = gv.build_parser().parse_args(["vad-gap-sweep", "rec.wav"])
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
            ["vad-gap-sweep", "rec.wav", "--thresholds", "0.3,0.5",
             "--min-silences", "200,400"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["vad-gap-sweep", "rec.wav", "--min-speeches", "50,100",
             "--speech-pads", "0,20"]
        )


def test_parser_json_csv_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-sweep", "rec.wav", "--json", "--csv"])


def test_parser_has_no_target_pick_args():
    """Unlike vad-sweep, the gap sweep has no --target / --top / --tie-break."""
    args = gv.build_parser().parse_args(["vad-gap-sweep", "rec.wav"])
    assert not hasattr(args, "target")
    assert not hasattr(args, "top")
    assert not hasattr(args, "tie_break")


# ---- pure core: vad_gap_sweep -------------------------------------------


def test_core_basic_two_value_threshold_sweep():
    rows = gv.vad_gap_sweep([0.3, 0.9], [_three(), _single()])
    assert rows[0] == {
        "threshold": 0.3,
        "num_segments": 3,
        "num_gaps": 2,
        "min_gap_s": 1.0,
        "mean_gap_s": 1.5,
        "max_gap_s": 2.0,
        "total_silence_s": 3.0,
    }
    # The single-segment row has no pause: aggregates are None, NOT 0.0.
    assert rows[1] == {
        "threshold": 0.9,
        "num_segments": 1,
        "num_gaps": 0,
        "min_gap_s": None,
        "mean_gap_s": None,
        "max_gap_s": None,
        "total_silence_s": 0.0,
    }


def test_core_axis_key_follows_axis_arg():
    rows = gv.vad_gap_sweep([200.0, 800.0], [_three(), _three()],
                            axis="min_silence_ms")
    assert "min_silence_ms" in rows[0]
    assert "threshold" not in rows[0]
    assert rows[0]["min_silence_ms"] == 200.0


def test_core_aggregates_match_vad_silence_gaps():
    """Each row's aggregates equal an independent vad_silence_gaps on its
    result — the sweep differences the SAME segmentation gap core does."""
    r = _three()
    direct = gv.vad_silence_gaps(r)
    row = gv.vad_gap_sweep([0.5], [r])[0]
    for key in ("num_segments", "num_gaps", "min_gap_s", "mean_gap_s",
                "max_gap_s", "total_silence_s"):
        assert row[key] == direct[key]


def test_core_min_gap_tracks_shortest_pause():
    rows = gv.vad_gap_sweep([0.5], [_three()])
    assert rows[0]["min_gap_s"] == 1.0  # the 1.0s pause, not the 2.0s one


def test_core_length_mismatch_raises():
    with pytest.raises(ValueError):
        gv.vad_gap_sweep([0.3, 0.5], [_three()])


def test_core_empty_sweep_is_empty():
    assert gv.vad_gap_sweep([], []) == []


# ---- renderer: render_vad_gap_sweep (human) -----------------------------


def test_render_human_header_and_rows():
    lines = gv.render_vad_gap_sweep([0.3, 0.9], [_three(), _single()],
                                    name="rec.wav")
    text = "\n".join(lines)
    assert lines[0] == "silero VAD gap sweep — rec.wav"
    # The header names the gap columns.
    assert "min_gap" in lines[1]
    assert "mean_gap" in lines[1]
    assert "max_gap" in lines[1]
    # The 3-segment row shows the numeric gaps; the single-segment row shows -.
    assert "1.000" in text  # min gap of the 3-segment row
    assert "  -  " in text or "-" in lines[-1]


def test_render_human_single_segment_row_dashes_aggregates():
    lines = gv.render_vad_gap_sweep([0.9], [_single()], name="rec.wav")
    # The lone data row prints dashes, never a fake 0.000 gap.
    data_row = lines[-1]
    assert "0.000" not in data_row
    assert "-" in data_row


def test_render_human_axis_label_for_min_silence():
    lines = gv.render_vad_gap_sweep([200.0, 800.0], [_three(), _three()],
                                    name="rec.wav", axis="min_silence_ms")
    assert "min_silence" in lines[1]


def test_render_human_unavailable():
    lines = gv.render_vad_gap_sweep([], [None], name="rec.wav")
    assert len(lines) == 1
    assert lines[0].startswith("silero VAD unavailable")


# ---- renderer: render_vad_gap_sweep_json --------------------------------


def test_render_json_shape():
    text = gv.render_vad_gap_sweep_json([0.3, 0.9], [_three(), _single()],
                                        name="rec.wav")
    payload = json.loads(text)
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["axis"] == "threshold"
    assert len(payload["sweep"]) == 2
    assert payload["sweep"][0]["num_gaps"] == 2
    assert payload["sweep"][0]["min_gap_s"] == 1.0


def test_render_json_single_segment_aggregates_null():
    text = gv.render_vad_gap_sweep_json([0.9], [_single()], name="rec.wav")
    row = json.loads(text)["sweep"][0]
    assert row["num_gaps"] == 0
    assert row["min_gap_s"] is None
    assert row["mean_gap_s"] is None
    assert row["max_gap_s"] is None
    assert row["total_silence_s"] == 0.0


def test_render_json_axis_name_carried():
    text = gv.render_vad_gap_sweep_json([200.0], [_three()], name="rec.wav",
                                        axis="min_silence_ms")
    payload = json.loads(text)
    assert payload["axis"] == "min_silence_ms"
    assert "min_silence_ms" in payload["sweep"][0]


def test_render_json_unavailable():
    payload = json.loads(gv.render_vad_gap_sweep_json([], [None], name="rec.wav"))
    assert payload["available"] is False
    assert "hint" in payload


# ---- renderer: render_vad_gap_sweep_csv ---------------------------------


def test_render_csv_header_and_rows():
    text = gv.render_vad_gap_sweep_csv([0.3, 0.9], [_three(), _single()],
                                       name="rec.wav")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == [
        "threshold", "num_segments", "num_gaps",
        "min_gap_s", "mean_gap_s", "max_gap_s", "total_silence_s",
    ]
    assert rows[1] == ["0.3", "3", "2", "1.0", "1.5", "2.0", "3.0"]


def test_render_csv_single_segment_empty_aggregate_cells():
    text = gv.render_vad_gap_sweep_csv([0.9], [_single()], name="rec.wav")
    rows = list(csv.reader(io.StringIO(text)))
    # The aggregate columns are empty (the CSV spelling of JSON null), not 0.0.
    assert rows[1] == ["0.9", "1", "0", "", "", "", "0"]


def test_render_csv_axis_header():
    text = gv.render_vad_gap_sweep_csv([200.0], [_three()], name="rec.wav",
                                       axis="min_silence_ms")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0][0] == "min_silence_ms"


def test_render_csv_unavailable():
    text = gv.render_vad_gap_sweep_csv([], [None], name="rec.wav")
    assert text.startswith("# silero VAD unavailable")


# ---- handler: cmd_vad_gap_sweep -----------------------------------------


def _run_handler(results, argv_extra=None, segmenter=None):
    """Drive cmd_vad_gap_sweep with an injected segmenter returning ``results``
    in order (one per swept value)."""
    lines: List[str] = []
    argv = ["vad-gap-sweep", "rec.wav", *(argv_extra or [])]
    args = gv.build_parser().parse_args(argv)
    it = iter(results)
    if segmenter is None:
        segmenter = lambda wav, params=None: next(it)  # noqa: E731
    gv.cmd_vad_gap_sweep(
        args, log=lines.append, segmenter=segmenter, availability=lambda: True
    )
    return lines


def test_cmd_human_path():
    lines = _run_handler(
        [_three(), _three(), _three(), _single()],  # 4 default thresholds
    )
    text = "\n".join(lines)
    assert "silero VAD gap sweep" in text
    assert "min_gap" in text


def test_cmd_json_path():
    lines = _run_handler([_three(), _single()],
                         argv_extra=["--thresholds", "0.3,0.9", "--json"])
    payload = json.loads("\n".join(lines))
    assert payload["available"] is True
    assert len(payload["sweep"]) == 2
    assert payload["sweep"][0]["num_gaps"] == 2


def test_cmd_csv_path():
    lines = _run_handler([_three(), _single()],
                         argv_extra=["--thresholds", "0.3,0.9", "--csv"])
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    assert rows[0][0] == "threshold"
    assert rows[0][2] == "num_gaps"


def test_cmd_min_silences_axis_switches_swept_dimension():
    seen = []

    def _seg(wav, params=None):
        seen.append(params.min_silence_ms)
        return _three()

    lines = _run_handler(
        [None],  # unused; _seg returns directly
        argv_extra=["--min-silences", "200,800", "--threshold", "0.7", "--json"],
        segmenter=_seg,
    )
    # The hangover was swept (200 then 800); the gate held fixed at 0.7.
    assert seen == [200.0, 800.0]
    payload = json.loads("\n".join(lines))
    assert payload["axis"] == "min_silence_ms"
    assert [r["min_silence_ms"] for r in payload["sweep"]] == [200.0, 800.0]


def test_cmd_threshold_axis_holds_other_knobs_fixed():
    captured = []

    def _seg(wav, params=None):
        captured.append((params.threshold, params.min_silence_ms))
        return _three()

    _run_handler(
        [None],
        argv_extra=["--thresholds", "0.3,0.9", "--min-silence-ms", "500"],
        segmenter=_seg,
    )
    # The gate is swept; the hangover is held at its scalar for every run.
    assert captured == [(0.3, 500.0), (0.9, 500.0)]


def test_cmd_max_speeches_seconds_axis():
    seen = []

    def _seg(wav, params=None):
        seen.append(params.max_speech_s)
        return _three()

    lines = _run_handler(
        [None],
        argv_extra=["--max-speeches", "5,inf", "--json"],
        segmenter=_seg,
    )
    assert seen[0] == 5.0
    assert seen[1] == float("inf")
    payload = json.loads("\n".join(lines))
    assert payload["axis"] == "max_speech_s"


def test_cmd_unavailable_human():
    lines: List[str] = []
    args = gv.build_parser().parse_args(["vad-gap-sweep", "rec.wav"])
    gv.cmd_vad_gap_sweep(
        args,
        log=lines.append,
        segmenter=lambda wav, params=None: pytest.fail("must not segment"),
        availability=lambda: False,
    )
    assert any("silero VAD unavailable" in ln for ln in lines)


def test_cmd_unavailable_json():
    lines: List[str] = []
    args = gv.build_parser().parse_args(["vad-gap-sweep", "rec.wav", "--json"])
    gv.cmd_vad_gap_sweep(
        args,
        log=lines.append,
        segmenter=lambda wav, params=None: pytest.fail("must not segment"),
        availability=lambda: False,
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is False


def test_cmd_unavailable_csv():
    lines: List[str] = []
    args = gv.build_parser().parse_args(["vad-gap-sweep", "rec.wav", "--csv"])
    gv.cmd_vad_gap_sweep(
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
        argv_extra=["--thresholds", "0.5", "--json"],
    )
    payload = json.loads("\n".join(lines))
    assert payload["name"] == "clean.wav"
