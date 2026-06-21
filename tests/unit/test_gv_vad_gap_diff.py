"""Tests for iter-334 — the ``gv vad-gap-diff`` subcommand (examples/gv.py).

iter-328 shipped ``gv vad-gaps`` — the inter-segment silence-gap distribution
at ONE knob setting; iter-330 added its 1-D sweep and iter-332 its 2-D grid.
This lap adds the two-point comparison: ``gv vad-gap-diff`` is the gap-side
analogue of ``gv vad-diff`` (iter-235) and the two-point degenerate of
``gv vad-gap-sweep``. Where ``vad-diff`` reports the segment-count /
speech-seconds delta between two ``--threshold`` settings, ``vad-gap-diff``
reports how the SILENCE-gap distribution shifts — the min/mean/max gap delta —
so an operator can watch whether a stricter gate lifts the shortest pause clear
of a target end-of-turn hangover (``--min-silence-ms`` / the live
``chat.vad.silence_duration``), buying merge headroom.

Like the rest of the VAD-analysis family, the handler takes injected
``segmenter`` / ``availability`` / ``log`` dependencies so every test runs
WITHOUT importing torch / silero-vad and without touching real audio — fast and
deterministic on the x86_64 Linux runner. The pure core (``vad_gap_delta``) and
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


# Recurring stand-ins:
#   _three : 3 segments → 2 gaps (1.0s and 2.0s), min 1.0 mean 1.5 max 2.0
#   _two   : 2 segments → 1 gap (3.0s)
#   _single: 1 segment  → no inter-segment pause (aggregates None)
def _three():
    return _result((0.0, 1.0), (2.0, 3.0), (5.0, 6.0))


def _two():
    return _result((0.0, 1.0), (4.0, 6.0))


def _single():
    return _result((0.0, 6.0))


# ---- parser: registration & defaults -----------------------------------


def test_vad_gap_diff_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-gap-diff"] is gv.cmd_vad_gap_diff


def test_parser_defaults_match_vad_diff():
    """The two-threshold + held-scalar knobs mirror ``gv vad-diff`` exactly."""
    args = gv.build_parser().parse_args(["vad-gap-diff", "rec.wav"])
    diff = gv.build_parser().parse_args(["vad-diff", "rec.wav"])
    assert args.command == "vad-gap-diff"
    assert args.threshold_a == diff.threshold_a
    assert args.threshold_b == diff.threshold_b == 0.7
    assert args.min_speech_ms == diff.min_speech_ms
    assert args.min_silence_ms == diff.min_silence_ms
    assert args.speech_pad_ms == diff.speech_pad_ms
    assert args.max_speech_s == diff.max_speech_s


def test_parser_accepts_explicit_thresholds():
    args = gv.build_parser().parse_args(
        ["vad-gap-diff", "rec.wav", "--threshold-a", "0.4", "--threshold-b", "0.8"]
    )
    assert args.threshold_a == 0.4
    assert args.threshold_b == 0.8


def test_parser_json_csv_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-diff", "rec.wav", "--json", "--csv"])


def test_parser_rejects_out_of_range_threshold():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-diff", "rec.wav", "--threshold-a", "1.5"])


# ---- pure core: vad_gap_delta ------------------------------------------


def test_gap_delta_both_sides_and_signed_deltas():
    d = gv.vad_gap_delta(_three(), _two())
    # counts
    assert d["num_segments_a"] == 3 and d["num_segments_b"] == 2
    assert d["num_segments_delta"] == -1
    assert d["num_gaps_a"] == 2 and d["num_gaps_b"] == 1
    assert d["num_gaps_delta"] == -1
    # aggregates: A has gaps {1.0, 2.0}; B has {3.0}
    assert d["min_gap_s_a"] == 1.0 and d["min_gap_s_b"] == 3.0
    assert d["min_gap_s_delta"] == 2.0
    assert d["mean_gap_s_a"] == 1.5 and d["mean_gap_s_b"] == 3.0
    assert d["mean_gap_s_delta"] == 1.5
    assert d["max_gap_s_a"] == 2.0 and d["max_gap_s_b"] == 3.0
    assert d["max_gap_s_delta"] == 1.0
    assert d["total_silence_s_a"] == 3.0 and d["total_silence_s_b"] == 3.0
    assert d["total_silence_s_delta"] == 0.0


def test_gap_delta_anchors_each_side_to_vad_silence_gaps():
    """Each side of the diff equals an independent ``vad_silence_gaps`` — the
    diff is exactly the difference of two standalone gap reports."""
    a, b = _three(), _two()
    ga, gb = gv.vad_silence_gaps(a), gv.vad_silence_gaps(b)
    d = gv.vad_gap_delta(a, b)
    for side, g in (("a", ga), ("b", gb)):
        assert d[f"num_segments_{side}"] == g["num_segments"]
        assert d[f"num_gaps_{side}"] == g["num_gaps"]
        assert d[f"min_gap_s_{side}"] == g["min_gap_s"]
        assert d[f"mean_gap_s_{side}"] == g["mean_gap_s"]
        assert d[f"max_gap_s_{side}"] == g["max_gap_s"]
        assert d[f"total_silence_s_{side}"] == g["total_silence_s"]


def test_gap_delta_missing_pause_yields_none_delta():
    """A side with <2 segments has None aggregates, so the aggregate deltas are
    None (a missing pause cannot be differenced) — but counts and total_silence
    (always a float) still difference."""
    d = gv.vad_gap_delta(_three(), _single())
    assert d["num_gaps_a"] == 2 and d["num_gaps_b"] == 0
    assert d["num_gaps_delta"] == -2
    assert d["min_gap_s_b"] is None
    assert d["min_gap_s_delta"] is None
    assert d["mean_gap_s_delta"] is None
    assert d["max_gap_s_delta"] is None
    # total_silence is 0.0 (not None) for a single-segment side, so it still
    # differences: 0.0 - 3.0 = -3.0.
    assert d["total_silence_s_b"] == 0.0
    assert d["total_silence_s_delta"] == -3.0


def test_gap_delta_both_sides_missing_pause():
    d = gv.vad_gap_delta(_single(), _single())
    assert d["num_gaps_delta"] == 0
    assert d["min_gap_s_delta"] is None
    assert d["mean_gap_s_delta"] is None
    assert d["max_gap_s_delta"] is None
    assert d["total_silence_s_delta"] == 0.0


def test_gap_delta_rounds_to_three_places():
    a = _result((0.0, 1.0), (1.333, 2.0))  # gap 0.333
    b = _result((0.0, 1.0), (1.111, 2.0))  # gap 0.111
    d = gv.vad_gap_delta(a, b)
    # 0.111 - 0.333 = -0.222 exactly at 3 places.
    assert d["min_gap_s_delta"] == -0.222


# ---- human renderer -----------------------------------------------------


def test_render_human_shape_and_deltas():
    lines = gv.render_vad_gap_diff(_three(), _two(), label_a=0.5, label_b=0.7)
    assert lines[0] == "silero VAD gap diff — rec.wav"
    assert any("threshold A:  0.50" in ln for ln in lines)
    assert any("threshold B:  0.70" in ln for ln in lines)
    assert any("segments:     3 → 2 (-1)" in ln for ln in lines)
    assert any("gaps:         2 → 1 (-1)" in ln for ln in lines)
    # min gap rose 1.0 → 3.0, signed at 3 places, naming the actionable knob.
    min_line = next(ln for ln in lines if "min gap:" in ln)
    assert "1.000s → 3.000s (+2.000s)" in min_line
    assert "--min-silence-ms" in min_line
    assert any("mean gap:" in ln and "(+1.500s)" in ln for ln in lines)
    assert any("max gap:" in ln and "(+1.000s)" in ln for ln in lines)


def test_render_human_missing_pause_prints_dash_and_na():
    lines = gv.render_vad_gap_diff(_three(), _single(), label_a=0.5, label_b=0.9)
    min_line = next(ln for ln in lines if "min gap:" in ln)
    # B has no pause: "-" for the value, "n/a" for the delta.
    assert "1.000s → - (n/a)" in min_line
    # total_silence still differences (0.0 is a real value, not a missing pause).
    total_line = next(ln for ln in lines if "total silence" in ln)
    assert "3.000s → 0.000s (-3.000s)" in total_line


def test_render_human_negative_delta_keeps_sign():
    """A looser gate that shortens the min gap shows a negative signed delta."""
    lines = gv.render_vad_gap_diff(_two(), _three(), label_a=0.7, label_b=0.5)
    min_line = next(ln for ln in lines if "min gap:" in ln)
    assert "3.000s → 1.000s (-2.000s)" in min_line


def test_render_human_unavailable():
    for pair in ((None, _two()), (_three(), None), (None, None)):
        lines = gv.render_vad_gap_diff(*pair, label_a=0.5, label_b=0.7)
        assert len(lines) == 1
        assert "silero VAD unavailable" in lines[0]


# ---- json renderer ------------------------------------------------------


def test_render_json_payload():
    out = gv.render_vad_gap_diff_json(_three(), _two(), label_a=0.5, label_b=0.7)
    payload = json.loads(out)
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["threshold_a"] == 0.5
    assert payload["threshold_b"] == 0.7
    # Carries every key vad_gap_delta returns.
    expected = gv.vad_gap_delta(_three(), _two())
    for key, value in expected.items():
        assert payload[key] == value


def test_render_json_null_delta_for_missing_pause():
    out = gv.render_vad_gap_diff_json(_three(), _single(), label_a=0.5, label_b=0.9)
    payload = json.loads(out)
    assert payload["min_gap_s_b"] is None
    assert payload["min_gap_s_delta"] is None
    assert payload["mean_gap_s_delta"] is None
    assert payload["max_gap_s_delta"] is None
    assert payload["total_silence_s_delta"] == -3.0


def test_render_json_unavailable():
    out = gv.render_vad_gap_diff_json(None, None, label_a=0.5, label_b=0.7)
    payload = json.loads(out)
    assert payload["available"] is False
    assert "install 'silero-vad'" in payload["hint"]


# ---- csv renderer -------------------------------------------------------


def test_render_csv_two_rows_and_header():
    out = gv.render_vad_gap_diff_csv(_three(), _two(), label_a=0.5, label_b=0.7)
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0] == [
        "threshold",
        "num_segments",
        "num_gaps",
        "min_gap_s",
        "mean_gap_s",
        "max_gap_s",
        "total_silence_s",
    ]
    assert rows[1] == ["0.5", "3", "2", "1.0", "1.5", "2.0", "3.0"]
    assert rows[2] == ["0.7", "2", "1", "3.0", "3.0", "3.0", "3.0"]


def test_render_csv_missing_pause_empty_cells():
    out = gv.render_vad_gap_diff_csv(_three(), _single(), label_a=0.5, label_b=0.9)
    rows = list(csv.reader(io.StringIO(out)))
    # The single-segment B row has empty aggregate cells (CSV spelling of null).
    assert rows[2] == ["0.9", "1", "0", "", "", "", "0"]


def test_render_csv_byte_identical_to_two_value_gap_sweep():
    """A gap diff IS the two-point degenerate of a gap sweep, so a
    ``vad-gap-diff --csv`` and a two-value ``vad-gap-sweep --csv`` over the same
    (threshold) pair must be byte-identical — the same contract iter-313 pins
    between ``vad-diff --csv`` and ``vad-sweep --csv``."""
    a, b = _three(), _two()
    diff_csv = gv.render_vad_gap_diff_csv(a, b, label_a=0.5, label_b=0.7)
    sweep_csv = gv.render_vad_gap_sweep_csv([0.5, 0.7], [a, b], name="rec.wav")
    assert diff_csv == sweep_csv


def test_render_csv_unavailable():
    out = gv.render_vad_gap_diff_csv(None, None, label_a=0.5, label_b=0.7)
    assert out.startswith("# silero VAD unavailable")
    assert "\n" not in out


# ---- handler: cmd_vad_gap_diff (injected deps) -------------------------


class _Args:
    """Minimal argparse.Namespace stand-in for the handler."""

    def __init__(self, **kw):
        self.wav = "rec.wav"
        self.threshold_a = 0.5
        self.threshold_b = 0.7
        self.min_speech_ms = 250.0
        self.min_silence_ms = 800.0
        self.speech_pad_ms = 30.0
        self.max_speech_s = float("inf")
        self.json = False
        self.csv = False
        for k, v in kw.items():
            setattr(self, k, v)


def _capture_handler(args, results_by_threshold):
    """Run cmd_vad_gap_diff with a stub segmenter keyed on threshold."""
    lines: List[str] = []
    calls: List[float] = []

    def segmenter(wav, *, params):
        calls.append(params.threshold)
        return results_by_threshold[params.threshold]

    gv.cmd_vad_gap_diff(
        args,
        log=lines.append,
        segmenter=segmenter,
        availability=lambda: True,
    )
    return lines, calls


def test_handler_human_output():
    args = _Args()
    lines, calls = _capture_handler(
        args, {0.5: _three(), 0.7: _two()}
    )
    # Segments A then B, both thresholds.
    assert calls == [0.5, 0.7]
    out = "\n".join(lines)
    assert "silero VAD gap diff — rec.wav" in out
    assert "segments:     3 → 2 (-1)" in out
    assert "min gap:" in out and "(+2.000s)" in out


def test_handler_holds_other_knobs_across_runs():
    """Every non-threshold knob is shared between the two runs."""
    captured = []

    def segmenter(wav, *, params):
        captured.append(params)
        return _two() if params.threshold == 0.7 else _three()

    args = _Args(min_silence_ms=600.0, min_speech_ms=100.0, speech_pad_ms=40.0)
    gv.cmd_vad_gap_diff(
        args, log=lambda _l: None, segmenter=segmenter, availability=lambda: True
    )
    assert len(captured) == 2
    for p in captured:
        assert p.min_silence_ms == 600.0
        assert p.min_speech_ms == 100.0
        assert p.speech_pad_ms == 40.0
    # Only the threshold differs between the runs.
    assert {p.threshold for p in captured} == {0.5, 0.7}


def test_handler_json_output():
    args = _Args(json=True)
    lines, _ = _capture_handler(args, {0.5: _three(), 0.7: _two()})
    payload = json.loads("\n".join(lines))
    assert payload["available"] is True
    assert payload["threshold_a"] == 0.5
    assert payload["min_gap_s_delta"] == 2.0


def test_handler_csv_output():
    args = _Args(csv=True)
    lines, _ = _capture_handler(args, {0.5: _three(), 0.7: _two()})
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    assert rows[0][0] == "threshold"
    assert rows[1][0] == "0.5"
    assert rows[2][0] == "0.7"


def test_handler_uses_result_name_not_raw_path():
    """The report names the segmenter's own basename, not the raw path arg."""
    args = _Args(wav="/abs/path/to/recording.wav")
    lines, _ = _capture_handler(
        args,
        {
            0.5: _result((0.0, 1.0), (2.0, 3.0), name="recording.wav"),
            0.7: _result((0.0, 3.0), name="recording.wav"),
        },
    )
    assert lines[0] == "silero VAD gap diff — recording.wav"


def test_handler_unavailable_human():
    lines: List[str] = []
    gv.cmd_vad_gap_diff(
        _Args(),
        log=lines.append,
        segmenter=lambda *a, **k: pytest.fail("segmenter must not be called"),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert "silero VAD unavailable" in lines[0]


def test_handler_unavailable_json():
    lines: List[str] = []
    gv.cmd_vad_gap_diff(
        _Args(json=True),
        log=lines.append,
        segmenter=lambda *a, **k: pytest.fail("segmenter must not be called"),
        availability=lambda: False,
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is False


def test_handler_unavailable_csv():
    lines: List[str] = []
    gv.cmd_vad_gap_diff(
        _Args(csv=True),
        log=lines.append,
        segmenter=lambda *a, **k: pytest.fail("segmenter must not be called"),
        availability=lambda: False,
    )
    assert lines[0].startswith("# silero VAD unavailable")
