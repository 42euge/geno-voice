"""Tests for iter-350 — the ``gv vad-gap-peak`` subcommand (examples/gv.py).

``gv vad-gap-cost`` (iter-349) reports the full marginal merge cost curve — for
every band between consecutive ``--cuts-ms`` values, how many ADDITIONAL pauses
merge and at what rate per +100 ms of hangover. ``gv vad-gap-peak`` is its
VERDICT companion: it reads that curve and names the single STEEPEST band — the
densest pause cluster, the steepest part of the CDF, the most expensive place to
raise ``--min-silence-ms``. It is the mirror image of ``gv vad-gap-recommend``
(which points at the cheapest zero-rate valley): peak says where NOT to cut,
recommend says where TO cut.

Like the rest of the VAD-analysis family, the handler takes injected
``segmenter`` / ``availability`` / ``log`` dependencies so every test runs
WITHOUT importing torch / silero-vad and without touching real audio — fast and
deterministic on the x86_64 Linux runner. The pure core (``vad_gap_peak``) and
the three renderers are exercised directly against lightweight stand-ins
mirroring just the ``SileroResult`` / ``SpeechSegment`` attributes they read.
"""

from __future__ import annotations

import csv
import io
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
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


# ---- parser: registration & defaults -----------------------------------


def test_vad_gap_peak_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-gap-peak"] is gv.cmd_vad_gap_peak


def test_parser_registers_vad_gap_peak():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-peak", "rec.wav"])
    assert args.command == "vad-gap-peak"
    assert args.wav == "rec.wav"


def test_parser_defaults_mirror_vad_gaps_knobs():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-peak", "rec.wav"])
    # Shares the gv vad segmenter knobs.
    assert args.threshold == pytest.approx(0.5)
    assert args.min_speech_ms == pytest.approx(250.0)
    assert args.min_silence_ms == pytest.approx(800.0)
    assert args.speech_pad_ms == pytest.approx(30.0)
    assert math.isinf(args.max_speech_s)
    # The cuts default (reuses vad-gap-cost's / vad-gap-cdf's cuts list).
    assert args.cuts_ms == [200.0, 400.0, 800.0, 1600.0]
    assert args.json is False
    assert args.csv is False


def test_parser_accepts_custom_cuts():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-peak", "rec.wav", "--cuts-ms", "100,500,1000"])
    assert args.cuts_ms == [100.0, 500.0, 1000.0]


def test_parser_rejects_bad_cuts():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-peak", "rec.wav", "--cuts-ms", "-5"])
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-peak", "rec.wav", "--cuts-ms", "nan"])


def test_parser_json_csv_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-peak", "rec.wav", "--json", "--csv"])


def test_parser_rejects_out_of_range_threshold():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-peak", "rec.wav", "--threshold", "1.5"])


# ---- pure core: vad_gap_peak --------------------------------------------


def test_peak_names_highest_rate_band():
    # Gaps (sorted): 1.0, 2.0, 4.0, 6.0 seconds.
    # cuts 500/2500/3500/5000 -> bands:
    #   500-2500 : {1.0, 2.0} -> +2, rate 0.100  <- the peak
    #   2500-3500: {}        -> +0, rate 0.000
    #   3500-5000: {4.0}     -> +1, rate 0.067
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    p = gv.vad_gap_peak(res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0])
    assert p["num_segments"] == 5
    assert p["num_gaps"] == 4
    assert p["num_bands"] == 3
    assert p["peak_found"] is True
    assert p["peak_from_ms"] == 500.0
    assert p["peak_to_ms"] == 2500.0
    assert p["peak_width_ms"] == 2000.0
    assert p["peak_merged_added"] == 2
    assert p["peak_rate_per_100ms"] == pytest.approx(0.1)


def test_peak_agrees_with_vad_gap_cost_band():
    # The named peak is literally the band of vad_gap_cost with the highest rate.
    res = _result((0, 1), (2, 3), (6, 7), (12, 13), (20, 21))
    cuts = [200.0, 1500.0, 3000.0, 5000.0, 9000.0]
    p = gv.vad_gap_peak(res, cuts_ms=cuts)
    c = gv.vad_gap_cost(res, cuts_ms=cuts)
    best = max(c["bands"], key=lambda b: b["rate_per_100ms"])
    assert p["peak_from_ms"] == best["from_ms"]
    assert p["peak_to_ms"] == best["to_ms"]
    assert p["peak_rate_per_100ms"] == best["rate_per_100ms"]
    assert p["peak_merged_added"] == best["merged_added"]


def test_peak_high_rate_band_inside_cluster():
    # A narrow band packed with pauses is the costliest.
    res = _result((0, 1), (1.5, 2.5), (3.0, 4.0), (4.5, 5.5))  # gaps 0.5,0.5,0.5
    p = gv.vad_gap_peak(res, cuts_ms=[400.0, 600.0, 2000.0])
    # [0.4, 0.6) holds all three 0.5s gaps; width 200ms -> 3/200*100 = 1.5.
    assert p["peak_found"] is True
    assert p["peak_from_ms"] == 400.0
    assert p["peak_to_ms"] == 600.0
    assert p["peak_rate_per_100ms"] == pytest.approx(1.5)
    assert p["peak_merged_added"] == 3


def test_peak_is_mirror_of_recommend_valley():
    # vad_gap_recommend points at the zero-rate valley; vad_gap_peak points at a
    # high-rate cluster. The recommended hangover must NOT fall inside the peak
    # band (they name opposite features of the same curve).
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    cuts = [500.0, 2500.0, 3500.0, 5000.0]
    p = gv.vad_gap_peak(res, cuts_ms=cuts)
    rec = gv.vad_gap_recommend(res)
    assert p["peak_found"] is True
    rec_ms = rec["recommended_ms"]
    assert not (p["peak_from_ms"] <= rec_ms < p["peak_to_ms"])


def test_peak_earliest_tie_wins():
    # Two bands with the SAME rate: the first (lower cut) band wins.
    # Gaps 0.5 and 1.5; cuts 0/1000/2000 -> [0,1): {0.5} +1 width 1000 rate 0.1;
    # [1000,2000): {1.5} +1 width 1000 rate 0.1. Equal rate -> earliest wins.
    res = _result((0, 1), (1.5, 2.0), (3.5, 4.0))  # gaps 0.5, 1.5
    p = gv.vad_gap_peak(res, cuts_ms=[0.0, 1000.0, 2000.0])
    assert p["peak_from_ms"] == 0.0
    assert p["peak_to_ms"] == 1000.0
    assert p["peak_rate_per_100ms"] == pytest.approx(0.1)


def test_peak_boundary_is_strict_less_than():
    # A pause EXACTLY at a cut is kept, not merged: it counts in the band ABOVE.
    res = _result((0, 1), (3, 4))  # one gap of exactly 2.0s
    p = gv.vad_gap_peak(res, cuts_ms=[1000.0, 2000.0, 3000.0])
    # [1.0,2.0): 2.0 not < 2.0 -> +0; [2.0,3.0): 2.0 < 3.0 -> +1. Peak is band 2.
    assert p["peak_from_ms"] == 2000.0
    assert p["peak_to_ms"] == 3000.0
    assert p["peak_merged_added"] == 1


def test_peak_anchors_to_vad_silence_gaps():
    res = _result((0, 1), (2, 3), (6, 7), (15, 16))
    p = gv.vad_gap_peak(res, cuts_ms=[800.0, 1600.0])
    d = gv.vad_silence_gaps(res)
    assert p["num_segments"] == d["num_segments"]
    assert p["num_gaps"] == d["num_gaps"]
    assert p["min_gap_s"] == d["min_gap_s"]
    assert p["max_gap_s"] == d["max_gap_s"]
    assert p["mean_gap_s"] == d["mean_gap_s"]
    assert p["total_silence_s"] == d["total_silence_s"]


def test_peak_all_valley_range_has_no_peak():
    # Every band empty (no cluster in the scanned range) -> no peak to name.
    res = _result((0, 1), (5, 6))  # one gap of 4.0s, outside all bands
    p = gv.vad_gap_peak(res, cuts_ms=[1000.0, 2000.0, 3000.0])
    assert p["num_bands"] == 2
    assert p["peak_found"] is False
    assert p["peak_from_ms"] is None
    assert p["peak_rate_per_100ms"] is None


def test_peak_empty_for_fewer_than_two_segments():
    res = _result((0, 1))
    p = gv.vad_gap_peak(res)
    assert p["num_gaps"] == 0
    assert p["num_bands"] == 0
    assert p["peak_found"] is False
    assert p["min_gap_s"] is None


def test_peak_empty_for_zero_segments():
    res = _result()
    p = gv.vad_gap_peak(res)
    assert p["num_segments"] == 0
    assert p["num_bands"] == 0
    assert p["peak_found"] is False


def test_peak_no_band_for_single_distinct_cut():
    # A single distinct cut forms no band (a degenerate axis) -> nothing to name.
    res = _result((0, 1), (2, 3), (5, 6))
    p = gv.vad_gap_peak(res, cuts_ms=[800.0])
    assert p["num_gaps"] == 2
    assert p["num_bands"] == 0
    assert p["peak_found"] is False


def test_peak_no_band_for_duplicate_cuts():
    res = _result((0, 1), (2, 3), (5, 6))
    p = gv.vad_gap_peak(res, cuts_ms=[800.0, 800.0, 800.0])
    assert p["num_bands"] == 0
    assert p["peak_found"] is False


def test_peak_cuts_sorted_and_deduplicated():
    # Like vad_gap_cost, the peak needs a monotone axis: unsorted/dup collapse.
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    p = gv.vad_gap_peak(res, cuts_ms=[3000.0, 500.0, 3000.0, 1500.0])
    # Bands [500,1500) {1.0} +1 rate 0.1; [1500,3000) {2.0} +1 rate 0.067.
    # Peak is the first (higher rate).
    assert p["peak_from_ms"] == 500.0
    assert p["peak_to_ms"] == 1500.0


def test_peak_rate_rounded_to_three_places():
    # single gap 4.0 in [3.5, 5.0): 1 / 1500 * 100 = 0.0666… -> 0.067.
    res = _result((0, 1), (5, 6))  # one gap of 4.0s
    p = gv.vad_gap_peak(res, cuts_ms=[3500.0, 5000.0])
    assert p["peak_rate_per_100ms"] == 0.067


def test_peak_default_cuts():
    # Gaps are 1.0s and 2.0s. Default cuts 200/400/800/1600 -> 3 bands; the 1.0s
    # gap (1000ms) falls in [800, 1600), so that band is the peak.
    res = _result((0, 1), (2, 3), (5, 6))
    p = gv.vad_gap_peak(res)
    assert p["num_bands"] == 3
    assert p["peak_found"] is True
    assert p["peak_from_ms"] == 800.0
    assert p["peak_to_ms"] == 1600.0
    assert p["peak_merged_added"] == 1


def test_peak_rejects_empty_cuts():
    res = _result((0, 1), (2, 3))
    with pytest.raises(ValueError):
        gv.vad_gap_peak(res, cuts_ms=[])


def test_peak_rejects_negative_cut():
    res = _result((0, 1), (2, 3))
    with pytest.raises(ValueError):
        gv.vad_gap_peak(res, cuts_ms=[-1.0])


def test_peak_rejects_nan_cut():
    res = _result((0, 1), (2, 3))
    with pytest.raises(ValueError):
        gv.vad_gap_peak(res, cuts_ms=[float("nan")])


def test_peak_handles_unsorted_segments():
    res = _result((9, 10), (0, 1), (5, 6), (2, 3))
    p = gv.vad_gap_peak(res, cuts_ms=[500.0, 2500.0, 5000.0])
    res_sorted = _result((0, 1), (2, 3), (5, 6), (9, 10))
    p_sorted = gv.vad_gap_peak(res_sorted, cuts_ms=[500.0, 2500.0, 5000.0])
    assert p == p_sorted


# ---- renderer: human-readable -------------------------------------------


def test_render_human_shape():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    lines = gv.render_vad_gap_peak(res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0])
    text = "\n".join(lines)
    assert "silero VAD gap cost peak — rec.wav" in lines[0]
    assert any("segments:" in ln for ln in lines)
    assert any("costliest band:" in ln for ln in lines)
    assert "per +100ms" in text
    assert "--min-silence-ms" in text


def test_render_human_no_gaps():
    res = _result((0, 1))
    lines = gv.render_vad_gap_peak(res)
    assert any("fewer than 2 segments" in ln for ln in lines)
    assert not any("costliest band:" in ln for ln in lines)


def test_render_human_single_cut_note():
    res = _result((0, 1), (2, 3), (5, 6))
    lines = gv.render_vad_gap_peak(res, cuts_ms=[800.0])
    assert any("at least 2 distinct cuts" in ln for ln in lines)
    assert not any("costliest band:" in ln for ln in lines)


def test_render_human_all_valley_note():
    res = _result((0, 1), (5, 6))  # one gap of 4.0s outside all bands
    lines = gv.render_vad_gap_peak(res, cuts_ms=[1000.0, 2000.0, 3000.0])
    assert any("no cost peak" in ln for ln in lines)
    assert not any("costliest band:" in ln for ln in lines)


def test_render_human_unavailable():
    lines = gv.render_vad_gap_peak(None)
    assert len(lines) == 1
    assert "silero VAD unavailable" in lines[0]


# ---- renderer: human golden output --------------------------------------
#
# The shape tests above assert structure + substrings. They do NOT pin the
# EXACT rendered block, so a silent alignment/label/header regression would slip
# through. These goldens freeze the byte-for-byte verdict for fixed stub
# segmentations, so the human face can only change deliberately.


def test_render_human_golden_peak_found():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    lines = gv.render_vad_gap_peak(res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0])
    assert lines == [
        "silero VAD gap cost peak — rec.wav",
        "  segments:     5",
        "  gaps:         4 (pauses between consecutive speech regions)",
        "  min gap:      1.000s",
        "  mean gap:     3.250s",
        "  max gap:      6.000s",
        "  total silence:  13.000s",
        "  costliest band: 500-2500ms (width 2000ms) — the densest pause cluster "
        "/ steepest part of the CDF",
        "  cost:         merges +2 pauses, 0.100 per +100ms (most expensive place "
        "to raise --min-silence-ms — don't cut through here) (iter-350)",
    ]


def test_render_human_golden_all_valley_block():
    res = _result((0, 1), (5, 6))  # one gap of 4.0s outside all bands
    lines = gv.render_vad_gap_peak(res, cuts_ms=[1000.0, 2000.0, 3000.0])
    assert lines == [
        "silero VAD gap cost peak — rec.wav",
        "  segments:     2",
        "  gaps:         1 (pauses between consecutive speech regions)",
        "  min gap:      4.000s",
        "  mean gap:     4.000s",
        "  max gap:      4.000s",
        "  total silence:   4.000s",
        "  (no cost peak — every band is an empty valley; no pause cluster in the "
        "scanned cut range, so raising the hangover costs nothing anywhere here)",
    ]


def test_render_human_golden_single_segment_block():
    res = _result((0, 1))
    lines = gv.render_vad_gap_peak(res)
    assert lines == [
        "silero VAD gap cost peak — rec.wav",
        "  segments:     1",
        "  gaps:         0 (pauses between consecutive speech regions)",
        "  (fewer than 2 segments — no inter-segment pause to measure)",
    ]


def test_render_human_golden_single_cut_block():
    res = _result((0, 1), (2, 3), (5, 6))
    lines = gv.render_vad_gap_peak(res, cuts_ms=[800.0])
    assert lines == [
        "silero VAD gap cost peak — rec.wav",
        "  segments:     3",
        "  gaps:         2 (pauses between consecutive speech regions)",
        "  min gap:      1.000s",
        "  mean gap:     1.500s",
        "  max gap:      2.000s",
        "  total silence:   3.000s",
        "  (need at least 2 distinct cuts to form a cost band — none to show)",
    ]


# ---- renderer: JSON -----------------------------------------------------


def test_render_json_shape():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    payload = json.loads(
        gv.render_vad_gap_peak_json(res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0])
    )
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["num_segments"] == 5
    assert payload["num_gaps"] == 4
    assert payload["num_bands"] == 3
    assert payload["peak_found"] is True
    assert payload["peak_from_ms"] == 500.0
    assert payload["peak_rate_per_100ms"] == pytest.approx(0.1)


def test_render_json_no_peak_for_all_valley():
    res = _result((0, 1), (5, 6))
    payload = json.loads(
        gv.render_vad_gap_peak_json(res, cuts_ms=[1000.0, 2000.0, 3000.0])
    )
    assert payload["peak_found"] is False
    assert payload["peak_from_ms"] is None


def test_render_json_no_gaps():
    res = _result((0, 1))
    payload = json.loads(gv.render_vad_gap_peak_json(res))
    assert payload["peak_found"] is False
    assert payload["min_gap_s"] is None
    assert payload["num_bands"] == 0


def test_render_json_core_agreement():
    res = _result((0, 1), (2, 3), (6, 7), (15, 16))
    cuts = [500.0, 2000.0, 8000.0]
    payload = json.loads(gv.render_vad_gap_peak_json(res, cuts_ms=cuts))
    core = gv.vad_gap_peak(res, cuts_ms=cuts)
    assert payload["peak_from_ms"] == core["peak_from_ms"]
    assert payload["peak_to_ms"] == core["peak_to_ms"]
    assert payload["peak_rate_per_100ms"] == core["peak_rate_per_100ms"]


def test_render_json_unavailable():
    payload = json.loads(gv.render_vad_gap_peak_json(None))
    assert payload["available"] is False
    assert "hint" in payload


# ---- renderer: CSV ------------------------------------------------------


def test_render_csv_shape():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    text = gv.render_vad_gap_peak_csv(res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0])
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == [
        "peak_found",
        "peak_from_ms",
        "peak_to_ms",
        "peak_width_ms",
        "peak_merged_added",
        "peak_rate_per_100ms",
    ]
    assert len(rows) == 2  # header + one verdict row
    assert rows[1][0] == "True"
    assert rows[1][1] == "500"


def test_render_csv_golden():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    text = gv.render_vad_gap_peak_csv(res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0])
    assert text == (
        "peak_found,peak_from_ms,peak_to_ms,peak_width_ms,peak_merged_added,"
        "peak_rate_per_100ms\r\n"
        "True,500,2500,2000,2,0.1"
    )


def test_render_csv_blanks_for_all_valley():
    res = _result((0, 1), (5, 6))
    text = gv.render_vad_gap_peak_csv(res, cuts_ms=[1000.0, 2000.0, 3000.0])
    rows = list(csv.reader(io.StringIO(text)))
    # Bands exist (num_bands > 0) so the row is emitted, but the peak measures
    # are blank because no peak was found.
    assert len(rows) == 2
    assert rows[1] == ["False", "", "", "", "", ""]


def test_render_csv_header_only_for_no_gaps():
    res = _result((0, 1))
    text = gv.render_vad_gap_peak_csv(res)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows == [
        [
            "peak_found",
            "peak_from_ms",
            "peak_to_ms",
            "peak_width_ms",
            "peak_merged_added",
            "peak_rate_per_100ms",
        ]
    ]


def test_render_csv_header_only_for_single_cut():
    res = _result((0, 1), (2, 3), (5, 6))
    text = gv.render_vad_gap_peak_csv(res, cuts_ms=[800.0])
    rows = list(csv.reader(io.StringIO(text)))
    assert len(rows) == 1  # header alone — no band


def test_render_csv_matches_json():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    cuts = [500.0, 2500.0, 3500.0, 5000.0]
    payload = json.loads(gv.render_vad_gap_peak_json(res, cuts_ms=cuts))
    text = gv.render_vad_gap_peak_csv(res, cuts_ms=cuts)
    row = list(csv.reader(io.StringIO(text)))[1]
    assert (row[0] == "True") == payload["peak_found"]
    assert int(row[4]) == payload["peak_merged_added"]
    assert float(row[5]) == pytest.approx(payload["peak_rate_per_100ms"])


def test_render_csv_unavailable():
    text = gv.render_vad_gap_peak_csv(None)
    assert text.startswith("# silero VAD unavailable")


# ---- handler: cmd_vad_gap_peak ------------------------------------------


def _run(args, **kw):
    lines: List[str] = []
    gv.cmd_vad_gap_peak(args, log=lines.append, **kw)
    return lines


def _args(**over):
    base = dict(
        wav="rec.wav",
        cuts_ms=[200.0, 400.0, 800.0, 1600.0],
        threshold=0.5,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        json=False,
        csv=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_handler_human():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    captured = {}

    def segmenter(wav, *, params):
        captured["wav"] = wav
        captured["params"] = params
        return res

    lines = _run(
        _args(cuts_ms=[500.0, 2500.0, 3500.0, 5000.0]),
        segmenter=segmenter,
        availability=lambda: True,
    )
    assert captured["wav"] == "rec.wav"
    assert any("costliest band:" in ln for ln in lines)


def test_handler_json():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    lines = _run(
        _args(cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], json=True),
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is True
    assert payload["peak_found"] is True


def test_handler_csv():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    lines = _run(
        _args(cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], csv=True),
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == [
        "peak_found",
        "peak_from_ms",
        "peak_to_ms",
        "peak_width_ms",
        "peak_merged_added",
        "peak_rate_per_100ms",
    ]


def test_handler_unavailable_human():
    called = []
    lines = _run(
        _args(),
        segmenter=lambda *a, **k: called.append(1),
        availability=lambda: False,
    )
    assert not called  # segmenter never invoked when unavailable
    assert any("silero VAD unavailable" in ln for ln in lines)


def test_handler_unavailable_json():
    lines = _run(
        _args(json=True),
        segmenter=lambda *a, **k: None,
        availability=lambda: False,
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is False


def test_handler_unavailable_csv():
    lines = _run(
        _args(csv=True),
        segmenter=lambda *a, **k: None,
        availability=lambda: False,
    )
    assert lines[0].startswith("# silero VAD unavailable")


def test_handler_passes_cuts_through():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    lines = _run(
        _args(cuts_ms=[100.0, 900.0, 2000.0], json=True),
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["num_bands"] == 2


def test_handler_builds_params_from_knobs():
    res = _result((0, 1), (2, 3))
    captured = {}

    class _Params:
        def __init__(self, **kw):
            captured.update(kw)

    args = _args(
        cuts_ms=[400.0, 800.0],
        threshold=0.7,
        min_speech_ms=100.0,
        min_silence_ms=400.0,
        speech_pad_ms=10.0,
        max_speech_s=5.0,
        json=True,
    )
    import vad.silero as silero_mod

    # The handler imports SileroParams lazily from vad.silero; patch it there.
    orig = getattr(silero_mod, "SileroParams", None)
    silero_mod.SileroParams = _Params
    try:
        _run(args, segmenter=lambda w, *, params: res, availability=lambda: True)
    finally:
        if orig is not None:
            silero_mod.SileroParams = orig
    assert captured["threshold"] == 0.7
    assert captured["min_speech_ms"] == 100.0
    assert captured["min_silence_ms"] == 400.0
    assert captured["speech_pad_ms"] == 10.0
    assert captured["max_speech_s"] == 5.0


# ---- iter-354: --top-n (name the N steepest cost bands) -----------------
#
# vad_gap_peak names the single steepest cost band by default. iter-354's
# --top-n N ranks the N steepest bands instead — a `peaks` list (descending
# rate, earliest band first on a tie) holding ONLY non-empty bands, so it may
# be shorter than N. The legacy scalar peak_* fields always echo peaks[0], so
# top_n=1 is byte-for-byte unchanged on all three faces.


def test_positive_int_type_accepts_one_and_up():
    assert gv.positive_int_type("1") == 1
    assert gv.positive_int_type("5") == 5
    assert gv.positive_int_type(3) == 3


def test_positive_int_type_rejects_zero_negative_fraction():
    import argparse

    for bad in ("0", "-1", "1.5", "abc", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            gv.positive_int_type(bad)


def test_parser_top_n_default_is_one():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-peak", "rec.wav"])
    assert args.top_n == 1


def test_parser_accepts_top_n():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-peak", "rec.wav", "--top-n", "3"])
    assert args.top_n == 3


def test_parser_rejects_bad_top_n():
    parser = gv.build_parser()
    for bad in ["0", "-2", "x"]:
        with pytest.raises(SystemExit):
            parser.parse_args(["vad-gap-peak", "rec.wav", "--top-n", bad])


def test_core_default_top_n_is_one_with_singleton_peaks():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    p = gv.vad_gap_peak(res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0])
    assert p["top_n"] == 1
    assert len(p["peaks"]) == 1
    # peaks[0] echoes the scalar peak_* fields.
    pk = p["peaks"][0]
    assert pk["from_ms"] == p["peak_from_ms"] == 500.0
    assert pk["to_ms"] == p["peak_to_ms"] == 2500.0
    assert pk["width_ms"] == p["peak_width_ms"] == 2000.0
    assert pk["merged_added"] == p["peak_merged_added"] == 2
    assert pk["rate_per_100ms"] == p["peak_rate_per_100ms"] == pytest.approx(0.1)


def test_core_top_n_ranks_descending_rate():
    # Gaps (sorted): 1.0, 2.0, 4.0, 6.0 seconds.
    # cuts 500/2500/3500/5000 -> bands:
    #   500-2500 : {1.0, 2.0} -> +2, rate 0.100  <- #1
    #   2500-3500: {}         -> +0, rate 0.000  (empty valley, NOT listed)
    #   3500-5000: {4.0}      -> +1, rate 0.067  <- #2
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    p = gv.vad_gap_peak(res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=5)
    assert p["top_n"] == 5
    # Only the two non-empty bands are listed (the empty valley is dropped).
    assert len(p["peaks"]) == 2
    rates = [pk["rate_per_100ms"] for pk in p["peaks"]]
    assert rates == sorted(rates, reverse=True)
    assert p["peaks"][0]["from_ms"] == 500.0
    assert p["peaks"][1]["from_ms"] == 3500.0


def test_core_top_n_truncates_to_n():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    p = gv.vad_gap_peak(res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=1)
    assert len(p["peaks"]) == 1
    assert p["peaks"][0]["from_ms"] == 500.0


def test_core_top_n_earliest_tie_first():
    # Two bands with equal rate 0.1; the earlier (lower-cut) band ranks first.
    res = _result((0, 1), (1.5, 2.0), (3.5, 4.0))  # gaps 0.5, 1.5
    p = gv.vad_gap_peak(res, cuts_ms=[0.0, 1000.0, 2000.0], top_n=2)
    assert len(p["peaks"]) == 2
    assert p["peaks"][0]["rate_per_100ms"] == pytest.approx(0.1)
    assert p["peaks"][1]["rate_per_100ms"] == pytest.approx(0.1)
    assert p["peaks"][0]["from_ms"] == 0.0
    assert p["peaks"][1]["from_ms"] == 1000.0


def test_core_top_n_all_valley_empty_peaks():
    res = _result((0, 1), (5, 6))  # one gap of 4.0s, outside all bands
    p = gv.vad_gap_peak(res, cuts_ms=[1000.0, 2000.0, 3000.0], top_n=3)
    assert p["peak_found"] is False
    assert p["peaks"] == []


def test_core_top_n_no_gaps_empty_peaks():
    res = _result((0, 1))
    p = gv.vad_gap_peak(res, top_n=4)
    assert p["num_bands"] == 0
    assert p["peaks"] == []
    assert p["top_n"] == 4


def test_core_top_n_below_one_raises():
    res = _result((0, 1), (2, 3), (5, 6))
    with pytest.raises(ValueError):
        gv.vad_gap_peak(res, top_n=0)
    with pytest.raises(ValueError):
        gv.vad_gap_peak(res, top_n=-1)


def test_core_top_n_each_peak_matches_vad_gap_cost_band():
    res = _result((0, 1), (2, 3), (6, 7), (12, 13), (20, 21))
    cuts = [200.0, 1500.0, 3000.0, 5000.0, 9000.0]
    p = gv.vad_gap_peak(res, cuts_ms=cuts, top_n=10)
    c = gv.vad_gap_cost(res, cuts_ms=cuts)
    nonempty = sorted(
        (b for b in c["bands"] if b["rate_per_100ms"] > 0),
        key=lambda b: -b["rate_per_100ms"],
    )
    assert len(p["peaks"]) == len(nonempty)
    for pk, b in zip(p["peaks"], nonempty):
        assert pk["from_ms"] == b["from_ms"]
        assert pk["to_ms"] == b["to_ms"]
        assert pk["rate_per_100ms"] == b["rate_per_100ms"]
        assert pk["merged_added"] == b["merged_added"]


# ---- renderer faces under --top-n ---------------------------------------


def test_render_human_top_n_one_is_unchanged():
    # top_n=1 reproduces the original single-peak golden exactly.
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    lines = gv.render_vad_gap_peak(res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0])
    lines_explicit = gv.render_vad_gap_peak(
        res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=1
    )
    assert lines == lines_explicit
    assert any("costliest band:" in ln for ln in lines)
    assert not any("top 1 costliest" in ln for ln in lines)


def test_render_human_top_n_multi_golden():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    lines = gv.render_vad_gap_peak(
        res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=3
    )
    assert lines == [
        "silero VAD gap cost peak — rec.wav",
        "  segments:     5",
        "  gaps:         4 (pauses between consecutive speech regions)",
        "  min gap:      1.000s",
        "  mean gap:     3.250s",
        "  max gap:      6.000s",
        "  total silence:  13.000s",
        "  top 3 costliest bands (steepest first — the densest pause clusters / "
        "steepest parts of the CDF; don't raise --min-silence-ms through these) "
        "(iter-354):",
        "    #1: 500-2500ms (width 2000ms) — merges +2 pauses, 0.100 per +100ms",
        "    #2: 3500-5000ms (width 1500ms) — merges +1 pauses, 0.067 per +100ms",
    ]


def test_render_human_top_n_all_valley_note_still_fires():
    res = _result((0, 1), (5, 6))
    lines = gv.render_vad_gap_peak(
        res, cuts_ms=[1000.0, 2000.0, 3000.0], top_n=3
    )
    assert any("no cost peak" in ln for ln in lines)
    assert not any("costliest band" in ln for ln in lines)


def test_render_json_top_n_one_superset_of_legacy():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    cuts = [500.0, 2500.0, 3500.0, 5000.0]
    payload = json.loads(gv.render_vad_gap_peak_json(res, cuts_ms=cuts))
    assert payload["top_n"] == 1
    assert len(payload["peaks"]) == 1
    assert payload["peaks"][0]["from_ms"] == payload["peak_from_ms"]


def test_render_json_top_n_multi():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    payload = json.loads(
        gv.render_vad_gap_peak_json(
            res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=5
        )
    )
    assert payload["top_n"] == 5
    assert len(payload["peaks"]) == 2
    assert payload["peaks"][0]["from_ms"] == 500.0
    assert payload["peaks"][1]["from_ms"] == 3500.0


def test_render_json_top_n_no_gaps_empty_peaks():
    res = _result((0, 1))
    payload = json.loads(gv.render_vad_gap_peak_json(res, top_n=3))
    assert payload["peaks"] == []
    assert payload["top_n"] == 3


def test_render_csv_top_n_one_is_legacy_golden():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    text = gv.render_vad_gap_peak_csv(
        res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=1
    )
    assert text == (
        "peak_found,peak_from_ms,peak_to_ms,peak_width_ms,peak_merged_added,"
        "peak_rate_per_100ms\r\n"
        "True,500,2500,2000,2,0.1"
    )


def test_render_csv_top_n_multi_one_row_per_peak():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    text = gv.render_vad_gap_peak_csv(
        res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=3
    )
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == [
        "peak_found",
        "peak_from_ms",
        "peak_to_ms",
        "peak_width_ms",
        "peak_merged_added",
        "peak_rate_per_100ms",
    ]
    assert len(rows) == 3  # header + two ranked rows (empty valley dropped)
    assert rows[1] == ["True", "500", "2500", "2000", "2", "0.1"]
    assert rows[2] == ["True", "3500", "5000", "1500", "1", "0.067"]


def test_render_csv_top_n_columns_match_single_surface():
    # The multi-row CSV unions cleanly with the single-peak CSV: same header.
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    cuts = [500.0, 2500.0, 3500.0, 5000.0]
    single = list(csv.reader(io.StringIO(gv.render_vad_gap_peak_csv(res, cuts_ms=cuts))))
    multi = list(
        csv.reader(io.StringIO(gv.render_vad_gap_peak_csv(res, cuts_ms=cuts, top_n=3)))
    )
    assert single[0] == multi[0]


def test_render_csv_top_n_all_valley_single_false_row():
    res = _result((0, 1), (5, 6))
    text = gv.render_vad_gap_peak_csv(
        res, cuts_ms=[1000.0, 2000.0, 3000.0], top_n=3
    )
    rows = list(csv.reader(io.StringIO(text)))
    assert len(rows) == 2
    assert rows[1] == ["False", "", "", "", "", ""]


# ---- handler threads --top-n through ------------------------------------


def test_handler_top_n_human_multi():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    lines = _run(
        _args(cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=3),
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    assert any("top 3 costliest bands" in ln for ln in lines)
    assert any(ln.strip().startswith("#1:") for ln in lines)
    assert any(ln.strip().startswith("#2:") for ln in lines)


def test_handler_top_n_json_multi():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    lines = _run(
        _args(cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=5, json=True),
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["top_n"] == 5
    assert len(payload["peaks"]) == 2


def test_handler_top_n_csv_multi():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    lines = _run(
        _args(cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=3, csv=True),
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    assert len(rows) == 3  # header + two ranked rows


def test_handler_top_n_defaults_to_one_when_absent():
    # Older callers without a top_n attr fall back to 1 (getattr default).
    # _args() omits top_n by default, so this exercises the getattr fallback.
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    args = _args(cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], json=True)
    assert not hasattr(args, "top_n")
    lines = _run(
        args,
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["top_n"] == 1
    assert len(payload["peaks"]) == 1


# ---- iter-355: --min-rate floor -----------------------------------------
#
# vad_gap_peak(... min_rate=X) drops cost bands whose rate_per_100ms is
# strictly below X BEFORE top_n truncation, so the ranked list holds only the
# bands worth worrying about. min_rate=0.0 keeps every non-empty band (the
# iter-354 behaviour, byte-for-byte). The canonical fixture below produces two
# non-empty bands: 500-2500ms @ 0.100 (#1) and 3500-5000ms @ 0.067 (#2), plus
# one empty valley at 2500-3500ms.


def test_parser_min_rate_default_is_zero():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-peak", "rec.wav"])
    assert args.min_rate == 0.0


def test_parser_accepts_min_rate():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-peak", "rec.wav", "--min-rate", "0.08"])
    assert args.min_rate == pytest.approx(0.08)


def test_parser_rejects_bad_min_rate():
    parser = gv.build_parser()
    for bad in ["-0.1", "nan", "x"]:
        with pytest.raises(SystemExit):
            parser.parse_args(["vad-gap-peak", "rec.wav", "--min-rate", bad])


def test_core_min_rate_default_keeps_every_non_empty_band():
    # min_rate=0.0 is byte-for-byte the iter-354 result.
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    cuts = [500.0, 2500.0, 3500.0, 5000.0]
    default = gv.vad_gap_peak(res, cuts_ms=cuts, top_n=5)
    explicit = gv.vad_gap_peak(res, cuts_ms=cuts, top_n=5, min_rate=0.0)
    assert default == explicit
    assert explicit["min_rate"] == 0.0
    assert len(explicit["peaks"]) == 2


def test_core_min_rate_drops_cheaper_bands():
    # Floor of 0.08 drops the 0.067 band, keeping only the 0.100 peak.
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    p = gv.vad_gap_peak(
        res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=5, min_rate=0.08
    )
    assert p["min_rate"] == pytest.approx(0.08)
    assert len(p["peaks"]) == 1
    assert p["peaks"][0]["from_ms"] == 500.0
    assert p["peaks"][0]["rate_per_100ms"] == pytest.approx(0.1)
    # The scalar peak_* fields still echo peaks[0].
    assert p["peak_from_ms"] == 500.0
    assert p["peak_found"] is True


def test_core_min_rate_boundary_is_inclusive():
    # A band exactly AT the floor is kept (>= floor, not strictly above).
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    p = gv.vad_gap_peak(
        res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=5, min_rate=0.067
    )
    rates = [pk["rate_per_100ms"] for pk in p["peaks"]]
    assert pytest.approx(0.067) in rates
    assert len(p["peaks"]) == 2


def test_core_min_rate_filters_all_bands():
    # A floor above every band's rate leaves nothing to name — same "no
    # structure" spelling as the all-valley case.
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    p = gv.vad_gap_peak(
        res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=5, min_rate=0.5
    )
    assert p["peak_found"] is False
    assert p["peaks"] == []
    assert p["peak_from_ms"] is None
    # num_bands still reflects the scanned bands (the floor only affects ranking).
    assert p["num_bands"] == 3


def test_core_min_rate_composes_with_top_n():
    # Floor keeps both bands; top_n=1 then truncates to the steepest.
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    p = gv.vad_gap_peak(
        res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=1, min_rate=0.05
    )
    assert len(p["peaks"]) == 1
    assert p["peaks"][0]["rate_per_100ms"] == pytest.approx(0.1)


def test_core_min_rate_negative_raises():
    res = _result((0, 1), (2, 3), (5, 6))
    with pytest.raises(ValueError):
        gv.vad_gap_peak(res, min_rate=-0.1)


def test_render_human_min_rate_zero_is_unchanged():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    cuts = [500.0, 2500.0, 3500.0, 5000.0]
    base = gv.render_vad_gap_peak(res, cuts_ms=cuts, top_n=3)
    explicit = gv.render_vad_gap_peak(res, cuts_ms=cuts, top_n=3, min_rate=0.0)
    assert base == explicit
    assert not any("rate floor" in ln for ln in base)


def test_render_human_min_rate_note_and_filtered_ranking():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    lines = gv.render_vad_gap_peak(
        res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=5, min_rate=0.08
    )
    assert any("rate floor:" in ln and "0.080" in ln for ln in lines)
    # Only the 0.100 band survives the floor.
    assert any("#1:" in ln for ln in lines)
    assert not any("#2:" in ln for ln in lines)


def test_render_human_min_rate_no_peak_meets_floor():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    lines = gv.render_vad_gap_peak(
        res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=5, min_rate=0.5
    )
    assert any("rate floor:" in ln for ln in lines)
    assert any("no cost peak meets the rate floor" in ln for ln in lines)
    assert not any("costliest band" in ln for ln in lines)


def test_render_json_min_rate_echoed_and_filters():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    payload = json.loads(
        gv.render_vad_gap_peak_json(
            res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=5, min_rate=0.08
        )
    )
    assert payload["min_rate"] == pytest.approx(0.08)
    assert len(payload["peaks"]) == 1
    assert payload["peaks"][0]["from_ms"] == 500.0


def test_render_json_min_rate_default_present():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    payload = json.loads(
        gv.render_vad_gap_peak_json(res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0])
    )
    assert payload["min_rate"] == 0.0


def test_render_csv_min_rate_filters_rows_columns_unchanged():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    cuts = [500.0, 2500.0, 3500.0, 5000.0]
    text = gv.render_vad_gap_peak_csv(res, cuts_ms=cuts, top_n=5, min_rate=0.08)
    rows = list(csv.reader(io.StringIO(text)))
    # Same six columns as the single-peak surface; one ranked row survives.
    single = list(csv.reader(io.StringIO(gv.render_vad_gap_peak_csv(res, cuts_ms=cuts))))
    assert rows[0] == single[0]
    assert len(rows) == 2
    assert rows[1] == ["True", "500", "2500", "2000", "2", "0.1"]


def test_render_csv_min_rate_all_filtered_blank_row():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    text = gv.render_vad_gap_peak_csv(
        res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=5, min_rate=0.5
    )
    rows = list(csv.reader(io.StringIO(text)))
    assert len(rows) == 2
    assert rows[1] == ["False", "", "", "", "", ""]


def test_handler_min_rate_threads_through_json():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    lines = _run(
        _args(cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=5, min_rate=0.08, json=True),
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["min_rate"] == pytest.approx(0.08)
    assert len(payload["peaks"]) == 1


def test_handler_min_rate_defaults_to_zero_when_absent():
    # Older callers without a min_rate attr fall back to 0.0 (getattr default).
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    args = _args(cuts_ms=[500.0, 2500.0, 3500.0, 5000.0], top_n=5, json=True)
    assert not hasattr(args, "min_rate")
    lines = _run(
        args,
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["min_rate"] == 0.0
    assert len(payload["peaks"]) == 2
