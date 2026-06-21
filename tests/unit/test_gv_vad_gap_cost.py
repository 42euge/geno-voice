"""Tests for iter-349 — the ``gv vad-gap-cost`` subcommand (examples/gv.py).

``gv vad-gap-cdf`` (iter-346) reports the cumulative empirical CDF: at candidate
hangover cut ``c``, what FRACTION of the inter-segment pauses are shorter than
``c`` (and so would MERGE). ``gv vad-gap-cost`` reports its DERIVATIVE — between
two consecutive cuts, how many ADDITIONAL pauses get swallowed and at what rate
per +100 ms of hangover. A high-rate band sits inside a pause cluster (expensive
to raise the hangover there); a zero-rate band is an empty valley where raising
the hangover costs nothing — exactly where ``gv vad-gap-recommend`` points.

The merge rule follows the segmenter's own convention: a pause STRICTLY ``< c``
merges, ``>= c`` is kept. Like the rest of the VAD-analysis family, the handler
takes injected ``segmenter`` / ``availability`` / ``log`` dependencies so every
test runs WITHOUT importing torch / silero-vad and without touching real audio —
fast and deterministic on the x86_64 Linux runner. The pure core
(``vad_gap_cost``) and the three renderers are exercised directly against
lightweight stand-ins mirroring just the ``SileroResult`` / ``SpeechSegment``
attributes they read.
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


def test_vad_gap_cost_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-gap-cost"] is gv.cmd_vad_gap_cost


def test_parser_registers_vad_gap_cost():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-cost", "rec.wav"])
    assert args.command == "vad-gap-cost"
    assert args.wav == "rec.wav"


def test_parser_defaults_mirror_vad_gaps_knobs():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-cost", "rec.wav"])
    # Shares the gv vad segmenter knobs.
    assert args.threshold == pytest.approx(0.5)
    assert args.min_speech_ms == pytest.approx(250.0)
    assert args.min_silence_ms == pytest.approx(800.0)
    assert args.speech_pad_ms == pytest.approx(30.0)
    assert math.isinf(args.max_speech_s)
    # The cuts default (reuses vad-gap-cdf's cuts list).
    assert args.cuts_ms == [200.0, 400.0, 800.0, 1600.0]
    assert args.json is False
    assert args.csv is False


def test_parser_accepts_custom_cuts():
    parser = gv.build_parser()
    args = parser.parse_args(
        ["vad-gap-cost", "rec.wav", "--cuts-ms", "100,500,1000"]
    )
    assert args.cuts_ms == [100.0, 500.0, 1000.0]


def test_parser_rejects_bad_cuts():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-cost", "rec.wav", "--cuts-ms", "-5"])
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-cost", "rec.wav", "--cuts-ms", "nan"])


def test_parser_json_csv_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-cost", "rec.wav", "--json", "--csv"])


def test_parser_rejects_out_of_range_threshold():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-cost", "rec.wav", "--threshold", "1.5"])


# ---- pure core: vad_gap_cost --------------------------------------------


def test_cost_basic_marginal_counts():
    # Gaps (sorted): 1.0, 2.0, 4.0, 6.0 seconds.
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    c = gv.vad_gap_cost(res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0])
    assert c["num_segments"] == 5
    assert c["num_gaps"] == 4
    bands = c["bands"]
    assert len(bands) == 3  # N distinct cuts -> N-1 bands
    # 500-2500ms band: gaps in [0.5, 2.5) are {1.0, 2.0} -> +2.
    assert bands[0]["from_ms"] == 500.0
    assert bands[0]["to_ms"] == 2500.0
    assert bands[0]["merged_added"] == 2
    assert bands[0]["merged_cumulative"] == 2
    assert bands[0]["width_ms"] == 2000.0
    assert bands[0]["rate_per_100ms"] == pytest.approx(0.1)  # 2 / 2000 * 100
    # 2500-3500ms band: no gap in [2.5, 3.5) -> a zero-rate valley.
    assert bands[1]["merged_added"] == 0
    assert bands[1]["rate_per_100ms"] == 0.0
    # 3500-5000ms band: gap 4.0 falls in [3.5, 5.0) -> +1.
    assert bands[2]["merged_added"] == 1
    assert bands[2]["merged_cumulative"] == 3


def test_cost_is_difference_of_cdf():
    # The cost curve is the derivative of the CDF: each band's merged_cumulative
    # equals exactly what vad_gap_cdf reports at the band's top cut, and the
    # marginal merged_added is the difference between consecutive cumulative
    # counts.
    res = _result((0, 1), (2, 3), (6, 7), (12, 13), (20, 21))
    cuts = [200.0, 1500.0, 3000.0, 5000.0, 9000.0]
    cost = gv.vad_gap_cost(res, cuts_ms=cuts)
    cdf = gv.vad_gap_cdf(res, cuts_ms=cuts)
    cdf_merged = {e["cut_ms"]: e["merged"] for e in cdf["cuts"]}
    for band in cost["bands"]:
        assert band["merged_cumulative"] == cdf_merged[band["to_ms"]]
        expected_added = cdf_merged[band["to_ms"]] - cdf_merged[band["from_ms"]]
        assert band["merged_added"] == expected_added


def test_cost_added_sums_to_total_merged_at_top():
    # The marginal counts telescope: the sum of merged_added equals the
    # cumulative merged at the top cut (every pause below the top counted once).
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    c = gv.vad_gap_cost(res, cuts_ms=[0.0, 1500.0, 3000.0, 8000.0])
    total_added = sum(b["merged_added"] for b in c["bands"])
    assert total_added == c["bands"][-1]["merged_cumulative"]


def test_cost_zero_rate_band_is_the_recommended_valley():
    # The flattest (zero-cost) band is where vad_gap_recommend points: the
    # recommended hangover sits in an empty band of the distribution.
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    rec = gv.vad_gap_recommend(res)
    assert rec["split_found"] is True
    rec_ms = rec["recommended_ms"]
    # Bracket the recommendation with cuts straddling it; the band containing it
    # must be a zero-rate valley.
    cuts = [rec_ms - 500.0, rec_ms + 500.0]
    c = gv.vad_gap_cost(res, cuts_ms=cuts)
    band = c["bands"][0]
    assert band["from_ms"] <= rec_ms <= band["to_ms"]
    assert band["rate_per_100ms"] == 0.0


def test_cost_boundary_is_strict_less_than():
    # A pause EXACTLY at a cut is kept (ends the turn), not merged: it counts in
    # the band ABOVE that cut, never below.
    res = _result((0, 1), (3, 4))  # one gap of exactly 2.0s
    c = gv.vad_gap_cost(res, cuts_ms=[1000.0, 2000.0, 3000.0])
    # [1.0, 2.0) band: 2.0 is NOT < 2.0 -> +0.
    assert c["bands"][0]["merged_added"] == 0
    # [2.0, 3.0) band: 2.0 is < 3.0 -> +1.
    assert c["bands"][1]["merged_added"] == 1


def test_cost_high_rate_band_inside_cluster():
    # A narrow band packed with pauses has a high marginal rate.
    res = _result((0, 1), (1.5, 2.5), (3.0, 4.0), (4.5, 5.5))  # gaps 0.5,0.5,0.5
    c = gv.vad_gap_cost(res, cuts_ms=[400.0, 600.0])
    band = c["bands"][0]
    # All three 0.5s gaps fall in [0.4, 0.6); width 200ms -> 3/200*100 = 1.5.
    assert band["merged_added"] == 3
    assert band["rate_per_100ms"] == pytest.approx(1.5)


def test_cost_cuts_sorted_and_deduplicated():
    # Unlike vad_gap_cdf (preserves column order), the cost curve needs a
    # monotone axis: unsorted / duplicate cuts collapse to sorted-unique.
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    c = gv.vad_gap_cost(res, cuts_ms=[3000.0, 500.0, 3000.0, 1500.0])
    froms = [b["from_ms"] for b in c["bands"]]
    tos = [b["to_ms"] for b in c["bands"]]
    assert froms == [500.0, 1500.0]
    assert tos == [1500.0, 3000.0]


def test_cost_anchors_to_vad_silence_gaps():
    res = _result((0, 1), (2, 3), (6, 7), (15, 16))
    c = gv.vad_gap_cost(res, cuts_ms=[800.0, 1600.0])
    d = gv.vad_silence_gaps(res)
    assert c["num_segments"] == d["num_segments"]
    assert c["num_gaps"] == d["num_gaps"]
    assert c["min_gap_s"] == d["min_gap_s"]
    assert c["max_gap_s"] == d["max_gap_s"]
    assert c["mean_gap_s"] == d["mean_gap_s"]
    assert c["total_silence_s"] == d["total_silence_s"]


def test_cost_empty_for_fewer_than_two_segments():
    res = _result((0, 1))
    c = gv.vad_gap_cost(res)
    assert c["num_gaps"] == 0
    assert c["bands"] == []
    assert c["min_gap_s"] is None


def test_cost_empty_for_zero_segments():
    res = _result()
    c = gv.vad_gap_cost(res)
    assert c["num_segments"] == 0
    assert c["bands"] == []


def test_cost_empty_bands_for_single_distinct_cut():
    # A single distinct cut forms no band (a degenerate axis).
    res = _result((0, 1), (2, 3), (5, 6))
    c = gv.vad_gap_cost(res, cuts_ms=[800.0])
    assert c["num_gaps"] == 2
    assert c["bands"] == []


def test_cost_duplicate_cuts_collapse_to_no_band():
    # All-duplicate cuts collapse to one distinct value -> no band.
    res = _result((0, 1), (2, 3), (5, 6))
    c = gv.vad_gap_cost(res, cuts_ms=[800.0, 800.0, 800.0])
    assert c["bands"] == []


def test_cost_rate_rounded_to_three_places():
    # single gap 4.0 in [3.5, 5.0): 1 / 1500 * 100 = 0.0666… -> 0.067.
    res = _result((0, 1), (5, 6))  # one gap of 4.0s
    c = gv.vad_gap_cost(res, cuts_ms=[3500.0, 5000.0])
    assert c["bands"][0]["rate_per_100ms"] == 0.067


def test_cost_default_cuts_form_three_bands():
    res = _result((0, 1), (2, 3), (5, 6))
    c = gv.vad_gap_cost(res)  # default cuts 200/400/800/1600 -> 3 bands
    assert [b["from_ms"] for b in c["bands"]] == [200.0, 400.0, 800.0]
    assert [b["to_ms"] for b in c["bands"]] == [400.0, 800.0, 1600.0]


def test_cost_rejects_empty_cuts():
    res = _result((0, 1), (2, 3))
    with pytest.raises(ValueError):
        gv.vad_gap_cost(res, cuts_ms=[])


def test_cost_rejects_negative_cut():
    res = _result((0, 1), (2, 3))
    with pytest.raises(ValueError):
        gv.vad_gap_cost(res, cuts_ms=[-1.0])


def test_cost_rejects_nan_cut():
    res = _result((0, 1), (2, 3))
    with pytest.raises(ValueError):
        gv.vad_gap_cost(res, cuts_ms=[float("nan")])


def test_cost_handles_unsorted_segments():
    res = _result((9, 10), (0, 1), (5, 6), (2, 3))
    c = gv.vad_gap_cost(res, cuts_ms=[500.0, 2500.0])
    res_sorted = _result((0, 1), (2, 3), (5, 6), (9, 10))
    c_sorted = gv.vad_gap_cost(res_sorted, cuts_ms=[500.0, 2500.0])
    assert c["bands"] == c_sorted["bands"]


# ---- renderer: human-readable -------------------------------------------


def test_render_human_shape():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    lines = gv.render_vad_gap_cost(res)
    text = "\n".join(lines)
    assert "silero VAD gap merge cost curve — rec.wav" in lines[0]
    assert any("segments:" in ln for ln in lines)
    assert any("gaps:" in ln for ln in lines)
    assert any("per +100ms" in ln for ln in lines)
    # Names the actionable knob.
    assert "--min-silence-ms" in text


def test_render_human_no_gaps():
    res = _result((0, 1))
    lines = gv.render_vad_gap_cost(res)
    assert any("fewer than 2 segments" in ln for ln in lines)
    assert not any("per +100ms" in ln for ln in lines)


def test_render_human_single_cut_note():
    res = _result((0, 1), (2, 3), (5, 6))
    lines = gv.render_vad_gap_cost(res, cuts_ms=[800.0])
    assert any("at least 2 distinct cuts" in ln for ln in lines)
    assert not any("per +100ms" in ln for ln in lines)


def test_render_human_unavailable():
    lines = gv.render_vad_gap_cost(None)
    assert len(lines) == 1
    assert "silero VAD unavailable" in lines[0]


def test_render_human_custom_cut_labels():
    res = _result((0, 1), (2, 3), (5, 6))
    lines = gv.render_vad_gap_cost(res, cuts_ms=[250.0, 750.5])
    text = "\n".join(lines)
    assert "250" in text
    assert "750.5" in text  # fractional cut keeps its decimals


# ---- renderer: human golden output --------------------------------------
#
# The shape tests above assert structure + substrings. They do NOT pin the
# EXACT rendered block, so a silent alignment/label/header regression — a column
# drifting by a space, the ``%g`` cut spelling losing its compact form, the
# ``+N`` merged-added field or the ``per +100ms`` rate column shifting — would
# slip through every one of them. These goldens freeze the byte-for-byte report
# for fixed stub segmentations, so the human face can only change deliberately.


def test_render_human_golden_varied_bands():
    # Gaps (sorted): [1.0, 2.0, 4.0, 6.0]. Cuts chosen to land a +2 band, a
    # zero-rate valley band, and a +1 band, so the merged ``+N`` column and the
    # rate column are each exercised at distinct values (0.100 / 0.000 / 0.067).
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    lines = gv.render_vad_gap_cost(res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0])
    assert lines == [
        "silero VAD gap merge cost curve — rec.wav",
        "  segments:     5",
        "  gaps:         4 (pauses between consecutive speech regions)",
        "  min gap:      1.000s (shortest real pause — keep --min-silence-ms "
        "below this to avoid merging turns)",
        "  mean gap:     3.250s",
        "  max gap:      6.000s",
        "  total silence:  13.000s",
        "  band (ms)        width   merged   per +100ms",
        "  500-2500        2000ms      +2      0.100",
        "  2500-3500       1000ms      +0      0.000",
        "  3500-5000       1500ms      +1      0.067",
    ]


def test_render_human_golden_single_segment_block():
    # The <2-segment branch: stats header truncates at the explanatory line,
    # WITHOUT the band table (no pauses to differentiate).
    res = _result((0, 1))
    lines = gv.render_vad_gap_cost(res)
    assert lines == [
        "silero VAD gap merge cost curve — rec.wav",
        "  segments:     1",
        "  gaps:         0 (pauses between consecutive speech regions)",
        "  (fewer than 2 segments — no inter-segment pause to measure)",
    ]


def test_render_human_golden_single_cut_block():
    # The single-distinct-cut branch: stats header truncates at the "need 2
    # distinct cuts" line, WITHOUT the band table (a degenerate axis).
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    lines = gv.render_vad_gap_cost(res, cuts_ms=[800.0])
    assert lines == [
        "silero VAD gap merge cost curve — rec.wav",
        "  segments:     5",
        "  gaps:         4 (pauses between consecutive speech regions)",
        "  min gap:      1.000s (shortest real pause — keep --min-silence-ms "
        "below this to avoid merging turns)",
        "  mean gap:     3.250s",
        "  max gap:      6.000s",
        "  total silence:  13.000s",
        "  (need at least 2 distinct cuts to form a cost band — none to show)",
    ]


# ---- renderer: JSON -----------------------------------------------------


def test_render_json_shape():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    payload = json.loads(gv.render_vad_gap_cost_json(res))
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["num_segments"] == 4
    assert payload["num_gaps"] == 3
    # Default cuts 200/400/800/1600 -> 3 bands.
    assert [b["from_ms"] for b in payload["bands"]] == [200.0, 400.0, 800.0]
    assert all("rate_per_100ms" in b for b in payload["bands"])
    assert all("merged_cumulative" in b for b in payload["bands"])


def test_render_json_empty_bands_for_no_gaps():
    res = _result((0, 1))
    payload = json.loads(gv.render_vad_gap_cost_json(res))
    assert payload["bands"] == []
    assert payload["min_gap_s"] is None


def test_render_json_core_agreement():
    res = _result((0, 1), (2, 3), (6, 7), (15, 16))
    payload = json.loads(gv.render_vad_gap_cost_json(res, cuts_ms=[500.0, 2000.0]))
    core = gv.vad_gap_cost(res, cuts_ms=[500.0, 2000.0])
    assert payload["bands"] == core["bands"]


def test_render_json_unavailable():
    payload = json.loads(gv.render_vad_gap_cost_json(None))
    assert payload["available"] is False
    assert "hint" in payload


# ---- renderer: CSV ------------------------------------------------------


def test_render_csv_shape():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    text = gv.render_vad_gap_cost_csv(res)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == [
        "from_ms",
        "to_ms",
        "width_ms",
        "merged_added",
        "merged_cumulative",
        "rate_per_100ms",
    ]
    assert len(rows) == 1 + 3  # header + 3 bands (4 default cuts)
    assert rows[1][0] == "200"


def test_render_csv_matches_json():
    res = _result((0, 1), (2, 3), (6, 7), (15, 16))
    payload = json.loads(gv.render_vad_gap_cost_json(res, cuts_ms=[500.0, 2000.0]))
    text = gv.render_vad_gap_cost_csv(res, cuts_ms=[500.0, 2000.0])
    rows = list(csv.reader(io.StringIO(text)))[1:]
    assert len(rows) == len(payload["bands"])
    for row, band in zip(rows, payload["bands"]):
        assert int(row[3]) == band["merged_added"]
        assert int(row[4]) == band["merged_cumulative"]
        assert float(row[5]) == pytest.approx(band["rate_per_100ms"])


def test_render_csv_golden():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    text = gv.render_vad_gap_cost_csv(res, cuts_ms=[500.0, 2500.0, 3500.0, 5000.0])
    assert text == (
        "from_ms,to_ms,width_ms,merged_added,merged_cumulative,rate_per_100ms\r\n"
        "500,2500,2000,2,2,0.1\r\n"
        "2500,3500,1000,0,2,0.0\r\n"
        "3500,5000,1500,1,3,0.067"
    )


def test_render_csv_header_only_for_no_gaps():
    res = _result((0, 1))
    text = gv.render_vad_gap_cost_csv(res)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows == [
        [
            "from_ms",
            "to_ms",
            "width_ms",
            "merged_added",
            "merged_cumulative",
            "rate_per_100ms",
        ]
    ]


def test_render_csv_header_only_for_single_cut():
    res = _result((0, 1), (2, 3), (5, 6))
    text = gv.render_vad_gap_cost_csv(res, cuts_ms=[800.0])
    rows = list(csv.reader(io.StringIO(text)))
    assert len(rows) == 1  # header alone — no band


def test_render_csv_unavailable():
    text = gv.render_vad_gap_cost_csv(None)
    assert text.startswith("# silero VAD unavailable")


# ---- handler: cmd_vad_gap_cost ------------------------------------------


def _run(args, **kw):
    lines: List[str] = []
    gv.cmd_vad_gap_cost(args, log=lines.append, **kw)
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
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    captured = {}

    def segmenter(wav, *, params):
        captured["wav"] = wav
        captured["params"] = params
        return res

    lines = _run(_args(), segmenter=segmenter, availability=lambda: True)
    assert captured["wav"] == "rec.wav"
    assert any("per +100ms" in ln for ln in lines)


def test_handler_json():
    res = _result((0, 1), (2, 3), (5, 6))
    lines = _run(
        _args(json=True),
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is True
    assert len(payload["bands"]) == 3  # 4 default cuts -> 3 bands


def test_handler_csv():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    lines = _run(
        _args(cuts_ms=[500.0, 2000.0], csv=True),
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == [
        "from_ms",
        "to_ms",
        "width_ms",
        "merged_added",
        "merged_cumulative",
        "rate_per_100ms",
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
    assert [b["from_ms"] for b in payload["bands"]] == [100.0, 900.0]
    assert [b["to_ms"] for b in payload["bands"]] == [900.0, 2000.0]


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
