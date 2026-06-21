"""Tests for iter-346 — the ``gv vad-gap-cdf`` subcommand (examples/gv.py).

``gv vad-gap-percentiles`` (iter-338) answers "what pause length sits at the
p90?" (fraction → value). ``gv vad-gap-cdf`` answers the operationally-direct
INVERSE — "if I set the end-of-turn hangover (``--min-silence-ms`` / the live
``chat.vad.silence_duration``) to candidate cut ``c``, what FRACTION of the
inter-segment pauses are shorter than ``c`` and would therefore be MERGED
(swallowed as within-turn silence rather than ending a turn)?" (value →
fraction). That is the empirical CDF of the gap distribution sampled at the
operator's candidate cuts — a direct "this hangover merges X% of your pauses"
answer.

The merge rule follows the segmenter's own convention: a region ends once the
trailing silence REACHES the hangover, so a pause ``>= c`` ends the turn (kept)
while a pause STRICTLY ``< c`` merges. Like the rest of the VAD-analysis family,
the handler takes injected ``segmenter`` / ``availability`` / ``log``
dependencies so every test runs WITHOUT importing torch / silero-vad and without
touching real audio — fast and deterministic on the x86_64 Linux runner. The
pure core (``vad_gap_cdf``) and the three renderers are exercised directly
against lightweight stand-ins mirroring just the ``SileroResult`` /
``SpeechSegment`` attributes they read.
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


# ---- parser: cut_ms_list_type validator ---------------------------------


def test_cut_ms_list_type_parses():
    assert gv.cut_ms_list_type("200,400,800,1600") == [200.0, 400.0, 800.0, 1600.0]


def test_cut_ms_list_type_preserves_order_and_dups():
    assert gv.cut_ms_list_type("800,400,800") == [800.0, 400.0, 800.0]


def test_cut_ms_list_type_strips_whitespace():
    assert gv.cut_ms_list_type(" 250 , 500 ") == [250.0, 500.0]


def test_cut_ms_list_type_allows_zero():
    # A zero hangover is legitimate — it merges nothing.
    assert gv.cut_ms_list_type("0") == [0.0]


def test_cut_ms_list_type_rejects_negative():
    with pytest.raises(gv.argparse.ArgumentTypeError):
        gv.cut_ms_list_type("-5")


def test_cut_ms_list_type_rejects_nan():
    with pytest.raises(gv.argparse.ArgumentTypeError):
        gv.cut_ms_list_type("nan")


def test_cut_ms_list_type_rejects_nonnumber():
    with pytest.raises(gv.argparse.ArgumentTypeError):
        gv.cut_ms_list_type("200,abc")


def test_cut_ms_list_type_rejects_empty():
    with pytest.raises(gv.argparse.ArgumentTypeError):
        gv.cut_ms_list_type("")
    with pytest.raises(gv.argparse.ArgumentTypeError):
        gv.cut_ms_list_type(",,")


def test_cut_ms_list_type_rejects_non_string():
    with pytest.raises(gv.argparse.ArgumentTypeError):
        gv.cut_ms_list_type(200)


# ---- parser: registration & defaults -----------------------------------


def test_vad_gap_cdf_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-gap-cdf"] is gv.cmd_vad_gap_cdf


def test_parser_registers_vad_gap_cdf():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-cdf", "rec.wav"])
    assert args.command == "vad-gap-cdf"
    assert args.wav == "rec.wav"


def test_parser_defaults_mirror_vad_gaps_knobs():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-cdf", "rec.wav"])
    # Shares the gv vad segmenter knobs.
    assert args.threshold == pytest.approx(0.5)
    assert args.min_speech_ms == pytest.approx(250.0)
    assert args.min_silence_ms == pytest.approx(800.0)
    assert args.speech_pad_ms == pytest.approx(30.0)
    assert math.isinf(args.max_speech_s)
    # The cuts default.
    assert args.cuts_ms == [200.0, 400.0, 800.0, 1600.0]
    assert args.json is False
    assert args.csv is False


def test_parser_accepts_custom_cuts():
    parser = gv.build_parser()
    args = parser.parse_args(
        ["vad-gap-cdf", "rec.wav", "--cuts-ms", "100,500,1000"]
    )
    assert args.cuts_ms == [100.0, 500.0, 1000.0]


def test_parser_rejects_bad_cuts():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-cdf", "rec.wav", "--cuts-ms", "-5"])
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-cdf", "rec.wav", "--cuts-ms", "nan"])


def test_parser_json_csv_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-cdf", "rec.wav", "--json", "--csv"])


def test_parser_rejects_out_of_range_threshold():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-cdf", "rec.wav", "--threshold", "1.5"])


# ---- pure core: vad_gap_cdf ---------------------------------------------


def test_cdf_basic_merge_counts():
    # Gaps (sorted): 1.0, 2.0, 3.0, 4.0 seconds.
    res = _result((0, 1), (2, 3), (5, 6), (9, 10), (14, 15))
    c = gv.vad_gap_cdf(res, cuts_ms=[2500.0])
    assert c["num_segments"] == 5
    assert c["num_gaps"] == 4
    # cut 2.5s: gaps < 2.5 are {1.0, 2.0} -> 2 merged of 4.
    entry = c["cuts"][0]
    assert entry["cut_ms"] == 2500.0
    assert entry["cut_s"] == 2.5
    assert entry["merged"] == 2
    assert entry["kept"] == 2
    assert entry["merge_fraction"] == pytest.approx(0.5)
    assert entry["keep_fraction"] == pytest.approx(0.5)


def test_cdf_boundary_is_strict_less_than():
    # A pause EXACTLY at the cut ends the turn (kept), not merged: the segmenter
    # ends a region once silence REACHES the hangover.
    res = _result((0, 1), (3, 4))  # one gap of exactly 2.0s
    c = gv.vad_gap_cdf(res, cuts_ms=[2000.0])
    entry = c["cuts"][0]
    assert entry["merged"] == 0  # 2.0 is NOT < 2.0
    assert entry["kept"] == 1
    assert entry["merge_fraction"] == 0.0


def test_cdf_below_min_merges_nothing():
    # gaps 1.0, 2.0; cut 0.5s is below the min gap -> merges nothing.
    res = _result((0, 1), (2, 3), (5, 6))
    c = gv.vad_gap_cdf(res, cuts_ms=[500.0])
    entry = c["cuts"][0]
    assert entry["merged"] == 0
    assert entry["merge_fraction"] == 0.0


def test_cdf_above_max_merges_everything():
    # gaps 1.0, 2.0; cut 5.0s exceeds the max gap -> merges everything.
    res = _result((0, 1), (2, 3), (5, 6))
    c = gv.vad_gap_cdf(res, cuts_ms=[5000.0])
    entry = c["cuts"][0]
    assert entry["merged"] == 2
    assert entry["merge_fraction"] == 1.0


def test_cdf_monotonic_non_decreasing_in_cut():
    # The merge fraction is a CDF — non-decreasing as the cut grows.
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (17, 18))
    c = gv.vad_gap_cdf(res, cuts_ms=[500.0, 1500.0, 2500.0, 3500.0, 8000.0])
    fracs = [e["merge_fraction"] for e in c["cuts"]]
    assert fracs == sorted(fracs)
    assert fracs[0] == 0.0
    assert fracs[-1] == 1.0


def test_cdf_order_preserved():
    # Cuts are NOT sorted — the operator's column order is preserved.
    res = _result((0, 1), (2, 3), (5, 6))
    c = gv.vad_gap_cdf(res, cuts_ms=[5000.0, 500.0])
    assert [e["cut_ms"] for e in c["cuts"]] == [5000.0, 500.0]


def test_cdf_anchors_to_vad_silence_gaps():
    res = _result((0, 1), (2, 3), (6, 7), (15, 16))
    c = gv.vad_gap_cdf(res, cuts_ms=[800.0])
    d = gv.vad_silence_gaps(res)
    assert c["num_segments"] == d["num_segments"]
    assert c["num_gaps"] == d["num_gaps"]
    assert c["min_gap_s"] == d["min_gap_s"]
    assert c["max_gap_s"] == d["max_gap_s"]
    assert c["mean_gap_s"] == d["mean_gap_s"]
    assert c["total_silence_s"] == d["total_silence_s"]


def test_cdf_empty_for_fewer_than_two_segments():
    res = _result((0, 1))
    c = gv.vad_gap_cdf(res)
    assert c["num_gaps"] == 0
    assert c["cuts"] == []
    assert c["min_gap_s"] is None


def test_cdf_empty_for_zero_segments():
    res = _result()
    c = gv.vad_gap_cdf(res)
    assert c["num_segments"] == 0
    assert c["cuts"] == []


def test_cdf_keep_plus_merge_equals_num_gaps():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    c = gv.vad_gap_cdf(res, cuts_ms=[200.0, 800.0, 2500.0, 5000.0])
    for entry in c["cuts"]:
        assert entry["merged"] + entry["kept"] == c["num_gaps"]


def test_cdf_zero_cut_merges_nothing():
    res = _result((0, 1), (2, 3), (5, 6))
    c = gv.vad_gap_cdf(res, cuts_ms=[0.0])
    entry = c["cuts"][0]
    assert entry["cut_s"] == 0.0
    assert entry["merged"] == 0  # nothing is < 0
    assert entry["merge_fraction"] == 0.0


def test_cdf_fractions_rounded_to_three_places():
    # 3 gaps: 1.0, 2.0, 3.0; cut 2.5 merges 2/3 -> 0.667 (not 0.6666…).
    res = _result((0, 1), (2, 3), (5, 6), (9, 10))
    c = gv.vad_gap_cdf(res, cuts_ms=[2500.0])
    assert c["cuts"][0]["merge_fraction"] == 0.667


def test_cdf_default_cuts_are_200_400_800_1600():
    res = _result((0, 1), (2, 3), (5, 6))
    c = gv.vad_gap_cdf(res)
    assert [e["cut_ms"] for e in c["cuts"]] == [200.0, 400.0, 800.0, 1600.0]


def test_cdf_rejects_empty_cuts():
    res = _result((0, 1), (2, 3))
    with pytest.raises(ValueError):
        gv.vad_gap_cdf(res, cuts_ms=[])


def test_cdf_rejects_negative_cut():
    res = _result((0, 1), (2, 3))
    with pytest.raises(ValueError):
        gv.vad_gap_cdf(res, cuts_ms=[-1.0])


def test_cdf_rejects_nan_cut():
    res = _result((0, 1), (2, 3))
    with pytest.raises(ValueError):
        gv.vad_gap_cdf(res, cuts_ms=[float("nan")])


def test_cdf_handles_unsorted_segments():
    res = _result((9, 10), (0, 1), (5, 6), (2, 3))
    c = gv.vad_gap_cdf(res, cuts_ms=[2500.0])
    res_sorted = _result((0, 1), (2, 3), (5, 6), (9, 10))
    c_sorted = gv.vad_gap_cdf(res_sorted, cuts_ms=[2500.0])
    assert c["cuts"] == c_sorted["cuts"]


# ---- renderer: human-readable -------------------------------------------


def test_render_human_shape():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    lines = gv.render_vad_gap_cdf(res)
    text = "\n".join(lines)
    assert "silero VAD gap merge-CDF — rec.wav" in lines[0]
    assert any("segments:" in ln for ln in lines)
    assert any("gaps:" in ln for ln in lines)
    assert any("merge%" in ln for ln in lines)
    # Names the actionable knob.
    assert "--min-silence-ms" in text


def test_render_human_no_gaps():
    res = _result((0, 1))
    lines = gv.render_vad_gap_cdf(res)
    assert any("fewer than 2 segments" in ln for ln in lines)
    assert not any("merge%" in ln for ln in lines)


def test_render_human_unavailable():
    lines = gv.render_vad_gap_cdf(None)
    assert len(lines) == 1
    assert "silero VAD unavailable" in lines[0]


def test_render_human_custom_cut_labels():
    res = _result((0, 1), (2, 3), (5, 6))
    lines = gv.render_vad_gap_cdf(res, cuts_ms=[250.0, 750.5])
    text = "\n".join(lines)
    assert "250" in text
    assert "750.5" in text  # fractional cut keeps its decimals


# ---- renderer: human golden output --------------------------------------
#
# The shape tests above assert structure + substrings. They do NOT pin the
# EXACT rendered block, so a silent alignment/label/header regression — a column
# drifting by a space, the ``%g`` cut spelling losing its compact form, the
# merged ``m/n`` count field or the ``merge%`` percentage column shifting —
# would slip through every one of them. These goldens freeze the byte-for-byte
# report for fixed stub segmentations, so the human face can only change
# deliberately.


def test_render_human_golden_default_cuts():
    # Gaps (sorted): [1.0, 2.0, 4.0] seconds; n=3. Cuts 200/400/800/1600 ms all
    # sit below the 1.0s min gap, so every default cut merges 0/3 — the table
    # row alignment is what this golden pins, not a varying merge count.
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    lines = gv.render_vad_gap_cdf(res)
    assert lines == [
        "silero VAD gap merge-CDF — rec.wav",
        "  segments:     4",
        "  gaps:         3 (pauses between consecutive speech regions)",
        "  min gap:      1.000s (shortest real pause — keep --min-silence-ms "
        "below this to avoid merging turns)",
        "  mean gap:     2.333s",
        "  max gap:      4.000s",
        "  total silence:   7.000s",
        "  cut (ms)  cut (s)    merged   merge%",
        "       200    0.200       0/3     0.0%",
        "       400    0.400       0/3     0.0%",
        "       800    0.800       0/3     0.0%",
        "      1600    1.600       1/3    33.3%",
    ]


def test_render_human_golden_varied_merge_fractions():
    # Gaps (sorted): [1.0, 2.0, 4.0]. Cuts chosen to land 0 / 1 / 2 / 3 merges so
    # the merged ``m/n`` count column and the ``merge%`` percentage column are
    # each exercised at a distinct value (0.0 / 33.3 / 66.7 / 100.0%). The
    # fractional 1500.5 cut also pins the ``%g`` compact label spelling.
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    lines = gv.render_vad_gap_cdf(res, cuts_ms=[500.0, 1500.5, 3000.0, 5000.0])
    assert lines == [
        "silero VAD gap merge-CDF — rec.wav",
        "  segments:     4",
        "  gaps:         3 (pauses between consecutive speech regions)",
        "  min gap:      1.000s (shortest real pause — keep --min-silence-ms "
        "below this to avoid merging turns)",
        "  mean gap:     2.333s",
        "  max gap:      4.000s",
        "  total silence:   7.000s",
        "  cut (ms)  cut (s)    merged   merge%",
        "       500    0.500       0/3     0.0%",
        "    1500.5    1.500       1/3    33.3%",
        "      3000    3.000       2/3    66.7%",
        "      5000    5.000       3/3   100.0%",
    ]
    # The cut-label column right-aligns: the widest label (``1500.5``) and the
    # shorter ``500`` share the same right edge in the ``{:>8}`` field.
    row_500 = next(ln for ln in lines if " 500    0.500" in ln)
    row_1500 = next(ln for ln in lines if "1500.5" in ln)
    assert row_500.index("0.500") == row_1500.index("1.500")


def test_render_human_golden_single_segment_block():
    # The <2-segment branch: stats header truncates at the explanatory line,
    # WITHOUT the cut table (no pauses to sample a CDF against).
    res = _result((0, 1))
    lines = gv.render_vad_gap_cdf(res)
    assert lines == [
        "silero VAD gap merge-CDF — rec.wav",
        "  segments:     1",
        "  gaps:         0 (pauses between consecutive speech regions)",
        "  (fewer than 2 segments — no inter-segment pause to measure)",
    ]


# ---- renderer: JSON -----------------------------------------------------


def test_render_json_shape():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    payload = json.loads(gv.render_vad_gap_cdf_json(res))
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["num_segments"] == 4
    assert payload["num_gaps"] == 3
    assert [e["cut_ms"] for e in payload["cuts"]] == [200.0, 400.0, 800.0, 1600.0]
    assert all("merge_fraction" in e for e in payload["cuts"])
    assert all("keep_fraction" in e for e in payload["cuts"])


def test_render_json_empty_cuts_for_no_gaps():
    res = _result((0, 1))
    payload = json.loads(gv.render_vad_gap_cdf_json(res))
    assert payload["cuts"] == []
    assert payload["min_gap_s"] is None


def test_render_json_unavailable():
    payload = json.loads(gv.render_vad_gap_cdf_json(None))
    assert payload["available"] is False
    assert "hint" in payload


# ---- renderer: CSV ------------------------------------------------------


def test_render_csv_shape():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    text = gv.render_vad_gap_cdf_csv(res)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["cut_ms", "cut_s", "merged", "merge_fraction"]
    assert len(rows) == 1 + 4  # header + 4 default cuts
    assert rows[1][0] == "200"


def test_render_csv_matches_json():
    res = _result((0, 1), (2, 3), (6, 7), (15, 16))
    payload = json.loads(gv.render_vad_gap_cdf_json(res))
    text = gv.render_vad_gap_cdf_csv(res)
    rows = list(csv.reader(io.StringIO(text)))[1:]
    assert len(rows) == len(payload["cuts"])
    for row, entry in zip(rows, payload["cuts"]):
        assert int(row[2]) == entry["merged"]
        assert float(row[3]) == pytest.approx(entry["merge_fraction"])


def test_render_csv_header_only_for_no_gaps():
    res = _result((0, 1))
    text = gv.render_vad_gap_cdf_csv(res)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows == [["cut_ms", "cut_s", "merged", "merge_fraction"]]


def test_render_csv_unavailable():
    text = gv.render_vad_gap_cdf_csv(None)
    assert text.startswith("# silero VAD unavailable")


# ---- handler: cmd_vad_gap_cdf -------------------------------------------


def _run(args, **kw):
    lines: List[str] = []
    gv.cmd_vad_gap_cdf(args, log=lines.append, **kw)
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
    assert any("merge%" in ln for ln in lines)


def test_handler_json():
    res = _result((0, 1), (2, 3), (5, 6))
    lines = _run(
        _args(json=True),
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is True
    assert len(payload["cuts"]) == 4


def test_handler_csv():
    res = _result((0, 1), (2, 3), (5, 6))
    lines = _run(
        _args(cuts_ms=[800.0], csv=True),
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["cut_ms", "cut_s", "merged", "merge_fraction"]


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


def test_handler_passes_cuts_through():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    lines = _run(
        _args(cuts_ms=[100.0, 900.0], json=True),
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert [e["cut_ms"] for e in payload["cuts"]] == [100.0, 900.0]


def test_handler_builds_params_from_knobs():
    res = _result((0, 1), (2, 3))
    captured = {}

    class _Params:
        def __init__(self, **kw):
            captured.update(kw)

    args = _args(
        cuts_ms=[800.0],
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
