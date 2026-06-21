"""Tests for iter-338 — the ``gv vad-gap-percentiles`` subcommand (examples/gv.py).

``gv vad-gaps`` (iter-328) summarises the inter-segment silence distribution
with min/mean/max — but each is fragile to a single outlier pause: one unusually
long between-paragraph silence drags the max (and the mean) up, hiding where the
bulk of the pauses actually sit. ``gv vad-gap-percentiles`` reports robust
percentiles (p50/p90/p99 by default) instead: a percentile is unmoved by a lone
outlier, so the median is the typical pause an operator sets the end-of-turn
hangover (``--min-silence-ms`` / the live ``chat.vad.silence_duration``) below to
never merge a typical turn, and p90/p99 size the long tail.

Like the rest of the VAD-analysis family, the handler takes injected
``segmenter`` / ``availability`` / ``log`` dependencies so every test runs
WITHOUT importing torch / silero-vad and without touching real audio — fast and
deterministic on the x86_64 Linux runner. The pure core (``vad_gap_percentiles``)
and the three renderers are exercised directly against lightweight stand-ins
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


# ---- parser: percentile_list_type validator ----------------------------


def test_percentile_list_type_parses():
    assert gv.percentile_list_type("50,90,99") == [50.0, 90.0, 99.0]


def test_percentile_list_type_preserves_order_and_dups():
    assert gv.percentile_list_type("90,50,90") == [90.0, 50.0, 90.0]


def test_percentile_list_type_strips_whitespace():
    assert gv.percentile_list_type(" 25 , 75 ") == [25.0, 75.0]


def test_percentile_list_type_allows_100():
    assert gv.percentile_list_type("100") == [100.0]


def test_percentile_list_type_rejects_zero():
    with pytest.raises(gv.argparse.ArgumentTypeError):
        gv.percentile_list_type("0")


def test_percentile_list_type_rejects_over_100():
    with pytest.raises(gv.argparse.ArgumentTypeError):
        gv.percentile_list_type("101")


def test_percentile_list_type_rejects_negative():
    with pytest.raises(gv.argparse.ArgumentTypeError):
        gv.percentile_list_type("-5")


def test_percentile_list_type_rejects_nan():
    with pytest.raises(gv.argparse.ArgumentTypeError):
        gv.percentile_list_type("nan")


def test_percentile_list_type_rejects_nonnumber():
    with pytest.raises(gv.argparse.ArgumentTypeError):
        gv.percentile_list_type("50,abc")


def test_percentile_list_type_rejects_empty():
    with pytest.raises(gv.argparse.ArgumentTypeError):
        gv.percentile_list_type("")
    with pytest.raises(gv.argparse.ArgumentTypeError):
        gv.percentile_list_type(",,")


def test_percentile_list_type_rejects_non_string():
    with pytest.raises(gv.argparse.ArgumentTypeError):
        gv.percentile_list_type(50)


# ---- parser: registration & defaults -----------------------------------


def test_vad_gap_percentiles_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-gap-percentiles"] is gv.cmd_vad_gap_percentiles


def test_parser_registers_vad_gap_percentiles():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-percentiles", "rec.wav"])
    assert args.command == "vad-gap-percentiles"
    assert args.wav == "rec.wav"


def test_parser_defaults_mirror_vad_gaps_knobs():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-percentiles", "rec.wav"])
    # Shares the gv vad segmenter knobs.
    assert args.threshold == pytest.approx(0.5)
    assert args.min_speech_ms == pytest.approx(250.0)
    assert args.min_silence_ms == pytest.approx(800.0)
    assert args.speech_pad_ms == pytest.approx(30.0)
    assert math.isinf(args.max_speech_s)
    # The percentiles default.
    assert args.percentiles == [50.0, 90.0, 99.0]
    assert args.json is False
    assert args.csv is False


def test_parser_accepts_custom_percentiles():
    parser = gv.build_parser()
    args = parser.parse_args(
        ["vad-gap-percentiles", "rec.wav", "--percentiles", "25,50,75,95"]
    )
    assert args.percentiles == [25.0, 50.0, 75.0, 95.0]


def test_parser_rejects_bad_percentiles():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-percentiles", "rec.wav", "--percentiles", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-percentiles", "rec.wav", "--percentiles", "150"])


def test_parser_json_csv_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-percentiles", "rec.wav", "--json", "--csv"])


def test_parser_rejects_out_of_range_threshold():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-percentiles", "rec.wav", "--threshold", "1.5"])


# ---- pure core: vad_gap_percentiles -------------------------------------


def test_percentiles_basic_interpolation():
    # Segments yield gaps 1.0, 2.0, 3.0, 4.0 (sorted).
    res = _result((0, 1), (2, 3), (5, 6), (9, 10), (14, 15))
    s = gv.vad_gap_percentiles(res, percentiles=[50.0])
    assert s["num_segments"] == 5
    assert s["num_gaps"] == 4
    # gaps = [1,2,3,4]; p50 → rank (0.5)*(3) = 1.5 → between gaps[1]=2 and
    # gaps[2]=3, frac 0.5 → 2.5.
    assert s["percentiles"] == [{"p": 50.0, "value_s": 2.5}]


def test_percentiles_p0_endpoint_equals_min_p100_equals_max():
    res = _result((0, 1), (2, 3), (5, 6), (9, 10))  # gaps 1.0, 2.0, 3.0
    s = gv.vad_gap_percentiles(res, percentiles=[100.0])
    # p100 → rank (1.0)*(2) = 2 → gaps[2] = 3.0 == max.
    assert s["percentiles"][0]["value_s"] == s["max_gap_s"] == 3.0


def test_percentiles_order_preserved():
    res = _result((0, 1), (2, 3), (5, 6), (9, 10))  # gaps 1,2,3
    s = gv.vad_gap_percentiles(res, percentiles=[90.0, 50.0, 10.0])
    assert [e["p"] for e in s["percentiles"]] == [90.0, 50.0, 10.0]


def test_percentiles_monotonic_non_decreasing_in_p():
    # For ascending percentiles, the values must be non-decreasing.
    res = _result((0, 1), (2, 3), (6, 7), (15, 16), (30, 31))  # gaps 1,3,8,14
    s = gv.vad_gap_percentiles(res, percentiles=[10.0, 50.0, 90.0, 99.0])
    vals = [e["value_s"] for e in s["percentiles"]]
    assert vals == sorted(vals)


def test_percentiles_robust_to_outlier():
    # One huge outlier gap should NOT move the median, but DOES move the max.
    # gaps without outlier: 1,1,1,1 ; add one 100.0 → median still ~1.0.
    res = _result((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (109, 110))
    # gaps = [1,1,1,1,100]
    s = gv.vad_gap_percentiles(res, percentiles=[50.0])
    assert s["max_gap_s"] == 100.0
    assert s["mean_gap_s"] > 5.0  # mean dragged up by the outlier
    assert s["percentiles"][0]["value_s"] == 1.0  # median unmoved


def test_percentiles_single_gap_returns_that_gap_for_all():
    res = _result((0, 1), (3.5, 4))  # one gap = 2.5
    s = gv.vad_gap_percentiles(res, percentiles=[1.0, 50.0, 99.0])
    assert [e["value_s"] for e in s["percentiles"]] == [2.5, 2.5, 2.5]


def test_percentiles_anchors_to_vad_silence_gaps():
    res = _result((0, 1), (2.5, 3), (5, 6.5))
    s = gv.vad_gap_percentiles(res)
    d = gv.vad_silence_gaps(res)
    for key in ("num_segments", "num_gaps", "min_gap_s", "max_gap_s",
                "mean_gap_s", "total_silence_s"):
        assert s[key] == d[key]


def test_percentiles_empty_for_fewer_than_two_segments():
    res = _result((0, 1))
    s = gv.vad_gap_percentiles(res)
    assert s["num_gaps"] == 0
    assert s["percentiles"] == []
    assert s["min_gap_s"] is None
    assert s["max_gap_s"] is None
    assert s["mean_gap_s"] is None


def test_percentiles_empty_for_zero_segments():
    res = _result()
    s = gv.vad_gap_percentiles(res)
    assert s["num_segments"] == 0
    assert s["percentiles"] == []


def test_percentiles_values_rounded_to_three_places():
    res = _result((0, 1), (2, 3), (5.001, 6))  # gaps 1.0, 2.001
    s = gv.vad_gap_percentiles(res, percentiles=[50.0])
    # p50 of [1.0, 2.001] → rank 0.5 → 1.0 + 0.5*(1.001) = 1.5005 → 1.5 or 1.501
    val = s["percentiles"][0]["value_s"]
    assert round(val, 3) == val


def test_percentiles_default_is_50_90_99():
    res = _result((0, 1), (2, 3), (5, 6))
    s = gv.vad_gap_percentiles(res)
    assert [e["p"] for e in s["percentiles"]] == [50.0, 90.0, 99.0]


def test_percentiles_rejects_empty_percentiles():
    res = _result((0, 1), (2, 3))
    with pytest.raises(ValueError):
        gv.vad_gap_percentiles(res, percentiles=[])


def test_percentiles_rejects_out_of_range():
    res = _result((0, 1), (2, 3))
    with pytest.raises(ValueError):
        gv.vad_gap_percentiles(res, percentiles=[0.0])
    with pytest.raises(ValueError):
        gv.vad_gap_percentiles(res, percentiles=[101.0])


def test_percentiles_rejects_nan():
    res = _result((0, 1), (2, 3))
    with pytest.raises(ValueError):
        gv.vad_gap_percentiles(res, percentiles=[float("nan")])


def test_percentiles_handles_unsorted_segments():
    # Out-of-order input: same gaps as sorted, percentiles identical.
    res = _result((9, 10), (0, 1), (5, 6), (2, 3))
    s = gv.vad_gap_percentiles(res, percentiles=[50.0])
    res_sorted = _result((0, 1), (2, 3), (5, 6), (9, 10))
    s_sorted = gv.vad_gap_percentiles(res_sorted, percentiles=[50.0])
    assert s["percentiles"] == s_sorted["percentiles"]


# ---- renderer: human-readable -------------------------------------------


def test_render_human_shape():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    lines = gv.render_vad_gap_percentiles(res)
    text = "\n".join(lines)
    assert "silero VAD gap percentiles — rec.wav" in lines[0]
    assert any("segments:" in ln for ln in lines)
    assert any("gaps:" in ln for ln in lines)
    assert any("p50" in ln for ln in lines)
    assert any("p90" in ln for ln in lines)
    assert any("p99" in ln for ln in lines)
    # Names the actionable knob on the median line.
    assert "--min-silence-ms" in text


def test_render_human_no_gaps():
    res = _result((0, 1))
    lines = gv.render_vad_gap_percentiles(res)
    assert any("fewer than 2 segments" in ln for ln in lines)
    assert not any("p50" in ln for ln in lines)


def test_render_human_unavailable():
    lines = gv.render_vad_gap_percentiles(None)
    assert len(lines) == 1
    assert "silero VAD unavailable" in lines[0]


def test_render_human_custom_percentile_labels():
    res = _result((0, 1), (2, 3), (5, 6))
    lines = gv.render_vad_gap_percentiles(res, percentiles=[25.0, 99.5])
    text = "\n".join(lines)
    assert "p25" in text
    assert "p99.5" in text  # fractional percentile keeps its decimals


# ---- renderer: human golden output (iter-340) ---------------------------
#
# The shape tests above assert structure + substrings (a p50 line exists, the
# knob is named *somewhere*). They do NOT pin the EXACT rendered block, so a
# silent alignment/label/suffix regression — a column drifting by a space, the
# median knob-hint moving onto p90, ``%g`` losing its compact spelling — would
# slip through every one of them. These two goldens freeze the byte-for-byte
# report for two fixed stub segmentations, so the human face of the surface
# can only change deliberately (the analogue of iter-339's next-item #1: "pin
# the exact aligned ``gv vad-gap-percentiles`` ``pNN`` block").


def test_render_human_golden_default_percentiles():
    # Gaps (sorted): [1.0, 2.0, 4.0]; n=3 so the R-7 ranks land p50 -> gaps[1]
    # exactly (2.000), p90/p99 interpolate up toward the 4.000 max. Pins the
    # aggregate header column, the ``pNN`` value alignment (``{label:<5}`` +
    # ``{value:7.3f}``), and the median-only knob-hint suffix placement.
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    lines = gv.render_vad_gap_percentiles(res)
    assert lines == [
        "silero VAD gap percentiles — rec.wav",
        "  segments:     4",
        "  gaps:         3 (pauses between consecutive speech regions)",
        "  min gap:      1.000s",
        "  mean gap:     2.333s",
        "  max gap:      4.000s",
        "  total silence:   7.000s",
        "  p50     2.000s  (typical pause — keep --min-silence-ms below this "
        "to avoid merging turns)",
        "  p90     3.600s",
        "  p99     3.960s",
    ]


def test_render_human_golden_fractional_label_alignment():
    # A 5-char label (``p99.5``) is the widest the ``{label:<5}`` field is
    # designed around: it consumes the pad exactly, so its value column must
    # still line up with the shorter ``p25`` row above it. No median requested,
    # so NO knob-hint suffix appears on any line.
    res = _result((0, 1), (2, 3), (5, 6))
    lines = gv.render_vad_gap_percentiles(res, percentiles=[25.0, 99.5])
    assert lines == [
        "silero VAD gap percentiles — rec.wav",
        "  segments:     3",
        "  gaps:         2 (pauses between consecutive speech regions)",
        "  min gap:      1.000s",
        "  mean gap:     1.500s",
        "  max gap:      2.000s",
        "  total silence:   3.000s",
        "  p25     1.250s",
        "  p99.5   1.995s",
    ]
    # The value columns align: both ``value_s`` numbers start at the same offset.
    p25_line = next(ln for ln in lines if ln.lstrip().startswith("p25"))
    p995_line = next(ln for ln in lines if ln.lstrip().startswith("p99.5"))
    assert p25_line.index("1.250") == p995_line.index("1.995")


# ---- renderer: JSON -----------------------------------------------------


def test_render_json_shape():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    payload = json.loads(gv.render_vad_gap_percentiles_json(res))
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["num_segments"] == 4
    assert payload["num_gaps"] == 3
    assert [e["p"] for e in payload["percentiles"]] == [50.0, 90.0, 99.0]
    assert all("value_s" in e for e in payload["percentiles"])


def test_render_json_empty_percentiles_for_no_gaps():
    res = _result((0, 1))
    payload = json.loads(gv.render_vad_gap_percentiles_json(res))
    assert payload["percentiles"] == []
    assert payload["min_gap_s"] is None


def test_render_json_unavailable():
    payload = json.loads(gv.render_vad_gap_percentiles_json(None))
    assert payload["available"] is False
    assert "hint" in payload


# ---- renderer: CSV ------------------------------------------------------


def test_render_csv_shape():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    text = gv.render_vad_gap_percentiles_csv(res)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["percentile", "value_s"]
    assert len(rows) == 1 + 3  # header + 3 default percentiles
    assert rows[1][0] == "50"


def test_render_csv_matches_json():
    res = _result((0, 1), (2, 3), (6, 7), (15, 16))
    payload = json.loads(gv.render_vad_gap_percentiles_json(res))
    text = gv.render_vad_gap_percentiles_csv(res)
    rows = list(csv.reader(io.StringIO(text)))[1:]
    assert len(rows) == len(payload["percentiles"])
    for row, entry in zip(rows, payload["percentiles"]):
        assert float(row[1]) == pytest.approx(entry["value_s"])


def test_render_csv_header_only_for_no_gaps():
    res = _result((0, 1))
    text = gv.render_vad_gap_percentiles_csv(res)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows == [["percentile", "value_s"]]


def test_render_csv_unavailable():
    text = gv.render_vad_gap_percentiles_csv(None)
    assert text.startswith("# silero VAD unavailable")


# ---- handler: cmd_vad_gap_percentiles -----------------------------------


def _run(args, **kw):
    lines: List[str] = []
    gv.cmd_vad_gap_percentiles(args, log=lines.append, **kw)
    return lines


def test_handler_human(monkeypatch):
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    captured = {}

    def segmenter(wav, *, params):
        captured["wav"] = wav
        captured["params"] = params
        return res

    args = SimpleNamespace(
        wav="rec.wav",
        percentiles=[50.0, 90.0, 99.0],
        threshold=0.5,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        json=False,
        csv=False,
    )
    lines = _run(args, segmenter=segmenter, availability=lambda: True)
    assert captured["wav"] == "rec.wav"
    assert any("p50" in ln for ln in lines)


def test_handler_json(monkeypatch):
    res = _result((0, 1), (2, 3), (5, 6))
    args = SimpleNamespace(
        wav="rec.wav",
        percentiles=[50.0, 90.0, 99.0],
        threshold=0.5,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        json=True,
        csv=False,
    )
    lines = _run(args, segmenter=lambda w, *, params: res, availability=lambda: True)
    payload = json.loads("\n".join(lines))
    assert payload["available"] is True
    assert len(payload["percentiles"]) == 3


def test_handler_csv():
    res = _result((0, 1), (2, 3), (5, 6))
    args = SimpleNamespace(
        wav="rec.wav",
        percentiles=[50.0],
        threshold=0.5,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        json=False,
        csv=True,
    )
    lines = _run(args, segmenter=lambda w, *, params: res, availability=lambda: True)
    text = "\n".join(lines)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["percentile", "value_s"]


def test_handler_unavailable_human():
    args = SimpleNamespace(
        wav="rec.wav",
        percentiles=[50.0, 90.0, 99.0],
        threshold=0.5,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        json=False,
        csv=False,
    )
    called = []
    lines = _run(
        args,
        segmenter=lambda *a, **k: called.append(1),
        availability=lambda: False,
    )
    assert not called  # segmenter never invoked when unavailable
    assert any("silero VAD unavailable" in ln for ln in lines)


def test_handler_unavailable_json():
    args = SimpleNamespace(
        wav="rec.wav",
        percentiles=[50.0],
        threshold=0.5,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        json=True,
        csv=False,
    )
    lines = _run(args, segmenter=lambda *a, **k: None, availability=lambda: False)
    payload = json.loads("\n".join(lines))
    assert payload["available"] is False


def test_handler_passes_percentiles_through():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    args = SimpleNamespace(
        wav="rec.wav",
        percentiles=[25.0, 75.0],
        threshold=0.5,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        json=True,
        csv=False,
    )
    lines = _run(args, segmenter=lambda w, *, params: res, availability=lambda: True)
    payload = json.loads("\n".join(lines))
    assert [e["p"] for e in payload["percentiles"]] == [25.0, 75.0]


def test_handler_builds_params_from_knobs():
    res = _result((0, 1), (2, 3))
    captured = {}

    class _Params:
        def __init__(self, **kw):
            captured.update(kw)

    args = SimpleNamespace(
        wav="rec.wav",
        percentiles=[50.0],
        threshold=0.7,
        min_speech_ms=100.0,
        min_silence_ms=400.0,
        speech_pad_ms=10.0,
        max_speech_s=5.0,
        json=True,
        csv=False,
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
