"""Tests for iter-372 — the ``gv vad-gap-recommend-knob-sweep`` subcommand (examples/gv.py).

iter-352's ``gv vad-gap-recommend-sweep`` sweeps the BIAS (short/balanced/long)
over ONE segmentation, naming the whole spread of defensible end-of-turn hangovers
plus the iter-348 confidence grade. This surface is its knob-sweep companion: it
sweeps a SEGMENTER knob (the gate ``--thresholds`` or one of the ms/seconds region
knobs) and reports, at EACH swept value, the whole bias spread AND the confidence
grade — so an operator sees not just which hangover to pick but at WHICH knob
setting the recommendation becomes trustworthy. It is to
``vad-gap-recommend-sweep`` what ``vad-gap-sweep`` is to ``vad-gaps``.

Each row is anchored to :func:`vad_gap_recommend_sweep` over that value's
segmentation, so the per-row bias spread, valley accounting, and grade agree
EXACTLY with ``gv vad-gap-recommend-sweep`` (and ``gv vad-gap-confidence``) at the
matching value.

Like the rest of the VAD-analysis family, the handler takes injected
``segmenter`` / ``availability`` / ``log`` dependencies so every test runs WITHOUT
importing torch / silero-vad and without touching real audio.
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
    return _result(
        (0, 1), (1.3, 2), (2.3, 3), (3.3, 4), (6, 7), name=name
    )


# Three segments, both gaps the same 1.0s → no valley (grades "none").
def _no_valley(name="rec.wav"):
    return _result((0, 1), (2, 3), (4, 5), name=name)


# A single segment → no gaps, nothing to recommend over.
def _single(name="rec.wav"):
    return _result((0, 1), name=name)


# ---- core: vad_gap_recommend_knob_sweep ----------------------------------


def test_core_row_per_value_keyed_by_axis():
    res = [_bimodal(), _bimodal()]
    rows = gv.vad_gap_recommend_knob_sweep([0.3, 0.7], res, axis="threshold")
    assert [r["threshold"] for r in rows] == [0.3, 0.7]


def test_core_each_row_matches_recommend_sweep_exactly():
    res = _bimodal()
    rows = gv.vad_gap_recommend_knob_sweep([0.5], [res])
    s = gv.vad_gap_recommend_sweep(res)
    row = rows[0]
    assert row["biases"] == s["biases"]
    assert row["spread_ms"] == s["spread_ms"]
    assert row["spread_s"] == s["spread_s"]
    assert row["split_found"] == s["split_found"]
    assert row["num_segments"] == s["num_segments"]
    assert row["num_gaps"] == s["num_gaps"]


def test_core_grade_matches_confidence_surface():
    # The genuinely-new signal: the confidence grade per row agrees exactly with
    # the single confidence surface over the same segmentation.
    res = _bimodal()
    rows = gv.vad_gap_recommend_knob_sweep([0.5], [res])
    c = gv.vad_gap_confidence(res)
    assert rows[0]["grade"] == c["grade"] == "strong"
    assert rows[0]["dominance"] == c["dominance"]
    assert rows[0]["separation_ratio"] == c["separation_ratio"]


def test_core_carries_all_three_biases_in_order():
    rows = gv.vad_gap_recommend_knob_sweep([0.5], [_bimodal()])
    assert [b["bias"] for b in rows[0]["biases"]] == ["short", "balanced", "long"]


def test_core_no_valley_row_grades_none():
    rows = gv.vad_gap_recommend_knob_sweep([0.5], [_no_valley()])
    row = rows[0]
    assert row["split_found"] is False
    assert row["grade"] == "none"
    assert row["dominance"] is None
    assert row["separation_ratio"] is None


def test_core_no_gaps_row_has_none_recommendations():
    rows = gv.vad_gap_recommend_knob_sweep([0.5], [_single()])
    row = rows[0]
    assert row["num_gaps"] == 0
    assert row["spread_ms"] is None
    assert row["spread_s"] is None
    for b in row["biases"]:
        assert b["recommended_ms"] is None
        assert b["recommended_s"] is None


def test_core_grade_can_shift_across_swept_values():
    # The whole point of the knob sweep: different segmentations at different
    # knob values can yield different grades. A bimodal value grades strong; a
    # uniform value grades none — they sit in the same sweep.
    rows = gv.vad_gap_recommend_knob_sweep(
        [0.3, 0.7], [_bimodal(), _no_valley()]
    )
    assert rows[0]["grade"] == "strong"
    assert rows[1]["grade"] == "none"


def test_core_axis_key_follows_axis_argument():
    rows = gv.vad_gap_recommend_knob_sweep(
        [200.0], [_bimodal()], axis="min_silence_ms"
    )
    assert "min_silence_ms" in rows[0]
    assert "threshold" not in rows[0]


def test_core_length_mismatch_raises():
    with pytest.raises(ValueError):
        gv.vad_gap_recommend_knob_sweep([0.3, 0.7], [_bimodal()])


# ---- human renderer ------------------------------------------------------


def test_render_human_header_and_columns():
    lines = gv.render_vad_gap_recommend_knob_sweep(
        [0.3, 0.7], [_bimodal(), _bimodal()], name="rec.wav"
    )
    text = "\n".join(lines)
    assert "recommended-hangover knob sweep" in text
    assert "short" in text
    assert "balanced" in text
    assert "confidence" in text


def test_render_human_shows_grade_per_row():
    lines = gv.render_vad_gap_recommend_knob_sweep(
        [0.3, 0.7], [_bimodal(), _no_valley()], name="rec.wav"
    )
    text = "\n".join(lines)
    assert "strong" in text
    assert "none" in text


def test_render_human_no_gaps_row_dashes():
    lines = gv.render_vad_gap_recommend_knob_sweep(
        [0.5], [_single()], name="rec.wav"
    )
    # The recommendation columns collapse to dashes for a <2-segment row.
    assert any("-" in ln for ln in lines[2:])


def test_render_human_unavailable_hint():
    lines = gv.render_vad_gap_recommend_knob_sweep([], [None], name="rec.wav")
    assert any("silero VAD unavailable" in ln for ln in lines)


# ---- JSON renderer -------------------------------------------------------


def test_render_json_shape():
    text = gv.render_vad_gap_recommend_knob_sweep_json(
        [0.3, 0.7], [_bimodal(), _no_valley()], name="rec.wav", axis="threshold"
    )
    payload = json.loads(text)
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["axis"] == "threshold"
    assert len(payload["sweep"]) == 2
    assert payload["sweep"][0]["grade"] == "strong"
    assert payload["sweep"][1]["grade"] == "none"
    # The per-bias spread is carried per row.
    assert [b["bias"] for b in payload["sweep"][0]["biases"]] == [
        "short",
        "balanced",
        "long",
    ]


def test_render_json_no_gaps_nulls():
    text = gv.render_vad_gap_recommend_knob_sweep_json(
        [0.5], [_single()], name="rec.wav"
    )
    row = json.loads(text)["sweep"][0]
    assert row["spread_ms"] is None
    assert row["dominance"] is None


def test_render_json_unavailable():
    text = gv.render_vad_gap_recommend_knob_sweep_json([], [None], name="rec.wav")
    payload = json.loads(text)
    assert payload["available"] is False
    assert "hint" in payload


# ---- CSV renderer --------------------------------------------------------


def test_render_csv_header_and_one_row_per_value():
    text = gv.render_vad_gap_recommend_knob_sweep_csv(
        [0.3, 0.7], [_bimodal(), _no_valley()], name="rec.wav", axis="threshold"
    )
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == [
        "threshold",
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
    # One row per swept value.
    assert len(rows) == 3
    assert rows[1][0] == "0.3"
    assert rows[1][7] == "strong"
    assert rows[2][7] == "none"


def test_render_csv_axis_header_follows_axis():
    text = gv.render_vad_gap_recommend_knob_sweep_csv(
        [200.0], [_bimodal()], name="rec.wav", axis="min_silence_ms"
    )
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0][0] == "min_silence_ms"


def test_render_csv_no_gaps_empty_cells():
    text = gv.render_vad_gap_recommend_knob_sweep_csv(
        [0.5], [_single()], name="rec.wav"
    )
    rows = list(csv.reader(io.StringIO(text)))
    # short_ms..separation_ratio all empty for a <2-segment row.
    assert rows[1][3:] == ["", "", "", "", "", "", ""]


def test_render_csv_unavailable():
    text = gv.render_vad_gap_recommend_knob_sweep_csv([], [None], name="rec.wav")
    assert text.startswith("# silero VAD unavailable")


# ---- handler: cmd_vad_gap_recommend_knob_sweep ---------------------------


def _run_handler(results, argv_extra=None, segmenter=None):
    """Drive cmd_vad_gap_recommend_knob_sweep with an injected segmenter returning
    ``results`` in order (one per swept value)."""
    lines: List[str] = []
    argv = ["vad-gap-recommend-knob-sweep", "rec.wav", *(argv_extra or [])]
    args = gv.build_parser().parse_args(argv)
    it = iter(results)
    if segmenter is None:
        segmenter = lambda wav, params=None: next(it)  # noqa: E731
    gv.cmd_vad_gap_recommend_knob_sweep(
        args, log=lines.append, segmenter=segmenter, availability=lambda: True
    )
    return lines


def test_cmd_human_path():
    lines = _run_handler(
        [_bimodal(), _bimodal(), _bimodal(), _no_valley()]  # 4 default thresholds
    )
    text = "\n".join(lines)
    assert "recommended-hangover knob sweep" in text
    assert "confidence" in text


def test_cmd_json_path():
    lines = _run_handler(
        [_bimodal(), _no_valley()],
        argv_extra=["--thresholds", "0.3,0.9", "--json"],
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is True
    assert len(payload["sweep"]) == 2
    assert payload["sweep"][0]["grade"] == "strong"


def test_cmd_csv_path():
    lines = _run_handler(
        [_bimodal(), _no_valley()],
        argv_extra=["--thresholds", "0.3,0.9", "--csv"],
    )
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    assert rows[0][0] == "threshold"
    assert rows[0][7] == "grade"


def test_cmd_min_speeches_axis_switches_swept_dimension():
    seen = []

    def _seg(wav, params=None):
        seen.append(params.min_speech_ms)
        return _bimodal()

    lines = _run_handler(
        [None],
        argv_extra=["--min-speeches", "50,200", "--threshold", "0.7", "--json"],
        segmenter=_seg,
    )
    assert seen == [50.0, 200.0]
    payload = json.loads("\n".join(lines))
    assert payload["axis"] == "min_speech_ms"
    assert [r["min_speech_ms"] for r in payload["sweep"]] == [50.0, 200.0]


def test_cmd_min_silences_axis_holds_gate_fixed():
    seen = []

    def _seg(wav, params=None):
        seen.append((params.min_silence_ms, params.threshold))
        return _bimodal()

    _run_handler(
        [None],
        argv_extra=["--min-silences", "400,800", "--threshold", "0.6"],
        segmenter=_seg,
    )
    assert seen == [(400.0, 0.6), (800.0, 0.6)]


def test_cmd_threshold_axis_holds_other_knobs_fixed():
    captured = []

    def _seg(wav, params=None):
        captured.append((params.threshold, params.min_silence_ms))
        return _bimodal()

    _run_handler(
        [None],
        argv_extra=["--thresholds", "0.3,0.9", "--min-silence-ms", "500"],
        segmenter=_seg,
    )
    assert captured == [(0.3, 500.0), (0.9, 500.0)]


def test_cmd_max_speeches_seconds_axis():
    seen = []

    def _seg(wav, params=None):
        seen.append(params.max_speech_s)
        return _bimodal()

    lines = _run_handler(
        [None],
        argv_extra=["--max-speeches", "5,inf", "--json"],
        segmenter=_seg,
    )
    assert seen[0] == 5.0
    assert seen[1] == float("inf")
    payload = json.loads("\n".join(lines))
    assert payload["axis"] == "max_speech_s"


def test_cmd_uses_result_name_not_raw_path():
    lines = _run_handler(
        [_bimodal(name="clean.wav")],
        argv_extra=["--thresholds", "0.5", "--json"],
    )
    payload = json.loads("\n".join(lines))
    assert payload["name"] == "clean.wav"


def test_cmd_unavailable_human():
    lines: List[str] = []
    args = gv.build_parser().parse_args(
        ["vad-gap-recommend-knob-sweep", "rec.wav"]
    )
    gv.cmd_vad_gap_recommend_knob_sweep(
        args,
        log=lines.append,
        segmenter=lambda wav, params=None: pytest.fail("must not segment"),
        availability=lambda: False,
    )
    assert any("silero VAD unavailable" in ln for ln in lines)


def test_cmd_unavailable_json():
    lines: List[str] = []
    args = gv.build_parser().parse_args(
        ["vad-gap-recommend-knob-sweep", "rec.wav", "--json"]
    )
    gv.cmd_vad_gap_recommend_knob_sweep(
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
        ["vad-gap-recommend-knob-sweep", "rec.wav", "--csv"]
    )
    gv.cmd_vad_gap_recommend_knob_sweep(
        args,
        log=lines.append,
        segmenter=lambda wav, params=None: pytest.fail("must not segment"),
        availability=lambda: False,
    )
    assert any("silero VAD unavailable" in ln for ln in lines)


def test_cmd_json_and_csv_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-gap-recommend-knob-sweep", "rec.wav", "--json", "--csv"]
        )


def test_cmd_axes_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            [
                "vad-gap-recommend-knob-sweep",
                "rec.wav",
                "--thresholds",
                "0.3,0.5",
                "--min-silences",
                "400,800",
            ]
        )


def test_cmd_registered_in_dispatch_table():
    assert gv.DEFAULT_HANDLERS["vad-gap-recommend-knob-sweep"] is (
        gv.cmd_vad_gap_recommend_knob_sweep
    )


# ---- iter-374: --bias column filter --------------------------------------


def test_bias_list_type_canonical_order_and_dedup():
    # The parser type returns the selected biases in canonical short..balanced..
    # long order regardless of how they were typed, and collapses duplicates.
    assert gv.gap_recommend_bias_list_type("long,short") == ["short", "long"]
    assert gv.gap_recommend_bias_list_type("balanced") == ["balanced"]
    assert gv.gap_recommend_bias_list_type("long,short,long") == ["short", "long"]
    assert gv.gap_recommend_bias_list_type(" short , long ") == ["short", "long"]


def test_bias_list_type_rejects_empty_and_unknown():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        gv.gap_recommend_bias_list_type("")
    with pytest.raises(argparse.ArgumentTypeError):
        gv.gap_recommend_bias_list_type(",, ,")
    with pytest.raises(argparse.ArgumentTypeError):
        gv.gap_recommend_bias_list_type("medium")
    with pytest.raises(argparse.ArgumentTypeError):
        gv.gap_recommend_bias_list_type("short,bogus")


def test_render_human_default_biases_unchanged():
    # The full triad is byte-identical to passing no biases at all (the default).
    res = [_bimodal(), _no_valley()]
    default = gv.render_vad_gap_recommend_knob_sweep(
        [0.3, 0.7], res, name="rec.wav", axis="threshold"
    )
    explicit = gv.render_vad_gap_recommend_knob_sweep(
        [0.3, 0.7], res, name="rec.wav", axis="threshold",
        biases=["short", "balanced", "long"],
    )
    assert default == explicit
    # The header still names all three biases.
    assert "short" in default[1] and "balanced" in default[1] and "long" in default[1]


def test_render_human_filtered_subset_columns():
    lines = gv.render_vad_gap_recommend_knob_sweep(
        [0.3], [_bimodal()], name="rec.wav", axis="threshold",
        biases=["short", "long"],
    )
    header = lines[1]
    # Only the selected per-bias columns appear; balanced is dropped.
    assert "short" in header
    assert "long" in header
    assert "balanced" not in header
    # The invariant spread + confidence columns are always kept.
    assert "spread" in header
    assert "confidence" in header


def test_render_human_filtered_single_bias():
    lines = gv.render_vad_gap_recommend_knob_sweep(
        [0.3], [_bimodal()], name="rec.wav", biases=["balanced"],
    )
    header = lines[1]
    assert "balanced" in header
    assert "short" not in header and "long" not in header
    assert "spread" in header and "confidence" in header


def test_render_human_filtered_no_gaps_dashes():
    # A <2-segment row still dashes the (narrowed) recommendation columns.
    lines = gv.render_vad_gap_recommend_knob_sweep(
        [0.5], [_single()], name="rec.wav", biases=["short"],
    )
    assert any("-" in ln for ln in lines[2:])


def test_render_json_filtered_narrows_biases_and_names_them():
    text = gv.render_vad_gap_recommend_knob_sweep_json(
        [0.3], [_bimodal()], name="rec.wav", biases=["short", "long"],
    )
    payload = json.loads(text)
    # Top-level biases key records which columns were kept.
    assert payload["biases"] == ["short", "long"]
    # Each row's nested biases list is narrowed to the same subset, in order.
    assert [b["bias"] for b in payload["sweep"][0]["biases"]] == ["short", "long"]
    # The valley/confidence fields are untouched by the filter.
    assert payload["sweep"][0]["grade"] == "strong"
    assert payload["sweep"][0]["spread_ms"] is not None


def test_render_json_default_has_no_top_level_biases_key():
    # The unfiltered payload is unchanged: no top-level biases key, full triad.
    text = gv.render_vad_gap_recommend_knob_sweep_json(
        [0.3], [_bimodal()], name="rec.wav",
    )
    payload = json.loads(text)
    assert "biases" not in payload
    assert [b["bias"] for b in payload["sweep"][0]["biases"]] == [
        "short", "balanced", "long",
    ]


def test_render_csv_filtered_header_and_columns():
    text = gv.render_vad_gap_recommend_knob_sweep_csv(
        [0.3, 0.7], [_bimodal(), _no_valley()], name="rec.wav",
        biases=["short", "long"],
    )
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == [
        "threshold",
        "num_segments",
        "num_gaps",
        "short_ms",
        "long_ms",
        "spread_ms",
        "grade",
        "dominance",
        "separation_ratio",
    ]
    # short_ms, long_ms present; balanced_ms dropped.
    assert rows[1][3] != "" and rows[1][4] != ""


def test_render_csv_filtered_no_gaps_empty_cells():
    text = gv.render_vad_gap_recommend_knob_sweep_csv(
        [0.5], [_single()], name="rec.wav", biases=["short"],
    )
    rows = list(csv.reader(io.StringIO(text)))
    # short_ms..separation_ratio (5 cols for one bias) all empty for a <2-seg row.
    assert rows[1][3:] == ["", "", "", "", ""]


def test_cmd_bias_filter_human_path():
    lines = _run_handler(
        [_bimodal(), _bimodal(), _bimodal(), _no_valley()],
        argv_extra=["--bias", "short,long"],
    )
    text = "\n".join(lines)
    header = lines[1]
    assert "short" in header and "long" in header
    assert "balanced" not in header
    assert "confidence" in header


def test_cmd_bias_filter_json_path():
    lines = _run_handler(
        [_bimodal(), _bimodal(), _bimodal(), _no_valley()],
        argv_extra=["--json", "--bias", "long"],
    )
    payload = json.loads("\n".join(lines))
    assert payload["biases"] == ["long"]
    assert [b["bias"] for b in payload["sweep"][0]["biases"]] == ["long"]


def test_cmd_bias_filter_csv_path():
    lines = _run_handler(
        [_bimodal(), _bimodal(), _bimodal(), _no_valley()],
        argv_extra=["--csv", "--bias", "balanced"],
    )
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    assert rows[0][3] == "balanced_ms"
    assert "short_ms" not in rows[0] and "long_ms" not in rows[0]


def test_cmd_bias_typed_order_canonicalized_in_output():
    # Typing "long,short" still renders short before long (canonical order).
    lines = _run_handler(
        [_bimodal(), _bimodal(), _bimodal(), _no_valley()],
        argv_extra=["--csv", "--bias", "long,short"],
    )
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    assert rows[0][3] == "short_ms"
    assert rows[0][4] == "long_ms"


def test_cmd_invalid_bias_rejected():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-gap-recommend-knob-sweep", "rec.wav", "--bias", "medium"]
        )
