"""Tests for iter-352 — the ``gv vad-gap-recommend-sweep`` subcommand (examples/gv.py).

iter-351 gave ``gv vad-gap-recommend`` a ``--bias {short,balanced,long}`` knob that
slides the recommended end-of-turn hangover within the valley between the short
within-turn pauses and the long between-turn pauses. This surface is the natural
companion: where the single-bias command names ONE defensible number, the sweep
names ALL THREE side by side — plus the short→long spread — so the operator sees
the whole range of defensible numbers in one shot, without re-running the command
per bias. It is the ``vad-gap-recommend`` analogue of how ``gv vad-gap-sweep``
shows a whole ``--min-silence-ms`` sweep instead of one ``gv vad-gaps`` snapshot.

The valley is the EMPTY band the largest-gap split finds, so every interior point
splits the pauses identically: ``split_found`` / ``below`` / ``at_or_above`` / the
valley endpoints are INVARIANT across biases (only the named number shifts). The
sweep therefore reports those shared fields ONCE and a compact per-bias row.

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


# A clean bimodal recording: three short within-turn pauses (~0.3s) and one long
# between-turn pause (~2.0s). The widest jump (the valley) sits between 0.3 and
# 2.0, so the balanced midpoint is 1.15s; short biases to 0.725s, long to 1.575s.
def _bimodal():
    return _result((0, 1), (1.3, 2), (2.3, 3), (3.3, 4), (6, 7))


# ---- core: vad_gap_recommend_sweep --------------------------------------


def test_sweep_carries_all_three_biases_in_order():
    s = gv.vad_gap_recommend_sweep(_bimodal())
    assert [row["bias"] for row in s["biases"]] == ["short", "balanced", "long"]


def test_sweep_each_bias_matches_single_surface_exactly():
    res = _bimodal()
    s = gv.vad_gap_recommend_sweep(res)
    for row in s["biases"]:
        single = gv.vad_gap_recommend(res, bias=row["bias"])
        assert row["recommended_ms"] == single["recommended_ms"]
        assert row["recommended_s"] == single["recommended_s"]


def test_sweep_recommended_numbers_monotone_non_decreasing():
    s = gv.vad_gap_recommend_sweep(_bimodal())
    nums = [row["recommended_s"] for row in s["biases"]]
    assert nums[0] < nums[1] < nums[2]


def test_sweep_shared_fields_match_balanced_single_surface():
    res = _bimodal()
    s = gv.vad_gap_recommend_sweep(res)
    base = gv.vad_gap_recommend(res, bias="balanced")
    for key in (
        "num_segments",
        "num_gaps",
        "min_gap_s",
        "max_gap_s",
        "mean_gap_s",
        "total_silence_s",
        "split_found",
        "below",
        "at_or_above",
        "gap_below_s",
        "gap_above_s",
        "valley_width_s",
    ):
        assert s[key] == base[key]


def test_sweep_merge_accounting_invariant_across_biases():
    # below / at_or_above are reported once and must equal the per-bias single
    # surface's accounting for EVERY bias (the valley is an empty band).
    res = _bimodal()
    s = gv.vad_gap_recommend_sweep(res)
    for bias in gv.GAP_RECOMMEND_BIAS_ORDER:
        single = gv.vad_gap_recommend(res, bias=bias)
        assert single["below"] == s["below"]
        assert single["at_or_above"] == s["at_or_above"]


def test_sweep_spread_is_long_minus_short():
    s = gv.vad_gap_recommend_sweep(_bimodal())
    lo = s["biases"][0]["recommended_s"]
    hi = s["biases"][-1]["recommended_s"]
    assert s["spread_s"] == pytest.approx(round(hi - lo, 3))
    assert s["spread_ms"] == pytest.approx(round((hi - lo) * 1000.0, 1))


def test_sweep_spread_positive_when_valley_found():
    s = gv.vad_gap_recommend_sweep(_bimodal())
    assert s["split_found"] is True
    assert s["spread_s"] > 0


def test_sweep_no_gaps_has_none_recommendations_and_spread():
    s = gv.vad_gap_recommend_sweep(_result((0, 1)))
    assert s["num_gaps"] == 0
    assert s["spread_s"] is None
    assert s["spread_ms"] is None
    for row in s["biases"]:
        assert row["recommended_ms"] is None
        assert row["recommended_s"] is None


def test_sweep_zero_segments_has_none_recommendations():
    s = gv.vad_gap_recommend_sweep(_result())
    assert s["num_segments"] == 0
    assert s["num_gaps"] == 0
    assert s["spread_s"] is None
    for row in s["biases"]:
        assert row["recommended_s"] is None


def test_sweep_no_valley_fallback_still_keeps_every_pause():
    # Two identical-length pauses → no valley; the no-valley fallback recommends
    # below the shortest pause, so every pause is kept regardless of bias.
    res = _result((0, 1), (2, 3), (4, 5))  # both gaps = 1.0s
    s = gv.vad_gap_recommend_sweep(res)
    assert s["split_found"] is False
    assert s["below"] == 0
    assert s["at_or_above"] == s["num_gaps"]
    # The named numbers still spread short < balanced < long (frac of min_gap).
    nums = [row["recommended_s"] for row in s["biases"]]
    assert nums[0] < nums[1] < nums[2]


def test_sweep_bias_order_constant():
    assert gv.GAP_RECOMMEND_BIAS_ORDER == ("short", "balanced", "long")
    assert set(gv.GAP_RECOMMEND_BIAS_ORDER) == set(gv.GAP_RECOMMEND_BIAS_FRACTIONS)


# ---- core: confidence grade folded in (iter-353) ------------------------


def test_sweep_carries_confidence_fields():
    s = gv.vad_gap_recommend_sweep(_bimodal())
    assert set(("grade", "dominance", "separation_ratio")) <= set(s)


def test_sweep_grade_matches_confidence_surface_exactly():
    # The grade is anchored to vad_gap_confidence, so it must agree field-for-field.
    res = _bimodal()
    s = gv.vad_gap_recommend_sweep(res)
    c = gv.vad_gap_confidence(res)
    assert s["grade"] == c["grade"]
    assert s["dominance"] == c["dominance"]
    assert s["separation_ratio"] == c["separation_ratio"]


def test_sweep_clean_bimodal_grades_strong():
    s = gv.vad_gap_recommend_sweep(_bimodal())
    assert s["grade"] == "strong"


def test_sweep_grade_invariant_across_biases():
    # The grade is a property of the valley, not of where in it the number sits;
    # it does not depend on bias. (Sanity: the single confidence surface ignores
    # bias entirely, and the sweep reports a single shared grade.)
    res = _bimodal()
    s = gv.vad_gap_recommend_sweep(res)
    # The grade sits in the shared block, not per-bias.
    assert "grade" not in s["biases"][0]
    assert s["grade"] == gv.vad_gap_confidence(res)["grade"]


def test_sweep_no_valley_grades_none():
    res = _result((0, 1), (2, 3), (4, 5))  # both gaps 1.0s → no valley
    s = gv.vad_gap_recommend_sweep(res)
    assert s["split_found"] is False
    assert s["grade"] == "none"
    assert s["dominance"] is None
    assert s["separation_ratio"] is None


def test_sweep_no_gaps_grades_none():
    s = gv.vad_gap_recommend_sweep(_result((0, 1)))
    assert s["num_gaps"] == 0
    assert s["grade"] is None
    assert s["dominance"] is None


# ---- human renderer -----------------------------------------------------


def test_render_human_unavailable():
    lines = gv.render_vad_gap_recommend_sweep(None)
    assert len(lines) == 1
    assert "silero VAD unavailable" in lines[0]


def test_render_human_no_gaps():
    lines = gv.render_vad_gap_recommend_sweep(_result((0, 1)))
    assert any("fewer than 2 segments" in ln for ln in lines)
    # No per-bias lines when there are no gaps.
    assert not any("short" in ln for ln in lines)


def test_render_human_golden_with_valley():
    res = _bimodal()
    lines = gv.render_vad_gap_recommend_sweep(res)
    assert lines == [
        "silero VAD recommended-hangover bias sweep — rec.wav",
        "  segments:     5",
        "  gaps:         4 (pauses between consecutive speech regions)",
        "  min gap:      0.300s",
        "  mean gap:     0.725s",
        "  max gap:      2.000s",
        "  total silence:   2.900s",
        "  valley:       between 0.300s (top of short pauses) and 2.000s "
        "(bottom of long pauses), width 1.700s",
        "  recommended --min-silence-ms by bias:",
        "    short    725 (0.725s)",
        "    balanced 1150 (1.150s)",
        "    long     1575 (1.575s)",
        "  spread:       850ms (0.850s) short→long (how much the bias choice "
        "moves the number)",
        "  confidence:   strong (valley is 100.0% of the gap spread; n/a (only "
        "one jump / clean split))",
        "  suggestion:   trust the recommendation — the valley is well separated "
        "(iter-348)",
        "  effect:       merges 3/4 within-turn pauses, keeps 1/4 as turn "
        "boundaries (invariant across biases, iter-352)",
    ]


def test_render_human_no_valley_block():
    res = _result((0, 1), (2, 3), (4, 5))  # both gaps 1.0s → no valley
    lines = gv.render_vad_gap_recommend_sweep(res)
    assert any("no valley" in ln for ln in lines)
    assert any("recommended --min-silence-ms by bias:" in ln for ln in lines)


def test_render_human_has_confidence_and_suggestion_lines():
    lines = gv.render_vad_gap_recommend_sweep(_bimodal())
    assert any(ln.startswith("  confidence:   strong") for ln in lines)
    assert any(ln.startswith("  suggestion:") for ln in lines)


def test_render_human_no_valley_confidence_is_none():
    res = _result((0, 1), (2, 3), (4, 5))  # both gaps 1.0s → no valley
    lines = gv.render_vad_gap_recommend_sweep(res)
    assert any(ln.startswith("  confidence:   none") for ln in lines)
    # The 'none' suggestion text from _gap_confidence_summary.
    assert any("conservative fallback" in ln for ln in lines)


# ---- json renderer ------------------------------------------------------


def test_render_json_shape():
    res = _bimodal()
    payload = json.loads(gv.render_vad_gap_recommend_sweep_json(res))
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert len(payload["biases"]) == 3
    assert [b["bias"] for b in payload["biases"]] == ["short", "balanced", "long"]
    assert payload["spread_ms"] is not None


def test_render_json_unavailable():
    payload = json.loads(gv.render_vad_gap_recommend_sweep_json(None))
    assert payload["available"] is False
    assert "hint" in payload


def test_render_json_no_gaps_null_recommendations():
    payload = json.loads(gv.render_vad_gap_recommend_sweep_json(_result((0, 1))))
    assert payload["available"] is True
    assert payload["spread_ms"] is None
    for b in payload["biases"]:
        assert b["recommended_ms"] is None


def test_render_json_matches_core():
    res = _bimodal()
    payload = json.loads(gv.render_vad_gap_recommend_sweep_json(res))
    s = gv.vad_gap_recommend_sweep(res)
    assert payload["biases"] == s["biases"]
    assert payload["below"] == s["below"]
    assert payload["spread_s"] == s["spread_s"]


def test_render_json_carries_confidence_fields():
    res = _bimodal()
    payload = json.loads(gv.render_vad_gap_recommend_sweep_json(res))
    c = gv.vad_gap_confidence(res)
    assert payload["grade"] == c["grade"] == "strong"
    assert payload["dominance"] == c["dominance"]
    assert payload["separation_ratio"] == c["separation_ratio"]


def test_render_json_no_gaps_grade_null():
    payload = json.loads(gv.render_vad_gap_recommend_sweep_json(_result((0, 1))))
    assert payload["grade"] is None
    assert payload["dominance"] is None


# ---- csv renderer -------------------------------------------------------


def test_render_csv_shape():
    res = _bimodal()
    text = gv.render_vad_gap_recommend_sweep_csv(res)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == [
        "bias",
        "recommended_ms",
        "recommended_s",
        "split_found",
        "below",
        "at_or_above",
        "num_gaps",
    ]
    # One row per bias.
    assert len(rows) == 4
    assert [r[0] for r in rows[1:]] == ["short", "balanced", "long"]


def test_render_csv_golden():
    res = _bimodal()
    text = gv.render_vad_gap_recommend_sweep_csv(res)
    assert text == (
        "bias,recommended_ms,recommended_s,split_found,below,at_or_above,num_gaps\r\n"
        "short,725,0.725,True,3,1,4\r\n"
        "balanced,1150,1.15,True,3,1,4\r\n"
        "long,1575,1.575,True,3,1,4"
    )


def test_render_csv_header_only_for_no_gaps():
    text = gv.render_vad_gap_recommend_sweep_csv(_result((0, 1)))
    rows = list(csv.reader(io.StringIO(text)))
    assert len(rows) == 1  # header only


def test_render_csv_unavailable():
    text = gv.render_vad_gap_recommend_sweep_csv(None)
    assert text.startswith("# silero VAD unavailable")


def test_render_csv_columns_match_single_surface():
    # The sweep CSV columns are identical to the single-bias --csv surface so the
    # rows union cleanly.
    res = _bimodal()
    sweep_header = gv.render_vad_gap_recommend_sweep_csv(res).splitlines()[0]
    single_header = gv.render_vad_gap_recommend_csv(res, bias="short").splitlines()[0]
    assert sweep_header == single_header


def test_render_csv_omits_confidence_grade():
    # The iter-353 confidence grade is a shared scalar (one per valley), so it is
    # deliberately NOT folded into the per-bias CSV rows — duplicating it would
    # break the clean union with gv vad-gap-recommend --csv. The grade lives on
    # the human / --json faces instead.
    header = gv.render_vad_gap_recommend_sweep_csv(_bimodal()).splitlines()[0]
    assert "grade" not in header
    assert "dominance" not in header


# ---- parser -------------------------------------------------------------


def test_parser_registers_sweep_subcommand():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-recommend-sweep", "rec.wav"])
    assert args.command == "vad-gap-recommend-sweep"
    assert args.wav == "rec.wav"


def test_parser_sweep_has_no_bias_flag():
    # The sweep covers all biases, so it has no --bias knob.
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-recommend-sweep", "rec.wav", "--bias", "short"])


def test_parser_sweep_json_csv_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["vad-gap-recommend-sweep", "rec.wav", "--json", "--csv"]
        )


def test_parser_sweep_shares_segmenter_knobs():
    parser = gv.build_parser()
    args = parser.parse_args(
        [
            "vad-gap-recommend-sweep",
            "rec.wav",
            "--threshold",
            "0.7",
            "--min-silence-ms",
            "400",
        ]
    )
    assert args.threshold == 0.7
    assert args.min_silence_ms == 400.0


# ---- handler ------------------------------------------------------------


def _args(**over):
    base = dict(
        wav="rec.wav",
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


def _run(args, **kw):
    lines: List[str] = []
    gv.cmd_vad_gap_recommend_sweep(args, log=lines.append, **kw)
    return lines


def test_handler_human():
    res = _bimodal()
    captured = {}

    def segmenter(wav, *, params):
        captured["wav"] = wav
        captured["params"] = params
        return res

    lines = _run(_args(), segmenter=segmenter, availability=lambda: True)
    assert captured["wav"] == "rec.wav"
    assert any("recommended --min-silence-ms by bias:" in ln for ln in lines)


def test_handler_json():
    res = _bimodal()
    lines = _run(
        _args(json=True),
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is True
    assert len(payload["biases"]) == 3


def test_handler_csv():
    res = _bimodal()
    lines = _run(
        _args(csv=True),
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    assert rows[0][0] == "bias"
    assert len(rows) == 4  # header + 3 biases


def test_handler_unavailable_human():
    called = []
    lines = _run(
        _args(),
        segmenter=lambda *a, **k: called.append(1),
        availability=lambda: False,
    )
    assert not called
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
    assert "\n".join(lines).startswith("# silero VAD unavailable")


def test_handler_builds_params_from_knobs():
    res = _bimodal()
    captured = {}

    class _Params:
        def __init__(self, **kw):
            captured.update(kw)

    args = _args(
        threshold=0.7,
        min_speech_ms=100.0,
        min_silence_ms=400.0,
        speech_pad_ms=10.0,
        max_speech_s=5.0,
        json=True,
    )
    import vad.silero as silero_mod

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
