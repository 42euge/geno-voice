"""Tests for iter-347 — the ``gv vad-gap-recommend`` subcommand (examples/gv.py).

The gap-analysis family — ``gv vad-gaps`` (iter-328), ``gv vad-gap-hist``
(iter-336), ``gv vad-gap-percentiles`` (iter-338), ``gv vad-gap-cdf`` (iter-346)
— all SHOW the operator the inter-segment pause distribution and leave the "so
what do I set ``--min-silence-ms`` to?" judgement to them. ``gv vad-gap-recommend``
is the VERDICT surface that answers it directly: it finds the valley between the
short within-turn pauses (which should merge) and the long between-turn pauses
(which should end a turn) — the WIDEST jump in the sorted gap distribution
(1-D largest-gap / Jenks split) — and names a single recommended hangover sitting
at that valley's midpoint.

Like the rest of the VAD-analysis family, the handler takes injected
``segmenter`` / ``availability`` / ``log`` dependencies so every test runs WITHOUT
importing torch / silero-vad and without touching real audio — fast and
deterministic on the x86_64 Linux runner. The pure core (``vad_gap_recommend``)
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


# ---- parser: registration & defaults -----------------------------------


def test_vad_gap_recommend_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-gap-recommend"] is gv.cmd_vad_gap_recommend


def test_parser_registers_vad_gap_recommend():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-recommend", "rec.wav"])
    assert args.command == "vad-gap-recommend"
    assert args.wav == "rec.wav"


def test_parser_defaults_mirror_vad_gaps_knobs():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-recommend", "rec.wav"])
    # Shares the gv vad segmenter knobs.
    assert args.threshold == pytest.approx(0.5)
    assert args.min_speech_ms == pytest.approx(250.0)
    assert args.min_silence_ms == pytest.approx(800.0)
    assert args.speech_pad_ms == pytest.approx(30.0)
    assert math.isinf(args.max_speech_s)
    assert args.json is False
    assert args.csv is False


def test_parser_accepts_custom_knobs():
    parser = gv.build_parser()
    args = parser.parse_args(
        ["vad-gap-recommend", "rec.wav", "--threshold", "0.7", "--min-silence-ms", "400"]
    )
    assert args.threshold == pytest.approx(0.7)
    assert args.min_silence_ms == pytest.approx(400.0)


def test_parser_json_csv_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-recommend", "rec.wav", "--json", "--csv"])


def test_parser_rejects_out_of_range_threshold():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-recommend", "rec.wav", "--threshold", "1.5"])


# ---- pure core: vad_gap_recommend ---------------------------------------


def test_recommend_finds_valley_midpoint():
    # Short pauses ~0.2s, long pauses 2.0s. Sorted gaps: 0.2, 0.2, 2.0, 3.0.
    # Widest jump is 0.2 -> 2.0 (width 1.8); midpoint 1.1s.
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    r = gv.vad_gap_recommend(res)
    assert r["num_segments"] == 5
    assert r["num_gaps"] == 4
    assert r["split_found"] is True
    assert r["gap_below_s"] == 0.2
    assert r["gap_above_s"] == 2.0
    assert r["valley_width_s"] == 1.8
    assert r["recommended_s"] == 1.1
    assert r["recommended_ms"] == 1100.0


def test_recommend_below_at_or_above_split():
    # Recommended 1.1s: gaps < 1.1 are {0.2, 0.2} -> 2 below, 2 at-or-above.
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    r = gv.vad_gap_recommend(res)
    assert r["below"] == 2
    assert r["at_or_above"] == 2
    assert r["below"] + r["at_or_above"] == r["num_gaps"]


def test_recommend_below_matches_cdf_at_cut():
    # The below / at_or_above split is exactly what vad_gap_cdf reports at the
    # recommended cut — the two surfaces agree on the merge accounting.
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10), (14, 15))
    r = gv.vad_gap_recommend(res)
    c = gv.vad_gap_cdf(res, cuts_ms=[r["recommended_ms"]])
    entry = c["cuts"][0]
    assert entry["merged"] == r["below"]
    assert entry["kept"] == r["at_or_above"]


def test_recommend_anchors_to_vad_silence_gaps():
    res = _result((0, 1), (2, 3), (6, 7), (15, 16))
    r = gv.vad_gap_recommend(res)
    d = gv.vad_silence_gaps(res)
    assert r["num_segments"] == d["num_segments"]
    assert r["num_gaps"] == d["num_gaps"]
    assert r["min_gap_s"] == d["min_gap_s"]
    assert r["max_gap_s"] == d["max_gap_s"]
    assert r["mean_gap_s"] == d["mean_gap_s"]
    assert r["total_silence_s"] == d["total_silence_s"]


def test_recommend_single_gap_no_valley():
    # One pause: no short/long split possible. Recommend just below it (half).
    res = _result((0, 1), (1.5, 2))  # gap 0.5s
    r = gv.vad_gap_recommend(res)
    assert r["num_gaps"] == 1
    assert r["split_found"] is False
    assert r["recommended_s"] == 0.25
    assert r["recommended_ms"] == 250.0
    assert r["gap_below_s"] is None
    assert r["gap_above_s"] is None
    assert r["valley_width_s"] is None
    # Just below the only pause -> it is kept as a boundary.
    assert r["below"] == 0
    assert r["at_or_above"] == 1


def test_recommend_all_equal_gaps_no_valley():
    # Several pauses all the same length: no valley (zero-width jumps). Recommend
    # just below the cluster (half the shortest); every pause kept.
    res = _result((0, 1), (2, 3), (4, 5), (6, 7))  # gaps all 1.0s
    r = gv.vad_gap_recommend(res)
    assert r["num_gaps"] == 3
    assert r["split_found"] is False
    assert r["recommended_s"] == 0.5
    assert r["below"] == 0
    assert r["at_or_above"] == 3


def test_recommend_earliest_widest_jump_wins_on_tie():
    # Two equal-width jumps (1.0->2.0 and 3.0->4.0, both width 1.0). The FIRST
    # widest jump wins — earliest-tie, matching the rest of the family.
    res = _result((0, 1), (2, 3), (5, 6), (10, 11), (15, 16))
    # gaps sorted: 1.0, 2.0, 4.0, 4.0  -> jumps 1.0, 2.0, 0.0; widest is 2.0->4.0
    r = gv.vad_gap_recommend(res)
    assert r["gap_below_s"] == 2.0
    assert r["gap_above_s"] == 4.0


def test_recommend_explicit_tie_takes_first():
    # Gaps sorted: 1.0, 2.0, 3.0, 4.0 -> jumps all width 1.0. First jump
    # (1.0->2.0) wins.
    res = _result((0, 1), (2, 3), (5, 6), (9, 10), (14, 15))
    r = gv.vad_gap_recommend(res)
    assert r["gap_below_s"] == 1.0
    assert r["gap_above_s"] == 2.0
    assert r["recommended_s"] == 1.5


def test_recommend_empty_for_fewer_than_two_segments():
    res = _result((0, 1))
    r = gv.vad_gap_recommend(res)
    assert r["num_gaps"] == 0
    assert r["recommended_ms"] is None
    assert r["recommended_s"] is None
    assert r["split_found"] is False
    assert r["min_gap_s"] is None
    assert r["below"] == 0
    assert r["at_or_above"] == 0


def test_recommend_empty_for_zero_segments():
    res = _result()
    r = gv.vad_gap_recommend(res)
    assert r["num_segments"] == 0
    assert r["num_gaps"] == 0
    assert r["recommended_ms"] is None


def test_recommend_handles_unsorted_segments():
    res = _result((9, 10), (0, 1), (5, 6), (2, 3))
    r = gv.vad_gap_recommend(res)
    res_sorted = _result((0, 1), (2, 3), (5, 6), (9, 10))
    r_sorted = gv.vad_gap_recommend(res_sorted)
    assert r["recommended_s"] == r_sorted["recommended_s"]
    assert r["gap_below_s"] == r_sorted["gap_below_s"]
    assert r["gap_above_s"] == r_sorted["gap_above_s"]


def test_recommend_rounding():
    # gaps sorted: 0.1, 0.7. Widest jump 0.1->0.7; midpoint 0.4. recommended_ms
    # 400.0. Confirm the 1-place ms rounding doesn't over-round.
    res = _result((0, 1), (1.1, 2), (2.7, 3))
    r = gv.vad_gap_recommend(res)
    assert r["recommended_s"] == 0.4
    assert r["recommended_ms"] == 400.0


def test_recommend_lies_within_min_max_when_split():
    # The recommended hangover sits strictly between the min and max gap when a
    # valley exists.
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    r = gv.vad_gap_recommend(res)
    assert r["min_gap_s"] < r["recommended_s"] < r["max_gap_s"]


# ---- renderer: human-readable -------------------------------------------


def test_render_human_shape():
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    lines = gv.render_vad_gap_recommend(res)
    text = "\n".join(lines)
    assert "silero VAD recommended hangover — rec.wav" in lines[0]
    assert any("segments:" in ln for ln in lines)
    assert any("recommended --min-silence-ms:" in ln for ln in lines)
    assert any("valley:" in ln for ln in lines)
    assert any("effect:" in ln for ln in lines)
    assert "iter-347" in text


def test_render_human_no_valley():
    res = _result((0, 1), (2, 3), (4, 5), (6, 7))  # all gaps equal
    lines = gv.render_vad_gap_recommend(res)
    assert any("no valley" in ln for ln in lines)
    assert not any("valley:" in ln and "between" in ln for ln in lines)


def test_render_human_no_gaps():
    res = _result((0, 1))
    lines = gv.render_vad_gap_recommend(res)
    assert any("fewer than 2 segments" in ln for ln in lines)
    assert not any("recommended --min-silence-ms" in ln for ln in lines)


def test_render_human_unavailable():
    lines = gv.render_vad_gap_recommend(None)
    assert len(lines) == 1
    assert "silero VAD unavailable" in lines[0]


# ---- renderer: human golden output --------------------------------------
#
# The shape tests above assert structure + substrings. They do NOT pin the EXACT
# rendered block, so a silent alignment/label/header regression would slip
# through them. These goldens freeze the byte-for-byte verdict for fixed stub
# segmentations, so the human face can only change deliberately.


def test_render_human_golden_with_valley():
    # Short pauses 0.2s, long pauses 2.0/3.0s. Sorted gaps: 0.2, 0.2, 2.0, 3.0.
    # Widest jump 0.2->2.0 (width 1.8); recommend midpoint 1.1s = 1100 ms.
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    lines = gv.render_vad_gap_recommend(res)
    assert lines == [
        "silero VAD recommended hangover — rec.wav",
        "  segments:     5",
        "  gaps:         4 (pauses between consecutive speech regions)",
        "  min gap:      0.200s",
        "  mean gap:     1.350s",
        "  max gap:      3.000s",
        "  total silence:   5.400s",
        "  recommended --min-silence-ms: 1100 (1.100s) [bias: balanced]",
        "  valley:       between 0.200s (top of short pauses) and 2.000s "
        "(bottom of long pauses), width 1.800s",
        "  effect:       merges 2/4 within-turn pauses, keeps 2/4 as turn "
        "boundaries (iter-347)",
    ]


def test_render_human_golden_no_valley():
    # All gaps equal 1.0s: no valley. Recommend just below (0.5s = 500 ms);
    # every pause kept.
    res = _result((0, 1), (2, 3), (4, 5), (6, 7))
    lines = gv.render_vad_gap_recommend(res)
    assert lines == [
        "silero VAD recommended hangover — rec.wav",
        "  segments:     4",
        "  gaps:         3 (pauses between consecutive speech regions)",
        "  min gap:      1.000s",
        "  mean gap:     1.000s",
        "  max gap:      1.000s",
        "  total silence:   3.000s",
        "  recommended --min-silence-ms: 500 (0.500s) [bias: balanced]",
        "  (no valley — pauses don't separate into short/long clusters; "
        "recommending just below the shortest pause so every pause is kept)",
        "  effect:       merges 0/3 within-turn pauses, keeps 3/3 as turn "
        "boundaries (iter-347)",
    ]


def test_render_human_golden_single_segment_block():
    res = _result((0, 1))
    lines = gv.render_vad_gap_recommend(res)
    assert lines == [
        "silero VAD recommended hangover — rec.wav",
        "  segments:     1",
        "  gaps:         0 (pauses between consecutive speech regions)",
        "  (fewer than 2 segments — no inter-segment pause to measure)",
    ]


def test_render_human_compact_ms_label():
    # recommended_ms is a float (e.g. 1100.0) but %g renders it compactly without
    # the trailing ``.0`` — the cut-label twin shared with the CDF surface. Since
    # cut_s rounds to 3 places, recommended_ms is always a whole number, so the
    # label is always integer-spelled.
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    lines = gv.render_vad_gap_recommend(res)
    rec_line = next(ln for ln in lines if "recommended --min-silence-ms" in ln)
    assert "1100 (" in rec_line  # not "1100.0 ("
    assert "1100.0" not in rec_line


# ---- renderer: JSON -----------------------------------------------------


def test_render_json_shape():
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    payload = json.loads(gv.render_vad_gap_recommend_json(res))
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["num_segments"] == 5
    assert payload["num_gaps"] == 4
    assert payload["bias"] == "balanced"
    assert payload["recommended_ms"] == 1100.0
    assert payload["recommended_s"] == 1.1
    assert payload["split_found"] is True
    assert payload["below"] == 2
    assert payload["at_or_above"] == 2
    assert payload["gap_below_s"] == 0.2
    assert payload["gap_above_s"] == 2.0
    assert payload["valley_width_s"] == 1.8


def test_render_json_no_gaps():
    res = _result((0, 1))
    payload = json.loads(gv.render_vad_gap_recommend_json(res))
    assert payload["recommended_ms"] is None
    assert payload["split_found"] is False
    assert payload["min_gap_s"] is None
    assert payload["below"] == 0


def test_render_json_no_valley():
    res = _result((0, 1), (2, 3), (4, 5))
    payload = json.loads(gv.render_vad_gap_recommend_json(res))
    assert payload["split_found"] is False
    assert payload["gap_below_s"] is None
    assert payload["recommended_ms"] is not None  # still recommends a number


def test_render_json_unavailable():
    payload = json.loads(gv.render_vad_gap_recommend_json(None))
    assert payload["available"] is False
    assert "hint" in payload


# ---- renderer: CSV ------------------------------------------------------


def test_render_csv_shape():
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    text = gv.render_vad_gap_recommend_csv(res)
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
    assert len(rows) == 2  # header + one summary row
    assert rows[1][0] == "balanced"
    assert rows[1][1] == "1100"
    assert rows[1][3] == "True"
    assert int(rows[1][4]) == 2
    assert int(rows[1][5]) == 2
    assert int(rows[1][6]) == 4


def test_render_csv_matches_json():
    res = _result((0, 1), (1.2, 2), (2.2, 3), (6, 7), (15, 16))
    payload = json.loads(gv.render_vad_gap_recommend_json(res))
    text = gv.render_vad_gap_recommend_csv(res)
    row = list(csv.reader(io.StringIO(text)))[1]
    assert row[0] == payload["bias"]
    assert float(row[2]) == pytest.approx(payload["recommended_s"])
    assert int(row[4]) == payload["below"]
    assert int(row[5]) == payload["at_or_above"]
    assert int(row[6]) == payload["num_gaps"]


def test_render_csv_header_only_for_no_gaps():
    res = _result((0, 1))
    text = gv.render_vad_gap_recommend_csv(res)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows == [
        [
            "bias",
            "recommended_ms",
            "recommended_s",
            "split_found",
            "below",
            "at_or_above",
            "num_gaps",
        ]
    ]


def test_render_csv_unavailable():
    text = gv.render_vad_gap_recommend_csv(None)
    assert text.startswith("# silero VAD unavailable")


# ---- handler: cmd_vad_gap_recommend -------------------------------------


def _run(args, **kw):
    lines: List[str] = []
    gv.cmd_vad_gap_recommend(args, log=lines.append, **kw)
    return lines


def _args(**over):
    base = dict(
        wav="rec.wav",
        threshold=0.5,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        bias="balanced",
        json=False,
        csv=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_handler_human():
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    captured = {}

    def segmenter(wav, *, params):
        captured["wav"] = wav
        captured["params"] = params
        return res

    lines = _run(_args(), segmenter=segmenter, availability=lambda: True)
    assert captured["wav"] == "rec.wav"
    assert any("recommended --min-silence-ms:" in ln for ln in lines)


def test_handler_json():
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6))
    lines = _run(
        _args(json=True),
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is True
    assert payload["recommended_ms"] is not None


def test_handler_csv():
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6))
    lines = _run(
        _args(csv=True),
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0][0] == "bias"
    assert len(rows) == 2


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
    assert "\n".join(lines).startswith("# silero VAD unavailable")


def test_handler_builds_params_from_knobs():
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6))
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


# ---- iter-351: --bias knob ----------------------------------------------
#
# The valley between the short within-turn cluster and the long between-turn
# cluster is an EMPTY band, so any interior point splits the pauses identically.
# --bias only shifts WHERE in that valley the recommended number sits — short
# (quarter up from the short cluster), balanced (midpoint, the iter-347 default),
# long (three quarters up, hugging the long cluster). The merge accounting
# (below / at_or_above) is therefore invariant across biases.


def test_parser_bias_default_is_balanced():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-recommend", "rec.wav"])
    assert args.bias == "balanced"


def test_parser_bias_accepts_each_choice():
    parser = gv.build_parser()
    for choice in ("short", "balanced", "long"):
        args = parser.parse_args(["vad-gap-recommend", "rec.wav", "--bias", choice])
        assert args.bias == choice


def test_parser_bias_rejects_unknown():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-recommend", "rec.wav", "--bias", "medium"])


def test_bias_fraction_table_and_default():
    # The fraction table and default constant are the single source of truth.
    assert gv.GAP_RECOMMEND_BIAS_FRACTIONS == {
        "short": 0.25,
        "balanced": 0.5,
        "long": 0.75,
    }
    assert gv.DEFAULT_GAP_RECOMMEND_BIAS == "balanced"


def test_core_balanced_equals_legacy_midpoint():
    # balanced (0.5) reproduces the original (best_lo + best_hi) / 2 midpoint.
    # Sorted gaps 0.2, 0.2, 2.0, 3.0; valley 0.2->2.0; midpoint 1.1s.
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    r = gv.vad_gap_recommend(res, bias="balanced")
    assert r["bias"] == "balanced"
    assert r["recommended_s"] == 1.1
    assert r["recommended_ms"] == 1100.0
    # Default arg == explicit balanced.
    assert gv.vad_gap_recommend(res) == r


def test_core_short_bias_smaller_than_balanced_than_long():
    # Valley 0.2 -> 2.0 (width 1.8). short=0.2+0.25*1.8=0.65; balanced=1.1;
    # long=0.2+0.75*1.8=1.55. Monotone short < balanced < long.
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    short = gv.vad_gap_recommend(res, bias="short")
    balanced = gv.vad_gap_recommend(res, bias="balanced")
    long = gv.vad_gap_recommend(res, bias="long")
    assert short["recommended_s"] == 0.65
    assert balanced["recommended_s"] == 1.1
    assert long["recommended_s"] == 1.55
    assert short["recommended_s"] < balanced["recommended_s"] < long["recommended_s"]


def test_bias_lands_strictly_inside_valley():
    # Every bias point sits strictly between the valley endpoints, so it is a
    # valid cut that merges the short cluster and keeps the long cluster.
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    for choice in ("short", "balanced", "long"):
        r = gv.vad_gap_recommend(res, bias=choice)
        assert r["gap_below_s"] < r["recommended_s"] < r["gap_above_s"]


def test_bias_does_not_change_merge_accounting():
    # The whole point: shifting the number within the EMPTY valley leaves the
    # below / at_or_above split identical across all three biases.
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    splits = {
        choice: (
            gv.vad_gap_recommend(res, bias=choice)["below"],
            gv.vad_gap_recommend(res, bias=choice)["at_or_above"],
        )
        for choice in ("short", "balanced", "long")
    }
    assert splits["short"] == splits["balanced"] == splits["long"] == (2, 2)


def test_bias_no_valley_fallback_scales_shortest_pause():
    # No valley (all gaps equal 1.0s): recommend a fraction of the shortest pause.
    # short=1.0*0.25=0.25; balanced=0.5; long=0.75. All below the shortest pause,
    # so every pause is still kept regardless of bias.
    res = _result((0, 1), (2, 3), (4, 5), (6, 7))
    assert gv.vad_gap_recommend(res, bias="short")["recommended_s"] == 0.25
    assert gv.vad_gap_recommend(res, bias="balanced")["recommended_s"] == 0.5
    assert gv.vad_gap_recommend(res, bias="long")["recommended_s"] == 0.75
    for choice in ("short", "balanced", "long"):
        r = gv.vad_gap_recommend(res, bias=choice)
        assert r["split_found"] is False
        assert r["below"] == 0
        assert r["at_or_above"] == 3


def test_core_rejects_unknown_bias():
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    with pytest.raises(ValueError, match="unknown bias"):
        gv.vad_gap_recommend(res, bias="medium")


def test_bias_present_for_no_gaps():
    # The bias field is echoed even when there are no gaps to recommend against.
    res = _result((0, 1))
    r = gv.vad_gap_recommend(res, bias="long")
    assert r["bias"] == "long"
    assert r["recommended_ms"] is None


def test_render_human_bias_echoed_on_recommended_line():
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    lines = gv.render_vad_gap_recommend(res, bias="short")
    rec_line = next(ln for ln in lines if "recommended --min-silence-ms" in ln)
    assert "[bias: short]" in rec_line
    assert "650 (0.650s)" in rec_line


def test_render_json_carries_bias():
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    payload = json.loads(gv.render_vad_gap_recommend_json(res, bias="long"))
    assert payload["bias"] == "long"
    assert payload["recommended_s"] == 1.55


def test_render_csv_carries_bias():
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    text = gv.render_vad_gap_recommend_csv(res, bias="long")
    row = list(csv.reader(io.StringIO(text)))[1]
    assert row[0] == "long"
    assert row[1] == "1550"


def test_render_unavailable_accepts_bias():
    # The unavailable path still takes the bias kwarg without crashing.
    assert gv.render_vad_gap_recommend(None, bias="short")[0].startswith(
        "silero VAD unavailable"
    )
    assert (
        json.loads(gv.render_vad_gap_recommend_json(None, bias="short"))["available"]
        is False
    )
    assert gv.render_vad_gap_recommend_csv(None, bias="short").startswith(
        "# silero VAD unavailable"
    )


def test_handler_passes_bias_through():
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    lines = _run(
        _args(bias="long", json=True),
        segmenter=lambda w, *, params: res,
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["bias"] == "long"
    assert payload["recommended_s"] == 1.55


def test_handler_bias_defaults_balanced_when_absent():
    # A namespace without a bias attr (older callers) falls back to balanced.
    res = _result((0, 1), (1.2, 2), (2.2, 3), (5, 6), (9, 10))
    args = _args(json=True)
    del args.bias
    lines = _run(
        args, segmenter=lambda w, *, params: res, availability=lambda: True
    )
    payload = json.loads("\n".join(lines))
    assert payload["bias"] == "balanced"
