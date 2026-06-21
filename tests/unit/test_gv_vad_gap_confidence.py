"""Tests for iter-348 — the ``gv vad-gap-confidence`` subcommand (examples/gv.py).

iter-347's ``gv vad-gap-recommend`` always names a recommended ``--min-silence-ms``,
but the number is only as good as the valley it sits in. ``gv vad-gap-confidence``
is the companion CONFIDENCE surface (iter-347's own next-item): it reads the same
gap distribution and grades how dominant the recommendation's valley (the widest
jump in the sorted gaps) is versus the total gap spread and the next-widest jump,
reporting ``strong`` / ``moderate`` / ``weak`` (or ``none`` when the pauses are
uniform and there is no valley to grade). A clean bimodal distribution grades
strong; a smear of similar pauses grades weak — so the operator knows whether to
trust the recommendation or tune by ear.

Like the rest of the VAD-analysis family, the handler takes injected
``segmenter`` / ``availability`` / ``log`` dependencies so every test runs WITHOUT
importing torch / silero-vad and without touching real audio — fast and
deterministic on the x86_64 Linux runner. The pure core (``vad_gap_confidence``)
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


def _result_from_gaps(gaps, *, name="rec.wav", sample_rate=16000):
    """Build a result whose inter-segment pauses are exactly ``gaps`` (in s).

    Each speech region is a fixed 1.0s long; the gap before region ``i+1`` is
    ``gaps[i]``. This keeps the gap list under direct test control while still
    flowing through the real ``vad_silence_gaps`` -> ``vad_gap_recommend`` ->
    ``vad_gap_confidence`` chain.
    """
    segs = [_Seg(0.0, 1.0)]
    t = 1.0
    for g in gaps:
        start = t + g
        segs.append(_Seg(start, start + 1.0))
        t = start + 1.0
    return _Result(name=name, sample_rate=sample_rate, duration_s=t, segments=segs)


def _result(*pairs, name="rec.wav", sample_rate=16000, duration_s=30.0):
    return _Result(
        name=name,
        sample_rate=sample_rate,
        duration_s=duration_s,
        segments=[_Seg(a, b) for a, b in pairs],
    )


# ---- parser: registration & defaults -----------------------------------


def test_vad_gap_confidence_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-gap-confidence"] is gv.cmd_vad_gap_confidence


def test_parser_registers_vad_gap_confidence():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-confidence", "rec.wav"])
    assert args.command == "vad-gap-confidence"
    assert args.wav == "rec.wav"


def test_parser_vad_gap_confidence_knob_defaults():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-confidence", "rec.wav"])
    # Shares the gv vad segmenter knobs.
    assert args.threshold == pytest.approx(0.5)
    assert args.min_speech_ms == pytest.approx(250.0)
    assert args.min_silence_ms == pytest.approx(800.0)
    assert args.speech_pad_ms == pytest.approx(30.0)
    assert math.isinf(args.max_speech_s)
    assert args.json is False
    assert args.csv is False


def test_parser_vad_gap_confidence_custom_knobs():
    parser = gv.build_parser()
    args = parser.parse_args(
        [
            "vad-gap-confidence",
            "rec.wav",
            "--threshold",
            "0.7",
            "--min-speech-ms",
            "120",
            "--min-silence-ms",
            "400",
            "--speech-pad-ms",
            "50",
            "--max-speech-s",
            "15",
        ]
    )
    assert args.threshold == 0.7
    assert args.min_speech_ms == 120
    assert args.min_silence_ms == 400
    assert args.speech_pad_ms == 50
    assert args.max_speech_s == 15


def test_parser_vad_gap_confidence_json_csv_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-confidence", "rec.wav", "--json", "--csv"])


def test_parser_vad_gap_confidence_threshold_range():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-confidence", "rec.wav", "--threshold", "1.5"])


# ---- pure core: vad_gap_confidence -------------------------------------


def test_confidence_strong_clean_bimodal():
    # Short cluster ~0.2-0.3s, long cluster ~1.4-1.6s, a wide empty band between.
    r = gv.vad_gap_confidence(_result_from_gaps([0.2, 0.3, 0.25, 1.5, 1.6, 1.4]))
    assert r["grade"] == "strong"
    assert r["split_found"] is True
    assert r["valley_width_s"] == 1.1  # 1.4 - 0.3
    assert r["spread_s"] == 1.4  # 1.6 - 0.2
    # dominance = valley / spread = 1.1 / 1.4
    assert r["dominance"] == round(1.1 / 1.4, 3)
    # separation = widest jump (1.1) / next-widest (0.1)
    assert r["separation_ratio"] == round(1.1 / 0.1, 3)
    assert r["dominance"] >= gv.GAP_CONFIDENCE_STRONG_DOMINANCE


def test_confidence_weak_uniform_smear():
    # Evenly-spaced pauses: every jump is the same width, no dominant valley.
    r = gv.vad_gap_confidence(_result_from_gaps([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    assert r["grade"] == "weak"
    assert r["split_found"] is True
    # All jumps equal (0.1), so dominance is small and separation is exactly 1.
    assert r["separation_ratio"] == 1.0
    assert r["dominance"] < gv.GAP_CONFIDENCE_MODERATE_DOMINANCE


def test_confidence_moderate_band():
    # A discernible-but-shallow valley: dominance lands in [0.25, 0.5).
    r = gv.vad_gap_confidence(_result_from_gaps([0.2, 0.5, 0.9, 1.2]))
    assert r["grade"] == "moderate"
    assert gv.GAP_CONFIDENCE_MODERATE_DOMINANCE <= r["dominance"]
    assert r["dominance"] < gv.GAP_CONFIDENCE_STRONG_DOMINANCE


def test_confidence_grade_boundaries():
    # Exactly at the strong boundary (dominance == 0.5) grades strong (>=).
    # gaps 0.0, 0.0... build a valley of half the spread.
    # Use three gaps: jumps must sum to spread; make the widest exactly half.
    r = gv.vad_gap_confidence(_result_from_gaps([0.0, 0.5, 1.0]))
    # sorted gaps 0.0, 0.5, 1.0 -> jumps 0.5, 0.5; spread 1.0; widest 0.5.
    assert r["dominance"] == 0.5
    assert r["grade"] == "strong"


def test_confidence_dominance_is_widest_over_spread():
    r = gv.vad_gap_confidence(_result_from_gaps([0.2, 0.3, 0.25, 1.5, 1.6, 1.4]))
    # Recompute independently from the sorted gaps.
    gaps = sorted([0.2, 0.3, 0.25, 1.5, 1.6, 1.4])
    jumps = [gaps[i] - gaps[i - 1] for i in range(1, len(gaps))]
    spread = sum(jumps)
    assert r["dominance"] == round(max(jumps) / spread, 3)
    assert r["spread_s"] == round(spread, 3)


def test_confidence_agrees_with_recommend_recommendation():
    # The confidence surface must carry EXACTLY the recommendation vad-gap-recommend
    # produces (it anchors to vad_gap_recommend).
    res = _result_from_gaps([0.2, 0.3, 1.5, 1.6])
    rec = gv.vad_gap_recommend(res)
    conf = gv.vad_gap_confidence(res)
    for key in (
        "recommended_ms",
        "recommended_s",
        "split_found",
        "below",
        "at_or_above",
        "valley_width_s",
        "gap_below_s",
        "gap_above_s",
    ):
        assert conf[key] == rec[key], key


def test_confidence_single_gap_no_valley():
    # Two segments -> one gap -> no jump between consecutive gaps -> no valley.
    r = gv.vad_gap_confidence(_result_from_gaps([0.5]))
    assert r["num_gaps"] == 1
    assert r["split_found"] is False
    assert r["grade"] == "none"
    assert r["dominance"] is None
    assert r["separation_ratio"] is None
    assert r["spread_s"] is None
    assert r["runner_up_width_s"] is None


def test_confidence_all_equal_gaps_no_valley():
    # Several identical pauses: every jump is zero-width, so no valley.
    r = gv.vad_gap_confidence(_result_from_gaps([0.5, 0.5, 0.5]))
    assert r["num_gaps"] == 3
    assert r["split_found"] is False
    assert r["grade"] == "none"
    assert r["dominance"] is None
    assert r["separation_ratio"] is None


def test_confidence_two_gaps_single_jump_separation_none():
    # Two gaps -> a single jump -> there is a valley but no runner-up jump, so
    # separation_ratio is None (the valley is infinitely dominant over rivals),
    # while dominance is 1.0 (the single jump IS the whole spread).
    r = gv.vad_gap_confidence(_result_from_gaps([0.2, 1.5]))
    assert r["num_gaps"] == 2
    assert r["split_found"] is True
    assert r["dominance"] == 1.0
    assert r["spread_s"] == round(1.5 - 0.2, 3)
    assert r["runner_up_width_s"] is None
    assert r["separation_ratio"] is None
    assert r["grade"] == "strong"


def test_confidence_fewer_than_two_segments():
    single = _result((0.0, 1.0))
    r = gv.vad_gap_confidence(single)
    assert r["num_gaps"] == 0
    assert r["grade"] is None
    assert r["recommended_ms"] is None
    assert r["dominance"] is None
    assert r["separation_ratio"] is None
    assert r["spread_s"] is None


def test_confidence_zero_segments():
    empty = _result()
    r = gv.vad_gap_confidence(empty)
    assert r["num_segments"] == 0
    assert r["num_gaps"] == 0
    assert r["grade"] is None


def test_confidence_unsorted_segments_robust():
    # Segments out of chronological order must still produce a stable grade
    # (vad_silence_gaps sorts; confidence re-sorts the gaps).
    a = _result((0.0, 1.0), (1.2, 2.2), (2.4, 3.4), (5.0, 6.0), (7.2, 8.2))
    b = _result((7.2, 8.2), (0.0, 1.0), (5.0, 6.0), (2.4, 3.4), (1.2, 2.2))
    ra = gv.vad_gap_confidence(a)
    rb = gv.vad_gap_confidence(b)
    assert ra["grade"] == rb["grade"]
    assert ra["dominance"] == rb["dominance"]
    assert ra["separation_ratio"] == rb["separation_ratio"]


def test_confidence_rounding_three_places():
    r = gv.vad_gap_confidence(_result_from_gaps([0.1, 0.13, 0.97, 1.0]))
    for key in ("dominance", "spread_s", "valley_width_s", "runner_up_width_s"):
        if r[key] is not None:
            assert round(r[key], 3) == r[key], key
    if r["separation_ratio"] is not None:
        assert round(r["separation_ratio"], 3) == r["separation_ratio"]


def test_confidence_dominance_in_unit_interval():
    r = gv.vad_gap_confidence(_result_from_gaps([0.2, 0.4, 1.0, 1.8, 2.0]))
    assert 0.0 < r["dominance"] <= 1.0


# ---- gap-confidence summary helper -------------------------------------


@pytest.mark.parametrize(
    "grade,needle",
    [
        ("strong", "trust the recommendation"),
        ("moderate", "shallow"),
        ("weak", "tune"),
        ("none", "conservative fallback"),
    ],
)
def test_gap_confidence_summary_per_grade(grade, needle):
    assert needle in gv._gap_confidence_summary(grade)


def test_gap_confidence_summary_defensive_unknown():
    out = gv._gap_confidence_summary("bogus")
    assert "unrecognized" in out.lower()


# ---- human renderer ----------------------------------------------------


def test_render_human_unavailable():
    lines = gv.render_vad_gap_confidence(None)
    assert len(lines) == 1
    assert "silero-vad" in lines[0]


def test_render_human_no_gaps():
    lines = gv.render_vad_gap_confidence(_result((0.0, 1.0)))
    assert any("fewer than 2 segments" in l for l in lines)
    # No grade line is printed when there are no gaps.
    assert not any("confidence:" in l for l in lines)


def test_render_human_strong_golden():
    res = _result_from_gaps([0.2, 0.3, 0.25, 1.5, 1.6, 1.4])
    assert gv.render_vad_gap_confidence(res) == [
        "silero VAD recommendation confidence — rec.wav",
        "  segments:     7",
        "  gaps:         6 (pauses between consecutive speech regions)",
        "  min gap:      0.200s",
        "  mean gap:     0.875s",
        "  max gap:      1.600s",
        "  total silence:   5.250s",
        "  recommended --min-silence-ms: 850 (0.850s)",
        "  confidence:   strong (valley 1.100s is 78.6% of the 1.400s gap "
        "spread; 11.000x the next-widest jump)",
        "  suggestion:   trust the recommendation — the valley is well "
        "separated (iter-348)",
    ]


def test_render_human_none_golden():
    res = _result_from_gaps([0.5, 0.5, 0.5])
    assert gv.render_vad_gap_confidence(res) == [
        "silero VAD recommendation confidence — rec.wav",
        "  segments:     4",
        "  gaps:         3 (pauses between consecutive speech regions)",
        "  min gap:      0.500s",
        "  mean gap:     0.500s",
        "  max gap:      0.500s",
        "  total silence:   1.500s",
        "  recommended --min-silence-ms: 250 (0.250s)",
        "  confidence:   none (no valley — pauses don't separate into "
        "short/long clusters)",
        "  suggestion:   no valley to grade — the pauses are uniform, so the "
        "recommendation is a conservative fallback, not a confident split "
        "(iter-348)",
    ]


def test_render_human_single_jump_separation_na():
    # Two gaps -> one jump -> separation_ratio None renders the "n/a" branch.
    lines = gv.render_vad_gap_confidence(_result_from_gaps([0.2, 1.5]))
    conf_line = next(l for l in lines if "confidence:" in l)
    assert "n/a (only one jump / clean split)" in conf_line


def test_render_human_single_segment_golden():
    res = _result((0.0, 1.0), name="solo.wav")
    assert gv.render_vad_gap_confidence(res) == [
        "silero VAD recommendation confidence — solo.wav",
        "  segments:     1",
        "  gaps:         0 (pauses between consecutive speech regions)",
        "  (fewer than 2 segments — no inter-segment pause to measure)",
    ]


# ---- JSON renderer -----------------------------------------------------


def test_render_json_unavailable():
    payload = json.loads(gv.render_vad_gap_confidence_json(None))
    assert payload["available"] is False
    assert "silero-vad" in payload["hint"]


def test_render_json_strong_shape():
    res = _result_from_gaps([0.2, 0.3, 0.25, 1.5, 1.6, 1.4])
    payload = json.loads(gv.render_vad_gap_confidence_json(res))
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["grade"] == "strong"
    assert payload["dominance"] == round(1.1 / 1.4, 3)
    assert payload["separation_ratio"] == 11.0
    assert payload["spread_s"] == 1.4
    assert payload["valley_width_s"] == 1.1
    assert payload["recommended_ms"] == 850.0
    assert payload["runner_up_width_s"] == 0.1


def test_render_json_matches_core():
    res = _result_from_gaps([0.2, 0.4, 1.2, 1.4])
    payload = json.loads(gv.render_vad_gap_confidence_json(res))
    core = gv.vad_gap_confidence(res)
    for key in (
        "num_segments",
        "num_gaps",
        "grade",
        "dominance",
        "separation_ratio",
        "spread_s",
        "valley_width_s",
        "recommended_ms",
        "split_found",
    ):
        assert payload[key] == core[key], key


def test_render_json_no_valley_nulls():
    payload = json.loads(gv.render_vad_gap_confidence_json(_result_from_gaps([0.5, 0.5])))
    assert payload["grade"] == "none"
    assert payload["dominance"] is None
    assert payload["separation_ratio"] is None
    assert payload["spread_s"] is None


def test_render_json_no_gaps_nulls():
    payload = json.loads(gv.render_vad_gap_confidence_json(_result((0.0, 1.0))))
    assert payload["num_gaps"] == 0
    assert payload["grade"] is None
    assert payload["dominance"] is None


# ---- CSV renderer ------------------------------------------------------


def test_render_csv_unavailable():
    out = gv.render_vad_gap_confidence_csv(None)
    assert out.startswith("# silero VAD unavailable")


def test_render_csv_strong_shape():
    res = _result_from_gaps([0.2, 0.3, 0.25, 1.5, 1.6, 1.4])
    out = gv.render_vad_gap_confidence_csv(res)
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0] == [
        "recommended_ms",
        "grade",
        "dominance",
        "separation_ratio",
        "valley_width_s",
        "spread_s",
    ]
    assert rows[1] == ["850", "strong", "0.786", "11.0", "1.1", "1.4"]


def test_render_csv_agrees_with_core():
    res = _result_from_gaps([0.2, 0.4, 1.2, 1.4])
    out = gv.render_vad_gap_confidence_csv(res)
    rows = list(csv.reader(io.StringIO(out)))
    core = gv.vad_gap_confidence(res)
    assert rows[1][0] == gv._format_cut_label(core["recommended_ms"])
    assert rows[1][1] == core["grade"]
    assert float(rows[1][2]) == core["dominance"]


def test_render_csv_no_valley_blanks():
    out = gv.render_vad_gap_confidence_csv(_result_from_gaps([0.5, 0.5]))
    rows = list(csv.reader(io.StringIO(out)))
    # grade "none", numeric measures blank.
    assert rows[1][1] == "none"
    assert rows[1][2] == ""  # dominance blank
    assert rows[1][3] == ""  # separation_ratio blank


def test_render_csv_header_only_no_gaps():
    out = gv.render_vad_gap_confidence_csv(_result((0.0, 1.0)))
    rows = list(csv.reader(io.StringIO(out)))
    assert len(rows) == 1  # header alone
    assert rows[0][0] == "recommended_ms"


# ---- handler: cmd_vad_gap_confidence -----------------------------------


def _args(**kw):
    base = dict(
        wav="rec.wav",
        threshold=0.5,
        min_speech_ms=250,
        min_silence_ms=800,
        speech_pad_ms=30,
        max_speech_s=float("inf"),
        json=False,
        csv=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_cmd_human(monkeypatch):
    res = _result_from_gaps([0.2, 0.3, 1.5, 1.6])
    lines = []
    gv.cmd_vad_gap_confidence(
        _args(),
        log=lines.append,
        segmenter=lambda wav, params=None: res,
        availability=lambda: True,
    )
    assert any("recommendation confidence" in l for l in lines)
    assert any("confidence:" in l for l in lines)


def test_cmd_json(monkeypatch):
    res = _result_from_gaps([0.2, 0.3, 1.5, 1.6])
    lines = []
    gv.cmd_vad_gap_confidence(
        _args(json=True),
        log=lines.append,
        segmenter=lambda wav, params=None: res,
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is True
    assert payload["grade"] in ("strong", "moderate", "weak")


def test_cmd_csv(monkeypatch):
    res = _result_from_gaps([0.2, 0.3, 1.5, 1.6])
    lines = []
    gv.cmd_vad_gap_confidence(
        _args(csv=True),
        log=lines.append,
        segmenter=lambda wav, params=None: res,
        availability=lambda: True,
    )
    out = "\n".join(lines)
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0][0] == "recommended_ms"


def test_cmd_unavailable_human():
    lines = []
    gv.cmd_vad_gap_confidence(
        _args(),
        log=lines.append,
        segmenter=lambda wav, params=None: None,
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert "silero-vad" in lines[0]


def test_cmd_unavailable_json():
    lines = []
    gv.cmd_vad_gap_confidence(
        _args(json=True),
        log=lines.append,
        segmenter=lambda wav, params=None: None,
        availability=lambda: False,
    )
    assert json.loads("\n".join(lines))["available"] is False


def test_cmd_unavailable_csv():
    lines = []
    gv.cmd_vad_gap_confidence(
        _args(csv=True),
        log=lines.append,
        segmenter=lambda wav, params=None: None,
        availability=lambda: False,
    )
    assert lines[0].startswith("# silero VAD unavailable")


def test_cmd_passes_knobs_to_params():
    captured = {}

    def fake_segmenter(wav, params=None):
        captured["wav"] = wav
        captured["params"] = params
        return _result_from_gaps([0.2, 0.3, 1.5, 1.6])

    gv.cmd_vad_gap_confidence(
        _args(threshold=0.7, min_silence_ms=400),
        log=lambda *_: None,
        segmenter=fake_segmenter,
        availability=lambda: True,
    )
    assert captured["wav"] == "rec.wav"
    assert captured["params"].threshold == 0.7
    assert captured["params"].min_silence_ms == 400
