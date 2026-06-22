"""Tests for iter-385 — the ``gv vad-gap-recommend-batch`` subcommand (examples/gv.py).

iter-384's ``gv vad-gap-recommend-diff`` compares exactly TWO recordings'
recommended end-of-turn hangovers. ``gv vad-gap-recommend-batch`` generalises that
to N recordings — the recommend analogue of how ``gv vad-gap-sweep`` generalises
``gv vad-gap-diff``: segment a whole corpus under the same shared knobs and report
each recording's ``vad_gap_recommend`` hangover (and its ``vad_gap_confidence``
grade) in one table, plus a corpus median / spread, so an operator can see at a
glance which recordings AGREE on a hangover and which are OUTLIERS.

Like the rest of the VAD-analysis family, the handler takes injected
``segmenter`` / ``availability`` / ``log`` dependencies so every test runs WITHOUT
importing torch / silero-vad and without touching real audio — fast and
deterministic on the x86_64 Linux runner. The pure core
(``vad_gap_recommend_batch``) and the three renderers are exercised directly
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


def _result_from_gaps(gaps, *, name="rec.wav", sample_rate=16000):
    """Build a result whose inter-segment pauses are exactly ``gaps`` (in s).

    Each speech region is a fixed 1.0s long; the gap before region ``i+1`` is
    ``gaps[i]``. Flows through the real ``vad_silence_gaps`` ->
    ``vad_gap_recommend`` / ``vad_gap_confidence`` chain.
    """
    segs = [_Seg(0.0, 1.0)]
    t = 1.0
    for g in gaps:
        start = t + g
        segs.append(_Seg(start, start + 1.0))
        t = start + 1.0
    return _Result(name=name, sample_rate=sample_rate, duration_s=t, segments=segs)


# A clean bimodal recording (strong grade), a clone whose valley sits lower, and
# a higher-valley clone — three distinct recommended hangovers for the corpus.
def _clean(name="a.wav"):
    # Short ~0.2-0.3s, long ~1.4-1.6s — wide empty valley -> strong, high number.
    return _result_from_gaps([0.2, 0.3, 0.25, 1.5, 1.6, 1.4], name=name)


def _lower(name="b.wav"):
    # Same short cluster, lower long cluster (~0.8-0.9s) -> valley sits lower.
    return _result_from_gaps([0.2, 0.3, 0.25, 0.85, 0.9, 0.8], name=name)


def _higher(name="c.wav"):
    # Same short cluster, higher long cluster (~2.4-2.6s) -> valley sits higher.
    return _result_from_gaps([0.2, 0.3, 0.25, 2.5, 2.6, 2.4], name=name)


def _flat(name="flat.wav"):
    # A single segment — no gaps, nothing to recommend.
    return _Result(name=name, sample_rate=16000, duration_s=1.0,
                   segments=[_Seg(0.0, 1.0)])


# ---- parser: registration & defaults -----------------------------------


def test_vad_gap_recommend_batch_in_handler_map():
    assert (
        gv.DEFAULT_HANDLERS["vad-gap-recommend-batch"]
        is gv.cmd_vad_gap_recommend_batch
    )


def test_parser_registers_vad_gap_recommend_batch():
    parser = gv.build_parser()
    args = parser.parse_args(
        ["vad-gap-recommend-batch", "a.wav", "b.wav", "c.wav"]
    )
    assert args.command == "vad-gap-recommend-batch"
    assert args.wavs == ["a.wav", "b.wav", "c.wav"]


def test_parser_vad_gap_recommend_batch_knob_defaults():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-recommend-batch", "a.wav", "b.wav"])
    assert args.threshold == pytest.approx(0.5)
    assert args.min_speech_ms == pytest.approx(250.0)
    assert args.min_silence_ms == pytest.approx(800.0)
    assert args.speech_pad_ms == pytest.approx(30.0)
    assert math.isinf(args.max_speech_s)
    assert args.bias == "balanced"
    assert args.json is False
    assert args.csv is False


def test_parser_vad_gap_recommend_batch_custom_knobs():
    parser = gv.build_parser()
    args = parser.parse_args(
        [
            "vad-gap-recommend-batch",
            "a.wav",
            "b.wav",
            "--threshold",
            "0.7",
            "--min-silence-ms",
            "400",
            "--bias",
            "long",
        ]
    )
    assert args.threshold == 0.7
    assert args.min_silence_ms == 400
    assert args.bias == "long"


def test_parser_vad_gap_recommend_batch_json_csv_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["vad-gap-recommend-batch", "a.wav", "b.wav", "--json", "--csv"]
        )


def test_parser_vad_gap_recommend_batch_bad_bias():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["vad-gap-recommend-batch", "a.wav", "b.wav", "--bias", "huge"]
        )


def test_parser_vad_gap_recommend_batch_requires_at_least_one_wav():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-recommend-batch"])


def test_parser_vad_gap_recommend_batch_accepts_single_wav():
    # nargs="+" allows one; the batch is degenerate but well-defined.
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-recommend-batch", "a.wav"])
    assert args.wavs == ["a.wav"]


# ---- pure helper: _median ----------------------------------------------


def test_median_odd():
    assert gv._median([3, 1, 2]) == 2


def test_median_even():
    assert gv._median([1, 2, 3, 4]) == 2.5


def test_median_single():
    assert gv._median([7]) == 7


# ---- pure core: vad_gap_recommend_batch --------------------------------


def test_batch_rows_agree_with_per_recording_recommend():
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    labels = ["a.wav", "b.wav", "c.wav"]
    d = gv.vad_gap_recommend_batch(results, labels)
    assert [r["recording"] for r in d["rows"]] == labels
    for result, row in zip(results, d["rows"]):
        rec = gv.vad_gap_recommend(result)
        conf = gv.vad_gap_confidence(result)
        assert row["recommended_ms"] == rec["recommended_ms"]
        assert row["recommended_s"] == rec["recommended_s"]
        assert row["grade"] == conf["grade"]
        assert row["num_segments"] == rec["num_segments"]
        assert row["num_gaps"] == rec["num_gaps"]


def test_batch_median_and_aggregates():
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    labels = ["a.wav", "b.wav", "c.wav"]
    d = gv.vad_gap_recommend_batch(results, labels)
    recs = sorted(
        gv.vad_gap_recommend(r)["recommended_ms"] for r in results
    )
    assert d["num_recordings"] == 3
    assert d["num_recommended"] == 3
    assert d["recommended_ms_median"] == round(recs[1], 1)
    assert d["recommended_ms_min"] == round(recs[0], 1)
    assert d["recommended_ms_max"] == round(recs[2], 1)
    assert d["recommended_ms_spread"] == round(recs[2] - recs[0], 1)


def test_batch_delta_from_median():
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    labels = ["a.wav", "b.wav", "c.wav"]
    d = gv.vad_gap_recommend_batch(results, labels)
    median = d["recommended_ms_median"]
    for row in d["rows"]:
        assert row["delta_from_median_ms"] == round(
            row["recommended_ms"] - median, 1
        )
    # The middle recording (median) has a zero delta.
    deltas = [row["delta_from_median_ms"] for row in d["rows"]]
    assert 0.0 in deltas


def test_batch_echoes_bias():
    d = gv.vad_gap_recommend_batch([_clean(), _lower()], ["a.wav", "b.wav"],
                                   bias="short")
    assert d["bias"] == "short"


def test_batch_bias_shifts_the_numbers():
    labels = ["a.wav", "b.wav"]
    results = [_clean("a.wav"), _lower("b.wav")]
    short = gv.vad_gap_recommend_batch(results, labels, bias="short")
    long = gv.vad_gap_recommend_batch(results, labels, bias="long")
    # A short bias sits lower in each valley than a long bias.
    for s_row, l_row in zip(short["rows"], long["rows"]):
        assert s_row["recommended_ms"] < l_row["recommended_ms"]


def test_batch_missing_recording_excluded_from_median():
    # _flat has <2 segments -> no recommendation; it must not feed the median.
    results = [_clean("a.wav"), _lower("b.wav"), _flat("flat.wav")]
    labels = ["a.wav", "b.wav", "flat.wav"]
    d = gv.vad_gap_recommend_batch(results, labels)
    assert d["num_recordings"] == 3
    assert d["num_recommended"] == 2
    flat_row = d["rows"][2]
    assert flat_row["recommended_ms"] is None
    assert flat_row["grade"] is None
    assert flat_row["delta_from_median_ms"] is None
    # The median is over the two recommending recordings only.
    recs = sorted(
        gv.vad_gap_recommend(r)["recommended_ms"]
        for r in (results[0], results[1])
    )
    assert d["recommended_ms_median"] == round((recs[0] + recs[1]) / 2, 1)


def test_batch_all_missing_yields_none_aggregates():
    results = [_flat("a.wav"), _flat("b.wav")]
    labels = ["a.wav", "b.wav"]
    d = gv.vad_gap_recommend_batch(results, labels)
    assert d["num_recommended"] == 0
    assert d["recommended_ms_median"] is None
    assert d["recommended_ms_min"] is None
    assert d["recommended_ms_max"] is None
    assert d["recommended_ms_spread"] is None
    for row in d["rows"]:
        assert row["recommended_ms"] is None
        assert row["delta_from_median_ms"] is None


# ---- human renderer: render_vad_gap_recommend_batch --------------------


def test_human_names_every_recording():
    labels = ["a.wav", "b.wav", "c.wav"]
    lines = gv.render_vad_gap_recommend_batch(
        [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")], labels
    )
    text = "\n".join(lines)
    for label in labels:
        assert label in text
    assert "recommended-hangover batch" in text
    assert "3 recordings" in text


def test_human_shows_corpus_summary():
    lines = gv.render_vad_gap_recommend_batch(
        [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")],
        ["a.wav", "b.wav", "c.wav"],
    )
    text = "\n".join(lines)
    assert "corpus:" in text
    assert "median" in text
    assert "spread" in text
    assert "3/3 recordings recommend" in text


def test_human_echoes_bias():
    lines = gv.render_vad_gap_recommend_batch(
        [_clean(), _lower()], ["a.wav", "b.wav"], bias="long"
    )
    assert any("bias: long" in ln for ln in lines)


def test_human_missing_recording_dashes():
    lines = gv.render_vad_gap_recommend_batch(
        [_clean("a.wav"), _flat("flat.wav")], ["a.wav", "flat.wav"]
    )
    # The flat recording's row carries dashes for its rec/grade/delta.
    flat_line = [ln for ln in lines if "flat.wav" in ln][0]
    assert "-" in flat_line
    # Corpus summary still appears (one recording recommends).
    assert any("1/2 recordings recommend" in ln for ln in lines)


def test_human_all_missing_note():
    lines = gv.render_vad_gap_recommend_batch(
        [_flat("a.wav"), _flat("b.wav")], ["a.wav", "b.wav"]
    )
    text = "\n".join(lines)
    assert "no recording carries a recommendation" in text


def test_human_unavailable_hint():
    lines = gv.render_vad_gap_recommend_batch(
        [None, None], ["a.wav", "b.wav"]
    )
    assert len(lines) == 1
    assert "silero-vad" in lines[0]


# ---- JSON renderer: render_vad_gap_recommend_batch_json ----------------


def test_json_shape():
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")],
            ["a.wav", "b.wav", "c.wav"],
        )
    )
    assert payload["available"] is True
    assert payload["num_recordings"] == 3
    assert payload["num_recommended"] == 3
    assert payload["recommended_ms_median"] is not None
    assert len(payload["rows"]) == 3
    assert payload["rows"][0]["recording"] == "a.wav"
    assert "delta_from_median_ms" in payload["rows"][0]


def test_json_missing_recording_null():
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            [_clean("a.wav"), _flat("flat.wav")], ["a.wav", "flat.wav"]
        )
    )
    flat = payload["rows"][1]
    assert flat["recommended_ms"] is None
    assert flat["grade"] is None
    assert flat["delta_from_median_ms"] is None
    assert payload["num_recommended"] == 1


def test_json_all_missing_null_aggregates():
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            [_flat("a.wav"), _flat("b.wav")], ["a.wav", "b.wav"]
        )
    )
    assert payload["recommended_ms_median"] is None
    assert payload["num_recommended"] == 0


def test_json_unavailable():
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json([None, None], ["a.wav", "b.wav"])
    )
    assert payload["available"] is False
    assert "hint" in payload


# ---- CSV renderer: render_vad_gap_recommend_batch_csv ------------------


def test_csv_one_row_per_recording():
    csv_text = gv.render_vad_gap_recommend_batch_csv(
        [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")],
        ["a.wav", "b.wav", "c.wav"],
    )
    rows = list(csv.reader(io.StringIO(csv_text)))
    assert rows[0] == [
        "recording",
        "bias",
        "num_segments",
        "num_gaps",
        "recommended_ms",
        "grade",
        "delta_from_median_ms",
    ]
    assert len(rows) == 4  # header + 3 recordings
    assert [r[0] for r in rows[1:]] == ["a.wav", "b.wav", "c.wav"]


def test_csv_missing_recording_empty_cells():
    csv_text = gv.render_vad_gap_recommend_batch_csv(
        [_clean("a.wav"), _flat("flat.wav")], ["a.wav", "flat.wav"]
    )
    rows = list(csv.reader(io.StringIO(csv_text)))
    flat = rows[2]
    assert flat[0] == "flat.wav"
    # recommended_ms / grade / delta_from_median_ms cells are empty.
    assert flat[4] == ""
    assert flat[5] == ""
    assert flat[6] == ""


def test_csv_unavailable_comment():
    csv_text = gv.render_vad_gap_recommend_batch_csv(
        [None, None], ["a.wav", "b.wav"]
    )
    assert csv_text.startswith("# silero VAD unavailable")


# ---- handler: cmd_vad_gap_recommend_batch ------------------------------


def _args(**kw):
    base = dict(
        wavs=["a.wav", "b.wav", "c.wav"],
        threshold=0.5,
        min_speech_ms=250,
        min_silence_ms=800,
        speech_pad_ms=30,
        max_speech_s=float("inf"),
        bias="balanced",
        json=False,
        csv=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _corpus_segmenter():
    """Return a segmenter mapping each label to a distinct result."""
    table = {"a.wav": _clean("a.wav"), "b.wav": _lower("b.wav"),
             "c.wav": _higher("c.wav")}

    def seg(wav, params=None):
        return table[wav]

    return seg


def test_cmd_human():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(),
        log=lines.append,
        segmenter=_corpus_segmenter(),
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "recommended-hangover batch" in text
    assert "a.wav" in text and "b.wav" in text and "c.wav" in text


def test_cmd_json():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(json=True),
        log=lines.append,
        segmenter=_corpus_segmenter(),
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is True
    assert payload["num_recordings"] == 3


def test_cmd_csv():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(csv=True),
        log=lines.append,
        segmenter=_corpus_segmenter(),
        availability=lambda: True,
    )
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    assert rows[0][0] == "recording"
    assert len(rows) == 4


def test_cmd_threads_bias():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(json=True, bias="long"),
        log=lines.append,
        segmenter=_corpus_segmenter(),
        availability=lambda: True,
    )
    assert json.loads("\n".join(lines))["bias"] == "long"


def test_cmd_passes_shared_knobs_to_every_run():
    captured = []

    def fake_segmenter(wav, params=None):
        captured.append((wav, params.threshold, params.min_silence_ms))
        return _corpus_segmenter()(wav)

    gv.cmd_vad_gap_recommend_batch(
        _args(threshold=0.7, min_silence_ms=400),
        log=lambda *_: None,
        segmenter=fake_segmenter,
        availability=lambda: True,
    )
    assert len(captured) == 3
    # Every run shares the same knobs.
    for wav, threshold, min_silence in captured:
        assert threshold == 0.7
        assert min_silence == 400
    assert [c[0] for c in captured] == ["a.wav", "b.wav", "c.wav"]


def test_cmd_unavailable_human():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(),
        log=lines.append,
        segmenter=lambda wav, params=None: None,
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert "silero-vad" in lines[0]


def test_cmd_unavailable_json():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(json=True),
        log=lines.append,
        segmenter=lambda wav, params=None: None,
        availability=lambda: False,
    )
    assert json.loads("\n".join(lines))["available"] is False


def test_cmd_unavailable_csv():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(csv=True),
        log=lines.append,
        segmenter=lambda wav, params=None: None,
        availability=lambda: False,
    )
    assert lines[0].startswith("# silero VAD unavailable")


# ============ iter-386 — --sort-by ordering for the batch ================
#
# Reorders the RECORDING rows of the batch so the most-useful read first:
#   recommended = shortest hangover first (recommended ms ascending),
#   grade       = most-trustworthy first (confidence descending),
#   delta       = biggest outliers first (|Δmedian| descending).
# Render-only: the core vad_gap_recommend_batch stays argument-order; the three
# renderers gain a sort_by kwarg, and the handler reads args.sort_by. The default
# None keeps argument order (byte-identical to the pre-sort output).


def _graded_corpus():
    """A corpus spanning every grade so a grade sort has something to reorder.

    Built so the recommended-ms order and the grade order DISAGREE — that way a
    grade sort and a recommended sort produce visibly different orderings, proving
    each key is applied independently.
    """
    # name -> (result, recommended_ms, grade) for reference in assertions:
    #   strong.wav  : _clean   ms=850.0  grade=strong
    #   strongL.wav : _lower   ms=550.0  grade=strong
    #   mod.wav     : moderate ms=550.0  grade=moderate
    #   none.wav    : smear    ms=250.0  grade=none
    return {
        "strong.wav": _clean("strong.wav"),
        "strongL.wav": _lower("strongL.wav"),
        "mod.wav": _result_from_gaps([0.2, 0.3, 0.4, 0.7, 0.9], name="mod.wav"),
        "none.wav": _result_from_gaps([0.5, 0.5, 0.5, 0.5], name="none.wav"),
    }


# ---- argparse type: gap_recommend_batch_sort_type ----------------------


@pytest.mark.parametrize("key", ["recommended", "grade", "delta"])
def test_batch_sort_type_accepts_each_key(key):
    assert gv.gap_recommend_batch_sort_type(key) == key


def test_batch_sort_type_is_case_insensitive_and_strips():
    assert gv.gap_recommend_batch_sort_type("  Delta ") == "delta"


def test_batch_sort_type_rejects_empty_and_unknown():
    for bad in ["", "spread", "median", "bogus"]:
        with pytest.raises(gv.argparse.ArgumentTypeError):
            gv.gap_recommend_batch_sort_type(bad)


def test_batch_sort_type_rejects_non_string():
    with pytest.raises(gv.argparse.ArgumentTypeError):
        gv.gap_recommend_batch_sort_type(3)


# ---- parser wiring -----------------------------------------------------


def test_parser_vad_gap_recommend_batch_sort_by_default_none():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-recommend-batch", "a.wav", "b.wav"])
    assert args.sort_by is None


def test_parser_vad_gap_recommend_batch_sort_by_value():
    parser = gv.build_parser()
    args = parser.parse_args(
        ["vad-gap-recommend-batch", "a.wav", "b.wav", "--sort-by", "delta"]
    )
    assert args.sort_by == "delta"


def test_parser_vad_gap_recommend_batch_sort_by_bad_key():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["vad-gap-recommend-batch", "a.wav", "b.wav", "--sort-by", "spread"]
        )


# ---- pure helper: _sort_batch_rows -------------------------------------


def _rows(*tuples):
    """Build batch-shaped rows from (recording, recommended_ms, grade, delta)."""
    return [
        {
            "recording": rec,
            "recommended_ms": ms,
            "grade": grade,
            "delta_from_median_ms": delta,
        }
        for rec, ms, grade, delta in tuples
    ]


def test_sort_batch_rows_none_keeps_argument_order():
    rows = _rows(("a", 300.0, "weak", 50.0), ("b", 100.0, "strong", -150.0))
    out = gv._sort_batch_rows(rows, None)
    assert [r["recording"] for r in out] == ["a", "b"]
    # A copy — the source is not mutated/aliased.
    assert out is not rows


def test_sort_batch_rows_recommended_ascending():
    rows = _rows(
        ("a", 300.0, "weak", 0.0),
        ("b", 100.0, "strong", -200.0),
        ("d", 500.0, "moderate", 200.0),
    )
    out = gv._sort_batch_rows(rows, "recommended")
    assert [r["recording"] for r in out] == ["b", "a", "d"]


def test_sort_batch_rows_grade_descending():
    rows = _rows(
        ("a", 300.0, "weak", 0.0),
        ("b", 100.0, "strong", -200.0),
        ("d", 500.0, "moderate", 200.0),
    )
    out = gv._sort_batch_rows(rows, "grade")
    assert [r["recording"] for r in out] == ["b", "d", "a"]


def test_sort_batch_rows_delta_by_absolute_value_descending():
    rows = _rows(
        ("a", 300.0, "weak", 50.0),
        ("b", 100.0, "strong", -200.0),
        ("d", 400.0, "moderate", 150.0),
    )
    out = gv._sort_batch_rows(rows, "delta")
    # |−200| > |150| > |50|
    assert [r["recording"] for r in out] == ["b", "d", "a"]


def test_sort_batch_rows_missing_sort_last_under_every_key():
    rows = _rows(
        ("miss", None, None, None),
        ("a", 300.0, "weak", 50.0),
        ("b", 100.0, "strong", -200.0),
    )
    for key in ("recommended", "grade", "delta"):
        out = gv._sort_batch_rows(rows, key)
        assert out[-1]["recording"] == "miss", key


def test_sort_batch_rows_is_stable_on_ties():
    # Two rows tie on grade; the one earlier in argument order stays first.
    rows = _rows(
        ("first", 200.0, "strong", 0.0),
        ("second", 800.0, "strong", 600.0),
    )
    out = gv._sort_batch_rows(rows, "grade")
    assert [r["recording"] for r in out] == ["first", "second"]


def test_sort_batch_rows_unknown_key_keeps_order():
    rows = _rows(("a", 300.0, "weak", 0.0), ("b", 100.0, "strong", -200.0))
    out = gv._sort_batch_rows(rows, "nonsense")
    assert [r["recording"] for r in out] == ["a", "b"]


def test_sort_batch_rows_does_not_mutate_source():
    rows = _rows(("a", 300.0, "weak", 50.0), ("b", 100.0, "strong", -200.0))
    gv._sort_batch_rows(rows, "recommended")
    assert [r["recording"] for r in rows] == ["a", "b"]


# ---- human renderer with sort_by ---------------------------------------


def test_human_sort_by_reorders_rows():
    labels = ["a.wav", "b.wav", "c.wav"]
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    lines = gv.render_vad_gap_recommend_batch(
        results, labels, sort_by="recommended"
    )
    # Body rows (skip the 3 header lines): lower(550) < clean(850) < higher(1350).
    body = [ln for ln in lines if any(w in ln for w in labels)]
    order = [next(w for w in labels if w in ln) for ln in body]
    assert order == ["b.wav", "a.wav", "c.wav"]


def test_human_sort_by_default_matches_argument_order():
    labels = ["a.wav", "b.wav", "c.wav"]
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    lines = gv.render_vad_gap_recommend_batch(results, labels)
    body = [ln for ln in lines if any(w in ln for w in labels)]
    order = [next(w for w in labels if w in ln) for ln in body]
    assert order == labels


def test_human_sort_by_names_the_ordering():
    lines = gv.render_vad_gap_recommend_batch(
        [_clean("a.wav"), _lower("b.wav")], ["a.wav", "b.wav"], sort_by="delta"
    )
    assert any("sorted by: delta" in ln for ln in lines)


def test_human_no_sort_omits_ordering_note():
    lines = gv.render_vad_gap_recommend_batch(
        [_clean("a.wav"), _lower("b.wav")], ["a.wav", "b.wav"]
    )
    assert not any("sorted by" in ln for ln in lines)


def test_human_sort_does_not_change_corpus_summary():
    labels = ["a.wav", "b.wav", "c.wav"]
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    unsorted = gv.render_vad_gap_recommend_batch(results, labels)
    srt = gv.render_vad_gap_recommend_batch(results, labels, sort_by="grade")
    summary_u = [ln for ln in unsorted if "corpus:" in ln][0]
    summary_s = [ln for ln in srt if "corpus:" in ln][0]
    assert summary_u == summary_s


# ---- JSON renderer with sort_by ----------------------------------------


def test_json_sort_by_reorders_rows_and_echoes_key():
    labels = ["a.wav", "b.wav", "c.wav"]
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            results, labels, sort_by="recommended"
        )
    )
    assert payload["sort_by"] == "recommended"
    order = [r["recording"] for r in payload["rows"]]
    assert order == ["b.wav", "a.wav", "c.wav"]


def test_json_no_sort_omits_sort_by_key():
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            [_clean("a.wav"), _lower("b.wav")], ["a.wav", "b.wav"]
        )
    )
    assert "sort_by" not in payload
    assert [r["recording"] for r in payload["rows"]] == ["a.wav", "b.wav"]


def test_json_sort_preserves_aggregates():
    labels = ["a.wav", "b.wav", "c.wav"]
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    base = json.loads(
        gv.render_vad_gap_recommend_batch_json(results, labels)
    )
    srt = json.loads(
        gv.render_vad_gap_recommend_batch_json(results, labels, sort_by="delta")
    )
    for key in (
        "recommended_ms_median",
        "recommended_ms_min",
        "recommended_ms_max",
        "recommended_ms_spread",
        "num_recommended",
    ):
        assert base[key] == srt[key]


# ---- CSV renderer with sort_by -----------------------------------------


def test_csv_sort_by_reorders_data_rows_same_header():
    labels = ["a.wav", "b.wav", "c.wav"]
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    base = list(csv.reader(io.StringIO(
        gv.render_vad_gap_recommend_batch_csv(results, labels)
    )))
    srt = list(csv.reader(io.StringIO(
        gv.render_vad_gap_recommend_batch_csv(
            results, labels, sort_by="recommended"
        )
    )))
    # Header unchanged; sorted run unions cleanly with the unsorted one.
    assert base[0] == srt[0]
    assert [row[0] for row in srt[1:]] == ["b.wav", "a.wav", "c.wav"]
    # Same set of recordings, just reordered.
    assert sorted(row[0] for row in base[1:]) == sorted(
        row[0] for row in srt[1:]
    )


# ---- handler threads sort_by -------------------------------------------


def test_cmd_threads_sort_by_human():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(sort_by="recommended"),
        log=lines.append,
        segmenter=_corpus_segmenter(),
        availability=lambda: True,
    )
    body = [ln for ln in lines if any(w in ln for w in ("a.wav", "b.wav", "c.wav"))]
    order = [next(w for w in ("a.wav", "b.wav", "c.wav") if w in ln) for ln in body]
    assert order == ["b.wav", "a.wav", "c.wav"]
    assert any("sorted by: recommended" in ln for ln in lines)


def test_cmd_threads_sort_by_json():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(json=True, sort_by="delta"),
        log=lines.append,
        segmenter=_corpus_segmenter(),
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["sort_by"] == "delta"


def test_cmd_sort_by_default_none_keeps_argument_order():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(json=True),
        log=lines.append,
        segmenter=_corpus_segmenter(),
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert "sort_by" not in payload
    assert [r["recording"] for r in payload["rows"]] == ["a.wav", "b.wav", "c.wav"]


def test_cmd_unavailable_still_threads_sort_by_json():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(json=True, sort_by="grade"),
        log=lines.append,
        segmenter=lambda wav, params=None: None,
        availability=lambda: False,
    )
    # Degrades cleanly — no crash — even with a sort requested.
    assert json.loads("\n".join(lines))["available"] is False
