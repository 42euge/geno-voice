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


# ========================================================================
# iter-387 — --top-n: keep only the N most-useful recordings.
# ========================================================================
# The count companion of iter-386's --sort-by (an ORDERING): --top-n N keeps a
# FIXED COUNT of recordings, applied AFTER --sort-by, so "--sort-by delta --top-n 3"
# shows the 3 biggest outliers and "--sort-by grade --top-n 3" the 3 most-trustworthy.
# The batch analogue of iter-380's knob-sweep --top-n. Render-only: the core
# vad_gap_recommend_batch stays the full-corpus primitive; the three renderers gain a
# top_n kwarg threaded by the handler from args.top_n. The corpus aggregates stay
# computed over the WHOLE corpus and are unaffected. The default None shows every
# recording (byte-identical to the pre-cap output).


# ---- parser wiring -----------------------------------------------------


def test_parser_vad_gap_recommend_batch_top_n_default_none():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-recommend-batch", "a.wav", "b.wav"])
    assert args.top_n is None


def test_parser_vad_gap_recommend_batch_top_n_value():
    parser = gv.build_parser()
    args = parser.parse_args(
        ["vad-gap-recommend-batch", "a.wav", "b.wav", "--top-n", "2"]
    )
    assert args.top_n == 2


def test_parser_vad_gap_recommend_batch_top_n_rejects_zero_and_negative():
    parser = gv.build_parser()
    for bad in ["0", "-1"]:
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["vad-gap-recommend-batch", "a.wav", "b.wav", "--top-n", bad]
            )


def test_parser_vad_gap_recommend_batch_top_n_composes_with_sort_by():
    parser = gv.build_parser()
    args = parser.parse_args(
        [
            "vad-gap-recommend-batch",
            "a.wav",
            "b.wav",
            "c.wav",
            "--sort-by",
            "delta",
            "--top-n",
            "2",
        ]
    )
    assert args.sort_by == "delta"
    assert args.top_n == 2


# ---- pure helper: _truncate_batch_rows ---------------------------------


def test_truncate_batch_rows_none_keeps_all_and_copies():
    rows = _rows(("a", 300.0, "weak", 50.0), ("b", 100.0, "strong", -150.0))
    out = gv._truncate_batch_rows(rows, None)
    assert [r["recording"] for r in out] == ["a", "b"]
    # A copy — the source is not aliased.
    assert out is not rows


def test_truncate_batch_rows_keeps_first_n():
    rows = _rows(
        ("a", 300.0, "weak", 0.0),
        ("b", 100.0, "strong", -200.0),
        ("d", 500.0, "moderate", 200.0),
    )
    out = gv._truncate_batch_rows(rows, 2)
    assert [r["recording"] for r in out] == ["a", "b"]


def test_truncate_batch_rows_n_larger_than_rows_keeps_all():
    rows = _rows(("a", 300.0, "weak", 0.0), ("b", 100.0, "strong", -200.0))
    out = gv._truncate_batch_rows(rows, 99)
    assert [r["recording"] for r in out] == ["a", "b"]


def test_truncate_batch_rows_does_not_mutate_source():
    rows = _rows(
        ("a", 300.0, "weak", 0.0),
        ("b", 100.0, "strong", -200.0),
        ("d", 500.0, "moderate", 200.0),
    )
    gv._truncate_batch_rows(rows, 1)
    assert [r["recording"] for r in rows] == ["a", "b", "d"]


def test_truncate_batch_rows_applied_after_sort():
    # Sort by recommended ASC then keep the 2 shortest hangovers.
    rows = _rows(
        ("a", 300.0, "weak", 0.0),
        ("b", 100.0, "strong", -200.0),
        ("d", 500.0, "moderate", 200.0),
    )
    out = gv._truncate_batch_rows(gv._sort_batch_rows(rows, "recommended"), 2)
    assert [r["recording"] for r in out] == ["b", "a"]


# ---- human renderer with top_n -----------------------------------------


def test_human_top_n_keeps_first_n_after_sort():
    labels = ["a.wav", "b.wav", "c.wav"]
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    lines = gv.render_vad_gap_recommend_batch(
        results, labels, sort_by="recommended", top_n=2
    )
    body = [ln for ln in lines if any(w in ln for w in labels)]
    order = [next(w for w in labels if w in ln) for ln in body]
    # lower(550) < clean(850) < higher(1350); keep the 2 shortest.
    assert order == ["b.wav", "a.wav"]


def test_human_top_n_default_shows_every_recording():
    labels = ["a.wav", "b.wav", "c.wav"]
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    lines = gv.render_vad_gap_recommend_batch(results, labels)
    body = [ln for ln in lines if any(w in ln for w in labels)]
    assert len(body) == 3


def test_human_top_n_names_the_cap_when_truncating():
    labels = ["a.wav", "b.wav", "c.wav"]
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    lines = gv.render_vad_gap_recommend_batch(results, labels, top_n=2)
    assert any("top 2 of 3" in ln for ln in lines)


def test_human_top_n_omits_note_when_cap_drops_nothing():
    labels = ["a.wav", "b.wav"]
    results = [_clean("a.wav"), _lower("b.wav")]
    # top_n >= row count: every row kept, so no truncation note.
    lines = gv.render_vad_gap_recommend_batch(results, labels, top_n=5)
    assert not any("top " in ln and " of " in ln for ln in lines)


def test_human_top_n_does_not_change_corpus_summary():
    labels = ["a.wav", "b.wav", "c.wav"]
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    full = gv.render_vad_gap_recommend_batch(results, labels)
    capped = gv.render_vad_gap_recommend_batch(
        results, labels, sort_by="delta", top_n=1
    )
    summary_full = [ln for ln in full if "corpus:" in ln][0]
    summary_capped = [ln for ln in capped if "corpus:" in ln][0]
    assert summary_full == summary_capped


# ---- JSON renderer with top_n ------------------------------------------


def test_json_top_n_truncates_rows_and_echoes_key():
    labels = ["a.wav", "b.wav", "c.wav"]
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            results, labels, sort_by="recommended", top_n=2
        )
    )
    assert payload["top_n"] == 2
    order = [r["recording"] for r in payload["rows"]]
    assert order == ["b.wav", "a.wav"]


def test_json_no_top_n_omits_key():
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            [_clean("a.wav"), _lower("b.wav")], ["a.wav", "b.wav"]
        )
    )
    assert "top_n" not in payload
    assert len(payload["rows"]) == 2


def test_json_top_n_preserves_corpus_aggregates():
    labels = ["a.wav", "b.wav", "c.wav"]
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    base = json.loads(gv.render_vad_gap_recommend_batch_json(results, labels))
    capped = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            results, labels, sort_by="delta", top_n=1
        )
    )
    for key in (
        "recommended_ms_median",
        "recommended_ms_min",
        "recommended_ms_max",
        "recommended_ms_spread",
        "num_recommended",
        "num_recordings",
    ):
        assert base[key] == capped[key]
    # Aggregates span the whole corpus even though only 1 row is shown.
    assert len(capped["rows"]) == 1


# ---- CSV renderer with top_n -------------------------------------------


def test_csv_top_n_truncates_data_rows_same_header():
    labels = ["a.wav", "b.wav", "c.wav"]
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    base = list(csv.reader(io.StringIO(
        gv.render_vad_gap_recommend_batch_csv(results, labels)
    )))
    capped = list(csv.reader(io.StringIO(
        gv.render_vad_gap_recommend_batch_csv(
            results, labels, sort_by="recommended", top_n=2
        )
    )))
    # Header unchanged; truncated run unions cleanly with the full one.
    assert base[0] == capped[0]
    assert [row[0] for row in capped[1:]] == ["b.wav", "a.wav"]


# ---- handler threads top_n ---------------------------------------------


def test_cmd_threads_top_n_human():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(sort_by="recommended", top_n=2),
        log=lines.append,
        segmenter=_corpus_segmenter(),
        availability=lambda: True,
    )
    body = [ln for ln in lines if any(w in ln for w in ("a.wav", "b.wav", "c.wav"))]
    order = [next(w for w in ("a.wav", "b.wav", "c.wav") if w in ln) for ln in body]
    assert order == ["b.wav", "a.wav"]
    assert any("top 2 of 3" in ln for ln in lines)


def test_cmd_threads_top_n_json():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(json=True, sort_by="delta", top_n=1),
        log=lines.append,
        segmenter=_corpus_segmenter(),
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["top_n"] == 1
    assert len(payload["rows"]) == 1


def test_cmd_top_n_default_none_shows_every_recording():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(json=True),
        log=lines.append,
        segmenter=_corpus_segmenter(),
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert "top_n" not in payload
    assert len(payload["rows"]) == 3


def test_cmd_unavailable_still_threads_top_n_json():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(json=True, top_n=2),
        log=lines.append,
        segmenter=lambda wav, params=None: None,
        availability=lambda: False,
    )
    # Degrades cleanly — no crash — even with a cap requested.
    assert json.loads("\n".join(lines))["available"] is False


# ---- iter-388: --summary single most-representative-recording verdict ----


def _batch_rows(*specs):
    """Build a minimal list of batch rows from (recording, recommended_ms, grade,
    delta_from_median_ms) tuples — enough for the _best_batch_row reduction.
    """
    return [
        {
            "recording": rec,
            "num_segments": 7,
            "num_gaps": 6,
            "recommended_ms": ms,
            "recommended_s": None if ms is None else ms / 1000.0,
            "grade": grade,
            "delta_from_median_ms": delta,
        }
        for rec, ms, grade, delta in specs
    ]


def test_best_batch_row_none_when_no_recommendation():
    rows = _batch_rows(("a.wav", None, None, None), ("b.wav", None, None, None))
    assert gv._best_batch_row(rows) is None


def test_best_batch_row_empty_is_none():
    assert gv._best_batch_row([]) is None


def test_best_batch_row_picks_nearest_median():
    # b.wav sits exactly at the median (Δ 0); the others are farther.
    rows = _batch_rows(
        ("a.wav", 1350.0, "strong", 500.0),
        ("b.wav", 850.0, "strong", 0.0),
        ("c.wav", 550.0, "strong", -300.0),
    )
    assert gv._best_batch_row(rows)["recording"] == "b.wav"


def test_best_batch_row_uses_absolute_delta():
    # The nearest by MAGNITUDE wins regardless of sign — -100 beats +250.
    rows = _batch_rows(
        ("a.wav", 1100.0, "strong", 250.0),
        ("b.wav", 750.0, "strong", -100.0),
    )
    assert gv._best_batch_row(rows)["recording"] == "b.wav"


def test_best_batch_row_ties_break_to_higher_grade():
    # Equal |Δ|; the more-trustworthy recording wins.
    rows = _batch_rows(
        ("a.wav", 950.0, "weak", 100.0),
        ("b.wav", 750.0, "strong", -100.0),
    )
    assert gv._best_batch_row(rows)["recording"] == "b.wav"


def test_best_batch_row_full_tie_breaks_to_earliest():
    # Equal |Δ| AND equal grade -> earliest argument position wins (stable min).
    rows = _batch_rows(
        ("a.wav", 950.0, "strong", 100.0),
        ("b.wav", 750.0, "strong", -100.0),
    )
    assert gv._best_batch_row(rows)["recording"] == "a.wav"


def test_best_batch_row_skips_missing():
    # A <2-segment row (delta None) is never picked even if listed first.
    rows = _batch_rows(
        ("a.wav", None, None, None),
        ("b.wav", 850.0, "moderate", 0.0),
    )
    assert gv._best_batch_row(rows)["recording"] == "b.wav"


def test_best_batch_row_does_not_mutate():
    rows = _batch_rows(
        ("a.wav", 1350.0, "strong", 500.0),
        ("b.wav", 850.0, "strong", 0.0),
    )
    before = [dict(r) for r in rows]
    gv._best_batch_row(rows)
    assert rows == before


def test_best_batch_row_independent_of_order():
    # Reordering the input does not change the pick (it is a MIN, not a head).
    specs = [
        ("a.wav", 1350.0, "strong", 500.0),
        ("b.wav", 850.0, "strong", 0.0),
        ("c.wav", 550.0, "strong", -300.0),
    ]
    forward = gv._best_batch_row(_batch_rows(*specs))
    backward = gv._best_batch_row(_batch_rows(*reversed(specs)))
    assert forward["recording"] == backward["recording"] == "b.wav"


def test_format_batch_summary_verdict_spells_the_call():
    row = _batch_rows(("b.wav", 850.0, "strong", 0.0))[0]
    line = gv._format_batch_summary_verdict(row, "balanced")
    assert "representative: b.wav" in line
    assert "--min-silence-ms 850" in line
    assert "[balanced]" in line
    assert "confidence strong" in line
    assert "Δmedian +0.0ms" in line or "Δmedian 0.0ms" in line


def test_format_batch_summary_verdict_signs_negative_delta():
    row = _batch_rows(("b.wav", 750.0, "moderate", -100.0))[0]
    line = gv._format_batch_summary_verdict(row, "short")
    assert "Δmedian -100.0ms" in line
    assert "[short]" in line


# ---- parser: --summary ---------------------------------------------------


def test_parser_summary_default_false():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-recommend-batch", "a.wav", "b.wav"])
    assert args.summary is False


def test_parser_summary_store_true():
    parser = gv.build_parser()
    args = parser.parse_args(
        ["vad-gap-recommend-batch", "a.wav", "b.wav", "--summary"]
    )
    assert args.summary is True


# ---- human renderer: --summary -------------------------------------------


def _corpus():
    return [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]


def test_render_human_summary_names_representative():
    lines = gv.render_vad_gap_recommend_batch(
        _corpus(), ["a.wav", "b.wav", "c.wav"], summary=True
    )
    assert lines[0] == "silero VAD recommended-hangover batch summary (3 recordings)"
    assert lines[1] == "  bias: balanced"
    # a.wav sits at the median in this corpus (clean is the central one).
    assert "representative: a.wav" in lines[2]


def test_render_human_summary_independent_of_sort_and_top_n():
    base = gv.render_vad_gap_recommend_batch(
        _corpus(), ["a.wav", "b.wav", "c.wav"], summary=True
    )
    shaped = gv.render_vad_gap_recommend_batch(
        _corpus(), ["a.wav", "b.wav", "c.wav"],
        summary=True, sort_by="delta", top_n=1,
    )
    assert base == shaped


def test_render_human_summary_no_recommendation_note():
    lines = gv.render_vad_gap_recommend_batch(
        [_flat("f.wav")], ["f.wav"], summary=True
    )
    assert "no recording carries a recommendation" in lines[-1]


def test_render_human_default_is_full_table():
    lines = gv.render_vad_gap_recommend_batch(
        _corpus(), ["a.wav", "b.wav", "c.wav"]
    )
    # Full table has a header row + one row per recording + corpus line.
    assert "summary" not in lines[0]
    assert any("corpus:" in ln for ln in lines)


# ---- JSON renderer: --summary --------------------------------------------


def test_render_json_summary_shape():
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            _corpus(), ["a.wav", "b.wav", "c.wav"], summary=True
        )
    )
    assert payload["available"] is True
    assert payload["summary"] is True
    assert payload["best"]["recording"] == "a.wav"
    # The single-best replaces the rows list.
    assert "rows" not in payload


def test_render_json_summary_preserves_corpus_aggregates():
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            _corpus(), ["a.wav", "b.wav", "c.wav"], summary=True
        )
    )
    # The representative is central WITHIN these aggregates, which are still carried.
    assert payload["num_recordings"] == 3
    assert payload["num_recommended"] == 3
    assert payload["recommended_ms_median"] is not None


def test_render_json_summary_default_has_no_summary_or_best_key():
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            _corpus(), ["a.wav", "b.wav", "c.wav"]
        )
    )
    assert "summary" not in payload
    assert "best" not in payload
    assert len(payload["rows"]) == 3


def test_render_json_summary_best_null_when_no_recommendation():
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            [_flat("f.wav")], ["f.wav"], summary=True
        )
    )
    assert payload["summary"] is True
    assert payload["best"] is None


def test_render_json_summary_independent_of_sort_and_top_n():
    base = gv.render_vad_gap_recommend_batch_json(
        _corpus(), ["a.wav", "b.wav", "c.wav"], summary=True
    )
    shaped = gv.render_vad_gap_recommend_batch_json(
        _corpus(), ["a.wav", "b.wav", "c.wav"],
        summary=True, sort_by="delta", top_n=1,
    )
    assert base == shaped


# ---- CSV renderer: --summary ---------------------------------------------


def test_render_csv_summary_one_best_row():
    text = gv.render_vad_gap_recommend_batch_csv(
        _corpus(), ["a.wav", "b.wav", "c.wav"], summary=True
    )
    rows = list(csv.reader(io.StringIO(text)))
    # Header + exactly one data row (the most-representative).
    assert len(rows) == 2
    assert rows[1][0] == "a.wav"


def test_render_csv_summary_header_unchanged():
    full = gv.render_vad_gap_recommend_batch_csv(
        _corpus(), ["a.wav", "b.wav", "c.wav"]
    )
    summ = gv.render_vad_gap_recommend_batch_csv(
        _corpus(), ["a.wav", "b.wav", "c.wav"], summary=True
    )
    # Same header -> a summary CSV unions cleanly with a full batch.
    assert full.splitlines()[0] == summ.splitlines()[0]


def test_render_csv_summary_header_only_when_no_recommendation():
    text = gv.render_vad_gap_recommend_batch_csv(
        [_flat("f.wav")], ["f.wav"], summary=True
    )
    assert len(text.splitlines()) == 1  # header only


# ---- handler threads summary ---------------------------------------------


def test_cmd_summary_human_path():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(summary=True),
        log=lines.append,
        segmenter=_corpus_segmenter(),
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "batch summary" in text
    assert "representative:" in text


def test_cmd_summary_json_path():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(json=True, summary=True),
        log=lines.append,
        segmenter=_corpus_segmenter(),
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["summary"] is True
    assert payload["best"]["recording"] == "a.wav"


def test_cmd_summary_default_false_shows_full_table():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(json=True),
        log=lines.append,
        segmenter=_corpus_segmenter(),
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert "summary" not in payload
    assert len(payload["rows"]) == 3


def test_cmd_unavailable_still_threads_summary_json():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(json=True, summary=True),
        log=lines.append,
        segmenter=lambda wav, params=None: None,
        availability=lambda: False,
    )
    assert json.loads("\n".join(lines))["available"] is False


# ========================================================================
# iter-389 --min-grade: drop recordings below a confidence floor before the
# table / summary. Applied FIRST (before --sort-by / --top-n / --summary); the
# corpus aggregates stay computed over the WHOLE corpus. The batch analogue of
# the iter-376 --min-grade on the knob sweep, reusing _filter_knob_rows_by_grade.
# ========================================================================


def _mixed_grade_corpus():
    """A corpus spanning strong / moderate / none grades, with labels.

    grades (verified): strong.wav=strong, mod.wav=moderate, none.wav=none.
    Returns (results, labels) so a --min-grade floor has rows to drop.
    """
    strong = _result_from_gaps([0.2, 0.3, 0.25, 1.5, 1.6, 1.4], name="strong.wav")
    mod = _result_from_gaps([0.2, 0.3, 0.4, 0.7, 0.9], name="mod.wav")
    none = _result_from_gaps([0.5, 0.5, 0.5, 0.5], name="none.wav")
    return [strong, mod, none], ["strong.wav", "mod.wav", "none.wav"]


def _mixed_grade_segmenter():
    results, labels = _mixed_grade_corpus()
    table = dict(zip(labels, results))

    def seg(wav, params=None):
        return table[wav]

    return seg


# ---- argparse type reuse: gap_confidence_grade_type --------------------


@pytest.mark.parametrize("grade", ["weak", "moderate", "strong"])
def test_batch_min_grade_type_accepts_each(grade):
    assert gv.gap_confidence_grade_type(grade) == grade


def test_batch_min_grade_type_rejects_none_and_empty():
    for bad in ["", "none", "bogus"]:
        with pytest.raises(gv.argparse.ArgumentTypeError):
            gv.gap_confidence_grade_type(bad)


# ---- parser wiring -----------------------------------------------------


def test_parser_vad_gap_recommend_batch_min_grade_default_none():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-recommend-batch", "a.wav", "b.wav"])
    assert args.min_grade is None


def test_parser_vad_gap_recommend_batch_min_grade_value():
    parser = gv.build_parser()
    args = parser.parse_args(
        ["vad-gap-recommend-batch", "a.wav", "b.wav", "--min-grade", "strong"]
    )
    assert args.min_grade == "strong"


def test_parser_vad_gap_recommend_batch_min_grade_case_insensitive():
    parser = gv.build_parser()
    args = parser.parse_args(
        ["vad-gap-recommend-batch", "a.wav", "b.wav", "--min-grade", "  Moderate "]
    )
    assert args.min_grade == "moderate"


def test_parser_vad_gap_recommend_batch_min_grade_rejects_none():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["vad-gap-recommend-batch", "a.wav", "b.wav", "--min-grade", "none"]
        )


def test_parser_vad_gap_recommend_batch_min_grade_composes_with_sort_and_top_n():
    parser = gv.build_parser()
    args = parser.parse_args(
        [
            "vad-gap-recommend-batch", "a.wav", "b.wav",
            "--min-grade", "moderate", "--sort-by", "delta", "--top-n", "1",
        ]
    )
    assert args.min_grade == "moderate"
    assert args.sort_by == "delta"
    assert args.top_n == 1


# ---- human renderer with min_grade -------------------------------------


def test_human_min_grade_drops_below_floor():
    results, labels = _mixed_grade_corpus()
    lines = gv.render_vad_gap_recommend_batch(results, labels, min_grade="strong")
    body = "\n".join(lines)
    assert "strong.wav" in body
    # moderate / none recordings are below the 'strong' floor.
    assert "mod.wav" not in body
    assert "none.wav" not in body


def test_human_min_grade_moderate_keeps_strong_and_moderate():
    results, labels = _mixed_grade_corpus()
    lines = gv.render_vad_gap_recommend_batch(results, labels, min_grade="moderate")
    body = "\n".join(lines)
    assert "strong.wav" in body and "mod.wav" in body
    assert "none.wav" not in body


def test_human_min_grade_names_the_floor():
    results, labels = _mixed_grade_corpus()
    lines = gv.render_vad_gap_recommend_batch(results, labels, min_grade="strong")
    assert any("min grade: strong" in ln for ln in lines)


def test_human_no_min_grade_omits_floor_note():
    results, labels = _mixed_grade_corpus()
    lines = gv.render_vad_gap_recommend_batch(results, labels)
    assert not any("min grade" in ln for ln in lines)


def test_human_min_grade_corpus_summary_over_whole_corpus():
    # The corpus line must reflect ALL 3 recordings, not just the survivors.
    results, labels = _mixed_grade_corpus()
    full = gv.render_vad_gap_recommend_batch(results, labels)
    filt = gv.render_vad_gap_recommend_batch(results, labels, min_grade="strong")
    corpus_full = [ln for ln in full if "corpus:" in ln][0]
    corpus_filt = [ln for ln in filt if "corpus:" in ln][0]
    assert corpus_full == corpus_filt


def test_human_min_grade_removes_every_row_note():
    # No recording reaches 'strong' -> a single note replaces the body.
    mod = _result_from_gaps([0.2, 0.3, 0.4, 0.7, 0.9], name="mod.wav")
    lines = gv.render_vad_gap_recommend_batch(
        [mod], ["mod.wav"], min_grade="strong"
    )
    assert any("no recording reaches confidence grade 'strong'" in ln for ln in lines)


def test_human_min_grade_applied_before_top_n():
    # min_grade drops to 1 survivor; top_n count is over the FILTERED set.
    results, labels = _mixed_grade_corpus()
    lines = gv.render_vad_gap_recommend_batch(
        results, labels, min_grade="strong", top_n=5
    )
    # Only 1 strong recording survives; the (top N of M) note uses the kept count,
    # so a top_n >= kept count is a no-op (no note).
    assert not any("top " in ln and " of " in ln for ln in lines)
    assert "strong.wav" in "\n".join(lines)


# ---- JSON renderer with min_grade --------------------------------------


def test_json_min_grade_filters_rows_and_echoes_key():
    results, labels = _mixed_grade_corpus()
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(results, labels, min_grade="strong")
    )
    assert payload["min_grade"] == "strong"
    recs = [r["recording"] for r in payload["rows"]]
    assert recs == ["strong.wav"]


def test_json_no_min_grade_omits_key():
    results, labels = _mixed_grade_corpus()
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(results, labels)
    )
    assert "min_grade" not in payload
    assert len(payload["rows"]) == 3


def test_json_min_grade_preserves_corpus_aggregates():
    results, labels = _mixed_grade_corpus()
    base = json.loads(gv.render_vad_gap_recommend_batch_json(results, labels))
    filt = json.loads(
        gv.render_vad_gap_recommend_batch_json(results, labels, min_grade="strong")
    )
    for key in (
        "recommended_ms_median",
        "recommended_ms_min",
        "recommended_ms_max",
        "recommended_ms_spread",
        "num_recommended",
        "num_recordings",
    ):
        assert base[key] == filt[key]


def test_json_min_grade_composes_with_sort_and_top_n():
    results, labels = _mixed_grade_corpus()
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            results, labels, min_grade="moderate", sort_by="recommended", top_n=1
        )
    )
    # moderate floor keeps strong.wav + mod.wav; sort recommended ascending puts
    # mod.wav (550) before strong.wav (850); top_n 1 keeps just mod.wav.
    assert payload["min_grade"] == "moderate"
    assert [r["recording"] for r in payload["rows"]] == ["mod.wav"]


# ---- CSV renderer with min_grade ---------------------------------------


def test_csv_min_grade_filters_data_rows_same_header():
    results, labels = _mixed_grade_corpus()
    base = list(csv.reader(io.StringIO(
        gv.render_vad_gap_recommend_batch_csv(results, labels)
    )))
    filt = list(csv.reader(io.StringIO(
        gv.render_vad_gap_recommend_batch_csv(results, labels, min_grade="strong")
    )))
    assert base[0] == filt[0]  # header unchanged
    assert [row[0] for row in filt[1:]] == ["strong.wav"]


# ---- summary respects min_grade ----------------------------------------


def test_human_summary_respects_min_grade():
    # With a 'strong' floor, only strong.wav survives, so it must be the
    # representative even though mod.wav sits exactly at the median.
    results, labels = _mixed_grade_corpus()
    lines = gv.render_vad_gap_recommend_batch(
        results, labels, min_grade="strong", summary=True
    )
    assert any("representative: strong.wav" in ln for ln in lines)


def test_json_summary_respects_min_grade():
    results, labels = _mixed_grade_corpus()
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            results, labels, min_grade="strong", summary=True
        )
    )
    assert payload["min_grade"] == "strong"
    assert payload["best"]["recording"] == "strong.wav"


def test_summary_min_grade_no_survivor_note():
    # A 'strong' floor over a moderate-only corpus leaves nothing to summarize.
    mod = _result_from_gaps([0.2, 0.3, 0.4, 0.7, 0.9], name="mod.wav")
    lines = gv.render_vad_gap_recommend_batch(
        [mod], ["mod.wav"], min_grade="strong", summary=True
    )
    body = "\n".join(lines)
    assert "reaching confidence grade 'strong' or better" in body
    assert "no recording" in body


def test_csv_summary_respects_min_grade():
    results, labels = _mixed_grade_corpus()
    text = gv.render_vad_gap_recommend_batch_csv(
        results, labels, min_grade="strong", summary=True
    )
    rows = list(csv.reader(io.StringIO(text)))
    assert len(rows) == 2  # header + the one surviving best
    assert rows[1][0] == "strong.wav"


# ---- handler threads min_grade -----------------------------------------


def test_cmd_threads_min_grade_human():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(wavs=["strong.wav", "mod.wav", "none.wav"], min_grade="strong"),
        log=lines.append,
        segmenter=_mixed_grade_segmenter(),
        availability=lambda: True,
    )
    body = "\n".join(lines)
    assert "strong.wav" in body
    assert "mod.wav" not in body and "none.wav" not in body
    assert "min grade: strong" in body


def test_cmd_threads_min_grade_json():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(wavs=["strong.wav", "mod.wav", "none.wav"], json=True,
              min_grade="moderate"),
        log=lines.append,
        segmenter=_mixed_grade_segmenter(),
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["min_grade"] == "moderate"
    recs = sorted(r["recording"] for r in payload["rows"])
    assert recs == ["mod.wav", "strong.wav"]


def test_cmd_default_no_min_grade_keeps_all():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(wavs=["strong.wav", "mod.wav", "none.wav"], json=True),
        log=lines.append,
        segmenter=_mixed_grade_segmenter(),
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert "min_grade" not in payload
    assert len(payload["rows"]) == 3


def test_cmd_unavailable_still_threads_min_grade_json():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(json=True, min_grade="strong"),
        log=lines.append,
        segmenter=lambda wav, params=None: None,
        availability=lambda: False,
    )
    assert json.loads("\n".join(lines))["available"] is False


# ========================================================================
# iter-390 — corpus confidence-grade histogram (grade_counts). The batch already
# reports WHERE the corpus agrees (median/spread); grade_counts reports HOW
# TRUSTWORTHY it is — how many recordings sit at each iter-348 grade. The companion
# to the iter-389 --min-grade filter (shows how many recordings each floor keeps),
# computed over the WHOLE corpus regardless of any render-time filter.
# ========================================================================


# ---- _batch_grade_counts primitive -------------------------------------


def test_batch_grade_counts_keys_always_present_and_ordered():
    counts = gv._batch_grade_counts([])
    assert list(counts.keys()) == list(gv.GAP_RECOMMEND_BATCH_GRADE_ORDER)
    assert all(v == 0 for v in counts.values())


def test_batch_grade_counts_tallies_each_grade():
    rows = [
        {"grade": "strong"},
        {"grade": "strong"},
        {"grade": "moderate"},
        {"grade": "weak"},
        {"grade": "none"},
        {"grade": None},  # ungraded (<2 segments)
    ]
    counts = gv._batch_grade_counts(rows)
    assert counts == {
        "strong": 2,
        "moderate": 1,
        "weak": 1,
        "none": 1,
        "ungraded": 1,
    }


def test_batch_grade_counts_none_grade_maps_to_ungraded():
    # The core's None grade is the <2-segment bucket, kept DISTINCT from "none".
    counts = gv._batch_grade_counts([{"grade": None}, {"grade": "none"}])
    assert counts["ungraded"] == 1
    assert counts["none"] == 1


def test_batch_grade_counts_unknown_grade_falls_back_to_ungraded():
    # A future/unrecognised grade must not vanish — counts still sum to len(rows).
    counts = gv._batch_grade_counts([{"grade": "superb"}])
    assert counts["ungraded"] == 1
    assert sum(counts.values()) == 1


def test_batch_grade_counts_sum_equals_row_count():
    rows = [{"grade": g} for g in ["strong", "moderate", None, "none", "weak"]]
    counts = gv._batch_grade_counts(rows)
    assert sum(counts.values()) == len(rows)


def test_batch_grade_counts_does_not_mutate_rows():
    rows = [{"grade": "strong"}, {"grade": None}]
    snapshot = [dict(r) for r in rows]
    gv._batch_grade_counts(rows)
    assert rows == snapshot


# ---- _format_batch_grade_counts renderer -------------------------------


def test_format_grade_counts_names_only_nonzero_in_order():
    counts = {"strong": 2, "moderate": 0, "weak": 1, "none": 0, "ungraded": 3}
    line = gv._format_batch_grade_counts(counts)
    assert line == "  grades: 2 strong, 1 weak, 3 ungraded"


def test_format_grade_counts_empty_corpus_reads_none():
    counts = {g: 0 for g in gv.GAP_RECOMMEND_BATCH_GRADE_ORDER}
    assert gv._format_batch_grade_counts(counts) == "  grades: (none)"


# ---- core key ----------------------------------------------------------


def test_batch_core_carries_grade_counts():
    results = [_clean("a.wav"), _lower("b.wav"), _flat("flat.wav")]
    labels = ["a.wav", "b.wav", "flat.wav"]
    d = gv.vad_gap_recommend_batch(results, labels)
    assert d["grade_counts"]["strong"] == 2
    assert d["grade_counts"]["ungraded"] == 1
    assert sum(d["grade_counts"].values()) == 3


# ---- human renderer ----------------------------------------------------


def test_human_shows_grades_line():
    results = [_clean("a.wav"), _lower("b.wav"), _flat("flat.wav")]
    labels = ["a.wav", "b.wav", "flat.wav"]
    lines = gv.render_vad_gap_recommend_batch(results, labels)
    grade_line = [ln for ln in lines if "grades:" in ln][0]
    assert "2 strong" in grade_line
    assert "1 ungraded" in grade_line


def test_human_grades_line_unaffected_by_min_grade():
    # The histogram describes the WHOLE corpus; the floor narrows which ROWS show.
    results = [_clean("a.wav"), _lower("b.wav"), _flat("flat.wav")]
    labels = ["a.wav", "b.wav", "flat.wav"]
    full = gv.render_vad_gap_recommend_batch(results, labels)
    filtered = gv.render_vad_gap_recommend_batch(results, labels, min_grade="strong")
    full_grade = [ln for ln in full if "grades:" in ln][0]
    filt_grade = [ln for ln in filtered if "grades:" in ln][0]
    assert full_grade == filt_grade
    assert "1 ungraded" in filt_grade


def test_human_grades_line_unaffected_by_top_n():
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    labels = ["a.wav", "b.wav", "c.wav"]
    full = gv.render_vad_gap_recommend_batch(results, labels)
    capped = gv.render_vad_gap_recommend_batch(results, labels, top_n=1)
    assert [ln for ln in full if "grades:" in ln][0] == (
        [ln for ln in capped if "grades:" in ln][0]
    )


# ---- JSON renderer -----------------------------------------------------


def test_json_carries_grade_counts():
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            [_clean("a.wav"), _lower("b.wav"), _flat("flat.wav")],
            ["a.wav", "b.wav", "flat.wav"],
        )
    )
    assert payload["grade_counts"] == {
        "strong": 2,
        "moderate": 0,
        "weak": 0,
        "none": 0,
        "ungraded": 1,
    }


def test_json_grade_counts_unaffected_by_min_grade():
    full = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            [_clean("a.wav"), _lower("b.wav"), _flat("flat.wav")],
            ["a.wav", "b.wav", "flat.wav"],
        )
    )
    filtered = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            [_clean("a.wav"), _lower("b.wav"), _flat("flat.wav")],
            ["a.wav", "b.wav", "flat.wav"],
            min_grade="strong",
        )
    )
    assert full["grade_counts"] == filtered["grade_counts"]


def test_json_summary_still_carries_grade_counts():
    # Summary pops rows but the whole-corpus histogram stays — a consumer sees
    # what the representative is central within.
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            [_clean("a.wav"), _lower("b.wav"), _flat("flat.wav")],
            ["a.wav", "b.wav", "flat.wav"],
            summary=True,
        )
    )
    assert "rows" not in payload
    assert payload["grade_counts"]["strong"] == 2
    assert payload["grade_counts"]["ungraded"] == 1


def test_json_all_missing_grade_counts_all_ungraded():
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            [_flat("a.wav"), _flat("b.wav")], ["a.wav", "b.wav"]
        )
    )
    assert payload["grade_counts"]["ungraded"] == 2
    assert sum(payload["grade_counts"].values()) == 2


# ---- handler -----------------------------------------------------------


def test_cmd_human_emits_grades_line():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(),
        log=lines.append,
        segmenter=_corpus_segmenter(),
        availability=lambda: True,
    )
    assert any("grades:" in ln for ln in lines)


def test_cmd_json_carries_grade_counts():
    lines = []
    gv.cmd_vad_gap_recommend_batch(
        _args(json=True),
        log=lines.append,
        segmenter=_corpus_segmenter(),
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["grade_counts"]["strong"] == 3


# ============================================================================
# iter-391 — corpus outlier-robust IQR (recommended_ms_q1 / _q3 / _iqr).
# The batch already reports the range-based ``spread`` (max - min), which a single
# flyer inflates. The IQR (Q3 - Q1) measures the width of the MIDDLE HALF of the
# corpus, so a lone outlier cannot widen it — the robust companion to the
# outlier-robust median. Reuses the iter-338 ``_percentile_of_sorted`` primitive.
# ============================================================================


def _iqr_of(values):
    """Reference IQR (Q3 - Q1) over a list, via the same R-7 primitive the code uses."""
    srt = sorted(values)
    q1 = gv._percentile_of_sorted(srt, 25)
    q3 = gv._percentile_of_sorted(srt, 75)
    return round(q1, 1), round(q3, 1), round(q3 - q1, 1)


# ---- core keys ---------------------------------------------------------


def test_batch_core_carries_iqr_keys():
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    labels = ["a.wav", "b.wav", "c.wav"]
    d = gv.vad_gap_recommend_batch(results, labels)
    recs = [gv.vad_gap_recommend(r)["recommended_ms"] for r in results]
    q1, q3, iqr = _iqr_of(recs)
    assert d["recommended_ms_q1"] == q1
    assert d["recommended_ms_q3"] == q3
    assert d["recommended_ms_iqr"] == iqr


def test_batch_iqr_matches_percentile_primitive():
    # Five recordings with known, spread recommended ms — IQR must equal the
    # R-7 25th/75th percentile difference exactly.
    results = [
        _result_from_gaps([0.2, 0.3, 0.25, lo, lo + 0.1, lo - 0.05], name=f"r{i}.wav")
        for i, lo in enumerate([0.6, 0.9, 1.2, 1.5, 2.4])
    ]
    labels = [r.name for r in results]
    d = gv.vad_gap_recommend_batch(results, labels)
    recs = [gv.vad_gap_recommend(r)["recommended_ms"] for r in results]
    q1, q3, iqr = _iqr_of(recs)
    assert d["recommended_ms_q1"] == q1
    assert d["recommended_ms_q3"] == q3
    assert d["recommended_ms_iqr"] == iqr
    assert iqr >= 0.0


def test_batch_iqr_robust_to_single_flyer():
    # A tight cluster of four plus one extreme outlier. The range-based spread
    # is dominated by the flyer; the IQR (middle half) is much smaller.
    base = [_lower(f"b{i}.wav") for i in range(4)]  # four near-identical valleys
    flyer = _higher("flyer.wav")                    # one much-higher valley
    results = base + [flyer]
    labels = [r.name for r in results]
    d = gv.vad_gap_recommend_batch(results, labels)
    # The four clones agree, so the middle half is narrow; the flyer blows the
    # range out, so the IQR is strictly smaller than the spread.
    assert d["recommended_ms_iqr"] < d["recommended_ms_spread"]


def test_batch_iqr_single_recording_is_zero():
    # One recommending recording: every percentile is that value, so IQR is 0.
    d = gv.vad_gap_recommend_batch([_clean("a.wav")], ["a.wav"])
    rec = gv.vad_gap_recommend(_clean("a.wav"))["recommended_ms"]
    assert d["recommended_ms_q1"] == round(rec, 1)
    assert d["recommended_ms_q3"] == round(rec, 1)
    assert d["recommended_ms_iqr"] == 0.0


def test_batch_iqr_none_when_no_recording_recommends():
    d = gv.vad_gap_recommend_batch([_flat("a.wav"), _flat("b.wav")],
                                   ["a.wav", "b.wav"])
    assert d["recommended_ms_q1"] is None
    assert d["recommended_ms_q3"] is None
    assert d["recommended_ms_iqr"] is None


def test_batch_iqr_ignores_non_recommending_recordings():
    # A flat recording carries no recommendation, so it must not feed the IQR —
    # the quartiles match those over the recommending recordings alone.
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav"), _flat("f.wav")]
    labels = [r.name for r in results]
    d = gv.vad_gap_recommend_batch(results, labels)
    recs = [
        gv.vad_gap_recommend(r)["recommended_ms"]
        for r in (results[0], results[1], results[2])
    ]
    q1, q3, iqr = _iqr_of(recs)
    assert d["recommended_ms_iqr"] == iqr


# ---- human renderer ----------------------------------------------------


def test_human_corpus_line_shows_iqr():
    lines = gv.render_vad_gap_recommend_batch(
        [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")],
        ["a.wav", "b.wav", "c.wav"],
    )
    corpus = [ln for ln in lines if "corpus:" in ln][0]
    assert "IQR" in corpus
    assert "spread" in corpus


def test_human_iqr_unaffected_by_min_grade_and_top_n():
    results = [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")]
    labels = ["a.wav", "b.wav", "c.wav"]
    full = [ln for ln in gv.render_vad_gap_recommend_batch(results, labels)
            if "corpus:" in ln][0]
    capped = [ln for ln in gv.render_vad_gap_recommend_batch(
        results, labels, top_n=1) if "corpus:" in ln][0]
    floored = [ln for ln in gv.render_vad_gap_recommend_batch(
        results, labels, min_grade="strong") if "corpus:" in ln][0]
    assert full == capped == floored


# ---- JSON renderer -----------------------------------------------------


def test_json_carries_iqr_keys():
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")],
            ["a.wav", "b.wav", "c.wav"],
        )
    )
    recs = [
        gv.vad_gap_recommend(r)["recommended_ms"]
        for r in (_clean("a.wav"), _lower("b.wav"), _higher("c.wav"))
    ]
    q1, q3, iqr = _iqr_of(recs)
    assert payload["recommended_ms_q1"] == q1
    assert payload["recommended_ms_q3"] == q3
    assert payload["recommended_ms_iqr"] == iqr


def test_json_iqr_null_when_no_recording_recommends():
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            [_flat("a.wav"), _flat("b.wav")], ["a.wav", "b.wav"]
        )
    )
    assert payload["recommended_ms_q1"] is None
    assert payload["recommended_ms_q3"] is None
    assert payload["recommended_ms_iqr"] is None


def test_json_summary_still_carries_iqr():
    # Summary pops rows but keeps the whole-corpus aggregates, IQR included.
    payload = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")],
            ["a.wav", "b.wav", "c.wav"],
            summary=True,
        )
    )
    assert "rows" not in payload
    assert payload["recommended_ms_iqr"] is not None


def test_json_iqr_unaffected_by_min_grade():
    full = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")],
            ["a.wav", "b.wav", "c.wav"],
        )
    )
    floored = json.loads(
        gv.render_vad_gap_recommend_batch_json(
            [_clean("a.wav"), _lower("b.wav"), _higher("c.wav")],
            ["a.wav", "b.wav", "c.wav"],
            min_grade="strong",
        )
    )
    assert full["recommended_ms_iqr"] == floored["recommended_ms_iqr"]
    assert full["recommended_ms_q1"] == floored["recommended_ms_q1"]
    assert full["recommended_ms_q3"] == floored["recommended_ms_q3"]
