"""Tests for iter-397 — the ``gv calibrate-base-wpm-batch`` subcommand.

iter-397 ships the batch calibration core (``calibrate_base_wpm_batch`` /
``BaseWpmCalibrationBatch``) — calibrate a CORPUS of voices and tabulate each
voice's implied base_wpm + dispersion grade + drift plus an outlier-robust
corpus median — and exposes it on the ``gv`` CLI, the calibration analogue of
``gv vad-gap-recommend-batch``. Each ``--voice`` group is one voice: a LABEL
followed by its ``words:audio_seconds[:speed]`` sample triples.

These tests exercise the parser wiring, the pure ``render_calibration_batch``
render, and the handler (driven with an injected ``log`` so no real I/O
happens). The engine is pure stdlib loaded by file path, so the handler runs on
this x86_64 Linux runner without pipecat.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples import gv  # noqa: E402


# ---- parser: wiring ----------------------------------------------------


def test_batch_in_handler_map():
    assert (
        gv.DEFAULT_HANDLERS["calibrate-base-wpm-batch"]
        is gv.cmd_calibrate_base_wpm_batch
    )


def test_batch_parses_voice_groups():
    args = gv.build_parser().parse_args(
        [
            "calibrate-base-wpm-batch",
            "--voice", "af_heart", "50:18.2", "50:9.1:2.0",
            "--voice", "am_adam", "40:15.0",
        ]
    )
    assert args.command == "calibrate-base-wpm-batch"
    assert args.voice == [
        ["af_heart", "50:18.2", "50:9.1:2.0"],
        ["am_adam", "40:15.0"],
    ]
    assert args.nominal == 165.0


def test_batch_requires_voice():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(["calibrate-base-wpm-batch"])
    assert exc.value.code == 2


def test_batch_nominal_override():
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm-batch", "--voice", "a", "50:18.2", "--nominal", "150"]
    )
    assert args.nominal == 150.0


# ---- render_calibration_batch: pure render -----------------------------


def _batch(voices, nominal=165.0):
    wm = gv._load_wpm_mirror()
    built = []
    for label, triples in voices:
        samples = [
            wm.CalibrationSample(words=int(w), audio_seconds=a, speed=s)
            for (w, a, s) in triples
        ]
        built.append((label, samples))
    return wm.calibrate_base_wpm_batch(built, default_base_wpm=nominal)


def test_render_header_counts_voices():
    batch = _batch([("a", [(165, 60.0, 1.0)]), ("b", [(150, 60.0, 1.0)])])
    lines = gv.render_calibration_batch(batch)
    assert "2 voices, 2 calibrated" in lines[0]


def test_render_lists_each_voice_with_grade_and_drift():
    batch = _batch([("af_heart", [(180, 60.0, 1.0)])], nominal=160.0)
    text = "\n".join(gv.render_calibration_batch(batch))
    assert "af_heart" in text
    assert "180.0 WPM" in text
    assert "agree" in text  # single-sample ⇒ zero spread ⇒ agree
    assert "drift +20.0" in text


def test_render_corpus_summary_line():
    batch = _batch(
        [
            ("slow", [(150, 60.0, 1.0)]),
            ("mid", [(165, 60.0, 1.0)]),
            ("fast", [(180, 60.0, 1.0)]),
        ]
    )
    text = "\n".join(gv.render_calibration_batch(batch))
    assert "corpus: median 165.0" in text
    assert "range 150.0 – 180.0" in text
    assert "spread 30.0" in text


def test_render_grades_histogram():
    batch = _batch([("tight", [(165, 60.0, 1.0)]), ("empty", [])])
    text = "\n".join(gv.render_calibration_batch(batch))
    assert "grades:" in text
    assert "1 agree" in text
    assert "1 uncalibrated" in text


def test_render_uncalibrated_voice_marked():
    batch = _batch([("real", [(165, 60.0, 1.0)]), ("empty", [])])
    text = "\n".join(gv.render_calibration_batch(batch))
    assert "empty: - (uncalibrated" in text


def test_render_all_uncalibrated_corpus_note():
    batch = _batch([("a", []), ("b", [])])
    text = "\n".join(gv.render_calibration_batch(batch))
    assert "no voice calibrated" in text


def test_render_delta_from_median_shown():
    batch = _batch(
        [
            ("slow", [(150, 60.0, 1.0)]),
            ("mid", [(165, 60.0, 1.0)]),
            ("fast", [(180, 60.0, 1.0)]),
        ]
    )
    by_text = {}
    for line in gv.render_calibration_batch(batch):
        for v in ("slow", "mid", "fast"):
            if line.strip().startswith(v + ":"):
                by_text[v] = line
    assert "Δmedian -15.0" in by_text["slow"]
    assert "Δmedian +0.0" in by_text["mid"]
    assert "Δmedian +15.0" in by_text["fast"]


# ---- handler: injected log ---------------------------------------------


def _run(argv):
    args = gv.build_parser().parse_args(argv)
    lines: list = []
    gv.cmd_calibrate_base_wpm_batch(args, log=lines.append)
    return lines


def test_handler_emits_report():
    lines = _run(
        [
            "calibrate-base-wpm-batch",
            "--voice", "a", "165:60.0",
            "--voice", "b", "150:60.0",
        ]
    )
    text = "\n".join(lines)
    assert "calibration batch" in text
    assert "corpus: median" in text


def test_handler_matches_render_directly():
    # The handler's output equals render_calibration_batch over the same fold.
    batch = _batch([("a", [(165, 60.0, 1.0)]), ("b", [(150, 60.0, 1.0)])])
    expected = gv.render_calibration_batch(batch)
    lines = _run(
        [
            "calibrate-base-wpm-batch",
            "--voice", "a", "165:60.0",
            "--voice", "b", "150:60.0",
        ]
    )
    assert lines == expected


def test_handler_malformed_triple_is_clean_error():
    # A malformed triple is rejected by the SAME calibration_sample_type
    # validator the single-voice --samples uses — a clean, descriptive
    # ArgumentTypeError rather than a forwarded-garbage traceback. (It surfaces
    # in the handler because the label-plus-triples shape is parsed there, not by
    # an argparse ``type=``.)
    import argparse

    with pytest.raises(argparse.ArgumentTypeError) as exc:
        _run(["calibrate-base-wpm-batch", "--voice", "a", "notatriple"])
    assert "words:audio_seconds" in str(exc.value)


def test_handler_voice_with_only_label_is_uncalibrated():
    lines = _run(["calibrate-base-wpm-batch", "--voice", "empty"])
    text = "\n".join(lines)
    assert "empty: - (uncalibrated" in text


def test_handler_default_log_is_print(capsys):
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm-batch", "--voice", "a", "165:60.0"]
    )
    gv.cmd_calibrate_base_wpm_batch(args)
    out = capsys.readouterr().out
    assert "calibration batch" in out


# ---- --json / --csv: parser wiring -------------------------------------


def test_batch_json_csv_mutually_exclusive():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(
            [
                "calibrate-base-wpm-batch",
                "--voice", "a", "165:60.0",
                "--json", "--csv",
            ]
        )
    assert exc.value.code == 2


def test_batch_json_flag_parses():
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm-batch", "--voice", "a", "165:60.0", "--json"]
    )
    assert args.json is True
    assert args.csv is False


def test_batch_csv_flag_parses():
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm-batch", "--voice", "a", "165:60.0", "--csv"]
    )
    assert args.csv is True
    assert args.json is False


# ---- render_calibration_batch_json: pure render ------------------------


def _json(voices, nominal=165.0):
    import json

    return json.loads(gv.render_calibration_batch_json(_batch(voices, nominal)))


def test_json_carries_corpus_aggregates():
    payload = _json(
        [
            ("slow", [(150, 60.0, 1.0)]),
            ("mid", [(165, 60.0, 1.0)]),
            ("fast", [(180, 60.0, 1.0)]),
        ]
    )
    assert payload["nominal"] == 165.0
    assert payload["num_voices"] == 3
    assert payload["num_calibrated"] == 3
    assert payload["implied_base_wpm_median"] == 165.0
    assert payload["implied_base_wpm_min"] == 150.0
    assert payload["implied_base_wpm_max"] == 180.0
    assert payload["implied_base_wpm_spread"] == 30.0


def test_json_grade_counts_has_all_four_buckets_summing_to_num_voices():
    payload = _json([("tight", [(165, 60.0, 1.0)]), ("empty", [])])
    counts = payload["grade_counts"]
    assert set(counts) == set(gv._wm_calib_batch_grade_order())
    assert sum(counts.values()) == payload["num_voices"] == 2
    assert counts["agree"] == 1
    assert counts["uncalibrated"] == 1


def test_json_row_matches_single_voice_json():
    # A batch row's calibration object equals the single-voice --json calibration
    # on the same samples — the batch is a true generalisation, not a re-derivation.
    import json

    wm = gv._load_wpm_mirror()
    samples = [
        wm.CalibrationSample(words=180, audio_seconds=60.0, speed=1.0),
        wm.CalibrationSample(words=165, audio_seconds=60.0, speed=1.0),
    ]
    single = json.loads(gv.render_calibration_json(samples, wm.calibrate_base_wpm(samples)))
    payload = _json([("af_heart", [(180, 60.0, 1.0), (165, 60.0, 1.0)])])
    row = payload["rows"][0]
    assert row["voice"] == "af_heart"
    assert row["calibration"] == single["calibration"]


def test_json_uncalibrated_voice_is_null():
    payload = _json([("real", [(165, 60.0, 1.0)]), ("empty", [])])
    rows = {r["voice"]: r for r in payload["rows"]}
    assert rows["empty"]["calibration"] is None
    assert rows["empty"]["delta_from_median_wpm"] is None


def test_json_delta_from_median_signed():
    payload = _json(
        [
            ("slow", [(150, 60.0, 1.0)]),
            ("mid", [(165, 60.0, 1.0)]),
            ("fast", [(180, 60.0, 1.0)]),
        ]
    )
    rows = {r["voice"]: r for r in payload["rows"]}
    assert rows["slow"]["delta_from_median_wpm"] == -15.0
    assert rows["mid"]["delta_from_median_wpm"] == 0.0
    assert rows["fast"]["delta_from_median_wpm"] == 15.0


def test_json_empty_corpus_aggregates_null():
    payload = _json([("a", []), ("b", [])])
    assert payload["num_calibrated"] == 0
    assert payload["implied_base_wpm_median"] is None
    assert payload["implied_base_wpm_min"] is None
    assert payload["implied_base_wpm_max"] is None
    assert payload["implied_base_wpm_spread"] is None
    assert payload["grade_counts"]["uncalibrated"] == 2


def test_json_rows_in_input_order():
    payload = _json([("z", [(165, 60.0, 1.0)]), ("a", [(150, 60.0, 1.0)])])
    assert [r["voice"] for r in payload["rows"]] == ["z", "a"]


def test_json_grade_is_voice_comparable_across_rates():
    # Same relative spread at a 100-WPM and a 300-WPM voice ⇒ same grade.
    slow = _json([("slow", [(95, 60.0, 1.0), (105, 60.0, 1.0)])])["rows"][0]
    fast = _json([("fast", [(285, 60.0, 1.0), (315, 60.0, 1.0)])])["rows"][0]
    assert (
        slow["calibration"]["dispersion_grade"]
        == fast["calibration"]["dispersion_grade"]
    )


# ---- render_calibration_batch_csv: pure render -------------------------


def _csv_rows(voices, nominal=165.0):
    text = gv.render_calibration_batch_csv(_batch(voices, nominal))
    return text.splitlines()


def test_csv_header():
    rows = _csv_rows([("a", [(165, 60.0, 1.0)])])
    assert rows[0] == (
        "voice,implied_base_wpm,n_samples,spread,relative_spread,"
        "dispersion_grade,dispersion_margin,drift,delta_from_median_wpm"
    )


def test_csv_one_data_row_per_voice():
    rows = _csv_rows([("a", [(165, 60.0, 1.0)]), ("b", [(150, 60.0, 1.0)])])
    data = [r for r in rows if not r.startswith("#")][1:]  # skip header
    assert len(data) == 2
    assert data[0].startswith("a,")
    assert data[1].startswith("b,")


def test_csv_uncalibrated_voice_blank_cells():
    rows = _csv_rows([("real", [(165, 60.0, 1.0)]), ("empty", [])])
    empty = next(r for r in rows if r.startswith("empty,"))
    # voice label then eight empty cells.
    assert empty == "empty,,,,,,,,"


def test_csv_summary_comments_carry_aggregates():
    rows = _csv_rows(
        [
            ("slow", [(150, 60.0, 1.0)]),
            ("mid", [(165, 60.0, 1.0)]),
            ("fast", [(180, 60.0, 1.0)]),
        ]
    )
    text = "\n".join(rows)
    assert "# nominal: 165.0" in text
    assert "# num_voices: 3" in text
    assert "# num_calibrated: 3" in text
    assert "# implied_base_wpm_median: 165.0" in text
    assert "# range: 150.0 - 180.0" in text
    assert "# implied_base_wpm_spread: 30.0" in text
    assert "# grades: 3 agree, 0 loose, 0 scattered, 0 uncalibrated" in text


def test_csv_empty_corpus_blank_aggregates():
    rows = _csv_rows([("a", []), ("b", [])])
    text = "\n".join(rows)
    assert "# num_calibrated: 0" in text
    assert "# implied_base_wpm_median: " in text  # blank value
    assert "# range:  - " in text


def test_csv_parses_with_pandas_comment_skip():
    # The # comment lines are skippable so the data grid stays pure.
    import csv as _csv
    import io as _io

    rows = _csv_rows([("a", [(165, 60.0, 1.0)]), ("b", [(150, 60.0, 1.0)])])
    data_lines = [r for r in rows if not r.startswith("#")]
    reader = list(_csv.DictReader(_io.StringIO("\n".join(data_lines))))
    assert len(reader) == 2
    assert reader[0]["voice"] == "a"
    assert reader[0]["implied_base_wpm"] == "165.0"


# ---- handler: --json / --csv -------------------------------------------


def test_handler_json_emits_single_string():
    import json

    lines = _run(
        ["calibrate-base-wpm-batch", "--voice", "a", "165:60.0", "--json"]
    )
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["num_voices"] == 1


def test_handler_json_matches_render():
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm-batch", "--voice", "a", "165:60.0", "--json"]
    )
    lines: list = []
    gv.cmd_calibrate_base_wpm_batch(args, log=lines.append)
    expected = gv.render_calibration_batch_json(
        _batch([("a", [(165, 60.0, 1.0)])])
    )
    assert lines == [expected]


def test_handler_csv_matches_render():
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm-batch", "--voice", "a", "165:60.0", "--csv"]
    )
    lines: list = []
    gv.cmd_calibrate_base_wpm_batch(args, log=lines.append)
    expected = gv.render_calibration_batch_csv(
        _batch([("a", [(165, 60.0, 1.0)])])
    )
    assert lines == [expected]


def test_handler_json_suppresses_human_report():
    lines = _run(
        ["calibrate-base-wpm-batch", "--voice", "a", "165:60.0", "--json"]
    )
    text = "\n".join(lines)
    assert "calibration batch" not in text  # the human header is gone


# ---- iter-399: --sort-by render-only ordering --------------------------


def test_sort_type_accepts_each_key():
    for key in ("base_wpm", "grade", "drift", "delta"):
        assert gv.calib_batch_sort_type(key) == key


def test_sort_type_is_case_insensitive_and_strips():
    assert gv.calib_batch_sort_type("  BASE_WPM ") == "base_wpm"


def test_sort_type_rejects_unknown():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        gv.calib_batch_sort_type("nonsense")
    with pytest.raises(argparse.ArgumentTypeError):
        gv.calib_batch_sort_type("")


def test_parser_sort_by_parses():
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm-batch", "--voice", "a", "165:60.0", "--sort-by", "grade"]
    )
    assert args.sort_by == "grade"


def test_parser_sort_by_default_none():
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm-batch", "--voice", "a", "165:60.0"]
    )
    assert args.sort_by is None


def test_parser_sort_by_rejects_unknown():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(
            ["calibrate-base-wpm-batch", "--voice", "a", "165:60.0",
             "--sort-by", "bogus"]
        )
    assert exc.value.code == 2


# The corpus the ordering tests share: three distinct base rates so every key
# produces a determinate permutation, plus an uncalibrated voice that must sort
# last under every ordering.
_SORT_VOICES = [
    ("mid", [(165, 60.0, 1.0)]),
    ("fast", [(180, 60.0, 1.0)]),
    ("slow", [(150, 60.0, 1.0)]),
    ("empty", []),
]


def _row_order(lines):
    """The voice labels in the order their rows appear in a human render."""
    order = []
    for line in lines:
        stripped = line.strip()
        for v in ("mid", "fast", "slow", "empty"):
            if stripped.startswith(v + ":"):
                order.append(v)
    return order


def test_sort_none_keeps_argument_order():
    batch = _batch(_SORT_VOICES)
    assert _row_order(gv.render_calibration_batch(batch)) == [
        "mid", "fast", "slow", "empty"
    ]


def test_sort_base_wpm_ascending_uncalibrated_last():
    batch = _batch(_SORT_VOICES)
    assert _row_order(gv.render_calibration_batch(batch, sort_by="base_wpm")) == [
        "slow", "mid", "fast", "empty"
    ]


def test_sort_delta_biggest_outlier_first():
    # median is 165 (mid); |Δ| = fast 15, slow 15, mid 0. fast/slow tie on |Δ|,
    # broken by input order (fast listed before slow). empty last.
    batch = _batch(_SORT_VOICES)
    assert _row_order(gv.render_calibration_batch(batch, sort_by="delta")) == [
        "fast", "slow", "mid", "empty"
    ]


def test_sort_drift_biggest_mover_first():
    # nominal 165: |drift| = slow 15, fast 15, mid 0. slow/fast tie, input order
    # keeps fast after... actually input order is mid,fast,slow so fast first.
    batch = _batch(_SORT_VOICES)
    assert _row_order(gv.render_calibration_batch(batch, sort_by="drift")) == [
        "fast", "slow", "mid", "empty"
    ]


def test_sort_grade_most_trustworthy_first():
    # A wide-spread voice grades worse than tight single-sample voices. Build a
    # corpus with one scattered voice and two agree voices.
    batch = _batch(
        [
            ("scattered", [(120, 60.0, 1.0), (240, 60.0, 1.0)]),
            ("tightA", [(165, 60.0, 1.0)]),
            ("tightB", [(150, 60.0, 1.0)]),
        ]
    )
    order = []
    for line in gv.render_calibration_batch(batch, sort_by="grade"):
        s = line.strip()
        for v in ("scattered", "tightA", "tightB"):
            if s.startswith(v + ":"):
                order.append(v)
    # agree voices (tightA, tightB) before the scattered one; ties keep input order.
    assert order == ["tightA", "tightB", "scattered"]


def test_sort_names_active_ordering_in_header():
    batch = _batch(_SORT_VOICES)
    text = "\n".join(gv.render_calibration_batch(batch, sort_by="delta"))
    assert "sorted by delta" in text
    # default render does NOT mention sorting
    assert "sorted by" not in "\n".join(gv.render_calibration_batch(batch))


def test_sort_does_not_change_corpus_summary():
    batch = _batch(_SORT_VOICES)
    plain = "\n".join(gv.render_calibration_batch(batch))
    sorted_text = "\n".join(gv.render_calibration_batch(batch, sort_by="base_wpm"))
    for marker in ("corpus: median 165.0", "range 150.0 – 180.0", "spread 30.0"):
        assert marker in plain
        assert marker in sorted_text


def test_json_sort_reorders_rows_and_names_key():
    import json as _json

    batch = _batch(_SORT_VOICES)
    obj = _json.loads(gv.render_calibration_batch_json(batch, sort_by="base_wpm"))
    assert obj["sort_by"] == "base_wpm"
    assert [r["voice"] for r in obj["rows"]] == ["slow", "mid", "fast", "empty"]
    # aggregates unaffected
    assert obj["implied_base_wpm_median"] == 165.0


def test_json_sort_none_omits_key():
    import json as _json

    batch = _batch(_SORT_VOICES)
    obj = _json.loads(gv.render_calibration_batch_json(batch))
    assert "sort_by" not in obj
    assert [r["voice"] for r in obj["rows"]] == ["mid", "fast", "slow", "empty"]


def test_csv_sort_reorders_rows_and_comments_key():
    batch = _batch(_SORT_VOICES)
    text = gv.render_calibration_batch_csv(batch, sort_by="base_wpm")
    assert "# sort_by: base_wpm" in text
    data_rows = [
        ln for ln in text.splitlines()
        if ln and not ln.startswith("#") and not ln.startswith("voice,")
    ]
    voices = [ln.split(",")[0] for ln in data_rows]
    assert voices == ["slow", "mid", "fast", "empty"]


def test_csv_sort_none_omits_comment():
    batch = _batch(_SORT_VOICES)
    text = gv.render_calibration_batch_csv(batch)
    assert "# sort_by" not in text


def test_handler_sort_threads_to_human_render():
    lines = _run(
        [
            "calibrate-base-wpm-batch",
            "--voice", "mid", "165:60.0",
            "--voice", "fast", "180:60.0",
            "--voice", "slow", "150:60.0",
            "--sort-by", "base_wpm",
        ]
    )
    assert _row_order(lines) == ["slow", "mid", "fast"]


def test_handler_sort_threads_to_json():
    import json as _json

    lines = _run(
        [
            "calibrate-base-wpm-batch",
            "--voice", "mid", "165:60.0",
            "--voice", "slow", "150:60.0",
            "--json", "--sort-by", "base_wpm",
        ]
    )
    obj = _json.loads(lines[0])
    assert obj["sort_by"] == "base_wpm"
    assert [r["voice"] for r in obj["rows"]] == ["slow", "mid"]


def test_handler_sort_matches_render_directly():
    batch = _batch([("mid", [(165, 60.0, 1.0)]), ("slow", [(150, 60.0, 1.0)])])
    expected = gv.render_calibration_batch(batch, sort_by="base_wpm")
    lines = _run(
        [
            "calibrate-base-wpm-batch",
            "--voice", "mid", "165:60.0",
            "--voice", "slow", "150:60.0",
            "--sort-by", "base_wpm",
        ]
    )
    assert lines == expected
