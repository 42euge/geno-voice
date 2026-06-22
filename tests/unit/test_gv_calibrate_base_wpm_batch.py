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


def test_render_corpus_summary_carries_iqr():
    # iter-403: the human corpus line names the outlier-robust IQR alongside spread.
    batch = _batch(
        [
            ("a", [(150, 60.0, 1.0)]),
            ("b", [(160, 60.0, 1.0)]),
            ("c", [(165, 60.0, 1.0)]),
            ("d", [(170, 60.0, 1.0)]),
            ("e", [(180, 60.0, 1.0)]),
        ]
    )
    text = "\n".join(gv.render_calibration_batch(batch))
    assert "spread 30.0" in text
    assert "IQR 10.0" in text


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


def test_json_carries_iqr_keys():
    # iter-403: q1 / q3 / iqr ride alongside the other corpus aggregates.
    payload = _json(
        [
            ("a", [(150, 60.0, 1.0)]),
            ("b", [(160, 60.0, 1.0)]),
            ("c", [(165, 60.0, 1.0)]),
            ("d", [(170, 60.0, 1.0)]),
            ("e", [(180, 60.0, 1.0)]),
        ]
    )
    assert payload["implied_base_wpm_q1"] == 160.0
    assert payload["implied_base_wpm_q3"] == 170.0
    assert payload["implied_base_wpm_iqr"] == 10.0


def test_json_empty_corpus_iqr_keys_null():
    payload = _json([("a", []), ("b", [])])
    assert payload["implied_base_wpm_q1"] is None
    assert payload["implied_base_wpm_q3"] is None
    assert payload["implied_base_wpm_iqr"] is None


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


def test_csv_summary_comments_carry_iqr():
    # iter-403: the IQR rides in the trailing # comment block beside spread.
    rows = _csv_rows(
        [
            ("a", [(150, 60.0, 1.0)]),
            ("b", [(160, 60.0, 1.0)]),
            ("c", [(165, 60.0, 1.0)]),
            ("d", [(170, 60.0, 1.0)]),
            ("e", [(180, 60.0, 1.0)]),
        ]
    )
    text = "\n".join(rows)
    assert "# implied_base_wpm_spread: 30.0" in text
    assert "# implied_base_wpm_iqr: 10.0" in text


def test_csv_empty_corpus_blank_aggregates():
    rows = _csv_rows([("a", []), ("b", [])])
    text = "\n".join(rows)
    assert "# num_calibrated: 0" in text
    assert "# implied_base_wpm_median: " in text  # blank value
    assert "# range:  - " in text
    assert "# implied_base_wpm_iqr: " in text  # blank value


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
    return _row_order_pair(
        lines, ("mid", "fast", "slow", "empty", "tight", "loosey", "wide")
    )


def _row_order_pair(lines, labels):
    """The labels (from ``labels``) in the order their rows appear in a render."""
    order = []
    for line in lines:
        stripped = line.strip()
        for v in labels:
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


# ---- iter-400: --top-n render-only count cap ---------------------------


def test_parser_top_n_parses():
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm-batch", "--voice", "a", "165:60.0", "--top-n", "3"]
    )
    assert args.top_n == 3


def test_parser_top_n_default_none():
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm-batch", "--voice", "a", "165:60.0"]
    )
    assert args.top_n is None


def test_parser_top_n_rejects_zero_and_negative():
    for bad in ("0", "-1"):
        with pytest.raises(SystemExit) as exc:
            gv.build_parser().parse_args(
                ["calibrate-base-wpm-batch", "--voice", "a", "165:60.0",
                 "--top-n", bad]
            )
        assert exc.value.code == 2


def test_top_n_none_keeps_every_row():
    batch = _batch(_SORT_VOICES)
    assert _row_order(gv.render_calibration_batch(batch, top_n=None)) == [
        "mid", "fast", "slow", "empty"
    ]


def test_top_n_caps_rows_after_sort():
    # --sort-by delta floats the biggest outliers up; --top-n 2 keeps only those.
    batch = _batch(_SORT_VOICES)
    assert _row_order(
        gv.render_calibration_batch(batch, sort_by="delta", top_n=2)
    ) == ["fast", "slow"]


def test_top_n_without_sort_keeps_argument_order_prefix():
    batch = _batch(_SORT_VOICES)
    assert _row_order(gv.render_calibration_batch(batch, top_n=2)) == [
        "mid", "fast"
    ]


def test_top_n_larger_than_corpus_keeps_all():
    batch = _batch(_SORT_VOICES)
    assert _row_order(gv.render_calibration_batch(batch, top_n=99)) == [
        "mid", "fast", "slow", "empty"
    ]


def test_top_n_header_names_cap_when_rows_dropped():
    batch = _batch(_SORT_VOICES)
    text = "\n".join(gv.render_calibration_batch(batch, top_n=2))
    assert "(top 2 of 4)" in text


def test_top_n_header_silent_when_nothing_dropped():
    batch = _batch(_SORT_VOICES)
    text = "\n".join(gv.render_calibration_batch(batch, top_n=4))
    assert "top 4 of" not in text
    assert "top " not in text  # no cap tag at all when nothing elided


def test_top_n_does_not_change_corpus_summary():
    # The corpus aggregates describe the WHOLE corpus even when rows are capped.
    batch = _batch(_SORT_VOICES)
    text = "\n".join(gv.render_calibration_batch(batch, sort_by="delta", top_n=1))
    assert "corpus: median 165.0" in text
    assert "range 150.0 – 180.0" in text
    assert "spread 30.0" in text
    # the grade histogram still counts all four voices
    assert "1 uncalibrated" in text


def test_json_top_n_caps_rows_and_names_key():
    import json as _json

    batch = _batch(_SORT_VOICES)
    obj = _json.loads(
        gv.render_calibration_batch_json(batch, sort_by="delta", top_n=2)
    )
    assert obj["top_n"] == 2
    assert [r["voice"] for r in obj["rows"]] == ["fast", "slow"]
    # aggregates still describe the whole corpus
    assert obj["num_voices"] == 4
    assert obj["implied_base_wpm_median"] == 165.0


def test_json_top_n_none_omits_key():
    import json as _json

    batch = _batch(_SORT_VOICES)
    obj = _json.loads(gv.render_calibration_batch_json(batch))
    assert "top_n" not in obj


def test_csv_top_n_caps_rows_and_comments_key():
    batch = _batch(_SORT_VOICES)
    text = gv.render_calibration_batch_csv(batch, sort_by="delta", top_n=2)
    assert "# top_n: 2" in text
    data_rows = [
        ln for ln in text.splitlines()
        if ln and not ln.startswith("#") and not ln.startswith("voice,")
    ]
    voices = [ln.split(",")[0] for ln in data_rows]
    assert voices == ["fast", "slow"]
    # corpus aggregate comments still describe the whole corpus
    assert "# num_voices: 4" in text


def test_csv_top_n_none_omits_comment():
    batch = _batch(_SORT_VOICES)
    text = gv.render_calibration_batch_csv(batch)
    assert "# top_n" not in text


def test_handler_top_n_threads_to_human_render():
    lines = _run(
        [
            "calibrate-base-wpm-batch",
            "--voice", "mid", "165:60.0",
            "--voice", "fast", "180:60.0",
            "--voice", "slow", "150:60.0",
            "--sort-by", "delta", "--top-n", "2",
        ]
    )
    assert _row_order(lines) == ["fast", "slow"]
    assert "(top 2 of 3)" in "\n".join(lines)


def test_handler_top_n_threads_to_json():
    import json as _json

    lines = _run(
        [
            "calibrate-base-wpm-batch",
            "--voice", "mid", "165:60.0",
            "--voice", "fast", "180:60.0",
            "--voice", "slow", "150:60.0",
            "--json", "--sort-by", "delta", "--top-n", "2",
        ]
    )
    obj = _json.loads(lines[0])
    assert obj["top_n"] == 2
    assert [r["voice"] for r in obj["rows"]] == ["fast", "slow"]


def test_handler_top_n_matches_render_directly():
    batch = _batch(
        [
            ("mid", [(165, 60.0, 1.0)]),
            ("fast", [(180, 60.0, 1.0)]),
            ("slow", [(150, 60.0, 1.0)]),
        ]
    )
    expected = gv.render_calibration_batch(batch, sort_by="delta", top_n=2)
    lines = _run(
        [
            "calibrate-base-wpm-batch",
            "--voice", "mid", "165:60.0",
            "--voice", "fast", "180:60.0",
            "--voice", "slow", "150:60.0",
            "--sort-by", "delta", "--top-n", "2",
        ]
    )
    assert lines == expected


# ---- iter-401: --min-grade render-only dispersion-grade floor ----------


def test_min_grade_type_accepts_each_grade():
    for g in ("scattered", "loose", "agree"):
        assert gv.calib_batch_min_grade_type(g) == g


def test_min_grade_type_is_case_insensitive_and_strips():
    assert gv.calib_batch_min_grade_type("  AGREE ") == "agree"


def test_min_grade_type_rejects_unknown_and_empty():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        gv.calib_batch_min_grade_type("uncalibrated")
    with pytest.raises(argparse.ArgumentTypeError):
        gv.calib_batch_min_grade_type("")


def test_calib_grade_meets_min_total_order():
    # uncalibrated (None) < scattered < loose < agree
    assert gv._calib_grade_meets_min("agree", "loose") is True
    assert gv._calib_grade_meets_min("loose", "loose") is True
    assert gv._calib_grade_meets_min("scattered", "loose") is False
    assert gv._calib_grade_meets_min(None, "scattered") is False
    # None floor passes everything (no filter requested)
    assert gv._calib_grade_meets_min(None, None) is True
    assert gv._calib_grade_meets_min("scattered", None) is True


def test_parser_min_grade_parses():
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm-batch", "--voice", "a", "165:60.0",
         "--min-grade", "loose"]
    )
    assert args.min_grade == "loose"


def test_parser_min_grade_default_none():
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm-batch", "--voice", "a", "165:60.0"]
    )
    assert args.min_grade is None


def test_parser_min_grade_rejects_unknown():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(
            ["calibrate-base-wpm-batch", "--voice", "a", "165:60.0",
             "--min-grade", "bogus"]
        )
    assert exc.value.code == 2


# A corpus with exactly one voice at each grade (agree / loose / scattered) plus an
# uncalibrated voice, so every floor produces a determinate surviving set.
_GRADE_VOICES = [
    ("tight", [(165, 60.0, 1.0)]),                    # agree
    ("loosey", [(155, 60.0, 1.0), (170, 60.0, 1.0)]),  # loose
    ("wide", [(120, 60.0, 1.0), (240, 60.0, 1.0)]),    # scattered
    ("empty", []),                                     # uncalibrated
]


def test_min_grade_none_keeps_every_voice():
    batch = _batch(_GRADE_VOICES)
    order = _row_order(gv.render_calibration_batch(batch))
    assert order == ["tight", "loosey", "wide", "empty"]


def test_min_grade_scattered_keeps_all_calibrated_drops_uncalibrated():
    batch = _batch(_GRADE_VOICES)
    order = _row_order(gv.render_calibration_batch(batch, min_grade="scattered"))
    assert order == ["tight", "loosey", "wide"]  # empty dropped


def test_min_grade_loose_drops_scattered_and_uncalibrated():
    batch = _batch(_GRADE_VOICES)
    order = _row_order(gv.render_calibration_batch(batch, min_grade="loose"))
    assert order == ["tight", "loosey"]


def test_min_grade_agree_keeps_only_tightest():
    batch = _batch(_GRADE_VOICES)
    order = _row_order(gv.render_calibration_batch(batch, min_grade="agree"))
    assert order == ["tight"]


def test_min_grade_applied_before_sort_and_top_n():
    # Floor first (keep agree+loose), then sort by base_wpm ascending: loosey 162.5
    # before tight 165.0, then cap to 1 → loosey only.
    batch = _batch(_GRADE_VOICES)
    order = _row_order(
        gv.render_calibration_batch(
            batch, min_grade="loose", sort_by="base_wpm", top_n=1
        )
    )
    assert order == ["loosey"]


def test_min_grade_names_floor_in_header():
    batch = _batch(_GRADE_VOICES)
    text = "\n".join(gv.render_calibration_batch(batch, min_grade="loose"))
    assert "min grade loose" in text
    # default render does NOT mention a floor
    assert "min grade" not in "\n".join(gv.render_calibration_batch(batch))


def test_min_grade_removes_every_voice_emits_note():
    # Only scattered + uncalibrated voices; an "agree" floor keeps none.
    batch = _batch(
        [
            ("wide", [(120, 60.0, 1.0), (240, 60.0, 1.0)]),
            ("empty", []),
        ]
    )
    text = "\n".join(gv.render_calibration_batch(batch, min_grade="agree"))
    assert "no voice calibrated to grade 'agree' or better" in text
    # but the corpus summary + histogram still describe the whole corpus
    assert "corpus:" in text
    assert "grades:" in text


def test_min_grade_does_not_change_corpus_summary():
    batch = _batch(_GRADE_VOICES)
    plain = "\n".join(gv.render_calibration_batch(batch))
    floored = "\n".join(gv.render_calibration_batch(batch, min_grade="agree"))
    # The histogram counts all four voices in both renders.
    for marker in ("1 agree", "1 loose", "1 scattered", "1 uncalibrated"):
        assert marker in plain
        assert marker in floored


def test_json_min_grade_filters_rows_and_names_key():
    import json as _json

    batch = _batch(_GRADE_VOICES)
    obj = _json.loads(
        gv.render_calibration_batch_json(batch, min_grade="loose")
    )
    assert obj["min_grade"] == "loose"
    assert [r["voice"] for r in obj["rows"]] == ["tight", "loosey"]
    # aggregates still describe the whole corpus
    assert obj["num_voices"] == 4
    assert obj["grade_counts"]["scattered"] == 1


def test_json_min_grade_none_omits_key():
    import json as _json

    batch = _batch(_GRADE_VOICES)
    obj = _json.loads(gv.render_calibration_batch_json(batch))
    assert "min_grade" not in obj
    assert len(obj["rows"]) == 4


def test_csv_min_grade_filters_rows_and_comments_key():
    batch = _batch(_GRADE_VOICES)
    text = gv.render_calibration_batch_csv(batch, min_grade="loose")
    assert "# min_grade: loose" in text
    data_rows = [
        ln for ln in text.splitlines()
        if ln and not ln.startswith("#") and not ln.startswith("voice,")
    ]
    voices = [ln.split(",")[0] for ln in data_rows]
    assert voices == ["tight", "loosey"]
    # corpus aggregate comments still describe the whole corpus
    assert "# num_voices: 4" in text


def test_csv_min_grade_none_omits_comment():
    batch = _batch(_GRADE_VOICES)
    text = gv.render_calibration_batch_csv(batch)
    assert "# min_grade" not in text


def test_handler_min_grade_threads_to_human_render():
    lines = _run(
        [
            "calibrate-base-wpm-batch",
            "--voice", "tight", "165:60.0",
            "--voice", "wide", "120:60.0", "240:60.0",
            "--min-grade", "agree",
        ]
    )
    assert _row_order_pair(lines, ("tight", "wide")) == ["tight"]
    assert "min grade agree" in "\n".join(lines)


def test_handler_min_grade_threads_to_json():
    import json as _json

    lines = _run(
        [
            "calibrate-base-wpm-batch",
            "--voice", "tight", "165:60.0",
            "--voice", "wide", "120:60.0", "240:60.0",
            "--json", "--min-grade", "agree",
        ]
    )
    obj = _json.loads(lines[0])
    assert obj["min_grade"] == "agree"
    assert [r["voice"] for r in obj["rows"]] == ["tight"]


def test_handler_min_grade_matches_render_directly():
    batch = _batch(
        [
            ("tight", [(165, 60.0, 1.0)]),
            ("loosey", [(155, 60.0, 1.0), (170, 60.0, 1.0)]),
        ]
    )
    expected = gv.render_calibration_batch(batch, min_grade="loose")
    lines = _run(
        [
            "calibrate-base-wpm-batch",
            "--voice", "tight", "165:60.0",
            "--voice", "loosey", "155:60.0", "170:60.0",
            "--min-grade", "loose",
        ]
    )
    assert lines == expected


# ---- iter-402: --summary single most-representative voice --------------


def _summary_lines(lines):
    """The 'representative:' / '(no voice ...)' verdict lines from a summary render."""
    return [
        ln.strip()
        for ln in lines
        if "representative:" in ln or "nothing to summarise" in ln
    ]


# slow=150, mid=165, fast=180 → corpus median 165, deltas -15 / 0 / +15. "mid" sits
# exactly on the median, so it is the unambiguous most-representative voice.
_REP_VOICES = [
    ("slow", [(150, 60.0, 1.0)]),
    ("fast", [(180, 60.0, 1.0)]),
    ("mid", [(165, 60.0, 1.0)]),
]


def test_best_calib_batch_row_picks_nearest_median():
    batch = _batch(_REP_VOICES)
    best = gv._best_calib_batch_row(batch.rows)
    assert best["voice"] == "mid"
    assert best["delta_from_median_wpm"] == 0.0


def test_best_calib_batch_row_none_when_no_delta():
    batch = _batch([("a", []), ("b", [])])
    assert gv._best_calib_batch_row(batch.rows) is None


def test_best_calib_batch_row_tie_breaks_to_higher_grade():
    # Two voices, median = (150+180)/2 = 165, both |Δ| = 15. The tie breaks toward the
    # higher dispersion grade. "lo" is a 2-sample loose calibration at 150; "hi" is a
    # single-sample agree calibration at 180. Despite "lo" being listed first, the
    # higher grade (agree) wins.
    batch = _batch(
        [
            ("lo", [(145, 60.0, 1.0), (155, 60.0, 1.0)]),  # 150, loose
            ("hi", [(180, 60.0, 1.0)]),                     # 180, agree
        ]
    )
    best = gv._best_calib_batch_row(batch.rows)
    assert best["voice"] == "hi"


def test_summary_human_names_representative_voice():
    batch = _batch(_REP_VOICES)
    lines = gv.render_calibration_batch(batch, summary=True)
    text = "\n".join(lines)
    assert "calibration batch summary" in text
    assert "representative: mid → base_wpm 165.0" in text
    assert "Δmedian 0.0" in text
    # the whole-corpus summary + histogram are still emitted
    assert "corpus: median 165.0" in text
    assert "grades:" in text
    # exactly one verdict line, no per-voice table rows for slow/fast
    assert _row_order_pair(lines, ("slow", "fast")) == []


def test_summary_independent_of_sort_and_top_n():
    batch = _batch(_REP_VOICES)
    base = gv.render_calibration_batch(batch, summary=True)
    # --sort-by / --top-n must NOT change the pick.
    assert _summary_lines(
        gv.render_calibration_batch(batch, summary=True, sort_by="delta", top_n=1)
    ) == _summary_lines(base)


def test_summary_respects_min_grade():
    # Floor out "mid" (the natural representative) by making it scattered; with an
    # "agree" floor only the tight voices remain, so the pick is chosen among them.
    batch = _batch(
        [
            ("mid", [(120, 60.0, 1.0), (210, 60.0, 1.0)]),  # 165 but scattered
            ("slow", [(150, 60.0, 1.0)]),                    # agree
            ("fast", [(180, 60.0, 1.0)]),                    # agree
        ]
    )
    text = "\n".join(
        gv.render_calibration_batch(batch, summary=True, min_grade="agree")
    )
    # mid is filtered out; slow/fast are equidistant (median 165) so earliest wins.
    assert "representative: slow" in text
    assert "min grade agree" in text


def test_summary_no_voice_emits_note():
    batch = _batch([("a", []), ("b", [])])
    text = "\n".join(gv.render_calibration_batch(batch, summary=True))
    assert "nothing to summarise" in text
    # corpus + histogram still present
    assert "grades:" in text


def test_summary_no_voice_after_floor_names_floor():
    # Only a scattered voice survives the corpus; an "agree" floor removes everyone.
    batch = _batch([("wide", [(120, 60.0, 1.0), (240, 60.0, 1.0)]), ("empty", [])])
    text = "\n".join(
        gv.render_calibration_batch(batch, summary=True, min_grade="agree")
    )
    assert "no voice calibrated to grade 'agree' or better" in text


def test_parser_summary_parses():
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm-batch", "--voice", "a", "165:60.0", "--summary"]
    )
    assert args.summary is True


def test_parser_summary_default_false():
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm-batch", "--voice", "a", "165:60.0"]
    )
    assert args.summary is False


# ---- iter-402: --summary in --json -------------------------------------


def test_json_summary_replaces_rows_with_best():
    import json as _json

    batch = _batch(_REP_VOICES)
    obj = _json.loads(gv.render_calibration_batch_json(batch, summary=True))
    assert obj["summary"] is True
    assert "rows" not in obj
    assert obj["best"]["voice"] == "mid"
    # corpus aggregates still carried
    assert obj["num_voices"] == 3
    assert obj["implied_base_wpm_median"] == 165.0


def test_json_summary_best_null_when_no_delta():
    import json as _json

    batch = _batch([("a", []), ("b", [])])
    obj = _json.loads(gv.render_calibration_batch_json(batch, summary=True))
    assert obj["summary"] is True
    assert obj["best"] is None


def test_json_summary_independent_of_sort_and_top_n():
    import json as _json

    batch = _batch(_REP_VOICES)
    base = _json.loads(gv.render_calibration_batch_json(batch, summary=True))
    other = _json.loads(
        gv.render_calibration_batch_json(batch, summary=True, sort_by="delta", top_n=1)
    )
    assert base["best"]["voice"] == other["best"]["voice"] == "mid"
    # the ordering/count keys are meaningless in summary mode and omitted
    assert "sort_by" not in other
    assert "top_n" not in other


def test_json_summary_omits_summary_key_when_false():
    import json as _json

    batch = _batch(_REP_VOICES)
    obj = _json.loads(gv.render_calibration_batch_json(batch))
    assert "summary" not in obj
    assert "best" not in obj
    assert "rows" in obj


# ---- iter-402: --summary in --csv --------------------------------------


def _csv_data_rows(text):
    return [
        ln for ln in text.splitlines()
        if ln and not ln.startswith("#") and not ln.startswith("voice,")
    ]


def test_csv_summary_emits_single_best_row():
    batch = _batch(_REP_VOICES)
    text = gv.render_calibration_batch_csv(batch, summary=True)
    assert "# summary: true" in text
    data = _csv_data_rows(text)
    assert len(data) == 1
    assert data[0].split(",")[0] == "mid"
    # header + corpus comments unchanged
    assert text.splitlines()[0].startswith("voice,")
    assert "# num_voices: 3" in text


def test_csv_summary_header_only_when_no_delta():
    batch = _batch([("a", []), ("b", [])])
    text = gv.render_calibration_batch_csv(batch, summary=True)
    assert "# summary: true" in text
    assert _csv_data_rows(text) == []


def test_csv_summary_omits_sort_and_top_n_comments():
    batch = _batch(_REP_VOICES)
    text = gv.render_calibration_batch_csv(
        batch, summary=True, sort_by="delta", top_n=1
    )
    assert "# sort_by" not in text
    assert "# top_n" not in text
    assert _csv_data_rows(text)[0].split(",")[0] == "mid"


def test_csv_summary_omits_comment_when_false():
    batch = _batch(_REP_VOICES)
    text = gv.render_calibration_batch_csv(batch)
    assert "# summary" not in text


# ---- iter-402: handler threading --------------------------------------


def test_handler_summary_threads_to_human_render():
    lines = _run(
        [
            "calibrate-base-wpm-batch",
            "--voice", "slow", "150:60.0",
            "--voice", "fast", "180:60.0",
            "--voice", "mid", "165:60.0",
            "--summary",
        ]
    )
    text = "\n".join(lines)
    assert "representative: mid" in text


def test_handler_summary_threads_to_json():
    import json as _json

    lines = _run(
        [
            "calibrate-base-wpm-batch",
            "--voice", "slow", "150:60.0",
            "--voice", "fast", "180:60.0",
            "--voice", "mid", "165:60.0",
            "--json", "--summary",
        ]
    )
    obj = _json.loads(lines[0])
    assert obj["summary"] is True
    assert obj["best"]["voice"] == "mid"


def test_handler_summary_matches_render_directly():
    batch = _batch(_REP_VOICES)
    expected = gv.render_calibration_batch(batch, summary=True)
    lines = _run(
        [
            "calibrate-base-wpm-batch",
            "--voice", "slow", "150:60.0",
            "--voice", "fast", "180:60.0",
            "--voice", "mid", "165:60.0",
            "--summary",
        ]
    )
    assert lines == expected
