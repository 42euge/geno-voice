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
