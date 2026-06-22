"""Tests for iter-221 — the ``gv calibrate-base-wpm`` subcommand (examples/gv.py).

iter-220 shipped the audio-free calibration core (``CalibrationSample`` /
``calibrate_base_wpm``): the pure arithmetic that turns a rendered TTS sample
(``words`` synthesized into ``audio_seconds`` of audio at a known Kokoro
``speed``) into a measured ``base_wpm``. iter-221 exposes that core on the
``gv`` CLI — the iter-218-style CLI-later twin of the iter-216 engine — so an
operator can fold rendered samples offline and read the implied ``base_wpm`` to
set ``DEFAULT_BASE_WPM`` from their own voice.

These tests exercise the ``--samples`` arg type, the pure ``render_calibration``
helper, and the handler (driven with an injected ``log`` so no real I/O
happens). The engine is pure stdlib loaded by file path, so the handler runs on
this x86_64 Linux runner without pipecat.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples import gv  # noqa: E402


# ---- parser: defaults & wiring -----------------------------------------


def test_calibrate_in_handler_map():
    assert gv.DEFAULT_HANDLERS["calibrate-base-wpm"] is gv.cmd_calibrate_base_wpm


def test_calibrate_defaults():
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm", "--samples", "50:18.2"]
    )
    assert args.command == "calibrate-base-wpm"
    assert args.samples == [(50.0, 18.2, 1.0)]
    # The nominal default is sourced from the engine seed.
    assert args.nominal == 165.0


def test_calibrate_requires_samples():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(["calibrate-base-wpm"])
    assert exc.value.code == 2


def test_calibrate_multiple_samples_and_nominal():
    args = gv.build_parser().parse_args(
        [
            "calibrate-base-wpm",
            "--samples", "50:18.2", "50:9.1:2.0", "30:12.0:1.0",
            "--nominal", "170",
        ]
    )
    assert args.samples == [
        (50.0, 18.2, 1.0),
        (50.0, 9.1, 2.0),
        (30.0, 12.0, 1.0),
    ]
    assert args.nominal == 170.0


# ---- calibration_sample_type: pure arg parsing -------------------------


def test_sample_type_two_fields_defaults_speed():
    assert gv.calibration_sample_type("50:18.2") == (50.0, 18.2, 1.0)


def test_sample_type_three_fields():
    assert gv.calibration_sample_type("50:9.1:2.0") == (50.0, 9.1, 2.0)


def test_sample_type_strips_whitespace():
    assert gv.calibration_sample_type(" 50 : 18.2 : 1.5 ") == (50.0, 18.2, 1.5)


@pytest.mark.parametrize(
    "raw",
    [
        "50",            # too few fields
        "50:18:1:9",     # too many fields
        "50::1",         # empty middle field
        ":18:1",         # empty leading field
        "",              # empty
    ],
)
def test_sample_type_rejects_bad_shape(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.calibration_sample_type(raw)


@pytest.mark.parametrize("raw", ["abc:18", "50:xx", "50:18:zz"])
def test_sample_type_rejects_non_numeric(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.calibration_sample_type(raw)


@pytest.mark.parametrize("raw", ["nan:18", "50:nan", "50:18:nan"])
def test_sample_type_rejects_nan(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.calibration_sample_type(raw)


@pytest.mark.parametrize("raw", ["0:18", "50:0", "50:18:0", "-5:18", "50:-1"])
def test_sample_type_rejects_non_positive(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.calibration_sample_type(raw)


def test_sample_type_rejects_non_string():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.calibration_sample_type(50)


def test_sample_type_end_to_end_systemexit():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(["calibrate-base-wpm", "--samples", "bad"])
    assert exc.value.code == 2


# ---- render_calibration: pure formatting -------------------------------


def test_render_calibration_none_no_samples():
    lines = gv.render_calibration(None)
    assert lines == [
        "base_wpm calibration: no samples (nothing to calibrate from)"
    ]


def test_render_calibration_all_fields():
    wm = gv._load_wpm_mirror()
    # 50 words / 18.2s at speed 1.0 → ~164.8 wpm.
    samples = [wm.CalibrationSample(words=50, audio_seconds=18.2)]
    calib = wm.calibrate_base_wpm(samples)
    lines = gv.render_calibration(calib)
    text = "\n".join(lines)
    assert "base_wpm calibration from rendered samples" in text
    assert "samples:" in text
    assert "implied base_wpm:" in text
    assert "range:" in text
    assert "spread:" in text
    assert "relative spread:" in text  # iter-393
    assert "nominal:" in text
    assert "drift:" in text
    assert f"{calib.implied_base_wpm:.1f}" in text
    assert f"{calib.relative_spread:.3f}" in text  # iter-393


def test_render_calibration_human_shows_dispersion_grade():
    # iter-394: the dispersion line carries the grade + a trust note.
    wm = gv._load_wpm_mirror()
    samples = [
        wm.CalibrationSample(words=164, audio_seconds=60.0),
        wm.CalibrationSample(words=165, audio_seconds=60.0),
        wm.CalibrationSample(words=166, audio_seconds=60.0),
    ]
    calib = wm.calibrate_base_wpm(samples)
    text = "\n".join(gv.render_calibration(calib))
    assert "dispersion:" in text
    assert calib.dispersion_grade == "agree"
    assert "dispersion:       agree" in text
    assert "cluster tightly" in text  # the "agree" trust note
    # iter-396: the margin note rides alongside the grade.
    assert f"{calib.dispersion_margin:.3f} relative-spread headroom" in text


def test_render_calibration_human_dispersion_grade_scattered():
    wm = gv._load_wpm_mirror()
    samples = [
        wm.CalibrationSample(words=130, audio_seconds=60.0),
        wm.CalibrationSample(words=160, audio_seconds=60.0),
        wm.CalibrationSample(words=190, audio_seconds=60.0),
    ]
    calib = wm.calibrate_base_wpm(samples)
    text = "\n".join(gv.render_calibration(calib))
    assert calib.dispersion_grade == "scattered"
    assert "dispersion:       scattered" in text
    assert "re-render more consistently" in text
    # iter-396: a scattered calibration has no lower grade to fall to.
    assert calib.dispersion_margin is None
    assert "no lower grade to fall to" in text


def test_calib_dispersion_summary_defensive_fallback():
    # An unexpected grade never drops the signal silently.
    assert "unrecognized" in gv._calib_dispersion_summary("bogus")
    for g in ("agree", "loose", "scattered"):
        assert "unrecognized" not in gv._calib_dispersion_summary(g)


def test_calib_dispersion_margin_note_finite_and_none():
    # iter-396: a finite margin shows headroom; None reads as "lowest grade".
    assert "0.030 relative-spread headroom" in gv._calib_dispersion_margin_note(0.03)
    assert "no lower grade to fall to" in gv._calib_dispersion_margin_note(None)


# ---- cmd_calibrate_base_wpm: handler with injected log -----------------


def _run(argv):
    """Parse argv and run the handler, capturing the emitted log lines."""
    args = gv.build_parser().parse_args(argv)
    lines: list = []
    gv.cmd_calibrate_base_wpm(args, log=lines.append)
    return lines


def test_handler_emits_report():
    lines = _run(["calibrate-base-wpm", "--samples", "50:18.2"])
    text = "\n".join(lines)
    assert "calibration from rendered samples" in text
    assert "implied base_wpm" in text


def test_handler_matches_engine_directly():
    # The handler's report reflects the same fold the engine runs directly.
    wm = gv._load_wpm_mirror()
    samples = [
        wm.CalibrationSample(words=50, audio_seconds=18.2),
        wm.CalibrationSample(words=50, audio_seconds=9.1, speed=2.0),
    ]
    calib = wm.calibrate_base_wpm(samples, default_base_wpm=165.0)
    expected = gv.render_calibration(calib)
    lines = _run(["calibrate-base-wpm", "--samples", "50:18.2", "50:9.1:2.0"])
    assert lines == expected


def test_handler_nominal_threads_to_drift():
    # The --nominal override is reported as the drift baseline.
    lines = _run(["calibrate-base-wpm", "--samples", "50:18.2", "--nominal", "100"])
    text = "\n".join(lines)
    assert "100.0" in text  # nominal line


def test_handler_default_log_is_print(capsys):
    args = gv.build_parser().parse_args(["calibrate-base-wpm", "--samples", "50:18.2"])
    gv.cmd_calibrate_base_wpm(args)
    out = capsys.readouterr().out
    assert "calibration from rendered samples" in out


def test_handler_dispatch_routes(capsys):
    # End-to-end through main(): the subcommand dispatches and prints.
    rc = gv.main(["calibrate-base-wpm", "--samples", "50:18.2"])
    assert rc == 0
    assert "calibration from rendered samples" in capsys.readouterr().out


# ---- iter-223: --verdict surface (parser, formatter, handler) ----------


def test_verdict_flag_defaults_off():
    # Without --verdict the flag is False and gates default from the engine.
    args = gv.build_parser().parse_args(["calibrate-base-wpm", "--samples", "50:18.2"])
    assert args.verdict is False
    assert args.spread_max == 10.0
    assert args.drift_min == 5.0
    assert args.min_samples == 3


def test_verdict_flag_and_gate_overrides_parse():
    args = gv.build_parser().parse_args(
        [
            "calibrate-base-wpm", "--samples", "50:18.2",
            "--verdict",
            "--spread-max", "20",
            "--drift-min", "2.5",
            "--min-samples", "1",
        ]
    )
    assert args.verdict is True
    assert args.spread_max == 20.0
    assert args.drift_min == 2.5
    assert args.min_samples == 1


def test_render_calibration_verdict_none():
    lines = gv.render_calibration_verdict(None)
    assert lines == ["base_wpm verdict: no samples (nothing to decide)"]


def test_render_calibration_verdict_recommend():
    wm = gv._load_wpm_mirror()
    # 50 words / 14.0s ≈ 214 wpm — well above the 165 nominal, 3 samples agree.
    samples = [wm.CalibrationSample(words=50, audio_seconds=14.0)] * 3
    calib = wm.calibrate_base_wpm(samples, default_base_wpm=165.0)
    verdict = wm.calibration_verdict(calib)
    lines = gv.render_calibration_verdict(verdict)
    text = "\n".join(lines)
    assert "base_wpm calibration verdict" in text
    assert "decision: re-seed base_wpm to" in text
    assert f"{verdict.implied_base_wpm:.1f}" in text
    assert "reason:" in text
    assert "gates:" in text


def test_render_calibration_verdict_keep():
    wm = gv._load_wpm_mirror()
    # A 165-nominal voice with tiny drift → keep.
    samples = [wm.CalibrationSample(words=50, audio_seconds=18.2)] * 3
    calib = wm.calibrate_base_wpm(samples, default_base_wpm=165.0)
    verdict = wm.calibration_verdict(calib)
    lines = gv.render_calibration_verdict(verdict)
    text = "\n".join(lines)
    assert "decision: keep the current nominal" in text


def test_render_calibration_verdict_shows_dispersion_grade():
    # iter-395: the verdict render carries a dispersion line echoing the grade.
    wm = gv._load_wpm_mirror()
    samples = [wm.CalibrationSample(words=50, audio_seconds=14.0)] * 3
    calib = wm.calibrate_base_wpm(samples, default_base_wpm=165.0)
    verdict = wm.calibration_verdict(calib)
    lines = gv.render_calibration_verdict(verdict)
    text = "\n".join(lines)
    assert "dispersion:" in text
    assert verdict.dispersion_grade in text
    assert "reading aid" in text


def test_render_calibration_verdict_reason_cites_grade():
    # iter-395: the recommend reason names the dispersion grade.
    wm = gv._load_wpm_mirror()
    samples = [wm.CalibrationSample(words=50, audio_seconds=14.0)] * 3
    calib = wm.calibrate_base_wpm(samples, default_base_wpm=165.0)
    verdict = wm.calibration_verdict(calib)
    assert verdict.recommend is True
    assert verdict.dispersion_grade in verdict.reason
    assert "dispersion" in verdict.reason


def test_handler_omits_verdict_by_default():
    lines = _run(["calibrate-base-wpm", "--samples", "50:14.0", "50:14.0", "50:14.0"])
    text = "\n".join(lines)
    assert "calibration from rendered samples" in text
    assert "calibration verdict" not in text


def test_handler_emits_verdict_when_flagged():
    lines = _run(
        ["calibrate-base-wpm", "--samples", "50:14.0", "50:14.0", "50:14.0",
         "--verdict"]
    )
    text = "\n".join(lines)
    # Both the raw report AND the verdict appear, raw first.
    assert "calibration from rendered samples" in text
    assert "calibration verdict" in text
    assert text.index("from rendered samples") < text.index("calibration verdict")
    assert "decision: re-seed base_wpm to" in text


def test_handler_verdict_matches_engine_directly():
    # The handler's verdict reflects the same fold the engine runs directly.
    wm = gv._load_wpm_mirror()
    samples = [wm.CalibrationSample(words=50, audio_seconds=14.0)] * 3
    calib = wm.calibrate_base_wpm(samples, default_base_wpm=165.0)
    verdict = wm.calibration_verdict(calib)
    expected = gv.render_calibration(calib) + gv.render_calibration_verdict(verdict)
    lines = _run(
        ["calibrate-base-wpm", "--samples", "50:14.0", "50:14.0", "50:14.0",
         "--verdict"]
    )
    assert lines == expected


def test_handler_verdict_gate_overrides_thread_through():
    # --min-samples 1 lets a single significant render recommend a re-seed.
    lines = _run(
        ["calibrate-base-wpm", "--samples", "50:14.0", "--verdict",
         "--min-samples", "1"]
    )
    text = "\n".join(lines)
    assert "decision: re-seed base_wpm to" in text


def test_handler_verdict_below_threshold_keeps():
    # A large --drift-min suppresses the re-seed even on a fast voice.
    lines = _run(
        ["calibrate-base-wpm", "--samples", "50:14.0", "50:14.0", "50:14.0",
         "--verdict", "--drift-min", "100"]
    )
    text = "\n".join(lines)
    assert "decision: keep the current nominal" in text


def test_handler_verdict_dispatch_routes(capsys):
    rc = gv.main(
        ["calibrate-base-wpm", "--samples", "50:14.0", "50:14.0", "50:14.0",
         "--verdict"]
    )
    assert rc == 0
    assert "calibration verdict" in capsys.readouterr().out


# ---- iter-316: --csv surface (parser, renderer, handler) ---------------

import csv as _csv  # noqa: E402
import io as _io  # noqa: E402


def test_csv_flag_defaults_off():
    args = gv.build_parser().parse_args(["calibrate-base-wpm", "--samples", "50:18.2"])
    assert args.csv is False


def test_csv_flag_sets_true():
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm", "--samples", "50:18.2", "--csv"]
    )
    assert args.csv is True


def _samples(specs):
    """Build CalibrationSample objects from (words, seconds[, speed]) tuples."""
    wm = gv._load_wpm_mirror()
    out = []
    for spec in specs:
        if len(spec) == 2:
            words, secs = spec
            out.append(wm.CalibrationSample(words=words, audio_seconds=secs))
        else:
            words, secs, speed = spec
            out.append(
                wm.CalibrationSample(words=words, audio_seconds=secs, speed=speed)
            )
    return out


def _rows(text):
    """Parse the non-comment CSV body into a list of dict rows."""
    data = "\n".join(
        line for line in text.splitlines() if not line.startswith("#")
    )
    return list(_csv.DictReader(_io.StringIO(data)))


def _comments(text):
    return [line for line in text.splitlines() if line.startswith("#")]


def test_render_calibration_csv_header_and_one_row_per_sample():
    wm = gv._load_wpm_mirror()
    samples = _samples([(50, 18.2), (50, 9.1, 2.0)])
    calib = wm.calibrate_base_wpm(samples)
    text = gv.render_calibration_csv(samples, calib)
    rows = _rows(text)
    assert len(rows) == 2
    assert list(rows[0].keys()) == [
        "sample", "words", "audio_seconds", "speed", "bot_wpm", "implied_base_wpm"
    ]
    assert [r["sample"] for r in rows] == ["1", "2"]


def test_render_calibration_csv_values_match_engine():
    wm = gv._load_wpm_mirror()
    samples = _samples([(50, 18.2), (50, 9.1, 2.0)])
    calib = wm.calibrate_base_wpm(samples)
    rows = _rows(gv.render_calibration_csv(samples, calib))
    for row, s in zip(rows, samples):
        assert float(row["words"]) == s.words
        assert float(row["audio_seconds"]) == round(s.audio_seconds, 3)
        assert float(row["speed"]) == round(s.speed, 3)
        assert float(row["bot_wpm"]) == round(s.bot_wpm, 3)
        assert float(row["implied_base_wpm"]) == round(s.implied_base_wpm, 3)


def test_render_calibration_csv_summary_comments():
    wm = gv._load_wpm_mirror()
    samples = _samples([(50, 18.2), (50, 9.1, 2.0)])
    calib = wm.calibrate_base_wpm(samples)
    text = gv.render_calibration_csv(samples, calib)
    comments = "\n".join(_comments(text))
    assert f"# implied_base_wpm (median): {round(calib.implied_base_wpm, 3)}" in comments
    assert "# range:" in comments
    assert f"# spread: {round(calib.spread, 3)}" in comments
    assert f"# relative_spread: {round(calib.relative_spread, 3)}" in comments  # iter-393
    assert f"# dispersion_grade: {calib.dispersion_grade}" in comments  # iter-394
    # iter-396: the margin trails as its own comment (finite for a non-scattered set).
    assert calib.dispersion_margin is not None
    assert f"# dispersion_margin: {round(calib.dispersion_margin, 3)}" in comments
    assert f"# nominal: {round(calib.default_base_wpm, 3)}" in comments
    assert f"# drift: {round(calib.drift, 3)}" in comments


def test_render_calibration_csv_dispersion_margin_blank_for_scattered():
    # iter-396: a scattered calibration has margin None ⇒ the comment is blank.
    wm = gv._load_wpm_mirror()
    samples = _samples([(130, 60.0), (160, 60.0), (190, 60.0)])
    calib = wm.calibrate_base_wpm(samples)
    assert calib.dispersion_grade == "scattered"
    assert calib.dispersion_margin is None
    comments = "\n".join(_comments(gv.render_calibration_csv(samples, calib)))
    assert "# dispersion_margin:" in comments
    # The value after the label is empty (no finite headroom to show).
    for line in comments.splitlines():
        if line.startswith("# dispersion_margin:"):
            assert line.strip() == "# dispersion_margin:"


def test_render_calibration_csv_none_calib_header_only():
    # No samples ⇒ None calib ⇒ header alone, no summary comments.
    text = gv.render_calibration_csv([], None)
    assert text == "sample,words,audio_seconds,speed,bot_wpm,implied_base_wpm"
    assert _comments(text) == []


def test_render_calibration_csv_no_trailing_newline():
    wm = gv._load_wpm_mirror()
    samples = _samples([(50, 18.2)])
    calib = wm.calibrate_base_wpm(samples)
    text = gv.render_calibration_csv(samples, calib)
    assert not text.endswith("\n")
    assert not text.endswith("\r")


def test_render_calibration_csv_nominal_threads_to_drift():
    wm = gv._load_wpm_mirror()
    samples = _samples([(50, 18.2)])
    calib = wm.calibrate_base_wpm(samples, default_base_wpm=100.0)
    text = gv.render_calibration_csv(samples, calib)
    assert "# nominal: 100.0" in text
    assert f"# drift: {round(calib.drift, 3)}" in text


def test_handler_csv_matches_renderer():
    wm = gv._load_wpm_mirror()
    samples = _samples([(50, 18.2), (50, 9.1, 2.0)])
    calib = wm.calibrate_base_wpm(samples, default_base_wpm=165.0)
    expected = [gv.render_calibration_csv(samples, calib)]
    lines = _run(["calibrate-base-wpm", "--samples", "50:18.2", "50:9.1:2.0", "--csv"])
    assert lines == expected


def test_handler_csv_suppresses_human_report():
    lines = _run(["calibrate-base-wpm", "--samples", "50:18.2", "--csv"])
    text = "\n".join(lines)
    assert "calibration from rendered samples" not in text
    # The CSV header is present instead.
    assert "sample,words,audio_seconds,speed,bot_wpm,implied_base_wpm" in text


def test_handler_csv_suppresses_verdict():
    # Even with --verdict, --csv wins and no prose decision is emitted.
    lines = _run(
        ["calibrate-base-wpm", "--samples", "50:14.0", "50:14.0", "50:14.0",
         "--verdict", "--csv"]
    )
    text = "\n".join(lines)
    assert "calibration verdict" not in text
    assert "decision:" not in text


def test_handler_csv_rows_are_parseable():
    lines = _run(["calibrate-base-wpm", "--samples", "50:18.2", "50:9.1:2.0", "--csv"])
    rows = _rows("\n".join(lines))
    assert len(rows) == 2
    assert [r["sample"] for r in rows] == ["1", "2"]


def test_handler_csv_default_log_is_print(capsys):
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm", "--samples", "50:18.2", "--csv"]
    )
    gv.cmd_calibrate_base_wpm(args)
    out = capsys.readouterr().out
    assert "sample,words,audio_seconds,speed,bot_wpm,implied_base_wpm" in out


def test_handler_csv_dispatch_routes(capsys):
    rc = gv.main(["calibrate-base-wpm", "--samples", "50:18.2", "--csv"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sample,words,audio_seconds,speed,bot_wpm,implied_base_wpm" in out
    assert "# implied_base_wpm (median):" in out


# ---- iter-317: --json surface (parser, renderer, handler) --------------

import json as _json  # noqa: E402


def test_json_flag_defaults_off():
    args = gv.build_parser().parse_args(["calibrate-base-wpm", "--samples", "50:18.2"])
    assert args.json is False


def test_json_flag_sets_true():
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm", "--samples", "50:18.2", "--json"]
    )
    assert args.json is True


def test_json_and_csv_mutually_exclusive():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(
            ["calibrate-base-wpm", "--samples", "50:18.2", "--json", "--csv"]
        )
    assert exc.value.code == 2


def test_render_calibration_json_samples_and_calibration():
    wm = gv._load_wpm_mirror()
    samples = _samples([(50, 18.2), (50, 9.1, 2.0)])
    calib = wm.calibrate_base_wpm(samples)
    payload = _json.loads(gv.render_calibration_json(samples, calib))
    assert len(payload["samples"]) == 2
    assert [s["sample"] for s in payload["samples"]] == [1, 2]
    assert set(payload["samples"][0]) == {
        "sample", "words", "audio_seconds", "speed", "bot_wpm", "implied_base_wpm"
    }
    cal = payload["calibration"]
    assert cal["implied_base_wpm"] == round(calib.implied_base_wpm, 3)
    assert cal["n_samples"] == calib.n_samples
    assert cal["spread"] == round(calib.spread, 3)
    assert cal["relative_spread"] == round(calib.relative_spread, 3)  # iter-393
    assert cal["dispersion_grade"] == calib.dispersion_grade  # iter-394
    # iter-396: the margin rounds to 3 places (finite for a non-scattered set).
    assert calib.dispersion_margin is not None
    assert cal["dispersion_margin"] == round(calib.dispersion_margin, 3)
    assert cal["nominal"] == round(calib.default_base_wpm, 3)
    assert cal["drift"] == round(calib.drift, 3)


def test_render_calibration_json_sample_values_match_engine():
    wm = gv._load_wpm_mirror()
    samples = _samples([(50, 18.2), (50, 9.1, 2.0)])
    calib = wm.calibrate_base_wpm(samples)
    payload = _json.loads(gv.render_calibration_json(samples, calib))
    for obj, s in zip(payload["samples"], samples):
        assert obj["words"] == s.words
        assert obj["audio_seconds"] == round(float(s.audio_seconds), 3)
        assert obj["speed"] == round(float(s.speed), 3)
        assert obj["bot_wpm"] == round(s.bot_wpm, 3)
        assert obj["implied_base_wpm"] == round(s.implied_base_wpm, 3)


def test_render_calibration_json_none_calib_null():
    # No samples ⇒ None calib ⇒ empty samples list, calibration null.
    payload = _json.loads(gv.render_calibration_json([], None))
    assert payload["samples"] == []
    assert payload["calibration"] is None


def test_render_calibration_json_relative_spread_normalizes_base():
    # iter-393: the SAME absolute spread reads as a larger relative spread at a
    # slow voice than at a fast one — surfaced verbatim through the JSON.
    wm = gv._load_wpm_mirror()
    slow = wm.calibrate_base_wpm(
        [wm.CalibrationSample(words=80, audio_seconds=60.0),
         wm.CalibrationSample(words=100, audio_seconds=60.0),
         wm.CalibrationSample(words=120, audio_seconds=60.0)]
    )
    fast = wm.calibrate_base_wpm(
        [wm.CalibrationSample(words=280, audio_seconds=60.0),
         wm.CalibrationSample(words=300, audio_seconds=60.0),
         wm.CalibrationSample(words=320, audio_seconds=60.0)]
    )
    slow_rel = _json.loads(gv.render_calibration_json([], slow))["calibration"]["relative_spread"]
    fast_rel = _json.loads(gv.render_calibration_json([], fast))["calibration"]["relative_spread"]
    assert slow_rel > fast_rel


def test_render_calibration_json_dispersion_grade_present():
    # iter-394: the grade is carried verbatim (a string, not rounded).
    wm = gv._load_wpm_mirror()
    samples = _samples([(50, 18.2), (50, 9.1, 2.0)])  # renders agree ⇒ "agree"
    calib = wm.calibrate_base_wpm(samples)
    cal = _json.loads(gv.render_calibration_json(samples, calib))["calibration"]
    assert cal["dispersion_grade"] == calib.dispersion_grade
    assert cal["dispersion_grade"] in ("agree", "loose", "scattered")


def test_render_calibration_json_dispersion_grade_voice_comparable():
    # iter-394: equal relative spread at slow/fast voices ⇒ equal grade.
    wm = gv._load_wpm_mirror()
    slow = wm.calibrate_base_wpm(
        [wm.CalibrationSample(words=90, audio_seconds=60.0),
         wm.CalibrationSample(words=100, audio_seconds=60.0),
         wm.CalibrationSample(words=110, audio_seconds=60.0)]
    )
    fast = wm.calibrate_base_wpm(
        [wm.CalibrationSample(words=270, audio_seconds=60.0),
         wm.CalibrationSample(words=300, audio_seconds=60.0),
         wm.CalibrationSample(words=330, audio_seconds=60.0)]
    )
    slow_g = _json.loads(gv.render_calibration_json([], slow))["calibration"]["dispersion_grade"]
    fast_g = _json.loads(gv.render_calibration_json([], fast))["calibration"]["dispersion_grade"]
    assert slow_g == fast_g


def test_render_calibration_json_dispersion_margin_null_for_scattered():
    # iter-396: a scattered calibration carries dispersion_margin null.
    wm = gv._load_wpm_mirror()
    samples = _samples([(130, 60.0), (160, 60.0), (190, 60.0)])
    calib = wm.calibrate_base_wpm(samples)
    assert calib.dispersion_grade == "scattered"
    cal = _json.loads(gv.render_calibration_json(samples, calib))["calibration"]
    assert cal["dispersion_margin"] is None


def test_render_calibration_json_dispersion_margin_voice_comparable():
    # iter-396: equal relative spread at slow/fast voices ⇒ equal margin.
    wm = gv._load_wpm_mirror()
    slow = wm.calibrate_base_wpm(
        [wm.CalibrationSample(words=98, audio_seconds=60.0),
         wm.CalibrationSample(words=100, audio_seconds=60.0),
         wm.CalibrationSample(words=102, audio_seconds=60.0)]
    )
    fast = wm.calibrate_base_wpm(
        [wm.CalibrationSample(words=294, audio_seconds=60.0),
         wm.CalibrationSample(words=300, audio_seconds=60.0),
         wm.CalibrationSample(words=306, audio_seconds=60.0)]
    )
    slow_m = _json.loads(gv.render_calibration_json([], slow))["calibration"]["dispersion_margin"]
    fast_m = _json.loads(gv.render_calibration_json([], fast))["calibration"]["dispersion_margin"]
    assert slow_m == fast_m


def test_render_calibration_json_nominal_threads_to_drift():
    wm = gv._load_wpm_mirror()
    samples = _samples([(50, 18.2)])
    calib = wm.calibrate_base_wpm(samples, default_base_wpm=100.0)
    payload = _json.loads(gv.render_calibration_json(samples, calib))
    assert payload["calibration"]["nominal"] == 100.0
    assert payload["calibration"]["drift"] == round(calib.drift, 3)


def test_render_calibration_json_omits_verdict():
    # Like the CSV, the JSON carries no adopt/keep decision — that is --verdict.
    wm = gv._load_wpm_mirror()
    samples = _samples([(50, 18.2)])
    calib = wm.calibrate_base_wpm(samples)
    payload = _json.loads(gv.render_calibration_json(samples, calib))
    assert "recommend" not in payload
    assert "verdict" not in payload


def test_handler_json_matches_renderer():
    wm = gv._load_wpm_mirror()
    samples = _samples([(50, 18.2), (50, 9.1, 2.0)])
    calib = wm.calibrate_base_wpm(samples, default_base_wpm=165.0)
    expected = [gv.render_calibration_json(samples, calib)]
    lines = _run(["calibrate-base-wpm", "--samples", "50:18.2", "50:9.1:2.0", "--json"])
    assert lines == expected


def test_handler_json_suppresses_human_report():
    lines = _run(["calibrate-base-wpm", "--samples", "50:18.2", "--json"])
    text = "\n".join(lines)
    assert "calibration from rendered samples" not in text
    payload = _json.loads(text)
    assert "samples" in payload


def test_handler_json_suppresses_verdict():
    # Even with --verdict, --json wins and no prose decision is emitted.
    lines = _run(
        ["calibrate-base-wpm", "--samples", "50:14.0", "50:14.0", "50:14.0",
         "--verdict", "--json"]
    )
    text = "\n".join(lines)
    assert "calibration verdict" not in text
    assert "decision:" not in text


def test_handler_json_rows_are_parseable():
    lines = _run(["calibrate-base-wpm", "--samples", "50:18.2", "50:9.1:2.0", "--json"])
    payload = _json.loads("\n".join(lines))
    assert len(payload["samples"]) == 2
    assert [s["sample"] for s in payload["samples"]] == [1, 2]


def test_handler_json_default_log_is_print(capsys):
    args = gv.build_parser().parse_args(
        ["calibrate-base-wpm", "--samples", "50:18.2", "--json"]
    )
    gv.cmd_calibrate_base_wpm(args)
    out = capsys.readouterr().out
    payload = _json.loads(out)
    assert "samples" in payload


def test_handler_json_dispatch_routes(capsys):
    rc = gv.main(["calibrate-base-wpm", "--samples", "50:18.2", "--json"])
    assert rc == 0
    payload = _json.loads(capsys.readouterr().out)
    assert payload["calibration"]["n_samples"] == 1
