"""Tests for iter-406 — the ``gv stt-rtf`` subcommand (examples/gv.py).

iter-405 shipped the audio-free STT real-time-factor profiling core
(``TranscriptionSample`` / ``profile_stt_rtf`` in ``stt/rtf_profile.py``): the
pure arithmetic that folds measured transcriptions
(``audio_seconds:transcribe_seconds``) into a robust median RTF with a speed
grade and dispersion/headroom diagnostics. iter-406 exposes that core on the
``gv`` CLI — the STT-side analogue of the iter-221 ``gv calibrate-base-wpm``
command — so an operator can fold a handful of offline-measured transcriptions
and read whether the transcriber keeps up with realtime or is the latency
bottleneck.

These tests exercise the ``--samples`` arg type, the pure ``render_stt_rtf`` /
``render_stt_rtf_csv`` / ``render_stt_rtf_json`` helpers, and the handler
(driven with an injected ``log`` so no real I/O happens). The core is pure
stdlib loaded by file path, so the handler runs on this x86_64 Linux runner
without torch or faster-whisper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples import gv  # noqa: E402


# ---- parser: defaults & wiring -----------------------------------------


def test_stt_rtf_in_handler_map():
    assert gv.DEFAULT_HANDLERS["stt-rtf"] is gv.cmd_stt_rtf


def test_stt_rtf_defaults():
    args = gv.build_parser().parse_args(["stt-rtf", "--samples", "10.0:1.2"])
    assert args.command == "stt-rtf"
    assert args.samples == [(10.0, 1.2)]
    assert args.json is False
    assert args.csv is False


def test_stt_rtf_requires_samples():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(["stt-rtf"])
    assert exc.value.code == 2


def test_stt_rtf_multiple_samples():
    args = gv.build_parser().parse_args(
        ["stt-rtf", "--samples", "10.0:1.2", "5.0:0.8", "20.0:30.0"]
    )
    assert args.samples == [(10.0, 1.2), (5.0, 0.8), (20.0, 30.0)]


def test_stt_rtf_json_csv_mutually_exclusive():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(
            ["stt-rtf", "--samples", "10.0:1.2", "--json", "--csv"]
        )
    assert exc.value.code == 2


# ---- stt_rtf_sample_type: pure arg parsing -----------------------------


def test_sample_type_two_fields():
    assert gv.stt_rtf_sample_type("10.0:1.2") == (10.0, 1.2)


def test_sample_type_integers_coerced_to_float():
    assert gv.stt_rtf_sample_type("10:2") == (10.0, 2.0)


def test_sample_type_strips_whitespace():
    assert gv.stt_rtf_sample_type(" 10.0 : 1.2 ") == (10.0, 1.2)


@pytest.mark.parametrize("raw", ["10.0", "10.0:1.2:3.0", "", "10.0:"])
def test_sample_type_wrong_field_count_rejected(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.stt_rtf_sample_type(raw)


def test_sample_type_non_numeric_rejected():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.stt_rtf_sample_type("ten:1.2")


def test_sample_type_nan_rejected():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.stt_rtf_sample_type("nan:1.2")


@pytest.mark.parametrize("raw", ["0:1.2", "10.0:0", "-1:1.2", "10.0:-2"])
def test_sample_type_nonpositive_rejected(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.stt_rtf_sample_type(raw)


def test_sample_type_non_string_rejected():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.stt_rtf_sample_type(12.0)


# ---- render_stt_rtf: pure human report ---------------------------------


def _profile(*pairs):
    """Build an SttRtfProfile from (audio, transcribe) pairs via the core."""
    mod = gv._load_stt_rtf_profile()
    samples = [
        mod.TranscriptionSample(audio_seconds=a, transcribe_seconds=t)
        for (a, t) in pairs
    ]
    return samples, mod.profile_stt_rtf(samples)


def test_render_none_is_no_samples():
    lines = gv.render_stt_rtf(None)
    assert lines == ["stt-rtf profile: no samples (nothing to profile from)"]


def test_render_reports_fields():
    _, profile = _profile((10.0, 1.2), (5.0, 0.8))
    lines = gv.render_stt_rtf(profile)
    text = "\n".join(lines)
    assert "STT real-time-factor profile" in lines[0]
    assert "samples:          2" in text
    assert "median RTF:       0.140" in text
    assert "range:            0.120 – 0.160" in text
    assert "spread:           0.040" in text
    assert "relative spread:  0.286" in text
    assert "speed:            fast" in text


def test_render_slow_grade_note_and_no_margin():
    _, profile = _profile((5.0, 8.0))
    text = "\n".join(gv.render_stt_rtf(profile))
    assert "speed:            slow" in text
    assert "latency bottleneck" in text
    # slow is the worst grade — no slower grade to fall to (margin None).
    assert "no slower grade to fall to" in text


def test_render_realtime_grade_shows_headroom():
    _, profile = _profile((10.0, 8.0))  # rtf 0.8 -> realtime
    text = "\n".join(gv.render_stt_rtf(profile))
    assert "speed:            realtime" in text
    # headroom to the realtime/slow knee at 1.0 is 0.2.
    assert "0.200 RTF headroom before the grade degrades" in text


# ---- render_stt_rtf_csv ------------------------------------------------


def test_render_csv_none_is_header_only():
    out = gv.render_stt_rtf_csv([], None)
    assert out == "sample,audio_seconds,transcribe_seconds,rtf"


def test_render_csv_rows_and_summary():
    samples, profile = _profile((10.0, 1.2), (5.0, 0.8))
    out = gv.render_stt_rtf_csv(samples, profile)
    lines = out.splitlines()
    assert lines[0] == "sample,audio_seconds,transcribe_seconds,rtf"
    assert lines[1] == "1,10.0,1.2,0.12"
    assert lines[2] == "2,5.0,0.8,0.16"
    assert "# median_rtf: 0.14" in out
    assert "# speed_grade: fast" in out
    assert "# speed_margin: 0.36" in out


def test_render_csv_slow_has_blank_margin():
    samples, profile = _profile((5.0, 8.0))
    out = gv.render_stt_rtf_csv(samples, profile)
    # slow grade -> speed_margin None -> rendered as empty after the colon.
    assert "# speed_grade: slow" in out
    assert "# speed_margin: " in out
    # the margin line carries no value (nothing after the trailing space).
    margin_line = next(
        ln for ln in out.splitlines() if ln.startswith("# speed_margin:")
    )
    assert margin_line == "# speed_margin: "


# ---- render_stt_rtf_json -----------------------------------------------


def test_render_json_none_profile():
    payload = json.loads(gv.render_stt_rtf_json([], None))
    assert payload == {"samples": [], "profile": None}


def test_render_json_full():
    samples, profile = _profile((10.0, 1.2), (5.0, 0.8))
    payload = json.loads(gv.render_stt_rtf_json(samples, profile))
    assert payload["samples"] == [
        {"sample": 1, "audio_seconds": 10.0, "transcribe_seconds": 1.2, "rtf": 0.12},
        {"sample": 2, "audio_seconds": 5.0, "transcribe_seconds": 0.8, "rtf": 0.16},
    ]
    prof = payload["profile"]
    assert prof["median_rtf"] == 0.14
    assert prof["n_samples"] == 2
    assert prof["min_rtf"] == 0.12
    assert prof["max_rtf"] == 0.16
    assert prof["spread"] == 0.04
    assert prof["relative_spread"] == 0.286
    assert prof["speed_grade"] == "fast"
    assert prof["speed_margin"] == 0.36


def test_render_json_slow_margin_null():
    samples, profile = _profile((5.0, 8.0))
    payload = json.loads(gv.render_stt_rtf_json(samples, profile))
    assert payload["profile"]["speed_grade"] == "slow"
    assert payload["profile"]["speed_margin"] is None


# ---- cmd_stt_rtf: handler dispatch (injected log) ----------------------


def _run(argv):
    args = gv.build_parser().parse_args(argv)
    out = []
    gv.cmd_stt_rtf(args, log=out.append)
    return out


def test_handler_human_default():
    out = _run(["stt-rtf", "--samples", "10.0:1.2", "5.0:0.8"])
    text = "\n".join(out)
    assert "STT real-time-factor profile" in text
    assert "median RTF:       0.140" in text
    assert "speed:            fast" in text


def test_handler_json():
    out = _run(["stt-rtf", "--samples", "10.0:1.2", "5.0:0.8", "--json"])
    assert len(out) == 1
    payload = json.loads(out[0])
    assert payload["profile"]["speed_grade"] == "fast"
    assert len(payload["samples"]) == 2


def test_handler_csv():
    out = _run(["stt-rtf", "--samples", "10.0:1.2", "5.0:0.8", "--csv"])
    assert len(out) == 1
    assert out[0].splitlines()[0] == "sample,audio_seconds,transcribe_seconds,rtf"
    assert "# median_rtf: 0.14" in out[0]


def test_handler_single_sample():
    out = _run(["stt-rtf", "--samples", "10.0:1.2"])
    text = "\n".join(out)
    assert "samples:          1" in text
    assert "median RTF:       0.120" in text


# ---- loader -------------------------------------------------------------


def test_load_stt_rtf_profile_is_cached():
    a = gv._load_stt_rtf_profile()
    b = gv._load_stt_rtf_profile()
    assert a is b
    assert hasattr(a, "TranscriptionSample")
    assert hasattr(a, "profile_stt_rtf")


# ---- iter-408: --verdict parser wiring ---------------------------------


def test_verdict_flag_defaults_off():
    args = gv.build_parser().parse_args(["stt-rtf", "--samples", "10.0:1.2"])
    assert args.verdict is False


def test_verdict_flag_sets_true():
    args = gv.build_parser().parse_args(
        ["stt-rtf", "--samples", "10.0:1.2", "--verdict"]
    )
    assert args.verdict is True


def test_verdict_gate_defaults_match_core():
    mod = gv._load_stt_rtf_profile()
    args = gv.build_parser().parse_args(["stt-rtf", "--samples", "10.0:1.2"])
    assert args.rel_spread_max == mod.DEFAULT_STT_RTF_REL_SPREAD_MAX
    assert args.min_samples == mod.DEFAULT_STT_RTF_MIN_SAMPLES


def test_verdict_gate_overrides_parse():
    args = gv.build_parser().parse_args(
        [
            "stt-rtf",
            "--samples",
            "10.0:1.2",
            "--rel-spread-max",
            "0.3",
            "--min-samples",
            "5",
        ]
    )
    assert args.rel_spread_max == 0.3
    assert args.min_samples == 5


# ---- iter-408: render_stt_rtf_verdict ----------------------------------


def _verdict(*pairs, **kw):
    """Build an SttRtfVerdict from (audio, transcribe) pairs via the core."""
    mod = gv._load_stt_rtf_profile()
    _, profile = _profile(*pairs)
    return mod.stt_rtf_verdict(profile, **kw)


def test_render_verdict_none_is_no_samples():
    lines = gv.render_stt_rtf_verdict(None)
    assert lines == ["stt-rtf verdict: no samples (nothing to decide)"]


def test_render_verdict_recommend_lightens():
    # slow + agree + enough samples -> recommend lightening the STT path.
    verdict = _verdict((5.0, 10.0), (5.0, 10.2), (5.0, 9.8))
    assert verdict.recommend
    text = "\n".join(gv.render_stt_rtf_verdict(verdict))
    assert "STT real-time-factor verdict" in text
    assert "decision: lighten the STT path" in text
    assert "reason:" in text
    assert "grade==slow" in text
    assert "median RTF 2.000 grades slow" in text
    assert "a reading aid, not a gate" in text


def test_render_verdict_keep_when_fast():
    # tight + fast: passes the sample and trust gates, fails significance.
    verdict = _verdict((10.0, 1.40), (10.0, 1.45), (10.0, 1.50))
    assert not verdict.recommend
    text = "\n".join(gv.render_stt_rtf_verdict(verdict))
    assert "decision: keep the current engine" in text
    assert "keeps pace" in text


def test_render_verdict_echoes_gate_thresholds():
    verdict = _verdict(
        (5.0, 10.0), (5.0, 10.2), (5.0, 9.8), rel_spread_max=0.2, min_samples=2
    )
    text = "\n".join(gv.render_stt_rtf_verdict(verdict))
    assert "relative_spread<=0.20" in text
    assert "samples>=2" in text


# ---- iter-408: cmd_stt_rtf --verdict dispatch --------------------------


def test_handler_verdict_appends_decision():
    out = _run(["stt-rtf", "--samples", "5.0:10.0", "5.0:10.2", "5.0:9.8", "--verdict"])
    text = "\n".join(out)
    # the human profile report still leads...
    assert "STT real-time-factor profile" in text
    # ...and the verdict block follows.
    assert "STT real-time-factor verdict" in text
    assert "decision: lighten the STT path" in text


def test_handler_no_verdict_by_default():
    out = _run(["stt-rtf", "--samples", "5.0:10.0", "5.0:10.2", "5.0:9.8"])
    text = "\n".join(out)
    assert "STT real-time-factor verdict" not in text


def test_handler_verdict_keep_when_too_few_samples():
    out = _run(["stt-rtf", "--samples", "5.0:10.0", "--verdict"])
    text = "\n".join(out)
    assert "decision: keep the current engine" in text
    assert "need 3+" in text


def test_handler_verdict_suppressed_under_json():
    out = _run(
        ["stt-rtf", "--samples", "5.0:10.0", "5.0:10.2", "5.0:9.8", "--verdict", "--json"]
    )
    assert len(out) == 1
    payload = json.loads(out[0])
    assert payload["profile"]["speed_grade"] == "slow"
    assert "verdict" not in out[0]


def test_handler_verdict_suppressed_under_csv():
    out = _run(
        ["stt-rtf", "--samples", "5.0:10.0", "5.0:10.2", "5.0:9.8", "--verdict", "--csv"]
    )
    assert len(out) == 1
    assert out[0].splitlines()[0] == "sample,audio_seconds,transcribe_seconds,rtf"
    assert "decision:" not in out[0]


def test_handler_verdict_honors_gate_overrides():
    # rel-spread-max 0.001 is tight enough to fail the trust gate on agreeing
    # runs, flipping the recommendation to keep.
    out = _run(
        [
            "stt-rtf",
            "--samples",
            "5.0:10.0",
            "5.0:10.2",
            "5.0:9.8",
            "--verdict",
            "--rel-spread-max",
            "0.001",
        ]
    )
    text = "\n".join(out)
    assert "decision: keep the current engine" in text
    assert "runs disagree" in text
