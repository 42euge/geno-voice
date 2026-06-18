"""Tests for iter-233 — the ``gv vad`` subcommand (examples/gv.py).

iter-231 shipped the Silero batch segmenter (``vad/silero.py``), reachable
only via the :5111 HTTP endpoint (``POST /vad/silero``) and the
``fixtures/replay_silero.py`` script. iter-233 brings it to the gv CLI:
``gv vad recording.wav`` segments any 16-bit PCM WAV into speech regions
offline — no server, no mic — the headless analogue of the live mic path.

These tests exercise the new parser arg-type validators, the pure
``render_vad_segments`` helper, and the ``cmd_vad`` handler. The handler takes
injected ``segmenter`` / ``availability`` / ``log`` dependencies (mirroring
``dispatch``'s handler injection), so every test runs WITHOUT importing torch /
silero-vad and without touching real audio — fast and deterministic on the
x86_64 Linux runner.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples import gv  # noqa: E402


# ---- lightweight stand-ins for the SileroResult / SpeechSegment shapes ----
# We don't import vad.silero (it pulls torch); these mirror just the attributes
# render_vad_segments / cmd_vad read.


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


# ---- parser: registration & defaults -----------------------------------


def test_vad_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad"] is gv.cmd_vad


def test_vad_defaults_mirror_silero_params():
    args = gv.build_parser().parse_args(["vad", "rec.wav"])
    assert args.command == "vad"
    assert args.wav == "rec.wav"
    # Defaults track SileroParams (iter-231) — the live pipecat stop_secs=0.8.
    assert args.threshold == 0.5
    assert args.min_speech_ms == 250.0
    assert args.min_silence_ms == 800.0
    assert args.speech_pad_ms == 30.0
    assert args.max_speech_s == float("inf")


def test_vad_requires_wav_positional():
    # The WAV path is required — argparse exits 2 when it is missing.
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(["vad"])
    assert exc.value.code == 2


def test_vad_overrides_parse():
    args = gv.build_parser().parse_args(
        [
            "vad",
            "clip.wav",
            "--threshold",
            "0.7",
            "--min-speech-ms",
            "100",
            "--min-silence-ms",
            "500",
            "--speech-pad-ms",
            "0",
            "--max-speech-s",
            "12",
        ]
    )
    assert args.threshold == 0.7
    assert args.min_speech_ms == 100.0
    assert args.min_silence_ms == 500.0
    assert args.speech_pad_ms == 0.0
    assert args.max_speech_s == 12.0


# ---- unit_interval_type: the --threshold validator ---------------------


@pytest.mark.parametrize("raw", ["0", "0.5", "1", "0.999"])
def test_unit_interval_accepts_in_range(raw):
    value = gv.unit_interval_type(raw)
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("raw", ["-0.1", "1.1", "2", "100"])
def test_unit_interval_rejects_out_of_range(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.unit_interval_type(raw)


@pytest.mark.parametrize("raw", ["high", "", "0.5x"])
def test_unit_interval_rejects_non_numbers(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.unit_interval_type(raw)


def test_unit_interval_rejects_nan():
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        gv.unit_interval_type("nan")
    assert "nan" in str(exc.value)


def test_parser_rejects_out_of_range_threshold_via_systemexit():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(["vad", "x.wav", "--threshold", "1.5"])
    assert exc.value.code == 2


# ---- nonneg_float_type: the millisecond knobs --------------------------


@pytest.mark.parametrize("raw", ["0", "0.0", "250", "800.5"])
def test_nonneg_float_accepts_zero_and_positive(raw):
    value = gv.nonneg_float_type(raw)
    assert isinstance(value, float)
    assert value >= 0


@pytest.mark.parametrize("raw", ["-1", "-0.5"])
def test_nonneg_float_rejects_negative(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.nonneg_float_type(raw)


@pytest.mark.parametrize("raw", ["lots", "", "12ms"])
def test_nonneg_float_rejects_non_numbers(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.nonneg_float_type(raw)


def test_nonneg_float_rejects_nan():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.nonneg_float_type("nan")


# ---- max_speech_type: the force-split bound ----------------------------


@pytest.mark.parametrize("raw", ["inf", "none", "off", "NONE", "Off"])
def test_max_speech_sentinels_mean_infinity(raw):
    assert gv.max_speech_type(raw) == float("inf")


def test_max_speech_accepts_positive_float():
    assert gv.max_speech_type("12.5") == 12.5


@pytest.mark.parametrize("raw", ["0", "-5"])
def test_max_speech_rejects_zero_and_negative(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.max_speech_type(raw)


def test_max_speech_rejects_non_number():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.max_speech_type("forever")


def test_max_speech_rejects_nan():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.max_speech_type("nan")


# ---- render_vad_segments: pure presentation ----------------------------


def test_render_none_is_install_hint():
    lines = gv.render_vad_segments(None)
    assert len(lines) == 1
    assert "silero-vad" in lines[0]


def test_render_empty_segments_notes_no_speech():
    result = _Result(name="quiet.wav", sample_rate=16000, duration_s=4.0)
    lines = gv.render_vad_segments(result, threshold=0.5)
    text = "\n".join(lines)
    assert "quiet.wav" in text
    assert "segments:     0" in text
    assert "no speech regions detected" in text


def test_render_lists_each_segment():
    result = _Result(
        name="rec.wav",
        sample_rate=48000,
        duration_s=31.3,
        segments=[_Seg(1.6, 2.1), _Seg(10.7, 18.5)],
    )
    lines = gv.render_vad_segments(result, threshold=0.5)
    text = "\n".join(lines)
    assert "rec.wav" in text
    assert "48000 Hz" in text
    assert "segments:     2" in text
    # speech total = 0.5 + 7.8 = 8.3s
    assert "8.3s" in text
    # both regions rendered with their durations
    assert "[ 1]" in text and "[ 2]" in text
    assert "1.60s" in text and "18.50s" in text


def test_render_omits_threshold_line_when_none():
    result = _Result(name="r.wav", sample_rate=16000, duration_s=1.0)
    lines = gv.render_vad_segments(result)  # no threshold kwarg
    assert not any("threshold" in ln for ln in lines)


# ---- cmd_vad: handler with injected deps -------------------------------


def _args(**over):
    base = dict(
        wav="rec.wav",
        threshold=0.5,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        json=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_cmd_vad_unavailable_prints_install_hint():
    lines: List[str] = []
    called = {"segmenter": False}

    def seg(*a, **k):  # should NOT be called when unavailable
        called["segmenter"] = True
        raise AssertionError("segmenter must not run when silero unavailable")

    gv.cmd_vad(
        _args(),
        log=lines.append,
        segmenter=seg,
        availability=lambda: False,
    )
    assert called["segmenter"] is False
    assert len(lines) == 1
    assert "silero-vad" in lines[0]


def test_cmd_vad_segments_and_renders():
    lines: List[str] = []
    captured = {}

    def seg(wav, params=None):
        captured["wav"] = wav
        captured["params"] = params
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=5.0,
            segments=[_Seg(0.5, 1.5), _Seg(2.0, 4.0)],
        )

    gv.cmd_vad(
        _args(threshold=0.6, min_silence_ms=500.0),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    # The WAV path was forwarded to the segmenter.
    assert captured["wav"] == "rec.wav"
    # SileroParams carried the CLI knobs through.
    params = captured["params"]
    assert params.threshold == 0.6
    assert params.min_silence_ms == 500.0
    # The rendered report names the file and both segments.
    text = "\n".join(lines)
    assert "rec.wav" in text
    assert "[ 1]" in text and "[ 2]" in text
    assert "threshold:    0.60" in text


def test_cmd_vad_builds_real_silero_params():
    # The params object the handler builds must be a genuine vad.silero
    # SileroParams (not a duck) so the field names stay in lock-step with the
    # engine. This is the one test that imports the engine; skip if absent.
    silero = pytest.importorskip("vad.silero")
    captured = {}

    def seg(wav, params=None):
        captured["params"] = params
        return _Result(name="r.wav", sample_rate=16000, duration_s=1.0)

    gv.cmd_vad(
        _args(threshold=0.4, min_speech_ms=120.0, max_speech_s=20.0),
        log=lambda *_: None,
        segmenter=seg,
        availability=lambda: True,
    )
    params = captured["params"]
    assert isinstance(params, silero.SileroParams)
    assert params.threshold == 0.4
    assert params.min_speech_ms == 120.0
    assert params.max_speech_s == 20.0


# ---- parser: the --json flag -------------------------------------------


def test_vad_json_defaults_false():
    args = gv.build_parser().parse_args(["vad", "rec.wav"])
    assert args.json is False


def test_vad_json_flag_sets_true():
    args = gv.build_parser().parse_args(["vad", "rec.wav", "--json"])
    assert args.json is True


# ---- render_vad_json: pure machine-readable presentation ---------------


def test_render_json_none_marks_unavailable():
    payload = json.loads(gv.render_vad_json(None))
    assert payload["available"] is False
    assert "silero-vad" in payload["hint"]
    # No segmentation keys leak onto the degraded payload.
    assert "segments" not in payload


def test_render_json_empty_segments():
    result = _Result(name="quiet.wav", sample_rate=16000, duration_s=4.0)
    payload = json.loads(gv.render_vad_json(result, threshold=0.5))
    assert payload["available"] is True
    assert payload["name"] == "quiet.wav"
    assert payload["sample_rate"] == 16000
    assert payload["num_segments"] == 0
    assert payload["speech_s"] == 0.0
    assert payload["segments"] == []
    assert payload["threshold"] == 0.5


def test_render_json_lists_each_segment():
    result = _Result(
        name="rec.wav",
        sample_rate=48000,
        duration_s=31.3,
        segments=[_Seg(1.6, 2.1), _Seg(10.7, 18.5)],
    )
    payload = json.loads(gv.render_vad_json(result, threshold=0.6))
    assert payload["num_segments"] == 2
    # speech total = 0.5 + 7.8 = 8.3s, rounded to 3 places
    assert payload["speech_s"] == 8.3
    segs = payload["segments"]
    assert len(segs) == 2
    assert segs[0] == {"start_s": 1.6, "end_s": 2.1, "duration_s": 0.5}
    assert segs[1]["start_s"] == 10.7 and segs[1]["end_s"] == 18.5
    assert payload["threshold"] == 0.6


def test_render_json_omits_threshold_when_none():
    result = _Result(name="r.wav", sample_rate=16000, duration_s=1.0)
    payload = json.loads(gv.render_vad_json(result))  # no threshold kwarg
    assert "threshold" not in payload


def test_render_json_rounds_to_three_places():
    # Sub-millisecond Silero boundaries must round to 3 places like to_dict().
    result = _Result(
        name="r.wav",
        sample_rate=16000,
        duration_s=1.23456,
        segments=[_Seg(0.123456, 0.987654)],
    )
    payload = json.loads(gv.render_vad_json(result))
    assert payload["duration_s"] == 1.235
    assert payload["segments"][0]["start_s"] == 0.123
    assert payload["segments"][0]["end_s"] == 0.988
    assert payload["segments"][0]["duration_s"] == 0.864


def test_render_json_matches_silero_to_dict_shape():
    # The hand-built payload must carry the same segmentation keys as the
    # engine's SileroResult.to_dict() so consumers can treat gv output and
    # the replay/server output interchangeably. Skip if the engine is absent.
    silero = pytest.importorskip("vad.silero")
    result = silero.SileroResult(
        name="rec.wav",
        sample_rate=16000,
        duration_s=5.0,
        segments=[silero.SpeechSegment(0.5, 1.5)],
    )
    via_to_dict = result.to_dict()
    via_render = json.loads(gv.render_vad_json(result))
    # Every key to_dict() emits is present in the render with equal values.
    for key, value in via_to_dict.items():
        assert via_render[key] == value


# ---- cmd_vad: the --json branch ----------------------------------------


def test_cmd_vad_json_unavailable_emits_unavailable_payload():
    lines: List[str] = []
    gv.cmd_vad(
        _args(json=True),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not segment when unavailable")
        ),
        availability=lambda: False,
    )
    # One JSON document, parseable, marking the degraded path.
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["available"] is False


def test_cmd_vad_json_emits_segmentation_payload():
    lines: List[str] = []

    def seg(wav, params=None):
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=5.0,
            segments=[_Seg(0.5, 1.5), _Seg(2.0, 4.0)],
        )

    gv.cmd_vad(
        _args(json=True, threshold=0.6),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["available"] is True
    assert payload["num_segments"] == 2
    assert payload["threshold"] == 0.6
    assert payload["segments"][0]["start_s"] == 0.5


def test_cmd_vad_without_json_stays_human_readable():
    # Regression guard: omitting --json keeps the multi-line text report, not
    # a single JSON blob.
    lines: List[str] = []

    def seg(wav, params=None):
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=5.0,
            segments=[_Seg(0.5, 1.5)],
        )

    gv.cmd_vad(_args(json=False), log=lines.append, segmenter=seg, availability=lambda: True)
    # Multiple human-readable lines, and the first is NOT valid JSON.
    assert len(lines) > 1
    with pytest.raises(json.JSONDecodeError):
        json.loads(lines[0])


# ====================================================================
# iter-235 — gv vad-diff: compare two thresholds (first gv vad --json consumer)
# ====================================================================


def _diff_args(**over):
    base = dict(
        wav="rec.wav",
        threshold_a=0.5,
        threshold_b=0.7,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        json=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


# ---- parser: registration & defaults -----------------------------------


def test_vad_diff_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-diff"] is gv.cmd_vad_diff


def test_vad_diff_defaults():
    args = gv.build_parser().parse_args(["vad-diff", "rec.wav"])
    assert args.command == "vad-diff"
    assert args.wav == "rec.wav"
    assert args.threshold_a == 0.5
    assert args.threshold_b == 0.7
    # Shared knobs default to the same SileroParams values as `gv vad`.
    assert args.min_speech_ms == 250.0
    assert args.min_silence_ms == 800.0
    assert args.speech_pad_ms == 30.0
    assert args.json is False


def test_vad_diff_overrides_thresholds():
    args = gv.build_parser().parse_args(
        ["vad-diff", "rec.wav", "--threshold-a", "0.3", "--threshold-b", "0.9"]
    )
    assert args.threshold_a == 0.3
    assert args.threshold_b == 0.9


def test_vad_diff_rejects_out_of_range_threshold():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-diff", "rec.wav", "--threshold-a", "1.5"])
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-diff", "rec.wav", "--threshold-b", "-0.1"])


def test_vad_diff_json_flag():
    args = gv.build_parser().parse_args(["vad-diff", "rec.wav", "--json"])
    assert args.json is True


# ---- vad_segmentation_delta: pure delta core ---------------------------


def test_delta_fewer_segments_at_higher_threshold():
    # A higher gate is typically a subset: fewer regions, less speech.
    a = _Result(
        name="r.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0), _Seg(5.0, 6.0)],
    )
    b = _Result(
        name="r.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0)],
    )
    d = gv.vad_segmentation_delta(a, b)
    assert d["num_segments_a"] == 3
    assert d["num_segments_b"] == 1
    assert d["num_segments_delta"] == -2
    assert d["speech_s_a"] == 3.0
    assert d["speech_s_b"] == 1.0
    assert d["speech_s_delta"] == -2.0


def test_delta_identical_segmentations_are_zero():
    segs = [_Seg(0.0, 1.0), _Seg(2.0, 3.5)]
    a = _Result(name="r.wav", sample_rate=16000, duration_s=5.0, segments=list(segs))
    b = _Result(name="r.wav", sample_rate=16000, duration_s=5.0, segments=list(segs))
    d = gv.vad_segmentation_delta(a, b)
    assert d["num_segments_delta"] == 0
    assert d["speech_s_delta"] == 0.0


def test_delta_positive_when_b_has_more():
    a = _Result(name="r.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    b = _Result(
        name="r.wav",
        sample_rate=16000,
        duration_s=5.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    d = gv.vad_segmentation_delta(a, b)
    assert d["num_segments_delta"] == 1
    assert d["speech_s_delta"] == 1.0


def test_delta_rounds_to_three_places():
    a = _Result(
        name="r.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 0.123456)]
    )
    b = _Result(
        name="r.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 0.987654)]
    )
    d = gv.vad_segmentation_delta(a, b)
    assert d["speech_s_a"] == 0.123
    assert d["speech_s_b"] == 0.988
    assert d["speech_s_delta"] == 0.865


# ---- render_vad_diff: human-readable -----------------------------------


def test_render_diff_none_marks_unavailable():
    lines = gv.render_vad_diff(None, None, label_a=0.5, label_b=0.7)
    assert len(lines) == 1
    assert "silero-vad" in lines[0]


def test_render_diff_one_none_marks_unavailable():
    r = _Result(name="r.wav", sample_rate=16000, duration_s=5.0)
    assert "silero-vad" in gv.render_vad_diff(r, None, label_a=0.5, label_b=0.7)[0]


def test_render_diff_shows_signed_deltas():
    a = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0), _Seg(5.0, 6.0)],
    )
    b = _Result(name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)])
    text = "\n".join(gv.render_vad_diff(a, b, label_a=0.5, label_b=0.7))
    assert "rec.wav" in text
    assert "0.50" in text and "0.70" in text
    assert "3 → 1" in text
    assert "(-2)" in text
    assert "(-2.0s)" in text


def test_render_diff_positive_delta_carries_plus_sign():
    a = _Result(name="rec.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    b = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=5.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    text = "\n".join(gv.render_vad_diff(a, b, label_a=0.3, label_b=0.5))
    assert "1 → 2" in text
    assert "(+1)" in text
    assert "(+1.0s)" in text


# ---- render_vad_diff_json: machine-readable ----------------------------


def test_render_diff_json_none_marks_unavailable():
    payload = json.loads(gv.render_vad_diff_json(None, None, label_a=0.5, label_b=0.7))
    assert payload["available"] is False
    assert "silero-vad" in payload["hint"]
    assert "num_segments_delta" not in payload


def test_render_diff_json_carries_both_sides_and_deltas():
    a = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0), _Seg(5.0, 6.0)],
    )
    b = _Result(name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)])
    payload = json.loads(gv.render_vad_diff_json(a, b, label_a=0.5, label_b=0.7))
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["threshold_a"] == 0.5
    assert payload["threshold_b"] == 0.7
    assert payload["num_segments_a"] == 3
    assert payload["num_segments_b"] == 1
    assert payload["num_segments_delta"] == -2
    assert payload["speech_s_delta"] == -2.0


# ---- cmd_vad_diff: the handler -----------------------------------------


def test_cmd_vad_diff_unavailable_emits_hint():
    lines: List[str] = []
    gv.cmd_vad_diff(
        _diff_args(),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not segment when unavailable")
        ),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert "silero-vad" in lines[0]


def test_cmd_vad_diff_unavailable_json():
    lines: List[str] = []
    gv.cmd_vad_diff(
        _diff_args(json=True),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert json.loads(lines[0])["available"] is False


def test_cmd_vad_diff_runs_both_thresholds():
    # The handler must segment twice — once per threshold — forwarding the
    # shared knobs both times. We capture the threshold of each call.
    seen = []

    def seg(wav, params=None):
        seen.append(params.threshold)
        # Higher threshold → fewer segments (subset behaviour).
        n = 3 if params.threshold < 0.6 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_diff(
        _diff_args(threshold_a=0.5, threshold_b=0.7),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert seen == [0.5, 0.7]  # A first, then B
    text = "\n".join(lines)
    assert "3 → 1" in text
    assert "(-2)" in text


def test_cmd_vad_diff_json_branch():
    def seg(wav, params=None):
        n = 3 if params.threshold < 0.6 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_diff(
        _diff_args(threshold_a=0.5, threshold_b=0.7, json=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["num_segments_delta"] == -2
    assert payload["threshold_a"] == 0.5
    assert payload["threshold_b"] == 0.7


def test_cmd_vad_diff_shares_knobs_across_both_runs():
    # Both runs must carry the SAME min_speech_ms / max_speech_s — only the
    # threshold differs. Build genuine SileroParams to lock field names.
    pytest.importorskip("vad.silero")
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        return _Result(name="rec.wav", sample_rate=16000, duration_s=1.0)

    gv.cmd_vad_diff(
        _diff_args(threshold_a=0.4, threshold_b=0.8, min_speech_ms=120.0, max_speech_s=20.0),
        log=lambda *_: None,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(captured) == 2
    assert [p.threshold for p in captured] == [0.4, 0.8]
    # Shared knobs identical across both runs.
    assert captured[0].min_speech_ms == captured[1].min_speech_ms == 120.0
    assert captured[0].max_speech_s == captured[1].max_speech_s == 20.0


# ====================================================================
# iter-236 — gv vad-sweep: tabulate segmentation over N thresholds
# ====================================================================


def _sweep_args(**over):
    base = dict(
        wav="rec.wav",
        thresholds=[0.3, 0.5, 0.7, 0.9],
        min_silences=None,
        min_speeches=None,
        threshold=0.5,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        json=False,
        csv=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


# ---- unit_interval_list_type: the --thresholds validator ---------------


def test_threshold_list_parses_comma_separated():
    values = gv.unit_interval_list_type("0.3,0.5,0.7,0.9")
    assert values == [0.3, 0.5, 0.7, 0.9]
    assert all(isinstance(v, float) for v in values)


def test_threshold_list_strips_whitespace_and_blanks():
    assert gv.unit_interval_list_type(" 0.2 , 0.8 ,") == [0.2, 0.8]


def test_threshold_list_preserves_order_and_duplicates():
    # The operator picks the column order; we don't sort or dedupe.
    assert gv.unit_interval_list_type("0.9,0.1,0.9") == [0.9, 0.1, 0.9]


@pytest.mark.parametrize("raw", ["", "  ", ",", " , "])
def test_threshold_list_rejects_empty(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.unit_interval_list_type(raw)


@pytest.mark.parametrize("raw", ["0.3,1.5", "0.3,-0.1", "2"])
def test_threshold_list_rejects_out_of_range_member(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.unit_interval_list_type(raw)


@pytest.mark.parametrize("raw", ["0.3,high", "x,0.5", "0.3,nan"])
def test_threshold_list_rejects_non_number_member(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.unit_interval_list_type(raw)


def test_threshold_list_rejects_non_string():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.unit_interval_list_type(0.5)


# ---- parser: registration & defaults -----------------------------------


def test_vad_sweep_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-sweep"] is gv.cmd_vad_sweep


def test_vad_sweep_defaults():
    args = gv.build_parser().parse_args(["vad-sweep", "rec.wav"])
    assert args.command == "vad-sweep"
    assert args.wav == "rec.wav"
    assert args.thresholds == [0.3, 0.5, 0.7, 0.9]
    # Shared knobs default to the same SileroParams values as `gv vad`.
    assert args.min_speech_ms == 250.0
    assert args.min_silence_ms == 800.0
    assert args.speech_pad_ms == 30.0
    assert args.max_speech_s == float("inf")
    assert args.json is False
    assert args.csv is False


def test_vad_sweep_overrides_thresholds():
    args = gv.build_parser().parse_args(
        ["vad-sweep", "rec.wav", "--thresholds", "0.1,0.6,0.95"]
    )
    assert args.thresholds == [0.1, 0.6, 0.95]


def test_vad_sweep_rejects_out_of_range_threshold():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-sweep", "rec.wav", "--thresholds", "0.5,1.5"])


def test_vad_sweep_json_flag():
    args = gv.build_parser().parse_args(["vad-sweep", "rec.wav", "--json"])
    assert args.json is True


def test_vad_sweep_csv_flag():
    args = gv.build_parser().parse_args(["vad-sweep", "rec.wav", "--csv"])
    assert args.csv is True
    assert args.json is False


def test_vad_sweep_json_and_csv_are_mutually_exclusive():
    # Two output formats can't both win; argparse rejects the combination with
    # the usual SystemExit(2) rather than silently picking one.
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-sweep", "rec.wav", "--json", "--csv"])


# ---- vad_segmentation_sweep: pure core ---------------------------------


def test_sweep_pairs_each_threshold_with_summary():
    r_lo = _Result(
        name="r.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0), _Seg(5.0, 6.0)],
    )
    r_hi = _Result(
        name="r.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    rows = gv.vad_segmentation_sweep([0.3, 0.9], [r_lo, r_hi])
    assert rows == [
        {"threshold": 0.3, "num_segments": 3, "speech_s": 3.0},
        {"threshold": 0.9, "num_segments": 1, "speech_s": 1.0},
    ]


def test_sweep_rounds_speech_to_three_places():
    r = _Result(
        name="r.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 0.123456)]
    )
    rows = gv.vad_segmentation_sweep([0.5], [r])
    assert rows[0]["speech_s"] == 0.123


def test_sweep_length_mismatch_raises():
    r = _Result(name="r.wav", sample_rate=16000, duration_s=5.0)
    with pytest.raises(ValueError):
        gv.vad_segmentation_sweep([0.3, 0.5], [r])


# ---- render_vad_sweep: human-readable ----------------------------------


def test_render_sweep_none_marks_unavailable():
    lines = gv.render_vad_sweep([], [None], name="rec.wav")
    assert len(lines) == 1
    assert "silero-vad" in lines[0]


def test_render_sweep_any_none_marks_unavailable():
    r = _Result(name="r.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    lines = gv.render_vad_sweep([0.3, 0.9], [r, None], name="rec.wav")
    assert "silero-vad" in lines[0]


def test_render_sweep_tabulates_each_threshold():
    r_lo = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0), _Seg(5.0, 6.0)],
    )
    r_hi = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    lines = gv.render_vad_sweep([0.3, 0.9], [r_lo, r_hi], name="rec.wav")
    text = "\n".join(lines)
    assert "rec.wav" in text
    assert "threshold" in text and "segments" in text and "speech" in text
    # one header line, one column-label line, one row per threshold
    assert len(lines) == 4
    assert "0.30" in text and "0.90" in text
    assert "3" in text and "1" in text


# ---- render_vad_sweep_json: machine-readable ---------------------------


def test_render_sweep_json_none_marks_unavailable():
    payload = json.loads(gv.render_vad_sweep_json([], [None], name="rec.wav"))
    assert payload["available"] is False
    assert "silero-vad" in payload["hint"]
    assert "sweep" not in payload


def test_render_sweep_json_carries_rows():
    r_lo = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0), _Seg(5.0, 6.0)],
    )
    r_hi = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    payload = json.loads(gv.render_vad_sweep_json([0.3, 0.9], [r_lo, r_hi], name="rec.wav"))
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["sweep"] == [
        {"threshold": 0.3, "num_segments": 3, "speech_s": 3.0},
        {"threshold": 0.9, "num_segments": 1, "speech_s": 1.0},
    ]


# ---- cmd_vad_sweep: the handler ----------------------------------------


def test_cmd_vad_sweep_unavailable_emits_hint():
    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not segment when unavailable")
        ),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert "silero-vad" in lines[0]


def test_cmd_vad_sweep_unavailable_json():
    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(json=True),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert json.loads(lines[0])["available"] is False


def test_cmd_vad_sweep_runs_every_threshold_in_order():
    seen = []

    def seg(wav, params=None):
        seen.append(params.threshold)
        # Higher threshold → fewer segments (subset behaviour).
        n = 3 if params.threshold < 0.6 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.2, 0.5, 0.8]),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert seen == [0.2, 0.5, 0.8]  # swept in order
    text = "\n".join(lines)
    assert "0.20" in text and "0.50" in text and "0.80" in text


def test_cmd_vad_sweep_json_branch():
    def seg(wav, params=None):
        n = 3 if params.threshold < 0.6 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.3, 0.9], json=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert [row["threshold"] for row in payload["sweep"]] == [0.3, 0.9]
    assert payload["sweep"][0]["num_segments"] == 3
    assert payload["sweep"][1]["num_segments"] == 1


def test_cmd_vad_sweep_shares_knobs_across_all_runs():
    # Every run must carry the SAME min_speech_ms / max_speech_s — only the
    # threshold differs. Build genuine SileroParams to lock field names.
    pytest.importorskip("vad.silero")
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        return _Result(name="rec.wav", sample_rate=16000, duration_s=1.0)

    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.2, 0.6, 0.95], min_speech_ms=120.0, max_speech_s=20.0),
        log=lambda *_: None,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(captured) == 3
    assert [p.threshold for p in captured] == [0.2, 0.6, 0.95]
    assert {p.min_speech_ms for p in captured} == {120.0}
    assert {p.max_speech_s for p in captured} == {20.0}


# ====================================================================
# iter-237 — gv vad-sweep --csv: flat spreadsheet/plot-friendly table
# ====================================================================


# ---- render_vad_sweep_csv: machine-readable CSV ------------------------


def test_render_sweep_csv_none_marks_unavailable():
    text = gv.render_vad_sweep_csv([], [None], name="rec.wav")
    # A degraded run is a single self-describing comment, not empty output.
    assert text.startswith("#")
    assert "silero-vad" in text


def test_render_sweep_csv_any_none_marks_unavailable():
    r = _Result(name="r.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    text = gv.render_vad_sweep_csv([0.3, 0.9], [r, None], name="rec.wav")
    assert text.startswith("#")
    assert "silero-vad" in text


def test_render_sweep_csv_header_and_rows():
    r_lo = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0), _Seg(5.0, 6.0)],
    )
    r_hi = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    text = gv.render_vad_sweep_csv([0.3, 0.9], [r_lo, r_hi], name="rec.wav")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["threshold", "num_segments", "speech_s"]
    assert rows[1] == ["0.3", "3", "3.0"]
    assert rows[2] == ["0.9", "1", "1.0"]
    # exactly the header + one row per threshold, nothing else.
    assert len(rows) == 3


def test_render_sweep_csv_no_trailing_newline():
    r = _Result(name="rec.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    text = gv.render_vad_sweep_csv([0.5], [r], name="rec.wav")
    # The renderer is pure text the caller logs; it must not carry a trailing
    # blank line (csv.writer's \r\n terminator stripped).
    assert not text.endswith("\n")
    assert not text.endswith("\r")


def test_render_sweep_csv_round_trips_to_sweep_rows():
    # The CSV body must describe the SAME segmentation as the JSON twin, so a
    # consumer reading either surface sees identical numbers.
    r_lo = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    r_hi = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    thresholds = [0.3, 0.9]
    results = [r_lo, r_hi]
    csv_text = gv.render_vad_sweep_csv(thresholds, results, name="rec.wav")
    json_rows = json.loads(
        gv.render_vad_sweep_json(thresholds, results, name="rec.wav")
    )["sweep"]
    csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert [
        {
            "threshold": float(row["threshold"]),
            "num_segments": int(row["num_segments"]),
            "speech_s": float(row["speech_s"]),
        }
        for row in csv_rows
    ] == json_rows


# ---- cmd_vad_sweep --csv: the handler ----------------------------------


def test_cmd_vad_sweep_csv_unavailable_emits_comment():
    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(csv=True),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert lines[0].startswith("#")
    assert "silero-vad" in lines[0]


def test_cmd_vad_sweep_csv_branch():
    def seg(wav, params=None):
        n = 3 if params.threshold < 0.6 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.3, 0.9], csv=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    # One CSV blob logged in a single call (not line-by-line like the table).
    assert len(lines) == 1
    rows = list(csv.reader(io.StringIO(lines[0])))
    assert rows[0] == ["threshold", "num_segments", "speech_s"]
    assert [row[0] for row in rows[1:]] == ["0.3", "0.9"]
    assert rows[1][1] == "3"
    assert rows[2][1] == "1"


def test_cmd_vad_sweep_csv_uses_segmenter_name():
    # CSV body is a pure data grid — the segmenter's basename name does not leak
    # into the rows (signature parity only). Verify the table is exactly
    # header + data, no name column.
    def seg(wav, params=None):
        return _Result(
            name="basename.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(0.0, 1.0)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.5], csv=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert "basename.wav" not in lines[0]
    rows = list(csv.reader(io.StringIO(lines[0])))
    assert len(rows) == 2  # header + single threshold row


# ====================================================================
# iter-238 — gv vad-sweep --min-silences: a second sweep axis (hangover)
# ====================================================================


# ---- nonneg_float_list_type: the --min-silences validator --------------


def test_min_silences_list_parses_comma_separated():
    assert gv.nonneg_float_list_type("400,600,800,1000") == [400.0, 600.0, 800.0, 1000.0]


def test_min_silences_list_strips_whitespace_and_blanks():
    assert gv.nonneg_float_list_type(" 400 , 800 ,") == [400.0, 800.0]


def test_min_silences_list_preserves_order_and_duplicates():
    assert gv.nonneg_float_list_type("800,400,800") == [800.0, 400.0, 800.0]


def test_min_silences_list_allows_zero():
    # 0 ms is legitimate (disable the minimum hangover), unlike thresholds which
    # are bounded to [0, 1] — here only negatives are rejected.
    assert gv.nonneg_float_list_type("0,400") == [0.0, 400.0]


@pytest.mark.parametrize("raw", ["", "  ", ",", " , "])
def test_min_silences_list_rejects_empty(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.nonneg_float_list_type(raw)


@pytest.mark.parametrize("raw", ["400,-1", "-50"])
def test_min_silences_list_rejects_negative_member(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.nonneg_float_list_type(raw)


@pytest.mark.parametrize("raw", ["400,abc", "x"])
def test_min_silences_list_rejects_non_number_member(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.nonneg_float_list_type(raw)


def test_min_silences_list_rejects_non_string():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.nonneg_float_list_type(800.0)


# ---- parser wiring: --min-silences axis --------------------------------


def test_vad_sweep_min_silences_parses():
    args = gv.build_parser().parse_args(
        ["vad-sweep", "rec.wav", "--min-silences", "400,600,800"]
    )
    assert args.min_silences == [400.0, 600.0, 800.0]
    # The threshold list keeps its default; the handler picks the silence axis.
    assert args.thresholds == [0.3, 0.5, 0.7, 0.9]


def test_vad_sweep_min_silences_default_is_none():
    # Without --min-silences the silence axis is off (None), so the handler
    # sweeps --thresholds (the iter-236 default).
    args = gv.build_parser().parse_args(["vad-sweep", "rec.wav"])
    assert args.min_silences is None


def test_vad_sweep_scalar_threshold_default_and_override():
    args = gv.build_parser().parse_args(["vad-sweep", "rec.wav"])
    assert args.threshold == 0.5
    args = gv.build_parser().parse_args(
        ["vad-sweep", "rec.wav", "--min-silences", "400,800", "--threshold", "0.7"]
    )
    assert args.threshold == 0.7


def test_vad_sweep_thresholds_and_min_silences_mutually_exclusive():
    # The two sweep axes can't both win; argparse rejects the combination with
    # SystemExit(2) rather than silently picking one.
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--thresholds", "0.3,0.5", "--min-silences", "400,800"]
        )


def test_vad_sweep_rejects_negative_min_silence_member():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--min-silences", "400,-1"]
        )


# ---- vad_segmentation_sweep: axis parameter ----------------------------


def test_sweep_axis_keys_rows_by_axis_name():
    r_lo = _Result(
        name="r.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    r_hi = _Result(
        name="r.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    rows = gv.vad_segmentation_sweep([400.0, 800.0], [r_lo, r_hi], axis="min_silence_ms")
    assert rows == [
        {"min_silence_ms": 400.0, "num_segments": 2, "speech_s": 2.0},
        {"min_silence_ms": 800.0, "num_segments": 1, "speech_s": 1.0},
    ]


def test_sweep_axis_defaults_to_threshold():
    # Omitting axis keeps the iter-236 row shape so old callers are unchanged.
    r = _Result(name="r.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    rows = gv.vad_segmentation_sweep([0.5], [r])
    assert rows == [{"threshold": 0.5, "num_segments": 1, "speech_s": 1.0}]


# ---- renderers: silence axis -------------------------------------------


def test_render_sweep_silence_axis_labels_column():
    r_lo = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    r_hi = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    lines = gv.render_vad_sweep(
        [400.0, 800.0], [r_lo, r_hi], name="rec.wav", axis="min_silence_ms"
    )
    text = "\n".join(lines)
    assert "min_silence" in text
    assert "threshold" not in text
    # Hangover values print as bare integers (400, 800), not 0.40 gates.
    assert "400" in text and "800" in text
    assert "0.40" not in text


def test_render_sweep_json_carries_axis():
    r = _Result(name="rec.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    payload = json.loads(
        gv.render_vad_sweep_json([400.0], [r], name="rec.wav", axis="min_silence_ms")
    )
    assert payload["axis"] == "min_silence_ms"
    assert payload["sweep"] == [
        {"min_silence_ms": 400.0, "num_segments": 1, "speech_s": 1.0}
    ]


def test_render_sweep_json_axis_defaults_to_threshold():
    r = _Result(name="rec.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    payload = json.loads(gv.render_vad_sweep_json([0.5], [r], name="rec.wav"))
    assert payload["axis"] == "threshold"


def test_render_sweep_csv_header_is_axis_name():
    r = _Result(name="rec.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    text = gv.render_vad_sweep_csv([400.0], [r], name="rec.wav", axis="min_silence_ms")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["min_silence_ms", "num_segments", "speech_s"]
    assert rows[1] == ["400.0", "1", "1.0"]


# ---- cmd_vad_sweep: silence axis end-to-end ----------------------------


def test_cmd_vad_sweep_silence_axis_sweeps_hangover():
    # When --min-silences is set, the segmenter sees the SWEPT min_silence_ms and
    # the gate held at scalar --threshold; --min-silence-ms is then ignored.
    pytest.importorskip("vad.silero")
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        # Longer hangover merges regions → fewer segments.
        n = 3 if params.min_silence_ms < 600 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(min_silences=[400.0, 800.0], threshold=0.7, min_silence_ms=999.0),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert [p.min_silence_ms for p in captured] == [400.0, 800.0]
    # Gate held at scalar --threshold for every run; the shared --min-silence-ms
    # scalar (999) is NOT used as a swept value.
    assert {p.threshold for p in captured} == {0.7}
    text = "\n".join(lines)
    assert "min_silence" in text
    assert "400" in text and "800" in text


def test_cmd_vad_sweep_silence_axis_json_branch():
    def seg(wav, params=None):
        n = 3 if params.min_silence_ms < 600 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(min_silences=[400.0, 800.0], json=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["axis"] == "min_silence_ms"
    assert [row["min_silence_ms"] for row in payload["sweep"]] == [400.0, 800.0]
    assert payload["sweep"][0]["num_segments"] == 3
    assert payload["sweep"][1]["num_segments"] == 1


def test_cmd_vad_sweep_silence_axis_csv_branch():
    def seg(wav, params=None):
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(0.0, 1.0)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(min_silences=[400.0, 800.0], csv=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    rows = list(csv.reader(io.StringIO(lines[0])))
    assert rows[0] == ["min_silence_ms", "num_segments", "speech_s"]
    assert [row[0] for row in rows[1:]] == ["400.0", "800.0"]


def test_cmd_vad_sweep_silence_axis_unavailable_uses_axis_label():
    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(min_silences=[400.0, 800.0], json=True),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert json.loads(lines[0])["available"] is False


# ====================================================================
# iter-239 — gv vad-sweep --min-speeches: a third sweep axis (floor)
# ====================================================================


# ---- parser wiring: --min-speeches axis --------------------------------


def test_vad_sweep_min_speeches_parses():
    args = gv.build_parser().parse_args(
        ["vad-sweep", "rec.wav", "--min-speeches", "50,100,200"]
    )
    assert args.min_speeches == [50.0, 100.0, 200.0]
    # The threshold list keeps its default; the handler picks the speech axis.
    assert args.thresholds == [0.3, 0.5, 0.7, 0.9]


def test_vad_sweep_min_speeches_default_is_none():
    # Without --min-speeches the speech axis is off (None), so the handler
    # sweeps --thresholds (the iter-236 default).
    args = gv.build_parser().parse_args(["vad-sweep", "rec.wav"])
    assert args.min_speeches is None


def test_vad_sweep_min_speeches_allows_zero():
    # 0 ms is legitimate (disable the floor — keep every region).
    args = gv.build_parser().parse_args(
        ["vad-sweep", "rec.wav", "--min-speeches", "0,100"]
    )
    assert args.min_speeches == [0.0, 100.0]


def test_vad_sweep_thresholds_and_min_speeches_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--thresholds", "0.3,0.5", "--min-speeches", "50,100"]
        )


def test_vad_sweep_min_silences_and_min_speeches_mutually_exclusive():
    # The two ms axes are also mutually exclusive — only one knob varies per run.
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--min-silences", "400,800", "--min-speeches", "50,100"]
        )


def test_vad_sweep_rejects_negative_min_speech_member():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--min-speeches", "100,-1"]
        )


# ---- vad_segmentation_sweep / renderers: speech axis -------------------


def test_sweep_axis_keys_rows_by_speech_axis_name():
    r_lo = _Result(
        name="r.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    r_hi = _Result(
        name="r.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    rows = gv.vad_segmentation_sweep([50.0, 400.0], [r_lo, r_hi], axis="min_speech_ms")
    assert rows == [
        {"min_speech_ms": 50.0, "num_segments": 2, "speech_s": 2.0},
        {"min_speech_ms": 400.0, "num_segments": 1, "speech_s": 1.0},
    ]


def test_render_sweep_speech_axis_labels_column():
    r_lo = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    r_hi = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    lines = gv.render_vad_sweep(
        [50.0, 400.0], [r_lo, r_hi], name="rec.wav", axis="min_speech_ms"
    )
    text = "\n".join(lines)
    assert "min_speech" in text
    assert "min_silence" not in text
    # Floor values print as bare integers (50, 400), not 0.50 gates.
    assert "50" in text and "400" in text
    assert "0.50" not in text


def test_render_sweep_json_carries_speech_axis():
    r = _Result(name="rec.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    payload = json.loads(
        gv.render_vad_sweep_json([100.0], [r], name="rec.wav", axis="min_speech_ms")
    )
    assert payload["axis"] == "min_speech_ms"
    assert payload["sweep"] == [
        {"min_speech_ms": 100.0, "num_segments": 1, "speech_s": 1.0}
    ]


def test_render_sweep_csv_header_is_speech_axis_name():
    r = _Result(name="rec.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    text = gv.render_vad_sweep_csv([100.0], [r], name="rec.wav", axis="min_speech_ms")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["min_speech_ms", "num_segments", "speech_s"]
    assert rows[1] == ["100.0", "1", "1.0"]


# ---- cmd_vad_sweep: speech axis end-to-end -----------------------------


def test_cmd_vad_sweep_speech_axis_sweeps_floor():
    # When --min-speeches is set, the segmenter sees the SWEPT min_speech_ms and
    # the gate held at scalar --threshold; the scalar --min-speech-ms is ignored.
    pytest.importorskip("vad.silero")
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        # A higher floor drops more short regions → fewer segments.
        n = 3 if params.min_speech_ms < 200 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(min_speeches=[50.0, 400.0], threshold=0.7, min_speech_ms=999.0),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert [p.min_speech_ms for p in captured] == [50.0, 400.0]
    # Gate held at scalar --threshold for every run; the shared --min-speech-ms
    # scalar (999) is NOT used as a swept value.
    assert {p.threshold for p in captured} == {0.7}
    text = "\n".join(lines)
    assert "min_speech" in text
    assert "50" in text and "400" in text


def test_cmd_vad_sweep_speech_axis_holds_silence_scalar():
    # The non-swept ms knob (--min-silence-ms) is shared across every run.
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        return _Result(
            name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
        )

    gv.cmd_vad_sweep(
        _sweep_args(min_speeches=[50.0, 400.0], min_silence_ms=750.0),
        log=lambda *a: None,
        segmenter=seg,
        availability=lambda: True,
    )
    assert {p.min_silence_ms for p in captured} == {750.0}


def test_cmd_vad_sweep_speech_axis_json_branch():
    def seg(wav, params=None):
        n = 3 if params.min_speech_ms < 200 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(min_speeches=[50.0, 400.0], json=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["axis"] == "min_speech_ms"
    assert [row["min_speech_ms"] for row in payload["sweep"]] == [50.0, 400.0]
    assert payload["sweep"][0]["num_segments"] == 3
    assert payload["sweep"][1]["num_segments"] == 1


def test_cmd_vad_sweep_speech_axis_csv_branch():
    def seg(wav, params=None):
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(0.0, 1.0)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(min_speeches=[50.0, 400.0], csv=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    rows = list(csv.reader(io.StringIO(lines[0])))
    assert rows[0] == ["min_speech_ms", "num_segments", "speech_s"]
    assert [row[0] for row in rows[1:]] == ["50.0", "400.0"]


def test_cmd_vad_sweep_speech_axis_unavailable():
    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(min_speeches=[50.0, 400.0], json=True),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert json.loads(lines[0])["available"] is False
