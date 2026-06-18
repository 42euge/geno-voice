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
