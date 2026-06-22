"""Tests for iter-384 — the ``gv vad-gap-recommend-diff`` subcommand (examples/gv.py).

The recommend-knob family went feature-complete at iter-383 (``--bias`` /
``--min-grade`` / ``--sort-by`` / ``--top-n`` / ``--summary`` across both the 1-D
sweep and 2-D grid). ``gv vad-gap-recommend-diff`` is the genuinely NEW surface
after it: where every prior diff (``gv vad-diff`` / ``gv vad-gap-diff``) compares
two SETTINGS of ONE recording, this compares ONE setting across TWO recordings —
segment ``wav_a`` and ``wav_b`` under the same shared knobs and report how each
WAV's ``vad_gap_recommend`` end-of-turn hangover (and its ``vad_gap_confidence``
grade) differs. The natural "did my tuning change help?" tool: re-record after a
mic / room / speaker change and see whether the recommended ``--min-silence-ms``
still holds.

Like the rest of the VAD-analysis family, the handler takes injected
``segmenter`` / ``availability`` / ``log`` dependencies so every test runs WITHOUT
importing torch / silero-vad and without touching real audio — fast and
deterministic on the x86_64 Linux runner. The pure core
(``vad_gap_recommend_delta``) and the three renderers are exercised directly
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


# A clean bimodal recording (strong grade) and a tighter-valley clone whose
# recommended hangover lands lower. Used across the delta/render tests.
def _clean(name="a.wav"):
    # Short ~0.2-0.3s, long ~1.4-1.6s — wide empty valley between -> strong.
    return _result_from_gaps([0.2, 0.3, 0.25, 1.5, 1.6, 1.4], name=name)


def _shifted(name="b.wav"):
    # Same short cluster, lower long cluster (~0.8-0.9s) -> valley sits lower,
    # so the recommended hangover moves DOWN vs _clean.
    return _result_from_gaps([0.2, 0.3, 0.25, 0.85, 0.9, 0.8], name=name)


def _flat(name="flat.wav"):
    # A single segment — no gaps, nothing to recommend.
    return _Result(name=name, sample_rate=16000, duration_s=1.0,
                   segments=[_Seg(0.0, 1.0)])


# ---- parser: registration & defaults -----------------------------------


def test_vad_gap_recommend_diff_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-gap-recommend-diff"] is gv.cmd_vad_gap_recommend_diff


def test_parser_registers_vad_gap_recommend_diff():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-recommend-diff", "a.wav", "b.wav"])
    assert args.command == "vad-gap-recommend-diff"
    assert args.wav_a == "a.wav"
    assert args.wav_b == "b.wav"


def test_parser_vad_gap_recommend_diff_knob_defaults():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-recommend-diff", "a.wav", "b.wav"])
    assert args.threshold == pytest.approx(0.5)
    assert args.min_speech_ms == pytest.approx(250.0)
    assert args.min_silence_ms == pytest.approx(800.0)
    assert args.speech_pad_ms == pytest.approx(30.0)
    assert math.isinf(args.max_speech_s)
    assert args.bias == "balanced"
    assert args.json is False
    assert args.csv is False


def test_parser_vad_gap_recommend_diff_custom_knobs():
    parser = gv.build_parser()
    args = parser.parse_args(
        [
            "vad-gap-recommend-diff",
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


def test_parser_vad_gap_recommend_diff_json_csv_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["vad-gap-recommend-diff", "a.wav", "b.wav", "--json", "--csv"]
        )


def test_parser_vad_gap_recommend_diff_bad_bias():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["vad-gap-recommend-diff", "a.wav", "b.wav", "--bias", "huge"]
        )


def test_parser_vad_gap_recommend_diff_requires_two_wavs():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-recommend-diff", "a.wav"])


# ---- pure core: vad_gap_recommend_delta --------------------------------


def test_delta_agrees_with_per_side_recommend():
    a, b = _clean(), _shifted()
    d = gv.vad_gap_recommend_delta(a, b)
    ra = gv.vad_gap_recommend(a)
    rb = gv.vad_gap_recommend(b)
    assert d["recommended_ms_a"] == ra["recommended_ms"]
    assert d["recommended_ms_b"] == rb["recommended_ms"]
    # Delta is b minus a, rounded to 1 place.
    assert d["recommended_ms_delta"] == round(
        rb["recommended_ms"] - ra["recommended_ms"], 1
    )
    assert d["recommended_s_delta"] == round(
        rb["recommended_s"] - ra["recommended_s"], 3
    )


def test_delta_carries_per_side_grades():
    a, b = _clean(), _shifted()
    d = gv.vad_gap_recommend_delta(a, b)
    assert d["grade_a"] == gv.vad_gap_confidence(a)["grade"]
    assert d["grade_b"] == gv.vad_gap_confidence(b)["grade"]


def test_delta_counts_and_int_deltas():
    a, b = _clean(), _shifted()
    d = gv.vad_gap_recommend_delta(a, b)
    assert d["num_segments_delta"] == d["num_segments_b"] - d["num_segments_a"]
    assert d["num_gaps_delta"] == d["num_gaps_b"] - d["num_gaps_a"]


def test_delta_echoes_bias():
    d = gv.vad_gap_recommend_delta(_clean(), _shifted(), bias="long")
    assert d["bias"] == "long"


def test_delta_bias_shifts_the_number():
    # short biases lower (eager), long biases higher (patient) within the valley.
    short = gv.vad_gap_recommend_delta(_clean(), _clean(), bias="short")
    long = gv.vad_gap_recommend_delta(_clean(), _clean(), bias="long")
    # A WAV vs itself has zero delta regardless of bias...
    assert short["recommended_ms_delta"] == 0.0
    assert long["recommended_ms_delta"] == 0.0
    # ...but the per-side number itself moves with bias.
    assert short["recommended_ms_a"] < long["recommended_ms_a"]


def test_delta_missing_side_yields_none_delta():
    # B has no gaps (one segment): nothing to recommend, so the delta is None.
    d = gv.vad_gap_recommend_delta(_clean(), _flat())
    assert d["recommended_ms_b"] is None
    assert d["recommended_ms_delta"] is None
    assert d["recommended_s_delta"] is None
    assert d["grade_b"] is None
    # The always-present counts still difference.
    assert d["num_segments_delta"] == d["num_segments_b"] - d["num_segments_a"]


def test_delta_both_missing_yields_none():
    d = gv.vad_gap_recommend_delta(_flat("x.wav"), _flat("y.wav"))
    assert d["recommended_ms_a"] is None
    assert d["recommended_ms_b"] is None
    assert d["recommended_ms_delta"] is None
    assert d["num_gaps_delta"] == 0


# ---- human renderer: render_vad_gap_recommend_diff ---------------------


def test_human_names_both_recordings_and_delta():
    lines = gv.render_vad_gap_recommend_diff(
        _clean(), _shifted(), label_a="before.wav", label_b="after.wav"
    )
    text = "\n".join(lines)
    assert "recommended-hangover diff" in text
    assert "before.wav" in text
    assert "after.wav" in text
    assert "recommended --min-silence-ms:" in text
    assert "→" in text


def test_human_shows_both_grades():
    lines = gv.render_vad_gap_recommend_diff(
        _clean(), _shifted(), label_a="a", label_b="b"
    )
    conf_line = next(l for l in lines if "confidence:" in l)
    assert "strong" in conf_line  # _clean grades strong


def test_human_echoes_bias():
    lines = gv.render_vad_gap_recommend_diff(
        _clean(), _shifted(), label_a="a", label_b="b", bias="long"
    )
    assert any("bias:" in l and "long" in l for l in lines)


def test_human_missing_side_dashes_and_na():
    lines = gv.render_vad_gap_recommend_diff(
        _clean(), _flat(), label_a="a", label_b="b"
    )
    rec_line = next(l for l in lines if "recommended --min-silence-ms:" in l)
    assert "-" in rec_line  # missing B number prints "-"
    assert "n/a" in rec_line  # undefined delta prints n/a
    conf_line = next(l for l in lines if "confidence:" in l)
    assert "-" in conf_line  # missing B grade prints "-"


def test_human_unavailable_hint():
    lines = gv.render_vad_gap_recommend_diff(
        None, None, label_a="a", label_b="b"
    )
    assert len(lines) == 1
    assert "silero-vad" in lines[0]


# ---- JSON renderer: render_vad_gap_recommend_diff_json -----------------


def test_json_shape():
    out = gv.render_vad_gap_recommend_diff_json(
        _clean(), _shifted(), label_a="before.wav", label_b="after.wav"
    )
    payload = json.loads(out)
    assert payload["available"] is True
    assert payload["recording_a"] == "before.wav"
    assert payload["recording_b"] == "after.wav"
    assert payload["bias"] == "balanced"
    assert "recommended_ms_delta" in payload
    assert payload["grade_a"] == "strong"


def test_json_missing_side_null_delta():
    out = gv.render_vad_gap_recommend_diff_json(
        _clean(), _flat(), label_a="a", label_b="b"
    )
    payload = json.loads(out)
    assert payload["recommended_ms_b"] is None
    assert payload["recommended_ms_delta"] is None
    assert payload["grade_b"] is None


def test_json_unavailable():
    out = gv.render_vad_gap_recommend_diff_json(
        None, None, label_a="a", label_b="b"
    )
    assert json.loads(out)["available"] is False


# ---- CSV renderer: render_vad_gap_recommend_diff_csv -------------------


def test_csv_one_row_per_recording():
    out = gv.render_vad_gap_recommend_diff_csv(
        _clean(), _shifted(), label_a="before.wav", label_b="after.wav"
    )
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0] == [
        "recording", "bias", "num_segments", "num_gaps", "recommended_ms", "grade"
    ]
    assert len(rows) == 3  # header + 2 recordings
    assert rows[1][0] == "before.wav"
    assert rows[2][0] == "after.wav"
    # Both rows carry the shared bias.
    assert rows[1][1] == "balanced"
    assert rows[2][1] == "balanced"


def test_csv_missing_side_empty_cells():
    out = gv.render_vad_gap_recommend_diff_csv(
        _clean(), _flat(), label_a="a", label_b="b"
    )
    rows = list(csv.reader(io.StringIO(out)))
    # B has no recommendation/grade — empty cells (CSV spelling of null).
    assert rows[2][4] == ""  # recommended_ms
    assert rows[2][5] == ""  # grade


def test_csv_unavailable_comment():
    out = gv.render_vad_gap_recommend_diff_csv(
        None, None, label_a="a", label_b="b"
    )
    assert out.startswith("# silero VAD unavailable")


# ---- handler: cmd_vad_gap_recommend_diff -------------------------------


def _args(**kw):
    base = dict(
        wav_a="a.wav",
        wav_b="b.wav",
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


def _two_segmenter(ra, rb):
    """Return a segmenter that yields ra for a.wav and rb for b.wav."""

    def seg(wav, params=None):
        return ra if wav == "a.wav" else rb

    return seg


def test_cmd_human():
    lines = []
    gv.cmd_vad_gap_recommend_diff(
        _args(),
        log=lines.append,
        segmenter=_two_segmenter(_clean(), _shifted()),
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "recommended-hangover diff" in text
    assert "a.wav" in text and "b.wav" in text


def test_cmd_json():
    lines = []
    gv.cmd_vad_gap_recommend_diff(
        _args(json=True),
        log=lines.append,
        segmenter=_two_segmenter(_clean(), _shifted()),
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is True
    assert payload["recording_a"] == "a.wav"
    assert payload["recording_b"] == "b.wav"


def test_cmd_csv():
    lines = []
    gv.cmd_vad_gap_recommend_diff(
        _args(csv=True),
        log=lines.append,
        segmenter=_two_segmenter(_clean(), _shifted()),
        availability=lambda: True,
    )
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    assert rows[0][0] == "recording"
    assert len(rows) == 3


def test_cmd_threads_bias():
    lines = []
    gv.cmd_vad_gap_recommend_diff(
        _args(json=True, bias="long"),
        log=lines.append,
        segmenter=_two_segmenter(_clean(), _shifted()),
        availability=lambda: True,
    )
    assert json.loads("\n".join(lines))["bias"] == "long"


def test_cmd_passes_shared_knobs_to_both_runs():
    captured = []

    def fake_segmenter(wav, params=None):
        captured.append((wav, params.threshold, params.min_silence_ms))
        return _clean() if wav == "a.wav" else _shifted()

    gv.cmd_vad_gap_recommend_diff(
        _args(threshold=0.7, min_silence_ms=400),
        log=lambda *_: None,
        segmenter=fake_segmenter,
        availability=lambda: True,
    )
    assert len(captured) == 2
    # Both runs share the same knobs.
    assert captured[0] == ("a.wav", 0.7, 400)
    assert captured[1] == ("b.wav", 0.7, 400)


def test_cmd_unavailable_human():
    lines = []
    gv.cmd_vad_gap_recommend_diff(
        _args(),
        log=lines.append,
        segmenter=lambda wav, params=None: None,
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert "silero-vad" in lines[0]


def test_cmd_unavailable_json():
    lines = []
    gv.cmd_vad_gap_recommend_diff(
        _args(json=True),
        log=lines.append,
        segmenter=lambda wav, params=None: None,
        availability=lambda: False,
    )
    assert json.loads("\n".join(lines))["available"] is False


def test_cmd_unavailable_csv():
    lines = []
    gv.cmd_vad_gap_recommend_diff(
        _args(csv=True),
        log=lines.append,
        segmenter=lambda wav, params=None: None,
        availability=lambda: False,
    )
    assert lines[0].startswith("# silero VAD unavailable")
