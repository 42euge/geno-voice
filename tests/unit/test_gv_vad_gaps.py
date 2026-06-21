"""Tests for iter-328 — the ``gv vad-gaps`` subcommand (examples/gv.py).

Every prior ``gv vad*`` surface reports *where the speech is* (segments,
counts, total speech-seconds). None reports *where the silence is* — the
inter-segment pauses. Yet that gap distribution is the direct signal for
tuning the end-of-turn hangover (``--min-silence-ms`` / the live
``chat.vad.silence_duration``): the SHORTEST real pause in a recording is the
floor above which raising the hangover starts merging two genuine turns into
one. ``gv vad-gaps recording.wav`` segments one WAV and reports that gap
distribution, the silence-side complement of ``gv vad``.

Like the rest of the VAD-analysis family, the handler takes injected
``segmenter`` / ``availability`` / ``log`` dependencies so every test runs
WITHOUT importing torch / silero-vad and without touching real audio — fast and
deterministic on the x86_64 Linux runner. The pure core (``vad_silence_gaps``)
and the three renderers are exercised directly against lightweight stand-ins
mirroring just the ``SileroResult`` / ``SpeechSegment`` attributes they read.
"""

from __future__ import annotations

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


def _result(*pairs, name="rec.wav", sample_rate=16000, duration_s=30.0):
    return _Result(
        name=name,
        sample_rate=sample_rate,
        duration_s=duration_s,
        segments=[_Seg(a, b) for a, b in pairs],
    )


# ---- parser: registration & defaults -----------------------------------


def test_vad_gaps_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-gaps"] is gv.cmd_vad_gaps


def test_vad_gaps_requires_wav_positional():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gaps"])


def test_vad_gaps_parses_wav_and_format_flags():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gaps", "rec.wav"])
    assert args.command == "vad-gaps"
    assert args.wav == "rec.wav"
    assert args.json is False
    assert args.csv is False


def test_vad_gaps_defaults_mirror_silero_params():
    # The segmenter knobs are shared with `gv vad`, so an operator gets the same
    # segmentation before measuring the gaps in it.
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gaps", "rec.wav"])
    from vad.silero import SileroParams

    p = SileroParams()
    assert args.threshold == p.threshold
    assert args.min_speech_ms == p.min_speech_ms
    assert args.min_silence_ms == p.min_silence_ms
    assert args.speech_pad_ms == p.speech_pad_ms


def test_vad_gaps_json_csv_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gaps", "rec.wav", "--json", "--csv"])


# ---- pure core: vad_silence_gaps ----------------------------------------


def test_silence_gaps_basic_two_segments():
    # speech [0,1] then [3,4] -> one 2.0s gap.
    d = gv.vad_silence_gaps(_result((0.0, 1.0), (3.0, 4.0)))
    assert d["num_segments"] == 2
    assert d["num_gaps"] == 1
    assert d["gaps"] == [2.0]
    assert d["min_gap_s"] == 2.0
    assert d["max_gap_s"] == 2.0
    assert d["mean_gap_s"] == 2.0
    assert d["total_silence_s"] == 2.0


def test_silence_gaps_three_segments_stats():
    # gaps: 3->4 =>... segments [0,1],[2,5],[6,7]: gaps 1.0 and 1.0
    d = gv.vad_silence_gaps(_result((0.0, 1.0), (2.0, 5.0), (6.0, 7.0)))
    assert d["num_gaps"] == 2
    assert d["gaps"] == [1.0, 1.0]
    assert d["min_gap_s"] == 1.0
    assert d["max_gap_s"] == 1.0
    assert d["total_silence_s"] == 2.0


def test_silence_gaps_min_is_shortest_pause():
    # gaps 2.0 then 0.5 -> min is the actionable shortest pause.
    d = gv.vad_silence_gaps(_result((0.0, 1.0), (3.0, 4.0), (4.5, 5.0)))
    assert d["gaps"] == [2.0, 0.5]
    assert d["min_gap_s"] == 0.5
    assert d["max_gap_s"] == 2.0
    assert round(d["mean_gap_s"], 3) == 1.25


def test_silence_gaps_after_segment_indices():
    d = gv.vad_silence_gaps(_result((0.0, 1.0), (3.0, 4.0), (6.0, 7.0)))
    # Each gap records which 1-based segment it follows and that segment's end.
    assert d["after_segment"] == [1, 2]
    assert d["after_segment_end_s"] == [1.0, 4.0]


def test_silence_gaps_single_segment_has_no_gaps():
    d = gv.vad_silence_gaps(_result((0.0, 1.0)))
    assert d["num_segments"] == 1
    assert d["num_gaps"] == 0
    assert d["gaps"] == []
    assert d["min_gap_s"] is None
    assert d["max_gap_s"] is None
    assert d["mean_gap_s"] is None
    assert d["total_silence_s"] == 0.0


def test_silence_gaps_empty_result():
    d = gv.vad_silence_gaps(_result())
    assert d["num_segments"] == 0
    assert d["num_gaps"] == 0
    assert d["gaps"] == []
    assert d["min_gap_s"] is None


def test_silence_gaps_unsorted_segments_are_sorted():
    # A robust core sorts by start before differencing, so out-of-order input
    # yields the same gaps as the sorted form.
    d = gv.vad_silence_gaps(_result((3.0, 4.0), (0.0, 1.0)))
    assert d["gaps"] == [2.0]
    assert d["after_segment_end_s"] == [1.0]


def test_silence_gaps_overlapping_segments_clamp_to_zero():
    # Padding can make adjacent regions touch/overlap; a negative raw difference
    # is not silence, so it clamps to 0.0 (no pause between them).
    d = gv.vad_silence_gaps(_result((0.0, 2.0), (1.5, 3.0)))
    assert d["gaps"] == [0.0]
    assert d["min_gap_s"] == 0.0


def test_silence_gaps_rounds_to_three_places():
    d = gv.vad_silence_gaps(_result((0.0, 1.0), (1.0001239, 2.0)))
    assert d["gaps"] == [0.0]
    d2 = gv.vad_silence_gaps(_result((0.0, 1.0), (1.123456, 2.0)))
    assert d2["gaps"] == [0.123]


# ---- human renderer ------------------------------------------------------


def test_render_vad_gaps_human_lines():
    lines = gv.render_vad_gaps(_result((0.0, 1.0), (3.0, 4.0), (4.5, 5.0)))
    text = "\n".join(lines)
    assert "silero VAD silence gaps" in text
    assert "rec.wav" in text
    assert "segments:" in text
    assert "gaps:" in text
    # The min-gap line names the actionable knob.
    assert "min gap" in text
    assert "min-silence-ms" in text
    # Per-gap detail rows.
    assert "after seg 1" in text
    assert "after seg 2" in text


def test_render_vad_gaps_single_segment_message():
    lines = gv.render_vad_gaps(_result((0.0, 1.0)))
    text = "\n".join(lines)
    assert "fewer than 2 segments" in text
    # No min-gap knob advice when there are no gaps to tune against.
    assert "min-silence-ms" not in text


def test_render_vad_gaps_unavailable():
    lines = gv.render_vad_gaps(None)
    assert len(lines) == 1
    assert "silero VAD unavailable" in lines[0]
    assert "silero-vad" in lines[0]


# ---- human renderer: byte-for-byte golden (iter-345) ---------------------
#
# The substring tests above (and the iter-328 originals) prove the report
# CONTAINS the right phrases, but never freeze the EXACT rendered block. So a
# silent regression in the label-column padding (the hand-tuned trailing spaces
# after "segments:" / "gaps:" / "min gap:" / "total silence:"), the
# `{:8.3f}` total-silence field, the per-gap `[{i:>2}]` index padding, the
# `{:6.3f}s` gap column, or the `(ends {end:.2f}s)` suffix would slip past every
# one of them. These two goldens close that gap — the SIXTH and final VAD-gap
# human surface to be byte-pinned (after percentiles iter-340, histogram
# iter-341, diff iter-342, grid iter-343, sweep iter-344), completing the
# gap-surface golden family.


def test_render_human_golden_multi_gap_block():
    # Three segments -> two gaps of distinct sizes (2.0s then 0.5s), so the
    # min/mean/max are all different and the per-gap rows carry distinct values.
    lines = gv.render_vad_gaps(_result((0.0, 1.0), (3.0, 4.0), (4.5, 5.0)))
    assert lines == [
        "silero VAD silence gaps — rec.wav",
        "  segments:     3",
        "  gaps:         2 (pauses between consecutive speech regions)",
        "  min gap:      0.500s (shortest real pause — keep --min-silence-ms "
        "below this to avoid merging turns)",
        "  mean gap:     1.250s",
        "  max gap:      2.000s",
        "  total silence:   2.500s",
        "  [ 1]  2.000s  after seg 1 (ends 1.00s)",
        "  [ 2]  0.500s  after seg 2 (ends 4.00s)",
    ]


def test_render_human_golden_single_segment_block():
    # The <2-segment branch: stats header is present but truncates at the
    # explanatory line, WITHOUT the min-gap knob advice (no pause to tune).
    lines = gv.render_vad_gaps(_result((0.0, 1.0)))
    assert lines == [
        "silero VAD silence gaps — rec.wav",
        "  segments:     1",
        "  gaps:         0 (pauses between consecutive speech regions)",
        "  (fewer than 2 segments — no inter-segment pause to measure)",
    ]


def test_render_human_golden_double_digit_index_alignment():
    # 11 segments -> 10 gaps forces the per-gap index past one digit, pinning
    # that `[{i:>2}]` right-aligns "[ 9]" against "[10]" (a width regression
    # would misalign the bracket column), and that the `{:8.3f}` total-silence
    # field widens cleanly from "   2.500s" to "  10.000s".
    pairs = [(i * 1.5, i * 1.5 + 0.5) for i in range(11)]
    lines = gv.render_vad_gaps(_result(*pairs))
    assert lines[:7] == [
        "silero VAD silence gaps — rec.wav",
        "  segments:     11",
        "  gaps:         10 (pauses between consecutive speech regions)",
        "  min gap:      1.000s (shortest real pause — keep --min-silence-ms "
        "below this to avoid merging turns)",
        "  mean gap:     1.000s",
        "  max gap:      1.000s",
        "  total silence:  10.000s",
    ]
    # The single-digit row and the two-digit row share the same bracket width:
    # "[ 9]" and "[10]" both end at the same character offset.
    assert lines[7] == "  [ 1]  1.000s  after seg 1 (ends 0.50s)"
    assert lines[15] == "  [ 9]  1.000s  after seg 9 (ends 12.50s)"
    assert lines[16] == "  [10]  1.000s  after seg 10 (ends 14.00s)"
    assert lines[15].index("]") == lines[16].index("]")


# ---- JSON renderer -------------------------------------------------------


def test_render_vad_gaps_json_shape():
    payload = json.loads(gv.render_vad_gaps_json(_result((0.0, 1.0), (3.0, 4.0))))
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["num_segments"] == 2
    assert payload["num_gaps"] == 1
    assert payload["min_gap_s"] == 2.0
    assert payload["max_gap_s"] == 2.0
    assert payload["mean_gap_s"] == 2.0
    assert payload["total_silence_s"] == 2.0
    assert payload["gaps"] == [
        {"index": 1, "after_segment": 1, "after_segment_end_s": 1.0, "gap_s": 2.0}
    ]


def test_render_vad_gaps_json_single_segment_nulls():
    payload = json.loads(gv.render_vad_gaps_json(_result((0.0, 1.0))))
    assert payload["num_gaps"] == 0
    assert payload["gaps"] == []
    assert payload["min_gap_s"] is None
    assert payload["max_gap_s"] is None
    assert payload["mean_gap_s"] is None


def test_render_vad_gaps_json_unavailable():
    payload = json.loads(gv.render_vad_gaps_json(None))
    assert payload["available"] is False
    assert "silero-vad" in payload["hint"]


# ---- CSV renderer --------------------------------------------------------


def test_render_vad_gaps_csv_rows():
    text = gv.render_vad_gaps_csv(_result((0.0, 1.0), (3.0, 4.0), (4.5, 5.0)))
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["index", "after_segment", "after_segment_end_s", "gap_s"]
    assert rows[1] == ["1", "1", "1.0", "2.0"]
    assert rows[2] == ["2", "2", "4.0", "0.5"]
    assert text == text.rstrip()  # no trailing newline


def test_render_vad_gaps_csv_single_segment_header_only():
    text = gv.render_vad_gaps_csv(_result((0.0, 1.0)))
    rows = list(csv.reader(io.StringIO(text)))
    assert rows == [["index", "after_segment", "after_segment_end_s", "gap_s"]]


def test_render_vad_gaps_csv_unavailable():
    text = gv.render_vad_gaps_csv(None)
    assert text.startswith("# silero VAD unavailable")


# ---- handler: cmd_vad_gaps ----------------------------------------------


def _run_handler(result, **flags):
    lines: List[str] = []
    args = gv.build_parser().parse_args(
        ["vad-gaps", "rec.wav", *[f"--{k}" for k, v in flags.items() if v]]
    )
    gv.cmd_vad_gaps(
        args,
        log=lines.append,
        segmenter=lambda wav, params=None: result,
        availability=lambda: True,
    )
    return lines


def test_cmd_vad_gaps_human_path():
    lines = _run_handler(_result((0.0, 1.0), (3.0, 4.0)))
    text = "\n".join(lines)
    assert "silero VAD silence gaps" in text
    assert "min gap" in text


def test_cmd_vad_gaps_json_path():
    lines = _run_handler(_result((0.0, 1.0), (3.0, 4.0)), json=True)
    payload = json.loads("\n".join(lines))
    assert payload["available"] is True
    assert payload["num_gaps"] == 1


def test_cmd_vad_gaps_csv_path():
    lines = _run_handler(_result((0.0, 1.0), (3.0, 4.0)), csv=True)
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    assert rows[0] == ["index", "after_segment", "after_segment_end_s", "gap_s"]


def test_cmd_vad_gaps_unavailable_human():
    lines: List[str] = []
    args = gv.build_parser().parse_args(["vad-gaps", "rec.wav"])
    gv.cmd_vad_gaps(
        args,
        log=lines.append,
        segmenter=lambda wav, params=None: pytest.fail("must not segment"),
        availability=lambda: False,
    )
    assert any("silero VAD unavailable" in ln for ln in lines)


def test_cmd_vad_gaps_unavailable_json():
    lines: List[str] = []
    args = gv.build_parser().parse_args(["vad-gaps", "rec.wav", "--json"])
    gv.cmd_vad_gaps(
        args,
        log=lines.append,
        segmenter=lambda wav, params=None: pytest.fail("must not segment"),
        availability=lambda: False,
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is False


def test_cmd_vad_gaps_passes_segmenter_params():
    seen = {}

    def _seg(wav, params=None):
        seen["wav"] = wav
        seen["threshold"] = params.threshold
        return _result((0.0, 1.0), (3.0, 4.0))

    args = gv.build_parser().parse_args(
        ["vad-gaps", "rec.wav", "--threshold", "0.7"]
    )
    gv.cmd_vad_gaps(args, log=lambda *_: None, segmenter=_seg, availability=lambda: True)
    assert seen["wav"] == "rec.wav"
    assert seen["threshold"] == 0.7
