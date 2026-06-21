"""Tests for iter-336 — the ``gv vad-gap-hist`` subcommand (examples/gv.py).

``gv vad-gaps`` (iter-328) collapses the inter-segment silence distribution to
three numbers (min/mean/max) — but those cannot distinguish a *bimodal* pause
pattern (a cluster of short within-turn pauses plus a cluster of long
between-turn pauses, with a valley between) from a uniform spread with the same
min/max. ``gv vad-gap-hist`` shows the shape: it buckets the gaps into
fixed-width bins (``--bin-width-s``) and reports a count per bin, so the valley
that marks the safe end-of-turn hangover (``--min-silence-ms`` / the live
``chat.vad.silence_duration``) is visible.

Like the rest of the VAD-analysis family, the handler takes injected
``segmenter`` / ``availability`` / ``log`` dependencies so every test runs
WITHOUT importing torch / silero-vad and without touching real audio — fast and
deterministic on the x86_64 Linux runner. The pure core (``vad_gap_histogram``)
and the three renderers are exercised directly against lightweight stand-ins
mirroring just the ``SileroResult`` / ``SpeechSegment`` attributes they read.
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


def _result(*pairs, name="rec.wav", sample_rate=16000, duration_s=30.0):
    return _Result(
        name=name,
        sample_rate=sample_rate,
        duration_s=duration_s,
        segments=[_Seg(a, b) for a, b in pairs],
    )


# ---- parser: registration & defaults -----------------------------------


def test_vad_gap_hist_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-gap-hist"] is gv.cmd_vad_gap_histogram


def test_parser_registers_vad_gap_hist():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-hist", "rec.wav"])
    assert args.command == "vad-gap-hist"
    assert args.wav == "rec.wav"


def test_parser_defaults_mirror_vad_gaps_knobs():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-hist", "rec.wav"])
    # Shares the gv vad segmenter knobs.
    assert args.threshold == pytest.approx(0.5)
    assert args.min_speech_ms == pytest.approx(250.0)
    assert args.min_silence_ms == pytest.approx(800.0)
    assert args.speech_pad_ms == pytest.approx(30.0)
    assert math.isinf(args.max_speech_s)
    # The histogram-specific knob.
    assert args.bin_width_s == pytest.approx(0.5)
    assert args.json is False
    assert args.csv is False


def test_parser_accepts_custom_bin_width():
    parser = gv.build_parser()
    args = parser.parse_args(["vad-gap-hist", "rec.wav", "--bin-width-s", "0.25"])
    assert args.bin_width_s == pytest.approx(0.25)


def test_parser_rejects_nonpositive_bin_width():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-hist", "rec.wav", "--bin-width-s", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-hist", "rec.wav", "--bin-width-s", "-1"])


def test_parser_json_csv_mutually_exclusive():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-hist", "rec.wav", "--json", "--csv"])


def test_parser_rejects_out_of_range_threshold():
    parser = gv.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vad-gap-hist", "rec.wav", "--threshold", "1.5"])


# ---- pure core: vad_gap_histogram --------------------------------------


def test_histogram_buckets_gaps():
    # Segments at [0,1],[2,3],[5,6],[10,11] → gaps of 1.0, 2.0, 4.0.
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    h = gv.vad_gap_histogram(res, bin_width_s=1.0)
    assert h["num_segments"] == 4
    assert h["num_gaps"] == 3
    assert h["bin_width_s"] == 1.0
    # max gap 4.0 → bins span [0,1)..[4,5), i.e. 5 bins.
    assert len(h["bins"]) == 5
    counts = [b["count"] for b in h["bins"]]
    # gap 1.0 → bin index 1; gap 2.0 → bin 2; gap 4.0 → bin 4.
    assert counts == [0, 1, 1, 0, 1]
    assert sum(counts) == h["num_gaps"]


def test_histogram_bin_ranges_are_half_open():
    res = _result((0, 1), (2, 3), (5, 6))  # gaps 1.0, 2.0
    h = gv.vad_gap_histogram(res, bin_width_s=0.5)
    # max gap 2.0 → floor(2.0/0.5)=4, +1 = 5 bins: [0,.5)..[2,2.5)
    assert [(b["lo_s"], b["hi_s"]) for b in h["bins"]] == [
        (0.0, 0.5),
        (0.5, 1.0),
        (1.0, 1.5),
        (1.5, 2.0),
        (2.0, 2.5),
    ]
    # gap 1.0 boundary goes to UPPER bin [1.0, 1.5); gap 2.0 to [2.0, 2.5).
    counts = [b["count"] for b in h["bins"]]
    assert counts == [0, 0, 1, 0, 1]


def test_histogram_anchors_to_vad_silence_gaps():
    res = _result((0, 1), (2.5, 3), (5, 6.5))
    h = gv.vad_gap_histogram(res, bin_width_s=0.5)
    d = gv.vad_silence_gaps(res)
    # Aggregates are passed through verbatim from the gap core.
    for key in ("num_segments", "num_gaps", "min_gap_s", "max_gap_s",
                "mean_gap_s", "total_silence_s"):
        assert h[key] == d[key]
    # Counts sum to the gap count.
    assert sum(b["count"] for b in h["bins"]) == d["num_gaps"]


def test_histogram_empty_for_fewer_than_two_segments():
    for res in (_result(), _result((0, 1))):
        h = gv.vad_gap_histogram(res)
        assert h["num_gaps"] == 0
        assert h["bins"] == []
        assert h["min_gap_s"] is None
        assert h["max_gap_s"] is None
        assert h["mean_gap_s"] is None
        assert h["total_silence_s"] == 0.0


def test_histogram_bimodal_shape_visible():
    # Two short pauses (~0.2s) and two long pauses (~2.0s), valley between.
    res = _result(
        (0, 1), (1.2, 2), (2.2, 3),  # two ~0.2s gaps
        (5, 6), (8, 9),              # ~2.0s, ~2.0s gaps
    )
    h = gv.vad_gap_histogram(res, bin_width_s=0.5)
    counts = [b["count"] for b in h["bins"]]
    # First bin [0,0.5) holds the two short pauses.
    assert counts[0] == 2
    # The long pauses (~2.0s) land in a later bin, with a zero valley between.
    assert counts[-1] >= 1
    assert 0 in counts[1:-1]  # at least one empty valley bin


def test_histogram_rejects_nonpositive_bin_width():
    res = _result((0, 1), (2, 3))
    for bad in (0, -0.5, float("nan")):
        with pytest.raises(ValueError):
            gv.vad_gap_histogram(res, bin_width_s=bad)


def test_histogram_boundary_gap_clamps_into_top_bin():
    # A gap landing exactly on the top boundary must not index out of range.
    res = _result((0, 1), (2.5, 3))  # one gap of 1.5
    h = gv.vad_gap_histogram(res, bin_width_s=0.5)
    counts = [b["count"] for b in h["bins"]]
    assert sum(counts) == 1
    assert len(h["bins"]) >= 1


# ---- renderers: human ---------------------------------------------------


def test_render_human_unavailable():
    lines = gv.render_vad_gap_histogram(None)
    assert len(lines) == 1
    assert "silero VAD unavailable" in lines[0]
    assert "install 'silero-vad'" in lines[0]


def test_render_human_no_gaps():
    lines = gv.render_vad_gap_histogram(_result((0, 1)))
    joined = "\n".join(lines)
    assert "gap histogram — rec.wav" in joined
    assert "fewer than 2 segments" in joined
    # No min-silence advice when there is no shortest pause.
    assert "--min-silence-ms" not in joined


def test_render_human_with_gaps():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    lines = gv.render_vad_gap_histogram(res, bin_width_s=1.0)
    joined = "\n".join(lines)
    assert "gap histogram — rec.wav" in joined
    assert "segments:     4" in joined
    assert "gaps:         3" in joined
    assert "bin width:    1.000s" in joined
    assert "--min-silence-ms" in joined  # named on the min-gap line
    # One bar line per bin, with the half-open range and an ASCII bar.
    bar_lines = [ln for ln in lines if ln.strip().startswith("[")]
    assert len(bar_lines) == 5
    assert "#" in joined  # the busiest bin draws at least one bar char


def test_render_human_bar_scales_to_busiest_bin():
    # Three gaps all in the first bin → that bin is busiest, full bar width.
    res = _result((0, 1), (1.1, 2), (2.1, 3), (3.1, 4))
    lines = gv.render_vad_gap_histogram(res, bin_width_s=0.5)
    bar_lines = [ln for ln in lines if ln.strip().startswith("[")]
    # Busiest bin's bar should be the longest run of '#'.
    bars = [ln.count("#") for ln in bar_lines]
    assert max(bars) == 40  # bar_width


# ---- renderer: human golden (byte-for-byte) -----------------------------
#
# The test_render_human_* tests above assert STRUCTURE + substrings (a bar line
# count, a "#" appears, the --min-silence-ms knob is named somewhere) but never
# freeze the EXACT rendered block. So a silent regression in the aggregate
# header column, the half-open ``[lo, hi)`` range formatting (``{:6.3f}``), the
# right-aligned count field (``{:>4}``), the two-space gutters around the bar,
# or the bar-scaling arithmetic would slip through every one of them. These two
# goldens pin the byte-for-byte report for two fixed stub segmentations, so the
# human face of the histogram can only change deliberately — the iter-341
# extension of iter-340's percentiles golden to its busiest-drift sibling
# surface (the gv vad-gap-hist aligned bar block, where column drift is most
# visible). The percentiles golden froze the ``pNN`` column; this freezes the
# bar column.


def test_render_human_golden_full_width_bars():
    # Gaps (sorted): [1.0, 2.0, 4.0] at bin width 1.0 → five [lo, hi) bins
    # spanning 0..5, each occupied bin holding exactly one gap. With max_count==1
    # every occupied bin draws the full 40-char bar and the two empty bins draw
    # none. Pins the aggregate header (the min-gap knob-hint line, the
    # right-aligned ``total silence:   7.000s``), the ``[{:6.3f}, {:6.3f})``
    # range column, the ``{:>4}`` count field, and the full-width bar.
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    lines = gv.render_vad_gap_histogram(res, bin_width_s=1.0)
    bar = "#" * 40
    assert lines == [
        "silero VAD gap histogram — rec.wav",
        "  segments:     4",
        "  gaps:         3 (pauses between consecutive speech regions)",
        "  bin width:    1.000s",
        "  min gap:      1.000s (shortest real pause — keep --min-silence-ms "
        "below this to avoid merging turns)",
        "  mean gap:     2.333s",
        "  max gap:      4.000s",
        "  total silence:   7.000s",
        "  [ 0.000,  1.000)     0  ",
        f"  [ 1.000,  2.000)     1  {bar}",
        f"  [ 2.000,  3.000)     1  {bar}",
        "  [ 3.000,  4.000)     0  ",
        f"  [ 4.000,  5.000)     1  {bar}",
    ]


def test_render_human_golden_partial_bar_scaling():
    # Bimodal: three short gaps (0.2s each) cluster in the first 0.5s bin, one
    # long 2.0s gap sits alone in the top bin. The busiest bin (count 3) draws
    # the full 40-char bar; the lone gap scales to round(1/3 * 40) == 13 chars,
    # pinning the bar-scaling arithmetic itself (not just "a bar exists").
    res = _result((0, 1), (1.2, 2), (2.2, 3), (3.2, 4), (6.0, 7))
    lines = gv.render_vad_gap_histogram(res, bin_width_s=0.5)
    assert lines == [
        "silero VAD gap histogram — rec.wav",
        "  segments:     5",
        "  gaps:         4 (pauses between consecutive speech regions)",
        "  bin width:    0.500s",
        "  min gap:      0.200s (shortest real pause — keep --min-silence-ms "
        "below this to avoid merging turns)",
        "  mean gap:     0.650s",
        "  max gap:      2.000s",
        "  total silence:   2.600s",
        "  [ 0.000,  0.500)     3  " + "#" * 40,
        "  [ 0.500,  1.000)     0  ",
        "  [ 1.000,  1.500)     0  ",
        "  [ 1.500,  2.000)     0  ",
        "  [ 2.000,  2.500)     1  " + "#" * 13,
    ]
    # The bar column starts at the same offset on the full and partial rows,
    # proving the range+count prefix is fixed-width regardless of bar length.
    busiest = next(ln for ln in lines if ln.endswith("#" * 40))
    partial = next(ln for ln in lines if ln.endswith("#" * 13))
    assert busiest.index("#") == partial.index("#")


# ---- renderers: json ----------------------------------------------------


def test_render_json_unavailable():
    payload = json.loads(gv.render_vad_gap_histogram_json(None))
    assert payload["available"] is False
    assert "install 'silero-vad'" in payload["hint"]


def test_render_json_with_gaps():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    payload = json.loads(gv.render_vad_gap_histogram_json(res, bin_width_s=1.0))
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["num_segments"] == 4
    assert payload["num_gaps"] == 3
    assert payload["bin_width_s"] == 1.0
    assert len(payload["bins"]) == 5
    assert sum(b["count"] for b in payload["bins"]) == 3
    # Each bin object carries lo_s/hi_s/count.
    assert set(payload["bins"][0].keys()) == {"lo_s", "hi_s", "count"}


def test_render_json_no_gaps_empty_bins():
    payload = json.loads(gv.render_vad_gap_histogram_json(_result((0, 1))))
    assert payload["available"] is True
    assert payload["num_gaps"] == 0
    assert payload["bins"] == []
    assert payload["min_gap_s"] is None


# ---- renderers: csv -----------------------------------------------------


def test_render_csv_unavailable():
    out = gv.render_vad_gap_histogram_csv(None)
    assert out.startswith("# silero VAD unavailable")


def test_render_csv_header_and_rows():
    res = _result((0, 1), (2, 3), (5, 6), (10, 11))
    out = gv.render_vad_gap_histogram_csv(res, bin_width_s=1.0)
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0] == ["bin_index", "lo_s", "hi_s", "count"]
    assert len(rows) == 1 + 5  # header + 5 bins
    # Counts in the CSV sum to the gap count.
    counts = [int(r[3]) for r in rows[1:]]
    assert sum(counts) == 3
    # bin_index is 1-based and contiguous.
    assert [int(r[0]) for r in rows[1:]] == [1, 2, 3, 4, 5]


def test_render_csv_no_gaps_header_only():
    out = gv.render_vad_gap_histogram_csv(_result((0, 1)))
    rows = list(csv.reader(io.StringIO(out)))
    assert rows == [["bin_index", "lo_s", "hi_s", "count"]]


def test_render_csv_matches_json_bins():
    res = _result((0, 1), (2, 3), (5, 6.5), (10, 11))
    payload = json.loads(gv.render_vad_gap_histogram_json(res, bin_width_s=0.5))
    out = gv.render_vad_gap_histogram_csv(res, bin_width_s=0.5)
    rows = list(csv.reader(io.StringIO(out)))[1:]
    assert len(rows) == len(payload["bins"])
    for row, b in zip(rows, payload["bins"]):
        assert float(row[1]) == b["lo_s"]
        assert float(row[2]) == b["hi_s"]
        assert int(row[3]) == b["count"]


# ---- handler: cmd_vad_gap_histogram (injected deps) --------------------


def _args(**kw):
    base = dict(
        wav="rec.wav",
        bin_width_s=0.5,
        threshold=0.5,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        json=False,
        csv=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_handler_unavailable_human():
    lines: List[str] = []
    gv.cmd_vad_gap_histogram(
        _args(),
        log=lines.append,
        segmenter=lambda *a, **k: pytest.fail("segmenter must not run"),
        availability=lambda: False,
    )
    assert any("silero VAD unavailable" in ln for ln in lines)


def test_handler_unavailable_json():
    lines: List[str] = []
    gv.cmd_vad_gap_histogram(
        _args(json=True),
        log=lines.append,
        segmenter=lambda *a, **k: pytest.fail("segmenter must not run"),
        availability=lambda: False,
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is False


def test_handler_unavailable_csv():
    lines: List[str] = []
    gv.cmd_vad_gap_histogram(
        _args(csv=True),
        log=lines.append,
        segmenter=lambda *a, **k: pytest.fail("segmenter must not run"),
        availability=lambda: False,
    )
    assert "\n".join(lines).startswith("# silero VAD unavailable")


def test_handler_human_runs_segmenter():
    res = _result((0, 1), (2, 3), (5, 6))
    captured = {}

    def seg(wav, *, params):
        captured["wav"] = wav
        captured["params"] = params
        return res

    lines: List[str] = []
    gv.cmd_vad_gap_histogram(
        _args(threshold=0.7),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert captured["wav"] == "rec.wav"
    assert captured["params"].threshold == pytest.approx(0.7)
    joined = "\n".join(lines)
    assert "gap histogram — rec.wav" in joined


def test_handler_json_path():
    res = _result((0, 1), (2, 3), (5, 6))
    lines: List[str] = []
    gv.cmd_vad_gap_histogram(
        _args(json=True),
        log=lines.append,
        segmenter=lambda *a, **k: res,
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["available"] is True
    assert payload["num_gaps"] == 2


def test_handler_csv_path():
    res = _result((0, 1), (2, 3), (5, 6))
    lines: List[str] = []
    gv.cmd_vad_gap_histogram(
        _args(csv=True),
        log=lines.append,
        segmenter=lambda *a, **k: res,
        availability=lambda: True,
    )
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    assert rows[0] == ["bin_index", "lo_s", "hi_s", "count"]


def test_handler_passes_bin_width_through():
    res = _result((0, 1), (2, 3), (5, 6))
    lines: List[str] = []
    gv.cmd_vad_gap_histogram(
        _args(bin_width_s=0.25, json=True),
        log=lines.append,
        segmenter=lambda *a, **k: res,
        availability=lambda: True,
    )
    payload = json.loads("\n".join(lines))
    assert payload["bin_width_s"] == 0.25
