"""iter-337 — End-to-end ``gv vad-gap-hist`` over the real corpus.

The histogram-side companion to :mod:`test_gv_vad_gaps_cli` (iter-329),
:mod:`test_gv_vad_gap_sweep_cli` (iter-331), :mod:`test_gv_vad_gap_grid_cli`
(iter-333) and :mod:`test_gv_vad_gap_diff_cli` (iter-335). Those modules pin the
point / sweep / grid / diff gap surfaces against the REAL Silero segmenter; this
one does the same for ``gv vad-gap-hist`` (iter-336), the distribution-shape
surface. Where ``gv vad-gaps`` answers "what are the min/mean/max pauses at THIS
setting?", ``gv vad-gap-hist`` answers "what is the full SHAPE of the pause-length
distribution?" — bucketing the inter-segment silence gaps into fixed-width
half-open ``[lo, hi)`` bins so a bimodal pattern (a short-pause mode plus a
long-pause mode with a valley between) becomes visible, where the three aggregate
numbers hide it. That valley is the safe place to set the end-of-turn hangover
(``--min-silence-ms`` / the live ``chat.vad.silence_duration``).

The anchoring property mirrors the family's: the histogram's gap count and
aggregates must equal an independent ``gv vad-gaps --json`` run segmented at the
same gate (so the totals always agree with ``gv vad-gaps``), every gap must fall
in the bin its duration indexes, the bin counts must sum to ``num_gaps``, and the
min/max gaps must land in the first / last NON-EMPTY bins. The 31s continuous
recording (which energy-VAD cannot split) is THE GATE here too: at the baseline
0.5 gate it must yield ≥2 segments, hence ≥1 measurable gap to bucket.

Skips cleanly when the recordings (large binary captures, not committed) or
``silero-vad`` (+ torch deps) are absent — same contract as the sibling modules.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples import gv  # noqa: E402
from vad.silero import silero_available  # noqa: E402

RECORDINGS_DIR = ROOT / "fixtures" / "recordings"
CONTINUOUS_31S = "voice-20260618-110355.wav"


def _recordings() -> list[Path]:
    if not RECORDINGS_DIR.is_dir():
        return []
    return sorted(RECORDINGS_DIR.glob("*.wav"))


RECORDINGS = _recordings()

pytestmark = [
    pytest.mark.skipif(
        not RECORDINGS,
        reason=f"no ground-truth recordings in {RECORDINGS_DIR} (rsync'd onto the loop host)",
    ),
    pytest.mark.skipif(
        not silero_available(),
        reason="silero-vad not installed (pulls torch + torchaudio)",
    ),
]


def _gaps_args(wav: Path, **over):
    """Namespace for a single-setting ``gv vad-gaps`` run (the anchor)."""
    base = dict(
        wav=str(wav),
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


def _run_gaps(wav: Path, **over) -> list[str]:
    lines: list[str] = []
    gv.cmd_vad_gaps(_gaps_args(wav, **over), log=lines.append)
    return lines


def _hist_args(wav: Path, **over):
    """Namespace for a ``gv vad-gap-hist`` run mirroring the parser defaults."""
    base = dict(
        wav=str(wav),
        bin_width_s=0.5,
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


def _run_hist(wav: Path, **over) -> list[str]:
    lines: list[str] = []
    gv.cmd_vad_gap_histogram(_hist_args(wav, **over), log=lines.append)
    return lines


def test_gv_vad_gap_hist_31s_has_a_measurable_gap_at_baseline():
    """THE GATE: histogramming the 31s continuous recording at the baseline 0.5
    gate must yield ≥2 segments and ≥1 inter-segment gap (energy-VAD cannot split
    this clip; Silero must), so there is a distribution to bucket — at least one
    non-empty bin."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    payload = json.loads(_run_hist(wav, threshold=0.5, json=True)[0])
    assert payload["available"] is True
    assert payload["name"] == CONTINUOUS_31S
    assert payload["num_segments"] >= 2
    assert payload["num_gaps"] == payload["num_segments"] - 1
    assert payload["num_gaps"] >= 1
    assert payload["bins"], "expected at least one bin for a recording with gaps"
    assert sum(b["count"] for b in payload["bins"]) >= 1


def test_gv_vad_gap_hist_matches_independent_vad_gaps():
    """The anchoring property: the histogram's gap count and aggregates must equal
    an independent ``gv vad-gaps --json`` run segmented at the same gate — proving
    the histogram buckets the SAME segmentation a standalone ``gv vad-gaps``
    produces, not a re-derived one."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    hist = json.loads(_run_hist(wav, threshold=0.3, json=True)[0])
    anchor = json.loads(_run_gaps(wav, threshold=0.3, json=True)[0])

    assert hist["num_segments"] == anchor["num_segments"]
    assert hist["num_gaps"] == anchor["num_gaps"]
    assert hist["min_gap_s"] == anchor["min_gap_s"]
    assert hist["mean_gap_s"] == anchor["mean_gap_s"]
    assert hist["max_gap_s"] == anchor["max_gap_s"]
    assert hist["total_silence_s"] == anchor["total_silence_s"]


def test_gv_vad_gap_hist_bin_counts_sum_to_num_gaps():
    """The bin counts must sum to ``num_gaps`` — every gap is bucketed exactly
    once, none dropped or double-counted, on a real segmentation."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    for bw in (0.25, 0.5, 1.0):
        hist = json.loads(_run_hist(wav, threshold=0.3, bin_width_s=bw, json=True)[0])
        assert sum(b["count"] for b in hist["bins"]) == hist["num_gaps"], (
            f"bin counts did not sum to num_gaps at bin_width_s={bw}"
        )


def test_gv_vad_gap_hist_every_gap_lands_in_its_indexed_bin():
    """Each individual gap (from the anchoring ``gv vad-gaps`` payload) must fall
    in the bin its duration indexes — ``lo_s <= gap < hi_s`` for exactly the bin
    at index ``floor(gap / bin_width_s)`` — so the histogram is a faithful
    bucketing of the real pauses, not an approximation."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    bw = 0.5
    anchor = json.loads(_run_gaps(wav, threshold=0.3, json=True)[0])
    gaps = [g["gap_s"] for g in anchor["gaps"]]
    if not gaps:
        pytest.skip("no gaps to place")
    hist = json.loads(_run_hist(wav, threshold=0.3, bin_width_s=bw, json=True)[0])
    bins = hist["bins"]

    # Re-derive the per-bin counts from the raw gaps and require they match.
    derived = [0] * len(bins)
    for gap in gaps:
        idx = int(gap / bw)
        if idx >= len(bins):  # defensive float clamp, mirrors the implementation
            idx = len(bins) - 1
        derived[idx] += 1
        # The chosen bin's half-open range must contain the gap.
        assert bins[idx]["lo_s"] <= gap < bins[idx]["hi_s"] + 1e-9
    assert derived == [b["count"] for b in bins]


def test_gv_vad_gap_hist_min_and_max_land_in_first_and_last_nonempty_bins():
    """The shortest gap must fall in the FIRST non-empty bin and the longest in
    the LAST non-empty bin — the histogram's extremes agree with the aggregate
    min/max gaps, on a real segmentation."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    hist = json.loads(_run_hist(wav, threshold=0.3, bin_width_s=0.5, json=True)[0])
    if hist["num_gaps"] == 0:
        pytest.skip("no gaps to bound")
    nonempty = [b for b in hist["bins"] if b["count"] > 0]
    assert nonempty, "num_gaps > 0 but no non-empty bins"
    first, last = nonempty[0], nonempty[-1]
    assert first["lo_s"] <= hist["min_gap_s"] < first["hi_s"] + 1e-9
    assert last["lo_s"] <= hist["max_gap_s"] < last["hi_s"] + 1e-9


def test_gv_vad_gap_hist_csv_matches_json():
    """``gv vad-gap-hist --csv`` describes the same bins as ``--json`` over THE
    GATE recording — same bin count, same per-bin ranges and counts — proving the
    two machine-readable surfaces agree."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    csv_lines = _run_hist(wav, threshold=0.3, bin_width_s=0.5, csv=True)
    csv_rows = list(csv.DictReader(io.StringIO(csv_lines[0])))

    hist = json.loads(_run_hist(wav, threshold=0.3, bin_width_s=0.5, json=True)[0])
    assert len(csv_rows) == len(hist["bins"])
    for i, (csv_row, jb) in enumerate(zip(csv_rows, hist["bins"]), start=1):
        assert int(csv_row["bin_index"]) == i
        assert abs(float(csv_row["lo_s"]) - jb["lo_s"]) <= 1e-9
        assert abs(float(csv_row["hi_s"]) - jb["hi_s"]) <= 1e-9
        assert int(csv_row["count"]) == jb["count"]


def test_gv_vad_gap_hist_human_report_shows_aggregates_and_bars():
    """The human report over THE GATE recording carries the title, the actionable
    ``--min-silence-ms`` knob hint, and at least one ASCII bar line (a ``#`` for a
    non-empty bin) — the distribution is rendered, not just summarized."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    lines = _run_hist(wav, threshold=0.3, bin_width_s=0.5)
    text = "\n".join(lines)
    assert "gap histogram" in text
    assert CONTINUOUS_31S in text
    assert "--min-silence-ms" in text
    assert "bin width:" in text
    assert any("#" in ln for ln in lines), "expected at least one ASCII bar"


@pytest.mark.parametrize("wav", RECORDINGS, ids=[p.name for p in RECORDINGS])
def test_gv_vad_gap_hist_emits_a_report_for_every_recording(wav: Path):
    """Every corpus recording produces a well-formed human histogram (title + the
    min-gap knob hint) and a parseable JSON payload whose gap count is one fewer
    than its segment count (or zero for a single region), with bin counts summing
    to the gap count and empty bins for a <2-segment clip."""
    lines = _run_hist(wav)
    text = "\n".join(lines)
    assert "gap histogram" in text
    assert wav.name in text

    payload = json.loads(_run_hist(wav, json=True)[0])
    assert payload["available"] is True
    assert payload["name"] == wav.name
    assert payload["bin_width_s"] == 0.5
    if payload["num_segments"] >= 2:
        assert payload["num_gaps"] == payload["num_segments"] - 1
        assert payload["min_gap_s"] is not None
        assert sum(b["count"] for b in payload["bins"]) == payload["num_gaps"]
        # bin spans are contiguous half-open [lo, hi) of the requested width.
        for b in payload["bins"]:
            assert math.isclose(b["hi_s"] - b["lo_s"], 0.5, abs_tol=1e-9)
    else:
        assert payload["num_gaps"] == 0
        assert payload["min_gap_s"] is None
        assert payload["mean_gap_s"] is None
        assert payload["max_gap_s"] is None
        assert payload["bins"] == []
