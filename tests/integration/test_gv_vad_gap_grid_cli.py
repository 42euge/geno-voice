"""iter-333 — End-to-end ``gv vad-gap-grid`` over the real corpus.

The grid-side companion to :mod:`test_gv_vad_gap_sweep_cli` (iter-331). That
module pins the 1-D ``gv vad-gap-sweep`` surface against the REAL Silero
segmenter; this one does the same for ``gv vad-gap-grid`` (iter-332), the 2-D
analogue that tabulates the inter-segment SILENCE-gap distribution across the
cartesian product of TWO knobs (the P(speech) gate × a column knob) at once.
Where ``gv vad-gap-sweep`` answers "how does the shortest pause MOVE as I
tighten ONE knob?", ``gv vad-gap-grid`` answers it across two dimensions in a
single pass — the floor an operator reads to find the (gate, hangover) pair
that lifts the min gap clear of a target end-of-turn hangover
(``--min-silence-ms`` / the live ``chat.vad.silence_duration``, iter-020).

The anchoring property mirrors the sweep test, lifted to a grid: every grid
cell's gap aggregates must equal an independent ``gv vad-gaps --json`` run
segmented at that cell's exact (threshold, column-knob) pair — proving the grid
differences the same per-cell segmentation a standalone ``gv vad-gaps`` would,
not a re-derived one. Plus the grid-specific properties the 1-D sweep cannot
exercise: row-major flattening order, and per-cell aggregate self-consistency.
The 31s continuous recording (which energy-VAD cannot split) is THE GATE here
too: its 0.5/default-hangover cell must yield ≥2 segments, hence ≥1 measurable
gap.

Skips cleanly when the recordings (large binary captures, not committed) or
``silero-vad`` (+ torch deps) are absent — same contract as the sibling
modules.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
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
    """Namespace for a single-setting ``gv vad-gaps`` run (the per-cell anchor)."""
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


def _grid_args(wav: Path, **over):
    """Namespace for a ``gv vad-gap-grid`` run mirroring the parser defaults."""
    base = dict(
        wav=str(wav),
        thresholds=[0.3, 0.5, 0.7, 0.9],
        min_silences=[400.0, 600.0, 800.0, 1000.0],
        min_speeches=None,
        speech_pads=None,
        max_speeches=None,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        json=False,
        csv=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _run_grid(wav: Path, **over) -> list[str]:
    lines: list[str] = []
    gv.cmd_vad_gap_grid(_grid_args(wav, **over), log=lines.append)
    return lines


def test_gv_vad_gap_grid_31s_has_a_measurable_gap_at_baseline():
    """THE GATE: the default gate × hangover grid over the 31s continuous
    recording includes the (0.5 gate, 800ms hangover) cell, which must report
    ≥2 segments and ≥1 inter-segment gap (energy-VAD cannot split this clip;
    Silero must)."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    payload = json.loads(
        _run_grid(wav, thresholds=[0.5], min_silences=[800.0], json=True)[0]
    )
    assert payload["available"] is True
    assert payload["name"] == CONTINUOUS_31S
    assert payload["row_axis"] == "threshold"
    assert payload["col_axis"] == "min_silence_ms"
    assert len(payload["grid"]) == 1
    cell = payload["grid"][0]
    assert cell["threshold"] == 0.5
    assert cell["min_silence_ms"] == 800.0
    assert cell["num_segments"] >= 2
    assert cell["num_gaps"] == cell["num_segments"] - 1
    assert cell["num_gaps"] >= 1
    assert cell["min_gap_s"] is not None


def test_gv_vad_gap_grid_cells_match_independent_vad_gaps():
    """The anchoring property, lifted to a grid: every grid cell's gap
    aggregates must equal an independent ``gv vad-gaps --json`` run segmented at
    that cell's exact (threshold, hangover) pair — proving the grid differences
    the same per-cell segmentation a standalone ``gv vad-gaps`` produces, across
    the full cartesian product of two knobs."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    thresholds = [0.3, 0.5, 0.7]
    hangovers = [400.0, 800.0]
    grid = json.loads(
        _run_grid(wav, thresholds=thresholds, min_silences=hangovers, json=True)[0]
    )
    by_cell = {
        (cell["threshold"], cell["min_silence_ms"]): cell for cell in grid["grid"]
    }
    assert len(by_cell) == len(thresholds) * len(hangovers)

    for thr in thresholds:
        for ms in hangovers:
            anchor = json.loads(
                _run_gaps(wav, threshold=thr, min_silence_ms=ms, json=True)[0]
            )
            cell = by_cell[(thr, ms)]
            assert cell["num_segments"] == anchor["num_segments"]
            assert cell["num_gaps"] == anchor["num_gaps"]
            assert cell["min_gap_s"] == anchor["min_gap_s"]
            assert cell["mean_gap_s"] == anchor["mean_gap_s"]
            assert cell["max_gap_s"] == anchor["max_gap_s"]
            assert cell["total_silence_s"] == anchor["total_silence_s"]


def test_gv_vad_gap_grid_column_axis_anchors_to_vad_gaps():
    """A non-default column axis (``--min-speeches``) anchors the same way: each
    cell equals an independent ``gv vad-gaps`` at that (gate, min-speech) pair
    with the hangover held at the scalar ``--min-silence-ms`` — covering a
    non-default column axis end to end and proving the held-scalar contract on
    the real engine."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    thresholds = [0.5, 0.7]
    min_speeches = [100.0, 250.0]
    grid = json.loads(
        _run_grid(
            wav,
            thresholds=thresholds,
            min_silences=None,
            min_speeches=min_speeches,
            min_silence_ms=800.0,
            json=True,
        )[0]
    )
    assert grid["col_axis"] == "min_speech_ms"
    by_cell = {
        (cell["threshold"], cell["min_speech_ms"]): cell for cell in grid["grid"]
    }

    for thr in thresholds:
        for ms in min_speeches:
            anchor = json.loads(
                _run_gaps(
                    wav,
                    threshold=thr,
                    min_speech_ms=ms,
                    min_silence_ms=800.0,
                    json=True,
                )[0]
            )
            cell = by_cell[(thr, ms)]
            assert cell["num_segments"] == anchor["num_segments"]
            assert cell["num_gaps"] == anchor["num_gaps"]
            assert cell["min_gap_s"] == anchor["min_gap_s"]
            assert cell["total_silence_s"] == anchor["total_silence_s"]


def test_gv_vad_gap_grid_emits_cells_in_row_major_order():
    """The grid flattens row-major: row 0's whole row of columns first, then
    row 1's, … — the same order :func:`vad_segmentation_grid` consumes. Reading
    the JSON ``grid`` list top-to-bottom, the threshold (row) is non-decreasing
    and the hangover (column) cycles through its values within each row block."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    thresholds = [0.3, 0.5, 0.7]
    hangovers = [400.0, 800.0]
    grid = json.loads(
        _run_grid(wav, thresholds=thresholds, min_silences=hangovers, json=True)[0]
    )["grid"]
    expected_order = [(t, h) for t in thresholds for h in hangovers]
    actual_order = [(c["threshold"], c["min_silence_ms"]) for c in grid]
    assert actual_order == expected_order


def test_gv_vad_gap_grid_aggregates_consistent_within_each_cell():
    """Within every grid cell that has ≥1 gap, the reported min ≤ mean ≤ max and
    total ≈ mean × count hold — the per-cell aggregates are internally
    self-consistent, on real segmentations across the whole default grid. A
    <2-segment cell reports null aggregates."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    grid = json.loads(_run_grid(wav, json=True)[0])["grid"]
    saw_gap = False
    for cell in grid:
        if cell["num_gaps"] == 0:
            assert cell["min_gap_s"] is None
            assert cell["mean_gap_s"] is None
            assert cell["max_gap_s"] is None
            continue
        saw_gap = True
        assert cell["min_gap_s"] <= cell["mean_gap_s"] <= cell["max_gap_s"]
        assert (
            abs(cell["total_silence_s"] - cell["mean_gap_s"] * cell["num_gaps"]) <= 0.05
        )
    assert saw_gap, "expected at least one grid cell with a measurable gap"


def test_gv_vad_gap_grid_longer_hangover_never_grows_gap_count_within_a_row():
    """Within a single gate row, reading across a rising ``--min-silences``
    column axis, the gap count is monotone non-increasing: a longer hangover can
    only merge adjacent regions (dropping the short pause between them), never
    split one — the grid-row echo of the iter-331 sweep monotonicity, held for
    every gate row at once."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    thresholds = [0.3, 0.5, 0.7]
    hangovers = [200.0, 400.0, 800.0, 1600.0]
    grid = json.loads(
        _run_grid(wav, thresholds=thresholds, min_silences=hangovers, json=True)[0]
    )["grid"]
    by_thr: dict[float, list[tuple[float, int]]] = {}
    for cell in grid:
        by_thr.setdefault(cell["threshold"], []).append(
            (cell["min_silence_ms"], cell["num_gaps"])
        )
    for thr, row in by_thr.items():
        row.sort()
        counts = [n for _, n in row]
        for lo, hi in zip(counts, counts[1:]):
            assert hi <= lo, f"gap count rose across rising hangovers at {thr}: {counts}"


def test_gv_vad_gap_grid_csv_matches_json():
    """``gv vad-gap-grid --csv`` describes the same per-cell rows as ``--json``
    over THE GATE recording — same (row, col) axis values, same segment/gap
    counts, same min gap, in the same row-major order — proving the two
    machine-readable surfaces agree."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    csv_lines = _run_grid(wav, csv=True)
    assert len(csv_lines) == 1
    csv_rows = list(csv.DictReader(io.StringIO(csv_lines[0])))

    json_cells = json.loads(_run_grid(wav, json=True)[0])["grid"]
    assert len(csv_rows) == len(json_cells)
    for csv_row, cell in zip(csv_rows, json_cells):
        assert abs(float(csv_row["threshold"]) - cell["threshold"]) <= 1e-9
        assert abs(float(csv_row["min_silence_ms"]) - cell["min_silence_ms"]) <= 1e-9
        assert int(csv_row["num_segments"]) == cell["num_segments"]
        assert int(csv_row["num_gaps"]) == cell["num_gaps"]
        if cell["num_gaps"] == 0:
            assert csv_row["min_gap_s"] == ""
        else:
            assert abs(float(csv_row["min_gap_s"]) - cell["min_gap_s"]) <= 0.01


@pytest.mark.parametrize("wav", RECORDINGS, ids=[p.name for p in RECORDINGS])
def test_gv_vad_gap_grid_emits_a_report_for_every_recording(wav: Path):
    """Every corpus recording produces a well-formed human grid table (title +
    a gap column header) and a parseable JSON payload with one cell per
    (threshold, hangover) pair of the default grid, each cell's gap count one
    fewer than its segment count (or zero for a single region)."""
    lines = _run_grid(wav)
    text = "\n".join(lines)
    assert wav.name in text
    assert "min_gap" in text

    payload = json.loads(_run_grid(wav, json=True)[0])
    assert payload["available"] is True
    assert payload["name"] == wav.name
    assert payload["row_axis"] == "threshold"
    assert payload["col_axis"] == "min_silence_ms"
    # default grid: 4 thresholds × 4 hangovers = 16 cells.
    assert len(payload["grid"]) == 16
    for cell in payload["grid"]:
        if cell["num_segments"] >= 2:
            assert cell["num_gaps"] == cell["num_segments"] - 1
            assert cell["min_gap_s"] is not None
        else:
            assert cell["num_gaps"] == 0
            assert cell["min_gap_s"] is None
            assert cell["max_gap_s"] is None
            assert cell["mean_gap_s"] is None
