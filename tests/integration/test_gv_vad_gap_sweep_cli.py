"""iter-331 — End-to-end ``gv vad-gap-sweep`` over the real corpus.

The sweep-side companion to :mod:`test_gv_vad_gaps_cli` (iter-329). That module
pins the single-setting ``gv vad-gaps`` surface against the REAL Silero
segmenter; this one does the same for ``gv vad-gap-sweep`` (iter-330), the
surface that tabulates how the inter-segment SILENCE-gap distribution moves as a
segmenter knob sweeps. Where ``gv vad-gaps`` answers "what are the pauses at THIS
setting?", ``gv vad-gap-sweep`` answers "how does the shortest pause MOVE as I
tighten the gate / lengthen the hangover?" — the floor an operator reads to find
the knob value that lifts the min gap clear of a target end-of-turn hangover
(``--min-silence-ms`` / the live ``chat.vad.silence_duration``, iter-020).

The anchoring property: every swept row's gap aggregates must equal an
independent ``gv vad-gaps --json`` run segmented at that row's exact knobs —
proving the sweep differences the same per-value segmentation a standalone
``gv vad-gaps`` would, not a re-derived one. The 31s continuous recording (which
energy-VAD cannot split) is THE GATE here too: it must yield ≥2 segments, hence
≥1 measurable gap, at the baseline knobs.

Skips cleanly when the recordings (large binary captures, not committed) or
``silero-vad`` (+ torch deps) are absent — same contract as the sibling modules.
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


def _sweep_args(wav: Path, **over):
    """Namespace for a ``gv vad-gap-sweep`` run mirroring the parser defaults."""
    base = dict(
        wav=str(wav),
        thresholds=[0.3, 0.5, 0.7, 0.9],
        min_silences=None,
        min_speeches=None,
        speech_pads=None,
        max_speeches=None,
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


def _run_sweep(wav: Path, **over) -> list[str]:
    lines: list[str] = []
    gv.cmd_vad_gap_sweep(_sweep_args(wav, **over), log=lines.append)
    return lines


def test_gv_vad_gap_sweep_31s_has_a_measurable_gap_at_baseline():
    """THE GATE: the default threshold sweep over the 31s continuous recording
    includes the baseline 0.5 gate, whose row must report ≥2 segments and ≥1
    inter-segment gap (energy-VAD cannot split this clip; Silero must)."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    payload = json.loads(_run_sweep(wav, json=True)[0])
    assert payload["available"] is True
    assert payload["name"] == CONTINUOUS_31S
    assert payload["axis"] == "threshold"
    rows = {row["threshold"]: row for row in payload["sweep"]}
    assert set(rows) == {0.3, 0.5, 0.7, 0.9}
    baseline = rows[0.5]
    assert baseline["num_segments"] >= 2
    assert baseline["num_gaps"] == baseline["num_segments"] - 1
    assert baseline["num_gaps"] >= 1
    assert baseline["min_gap_s"] is not None


def test_gv_vad_gap_sweep_rows_match_independent_vad_gaps():
    """The anchoring property: each swept row's gap aggregates must equal an
    independent ``gv vad-gaps --json`` run segmented at that row's exact knobs —
    proving the sweep differences the same per-value segmentation a standalone
    ``gv vad-gaps`` produces, not a re-derived one."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    thresholds = [0.3, 0.5, 0.7]
    sweep = json.loads(_run_sweep(wav, thresholds=thresholds, json=True)[0])
    by_threshold = {row["threshold"]: row for row in sweep["sweep"]}

    for thr in thresholds:
        anchor = json.loads(_run_gaps(wav, threshold=thr, json=True)[0])
        row = by_threshold[thr]
        assert row["num_segments"] == anchor["num_segments"]
        assert row["num_gaps"] == anchor["num_gaps"]
        assert row["min_gap_s"] == anchor["min_gap_s"]
        assert row["mean_gap_s"] == anchor["mean_gap_s"]
        assert row["max_gap_s"] == anchor["max_gap_s"]
        assert row["total_silence_s"] == anchor["total_silence_s"]


def test_gv_vad_gap_sweep_hangover_axis_anchors_to_vad_gaps():
    """Sweeping the hangover axis (``--min-silences``) anchors the same way:
    each row equals an independent ``gv vad-gaps`` at that hangover with the gate
    held at the scalar ``--threshold`` — covering a non-default axis end to
    end."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    hangovers = [200.0, 400.0, 800.0]
    sweep = json.loads(
        _run_sweep(wav, min_silences=hangovers, threshold=0.5, json=True)[0]
    )
    assert sweep["axis"] == "min_silence_ms"
    by_ms = {row["min_silence_ms"]: row for row in sweep["sweep"]}

    for ms in hangovers:
        anchor = json.loads(
            _run_gaps(wav, min_silence_ms=ms, threshold=0.5, json=True)[0]
        )
        row = by_ms[ms]
        assert row["num_segments"] == anchor["num_segments"]
        assert row["num_gaps"] == anchor["num_gaps"]
        assert row["min_gap_s"] == anchor["min_gap_s"]
        assert row["total_silence_s"] == anchor["total_silence_s"]


def test_gv_vad_gap_sweep_longer_hangover_never_grows_gap_count():
    """Reading down a rising ``--min-silences`` sweep, the gap count is monotone
    non-increasing: a longer hangover can only merge adjacent regions (dropping
    the short pause between them), never split one — the silence-side echo of the
    vad-sweep segment-count monotonicity, observed within a SINGLE sweep run."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    sweep = json.loads(
        _run_sweep(wav, min_silences=[200.0, 400.0, 800.0, 1600.0], json=True)[0]
    )
    counts = [row["num_gaps"] for row in sweep["sweep"]]
    for lo, hi in zip(counts, counts[1:]):
        assert hi <= lo, f"gap count rose across rising hangovers: {counts}"


def test_gv_vad_gap_sweep_aggregates_consistent_within_each_row():
    """Within every swept row that has ≥1 gap, the reported min ≤ mean ≤ max and
    total ≈ mean × count hold — the per-row aggregates are internally
    self-consistent, on real segmentations across the whole default sweep."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    sweep = json.loads(_run_sweep(wav, json=True)[0])
    saw_gap = False
    for row in sweep["sweep"]:
        if row["num_gaps"] == 0:
            assert row["min_gap_s"] is None
            assert row["mean_gap_s"] is None
            assert row["max_gap_s"] is None
            continue
        saw_gap = True
        assert row["min_gap_s"] <= row["mean_gap_s"] <= row["max_gap_s"]
        assert (
            abs(row["total_silence_s"] - row["mean_gap_s"] * row["num_gaps"]) <= 0.05
        )
    assert saw_gap, "expected at least one swept row with a measurable gap"


def test_gv_vad_gap_sweep_csv_matches_json():
    """``gv vad-gap-sweep --csv`` describes the same per-value rows as ``--json``
    over THE GATE recording — same swept-axis values, same segment/gap counts,
    same min gap — proving the two machine-readable surfaces agree."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    csv_lines = _run_sweep(wav, csv=True)
    assert len(csv_lines) == 1
    csv_rows = list(csv.DictReader(io.StringIO(csv_lines[0])))

    json_rows = json.loads(_run_sweep(wav, json=True)[0])["sweep"]
    assert len(csv_rows) == len(json_rows)
    for csv_row, json_row in zip(csv_rows, json_rows):
        assert abs(float(csv_row["threshold"]) - json_row["threshold"]) <= 1e-9
        assert int(csv_row["num_segments"]) == json_row["num_segments"]
        assert int(csv_row["num_gaps"]) == json_row["num_gaps"]
        if json_row["num_gaps"] == 0:
            assert csv_row["min_gap_s"] == ""
        else:
            assert abs(float(csv_row["min_gap_s"]) - json_row["min_gap_s"]) <= 0.01


@pytest.mark.parametrize("wav", RECORDINGS, ids=[p.name for p in RECORDINGS])
def test_gv_vad_gap_sweep_emits_a_report_for_every_recording(wav: Path):
    """Every corpus recording produces a well-formed human sweep table (title +
    column header) and a parseable JSON payload with one row per swept threshold,
    each row's gap count one fewer than its segment count (or zero for a single
    region)."""
    lines = _run_sweep(wav)
    text = "\n".join(lines)
    assert "gap sweep" in text
    assert wav.name in text
    assert "min_gap" in text

    payload = json.loads(_run_sweep(wav, json=True)[0])
    assert payload["available"] is True
    assert payload["name"] == wav.name
    assert payload["axis"] == "threshold"
    assert len(payload["sweep"]) == 4
    for row in payload["sweep"]:
        if row["num_segments"] >= 2:
            assert row["num_gaps"] == row["num_segments"] - 1
            assert row["min_gap_s"] is not None
        else:
            assert row["num_gaps"] == 0
            assert row["min_gap_s"] is None
            assert row["max_gap_s"] is None
            assert row["mean_gap_s"] is None
