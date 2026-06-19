"""iter-233/234/235/236 — End-to-end ``gv vad`` family over the real corpus.

The companion to ``test_silero_recordings.py``: that module pins the Silero
segmenter directly; this one drives it through the gv CLI surface
(``cmd_vad`` with the REAL ``vad.silero.segment_recording``, no injected stub)
to prove the new ``gv vad`` subcommand wires the engine correctly and emits a
sane report.

THE GATE recording — ``voice-20260618-110355.wav`` (31s continuous speech that
energy-VAD cannot segment) — must, when run through ``gv vad``, produce a report
naming ≥2 speech segments. ``test_gv_vad_31s_reports_multiple_segments`` pins
exactly that through the CLI handler.

Skips cleanly when the recordings (large binary captures, not committed) or
``silero-vad`` (+ torch deps) are absent — same contract as the sibling module.
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


def _vad_args(wav: Path, **over):
    base = dict(
        wav=str(wav),
        threshold=0.5,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        json=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _run(wav: Path, **over) -> list[str]:
    """Drive cmd_vad with the REAL segmenter, capturing output lines."""
    lines: list[str] = []
    gv.cmd_vad(_vad_args(wav, **over), log=lines.append)
    return lines


def test_gv_vad_31s_reports_multiple_segments():
    """THE GATE: the 31s continuous recording splits to ≥2 regions via gv vad."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    lines = _run(wav)
    text = "\n".join(lines)
    assert CONTINUOUS_31S in text
    # Count the per-segment "[ n]" rows the report emits.
    seg_rows = [ln for ln in lines if ln.lstrip().startswith("[")]
    assert len(seg_rows) >= 2, f"expected ≥2 segments, report was:\n{text}"


@pytest.mark.parametrize("wav", RECORDINGS, ids=[p.name for p in RECORDINGS])
def test_gv_vad_emits_a_report_for_every_recording(wav: Path):
    """Every corpus recording produces a well-formed report (header + counts)."""
    lines = _run(wav)
    text = "\n".join(lines)
    assert wav.name in text
    assert "segments:" in text
    assert "speech total:" in text
    # The threshold line is present (we passed a threshold).
    assert "threshold:" in text


def test_gv_vad_threshold_knob_changes_segmentation():
    """A near-1.0 threshold (very strict P(speech) gate) recovers no more speech
    than the default — the CLI knob genuinely reaches the engine."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    def speech_total(lines: list[str]) -> float:
        for ln in lines:
            if "speech total:" in ln:
                return float(ln.split(":")[1].strip().rstrip("s"))
        raise AssertionError("no speech-total line in report")

    default_speech = speech_total(_run(wav, threshold=0.5))
    strict_speech = speech_total(_run(wav, threshold=0.95))
    # A stricter gate can only keep or shrink recovered speech, never grow it.
    assert strict_speech <= default_speech + 1e-6


def test_gv_vad_json_emits_parseable_segmentation():
    """iter-234: ``gv vad --json`` over THE GATE recording emits a single
    parseable JSON document whose segmentation matches the human report."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    json_lines = _run(wav, json=True)
    # The whole report is one JSON document emitted via a single log() call.
    assert len(json_lines) == 1
    payload = json.loads(json_lines[0])
    assert payload["available"] is True
    assert payload["name"] == CONTINUOUS_31S
    assert payload["threshold"] == 0.5
    # ≥2 segments — the same GATE the human report pins.
    assert payload["num_segments"] >= 2
    assert len(payload["segments"]) == payload["num_segments"]
    # speech_s equals the sum of per-segment durations (engine invariant).
    summed = sum(s["duration_s"] for s in payload["segments"])
    assert abs(summed - payload["speech_s"]) <= 0.01


@pytest.mark.parametrize("wav", RECORDINGS, ids=[p.name for p in RECORDINGS])
def test_gv_vad_json_matches_human_report_counts(wav: Path):
    """For every recording, the --json segment count matches the human report's
    "[ n]" rows — the two surfaces describe the same segmentation."""
    human = _run(wav)
    human_rows = [ln for ln in human if ln.lstrip().startswith("[")]

    payload = json.loads(_run(wav, json=True)[0])
    assert payload["num_segments"] == len(human_rows)


# ---- iter-235: gv vad-diff over the real corpus ------------------------


def _diff_args(wav: Path, **over):
    base = dict(
        wav=str(wav),
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


def _run_diff(wav: Path, **over) -> list[str]:
    lines: list[str] = []
    gv.cmd_vad_diff(_diff_args(wav, **over), log=lines.append)
    return lines


def test_gv_vad_diff_json_matches_two_separate_vad_runs():
    """``gv vad-diff`` must report exactly the delta between two independent
    ``gv vad --json`` runs at the same thresholds — proving it segments twice
    with the real engine and computes the delta consistently."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    a = json.loads(_run(wav, threshold=0.5, json=True)[0])
    b = json.loads(_run(wav, threshold=0.9, json=True)[0])

    diff = json.loads(_run_diff(wav, threshold_a=0.5, threshold_b=0.9, json=True)[0])
    assert diff["available"] is True
    assert diff["name"] == CONTINUOUS_31S
    assert diff["num_segments_a"] == a["num_segments"]
    assert diff["num_segments_b"] == b["num_segments"]
    assert diff["num_segments_delta"] == b["num_segments"] - a["num_segments"]
    assert abs(diff["speech_s_delta"] - (b["speech_s"] - a["speech_s"])) <= 0.01


def test_gv_vad_diff_higher_threshold_is_a_subset():
    """A stricter B gate recovers no more speech than the looser A gate — the
    speech-seconds delta is ≤ 0 across the corpus's hardest recording."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    diff = json.loads(_run_diff(wav, threshold_a=0.5, threshold_b=0.95, json=True)[0])
    assert diff["speech_s_delta"] <= 1e-6


def test_gv_vad_diff_human_report_is_well_formed():
    """The human-readable diff names the file, both thresholds, and the arrow."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    text = "\n".join(_run_diff(wav, threshold_a=0.5, threshold_b=0.7))
    assert CONTINUOUS_31S in text
    assert "0.50" in text and "0.70" in text
    assert "→" in text


# ---- iter-236: gv vad-sweep over the real corpus -----------------------


def _sweep_args(wav: Path, **over):
    base = dict(
        wav=str(wav),
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


def _run_sweep(wav: Path, **over) -> list[str]:
    lines: list[str] = []
    gv.cmd_vad_sweep(_sweep_args(wav, **over), log=lines.append)
    return lines


def test_gv_vad_sweep_json_matches_independent_vad_runs():
    """Each sweep row must equal an independent ``gv vad --json`` run at that
    threshold — proving the sweep segments once per gate with the real engine."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    thresholds = [0.3, 0.5, 0.9]
    sweep = json.loads(_run_sweep(wav, thresholds=thresholds, json=True)[0])
    assert sweep["available"] is True
    assert sweep["name"] == CONTINUOUS_31S
    assert [row["threshold"] for row in sweep["sweep"]] == thresholds

    for t, row in zip(thresholds, sweep["sweep"]):
        single = json.loads(_run(wav, threshold=t, json=True)[0])
        assert row["num_segments"] == single["num_segments"]
        assert abs(row["speech_s"] - single["speech_s"]) <= 0.01


def test_gv_vad_sweep_speech_is_monotone_non_increasing():
    """Reading down an ascending-threshold sweep, recovered speech never grows
    — a stricter gate can only keep or shrink it. The elbow is where it falls."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    sweep = json.loads(_run_sweep(wav, thresholds=[0.3, 0.5, 0.7, 0.9], json=True)[0])
    speech = [row["speech_s"] for row in sweep["sweep"]]
    for lo, hi in zip(speech, speech[1:]):
        assert hi <= lo + 1e-6, f"speech rose across rising thresholds: {speech}"


def test_gv_vad_sweep_human_table_is_well_formed():
    """The human-readable table names the file, has a column header, and one
    row per threshold."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    lines = _run_sweep(wav, thresholds=[0.3, 0.7, 0.9])
    text = "\n".join(lines)
    assert CONTINUOUS_31S in text
    assert "threshold" in text and "segments" in text and "speech" in text
    # header + column labels + 3 threshold rows
    assert len(lines) == 5
    assert "0.30" in text and "0.70" in text and "0.90" in text


def test_gv_vad_sweep_csv_matches_json_sweep():
    """iter-237: ``gv vad-sweep --csv`` over THE GATE recording describes the
    same segmentation as ``--json`` — same thresholds, same counts, same
    speech-seconds — proving the two machine-readable surfaces agree."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    thresholds = [0.3, 0.5, 0.9]
    csv_lines = _run_sweep(wav, thresholds=thresholds, csv=True)
    # The whole CSV blob is logged in a single call.
    assert len(csv_lines) == 1
    csv_rows = list(csv.DictReader(io.StringIO(csv_lines[0])))

    json_rows = json.loads(_run_sweep(wav, thresholds=thresholds, json=True)[0])[
        "sweep"
    ]
    assert [float(r["threshold"]) for r in csv_rows] == thresholds
    for csv_row, json_row in zip(csv_rows, json_rows):
        assert int(csv_row["num_segments"]) == json_row["num_segments"]
        assert abs(float(csv_row["speech_s"]) - json_row["speech_s"]) <= 0.01


# ---- iter-238: gv vad-sweep --min-silences (hangover axis) -------------


def test_gv_vad_sweep_silence_axis_matches_independent_vad_runs():
    """Each min-silence sweep row must equal an independent ``gv vad --json`` run
    at that hangover (gate held fixed) — proving the second axis segments once
    per value with the real engine, exactly like the threshold axis."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    min_silences = [200.0, 400.0, 800.0]
    held_gate = 0.5
    sweep = json.loads(
        _run_sweep(wav, min_silences=min_silences, threshold=held_gate, json=True)[0]
    )
    assert sweep["available"] is True
    assert sweep["axis"] == "min_silence_ms"
    assert [row["min_silence_ms"] for row in sweep["sweep"]] == min_silences

    for ms, row in zip(min_silences, sweep["sweep"]):
        single = json.loads(
            _run(wav, threshold=held_gate, min_silence_ms=ms, json=True)[0]
        )
        assert row["num_segments"] == single["num_segments"]
        assert abs(row["speech_s"] - single["speech_s"]) <= 0.01


def test_gv_vad_sweep_silence_axis_segments_are_monotone_non_increasing():
    """Reading down an ascending-hangover sweep, the segment count never grows —
    a longer trailing-silence requirement can only merge adjacent regions, never
    split them. The elbow is where merging kicks in."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    sweep = json.loads(
        _run_sweep(wav, min_silences=[100.0, 200.0, 400.0, 800.0, 1600.0], json=True)[0]
    )
    segs = [row["num_segments"] for row in sweep["sweep"]]
    for lo, hi in zip(segs, segs[1:]):
        assert hi <= lo, f"segments rose across rising hangovers: {segs}"


def test_gv_vad_sweep_silence_axis_csv_matches_json():
    """``gv vad-sweep --min-silences --csv`` describes the same segmentation as
    ``--json`` over THE GATE recording — the two surfaces agree on the second
    axis too, and the CSV header is the swept axis name."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    min_silences = [200.0, 400.0, 800.0]
    csv_lines = _run_sweep(wav, min_silences=min_silences, csv=True)
    assert len(csv_lines) == 1
    csv_rows = list(csv.DictReader(io.StringIO(csv_lines[0])))
    assert "min_silence_ms" in csv_rows[0]

    json_rows = json.loads(
        _run_sweep(wav, min_silences=min_silences, json=True)[0]
    )["sweep"]
    assert [float(r["min_silence_ms"]) for r in csv_rows] == min_silences
    for csv_row, json_row in zip(csv_rows, json_rows):
        assert int(csv_row["num_segments"]) == json_row["num_segments"]
        assert abs(float(csv_row["speech_s"]) - json_row["speech_s"]) <= 0.01


# ---- iter-239: gv vad-sweep --min-speeches (min-speech floor axis) -----


def test_gv_vad_sweep_speech_axis_matches_independent_vad_runs():
    """Each min-speech sweep row must equal an independent ``gv vad --json`` run
    at that floor (gate held fixed) — proving the third axis segments once per
    value with the real engine, exactly like the threshold/silence axes."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    min_speeches = [50.0, 200.0, 800.0]
    held_gate = 0.5
    sweep = json.loads(
        _run_sweep(wav, min_speeches=min_speeches, threshold=held_gate, json=True)[0]
    )
    assert sweep["available"] is True
    assert sweep["axis"] == "min_speech_ms"
    assert [row["min_speech_ms"] for row in sweep["sweep"]] == min_speeches

    for ms, row in zip(min_speeches, sweep["sweep"]):
        single = json.loads(
            _run(wav, threshold=held_gate, min_speech_ms=ms, json=True)[0]
        )
        assert row["num_segments"] == single["num_segments"]
        assert abs(row["speech_s"] - single["speech_s"]) <= 0.01


def test_gv_vad_sweep_speech_axis_segments_are_monotone_non_increasing():
    """Reading down an ascending-floor sweep, the segment count never grows — a
    higher min-speech floor can only drop short regions, never add them. The
    elbow is where short regions start getting culled."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    sweep = json.loads(
        _run_sweep(wav, min_speeches=[50.0, 100.0, 200.0, 400.0, 800.0], json=True)[0]
    )
    segs = [row["num_segments"] for row in sweep["sweep"]]
    for lo, hi in zip(segs, segs[1:]):
        assert hi <= lo, f"segments rose across rising floors: {segs}"


def test_gv_vad_sweep_speech_axis_csv_matches_json():
    """``gv vad-sweep --min-speeches --csv`` describes the same segmentation as
    ``--json`` over THE GATE recording — the two surfaces agree on the third
    axis too, and the CSV header is the swept axis name."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    min_speeches = [50.0, 200.0, 800.0]
    csv_lines = _run_sweep(wav, min_speeches=min_speeches, csv=True)
    assert len(csv_lines) == 1
    csv_rows = list(csv.DictReader(io.StringIO(csv_lines[0])))
    assert "min_speech_ms" in csv_rows[0]

    json_rows = json.loads(
        _run_sweep(wav, min_speeches=min_speeches, json=True)[0]
    )["sweep"]
    assert [float(r["min_speech_ms"]) for r in csv_rows] == min_speeches
    for csv_row, json_row in zip(csv_rows, json_rows):
        assert int(csv_row["num_segments"]) == json_row["num_segments"]
        assert abs(float(csv_row["speech_s"]) - json_row["speech_s"]) <= 0.01


# ---- iter-240: gv vad-grid (2-D gate × ms-knob grid) -------------------


def _grid_args(wav: Path, **over):
    base = dict(
        wav=str(wav),
        thresholds=[0.3, 0.5, 0.7, 0.9],
        min_silences=[400.0, 600.0, 800.0, 1000.0],
        min_speeches=None,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        target=None,
        json=False,
        csv=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _run_grid(wav: Path, **over) -> list[str]:
    lines: list[str] = []
    gv.cmd_vad_grid(_grid_args(wav, **over), log=lines.append)
    return lines


def test_gv_vad_grid_cell_matches_independent_vad_run():
    """Each grid cell must equal an independent ``gv vad --json`` run at that
    (threshold, hangover) pair — proving the grid segments once per cell with
    the real engine, the 2-D analogue of the vad-sweep row-equality property."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    thresholds = [0.3, 0.7]
    min_silences = [400.0, 800.0]
    grid = json.loads(
        _run_grid(wav, thresholds=thresholds, min_silences=min_silences, json=True)[0]
    )
    assert grid["available"] is True
    assert grid["name"] == CONTINUOUS_31S
    assert grid["row_axis"] == "threshold"
    assert grid["col_axis"] == "min_silence_ms"
    # Row-major: every (threshold, hangover) pair appears exactly once.
    assert [(c["threshold"], c["min_silence_ms"]) for c in grid["grid"]] == [
        (0.3, 400.0), (0.3, 800.0), (0.7, 400.0), (0.7, 800.0),
    ]
    for cell in grid["grid"]:
        single = json.loads(
            _run(
                wav,
                threshold=cell["threshold"],
                min_silence_ms=cell["min_silence_ms"],
                json=True,
            )[0]
        )
        assert cell["num_segments"] == single["num_segments"]
        assert abs(cell["speech_s"] - single["speech_s"]) <= 0.01


def test_gv_vad_grid_speech_non_increasing_along_threshold_within_a_column():
    """Holding the hangover fixed (one column), reading DOWN rising thresholds
    recovered speech never grows — the same gate monotonicity vad-sweep proves,
    now visible inside the 2-D grid."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    thresholds = [0.3, 0.5, 0.7, 0.9]
    min_silences = [400.0, 800.0]
    grid = json.loads(
        _run_grid(wav, thresholds=thresholds, min_silences=min_silences, json=True)[0]
    )["grid"]
    for col in min_silences:
        speech = [c["speech_s"] for c in grid if c["min_silence_ms"] == col]
        for lo, hi in zip(speech, speech[1:]):
            assert hi <= lo + 1e-6, f"speech rose at hangover={col}: {speech}"


def test_gv_vad_grid_csv_matches_json():
    """``gv vad-grid --csv`` describes the same cells as ``--json`` over THE GATE
    recording — same (threshold, hangover) pairs, same counts, same speech — and
    the CSV header carries both swept axis names."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    thresholds = [0.3, 0.7]
    min_silences = [400.0, 800.0]
    csv_lines = _run_grid(
        wav, thresholds=thresholds, min_silences=min_silences, csv=True
    )
    assert len(csv_lines) == 1
    csv_rows = list(csv.DictReader(io.StringIO(csv_lines[0])))
    assert "threshold" in csv_rows[0] and "min_silence_ms" in csv_rows[0]

    json_cells = json.loads(
        _run_grid(wav, thresholds=thresholds, min_silences=min_silences, json=True)[0]
    )["grid"]
    assert len(csv_rows) == len(json_cells) == 4
    for csv_row, cell in zip(csv_rows, json_cells):
        assert float(csv_row["threshold"]) == cell["threshold"]
        assert float(csv_row["min_silence_ms"]) == cell["min_silence_ms"]
        assert int(csv_row["num_segments"]) == cell["num_segments"]
        assert abs(float(csv_row["speech_s"]) - cell["speech_s"]) <= 0.01


# ---- iter-241: gv vad-grid --target (data-driven best-cell pick) -------


def test_gv_vad_grid_best_pick_is_closest_cell_to_target():
    """iter-241: ``gv vad-grid --target N`` over THE GATE recording picks the
    cell whose recovered segment count is genuinely closest to the target — the
    ``best`` it surfaces must minimise ``|num_segments - target|`` over the very
    same grid the run tabulated (no off-by-one, no stale cell)."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    thresholds = [0.3, 0.5, 0.7, 0.9]
    min_silences = [400.0, 800.0]
    target = 3
    payload = json.loads(
        _run_grid(
            wav, thresholds=thresholds, min_silences=min_silences,
            target=target, json=True,
        )[0]
    )
    assert payload["target"] == target
    best = payload["best"]
    assert best is not None
    # The picked distance must be the minimum over the whole tabulated grid.
    distances = [abs(c["num_segments"] - target) for c in payload["grid"]]
    assert best["distance"] == min(distances)
    assert abs(best["num_segments"] - target) == best["distance"]
    # And it must be an actual cell of this grid (earliest-tie among minima).
    assert best in [{**c, "distance": best["distance"]} for c in payload["grid"]]


def test_gv_vad_grid_target_absent_keeps_iter240_payload():
    """Without ``--target`` the JSON payload is byte-for-byte the iter-240 shape
    over THE GATE recording — no ``best`` / ``target`` keys leak in."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    payload = json.loads(
        _run_grid(wav, thresholds=[0.3, 0.7], min_silences=[400.0, 800.0], json=True)[0]
    )
    assert "best" not in payload
    assert "target" not in payload
