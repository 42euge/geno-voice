"""iter-335 — End-to-end ``gv vad-gap-diff`` over the real corpus.

The diff-side companion to :mod:`test_gv_vad_gap_sweep_cli` (iter-331) and
:mod:`test_gv_vad_gaps_cli` (iter-329). Those modules pin the single-setting
``gv vad-gaps`` surface and the N-value ``gv vad-gap-sweep`` sweep against the
REAL Silero segmenter; this one does the same for ``gv vad-gap-diff`` (iter-334),
the two-point degenerate that segments one WAV at two thresholds and reports how
the inter-segment SILENCE-gap distribution SHIFTS (signed min/mean/max gap and
total-silence deltas). Where ``gv vad-gaps`` answers "what are the pauses at THIS
setting?" and ``gv vad-gap-sweep`` "how does the shortest pause move across a
sweep?", ``gv vad-gap-diff`` answers "exactly how much does each gap aggregate
move between gate A and gate B?" — the headline being how the MIN gap shifts (the
floor above which raising ``--min-silence-ms`` / the live
``chat.vad.silence_duration`` starts merging two genuine turns into one).

The anchoring property mirrors the sweep's: each side's gap aggregates must equal
an independent ``gv vad-gaps --json`` run segmented at that side's exact gate, and
every reported delta must equal the difference of the two standalone gap reports
(b minus a) — proving the diff differences the same per-gate segmentation a
standalone ``gv vad-gaps`` produces, not a re-derived one. The 31s continuous
recording (which energy-VAD cannot split) is THE GATE here too: at the baseline
0.5 gate it must yield ≥2 segments, hence ≥1 measurable gap.

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


def _diff_args(wav: Path, **over):
    """Namespace for a ``gv vad-gap-diff`` run mirroring the parser defaults."""
    base = dict(
        wav=str(wav),
        threshold_a=0.5,
        threshold_b=0.7,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        json=False,
        csv=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _run_diff(wav: Path, **over) -> list[str]:
    lines: list[str] = []
    gv.cmd_vad_gap_diff(_diff_args(wav, **over), log=lines.append)
    return lines


def test_gv_vad_gap_diff_31s_has_a_measurable_gap_at_baseline():
    """THE GATE: diffing the 31s continuous recording from the baseline 0.5 gate
    must yield a side A with ≥2 segments and ≥1 inter-segment gap (energy-VAD
    cannot split this clip; Silero must), so the min-gap floor is measurable."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    payload = json.loads(_run_diff(wav, threshold_a=0.5, threshold_b=0.7, json=True)[0])
    assert payload["available"] is True
    assert payload["name"] == CONTINUOUS_31S
    assert payload["threshold_a"] == 0.5
    assert payload["threshold_b"] == 0.7
    assert payload["num_segments_a"] >= 2
    assert payload["num_gaps_a"] == payload["num_segments_a"] - 1
    assert payload["num_gaps_a"] >= 1
    assert payload["min_gap_s_a"] is not None


def test_gv_vad_gap_diff_each_side_matches_independent_vad_gaps():
    """The anchoring property: each side's gap aggregates must equal an
    independent ``gv vad-gaps --json`` run segmented at that side's exact gate —
    proving the diff differences the same per-gate segmentation a standalone
    ``gv vad-gaps`` produces, not a re-derived one."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    thr_a, thr_b = 0.3, 0.7
    diff = json.loads(_run_diff(wav, threshold_a=thr_a, threshold_b=thr_b, json=True)[0])

    anchor_a = json.loads(_run_gaps(wav, threshold=thr_a, json=True)[0])
    anchor_b = json.loads(_run_gaps(wav, threshold=thr_b, json=True)[0])

    for anchor, side in ((anchor_a, "a"), (anchor_b, "b")):
        assert diff[f"num_segments_{side}"] == anchor["num_segments"]
        assert diff[f"num_gaps_{side}"] == anchor["num_gaps"]
        assert diff[f"min_gap_s_{side}"] == anchor["min_gap_s"]
        assert diff[f"mean_gap_s_{side}"] == anchor["mean_gap_s"]
        assert diff[f"max_gap_s_{side}"] == anchor["max_gap_s"]
        assert diff[f"total_silence_s_{side}"] == anchor["total_silence_s"]


def test_gv_vad_gap_diff_deltas_equal_difference_of_two_gap_reports():
    """Every reported delta equals the difference of the two standalone
    ``gv vad-gaps`` reports (b minus a) — the diff is a faithful subtraction of
    two independent gap measurements, with a missing pause on either side
    yielding a ``null`` delta (a missing pause cannot be differenced)."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    thr_a, thr_b = 0.3, 0.7
    diff = json.loads(_run_diff(wav, threshold_a=thr_a, threshold_b=thr_b, json=True)[0])
    anchor_a = json.loads(_run_gaps(wav, threshold=thr_a, json=True)[0])
    anchor_b = json.loads(_run_gaps(wav, threshold=thr_b, json=True)[0])

    # Integer counts always difference.
    assert diff["num_segments_delta"] == anchor_b["num_segments"] - anchor_a["num_segments"]
    assert diff["num_gaps_delta"] == anchor_b["num_gaps"] - anchor_a["num_gaps"]
    # total_silence_s is always a float (0.0 when no gaps), so always differences.
    assert diff["total_silence_s_delta"] == pytest.approx(
        round(anchor_b["total_silence_s"] - anchor_a["total_silence_s"], 3)
    )

    # The gap aggregates: a delta is null whenever EITHER side has no pause.
    for key in ("min_gap_s", "mean_gap_s", "max_gap_s"):
        a, b = anchor_a[key], anchor_b[key]
        if a is None or b is None:
            assert diff[f"{key}_delta"] is None
        else:
            assert diff[f"{key}_delta"] == pytest.approx(round(b - a, 3))


def test_gv_vad_gap_diff_aggregates_self_consistent_per_side():
    """For each side that has ≥1 gap, the reported min ≤ mean ≤ max and
    total ≈ mean × count hold — the per-side aggregates are internally
    self-consistent, on real segmentations at both gates."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    diff = json.loads(_run_diff(wav, threshold_a=0.3, threshold_b=0.7, json=True)[0])
    saw_gap = False
    for side in ("a", "b"):
        if diff[f"num_gaps_{side}"] == 0:
            assert diff[f"min_gap_s_{side}"] is None
            assert diff[f"mean_gap_s_{side}"] is None
            assert diff[f"max_gap_s_{side}"] is None
            continue
        saw_gap = True
        assert diff[f"min_gap_s_{side}"] <= diff[f"mean_gap_s_{side}"] <= diff[f"max_gap_s_{side}"]
        assert (
            abs(diff[f"total_silence_s_{side}"]
                - diff[f"mean_gap_s_{side}"] * diff[f"num_gaps_{side}"])
            <= 0.05
        )
    assert saw_gap, "expected at least one side with a measurable gap"


def test_gv_vad_gap_diff_csv_byte_identical_to_two_value_gap_sweep():
    """``gv vad-gap-diff --csv`` over a pair (A, B) produces a table BYTE-IDENTICAL
    to a two-value ``gv vad-gap-sweep --csv`` over the same thresholds — the
    contract iter-313 holds between ``vad-diff`` and ``vad-sweep``, here on a real
    segmentation. A diff IS the two-point degenerate of a sweep."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    diff_csv = _run_diff(wav, threshold_a=0.3, threshold_b=0.7, csv=True)
    assert len(diff_csv) == 1

    sweep_lines: list[str] = []
    gv.cmd_vad_gap_sweep(
        argparse.Namespace(
            wav=str(wav),
            thresholds=[0.3, 0.7],
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
            csv=True,
        ),
        log=sweep_lines.append,
    )
    assert len(sweep_lines) == 1
    assert diff_csv[0] == sweep_lines[0]


def test_gv_vad_gap_diff_csv_matches_json():
    """``gv vad-gap-diff --csv`` describes the same two sides as ``--json`` over
    THE GATE recording — same thresholds, same segment/gap counts, same min gap —
    proving the two machine-readable surfaces agree."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    csv_lines = _run_diff(wav, threshold_a=0.3, threshold_b=0.7, csv=True)
    csv_rows = list(csv.DictReader(io.StringIO(csv_lines[0])))
    assert len(csv_rows) == 2

    diff = json.loads(_run_diff(wav, threshold_a=0.3, threshold_b=0.7, json=True)[0])
    for csv_row, side in zip(csv_rows, ("a", "b")):
        assert abs(float(csv_row["threshold"]) - diff[f"threshold_{side}"]) <= 1e-9
        assert int(csv_row["num_segments"]) == diff[f"num_segments_{side}"]
        assert int(csv_row["num_gaps"]) == diff[f"num_gaps_{side}"]
        if diff[f"num_gaps_{side}"] == 0:
            assert csv_row["min_gap_s"] == ""
        else:
            assert abs(float(csv_row["min_gap_s"]) - diff[f"min_gap_s_{side}"]) <= 0.01


def test_gv_vad_gap_diff_stricter_gate_never_shortens_min_gap():
    """Reading A → B as the gate TIGHTENS (0.3 → 0.7), the shortest surviving
    pause is non-decreasing: a stricter gate gates out marginal speech, which can
    only drop or merge adjacent regions (lengthening the shortest pause), never
    introduce a shorter one — the diff-side echo of the sweep's min-gap-rises
    trend. Asserted only when both sides have a measurable pause."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    diff = json.loads(_run_diff(wav, threshold_a=0.3, threshold_b=0.7, json=True)[0])
    if diff["min_gap_s_a"] is None or diff["min_gap_s_b"] is None:
        pytest.skip("a side had fewer than 2 segments — no min gap to compare")
    assert diff["min_gap_s_b"] >= diff["min_gap_s_a"] - 1e-9, (
        f"stricter gate shortened the min gap: "
        f"{diff['min_gap_s_a']} → {diff['min_gap_s_b']}"
    )


@pytest.mark.parametrize("wav", RECORDINGS, ids=[p.name for p in RECORDINGS])
def test_gv_vad_gap_diff_emits_a_report_for_every_recording(wav: Path):
    """Every corpus recording produces a well-formed human diff (title + the
    min-gap knob hint) and a parseable JSON payload carrying both sides, each
    side's gap count one fewer than its segment count (or zero for a single
    region)."""
    lines = _run_diff(wav)
    text = "\n".join(lines)
    assert "gap diff" in text
    assert wav.name in text
    assert "--min-silence-ms" in text

    payload = json.loads(_run_diff(wav, json=True)[0])
    assert payload["available"] is True
    assert payload["name"] == wav.name
    for side in ("a", "b"):
        if payload[f"num_segments_{side}"] >= 2:
            assert payload[f"num_gaps_{side}"] == payload[f"num_segments_{side}"] - 1
            assert payload[f"min_gap_s_{side}"] is not None
        else:
            assert payload[f"num_gaps_{side}"] == 0
            assert payload[f"min_gap_s_{side}"] is None
            assert payload[f"mean_gap_s_{side}"] is None
            assert payload[f"max_gap_s_{side}"] is None
