"""iter-329 — End-to-end ``gv vad-gaps`` over the real corpus.

The silence-side companion to ``test_gv_vad_cli.py``: that module drives the
speech-reporting ``gv vad`` family through the CLI with the REAL Silero
segmenter; this one does the same for ``gv vad-gaps`` (iter-328), the
inter-segment silence-gap analysis surface. Where ``gv vad`` reports where the
SPEECH is, ``gv vad-gaps`` reports where the SILENCE is — the pauses BETWEEN
consecutive speech regions, the gap distribution an operator reads to choose
the end-of-turn hangover (``--min-silence-ms`` / the live
``chat.vad.silence_duration``, iter-020).

The anchoring property: the gaps ``gv vad-gaps`` reports must be exactly the
pauses between the segments an independent ``gv vad --json`` run produces at the
same knobs — proving the new surface segments once with the real engine and
differences the same segmentation, not a re-derived one. The 31s continuous
recording (which energy-VAD cannot split) is THE GATE here too: it must yield
≥2 segments, hence ≥1 measurable gap.

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


def _run_vad(wav: Path, **over) -> list[str]:
    """Drive cmd_vad with the REAL segmenter, capturing output lines."""
    lines: list[str] = []
    gv.cmd_vad(_vad_args(wav, **over), log=lines.append)
    return lines


def _gaps_args(wav: Path, **over):
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


def test_gv_vad_gaps_31s_reports_at_least_one_gap():
    """THE GATE: the 31s continuous recording splits to ≥2 regions, so it has
    ≥1 measurable inter-segment pause via gv vad-gaps."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    payload = json.loads(_run_gaps(wav, json=True)[0])
    assert payload["available"] is True
    assert payload["name"] == CONTINUOUS_31S
    assert payload["num_segments"] >= 2
    assert payload["num_gaps"] == payload["num_segments"] - 1
    assert payload["num_gaps"] >= 1
    assert len(payload["gaps"]) == payload["num_gaps"]


def test_gv_vad_gaps_match_pauses_between_independent_vad_segments():
    """The anchoring property: each gap ``gv vad-gaps`` reports must equal the
    pause between the corresponding consecutive segments an independent
    ``gv vad --json`` run produces at the same knobs — proving the surface
    segments once with the real engine and differences that exact segmentation,
    not a re-derived one."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    vad = json.loads(_run_vad(wav, json=True)[0])
    segs = vad["segments"]
    assert len(segs) >= 2

    # Reconstruct the inter-segment gaps from the vad segmentation. The segments
    # come out start-sorted, so consecutive differencing matches the core.
    expected = [
        round(max(0.0, segs[i]["start_s"] - segs[i - 1]["end_s"]), 3)
        for i in range(1, len(segs))
    ]

    gaps = json.loads(_run_gaps(wav, json=True)[0])
    actual = [g["gap_s"] for g in gaps["gaps"]]
    assert actual == expected
    # Each gap names the (1-based) segment it follows and that segment's end.
    for i, g in enumerate(gaps["gaps"], start=1):
        assert g["index"] == i
        assert g["after_segment"] == i
        assert abs(g["after_segment_end_s"] - round(segs[i - 1]["end_s"], 3)) <= 0.01


def test_gv_vad_gaps_aggregates_are_consistent_with_the_gap_list():
    """Over THE GATE recording the reported min/mean/max/total aggregates must
    be derivable from the per-gap list — no stale or independently-computed
    summary."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    payload = json.loads(_run_gaps(wav, json=True)[0])
    gaps = [g["gap_s"] for g in payload["gaps"]]
    assert gaps  # GATE guarantees ≥1 gap
    assert payload["min_gap_s"] == min(gaps)
    assert payload["max_gap_s"] == max(gaps)
    assert abs(payload["mean_gap_s"] - sum(gaps) / len(gaps)) <= 0.01
    assert abs(payload["total_silence_s"] - sum(gaps)) <= 0.01


def test_gv_vad_gaps_silence_plus_speech_spans_the_segmented_window():
    """Physical invariant over THE GATE recording: total inter-segment silence
    plus total speech equals the span from the first segment's start to the last
    segment's end — every instant inside the segmented window is either speech
    or an inter-segment pause."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    vad = json.loads(_run_vad(wav, json=True)[0])
    segs = vad["segments"]
    assert len(segs) >= 2
    window = segs[-1]["end_s"] - segs[0]["start_s"]

    payload = json.loads(_run_gaps(wav, json=True)[0])
    # speech_s from vad + total_silence_s from gaps tile the window exactly.
    assert abs((vad["speech_s"] + payload["total_silence_s"]) - window) <= 0.05


def test_gv_vad_gaps_longer_hangover_never_grows_gap_count():
    """A longer ``--min-silence-ms`` can only merge adjacent regions (dropping
    the short pause between them), never split one, so the number of
    inter-segment gaps is monotone non-increasing as the hangover rises — the
    silence-side echo of the vad-sweep segment-count monotonicity."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    counts = [
        json.loads(_run_gaps(wav, min_silence_ms=ms, json=True)[0])["num_gaps"]
        for ms in (200.0, 400.0, 800.0, 1600.0)
    ]
    for lo, hi in zip(counts, counts[1:]):
        assert hi <= lo, f"gap count rose across rising hangovers: {counts}"


def test_gv_vad_gaps_csv_matches_json():
    """``gv vad-gaps --csv`` describes the same per-gap rows as ``--json`` over
    THE GATE recording — same indices, same after-segment anchors, same gap
    seconds — proving the two machine-readable surfaces agree."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    csv_lines = _run_gaps(wav, csv=True)
    assert len(csv_lines) == 1
    csv_rows = list(csv.DictReader(io.StringIO(csv_lines[0])))

    json_gaps = json.loads(_run_gaps(wav, json=True)[0])["gaps"]
    assert len(csv_rows) == len(json_gaps)
    for csv_row, json_gap in zip(csv_rows, json_gaps):
        assert int(csv_row["index"]) == json_gap["index"]
        assert int(csv_row["after_segment"]) == json_gap["after_segment"]
        assert abs(
            float(csv_row["after_segment_end_s"]) - json_gap["after_segment_end_s"]
        ) <= 0.01
        assert abs(float(csv_row["gap_s"]) - json_gap["gap_s"]) <= 0.01


@pytest.mark.parametrize("wav", RECORDINGS, ids=[p.name for p in RECORDINGS])
def test_gv_vad_gaps_emits_a_report_for_every_recording(wav: Path):
    """Every corpus recording produces a well-formed human report (header +
    counts) and a parseable JSON payload whose gap count is one fewer than its
    segment count (or zero when the recording yields a single region)."""
    lines = _run_gaps(wav)
    text = "\n".join(lines)
    assert wav.name in text
    assert "segments:" in text
    assert "gaps:" in text

    payload = json.loads(_run_gaps(wav, json=True)[0])
    assert payload["available"] is True
    assert payload["name"] == wav.name
    if payload["num_segments"] >= 2:
        assert payload["num_gaps"] == payload["num_segments"] - 1
        assert payload["min_gap_s"] is not None
    else:
        assert payload["num_gaps"] == 0
        assert payload["min_gap_s"] is None
        assert payload["max_gap_s"] is None
        assert payload["mean_gap_s"] is None
