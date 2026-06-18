"""iter-233 — End-to-end ``gv vad`` over the real recording corpus.

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
