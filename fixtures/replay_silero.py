#!/usr/bin/env python3
"""iter-231 — Headless Silero-VAD replay harness over the real recording corpus.

The sibling ``replay_vad.py`` replays the energy-RMS state machine. This harness
runs the **Silero neural VAD** (``vad/silero.py``) over every recording in
``fixtures/recordings/`` and reports the speech segments per file — the headless
proof that Silero succeeds where energy-VAD fails.

GROUND TRUTH that motivates this (the steering's finding):
    ``voice-20260618-110355.wav`` (31s continuous speech) collapses to a SINGLE
    segment under energy-RMS VAD no matter how threshold/silence/hysteresis are
    tuned — the in-speech noise floor (~0.016 RMS) sits too close to the speech
    median (~0.023), so the utterance never closes. Silero distinguishes speech
    from room-tone regardless of the energy floor and splits it into multiple
    sensible regions. This harness measures exactly that, with no mic and no GUI.

Usage (CLI):
    python fixtures/replay_silero.py                    # all recordings, defaults
    python fixtures/replay_silero.py --threshold 0.5 --min-silence-ms 800
    python fixtures/replay_silero.py --json             # machine-readable
    python fixtures/replay_silero.py --compare          # Silero vs energy-VAD counts

Usage (library):
    from fixtures.replay_silero import replay_silero_all
    results = replay_silero_all()  # List[SileroResult]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vad.silero import (  # noqa: E402
    SileroParams,
    SileroResult,
    segment_recording,
    silero_available,
)

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"


def replay_silero_all(
    recordings_dir: Path = RECORDINGS_DIR,
    params: Optional[SileroParams] = None,
    model=None,
) -> List[SileroResult]:
    """Segment every ``*.wav`` in ``recordings_dir`` through Silero VAD.

    The model is loaded once (via ``segment_recording`` lazily, or pass
    ``model`` to reuse one) and applied to each recording in sorted order.
    """
    params = params or SileroParams()
    from vad.silero import load_model

    if model is None:
        model = load_model()
    return [
        segment_recording(p, params=params, model=model)
        for p in sorted(recordings_dir.glob("*.wav"))
    ]


def _energy_onsets(wav_path: Path) -> int:
    """How many segments the energy-RMS state machine finds (for --compare)."""
    from fixtures.replay_vad import VadParams, replay_recording

    return replay_recording(wav_path, VadParams(threshold=0.006)).onsets


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay recordings through Silero neural VAD."
    )
    parser.add_argument("--threshold", type=float, default=SileroParams.threshold)
    parser.add_argument("--min-speech-ms", type=float, default=SileroParams.min_speech_ms)
    parser.add_argument("--min-silence-ms", type=float, default=SileroParams.min_silence_ms)
    parser.add_argument("--speech-pad-ms", type=float, default=SileroParams.speech_pad_ms)
    parser.add_argument("--dir", type=Path, default=RECORDINGS_DIR)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="show Silero segment counts beside the energy-VAD onset counts",
    )
    args = parser.parse_args(argv)

    if not silero_available():
        print(
            "silero-vad is not installed; cannot replay. "
            "Install with: pip install silero-vad"
        )
        return 3

    params = SileroParams(
        threshold=args.threshold,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
    )

    results = replay_silero_all(args.dir, params)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0

    if not results:
        print(f"No recordings found in {args.dir}")
        return 1

    print(
        f"Silero VAD replay — threshold={params.threshold} "
        f"min_speech={params.min_speech_ms}ms min_silence={params.min_silence_ms}ms "
        f"speech_pad={params.speech_pad_ms}ms"
    )
    print("-" * 100)
    if args.compare:
        print(f"{'recording':<32} {'silero_segs':>11} {'energy_onsets':>14}")
        for r in results:
            energy = _energy_onsets(args.dir / r.name)
            flag = "  <-- energy-VAD under-segments" if r.num_segments > energy else ""
            print(f"{r.name:<32} {r.num_segments:>11} {energy:>14}{flag}")
    else:
        for r in results:
            print(r.summary_line())
    print("-" * 100)
    total = sum(r.num_segments for r in results)
    print(f"{total} total speech segments across {len(results)} recordings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
