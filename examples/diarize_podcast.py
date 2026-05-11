#!/usr/bin/env python3
"""
Speaker diarization on podcast audio using pyannote.audio.

Separates therapist vs client, then creates per-speaker transcript
segments for training data.

Usage:
    .venv/bin/python examples/diarize_podcast.py [--duration 300]
"""

import argparse
import json
import sys
import wave
from pathlib import Path

from pyannote.audio import Pipeline

AUDIO = Path(__file__).parent.parent / "test-data" / "esther-perel-session.wav"
OUT = Path(__file__).parent.parent / "test-data" / "diarization"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=60)
    parser.add_argument("--duration", type=int, default=300)
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)

    # Extract segment to process
    segment_path = OUT / "segment.wav"
    with wave.open(str(AUDIO), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        w.setpos(int(args.start * sr))
        frames = w.readframes(int(args.duration * sr))
    with wave.open(str(segment_path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(sw)
        w.setframerate(sr)
        w.writeframes(frames)
    print(f"Segment: {args.start}s - {args.start + args.duration}s ({segment_path})")

    # Run diarization
    print("Loading pyannote pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    print("Running diarization...")
    diarization = pipeline(str(segment_path))

    # Process results
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "duration": round(turn.end - turn.start, 2),
        })

    # Summary
    speakers = {}
    for s in segments:
        sp = s["speaker"]
        if sp not in speakers:
            speakers[sp] = {"count": 0, "total_secs": 0}
        speakers[sp]["count"] += 1
        speakers[sp]["total_secs"] += s["duration"]

    print(f"\n{len(segments)} segments, {len(speakers)} speakers:")
    for sp, info in sorted(speakers.items(), key=lambda x: -x[1]["total_secs"]):
        print(f"  {sp}: {info['count']} segments, {info['total_secs']:.0f}s total")

    # Save
    results = {"speakers": speakers, "segments": segments}
    (OUT / "diarization.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT / 'diarization.json'}")

    # Show first 10 segments
    print("\nFirst 10 turns:")
    for s in segments[:10]:
        print(f"  [{s['start']:6.1f}s - {s['end']:6.1f}s] {s['speaker']} ({s['duration']:.1f}s)")


if __name__ == "__main__":
    main()
