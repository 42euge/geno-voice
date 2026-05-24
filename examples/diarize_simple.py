#!/usr/bin/env python3
"""
Simple speaker diarization using resemblyzer (no gated models).

Splits podcast audio into segments, computes speaker embeddings,
clusters into 2-3 speakers, then produces per-speaker transcript
chunks for training data.

Usage:
    .venv/bin/python examples/diarize_simple.py test-data/esther-perel-session.wav --start 120 --duration 300
"""

import argparse
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

SAMPLE_RATE = 16000
CHUNK_SECS = 8
OUT = Path(__file__).parent.parent / "test-data" / "diarization"


def load_segment(path, start, duration):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        w.setpos(int(start * sr))
        frames = w.readframes(int(duration * sr))
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * SAMPLE_RATE / sr))
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="WAV file")
    parser.add_argument("--start", type=int, default=60)
    parser.add_argument("--duration", type=int, default=300)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.audio} from {args.start}s for {args.duration}s...")
    audio = load_segment(args.audio, args.start, args.duration)
    print(f"Audio: {len(audio)} samples ({len(audio)/SAMPLE_RATE:.0f}s)")

    # Split into chunks
    chunk_samples = CHUNK_SECS * SAMPLE_RATE
    chunks = []
    for i in range(0, len(audio), chunk_samples):
        c = audio[i:i + chunk_samples]
        if len(c) < SAMPLE_RATE:
            break
        chunks.append(c)
    print(f"Chunks: {len(chunks)}")

    # Compute speaker embeddings for each chunk
    print("Loading speaker encoder...")
    encoder = VoiceEncoder()

    print("Computing embeddings...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        wav = preprocess_wav(chunk, SAMPLE_RATE)
        if len(wav) < SAMPLE_RATE * 0.5:
            embeddings.append(None)
            continue
        embed = encoder.embed_utterance(wav)
        embeddings.append(embed)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(chunks)}")

    # Filter out None embeddings
    valid = [(i, e) for i, e in enumerate(embeddings) if e is not None]
    if len(valid) < 2:
        print("Not enough valid chunks for clustering")
        return

    # Cluster into 2 speakers (therapist + client)
    from sklearn.cluster import KMeans
    X = np.array([e for _, e in valid])
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
    labels = kmeans.labels_

    # Assign speaker labels
    speaker_map = {}
    for idx, (chunk_idx, _) in enumerate(valid):
        speaker_map[chunk_idx] = f"SPEAKER_{labels[idx]}"

    # Count per speaker
    counts = {}
    for s in speaker_map.values():
        counts[s] = counts.get(s, 0) + 1
    print(f"\nSpeakers: {counts}")

    # Save results
    results = []
    for i, chunk in enumerate(chunks):
        speaker = speaker_map.get(i, "UNKNOWN")
        results.append({
            "chunk": i,
            "time_offset": args.start + i * CHUNK_SECS,
            "speaker": speaker,
            "duration": len(chunk) / SAMPLE_RATE,
        })

    out_path = OUT / "speaker-labels.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved: {out_path}")

    # Show first 20
    print("\nFirst 20 chunks:")
    for r in results[:20]:
        print(f"  [{r['time_offset']:5d}s] {r['speaker']}")


if __name__ == "__main__":
    main()
