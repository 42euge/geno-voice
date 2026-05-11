#!/usr/bin/env python3
"""
Test MindReflect with real therapy audio from "Where Should We Begin" podcast.

Feeds chunks of the podcast directly to the STT endpoint, bypassing the
browser entirely. Tests: Whisper accuracy, trigger detection, session notes,
hallucination filtering — all on real therapeutic dialogue.

Usage:
    .venv/bin/python examples/podcast_test.py [--start 60] [--duration 120]

Default: starts at 60s (skip intro), runs for 120s.
"""

import argparse
import io
import json
import sys
import time
import wave
from pathlib import Path

import requests

VOICE_URL = "http://127.0.0.1:5111"
AUDIO_FILE = Path(__file__).parent.parent / "test-data" / "esther-perel-session.wav"
CHUNK_SECS = 8  # send 8-second chunks


def load_audio(path, start_sec, duration_sec):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        w.setpos(int(start_sec * sr))
        frames = w.readframes(int(duration_sec * sr))
    return frames, sr, ch, sw


def chunk_audio(frames, sr, ch, sw, chunk_secs):
    bytes_per_sec = sr * ch * sw
    chunk_bytes = int(chunk_secs * bytes_per_sec)
    chunks = []
    for i in range(0, len(frames), chunk_bytes):
        chunk = frames[i:i + chunk_bytes]
        if len(chunk) < bytes_per_sec:  # skip < 1s
            break
        # Wrap in WAV
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(sw)
            w.setframerate(sr)
            w.writeframes(chunk)
        chunks.append(buf.getvalue())
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=60, help="Start time in seconds (skip intro)")
    parser.add_argument("--duration", type=int, default=120, help="Duration to test in seconds")
    args = parser.parse_args()

    if not AUDIO_FILE.exists():
        print(f"Audio not found: {AUDIO_FILE}")
        print("Download: yt-dlp -x --audio-format wav -o test-data/esther-perel-session.wav 'https://www.youtube.com/watch?v=8XemqwxWW8M'")
        sys.exit(1)

    print(f"Loading {AUDIO_FILE.name} from {args.start}s for {args.duration}s...")
    frames, sr, ch, sw = load_audio(AUDIO_FILE, args.start, args.duration)
    chunks = chunk_audio(frames, sr, ch, sw, CHUNK_SECS)
    print(f"Audio: {sr}Hz, {ch}ch, {sw*8}bit — {len(chunks)} chunks of {CHUNK_SECS}s")
    print()

    results = []
    for i, wav_bytes in enumerate(chunks):
        t0 = time.time()
        try:
            resp = requests.post(
                f"{VOICE_URL}/stt/transcribe",
                data=wav_bytes,
                headers={"Content-Type": "audio/wav"},
                timeout=30,
            )
            elapsed = time.time() - t0
            data = resp.json()

            text = data.get("text", "")
            trigger = data.get("trigger")
            filtered = data.get("filtered", False)

            status = "FILTERED" if filtered else ("TRIGGER" if trigger else "ok")
            trigger_info = f" [{trigger['type']}:{trigger['hint']}]" if trigger else ""

            print(f"[{i+1:3d}/{len(chunks)}] {elapsed:.1f}s {status}{trigger_info}")
            if text:
                print(f"    \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
            print()

            results.append({
                "chunk": i + 1,
                "time_offset": args.start + i * CHUNK_SECS,
                "text": text,
                "trigger": trigger,
                "filtered": filtered,
                "elapsed": round(elapsed, 2),
            })

        except Exception as e:
            print(f"[{i+1:3d}] ERROR: {e}")
            results.append({"chunk": i + 1, "error": str(e)})

    # Summary
    print("=" * 60)
    total_text = sum(len(r.get("text", "")) for r in results)
    triggers = [r for r in results if r.get("trigger")]
    filtered = [r for r in results if r.get("filtered")]
    errors = [r for r in results if r.get("error")]

    print(f"Chunks: {len(results)}")
    print(f"Text: {total_text} chars")
    print(f"Triggers: {len(triggers)}")
    print(f"Filtered: {len(filtered)}")
    print(f"Errors: {len(errors)}")

    if triggers:
        print("\nTriggers found:")
        for r in triggers:
            print(f"  [{r['time_offset']}s] {r['trigger']['type']}: \"{r['text'][:60]}\"")

    # Save results
    out = Path(__file__).parent.parent / "test-data" / "podcast-test-results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out}")

    # Check session notes
    try:
        themes = requests.get(f"{VOICE_URL}/notes/themes").json()
        print(f"\nSession themes: {themes.get('themes', [])}")
        print(f"Summary: {themes.get('summary', '')[:200]}")
    except:
        pass


if __name__ == "__main__":
    main()
