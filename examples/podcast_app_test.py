#!/usr/bin/env python3
"""
Feed podcast audio through the running MindReflect app via loopback.

Sets system audio to Loopback Audio, plays podcast chunks through it,
and monitors the voice server for transcriptions + session notes.

Usage:
    # With MindReflect app already running:
    .venv/bin/python examples/podcast_app_test.py --start 120 --duration 60
"""

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import requests

VOICE_URL = "http://127.0.0.1:5111"
AUDIO_FILE = Path(__file__).parent.parent / "test-data" / "esther-perel-session.wav"


def set_audio(device, type_):
    subprocess.run(["SwitchAudioSource", "-s", device, "-t", type_], capture_output=True)


def play_wav(wav_bytes, volume_boost=10):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        src = f.name
    boosted = src + ".boosted.wav"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", src,
            "-filter:a", f"volume={volume_boost}",
            boosted,
        ], capture_output=True, timeout=10)
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", boosted], check=True, timeout=30)
    finally:
        for f in [src, boosted]:
            try: os.unlink(f)
            except: pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=120)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--chunk", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading {AUDIO_FILE.name} from {args.start}s for {args.duration}s...")
    with wave.open(str(AUDIO_FILE), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        w.setpos(int(args.start * sr))
        frames = w.readframes(int(args.duration * sr))

    bytes_per_sec = sr * ch * sw
    chunk_bytes = int(args.chunk * bytes_per_sec)
    chunks = []
    for i in range(0, len(frames), chunk_bytes):
        c = frames[i:i + chunk_bytes]
        if len(c) < bytes_per_sec:
            break
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(sw)
            w.setframerate(sr)
            w.writeframes(c)
        chunks.append(buf.getvalue())

    print(f"{len(chunks)} chunks of {args.chunk}s")

    # Switch to loopback
    set_audio("Loopback Audio", "output")
    set_audio("Loopback Audio", "input")
    print("Audio: Loopback Audio (internal routing)")

    # Get initial state
    initial = requests.get(f"{VOICE_URL}/notes/themes").json()
    print(f"Initial chunks: {initial.get('chunks', 0)}")
    print()

    try:
        for i, wav_bytes in enumerate(chunks):
            print(f"[{i+1}/{len(chunks)}] Playing chunk ({args.start + i*args.chunk}s - {args.start + (i+1)*args.chunk}s)...")
            play_wav(wav_bytes)
            time.sleep(3)  # wait for transcription

            # Check what happened
            themes = requests.get(f"{VOICE_URL}/notes/themes").json()
            print(f"  Chunks processed: {themes.get('chunks', 0)}, Themes: {themes.get('themes', [])}")

            # Check recordings
            sessions = sorted(Path.home().glob(".mindreflect/sessions/*/recordings/full-session.wav"))
            if sessions:
                size = sessions[-1].stat().st_size
                print(f"  Recording: {size/1024:.0f}KB")
            print()

    finally:
        # Restore audio
        set_audio("MacBook Air Speakers", "output")
        set_audio("MacBook Air Microphone", "input")
        print("Audio restored to defaults")

    # Final state
    final = requests.get(f"{VOICE_URL}/notes/themes").json()
    print(f"\nFinal: {final.get('chunks', 0)} chunks, themes: {final.get('themes', [])}")
    print(f"Summary: {final.get('summary', '')[:200]}")


if __name__ == "__main__":
    main()
