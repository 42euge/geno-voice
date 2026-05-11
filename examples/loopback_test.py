#!/usr/bin/env python3
"""
Loopback test — play speech through speakers so the Electron mic picks it up.

Generates TTS audio via Kokoro and plays it through afplay. The running
MindReflect app's always-listening mode should capture it, transcribe it,
and submit it as a chat message.

Usage:
    # With MindReflect running:
    .venv/bin/python examples/loopback_test.py "I have been feeling stressed"
    .venv/bin/python examples/loopback_test.py  # uses default test phrases

Prerequisites:
    - MindReflect Electron app running (npm start from MindReflect/)
    - geno-voice server running at :5111
    - System volume at 60-80%
"""

import os
import subprocess
import sys
import tempfile
import time

import requests

VOICE_URL = os.environ.get("GENO_VOICE_URL", "http://127.0.0.1:5111")
VOICE = os.environ.get("LOOPBACK_VOICE", "af_heart")  # use different voice than app's am_michael
SPEED = 1.0
PAUSE_AFTER = 3.0  # seconds to wait after playback for transcription

DEFAULT_PHRASES = [
    "I have been feeling really stressed about work lately.",
    "I don't know what to do. Everything feels overwhelming.",
    "What do you think I should do?",
]


def synthesize(text):
    resp = requests.post(
        f"{VOICE_URL}/tts/synthesize",
        json={"text": text, "voice": VOICE, "speed": SPEED},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.content


def play(wav_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp = f.name
    try:
        subprocess.run(["afplay", tmp], check=True)
    finally:
        os.unlink(tmp)


def set_volume(level):
    subprocess.run(["osascript", "-e", f"set volume output volume {level}"], check=True)


def main():
    phrases = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PHRASES

    # Check voice server
    try:
        r = requests.get(f"{VOICE_URL}/health", timeout=5)
        r.raise_for_status()
        print(f"Voice server: OK")
    except Exception as e:
        print(f"Voice server not available: {e}")
        sys.exit(1)

    set_volume(70)
    print(f"Volume: 70%")
    print(f"Voice: {VOICE} (distinct from app's am_michael)")
    print(f"Phrases: {len(phrases)}")
    print()

    for i, phrase in enumerate(phrases):
        print(f"[{i+1}/{len(phrases)}] \"{phrase}\"")
        wav = synthesize(phrase)
        print(f"  Synthesized: {len(wav)} bytes")
        print(f"  Playing...")
        play(wav)
        print(f"  Waiting {PAUSE_AFTER}s for transcription...")
        time.sleep(PAUSE_AFTER)
        print()

    set_volume(50)
    print("Done. Check the MindReflect window for transcribed messages.")


if __name__ == "__main__":
    main()
