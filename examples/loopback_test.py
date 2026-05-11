#!/usr/bin/env python3
"""
Internal loopback test — route TTS audio through virtual audio device.

Plays TTS audio to "Loopback Audio" virtual device. The Electron app
captures from the same device. Pure internal routing — no physical
speakers, no echo cancellation issues.

Setup:
    1. Set macOS input device to "Loopback Audio"
    2. Run this script — it plays to "Loopback Audio" output
    3. MindReflect's mic captures the audio internally

Usage:
    .venv/bin/python examples/loopback_test.py "I feel stressed"
    .venv/bin/python examples/loopback_test.py  # default phrases
"""

import os
import subprocess
import sys
import tempfile
import time

import requests

VOICE_URL = os.environ.get("GENO_VOICE_URL", "http://127.0.0.1:5111")
VOICE = os.environ.get("LOOPBACK_VOICE", "af_heart")
SPEED = 1.0
PAUSE_AFTER = 5.0
LOOPBACK_DEVICE = "Loopback Audio"

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


def play_to_device(wav_bytes, device):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp = f.name
    try:
        # Use SoX (play command) to route to specific device, fallback to afplay
        try:
            subprocess.run(
                ["play", "-q", tmp],
                env={**os.environ, "AUDIODEV": device},
                check=True, timeout=30,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            # Fallback: set system output to loopback, play, restore
            set_output_device(device)
            subprocess.run(["afplay", tmp], check=True, timeout=30)
    finally:
        os.unlink(tmp)


def set_output_device(name):
    """Set macOS output audio device using osascript."""
    subprocess.run([
        "osascript", "-e",
        f'tell application "System Preferences" to quit',
    ], capture_output=True)
    # Use SwitchAudioSource if available
    try:
        subprocess.run(["SwitchAudioSource", "-s", name, "-t", "output"], check=True, capture_output=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return False


def set_input_device(name):
    """Set macOS input audio device."""
    try:
        subprocess.run(["SwitchAudioSource", "-s", name, "-t", "input"], check=True, capture_output=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return False


def main():
    phrases = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PHRASES

    try:
        r = requests.get(f"{VOICE_URL}/health", timeout=5)
        r.raise_for_status()
        print(f"Voice server: OK")
    except Exception as e:
        print(f"Voice server not available: {e}")
        sys.exit(1)

    # Try to set up internal loopback
    has_switch = set_input_device(LOOPBACK_DEVICE)
    if has_switch:
        set_output_device(LOOPBACK_DEVICE)
        print(f"Audio routing: internal loopback ({LOOPBACK_DEVICE})")
    else:
        print(f"SwitchAudioSource not found. Install: brew install switchaudio-osx")
        print(f"Falling back to speaker playback. Set input to '{LOOPBACK_DEVICE}' manually in System Settings.")
        print()

    print(f"Voice: {VOICE}")
    print(f"Phrases: {len(phrases)}")
    print()

    for i, phrase in enumerate(phrases):
        print(f"[{i+1}/{len(phrases)}] \"{phrase}\"")
        wav = synthesize(phrase)
        print(f"  Synthesized: {len(wav)} bytes")
        print(f"  Playing to {LOOPBACK_DEVICE if has_switch else 'speakers'}...")

        if has_switch:
            play_to_device(wav, LOOPBACK_DEVICE)
        else:
            subprocess.run(["afplay", tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name], check=False)
            # Actually just play normally as fallback
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav)
                subprocess.run(["afplay", f.name], check=True)
                os.unlink(f.name)

        print(f"  Waiting {PAUSE_AFTER}s for transcription...")
        time.sleep(PAUSE_AFTER)
        print()

    # Restore default audio devices
    if has_switch:
        set_output_device("MacBook Air Speakers")
        set_input_device("MacBook Air Microphone")
        print("Audio restored to defaults.")

    print("Done. Check MindReflect for transcribed messages.")


if __name__ == "__main__":
    main()
