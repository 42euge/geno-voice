#!/usr/bin/env python3
"""
Generate backchannel cue audio bank using Kokoro TTS.

Pre-synthesizes short affirmation audio clips ("mhmm", "I see", etc.)
with varied prosody across multiple voices. The turn-taking engine
plays these at runtime instead of generating TTS on-the-fly.

Output: session/cues/<cue_type>/<variant>.wav

Usage:
    cd /Users/euge/code-red/mind-reflect-ws/geno-voice
    .venv/bin/python session/generate_cues.py
"""

import json
import os
import sys
from pathlib import Path

import requests

VOICE_SERVER = os.environ.get("GENO_VOICE_URL", "http://127.0.0.1:5111")
CUES_DIR = Path(__file__).parent / "cues"

CUE_BANK = {
    "mhmm": [
        ("Mhmm.", 0.85),
        ("Mm-hmm.", 0.9),
        ("Mhm.", 0.8),
    ],
    "i_see": [
        ("I see.", 0.95),
        ("Ah, I see.", 1.0),
        ("I see what you mean.", 1.0),
    ],
    "right": [
        ("Right.", 0.9),
        ("Right, right.", 0.85),
        ("Yeah.", 0.9),
    ],
    "go_on": [
        ("Go on.", 1.0),
        ("Mm, go on.", 0.95),
        ("Keep going.", 1.0),
    ],
    "tell_me_more": [
        ("Tell me more.", 1.0),
        ("Tell me more about that.", 1.0),
        ("Say more about that.", 1.0),
    ],
    "okay": [
        ("Okay.", 0.9),
        ("Okay, okay.", 0.85),
        ("Alright.", 0.95),
    ],
    "hmm": [
        ("Hmm.", 0.85),
        ("Hm.", 0.8),
        ("Hmm, interesting.", 0.95),
    ],
}

VOICES = ["af_heart", "af_sarah", "af_nova"]


def synthesize(text: str, voice: str, speed: float) -> bytes:
    resp = requests.post(
        f"{VOICE_SERVER}/tts/synthesize",
        json={"text": text, "voice": voice, "speed": speed},
    )
    resp.raise_for_status()
    return resp.content


def main():
    print(f"Generating cue bank at: {CUES_DIR}")
    print(f"Voice server: {VOICE_SERVER}")
    print(f"Voices: {', '.join(VOICES)}")
    print()

    total = 0
    for cue_type, variants in CUE_BANK.items():
        cue_dir = CUES_DIR / cue_type
        cue_dir.mkdir(parents=True, exist_ok=True)

        for vi, (text, speed) in enumerate(variants):
            for voice in VOICES:
                filename = f"{vi:02d}_{voice}.wav"
                filepath = cue_dir / filename

                if filepath.exists():
                    print(f"  skip {cue_type}/{filename} (exists)")
                    total += 1
                    continue

                try:
                    audio = synthesize(text, voice, speed)
                    filepath.write_bytes(audio)
                    size_kb = len(audio) / 1024
                    print(f"  {cue_type}/{filename} ({size_kb:.0f}KB) — \"{text}\"")
                    total += 1
                except Exception as e:
                    print(f"  FAIL {cue_type}/{filename}: {e}", file=sys.stderr)

    print(f"\nGenerated {total} cue files across {len(CUE_BANK)} types")

    manifest = {}
    for cue_type in CUE_BANK:
        cue_dir = CUES_DIR / cue_type
        files = sorted(str(p.relative_to(CUES_DIR)) for p in cue_dir.glob("*.wav"))
        manifest[cue_type] = files

    manifest_path = CUES_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
