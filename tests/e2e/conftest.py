"""
E2E integration test fixtures.

Provides clip queues, loopback audio routing, voice server clients,
and sidecar WebSocket consumers for full pipeline testing.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import pytest
import requests

VOICE_URL = os.environ.get("GENO_VOICE_URL", "http://127.0.0.1:5111")
SIDECAR_WS = os.environ.get("SIDECAR_WS", "ws://127.0.0.1:8765")
DIARIZATION_ROOT = Path(__file__).parent.parent.parent / "test-data" / "diarization"
LOOPBACK_DEVICE = "Loopback Audio"


class ClipQueue:
    """Load and iterate diarized clips from a source/episode directory."""

    def __init__(self, source, episode):
        self.root = DIARIZATION_ROOT / source / episode
        self.segments = json.loads((self.root / "segments.json").read_text())
        self.transcripts = {}
        t_path = self.root / "transcripts.json"
        if t_path.exists():
            for t in json.loads(t_path.read_text()):
                self.transcripts[f"{t['speaker']}/{t['clip']}"] = t

    @property
    def speakers(self):
        return sorted(set(s["speaker"] for s in self.segments))

    def clips_for(self, speaker=None):
        segs = self.segments if speaker is None else [s for s in self.segments if s["speaker"] == speaker]
        results = []
        speaker_counts = {}
        for s in segs:
            sp = s["speaker"]
            speaker_counts[sp] = speaker_counts.get(sp, 0) + 1
            idx = speaker_counts[sp]
            clip_name = f"{idx:03d}.wav"
            clip_path = self.root / sp / clip_name
            transcript = self.transcripts.get(f"{sp}/{clip_name}", {})
            results.append({
                "path": str(clip_path),
                "speaker": sp,
                "clip": clip_name,
                "start": s["start"],
                "end": s["end"],
                "duration": s["duration"],
                "text": transcript.get("text", ""),
            })
        return results

    def clips_with_text(self, speaker=None):
        return [c for c in self.clips_for(speaker) if c["text"]]


class VoiceServerClient:
    """Thin wrapper around the voice server HTTP API."""

    def __init__(self, base_url=VOICE_URL):
        self.base_url = base_url

    def health(self):
        r = requests.get(f"{self.base_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()

    def transcribe(self, wav_bytes):
        r = requests.post(
            f"{self.base_url}/stt/transcribe",
            data=wav_bytes,
            headers={"Content-Type": "audio/wav"},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def synthesize(self, text, voice="am_michael", speed=1.0):
        r = requests.post(
            f"{self.base_url}/tts/synthesize",
            json={"text": text, "voice": voice, "speed": speed},
            timeout=30,
        )
        r.raise_for_status()
        return r.content

    def notes_themes(self):
        r = requests.get(f"{self.base_url}/notes/themes", timeout=5)
        r.raise_for_status()
        return r.json()

    def process_note(self, text):
        r = requests.post(
            f"{self.base_url}/notes/process",
            json={"text": text},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()


class LoopbackPlayer:
    """Play WAV clips through the Loopback Audio virtual device."""

    def __init__(self, device=LOOPBACK_DEVICE):
        self.device = device
        self._has_switch = self._check_switch()

    def _check_switch(self):
        try:
            subprocess.run(
                ["SwitchAudioSource", "-s", self.device, "-t", "output"],
                check=True, capture_output=True, timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    @property
    def available(self):
        return self._has_switch

    def setup(self):
        if not self._has_switch:
            return False
        subprocess.run(
            ["SwitchAudioSource", "-s", self.device, "-t", "input"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["SwitchAudioSource", "-s", self.device, "-t", "output"],
            check=True, capture_output=True,
        )
        return True

    def restore(self):
        if not self._has_switch:
            return
        subprocess.run(
            ["SwitchAudioSource", "-s", "MacBook Air Speakers", "-t", "output"],
            capture_output=True,
        )
        subprocess.run(
            ["SwitchAudioSource", "-s", "MacBook Air Microphone", "-t", "input"],
            capture_output=True,
        )

    def play(self, wav_path, wait=True):
        if not self._has_switch:
            raise RuntimeError("SwitchAudioSource not available. Install: brew install switchaudio-osx")
        proc = subprocess.Popen(
            ["afplay", str(wav_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if wait:
            proc.wait()
        return proc

    def play_bytes(self, wav_bytes, wait=True):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp = f.name
        try:
            return self.play(tmp, wait=wait)
        finally:
            if wait:
                os.unlink(tmp)


class SidecarCollector:
    """Connect to sidecar WebSocket and collect messages."""

    def __init__(self, ws_url=SIDECAR_WS, timeout=30):
        self.ws_url = ws_url
        self.timeout = timeout
        self.messages = []

    async def collect(self, count=None, duration=None):
        import websockets
        deadline = time.time() + (duration or self.timeout)
        async with websockets.connect(self.ws_url) as ws:
            while time.time() < deadline:
                try:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    msg = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 2))
                    data = json.loads(msg)
                    self.messages.append(data)
                    if count and len(self.messages) >= count:
                        break
                except asyncio.TimeoutError:
                    continue
        return self.messages

    @property
    def transcripts(self):
        return [m for m in self.messages if m.get("type") == "transcript" and m.get("text")]

    @property
    def triggers(self):
        return [m for m in self.messages if m.get("type") == "transcript" and m.get("trigger")]

    @property
    def vad_events(self):
        return [m for m in self.messages if m.get("type") == "vad"]

    @property
    def cues(self):
        return [m for m in self.messages if m.get("type") == "cue"]


# --- Pytest fixtures ---

@pytest.fixture
def voice_server():
    client = VoiceServerClient()
    try:
        client.health()
    except Exception:
        pytest.skip("Voice server not running at " + VOICE_URL)
    return client


@pytest.fixture
def loopback():
    player = LoopbackPlayer()
    if not player.available:
        pytest.skip("Loopback Audio not available (install switchaudio-osx)")
    player.setup()
    yield player
    player.restore()


@pytest.fixture
def esther_ep1():
    q = ClipQueue("esther-perel", "ep1")
    if not q.segments:
        pytest.skip("Esther Perel ep1 clips not available")
    return q


@pytest.fixture
def daic_300():
    q = ClipQueue("daic-woz", "300")
    if not q.segments:
        pytest.skip("DAIC-WOZ 300 clips not available")
    return q


@pytest.fixture
def sidecar_collector():
    return SidecarCollector()
