"""Talk mode — full voice loop with NLP canned responses (no LLM).

STT → trigger detection → pick canned response → TTS → play back through speakers.
Measures and displays timing for every pipeline stage.
"""

import io
import os
import random
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stt import WhisperEngine
from tts import get_engine as get_tts_engine

# Import triggers directly to avoid session/__init__.py pulling in pipecat
import importlib.util
_triggers_spec = importlib.util.spec_from_file_location(
    "triggers", Path(__file__).resolve().parent.parent / "session" / "triggers.py"
)
_triggers = importlib.util.module_from_spec(_triggers_spec)
_triggers_spec.loader.exec_module(_triggers)
detect_triggers = _triggers.detect_triggers
filter_noise = _triggers.filter_noise
TriggerType = _triggers.TriggerType
ResponseHint = _triggers.ResponseHint

RATE = 16000
CHANNELS = 1
CHUNK = 1024
SILENCE_THRESHOLD = 0.02
SILENCE_DURATION = 0.8
MIN_SPEECH_DURATION = 0.3

DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

# Canned responses keyed by trigger type + hint
RESPONSES = {
    TriggerType.DIRECT_QUESTION: [
        "That's a good question. Let me think about that.",
        "Hmm, I think there are a few ways to look at that.",
        "That's worth exploring further.",
        "I'd say it depends on what matters most to you right now.",
    ],
    TriggerType.INVITATION: [
        "I think you're on the right track.",
        "From what you've said, it sounds like you already know what to do.",
        "That makes sense to me. Keep going.",
        "I hear you. What feels right?",
    ],
    TriggerType.RESIGNATION: [
        "That sounds really tough. It's okay to feel that way.",
        "You don't have to have it figured out right now.",
        "Take a breath. One thing at a time.",
        "I hear you. That's a lot to carry.",
    ],
    TriggerType.EMOTIONAL_PEAK: [
        "I can tell this means a lot to you.",
        "That's a strong feeling. It makes sense.",
        "Yeah, that's real.",
    ],
    TriggerType.TRAILING_OFF: [
        "Take your time.",
        "I'm listening.",
        "Go on, whenever you're ready.",
    ],
}

FALLBACK_RESPONSES = [
    "Mm hmm.",
    "I see.",
    "Interesting.",
    "Tell me more.",
    "Got it.",
]


def pick_response(trigger_type: TriggerType | None) -> str:
    if trigger_type and trigger_type in RESPONSES:
        return random.choice(RESPONSES[trigger_type])
    return random.choice(FALLBACK_RESPONSES)


def rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame ** 2)))


@dataclass
class TurnMetrics:
    speech_duration: float = 0.0
    stt_time: float = 0.0
    trigger_time: float = 0.0
    response_pick_time: float = 0.0
    tts_time: float = 0.0
    playback_time: float = 0.0
    total_e2e: float = 0.0
    trigger_type: str = ""
    transcript: str = ""
    response: str = ""

    def print(self, turn: int):
        print()
        print(f"  {DIM}{'─' * 56}{RESET}")
        print(f"  {BOLD}Turn {turn}{RESET}")
        print(f"  {DIM}You:{RESET}  \"{self.transcript}\"")
        print(f"  {CYAN}Bot:{RESET}  \"{self.response}\"")
        print()
        print(f"  {DIM}┌─ PIPELINE{RESET}")
        print(f"  {DIM}│{RESET}  Speech:        {self.speech_duration*1000:>7.0f}ms")
        print(f"  {DIM}│{RESET}  STT:           {self.stt_time*1000:>7.0f}ms")
        print(f"  {DIM}│{RESET}  Trigger NLP:   {self.trigger_time*1000:>7.2f}ms  → {self.trigger_type or 'none'}")
        print(f"  {DIM}│{RESET}  Response pick:  {self.response_pick_time*1000:>7.2f}ms")
        print(f"  {DIM}│{RESET}  TTS synth:     {self.tts_time*1000:>7.0f}ms")
        print(f"  {DIM}│{RESET}  Playback:      {self.playback_time*1000:>7.0f}ms")
        print(f"  {DIM}│{RESET}")
        # Latency = time from end-of-speech to first audio out
        response_latency = self.stt_time + self.trigger_time + self.response_pick_time + self.tts_time
        color = GREEN if response_latency < 2.0 else YELLOW
        print(f"  {DIM}└─{RESET} {BOLD}Response latency:{RESET} {color}{response_latency*1000:.0f}ms{RESET}  (silence → first audio)")
        total_color = GREEN if self.total_e2e < 4.0 else YELLOW
        print(f"     {BOLD}Total turn:{RESET}        {total_color}{self.total_e2e*1000:.0f}ms{RESET}")
        print(f"  {DIM}{'─' * 56}{RESET}")
        print()


def record_utterance(stream) -> tuple[bytes, float]:
    """Record until silence. Returns (wav_bytes, speech_duration)."""
    frames = []
    speaking = False
    speech_start = None
    silence_start = None

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        level = rms(audio)

        if level > SILENCE_THRESHOLD:
            if not speaking:
                speaking = True
                speech_start = time.monotonic()
                print(f"  {CYAN}● listening...{RESET}", end="", flush=True)
            silence_start = None
            frames.append(data)
        elif speaking:
            frames.append(data)
            if silence_start is None:
                silence_start = time.monotonic()
            elif time.monotonic() - silence_start >= SILENCE_DURATION:
                break

    speech_duration = time.monotonic() - speech_start - SILENCE_DURATION
    if speech_duration < MIN_SPEECH_DURATION:
        return b"", 0.0

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
    return buf.getvalue(), speech_duration


def play_wav(wav_bytes: bytes) -> float:
    """Play WAV through speakers, return playback duration."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp = f.name
    t0 = time.monotonic()
    try:
        subprocess.run(["afplay", tmp], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    elapsed = time.monotonic() - t0
    os.unlink(tmp)
    return elapsed


def run_talk(model_repo: str, voice: str = "af_heart", speed: float = 1.0):
    """Main talk loop."""
    model_short = model_repo.split("/")[-1]
    print(f"\n{BOLD}gv talk{RESET}")
    print(f"{DIM}stt: {model_short} │ tts: kokoro/{voice} │ speed: {speed}x{RESET}")
    print()

    # Load STT
    t0 = time.monotonic()
    stt_engine = WhisperEngine(model_repo=model_repo)
    stt_engine._load()
    stt_load = time.monotonic() - t0

    # Load TTS
    t1 = time.monotonic()
    tts_engine = get_tts_engine("kokoro")
    tts_engine._load()
    tts_load = time.monotonic() - t1

    print(f"  STT loaded in {stt_load*1000:.0f}ms")
    print(f"  TTS loaded in {tts_load*1000:.0f}ms")
    print(f"  {GREEN}Ready.{RESET} Speak and I'll respond. Ctrl+C to quit.\n")

    pa = pyaudio.PyAudio()
    mic = pa.open(format=pyaudio.paInt16, channels=CHANNELS,
                  rate=RATE, input=True, frames_per_buffer=CHUNK)

    all_metrics = []

    try:
        turn = 0
        while True:
            print(f"  {DIM}[{turn + 1}] waiting...{RESET}", end=" ", flush=True)
            wav_bytes, speech_dur = record_utterance(mic)
            if not wav_bytes:
                print(f" {YELLOW}(too short){RESET}")
                continue

            print(f" {DIM}processing...{RESET}", end="", flush=True)
            metrics = TurnMetrics(speech_duration=speech_dur)
            turn_start = time.monotonic()

            # STT
            t = time.monotonic()
            text, _ = stt_engine.transcribe(wav_bytes)
            metrics.stt_time = time.monotonic() - t

            if not text:
                print(f" {YELLOW}(no transcription){RESET}")
                continue

            # Filter noise
            cleaned = filter_noise(text)
            if not cleaned:
                print(f" {DIM}(noise: \"{text}\"){RESET}")
                continue
            metrics.transcript = cleaned

            # Trigger detection
            t = time.monotonic()
            trigger = detect_triggers(cleaned)
            metrics.trigger_time = time.monotonic() - t
            metrics.trigger_type = trigger.trigger_type.value if trigger.trigger_type else ""

            # Pick response
            t = time.monotonic()
            response = pick_response(trigger.trigger_type)
            metrics.response_pick_time = time.monotonic() - t
            metrics.response = response

            # TTS
            t = time.monotonic()
            response_wav = tts_engine.synthesize(response, voice=voice, speed=speed)
            metrics.tts_time = time.monotonic() - t

            # Playback
            t = time.monotonic()
            metrics.playback_time = play_wav(response_wav)

            metrics.total_e2e = time.monotonic() - turn_start
            metrics.print(turn + 1)
            all_metrics.append(metrics)
            turn += 1

    except KeyboardInterrupt:
        print(f"\n\n{DIM}{'─' * 56}{RESET}")
        if all_metrics:
            stt_times = [m.stt_time for m in all_metrics]
            tts_times = [m.tts_time for m in all_metrics]
            latencies = [m.stt_time + m.trigger_time + m.response_pick_time + m.tts_time for m in all_metrics]
            print(f"{BOLD}  Session Summary ({len(all_metrics)} turns){RESET}")
            print(f"    Median STT:       {sorted(stt_times)[len(stt_times)//2]*1000:.0f}ms")
            print(f"    Median TTS:       {sorted(tts_times)[len(tts_times)//2]*1000:.0f}ms")
            print(f"    Median latency:   {sorted(latencies)[len(latencies)//2]*1000:.0f}ms  (silence → audio)")
            print(f"    Best latency:     {min(latencies)*1000:.0f}ms")
            print(f"    Model:            {model_short}")
        print()
    finally:
        mic.stop_stream()
        mic.close()
        pa.terminate()
