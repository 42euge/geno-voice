#!/usr/bin/env python3
"""Live mic → STT latency benchmark with detailed per-step timing.

Opens the mic, detects speech via energy threshold, transcribes on silence,
and prints granular timing for every step in the pipeline.

Usage:
    cd ~/code-purp/geno-voice
    .venv/bin/python examples/mic_bench.py
    .venv/bin/python examples/mic_bench.py --model mlx-community/whisper-tiny
"""

import argparse
import io
import os
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import pyaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stt import WhisperEngine

RATE = 16000
CHANNELS = 1
CHUNK = 1024  # ~64ms at 16kHz
SILENCE_THRESHOLD = 0.02
SILENCE_DURATION = 0.8
MIN_SPEECH_DURATION = 0.3

DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"


def ms(t: float) -> str:
    """Format seconds as ms string."""
    return f"{t * 1000:.1f}ms"


def rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame ** 2)))


def record_utterance(stream) -> tuple[bytes, float, dict]:
    """Record until silence after speech. Returns (wav_bytes, speech_duration, timings)."""
    frames = []
    speaking = False
    speech_start = None
    silence_start = None
    frame_count = 0
    peak_rms = 0.0
    vad_trigger_time = None

    t_wait_start = time.monotonic()

    while True:
        t_read = time.monotonic()
        data = stream.read(CHUNK, exception_on_overflow=False)
        read_latency = time.monotonic() - t_read

        t_proc = time.monotonic()
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        level = rms(audio)
        proc_latency = time.monotonic() - t_proc

        if level > SILENCE_THRESHOLD:
            peak_rms = max(peak_rms, level)
            if not speaking:
                speaking = True
                speech_start = time.monotonic()
                vad_trigger_time = speech_start - t_wait_start
                print(f"  {CYAN}● recording{RESET}", end="", flush=True)
            silence_start = None
            frames.append(data)
            frame_count += 1
        elif speaking:
            frames.append(data)
            frame_count += 1
            if silence_start is None:
                silence_start = time.monotonic()
            elif time.monotonic() - silence_start >= SILENCE_DURATION:
                break

    t_end = time.monotonic()
    speech_duration = t_end - speech_start - SILENCE_DURATION

    if speech_duration < MIN_SPEECH_DURATION:
        return b"", 0.0, {}

    t_encode = time.monotonic()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
    wav_bytes = buf.getvalue()
    encode_time = time.monotonic() - t_encode

    timings = {
        "vad_trigger": vad_trigger_time,
        "speech_duration": speech_duration,
        "frames_captured": frame_count,
        "wav_encode": encode_time,
        "wav_size_kb": len(wav_bytes) / 1024,
        "peak_rms": peak_rms,
    }
    return wav_bytes, speech_duration, timings


def transcribe_with_timing(engine: WhisperEngine, wav_bytes: bytes) -> tuple[str | None, dict]:
    """Transcribe with granular step timing."""
    timings = {}

    # Write to temp file
    t0 = time.monotonic()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name
    timings["file_write"] = time.monotonic() - t0

    # Transcribe (model already loaded)
    t1 = time.monotonic()
    try:
        import mlx_whisper
        result = mlx_whisper.transcribe(tmp_path, path_or_hf_repo=engine.model_repo)
        timings["inference"] = time.monotonic() - t1

        t2 = time.monotonic()
        text = result["text"].strip()
        timings["post_process"] = time.monotonic() - t2

        # Extract segment-level info if available
        segments = result.get("segments", [])
        if segments:
            timings["segments"] = len(segments)
            timings["no_speech_prob"] = max(s.get("no_speech_prob", 0) for s in segments)

        return text, timings
    except Exception as e:
        timings["inference"] = time.monotonic() - t1
        timings["error"] = str(e)
        return None, timings
    finally:
        os.unlink(tmp_path)


def print_timing_block(rec_timings: dict, stt_timings: dict, text: str | None):
    """Print a detailed timing breakdown."""
    print()

    if text:
        print(f"  {BOLD}\"{text}\"{RESET}")
    else:
        print(f"  {YELLOW}(no transcription){RESET}")

    print()
    print(f"  {DIM}{'─' * 50}{RESET}")
    print(f"  {BOLD}Pipeline Breakdown:{RESET}")
    print()

    # Recording phase
    print(f"  {DIM}┌─ CAPTURE{RESET}")
    print(f"  {DIM}│{RESET}  VAD trigger:      {ms(rec_timings['vad_trigger']):>10}")
    print(f"  {DIM}│{RESET}  Speech duration:   {ms(rec_timings['speech_duration']):>10}  ({rec_timings['frames_captured']} frames)")
    print(f"  {DIM}│{RESET}  WAV encode:        {ms(rec_timings['wav_encode']):>10}  ({rec_timings['wav_size_kb']:.1f} KB)")
    print(f"  {DIM}│{RESET}  Peak RMS:          {rec_timings['peak_rms']:.4f}")
    print(f"  {DIM}│{RESET}")

    # STT phase
    print(f"  {DIM}├─ STT{RESET}")
    print(f"  {DIM}│{RESET}  File write:        {ms(stt_timings['file_write']):>10}")
    print(f"  {DIM}│{RESET}  {BOLD}Inference:         {ms(stt_timings['inference']):>10}{RESET}")
    if "post_process" in stt_timings:
        print(f"  {DIM}│{RESET}  Post-process:      {ms(stt_timings['post_process']):>10}")
    if "segments" in stt_timings:
        print(f"  {DIM}│{RESET}  Segments:          {stt_timings['segments']:>10}")
    if "no_speech_prob" in stt_timings:
        print(f"  {DIM}│{RESET}  No-speech prob:    {stt_timings['no_speech_prob']:>10.3f}")
    print(f"  {DIM}│{RESET}")

    # Totals
    total_stt = stt_timings["file_write"] + stt_timings["inference"] + stt_timings.get("post_process", 0)
    total_e2e = rec_timings["speech_duration"] + SILENCE_DURATION + total_stt
    rtf = stt_timings["inference"] / rec_timings["speech_duration"] if rec_timings["speech_duration"] > 0 else 0

    print(f"  {DIM}└─ TOTALS{RESET}")
    stt_color = GREEN if total_stt < 0.5 else YELLOW
    print(f"     STT latency:     {stt_color}{ms(total_stt):>10}{RESET}")
    print(f"     Real-time factor:{stt_color}     {rtf:.2f}x{RESET}")
    e2e_color = GREEN if total_e2e < 2.0 else YELLOW
    print(f"     End-to-end:      {e2e_color}{ms(total_e2e):>10}{RESET}  (speech + silence + STT)")
    print(f"  {DIM}{'─' * 50}{RESET}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Live mic STT benchmark")
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-turbo",
                        help="HuggingFace model repo")
    args = parser.parse_args()

    model_short = args.model.split("/")[-1]
    print(f"\n{BOLD}geno-voice mic bench{RESET}")
    print(f"{DIM}model: {model_short}{RESET}")
    print()

    t_load = time.monotonic()
    engine = WhisperEngine(model_repo=args.model)
    engine._load()
    load_time = time.monotonic() - t_load
    print(f"  Model loaded in {ms(load_time)}")
    print()

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=CHANNELS,
                     rate=RATE, input=True, frames_per_buffer=CHUNK)

    print(f"  {GREEN}Ready.{RESET} Speak into your mic. Ctrl+C to quit.\n")

    try:
        turn = 0
        latencies = []
        while True:
            print(f"  {DIM}[{turn + 1}] waiting...{RESET}", end=" ", flush=True)
            wav_bytes, speech_dur, rec_timings = record_utterance(stream)
            if not wav_bytes:
                print(f" {YELLOW}(too short){RESET}")
                continue

            print(f" {DIM}transcribing...{RESET}", end="", flush=True)
            text, stt_timings = transcribe_with_timing(engine, wav_bytes)
            print_timing_block(rec_timings, stt_timings, text)

            latencies.append(stt_timings["inference"])
            turn += 1

    except KeyboardInterrupt:
        print(f"\n\n{DIM}{'─' * 50}{RESET}")
        if latencies:
            print(f"{BOLD}  Session Summary ({turn} utterances){RESET}")
            print(f"    Median STT:  {ms(sorted(latencies)[len(latencies)//2])}")
            print(f"    Min STT:     {ms(min(latencies))}")
            print(f"    Max STT:     {ms(max(latencies))}")
            print(f"    Model:       {model_short}")
        print()
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
