"""Chat mode — full voice loop with LLM responses via LiteLLM.

STT → LLM (streaming) → TTS (per-sentence) → speaker playback.
Text streams to terminal in sync with speech output.
Maintains conversation history for context.
Reads config from config.local.yaml (not checked in).
"""

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyaudio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stt import WhisperEngine
from tts import get_engine as get_tts_engine

# Pure helpers — extracted for testability, see examples/_chat_helpers.py.
from examples._chat_helpers import (
    SENTENCE_END,
    flush_pending_audio,
    split_complete_sentences,
    trim_history,
)

# Recording loop — extracted to a pyaudio-free module so tests can drive
# it with examples.virtual_audio.VirtualMicStream + a stub transcriber.
from examples._chat_recording import (
    CHANNELS,
    CHUNK,
    CLEAR_LINE,
    INFERENCE_INTERVAL,
    MIN_SPEECH_DURATION,
    RATE,
    SILENCE_DURATION,
    SILENCE_THRESHOLD,
    _buffer_to_wav,
    _transcribe_quick,
    record_utterance_streaming,
    rms,
)

DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.local.yaml"


def load_llm_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"  {YELLOW}Missing config.local.yaml — create it with llm settings{RESET}")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    llm = cfg.get("llm", {})
    api_key = llm.get("api_key", "")
    if api_key.startswith("${") and api_key.endswith("}"):
        env_var = api_key[2:-1]
        api_key = os.environ.get(env_var, "")
        if not api_key:
            print(f"  {YELLOW}Env var {env_var} not set{RESET}")
            sys.exit(1)
    llm["api_key"] = api_key
    return llm


def llm_stream(messages: list[dict], config: dict):
    """Stream LLM tokens. Yields (token, is_done)."""
    import requests

    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config["model"],
        "messages": messages,
        "max_tokens": config.get("max_tokens", 150),
        "stream": True,
    }
    resp = requests.post(
        f"{config['base_url']}/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
        stream=True,
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token
        except (json.JSONDecodeError, KeyError, IndexError):
            continue


@dataclass
class TurnMetrics:
    speech_duration: float = 0.0
    stt_time: float = 0.0
    llm_first_token: float = 0.0
    llm_total: float = 0.0
    tts_time: float = 0.0
    playback_time: float = 0.0
    ttfs: float = 0.0
    total_e2e: float = 0.0
    sentences_spoken: int = 0
    transcript: str = ""
    response: str = ""
    model: str = ""

    def print(self, turn: int):
        print()
        print(f"  {DIM}{'─' * 56}{RESET}")
        print(f"  {BOLD}Turn {turn}{RESET}")
        print(f"  {DIM}You:{RESET} \"{self.transcript}\"")
        print()
        print(f"  {DIM}┌─ PIPELINE{RESET}")
        print(f"  {DIM}│{RESET}  Speech:        {self.speech_duration*1000:>7.0f}ms")
        print(f"  {DIM}│{RESET}  STT:           {self.stt_time*1000:>7.0f}ms")
        print(f"  {DIM}│{RESET}  LLM 1st tok:   {self.llm_first_token*1000:>7.0f}ms")
        print(f"  {DIM}│{RESET}  LLM total:     {self.llm_total*1000:>7.0f}ms  ({self.model})")
        print(f"  {DIM}│{RESET}  TTS:           {self.tts_time*1000:>7.0f}ms  ({self.sentences_spoken} sentences)")
        print(f"  {DIM}│{RESET}  Playback:      {self.playback_time*1000:>7.0f}ms")
        print(f"  {DIM}│{RESET}")
        ttfs_color = GREEN if self.ttfs < 3.0 else YELLOW
        print(f"  {DIM}├─{RESET} {BOLD}TTFS:{RESET}            {ttfs_color}{self.ttfs*1000:>7.0f}ms{RESET}  (speech stop → speaker)")
        total_color = GREEN if self.total_e2e < 6.0 else YELLOW
        print(f"  {DIM}└─{RESET} {BOLD}Total turn:{RESET}      {total_color}{self.total_e2e*1000:>7.0f}ms{RESET}")
        print(f"  {DIM}{'─' * 56}{RESET}")
        print()


TTS_RATE = 24000


def synthesize_with_alignment(tts_engine, text: str, voice: str, speed: float):
    """Synthesize text and return (audio_np, tokens_with_timing).

    Each token has .text, .start_ts, .end_ts.
    Audio is float32 numpy array at 24kHz.
    """
    import torch
    tts_engine._load()
    all_audio = []
    all_tokens = []
    offset = 0.0

    for result in tts_engine._pipeline(text, voice=voice, speed=speed):
        audio = result.audio
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        duration = len(audio) / TTS_RATE

        for tok in result.tokens:
            all_tokens.append({
                "text": tok.text,
                "start": tok.start_ts + offset,
                "end": tok.end_ts + offset,
            })
        all_audio.append(audio)
        offset += duration

    if not all_audio:
        return np.array([], dtype=np.float32), []

    combined = np.concatenate(all_audio)
    return combined, all_tokens


def play_aligned(pa, audio_np, tokens, is_first_sentence=False):
    """Play audio via PyAudio, revealing words in sync.

    Prints words incrementally (no \r rewrite). On first sentence,
    prints the "Bot: " prefix. Returns playback duration.
    """
    # Convert float32 to int16
    audio_int16 = (audio_np * 32767).astype(np.int16)
    total_samples = len(audio_int16)
    play_chunk = 1024  # ~42ms at 24kHz

    out_stream = pa.open(format=pyaudio.paInt16, channels=1,
                         rate=TTS_RATE, output=True,
                         frames_per_buffer=play_chunk)

    if is_first_sentence:
        # Clear any leftover "[N] waiting..." or live-preview line on the
        # current row before printing "Bot:". Without this, we get duplicate
        # "Bot:" lines on multi-sentence responses (bug #1).
        sys.stdout.write(f"\r{CLEAR_LINE}  {CYAN}Bot:{RESET} ")
        sys.stdout.flush()

    t0 = time.monotonic()
    samples_played = 0
    token_idx = 0

    try:
        while samples_played < total_samples:
            end = min(samples_played + play_chunk, total_samples)
            chunk_bytes = audio_int16[samples_played:end].tobytes()
            out_stream.write(chunk_bytes)
            samples_played = end

            # Current playback position in seconds
            pos = samples_played / TTS_RATE

            # Reveal words whose start_ts we've passed
            while token_idx < len(tokens) and tokens[token_idx]["start"] <= pos:
                word = tokens[token_idx]["text"]
                if word.strip() and not all(c in '.,!?;:' for c in word.strip()):
                    sys.stdout.write(f"{BOLD}{word}{RESET} ")
                    sys.stdout.flush()
                elif word.strip():
                    # Punctuation: backspace over trailing space, attach
                    sys.stdout.write(f"\b{word} ")
                    sys.stdout.flush()
                token_idx += 1

    finally:
        out_stream.stop_stream()
        out_stream.close()

    elapsed = time.monotonic() - t0

    # Flush any remaining tokens
    while token_idx < len(tokens):
        word = tokens[token_idx]["text"]
        if word.strip() and not all(c in '.,!?;:' for c in word.strip()):
            sys.stdout.write(f"{word} ")
        elif word.strip():
            sys.stdout.write(f"\b{word} ")
        token_idx += 1
    sys.stdout.flush()

    return elapsed


def run_chat(model_repo: str, voice: str = "af_heart", speed: float = 1.0):
    """Main chat loop with streaming LLM + sentence-by-sentence TTS."""
    model_short = model_repo.split("/")[-1]
    llm_config = load_llm_config()

    print(f"\n{BOLD}gv chat{RESET}")
    print(f"{DIM}stt: {model_short} │ llm: {llm_config['model']} │ tts: kokoro/{voice}{RESET}")
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

    system_prompt = llm_config.get("system_prompt", "You are a concise voice assistant.")
    messages = [{"role": "system", "content": system_prompt}]
    all_metrics = []

    try:
        turn = 0
        while True:
            print(f"  {DIM}[{turn + 1}] waiting...{RESET}", end="", flush=True)
            wav_bytes, speech_dur, stt_time = record_utterance_streaming(mic, stt_engine)
            if not wav_bytes:
                continue

            metrics = TurnMetrics(speech_duration=speech_dur, model=llm_config["model"])
            speech_ended_at = time.monotonic() - SILENCE_DURATION
            turn_start = time.monotonic()

            metrics.stt_time = stt_time
            text = stt_engine._last_text

            if not text or len(text.strip()) < 2:
                print(f"  {YELLOW}(no transcription){RESET}")
                continue
            metrics.transcript = text.strip()

            # Stream LLM → accumulate sentences → TTS with alignment → play
            messages.append({"role": "user", "content": metrics.transcript})

            llm_start = time.monotonic()
            first_token_at = None
            token_buffer = ""
            full_response = ""
            sentences_spoken = 0
            total_tts_time = 0.0
            total_playback_time = 0.0
            ttfs_recorded = False

            llm_stream_done_at = None
            try:
                for token in llm_stream(messages, llm_config):
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    token_buffer += token
                    full_response += token

                    # Check if we have a complete sentence
                    complete, token_buffer_next = split_complete_sentences(token_buffer)
                    if complete:
                        for sentence in complete:
                            # Synthesize with word-level timing
                            t = time.monotonic()
                            audio_np, tokens = synthesize_with_alignment(
                                tts_engine, sentence, voice, speed
                            )
                            total_tts_time += time.monotonic() - t

                            if len(audio_np) == 0:
                                continue

                            # Record TTFS on first sentence
                            if not ttfs_recorded:
                                metrics.ttfs = time.monotonic() - speech_ended_at
                                ttfs_recorded = True

                            # Play with word-aligned text reveal
                            is_first = sentences_spoken == 0
                            elapsed = play_aligned(
                                pa, audio_np, tokens, is_first_sentence=is_first
                            )
                            total_playback_time += elapsed
                            sentences_spoken += 1

                        token_buffer = token_buffer_next

                # Stamp end-of-stream BEFORE any trailing synth/playback so
                # llm_total measures only the LLM streaming window, not the
                # follow-on TTS/playback work (bug #2).
                llm_stream_done_at = time.monotonic()

                # Handle remaining text after stream ends
                remaining = token_buffer.strip()
                if remaining:
                    t = time.monotonic()
                    audio_np, tokens = synthesize_with_alignment(
                        tts_engine, remaining, voice, speed
                    )
                    total_tts_time += time.monotonic() - t

                    if len(audio_np) > 0:
                        if not ttfs_recorded:
                            metrics.ttfs = time.monotonic() - speech_ended_at
                            ttfs_recorded = True

                        is_first = sentences_spoken == 0
                        elapsed = play_aligned(
                            pa, audio_np, tokens, is_first_sentence=is_first
                        )
                        total_playback_time += elapsed
                        sentences_spoken += 1

                print()  # newline after streamed text

                metrics.llm_first_token = (first_token_at - llm_start) if first_token_at else 0
                # Use the stamped end-of-stream time, not "now" — "now" includes
                # all the trailing TTS/playback work and produces absurd values.
                metrics.llm_total = (
                    (llm_stream_done_at - llm_start) if llm_stream_done_at else 0
                )
                metrics.tts_time = total_tts_time
                metrics.playback_time = total_playback_time
                metrics.sentences_spoken = sentences_spoken
                metrics.response = full_response.strip()
                metrics.total_e2e = time.monotonic() - turn_start

                messages.append({"role": "assistant", "content": metrics.response})

            except Exception as e:
                print(f"\n  {YELLOW}LLM error: {e}{RESET}")
                messages.pop()
                # The mic stream has been silently buffering during the
                # (possibly long) failed LLM call. Drain it so we don't
                # immediately trigger STT on stale audio. Bug #3.
                drained = flush_pending_audio(mic, chunk_size=CHUNK)
                if drained:
                    print(
                        f"  {DIM}flushed {drained} stale audio frames "
                        f"({drained / RATE:.1f}s){RESET}"
                    )
                continue

            metrics.print(turn + 1)
            all_metrics.append(metrics)
            turn += 1

            messages = trim_history(messages, max_user_assistant=20)

    except KeyboardInterrupt:
        print(f"\n\n{DIM}{'─' * 56}{RESET}")
        if all_metrics:
            stt_times = [m.stt_time for m in all_metrics]
            llm_ft = [m.llm_first_token for m in all_metrics]
            tts_times = [m.tts_time for m in all_metrics]
            ttfs_times = [m.ttfs for m in all_metrics]
            print(f"{BOLD}  Session Summary ({len(all_metrics)} turns){RESET}")
            print(f"    Median STT:       {sorted(stt_times)[len(stt_times)//2]*1000:.0f}ms")
            print(f"    Median LLM 1st:   {sorted(llm_ft)[len(llm_ft)//2]*1000:.0f}ms")
            print(f"    Median TTS:       {sorted(tts_times)[len(tts_times)//2]*1000:.0f}ms")
            print(f"    {BOLD}Median TTFS:      {sorted(ttfs_times)[len(ttfs_times)//2]*1000:.0f}ms{RESET}")
            print(f"    Best TTFS:        {min(ttfs_times)*1000:.0f}ms")
            print(f"    Model:            {llm_config['model']}")
        print()
    finally:
        mic.stop_stream()
        mic.close()
        pa.terminate()
