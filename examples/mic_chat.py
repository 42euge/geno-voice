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

# Playback loop — extracted similarly, takes any speaker-shaped stream.
from examples._chat_playback import (
    TTS_RATE,
    play_aligned as _play_aligned_core,
)

# Streaming-overlap worker + barge-in primitives. Re-exported here
# so external imports keep working; the actual orchestration lives
# in examples/_chat_loop.py (iter-015).
from examples._chat_pipeline import (  # noqa: F401
    BargeInCoordinator,
    BargeInWatcher,
    SentenceWorker,
)
from examples._chat_loop import ChatLoop

DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.local.yaml"


def load_chat_config() -> dict:
    """Read the optional ``chat`` section of config.local.yaml.

    Returns a dict — empty if the section is missing. Used for
    iter-011 filler-word config (``chat.fillers``,
    ``chat.fillers_idle_threshold``).
    """
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("chat", {}) or {}


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


# llm_stream now lives in examples/_chat_llm.py. Re-exported here so
# any external callers / tests that import it from mic_chat keep
# working.
from examples._chat_llm import stream_chat_completion as llm_stream  # noqa: E402,F401


# TurnMetrics moved to examples/_chat_metrics.py so it's importable
# without pulling in mic_chat's top-level pyaudio dependency. iter-014.
from examples._chat_metrics import TurnMetrics  # noqa: E402,F401


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
    """Open a per-sentence PyAudio output stream and run the core
    play loop from examples/_chat_playback.py against it.

    The persistent-stream optimization (open once, reuse across
    sentences) is iter-008 streaming-overlap territory; for now we
    preserve the current open-per-sentence behavior so the iter-007
    extraction is purely structural.
    """
    play_chunk = 1024  # ~42ms at 24kHz
    out_stream = pa.open(
        format=pyaudio.paInt16, channels=1,
        rate=TTS_RATE, output=True,
        frames_per_buffer=play_chunk,
    )
    try:
        return _play_aligned_core(
            out_stream,
            audio_np,
            tokens,
            is_first_sentence=is_first_sentence,
            play_chunk=play_chunk,
            rate=TTS_RATE,
        )
    finally:
        out_stream.stop_stream()
        out_stream.close()


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

    # Pre-render fillers (iter-011). Empty by default; opt in via
    # config.local.yaml:
    #   chat:
    #     fillers: ["hmm", "let me think", "well,"]
    #     fillers_idle_threshold: 0.6
    chat_cfg = load_chat_config()
    filler_texts: list[str] = list(chat_cfg.get("fillers") or [])
    filler_idle_threshold: float = float(chat_cfg.get("fillers_idle_threshold", 0.6))
    rendered_fillers: list[tuple] = []
    if filler_texts:
        t_fill = time.monotonic()
        for text in filler_texts:
            try:
                audio_np, tokens = synthesize_with_alignment(
                    tts_engine, text, voice, speed,
                )
                if len(audio_np) > 0:
                    rendered_fillers.append((audio_np, tokens))
            except Exception as e:
                print(f"  {YELLOW}filler synth failed for {text!r}: {e}{RESET}")
        print(
            f"  Pre-rendered {len(rendered_fillers)}/{len(filler_texts)} "
            f"fillers in {(time.monotonic() - t_fill)*1000:.0f}ms "
            f"(idle threshold {filler_idle_threshold:.2f}s)"
        )

    print(f"  {GREEN}Ready.{RESET} Speak and I'll respond. Ctrl+C to quit.\n")

    pa = pyaudio.PyAudio()
    mic = pa.open(format=pyaudio.paInt16, channels=CHANNELS,
                  rate=RATE, input=True, frames_per_buffer=CHUNK)

    # Wire up real dependencies for ChatLoop. The class itself is
    # platform-agnostic and dep-injected (iter-015); these closures
    # bind it to PyAudio + kokoro + the requests-backed LLM.
    def _speaker_factory():
        return pa.open(
            format=pyaudio.paInt16, channels=1,
            rate=TTS_RATE, output=True,
            frames_per_buffer=1024,
        )

    def _synth(sentence: str):
        return synthesize_with_alignment(tts_engine, sentence, voice, speed)

    def _play(speaker, audio_np, tokens, *, is_first_sentence=False, cancel_event=None):
        return _play_aligned_core(
            speaker, audio_np, tokens,
            is_first_sentence=is_first_sentence,
            rate=TTS_RATE,
            cancel_event=cancel_event,
        )

    chat_loop = ChatLoop(
        mic=mic,
        speaker_factory=_speaker_factory,
        rate=RATE,
        chunk=CHUNK,
        silence_duration=SILENCE_DURATION,
        stt_engine=stt_engine,
        llm_stream_fn=llm_stream,
        llm_config=llm_config,
        synth_fn=_synth,
        play_fn=_play,
        fillers=rendered_fillers,
        idle_threshold=filler_idle_threshold,
    )

    system_prompt = llm_config.get("system_prompt", "You are a concise voice assistant.")
    messages = [{"role": "system", "content": system_prompt}]
    all_metrics = []
    primed_frames: list[bytes] | None = None

    try:
        turn = 0
        while True:
            print(f"  {DIM}[{turn + 1}] waiting...{RESET}", end="", flush=True)
            result = chat_loop.run_one_turn(messages, primed_frames=primed_frames)
            primed_frames = result.next_primed_frames
            if result.had_error:
                continue
            if result.metrics is None:
                continue
            print()  # newline after the streamed bot text
            result.metrics.print(turn + 1)
            all_metrics.append(result.metrics)
            turn += 1
            messages = ChatLoop.trim_messages(messages, max_user_assistant=20)

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
