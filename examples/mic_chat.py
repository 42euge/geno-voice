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

# Streaming-overlap worker + barge-in primitives — runs synth + play
# on a background thread (iter-008) and lets a mic-side watcher
# cancel mid-sentence when the user starts speaking (iter-009/010).
from examples._chat_pipeline import BargeInWatcher, SentenceWorker

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

    system_prompt = llm_config.get("system_prompt", "You are a concise voice assistant.")
    messages = [{"role": "system", "content": system_prompt}]
    all_metrics = []

    # Frames captured by a BargeInWatcher during the previous bot
    # response, if any. Fed into the next record_utterance_streaming
    # call so the user's first syllables aren't dropped. iter-010.
    primed_frames: list[bytes] | None = None

    try:
        turn = 0
        while True:
            print(f"  {DIM}[{turn + 1}] waiting...{RESET}", end="", flush=True)
            wav_bytes, speech_dur, stt_time = record_utterance_streaming(
                mic, stt_engine, primed_frames=primed_frames,
            )
            primed_frames = None  # consumed
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

            # Spin up a background SentenceWorker that owns one persistent
            # speaker stream and processes complete sentences off a queue.
            # While it synthesizes + plays, the main thread stays in the
            # for-token loop so LLM receipt is no longer blocked. iter-008.
            def _speaker_factory():
                return pa.open(
                    format=pyaudio.paInt16, channels=1,
                    rate=TTS_RATE, output=True,
                    frames_per_buffer=1024,
                )

            def _synth(sentence: str):
                return synthesize_with_alignment(tts_engine, sentence, voice, speed)

            def _play(speaker, audio_np, tokens, *, is_first_sentence=False):
                return _play_aligned_core(
                    speaker, audio_np, tokens,
                    is_first_sentence=is_first_sentence,
                    rate=TTS_RATE,
                )

            worker = SentenceWorker(
                speaker_factory=_speaker_factory,
                synth_fn=_synth,
                play_fn=_play,
                fillers=rendered_fillers,
                idle_threshold=(
                    filler_idle_threshold if rendered_fillers else 0.0
                ),
            )
            worker.start()

            # Drop any audio the mic has buffered during the LLM
            # round-trip — otherwise the watcher would interpret it
            # as fresh user speech the moment we start it.
            flush_pending_audio(mic, chunk_size=CHUNK)

            # Watch the mic for user barge-in while the bot is
            # speaking. On detection, fire worker.cancel so the
            # current sentence breaks mid-stream and pending ones
            # are dropped. Captured frames feed the next record loop
            # so the user's first syllables aren't lost. iter-010.
            watcher = BargeInWatcher(
                mic=mic,
                on_speech_detected=lambda: worker.cancel(timeout=5.0),
                chunk_size=CHUNK,
                rate=RATE,
            )
            watcher.start()

            llm_stream_done_at = None
            try:
                for token in llm_stream(messages, llm_config):
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    token_buffer += token
                    full_response += token

                    complete, token_buffer = split_complete_sentences(token_buffer)
                    for sentence in complete:
                        worker.submit(sentence)

                # Stamp end-of-stream BEFORE waiting on TTS/playback so
                # llm_total measures only the LLM streaming window
                # (bug #2 from iter-001 still applies in this code path).
                llm_stream_done_at = time.monotonic()

                remaining = token_buffer.strip()
                if remaining:
                    worker.submit(remaining)

                worker.submit_done()
                worker.wait_done(timeout=120.0)
                # Stop the watcher; if it detected user speech, save
                # its captured frames so the next record_utterance
                # call can replay them and not lose the user's
                # first syllables.
                watcher.stop(timeout=2.0)
                if watcher.detected:
                    primed_frames = list(watcher.frames)
                    print(
                        f"\n  {DIM}barge-in: replaying "
                        f"{len(primed_frames)} captured frames "
                        f"({len(primed_frames) * CHUNK / RATE:.1f}s){RESET}"
                    )
                print()  # newline after the streamed bot text

                # Pull TTFS / metrics from the worker.
                if worker.first_audio_at is not None:
                    metrics.ttfs = worker.first_audio_at - speech_ended_at
                metrics.llm_first_token = (first_token_at - llm_start) if first_token_at else 0
                metrics.llm_total = (
                    (llm_stream_done_at - llm_start) if llm_stream_done_at else 0
                )
                metrics.tts_time = worker.tts_time
                metrics.playback_time = worker.playback_time
                metrics.sentences_spoken = worker.sentences_spoken
                metrics.response = full_response.strip()
                metrics.total_e2e = time.monotonic() - turn_start

                messages.append({"role": "assistant", "content": metrics.response})

                if worker.errors:
                    # Surface any synth/play errors that occurred in the
                    # background but didn't take down the worker.
                    for err in worker.errors:
                        print(f"  {YELLOW}worker error: {err}{RESET}")

            except Exception as e:
                print(f"\n  {YELLOW}LLM error: {e}{RESET}")
                messages.pop()
                watcher.stop(timeout=2.0)
                worker.stop(timeout=5.0)  # drop pending sentences, close speaker
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
