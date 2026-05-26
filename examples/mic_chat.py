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


# Pure parse + validate logic lives in examples/_chat_config; this
# module just handles file I/O and CLI exit behavior. iter-018.
from examples._chat_config import (  # noqa: E402
    ConfigError,
    parse_chat_config,
    parse_llm_config,
)


def _read_yaml_or_exit() -> object:
    """Read CONFIG_PATH and return the parsed YAML object (or None
    if the file is missing). Exits the process on missing file
    or YAML parse errors — both are unrecoverable for the chat CLI.
    """
    if not CONFIG_PATH.exists():
        print(f"  {YELLOW}Missing config.local.yaml — create it with llm settings{RESET}")
        sys.exit(1)
    try:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"  {YELLOW}config.local.yaml is invalid YAML: {e}{RESET}")
        sys.exit(1)


def load_chat_config() -> dict:
    """Read the optional ``chat`` section of config.local.yaml.

    Returns an empty dict if the file is missing or the section is
    absent. Used for iter-011 filler-word config (``chat.fillers``,
    ``chat.fillers_idle_threshold``).
    """
    if not CONFIG_PATH.exists():
        return {}
    cfg = _read_yaml_or_exit()
    return parse_chat_config(cfg)


def load_llm_config() -> dict:
    """Read + validate the ``llm`` section of config.local.yaml.

    On any structural problem (missing file, missing section,
    missing required fields, unresolved ``${ENV_VAR}`` placeholder
    in ``api_key``) prints a helpful message and exits.
    """
    cfg = _read_yaml_or_exit()
    try:
        return parse_llm_config(cfg)
    except ConfigError as e:
        print(f"  {YELLOW}{e}{RESET}")
        sys.exit(1)


# llm_stream now lives in examples/_chat_llm.py. Re-exported here so
# any external callers / tests that import it from mic_chat keep
# working.
from examples._chat_llm import stream_chat_completion as llm_stream  # noqa: E402,F401


# TurnMetrics moved to examples/_chat_metrics.py so it's importable
# without pulling in mic_chat's top-level pyaudio dependency. iter-014.
from examples._chat_metrics import TurnMetrics  # noqa: E402,F401


# synthesize_with_alignment moved to examples/_chat_tts.py so it's
# importable + testable without pyaudio at module scope. iter-019.
from examples._chat_tts import synthesize_with_alignment  # noqa: E402,F401


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
    """Main chat loop with streaming LLM + sentence-by-sentence TTS.

    ``model_repo`` is the FALLBACK STT model when chat config
    omits ``stt_model``. iter-119 lets operators set
    ``chat.stt_engine`` / ``stt_model`` in config.local.yaml to
    pick faster_whisper for x86_64 Linux; the function arg is
    still honored for backwards compatibility.
    """
    llm_config = load_llm_config()

    # iter-119: load chat_cfg up front so stt_config is available
    # before load_engines. The original code loaded chat_cfg AFTER
    # the STT engine was constructed, hardcoding WhisperEngine.
    # Now stt_cfg drives the engine choice.
    from examples._chat_config import (
        parse_filler_config, parse_stt_config, parse_vad_config,
    )
    chat_cfg = load_chat_config()
    stt_cfg = parse_stt_config(chat_cfg)
    # Empty `stt_model` → use the function-arg `model_repo` so
    # legacy callers without yaml-set models keep working.
    stt_model = stt_cfg["model"] or model_repo
    stt_engine_name = stt_cfg["engine"]

    model_short = stt_model.split("/")[-1] if stt_model else "(default)"
    print(f"\n{BOLD}gv chat{RESET}")
    print(
        f"{DIM}stt: {stt_engine_name}/{model_short} │ "
        f"llm: {llm_config['model']} │ tts: kokoro/{voice}{RESET}"
    )
    print()

    # iter-108: engine loading + timing + log moved to
    # examples/_chat_engines so the sequence is testable without
    # importing mlx-whisper / kokoro at the test level.
    from examples._chat_engines import load_engines
    from stt import get_engine as _get_stt_engine

    def _stt_factory():
        # iter-119: route to whichever engine the chat config
        # selected. faster_whisper takes device + compute_type;
        # whisper accepts only model_repo.
        kwargs = {}
        if stt_model:
            kwargs["model_repo"] = stt_model
        if stt_engine_name == "faster_whisper":
            kwargs["device"] = stt_cfg["device"]
            kwargs["compute_type"] = stt_cfg["compute_type"]
        return _get_stt_engine(stt_engine_name, **kwargs)

    engines = load_engines(
        stt_factory=_stt_factory,
        tts_factory=lambda: get_tts_engine("kokoro"),
        log=lambda line: print(f"  {line}"),
    )
    stt_engine = engines.stt
    tts_engine = engines.tts

    # Pre-render fillers (iter-011). Empty by default; opt in via
    # config.local.yaml:
    #   chat:
    #     fillers: ["hmm", "let me think", "well,"]
    #     fillers_idle_threshold: 0.6
    # iter-119: chat_cfg + parse_*_config imports moved up to
    # support stt_cfg. Reused here.
    filler_cfg = parse_filler_config(chat_cfg)
    filler_texts: list[str] = filler_cfg["texts"]
    filler_idle_threshold: float = filler_cfg["idle_threshold"]
    # iter-020: optional VAD tuning. parse_vad_config defaults
    # match the _chat_recording module constants and tolerates
    # malformed user input.
    vad_cfg = parse_vad_config(chat_cfg)
    # iter-107: filler pre-rendering moved to examples/_chat_fillers
    # so the loop is testable without a real TTS engine. The
    # caller supplies a closure that wraps synthesize_with_alignment,
    # plus a log callable that re-applies the YELLOW/leading-space
    # styling for failure lines + plain styling for the summary.
    from examples._chat_fillers import prerender_fillers

    def _filler_log(line: str) -> None:
        if line.startswith("filler synth failed"):
            print(f"  {YELLOW}{line}{RESET}")
        else:
            print(f"  {line}")

    rendered_fillers = prerender_fillers(
        lambda text: synthesize_with_alignment(tts_engine, text, voice, speed),
        filler_texts,
        idle_threshold=filler_idle_threshold,
        log=_filler_log,
    )

    print(f"  {GREEN}Ready.{RESET} Speak and I'll respond. Ctrl+C to quit.\n")

    pa = pyaudio.PyAudio()
    mic = pa.open(format=pyaudio.paInt16, channels=CHANNELS,
                  rate=RATE, input=True, frames_per_buffer=CHUNK)

    # iter-109: speaker_factory + synth + play closures moved to
    # examples/_chat_audio_io. ChatLoop is dep-injected (iter-015);
    # build_audio_io produces the production wiring (pyaudio +
    # synthesize_with_alignment + _play_aligned_core).
    from examples._chat_audio_io import build_audio_io

    audio_io = build_audio_io(pa, tts_engine, voice, speed)

    # iter-088: optional aggressive first-sentence splitter. Reduces
    # TTFS on long-preamble responses at the cost of some prosody.
    # Read from chat_cfg with a False default — opt-in.
    aggressive_first_sentence = bool(
        chat_cfg.get("aggressive_first_sentence", False)
    )
    # iter-093: optional auto-aggressive-on-stall threshold. When
    # >0, a mid-stream LLM token gap above this many seconds
    # flips the splitter to aggressive mode mid-turn so audio
    # recovers faster from the stall. 0.0 = disabled.
    auto_aggressive_threshold = float(
        chat_cfg.get("auto_aggressive_threshold", 0.0)
    )
    chat_loop = ChatLoop(
        mic=mic,
        speaker_factory=audio_io.speaker_factory,
        rate=RATE,
        chunk=CHUNK,
        silence_threshold=vad_cfg["silence_threshold"],
        silence_duration=vad_cfg["silence_duration"],
        min_speech_duration=vad_cfg["min_speech_duration"],
        stt_engine=stt_engine,
        llm_stream_fn=llm_stream,
        llm_config=llm_config,
        synth_fn=audio_io.synth_fn,
        play_fn=audio_io.play_fn,
        fillers=rendered_fillers,
        idle_threshold=filler_idle_threshold,
        aggressive_first_sentence=aggressive_first_sentence,
        auto_aggressive_threshold=auto_aggressive_threshold,
    )

    # iter-110: main turn loop + KeyboardInterrupt handler moved
    # to examples/_chat_session.run_session. State that previously
    # lived as 6 mutable locals now flows through SessionState.
    from examples._chat_session import run_session

    system_prompt = llm_config.get(
        "system_prompt", "You are a concise voice assistant.",
    )
    state = run_session(
        chat_loop,
        system_prompt,
        max_user_assistant=20,
        prompt_log=lambda turn: print(
            f"  {DIM}[{turn}] waiting...{RESET}", end="", flush=True,
        ),
    )

    # iter-017 / iter-086: hand the populated SessionState to the
    # session-summary aggregator. Field names line up 1:1 with
    # SessionMeta so this is mechanical.
    try:
        from examples._chat_metrics import (
            print_session_summary,
            SessionMeta,
        )
        print_session_summary(
            state.all_metrics, llm_config,
            meta=SessionMeta(
                false_triggers=state.false_triggers,
                session_seconds=time.monotonic() - state.session_start,
                llm_errors=state.llm_errors,
                trim_events=state.trim_events,
                trim_messages_evicted=state.trim_messages_evicted,
                idle_threshold=filler_idle_threshold,
            ),
        )
    finally:
        mic.stop_stream()
        mic.close()
        pa.terminate()
