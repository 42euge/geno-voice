#!/usr/bin/env python3
"""THROWAWAY PROTOTYPE — feed a WAV through one real agent turn.

Question: can geno-voice replace the physical microphone/speaker with its
existing virtual devices and still complete STT -> Blue/custom LLM -> TTS?

Run automated (silent; never opens a hardware output device):
  .venv/bin/python examples/prototype_wav_agent.py INPUT.wav

Run as a human test (plays the captured response after the turn):
  .venv/bin/python examples/prototype_wav_agent.py INPUT.wav --human-test

The virtual microphone receives normalized 16 kHz mono PCM plus trailing
silence so the production VAD closes the turn. The virtual speaker captures
Kokoro's 24 kHz response into /tmp/geno-voice-wav-agent-response.wav.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examples._chat_llm import stream_chat_completion
from examples._chat_loop import ChatLoop
from examples._chat_tts import synthesize_with_alignment
from examples.mic_chat import load_llm_config
from examples.virtual_audio import VirtualMicStream, VirtualSpeakerStream, concat, make_silence
from stt import get_engine as get_stt_engine
from tts import get_engine as get_tts_engine

RATE = 16_000
TTS_RATE = 24_000


def load_input(path: Path) -> np.ndarray:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if source_rate != RATE:
        source_x = np.arange(len(mono), dtype=np.float64)
        target_n = round(len(mono) * RATE / source_rate)
        target_x = np.linspace(0, max(0, len(mono) - 1), target_n)
        mono = np.interp(target_x, source_x, mono).astype(np.float32)
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    if peak > 0:
        mono = mono * (0.7 / peak)
    pcm = (np.clip(mono, -1, 1) * 32767).astype(np.int16)
    return concat(make_silence(0.25, RATE), pcm, make_silence(1.25, RATE))


def render(state: dict) -> None:
    print("\nPROTOTYPE STATE")
    print(json.dumps(state, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--human-test",
        "--play",
        dest="human_test",
        action="store_true",
        help="play the captured response; automated runs are silent by default",
    )
    output_mode.add_argument(
        "--fake-output",
        dest="human_test",
        action="store_false",
        help="capture response audio without opening an output device (default)",
    )
    parser.set_defaults(human_test=False)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/geno-voice-wav-agent-response.wav"),
    )
    parser.add_argument(
        "--stt-model",
        default="mlx-community/whisper-large-v3-turbo",
    )
    args = parser.parse_args()

    state = {
        "phase": "loading",
        "input": str(args.wav),
        "transcript": None,
        "response": None,
        "output": str(args.output),
        "output_mode": "human" if args.human_test else "fake",
    }
    render(state)

    mic = VirtualMicStream(rate=RATE, chunk_size=1024)
    mic.push(load_input(args.wav))
    speaker = VirtualSpeakerStream(rate=TTS_RATE)
    stt = get_stt_engine("whisper", model_repo=args.stt_model)
    tts = get_tts_engine("kokoro")
    llm_config = load_llm_config()

    def play(speaker_stream, audio, tokens, **kwargs):
        speaker_stream.write(np.clip(audio, -1, 1))
        return len(audio) / TTS_RATE

    loop = ChatLoop(
        mic=mic,
        speaker_factory=lambda: speaker,
        stt_engine=stt,
        llm_stream_fn=stream_chat_completion,
        llm_config=llm_config,
        synth_fn=lambda text: synthesize_with_alignment(tts, text, "af_heart", 1.0),
        play_fn=play,
        barge_in_enabled=False,
    )

    state["phase"] = "running"
    render(state)
    result = loop.run_one_turn(
        [{"role": "system", "content": "You are a concise voice assistant."}]
    )
    if result.metrics is None:
        state["phase"] = "failed"
        render(state)
        return 1

    state["transcript"] = result.metrics.transcript
    state["response"] = result.metrics.response
    state["phase"] = "complete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, speaker.captured_float32, TTS_RATE, subtype="PCM_16")
    render(state)

    if args.human_test:
        subprocess.run(["afplay", str(args.output)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
