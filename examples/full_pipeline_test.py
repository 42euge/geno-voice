#!/usr/bin/env python3
"""
MindReflect full pipeline — mic → VAD → Whisper → session notes.

Combines the STT spike and SessionNoteProcessor into a single
Pipecat pipeline. Speak into the mic; transcriptions flow to
the LLM which maintains live session notes via tool use.

Architecture:
    LocalAudioTransport (mic)
        → SileroVAD
        → MLXWhisperContinuousSTT
        → SessionNoteProcessor (gemma4:e2b, background tool use)
        → TranscriptPrinter (terminal display)

Usage:
    cd /Users/euge/code-red/mind-reflect-ws/geno-voice
    .venv/bin/python examples/full_pipeline_test.py

Session notes are written to /tmp/mindreflect-session-<timestamp>/
Press Ctrl+C to stop.
"""

import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from session.notes import SessionNoteProcessor

SAMPLE_RATE = 16000
executor = ThreadPoolExecutor(max_workers=1)


class MLXWhisperContinuousSTT(FrameProcessor):
    """Accumulates audio while user speaks, transcribes on silence."""

    def __init__(self, model_repo="mlx-community/whisper-large-v3-turbo"):
        super().__init__()
        self._audio_buffer = bytearray()
        self._speaking = False
        self._whisper = None
        self._model_repo = model_repo

    def _ensure_model(self):
        if self._whisper is None:
            import mlx_whisper
            self._whisper = mlx_whisper
            print("[mlx-whisper] Model ready", file=sys.stderr)

    def _transcribe(self, audio_bytes: bytes) -> str:
        self._ensure_model()
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio_np) < SAMPLE_RATE * 0.5:
            return ""
        t0 = time.time()
        result = self._whisper.transcribe(
            audio_np,
            path_or_hf_repo=self._model_repo,
            no_speech_threshold=0.6,
        )
        elapsed = time.time() - t0
        text = result.get("text", "").strip()
        duration = len(audio_np) / SAMPLE_RATE
        if text:
            print(
                f"\n[stt] {duration:.1f}s audio → {elapsed:.1f}s → \"{text}\"",
                file=sys.stderr,
            )
        return text

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._speaking = True
            self._audio_buffer = bytearray()
            print("\n🎙️  Speaking...", file=sys.stderr, end="", flush=True)

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speaking = False
            print(" stopped.", file=sys.stderr)
            if len(self._audio_buffer) > SAMPLE_RATE:
                audio = bytes(self._audio_buffer)
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(executor, self._transcribe, audio)
                if text:
                    await self.push_frame(
                        TranscriptionFrame(text=text, user_id="user", timestamp=str(time.time())),
                        direction,
                    )
            self._audio_buffer = bytearray()

        elif isinstance(frame, InputAudioRawFrame) and self._speaking:
            self._audio_buffer.extend(frame.audio)

        await self.push_frame(frame, direction)


class TranscriptDisplay(FrameProcessor):
    """Displays transcription frames in the terminal."""

    def __init__(self):
        super().__init__()
        self._count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                self._count += 1
                print(f"\n📝 [{self._count}] {text}", flush=True)
        await self.push_frame(frame, direction)


async def main():
    session_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    session_dir = f"/tmp/mindreflect-session-{session_id}"

    print("=" * 60)
    print("  MindReflect — Full Pipeline Test")
    print()
    print("  mic → VAD → Whisper STT → Session Notes (Ollama)")
    print()
    print(f"  Session notes: {session_dir}")
    print(f"  Notes model:   gemma4:e2b")
    print(f"  STT model:     whisper-large-v3-turbo (MLX)")
    print()
    print("  Speak into your microphone.")
    print("  Transcriptions appear as 📝, notes are written live.")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(min_volume=0.02, stop_secs=0.8),
            ),
        )
    )

    stt = MLXWhisperContinuousSTT()
    notes = SessionNoteProcessor(session_dir=session_dir, model="gemma4:e2b")
    display = TranscriptDisplay()

    pipeline = Pipeline([
        transport.input(),
        stt,
        notes,
        display,
    ])

    runner = PipelineRunner()
    task = PipelineTask(pipeline)

    print("\n[pipeline] Listening...\n", file=sys.stderr)

    try:
        await runner.run(task)
    except KeyboardInterrupt:
        notes._update_meta()
        print(f"\n\n[pipeline] Stopped. Session notes at: {session_dir}", file=sys.stderr)

        for f in ["summary.md", "moments.jsonl"]:
            p = Path(session_dir) / f
            if p.exists() and p.stat().st_size > 0:
                print(f"\n--- {f} ---")
                print(p.read_text()[:500])


if __name__ == "__main__":
    asyncio.run(main())
