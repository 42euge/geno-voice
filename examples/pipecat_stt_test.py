#!/usr/bin/env python3
"""
Pipecat pipeline spike — VAD + mlx-whisper continuous STT.

Proves Pipecat can orchestrate local audio capture with Silero VAD
and our mlx-whisper engine for continuous transcription.

Usage:
    cd /Users/euge/code-red/mind-reflect-ws/geno-voice
    .venv/bin/python examples/pipecat_stt_test.py

Press Ctrl+C to stop.
"""

import asyncio
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor

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
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

SAMPLE_RATE = 16000
executor = ThreadPoolExecutor(max_workers=1)


class MLXWhisperContinuousSTT(FrameProcessor):
    """Accumulates audio while user speaks, transcribes on silence."""

    def __init__(self):
        super().__init__()
        self._audio_buffer = bytearray()
        self._speaking = False
        self._whisper = None

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
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
            no_speech_threshold=0.6,
        )
        elapsed = time.time() - t0
        text = result.get("text", "").strip()
        duration = len(audio_np) / SAMPLE_RATE
        if text:
            print(
                f"\n[mlx-whisper] {duration:.1f}s audio → {elapsed:.1f}s inference → \"{text}\"",
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


class TranscriptPrinter(FrameProcessor):
    """Prints transcription frames as they arrive."""

    def __init__(self):
        super().__init__()
        self._lines = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                self._lines.append(text)
                print(f"\n📝 {text}", flush=True)
        await self.push_frame(frame, direction)


async def main():
    print("=" * 60)
    print("  MindReflect — Pipecat Continuous STT Spike")
    print("  Speak into your microphone.")
    print("  VAD detects speech → mlx-whisper transcribes.")
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
    printer = TranscriptPrinter()

    pipeline = Pipeline([
        transport.input(),
        stt,
        printer,
    ])

    runner = PipelineRunner()
    task = PipelineTask(
        pipeline,
        PipelineParams(allow_interruptions=True),
    )

    print("\n[pipeline] Listening...\n", file=sys.stderr)

    try:
        await runner.run(task)
    except KeyboardInterrupt:
        print("\n\n[pipeline] Stopped.", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
