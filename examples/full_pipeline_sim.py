#!/usr/bin/env python3
"""
Full M1 pipeline simulation — no mic needed.

Simulates a realistic reflection session through all components:
  VAD events → Whisper transcription → activation tracking →
  NLP triggers → turn-taking engine → session notes (Ollama)

Exercises the entire M1 architecture with realistic timing and
content, producing real session notes via Ollama tool use.

Usage:
    cd /Users/euge/code-red/mind-reflect-ws/geno-voice
    .venv/bin/python examples/full_pipeline_sim.py
"""

import asyncio
import json
import logging
import struct
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipecat.frames.frames import (
    EndFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from session.activation import ActivationTracker
from session.compute import ComputeMonitor, ComputeMonitorProcessor
from session.notes import SessionNoteProcessor
from session.triggers import detect_triggers
from session.turn_taking import TurnTakingEngine, Action

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("sim")

ACTION_ICONS = {
    Action.STAY_SILENT: "🤫",
    Action.PLAY_CUE: "💬",
    Action.SPEAK_BRIEF: "🗣️",
    Action.SPEAK_FULL: "📢",
    Action.GENTLE_PROMPT: "🌿",
}

SCENARIO = [
    {
        "speech_secs": 8,
        "silence_secs": 1.5,
        "text": "I've been thinking a lot about work lately. It's been really stressful, you know? My manager keeps piling on projects and I don't feel like I can say no.",
        "tone": "normal",
        "pitch": 150,
        "amplitude": 0.12,
    },
    {
        "speech_secs": 7,
        "silence_secs": 1.0,
        "text": "And like, it's not just the workload. It's the feeling that nothing I do is ever enough. I finished that big report last week and didn't even get a thank you.",
        "tone": "frustrated",
        "pitch": 180,
        "amplitude": 0.18,
    },
    {
        "speech_secs": 9,
        "silence_secs": 2.0,
        "text": "I don't know, maybe I'm overthinking it. But it's been keeping me up at night. I just lie there thinking about all the things I should have done differently.",
        "tone": "sad",
        "pitch": 130,
        "amplitude": 0.08,
    },
    {
        "speech_secs": 4,
        "silence_secs": 5.0,
        "text": "Yeah I don't know, it's just hard.",
        "tone": "resigned",
        "pitch": 120,
        "amplitude": 0.07,
    },
    {
        "speech_secs": 6,
        "silence_secs": 1.0,
        "text": "I keep thinking about whether I should just quit. But then I think about the mortgage and the kids and I feel trapped.",
        "tone": "anxious",
        "pitch": 200,
        "amplitude": 0.22,
    },
    {
        "speech_secs": 3,
        "silence_secs": 2.0,
        "text": "What do you think? Am I being unreasonable?",
        "tone": "seeking",
        "pitch": 170,
        "amplitude": 0.14,
    },
]


def make_audio(freq, amplitude, duration_s, sr=16000):
    t = np.arange(int(sr * duration_s)) / sr
    noise = np.random.normal(0, 0.01, len(t))
    samples = ((amplitude * np.sin(2 * np.pi * freq * t) + noise) * 32767).astype(np.int16)
    return samples.tobytes()


class SimulatedPipelineProcessor(FrameProcessor):
    """Processes frames through compute monitor + activation + turn-taking."""

    def __init__(self, engine: TurnTakingEngine, activation: ActivationTracker, monitor: ComputeMonitor):
        super().__init__()
        self._engine = engine
        self._activation = activation
        self._monitor = monitor
        self._last_speech_end = time.time()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._last_speech_end = time.time()
            self._monitor.on_stt_done()

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if not text:
                await self.push_frame(frame, direction)
                return

            silence = time.time() - self._last_speech_end
            activation_state = self._activation.state

            self._engine.update_state(
                emotional_content=activation_state.is_elevated,
                user_crying=activation_state.is_crying,
            )

            decision = self._engine.decide(
                silence_duration_secs=silence,
                smart_turn_confidence=0.5,
                transcript_chunk=text,
            )

            icon = ACTION_ICONS.get(decision.action, "❓")
            pipe = self._monitor.state.pipeline.value
            bg_ok = self._monitor.can_run_background_llm()

            print(f"\n📝 \"{text}\"")
            print(f"   Activation: {activation_state.score:.2f} (fast={activation_state.fast_ema:.2f} slow={activation_state.slow_ema:.2f} traj={activation_state.trajectory:+.2f} cry={activation_state.is_crying})")
            print(f"   Compute: {pipe} | bg_llm={bg_ok} | cancel_llm={self._monitor.should_cancel_llm()}")
            print(f"   {icon} Turn: {decision.action.value} — {decision.reason}")
            if decision.cue:
                print(f"   Cue: {decision.cue.cue_type}")

        await self.push_frame(frame, direction)


class ChunkFeeder(FrameProcessor):
    """Feeds simulated speech through the pipeline with realistic timing."""

    def __init__(self, scenario, activation_tracker):
        super().__init__()
        self._scenario = scenario
        self._activation = activation_tracker

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def run(self, task):
        for i, chunk in enumerate(self._scenario):
            print(f"\n{'='*60}")
            print(f"  Chunk {i+1}/{len(self._scenario)}: [{chunk['tone']}] {chunk['speech_secs']}s speech, {chunk['silence_secs']}s silence")
            print(f"{'='*60}")

            audio = make_audio(chunk["pitch"], chunk["amplitude"], chunk["speech_secs"])
            self._activation.process_chunk(audio)

            await task.queue_frame(VADUserStartedSpeakingFrame())
            await asyncio.sleep(0.1)
            await task.queue_frame(VADUserStoppedSpeakingFrame())
            await asyncio.sleep(chunk["silence_secs"])

            await task.queue_frame(
                TranscriptionFrame(
                    text=chunk["text"],
                    user_id="sim-user",
                    timestamp=str(time.time()),
                )
            )

            await asyncio.sleep(12)

        await asyncio.sleep(5)
        await task.queue_frame(EndFrame())


async def main():
    session_dir = tempfile.mkdtemp(prefix="mindreflect-sim-")

    engine = TurnTakingEngine()
    engine.state.session_start = time.time() - 300
    activation = ActivationTracker(baseline_chunks=2)
    monitor = ComputeMonitor()

    # Pre-fill baseline with first two "calm" chunks
    for chunk in SCENARIO[:2]:
        audio = make_audio(chunk["pitch"], chunk["amplitude"], 2.0)
        activation.process_chunk(audio)

    print("=" * 60)
    print("  MindReflect — Full M1 Pipeline Simulation")
    print()
    print("  All 7 modules: VAD → Compute → Activation → Turn-Taking → Notes")
    print()
    print(f"  Session notes: {session_dir}")
    print(f"  Scenario: {len(SCENARIO)} chunks, ~37s speech")
    print(f"  Notes model: gemma4:e2b (Ollama)")
    print("=" * 60)

    feeder = ChunkFeeder(SCENARIO, activation)
    compute_proc = ComputeMonitorProcessor(monitor)
    sim = SimulatedPipelineProcessor(engine, activation, monitor)
    notes = SessionNoteProcessor(session_dir=session_dir, model="gemma4:e2b")

    pipeline = Pipeline([feeder, compute_proc, sim, notes])
    runner = PipelineRunner()
    task = PipelineTask(pipeline)

    runner_task = asyncio.create_task(runner.run(task))
    await asyncio.sleep(2)
    await feeder.run(task)
    await runner_task

    notes._update_meta()

    print("\n" + "=" * 60)
    print("  Session Results")
    print("=" * 60)

    print(f"\n  Speaking total: {engine.state.user_speaking_total_secs:.0f}s")
    print(f"  Chunks processed: {engine.state.chunks_since_last_response}")
    print(f"  Activation: score={activation.state.score:.2f} traj={activation.state.trajectory:+.2f}")

    for f in ["summary.md", "moments.jsonl", "meta.json"]:
        p = Path(session_dir) / f
        if p.exists() and p.stat().st_size > 0:
            content = p.read_text()
            print(f"\n--- {f} ---")
            if len(content) > 600:
                print(content[:600] + "...")
            else:
                print(content)

    wiki_dir = Path(session_dir) / "wiki"
    pages = list(wiki_dir.glob("*.md"))
    if pages:
        print(f"\n--- wiki/ ({len(pages)} pages) ---")
        for p in pages:
            print(f"  {p.name}")

    print(f"\nFull output: {session_dir}")


if __name__ == "__main__":
    asyncio.run(main())
