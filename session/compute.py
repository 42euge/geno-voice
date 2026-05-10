"""
Compute monitor — lightweight resource orchestration for Pipecat.

Tracks pipeline state (user speaking, thinking, responding) and
implements priority-based gating:
  STT > TTS > LLM conversation > LLM background

Preemption is cooperative:
  - LLM: cancel between token generations when user starts speaking
  - TTS: stop feeding new text (existing audio buffer plays out)
  - STT: never preempted

Does NOT manage GPU scheduling (macOS Metal handles that) or memory
(Ollama/MLX handle their own). This is a ~150-line state tracker,
not a kernel scheduler.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from pipecat.frames.frames import (
    Frame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

log = logging.getLogger("compute-monitor")


class PipelineState(Enum):
    IDLE = "idle"
    USER_SPEAKING = "user_speaking"
    PROCESSING_STT = "processing_stt"
    THINKING = "thinking"
    RESPONDING = "responding"


@dataclass
class ComputeState:
    pipeline: PipelineState = PipelineState.IDLE
    user_speaking: bool = False
    stt_active: bool = False
    llm_active: bool = False
    tts_active: bool = False
    last_vad_start: float | None = None
    last_vad_stop: float | None = None


class ComputeMonitor:
    """Resource orchestration for the voice pipeline.

    Provides gating and preemption signals that pipeline components
    check before and during inference.
    """

    def __init__(self):
        self.state = ComputeState()
        self._llm_gate = asyncio.Event()
        self._llm_gate.set()
        self._cancel_llm = asyncio.Event()
        self._cancel_tts = asyncio.Event()

    def on_vad_start(self):
        self.state.user_speaking = True
        self.state.pipeline = PipelineState.USER_SPEAKING
        self.state.last_vad_start = time.time()
        self._cancel_llm.set()
        self._cancel_tts.set()
        self._llm_gate.clear()
        log.debug("VAD start → user speaking, LLM gated, TTS cancelled")

    def on_vad_stop(self):
        self.state.user_speaking = False
        self.state.pipeline = PipelineState.PROCESSING_STT
        self.state.last_vad_stop = time.time()
        self._cancel_llm.clear()
        self._cancel_tts.clear()
        self._llm_gate.set()
        log.debug("VAD stop → STT processing, LLM ungated")

    def on_stt_start(self):
        self.state.stt_active = True

    def on_stt_done(self):
        self.state.stt_active = False
        if not self.state.user_speaking:
            self.state.pipeline = PipelineState.THINKING

    def on_llm_start(self):
        self.state.llm_active = True
        self.state.pipeline = PipelineState.THINKING

    def on_llm_done(self):
        self.state.llm_active = False
        if not self.state.tts_active:
            self.state.pipeline = PipelineState.IDLE

    def on_tts_start(self):
        self.state.tts_active = True
        self.state.pipeline = PipelineState.RESPONDING

    def on_tts_done(self):
        self.state.tts_active = False
        if not self.state.llm_active:
            self.state.pipeline = PipelineState.IDLE

    async def gate_llm(self):
        """Block until LLM inference is allowed (user not speaking)."""
        await self._llm_gate.wait()

    def should_cancel_llm(self) -> bool:
        """Check between token generations — cancel if user started speaking."""
        return self._cancel_llm.is_set()

    def should_cancel_tts(self) -> bool:
        """Check before feeding new text to TTS."""
        return self._cancel_tts.is_set()

    def can_run_background_llm(self) -> bool:
        """Whether background LLM work (notes, summary) is allowed."""
        return (
            not self.state.user_speaking
            and not self.state.stt_active
            and not self.state.llm_active
        )


class ComputeMonitorProcessor(FrameProcessor):
    """Pipecat FrameProcessor that updates the compute monitor from VAD events."""

    def __init__(self, monitor: ComputeMonitor):
        super().__init__()
        self.monitor = monitor

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self.monitor.on_vad_start()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self.monitor.on_vad_stop()

        await self.push_frame(frame, direction)
