#!/usr/bin/env python3
"""
RestReflect voice sidecar — Python captures audio, Electron displays.

Captures audio directly via PyAudio (reliable, no browser quirks).
Runs VAD, STT, triggers, session notes, and broadcasts everything
to the Electron app via WebSocket.

Usage:
    .venv/bin/python pipecat_server.py
"""

import asyncio
import json
import logging
import os
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import websockets

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame, InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from session.triggers import detect_triggers, filter_noise
from session.turn_taking import TurnTakingEngine, Action
from session.text_eou import TextAwareTurnDecider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("sidecar")

PORT = int(os.environ.get("PIPECAT_PORT", "8765"))
VOICE_SERVER = os.environ.get("VOICE_SERVER_URL", "http://127.0.0.1:5111")
SAMPLE_RATE = 16000
ws_clients: set = set()
executor = ThreadPoolExecutor(max_workers=1)
_http_session = None


async def get_http_session():
    global _http_session
    if _http_session is None:
        import aiohttp as _aiohttp
        _http_session = _aiohttp.ClientSession()
    return _http_session


async def post_notes(text):
    """Forward transcript to voice server for background note processing."""
    try:
        import aiohttp
        session = await get_http_session()
        async with session.post(
            f"{VOICE_SERVER}/notes/process",
            json={"text": text},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status == 200:
                log.info("Notes: posted chunk (%d chars)", len(text))
            else:
                log.warning("Notes endpoint returned %d", resp.status)
    except Exception as e:
        log.warning("Notes post failed: %s", e)

# Session recording
session_dir = None
raw_wav = None
raw_wav_bytes = 0


def init_session():
    global session_dir, raw_wav, raw_wav_bytes
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    session_dir = Path.home() / ".restreflect" / "sessions" / ts
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "recordings").mkdir(exist_ok=True)

    wav_path = session_dir / "recordings" / "full-session.wav"
    raw_wav = {"path": str(wav_path), "total_bytes": 0}
    import struct
    with open(wav_path, "wb") as f:
        sr = 16000
        f.write(b"RIFF")
        f.write(struct.pack("<I", 0))
        f.write(b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", 0))
    log.info("Session: %s", session_dir)


def record_audio(pcm_bytes):
    if not raw_wav:
        return
    import struct
    wav_path = raw_wav["path"]
    with open(wav_path, "ab") as f:
        f.write(pcm_bytes)
    raw_wav["total_bytes"] += len(pcm_bytes)
    with open(wav_path, "r+b") as f:
        f.seek(4)
        f.write(struct.pack("<I", 36 + raw_wav["total_bytes"]))
        f.seek(40)
        f.write(struct.pack("<I", raw_wav["total_bytes"]))


class STTProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._buf = bytearray()
        self._speaking = False
        self._whisper = None
        self._chunk_num = 0

    def _load(self):
        if not self._whisper:
            import mlx_whisper
            self._whisper = mlx_whisper
            log.info("Whisper loaded")

    def _transcribe(self, audio_bytes):
        self._load()
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) < SAMPLE_RATE * 0.5:
            return ""
        result = self._whisper.transcribe(audio, path_or_hf_repo="mlx-community/whisper-large-v3-turbo", no_speech_threshold=0.6)
        return result.get("text", "").strip()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._speaking = True
            self._buf = bytearray()

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speaking = False
            if len(self._buf) > SAMPLE_RATE * 2:
                audio = bytes(self._buf)
                text = await asyncio.get_event_loop().run_in_executor(executor, self._transcribe, audio)
                text = filter_noise(text)
                if text:
                    self._chunk_num += 1
                    # Save recording
                    if session_dir:
                        p = session_dir / "recordings" / f"chunk-{self._chunk_num:04d}.wav"
                        import io, wave as wavmod
                        buf = io.BytesIO()
                        with wavmod.open(buf, "wb") as w:
                            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
                            w.writeframes(audio)
                        p.write_bytes(buf.getvalue())

                    await self.push_frame(
                        TranscriptionFrame(text=text, user_id="user", timestamp=str(time.time())),
                        direction,
                    )
            self._buf = bytearray()

        elif isinstance(frame, InputAudioRawFrame) and self._speaking:
            self._buf.extend(frame.audio)

        # Record ALL audio
        if isinstance(frame, InputAudioRawFrame):
            record_audio(frame.audio)

        await self.push_frame(frame, direction)


class Broadcaster(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._engine = TurnTakingEngine()
        self._engine.state.session_start = time.time() - 300
        self._last_stop = time.time()
        # Source smart-turn confidence from a swappable decider instead of a
        # hardcoded 0.5 (which sat below the engine's backchannel threshold,
        # leaving the silence-driven tiers dead). The decider now also folds in
        # a rule-based text EOU signal: a transcript trailing off on a
        # conjunction / preposition / filler / ellipsis dampens the
        # silence-derived confidence, so the engine stays silent through a
        # mid-thought pause. A later lap swaps in pipecat smart-turn behind the
        # same interface. See session/text_eou.py + organic-turn-taking.md #2/#4.
        self._turn_decider = TextAwareTurnDecider()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await broadcast({"type": "vad", "speaking": True})

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._last_stop = time.time()
            await broadcast({"type": "vad", "speaking": False})

        elif isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if not text:
                return

            self._engine.update_state(user_spoke_secs=5)
            silence = time.time() - self._last_stop
            trigger = detect_triggers(text)
            confidence = self._turn_decider.confidence(
                silence_duration_secs=silence, transcript_chunk=text
            )
            decision = self._engine.decide(silence, confidence, text)

            await broadcast({
                "type": "transcript",
                "text": text,
                "trigger": {
                    "type": trigger.trigger_type.value,
                    "hint": trigger.hint.value,
                } if trigger.triggered else None,
                "turn": decision.action.value,
            })
            asyncio.create_task(post_notes(text))

            if decision.action == Action.play_cue and not trigger.triggered:
                asyncio.create_task(broadcast_cue())

        await self.push_frame(frame, direction)


CUE_TYPES = ["mhmm", "hmm", "okay", "i_see", "right", "go_on"]


async def broadcast_cue():
    """Fetch a random cue from the voice server and broadcast as base64 audio."""
    import base64, random
    cue_type = random.choice(CUE_TYPES)
    try:
        session = await get_http_session()
        import aiohttp
        async with session.get(
            f"{VOICE_SERVER}/cue/{cue_type}",
            timeout=aiohttp.ClientTimeout(total=3),
        ) as resp:
            if resp.status == 200:
                wav = await resp.read()
                await broadcast({
                    "type": "cue",
                    "cue_type": cue_type,
                    "audio": base64.b64encode(wav).decode(),
                })
                log.info("Cue: %s (%d bytes)", cue_type, len(wav))
    except Exception as e:
        log.debug("Cue fetch failed: %s", e)


async def broadcast(msg):
    global ws_clients
    if not ws_clients:
        return
    data = json.dumps(msg)
    dead = set()
    for ws in ws_clients:
        try:
            await ws.send(data)
        except:
            dead.add(ws)
    ws_clients -= dead


async def ws_handler(ws):
    ws_clients.add(ws)
    log.info("Client connected (%d)", len(ws_clients))
    try:
        async for _ in ws:
            pass
    finally:
        ws_clients.discard(ws)
        log.info("Client disconnected (%d)", len(ws_clients))


async def run_mic():
    """Normal mode: capture from system mic. Retries on audio device errors."""
    for attempt in range(5):
        try:
            transport = LocalAudioTransport(
                LocalAudioTransportParams(
                    audio_in_enabled=True,
                    audio_out_enabled=False,
                    vad_enabled=True,
                    vad_analyzer=SileroVADAnalyzer(params=VADParams(min_volume=0.01, stop_secs=0.8)),
                )
            )
            pipeline = Pipeline([transport.input(), STTProcessor(), Broadcaster()])
            runner = PipelineRunner()
            task = PipelineTask(pipeline, idle_timeout_secs=None)
            log.info("Listening on system mic...")
            await runner.run(task)
            break
        except Exception as e:
            log.warning("Mic capture failed (attempt %d): %s", attempt + 1, e)
            if attempt < 4:
                await asyncio.sleep(3)
            else:
                log.error("Mic capture failed after 5 attempts, giving up")


async def run_test_audio(audio_path, start=60, duration=120):
    """Test mode: feed audio from file through the pipeline."""
    import io
    import wave as wavmod

    log.info("Test mode: %s from %ds for %ds", audio_path, start, duration)

    with wavmod.open(audio_path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        w.setpos(int(start * sr))
        frames = w.readframes(int(duration * sr))

    # Convert to mono 16-bit
    audio = np.frombuffer(frames, dtype=np.int16)
    if ch == 2:
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        from scipy.signal import resample
        target_len = int(len(audio) * SAMPLE_RATE / sr)
        audio = resample(audio.astype(np.float32), target_len).astype(np.int16)
        log.info("Resampled %dHz → %dHz (%d samples)", sr, SAMPLE_RATE, len(audio))
        sr = SAMPLE_RATE

    stt = STTProcessor()
    broadcaster = Broadcaster()

    chunk_samples = sr * 8
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) < sr:
            break

        # Check energy for VAD
        rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2)) / 32768
        if rms < 0.005:
            continue

        # Transcribe directly
        audio_bytes = chunk.astype(np.int16).tobytes()
        record_audio(audio_bytes)

        text = await asyncio.get_event_loop().run_in_executor(
            executor, stt._transcribe, audio_bytes
        )
        text = filter_noise(text)
        if not text:
            continue

        stt._chunk_num += 1
        # Save chunk recording
        if session_dir:
            p = session_dir / "recordings" / f"chunk-{stt._chunk_num:04d}.wav"
            buf = io.BytesIO()
            with wavmod.open(buf, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes(audio_bytes)
            p.write_bytes(buf.getvalue())

        trigger = detect_triggers(text)
        # Offline replay: silence isn't measured between fixed-size chunks, so
        # pass the chunk's nominal silence through the same decider seam rather
        # than a hardcoded 0.5. Triggers/LLM assessment still drive responses.
        replay_silence = 1.0
        confidence = broadcaster._turn_decider.confidence(
            silence_duration_secs=replay_silence, transcript_chunk=text
        )
        decision = broadcaster._engine.decide(replay_silence, confidence, text)

        msg = {
            "type": "transcript",
            "text": text,
            "trigger": {"type": trigger.trigger_type.value, "hint": trigger.hint.value} if trigger.triggered else None,
            "turn": decision.action.value,
        }
        await broadcast(msg)
        await post_notes(text)
        log.info("[%d] %s %s", stt._chunk_num, "TRIGGER" if trigger.triggered else "ok", text[:60])

    log.info("Test complete: %d chunks", stt._chunk_num)
    if _http_session:
        await _http_session.close()


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-audio", help="WAV file to use instead of mic")
    parser.add_argument("--start", type=int, default=60)
    parser.add_argument("--duration", type=int, default=120)
    args = parser.parse_args()

    init_session()

    ws_server = await websockets.serve(ws_handler, "127.0.0.1", PORT)
    log.info("WebSocket on ws://127.0.0.1:%d", PORT)

    if args.test_audio:
        # Wait for a WebSocket client to connect before sending test audio
        log.info("Waiting for client to connect...")
        while not ws_clients:
            await asyncio.sleep(0.5)
        log.info("Client connected — starting test audio")
        await asyncio.sleep(2)  # let client settle
        await run_test_audio(args.test_audio, args.start, args.duration)
    else:
        while True:
            await run_mic()
            log.warning("Mic pipeline exited, restarting in 3s...")
            await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(main())
