"""Recording loop extracted from mic_chat.py — pyaudio-free at module scope.

Originally everything lived inside examples/mic_chat.py, which imports
pyaudio at the top. That kept record_utterance_streaming() unimportable
on machines without ALSA dev headers (most of CI, this Linux box, etc.),
so the function couldn't be tested even though it's mostly logic.

This module hosts the parts that don't need pyaudio:
  - audio + VAD constants (RATE, CHUNK, SILENCE_*, MIN_SPEECH_DURATION)
  - rms(), _buffer_to_wav(), _transcribe_quick()
  - record_utterance_streaming()

The record function now accepts optional `transcribe_fn`, `clock`, and
`output` kwargs so a test can:
  - inject a stub transcriber (no mlx-whisper required)
  - drive a deterministic monotonic clock
  - capture the live-preview output without touching real stdout

mic_chat.py re-exports the constants and the function so existing
imports keep working.
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import time
import wave
from typing import Callable

import numpy as np

from examples._chat_helpers import VadEvent, VadState, render_preview

# --- audio + VAD parameters ---------------------------------------------------
RATE = 16000
CHANNELS = 1
CHUNK = 1024
SILENCE_THRESHOLD = 0.02
SILENCE_DURATION = 0.8
MIN_SPEECH_DURATION = 0.3
INFERENCE_INTERVAL = 1.0

# --- ANSI helpers for the inline "You:" final-line print ---------------------
_BOLD = "\033[1m"
_RESET = "\033[0m"
CLEAR_LINE = "\033[2K"


def rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame ** 2)))


def _buffer_to_wav(
    frames: list[bytes],
    *,
    channels: int = CHANNELS,
    rate: int = RATE,
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


def _transcribe_quick(engine, wav_bytes: bytes) -> str | None:
    """Run mlx-whisper transcription on a wav blob. Returns text or None.

    On hosts without mlx_whisper (e.g. Linux x86_64), the import fails
    inside the try/except and we return None — callers should handle that.
    """
    if not wav_bytes:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp = f.name
    try:
        import mlx_whisper  # noqa: F401  (only available on Apple Silicon)
        result = mlx_whisper.transcribe(tmp, path_or_hf_repo=engine.model_repo)
        return result["text"].strip()
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def record_utterance_streaming(
    stream,
    stt_engine,
    *,
    transcribe_fn: Callable[[bytes], str | None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    output=None,
) -> tuple[bytes, float, float]:
    """Record one utterance with live STT preview.

    Returns ``(wav_bytes, speech_duration, stt_time)``. The final text is
    also stashed on ``stt_engine._last_text`` for backward compat with the
    existing chat loop.

    ``stream`` only needs the pyaudio shape: ``.read(n_frames,
    exception_on_overflow=False) -> bytes``. Both the real
    ``pyaudio.Stream`` and ``examples.virtual_audio.VirtualMicStream``
    satisfy this.

    Parameters that exist purely to make this testable:
      ``transcribe_fn`` — callable taking the in-progress wav blob and
        returning either a transcript or None. Defaults to wrapping
        ``_transcribe_quick(stt_engine, wav_bytes)``. A test stub can
        return canned text without needing mlx-whisper.
      ``clock`` — monotonic-clock function. Defaults to
        ``time.monotonic``. Tests can pass a generator-backed clock so
        the loop doesn't depend on real wall time.
      ``output`` — file-like object for the preview line. Defaults to
        ``sys.stdout``. Tests can pass an ``io.StringIO``.
    """
    if transcribe_fn is None:
        transcribe_fn = lambda wav: _transcribe_quick(stt_engine, wav)
    if output is None:
        output = sys.stdout

    frames: list[bytes] = []
    last_inference_at = 0.0
    preview_text = ""
    vad = VadState(
        silence_threshold=SILENCE_THRESHOLD,
        silence_duration=SILENCE_DURATION,
        min_speech_duration=MIN_SPEECH_DURATION,
    )
    too_short = False

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        level = rms(audio)
        now = clock()

        event = vad.feed(level, now)

        if event is VadEvent.IDLE:
            continue

        # ACTIVE / DONE_OK / DONE_TOO_SHORT all include this frame in the
        # buffer (parity with the pre-VAD-extraction code that appended
        # before the break check).
        frames.append(data)
        if last_inference_at == 0.0:
            last_inference_at = now

        if event is VadEvent.DONE_OK:
            break
        if event is VadEvent.DONE_TOO_SHORT:
            too_short = True
            break

        # Periodic STT preview while speaking.
        if frames and (now - last_inference_at) >= INFERENCE_INTERVAL:
            last_inference_at = now
            wav = _buffer_to_wav(frames)
            text = transcribe_fn(wav)
            if text and text != preview_text:
                preview_text = text
                term_cols = shutil.get_terminal_size(fallback=(80, 24)).columns
                render_preview(
                    preview_text,
                    max_width=term_cols,
                    prefix="  You: ",
                    file=output,
                )

    if too_short:
        output.write(f"\r{CLEAR_LINE}")
        output.flush()
        stt_engine._last_text = None
        return b"", 0.0, 0.0

    speech_duration = vad.last_speech_duration

    wav_bytes = _buffer_to_wav(frames)
    t = clock()
    final_text = transcribe_fn(wav_bytes)
    stt_time = clock() - t

    if final_text:
        output.write(f"\r{CLEAR_LINE}  {_BOLD}You:{_RESET} \"{final_text}\"\n")
        output.flush()

    stt_engine._last_text = final_text
    return wav_bytes, speech_duration, stt_time
