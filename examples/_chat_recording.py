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
    """Root-mean-square level for a frame of audio samples.

    Returns 0.0 for empty input — without the guard, ``np.mean`` of
    an empty slice raises RuntimeWarning and returns NaN, which then
    poisons every downstream comparison (``NaN > threshold`` is
    always False, so the loop silently behaves as IDLE forever).
    Empty buffers do happen in practice: a torn read from PyAudio,
    a virtual mic flushed mid-iteration, etc.
    """
    if frame is None or len(frame) == 0:
        return 0.0
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
    primed_frames: list[bytes] | None = None,
    silence_threshold: float = SILENCE_THRESHOLD,
    silence_duration: float = SILENCE_DURATION,
    min_speech_duration: float = MIN_SPEECH_DURATION,
    out_metrics: dict | None = None,
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
      ``primed_frames`` — optional list of pre-captured byte chunks
        that get fed through the VAD before any live mic reads. The
        iter-009 ``BargeInWatcher`` produces these; passing them here
        keeps the user's first syllables in the recorded wav instead
        of dropping them. Each chunk must be ``CHUNK * 2`` bytes
        (int16 mono at ``RATE``). During the priming phase, the VAD
        is fed virtual timestamps that advance at audio rate so its
        time-based logic (silence-window) matches what would happen
        with live mic reads.
      ``silence_threshold`` / ``silence_duration`` / ``min_speech_duration`` —
        VAD tuning knobs (iter-020). Default to the module-level
        constants. Override per-call to handle noisy environments
        (raise threshold), faster turn-taking (shorten silence_duration),
        or stricter speech detection (raise min_speech_duration).
      ``out_metrics`` — optional dict that, if provided, is populated
        with extra measurements that don't fit the return tuple
        (iter-063). Keys:
          - ``"eot_latency"`` (seconds from last in-speech frame to
            ``DONE_OK`` firing) — populated on the DONE_OK path only.
          - ``"stt_preview_divergence"`` (iter-072) in [0.0, 1.0]:
            ``1 - SequenceMatcher.ratio(preview, final)``. 0 = live
            preview matched the final transcript perfectly; 1 =
            completely different. Populated when both preview and
            final are non-empty; ``DONE_TOO_SHORT`` returns early
            without writing either key.
    """
    if transcribe_fn is None:
        transcribe_fn = lambda wav: _transcribe_quick(stt_engine, wav)
    if output is None:
        output = sys.stdout

    frames: list[bytes] = []
    last_inference_at = 0.0
    preview_text = ""
    vad = VadState(
        silence_threshold=silence_threshold,
        silence_duration=silence_duration,
        min_speech_duration=min_speech_duration,
    )
    too_short = False

    primed = list(primed_frames or [])
    primed_idx = 0
    # Use one frame-aligned virtual clock for ALL frames (primed and
    # live). The naive approach — clock() per frame — has two
    # failure modes:
    #   1. With a real ``time.monotonic`` and primed frames served
    #      in microseconds, the silence window appears to close
    #      almost instantly because per-frame deltas are tiny.
    #   2. With a per-call test FrameClock, the first live read's
    #      timestamp can land BEFORE the last primed timestamp
    #      (clock advances per call, but virtual primed time
    #      advances per frame), corrupting ``last_speech_duration``.
    # Capturing ``t_origin`` once and computing
    # ``now = t_origin + frame_idx * dt`` keeps time monotonic and
    # at audio rate regardless of how the underlying clock behaves.
    # In production this matches real wall time because PyAudio
    # blocks at audio rate; in tests it matches whatever virtual
    # cadence the test wants.
    t_origin = clock()
    frame_dt = CHUNK / RATE
    frame_idx = 0
    # iter-063: track the timestamp of the last frame whose RMS level
    # crossed ``silence_threshold`` — i.e. the last frame the VAD
    # actually heard speech. The gap between this and DONE_OK firing
    # is the user-perceived EoT latency. Lower bound is roughly
    # ``silence_duration`` (the VAD has to wait that long); the gap
    # above that is implementation overhead (chunk granularity,
    # processing). Stays None when the loop exits via DONE_TOO_SHORT
    # or never sees speech.
    last_speech_at: float | None = None

    while True:
        if primed_idx < len(primed):
            data = primed[primed_idx]
            primed_idx += 1
        else:
            data = stream.read(CHUNK, exception_on_overflow=False)
        now = t_origin + frame_idx * frame_dt
        frame_idx += 1
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        level = rms(audio)

        # iter-063: record this frame as "in speech" before the VAD
        # event branch so DONE_OK can read the latest value. Any
        # frame above the threshold counts, regardless of which
        # VadEvent it produces.
        if level > silence_threshold:
            last_speech_at = now

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
            # iter-063: populate eot_latency on the success path.
            # DONE_OK is fired by VadState.feed when silence has
            # persisted >= silence_duration AND speech_duration was
            # large enough; ``last_speech_at`` is guaranteed non-None
            # here because we only reach DONE_OK after at least one
            # speaking frame (the speech_start latch in VadState).
            if out_metrics is not None and last_speech_at is not None:
                out_metrics["eot_latency"] = now - last_speech_at
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

    # iter-072: STT preview-vs-final divergence. Populate the
    # side-band dict with ``1 - SequenceMatcher.ratio()``, where 0
    # = preview matched final perfectly (live STT was useful), 1 =
    # totally different (live STT was misleading and the user had
    # to wait for the final). Only emits when both preview and
    # final are non-empty — turns where the user spoke but no
    # preview managed to fire (very short utterance) leave the
    # field at default. Metric 1.8 in the perf-metrics taxonomy.
    if (
        out_metrics is not None
        and preview_text
        and final_text
    ):
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, preview_text, final_text).ratio()
        out_metrics["stt_preview_divergence"] = max(0.0, 1.0 - ratio)

    stt_engine._last_text = final_text
    return wav_bytes, speech_duration, stt_time
