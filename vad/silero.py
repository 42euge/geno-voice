"""iter-231 — Silero neural VAD segmenter (the real fix for continuous speech).

GROUND TRUTH (fixtures/recordings/voice-20260618-110355.wav, 31s continuous):
energy-RMS VAD CANNOT segment this audio. Its in-speech noise floor (~0.016
RMS) sits too close to the speech median (~0.023): NO threshold×silence
combination breaks the utterance into more than one segment, so it never
closes ("VAD triggered but wouldn't untrigger"). The RMS state machine in
``vad/silence.py`` / ``client/voice-capture.js`` (replayed by
``fixtures/replay_vad.py``) is a dead end for this audio and stays only as the
fallback path.

Silero VAD is a small neural model that distinguishes *speech* from room-tone
regardless of the energy floor, so it segments the 31s recording into 5 sensible
speech regions where energy-VAD gave 1. The engine already depends on it
indirectly: ``pipecat_server.py`` runs ``SileroVADAnalyzer(params=VADParams(
min_volume=0.01, stop_secs=0.8))`` on the live mic path. This module exposes the
same model to the :5111 server and to a headless replay harness so the
recording corpus — not a live mic — is the proof it works.

Design (mirrors the mic_chat.py extraction pattern in GENO.md):
  * The model is loaded once and cached (``load_model``); callers inject it or
    let the segmenter lazily load the singleton.
  * ``silero-vad`` (and its ``torch`` / ``torchaudio`` deps) are imported lazily
    inside the functions, NOT at module scope, so this file is importable on a
    host without the package — tests and the server degrade to a clean skip /
    503 rather than an ImportError at import time. ``silero_available()`` reports
    whether the real model can be loaded.
  * ``SileroParams`` mirrors the pipecat ``VADParams`` knobs (min speech / min
    silence durations) so a tuning experiment here ports straight to the live
    path. ``min_silence_ms`` defaults to 800 — the pipecat ``stop_secs=0.8``.
  * Results are dataclasses (``SpeechSegment`` / ``SileroResult``) so future
    fields extend without breaking call sites, same as ``replay_vad``.

The endpoint contract (server ``/vad/silero``) and the per-recording segment
counts are documented in ``docs/research/voice-capture-tuning.md``.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Silero runs at 16 kHz (it also supports 8 kHz, but 16 kHz is the canonical
# window the pretrained model expects and what pipecat feeds it). Recordings in
# the corpus are 44.1/48 kHz, so we resample to this before inference.
SILERO_SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Parameters — mirror the pipecat VADParams knobs so a tuning experiment here
# ports straight to pipecat_server.py's live SileroVADAnalyzer.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SileroParams:
    """One parameter set for the Silero segmenter.

    ``threshold``       — speech-probability gate (Silero emits a per-window
                          P(speech) in [0,1]; default 0.5 is the model's own
                          recommended operating point). This is NOT an energy
                          threshold — it is the neural model's confidence, which
                          is exactly why it works where the RMS gate fails.
    ``min_speech_ms``   — drop speech regions shorter than this (Silero's
                          ``min_speech_duration_ms``); the analogue of the RMS
                          state machine's ``min_speech_ms`` noise gate.
    ``min_silence_ms``  — how long P(speech) must stay low before a region ends
                          (Silero's ``min_silence_duration_ms``). Defaults to
                          800 — the pipecat ``stop_secs=0.8`` the live mic path
                          already uses, so headless replay matches production.
    ``speech_pad_ms``   — symmetric padding added to each detected region
                          (Silero's ``speech_pad_ms``), recovering the soft
                          attack/decay the model trims — the neural analogue of
                          ``replay_vad``'s ``preroll_ms`` opening-recovery.
    ``max_speech_s``    — split any region longer than this (Silero's
                          ``max_speech_duration_s``); ``inf`` (default) never
                          force-splits.
    """

    threshold: float = 0.5
    min_speech_ms: float = 250.0
    min_silence_ms: float = 800.0
    speech_pad_ms: float = 30.0
    max_speech_s: float = float("inf")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class SpeechSegment:
    """A Silero-detected speech region (seconds, relative to recording start)."""

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict:
        return {
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "duration_s": round(self.duration_s, 3),
        }


@dataclass
class SileroResult:
    """Outcome of segmenting one audio buffer through Silero VAD."""

    name: str
    sample_rate: int
    duration_s: float
    segments: List[SpeechSegment] = field(default_factory=list)

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    @property
    def speech_s(self) -> float:
        return sum(s.duration_s for s in self.segments)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sample_rate": self.sample_rate,
            "duration_s": round(self.duration_s, 3),
            "num_segments": self.num_segments,
            "speech_s": round(self.speech_s, 3),
            "segments": [s.to_dict() for s in self.segments],
        }

    def summary_line(self) -> str:
        preview = ", ".join(
            f"({s.start_s:.1f}-{s.end_s:.1f})" for s in self.segments[:6]
        )
        more = " …" if len(self.segments) > 6 else ""
        return (
            f"{self.name:<32} sr={self.sample_rate:<5} dur={self.duration_s:6.1f}s "
            f"segs={self.num_segments:<2} speech={self.speech_s:6.1f}s  {preview}{more}"
        )


# ---------------------------------------------------------------------------
# Model loading — lazy + cached so the import is cheap and dependency-free.
# ---------------------------------------------------------------------------


_MODEL = None  # cached singleton (RecursiveScriptModule)


def silero_available() -> bool:
    """Whether the ``silero-vad`` package (and its torch deps) can be imported.

    Used by the server to return a clean 503 and by tests to skip, rather than
    crashing at import time on a host without the package installed.
    """
    try:
        import silero_vad  # noqa: F401
    except Exception:
        return False
    return True


def load_model(force_reload: bool = False):
    """Load (once) and return the Silero VAD model.

    Mirrors how pipecat's ``SileroVADAnalyzer`` and ``silero_vad.load_silero_vad``
    obtain the model: the pretrained weights ship inside the ``silero-vad``
    package, so this works fully offline (no torch.hub download). Cached as a
    module singleton so the :5111 server pays the load cost exactly once.
    """
    global _MODEL
    if _MODEL is not None and not force_reload:
        return _MODEL
    from silero_vad import load_silero_vad

    _MODEL = load_silero_vad()
    return _MODEL


# ---------------------------------------------------------------------------
# Core: segment float32 mono samples / WAV bytes / a recording file
# ---------------------------------------------------------------------------


def _resample_to_16k(samples, sample_rate: int):
    """Resample a float32 mono tensor/array to 16 kHz, returning a torch tensor.

    Silero is trained at 16 kHz; the corpus is 44.1/48 kHz. ``torchaudio`` ships
    with ``silero-vad`` so it is always present when this code path runs.
    """
    import torch

    if not hasattr(samples, "dim"):  # numpy / list → tensor
        samples = torch.as_tensor(samples, dtype=torch.float32)
    samples = samples.to(torch.float32)
    if sample_rate == SILERO_SAMPLE_RATE:
        return samples
    import torchaudio.functional as AF

    return AF.resample(samples, sample_rate, SILERO_SAMPLE_RATE)


def segment_samples(
    samples,
    sample_rate: int,
    params: Optional[SileroParams] = None,
    model=None,
) -> List[SpeechSegment]:
    """Run Silero over float32 mono ``samples`` in [-1, 1] and return segments.

    Timestamps are returned in seconds **relative to the original recording**
    (Silero works on the resampled signal but seconds are sample-rate
    independent, so no rescaling of the timestamps is needed).
    """
    params = params or SileroParams()
    if model is None:
        model = load_model()
    from silero_vad import get_speech_timestamps

    audio = _resample_to_16k(samples, sample_rate)
    if audio.numel() == 0:
        return []

    max_speech_s = (
        params.max_speech_s if params.max_speech_s != float("inf") else float("inf")
    )
    ts = get_speech_timestamps(
        audio,
        model,
        sampling_rate=SILERO_SAMPLE_RATE,
        threshold=params.threshold,
        min_speech_duration_ms=int(params.min_speech_ms),
        min_silence_duration_ms=int(params.min_silence_ms),
        speech_pad_ms=int(params.speech_pad_ms),
        max_speech_duration_s=max_speech_s,
        return_seconds=True,
    )
    return [SpeechSegment(start_s=float(s["start"]), end_s=float(s["end"])) for s in ts]


def _read_wav_mono(wav_bytes: bytes) -> tuple:
    """Decode 16-bit PCM WAV bytes to (float32 mono numpy array, sample_rate)."""
    import numpy as np

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        channels = wf.getnchannels()
        raw = wf.readframes(n_frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def segment_wav_bytes(
    wav_bytes: bytes,
    params: Optional[SileroParams] = None,
    model=None,
    name: str = "<wav>",
) -> SileroResult:
    """Segment a 16-bit PCM WAV byte string and return a ``SileroResult``."""
    samples, sample_rate = _read_wav_mono(wav_bytes)
    duration_s = (len(samples) / sample_rate) if sample_rate else 0.0
    segments = segment_samples(samples, sample_rate, params, model=model)
    return SileroResult(
        name=name,
        sample_rate=sample_rate,
        duration_s=duration_s,
        segments=segments,
    )


def segment_recording(
    wav_path: Path,
    params: Optional[SileroParams] = None,
    model=None,
) -> SileroResult:
    """Segment one recording WAV file and return a ``SileroResult``."""
    wav_path = Path(wav_path)
    result = segment_wav_bytes(
        wav_path.read_bytes(), params=params, model=model, name=wav_path.name
    )
    return result


# ---------------------------------------------------------------------------
# Streaming (iter-232) — frame-by-frame Silero for live capture.
#
# ``segment_*`` above is BATCH: it needs the whole utterance buffered before it
# can emit a single segment. For live capture the desktop ContinuousListener
# wants *incremental* speech-start / speech-end decisions as audio arrives, so a
# turn can be cut the instant Silero sees ``min_silence_ms`` of trailing silence
# — without waiting for the user to stop talking AND a whole-WAV round-trip.
#
# ``silero-vad`` ships ``VADIterator`` for exactly this "stream imitation". It is
# a stateful per-window classifier that returns ``{"start": t}`` when speech
# opens and ``{"end": t}`` when it closes (after the silence hangover). It only
# accepts the model's native window (512 samples @ 16 kHz, 256 @ 8 kHz) per call
# and supports a REDUCED knob set vs the batch path: ``threshold``,
# ``min_silence_ms`` and ``speech_pad_ms`` apply; ``min_speech_ms`` /
# ``max_speech_s`` do NOT (the iterator has no look-back to drop a short region
# after the fact). ``SileroStream`` wraps it with sample buffering (callers push
# arbitrary-length chunks) and an end-of-stream ``flush()`` that closes a still-
# open segment — a property the batch path gets for free but a live stream must
# handle explicitly (a stream can end mid-speech).
# ---------------------------------------------------------------------------

# Silero's required per-call window, in samples, keyed by the two supported
# sample rates. ``VADIterator`` raises on any other window length.
_WINDOW_SAMPLES = {16000: 512, 8000: 256}
WINDOW_SAMPLES = _WINDOW_SAMPLES[SILERO_SAMPLE_RATE]  # 512 — the 16 kHz default


@dataclass
class StreamEvent:
    """One incremental VAD decision emitted by ``SileroStream``.

    ``kind``    — ``"start"`` (speech began) or ``"end"`` (speech ended).
    ``time_s``  — seconds from the start of the stream, matching the batch
                  segmenter's timestamps (Silero pads/​hangovers are already
                  applied by the iterator, exactly as in the batch path).
    """

    kind: str
    time_s: float

    def to_dict(self) -> dict:
        return {"type": self.kind, "time_s": round(self.time_s, 3)}


class SileroStream:
    """Stateful frame-by-frame Silero VAD for live capture.

    Push arbitrary-length float32 mono chunks in [-1, 1] via :meth:`push`; get
    back the ``StreamEvent``\\ s that crossed a window boundary in that chunk.
    Call :meth:`flush` at end-of-stream to close a segment still open because the
    audio ended mid-speech. :meth:`reset` re-arms the same stream for a new
    utterance without reloading the model.

    Sample rate is fixed at construction to one Silero supports (16 kHz default,
    8 kHz allowed). The stream does NOT resample — feed it audio already at that
    rate (the live pipecat path delivers 16 kHz; the corpus replay resamples the
    whole signal once up front via :func:`stream_samples`). Sub-window remainders
    are buffered across pushes, so chunk sizes need not be window-aligned.
    """

    def __init__(
        self,
        params: Optional[SileroParams] = None,
        model=None,
        sample_rate: int = SILERO_SAMPLE_RATE,
    ):
        if sample_rate not in _WINDOW_SAMPLES:
            raise ValueError(
                f"SileroStream supports sample rates {sorted(_WINDOW_SAMPLES)}, "
                f"got {sample_rate}"
            )
        self.params = params or SileroParams()
        self.sample_rate = sample_rate
        self.window = _WINDOW_SAMPLES[sample_rate]
        if model is None:
            model = load_model()
        from silero_vad import VADIterator

        self._iter = VADIterator(
            model,
            threshold=self.params.threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=int(self.params.min_silence_ms),
            speech_pad_ms=int(self.params.speech_pad_ms),
        )
        import torch

        self._buf = torch.zeros(0, dtype=torch.float32)
        self._triggered = False
        self._total_samples = 0  # everything pushed (for the flush() end time)

    @property
    def triggered(self) -> bool:
        """Whether the stream is currently inside an open speech region."""
        return self._triggered

    def push(self, samples) -> List[StreamEvent]:
        """Feed a chunk of float32 mono samples; return events it produced.

        Remainders shorter than one window are buffered for the next push, so
        the caller can pass any chunk length (a mic callback's frame size, a
        WebSocket message, a whole file).
        """
        import torch

        x = samples if hasattr(samples, "dim") else torch.as_tensor(
            samples, dtype=torch.float32
        )
        x = x.to(torch.float32).reshape(-1)
        self._total_samples += int(x.numel())
        if x.numel():
            self._buf = torch.cat([self._buf, x]) if self._buf.numel() else x

        events: List[StreamEvent] = []
        while self._buf.numel() >= self.window:
            window = self._buf[: self.window]
            self._buf = self._buf[self.window :]
            out = self._iter(window, return_seconds=True)
            if not out:
                continue
            if "start" in out:
                self._triggered = True
                events.append(StreamEvent("start", float(out["start"])))
            if "end" in out:
                self._triggered = False
                events.append(StreamEvent("end", float(out["end"])))
        return events

    def flush(self) -> List[StreamEvent]:
        """Close a segment left open because the stream ended mid-speech.

        Returns a single synthetic ``"end"`` at the total elapsed time when a
        region was open, else an empty list. Idempotent: a second call (or a
        call on a stream that ended in silence) returns ``[]``.
        """
        if not self._triggered:
            return []
        end_s = self._total_samples / self.sample_rate
        self._triggered = False
        return [StreamEvent("end", round(end_s, 3))]

    def reset(self) -> None:
        """Re-arm for a new utterance: clear iterator state and buffers."""
        import torch

        self._iter.reset_states()
        self._buf = torch.zeros(0, dtype=torch.float32)
        self._triggered = False
        self._total_samples = 0


def _events_to_segments(events: List[StreamEvent]) -> List[SpeechSegment]:
    """Pair ``start``/``end`` events into ``SpeechSegment``\\ s.

    Tolerant of a dangling ``start`` with no matching ``end`` (drops it — a
    correct stream flushes its open segment, so a leftover ``start`` means the
    caller forgot to ``flush()``).
    """
    segments: List[SpeechSegment] = []
    start: Optional[float] = None
    for ev in events:
        if ev.kind == "start":
            start = ev.time_s
        elif ev.kind == "end" and start is not None:
            segments.append(SpeechSegment(start_s=start, end_s=ev.time_s))
            start = None
    return segments


def stream_samples(
    samples,
    sample_rate: int,
    params: Optional[SileroParams] = None,
    model=None,
    chunk_samples: Optional[int] = None,
) -> SileroResult:
    """Drive a :class:`SileroStream` over ``samples`` and rebuild a ``SileroResult``.

    The streaming analogue of :func:`segment_samples`: it resamples the whole
    signal to 16 kHz once (so timestamps match the batch path exactly), then
    feeds it through ``SileroStream`` in ``chunk_samples`` slices, flushing at
    the end. Returns the same ``SileroResult`` shape as the batch path so callers
    and tests can assert streaming reconstructs batch segmentation.
    """
    params = params or SileroParams()
    audio = _resample_to_16k(samples, sample_rate)
    n = int(audio.numel())
    duration_s = (n / SILERO_SAMPLE_RATE) if n else 0.0
    if n == 0:
        return SileroResult(
            name="<stream>", sample_rate=SILERO_SAMPLE_RATE, duration_s=0.0
        )

    stream = SileroStream(params=params, model=model, sample_rate=SILERO_SAMPLE_RATE)
    step = chunk_samples or (WINDOW_SAMPLES * 4)
    events: List[StreamEvent] = []
    for i in range(0, n, step):
        events.extend(stream.push(audio[i : i + step]))
    events.extend(stream.flush())

    return SileroResult(
        name="<stream>",
        sample_rate=SILERO_SAMPLE_RATE,
        duration_s=duration_s,
        segments=_events_to_segments(events),
    )


# ---------------------------------------------------------------------------
# WebSocket protocol glue (iter-232) — the message-level state machine the
# server's ``/vad/silero/stream`` endpoint drives, extracted so it is testable
# WITHOUT importing ``server`` (which pulls in pipecat and won't import on the
# x86_64 Linux loop host). Mirrors the mic_chat.py extraction pattern: inject a
# stream factory, return plain dicts, lazy-import nothing platform-specific.
# ---------------------------------------------------------------------------

# float32 PCM is 4 bytes/sample, little-endian — the wire format the client
# (a Web Audio ScriptProcessor / AudioWorklet) emits and the protocol decodes.
_BYTES_PER_SAMPLE = 4


def decode_float32_le(data: bytes):
    """Decode a little-endian float32 byte frame to a list of floats.

    Trailing bytes that don't complete a sample are ignored (a partial frame on
    the wire). Used by the WebSocket path to turn a binary message into samples.
    """
    import struct

    n = len(data) // _BYTES_PER_SAMPLE
    if n == 0:
        return []
    return list(struct.unpack(f"<{n}f", data[: n * _BYTES_PER_SAMPLE]))


class StreamProtocol:
    """Message-level state machine for the ``/vad/silero/stream`` WebSocket.

    Owns the contract: a JSON config message arms / re-configures the underlying
    ``SileroStream``; ``{"cmd":"flush"}`` closes an open segment at EOF;
    ``{"cmd":"reset"}`` re-arms for a new utterance; a binary frame is decoded as
    little-endian float32 PCM and pushed. Each handler returns the JSON-able dict
    the server sends back, so the server endpoint is thin transport glue and ALL
    branching logic is unit-tested here without a live socket or pipecat.

    ``stream_factory(cfg: dict) -> SileroStream`` builds the stream from a config
    message (so the server can inject the loaded model); a binary frame arriving
    before any config arms a default-params stream via ``stream_factory({})``.
    """

    def __init__(self, stream_factory):
        self._factory = stream_factory
        self._stream = None

    @property
    def armed(self) -> bool:
        return self._stream is not None

    def handle_text(self, msg: dict) -> dict:
        """Handle a decoded JSON control/config message; return the reply dict."""
        cmd = msg.get("cmd")
        if cmd == "flush":
            events = self._stream.flush() if self._stream is not None else []
            return {"events": [e.to_dict() for e in events], "flushed": True}
        if cmd == "reset":
            if self._stream is not None:
                self._stream.reset()
            return {"events": [], "reset": True}
        # Anything else is a params config message: (re)arm the stream.
        self._stream = self._factory(msg)
        return {"events": [], "armed": True}

    def handle_binary(self, data: bytes) -> dict:
        """Decode a float32 PCM frame, push it, and return the events dict."""
        if self._stream is None:
            self._stream = self._factory({})
        samples = decode_float32_le(data)
        events = self._stream.push(samples)
        return {"events": [e.to_dict() for e in events]}
