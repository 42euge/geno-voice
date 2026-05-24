"""Virtual audio interfaces — software-only mic/speaker streams that
mimic pyaudio's API. Used to drive the chat pipeline without hardware.

Why bother:
  - mlx-whisper / pyaudio aren't reliably available off-Apple-Silicon, so
    end-to-end exercise of mic_chat.record_utterance_streaming and the
    VAD/playback path needs a software stand-in.
  - Tests can push known audio into the virtual mic and assert what the
    pipeline does with it.
  - The simulation can use the project's own TTS to synthesize speech
    and feed it back through the loop — closing the test loop with the
    same components that ship in production.

Contract: each stream below exposes the *subset* of pyaudio.Stream
methods that mic_chat.py and examples/_chat_helpers.flush_pending_audio
actually call. Specifically:

    .read(n_frames, exception_on_overflow=False) -> bytes
    .get_read_available() -> int
    .write(bytes) -> None
    .stop_stream() -> None
    .close() -> None

If you find the production code calling something else, add it here
rather than monkey-patching tests. The point is contract parity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

DEFAULT_RATE = 16000
DEFAULT_CHANNELS = 1
SAMPLE_WIDTH = 2  # int16


def _ensure_int16_bytes(data) -> bytes:
    """Accept ndarray (int16/float32/float64) or bytes; return int16 bytes."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    arr = np.asarray(data)
    if arr.dtype == np.int16:
        return arr.tobytes()
    if arr.dtype.kind == "f":
        # Assume floats are in [-1, 1].
        clipped = np.clip(arr, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16).tobytes()
    raise TypeError(f"Unsupported audio dtype: {arr.dtype}")


@dataclass
class VirtualMicStream:
    """Software input stream — push audio in, pyaudio-style read out.

    Internal buffer is bytes; reads pop from the front. If `read` is called
    with more frames than are available, the missing bytes are zero-padded
    (matches pyaudio behavior on a quiet mic — never blocks tests).

    Track `closed`/`stopped` flags for parity. Reads against a closed stream
    raise OSError, matching pyaudio.
    """

    rate: int = DEFAULT_RATE
    channels: int = DEFAULT_CHANNELS
    chunk_size: int = 1024
    pad_with_silence: bool = True

    _buffer: bytearray = field(default_factory=bytearray)
    _closed: bool = False
    _stopped: bool = False
    reads: list[int] = field(default_factory=list)

    def push(self, audio) -> None:
        """Inject audio. Accepts bytes (int16 PCM) or numpy array."""
        if self._closed:
            raise OSError("VirtualMicStream is closed")
        self._buffer.extend(_ensure_int16_bytes(audio))

    def push_silence(self, seconds: float) -> None:
        n = int(seconds * self.rate) * self.channels
        self.push(np.zeros(n, dtype=np.int16))

    @property
    def frames_buffered(self) -> int:
        return len(self._buffer) // (SAMPLE_WIDTH * self.channels)

    # --- pyaudio-shaped surface ---

    def get_read_available(self) -> int:
        if self._closed:
            return 0
        return self.frames_buffered

    def read(self, n_frames: int, exception_on_overflow: bool = False) -> bytes:
        if self._closed:
            raise OSError("VirtualMicStream is closed")
        self.reads.append(n_frames)
        n_bytes = n_frames * SAMPLE_WIDTH * self.channels
        available = len(self._buffer)
        if available >= n_bytes:
            out = bytes(self._buffer[:n_bytes])
            del self._buffer[:n_bytes]
            return out
        # Drain what we have and zero-pad the rest.
        head = bytes(self._buffer)
        self._buffer.clear()
        if not self.pad_with_silence:
            return head
        return head + bytes(n_bytes - available)

    def stop_stream(self) -> None:
        self._stopped = True

    def close(self) -> None:
        self._closed = True
        self._buffer.clear()


@dataclass
class VirtualSpeakerStream:
    """Software output stream — write captures audio for inspection.

    `captured` is the concatenated bytes written so far. `write` returns
    None, matching pyaudio. If `loopback_to` is set, every write is also
    pushed into the linked mic — useful for testing barge-in scenarios.
    """

    rate: int = 24000
    channels: int = DEFAULT_CHANNELS
    captured: bytearray = field(default_factory=bytearray)
    loopback_to: VirtualMicStream | None = None
    _closed: bool = False
    _stopped: bool = False
    writes: list[int] = field(default_factory=list)

    def write(self, data) -> None:
        if self._closed:
            raise OSError("VirtualSpeakerStream is closed")
        b = _ensure_int16_bytes(data)
        self.captured.extend(b)
        self.writes.append(len(b))
        if self.loopback_to is not None and not self.loopback_to._closed:
            # Loopback: rate-mismatch is the caller's responsibility for now.
            self.loopback_to.push(b)

    def stop_stream(self) -> None:
        self._stopped = True

    def close(self) -> None:
        self._closed = True

    @property
    def captured_int16(self) -> np.ndarray:
        return np.frombuffer(bytes(self.captured), dtype=np.int16)

    @property
    def captured_float32(self) -> np.ndarray:
        return self.captured_int16.astype(np.float32) / 32768.0


@dataclass
class VirtualAudioInterface:
    """A drop-in replacement for `pyaudio.PyAudio()` in tests.

    `open(input=True, ...)` returns a VirtualMicStream, `output=True` gives
    a VirtualSpeakerStream. The pair can be linked via `loopback=True` so
    speaker writes appear as mic reads — that's the simulation hook for
    barge-in and self-loop tests.
    """

    input_rate: int = DEFAULT_RATE
    output_rate: int = 24000
    chunk_size: int = 1024
    loopback: bool = False

    mics: list[VirtualMicStream] = field(default_factory=list)
    speakers: list[VirtualSpeakerStream] = field(default_factory=list)
    _terminated: bool = False

    def open(
        self,
        format=None,
        channels: int = DEFAULT_CHANNELS,
        rate: int | None = None,
        input: bool = False,
        output: bool = False,
        frames_per_buffer: int | None = None,
    ):
        if self._terminated:
            raise OSError("VirtualAudioInterface terminated")
        chunk = frames_per_buffer or self.chunk_size
        if input:
            mic = VirtualMicStream(
                rate=rate or self.input_rate,
                channels=channels,
                chunk_size=chunk,
            )
            self.mics.append(mic)
            return mic
        if output:
            spk = VirtualSpeakerStream(
                rate=rate or self.output_rate,
                channels=channels,
            )
            if self.loopback and self.mics:
                spk.loopback_to = self.mics[-1]
            self.speakers.append(spk)
            return spk
        raise ValueError("must pass input=True or output=True")

    def terminate(self) -> None:
        self._terminated = True
        for m in self.mics:
            m.close()
        for s in self.speakers:
            s.close()


# ---- Audio fixtures ----------------------------------------------------------

def make_silence(seconds: float, rate: int = DEFAULT_RATE) -> np.ndarray:
    """Pure zeros. Useful as gap-padding around speech-like bursts."""
    return np.zeros(int(seconds * rate), dtype=np.int16)


def make_tone_burst(
    seconds: float,
    rate: int = DEFAULT_RATE,
    freq: float = 440.0,
    amp: float = 0.3,
) -> np.ndarray:
    """Sine wave — VAD treats it as speech because RMS is high.

    Not realistic speech, but reliable for VAD state-machine tests where
    we just need "the level is above threshold for N seconds."
    """
    n = int(seconds * rate)
    t = np.arange(n) / rate
    wave = amp * np.sin(2 * np.pi * freq * t)
    return (wave * 32767).astype(np.int16)


def make_noise_burst(
    seconds: float,
    rate: int = DEFAULT_RATE,
    amp: float = 0.2,
    seed: int | None = 0,
) -> np.ndarray:
    """White-noise burst at the given amplitude. Closer to speech-like
    spectrum than a single tone, while still being deterministic with a seed.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    noise = rng.normal(0.0, amp, n)
    np.clip(noise, -1.0, 1.0, out=noise)
    return (noise * 32767).astype(np.int16)


def concat(*chunks: np.ndarray) -> np.ndarray:
    """Concatenate audio chunks; convenient for `silence + speech + silence`."""
    if not chunks:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(chunks).astype(np.int16)


# ---- TTS feeder --------------------------------------------------------------

def _import_kokoro_engine():
    """Return a loaded KokoroEngine, or raise an informative error."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tts import get_engine  # noqa: E402

    eng = get_engine("kokoro")
    eng._load()
    return eng


def _resample_int16(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Cheap linear resampler — fine for VAD-feeding, not for high-fidelity playback."""
    if src_rate == dst_rate:
        return audio
    src_n = len(audio)
    dst_n = int(round(src_n * dst_rate / src_rate))
    if dst_n <= 0:
        return np.zeros(0, dtype=np.int16)
    src_idx = np.linspace(0, src_n - 1, dst_n)
    floor = src_idx.astype(np.int64)
    frac = src_idx - floor
    floor_clip = np.clip(floor, 0, src_n - 1)
    ceil_clip = np.clip(floor + 1, 0, src_n - 1)
    a = audio[floor_clip].astype(np.float32)
    b = audio[ceil_clip].astype(np.float32)
    out = a + (b - a) * frac.astype(np.float32)
    return out.astype(np.int16)


def feed_tts(
    mic: VirtualMicStream,
    text: str,
    *,
    voice: str = "af_heart",
    speed: float = 1.0,
    leading_silence_s: float = 0.2,
    trailing_silence_s: float = 1.0,
    engine=None,
) -> int:
    """Render `text` via the project's TTS engine and push it into `mic`.

    Pads with leading silence (so VAD starts in IDLE) and trailing silence
    (so VAD's silence_duration window fires DONE_OK after the speech ends).

    Returns the total number of samples pushed (post-resample, at mic.rate).

    Raises `RuntimeError` with a clear message if kokoro fails to load —
    callers should catch and skip the test rather than failing.
    """
    try:
        eng = engine or _import_kokoro_engine()
    except Exception as e:
        raise RuntimeError(f"TTS unavailable: {e}") from e

    import io
    import wave

    wav_bytes = eng.synthesize(text, voice=voice, speed=speed)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        src_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    resampled = _resample_int16(audio_int16, src_rate, mic.rate)

    pushed = 0
    if leading_silence_s > 0:
        sil = make_silence(leading_silence_s, rate=mic.rate)
        mic.push(sil)
        pushed += len(sil)
    mic.push(resampled)
    pushed += len(resampled)
    if trailing_silence_s > 0:
        sil = make_silence(trailing_silence_s, rate=mic.rate)
        mic.push(sil)
        pushed += len(sil)
    return pushed


# ---- VAD simulation driver ---------------------------------------------------

def simulate_vad_over_audio(
    audio_int16: np.ndarray,
    *,
    rate: int = DEFAULT_RATE,
    chunk_size: int = 1024,
    silence_threshold: float = 0.02,
    silence_duration: float = 0.8,
    min_speech_duration: float = 0.3,
):
    """Drive a VadState through `audio_int16` chunk-by-chunk and return the
    full event sequence. Pure: no I/O, deterministic.

    Useful for asserting "given this audio shape, the VAD should fire
    DONE_OK exactly once at frame N."
    """
    from examples._chat_helpers import VadEvent, VadState  # local to avoid cycle

    vad = VadState(
        silence_threshold=silence_threshold,
        silence_duration=silence_duration,
        min_speech_duration=min_speech_duration,
    )
    events: list[VadEvent] = []
    n = len(audio_int16)
    pos = 0
    frame_idx = 0
    while pos < n:
        end = min(pos + chunk_size, n)
        chunk = audio_int16[pos:end]
        pos = end
        # Convert to float32 [-1, 1] to compute RMS the same way mic_chat does.
        f = chunk.astype(np.float32) / 32768.0
        level = float(np.sqrt(np.mean(f ** 2))) if len(f) else 0.0
        now = frame_idx * chunk_size / rate
        events.append(vad.feed(level, now))
        frame_idx += 1
    return events, vad
