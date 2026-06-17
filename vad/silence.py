import io
import math
import struct
import time
import wave

RATE = 16000
SAMPLE_WIDTH = 2
FRAME_SIZE = RATE * SAMPLE_WIDTH


def _require_positive(name: str, value: float) -> float:
    """Validate that a ``SilenceDetector`` timing/threshold knob is a
    finite, strictly-positive number.

    Raises :class:`ValueError` on a non-number, ``NaN``/``inf``, or a
    value ``<= 0``. This mirrors the ``val > 0`` rule that
    ``examples/_chat_config.parse_vad_config`` applies when it sanitizes
    the optional ``vad`` config section (iter-033) — but where that
    parser is *tolerant* (falls back to defaults so a typo'd config can't
    take down the chat loop), the constructor is *strict*: a caller
    building a ``SilenceDetector`` directly in code has no defaults to
    fall back on, and a non-positive ``threshold``/duration would silently
    break the state machine (e.g. ``silence_duration=0`` emits on the
    first silent frame; ``threshold<=0`` marks every frame as speech).
    Fail fast at construction with a message that names the offending
    knob, the same garbage-in contract the ``gv`` CLI trio established
    (iter-182 ``--speed`` / iter-183 ``--voice`` / iter-184 ``--model``).

    ``bool`` is rejected explicitly: ``True``/``False`` are ``int``
    subclasses that would otherwise sail through as ``1``/``0``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return float(value)


def make_wav(pcm_bytes: bytes, rate: int = RATE, channels: int = 1, sample_width: int = SAMPLE_WIDTH) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class SilenceDetector:
    def __init__(
        self,
        threshold: float = 0.02,
        silence_duration: float = 0.8,
        min_chunk_duration: float = 0.5,
        max_chunk_duration: float = 25,
    ):
        self.threshold = _require_positive("threshold", threshold)
        self.silence_duration = _require_positive("silence_duration", silence_duration)
        self.min_chunk_duration = _require_positive("min_chunk_duration", min_chunk_duration)
        self.max_chunk_duration = _require_positive("max_chunk_duration", max_chunk_duration)

        # max must exceed min, else the max-duration cut would fire on a
        # buffer the min-duration floor then drops as "too short" — every
        # emit would silently return None and no chunk would ever surface.
        if self.max_chunk_duration <= self.min_chunk_duration:
            raise ValueError(
                "max_chunk_duration must be greater than min_chunk_duration, "
                f"got max={self.max_chunk_duration!r} <= min={self.min_chunk_duration!r}"
            )

        self.audio_buffer = bytearray()
        self.silence_start: float | None = None
        self.speaking = False

    @staticmethod
    def rms_amplitude(pcm_bytes: bytes) -> float:
        if len(pcm_bytes) < 2:
            return 0.0
        n = len(pcm_bytes) // 2
        samples = struct.unpack(f"<{n}h", pcm_bytes[: n * 2])
        return (sum(s * s for s in samples) / n) ** 0.5 / 32768.0

    def feed(self, raw_pcm: bytes) -> tuple[float, bytes | None]:
        amp = self.rms_amplitude(raw_pcm)
        is_silent = amp < self.threshold
        result = None

        if not is_silent:
            self.speaking = True
            self.silence_start = None
            self.audio_buffer.extend(raw_pcm)
        elif self.speaking:
            self.audio_buffer.extend(raw_pcm)
            if self.silence_start is None:
                self.silence_start = time.monotonic()
            elif time.monotonic() - self.silence_start >= self.silence_duration:
                result = self._emit_chunk()

        if self.speaking:
            chunk_dur = len(self.audio_buffer) / FRAME_SIZE
            if chunk_dur >= self.max_chunk_duration:
                result = self._emit_chunk()

        return amp, result

    def flush(self) -> bytes | None:
        if self.audio_buffer:
            chunk_dur = len(self.audio_buffer) / FRAME_SIZE
            if chunk_dur >= self.min_chunk_duration:
                wav = make_wav(bytes(self.audio_buffer))
                self._reset()
                return wav
        self._reset()
        return None

    def _emit_chunk(self) -> bytes | None:
        chunk_dur = len(self.audio_buffer) / FRAME_SIZE
        if chunk_dur >= self.min_chunk_duration:
            wav = make_wav(bytes(self.audio_buffer))
            self._reset()
            return wav
        self._reset()
        return None

    def _reset(self):
        self.audio_buffer = bytearray()
        self.silence_start = None
        self.speaking = False

    @property
    def buffer_duration(self) -> float:
        return len(self.audio_buffer) / FRAME_SIZE if self.audio_buffer else 0.0
