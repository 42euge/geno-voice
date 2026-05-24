import io
import struct
import time
import wave

RATE = 16000
SAMPLE_WIDTH = 2
FRAME_SIZE = RATE * SAMPLE_WIDTH


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
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.min_chunk_duration = min_chunk_duration
        self.max_chunk_duration = max_chunk_duration

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
