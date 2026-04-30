import io
import re
import wave
from typing import Iterator

from .base import TTSEngine

VOICES = [
    {"id": "af_heart", "name": "Heart", "gender": "female"},
    {"id": "af_bella", "name": "Bella", "gender": "female"},
    {"id": "af_sarah", "name": "Sarah", "gender": "female"},
    {"id": "af_nova", "name": "Nova", "gender": "female"},
    {"id": "af_nicole", "name": "Nicole", "gender": "female"},
    {"id": "af_sky", "name": "Sky", "gender": "female"},
    {"id": "af_river", "name": "River", "gender": "female"},
    {"id": "af_jessica", "name": "Jessica", "gender": "female"},
    {"id": "af_alloy", "name": "Alloy", "gender": "female"},
    {"id": "af_aoede", "name": "Aoede", "gender": "female"},
    {"id": "am_adam", "name": "Adam", "gender": "male"},
    {"id": "am_michael", "name": "Michael", "gender": "male"},
    {"id": "am_fenrir", "name": "Fenrir", "gender": "male"},
    {"id": "am_puck", "name": "Puck", "gender": "male"},
    {"id": "am_echo", "name": "Echo", "gender": "male"},
    {"id": "am_eric", "name": "Eric", "gender": "male"},
    {"id": "am_liam", "name": "Liam", "gender": "male"},
    {"id": "am_onyx", "name": "Onyx", "gender": "male"},
    {"id": "am_santa", "name": "Santa", "gender": "male"},
]

SAMPLE_RATE = 24000
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _pcm_to_wav(pcm_float_samples, rate: int = SAMPLE_RATE) -> bytes:
    import numpy as np

    pcm_int16 = (pcm_float_samples * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


class KokoroEngine(TTSEngine):
    name = "kokoro"

    def __init__(self, language: str = "a", **kwargs):
        self.language = language
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code=self.language)

    def synthesize(self, text: str, voice: str = "af_heart", speed: float = 1.0) -> bytes:
        self._load()
        import numpy as np

        all_samples = []
        for result in self._pipeline(text, voice=voice, speed=speed):
            all_samples.append(result.audio)
        if not all_samples:
            return _pcm_to_wav(np.array([], dtype=np.float32))
        combined = np.concatenate(all_samples)
        return _pcm_to_wav(combined)

    def stream(self, text: str, voice: str = "af_heart", speed: float = 1.0) -> Iterator[bytes]:
        self._load()
        sentences = _SENTENCE_RE.split(text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return

        for sentence in sentences:
            import numpy as np
            chunks = []
            for result in self._pipeline(sentence, voice=voice, speed=speed):
                chunks.append(result.audio)
            if chunks:
                combined = np.concatenate(chunks)
                yield _pcm_to_wav(combined)

    def list_voices(self) -> list[dict]:
        return list(VOICES)
