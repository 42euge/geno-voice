import base64
import io
import time
import wave

import requests

from .base import STTEngine


class Gemma4Engine(STTEngine):
    name = "gemma4"

    SYSTEM_PROMPT = (
        "You are a verbatim speech transcriber. "
        "Output only the exact words spoken. Never repeat yourself."
    )

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434/api/chat",
        model: str = "gemma4:e4b",
        chunk_seconds: int = 30,
        overlap_seconds: int = 2,
        max_retries: int = 3,
        repeat_penalty: float = 1.5,
        num_predict: int = 256,
        **kwargs,
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.max_retries = max_retries
        self.repeat_penalty = repeat_penalty
        self.num_predict = num_predict

    def transcribe(self, wav_bytes: bytes) -> tuple[str | None, float]:
        t0 = time.monotonic()
        chunks = self._split_chunks(wav_bytes)
        texts = []
        for chunk_wav in chunks:
            text = self._transcribe_chunk(chunk_wav)
            if text:
                texts.append(text)
        elapsed = time.monotonic() - t0
        if not texts:
            return None, elapsed
        return " ".join(texts), elapsed

    def _split_chunks(self, wav_bytes: bytes) -> list[bytes]:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            total_frames = wf.getnframes()
            pcm = wf.readframes(total_frames)

        chunk_frames = self.chunk_seconds * rate
        total_seconds = total_frames / rate
        if total_seconds <= self.chunk_seconds + 5:
            return [wav_bytes]

        overlap_frames = self.overlap_seconds * rate
        step = chunk_frames - overlap_frames
        chunks = []
        start = 0
        frame_size = channels * sampwidth
        while start < total_frames:
            end = min(start + chunk_frames, total_frames)
            chunk_pcm = pcm[start * frame_size : end * frame_size]
            out = io.BytesIO()
            with wave.open(out, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sampwidth)
                wf.setframerate(rate)
                wf.writeframes(chunk_pcm)
            chunks.append(out.getvalue())
            start += step
        return chunks

    def _transcribe_chunk(self, wav_bytes: bytes) -> str | None:
        audio_b64 = base64.b64encode(wav_bytes).decode()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": "Transcribe:", "images": [audio_b64]},
            ],
            "stream": False,
            "options": {
                "repeat_penalty": self.repeat_penalty,
                "num_predict": self.num_predict,
            },
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.ollama_url, json=payload, timeout=120)
                data = resp.json()
                if "error" in data:
                    if attempt < self.max_retries - 1:
                        time.sleep(3)
                        continue
                    return None
                return data.get("message", {}).get("content", "").strip() or None
            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(3)
                    continue
                return None
        return None
