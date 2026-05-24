from abc import ABC, abstractmethod


class STTEngine(ABC):
    name: str

    @abstractmethod
    def transcribe(self, wav_bytes: bytes) -> tuple[str | None, float]:
        """Transcribe WAV audio bytes. Returns (text, elapsed_seconds)."""
        ...
