from abc import ABC, abstractmethod
from typing import Iterator


class TTSEngine(ABC):
    name: str

    @abstractmethod
    def synthesize(self, text: str, voice: str, speed: float) -> bytes:
        """Synthesize text to WAV audio bytes."""
        ...

    @abstractmethod
    def stream(self, text: str, voice: str, speed: float) -> Iterator[bytes]:
        """Yield WAV audio chunks per sentence."""
        ...

    @abstractmethod
    def list_voices(self) -> list[dict]:
        """Return available voices as [{id, name, gender}]."""
        ...
