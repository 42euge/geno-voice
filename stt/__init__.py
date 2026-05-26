from .whisper_engine import WhisperEngine
from .gemma4_engine import Gemma4Engine
from .faster_whisper_engine import FasterWhisperEngine

ENGINES = {
    "whisper": WhisperEngine,                # MLX — Apple Silicon only
    "gemma4": Gemma4Engine,
    # iter-118: x86_64 Linux + CUDA STT path. Lazy-imports
    # faster_whisper inside _load(), so this module is importable
    # even when the package isn't installed.
    "faster_whisper": FasterWhisperEngine,
}


def get_engine(name: str, **kwargs):
    cls = ENGINES.get(name)
    if cls is None:
        raise ValueError(f"Unknown STT engine: {name!r}. Available: {list(ENGINES)}")
    return cls(**kwargs)
