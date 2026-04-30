from .whisper_engine import WhisperEngine
from .gemma4_engine import Gemma4Engine

ENGINES = {
    "whisper": WhisperEngine,
    "gemma4": Gemma4Engine,
}


def get_engine(name: str, **kwargs):
    cls = ENGINES.get(name)
    if cls is None:
        raise ValueError(f"Unknown STT engine: {name!r}. Available: {list(ENGINES)}")
    return cls(**kwargs)
