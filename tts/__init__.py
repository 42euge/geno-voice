from .kokoro_engine import KokoroEngine

ENGINES = {
    "kokoro": KokoroEngine,
}


def get_engine(name: str, **kwargs):
    cls = ENGINES.get(name)
    if cls is None:
        raise ValueError(f"Unknown TTS engine: {name!r}. Available: {list(ENGINES)}")
    return cls(**kwargs)
