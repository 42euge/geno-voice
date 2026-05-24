import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AUDIO_DIR = Path(__file__).resolve().parent.parent / "test-data"
DEFAULT_AUDIO = AUDIO_DIR / "diarization" / "segment.wav"


@pytest.fixture(scope="session")
def audio_bytes():
    """Load default benchmark audio sample."""
    assert DEFAULT_AUDIO.exists(), f"Missing test audio: {DEFAULT_AUDIO}"
    return DEFAULT_AUDIO.read_bytes()


@pytest.fixture(scope="session")
def audio_dir():
    """Path to test-data directory for multi-file benchmarks."""
    return AUDIO_DIR


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that load large models")
    config.addinivalue_line("markers", "quick: marks tests with small/fast models")
