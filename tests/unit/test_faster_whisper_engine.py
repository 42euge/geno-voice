"""Tests for iter-118 — FasterWhisperEngine.

Three concerns:
  - Construction + STTEngine contract conformance (always run)
  - Model-repo string translation (pure function, always run)
  - Real round-trip via faster-whisper (skip when unavailable)
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stt import ENGINES, get_engine  # noqa: E402
from stt.base import STTEngine  # noqa: E402
from stt.faster_whisper_engine import (  # noqa: E402
    FasterWhisperEngine,
    _resolve_model_repo,
)


# ---- Repo-string translation (pure function) ---------------------------


@pytest.mark.parametrize("repo,expected", [
    ("tiny", "tiny"),
    ("base", "base"),
    ("small", "small"),
    ("large-v3", "large-v3"),
    ("Systran/faster-whisper-large-v3", "Systran/faster-whisper-large-v3"),
    ("/local/path/to/model", "/local/path/to/model"),
])
def test_passthrough_for_native_strings(repo, expected):
    """Plain aliases + HF repo IDs + local paths pass through
    unchanged."""
    assert _resolve_model_repo(repo) == expected


@pytest.mark.parametrize("mlx_repo,size", [
    ("mlx-community/whisper-tiny", "tiny"),
    ("mlx-community/whisper-base", "base"),
    ("mlx-community/whisper-small", "small"),
    ("mlx-community/whisper-medium", "medium"),
    ("mlx-community/whisper-large-v3", "large-v3"),
    ("mlx-community/whisper-large-v3-turbo", "large-v3-turbo"),
])
def test_strips_mlx_namespace(mlx_repo, size):
    """MLX-style repos lose the namespace + 'whisper-' prefix."""
    assert _resolve_model_repo(mlx_repo) == size


def test_unrecognized_mlx_pattern_passes_through():
    """MLX-namespace strings that don't match the known shape
    pass through (caller can debug)."""
    assert _resolve_model_repo(
        "mlx-community/whisper-experimental-foo",
    ) == "mlx-community/whisper-experimental-foo"


# ---- Construction + contract --------------------------------------------


def test_implements_stt_engine():
    """Class is registered as an STTEngine subclass."""
    assert issubclass(FasterWhisperEngine, STTEngine)


def test_default_construction():
    """Default kwargs match the iter-117 audio-fixture choices
    (tiny / cpu / int8) so a no-arg constructor is operator-
    friendly on x86_64 Linux."""
    e = FasterWhisperEngine()
    assert e.model_repo == "tiny"
    assert e.device == "cpu"
    assert e.compute_type == "int8"
    assert e.name == "faster_whisper"


def test_construction_with_overrides():
    """Operators can override every kwarg."""
    e = FasterWhisperEngine(
        model_repo="large-v3",
        device="cuda",
        compute_type="float16",
    )
    assert e.model_repo == "large-v3"
    assert e.device == "cuda"
    assert e.compute_type == "float16"


def test_construction_accepts_extra_kwargs():
    """Forward-compat: unknown kwargs accepted (matches
    WhisperEngine which has the same **kwargs swallow)."""
    e = FasterWhisperEngine(model_repo="tiny", future_param="ok")
    assert e.model_repo == "tiny"


def test_no_load_at_construction():
    """The constructor must not load the model — mirrors
    WhisperEngine. Ensures `mic_chat.py:run_chat`'s lazy load
    semantics work."""
    e = FasterWhisperEngine()
    assert e._model is None


# ---- Cache key behavior ----------------------------------------------


def test_cache_key_uses_resolved_repo():
    """Cache keys use the RESOLVED repo string, not the raw
    input. Two MLX repos that resolve to the same size share
    the cache."""
    e1 = FasterWhisperEngine(model_repo="mlx-community/whisper-tiny")
    e2 = FasterWhisperEngine(model_repo="tiny")
    assert e1._cache_key() == e2._cache_key()


def test_cache_key_includes_device_and_compute_type():
    """Different device or compute_type → different cache slot."""
    cpu = FasterWhisperEngine(device="cpu", compute_type="int8")
    cuda = FasterWhisperEngine(device="cuda", compute_type="float16")
    assert cpu._cache_key() != cuda._cache_key()


# ---- Factory registration -------------------------------------------


def test_engine_registered_in_factory():
    """get_engine('faster_whisper') returns an instance."""
    e = get_engine("faster_whisper")
    assert isinstance(e, FasterWhisperEngine)


def test_factory_forwards_kwargs():
    """Constructor kwargs flow through get_engine."""
    e = get_engine("faster_whisper", model_repo="base", device="cpu")
    assert e.model_repo == "base"
    assert e.device == "cpu"


def test_factory_rejects_unknown_engine():
    """Bogus name → ValueError listing the available engines.
    Existing behavior, asserted here so this iter doesn't break it."""
    with pytest.raises(ValueError, match="Unknown STT engine"):
        get_engine("not_a_real_engine")


def test_engines_dict_has_faster_whisper():
    """Sanity: the entry exists and points to the right class."""
    assert "faster_whisper" in ENGINES
    assert ENGINES["faster_whisper"] is FasterWhisperEngine


# ---- Real round-trip (skip when unavailable) -------------------------


try:
    import faster_whisper  # noqa: F401
    _FW_AVAILABLE = True
except Exception as e:
    _FW_AVAILABLE = False
    _FW_IMPORT_ERROR = str(e)


@pytest.fixture(scope="module")
def known_audio_bytes():
    """Read the iter-117 clean.wav fixture as bytes — the same
    audio the iter-117 integration test uses, so behavior is
    consistent across both."""
    fixture = ROOT / "tests" / "fixtures" / "wer" / "clean.wav"
    if not fixture.exists():
        pytest.skip(f"clean.wav fixture missing: {fixture}")
    return fixture.read_bytes()


def test_real_transcription_round_trips(known_audio_bytes):
    """End-to-end: construct engine → transcribe wav bytes →
    get back a sensible English transcript."""
    if not _FW_AVAILABLE:
        pytest.skip(f"faster-whisper unavailable: {_FW_IMPORT_ERROR}")
    try:
        engine = FasterWhisperEngine(
            model_repo="tiny", device="cpu", compute_type="int8",
        )
    except Exception as e:
        pytest.skip(f"engine construction failed: {e}")

    text, elapsed = engine.transcribe(known_audio_bytes)
    if text is None:
        pytest.skip(
            "transcribe returned None — model probably failed to load "
            "(no cache + no network)"
        )
    # Reference: "what is the weather today" → expect at least
    # one of the content words to come through cleanly. Don't
    # assert exact match; faster-whisper output varies by model
    # version + audio quality.
    text_lower = text.lower()
    assert "weather" in text_lower or "today" in text_lower, (
        f"transcript missing expected words: {text!r}"
    )
    assert elapsed > 0.0
    assert elapsed < 30.0  # generous upper bound — tiny model, short clip


def test_transcribe_returns_elapsed_on_failure():
    """Even when the model load or transcribe step fails, the
    return shape is preserved: ``(None, elapsed)`` — never
    raises. This is what mic_chat.py's error-handling path
    expects."""
    # Force failure by passing an invalid model name; the model
    # load inside _load() will raise, the except clause should
    # catch it.
    engine = FasterWhisperEngine(
        model_repo="this-model-does-not-exist-anywhere",
        device="cpu", compute_type="int8",
    )
    if not _FW_AVAILABLE:
        pytest.skip(f"faster-whisper unavailable: {_FW_IMPORT_ERROR}")

    # Use a minimal valid wav header so we don't fail before _load().
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)  # 0.1s of silence
    text, elapsed = engine.transcribe(buf.getvalue())
    assert text is None
    assert elapsed > 0.0
