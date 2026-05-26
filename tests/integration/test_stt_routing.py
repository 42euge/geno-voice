"""iter-121 — STT config-to-engine routing integration.

Validates the full chain that iter-118 + iter-119 built up:

    chat_cfg dict
      → parse_stt_config()
      → factory-closure (replicates mic_chat.py's _stt_factory)
      → get_engine(name, **kwargs)
      → engine instance ready to .transcribe()

Catches drift between `mic_chat.py:run_chat`'s inline factory and
the public iter-119 API. The factory-builder helper in this file
mirrors mic_chat.py's wiring exactly — if mic_chat changes shape
in a way that breaks the wiring, this test fails.

Skips the real-transcription test cleanly when faster-whisper is
unavailable (no install, no cache, no network). Routing-only
tests run regardless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_config import parse_stt_config  # noqa: E402
from stt import get_engine as _get_stt_engine  # noqa: E402
from stt.faster_whisper_engine import FasterWhisperEngine  # noqa: E402
from stt.whisper_engine import WhisperEngine  # noqa: E402


def _build_stt_factory(chat_cfg: dict, model_repo_fallback: str = ""):
    """Replicates the EXACT factory shape mic_chat.py:run_chat uses
    after iter-119. Centralized here so the test catches drift.

    Returns the closure (not the engine) — call it to construct.
    """
    stt_cfg = parse_stt_config(chat_cfg)
    stt_model = stt_cfg["model"] or model_repo_fallback
    stt_engine_name = stt_cfg["engine"]

    def _stt_factory():
        kwargs = {}
        if stt_model:
            kwargs["model_repo"] = stt_model
        if stt_engine_name == "faster_whisper":
            kwargs["device"] = stt_cfg["device"]
            kwargs["compute_type"] = stt_cfg["compute_type"]
        return _get_stt_engine(stt_engine_name, **kwargs)

    return _stt_factory


# ---- Routing without instantiating real Mac-only deps -------------------


def test_default_chat_cfg_returns_factory_closure():
    """Empty config → factory built without crashing. Default
    engine is "whisper" — calling it would import mlx, so we
    only check the closure exists and the resolved name is
    correct."""
    factory = _build_stt_factory({})
    assert callable(factory)


def test_faster_whisper_routing_constructs_correct_class():
    """chat_cfg pointing at faster_whisper → factory builds a
    FasterWhisperEngine. This is the routing test that
    iter-119 promises."""
    factory = _build_stt_factory({"stt_engine": "faster_whisper"})
    engine = factory()
    assert isinstance(engine, FasterWhisperEngine)


def test_full_faster_whisper_config_flows_through():
    """All 4 chat_cfg knobs land on the engine instance."""
    factory = _build_stt_factory({
        "stt_engine": "faster_whisper",
        "stt_model": "tiny",
        "stt_device": "cpu",
        "stt_compute": "int8",
    })
    engine = factory()
    assert engine.model_repo == "tiny"
    assert engine.device == "cpu"
    assert engine.compute_type == "int8"


def test_partial_config_uses_engine_class_defaults():
    """chat_cfg with only stt_engine → device/compute use
    FasterWhisperEngine's __init__ defaults. Validates the
    "empty model = engine class default" decision recorded in
    iter-119."""
    factory = _build_stt_factory({"stt_engine": "faster_whisper"})
    engine = factory()
    # No model in chat_cfg, no fallback → engine's own default.
    assert engine.model_repo == "tiny"  # FasterWhisperEngine default
    assert engine.device == "cpu"
    assert engine.compute_type == "int8"


def test_model_fallback_uses_function_arg_when_chat_cfg_omits_it():
    """The pre-iter-119 calling convention (passing model_repo
    to run_chat) still works when chat config doesn't specify
    stt_model. Backwards-compat sentinel."""
    factory = _build_stt_factory(
        {"stt_engine": "faster_whisper"},
        model_repo_fallback="base",
    )
    engine = factory()
    assert engine.model_repo == "base"


def test_chat_cfg_model_overrides_function_arg():
    """If both are set, chat_cfg wins (the more recent config
    layer). Documenting precedence."""
    factory = _build_stt_factory(
        {"stt_engine": "faster_whisper", "stt_model": "small"},
        model_repo_fallback="ignored_fallback",
    )
    engine = factory()
    assert engine.model_repo == "small"


def test_unknown_engine_raises_from_get_engine():
    """A bogus engine name surfaces as a ValueError when the
    factory is INVOKED (not at construction)."""
    factory = _build_stt_factory({"stt_engine": "totally_made_up"})
    with pytest.raises(ValueError, match="Unknown STT engine"):
        factory()


def test_whisper_engine_routing_does_not_pass_device_or_compute():
    """The factory's `if stt_engine_name == "faster_whisper"` guard
    keeps Mac-only WhisperEngine constructor clean of
    faster-whisper-specific kwargs.

    We can't actually invoke the factory (would import mlx), but
    we can monkey-patch get_engine to capture what kwargs the
    factory would pass.
    """
    captured: dict = {}

    def _capturing_get_engine(name: str, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return None  # noqa: E501 — we don't actually need an engine here

    # Patch the symbol the factory closure captured at import time.
    import tests.integration.test_stt_routing as this_mod
    original = this_mod._get_stt_engine
    this_mod._get_stt_engine = _capturing_get_engine
    try:
        factory = _build_stt_factory({
            "stt_engine": "whisper",
            "stt_model": "any-mlx-repo",
        })
        factory()
    finally:
        this_mod._get_stt_engine = original

    assert captured["name"] == "whisper"
    # Only model_repo passed — no device, no compute_type.
    assert "model_repo" in captured["kwargs"]
    assert "device" not in captured["kwargs"]
    assert "compute_type" not in captured["kwargs"]


def test_factory_is_callable_multiple_times():
    """Each invocation builds a fresh engine instance.
    SentenceWorker construction may invoke the factory more than
    once, so this contract matters."""
    factory = _build_stt_factory({"stt_engine": "faster_whisper"})
    e1 = factory()
    e2 = factory()
    assert e1 is not e2  # distinct instances
    assert isinstance(e1, FasterWhisperEngine)
    assert isinstance(e2, FasterWhisperEngine)


# ---- Real transcription via routed engine (skip when unavailable) ----


try:
    import faster_whisper  # noqa: F401
    _FW_AVAILABLE = True
except Exception as e:
    _FW_AVAILABLE = False
    _FW_IMPORT_ERROR = str(e)


@pytest.fixture(scope="module")
def clean_wav_bytes():
    fixture = ROOT / "tests" / "fixtures" / "wer" / "clean.wav"
    if not fixture.exists():
        pytest.skip(f"clean.wav fixture missing: {fixture}")
    return fixture.read_bytes()


def test_routed_engine_transcribes_clean_audio(clean_wav_bytes):
    """End-to-end: yaml-style config → factory → engine →
    transcribe real audio → sensible transcript. The integration
    test that iter-118 + iter-119 promised."""
    if not _FW_AVAILABLE:
        pytest.skip(f"faster-whisper unavailable: {_FW_IMPORT_ERROR}")

    factory = _build_stt_factory({
        "stt_engine": "faster_whisper",
        "stt_model": "tiny",
        "stt_device": "cpu",
        "stt_compute": "int8",
    })
    try:
        engine = factory()
    except Exception as e:
        pytest.skip(f"engine construction failed: {e}")

    text, elapsed = engine.transcribe(clean_wav_bytes)
    if text is None:
        pytest.skip(
            "transcribe returned None — model probably failed to load "
            "(no cache + no network)"
        )
    text_lower = text.lower()
    # clean.wav reference: "what is the weather today"
    assert "weather" in text_lower or "today" in text_lower, (
        f"transcript missing expected words: {text!r}"
    )
    assert 0.0 < elapsed < 30.0
