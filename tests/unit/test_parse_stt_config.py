"""Tests for iter-119 — parse_stt_config helper.

Mirrors iter-020's parse_vad_config and iter-034's
parse_filler_config conventions: tolerant of malformed input,
returns a dict with all default keys present, never raises.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_config import (  # noqa: E402
    STT_DEFAULTS,
    parse_stt_config,
)


# ---- Defaults ------------------------------------------------------------


def test_returns_all_default_keys():
    out = parse_stt_config({})
    assert set(out.keys()) == {"engine", "model", "device", "compute_type"}


def test_default_engine_is_whisper():
    """Default preserves Mac-first behavior — Linux operators
    must explicitly opt into faster_whisper."""
    out = parse_stt_config({})
    assert out["engine"] == "whisper"


def test_default_model_is_empty_string():
    """Empty string = "let the engine class default kick in".
    Avoids hardcoding a Mac-only model name that would mislead
    Linux operators."""
    out = parse_stt_config({})
    assert out["model"] == ""


def test_default_device_and_compute():
    out = parse_stt_config({})
    assert out["device"] == "cpu"
    assert out["compute_type"] == "int8"


def test_constants_match_dict_returned_for_empty_input():
    """STT_DEFAULTS is the source of truth — parsing empty
    input must return its values verbatim."""
    out = parse_stt_config({})
    for key, val in STT_DEFAULTS.items():
        assert out[key] == val


# ---- Malformed input is tolerated --------------------------------------


def test_non_mapping_input_returns_defaults():
    """A non-dict (None, list, string) → defaults, no crash.
    Matches iter-020 / iter-034 tolerance."""
    for bad in [None, [], "not a dict", 42, object()]:
        out = parse_stt_config(bad)
        assert out == STT_DEFAULTS


def test_missing_section_returns_defaults():
    """An empty chat dict with no stt_* keys → defaults."""
    out = parse_stt_config({"unrelated": "value"})
    assert out == STT_DEFAULTS


def test_engine_not_string_falls_back():
    """stt_engine: 42 → default "whisper"."""
    out = parse_stt_config({"stt_engine": 42})
    assert out["engine"] == "whisper"


def test_engine_empty_string_falls_back():
    """stt_engine: "" → default. Empty string is treated as
    "not specified," not as "use the empty engine."""
    out = parse_stt_config({"stt_engine": ""})
    assert out["engine"] == "whisper"


def test_engine_whitespace_only_falls_back():
    """stt_engine: "   " → default after strip."""
    out = parse_stt_config({"stt_engine": "   "})
    assert out["engine"] == "whisper"


def test_engine_with_surrounding_whitespace_is_stripped():
    """stt_engine: " faster_whisper " → "faster_whisper"."""
    out = parse_stt_config({"stt_engine": " faster_whisper "})
    assert out["engine"] == "faster_whisper"


# ---- Model -------------------------------------------------------------


def test_model_string_is_extracted():
    out = parse_stt_config({"stt_model": "tiny"})
    assert out["model"] == "tiny"


def test_model_full_repo_path_passes_through():
    """parse_stt_config doesn't translate repo strings —
    that's _resolve_model_repo's job inside the engine."""
    out = parse_stt_config({"stt_model": "Systran/faster-whisper-large-v3"})
    assert out["model"] == "Systran/faster-whisper-large-v3"


def test_model_strips_whitespace():
    out = parse_stt_config({"stt_model": "  base  "})
    assert out["model"] == "base"


def test_model_non_string_falls_back():
    out = parse_stt_config({"stt_model": 123})
    assert out["model"] == ""


def test_model_empty_string_falls_back():
    """Empty model string → default empty (engine uses its own)."""
    out = parse_stt_config({"stt_model": ""})
    assert out["model"] == ""


# ---- Device ------------------------------------------------------------


def test_device_cpu_explicit():
    out = parse_stt_config({"stt_device": "cpu"})
    assert out["device"] == "cpu"


def test_device_cuda():
    out = parse_stt_config({"stt_device": "cuda"})
    assert out["device"] == "cuda"


def test_device_strips_whitespace():
    out = parse_stt_config({"stt_device": "  cuda  "})
    assert out["device"] == "cuda"


def test_device_non_string_falls_back():
    out = parse_stt_config({"stt_device": 123})
    assert out["device"] == "cpu"


# ---- Compute type ------------------------------------------------------


def test_compute_type_extracted():
    out = parse_stt_config({"stt_compute": "float16"})
    assert out["compute_type"] == "float16"


def test_compute_type_strips_whitespace():
    out = parse_stt_config({"stt_compute": " int8  "})
    assert out["compute_type"] == "int8"


def test_compute_type_non_string_falls_back():
    out = parse_stt_config({"stt_compute": None})
    assert out["compute_type"] == "int8"


# ---- Composite -----------------------------------------------------------


def test_full_config_passes_through():
    """All four keys set explicitly → all four reflected in output."""
    out = parse_stt_config({
        "stt_engine": "faster_whisper",
        "stt_model": "large-v3",
        "stt_device": "cuda",
        "stt_compute": "float16",
    })
    assert out == {
        "engine": "faster_whisper",
        "model": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
    }


def test_partial_config_backfills_defaults():
    """Setting only engine + model → device + compute_type
    keep their defaults."""
    out = parse_stt_config({
        "stt_engine": "faster_whisper",
        "stt_model": "tiny",
    })
    assert out["engine"] == "faster_whisper"
    assert out["model"] == "tiny"
    assert out["device"] == STT_DEFAULTS["device"]
    assert out["compute_type"] == STT_DEFAULTS["compute_type"]


def test_does_not_mutate_input_dict():
    """The function must not mutate its argument — callers reuse
    the same chat_cfg for parse_filler_config and parse_vad_config."""
    cfg = {"stt_engine": "faster_whisper", "stt_model": "tiny"}
    snapshot = dict(cfg)
    parse_stt_config(cfg)
    assert cfg == snapshot


def test_does_not_mutate_defaults_constant():
    """Repeated calls with mutating side effects don't leak
    into STT_DEFAULTS."""
    snapshot = dict(STT_DEFAULTS)
    parse_stt_config({"stt_engine": "faster_whisper"})
    parse_stt_config({"stt_engine": "different"})
    assert STT_DEFAULTS == snapshot


# ---- Independence from other config sections ---------------------------


def test_irrelevant_keys_ignored():
    """Other chat-cfg sections (vad, fillers, etc.) don't bleed
    into the STT result."""
    out = parse_stt_config({
        "vad": {"silence_threshold": 0.05},
        "fillers": ["hi"],
        "stt_engine": "faster_whisper",
    })
    assert out["engine"] == "faster_whisper"
    # Defaults preserved for unrelated keys.
    assert out["model"] == ""
