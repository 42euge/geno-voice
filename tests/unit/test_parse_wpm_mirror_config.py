"""Tests for iter-214 — parse_wpm_mirror_config.

The tolerant parser for the optional ``chat.wpm_mirror`` section that drives the
live WPM-mirroring path (iter-213 seam). Contract:
  - ALWAYS returns a dict carrying ``enabled`` (a bool).
  - Off-by-default to the bone: missing / non-mapping section, or ``enabled``
    absent / non-bool ⇒ ``{"enabled": False}``.
  - Each numeric tunable is included ONLY when present-and-valid; otherwise it
    is omitted so WpmMirrorConfig backfills its own default.
  - Malformed values fall back rather than raising; with a ``warn`` callable a
    one-liner is emitted per present-but-rejected entry (iter-187 template).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_config import parse_wpm_mirror_config  # noqa: E402


# ---- Off-by-default ---------------------------------------------------------


def test_missing_section_disabled():
    assert parse_wpm_mirror_config({}) == {"enabled": False}


def test_non_mapping_chat_cfg_disabled():
    assert parse_wpm_mirror_config(None) == {"enabled": False}
    assert parse_wpm_mirror_config("nope") == {"enabled": False}


def test_non_mapping_section_disabled():
    assert parse_wpm_mirror_config({"wpm_mirror": "yes"}) == {"enabled": False}
    assert parse_wpm_mirror_config({"wpm_mirror": [1, 2]}) == {"enabled": False}


def test_empty_section_disabled():
    assert parse_wpm_mirror_config({"wpm_mirror": {}}) == {"enabled": False}


def test_enabled_false_explicit():
    out = parse_wpm_mirror_config({"wpm_mirror": {"enabled": False}})
    assert out == {"enabled": False}


# ---- enabled must be a real bool -------------------------------------------


def test_enabled_true_bool():
    out = parse_wpm_mirror_config({"wpm_mirror": {"enabled": True}})
    assert out["enabled"] is True


def test_truthy_int_does_not_enable():
    out = parse_wpm_mirror_config({"wpm_mirror": {"enabled": 1}})
    assert out["enabled"] is False


def test_truthy_string_does_not_enable():
    out = parse_wpm_mirror_config({"wpm_mirror": {"enabled": "yes"}})
    assert out["enabled"] is False


# ---- Float tunables: included only when present-and-valid ------------------


def test_valid_floats_included():
    out = parse_wpm_mirror_config({"wpm_mirror": {
        "enabled": True,
        "base_wpm": 170,
        "strength": 0.4,
        "min_speed": 0.7,
        "max_speed": 1.4,
        "min_delta": 0.02,
    }})
    assert out == {
        "enabled": True,
        "base_wpm": 170.0,
        "strength": 0.4,
        "min_speed": 0.7,
        "max_speed": 1.4,
        "min_delta": 0.02,
    }
    # all coerced to float
    assert all(isinstance(out[k], float) for k in (
        "base_wpm", "strength", "min_speed", "max_speed", "min_delta"))


def test_partial_floats_only_present_keys():
    out = parse_wpm_mirror_config({"wpm_mirror": {
        "enabled": True, "strength": 0.3,
    }})
    assert out == {"enabled": True, "strength": 0.3}


def test_min_delta_zero_accepted():
    """min_delta == 0 is the legitimate 'deadband off' value."""
    out = parse_wpm_mirror_config({"wpm_mirror": {
        "enabled": True, "min_delta": 0,
    }})
    assert out["min_delta"] == 0.0


def test_min_delta_negative_rejected():
    out = parse_wpm_mirror_config({"wpm_mirror": {
        "enabled": True, "min_delta": -0.1,
    }})
    assert "min_delta" not in out


def test_non_min_delta_zero_rejected():
    """base_wpm/strength/min_speed/max_speed require > 0; zero is dropped."""
    out = parse_wpm_mirror_config({"wpm_mirror": {
        "enabled": True, "base_wpm": 0, "min_speed": 0,
    }})
    assert "base_wpm" not in out
    assert "min_speed" not in out


def test_negative_float_rejected():
    out = parse_wpm_mirror_config({"wpm_mirror": {
        "enabled": True, "base_wpm": -50,
    }})
    assert "base_wpm" not in out


def test_bool_float_value_rejected():
    """A bool sneaking into a numeric slot must not become 1.0/0.0."""
    out = parse_wpm_mirror_config({"wpm_mirror": {
        "enabled": True, "strength": True, "min_delta": False,
    }})
    assert "strength" not in out
    assert "min_delta" not in out


def test_string_float_value_rejected():
    out = parse_wpm_mirror_config({"wpm_mirror": {
        "enabled": True, "base_wpm": "fast",
    }})
    assert "base_wpm" not in out


def test_unknown_keys_ignored():
    out = parse_wpm_mirror_config({"wpm_mirror": {
        "enabled": True, "frobnicate": 9,
    }})
    assert "frobnicate" not in out
    assert out == {"enabled": True}


def test_tunables_present_even_when_disabled():
    """Tunables are parsed regardless of enabled — enabled gates them at the
    WpmMirrorConfig level, not here."""
    out = parse_wpm_mirror_config({"wpm_mirror": {
        "enabled": False, "strength": 0.2,
    }})
    assert out == {"enabled": False, "strength": 0.2}


# ---- warn seam --------------------------------------------------------------


def test_warn_on_non_mapping_section():
    warnings = []
    parse_wpm_mirror_config(
        {"wpm_mirror": "on"}, warn=warnings.append,
    )
    assert len(warnings) == 1
    assert "wpm_mirror config ignored" in warnings[0]


def test_no_warn_when_section_absent():
    warnings = []
    parse_wpm_mirror_config({}, warn=warnings.append)
    assert warnings == []


def test_warn_on_bad_enabled():
    warnings = []
    parse_wpm_mirror_config(
        {"wpm_mirror": {"enabled": "yes"}}, warn=warnings.append,
    )
    assert len(warnings) == 1
    assert "wpm_mirror.enabled ignored" in warnings[0]


def test_warn_on_bad_float():
    warnings = []
    parse_wpm_mirror_config(
        {"wpm_mirror": {"enabled": True, "strength": "fast"}},
        warn=warnings.append,
    )
    assert len(warnings) == 1
    assert "wpm_mirror.strength ignored" in warnings[0]
    assert "a positive number" in warnings[0]


def test_warn_min_delta_bound_message():
    warnings = []
    parse_wpm_mirror_config(
        {"wpm_mirror": {"enabled": True, "min_delta": -1}},
        warn=warnings.append,
    )
    assert len(warnings) == 1
    assert ">= 0" in warnings[0]


def test_no_warn_on_missing_float_keys():
    warnings = []
    parse_wpm_mirror_config(
        {"wpm_mirror": {"enabled": True}}, warn=warnings.append,
    )
    assert warnings == []


def test_warn_none_is_silent():
    # No warn callable → no raise, just the fallback dict.
    out = parse_wpm_mirror_config({"wpm_mirror": {"strength": "bad"}})
    assert out == {"enabled": False}


# ---- Integration with WpmMirrorConfig (structural splat) -------------------


def test_result_splats_into_wpm_mirror_config():
    """The parsed dict is valid kwargs for WpmMirrorConfig — loaded by file
    path to avoid the pipecat-eager session package __init__."""
    import importlib.util

    wm_path = ROOT / "session" / "wpm_mirror.py"
    spec = importlib.util.spec_from_file_location("_wm_cfg_test", wm_path)
    wm = importlib.util.module_from_spec(spec)
    sys.modules["_wm_cfg_test"] = wm
    spec.loader.exec_module(wm)

    out = parse_wpm_mirror_config({"wpm_mirror": {
        "enabled": True, "base_wpm": 170, "strength": 0.4,
    }})
    cfg = wm.WpmMirrorConfig(**out)
    assert cfg.enabled is True
    assert cfg.base_wpm == 170.0
    assert cfg.strength == 0.4
    # un-set tunables fall back to the WpmMirrorConfig defaults
    assert cfg.min_speed == wm.DEFAULT_MIN_SPEED
