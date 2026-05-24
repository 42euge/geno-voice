"""Config parsing + validation extracted from mic_chat.

Until iter-017 the ``load_llm_config`` function in mic_chat did
three things at once:
  1. Read the YAML file (file I/O)
  2. Parse and validate the contents (pure logic)
  3. Print errors and ``sys.exit(1)`` (process termination)

That coupling made (2) untestable. It also failed in some real
ways:
  - An empty YAML file loaded as ``None`` and crashed with
    ``AttributeError`` instead of a useful "config is empty"
    message.
  - Missing required fields (``model``, ``base_url``) didn't
    raise — the user's first hint was an HTTP 400 deep in the
    request stack.
  - ``${ENV_VAR}`` placeholders that weren't resolved (env var
    unset) caused ``sys.exit`` from inside library code, which
    is hard to test and surprising for a future caller that
    might want to handle the error.

This module hosts the pure-data parts:
  - ``ConfigError`` — clean exception type the caller can catch.
  - ``parse_llm_config(cfg_dict, env=os.environ)`` —
    validate-and-resolve. Raises ``ConfigError`` on any problem.
  - ``parse_chat_config(cfg_dict)`` — extract the optional
    ``chat`` section, returning an empty dict when it's missing.

mic_chat's ``load_llm_config`` becomes a thin wrapper that does
the file I/O and converts ``ConfigError`` into a printed message
+ ``sys.exit(1)``, preserving the existing CLI behavior.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

REQUIRED_LLM_FIELDS = ("model", "base_url", "api_key")
DEFAULT_MAX_TOKENS = 150


class ConfigError(ValueError):
    """Raised when the config is structurally bad or has unresolved
    references. The message is intended to be shown directly to the
    user.
    """


def parse_llm_config(
    cfg: Any,
    *,
    env: Mapping[str, str] | None = None,
) -> dict:
    """Extract and validate the ``llm`` section of a config dict.

    ``cfg`` is whatever ``yaml.safe_load`` produced — typically a
    dict, but could be ``None`` (empty file) or any other type if
    the YAML is malformed at the top level.

    Returns a normalized dict suitable for passing to
    ``stream_chat_completion``. Always contains:
      ``model``      — string, required
      ``base_url``   — string, required (no trailing slash)
      ``api_key``    — string, required, ``${ENV}`` placeholders
                       resolved against ``env`` (default ``os.environ``)
      ``max_tokens`` — int, optional, defaults to 150
      Other keys from the input are passed through unchanged.

    Raises ``ConfigError`` on:
      - cfg is None or not a Mapping
      - missing ``llm`` section
      - missing or empty required fields
      - unresolved ``${ENV_VAR}`` placeholders in ``api_key``
    """
    if env is None:
        env = os.environ

    if cfg is None:
        raise ConfigError(
            "config.local.yaml is empty — add an 'llm' section "
            "with model / base_url / api_key"
        )
    if not isinstance(cfg, Mapping):
        raise ConfigError(
            f"top-level config must be a mapping, got {type(cfg).__name__}"
        )

    llm = cfg.get("llm")
    if llm is None:
        raise ConfigError(
            "config.local.yaml has no 'llm' section — add one with "
            "model / base_url / api_key"
        )
    if not isinstance(llm, Mapping):
        raise ConfigError(
            f"'llm' section must be a mapping, got {type(llm).__name__}"
        )

    out = dict(llm)

    for field in REQUIRED_LLM_FIELDS:
        if field not in out:
            raise ConfigError(
                f"'llm.{field}' is missing from config.local.yaml"
            )
        if not isinstance(out[field], str) or not out[field]:
            raise ConfigError(
                f"'llm.{field}' must be a non-empty string"
            )

    # Resolve ``${ENV_VAR}`` placeholders in api_key.
    api_key = out["api_key"]
    if api_key.startswith("${") and api_key.endswith("}"):
        var_name = api_key[2:-1]
        if not var_name:
            raise ConfigError(
                "'llm.api_key' has an empty ${} placeholder — give it a "
                "variable name like ${ANTHROPIC_API_KEY}"
            )
        resolved = env.get(var_name)
        if not resolved:
            raise ConfigError(
                f"environment variable {var_name!r} is not set "
                f"(referenced by 'llm.api_key' in config.local.yaml)"
            )
        out["api_key"] = resolved

    # Strip a trailing slash on base_url so URL joining works
    # consistently downstream.
    out["base_url"] = out["base_url"].rstrip("/")

    # Default max_tokens.
    out.setdefault("max_tokens", DEFAULT_MAX_TOKENS)

    return out


def parse_chat_config(cfg: Any) -> dict:
    """Extract the optional ``chat`` section of a config dict.

    Returns an empty dict if:
      - cfg is None / not a dict
      - cfg has no ``chat`` key
      - ``chat`` is None

    Never raises — the chat section is purely optional in the
    iter-011 / iter-017 design.
    """
    if not isinstance(cfg, Mapping):
        return {}
    chat = cfg.get("chat")
    if not isinstance(chat, Mapping):
        return {}
    return dict(chat)


# iter-020: VAD tuning config. The defaults match the
# ``_chat_recording`` module constants — same values that have
# served fine on a quiet desk mic. Users with noisier environments
# bump ``silence_threshold``; users who want faster turn-taking
# shorten ``silence_duration``.
VAD_DEFAULTS = {
    "silence_threshold": 0.02,
    "silence_duration": 0.8,
    "min_speech_duration": 0.3,
}


def parse_vad_config(chat_cfg: Any) -> dict:
    """Extract + validate the optional ``vad`` section of a parsed
    chat config (i.e. the dict returned by ``parse_chat_config``).

    Returns a dict with always-present keys
    (``silence_threshold``, ``silence_duration``,
    ``min_speech_duration``) backfilled from
    ``VAD_DEFAULTS`` for any missing/invalid entries.

    Tolerant — a malformed ``vad`` section (wrong type, bad value)
    falls back to defaults rather than raising. The reasoning:
    typo'd VAD config shouldn't take down the chat loop, just
    silently use the safe defaults. Users see misbehavior fast
    enough to debug.

    Caveat for callers writing tests: passing in a dict with
    out-of-range values (e.g. ``silence_threshold=10.0``) is
    accepted as-is. The parser only sanity-checks types and
    positivity; semantic validity is up to the caller / runtime.
    """
    out = dict(VAD_DEFAULTS)
    if not isinstance(chat_cfg, Mapping):
        return out
    vad = chat_cfg.get("vad")
    if not isinstance(vad, Mapping):
        return out
    for key, default in VAD_DEFAULTS.items():
        val = vad.get(key, default)
        if isinstance(val, (int, float)) and val > 0:
            out[key] = float(val)
        # else fall through to default — bad type / non-positive
        # number / missing key all hit the default.
    return out


# iter-034: filler-word config. iter-011 introduced fillers but
# the parsing lived inline in ``mic_chat.run_chat`` and was
# brittle:
#   - ``chat.fillers: "hi"`` (string instead of list) became
#     ``["h", "i"]`` because ``list("hi")`` iterates chars.
#   - ``chat.fillers_idle_threshold: "abc"`` crashed startup
#     with a ValueError from ``float(...)``.
#   - Non-string list items (numbers, dicts) reached the TTS
#     synth where they failed late with confusing errors.
# This parser is tolerant in the same shape as parse_vad_config:
# typo'd config silently falls back to defaults / drops bad
# items.
FILLER_DEFAULTS = {
    "texts": [],
    "idle_threshold": 0.6,
}


def parse_filler_config(chat_cfg: Any) -> dict:
    """Extract the optional filler config from a parsed chat config.

    Returns a dict with two keys:
      ``texts`` — list of strings to pre-render as filler clips.
        Empty list if config is missing / malformed.
      ``idle_threshold`` — seconds the worker waits before playing
        a filler. Default 0.6.

    Tolerant of malformed input:
      - ``fillers`` not a list → empty list (no fillers).
      - Non-string items in ``fillers`` → silently dropped from
        the output list. (Numbers, dicts, etc. would only fail
        later inside TTS with confusing errors.)
      - Empty / whitespace-only strings → dropped.
      - ``fillers_idle_threshold`` not a positive number →
        default 0.6.

    Mirrors the iter-020 parse_vad_config pattern.
    """
    out = {
        "texts": list(FILLER_DEFAULTS["texts"]),
        "idle_threshold": FILLER_DEFAULTS["idle_threshold"],
    }
    if not isinstance(chat_cfg, Mapping):
        return out

    raw_texts = chat_cfg.get("fillers")
    if isinstance(raw_texts, list):
        # Drop non-strings and empty / whitespace-only strings.
        cleaned = []
        for item in raw_texts:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    cleaned.append(stripped)
        out["texts"] = cleaned

    raw_threshold = chat_cfg.get("fillers_idle_threshold")
    if isinstance(raw_threshold, (int, float)) and raw_threshold > 0:
        out["idle_threshold"] = float(raw_threshold)

    return out
