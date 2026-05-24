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
