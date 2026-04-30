import os
from pathlib import Path
from copy import deepcopy

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"
_USER_CONFIG_PATH = Path.home() / ".geno-tools" / "geno-voice" / "config.yaml"

_ENV_MAP = {
    ("server", "host"): "GENO_VOICE_HOST",
    ("server", "port"): "GENO_VOICE_PORT",
    ("stt", "engine"): "GENO_VOICE_STT_ENGINE",
    ("tts", "voice"): "GENO_VOICE_TTS_VOICE",
}

_cache = None


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def _apply_env(cfg: dict) -> dict:
    for path, env_var in _ENV_MAP.items():
        val = os.environ.get(env_var)
        if val is None:
            continue
        d = cfg
        for key in path[:-1]:
            d = d.setdefault(key, {})
        leaf = path[-1]
        existing = d.get(leaf)
        if isinstance(existing, int):
            val = int(val)
        elif isinstance(existing, float):
            val = float(val)
        d[leaf] = val
    return cfg


def load_config() -> dict:
    global _cache
    if _cache is not None:
        return _cache

    with open(_DEFAULT_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    if _USER_CONFIG_PATH.exists():
        with open(_USER_CONFIG_PATH) as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)

    cfg = _apply_env(cfg)
    _cache = cfg
    return cfg


def get(section: str, key: str | None = None, default=None):
    cfg = load_config()
    s = cfg.get(section, {})
    if key is None:
        return s
    return s.get(key, default)


def update(updates: dict) -> dict:
    global _cache
    cfg = load_config()
    cfg = _deep_merge(cfg, updates)
    _cache = cfg

    _USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_USER_CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return cfg


def reload():
    global _cache
    _cache = None
    return load_config()
