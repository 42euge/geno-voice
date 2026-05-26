"""Tests for iter-112 — _yield_tokens context_factor knob.

The perf suite's _yield_tokens factory is the seam through which
all LLM-stub timing flows. iter-100 added stall_after/stall_seconds;
iter-112 adds context_factor for KV-fill TTFB scaling.

Tested as a pure-function unit test so the cost-of-running the
perf suite isn't paid here.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# _yield_tokens is module-level in the perf test file.
from tests.performance.test_pipeline_perf import _yield_tokens  # noqa: E402


def _consume(factory, messages, llm_config=None) -> list[str]:
    """Drive the factory's generator and return tokens emitted."""
    if llm_config is None:
        llm_config = {"model": "stub"}
    return list(factory(messages, llm_config))


# ---- Default behavior (context_factor=0) ---------------------------------


def test_default_yields_tokens_immediately():
    """No context_factor + no per_token_delay → tokens yield as
    fast as Python can produce them."""
    factory = _yield_tokens("hello world")
    t0 = time.monotonic()
    tokens = _consume(factory, [])
    elapsed = time.monotonic() - t0
    # Two words + one delimiter (the trailing space).
    assert any("hello" in t for t in tokens)
    assert any("world" in t for t in tokens)
    # Must be fast — no sleep was invoked.
    assert elapsed < 0.05


def test_default_ignores_messages_argument():
    """When context_factor=0, the messages list is irrelevant —
    the factory doesn't introspect it."""
    factory = _yield_tokens("ok")
    t0 = time.monotonic()
    # Pass a deliberately huge messages list; should not slow down.
    big_messages = [
        {"role": "user", "content": "a" * 10000} for _ in range(20)
    ]
    _consume(factory, big_messages)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.05


# ---- context_factor behavior ---------------------------------------------


def test_context_factor_delays_first_token_by_total_chars():
    """context_factor=0.001 + 100 chars of messages → ≥100ms
    delay before the first token yields."""
    messages = [{"role": "user", "content": "a" * 100}]
    factory = _yield_tokens("ok", context_factor=0.001)

    t0 = time.monotonic()
    gen = factory(messages, {"model": "stub"})
    first_token = next(gen)
    elapsed = time.monotonic() - t0

    assert "ok" in first_token
    # 100 chars × 0.001 = 0.1s minimum.
    assert elapsed >= 0.09  # permissive lower bound for sleep granularity


def test_context_factor_sums_across_messages():
    """Total chars is the sum across ALL messages — bounded by
    cap=N for cap-trim turns, full history otherwise."""
    messages = [
        {"role": "system", "content": "x" * 50},
        {"role": "user", "content": "y" * 50},
    ]
    factory = _yield_tokens("ok", context_factor=0.001)

    t0 = time.monotonic()
    next(factory(messages, {}))
    elapsed = time.monotonic() - t0
    # 100 chars total → 100ms delay.
    assert elapsed >= 0.09


def test_context_factor_zero_disables_delay_even_with_messages():
    """context_factor=0 explicitly → no delay, regardless of
    message size. This is the default and protects existing
    iter-098–102 scenarios from accidentally regressing."""
    messages = [{"role": "user", "content": "z" * 5000}]
    factory = _yield_tokens("ok", context_factor=0.0)

    t0 = time.monotonic()
    list(factory(messages, {}))
    elapsed = time.monotonic() - t0
    assert elapsed < 0.05


def test_context_factor_handles_missing_content():
    """Defensive: a malformed message dict (no 'content' key)
    becomes an empty string, contributing 0 chars. No KeyError."""
    messages = [
        {"role": "user", "content": "real"},
        {"role": "system"},  # no content key
    ]
    factory = _yield_tokens("ok", context_factor=0.001)
    # Should not raise.
    _consume(factory, messages)


def test_context_factor_combines_with_per_token_delay():
    """Both delays apply: context_factor at start, per_token_delay
    between each token."""
    messages = [{"role": "user", "content": "a" * 50}]
    factory = _yield_tokens(
        "one two", per_token_delay=0.02, context_factor=0.001,
    )

    t0 = time.monotonic()
    list(factory(messages, {}))
    elapsed = time.monotonic() - t0
    # context_factor: 50 × 0.001 = 50ms
    # per_token_delay: ~3 tokens × 20ms = 60ms (depends on regex)
    # Total ≥ 100ms — permissive lower bound.
    assert elapsed >= 0.09


# ---- Backwards compat ----------------------------------------------------


def test_old_kwarg_combinations_still_work():
    """All pre-iter-112 callers used positional-only or
    per_token_delay/stall_after. Make sure those signatures still
    work without the new kwarg."""
    factory = _yield_tokens("ok", per_token_delay=0.0)
    tokens = _consume(factory, [])
    assert any("ok" in t for t in tokens)

    factory2 = _yield_tokens(
        "ok",
        per_token_delay=0.0,
        stall_after="moment",
        stall_seconds=0.0,
    )
    tokens2 = _consume(factory2, [])
    assert any("ok" in t for t in tokens2)
