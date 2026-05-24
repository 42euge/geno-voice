"""LLM streaming pieces — pure SSE parsing pulled out of mic_chat so
it's testable without touching the network.

The previous ``llm_stream`` function did three things at once:
  1. Build the HTTP request to the OpenAI-compatible endpoint
  2. Open the streaming response
  3. Parse the SSE-formatted token chunks

The parsing was the only part that ever needed unit testing, but
because it was tangled with ``requests.post``, you couldn't exercise
it without mocking the HTTP transport.

This module hosts:
  - ``parse_sse_token_stream(lines)`` — pure parser, takes any
    iterable of lines (bytes or str), yields content tokens.
  - ``stream_chat_completion(messages, config)`` — thin generator
    that does the HTTP request and runs the parser, with a
    ``try/finally`` that closes the response when the generator is
    closed. iter-013 barge-in completion: calling
    ``generator.close()`` from outside (typically via
    ``BargeInCoordinator.on_trigger``) releases the upstream TCP
    connection promptly instead of dangling.
"""

from __future__ import annotations

import json
from typing import Iterable, Iterator


def parse_sse_token_stream(lines: Iterable) -> Iterator[str]:
    """Parse OpenAI-compatible SSE response lines, yielding content tokens.

    Behavior:
      - Empty lines are skipped (SSE keep-alive / comment markers).
      - Lines that don't start with ``"data: "`` are skipped (e.g.
        comment lines starting with ``":"``, event-type markers).
      - The ``"[DONE]"`` sentinel ends iteration cleanly.
      - Malformed JSON, missing ``choices``/``delta``/``content``
        fields, and other parser hiccups are silently skipped so a
        single bad chunk doesn't kill the whole stream.
      - Bytes input is UTF-8-decoded with ``errors="replace"`` to
        avoid raising on the rare malformed multibyte sequence.
      - Empty content (``""``) is also skipped — only non-empty
        token deltas are yielded.

    The function is a pure generator: no I/O, no side effects.
    Tests drive it with synthetic line iterables.
    """
    for line in lines:
        if not line:
            continue
        if isinstance(line, (bytes, bytearray)):
            line = line.decode("utf-8", errors="replace")
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            continue
        if token:
            yield token


def stream_chat_completion(messages: list[dict], config: dict) -> Iterator[str]:
    """Open an OpenAI-compatible streaming chat completion and yield
    content tokens.

    The function is a generator. The caller can stop early by either:
      - Breaking out of the for-loop normally, or
      - Calling ``generator.close()`` (e.g. via
        ``BargeInCoordinator.on_trigger``).

    In both cases the underlying ``requests.Response`` is closed in
    the ``finally`` block, releasing the upstream TCP connection.
    Without this, a barge-in left the connection hanging until the
    generator got garbage collected.
    """
    import requests

    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config["model"],
        "messages": messages,
        "max_tokens": config.get("max_tokens", 150),
        "stream": True,
    }
    resp = requests.post(
        f"{config['base_url']}/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
        stream=True,
    )
    resp.raise_for_status()
    try:
        yield from parse_sse_token_stream(resp.iter_lines())
    finally:
        try:
            resp.close()
        except Exception:
            pass
