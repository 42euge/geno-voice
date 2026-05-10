#!/usr/bin/env python3
"""
Benchmark SessionNoteProcessor latency across models and tool configs.

Tests:
  1. gemma4:e4b — all 5 tools (current)
  2. gemma4:e2b — all 5 tools
  3. gemma4:e2b — fast path only (verbatim + assess_moment)

Usage:
    cd /Users/euge/code-red/mind-reflect-ws/geno-voice
    .venv/bin/python examples/notes_latency_bench.py
"""

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from session.notes import SYSTEM_PROMPT, TOOL_SCHEMAS

CHUNK = (
    "I've been thinking a lot about work lately. It's been really stressful, you know? "
    "My manager keeps piling on projects and I don't feel like I can say no."
)

FAST_TOOLS = [t for t in TOOL_SCHEMAS if t["function"]["name"] in ("write_verbatim", "assess_moment")]

FAST_PROMPT = """You are a background processor for a reflection app. For each transcript chunk, call:
1. write_verbatim — the raw transcript as spoken
2. assess_moment — should the system respond? Default is stay_silent."""


async def bench_one(model: str, tools: list, system: str, label: str):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Chunk 1.\n\nNew transcript chunk:\n{CHUNK}"},
    ]
    payload = {"model": model, "messages": messages, "tools": tools, "stream": False}

    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post("http://127.0.0.1:11434/api/chat", json=payload) as resp:
            result = await resp.json()
    elapsed = time.time() - t0

    tool_calls = result.get("message", {}).get("tool_calls", [])
    tool_names = [tc["function"]["name"] for tc in tool_calls]

    print(f"  {label}")
    print(f"    Time: {elapsed:.1f}s")
    print(f"    Tools called: {', '.join(tool_names) or 'none'}")
    print()
    return elapsed, tool_names


async def main():
    print("=" * 60)
    print("  Session Notes Latency Benchmark")
    print("=" * 60)
    print()

    # Warm up models
    for model in ["gemma4:e4b", "gemma4:e2b"]:
        print(f"Warming up {model}...")
        async with aiohttp.ClientSession() as session:
            payload = {"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": False}
            async with session.post("http://127.0.0.1:11434/api/chat", json=payload) as resp:
                await resp.json()
        print(f"  {model} warm")

    print()
    results = []

    # Test 1: e4b, all tools
    e, t = await bench_one("gemma4:e4b", TOOL_SCHEMAS, SYSTEM_PROMPT, "gemma4:e4b — all 5 tools")
    results.append(("e4b-all", e, t))

    # Test 2: e2b, all tools
    e, t = await bench_one("gemma4:e2b", TOOL_SCHEMAS, SYSTEM_PROMPT, "gemma4:e2b — all 5 tools")
    results.append(("e2b-all", e, t))

    # Test 3: e2b, fast path
    e, t = await bench_one("gemma4:e2b", FAST_TOOLS, FAST_PROMPT, "gemma4:e2b — fast path (2 tools)")
    results.append(("e2b-fast", e, t))

    # Test 4: e4b, fast path
    e, t = await bench_one("gemma4:e4b", FAST_TOOLS, FAST_PROMPT, "gemma4:e4b — fast path (2 tools)")
    results.append(("e4b-fast", e, t))

    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print()
    print(f"  {'Config':<25} {'Time':>8}  {'Tools':>5}  {'Viable for RT?'}")
    print(f"  {'-'*25} {'-'*8}  {'-'*5}  {'-'*14}")
    for label, elapsed, tools in results:
        viable = "YES" if elapsed < 10 else ("MAYBE" if elapsed < 15 else "NO")
        print(f"  {label:<25} {elapsed:>7.1f}s  {len(tools):>5}  {viable}")


if __name__ == "__main__":
    asyncio.run(main())
