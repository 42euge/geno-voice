#!/usr/bin/env python3
"""
Test SessionNoteProcessor with simulated transcript frames.

Feeds fake TranscriptionFrames to the processor and verifies Ollama
tool calls produce session notes. No mic needed — tests the LLM
tool-use pipeline independently.

Usage:
    cd /Users/euge/code-red/mind-reflect-ws/geno-voice
    .venv/bin/python examples/session_notes_test.py

Requires Ollama running with gemma4:e4b pulled.
"""

import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipecat.frames.frames import StartFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection

from session.notes import SessionNoteProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("test")

SAMPLE_CHUNKS = [
    "I've been thinking a lot about work lately. It's been really stressful, you know? "
    "My manager keeps piling on projects and I don't feel like I can say no.",

    "And like, it's not just the workload. It's the feeling that nothing I do is ever enough. "
    "I finished that big report last week and didn't even get a thank you.",

    "I don't know, maybe I'm overthinking it. But it's been keeping me up at night. "
    "I just lie there thinking about all the things I should have done differently.",

    "What do you think? Am I being unreasonable here?",
]


async def main():
    session_dir = tempfile.mkdtemp(prefix="mindreflect-test-")
    log.info("Session directory: %s", session_dir)

    processor = SessionNoteProcessor(
        session_dir=session_dir,
        model="gemma4:e4b",
    )

    print("=" * 60)
    print("  SessionNoteProcessor Test")
    print(f"  Output: {session_dir}")
    print("=" * 60)

    await processor.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)

    for i, chunk in enumerate(SAMPLE_CHUNKS):
        print(f"\n--- Chunk {i + 1}/{len(SAMPLE_CHUNKS)} ---")
        print(f"  \"{chunk[:80]}...\"" if len(chunk) > 80 else f"  \"{chunk}\"")

        frame = TranscriptionFrame(
            text=chunk,
            user_id="test-user",
            timestamp=str(i),
        )
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        await asyncio.sleep(20)

    processor._update_meta()

    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)

    for filename in ["verbatim.md", "clean.md", "summary.md", "moments.jsonl", "meta.json"]:
        path = Path(session_dir) / filename
        if path.exists() and path.stat().st_size > 0:
            content = path.read_text()
            print(f"\n--- {filename} ({len(content)} bytes) ---")
            print(content[:500] + ("..." if len(content) > 500 else ""))
        else:
            print(f"\n--- {filename}: EMPTY or MISSING ---")

    wiki_dir = Path(session_dir) / "wiki"
    wiki_pages = list(wiki_dir.glob("*.md"))
    if wiki_pages:
        print(f"\n--- wiki/ ({len(wiki_pages)} pages) ---")
        for p in wiki_pages:
            print(f"  {p.name} ({p.stat().st_size} bytes)")
    else:
        print("\n--- wiki/: no pages created ---")

    print(f"\nFull output at: {session_dir}")


if __name__ == "__main__":
    asyncio.run(main())
