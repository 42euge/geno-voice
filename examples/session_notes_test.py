#!/usr/bin/env python3
"""
Test SessionNoteProcessor with simulated transcript frames.

Feeds fake TranscriptionFrames through a real Pipecat pipeline.
No mic needed — tests the LLM tool-use pipeline independently.

Usage:
    cd /Users/euge/code-red/rest-reflect-ws/geno-voice
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

from pipecat.frames.frames import EndFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

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


class ChunkFeeder(FrameProcessor):
    """Feeds simulated transcript chunks into the pipeline with delays."""

    def __init__(self, chunks: list[str], delay: float = 15.0):
        super().__init__()
        self._chunks = chunks
        self._delay = delay

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def run(self, task: PipelineTask):
        for i, chunk in enumerate(self._chunks):
            print(f"\n--- Chunk {i + 1}/{len(self._chunks)} ---")
            print(f"  \"{chunk[:80]}{'...' if len(chunk) > 80 else ''}\"")

            frame = TranscriptionFrame(
                text=chunk,
                user_id="test-user",
                timestamp=str(i),
            )
            await task.queue_frame(frame)

            if i < len(self._chunks) - 1:
                await asyncio.sleep(self._delay)
            else:
                await asyncio.sleep(self._delay)
                await task.queue_frame(EndFrame())


async def main():
    session_dir = tempfile.mkdtemp(prefix="restreflect-test-")
    log.info("Session directory: %s", session_dir)

    processor = SessionNoteProcessor(
        session_dir=session_dir,
        model="gemma4:e4b",
    )

    feeder = ChunkFeeder(SAMPLE_CHUNKS, delay=15.0)

    pipeline = Pipeline([feeder, processor])
    runner = PipelineRunner()
    task = PipelineTask(pipeline, params=PipelineParams())

    print("=" * 60)
    print("  SessionNoteProcessor Test")
    print(f"  Output: {session_dir}")
    print(f"  Chunks: {len(SAMPLE_CHUNKS)}, delay: 15s each")
    print("=" * 60)

    runner_task = asyncio.create_task(runner.run(task))
    await asyncio.sleep(2)
    await feeder.run(task)
    await runner_task

    processor._update_meta()

    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)

    for filename in ["verbatim.md", "clean.md", "summary.md", "moments.jsonl", "meta.json"]:
        path = Path(session_dir) / filename
        if path.exists() and path.stat().st_size > 0:
            content = path.read_text()
            print(f"\n--- {filename} ({len(content)} bytes) ---")
            lines = content.strip().split("\n")
            if len(lines) > 20:
                print("\n".join(lines[:20]))
                print(f"  ... ({len(lines) - 20} more lines)")
            else:
                print(content)
        else:
            print(f"\n--- {filename}: EMPTY ---")

    wiki_dir = Path(session_dir) / "wiki"
    wiki_pages = list(wiki_dir.glob("*.md"))
    if wiki_pages:
        print(f"\n--- wiki/ ({len(wiki_pages)} pages) ---")
        for p in wiki_pages:
            print(f"  {p.name}: {p.read_text()[:200]}")
    else:
        print("\n--- wiki/: no pages ---")

    print(f"\nFull output: {session_dir}")


if __name__ == "__main__":
    asyncio.run(main())
