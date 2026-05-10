"""
SessionNoteProcessor — Pipecat FrameProcessor that maintains live
session notes via Ollama tool use.

Receives TranscriptionFrames, calls Gemma 4 with tool schemas to
produce: verbatim transcript, cleaned transcript, running summary,
wiki entries, and turn-taking moment assessments.

Designed to run in a ParallelPipeline branch — never blocks the
main conversation pipeline.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import aiohttp

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

log = logging.getLogger("session-notes")

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "write_verbatim",
            "description": "Append the user's words exactly as transcribed, preserving filler words, false starts, and repetitions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Verbatim transcript of this chunk"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_clean",
            "description": "Append a cleaned, readable version of what the user said. Fix grammar, remove filler words, structure into sentences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Cleaned transcript"},
                    "correction": {"type": "boolean", "description": "True if this corrects a previous statement"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "Rewrite the running summary to incorporate this new chunk. The summary should be 100-300 words covering all themes discussed so far.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Complete updated summary"},
                    "themes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Active themes (3-7 short phrases)",
                    },
                },
                "required": ["summary", "themes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_update",
            "description": "Create or update a wiki page for a topic that emerged from the conversation. Only call this when a distinct topic warrants its own page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string", "description": "URL-safe page slug"},
                    "title": {"type": "string", "description": "Page title"},
                    "content": {"type": "string", "description": "Full page content in markdown"},
                },
                "required": ["slug", "title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_moment",
            "description": "Assess whether the system should respond right now. Default is stay_silent. Only recommend speaking when the user has clearly invited a response or reached a natural stopping point.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["stay_silent", "play_cue", "speak_brief", "speak_full"],
                        "description": "Recommended action",
                    },
                    "reason": {"type": "string", "description": "Why this action"},
                    "cue_type": {
                        "type": "string",
                        "enum": ["mhmm", "i_see", "go_on", "right", "tell_me_more"],
                        "description": "Which cue to play (only if action=play_cue)",
                    },
                },
                "required": ["action", "reason"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a background processor for a reflection/mindfulness app. The user is speaking aloud and you are processing their transcript in real time.

Your job is to maintain session notes using the provided tools. For each transcript chunk, you MUST call:
1. write_verbatim — the raw transcript as spoken
2. write_clean — a cleaned, readable version
3. summarize — update the running summary with this new information

Optionally call:
4. wiki_update — only when a distinct topic emerges that warrants its own page
5. assess_moment — evaluate whether the system should respond. Default is stay_silent. This is the USER's space to speak. Only recommend play_cue or speak when there is a clear invitation or a natural reflective pause.

Be concise. The summary should never exceed 300 words. Wiki pages should be brief (1-3 paragraphs)."""


class SessionNoteProcessor(FrameProcessor):
    def __init__(
        self,
        session_dir: str,
        ollama_url: str = "http://localhost:11434",
        model: str = "gemma4:e2b",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.session_dir = Path(session_dir)
        self.ollama_url = ollama_url
        self.model = model
        self.chunk_index = 0
        self.running_summary = ""
        self.active_themes: list[str] = []
        self._setup_session_dir()

    def _setup_session_dir(self):
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "wiki").mkdir(exist_ok=True)
        meta = {
            "started_at": datetime.now().isoformat(),
            "model": self.model,
            "chunks_processed": 0,
        }
        (self.session_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        for f in ["verbatim.md", "clean.md", "summary.md"]:
            path = self.session_dir / f
            if not path.exists():
                path.write_text("")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                self.chunk_index += 1
                asyncio.create_task(self._process_chunk(text, self.chunk_index))

        await self.push_frame(frame, direction)

    async def _process_chunk(self, text: str, chunk_idx: int):
        t0 = time.time()
        context = f"Chunk {chunk_idx}."
        if self.running_summary:
            context += f"\n\nRunning summary:\n{self.running_summary}"
        if self.active_themes:
            context += f"\n\nActive themes: {', '.join(self.active_themes)}"
        context += f"\n\nNew transcript chunk:\n{text}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "tools": TOOL_SCHEMAS,
                    "stream": False,
                }
                async with session.post(
                    f"{self.ollama_url}/api/chat", json=payload
                ) as resp:
                    result = await resp.json()

            tool_calls = result.get("message", {}).get("tool_calls", [])
            for tc in tool_calls:
                fn = tc["function"]["name"]
                args = tc["function"]["arguments"]
                await self._execute_tool(fn, args, chunk_idx)

            elapsed = time.time() - t0
            log.info("chunk %d: %d tool calls in %.1fs", chunk_idx, len(tool_calls), elapsed)

        except Exception as e:
            log.error("chunk %d failed: %s", chunk_idx, e)

    async def _execute_tool(self, name: str, args: dict, chunk_idx: int):
        if name == "write_verbatim":
            self._append_file("verbatim.md", f"\n<!-- chunk {chunk_idx} -->\n{args['text']}\n")
        elif name == "write_clean":
            prefix = "\n<!-- correction -->\n" if args.get("correction") else f"\n<!-- chunk {chunk_idx} -->\n"
            self._append_file("clean.md", f"{prefix}{args['text']}\n")
        elif name == "summarize":
            self.running_summary = args["summary"]
            self.active_themes = args.get("themes", [])
            (self.session_dir / "summary.md").write_text(
                f"# Session Summary\n\n{args['summary']}\n\n## Themes\n\n"
                + "\n".join(f"- {t}" for t in self.active_themes)
                + "\n"
            )
        elif name == "wiki_update":
            slug = args["slug"]
            wiki_path = self.session_dir / "wiki" / f"{slug}.md"
            wiki_path.write_text(f"# {args['title']}\n\n{args['content']}\n")
            log.info("wiki page: %s", slug)
        elif name == "assess_moment":
            moment = {"chunk": chunk_idx, "timestamp": time.time(), **args}
            self._append_file("moments.jsonl", json.dumps(moment) + "\n")
            if args.get("action") != "stay_silent":
                log.info("moment assessment: %s — %s", args["action"], args.get("reason", ""))

        log.debug("tool %s executed for chunk %d", name, chunk_idx)

    def _append_file(self, filename: str, content: str):
        with open(self.session_dir / filename, "a") as f:
            f.write(content)

    async def finalize(self) -> str | None:
        """Generate a closing summary for the session.

        Called when the session ends. Returns a closing message that
        can be displayed to the user or spoken via TTS.
        """
        if not self.running_summary:
            return None

        prompt = (
            "The reflection session is ending. Here is what was discussed:\n\n"
            f"{self.running_summary}\n\n"
            f"Themes: {', '.join(self.active_themes)}\n\n"
            "Write a brief, warm closing message (2-3 sentences). "
            "Summarize what was explored, offer one gentle takeaway thought, "
            "and end with an encouraging note. Do not give advice. "
            "Do not use the phrase 'I hear you' or 'that's valid.'"
        )

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a reflective companion closing a session."},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                }
                async with session.post(
                    f"{self.ollama_url}/api/chat", json=payload
                ) as resp:
                    result = await resp.json()

            closing = result.get("message", {}).get("content", "").strip()

            if closing:
                self._append_file("closing.md", f"# Closing\n\n{closing}\n")
                log.info("session closing generated: %s", closing[:80])

            return closing

        except Exception as e:
            log.error("finalize failed: %s", e)
            return None

    async def export_to_journal(self, journal_dir: str | None = None) -> str | None:
        """Export session themes to a local journal file.

        Opt-in only — never called automatically. Writes a timestamped
        markdown file with session summary, themes, and duration.
        Returns the file path, or None if nothing to export.
        """
        if not self.running_summary and not self.active_themes:
            return None

        if journal_dir is None:
            journal_dir = str(Path.home() / ".mindreflect" / "journal")

        journal_path = Path(journal_dir)
        journal_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = f"session-{timestamp}.md"
        filepath = journal_path / filename

        lines = [
            f"# Session — {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            "",
        ]

        if self.active_themes:
            lines.append("## Themes")
            lines.append("")
            for theme in self.active_themes:
                lines.append(f"- {theme}")
            lines.append("")

        if self.running_summary:
            lines.append("## Summary")
            lines.append("")
            lines.append(self.running_summary)
            lines.append("")

        lines.append(f"*{self.chunk_index} exchanges · Exported from MindReflect*")

        filepath.write_text("\n".join(lines))
        log.info("journal exported: %s", filepath)
        return str(filepath)

    def _update_meta(self):
        meta_path = self.session_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["chunks_processed"] = self.chunk_index
        meta["themes"] = self.active_themes
        meta["last_updated"] = datetime.now().isoformat()
        meta_path.write_text(json.dumps(meta, indent=2))
