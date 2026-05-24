"""Pure helpers for mic_chat.py — split out so they're testable without
pyaudio, mlx-whisper, or kokoro on the import path.

Keep this module dependency-free (stdlib only) and side-effect-free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def split_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """Split a streaming token buffer into (complete_sentences, remainder).

    A "complete" sentence is one terminated by . ! or ? followed by whitespace.
    The remainder is whatever trails the last terminator (the in-progress
    sentence that hasn't ended yet).

    Empty / whitespace-only sentences are dropped.

    Examples:
        >>> split_complete_sentences("Hello world. How are")
        (['Hello world.'], 'How are')
        >>> split_complete_sentences("One. Two! Three?")
        (['One.', 'Two!'], 'Three?')
        >>> split_complete_sentences("no terminator yet")
        ([], 'no terminator yet')
        >>> split_complete_sentences("")
        ([], '')
    """
    if not buffer:
        return [], ""
    parts = SENTENCE_END.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    complete = [p.strip() for p in parts[:-1] if p.strip()]
    remainder = parts[-1]
    return complete, remainder


def trim_history(messages: list[dict], max_user_assistant: int = 20) -> list[dict]:
    """Keep system prompt + last N non-system messages.

    Mirrors the trim logic in mic_chat.py so it can be unit-tested.
    Returns a new list (does not mutate input).
    """
    if not messages:
        return []
    if messages[0].get("role") == "system":
        head = [messages[0]]
        tail = messages[1:]
    else:
        head, tail = [], messages
    if len(tail) <= max_user_assistant:
        return head + tail
    return head + tail[-max_user_assistant:]


@dataclass
class TurnTimings:
    """Pure timing accumulator with a clear contract:

    - llm_start: monotonic time when the request was sent
    - llm_first_token_at: monotonic time of first streamed token (or None)
    - llm_stream_done_at: monotonic time when stream finished (or None)
    - tts_time: cumulative seconds spent in synthesis
    - playback_time: cumulative seconds spent writing to the speaker

    The crucial invariant: llm_total is computed from
    (llm_stream_done_at - llm_start), NOT from "now - llm_start". That fixes
    bug #2 (timer reading absurd values because it included playback time).
    """

    llm_start: float = 0.0
    llm_first_token_at: float | None = None
    llm_stream_done_at: float | None = None
    tts_time: float = 0.0
    playback_time: float = 0.0
    sentences_spoken: int = 0
    ttfs: float = 0.0
    speech_ended_at: float | None = None

    @property
    def llm_first_token(self) -> float:
        if self.llm_first_token_at is None:
            return 0.0
        return self.llm_first_token_at - self.llm_start

    @property
    def llm_total(self) -> float:
        """Total LLM stream duration. Returns 0 if stream never completed."""
        if self.llm_stream_done_at is None:
            return 0.0
        return self.llm_stream_done_at - self.llm_start

    def record_first_token(self, now: float) -> None:
        if self.llm_first_token_at is None:
            self.llm_first_token_at = now

    def record_stream_done(self, now: float) -> None:
        self.llm_stream_done_at = now

    def record_ttfs(self, now: float) -> bool:
        """Record TTFS (time-to-first-speech) once. Returns True if recorded."""
        if self.ttfs > 0 or self.speech_ended_at is None:
            return False
        self.ttfs = now - self.speech_ended_at
        return True


def flush_pending_audio(stream, chunk_size: int = 1024, max_iterations: int = 1024) -> int:
    """Drain any audio frames currently buffered in `stream`, non-blocking.

    `stream` must expose:
        - get_read_available() -> int  (number of frames waiting)
        - read(n_frames, exception_on_overflow: bool = False) -> bytes

    PyAudio's input stream satisfies both. The fake-stream tests in
    tests/unit/test_chat_helpers.py mimic the same shape.

    Returns the total number of frames drained. Stops when:
      - get_read_available() reports fewer than chunk_size frames, OR
      - max_iterations is reached (safety cap so a misbehaving stream
        can't trap us in an infinite drain loop).

    This is the fix for bug #3: when the LLM call fails after a long
    timeout, the mic stream has been silently filling. Without a flush,
    the next call to record_utterance_streaming() reads that backlog
    immediately, triggers the VAD as "speech," and feeds garbage into
    STT.
    """
    drained = 0
    for _ in range(max_iterations):
        try:
            available = stream.get_read_available()
        except Exception:
            # If we can't even ask the stream how much it has buffered,
            # there's nothing safe to do here — just stop.
            break
        if available < chunk_size:
            break
        try:
            stream.read(chunk_size, exception_on_overflow=False)
        except Exception:
            break
        drained += chunk_size
    return drained


def format_preview_line(text: str, max_width: int = 80, prefix_len: int = 7) -> str:
    """Truncate a live STT preview so it fits on one terminal row.

    `prefix_len` accounts for the visible "  You: " prefix (sans ANSI).
    Returns the (possibly truncated) text. If truncated, ends with an ellipsis.

    This is the fix for bug #4 (wrap-on-reprint) — by ensuring preview never
    exceeds available width, the `\\r` rewrite stays on a single row.
    """
    available = max(10, max_width - prefix_len)
    if len(text) <= available:
        return text
    return text[: available - 1] + "…"
