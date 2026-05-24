"""Pure helpers for mic_chat.py — split out so they're testable without
pyaudio, mlx-whisper, or kokoro on the import path.

Keep this module dependency-free (stdlib only) and side-effect-free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

# Common abbreviations that end with a period but should NOT terminate
# a sentence in voice context. Lowercased; the splitter checks the
# preceding word (also lowercased) against this set. iter-016.
#
# Coverage is single-word abbreviations only ("mr.", "dr.", "etc."),
# which catches the vast majority of real cases. Multi-period
# abbreviations like "i.e." and "U.S.A." are handled too because we
# look at the substring of letters/dots immediately preceding the
# period, so "I.e" and "U.S.A" both match against the lowercased
# entries "i.e" and "u.s.a" in the set.
NON_TERMINATING_ABBREVIATIONS = frozenset({
    # Titles
    "mr", "mrs", "ms", "dr", "prof", "rev", "fr", "sr", "jr", "st",
    # Latin abbreviations
    "etc", "i.e", "e.g", "vs", "cf", "viz",
    # Business / legal
    "inc", "ltd", "corp", "co", "llc", "lp",
    # Academic / professional titles
    "ph.d", "m.d", "b.a", "m.a", "b.s", "m.s", "esq",
    # Geographic / address
    "u.s", "u.k", "u.s.a", "ave", "blvd", "rd", "ln", "ct", "pl",
    "mt", "mts", "ft", "n", "s", "e", "w", "ne", "nw", "se", "sw",
    # Months / days
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri",
    "sat", "sun",
    # Time of day (iter-017): "9:30 a.m. Hi." should not split.
    "a.m", "p.m",
    # Misc
    "no", "nos", "vol", "p", "pp", "fig", "figs", "approx", "incl",
    "min", "mins", "max", "sec", "secs", "hr", "hrs",
})


class VadEvent(str, Enum):
    """One-frame outcome for the VAD state machine.

    IDLE — still waiting for speech to start; caller can drop the frame.
    ACTIVE — actively recording; caller should append the frame to its buffer.
    DONE_OK — utterance ended with enough speech; caller should return frames.
    DONE_TOO_SHORT — utterance ended but speech was below min_speech_duration;
        caller should reset and keep listening (often a cough or click).
    """

    IDLE = "idle"
    ACTIVE = "active"
    DONE_OK = "done_ok"
    DONE_TOO_SHORT = "done_too_short"


@dataclass
class VadState:
    """Pure VAD state machine — RMS in, event out, no I/O.

    Mirrors the speaking/silence tracking that used to be inlined in
    record_utterance_streaming() but is now testable without pyaudio.

    Behavior:
      - level > silence_threshold → ACTIVE (and starts speech timer if new)
      - level <= threshold while speaking:
        - first such frame starts the silence timer
        - if silence persists ≥ silence_duration:
            * total speech_duration ≥ min_speech_duration → DONE_OK
            * otherwise → DONE_TOO_SHORT
      - level <= threshold while not speaking → IDLE (do nothing)

    speech_duration excludes the trailing silence_duration window, matching
    the original calculation in mic_chat.py.
    """

    silence_threshold: float = 0.02
    silence_duration: float = 0.8
    min_speech_duration: float = 0.3

    speaking: bool = False
    speech_start: float | None = None
    silence_start: float | None = None
    last_speech_duration: float = 0.0

    def feed(self, level: float, now: float) -> VadEvent:
        if level > self.silence_threshold:
            if not self.speaking:
                self.speaking = True
                self.speech_start = now
            self.silence_start = None
            return VadEvent.ACTIVE
        # Below threshold below this point.
        if not self.speaking:
            return VadEvent.IDLE
        # Speaking but quiet → trailing silence.
        if self.silence_start is None:
            self.silence_start = now
            return VadEvent.ACTIVE
        if now - self.silence_start >= self.silence_duration:
            assert self.speech_start is not None  # set when speaking flipped on
            total = now - self.speech_start - self.silence_duration
            self.last_speech_duration = max(0.0, total)
            event = (
                VadEvent.DONE_OK
                if total >= self.min_speech_duration
                else VadEvent.DONE_TOO_SHORT
            )
            self.reset()
            return event
        return VadEvent.ACTIVE

    def reset(self) -> None:
        self.speaking = False
        self.speech_start = None
        self.silence_start = None


def _word_before_period(buffer: str, period_idx: int) -> str:
    """Return the lowercase substring of letters and inner periods
    immediately before ``buffer[period_idx]`` (which must itself be
    a period).

    Used to detect non-terminating abbreviations. Walking back over
    ``[a-zA-Z.]`` lets us recognize multi-period forms like
    ``i.e.`` or ``U.S.A.`` where the relevant token is more than
    just letters.

    iter-021: numeric ordinals get special-cased to empty string.
    Without this, ``1st.`` walks back over ``st`` (skipping the
    digit) and matches ``st`` in the abbreviation set (which is
    the Street abbreviation), so ``"He came 1st. Then we go."``
    wouldn't split. ``3rd.`` has the same problem (rd = Road).
    Detecting "digit immediately precedes the alpha sequence"
    catches all ordinals (1st/2nd/3rd/4th/...) without needing a
    separate ordinal regex.
    """
    end = period_idx
    start = end
    while start > 0 and (buffer[start - 1].isalpha() or buffer[start - 1] == "."):
        start -= 1
    # If a digit immediately precedes the alpha sequence we just
    # walked over, this is a numeric ordinal (1st, 2nd, ...) or
    # similar digit-prefixed form — NOT an abbreviation.
    if start < end and start > 0 and buffer[start - 1].isdigit():
        return ""
    # Trim a stray leading dot if any (e.g. " .e.g." would otherwise
    # produce a leading "." that doesn't match anything useful).
    word = buffer[start:end].lower().lstrip(".")
    return word


def split_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """Split a streaming token buffer into (complete_sentences, remainder).

    A "complete" sentence is one terminated by . ! or ? followed by
    whitespace. The remainder is whatever trails the last terminator
    (the in-progress sentence that hasn't ended yet).

    Common abbreviations (Mr., Dr., etc., i.e., e.g., U.S.A., …) are
    recognized via ``NON_TERMINATING_ABBREVIATIONS`` and do not split
    the sentence. iter-016.

    Empty / whitespace-only sentences are dropped.

    Examples:
        >>> split_complete_sentences("Hello world. How are")
        (['Hello world.'], 'How are')
        >>> split_complete_sentences("One. Two! Three?")
        (['One.', 'Two!'], 'Three?')
        >>> split_complete_sentences("Mr. Smith arrived. Hello.")
        (['Mr. Smith arrived.'], 'Hello.')
        >>> split_complete_sentences("no terminator yet")
        ([], 'no terminator yet')
        >>> split_complete_sentences("")
        ([], '')
    """
    if not buffer:
        return [], ""

    matches = list(SENTENCE_END.finditer(buffer))
    if not matches:
        return [], buffer

    real_splits = []
    for m in matches:
        # The terminator character is at m.start() - 1 (the regex
        # uses lookbehind so the match itself is the whitespace).
        # Only check abbreviation status if it's a period — `!` and
        # `?` always terminate.
        terminator_idx = m.start() - 1
        if buffer[terminator_idx] == ".":
            word = _word_before_period(buffer, terminator_idx)
            if word in NON_TERMINATING_ABBREVIATIONS:
                continue
        real_splits.append(m)

    if not real_splits:
        return [], buffer

    segments: list[str] = []
    last_end = 0
    for m in real_splits:
        segments.append(buffer[last_end:m.start()])
        last_end = m.end()
    segments.append(buffer[last_end:])

    complete = [s.strip() for s in segments[:-1] if s.strip()]
    remainder = segments[-1]
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


# ANSI helpers used by render_preview. Kept local so this module stays
# stdlib-only and matches the constants in mic_chat.py without importing.
_ANSI_DIM = "\033[2m"
_ANSI_RESET = "\033[0m"
_ANSI_CLEAR_LINE = "\033[2K"


def render_preview(
    text: str,
    *,
    max_width: int,
    prefix: str = "  You: ",
    file=None,
    dim: bool = True,
) -> str:
    """Write a single-line live STT preview to `file` (default sys.stdout).

    Uses `\\r\\033[2K` to overwrite the current row, then writes the prefix
    plus a width-truncated version of `text` so it cannot wrap. This is the
    fix for bug #4 — the original code did `\\r{CLEAR_LINE}  You: {text}`
    with no length cap, so any text wider than the terminal pushed onto a
    second row and broke the rewrite on the next iteration.

    Returns the exact string written, primarily so tests can assert on it
    without needing to capture file I/O.

    `prefix` defaults to the user-facing "  You: " label. Pass a different
    prefix (e.g. "  Bot: ") to reuse this for other live-update lines.
    `dim` wraps the body in ANSI dim; set False for plain output.
    """
    import sys as _sys
    out = file if file is not None else _sys.stdout
    visible_prefix_len = len(prefix)
    body = format_preview_line(text, max_width=max_width, prefix_len=visible_prefix_len)
    if dim:
        line = f"\r{_ANSI_CLEAR_LINE}{prefix}{_ANSI_DIM}{body}{_ANSI_RESET}"
    else:
        line = f"\r{_ANSI_CLEAR_LINE}{prefix}{body}"
    out.write(line)
    out.flush()
    return line
