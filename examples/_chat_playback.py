"""Playback loop extracted from mic_chat.py — pyaudio-free at module
scope.

play_aligned() used to call ``pa.open(...)`` to mint a fresh PyAudio
output stream per sentence, write into it, then close. Two problems:

  - It was unimportable on hosts without pyaudio (most of CI), so the
    token-reveal logic and the byte-write contract were untested.
  - Opening per-sentence costs a few ms each time and prevents
    streaming overlap (iter-008) from holding a single persistent
    speaker.

This module hosts the same play loop with a different contract: the
caller passes in a speaker-shaped stream (anything with ``.write(bytes)``
will do — real PyAudio output stream or
``examples.virtual_audio.VirtualSpeakerStream``). The caller manages
open/close. mic_chat.py does open-per-sentence around the call to
preserve current behavior; iter-008 will switch to a persistent stream.

Injection points (same pattern as iter-006):
  ``output`` — file-like for the token-reveal text.
  ``clock`` — monotonic-clock callable for tests.
"""

from __future__ import annotations

import sys
import time
from typing import Callable

import numpy as np

TTS_RATE = 24000
DEFAULT_PLAY_CHUNK = 1024  # ~42ms at 24kHz

# Inline ANSI codes — duplicated from mic_chat.py constants on purpose so
# this module remains a clean leaf with no dependency on mic_chat itself.
_BOLD = "\033[1m"
_RESET = "\033[0m"
_CYAN = "\033[36m"
CLEAR_LINE = "\033[2K"

_PUNCT_CHARS = frozenset('.,!?;:')


def _is_punct_only(token_text: str) -> bool:
    s = token_text.strip()
    return bool(s) and all(c in _PUNCT_CHARS for c in s)


def _emit_token(output, word: str, *, bold: bool, flush: bool) -> None:
    """Write a single token to `output`, attaching to the previous word
    if it's pure punctuation (via backspace), or printing as a bold
    standalone word otherwise.

    `bold` toggles ANSI bold/reset wrappers — the original code only
    bolded inside the playback loop and emitted plain text in the
    post-loop flush. We preserve that quirk here (callers pass
    bold=False for the post-loop case) so behavior is identical.
    Empty / whitespace-only tokens are skipped.
    """
    s = word.strip()
    if not s:
        return
    if _is_punct_only(word):
        output.write(f"\b{word} ")
    else:
        if bold:
            output.write(f"{_BOLD}{word}{_RESET} ")
        else:
            output.write(f"{word} ")
    if flush:
        output.flush()


def play_aligned(
    speaker_stream,
    audio_np: np.ndarray,
    tokens: list[dict],
    *,
    is_first_sentence: bool = False,
    output=None,
    clock: Callable[[], float] = time.monotonic,
    play_chunk: int = DEFAULT_PLAY_CHUNK,
    rate: int = TTS_RATE,
    cancel_event=None,
    lag_out: dict | None = None,
) -> float:
    """Stream `audio_np` into `speaker_stream` chunk-by-chunk and reveal
    `tokens` in real-time as playback advances.

    `speaker_stream` only needs ``.write(bytes)``. PyAudio's output
    stream and ``VirtualSpeakerStream`` both qualify. Lifecycle
    (open/close) is the caller's responsibility.

    `tokens` is a list of dicts with at least a ``"text"`` field and a
    ``"start"`` field measured in seconds from the start of this audio
    blob. Tokens whose ``start`` falls before the current playback
    position get printed when their chunk is written; any tokens whose
    start exceeds the audio duration (rare, but possible) get flushed
    after the loop.

    `cancel_event` is an optional ``threading.Event``-shaped object
    (anything with ``.is_set()``). When set, the play loop breaks
    between chunks — the current chunk in flight finishes writing,
    but no further chunks are queued. This is the iter-009 barge-in
    primitive: a watcher on the mic side can flip the flag the
    instant it detects user speech, and the worker stops mid-sentence.

    `lag_out` (iter-071) is an optional dict that, if provided, gets
    populated with per-call token-reveal lag statistics:
        ``"sum"``    — total lag (seconds) summed across emitted tokens
        ``"count"``  — number of tokens with a lag observation
        ``"max"``    — single-token worst-case lag (seconds)
    Lag is ``(clock_at_emit - t0) - token["start"]``. Positive means
    text was emitted AFTER the audio second it claims to align with;
    negative means text led audio (the bot got "spoiled"). Production
    play_aligned in mic_chat.py wires this through SentenceWorker so
    the per-turn mean ends up on TurnMetrics. Tests can call directly
    with their own dict.

    Returns elapsed wall-clock seconds spent inside the play loop.
    """
    if output is None:
        output = sys.stdout

    # Float [-1, 1] → int16 PCM.
    audio_int16 = (audio_np * 32767).astype(np.int16)
    total_samples = len(audio_int16)

    if is_first_sentence:
        # Clears any leftover "[N] waiting..." / live-preview row before
        # printing the "Bot: " prefix (bug #1 fix from iter-001).
        output.write(f"\r{CLEAR_LINE}  {_CYAN}Bot:{_RESET} ")
        output.flush()

    t0 = clock()
    samples_played = 0
    token_idx = 0
    # iter-071: token-reveal lag accumulators. Only updated when
    # ``lag_out`` was provided by the caller — keeps the hot loop
    # branch-free in the common case.
    lag_sum = 0.0
    lag_count = 0
    lag_max = 0.0

    while samples_played < total_samples:
        if cancel_event is not None and cancel_event.is_set():
            break
        end = min(samples_played + play_chunk, total_samples)
        chunk_bytes = audio_int16[samples_played:end].tobytes()
        speaker_stream.write(chunk_bytes)
        samples_played = end

        # Current playback position in seconds (assumes the stream
        # consumes at `rate` samples/sec — true for a real soundcard
        # and for VirtualSpeakerStream during tests).
        pos = samples_played / rate

        while token_idx < len(tokens) and tokens[token_idx]["start"] <= pos:
            if lag_out is not None:
                # iter-071: lag for this token. ``clock() - t0`` is
                # wall-clock elapsed since play start; the token
                # claims its audio plays at ``token["start"]``.
                # Difference = lag (positive = text behind audio).
                lag = (clock() - t0) - tokens[token_idx]["start"]
                lag_sum += lag
                lag_count += 1
                if abs(lag) > abs(lag_max):
                    lag_max = lag
            _emit_token(output, tokens[token_idx]["text"], bold=True, flush=True)
            token_idx += 1

    elapsed = clock() - t0

    # Flush tokens whose start_ts was beyond the audio duration. The
    # original code emitted these without bold codes — preserve that
    # quirk via bold=False so test snapshots stay stable.
    #
    # iter-026: skip the trailing-token flush when the loop exited
    # via cancel_event. The user has barged in; the bot's voice has
    # been cut. Continuing to print the rest of the bot's text to
    # the terminal (when the user is talking, possibly to interrupt)
    # is jarring UX. Match the audio: when audio stops, text stops.
    cancelled = cancel_event is not None and cancel_event.is_set()
    if not cancelled:
        while token_idx < len(tokens):
            _emit_token(output, tokens[token_idx]["text"], bold=False, flush=False)
            token_idx += 1
        output.flush()

    # iter-071: publish lag stats. Skip when no tokens had a
    # lag observation (cancel-before-first-token, or the stream
    # had zero tokens).
    if lag_out is not None and lag_count > 0:
        lag_out["sum"] = lag_sum
        lag_out["count"] = lag_count
        lag_out["max"] = lag_max

    return elapsed
