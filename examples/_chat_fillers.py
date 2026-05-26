"""Filler pre-rendering — extracted from mic_chat.py:run_chat.

iter-107: The filler-words feature (iter-011) needs each filler
text rendered to audio at startup so the SentenceWorker can pick
one off the shelf when the LLM stalls. The original code lived
inline in `run_chat`, mixing TTS calls with print() logging and
ANSI color codes. This module pulls the rendering loop out so
tests can drive it with a stub `synth_fn`.

Design choices:

  - **Caller passes `synth_fn`, not the engine.** Existing
    callers in mic_chat.py wrap the engine + voice + speed in a
    closure (matching iter-019's `synthesize_with_alignment`
    shape). Tests pass any callable returning ``(audio,
    tokens)`` — no real TTS engine needed.

  - **Caller passes `log`, not stdout.** Mirrors iter-009's
    pattern. Default to `print` for production parity; tests
    capture into a list.

  - **No ANSI color codes in this module.** Color is a
    presentation concern owned by `mic_chat.py`. The log
    callable is responsible for any styling.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable

# Type alias — every entry is ``(audio_np, tokens)`` matching the
# tuple shape the SentenceWorker expects.
FillerClip = tuple[Any, Any]
SynthFn = Callable[[str], FillerClip]
LogFn = Callable[[str], None]


def prerender_fillers(
    synth_fn: SynthFn,
    texts: Iterable[str],
    *,
    idle_threshold: float = 0.0,
    log: LogFn = print,
) -> list[FillerClip]:
    """Render each filler text to a (audio, tokens) tuple.

    Args:
        synth_fn: callable taking a filler-text string, returning
            ``(audio_np, tokens)``. Real callers wrap
            ``synthesize_with_alignment(engine, text, voice, speed)``;
            tests pass a stub.
        texts: filler-text strings to render. Empty iterable is
            allowed — the function returns ``[]`` and emits no log.
        idle_threshold: surfaced in the summary log line so the
            operator sees what threshold the rendered clips will
            actually trigger at. Doesn't affect rendering itself.
        log: single-line emit callable. Defaults to ``print``.

    Returns:
        List of successfully rendered (audio, tokens) tuples.
        Length may be < len(texts) if any synth raised or
        produced empty audio.

    Behavior:
        - Per-filler ``try/except`` — one bad filler doesn't kill
          the rest. Failures emit a single log line per failure.
        - Empty audio (``len(audio_np) == 0``) is silently
          dropped (treated like a failure but no log line —
          matches the original inline behavior, which only logged
          on exception).
        - When ``texts`` is non-empty, a final summary line emits
          with the wall-clock total + success rate. When empty,
          no summary line emits (matches the inline ``if
          filler_texts:`` guard).
    """
    rendered: list[FillerClip] = []
    texts_list = list(texts)
    if not texts_list:
        return rendered

    t0 = time.monotonic()
    for text in texts_list:
        try:
            audio_np, tokens = synth_fn(text)
            if len(audio_np) > 0:
                rendered.append((audio_np, tokens))
        except Exception as e:
            # iter-107: matches the original mic_chat.py log line.
            # Caller (mic_chat) wraps this in YELLOW; here we just
            # emit plain text and let the log callable decide.
            log(f"filler synth failed for {text!r}: {e}")

    elapsed_ms = (time.monotonic() - t0) * 1000
    log(
        f"Pre-rendered {len(rendered)}/{len(texts_list)} "
        f"fillers in {elapsed_ms:.0f}ms "
        f"(idle threshold {idle_threshold:.2f}s)"
    )
    return rendered
