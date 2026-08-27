"""TTS synthesis with token alignment — pulled out of mic_chat.

Until iter-018 this function lived in ``mic_chat.py`` next to the
pyaudio imports, which meant it was only importable on hosts with
ALSA dev headers. Tests on x86_64 Linux couldn't reach it. Same
iter-006/007 pattern: relocate the pyaudio-free logic to its own
leaf module so it composes with the iter-005 virtual audio + the
iter-015 ChatLoop without dragging real pyaudio along.

The function is small but does have one subtle thing: it
accumulates token timings across multiple pipeline yields by
adding a running ``offset`` (in seconds) so tokens emitted in the
second / third / Nth chunk get start/end times that are
contiguous with the first chunk. Without that, every chunk's
tokens would re-start at 0, breaking the iter-007 alignment
logic that reveals words during playback.

Tests verify:
  - Empty pipeline returns ``(empty audio, empty tokens)``
  - Single-chunk yield concatenates correctly
  - Multi-chunk yields offset tokens by accumulated duration
  - ``torch.Tensor`` audio is converted to numpy
  - Numpy audio passes through unchanged
"""

from __future__ import annotations

import numpy as np

# Kokoro emits 24kHz mono float32. Hardcoded throughout because we
# only target kokoro right now; if we ever support multiple TTS
# backends, this becomes per-engine config.
TTS_RATE = 24000


def synthesize_with_alignment(tts_engine, text: str, voice: str, speed: float):
    """Synthesize ``text`` via ``tts_engine`` and return
    ``(audio_np, tokens_with_timing)``.

    ``audio_np`` is a 1-D ``numpy.ndarray[float32]`` at
    ``TTS_RATE`` (24kHz mono).

    ``tokens_with_timing`` is a list of dicts:
        ``{"text": str, "start": float_seconds, "end": float_seconds}``
    where ``start``/``end`` are measured from the beginning of the
    full ``audio_np`` blob (NOT the per-chunk offsets the
    underlying TTS engine yields).

    Contract on ``tts_engine``:
      - ``._load()`` is callable (idempotent) — the kokoro engine
        uses this to lazy-load the model.
      - ``._pipeline(text, voice, speed)`` returns an iterable of
        result objects, each with:
          - ``.audio`` — either ``torch.Tensor`` or
            ``numpy.ndarray`` (1-D, float32, at TTS_RATE)
          - ``.tokens`` — iterable of token objects, each with
            ``.text``, ``.start_ts``, ``.end_ts`` in seconds
            measured from the start of THAT chunk (we accumulate
            offsets to make timings contiguous across chunks).

    Tests construct fake pipelines / engines / tokens to exercise
    this contract without loading kokoro.
    """
    tts_engine._load()
    all_audio = []
    all_tokens = []
    offset = 0.0

    for result in tts_engine._pipeline(text, voice=voice, speed=speed):
        audio = result.audio
        # ``torch.Tensor`` lazy-imports torch only when we actually
        # need to convert; pure-numpy callers (and tests) skip the
        # import entirely.
        if hasattr(audio, "numpy") and not isinstance(audio, np.ndarray):
            audio = audio.numpy()
        duration = len(audio) / TTS_RATE

        for tok in result.tokens:
            # Kokoro occasionally emits metadata-only / punctuation tokens
            # without alignment timestamps. They do not describe a playable
            # interval, so leave them out of the reveal timeline while keeping
            # the synthesized audio. Attempting to add the chunk offset to a
            # missing timestamp used to abort the rest of the response.
            if tok.start_ts is None or tok.end_ts is None:
                continue
            all_tokens.append({
                "text": tok.text,
                "start": tok.start_ts + offset,
                "end": tok.end_ts + offset,
            })
        all_audio.append(audio)
        offset += duration

    if not all_audio:
        return np.array([], dtype=np.float32), []

    combined = np.concatenate(all_audio)
    return combined, all_tokens
