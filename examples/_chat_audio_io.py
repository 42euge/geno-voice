"""Audio I/O closures — extracted from mic_chat.py:run_chat.

iter-109: ChatLoop is dep-injected (iter-015) — it accepts a
``speaker_factory`` callable, ``synth_fn``, and ``play_fn`` and
runs against any backend. Production wires those to PyAudio +
kokoro + the existing playback core; tests wire them to virtual
audio streams.

This module owns the production wiring. Three closures used to
live inline in `run_chat`:

  - ``speaker_factory()`` — open a per-sentence PyAudio output stream
  - ``synth(sentence)`` — call synthesize_with_alignment with the engine
  - ``play(speaker, audio, tokens, **)`` — call _play_aligned_core

`build_audio_io()` returns all three as a single ``AudioIO``
dataclass so `run_chat` can pass them straight into
``ChatLoop(...)`` without juggling individual variables.

Same factory-shaped pattern as iter-107 + iter-108 — the third
instance, which is the threshold for documenting it as the
**default mic_chat.py extraction shape**:

  1. Caller passes resource handles + config (here: ``pa``,
     ``tts_engine``, ``voice``, ``speed``).
  2. Module returns a dataclass bundle of callables / values.
  3. No log injection here because the closures don't log;
     log-injection joins the pattern when the extracted code
     emits user-facing strings (iter-107, iter-108).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

# Re-imports that match the original mic_chat closures. We import
# at module level — both are pure-Python and don't pull in
# platform-specific deps at import time.
from examples._chat_playback import TTS_RATE, play_aligned as _play_aligned_core
from examples._chat_tts import synthesize_with_alignment


# Type aliases — match the shapes ChatLoop expects.
SpeakerFactory = Callable[[], Any]
SynthFn = Callable[[str], tuple[Any, Any]]
PlayFn = Callable[..., float]


@dataclass
class AudioIO:
    """Bundle of the three audio-side callables ChatLoop needs."""

    speaker_factory: SpeakerFactory
    synth_fn: SynthFn
    play_fn: PlayFn


def build_audio_io(
    pa: Any,
    tts_engine: Any,
    voice: str,
    speed: Union[float, Callable[[], float]],
    *,
    pyaudio_module: Optional[Any] = None,
    speaker_chunk: int = 1024,
    rate: int = TTS_RATE,
) -> AudioIO:
    """Build the production audio-I/O closure bundle.

    Args:
        pa: a `pyaudio.PyAudio()` instance owning the audio
            subsystem. The caller is responsible for its
            lifecycle (open + terminate).
        tts_engine: a TTS engine ready to synthesize (already
            ``_load()``-ed).
        voice: voice id passed through to synth.
        speed: speech-rate multiplier passed through to synth. Either a
            **constant float** (the historical shape — baked into the
            ``synth_fn`` closure once) OR a **zero-arg callable** returning
            the current speed (iter-214). The callable form lets the
            WPM-mirroring path (``SpeedController.current``) adapt the rate
            turn-to-turn: ``synth_fn`` resolves it fresh on every sentence, so
            a speed updated between turns takes effect on the next sentence
            synthesized. A plain float is read once and used unchanged — the
            proven constant-speed behavior.
        pyaudio_module: defaults to the runtime ``pyaudio``
            module. Tests pass a stub exposing ``paInt16`` so
            the module is importable without pyaudio installed
            (matching the pattern other modules already use:
            virtual_audio.VirtualSpeakerStream takes the place
            of pa.open).
        speaker_chunk: frames_per_buffer for the speaker stream.
            Default 1024 matches the previous inline value.
        rate: speaker sample rate. Default TTS_RATE matches the
            previous inline value.

    Returns:
        An ``AudioIO`` with the three callables. Each closure
        captures the args above; calling ``speaker_factory()``,
        ``synth_fn(sentence)``, or ``play_fn(speaker, audio,
        tokens, ...)`` is identical to the original inline code.
    """
    def speaker_factory():
        # Lazy import — keeps build_audio_io itself importable on
        # systems without pyaudio (CI, x86_64 Linux). Real callers
        # in mic_chat.py have already imported pyaudio at module
        # scope, so this fallback only fires in test environments.
        nonlocal pyaudio_module
        if pyaudio_module is None:
            import pyaudio as pyaudio_module  # type: ignore
        return pa.open(
            format=pyaudio_module.paInt16,
            channels=1,
            rate=rate,
            output=True,
            frames_per_buffer=speaker_chunk,
        )

    # iter-214: resolve `speed` per sentence. A callable is invoked fresh each
    # synth (the WPM-mirroring path, where SpeedController.current returns the
    # live per-session speed); a plain number is used as-is. `callable(...)`
    # cleanly distinguishes the two — floats/ints aren't callable.
    speed_is_callable = callable(speed)

    def synth_fn(sentence: str):
        current_speed = speed() if speed_is_callable else speed
        return synthesize_with_alignment(
            tts_engine, sentence, voice, current_speed,
        )

    def play_fn(speaker, audio_np, tokens, *, is_first_sentence=False,
                cancel_event=None, lag_out=None):
        # iter-071: forward lag_out so SentenceWorker can collect
        # per-token reveal-lag stats on the live mic_chat path.
        return _play_aligned_core(
            speaker, audio_np, tokens,
            is_first_sentence=is_first_sentence,
            rate=rate,
            cancel_event=cancel_event,
            lag_out=lag_out,
        )

    return AudioIO(
        speaker_factory=speaker_factory,
        synth_fn=synth_fn,
        play_fn=play_fn,
    )
