"""iter-122 — Full ChatLoop integration with FasterWhisperEngine.

Drives `mic_chat.py:run_one_turn` through the entire pipeline on
x86_64 Linux:

    VirtualMicStream
      → recorder (with VAD)
      → FasterWhisperEngine.transcribe (real STT)
      → stub LLM yields tokens
      → SentenceWorker
      → stub synth + slow_play
      → VirtualSpeakerStream

This is the integration test iter-118 + iter-119 + iter-121
together promised: a Linux operator can configure mic_chat for
faster-whisper and the entire pipeline works end-to-end.

Skips cleanly when faster-whisper is unavailable (no install,
no model cache, no network).
"""

from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_loop import ChatLoop  # noqa: E402
from examples._chat_recording import CHUNK, RATE  # noqa: E402
from examples.virtual_audio import (  # noqa: E402
    VirtualMicStream,
    VirtualSpeakerStream,
    concat,
    make_silence,
)


# ---- Skip plumbing ------------------------------------------------------


try:
    from stt.faster_whisper_engine import FasterWhisperEngine  # noqa: E402
    import faster_whisper  # noqa: F401
    _FW_AVAILABLE = True
except Exception as e:
    _FW_AVAILABLE = False
    _FW_IMPORT_ERROR = str(e)


@pytest.fixture(scope="module")
def stt_engine():
    if not _FW_AVAILABLE:
        pytest.skip(
            f"faster-whisper unavailable: {_FW_IMPORT_ERROR}"
        )
    try:
        engine = FasterWhisperEngine(
            model_repo="tiny", device="cpu", compute_type="int8",
        )
        engine._load()  # warm cache
        return engine
    except Exception as e:
        pytest.skip(f"FasterWhisperEngine load failed: {e}")


# ---- Fixture loading ----------------------------------------------------


CLEAN_16K = ROOT / "tests" / "fixtures" / "wer" / "clean_16khz.wav"


def _read_wav_samples(path: Path) -> np.ndarray:
    """Read a 16-bit mono WAV at the recorder's sample rate.

    Asserts the rate matches RATE (16000 Hz) — if a future
    fixture commit drifts, this surfaces fast.
    """
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1, f"{path}: not mono"
        assert w.getsampwidth() == 2, f"{path}: not 16-bit"
        assert w.getframerate() == RATE, (
            f"{path}: rate {w.getframerate()} != {RATE}"
        )
        nframes = w.getnframes()
        return np.frombuffer(w.readframes(nframes), dtype=np.int16)


@pytest.fixture(scope="module")
def clean_audio_samples():
    if not CLEAN_16K.exists():
        pytest.skip(f"clean_16khz.wav fixture missing: {CLEAN_16K}")
    return _read_wav_samples(CLEAN_16K)


# ---- Stubs (non-STT layers stay deterministic) -------------------------


def _slow_play(speaker, audio, tokens, *, is_first_sentence=False, cancel_event=None):
    """Same shape as tests/performance/test_pipeline_perf.py's
    _slow_play — writes the audio in small chunks with a tiny
    sleep so timing-sensitive metrics (TTFS, etc.) get
    realistic-ish values.
    """
    audio_int16 = (audio * 32767).astype(np.int16)
    chunk = 256
    written = 0
    while written < len(audio_int16):
        if cancel_event is not None and cancel_event.is_set():
            break
        end = min(written + chunk, len(audio_int16))
        speaker.write(audio_int16[written:end].tobytes())
        written = end
        time.sleep(0.005)
    return 0.0


def _const_synth(samples=2048):
    """Stub TTS — constant 2048-sample tone. Same as the perf
    suite stub. We don't care about TTS quality; the test
    targets the STT path."""
    def synth(sentence):
        return np.full(samples, 0.5, dtype=np.float32), []
    return synth


def _yield_tokens_for(text):
    """Stub LLM — yields whitespace-split tokens of `text`."""
    def factory(messages, config):
        for tok in text.split():
            yield tok + " "
    return factory


# ---- The actual integration test ---------------------------------------


def test_one_turn_through_real_stt(stt_engine, clean_audio_samples):
    """Single turn: push clean.wav samples to virtual mic →
    recorder + FasterWhisperEngine produce a transcript →
    stub LLM streams response → stub TTS → speaker plays it.

    Assertions:
      - The turn completes without `had_error`.
      - `result.metrics` is populated.
      - The transcript contains at least one of the expected
        words ("weather" or "today" — clean.wav reference is
        "what is the weather today"). Loose match because tiny
        model output varies.
      - At least one sentence was spoken.
      - STT time is positive.
    """
    mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
    # iter-117 fixture references "what is the weather today",
    # ~1.5s of audio. Pad with leading + trailing silence so the
    # VAD's silence-duration window can fire DONE_OK after the
    # speech ends.
    leading_silence = make_silence(0.3, rate=RATE)
    trailing_silence = make_silence(1.5, rate=RATE)
    # All three arrays are int16 (make_silence returns int16,
    # clean_audio_samples comes from wave.readframes as int16);
    # concat preserves int16 so the recorder's normalization
    # (int16 → float32 / 32768.0) sees the right amplitude.
    full = concat(leading_silence, clean_audio_samples, trailing_silence)
    mic.push(full)

    bot_response = "Got it."
    loop = ChatLoop(
        mic=mic,
        speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
        stt_engine=stt_engine,
        transcribe_fn=lambda wav: stt_engine.transcribe(wav)[0],
        llm_stream_fn=_yield_tokens_for(bot_response),
        llm_config={"model": "stub"},
        synth_fn=_const_synth(),
        play_fn=_slow_play,
    )

    result = loop.run_one_turn([])

    assert not result.had_error, "turn produced an error"
    assert result.metrics is not None, "no metrics produced"
    m = result.metrics

    # The headline assertion: real STT got something usable.
    transcript_lower = (m.transcript or "").lower()
    assert "weather" in transcript_lower or "today" in transcript_lower, (
        f"transcript missing expected words: {m.transcript!r}"
    )

    # Pipeline ran end-to-end.
    assert m.sentences_spoken >= 1
    assert m.stt_time > 0
    assert m.speech_duration > 0


def test_transcribe_fn_signature_is_what_chat_loop_expects(stt_engine):
    """Sanity: ChatLoop's transcribe_fn contract is `wav_bytes
    → str | None`. The closure we use unwraps the (text, elapsed)
    tuple FasterWhisperEngine returns. If either side changes
    shape, this fails fast."""
    closure = lambda wav: stt_engine.transcribe(wav)[0]
    # Use the iter-117 16khz clean.wav directly.
    wav_bytes = CLEAN_16K.read_bytes()
    text = closure(wav_bytes)
    assert isinstance(text, str)
    assert len(text) > 0
