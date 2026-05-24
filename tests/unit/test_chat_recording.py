"""End-to-end-ish tests for record_utterance_streaming using the
virtual mic + virtual clock + stub transcriber. No pyaudio, no
mlx-whisper, no real wall clock.

This is the first test that exercises the full recording loop on
x86_64 Linux. The injection points (transcribe_fn, clock, output) are
the contract — keep them stable, downstream tests will rely on them.
"""

from __future__ import annotations

import io
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_recording import (  # noqa: E402
    CHUNK,
    RATE,
    record_utterance_streaming,
)
from examples.virtual_audio import (  # noqa: E402
    VirtualMicStream,
    concat,
    feed_tts,
    make_silence,
    make_tone_burst,
)


class FrameClock:
    """Deterministic monotonic clock — advances by `chunk / rate` per
    call, matching the rhythm of one PyAudio CHUNK read at the project
    sample rate. Lets the recording loop run without ever touching real
    time.
    """

    def __init__(self, chunk: int = CHUNK, rate: int = RATE):
        self._dt = chunk / rate
        self._t = 0.0

    def __call__(self) -> float:
        t = self._t
        self._t += self._dt
        return t


def _stub_engine() -> SimpleNamespace:
    """Minimal stand-in for WhisperEngine: only `_last_text` is touched."""
    return SimpleNamespace(_last_text=None, model_repo="stub")


def _utterance_audio(rate: int = RATE) -> np.ndarray:
    """Standard test fixture: 0.3s lead silence + 1.0s tone burst + 1.2s
    trailing silence. Long enough to fire DONE_OK, short enough to keep
    tests fast.
    """
    return concat(
        make_silence(0.3, rate=rate),
        make_tone_burst(1.0, rate=rate, amp=0.3),
        make_silence(1.2, rate=rate),
    )


class TestRecordUtteranceStreaming:
    def test_returns_three_tuple_with_plausible_values(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(_utterance_audio())
        engine = _stub_engine()

        wav, dur, stt_time = record_utterance_streaming(
            mic,
            engine,
            transcribe_fn=lambda wav_bytes: "test transcript",
            clock=FrameClock(),
            output=io.StringIO(),
        )

        assert isinstance(wav, bytes) and len(wav) > 0
        # speech_duration should be in the rough ballpark of the 1s tone burst.
        assert 0.5 < dur < 2.0
        # stt_time is computed by clock() difference around the transcribe call.
        # Our FrameClock advances ~64ms per call, so two calls ≈ 0.064s.
        assert 0.0 <= stt_time < 1.0

    def test_transcript_stashed_on_engine_for_back_compat(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(_utterance_audio())
        engine = _stub_engine()

        record_utterance_streaming(
            mic,
            engine,
            transcribe_fn=lambda wav: "hello world",
            clock=FrameClock(),
            output=io.StringIO(),
        )
        assert engine._last_text == "hello world"

    def test_returned_wav_bytes_are_a_valid_wav_file(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(_utterance_audio())
        engine = _stub_engine()

        wav, _, _ = record_utterance_streaming(
            mic,
            engine,
            transcribe_fn=lambda w: None,
            clock=FrameClock(),
            output=io.StringIO(),
        )
        with wave.open(io.BytesIO(wav), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == RATE
            assert wf.getnframes() > 0

    def test_too_short_utterance_returns_empty_tuple(self):
        # 0.3s silence + 0.1s tone (below 0.3 min) + 1.2s silence.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(
            concat(
                make_silence(0.3),
                make_tone_burst(0.1, amp=0.3),
                make_silence(1.2),
            )
        )
        engine = _stub_engine()

        wav, dur, stt_time = record_utterance_streaming(
            mic,
            engine,
            transcribe_fn=lambda w: "should not be called",
            clock=FrameClock(),
            output=io.StringIO(),
        )
        assert wav == b""
        assert dur == 0.0
        assert stt_time == 0.0
        # On the too-short path the transcribe_fn is never invoked,
        # so engine._last_text stays at its sentinel (None).
        assert engine._last_text is None

    def test_transcribe_fn_receives_growing_buffer_during_speech(self):
        """As speech accumulates and we cross INFERENCE_INTERVAL, the
        preview path calls transcribe_fn with progressively larger wav
        blobs. Verify by recording each call's input size.
        """
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        # Long enough utterance (3s tone) to cross multiple 1.0s intervals.
        mic.push(
            concat(
                make_silence(0.3),
                make_tone_burst(3.0, amp=0.3),
                make_silence(1.2),
            )
        )
        engine = _stub_engine()
        sizes: list[int] = []

        def stub(wav_bytes):
            sizes.append(len(wav_bytes))
            return f"preview-{len(sizes)}"

        record_utterance_streaming(
            mic,
            engine,
            transcribe_fn=stub,
            clock=FrameClock(),
            output=io.StringIO(),
        )

        # At least one preview call + one final call.
        assert len(sizes) >= 2
        # Each call should see strictly more bytes than the previous —
        # the buffer grows monotonically while speaking.
        for a, b in zip(sizes, sizes[1:]):
            assert b > a

    def test_preview_writes_only_when_transcript_changes(self):
        """If transcribe_fn returns the same text twice in a row, we
        should NOT re-render the preview. Avoids flicker.
        """
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(
            concat(
                make_silence(0.3),
                make_tone_burst(3.0, amp=0.3),  # 3 preview intervals
                make_silence(1.2),
            )
        )
        engine = _stub_engine()
        out = io.StringIO()

        # Always return the same string — preview should only render once.
        record_utterance_streaming(
            mic,
            engine,
            transcribe_fn=lambda w: "same text every time",
            clock=FrameClock(),
            output=out,
        )

        rendered = out.getvalue()
        # Each render emits exactly one \r. We expect: maybe 1 preview
        # render + 1 final "You:" line. So between 1 and 3 \r's.
        assert 1 <= rendered.count("\r") <= 3

    def test_default_transcribe_fn_falls_back_to_mlx_whisper(self):
        """When transcribe_fn is None, the helper builds a default that
        wraps _transcribe_quick(stt_engine, wav). On Linux mlx-whisper
        is unavailable so it returns None — we just verify the loop
        completes without crashing.
        """
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(_utterance_audio())
        engine = _stub_engine()

        wav, dur, _ = record_utterance_streaming(
            mic,
            engine,
            clock=FrameClock(),
            output=io.StringIO(),
        )
        assert len(wav) > 0
        assert dur > 0.0
        # Without mlx-whisper, _transcribe_quick returns None.
        assert engine._last_text is None


# ---- TTS-fed end-to-end test (opt-in) ----------------------------------------

def _kokoro_loadable() -> bool:
    try:
        from examples.virtual_audio import _import_kokoro_engine
        _import_kokoro_engine()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _kokoro_loadable(), reason="kokoro TTS not loadable")
class TestRecordingFromTTSAudio:
    def test_records_synthesized_speech_end_to_end(self):
        """Render a sentence with kokoro, push into a VirtualMicStream,
        run the full record_utterance_streaming loop with a stub
        transcriber, and assert we got back a valid wav with plausible
        duration.
        """
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK, pad_with_silence=True)
        feed_tts(mic, "Hello, this is a recording test.", trailing_silence_s=1.2)
        engine = _stub_engine()

        wav, dur, _ = record_utterance_streaming(
            mic,
            engine,
            transcribe_fn=lambda w: "Hello, this is a recording test.",
            clock=FrameClock(),
            output=io.StringIO(),
        )
        assert len(wav) > 0
        # A short sentence is typically 1-2s of speech.
        assert 0.5 < dur < 4.0
        assert engine._last_text == "Hello, this is a recording test."
        # Verify the captured wav is well-formed.
        with wave.open(io.BytesIO(wav), "rb") as wf:
            assert wf.getframerate() == RATE
            # The captured audio should be at least as long as the
            # speech_duration we measured (in frames).
            assert wf.getnframes() >= int(dur * RATE * 0.5)


# ---- iter-010: primed_frames ------------------------------------------------


def _audio_to_chunks(audio: np.ndarray, chunk: int = CHUNK) -> list[bytes]:
    """Slice an int16 ndarray into byte chunks of size ``chunk * 2`` —
    the same shape ``BargeInWatcher.frames`` produces.

    Pads the final chunk with zeros so callers don't need to size
    their fixtures to be an exact multiple of ``chunk``.
    """
    n = len(audio)
    if n == 0:
        return []
    n_padded = ((n + chunk - 1) // chunk) * chunk
    if n_padded > n:
        audio = np.concatenate(
            [audio, np.zeros(n_padded - n, dtype=np.int16)]
        )
    chunks: list[bytes] = []
    for i in range(0, len(audio), chunk):
        chunks.append(audio[i : i + chunk].astype(np.int16).tobytes())
    return chunks


class TestRecordUtteranceWithPrimedFrames:
    def test_no_primed_frames_is_a_noop(self):
        """Backward compat: omitting the kwarg or passing None / [] is
        equivalent to the legacy behavior.
        """
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(_utterance_audio())
        engine = _stub_engine()

        wav, dur, _ = record_utterance_streaming(
            mic,
            engine,
            transcribe_fn=lambda w: None,
            clock=FrameClock(),
            output=io.StringIO(),
            primed_frames=None,
        )
        assert len(wav) > 0
        assert dur > 0.0

    def test_empty_primed_list_is_a_noop(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(_utterance_audio())
        engine = _stub_engine()

        wav, dur, _ = record_utterance_streaming(
            mic,
            engine,
            transcribe_fn=lambda w: None,
            clock=FrameClock(),
            output=io.StringIO(),
            primed_frames=[],
        )
        assert len(wav) > 0

    def test_primed_speech_chunks_appear_in_recorded_wav(self):
        """The primed chunks should be written to the resulting wav as
        if they had come from the live mic. After a primed speech burst
        plus enough live trailing silence to close the VAD window,
        DONE_OK fires and the wav covers both regions.
        """
        # Primed = 0.6s of tone (the user's barge-in audio).
        primed_audio = make_tone_burst(0.6, rate=RATE, amp=0.3)
        primed = _audio_to_chunks(primed_audio)
        # Live mic delivers more than 0.8s of silence so VAD closes.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(make_silence(1.2, rate=RATE))
        engine = _stub_engine()

        wav, dur, _ = record_utterance_streaming(
            mic,
            engine,
            transcribe_fn=lambda w: None,
            clock=FrameClock(),
            output=io.StringIO(),
            primed_frames=primed,
        )
        with wave.open(io.BytesIO(wav), "rb") as wf:
            n_frames = wf.getnframes()
            recorded = wf.readframes(n_frames)
        decoded = np.frombuffer(recorded, dtype=np.int16)
        # Recorded wav should contain at least roughly the primed
        # tone burst plus some trailing silence (the silence frames
        # that VAD appended before firing DONE_OK).
        assert n_frames >= len(primed_audio) - CHUNK  # allow off-by-one
        # Speech-duration measurement should be in the right ballpark
        # for ~0.6s of speech.
        assert 0.3 < dur < 1.5
        # First chunk of the recording should non-trivially overlap
        # the primed tone burst (RMS > 0).
        first_chunk = decoded[:CHUNK].astype(np.float32) / 32768.0
        assert float(np.sqrt(np.mean(first_chunk ** 2))) > 0.05

    def test_primed_silence_only_falls_through_to_live_mic(self):
        """Primed silence shouldn't trigger speech. The live mic should
        provide the actual utterance — this is the case where
        BargeInWatcher captured a brief silent moment but the user
        isn't actually speaking.
        """
        primed = _audio_to_chunks(make_silence(0.2, rate=RATE))
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(_utterance_audio())  # 0.3 silence + 1.0 tone + 1.2 silence
        engine = _stub_engine()

        wav, dur, _ = record_utterance_streaming(
            mic,
            engine,
            transcribe_fn=lambda w: None,
            clock=FrameClock(),
            output=io.StringIO(),
            primed_frames=primed,
        )
        # Should still record the live utterance.
        assert len(wav) > 0
        assert 0.5 < dur < 2.0

    def test_primed_too_short_burst_yields_done_too_short(self):
        """If the primed buffer is just a tiny blip and live mic is
        silent, VAD should fire DONE_TOO_SHORT and we get the empty
        return tuple.
        """
        primed = _audio_to_chunks(make_tone_burst(0.05, rate=RATE, amp=0.3))
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        # Live: just silence so VAD's silence window can fire.
        mic.push(make_silence(1.5, rate=RATE))
        engine = _stub_engine()

        wav, dur, stt_time = record_utterance_streaming(
            mic,
            engine,
            transcribe_fn=lambda w: "should not be called",
            clock=FrameClock(),
            output=io.StringIO(),
            primed_frames=primed,
        )
        assert wav == b""
        assert dur == 0.0
        assert stt_time == 0.0
        assert engine._last_text is None

    def test_primed_frames_count_in_returned_wav_byte_size(self):
        """A pedantic accounting check — the wav frame count should be
        roughly primed_frames + live_frames_to_close_vad. Helps catch
        regressions where primed frames are skipped or double-counted.
        """
        primed_audio = make_tone_burst(0.4, rate=RATE, amp=0.3)
        primed = _audio_to_chunks(primed_audio)
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(make_silence(1.2, rate=RATE))
        engine = _stub_engine()

        wav, _, _ = record_utterance_streaming(
            mic,
            engine,
            transcribe_fn=lambda w: None,
            clock=FrameClock(),
            output=io.StringIO(),
            primed_frames=primed,
        )
        with wave.open(io.BytesIO(wav), "rb") as wf:
            n_frames = wf.getnframes()
        # We expect at least all the primed frames + one chunk of
        # silence + however many silence frames it took to fire DONE_OK.
        # The minimum is len(primed_audio); upper bound is loose.
        assert n_frames >= len(primed_audio) - CHUNK
        assert n_frames <= len(primed_audio) + int(2.0 * RATE)
