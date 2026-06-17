"""Tests for ``vad/silence.py:SilenceDetector`` — the core VAD state machine.

Despite being the most fragile part of the live loop (a four-way decision
between "still speaking", "silence beat → emit", "max duration → cut", and
"too short → drop"), the ``SilenceDetector`` class had **zero direct unit
tests** prior to iter-185 — only its downstream config wiring
(``test_vad_config.py``, ``test_watcher_vad_threshold.py``) was covered. This
file closes that gap.

Two layers are exercised:

  1. **Pure helpers** — ``rms_amplitude`` (the silence/speech discriminator)
     and ``make_wav`` (the PCM → WAV wrapper). No clock needed.

  2. **The ``feed`` / ``flush`` state machine** — the speaking flag, the
     silence-duration emit, the max-duration cut, and the min-duration drop.
     ``feed`` reads ``time.monotonic()`` to time the silence beat, so we
     monkeypatch ``vad.silence.time.monotonic`` with a fake clock list and
     advance it explicitly — the same deterministic-clock trick the barge-in
     coordinator tests use. No real time passes and no audio hardware is
     touched.
"""

from __future__ import annotations

import io
import struct
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import vad.silence as silence_mod  # noqa: E402
from vad.silence import (  # noqa: E402
    FRAME_SIZE,
    RATE,
    SAMPLE_WIDTH,
    SilenceDetector,
    make_wav,
)


# --------------------------------------------------------------------------- #
# helpers — build PCM frames at a known amplitude / duration                  #
# --------------------------------------------------------------------------- #

def _pcm(amplitude: int, seconds: float) -> bytes:
    """Return ``seconds`` of mono 16-bit PCM at constant ``amplitude``."""
    n = int(RATE * seconds)
    return struct.pack(f"<{n}h", *([amplitude] * n))


def _loud(seconds: float) -> bytes:
    # 20000 / 32768 ≈ 0.61 RMS — well above the default 0.02 threshold.
    return _pcm(20000, seconds)


def _quiet(seconds: float) -> bytes:
    return _pcm(0, seconds)


@pytest.fixture
def fake_clock(monkeypatch):
    """Replace ``vad.silence.time.monotonic`` with a controllable clock.

    Yields a single-element list; set ``clock[0]`` to advance time. The
    detector reads the patched ``monotonic`` only inside ``feed`` when it is
    in the silence-after-speech branch.
    """
    clock = [0.0]
    monkeypatch.setattr(silence_mod.time, "monotonic", lambda: clock[0])
    return clock


def _wav_frames(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        return wf.getnframes()


# --------------------------------------------------------------------------- #
# rms_amplitude — the silence/speech discriminator                            #
# --------------------------------------------------------------------------- #

class TestRmsAmplitude:
    def test_empty_is_zero(self):
        assert SilenceDetector.rms_amplitude(b"") == 0.0

    def test_single_byte_is_zero(self):
        # Fewer than 2 bytes can't form a sample → guarded to 0.0.
        assert SilenceDetector.rms_amplitude(b"\x00") == 0.0

    def test_all_zero_samples_is_zero(self):
        assert SilenceDetector.rms_amplitude(_quiet(0.01)) == 0.0

    def test_full_scale_is_near_one(self):
        amp = SilenceDetector.rms_amplitude(_pcm(32767, 0.01))
        assert amp == pytest.approx(1.0, abs=1e-3)

    def test_normalized_to_unit_range(self):
        # Half-scale constant signal → ~0.5 RMS (constant ⇒ RMS == |level|).
        amp = SilenceDetector.rms_amplitude(_pcm(16384, 0.01))
        assert amp == pytest.approx(0.5, abs=1e-3)

    def test_odd_trailing_byte_ignored(self):
        # An extra dangling byte must not corrupt the unpack; result matches
        # the clean even-length signal.
        clean = _pcm(10000, 0.01)
        amp_clean = SilenceDetector.rms_amplitude(clean)
        amp_odd = SilenceDetector.rms_amplitude(clean + b"\x07")
        assert amp_odd == pytest.approx(amp_clean, abs=1e-6)

    def test_louder_signal_has_higher_rms(self):
        soft = SilenceDetector.rms_amplitude(_pcm(1000, 0.01))
        loud = SilenceDetector.rms_amplitude(_pcm(20000, 0.01))
        assert loud > soft > 0.0


# --------------------------------------------------------------------------- #
# make_wav — the PCM → WAV wrapper                                            #
# --------------------------------------------------------------------------- #

class TestMakeWav:
    def test_header_defaults(self):
        wav = make_wav(b"\x01\x02\x03\x04")
        with wave.open(io.BytesIO(wav)) as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == SAMPLE_WIDTH
            assert wf.getframerate() == RATE

    def test_roundtrips_payload(self):
        payload = _pcm(123, 0.01)
        wav = make_wav(payload)
        with wave.open(io.BytesIO(wav)) as wf:
            assert wf.readframes(wf.getnframes()) == payload

    def test_frame_count_matches_payload(self):
        payload = _pcm(0, 0.02)  # 0.02 s
        wav = make_wav(payload)
        assert _wav_frames(wav) == len(payload) // SAMPLE_WIDTH

    def test_custom_rate_and_channels(self):
        wav = make_wav(b"\x00\x00" * 8, rate=8000, channels=2, sample_width=2)
        with wave.open(io.BytesIO(wav)) as wf:
            assert wf.getframerate() == 8000
            assert wf.getnchannels() == 2

    def test_empty_payload_is_valid_wav(self):
        wav = make_wav(b"")
        assert _wav_frames(wav) == 0


# --------------------------------------------------------------------------- #
# feed — the state machine                                                    #
# --------------------------------------------------------------------------- #

class TestFeedStateMachine:
    def test_loud_input_starts_speaking_no_emit(self, fake_clock):
        d = SilenceDetector()
        amp, chunk = d.feed(_loud(0.6))
        assert d.speaking is True
        assert chunk is None
        assert amp > d.threshold
        assert d.buffer_duration == pytest.approx(0.6, abs=1e-3)

    def test_quiet_before_speech_is_ignored(self, fake_clock):
        # Silence with no prior speech accumulates nothing and never emits.
        d = SilenceDetector()
        amp, chunk = d.feed(_quiet(0.5))
        assert d.speaking is False
        assert chunk is None
        assert d.buffer_duration == 0.0

    def test_silence_beat_emits_chunk(self, fake_clock):
        d = SilenceDetector(silence_duration=0.8, min_chunk_duration=0.5)
        d.feed(_loud(0.6))  # speaking
        # First quiet frame arms the silence timer (monotonic == 0.0).
        d.feed(_quiet(0.1))
        assert d.silence_start == 0.0
        # Advance past the silence window, then feed one more quiet frame.
        fake_clock[0] = 0.85
        amp, chunk = d.feed(_quiet(0.1))
        assert chunk is not None
        assert _wav_frames(chunk) > 0
        # Emitting resets the machine.
        assert d.speaking is False
        assert d.buffer_duration == 0.0

    def test_silence_shorter_than_window_holds(self, fake_clock):
        d = SilenceDetector(silence_duration=0.8, min_chunk_duration=0.5)
        d.feed(_loud(0.6))
        d.feed(_quiet(0.1))  # arms timer at 0.0
        fake_clock[0] = 0.5  # still within the 0.8 s window
        amp, chunk = d.feed(_quiet(0.1))
        assert chunk is None
        assert d.speaking is True  # still mid-utterance

    def test_resumed_speech_clears_silence_timer(self, fake_clock):
        d = SilenceDetector(silence_duration=0.8)
        d.feed(_loud(0.4))
        d.feed(_quiet(0.1))  # arm timer at 0.0
        assert d.silence_start == 0.0
        d.feed(_loud(0.2))  # speech resumes → timer cleared
        assert d.silence_start is None
        assert d.speaking is True

    def test_short_utterance_dropped_on_silence(self, fake_clock):
        # An utterance below min_chunk_duration is discarded, not emitted.
        d = SilenceDetector(silence_duration=0.8, min_chunk_duration=0.5)
        d.feed(_loud(0.2))  # under the 0.5 s floor
        d.feed(_quiet(0.1))
        fake_clock[0] = 0.9
        amp, chunk = d.feed(_quiet(0.1))
        assert chunk is None
        assert d.speaking is False  # still reset, just nothing emitted

    def test_max_duration_cuts_chunk(self, fake_clock):
        d = SilenceDetector(max_chunk_duration=1.0, min_chunk_duration=0.5)
        amp, chunk = d.feed(_loud(1.2))  # exceeds max in one frame
        assert chunk is not None
        assert d.speaking is False
        assert d.buffer_duration == 0.0

    def test_max_duration_cut_includes_buffered_audio(self, fake_clock):
        d = SilenceDetector(max_chunk_duration=1.0, min_chunk_duration=0.5)
        d.feed(_loud(0.6))
        amp, chunk = d.feed(_loud(0.6))  # total 1.2 s ≥ max
        assert chunk is not None
        # ~1.2 s at RATE — the whole buffer is flushed, not just one frame.
        assert _wav_frames(chunk) == pytest.approx(int(RATE * 1.2), rel=0.01)

    def test_amp_returned_each_call(self, fake_clock):
        d = SilenceDetector()
        amp_loud, _ = d.feed(_loud(0.1))
        amp_quiet, _ = d.feed(_quiet(0.1))
        assert amp_loud > amp_quiet


# --------------------------------------------------------------------------- #
# flush — drain whatever is buffered at shutdown / turn end                   #
# --------------------------------------------------------------------------- #

class TestFlush:
    def test_flush_empty_returns_none(self):
        assert SilenceDetector().flush() is None

    def test_flush_long_buffer_emits(self, fake_clock):
        d = SilenceDetector(min_chunk_duration=0.5)
        d.feed(_loud(0.6))
        chunk = d.flush()
        assert chunk is not None
        assert _wav_frames(chunk) > 0

    def test_flush_short_buffer_dropped(self, fake_clock):
        d = SilenceDetector(min_chunk_duration=0.5)
        d.feed(_loud(0.2))  # below floor
        assert d.flush() is None

    def test_flush_resets_state(self, fake_clock):
        d = SilenceDetector(min_chunk_duration=0.5)
        d.feed(_loud(0.6))
        d.flush()
        assert d.speaking is False
        assert d.silence_start is None
        assert d.buffer_duration == 0.0

    def test_flush_is_idempotent(self, fake_clock):
        d = SilenceDetector(min_chunk_duration=0.5)
        d.feed(_loud(0.6))
        assert d.flush() is not None
        assert d.flush() is None  # nothing left to drain


# --------------------------------------------------------------------------- #
# buffer_duration property                                                     #
# --------------------------------------------------------------------------- #

class TestBufferDuration:
    def test_empty_is_zero(self):
        assert SilenceDetector().buffer_duration == 0.0

    def test_tracks_accumulated_speech(self, fake_clock):
        d = SilenceDetector()
        d.feed(_loud(0.3))
        assert d.buffer_duration == pytest.approx(0.3, abs=1e-3)
        d.feed(_loud(0.2))
        assert d.buffer_duration == pytest.approx(0.5, abs=1e-3)

    def test_frame_size_constant(self):
        # FRAME_SIZE is one second of mono 16-bit PCM; buffer_duration divides
        # the byte count by it.
        assert FRAME_SIZE == RATE * SAMPLE_WIDTH
