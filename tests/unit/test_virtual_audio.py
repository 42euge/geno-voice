"""Unit tests for examples/virtual_audio.py.

These tests are pure-Python + numpy. The kokoro-based TTS smoke test at
the bottom is opt-in and skips cleanly when kokoro can't load on this
machine (e.g. wrong torch version).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_helpers import VadEvent, VadState  # noqa: E402
from examples.virtual_audio import (  # noqa: E402
    DEFAULT_RATE,
    SAMPLE_WIDTH,
    VirtualAudioInterface,
    VirtualMicStream,
    VirtualSpeakerStream,
    concat,
    feed_tts,
    make_noise_burst,
    make_silence,
    make_tone_burst,
    simulate_vad_over_audio,
)


class TestVirtualMicStream:
    def test_push_and_read_round_trips(self):
        mic = VirtualMicStream(rate=16000, chunk_size=1024)
        mic.push(np.full(2048, 1234, dtype=np.int16))
        assert mic.frames_buffered == 2048
        assert mic.get_read_available() == 2048
        out = mic.read(1024)
        assert len(out) == 1024 * SAMPLE_WIDTH
        decoded = np.frombuffer(out, dtype=np.int16)
        assert np.all(decoded == 1234)
        assert mic.get_read_available() == 1024

    def test_push_silence_helper(self):
        mic = VirtualMicStream(rate=16000)
        mic.push_silence(0.5)
        assert mic.frames_buffered == 8000
        out = mic.read(8000)
        decoded = np.frombuffer(out, dtype=np.int16)
        assert np.all(decoded == 0)

    def test_underflow_zero_pads_by_default(self):
        mic = VirtualMicStream(rate=16000)
        mic.push(np.full(500, 1000, dtype=np.int16))
        out = mic.read(1024)  # asked for more than buffered
        assert len(out) == 1024 * SAMPLE_WIDTH
        decoded = np.frombuffer(out, dtype=np.int16)
        assert np.all(decoded[:500] == 1000)
        assert np.all(decoded[500:] == 0)
        assert mic.get_read_available() == 0

    def test_underflow_no_padding_returns_short_read(self):
        mic = VirtualMicStream(rate=16000, pad_with_silence=False)
        mic.push(np.full(500, 1000, dtype=np.int16))
        out = mic.read(1024)
        assert len(out) == 500 * SAMPLE_WIDTH

    def test_close_blocks_further_use(self):
        mic = VirtualMicStream(rate=16000)
        mic.push(np.zeros(1024, dtype=np.int16))
        mic.close()
        assert mic.get_read_available() == 0
        with pytest.raises(OSError):
            mic.read(100)
        with pytest.raises(OSError):
            mic.push(np.zeros(10, dtype=np.int16))

    def test_reads_are_recorded_for_assertion(self):
        mic = VirtualMicStream(rate=16000)
        mic.push_silence(0.5)
        mic.read(1024)
        mic.read(2048)
        assert mic.reads == [1024, 2048]

    def test_accepts_float_audio(self):
        mic = VirtualMicStream(rate=16000)
        mic.push(np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32))
        out = mic.read(4)
        decoded = np.frombuffer(out, dtype=np.int16)
        # 0.5 * 32767 ≈ 16383
        assert decoded[0] == pytest.approx(16383, abs=2)
        assert decoded[1] == pytest.approx(-16383, abs=2)
        assert decoded[2] == 32767
        assert decoded[3] == -32767


class TestVirtualSpeakerStream:
    def test_write_captures_bytes(self):
        spk = VirtualSpeakerStream(rate=24000)
        data = np.full(512, 200, dtype=np.int16).tobytes()
        spk.write(data)
        spk.write(data)
        assert len(spk.captured) == 2 * 512 * SAMPLE_WIDTH
        decoded = spk.captured_int16
        assert np.all(decoded == 200)

    def test_captured_float32_normalizes(self):
        spk = VirtualSpeakerStream()
        spk.write(np.array([16384, -16384], dtype=np.int16))
        assert spk.captured_float32[0] == pytest.approx(0.5, abs=0.01)
        assert spk.captured_float32[1] == pytest.approx(-0.5, abs=0.01)

    def test_loopback_routes_writes_to_mic(self):
        mic = VirtualMicStream(rate=24000)
        spk = VirtualSpeakerStream(rate=24000, loopback_to=mic)
        spk.write(np.full(1024, 5000, dtype=np.int16))
        assert mic.frames_buffered == 1024
        decoded = np.frombuffer(mic.read(1024), dtype=np.int16)
        assert np.all(decoded == 5000)

    def test_close_blocks_writes(self):
        spk = VirtualSpeakerStream()
        spk.close()
        with pytest.raises(OSError):
            spk.write(b"\x00\x00")


class TestVirtualAudioInterface:
    def test_open_input_returns_mic(self):
        pa = VirtualAudioInterface(input_rate=16000)
        s = pa.open(channels=1, rate=16000, input=True, frames_per_buffer=1024)
        assert isinstance(s, VirtualMicStream)
        assert s.rate == 16000
        assert s.chunk_size == 1024

    def test_open_output_returns_speaker(self):
        pa = VirtualAudioInterface(output_rate=24000)
        s = pa.open(channels=1, rate=24000, output=True)
        assert isinstance(s, VirtualSpeakerStream)
        assert s.rate == 24000

    def test_loopback_links_speaker_to_existing_mic(self):
        pa = VirtualAudioInterface(loopback=True)
        mic = pa.open(rate=16000, input=True)
        spk = pa.open(rate=16000, output=True)
        assert spk.loopback_to is mic

    def test_terminate_closes_all_streams(self):
        pa = VirtualAudioInterface()
        mic = pa.open(rate=16000, input=True)
        spk = pa.open(rate=16000, output=True)
        pa.terminate()
        assert mic._closed and spk._closed
        with pytest.raises(OSError):
            pa.open(rate=16000, input=True)


class TestAudioFixtures:
    def test_silence_is_zeros_at_correct_length(self):
        s = make_silence(0.5, rate=16000)
        assert s.dtype == np.int16
        assert len(s) == 8000
        assert np.all(s == 0)

    def test_tone_burst_has_high_rms(self):
        t = make_tone_burst(0.5, rate=16000, amp=0.3)
        f = t.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(f ** 2)))
        # 0.3 amplitude sine has theoretical RMS = 0.3 / sqrt(2) ≈ 0.212
        assert 0.18 < rms < 0.25

    def test_noise_burst_is_deterministic_with_seed(self):
        a = make_noise_burst(0.1, seed=42)
        b = make_noise_burst(0.1, seed=42)
        assert np.array_equal(a, b)

    def test_concat_combines_chunks(self):
        out = concat(make_silence(0.1), make_tone_burst(0.1), make_silence(0.1))
        assert len(out) == 4800  # 3 * 0.1s @ 16kHz
        # First 1600 should be zero, middle 1600 nonzero, last 1600 zero.
        assert np.all(out[:1600] == 0)
        assert np.any(out[1600:3200] != 0)
        assert np.all(out[3200:] == 0)


class TestSimulateVadOverAudio:
    """End-to-end style: feed a fixture through a VadState and assert the
    event sequence makes sense for the input shape.
    """

    def test_pure_silence_stays_idle(self):
        audio = make_silence(2.0, rate=16000)
        events, vad = simulate_vad_over_audio(audio, rate=16000)
        assert all(e is VadEvent.IDLE for e in events)
        assert not vad.speaking

    def test_long_speech_then_silence_fires_done_ok(self):
        audio = concat(
            make_silence(0.3),
            make_tone_burst(1.5, amp=0.3),
            make_silence(1.2),
        )
        events, vad = simulate_vad_over_audio(audio, rate=16000)
        assert VadEvent.DONE_OK in events
        # Exactly one DONE_OK for one utterance.
        assert sum(1 for e in events if e is VadEvent.DONE_OK) == 1
        # speech_duration should be in the right ballpark for ~1.5s of tone.
        assert 1.0 < vad.last_speech_duration < 2.0

    def test_short_blip_fires_done_too_short(self):
        audio = concat(
            make_silence(0.3),
            make_tone_burst(0.1, amp=0.3),  # well below 0.3s min
            make_silence(1.2),
        )
        events, vad = simulate_vad_over_audio(audio, rate=16000)
        assert VadEvent.DONE_TOO_SHORT in events
        assert VadEvent.DONE_OK not in events

    def test_two_separate_utterances_fire_two_done_events(self):
        audio = concat(
            make_silence(0.3),
            make_tone_burst(0.5),
            make_silence(1.2),  # ends utterance 1
            make_tone_burst(0.5),
            make_silence(1.2),  # ends utterance 2
        )
        events, _ = simulate_vad_over_audio(audio, rate=16000)
        ok_count = sum(1 for e in events if e is VadEvent.DONE_OK)
        too_short = sum(1 for e in events if e is VadEvent.DONE_TOO_SHORT)
        assert ok_count + too_short == 2

    def test_low_amplitude_speech_stays_idle(self):
        # amp 0.005 → RMS well below 0.02 threshold → never enters speech.
        audio = concat(
            make_silence(0.3),
            make_tone_burst(1.5, amp=0.005),
            make_silence(1.2),
        )
        events, _ = simulate_vad_over_audio(audio, rate=16000)
        assert all(e is VadEvent.IDLE for e in events)


class TestVirtualMicDrivesVad:
    """Push fixture audio into a VirtualMicStream and read it like
    mic_chat.record_utterance_streaming does. The byte-level path matters
    because that's what the production code traverses, not the ndarray.
    """

    def test_full_drain_produces_expected_event_sequence(self):
        mic = VirtualMicStream(rate=16000, chunk_size=1024, pad_with_silence=False)
        audio = concat(
            make_silence(0.3),
            make_tone_burst(1.0, amp=0.3),
            make_silence(1.2),
        )
        mic.push(audio)
        vad = VadState(silence_threshold=0.02, silence_duration=0.8, min_speech_duration=0.3)
        events: list[VadEvent] = []
        frame_idx = 0
        while mic.get_read_available() >= 1024:
            data = mic.read(1024)
            arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            level = float(np.sqrt(np.mean(arr ** 2)))
            now = frame_idx * 1024 / 16000
            events.append(vad.feed(level, now))
            frame_idx += 1
        assert VadEvent.DONE_OK in events


# ---- TTS smoke test (opt-in) -------------------------------------------------

def _kokoro_loadable() -> bool:
    """Try to instantiate kokoro; return False on any failure so the test
    skips cleanly on platforms where torch/kokoro aren't usable.
    """
    try:
        from examples.virtual_audio import _import_kokoro_engine
        _import_kokoro_engine()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _kokoro_loadable(), reason="kokoro TTS not loadable on this host")
class TestTTSFedSimulation:
    def test_synthesized_speech_triggers_done_ok(self):
        """Render a sentence via TTS, push into a virtual mic, drive VAD —
        expect DONE_OK to fire and speech_duration to be plausible.
        """
        mic = VirtualMicStream(rate=16000, chunk_size=1024, pad_with_silence=False)
        feed_tts(mic, "Hello, this is a simulation test.", trailing_silence_s=1.2)
        vad = VadState(silence_threshold=0.02, silence_duration=0.8, min_speech_duration=0.3)
        events: list[VadEvent] = []
        frame_idx = 0
        while mic.get_read_available() >= 1024:
            data = mic.read(1024)
            arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            level = float(np.sqrt(np.mean(arr ** 2)))
            now = frame_idx * 1024 / 16000
            events.append(vad.feed(level, now))
            frame_idx += 1
        assert VadEvent.DONE_OK in events
        # A typical 5-word sentence at speed 1.0 is ~1-2s.
        assert vad.last_speech_duration > 0.5
