"""iter-189 — Unit tests for the headless VAD replay harness.

These exercise the state-machine port (``frame_rms`` + ``simulate_vad`` +
``replay_recording``) with *synthetic* signals written to tiny temp WAVs,
so they run fast and need none of the large recording fixtures. The
companion ``tests/integration/test_vad_recordings.py`` runs the same
harness against the real ground-truth corpus when it is present.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fixtures.replay_vad import (  # noqa: E402
    VadParams,
    frame_rms,
    simulate_vad,
    load_wav_mono,
    replay_recording,
)


# ---------------------------------------------------------------------------
# Helpers — build synthetic signals and write them to a temp WAV.
# ---------------------------------------------------------------------------


SR = 16000


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = SR) -> Path:
    clamped = np.clip(samples, -1.0, 1.0)
    pcm = (clamped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return path


def _tone(n_samples: int, amplitude: float, freq: float = 220.0, sample_rate: int = SR) -> np.ndarray:
    t = np.arange(n_samples) / sample_rate
    return amplitude * np.sin(2 * np.pi * freq * t)


def _silence(n_samples: int) -> np.ndarray:
    return np.zeros(n_samples, dtype=np.float32)


# ---------------------------------------------------------------------------
# frame_rms
# ---------------------------------------------------------------------------


class TestFrameRms:
    def test_empty_signal_returns_empty(self):
        assert frame_rms(np.zeros(0), frame_size=1024).size == 0

    def test_silence_is_zero_rms(self):
        rms = frame_rms(_silence(4096), frame_size=1024)
        assert rms.size == 4
        assert np.all(rms == 0.0)

    def test_constant_amplitude_rms(self):
        # A constant DC level c has RMS == |c| over any window.
        rms = frame_rms(np.full(2048, 0.5, dtype=np.float32), frame_size=1024)
        assert rms.size == 2
        assert np.allclose(rms, 0.5)

    def test_trailing_partial_frame_included(self):
        # 1024 + 500 samples → 2 frames (the client processes the partial tail).
        rms = frame_rms(np.full(1524, 0.3, dtype=np.float32), frame_size=1024)
        assert rms.size == 2
        assert np.allclose(rms, 0.3)

    def test_gain_scales_rms_linearly(self):
        base = frame_rms(np.full(1024, 0.1, dtype=np.float32), frame_size=1024, gain=1.0)
        amped = frame_rms(np.full(1024, 0.1, dtype=np.float32), frame_size=1024, gain=3.0)
        assert np.allclose(amped, base * 3.0)

    def test_invalid_frame_size_raises(self):
        with pytest.raises(ValueError):
            frame_rms(np.ones(10), frame_size=0)


# ---------------------------------------------------------------------------
# simulate_vad — the state machine
# ---------------------------------------------------------------------------


class TestSimulateVad:
    def _frame_dur(self, frame_size: int = 1024) -> float:
        return (frame_size / SR) * 1000.0  # 64ms at 16k/1024

    def test_all_silence_no_segments(self):
        rms = np.zeros(50)
        segs, speaking = simulate_vad(rms, self._frame_dur(), VadParams())
        assert segs == []
        assert speaking == 0

    def test_sustained_speech_commits_one_segment(self):
        # 64ms/frame; debounce 200ms needs >3 frames; min_speech 500ms needs ~8.
        rms = np.full(40, 0.05)
        segs, speaking = simulate_vad(rms, self._frame_dur(), VadParams())
        assert len(segs) == 1
        assert speaking > 0
        assert segs[0].duration_ms >= VadParams().min_speech_ms

    def test_short_blip_below_debounce_never_commits(self):
        # Only 2 over-threshold frames (~128ms) < 200ms debounce.
        rms = np.concatenate([np.zeros(10), np.full(2, 0.05), np.zeros(10)])
        segs, speaking = simulate_vad(rms, self._frame_dur(), VadParams())
        assert segs == []
        assert speaking == 0

    def test_committed_but_too_short_segment_dropped(self):
        # Commit (debounce needs the candidate to hold > 200ms, ~5 frames)
        # then end on the first silence frame. The segment spans onset
        # (~0ms) to ~384ms — under the 500ms min_speech gate → dropped.
        # Tiny silence_ms ends the segment immediately so it stays short.
        rms = np.concatenate([np.full(5, 0.05), np.zeros(10)])
        params = VadParams(silence_ms=64.0)
        segs, speaking = simulate_vad(rms, self._frame_dur(), params)
        assert segs == []
        # It DID enter speaking state, just got dropped on length.
        assert speaking > 0

    def test_two_segments_split_by_long_silence(self):
        block = np.full(20, 0.05)
        gap = np.zeros(20)  # 20*64ms = 1280ms > 800ms silence → splits
        rms = np.concatenate([block, gap, block])
        segs, _ = simulate_vad(rms, self._frame_dur(), VadParams())
        assert len(segs) == 2

    def test_short_silence_does_not_split(self):
        block = np.full(20, 0.05)
        gap = np.zeros(5)  # 5*64ms = 320ms < 800ms silence → no split
        rms = np.concatenate([block, gap, block])
        segs, _ = simulate_vad(rms, self._frame_dur(), VadParams())
        assert len(segs) == 1

    def test_open_segment_closed_at_eof(self):
        # Speech runs to the very end with no trailing silence.
        rms = np.full(30, 0.05)
        segs, _ = simulate_vad(rms, self._frame_dur(), VadParams())
        assert len(segs) == 1
        assert segs[0].end_frame == len(rms)

    def test_lower_threshold_detects_quiet_speech(self):
        # Quiet speech at 0.008 RMS: misses at 0.015, catches at 0.006.
        rms = np.full(30, 0.008)
        high = simulate_vad(rms, self._frame_dur(), VadParams(threshold=0.015))
        low = simulate_vad(rms, self._frame_dur(), VadParams(threshold=0.006))
        assert high[0] == []
        assert len(low[0]) == 1


# ---------------------------------------------------------------------------
# replay_recording — end-to-end over a synthetic WAV
# ---------------------------------------------------------------------------


class TestReplayRecording:
    def test_speech_then_silence_triggers(self, tmp_path):
        signal = np.concatenate([_silence(SR), _tone(SR, 0.3), _silence(SR)])
        wav = _write_wav(tmp_path / "speech.wav", signal)
        result = replay_recording(wav, VadParams())
        assert result.onsets >= 1
        assert result.speaking_frames > 0
        assert result.pct_over_threshold > 0
        assert result.duration_s == pytest.approx(3.0, abs=0.05)

    def test_pure_silence_does_not_trigger(self, tmp_path):
        wav = _write_wav(tmp_path / "quiet.wav", _silence(2 * SR))
        result = replay_recording(wav, VadParams())
        assert result.onsets == 0
        assert result.speaking_frames == 0
        assert result.known_speech_would_trigger is False

    def test_meta_peak_rms_drives_known_speech_verdict(self, tmp_path):
        signal = np.concatenate([_silence(SR // 2), _tone(SR, 0.3), _silence(SR // 2)])
        wav = _write_wav(tmp_path / "withmeta.wav", signal)
        (tmp_path / "withmeta.json").write_text('{"peak_rms": "0.03", "click_to_capture_ms": "3500"}')
        result = replay_recording(wav, VadParams())
        assert result.meta_peak_rms == pytest.approx(0.03)
        assert result.meta_click_to_capture_ms == pytest.approx(3500)
        assert result.known_speech_would_trigger is True

    def test_missing_meta_is_tolerated(self, tmp_path):
        signal = np.concatenate([_silence(SR // 2), _tone(SR, 0.3), _silence(SR // 2)])
        wav = _write_wav(tmp_path / "nometa.wav", signal)
        result = replay_recording(wav, VadParams())
        assert result.meta_peak_rms is None
        # No metadata to contradict a real onset → still counts as a trigger.
        assert result.known_speech_would_trigger is True

    def test_corrupt_meta_json_is_tolerated(self, tmp_path):
        signal = _tone(SR, 0.3)
        wav = _write_wav(tmp_path / "bad.wav", signal)
        (tmp_path / "bad.json").write_text("{ not valid json ")
        result = replay_recording(wav, VadParams())
        assert result.meta_peak_rms is None

    def test_load_wav_mono_roundtrip(self, tmp_path):
        signal = _tone(SR, 0.5)
        wav = _write_wav(tmp_path / "rt.wav", signal)
        samples, sr = load_wav_mono(wav)
        assert sr == SR
        assert samples.size == signal.size
        assert float(np.max(np.abs(samples))) == pytest.approx(0.5, abs=0.01)
