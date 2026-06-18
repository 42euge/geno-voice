"""iter-228 — Unit tests for the auto-gain recommendation (STEER.md item #2).

The steering asks for a gain that lifts the quietest real speech clearly over
the detection threshold WITHOUT lifting silence over a hard ceiling. A blind
gain sweep can't answer that — it just maximizes onsets. ``recommend_gain``
encodes the constraint as code: pick the largest gain that keeps every
recording's silence floor under the ceiling, then report whether that safe gain
clears the speech target. These tests pin every branch on synthetic corpora
(silence floor over/under the ceiling, quiet speech clearing / not clearing,
the binding-recording selection, CLI text + JSON) so the analyzer stays honest
as new recordings land.
"""

from __future__ import annotations

import json
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
    recommend_gain,
    _silence_floor_rms,
    _quiet_speech_rms,
    GainRecommendation,
    main,
)

SR = 16000


# ---------------------------------------------------------------------------
# Synthetic-signal helpers. Unlike the pure-zeros `_silence` in
# test_replay_vad, these inject a low-amplitude *noise floor* so a recording
# has a non-zero silence floor — the whole point of the gain constraint.
# ---------------------------------------------------------------------------


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
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise_floor(n_samples: int, rms_level: float) -> np.ndarray:
    """A constant-amplitude low signal whose per-frame RMS == ``rms_level``.

    A constant amplitude ``a`` gives RMS ``a`` per frame, so the silence floor
    is exactly ``rms_level`` — deterministic, no randomness needed (and
    ``np.random`` is unavailable to workflow scripts anyway; a constant is the
    robust choice here).
    """
    return np.full(n_samples, rms_level, dtype=np.float32)


def _make_recording(
    path: Path,
    *,
    floor_rms: float,
    speech_amp: float,
    speech_freq: float = 220.0,
    peak_rms: float = 0.05,
) -> Path:
    """Write a recording: a noise floor with one speech burst in the middle."""
    floor = _noise_floor(SR, floor_rms)
    speech = _tone(SR, speech_amp, freq=speech_freq) + floor_rms
    samples = np.concatenate([floor, speech, floor])
    _write_wav(path, samples)
    path.with_suffix(".json").write_text(json.dumps({"peak_rms": peak_rms}))
    return path


# ---------------------------------------------------------------------------
# Floor / quiet-speech estimators
# ---------------------------------------------------------------------------


class TestEstimators:
    def test_silence_floor_empty_is_zero(self):
        assert _silence_floor_rms(np.zeros(0)) == 0.0

    def test_silence_floor_is_median_of_frame_rms(self):
        # Mostly-floor recording: median picks the floor, ignores the rare loud tail.
        rms = np.array([0.001] * 90 + [0.05] * 10)
        assert _silence_floor_rms(rms) == pytest.approx(0.001)

    def test_quiet_speech_empty_is_zero(self):
        assert _quiet_speech_rms(np.zeros(0), threshold=0.006) == 0.0

    def test_quiet_speech_is_low_percentile_of_over_threshold(self):
        rms = np.array([0.0001] * 50 + [0.007, 0.01, 0.02, 0.03])
        # p10 of the four over-threshold frames is near the smallest, 0.007.
        q = _quiet_speech_rms(rms, threshold=0.006)
        assert 0.007 <= q <= 0.01

    def test_quiet_speech_falls_back_when_nothing_over_threshold(self):
        # All speech sits under the gate — the case auto-gain exists to fix.
        rms = np.array([0.001, 0.002, 0.004, 0.005])
        q = _quiet_speech_rms(rms, threshold=0.006)
        # Falls back to the loud tail (p90) so there is a non-zero target.
        assert q > 0.004


# ---------------------------------------------------------------------------
# recommend_gain — core verdict logic
# ---------------------------------------------------------------------------


class TestRecommendGain:
    def test_empty_corpus(self, tmp_path):
        rec = recommend_gain(tmp_path)
        assert isinstance(rec, GainRecommendation)
        assert rec.recommended_gain == 1.0
        assert rec.per_recording == {}
        assert rec.silence_floor == 0.0

    def test_quiet_floor_allows_amplification(self, tmp_path):
        # Floor well under the ceiling: headroom to amplify. quiet speech under
        # the threshold so a gain >1 is genuinely useful.
        corpus = tmp_path / "rec"
        corpus.mkdir()
        _make_recording(corpus / "a.wav", floor_rms=0.0001, speech_amp=0.004)
        rec = recommend_gain(corpus, silence_ceiling=0.0003)
        # Largest safe gain = ceiling / floor ≈ 0.0003 / 0.0001 = 3.0x (int16
        # quantization of the tiny floor widens the tolerance a little).
        assert rec.recommended_gain == pytest.approx(3.0, abs=0.4)
        assert rec.recommended_gain > 1.0
        assert rec.headroom == pytest.approx(3.0, abs=0.4)

    def test_floor_over_ceiling_caps_gain_at_one(self, tmp_path):
        # Floor already above the ceiling — no amplification is safe; never
        # recommend cutting signal (<1.0) to chase a noise target.
        corpus = tmp_path / "rec"
        corpus.mkdir()
        _make_recording(corpus / "a.wav", floor_rms=0.0006, speech_amp=0.02)
        rec = recommend_gain(corpus, silence_ceiling=0.0003)
        assert rec.recommended_gain == 1.0
        assert rec.headroom < 1.0

    def test_max_gain_caps_recommendation(self, tmp_path):
        # Extremely quiet floor would allow huge gain, but max_gain bounds it.
        corpus = tmp_path / "rec"
        corpus.mkdir()
        _make_recording(corpus / "a.wav", floor_rms=0.00001, speech_amp=0.004)
        rec = recommend_gain(corpus, silence_ceiling=0.0003, max_gain=8.0)
        assert rec.recommended_gain == 8.0

    def test_noisiest_recording_binds_the_cap(self, tmp_path):
        # Two recordings, different floors. The LOUDER floor must bind so the
        # recommendation is safe for the whole corpus, not the average.
        corpus = tmp_path / "rec"
        corpus.mkdir()
        _make_recording(corpus / "quiet.wav", floor_rms=0.00005, speech_amp=0.02)
        _make_recording(corpus / "noisy.wav", floor_rms=0.0002, speech_amp=0.02)
        rec = recommend_gain(corpus, silence_ceiling=0.0003)
        # Binding floor is the louder one (0.0002): safe gain = 0.0003/0.0002 = 1.5.
        assert rec.silence_floor == pytest.approx(0.0002, abs=0.00002)
        assert rec.recommended_gain == pytest.approx(1.5, abs=0.2)

    def test_softest_speech_is_the_target(self, tmp_path):
        # quiet_speech is the MIN across recordings (the weakest utterance must
        # still clear), not the mean.
        corpus = tmp_path / "rec"
        corpus.mkdir()
        _make_recording(corpus / "loud.wav", floor_rms=0.00005, speech_amp=0.05)
        _make_recording(corpus / "soft.wav", floor_rms=0.00005, speech_amp=0.008)
        rec = recommend_gain(corpus, silence_ceiling=0.0003)
        # The soft recording's quiet speech is far below the loud one's.
        floors_quiets = list(rec.per_recording.values())
        assert rec.quiet_speech == min(q for _, q in floors_quiets)

    def test_clears_speech_target_true_when_gain_lifts_quiet_speech(self, tmp_path):
        corpus = tmp_path / "rec"
        corpus.mkdir()
        # Floor low enough to allow 3x; quiet speech 0.004 * 3 = 0.012 > 0.006.
        _make_recording(corpus / "a.wav", floor_rms=0.0001, speech_amp=0.0057)
        rec = recommend_gain(corpus, silence_ceiling=0.0003, speech_target=0.006)
        assert rec.clears_speech_target is True

    def test_clears_speech_target_false_when_gain_insufficient(self, tmp_path):
        # Floor near the ceiling (little headroom) AND speech far below target:
        # the safe gain cannot rescue it — gain alone is insufficient.
        corpus = tmp_path / "rec"
        corpus.mkdir()
        _make_recording(corpus / "a.wav", floor_rms=0.00028, speech_amp=0.0012)
        rec = recommend_gain(corpus, silence_ceiling=0.0003, speech_target=0.006)
        assert rec.recommended_gain == pytest.approx(0.0003 / 0.00028, abs=0.1)
        assert rec.clears_speech_target is False

    def test_speech_target_defaults_to_threshold(self, tmp_path):
        corpus = tmp_path / "rec"
        corpus.mkdir()
        _make_recording(corpus / "a.wav", floor_rms=0.0001, speech_amp=0.02)
        rec = recommend_gain(corpus, base=VadParams(threshold=0.009))
        assert rec.speech_target == 0.009

    def test_per_recording_has_floor_and_quiet_for_each(self, tmp_path):
        corpus = tmp_path / "rec"
        corpus.mkdir()
        _make_recording(corpus / "a.wav", floor_rms=0.0001, speech_amp=0.02)
        _make_recording(corpus / "b.wav", floor_rms=0.00012, speech_amp=0.02)
        rec = recommend_gain(corpus)
        assert set(rec.per_recording) == {"a.wav", "b.wav"}
        for floor, quiet in rec.per_recording.values():
            assert floor > 0
            assert quiet > 0


# ---------------------------------------------------------------------------
# summary_line formatting
# ---------------------------------------------------------------------------


class TestSummaryLine:
    def test_ok_verdict_in_line(self, tmp_path):
        corpus = tmp_path / "rec"
        corpus.mkdir()
        _make_recording(corpus / "a.wav", floor_rms=0.0001, speech_amp=0.02)
        rec = recommend_gain(corpus)
        line = rec.summary_line()
        assert "recommended_gain=" in line
        assert "OK" in line
        assert "silence_floor=" in line
        assert "headroom=" in line

    def test_insufficient_verdict_in_line(self, tmp_path):
        corpus = tmp_path / "rec"
        corpus.mkdir()
        _make_recording(corpus / "a.wav", floor_rms=0.00028, speech_amp=0.0012)
        rec = recommend_gain(corpus, speech_target=0.006)
        assert "INSUFFICIENT" in rec.summary_line()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_recommend_gain_text(self, tmp_path, capsys):
        corpus = tmp_path / "rec"
        corpus.mkdir()
        _make_recording(corpus / "a.wav", floor_rms=0.0001, speech_amp=0.02)
        rc = main(["--recommend-gain", "--dir", str(corpus)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "auto-gain recommendation" in out
        assert "recommended_gain=" in out

    def test_recommend_gain_json(self, tmp_path, capsys):
        corpus = tmp_path / "rec"
        corpus.mkdir()
        _make_recording(corpus / "a.wav", floor_rms=0.0001, speech_amp=0.02)
        rc = main(["--recommend-gain", "--json", "--dir", str(corpus)])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "recommended_gain" in payload
        assert "per_recording" in payload
        assert "a.wav" in payload["per_recording"]
        # per_recording values are JSON lists [floor, quiet].
        assert len(payload["per_recording"]["a.wav"]) == 2

    def test_recommend_gain_custom_ceiling(self, tmp_path, capsys):
        corpus = tmp_path / "rec"
        corpus.mkdir()
        _make_recording(corpus / "a.wav", floor_rms=0.0001, speech_amp=0.02)
        rc = main(
            ["--recommend-gain", "--silence-ceiling", "0.0005", "--json", "--dir", str(corpus)]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["silence_ceiling"] == 0.0005
        # Higher ceiling -> more headroom -> larger safe gain (0.0005/0.0001=5).
        assert payload["recommended_gain"] == pytest.approx(5.0, abs=0.5)

    def test_recommend_gain_empty_corpus(self, tmp_path, capsys):
        corpus = tmp_path / "empty"
        corpus.mkdir()
        rc = main(["--recommend-gain", "--dir", str(corpus)])
        assert rc == 1
        assert "No recordings found" in capsys.readouterr().out
