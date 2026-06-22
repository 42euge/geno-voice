"""Tests for iter-405 — on-device STT real-time-factor (RTF) profiling.

The STT-side analogue of the iter-220 ``base_wpm`` calibration family. Where
that folds TTS *render* timings into a robust median base rate, this folds STT
*transcription* timings into a robust median **RTF** (transcribe_seconds /
audio_seconds) plus speed-grade, dispersion, and headroom diagnostics — the
first gv-family analysis surface on the STT pipeline stage (every prior
calibration lap iter-220..404 lived on the TTS side).

Pure arithmetic over injected timings — no torch, no faster-whisper, no audio
I/O — so it runs in the unit gate regardless of platform, exactly like the
``calibrate_base_wpm`` core.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stt.rtf_profile import (  # noqa: E402
    RTF_FAST_MAX,
    RTF_REALTIME_MAX,
    SttRtfProfile,
    TranscriptionSample,
    profile_stt_rtf,
    rtf_speed_grade,
    rtf_speed_margin,
)


# ---- TranscriptionSample ------------------------------------------------


def test_sample_rtf_basic():
    """rtf = transcribe_seconds / audio_seconds."""
    s = TranscriptionSample(audio_seconds=10.0, transcribe_seconds=2.5)
    assert s.rtf == pytest.approx(0.25)


def test_sample_rtf_realtime_boundary():
    """transcribe == audio ⇒ rtf 1.0 (exactly realtime)."""
    s = TranscriptionSample(audio_seconds=4.0, transcribe_seconds=4.0)
    assert s.rtf == pytest.approx(1.0)


def test_sample_rtf_slower_than_realtime():
    s = TranscriptionSample(audio_seconds=2.0, transcribe_seconds=5.0)
    assert s.rtf == pytest.approx(2.5)


@pytest.mark.parametrize("audio", [0.0, -1.0])
def test_sample_rejects_nonpositive_audio(audio):
    with pytest.raises(ValueError, match="audio_seconds"):
        TranscriptionSample(audio_seconds=audio, transcribe_seconds=1.0)


@pytest.mark.parametrize("trans", [0.0, -0.5])
def test_sample_rejects_nonpositive_transcribe(trans):
    with pytest.raises(ValueError, match="transcribe_seconds"):
        TranscriptionSample(audio_seconds=1.0, transcribe_seconds=trans)


# ---- rtf_speed_grade ----------------------------------------------------


def test_grade_fast():
    assert rtf_speed_grade(0.1) == "fast"
    assert rtf_speed_grade(RTF_FAST_MAX) == "fast"  # knee → favourable side


def test_grade_realtime():
    assert rtf_speed_grade(RTF_FAST_MAX + 0.01) == "realtime"
    assert rtf_speed_grade(RTF_REALTIME_MAX) == "realtime"  # knee


def test_grade_slow():
    assert rtf_speed_grade(RTF_REALTIME_MAX + 0.01) == "slow"
    assert rtf_speed_grade(5.0) == "slow"


# ---- rtf_speed_margin ---------------------------------------------------


def test_margin_fast_headroom_to_realtime_knee():
    # fast band: headroom up to RTF_FAST_MAX
    assert rtf_speed_margin(0.2) == pytest.approx(RTF_FAST_MAX - 0.2)


def test_margin_realtime_headroom_to_slow_knee():
    rtf = RTF_FAST_MAX + 0.2
    assert rtf_speed_margin(rtf) == pytest.approx(RTF_REALTIME_MAX - rtf)


def test_margin_slow_is_none():
    """Worst grade has no worse grade to degrade into."""
    assert rtf_speed_margin(RTF_REALTIME_MAX + 0.5) is None


def test_margin_on_knee_is_zero_favourable_side():
    # exactly on the fast/realtime knee grades "fast" with 0 margin
    assert rtf_speed_grade(RTF_FAST_MAX) == "fast"
    assert rtf_speed_margin(RTF_FAST_MAX) == pytest.approx(0.0)


# ---- profile_stt_rtf ----------------------------------------------------


def test_profile_empty_is_none():
    assert profile_stt_rtf([]) is None


def test_profile_single_sample():
    p = profile_stt_rtf([TranscriptionSample(10.0, 3.0)])
    assert isinstance(p, SttRtfProfile)
    assert p.median_rtf == pytest.approx(0.3)
    assert p.n_samples == 1
    assert p.min_rtf == pytest.approx(0.3)
    assert p.max_rtf == pytest.approx(0.3)
    assert p.spread == pytest.approx(0.0)
    assert p.relative_spread == pytest.approx(0.0)
    assert p.speed_grade == "fast"


def test_profile_median_robust_to_outlier():
    # rtfs: 0.2, 0.25, 5.0 ⇒ median 0.25 (mean would be skewed to ~1.8)
    samples = [
        TranscriptionSample(10.0, 2.0),   # 0.2
        TranscriptionSample(10.0, 2.5),   # 0.25
        TranscriptionSample(10.0, 50.0),  # 5.0
    ]
    p = profile_stt_rtf(samples)
    assert p.median_rtf == pytest.approx(0.25)
    assert p.min_rtf == pytest.approx(0.2)
    assert p.max_rtf == pytest.approx(5.0)
    assert p.n_samples == 3


def test_profile_spread_and_relative_spread():
    samples = [
        TranscriptionSample(10.0, 2.0),  # 0.2
        TranscriptionSample(10.0, 4.0),  # 0.4
    ]
    p = profile_stt_rtf(samples)
    assert p.median_rtf == pytest.approx(0.3)
    assert p.spread == pytest.approx(0.2)
    assert p.relative_spread == pytest.approx(0.2 / 0.3)


def test_profile_grade_realtime():
    samples = [TranscriptionSample(10.0, 8.0)]  # rtf 0.8 (0.5 < r <= 1.0)
    p = profile_stt_rtf(samples)
    assert p.median_rtf == pytest.approx(0.8)
    assert p.speed_grade == "realtime"


def test_profile_grade_slow():
    samples = [TranscriptionSample(10.0, 30.0)]  # rtf 3.0
    p = profile_stt_rtf(samples)
    assert p.speed_grade == "slow"
    assert p.speed_margin is None


def test_profile_speed_margin_matches_helper():
    samples = [TranscriptionSample(10.0, 3.0)]  # rtf 0.3
    p = profile_stt_rtf(samples)
    assert p.speed_margin == pytest.approx(rtf_speed_margin(0.3))


def test_profile_accepts_iterable_not_just_list():
    p = profile_stt_rtf(iter([TranscriptionSample(10.0, 3.0)]))
    assert p is not None
    assert p.n_samples == 1
