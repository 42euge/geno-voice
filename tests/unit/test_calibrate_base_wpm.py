"""Tests for iter-220 — on-device ``base_wpm`` calibration from rendered samples.

iter-219 established that ``base_wpm`` cannot be tuned by replay: it is the
bot's actual ``bot_wpm`` at Kokoro ``speed=1.0`` (the hardware/voice calibration
the simulator *uses* to define the convergence target ``ideal = user_wpm /
base_wpm``). The right value is therefore an on-device measurement: synthesize a
known-length script, measure the audio duration, and back out the implied
``base_wpm``.

This module pins the **pure measurement core** of that calibration:

- ``CalibrationSample`` carries one render (``words``, ``audio_seconds``,
  ``speed``) and derives ``bot_wpm`` (iter-046's ``words·60/audio_seconds``) and
  ``implied_base_wpm`` (that rate normalized back to speed 1.0).
- ``calibrate_base_wpm`` folds one-or-more samples into a robust median
  ``implied_base_wpm`` plus spread/drift diagnostics, so an operator can set
  ``base_wpm`` from their own voice rather than the 165 nominal.

The real Kokoro render that *produces* the samples is the on-device follow-on;
this lap ships the audio-free arithmetic that turns a measured duration into a
``base_wpm`` verdict, unit-tested in isolation exactly like the iter-216/217/219
simulator engine it complements.

Like the rest of the wpm_mirror suite, the module is loaded by file path to
bypass ``session/__init__``'s eager pipecat import (not installable on this
x86_64 Linux runner).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_WM_PATH = Path(__file__).resolve().parents[2] / "session" / "wpm_mirror.py"
_spec = importlib.util.spec_from_file_location("_wm_calib_under_test", _WM_PATH)
_wm = importlib.util.module_from_spec(_spec)
sys.modules["_wm_calib_under_test"] = _wm
_spec.loader.exec_module(_wm)

CalibrationSample = _wm.CalibrationSample
BaseWpmCalibration = _wm.BaseWpmCalibration
calibrate_base_wpm = _wm.calibrate_base_wpm
dispersion_grade = _wm.dispersion_grade
dispersion_margin = _wm.dispersion_margin
CALIB_AGREE_REL_SPREAD = _wm.CALIB_AGREE_REL_SPREAD
CALIB_LOOSE_REL_SPREAD = _wm.CALIB_LOOSE_REL_SPREAD
DEFAULT_BASE_WPM = _wm.DEFAULT_BASE_WPM


# --------------------------------------------------------------------------
# CalibrationSample — derived rate from one render
# --------------------------------------------------------------------------

def test_sample_bot_wpm_is_words_per_minute():
    # 165 words in 60 s ⇒ 165 WPM, the iter-046 convention.
    s = CalibrationSample(words=165, audio_seconds=60.0)
    assert s.bot_wpm == pytest.approx(165.0)


def test_sample_bot_wpm_scales_with_duration():
    # 100 words in 30 s ⇒ 200 WPM.
    s = CalibrationSample(words=100, audio_seconds=30.0)
    assert s.bot_wpm == pytest.approx(200.0)


def test_sample_implied_base_wpm_equals_bot_wpm_at_speed_one():
    # At speed 1.0 the implied base IS the measured rate.
    s = CalibrationSample(words=180, audio_seconds=60.0, speed=1.0)
    assert s.implied_base_wpm == pytest.approx(s.bot_wpm)
    assert s.implied_base_wpm == pytest.approx(180.0)


def test_sample_implied_base_wpm_normalizes_out_speed():
    # Rendered at speed 2.0: measured rate is doubled, so the implied
    # base (rate at speed 1.0) is the measured rate halved.
    s = CalibrationSample(words=200, audio_seconds=30.0, speed=2.0)
    assert s.bot_wpm == pytest.approx(400.0)
    assert s.implied_base_wpm == pytest.approx(200.0)


def test_sample_implied_base_wpm_speed_half():
    # Rendered slow (speed 0.5): measured rate is halved, implied base doubled.
    s = CalibrationSample(words=75, audio_seconds=60.0, speed=0.5)
    assert s.bot_wpm == pytest.approx(75.0)
    assert s.implied_base_wpm == pytest.approx(150.0)


def test_sample_default_speed_is_one():
    s = CalibrationSample(words=10, audio_seconds=5.0)
    assert s.speed == 1.0


@pytest.mark.parametrize(
    "words,audio_seconds,speed",
    [
        (0, 60.0, 1.0),     # no words
        (-5, 60.0, 1.0),    # negative words
        (165, 0.0, 1.0),    # zero duration
        (165, -1.0, 1.0),   # negative duration
        (165, 60.0, 0.0),   # zero speed
        (165, 60.0, -1.0),  # negative speed
    ],
)
def test_sample_rejects_nonpositive_inputs(words, audio_seconds, speed):
    with pytest.raises(ValueError):
        CalibrationSample(words=words, audio_seconds=audio_seconds, speed=speed)


def test_sample_is_frozen():
    s = CalibrationSample(words=165, audio_seconds=60.0)
    with pytest.raises(Exception):
        s.words = 200  # type: ignore[misc]


# --------------------------------------------------------------------------
# calibrate_base_wpm — fold samples into a verdict
# --------------------------------------------------------------------------

def test_calibrate_single_sample():
    cal = calibrate_base_wpm([CalibrationSample(words=170, audio_seconds=60.0)])
    assert cal is not None
    assert cal.implied_base_wpm == pytest.approx(170.0)
    assert cal.n_samples == 1
    assert cal.min_base_wpm == pytest.approx(170.0)
    assert cal.max_base_wpm == pytest.approx(170.0)
    assert cal.spread == pytest.approx(0.0)


def test_calibrate_uses_median():
    # Median of {150, 160, 200} is 160 — robust to the high outlier.
    samples = [
        CalibrationSample(words=150, audio_seconds=60.0),
        CalibrationSample(words=160, audio_seconds=60.0),
        CalibrationSample(words=200, audio_seconds=60.0),
    ]
    cal = calibrate_base_wpm(samples)
    assert cal.implied_base_wpm == pytest.approx(160.0)
    assert cal.n_samples == 3
    assert cal.min_base_wpm == pytest.approx(150.0)
    assert cal.max_base_wpm == pytest.approx(200.0)
    assert cal.spread == pytest.approx(50.0)


def test_calibrate_normalizes_mixed_speeds():
    # Two renders of the same voice at different speeds must agree on base.
    samples = [
        CalibrationSample(words=165, audio_seconds=60.0, speed=1.0),   # 165
        CalibrationSample(words=330, audio_seconds=60.0, speed=2.0),   # 330/2 = 165
    ]
    cal = calibrate_base_wpm(samples)
    assert cal.implied_base_wpm == pytest.approx(165.0)
    assert cal.spread == pytest.approx(0.0)


def test_calibrate_drift_against_default():
    cal = calibrate_base_wpm(
        [CalibrationSample(words=185, audio_seconds=60.0)],
        default_base_wpm=DEFAULT_BASE_WPM,
    )
    # Implied 185 vs nominal 165 ⇒ +20 drift.
    assert cal.default_base_wpm == pytest.approx(165.0)
    assert cal.drift == pytest.approx(20.0)


def test_calibrate_drift_negative_when_voice_slower():
    cal = calibrate_base_wpm([CalibrationSample(words=150, audio_seconds=60.0)])
    assert cal.drift == pytest.approx(150.0 - DEFAULT_BASE_WPM)
    assert cal.drift < 0


def test_calibrate_empty_returns_none():
    assert calibrate_base_wpm([]) is None


def test_calibrate_custom_default():
    cal = calibrate_base_wpm(
        [CalibrationSample(words=170, audio_seconds=60.0)],
        default_base_wpm=170.0,
    )
    assert cal.drift == pytest.approx(0.0)


def test_calibrate_result_is_frozen():
    cal = calibrate_base_wpm([CalibrationSample(words=165, audio_seconds=60.0)])
    with pytest.raises(Exception):
        cal.implied_base_wpm = 200.0  # type: ignore[misc]


def test_calibrate_does_not_mutate_input():
    samples = [
        CalibrationSample(words=150, audio_seconds=60.0),
        CalibrationSample(words=180, audio_seconds=60.0),
    ]
    snapshot = list(samples)
    calibrate_base_wpm(samples)
    assert samples == snapshot


def test_calibrate_is_deterministic():
    samples = [
        CalibrationSample(words=150, audio_seconds=60.0),
        CalibrationSample(words=170, audio_seconds=55.0),
        CalibrationSample(words=200, audio_seconds=58.0),
    ]
    a = calibrate_base_wpm(samples)
    b = calibrate_base_wpm(samples)
    assert a == b


def test_calibrate_even_count_median_averages_middle_two():
    # statistics.median of {150,160,180,200} = (160+180)/2 = 170.
    samples = [
        CalibrationSample(words=150, audio_seconds=60.0),
        CalibrationSample(words=160, audio_seconds=60.0),
        CalibrationSample(words=180, audio_seconds=60.0),
        CalibrationSample(words=200, audio_seconds=60.0),
    ]
    cal = calibrate_base_wpm(samples)
    assert cal.implied_base_wpm == pytest.approx(170.0)


# --------------------------------------------------------------------------
# iter-393 — relative_spread: spread normalized by the median.
#
# The absolute ``spread`` (max − min, in WPM) is base-dependent: a 10 WPM
# spread is tight at a 300-WPM voice but wide at a 100-WPM voice. The
# ``relative_spread`` (= spread / implied_base_wpm) is the dimensionless
# coefficient that lets an operator judge whether the renders AGREE
# independent of the voice's nominal rate. This mirrors the iter-391 move on
# the VAD side (an outlier-aware companion sitting beside the absolute number).
# --------------------------------------------------------------------------

def test_calibrate_relative_spread_is_spread_over_median():
    # spread 50 over median 160 ⇒ 0.3125.
    samples = [
        CalibrationSample(words=150, audio_seconds=60.0),
        CalibrationSample(words=160, audio_seconds=60.0),
        CalibrationSample(words=200, audio_seconds=60.0),
    ]
    cal = calibrate_base_wpm(samples)
    assert cal.spread == pytest.approx(50.0)
    assert cal.relative_spread == pytest.approx(50.0 / 160.0)


def test_calibrate_relative_spread_zero_when_renders_agree():
    samples = [
        CalibrationSample(words=165, audio_seconds=60.0, speed=1.0),
        CalibrationSample(words=330, audio_seconds=60.0, speed=2.0),  # 165
    ]
    cal = calibrate_base_wpm(samples)
    assert cal.spread == pytest.approx(0.0)
    assert cal.relative_spread == pytest.approx(0.0)


def test_calibrate_relative_spread_normalizes_base_dependence():
    # The SAME absolute spread (40 WPM) reads as a much larger relative spread
    # at a slow voice than at a fast one.
    slow = calibrate_base_wpm(
        [
            CalibrationSample(words=80, audio_seconds=60.0),   # 80
            CalibrationSample(words=100, audio_seconds=60.0),  # 100 (median)
            CalibrationSample(words=120, audio_seconds=60.0),  # 120
        ]
    )
    fast = calibrate_base_wpm(
        [
            CalibrationSample(words=280, audio_seconds=60.0),  # 280
            CalibrationSample(words=300, audio_seconds=60.0),  # 300 (median)
            CalibrationSample(words=320, audio_seconds=60.0),  # 320
        ]
    )
    assert slow.spread == pytest.approx(40.0)
    assert fast.spread == pytest.approx(40.0)
    assert slow.relative_spread == pytest.approx(40.0 / 100.0)
    assert fast.relative_spread == pytest.approx(40.0 / 300.0)
    assert slow.relative_spread > fast.relative_spread


def test_calibrate_relative_spread_single_sample_is_zero():
    cal = calibrate_base_wpm([CalibrationSample(words=170, audio_seconds=60.0)])
    assert cal.relative_spread == pytest.approx(0.0)


# --------------------------------------------------------------------------
# iter-394 — dispersion_grade: a categorical trust grade over relative_spread
# --------------------------------------------------------------------------

def test_dispersion_thresholds_are_ordered():
    # The boundaries must be a strictly increasing, positive sequence so the
    # three bands are non-empty and well-ordered.
    assert 0.0 < CALIB_AGREE_REL_SPREAD < CALIB_LOOSE_REL_SPREAD


def test_dispersion_grade_agree_for_tight_spread():
    assert dispersion_grade(0.0) == "agree"
    assert dispersion_grade(CALIB_AGREE_REL_SPREAD / 2.0) == "agree"


def test_dispersion_grade_loose_for_moderate_spread():
    midpoint = (CALIB_AGREE_REL_SPREAD + CALIB_LOOSE_REL_SPREAD) / 2.0
    assert dispersion_grade(midpoint) == "loose"


def test_dispersion_grade_scattered_for_wide_spread():
    assert dispersion_grade(CALIB_LOOSE_REL_SPREAD + 0.05) == "scattered"
    assert dispersion_grade(1.0) == "scattered"


def test_dispersion_grade_boundaries_are_inclusive_lower_band():
    # On each knee the more favourable (lower) band wins.
    assert dispersion_grade(CALIB_AGREE_REL_SPREAD) == "agree"
    assert dispersion_grade(CALIB_LOOSE_REL_SPREAD) == "loose"


def test_calibrate_carries_dispersion_grade():
    # A tight 3-render set (spread 2 over median ~165 ⇒ rel ~0.012) is "agree".
    samples = [
        CalibrationSample(words=164, audio_seconds=60.0),
        CalibrationSample(words=165, audio_seconds=60.0),
        CalibrationSample(words=166, audio_seconds=60.0),
    ]
    cal = calibrate_base_wpm(samples)
    assert cal.relative_spread < CALIB_AGREE_REL_SPREAD
    assert cal.dispersion_grade == "agree"


def test_calibrate_dispersion_grade_matches_helper():
    # The field is exactly dispersion_grade(relative_spread) — no drift.
    samples = [
        CalibrationSample(words=140, audio_seconds=60.0),
        CalibrationSample(words=165, audio_seconds=60.0),
        CalibrationSample(words=200, audio_seconds=60.0),
    ]
    cal = calibrate_base_wpm(samples)
    assert cal.dispersion_grade == dispersion_grade(cal.relative_spread)


def test_calibrate_dispersion_grade_scattered_for_disagreeing_renders():
    # Wide spread (60 over median 160 ⇒ 0.375 > 0.15) ⇒ scattered.
    samples = [
        CalibrationSample(words=130, audio_seconds=60.0),
        CalibrationSample(words=160, audio_seconds=60.0),
        CalibrationSample(words=190, audio_seconds=60.0),
    ]
    cal = calibrate_base_wpm(samples)
    assert cal.relative_spread > CALIB_LOOSE_REL_SPREAD
    assert cal.dispersion_grade == "scattered"


def test_calibrate_dispersion_grade_single_sample_is_agree():
    # One render has zero spread, so it grades "agree" — it cannot disagree with
    # itself; the iter-222 min_samples gate, not this grade, flags "too few".
    cal = calibrate_base_wpm([CalibrationSample(words=170, audio_seconds=60.0)])
    assert cal.dispersion_grade == "agree"


def test_calibrate_dispersion_grade_voice_comparable():
    # The SAME relative spread at a slow and a fast voice grades the same — the
    # grade inherits relative_spread's voice-independence.
    slow = calibrate_base_wpm(
        [
            CalibrationSample(words=90, audio_seconds=60.0),
            CalibrationSample(words=100, audio_seconds=60.0),
            CalibrationSample(words=110, audio_seconds=60.0),
        ]
    )
    fast = calibrate_base_wpm(
        [
            CalibrationSample(words=270, audio_seconds=60.0),
            CalibrationSample(words=300, audio_seconds=60.0),
            CalibrationSample(words=330, audio_seconds=60.0),
        ]
    )
    assert slow.relative_spread == pytest.approx(fast.relative_spread)
    assert slow.dispersion_grade == fast.dispersion_grade == "scattered"


# --------------------------------------------------------------------------
# iter-396 — dispersion_margin: headroom before the grade degrades
# --------------------------------------------------------------------------

def test_dispersion_margin_agree_is_headroom_to_agree_knee():
    # Inside "agree", the margin is the distance up to the agree/loose knee.
    rs = CALIB_AGREE_REL_SPREAD / 2.0
    assert dispersion_grade(rs) == "agree"
    assert dispersion_margin(rs) == pytest.approx(CALIB_AGREE_REL_SPREAD - rs)


def test_dispersion_margin_loose_is_headroom_to_loose_knee():
    # Inside "loose", the margin is the distance up to the loose/scattered knee.
    rs = (CALIB_AGREE_REL_SPREAD + CALIB_LOOSE_REL_SPREAD) / 2.0
    assert dispersion_grade(rs) == "loose"
    assert dispersion_margin(rs) == pytest.approx(CALIB_LOOSE_REL_SPREAD - rs)


def test_dispersion_margin_scattered_is_none():
    # The worst grade has no worse grade to fall to ⇒ margin is None.
    rs = CALIB_LOOSE_REL_SPREAD + 0.1
    assert dispersion_grade(rs) == "scattered"
    assert dispersion_margin(rs) is None


def test_dispersion_margin_zero_on_each_knee():
    # A value sitting exactly on a knee grades the lower band with a 0.0 margin —
    # in the better band, but one hair from leaving it (inclusive-lower-band).
    assert dispersion_grade(CALIB_AGREE_REL_SPREAD) == "agree"
    assert dispersion_margin(CALIB_AGREE_REL_SPREAD) == pytest.approx(0.0)
    assert dispersion_grade(CALIB_LOOSE_REL_SPREAD) == "loose"
    assert dispersion_margin(CALIB_LOOSE_REL_SPREAD) == pytest.approx(0.0)


def test_dispersion_margin_larger_for_tighter_agree():
    # A rock-solid "agree" (tiny relative spread) has more headroom than a
    # knife-edge one near the knee — the whole point of the margin.
    solid = dispersion_margin(0.005)
    knife_edge = dispersion_margin(0.049)
    assert solid > knife_edge > 0.0


def test_calibrate_carries_dispersion_margin():
    # A tight 3-render set is "agree" and carries a positive headroom.
    samples = [
        CalibrationSample(words=164, audio_seconds=60.0),
        CalibrationSample(words=165, audio_seconds=60.0),
        CalibrationSample(words=166, audio_seconds=60.0),
    ]
    cal = calibrate_base_wpm(samples)
    assert cal.dispersion_grade == "agree"
    assert cal.dispersion_margin == pytest.approx(dispersion_margin(cal.relative_spread))
    assert cal.dispersion_margin > 0.0


def test_calibrate_dispersion_margin_none_for_scattered():
    # A scattered calibration carries dispersion_margin None on the field.
    samples = [
        CalibrationSample(words=130, audio_seconds=60.0),
        CalibrationSample(words=160, audio_seconds=60.0),
        CalibrationSample(words=190, audio_seconds=60.0),
    ]
    cal = calibrate_base_wpm(samples)
    assert cal.dispersion_grade == "scattered"
    assert cal.dispersion_margin is None


def test_calibrate_dispersion_margin_matches_helper():
    # The field is exactly dispersion_margin(relative_spread) — no drift.
    samples = [
        CalibrationSample(words=150, audio_seconds=60.0),
        CalibrationSample(words=165, audio_seconds=60.0),
        CalibrationSample(words=178, audio_seconds=60.0),
    ]
    cal = calibrate_base_wpm(samples)
    assert cal.dispersion_margin == pytest.approx(
        dispersion_margin(cal.relative_spread)
    )


def test_dispersion_margin_voice_comparable():
    # The SAME relative spread at a slow and a fast voice yields the same margin
    # — it inherits relative_spread's voice-independence (a function of rel only).
    slow = calibrate_base_wpm(
        [
            CalibrationSample(words=98, audio_seconds=60.0),
            CalibrationSample(words=100, audio_seconds=60.0),
            CalibrationSample(words=102, audio_seconds=60.0),
        ]
    )
    fast = calibrate_base_wpm(
        [
            CalibrationSample(words=294, audio_seconds=60.0),
            CalibrationSample(words=300, audio_seconds=60.0),
            CalibrationSample(words=306, audio_seconds=60.0),
        ]
    )
    assert slow.relative_spread == pytest.approx(fast.relative_spread)
    assert slow.dispersion_margin == pytest.approx(fast.dispersion_margin)
