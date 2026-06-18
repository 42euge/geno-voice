"""Unit tests for the iter-222 ``base_wpm`` calibration verdict.

iter-220 measured ``implied_base_wpm`` (median + spread + drift), iter-221
surfaced it on the CLI. This lap adds ``calibration_verdict`` — the data-driven
decision of whether the measured base is trustworthy AND significant enough to
re-seed ``DEFAULT_BASE_WPM``, mirroring the iter-219 strength verdict shape.

Loads ``session/wpm_mirror.py`` by file path (the same trick the iter-220
calibration test uses) to bypass ``session/__init__``'s eager pipecat import
(not installable on this x86_64 Linux host).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_WM_PATH = Path(__file__).resolve().parents[2] / "session" / "wpm_mirror.py"
_spec = importlib.util.spec_from_file_location("_wm_verdict_under_test", _WM_PATH)
_wm = importlib.util.module_from_spec(_spec)
sys.modules["_wm_verdict_under_test"] = _wm
_spec.loader.exec_module(_wm)

CalibrationSample = _wm.CalibrationSample
BaseWpmCalibration = _wm.BaseWpmCalibration
calibrate_base_wpm = _wm.calibrate_base_wpm
CalibrationVerdict = _wm.CalibrationVerdict
calibration_verdict = _wm.calibration_verdict
DEFAULT_BASE_WPM = _wm.DEFAULT_BASE_WPM
DEFAULT_CALIB_SPREAD_MAX = _wm.DEFAULT_CALIB_SPREAD_MAX
DEFAULT_CALIB_DRIFT_MIN = _wm.DEFAULT_CALIB_DRIFT_MIN
DEFAULT_CALIB_MIN_SAMPLES = _wm.DEFAULT_CALIB_MIN_SAMPLES


def _calibration(
    implied_base_wpm,
    *,
    n_samples=3,
    spread=0.0,
    default_base_wpm=DEFAULT_BASE_WPM,
):
    """Build a BaseWpmCalibration directly, controlling each gate input."""
    return BaseWpmCalibration(
        implied_base_wpm=implied_base_wpm,
        n_samples=n_samples,
        min_base_wpm=implied_base_wpm - spread / 2.0,
        max_base_wpm=implied_base_wpm + spread / 2.0,
        spread=spread,
        default_base_wpm=default_base_wpm,
        drift=implied_base_wpm - default_base_wpm,
    )


# --------------------------------------------------------------------------
# Defaults sanity
# --------------------------------------------------------------------------

def test_default_thresholds_are_sane():
    assert DEFAULT_CALIB_SPREAD_MAX > 0
    assert DEFAULT_CALIB_DRIFT_MIN > 0
    assert DEFAULT_CALIB_MIN_SAMPLES >= 1


# --------------------------------------------------------------------------
# None passthrough (mirrors calibrate_base_wpm's empty contract)
# --------------------------------------------------------------------------

def test_none_calibration_returns_none():
    assert calibration_verdict(None) is None


# --------------------------------------------------------------------------
# The recommend=True path — all three gates pass
# --------------------------------------------------------------------------

def test_recommends_when_all_gates_pass():
    # Tight spread, plenty of samples, drift well past the threshold.
    cal = _calibration(180.0, n_samples=5, spread=2.0)  # drift +15 vs 165
    v = calibration_verdict(cal)
    assert v is not None
    assert v.recommend is True
    assert "re-seed" in v.reason
    assert "180.0" in v.reason
    assert v.implied_base_wpm == pytest.approx(180.0)


def test_recommends_for_negative_drift_too():
    # A voice slower than nominal still triggers a re-seed if significant.
    cal = _calibration(150.0, n_samples=4, spread=1.0)  # drift -15
    v = calibration_verdict(cal)
    assert v.recommend is True
    assert v.drift == pytest.approx(-15.0)


# --------------------------------------------------------------------------
# Gate 1 — too few samples (checked first)
# --------------------------------------------------------------------------

def test_rejects_too_few_samples():
    # Big drift, tight spread, but only 1 sample.
    cal = _calibration(185.0, n_samples=1, spread=0.0)
    v = calibration_verdict(cal)
    assert v.recommend is False
    assert "sample" in v.reason


def test_sample_gate_is_checked_before_spread_and_drift():
    # Few samples AND wide spread AND tiny drift — sample reason wins.
    cal = _calibration(166.0, n_samples=1, spread=50.0)
    v = calibration_verdict(cal)
    assert v.recommend is False
    assert "sample" in v.reason


def test_exactly_min_samples_passes_sample_gate():
    cal = _calibration(180.0, n_samples=DEFAULT_CALIB_MIN_SAMPLES, spread=1.0)
    v = calibration_verdict(cal)
    assert v.recommend is True


# --------------------------------------------------------------------------
# Gate 2 — renders disagree (spread too wide)
# --------------------------------------------------------------------------

def test_rejects_wide_spread():
    # Enough samples, big drift, but the renders disagree badly.
    cal = _calibration(180.0, n_samples=5, spread=DEFAULT_CALIB_SPREAD_MAX + 1.0)
    v = calibration_verdict(cal)
    assert v.recommend is False
    assert "disagree" in v.reason


def test_spread_exactly_at_max_is_trusted():
    cal = _calibration(180.0, n_samples=5, spread=DEFAULT_CALIB_SPREAD_MAX)
    v = calibration_verdict(cal)
    assert v.recommend is True


def test_spread_gate_checked_before_drift():
    # Enough samples, wide spread, tiny drift — spread reason wins over drift.
    cal = _calibration(166.0, n_samples=5, spread=DEFAULT_CALIB_SPREAD_MAX + 5.0)
    v = calibration_verdict(cal)
    assert v.recommend is False
    assert "disagree" in v.reason


# --------------------------------------------------------------------------
# Gate 3 — drift below the significance threshold
# --------------------------------------------------------------------------

def test_rejects_small_drift():
    # Enough samples, tight spread, but drift is noise.
    cal = _calibration(167.0, n_samples=5, spread=1.0)  # drift +2 vs 165
    v = calibration_verdict(cal)
    assert v.recommend is False
    assert "below" in v.reason


def test_zero_drift_is_rejected():
    cal = _calibration(DEFAULT_BASE_WPM, n_samples=5, spread=0.0)
    v = calibration_verdict(cal)
    assert v.recommend is False
    assert v.drift == pytest.approx(0.0)


def test_drift_exactly_at_min_is_significant():
    cal = _calibration(
        DEFAULT_BASE_WPM + DEFAULT_CALIB_DRIFT_MIN, n_samples=5, spread=1.0
    )
    v = calibration_verdict(cal)
    assert v.recommend is True


def test_negative_drift_just_below_min_is_rejected():
    cal = _calibration(
        DEFAULT_BASE_WPM - (DEFAULT_CALIB_DRIFT_MIN - 0.1), n_samples=5, spread=1.0
    )
    v = calibration_verdict(cal)
    assert v.recommend is False
    assert "below" in v.reason


# --------------------------------------------------------------------------
# Custom thresholds
# --------------------------------------------------------------------------

def test_custom_min_samples():
    cal = _calibration(180.0, n_samples=2, spread=1.0)
    assert calibration_verdict(cal, min_samples=2).recommend is True
    assert calibration_verdict(cal, min_samples=3).recommend is False


def test_custom_spread_max():
    cal = _calibration(180.0, n_samples=5, spread=8.0)
    assert calibration_verdict(cal, spread_max=10.0).recommend is True
    assert calibration_verdict(cal, spread_max=5.0).recommend is False


def test_custom_drift_min():
    cal = _calibration(168.0, n_samples=5, spread=1.0)  # drift +3
    assert calibration_verdict(cal, drift_min=2.0).recommend is True
    assert calibration_verdict(cal, drift_min=5.0).recommend is False


# --------------------------------------------------------------------------
# Echoed fields / result shape
# --------------------------------------------------------------------------

def test_verdict_echoes_calibration_and_thresholds():
    cal = _calibration(180.0, n_samples=5, spread=2.0)
    v = calibration_verdict(cal, spread_max=12.0, drift_min=4.0, min_samples=2)
    assert v.implied_base_wpm == pytest.approx(180.0)
    assert v.drift == pytest.approx(15.0)
    assert v.spread == pytest.approx(2.0)
    assert v.n_samples == 5
    assert v.spread_max == pytest.approx(12.0)
    assert v.drift_min == pytest.approx(4.0)
    assert v.min_samples == 2


def test_verdict_is_frozen():
    cal = _calibration(180.0, n_samples=5, spread=2.0)
    v = calibration_verdict(cal)
    with pytest.raises((AttributeError, TypeError)):
        v.recommend = False  # type: ignore[misc]


# --------------------------------------------------------------------------
# End-to-end over the real iter-220 fold
# --------------------------------------------------------------------------

def test_end_to_end_from_calibrate_base_wpm():
    # Three agreeing renders that all clock ~185 WPM at speed 1.0 — should
    # recommend a re-seed off the 165 nominal.
    samples = [
        CalibrationSample(words=185, audio_seconds=60.0),
        CalibrationSample(words=370, audio_seconds=120.0),
        CalibrationSample(words=92, audio_seconds=30.0),  # ~184
    ]
    cal = calibrate_base_wpm(samples)
    v = calibration_verdict(cal)
    assert v.recommend is True
    assert v.implied_base_wpm == pytest.approx(185.0, abs=2.0)


def test_end_to_end_on_nominal_voice_keeps_seed():
    # A voice that clocks right at the 165 nominal — no re-seed.
    samples = [
        CalibrationSample(words=165, audio_seconds=60.0),
        CalibrationSample(words=165, audio_seconds=60.0),
        CalibrationSample(words=165, audio_seconds=60.0),
    ]
    cal = calibrate_base_wpm(samples)
    v = calibration_verdict(cal)
    assert v.recommend is False
    assert "below" in v.reason


def test_does_not_mutate_calibration():
    cal = _calibration(180.0, n_samples=5, spread=2.0)
    before = (cal.implied_base_wpm, cal.n_samples, cal.spread, cal.drift)
    calibration_verdict(cal)
    after = (cal.implied_base_wpm, cal.n_samples, cal.spread, cal.drift)
    assert before == after


def test_is_deterministic():
    cal = _calibration(180.0, n_samples=5, spread=2.0)
    v1 = calibration_verdict(cal)
    v2 = calibration_verdict(cal)
    assert v1 == v2
