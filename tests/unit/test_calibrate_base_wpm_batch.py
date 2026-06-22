"""Tests for iter-397 — batch base_wpm calibration over a CORPUS of voices.

Where iter-220's ``calibrate_base_wpm`` folds ONE voice's renders into a single
median, ``calibrate_base_wpm_batch`` generalises to N voices — the calibration
analogue of iter-385's ``vad_gap_recommend_batch``. Each voice is calibrated
independently against the shared nominal seed and the per-voice
``implied_base_wpm`` values are summarised by an outlier-robust corpus median,
so an operator picking a fleet-wide ``DEFAULT_BASE_WPM`` sees which voices agree
and which are outliers.

This module pins the pure engine (``BaseWpmCalibrationBatch`` /
``calibrate_base_wpm_batch`` / ``CALIB_BATCH_GRADE_ORDER``). The gv-side human
render is pinned in ``test_gv_calibrate_base_wpm_batch.py``.

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
_spec = importlib.util.spec_from_file_location("_wm_calib_batch_under_test", _WM_PATH)
_wm = importlib.util.module_from_spec(_spec)
sys.modules["_wm_calib_batch_under_test"] = _wm
_spec.loader.exec_module(_wm)

CalibrationSample = _wm.CalibrationSample
BaseWpmCalibrationBatch = _wm.BaseWpmCalibrationBatch
calibrate_base_wpm_batch = _wm.calibrate_base_wpm_batch
CALIB_BATCH_GRADE_ORDER = _wm.CALIB_BATCH_GRADE_ORDER
calibrate_base_wpm = _wm.calibrate_base_wpm
DEFAULT_BASE_WPM = _wm.DEFAULT_BASE_WPM


def _samples(*triples):
    """Build CalibrationSamples from (words, audio_seconds[, speed]) tuples."""
    out = []
    for t in triples:
        if len(t) == 2:
            out.append(CalibrationSample(words=t[0], audio_seconds=t[1]))
        else:
            out.append(CalibrationSample(words=t[0], audio_seconds=t[1], speed=t[2]))
    return out


# --------------------------------------------------------------------------
# shape / counting
# --------------------------------------------------------------------------

def test_empty_corpus_has_no_aggregates():
    batch = calibrate_base_wpm_batch([])
    assert batch.num_voices == 0
    assert batch.num_calibrated == 0
    assert batch.implied_base_wpm_median is None
    assert batch.implied_base_wpm_min is None
    assert batch.implied_base_wpm_max is None
    assert batch.implied_base_wpm_spread is None
    assert batch.rows == ()


def test_one_row_per_voice_in_input_order():
    batch = calibrate_base_wpm_batch(
        [
            ("a", _samples((165, 60.0))),
            ("b", _samples((150, 60.0))),
            ("c", _samples((180, 60.0))),
        ]
    )
    assert batch.num_voices == 3
    assert [r["voice"] for r in batch.rows] == ["a", "b", "c"]


def test_each_row_carries_its_own_calibration():
    # The embedded calibration must agree EXACTLY with calibrate_base_wpm on
    # that voice's own samples.
    samples_a = _samples((165, 60.0), (170, 60.0))
    batch = calibrate_base_wpm_batch([("a", samples_a)])
    solo = calibrate_base_wpm(samples_a)
    assert batch.rows[0]["calibration"] == solo


# --------------------------------------------------------------------------
# corpus median / extremes / spread
# --------------------------------------------------------------------------

def test_corpus_median_is_robust_to_an_outlier():
    # Three voices at ~150/165/180 and one wild outlier at 600 — the median sits
    # with the cluster, not dragged to the outlier the way a mean would be.
    batch = calibrate_base_wpm_batch(
        [
            ("a", _samples((150, 60.0))),
            ("b", _samples((165, 60.0))),
            ("c", _samples((180, 60.0))),
            ("wild", _samples((600, 60.0))),
        ]
    )
    # medians sorted: 150, 165, 180, 600 ⇒ median = (165+180)/2 = 172.5
    assert batch.implied_base_wpm_median == pytest.approx(172.5)
    assert batch.implied_base_wpm_min == pytest.approx(150.0)
    assert batch.implied_base_wpm_max == pytest.approx(600.0)
    assert batch.implied_base_wpm_spread == pytest.approx(450.0)


def test_delta_from_median_is_signed_distance():
    batch = calibrate_base_wpm_batch(
        [
            ("slow", _samples((150, 60.0))),
            ("mid", _samples((165, 60.0))),
            ("fast", _samples((180, 60.0))),
        ]
    )
    # median = 165
    by_voice = {r["voice"]: r for r in batch.rows}
    assert by_voice["slow"]["delta_from_median_wpm"] == pytest.approx(-15.0)
    assert by_voice["mid"]["delta_from_median_wpm"] == pytest.approx(0.0)
    assert by_voice["fast"]["delta_from_median_wpm"] == pytest.approx(15.0)


def test_single_voice_corpus_spread_is_zero():
    batch = calibrate_base_wpm_batch([("solo", _samples((165, 60.0)))])
    assert batch.num_calibrated == 1
    assert batch.implied_base_wpm_median == pytest.approx(165.0)
    assert batch.implied_base_wpm_spread == pytest.approx(0.0)
    assert batch.rows[0]["delta_from_median_wpm"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# uncalibrated voices (no samples)
# --------------------------------------------------------------------------

def test_voice_with_no_samples_is_uncalibrated_and_excluded():
    batch = calibrate_base_wpm_batch(
        [
            ("real", _samples((165, 60.0))),
            ("empty", []),
        ]
    )
    assert batch.num_voices == 2
    assert batch.num_calibrated == 1
    by_voice = {r["voice"]: r for r in batch.rows}
    assert by_voice["empty"]["calibration"] is None
    assert by_voice["empty"]["delta_from_median_wpm"] is None
    # The empty voice does not move the corpus median.
    assert batch.implied_base_wpm_median == pytest.approx(165.0)


def test_all_voices_uncalibrated_has_no_aggregates():
    batch = calibrate_base_wpm_batch([("a", []), ("b", [])])
    assert batch.num_voices == 2
    assert batch.num_calibrated == 0
    assert batch.implied_base_wpm_median is None
    assert batch.grade_counts["uncalibrated"] == 2


# --------------------------------------------------------------------------
# grade histogram
# --------------------------------------------------------------------------

def test_grade_order_is_canonical():
    assert CALIB_BATCH_GRADE_ORDER == ("agree", "loose", "scattered", "uncalibrated")


def test_grade_counts_always_present_and_sum_to_num_voices():
    batch = calibrate_base_wpm_batch(
        [
            ("tight", _samples((165, 60.0), (165, 60.0))),  # zero spread ⇒ agree
            ("empty", []),
        ]
    )
    counts = batch.grade_counts
    assert set(counts) == set(CALIB_BATCH_GRADE_ORDER)
    assert sum(counts.values()) == batch.num_voices == 2
    assert counts["agree"] == 1
    assert counts["uncalibrated"] == 1


def test_scattered_voice_counts_in_its_grade_bucket():
    # A voice whose two renders disagree wildly grades "scattered".
    batch = calibrate_base_wpm_batch(
        [("noisy", _samples((100, 60.0), (300, 60.0)))]
    )
    assert batch.rows[0]["calibration"].dispersion_grade == "scattered"
    assert batch.grade_counts["scattered"] == 1


# --------------------------------------------------------------------------
# nominal / drift threading
# --------------------------------------------------------------------------

def test_nominal_threads_into_every_voice_drift():
    batch = calibrate_base_wpm_batch(
        [("a", _samples((180, 60.0)))], default_base_wpm=160.0
    )
    assert batch.default_base_wpm == pytest.approx(160.0)
    # drift = implied (180) - nominal (160) = +20
    assert batch.rows[0]["calibration"].drift == pytest.approx(20.0)


def test_default_nominal_is_module_default():
    batch = calibrate_base_wpm_batch([("a", _samples((165, 60.0)))])
    assert batch.default_base_wpm == pytest.approx(DEFAULT_BASE_WPM)


# --------------------------------------------------------------------------
# voice-comparability — grade is independent of the voice's absolute rate
# --------------------------------------------------------------------------

def test_grade_is_voice_comparable_across_rates():
    # Two voices with the SAME relative spread at very different absolute rates
    # grade identically — the per-voice grade inherits relative_spread's
    # voice-independence (iter-393/394).
    # slow voice ~100 WPM, 5% relative spread:
    slow = _samples((100, 60.0), (105, 60.0))
    # fast voice ~300 WPM, 5% relative spread:
    fast = _samples((300, 60.0), (315, 60.0))
    batch = calibrate_base_wpm_batch([("slow", slow), ("fast", fast)])
    g = {r["voice"]: r["calibration"].dispersion_grade for r in batch.rows}
    assert g["slow"] == g["fast"]
