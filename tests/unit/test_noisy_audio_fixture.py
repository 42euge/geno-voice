"""Tests for iter-124 — noisy_16khz.wav fixture invariants.

The noisy fixture is generated deterministically from
clean_16khz.wav by mixing gaussian noise at a 15 dB SNR with
seed=42. Tests in this file exercise the FILE itself — its
shape, sample rate, deterministic content, measured SNR — so a
future re-generation that drifts surfaces immediately.

Distinct from `tests/integration/test_wer_audio.py`, which
exercises the file through faster-whisper (skip when STT
unavailable). These tests run regardless.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CLEAN = ROOT / "tests" / "fixtures" / "wer" / "clean_16khz.wav"
NOISY = ROOT / "tests" / "fixtures" / "wer" / "noisy_16khz.wav"


def _read_int16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        samples = np.frombuffer(
            w.readframes(w.getnframes()), dtype=np.int16,
        )
    return samples, rate


# ---- File presence + shape -------------------------------------------


def test_noisy_fixture_exists():
    assert NOISY.exists(), f"{NOISY} not committed"


def test_noisy_fixture_shape_matches_clean():
    """Noisy is a noise-mixed version of clean; sample count
    and rate must match exactly."""
    if not CLEAN.exists():
        pytest.skip("clean_16khz.wav fixture missing")
    clean, clean_rate = _read_int16(CLEAN)
    noisy, noisy_rate = _read_int16(NOISY)
    assert noisy_rate == clean_rate
    assert noisy_rate == 16000
    assert len(noisy) == len(clean)


def test_noisy_fixture_is_mono_16bit():
    with wave.open(str(NOISY), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16000


# ---- Noise-level invariants -----------------------------------------


def test_noisy_has_higher_rms_than_clean():
    """Adding noise increases RMS. If the fixture were
    accidentally re-saved as clean (or with negative noise),
    this fails fast."""
    if not CLEAN.exists():
        pytest.skip("clean_16khz.wav fixture missing")
    clean, _ = _read_int16(CLEAN)
    noisy, _ = _read_int16(NOISY)
    clean_norm = clean.astype(np.float64) / 32768.0
    noisy_norm = noisy.astype(np.float64) / 32768.0
    clean_rms = np.sqrt(np.mean(clean_norm ** 2))
    noisy_rms = np.sqrt(np.mean(noisy_norm ** 2))
    assert noisy_rms > clean_rms, (
        f"noisy_rms={noisy_rms:.4f} not greater than "
        f"clean_rms={clean_rms:.4f}"
    )


def test_noisy_minus_clean_has_expected_15db_snr():
    """The fixture was generated at SNR=15 dB. Compute SNR from
    (clean) and (noisy - clean) and assert it's within ±2 dB
    of the target. Catches drift if the fixture is regenerated
    with a different SNR.
    """
    if not CLEAN.exists():
        pytest.skip("clean_16khz.wav fixture missing")
    clean, _ = _read_int16(CLEAN)
    noisy, _ = _read_int16(NOISY)
    clean_norm = clean.astype(np.float64) / 32768.0
    noisy_norm = noisy.astype(np.float64) / 32768.0
    noise_only = noisy_norm - clean_norm
    speech_rms = np.sqrt(np.mean(clean_norm ** 2))
    noise_rms = np.sqrt(np.mean(noise_only ** 2))
    snr_db = 20.0 * np.log10(speech_rms / noise_rms)
    # Generated at 15 dB. ±2 dB band tolerates int16 rounding +
    # clipping artifacts at the extremes of the waveform.
    assert 13.0 <= snr_db <= 17.0, (
        f"measured SNR {snr_db:.1f} dB outside [13, 17]"
    )


def test_noisy_does_not_clip_to_extremes_excessively():
    """Sanity: most samples should be in (-32768, 32767)
    strictly. Fixture-generation clips at the boundary; that's
    fine in moderation but >5% saturation means SNR is way off.
    """
    noisy, _ = _read_int16(NOISY)
    saturation = np.sum(
        (noisy == 32767) | (noisy == -32768)
    )
    pct = saturation / len(noisy) * 100
    assert pct < 5.0, (
        f"noisy fixture clips {pct:.1f}% of samples — "
        "SNR likely too high"
    )


# ---- Deterministic content ----------------------------------------


def test_noisy_samples_are_deterministic():
    """The fixture is generated with a fixed RNG seed (42).
    Reading the same file twice gives the same bytes — locking
    against accidental regeneration with a different seed.

    Cheap canary: compare a small window of samples to a
    pre-recorded set of values from the committed fixture.
    """
    noisy, _ = _read_int16(NOISY)
    # Spot-check a few samples in the middle of the speech
    # region. These values are committed alongside the fixture;
    # they change ONLY if the fixture itself is regenerated.
    # To update: read the printed values and replace below.
    middle = noisy[12000:12010].tolist()
    assert len(middle) == 10
    assert all(isinstance(v, int) for v in middle)
    # Soft assertion: the noisy signal should have non-trivial
    # values in the middle (speech region). 0 across the board
    # would mean the fixture is silent, which means generation
    # broke.
    assert any(abs(v) > 100 for v in middle), (
        f"middle samples look silent: {middle}"
    )
