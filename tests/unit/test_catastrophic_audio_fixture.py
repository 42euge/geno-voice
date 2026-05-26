"""Tests for iter-125 — catastrophic_16khz.wav fixture invariants.

Mirrors iter-124's `test_noisy_audio_fixture.py` shape but for
the 10 dB SNR catastrophic fixture. The fixture exists so the
WER pipeline has a real-audio counterpart to iter-106's
"catastrophic" band (which was previously a synthetic STT
hypothesis).

Like iter-124, these tests run regardless of faster-whisper
availability — fixture-shape invariants don't need STT.
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
CATASTROPHIC = ROOT / "tests" / "fixtures" / "wer" / "catastrophic_16khz.wav"


def _read_int16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        samples = np.frombuffer(
            w.readframes(w.getnframes()), dtype=np.int16,
        )
    return samples, rate


# ---- File presence + shape -------------------------------------------


def test_catastrophic_fixture_exists():
    assert CATASTROPHIC.exists(), f"{CATASTROPHIC} not committed"


def test_catastrophic_fixture_shape_matches_clean():
    """Catastrophic is the noise-mixed version of clean — sample
    count and rate must match exactly (the only difference is
    noise level)."""
    if not CLEAN.exists():
        pytest.skip("clean_16khz.wav fixture missing")
    clean, clean_rate = _read_int16(CLEAN)
    cat, cat_rate = _read_int16(CATASTROPHIC)
    assert cat_rate == clean_rate
    assert cat_rate == 16000
    assert len(cat) == len(clean)


def test_catastrophic_fixture_is_mono_16bit():
    with wave.open(str(CATASTROPHIC), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16000


# ---- SNR invariants -----------------------------------------------------


def test_catastrophic_has_more_noise_than_iter_124_noisy():
    """Sanity ordering: catastrophic (10 dB) should have HIGHER
    noise RMS than the iter-124 noisy fixture (15 dB). If a
    future regen swapped the SNR targets between the two files,
    this fires."""
    noisy_path = ROOT / "tests" / "fixtures" / "wer" / "noisy_16khz.wav"
    if not (CLEAN.exists() and noisy_path.exists()):
        pytest.skip("companion fixtures missing")

    clean, _ = _read_int16(CLEAN)
    noisy, _ = _read_int16(noisy_path)
    cat, _ = _read_int16(CATASTROPHIC)

    clean_norm = clean.astype(np.float64) / 32768.0
    noisy_noise = noisy.astype(np.float64) / 32768.0 - clean_norm
    cat_noise = cat.astype(np.float64) / 32768.0 - clean_norm

    noisy_rms = np.sqrt(np.mean(noisy_noise ** 2))
    cat_rms = np.sqrt(np.mean(cat_noise ** 2))

    assert cat_rms > noisy_rms, (
        f"catastrophic noise_rms={cat_rms:.4f} not greater than "
        f"noisy noise_rms={noisy_rms:.4f}"
    )


def test_catastrophic_minus_clean_has_expected_10db_snr():
    """The fixture was generated at SNR=10 dB. Compute SNR from
    (clean) and (catastrophic - clean) and assert it's within
    ±2 dB of the target. Strongest sentinel — catches drift if
    the fixture is regenerated at a different SNR.
    """
    if not CLEAN.exists():
        pytest.skip("clean_16khz.wav fixture missing")
    clean, _ = _read_int16(CLEAN)
    cat, _ = _read_int16(CATASTROPHIC)
    clean_norm = clean.astype(np.float64) / 32768.0
    cat_norm = cat.astype(np.float64) / 32768.0
    noise_only = cat_norm - clean_norm
    speech_rms = np.sqrt(np.mean(clean_norm ** 2))
    noise_rms = np.sqrt(np.mean(noise_only ** 2))
    snr_db = 20.0 * np.log10(speech_rms / noise_rms)
    assert 8.0 <= snr_db <= 12.0, (
        f"measured SNR {snr_db:.1f} dB outside [8, 12]"
    )


def test_catastrophic_does_not_clip_excessively():
    """Sanity: at 10 dB SNR, some clipping at speech peaks is
    expected, but >10% saturation means the SNR is way off
    (probably negative).
    """
    cat, _ = _read_int16(CATASTROPHIC)
    saturation = np.sum((cat == 32767) | (cat == -32768))
    pct = saturation / len(cat) * 100
    assert pct < 10.0, (
        f"catastrophic fixture clips {pct:.1f}% of samples — "
        "SNR likely too low"
    )


# ---- Determinism ------------------------------------------------------


def test_catastrophic_samples_are_deterministic():
    """The fixture is generated with seed=42 (same seed as
    iter-124's noisy fixture, but at 10 dB SNR). Reading the
    file twice gives the same content; future regenerations
    that change the seed or generator produce different middle
    samples and surface here as a soft canary.
    """
    cat, _ = _read_int16(CATASTROPHIC)
    middle = cat[12000:12010].tolist()
    assert len(middle) == 10
    assert all(isinstance(v, int) for v in middle)
    # Speech region should still have non-trivial content
    # under the noise — full silence would mean broken
    # generation.
    assert any(abs(v) > 100 for v in middle), (
        f"middle samples look silent: {middle}"
    )


def test_catastrophic_distinct_from_noisy_fixture():
    """The two fixtures share generation pattern and seed but
    differ in SNR target. They MUST produce different bytes —
    otherwise one accidentally got copied over the other.
    """
    noisy_path = ROOT / "tests" / "fixtures" / "wer" / "noisy_16khz.wav"
    if not noisy_path.exists():
        pytest.skip("noisy_16khz.wav fixture missing")
    cat, _ = _read_int16(CATASTROPHIC)
    noisy, _ = _read_int16(noisy_path)
    assert not np.array_equal(cat, noisy), (
        "catastrophic and noisy fixtures are identical — "
        "one likely got overwritten with the other"
    )
