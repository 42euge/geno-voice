"""Tests for iter-127 — multispeaker_16khz.wav fixture invariants.

Mirrors iter-124/iter-125's `test_*_audio_fixture.py` shape but
for the cross-talk fixture: two espeak-ng voices mixed at a
fixed offset and amplitude.

Like iter-124/iter-125, these tests run regardless of
faster-whisper availability — fixture-shape invariants don't
need STT.
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
CATASTROPHIC = ROOT / "tests" / "fixtures" / "wer" / "catastrophic_16khz.wav"
MULTISPEAKER = ROOT / "tests" / "fixtures" / "wer" / "multispeaker_16khz.wav"


def _read_int16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        samples = np.frombuffer(
            w.readframes(w.getnframes()), dtype=np.int16,
        )
    return samples, rate


# ---- File presence + shape -----------------------------------------------


def test_multispeaker_fixture_exists():
    assert MULTISPEAKER.exists(), f"{MULTISPEAKER} not committed"


def test_multispeaker_fixture_is_mono_16bit_16khz():
    with wave.open(str(MULTISPEAKER), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16000


def test_multispeaker_fixture_longer_than_clean():
    """The mix offsets the distractor by 0.4s, so total length
    is at least 1.5s (reference) + small extension. Catches a
    regen that accidentally truncated the distractor."""
    if not CLEAN.exists():
        pytest.skip("clean_16khz.wav fixture missing")
    clean, _ = _read_int16(CLEAN)
    multi, _ = _read_int16(MULTISPEAKER)
    assert len(multi) >= len(clean), (
        f"multispeaker {len(multi)} samples shorter than "
        f"clean {len(clean)} — distractor was truncated"
    )


# ---- Distinct from other fixtures --------------------------------------


def test_multispeaker_distinct_from_clean():
    """Mixing in a second voice produces different bytes than
    the clean original."""
    if not CLEAN.exists():
        pytest.skip("clean_16khz.wav fixture missing")
    clean, _ = _read_int16(CLEAN)
    multi, _ = _read_int16(MULTISPEAKER)
    # Multi is longer — compare the leading region only.
    head = min(len(clean), len(multi))
    assert not np.array_equal(clean[:head], multi[:head]), (
        "multispeaker leading region matches clean — "
        "distractor failed to mix in"
    )


def test_multispeaker_distinct_from_noisy_fixtures():
    """Multispeaker ≠ noisy ≠ catastrophic. All three are
    derived from clean speech but differ in how degradation
    was applied (gaussian noise vs second voice).
    """
    multi, _ = _read_int16(MULTISPEAKER)
    if NOISY.exists():
        noisy, _ = _read_int16(NOISY)
        head = min(len(noisy), len(multi))
        assert not np.array_equal(noisy[:head], multi[:head])
    if CATASTROPHIC.exists():
        cat, _ = _read_int16(CATASTROPHIC)
        head = min(len(cat), len(multi))
        assert not np.array_equal(cat[:head], multi[:head])


# ---- Amplitude / saturation invariants ----------------------------------


def test_multispeaker_does_not_clip_excessively():
    """At distractor amp=0.75, mixing two voices may push
    samples toward saturation but shouldn't push >5% to the
    extremes. Catches a regen that increased the amplitude
    sufficiently to break the test."""
    multi, _ = _read_int16(MULTISPEAKER)
    saturation = np.sum((multi == 32767) | (multi == -32768))
    pct = saturation / len(multi) * 100
    assert pct < 5.0, (
        f"multispeaker fixture clips {pct:.1f}% of samples — "
        "distractor amplitude likely too high"
    )


def test_multispeaker_amp_in_overlap_region():
    """In the overlap region (after t=0.4s), both voices are
    present so the RMS should be HIGHER than in the
    pre-overlap region (only voice A). Sanity check that the
    distractor actually got mixed in.
    """
    multi, _ = _read_int16(MULTISPEAKER)
    rate = 16000
    overlap_start = int(0.4 * rate)
    # Pre-overlap: first 0.3s. Overlap: 0.5s after the offset.
    pre = multi[: int(0.3 * rate)].astype(np.float64) / 32768.0
    overlap_window = multi[overlap_start: overlap_start + int(0.5 * rate)]
    overlap = overlap_window.astype(np.float64) / 32768.0

    pre_rms = np.sqrt(np.mean(pre ** 2))
    overlap_rms = np.sqrt(np.mean(overlap ** 2))

    # The overlap region carries ~150% the energy of pre-overlap.
    # Allow some slack — different speakers have different RMS.
    assert overlap_rms > pre_rms, (
        f"overlap region RMS ({overlap_rms:.4f}) not greater than "
        f"pre-overlap ({pre_rms:.4f}) — distractor likely missing"
    )


# ---- Determinism / generation locked --------------------------------


def test_multispeaker_samples_have_speech_in_distractor_region():
    """Samples in the distractor window (around t=1.0s, 0.6s
    after the distractor starts) must have non-trivial values.
    Catches a regeneration that silenced the distractor."""
    multi, _ = _read_int16(MULTISPEAKER)
    rate = 16000
    # Sample around the middle of the distractor's speech.
    window = multi[int(1.0 * rate): int(1.0 * rate) + 1000]
    assert any(abs(v) > 100 for v in window), (
        f"distractor window looks silent: max abs = {max(abs(window))}"
    )
