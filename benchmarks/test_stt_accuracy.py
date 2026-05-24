"""STT engine accuracy benchmarks (WER).

Requires ground-truth transcripts alongside audio files.
Place a .txt file next to each .wav in test-data/ with the expected transcript.

Run:  pytest benchmarks/test_stt_accuracy.py -v
"""

import io
import wave
from pathlib import Path

import pytest

from stt import WhisperEngine

WHISPER_VARIANTS = [
    pytest.param("mlx-community/whisper-large-v3-turbo", id="turbo", marks=pytest.mark.slow),
    pytest.param("mlx-community/whisper-medium", id="medium", marks=pytest.mark.slow),
    pytest.param("mlx-community/whisper-small", id="small", marks=pytest.mark.quick),
    pytest.param("mlx-community/whisper-tiny", id="tiny", marks=pytest.mark.quick),
]

AUDIO_DIR = Path(__file__).resolve().parent.parent / "test-data"


def _find_samples_with_transcripts():
    """Find WAV files that have a matching .txt ground truth."""
    samples = []
    for wav in AUDIO_DIR.rglob("*.wav"):
        txt = wav.with_suffix(".txt")
        if txt.exists():
            samples.append((wav, txt.read_text().strip()))
    return samples


def _compute_wer(reference: str, hypothesis: str) -> float:
    try:
        from jiwer import wer
        return wer(reference, hypothesis)
    except ImportError:
        pytest.skip("jiwer not installed — run: pip install jiwer")


SAMPLES = _find_samples_with_transcripts()


@pytest.mark.skipif(not SAMPLES, reason="No audio samples with ground-truth .txt files found")
@pytest.mark.parametrize("model_repo", WHISPER_VARIANTS)
def test_word_error_rate(model_repo):
    """Measure WER across all available ground-truth samples."""
    engine = WhisperEngine(model_repo=model_repo)

    wer_scores = []
    for wav_path, reference in SAMPLES:
        audio_bytes = wav_path.read_bytes()
        text, _elapsed = engine.transcribe(audio_bytes)
        if text is None:
            wer_scores.append(1.0)
            continue
        wer_scores.append(_compute_wer(reference, text))

    avg_wer = sum(wer_scores) / len(wer_scores)
    assert avg_wer < 0.5, f"Average WER {avg_wer:.2%} too high for {model_repo}"


@pytest.mark.skipif(not SAMPLES, reason="No audio samples with ground-truth .txt files found")
@pytest.mark.parametrize("model_repo", WHISPER_VARIANTS)
def test_transcription_not_empty(audio_bytes, model_repo):
    """Basic sanity — engine produces non-empty output."""
    engine = WhisperEngine(model_repo=model_repo)
    text, elapsed = engine.transcribe(audio_bytes)
    assert text is not None
    assert len(text.strip()) > 0
    assert elapsed > 0
