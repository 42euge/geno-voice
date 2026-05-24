"""STT engine latency benchmarks.

Run all:        pytest benchmarks/test_stt_latency.py -v
Quick only:     pytest benchmarks/test_stt_latency.py -m quick
Single engine:  pytest benchmarks/test_stt_latency.py -k turbo
With JSON out:  pytest benchmarks/test_stt_latency.py --benchmark-json=benchmarks/results/latest.json
"""

import io
import wave

import pytest

from stt import WhisperEngine

WHISPER_VARIANTS = [
    pytest.param(
        "mlx-community/whisper-large-v3-turbo",
        id="turbo",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        "mlx-community/whisper-medium",
        id="medium",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        "mlx-community/whisper-small",
        id="small",
        marks=pytest.mark.quick,
    ),
    pytest.param(
        "mlx-community/whisper-tiny",
        id="tiny",
        marks=pytest.mark.quick,
    ),
]


def _audio_duration_secs(wav_bytes: bytes) -> float:
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _ensure_model_available(engine: WhisperEngine):
    """Skip test if model can't be loaded (not cached + no network)."""
    try:
        engine._load()
    except Exception as e:
        pytest.skip(f"Model unavailable (not cached / no network): {e}")


@pytest.fixture(scope="module")
def audio_duration(audio_bytes):
    return _audio_duration_secs(audio_bytes)


@pytest.mark.parametrize("model_repo", WHISPER_VARIANTS)
def test_transcribe_latency(benchmark, audio_bytes, audio_duration, model_repo):
    """Benchmark transcription latency for a Whisper variant."""
    engine = WhisperEngine(model_repo=model_repo)
    _ensure_model_available(engine)

    def run():
        text, elapsed = engine.transcribe(audio_bytes)
        assert text is not None, f"Transcription returned None for {model_repo}"
        return text, elapsed

    result = benchmark.pedantic(run, warmup_rounds=1, rounds=3, iterations=1)

    text, elapsed = result
    rtf = elapsed / audio_duration
    benchmark.extra_info["real_time_factor"] = round(rtf, 3)
    benchmark.extra_info["audio_duration_s"] = round(audio_duration, 2)
    benchmark.extra_info["transcript_length"] = len(text)


@pytest.mark.parametrize("model_repo", WHISPER_VARIANTS)
def test_real_time_factor(audio_bytes, audio_duration, model_repo):
    """Assert engine is faster than real-time (RTF < 1.0)."""
    engine = WhisperEngine(model_repo=model_repo)
    _ensure_model_available(engine)
    text, elapsed = engine.transcribe(audio_bytes)
    assert text is not None, f"Transcription returned None for {model_repo}"
    rtf = elapsed / audio_duration
    assert rtf < 1.0, f"RTF {rtf:.2f} — slower than real-time for {model_repo}"
