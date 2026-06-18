"""iter-231 — Fast, deterministic unit coverage for ``vad/silero.py``.

These tests cover the segmenter's pure logic — dataclasses, params plumbing,
WAV decoding, resampling, and the argument mapping to ``get_speech_timestamps``
— WITHOUT loading the real Silero model or touching the recording corpus. The
heavy "does the real model segment continuous speech" proof lives in
``tests/integration/test_silero_recordings.py`` and runs only when both the
model and the corpus are present.

The real ``silero_vad`` functions are stubbed via ``sys.modules`` injection so
the math (resampling, timestamp passthrough, dataclass derivation) is exercised
deterministically and the call into Silero is asserted to receive the mapped
``SileroParams``.
"""

from __future__ import annotations

import io
import sys
import wave

import numpy as np
import pytest

from vad.silero import (
    SileroParams,
    SileroResult,
    SpeechSegment,
    segment_samples,
    segment_wav_bytes,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSileroModule:
    """Stand-in for the ``silero_vad`` package recording the last call.

    ``get_speech_timestamps`` returns a fixed list and stashes the kwargs it
    was called with so tests can assert the ``SileroParams`` mapping.
    """

    def __init__(self, timestamps):
        self.timestamps = timestamps
        self.last_kwargs = None
        self.load_count = 0

    def install(self, monkeypatch):
        mod = type(sys)("silero_vad")

        def load_silero_vad(*a, **k):
            self.load_count += 1
            return "FAKE_MODEL"

        def get_speech_timestamps(audio, model, **kwargs):
            self.last_kwargs = kwargs
            self.last_audio = audio
            self.last_model = model
            return list(self.timestamps)

        mod.load_silero_vad = load_silero_vad
        mod.get_speech_timestamps = get_speech_timestamps
        monkeypatch.setitem(sys.modules, "silero_vad", mod)
        return self


def _make_wav_bytes(samples: np.ndarray, sample_rate: int, channels: int = 1) -> bytes:
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestSpeechSegment:
    def test_duration_is_end_minus_start(self):
        seg = SpeechSegment(start_s=1.5, end_s=4.0)
        assert seg.duration_s == pytest.approx(2.5)

    def test_to_dict_rounds(self):
        seg = SpeechSegment(start_s=1.23456, end_s=4.98765)
        d = seg.to_dict()
        assert d == {"start_s": 1.235, "end_s": 4.988, "duration_s": 3.753}


class TestSileroResult:
    def _result(self):
        return SileroResult(
            name="r.wav",
            sample_rate=48000,
            duration_s=31.3,
            segments=[
                SpeechSegment(0.0, 2.0),
                SpeechSegment(5.0, 6.5),
            ],
        )

    def test_num_segments_and_speech_s(self):
        r = self._result()
        assert r.num_segments == 2
        assert r.speech_s == pytest.approx(3.5)

    def test_empty_result(self):
        r = SileroResult(name="x", sample_rate=16000, duration_s=1.0)
        assert r.num_segments == 0
        assert r.speech_s == 0.0

    def test_to_dict_shape(self):
        d = self._result().to_dict()
        assert d["num_segments"] == 2
        assert d["speech_s"] == pytest.approx(3.5)
        assert len(d["segments"]) == 2
        assert d["segments"][0] == {"start_s": 0.0, "end_s": 2.0, "duration_s": 2.0}

    def test_summary_line_mentions_counts(self):
        line = self._result().summary_line()
        assert "segs=2" in line
        assert "r.wav" in line

    def test_summary_line_truncates_many_segments(self):
        segs = [SpeechSegment(i, i + 0.5) for i in range(10)]
        r = SileroResult(name="m.wav", sample_rate=16000, duration_s=20.0, segments=segs)
        line = r.summary_line()
        assert "…" in line  # preview capped at 6


class TestSileroParams:
    def test_defaults_match_pipecat_stop_secs(self):
        # The live mic path uses stop_secs=0.8; our default min_silence mirrors it.
        assert SileroParams().min_silence_ms == 800.0
        assert SileroParams().threshold == 0.5

    def test_frozen(self):
        with pytest.raises(Exception):
            SileroParams().threshold = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# segment_samples — argument mapping + timestamp passthrough
# ---------------------------------------------------------------------------


class TestSegmentSamples:
    def test_maps_params_to_get_speech_timestamps(self, monkeypatch):
        fake = _FakeSileroModule([{"start": 0.5, "end": 2.0}]).install(monkeypatch)
        samples = np.zeros(16000, dtype=np.float32)
        params = SileroParams(
            threshold=0.7,
            min_speech_ms=300,
            min_silence_ms=900,
            speech_pad_ms=40,
        )
        segs = segment_samples(samples, 16000, params, model="M")
        assert fake.last_model == "M"
        kw = fake.last_kwargs
        assert kw["threshold"] == 0.7
        assert kw["min_speech_duration_ms"] == 300
        assert kw["min_silence_duration_ms"] == 900
        assert kw["speech_pad_ms"] == 40
        assert kw["sampling_rate"] == 16000
        assert kw["return_seconds"] is True
        assert segs == [SpeechSegment(0.5, 2.0)]

    def test_lazy_loads_model_when_none(self, monkeypatch):
        # Reset the cached singleton so load is exercised.
        import vad.silero as sv

        monkeypatch.setattr(sv, "_MODEL", None)
        fake = _FakeSileroModule([]).install(monkeypatch)
        segment_samples(np.zeros(16000, dtype=np.float32), 16000)
        assert fake.load_count == 1

    def test_empty_samples_short_circuit(self, monkeypatch):
        fake = _FakeSileroModule([{"start": 0, "end": 1}]).install(monkeypatch)
        segs = segment_samples(np.zeros(0, dtype=np.float32), 16000, model="M")
        assert segs == []
        # get_speech_timestamps must not be called on empty audio.
        assert fake.last_kwargs is None

    def test_resamples_non_16k(self, monkeypatch):
        fake = _FakeSileroModule([]).install(monkeypatch)
        # 48k for 1s → after resample the tensor handed to Silero is ~16k samples.
        samples = np.zeros(48000, dtype=np.float32)
        segment_samples(samples, 48000, model="M")
        n = int(fake.last_audio.shape[-1])
        assert abs(n - 16000) <= 16, f"expected ~16000 resampled samples, got {n}"

    def test_inf_max_speech_passed_through(self, monkeypatch):
        fake = _FakeSileroModule([]).install(monkeypatch)
        segment_samples(np.zeros(16000, dtype=np.float32), 16000, model="M")
        assert fake.last_kwargs["max_speech_duration_s"] == float("inf")


# ---------------------------------------------------------------------------
# segment_wav_bytes — decode + derive duration
# ---------------------------------------------------------------------------


class TestSegmentWavBytes:
    def test_decodes_and_segments(self, monkeypatch):
        _FakeSileroModule([{"start": 0.1, "end": 0.9}]).install(monkeypatch)
        wav = _make_wav_bytes(np.zeros(16000, dtype=np.float32), 16000)
        result = segment_wav_bytes(wav, model="M", name="clip.wav")
        assert result.name == "clip.wav"
        assert result.sample_rate == 16000
        assert result.duration_s == pytest.approx(1.0, abs=0.01)
        assert result.num_segments == 1

    def test_downmixes_stereo(self, monkeypatch):
        fake = _FakeSileroModule([]).install(monkeypatch)
        # Two channels interleaved; decode must average to mono of length 8000.
        stereo = np.zeros(16000, dtype=np.float32)  # 8000 frames * 2 ch
        wav = _make_wav_bytes(stereo, 16000, channels=2)
        result = segment_wav_bytes(wav, model="M")
        assert result.duration_s == pytest.approx(0.5, abs=0.01)
        # mono length 8000 at 16k needs no resample → audio handed in is 8000.
        assert int(fake.last_audio.shape[-1]) == 8000
