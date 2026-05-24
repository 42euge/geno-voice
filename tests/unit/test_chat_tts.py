"""Tests for examples/_chat_tts.synthesize_with_alignment.

Two layers:
  1. Unit tests with fake pipeline / engine / token objects.
     Cheap (microseconds), no kokoro / torch dependency.
  2. Integration test with real kokoro. Skipped on hosts where
     kokoro doesn't load. ~1-2s when it does — we keep it because
     it's the only test that exercises the actual production
     synth path end to end.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_tts import TTS_RATE, synthesize_with_alignment  # noqa: E402


# ---- Fake engine / pipeline / tokens ----------------------------------------


class FakeToken:
    """Mimics kokoro's per-token timing object."""

    def __init__(self, text: str, start_ts: float, end_ts: float):
        self.text = text
        self.start_ts = start_ts
        self.end_ts = end_ts


class FakeResult:
    """Mimics one yield from kokoro's pipeline iterator."""

    def __init__(self, audio, tokens):
        self.audio = audio
        self.tokens = list(tokens)


def _fake_engine(results: list[FakeResult]):
    """Build a SimpleNamespace that quacks like kokoro's engine
    enough for synthesize_with_alignment.
    """
    load_calls = {"n": 0}

    def _load():
        load_calls["n"] += 1

    def _pipeline(text, voice, speed):
        # Return a generator so we can verify synthesize_with_alignment
        # iterates exactly once (not, e.g., consuming it twice).
        for r in results:
            yield r

    eng = SimpleNamespace(_load=_load, _pipeline=_pipeline)
    eng._load_calls = load_calls  # for test inspection
    return eng


# ---- Unit tests -------------------------------------------------------------


class TestSynthesizeWithAlignment:
    def test_empty_pipeline_returns_empty_audio_and_tokens(self):
        eng = _fake_engine([])
        audio, tokens = synthesize_with_alignment(eng, "hello", "voice", 1.0)
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) == 0
        assert tokens == []
        # _load was called even on the empty path.
        assert eng._load_calls["n"] == 1

    def test_single_chunk_round_trips_audio(self):
        chunk = np.full(TTS_RATE, 0.5, dtype=np.float32)  # 1 second
        eng = _fake_engine([
            FakeResult(audio=chunk, tokens=[
                FakeToken("Hello", 0.0, 0.5),
                FakeToken("world", 0.5, 1.0),
            ]),
        ])
        audio, tokens = synthesize_with_alignment(eng, "Hello world", "v", 1.0)
        assert len(audio) == TTS_RATE
        assert tokens == [
            {"text": "Hello", "start": 0.0, "end": 0.5},
            {"text": "world", "start": 0.5, "end": 1.0},
        ]

    def test_multi_chunk_offsets_token_timings(self):
        """The whole point of the function: when the pipeline yields
        in multiple chunks, tokens emitted in the second chunk get
        timestamps relative to the START of the full audio, not
        relative to the start of THEIR chunk.
        """
        # Each chunk is 0.5 seconds at TTS_RATE.
        chunk_a = np.full(TTS_RATE // 2, 0.3, dtype=np.float32)
        chunk_b = np.full(TTS_RATE // 2, 0.7, dtype=np.float32)
        eng = _fake_engine([
            FakeResult(audio=chunk_a, tokens=[FakeToken("foo", 0.0, 0.5)]),
            FakeResult(audio=chunk_b, tokens=[FakeToken("bar", 0.0, 0.5)]),
        ])
        audio, tokens = synthesize_with_alignment(eng, "foo bar", "v", 1.0)
        assert len(audio) == TTS_RATE
        # Second chunk's "bar" should start at 0.5 (after chunk_a)
        # and end at 1.0, NOT 0.0 / 0.5.
        assert tokens == [
            {"text": "foo", "start": 0.0, "end": 0.5},
            {"text": "bar", "start": 0.5, "end": 1.0},
        ]

    def test_three_chunks_offset_correctly(self):
        # Verify offset accumulation works for >2 chunks.
        chunks = [
            np.full(TTS_RATE // 4, 0.1, dtype=np.float32),  # 0.25s
            np.full(TTS_RATE // 2, 0.2, dtype=np.float32),  # 0.50s
            np.full(TTS_RATE, 0.3, dtype=np.float32),       # 1.00s
        ]
        eng = _fake_engine([
            FakeResult(audio=chunks[0], tokens=[FakeToken("a", 0.0, 0.25)]),
            FakeResult(audio=chunks[1], tokens=[FakeToken("b", 0.0, 0.5)]),
            FakeResult(audio=chunks[2], tokens=[FakeToken("c", 0.0, 1.0)]),
        ])
        _, tokens = synthesize_with_alignment(eng, "a b c", "v", 1.0)
        assert tokens == [
            {"text": "a", "start": 0.0, "end": 0.25},
            {"text": "b", "start": 0.25, "end": 0.75},
            {"text": "c", "start": 0.75, "end": 1.75},
        ]

    def test_concatenated_audio_preserves_chunk_values(self):
        """Visible boundary check: the concatenated audio should
        contain the chunks back-to-back unchanged.
        """
        chunk_a = np.full(100, 0.1, dtype=np.float32)
        chunk_b = np.full(50, 0.9, dtype=np.float32)
        eng = _fake_engine([
            FakeResult(audio=chunk_a, tokens=[]),
            FakeResult(audio=chunk_b, tokens=[]),
        ])
        audio, _ = synthesize_with_alignment(eng, "x", "v", 1.0)
        assert np.allclose(audio[:100], 0.1)
        assert np.allclose(audio[100:150], 0.9)

    def test_tensor_audio_converted_to_numpy(self):
        """If the engine yields ``torch.Tensor`` audio, the
        function should convert to numpy via the duck-typed
        ``.numpy()`` call. We simulate this without importing
        torch by providing a fake object that has both
        ``.numpy()`` and is NOT an ndarray.
        """
        class FakeTensor:
            def __init__(self, arr):
                self._arr = arr

            def __len__(self):
                return len(self._arr)

            def numpy(self):
                return self._arr

        target = np.full(50, 0.5, dtype=np.float32)
        fake_tensor = FakeTensor(target)
        eng = _fake_engine([FakeResult(audio=fake_tensor, tokens=[])])
        audio, _ = synthesize_with_alignment(eng, "x", "v", 1.0)
        assert isinstance(audio, np.ndarray)
        assert np.array_equal(audio, target)

    def test_numpy_audio_passes_through_without_conversion(self):
        target = np.full(50, 0.5, dtype=np.float32)
        eng = _fake_engine([FakeResult(audio=target, tokens=[])])
        audio, _ = synthesize_with_alignment(eng, "x", "v", 1.0)
        assert isinstance(audio, np.ndarray)
        # Should be equal (concat of single chunk).
        assert np.array_equal(audio, target)

    def test_load_called_each_invocation(self):
        """_load is meant to be idempotent on the engine side; the
        function calls it on every invocation. Document and verify
        the contract.
        """
        eng = _fake_engine([FakeResult(audio=np.zeros(10, dtype=np.float32), tokens=[])])
        synthesize_with_alignment(eng, "a", "v", 1.0)
        synthesize_with_alignment(eng, "b", "v", 1.0)
        assert eng._load_calls["n"] == 2

    def test_results_with_no_tokens_just_contribute_audio(self):
        chunk = np.full(100, 0.2, dtype=np.float32)
        eng = _fake_engine([FakeResult(audio=chunk, tokens=[])])
        audio, tokens = synthesize_with_alignment(eng, "x", "v", 1.0)
        assert len(audio) == 100
        assert tokens == []


# ---- Integration test with real kokoro --------------------------------------


def _kokoro_loadable() -> bool:
    try:
        from examples.virtual_audio import _import_kokoro_engine
        _import_kokoro_engine()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _kokoro_loadable(), reason="kokoro TTS not loadable")
class TestSynthesizeWithRealKokoro:
    def test_short_sentence_produces_plausible_output(self):
        """Real kokoro pipeline. Verifies the contract we depend on:
        engine exposes ``_load`` and ``_pipeline``, results have
        ``.audio`` and ``.tokens`` with the right shapes.
        """
        from examples.virtual_audio import _import_kokoro_engine

        engine = _import_kokoro_engine()
        audio, tokens = synthesize_with_alignment(
            engine, "Hello, world.", "af_heart", 1.0,
        )

        # Audio should be a non-empty numpy float32 array.
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        # Anything from ~0.5s to ~3s for a 2-word utterance.
        assert TTS_RATE * 0.3 < len(audio) < TTS_RATE * 4

        # At least one token, with the right shape.
        assert len(tokens) >= 1
        for t in tokens:
            assert "text" in t and "start" in t and "end" in t
            assert isinstance(t["start"], (int, float))
            assert isinstance(t["end"], (int, float))
            # Within audio duration.
            duration = len(audio) / TTS_RATE
            assert 0 <= t["start"] <= duration + 0.5  # tolerate small overshoot
            # end >= start (sometimes equal for very short tokens).
            assert t["end"] >= t["start"]

    def test_amplitude_in_range(self):
        """Real kokoro audio should be in [-1, 1] (it's float32
        normalized). Catch regressions if a pipeline change breaks
        the convention.
        """
        from examples.virtual_audio import _import_kokoro_engine

        engine = _import_kokoro_engine()
        audio, _ = synthesize_with_alignment(engine, "Hi.", "af_heart", 1.0)
        assert audio.min() >= -1.0
        assert audio.max() <= 1.0
        # Should actually have signal (not just zeros).
        assert float(np.sqrt(np.mean(audio ** 2))) > 0.01
