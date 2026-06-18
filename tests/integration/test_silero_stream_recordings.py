"""iter-232 — Ground-truth streaming-Silero regression over real recordings.

The companion to ``test_silero_recordings.py`` (which pins the BATCH segmenter).
This module drives the **frame-by-frame** ``SileroStream`` / ``stream_samples``
over every ``fixtures/recordings/*.wav`` with the REAL model and asserts:

  1. Streaming reconstructs the SAME segmentation as the batch path — feeding the
     audio window-by-window through ``VADIterator`` must give the same speech
     regions as ``get_speech_timestamps`` over the whole buffer (modulo the
     final-segment flush). This is the proof the live path won't drift from the
     validated batch behaviour.
  2. THE GATE survives streaming: the 31s recording energy-VAD collapses to 1
     segment still splits into >=2 when streamed incrementally.
  3. Chunk size is irrelevant: a 320-sample (20ms @ 16k) mic-frame chunking and
     a 4096-sample chunking yield identical events — the buffering is correct on
     real audio, not just the stub.

Skips cleanly when the corpus or ``silero-vad`` is absent (same guards as the
batch integration test).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vad.silero import SileroParams, silero_available  # noqa: E402

RECORDINGS_DIR = ROOT / "fixtures" / "recordings"
CONTINUOUS_31S = "voice-20260618-110355.wav"
PROD_PARAMS = SileroParams(min_silence_ms=800.0)


def _recordings() -> list[Path]:
    if not RECORDINGS_DIR.is_dir():
        return []
    return sorted(RECORDINGS_DIR.glob("*.wav"))


RECORDINGS = _recordings()
_IDS = [p.name for p in RECORDINGS]

pytestmark = [
    pytest.mark.skipif(
        not RECORDINGS,
        reason=f"no ground-truth recordings in {RECORDINGS_DIR} (rsync'd onto the loop host)",
    ),
    pytest.mark.skipif(
        not silero_available(),
        reason="silero-vad not installed (model cannot load)",
    ),
]


@pytest.fixture(scope="module")
def model():
    from vad.silero import load_model

    return load_model()


@pytest.fixture(params=RECORDINGS, ids=_IDS)
def recording(request) -> Path:
    return request.param


def _stream_recording(path: Path, model, params=PROD_PARAMS, chunk_samples=None):
    """Stream a recording's resampled samples through stream_samples()."""
    from vad.silero import _read_wav_mono, stream_samples

    samples, sr = _read_wav_mono(path.read_bytes())
    return stream_samples(
        samples, sr, params=params, model=model, chunk_samples=chunk_samples
    )


# The batch path's get_speech_timestamps applies min_speech_duration_ms=250 to
# DROP regions shorter than this. The live VADIterator has NO such look-back (it
# emits "start" the instant P>=threshold, so it cannot retroactively suppress a
# region that turns out too short). So the streaming path is a SUPERSET of the
# batch path: identical except it may also emit sub-min_speech blips. Applying
# the same min_speech filter to the stream output reconstructs batch exactly.
MIN_SPEECH_S = PROD_PARAMS.min_speech_ms / 1000.0


def _keep_long(segments, min_s=MIN_SPEECH_S):
    return [s for s in segments if s.duration_s >= min_s]


class TestStreamingMatchesBatch:
    """Streaming window-by-window reconstructs the batch segmentation, once the
    min_speech_ms filter (which the live VADIterator structurally cannot apply)
    is replayed over the stream output.

    This pins the contract documented on ``SileroStream``: the streaming path is
    the batch path WITHOUT min_speech_ms / max_speech_s. Validating it as a
    superset-modulo-min_speech proves the live path won't drift from the
    corpus-validated batch behaviour.
    """

    def test_segment_count_matches_batch_after_min_speech_filter(self, recording, model):
        from vad.silero import segment_recording

        batch = segment_recording(recording, params=PROD_PARAMS, model=model)
        # Reset model state between the two passes (VADIterator and the batch
        # call both mutate the model's internal state).
        model.reset_states()
        stream = _stream_recording(recording, model)
        model.reset_states()
        kept = _keep_long(stream.segments)
        assert len(kept) == batch.num_segments, (
            f"{recording.name}: streaming (>= {MIN_SPEECH_S}s) gave {len(kept)} "
            f"segments, batch gave {batch.num_segments}\n"
            f"  stream(filtered)={[s.to_dict() for s in kept]}\n"
            f"  stream(raw)     ={[s.to_dict() for s in stream.segments]}\n"
            f"  batch           ={[s.to_dict() for s in batch.segments]}"
        )

    def test_stream_is_a_superset_of_batch(self, recording, model):
        """Every batch segment's start appears (within a window) in the raw
        stream output — streaming never MISSES a region batch found."""
        from vad.silero import segment_recording

        batch = segment_recording(recording, params=PROD_PARAMS, model=model)
        model.reset_states()
        stream = _stream_recording(recording, model)
        model.reset_states()
        stream_starts = [s.start_s for s in stream.segments]
        for b_seg in batch.segments:
            assert any(abs(b_seg.start_s - ss) <= 0.05 for ss in stream_starts), (
                f"{recording.name}: batch segment starting {b_seg.start_s}s not "
                f"found in stream starts {stream_starts}"
            )

    def test_segment_starts_align_with_batch(self, recording, model):
        from vad.silero import segment_recording

        batch = segment_recording(recording, params=PROD_PARAMS, model=model)
        model.reset_states()
        stream = _stream_recording(recording, model)
        model.reset_states()
        kept = _keep_long(stream.segments)
        if len(kept) != batch.num_segments:
            pytest.skip("count mismatch covered by the count test")
        for s_seg, b_seg in zip(kept, batch.segments):
            # Window-quantization can shift a boundary by <= one window (32ms).
            assert abs(s_seg.start_s - b_seg.start_s) <= 0.05, (
                f"{recording.name}: start drift {s_seg.start_s} vs {b_seg.start_s}"
            )


class TestStreamingGate:
    """The hard gate survives streaming: the 31s recording still splits >=2."""

    def test_31s_recording_splits_when_streamed(self, model):
        path = RECORDINGS_DIR / CONTINUOUS_31S
        if not path.exists():
            pytest.skip(f"{CONTINUOUS_31S} not present in corpus")
        model.reset_states()
        result = _stream_recording(path, model)
        model.reset_states()
        assert result.num_segments >= 2, (
            f"{CONTINUOUS_31S}: streaming gave {result.num_segments} segment(s); "
            f"the GATE requires >=2. segments={[s.to_dict() for s in result.segments]}"
        )
        assert result.speech_s >= 5.0


class TestChunkSizeIndependenceOnRealAudio:
    """Mic-frame (320-sample) vs bulk (4096-sample) chunking → identical events."""

    def test_chunking_is_irrelevant(self, recording, model):
        model.reset_states()
        fine = _stream_recording(recording, model, chunk_samples=320)
        model.reset_states()
        coarse = _stream_recording(recording, model, chunk_samples=4096)
        model.reset_states()
        fine_segs = [(round(s.start_s, 3), round(s.end_s, 3)) for s in fine.segments]
        coarse_segs = [(round(s.start_s, 3), round(s.end_s, 3)) for s in coarse.segments]
        assert fine_segs == coarse_segs, (
            f"{recording.name}: chunk size changed segmentation\n"
            f"  fine(320)  ={fine_segs}\n"
            f"  coarse(4096)={coarse_segs}"
        )
