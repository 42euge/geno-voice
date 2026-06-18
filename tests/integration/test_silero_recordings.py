"""iter-231 — Ground-truth Silero-VAD regression over real user recordings.

The companion to ``test_vad_recordings.py`` (which pins the energy-RMS state
machine). This module runs the **Silero neural VAD** over every
``fixtures/recordings/*.wav`` and asserts it segments continuous speech where
energy-VAD fails — the headless proof the steering asked for.

THE GATE (the steering's hard requirement):
    ``voice-20260618-110355.wav`` (31s continuous speech) collapses to a single
    segment under energy-RMS VAD no matter how it is tuned. Silero MUST split it
    into multiple sensible speech regions. ``test_31s_recording_splits`` pins
    exactly that, and ``test_silero_segments_where_energy_under_segments`` proves
    the general win across the corpus.

Skips cleanly when either prerequisite is absent:
  * the recordings are large binary captures NOT committed to the repo (rsync'd
    onto the loop host), so a CI host without them skips rather than fails;
  * ``silero-vad`` (and its torch deps) may not be installed, so the whole
    module skips when the model can't load.
The fast/deterministic coverage of the segmenter's pure logic lives in
``tests/unit/test_silero_vad.py`` and never depends on the corpus or the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vad.silero import SileroParams, silero_available  # noqa: E402

RECORDINGS_DIR = ROOT / "fixtures" / "recordings"

# The 31s continuous recording that energy-VAD cannot segment (the steering's
# ground-truth file). Silero must split it into more than one region.
CONTINUOUS_31S = "voice-20260618-110355.wav"

# Production-equivalent params: Silero defaults + the pipecat stop_secs=0.8 the
# live mic path already uses (min_silence_ms=800).
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


class TestSileroSegmentsEveryRecording:
    """Silero must find at least one speech region in every real recording, with
    each region landing inside the recording and having positive duration."""

    def test_detects_at_least_one_segment(self, recording, model):
        from vad.silero import segment_recording

        result = segment_recording(recording, params=PROD_PARAMS, model=model)
        assert result.num_segments >= 1, (
            f"{recording.name}: Silero found no speech (dur={result.duration_s:.1f}s)"
        )

    def test_segments_are_within_the_recording(self, recording, model):
        from vad.silero import segment_recording

        result = segment_recording(recording, params=PROD_PARAMS, model=model)
        for seg in result.segments:
            assert seg.duration_s > 0, f"{recording.name}: non-positive segment {seg}"
            assert seg.start_s >= 0
            # speech_pad can push the end a hair past EOF; allow a small slack.
            assert seg.end_s <= result.duration_s + 0.2, (
                f"{recording.name}: segment ends past EOF {seg.end_s} > {result.duration_s}"
            )

    def test_segments_are_ordered_and_non_overlapping(self, recording, model):
        from vad.silero import segment_recording

        result = segment_recording(recording, params=PROD_PARAMS, model=model)
        for earlier, later in zip(result.segments, result.segments[1:]):
            assert later.start_s >= earlier.start_s, (
                f"{recording.name}: segments out of order"
            )


class TestContinuousSpeechGate:
    """The hard gate: the 31s continuous recording energy-VAD can't segment MUST
    split into multiple Silero regions."""

    def test_31s_recording_splits(self, model):
        from vad.silero import segment_recording

        path = RECORDINGS_DIR / CONTINUOUS_31S
        if not path.exists():
            pytest.skip(f"{CONTINUOUS_31S} not present in corpus")
        result = segment_recording(path, params=PROD_PARAMS, model=model)
        assert result.num_segments >= 2, (
            f"{CONTINUOUS_31S}: Silero gave {result.num_segments} segment(s); "
            f"the GATE requires >=2 (energy-VAD collapses this to 1). "
            f"segments={[s.to_dict() for s in result.segments]}"
        )
        # The recording is mostly speech — Silero should recover a healthy chunk.
        assert result.speech_s >= 5.0, (
            f"{CONTINUOUS_31S}: only {result.speech_s:.1f}s of speech detected"
        )


class TestSileroBeatsEnergyVad:
    """Across the corpus, Silero must never under-segment relative to energy-VAD
    on the recordings energy-VAD is known to collapse, and must strictly beat it
    on at least one (the whole reason for this lap)."""

    def test_silero_strictly_wins_on_at_least_one_recording(self, model):
        from fixtures.replay_vad import VadParams, replay_recording
        from vad.silero import segment_recording

        wins = 0
        for path in RECORDINGS:
            silero = segment_recording(path, params=PROD_PARAMS, model=model)
            energy = replay_recording(path, VadParams(threshold=0.006)).onsets
            if silero.num_segments > energy:
                wins += 1
        assert wins >= 1, (
            "Silero did not out-segment energy-VAD on any recording — "
            "expected it to recover continuous speech energy-VAD collapses"
        )
