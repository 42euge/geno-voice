"""iter-189 — Ground-truth VAD regression over real user recordings.

Every ``fixtures/recordings/*.wav`` is a real session captured from the
desktop app. This module replays each one through the current production
VAD parameters (``threshold=0.006``) and asserts the known speech is
detected with a healthy speaking-frame count. The more the user talks to
the app, the more recordings land here — each becomes a regression test
(the data flywheel from the iter-189 steering).

Skips cleanly when the corpus is absent: the recordings are large binary
captures that are NOT committed to the repo (they are rsync'd onto the
machine that runs the loop), so a CI host without them should not fail —
it should skip. The harness logic itself is covered fast and
deterministically by ``tests/unit/test_replay_vad.py`` using synthetic
WAVs, so coverage never depends on the corpus being present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fixtures.replay_vad import VadParams, replay_recording, sweep_grid  # noqa: E402

RECORDINGS_DIR = ROOT / "fixtures" / "recordings"

# Current production parameters (client/voice-capture.js, desktop client).
PROD_PARAMS = VadParams(threshold=0.006)
# Upstream default that under-detected far-field speech (regression guard).
UPSTREAM_PARAMS = VadParams(threshold=0.015)


def _recordings() -> list[Path]:
    if not RECORDINGS_DIR.is_dir():
        return []
    return sorted(RECORDINGS_DIR.glob("*.wav"))


RECORDINGS = _recordings()

# Parametrize by filename so a failing recording is named in the report.
_IDS = [p.name for p in RECORDINGS]

pytestmark = pytest.mark.skipif(
    not RECORDINGS,
    reason=f"no ground-truth recordings in {RECORDINGS_DIR} (rsync'd onto the loop host)",
)


@pytest.fixture(params=RECORDINGS, ids=_IDS)
def recording(request) -> Path:
    return request.param


class TestProductionParams:
    """At threshold 0.006 every real recording must detect its speech."""

    def test_detects_at_least_one_onset(self, recording):
        result = replay_recording(recording, PROD_PARAMS)
        assert result.onsets >= 1, (
            f"{recording.name}: no speech onset detected at threshold "
            f"{PROD_PARAMS.threshold} (peakRMS={result.peak_rms:.4f}, "
            f"meanRMS={result.mean_rms:.4f}, over={result.pct_over_threshold:.1f}%)"
        )

    def test_healthy_speaking_frame_count(self, recording):
        # A real utterance occupies many frames; a handful is suspicious.
        # 30 frames at ~64ms ≈ 2s of committed speech — a conservative floor
        # that still clears all recordings in the seed corpus.
        result = replay_recording(recording, PROD_PARAMS)
        assert result.speaking_frames >= 30, (
            f"{recording.name}: only {result.speaking_frames} speaking frames "
            f"(expected >= 30); detection is anemic"
        )

    def test_known_speech_would_trigger(self, recording):
        result = replay_recording(recording, PROD_PARAMS)
        assert result.known_speech_would_trigger, (
            f"{recording.name}: known speech (meta peak_rms="
            f"{result.meta_peak_rms}) would NOT have triggered the live client"
        )

    def test_some_frames_clear_the_gate(self, recording):
        result = replay_recording(recording, PROD_PARAMS)
        assert result.frames_over_threshold > 0


class TestThresholdRegression:
    """Document the threshold lift: 0.006 must recover at least as much speech
    as the old 0.015 default did. This guards against anyone silently raising
    the threshold back toward the under-detecting value.

    NOTE (iter-231): this asserts monotonicity on *speech recovered*
    (``frames_over_threshold`` / ``speaking_frames``), NOT on onset *count*.
    Onset count is NOT monotonic in threshold on real audio: when a recording's
    inter-utterance noise floor sits between the two thresholds (as
    ``voice-20260617-161615.wav`` does), the LOWER gate keeps that gap above
    threshold and *merges* two utterances into one segment, while the higher
    gate drops the gap below threshold and *splits* them — so the lower
    threshold can legitimately show FEWER onsets while clearing strictly more
    frames. That merging-vs-splitting failure of energy-RMS VAD is exactly the
    dead-end the iter-231 Silero work replaces; counting onsets here was the
    wrong invariant. A lower gate can only ever let MORE frames clear it, so
    ``frames_over_threshold`` is the property that is genuinely monotonic and
    the one this regression guard should pin.
    """

    def test_lower_threshold_recovers_at_least_as_much_speech(self, recording):
        prod = replay_recording(recording, PROD_PARAMS)
        upstream = replay_recording(recording, UPSTREAM_PARAMS)
        # A lower gate is strictly more permissive: every frame over 0.015 is
        # also over 0.006, so the count of gate-clearing frames can only grow.
        assert prod.frames_over_threshold >= upstream.frames_over_threshold, (
            f"{recording.name}: threshold 0.006 cleared "
            f"{prod.frames_over_threshold} frames but 0.015 cleared "
            f"{upstream.frames_over_threshold} — lowering the gate should never "
            f"reduce the frames over threshold"
        )
        # The committed-speaking frames likewise can't drop (the state machine
        # sees at least as many over-threshold frames to commit on).
        assert prod.speaking_frames >= upstream.speaking_frames, (
            f"{recording.name}: threshold 0.006 committed {prod.speaking_frames} "
            f"speaking frames but 0.015 committed {upstream.speaking_frames} — "
            f"lowering the gate should never reduce recovered speech"
        )


class TestPrerollRecoversOpening:
    """iter-191 — A pre-roll buffer must pull each utterance's onset earlier
    (recovering the speech clipped during the onset/debounce window) without
    ever overlapping the previous segment. The aggregate sweep counts don't
    move (pre-roll changes onset *timing*, not onset *count*), so this is the
    test the research doc's backlog item 2 calls for: assert the first
    segment's ``onset_ms`` moves earlier.
    """

    PREROLL = VadParams(threshold=0.006, preroll_ms=512.0)

    def test_first_onset_moves_earlier_or_at_start(self, recording):
        base = replay_recording(recording, PROD_PARAMS)
        pre = replay_recording(recording, self.PREROLL)
        if not base.segments or not pre.segments:
            pytest.skip(f"{recording.name}: no committed segment to compare")
        assert pre.segments[0].onset_ms <= base.segments[0].onset_ms, (
            f"{recording.name}: pre-roll moved the first onset later "
            f"({pre.segments[0].onset_ms:.1f}ms vs {base.segments[0].onset_ms:.1f}ms)"
        )
        # The recovered opening extends the committed segment too.
        assert pre.segments[0].frames >= base.segments[0].frames

    def test_segments_never_overlap_with_preroll(self, recording):
        pre = replay_recording(recording, self.PREROLL)
        for earlier, later in zip(pre.segments, pre.segments[1:]):
            assert later.onset_frame >= earlier.end_frame, (
                f"{recording.name}: pre-roll caused overlapping segments "
                f"({later.onset_frame} < {earlier.end_frame})"
            )

    def test_onset_count_unchanged_by_preroll(self, recording):
        # Pre-roll is a timing recovery, not a detection change: it must not
        # create or destroy onsets.
        base = replay_recording(recording, PROD_PARAMS)
        pre = replay_recording(recording, self.PREROLL)
        assert pre.onsets == base.onsets


class TestGridSweep:
    """iter-192 — A 2-D threshold × gain grid over the corpus must behave
    monotonically along each axis: lowering the threshold (more sensitive) or
    raising the gain (louder signal) can only detect at least as much. This
    pins the joint operating-point search the research doc's backlog item 4
    asks for — a single-axis sweep can't see threshold×gain interaction.
    """

    THRESHOLDS = [0.006, 0.015]
    GAINS = [1.0, 2.0]

    def _grid(self):
        return sweep_grid(
            "threshold", self.THRESHOLDS, "gain", self.GAINS, recordings_dir=RECORDINGS_DIR
        )

    def test_grid_covers_every_cell(self):
        points = self._grid()
        assert len(points) == len(self.THRESHOLDS) * len(self.GAINS)
        for p in points:
            assert p.recordings == len(RECORDINGS)

    def test_lower_threshold_never_detects_fewer_at_fixed_gain(self):
        # Cells are row-major (threshold outer, gain inner). For each gain
        # column, the lower threshold (first row) must detect >= the higher.
        points = self._grid()
        n_gain = len(self.GAINS)
        low_row = points[:n_gain]            # threshold 0.006
        high_row = points[n_gain:2 * n_gain]  # threshold 0.015
        for low, high in zip(low_row, high_row):
            assert low.total_onsets >= high.total_onsets
            assert low.triggered >= high.triggered

    def test_higher_gain_never_detects_fewer_at_fixed_threshold(self):
        # Within each threshold row, more gain can only detect >= onsets.
        points = self._grid()
        n_gain = len(self.GAINS)
        for row_start in range(0, len(points), n_gain):
            row = points[row_start:row_start + n_gain]
            for lo, hi in zip(row, row[1:]):
                assert hi.total_onsets >= lo.total_onsets


def test_corpus_is_non_trivial():
    """Sanity: when present, the corpus has at least one recording and
    every entry has measurable audio (peak RMS above the silence floor)."""
    for path in RECORDINGS:
        result = replay_recording(path, PROD_PARAMS)
        assert result.peak_rms > 0.001, (
            f"{path.name}: peak RMS {result.peak_rms:.5f} is at the silence "
            f"floor — recording may be empty or corrupt"
        )
