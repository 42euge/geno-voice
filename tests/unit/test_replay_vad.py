"""iter-189 — Unit tests for the headless VAD replay harness.

These exercise the state-machine port (``frame_rms`` + ``simulate_vad`` +
``replay_recording``) with *synthetic* signals written to tiny temp WAVs,
so they run fast and need none of the large recording fixtures. The
companion ``tests/integration/test_vad_recordings.py`` runs the same
harness against the real ground-truth corpus when it is present.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fixtures.replay_vad import (  # noqa: E402
    VadParams,
    frame_rms,
    simulate_vad,
    load_wav_mono,
    replay_recording,
    replay_all,
    aggregate_results,
    sweep_param,
    sweep_grid,
    _parse_value_list,
    main,
)


# ---------------------------------------------------------------------------
# Helpers — build synthetic signals and write them to a temp WAV.
# ---------------------------------------------------------------------------


SR = 16000


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = SR) -> Path:
    clamped = np.clip(samples, -1.0, 1.0)
    pcm = (clamped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return path


def _tone(n_samples: int, amplitude: float, freq: float = 220.0, sample_rate: int = SR) -> np.ndarray:
    t = np.arange(n_samples) / sample_rate
    return amplitude * np.sin(2 * np.pi * freq * t)


def _silence(n_samples: int) -> np.ndarray:
    return np.zeros(n_samples, dtype=np.float32)


# ---------------------------------------------------------------------------
# frame_rms
# ---------------------------------------------------------------------------


class TestFrameRms:
    def test_empty_signal_returns_empty(self):
        assert frame_rms(np.zeros(0), frame_size=1024).size == 0

    def test_silence_is_zero_rms(self):
        rms = frame_rms(_silence(4096), frame_size=1024)
        assert rms.size == 4
        assert np.all(rms == 0.0)

    def test_constant_amplitude_rms(self):
        # A constant DC level c has RMS == |c| over any window.
        rms = frame_rms(np.full(2048, 0.5, dtype=np.float32), frame_size=1024)
        assert rms.size == 2
        assert np.allclose(rms, 0.5)

    def test_trailing_partial_frame_included(self):
        # 1024 + 500 samples → 2 frames (the client processes the partial tail).
        rms = frame_rms(np.full(1524, 0.3, dtype=np.float32), frame_size=1024)
        assert rms.size == 2
        assert np.allclose(rms, 0.3)

    def test_gain_scales_rms_linearly(self):
        base = frame_rms(np.full(1024, 0.1, dtype=np.float32), frame_size=1024, gain=1.0)
        amped = frame_rms(np.full(1024, 0.1, dtype=np.float32), frame_size=1024, gain=3.0)
        assert np.allclose(amped, base * 3.0)

    def test_invalid_frame_size_raises(self):
        with pytest.raises(ValueError):
            frame_rms(np.ones(10), frame_size=0)


# ---------------------------------------------------------------------------
# simulate_vad — the state machine
# ---------------------------------------------------------------------------


class TestSimulateVad:
    def _frame_dur(self, frame_size: int = 1024) -> float:
        return (frame_size / SR) * 1000.0  # 64ms at 16k/1024

    def test_all_silence_no_segments(self):
        rms = np.zeros(50)
        segs, speaking = simulate_vad(rms, self._frame_dur(), VadParams())
        assert segs == []
        assert speaking == 0

    def test_sustained_speech_commits_one_segment(self):
        # 64ms/frame; debounce 200ms needs >3 frames; min_speech 500ms needs ~8.
        rms = np.full(40, 0.05)
        segs, speaking = simulate_vad(rms, self._frame_dur(), VadParams())
        assert len(segs) == 1
        assert speaking > 0
        assert segs[0].duration_ms >= VadParams().min_speech_ms

    def test_short_blip_below_debounce_never_commits(self):
        # Only 2 over-threshold frames (~128ms) < 200ms debounce.
        rms = np.concatenate([np.zeros(10), np.full(2, 0.05), np.zeros(10)])
        segs, speaking = simulate_vad(rms, self._frame_dur(), VadParams())
        assert segs == []
        assert speaking == 0

    def test_committed_but_too_short_segment_dropped(self):
        # Commit (debounce needs the candidate to hold > 200ms, ~5 frames)
        # then end on the first silence frame. The segment spans onset
        # (~0ms) to ~384ms — under the 500ms min_speech gate → dropped.
        # Tiny silence_ms ends the segment immediately so it stays short.
        rms = np.concatenate([np.full(5, 0.05), np.zeros(10)])
        params = VadParams(silence_ms=64.0)
        segs, speaking = simulate_vad(rms, self._frame_dur(), params)
        assert segs == []
        # It DID enter speaking state, just got dropped on length.
        assert speaking > 0

    def test_two_segments_split_by_long_silence(self):
        block = np.full(20, 0.05)
        gap = np.zeros(20)  # 20*64ms = 1280ms > 800ms silence → splits
        rms = np.concatenate([block, gap, block])
        segs, _ = simulate_vad(rms, self._frame_dur(), VadParams())
        assert len(segs) == 2

    def test_short_silence_does_not_split(self):
        block = np.full(20, 0.05)
        gap = np.zeros(5)  # 5*64ms = 320ms < 800ms silence → no split
        rms = np.concatenate([block, gap, block])
        segs, _ = simulate_vad(rms, self._frame_dur(), VadParams())
        assert len(segs) == 1

    def test_open_segment_closed_at_eof(self):
        # Speech runs to the very end with no trailing silence.
        rms = np.full(30, 0.05)
        segs, _ = simulate_vad(rms, self._frame_dur(), VadParams())
        assert len(segs) == 1
        assert segs[0].end_frame == len(rms)

    def test_lower_threshold_detects_quiet_speech(self):
        # Quiet speech at 0.008 RMS: misses at 0.015, catches at 0.006.
        rms = np.full(30, 0.008)
        high = simulate_vad(rms, self._frame_dur(), VadParams(threshold=0.015))
        low = simulate_vad(rms, self._frame_dur(), VadParams(threshold=0.006))
        assert high[0] == []
        assert len(low[0]) == 1


# ---------------------------------------------------------------------------
# simulate_vad — pre-roll buffer (backlog item 2)
# ---------------------------------------------------------------------------


class TestPrerollBuffer:
    def _frame_dur(self, frame_size: int = 1024) -> float:
        return (frame_size / SR) * 1000.0  # 64ms at 16k/1024

    def test_zero_preroll_leaves_onset_unchanged(self):
        # preroll_ms=0 (the default) reproduces today's clip-the-opening behaviour.
        rms = np.concatenate([np.zeros(10), np.full(30, 0.05)])
        segs, _ = simulate_vad(rms, self._frame_dur(), VadParams(preroll_ms=0.0))
        base, _ = simulate_vad(rms, self._frame_dur(), VadParams())
        assert len(segs) == 1
        assert segs[0].onset_frame == base[0].onset_frame
        assert segs[0].onset_ms == base[0].onset_ms

    def test_preroll_moves_onset_earlier(self):
        # 10 silent frames then sustained speech. Pre-roll should pull the
        # emitted onset back into the silent lead-in.
        rms = np.concatenate([np.zeros(10), np.full(30, 0.05)])
        frame_dur = self._frame_dur()
        no_pre, _ = simulate_vad(rms, frame_dur, VadParams(preroll_ms=0.0))
        # 192ms ≈ 3 frames of pre-roll.
        pre, _ = simulate_vad(rms, frame_dur, VadParams(preroll_ms=192.0))
        assert pre[0].onset_frame == no_pre[0].onset_frame - 3
        assert pre[0].onset_ms < no_pre[0].onset_ms
        # The extended segment also covers more frames.
        assert pre[0].frames > no_pre[0].frames

    def test_preroll_clamps_to_recording_start(self):
        # Speech from frame 0 — there is no lead-in, so pre-roll can't go
        # below frame 0 even with a large preroll_ms.
        rms = np.full(30, 0.05)
        segs, _ = simulate_vad(rms, self._frame_dur(), VadParams(preroll_ms=5000.0))
        assert len(segs) == 1
        assert segs[0].onset_frame == 0
        assert segs[0].onset_ms == 0.0

    def test_preroll_clamps_to_previous_segment_end(self):
        # Two segments split by a long silence. The second segment's pre-roll
        # must not reach back past the first segment's end (no overlap).
        block = np.full(20, 0.05)
        gap = np.zeros(20)  # 1280ms > 800ms → splits
        rms = np.concatenate([block, gap, block])
        frame_dur = self._frame_dur()
        # Huge pre-roll that would otherwise swallow the whole gap.
        segs, _ = simulate_vad(rms, frame_dur, VadParams(preroll_ms=10_000.0))
        assert len(segs) == 2
        assert segs[1].onset_frame >= segs[0].end_frame

    def test_preroll_does_not_rescue_too_short_segment(self):
        # The min_speech gate measures the *committed* speech, not the
        # pre-roll padding: a segment that is only long because of pre-roll
        # is still dropped.
        rms = np.concatenate([np.full(5, 0.05), np.zeros(10)])
        params = VadParams(silence_ms=64.0, preroll_ms=10_000.0)
        segs, speaking = simulate_vad(rms, self._frame_dur(), params)
        assert segs == []
        assert speaking > 0


# ---------------------------------------------------------------------------
# replay_recording — end-to-end over a synthetic WAV
# ---------------------------------------------------------------------------


class TestReplayRecording:
    def test_speech_then_silence_triggers(self, tmp_path):
        signal = np.concatenate([_silence(SR), _tone(SR, 0.3), _silence(SR)])
        wav = _write_wav(tmp_path / "speech.wav", signal)
        result = replay_recording(wav, VadParams())
        assert result.onsets >= 1
        assert result.speaking_frames > 0
        assert result.pct_over_threshold > 0
        assert result.duration_s == pytest.approx(3.0, abs=0.05)

    def test_pure_silence_does_not_trigger(self, tmp_path):
        wav = _write_wav(tmp_path / "quiet.wav", _silence(2 * SR))
        result = replay_recording(wav, VadParams())
        assert result.onsets == 0
        assert result.speaking_frames == 0
        assert result.known_speech_would_trigger is False

    def test_meta_peak_rms_drives_known_speech_verdict(self, tmp_path):
        signal = np.concatenate([_silence(SR // 2), _tone(SR, 0.3), _silence(SR // 2)])
        wav = _write_wav(tmp_path / "withmeta.wav", signal)
        (tmp_path / "withmeta.json").write_text('{"peak_rms": "0.03", "click_to_capture_ms": "3500"}')
        result = replay_recording(wav, VadParams())
        assert result.meta_peak_rms == pytest.approx(0.03)
        assert result.meta_click_to_capture_ms == pytest.approx(3500)
        assert result.known_speech_would_trigger is True

    def test_missing_meta_is_tolerated(self, tmp_path):
        signal = np.concatenate([_silence(SR // 2), _tone(SR, 0.3), _silence(SR // 2)])
        wav = _write_wav(tmp_path / "nometa.wav", signal)
        result = replay_recording(wav, VadParams())
        assert result.meta_peak_rms is None
        # No metadata to contradict a real onset → still counts as a trigger.
        assert result.known_speech_would_trigger is True

    def test_corrupt_meta_json_is_tolerated(self, tmp_path):
        signal = _tone(SR, 0.3)
        wav = _write_wav(tmp_path / "bad.wav", signal)
        (tmp_path / "bad.json").write_text("{ not valid json ")
        result = replay_recording(wav, VadParams())
        assert result.meta_peak_rms is None

    def test_load_wav_mono_roundtrip(self, tmp_path):
        signal = _tone(SR, 0.5)
        wav = _write_wav(tmp_path / "rt.wav", signal)
        samples, sr = load_wav_mono(wav)
        assert sr == SR
        assert samples.size == signal.size
        assert float(np.max(np.abs(samples))) == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Helpers for the sweep tests — build a tiny synthetic corpus on disk.
# ---------------------------------------------------------------------------


def _make_corpus(tmp_path: Path) -> Path:
    """A 2-recording corpus: one loud (clear speech), one quiet (far-field).

    The quiet recording sits between the 0.006 and 0.015 thresholds so a
    threshold sweep shows a real detection difference across values.
    """
    corpus = tmp_path / "rec"
    corpus.mkdir()
    loud = np.concatenate([_silence(SR), _tone(SR, 0.3), _silence(SR)])
    _write_wav(corpus / "loud.wav", loud)
    (corpus / "loud.json").write_text('{"peak_rms": 0.05}')
    # ~0.0085 RMS sine: clears 0.006, misses 0.015.
    quiet = np.concatenate([_silence(SR // 2), _tone(SR, 0.012), _silence(SR // 2)])
    _write_wav(corpus / "quiet.wav", quiet)
    (corpus / "quiet.json").write_text('{"peak_rms": 0.0085}')
    return corpus


# ---------------------------------------------------------------------------
# aggregate_results
# ---------------------------------------------------------------------------


class TestAggregateResults:
    def test_empty_corpus_aggregates_to_zeros(self):
        point = aggregate_results(VadParams(), [])
        assert point.recordings == 0
        assert point.triggered == 0
        assert point.total_onsets == 0
        assert point.total_speaking_frames == 0
        assert point.mean_pct_over == 0.0
        assert point.min_onsets == 0
        assert point.max_onsets == 0
        assert point.mean_first_onset_ms == 0.0
        assert point.max_first_onset_ms == 0.0
        assert point.min_first_onset_ms == 0.0
        assert point.std_first_onset_ms == 0.0
        assert point.max_segment_ms == 0.0

    def test_aggregates_across_corpus(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        results = replay_all(corpus, VadParams(threshold=0.006))
        point = aggregate_results(VadParams(threshold=0.006), results)
        assert point.recordings == 2
        assert point.total_onsets == sum(r.onsets for r in results)
        assert point.total_speaking_frames == sum(r.speaking_frames for r in results)
        assert point.min_onsets == min(r.onsets for r in results)
        # min_onsets is the worst single recording, never above the total.
        assert point.min_onsets <= point.total_onsets

    def test_max_onsets_is_busiest_single_recording(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        results = replay_all(corpus, VadParams(threshold=0.006))
        point = aggregate_results(VadParams(threshold=0.006), results)
        # The over-split ceiling is the most onsets any single recording got.
        assert point.max_onsets == max(r.onsets for r in results)
        # It brackets the per-recording count from above: min <= max, and the
        # ceiling never exceeds the corpus total.
        assert point.min_onsets <= point.max_onsets
        assert point.max_onsets <= point.total_onsets

    def test_max_onsets_rises_when_short_silence_oversplits(self, tmp_path):
        # The silence-timeout failure mode: a long utterance with a mid-gap
        # shorter than the (large) silence timeout reads as ONE segment, but a
        # tiny silence_ms splits it into TWO. max_onsets is the aggregate that
        # makes that fragmentation visible across the corpus.
        corpus = tmp_path / "gap"
        corpus.mkdir()
        # speech — 300ms gap — speech, all loud. 300ms < 800ms default silence
        # (stays merged) but > a 100ms silence timeout (splits).
        gap = np.concatenate([
            _tone(SR, 0.3),
            _silence(int(SR * 0.3)),
            _tone(SR, 0.3),
        ])
        _write_wav(corpus / "gap.wav", gap)
        (corpus / "gap.json").write_text('{"peak_rms": 0.05}')

        merged = aggregate_results(
            VadParams(silence_ms=800.0),
            replay_all(corpus, VadParams(silence_ms=800.0)),
        )
        split = aggregate_results(
            VadParams(silence_ms=100.0),
            replay_all(corpus, VadParams(silence_ms=100.0)),
        )
        # The short timeout fragments the single utterance — the ceiling climbs.
        assert split.max_onsets > merged.max_onsets

    def test_mean_first_onset_averages_detected_recordings(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        results = replay_all(corpus, VadParams(threshold=0.006))
        point = aggregate_results(VadParams(threshold=0.006), results)
        # Hand-compute the expected mean over only the recordings that
        # detected at least one segment.
        first_onsets = [r.segments[0].onset_ms for r in results if r.segments]
        assert first_onsets, "fixture should detect speech in at least one rec"
        assert point.mean_first_onset_ms == pytest.approx(
            sum(first_onsets) / len(first_onsets)
        )

    def test_mean_first_onset_excludes_missed_recordings(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        # At threshold 0.015 the quiet recording misses (no segments) while the
        # loud one detects. The mean must reflect ONLY the loud recording's
        # onset, not be dragged to 0 by the miss.
        results = replay_all(corpus, VadParams(threshold=0.015))
        detected = [r for r in results if r.segments]
        missed = [r for r in results if not r.segments]
        assert detected and missed, "fixture must mix a hit and a miss at 0.015"
        point = aggregate_results(VadParams(threshold=0.015), results)
        assert point.mean_first_onset_ms == pytest.approx(
            detected[0].segments[0].onset_ms
        )
        assert point.mean_first_onset_ms > 0.0

    def test_mean_first_onset_zero_when_nothing_detected(self, tmp_path):
        # A corpus where every recording misses → no onset times to average,
        # so the timing aggregate is 0.0 (the documented "no data" sentinel),
        # never a spurious early value.
        corpus = tmp_path / "silent"
        corpus.mkdir()
        _write_wav(corpus / "a.wav", _silence(SR))
        (corpus / "a.json").write_text('{"peak_rms": 0.0001}')
        results = replay_all(corpus, VadParams(threshold=0.006))
        assert all(not r.segments for r in results)
        point = aggregate_results(VadParams(threshold=0.006), results)
        assert point.mean_first_onset_ms == 0.0
        assert point.max_first_onset_ms == 0.0
        assert point.min_first_onset_ms == 0.0
        assert point.std_first_onset_ms == 0.0
        assert point.max_segment_ms == 0.0

    def test_max_first_onset_is_latest_detected_recording(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        results = replay_all(corpus, VadParams(threshold=0.006))
        point = aggregate_results(VadParams(threshold=0.006), results)
        first_onsets = [r.segments[0].onset_ms for r in results if r.segments]
        assert first_onsets, "fixture should detect speech in at least one rec"
        # The worst-case ceiling is the max over the same detected onsets the
        # mean averages — never below the mean, never above any actual onset.
        assert point.max_first_onset_ms == pytest.approx(max(first_onsets))
        assert point.max_first_onset_ms >= point.mean_first_onset_ms

    def test_max_first_onset_excludes_missed_recordings(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        # At 0.015 one recording misses; the ceiling must reflect only the
        # detected recording's onset, not be skewed by the miss (which has no
        # onset time at all).
        results = replay_all(corpus, VadParams(threshold=0.015))
        detected = [r for r in results if r.segments]
        missed = [r for r in results if not r.segments]
        assert detected and missed, "fixture must mix a hit and a miss at 0.015"
        point = aggregate_results(VadParams(threshold=0.015), results)
        assert point.max_first_onset_ms == pytest.approx(
            max(r.segments[0].onset_ms for r in detected)
        )
        assert point.max_first_onset_ms > 0.0

    def test_min_first_onset_is_earliest_detected_recording(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        results = replay_all(corpus, VadParams(threshold=0.006))
        point = aggregate_results(VadParams(threshold=0.006), results)
        first_onsets = [r.segments[0].onset_ms for r in results if r.segments]
        assert first_onsets, "fixture should detect speech in at least one rec"
        # The best-case floor is the min over the same detected onsets the mean
        # averages — never above the mean, never below any actual onset. It
        # bounds the full spread together with mean/max:
        # min <= mean <= max.
        assert point.min_first_onset_ms == pytest.approx(min(first_onsets))
        assert point.min_first_onset_ms <= point.mean_first_onset_ms
        assert point.min_first_onset_ms <= point.max_first_onset_ms

    def test_min_first_onset_excludes_missed_recordings(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        # At 0.015 one recording misses; the floor must reflect only the
        # detected recording's onset, not be dragged to 0 by the miss (which
        # has no onset time at all — a 0.0 would falsely read as "earliest").
        results = replay_all(corpus, VadParams(threshold=0.015))
        detected = [r for r in results if r.segments]
        missed = [r for r in results if not r.segments]
        assert detected and missed, "fixture must mix a hit and a miss at 0.015"
        point = aggregate_results(VadParams(threshold=0.015), results)
        assert point.min_first_onset_ms == pytest.approx(
            min(r.segments[0].onset_ms for r in detected)
        )
        assert point.min_first_onset_ms > 0.0

    def test_std_first_onset_is_population_std_of_detected(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        results = replay_all(corpus, VadParams(threshold=0.006))
        point = aggregate_results(VadParams(threshold=0.006), results)
        first_onsets = [r.segments[0].onset_ms for r in results if r.segments]
        assert first_onsets, "fixture should detect speech in at least one rec"
        # Population (ddof=0) std over the same detected onsets the mean uses.
        assert point.std_first_onset_ms == pytest.approx(float(np.std(first_onsets)))
        # The spread is never negative and, when more than one recording is
        # detected with differing onsets, is strictly positive.
        assert point.std_first_onset_ms >= 0.0
        if len(set(first_onsets)) > 1:
            assert point.std_first_onset_ms > 0.0

    def test_std_first_onset_zero_for_single_detected_recording(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        # At 0.015 only the loud recording detects → a single onset → zero
        # spread (population std of one point is 0.0, the documented
        # "perfectly consistent given one point" reading, not undefined).
        results = replay_all(corpus, VadParams(threshold=0.015))
        detected = [r for r in results if r.segments]
        assert len(detected) == 1, "fixture must leave exactly one hit at 0.015"
        point = aggregate_results(VadParams(threshold=0.015), results)
        assert point.std_first_onset_ms == 0.0

    def test_std_first_onset_excludes_missed_recordings(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        # The std must be computed over only the detected recordings; a missed
        # recording contributes no onset time, so folding in a 0.0 would inflate
        # the spread with a phantom early value.
        results_all = replay_all(corpus, VadParams(threshold=0.006))
        results_mixed = replay_all(corpus, VadParams(threshold=0.015))
        detected_mixed = [r for r in results_mixed if r.segments]
        missed_mixed = [r for r in results_mixed if not r.segments]
        assert detected_mixed and missed_mixed, "fixture must mix a hit and a miss at 0.015"
        point = aggregate_results(VadParams(threshold=0.015), results_mixed)
        # Only the one detected recording counts → zero spread, NOT the spread
        # that a phantom 0.0 onset for the miss would produce.
        assert point.std_first_onset_ms == 0.0

    def test_max_segment_is_longest_committed_segment(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        results = replay_all(corpus, VadParams(threshold=0.006))
        point = aggregate_results(VadParams(threshold=0.006), results)
        # The over-merge ceiling is the longest single segment's duration across
        # every detected segment in the corpus.
        all_durations = [s.duration_ms for r in results for s in r.segments]
        assert all_durations, "fixture should detect at least one segment"
        assert point.max_segment_ms == pytest.approx(max(all_durations))
        # A real segment is at least the min_speech gate long (the gate measures
        # committed duration), so the ceiling is a positive, plausible duration.
        assert point.max_segment_ms >= VadParams().min_speech_ms

    def test_max_segment_rises_when_long_silence_overmerges(self, tmp_path):
        # The over-MERGE failure mode (the other end of the silence lever from
        # the over-split max_onsets catches): a long utterance with a mid-gap
        # reads as TWO short segments under a tiny silence_ms but fuses into ONE
        # long run-on segment under a large silence_ms. max_segment_ms is the
        # aggregate that makes that merge visible — the count stays flat (or
        # falls), only the duration balloons.
        corpus = tmp_path / "gap"
        corpus.mkdir()
        # speech — 300ms gap — speech, all loud. Splits at silence_ms=100 (the
        # 300ms gap exceeds the timeout) but merges at silence_ms=800.
        gap = np.concatenate([
            _tone(SR, 0.3),
            _silence(int(SR * 0.3)),
            _tone(SR, 0.3),
        ])
        _write_wav(corpus / "gap.wav", gap)
        (corpus / "gap.json").write_text('{"peak_rms": 0.05}')

        split = aggregate_results(
            VadParams(silence_ms=100.0),
            replay_all(corpus, VadParams(silence_ms=100.0)),
        )
        merged = aggregate_results(
            VadParams(silence_ms=800.0),
            replay_all(corpus, VadParams(silence_ms=800.0)),
        )
        # Merging fuses the two halves (plus the bridged gap) into one segment
        # whose duration exceeds either half alone — the over-merge signal.
        assert merged.max_onsets < split.max_onsets  # count fell (the merge)
        assert merged.max_segment_ms > split.max_segment_ms  # duration ballooned

    def test_max_segment_zero_when_nothing_detected(self, tmp_path):
        # A corpus where every recording misses → no committed segments, so the
        # over-merge ceiling is 0.0 (the documented "no data" sentinel).
        corpus = tmp_path / "silent"
        corpus.mkdir()
        _write_wav(corpus / "a.wav", _silence(SR))
        (corpus / "a.json").write_text('{"peak_rms": 0.0001}')
        results = replay_all(corpus, VadParams(threshold=0.006))
        assert all(not r.segments for r in results)
        point = aggregate_results(VadParams(threshold=0.006), results)
        assert point.max_segment_ms == 0.0


# ---------------------------------------------------------------------------
# sweep_param
# ---------------------------------------------------------------------------


class TestSweepParam:
    def test_one_point_per_value_in_order(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        points = sweep_param("threshold", [0.006, 0.015], recordings_dir=corpus)
        assert len(points) == 2
        assert points[0].params.threshold == 0.006
        assert points[1].params.threshold == 0.015

    def test_lower_threshold_detects_at_least_as_much(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        points = sweep_param("threshold", [0.006, 0.015], recordings_dir=corpus)
        low, high = points
        # The quiet recording drops out at 0.015 → fewer total onsets and a
        # lower trigger count at the higher threshold.
        assert low.total_onsets >= high.total_onsets
        assert low.triggered >= high.triggered

    def test_base_params_are_held_fixed(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        base = VadParams(threshold=0.006, gain=2.0)
        points = sweep_param("debounce_ms", [100.0, 200.0], base=base, recordings_dir=corpus)
        # Only debounce varies; the base's gain and threshold ride along.
        for p in points:
            assert p.params.gain == 2.0
            assert p.params.threshold == 0.006
        assert [p.params.debounce_ms for p in points] == [100.0, 200.0]

    def test_gain_sweep_recovers_quiet_recording(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        # At a strict threshold the quiet recording misses; enough gain lifts
        # it over the gate. Sweep proves the recovery is monotone-ish.
        points = sweep_param(
            "gain", [1.0, 4.0], base=VadParams(threshold=0.02), recordings_dir=corpus
        )
        assert points[1].total_onsets >= points[0].total_onsets

    def test_preroll_sweep_pulls_mean_first_onset_earlier(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        # Pre-roll only shifts onset *timing*, not count. The timing aggregate
        # is the lever that exposes that in a sweep (the count aggregates stay
        # flat). A larger pre-roll must not push the mean first onset later.
        points = sweep_param(
            "preroll_ms", [0.0, 256.0], base=VadParams(threshold=0.006), recordings_dir=corpus
        )
        no_pre, pre = points
        assert pre.total_onsets == no_pre.total_onsets  # count unchanged
        assert pre.mean_first_onset_ms <= no_pre.mean_first_onset_ms  # earlier

    def test_unknown_field_raises(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        with pytest.raises(ValueError):
            sweep_param("nonexistent", [1.0], recordings_dir=corpus)


# ---------------------------------------------------------------------------
# sweep_grid — 2-D grid (backlog item 4)
# ---------------------------------------------------------------------------


class TestSweepGrid:
    def test_one_point_per_cell_in_row_major_order(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        points = sweep_grid(
            "threshold", [0.006, 0.015], "gain", [1.0, 4.0], recordings_dir=corpus
        )
        # 2×2 grid → 4 cells, row-major: param_a outer, param_b inner.
        assert len(points) == 4
        observed = [(p.params.threshold, p.params.gain) for p in points]
        assert observed == [
            (0.006, 1.0),
            (0.006, 4.0),
            (0.015, 1.0),
            (0.015, 4.0),
        ]

    def test_base_params_ride_along(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        base = VadParams(silence_ms=600.0, min_speech_ms=400.0)
        points = sweep_grid(
            "threshold", [0.006], "gain", [1.0, 2.0], base=base, recordings_dir=corpus
        )
        for p in points:
            assert p.params.silence_ms == 600.0
            assert p.params.min_speech_ms == 400.0

    def test_gain_recovers_quiet_recording_at_strict_threshold(self, tmp_path):
        # At threshold 0.02 the quiet recording misses with gain 1.0 but enough
        # gain lifts it over the gate — the grid exposes the joint operating
        # point a single-axis sweep would miss.
        corpus = _make_corpus(tmp_path)
        points = sweep_grid(
            "threshold", [0.02], "gain", [1.0, 4.0], recordings_dir=corpus
        )
        low_gain, high_gain = points
        assert high_gain.total_onsets >= low_gain.total_onsets
        assert high_gain.triggered >= low_gain.triggered

    def test_rectangular_grid_dimensions(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        points = sweep_grid(
            "threshold", [0.006, 0.010, 0.015], "gain", [1.0, 2.0], recordings_dir=corpus
        )
        assert len(points) == 6  # 3×2

    def test_unknown_field_raises(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        with pytest.raises(ValueError):
            sweep_grid("threshold", [0.006], "bogus", [1.0], recordings_dir=corpus)
        with pytest.raises(ValueError):
            sweep_grid("bogus", [0.006], "gain", [1.0], recordings_dir=corpus)

    def test_identical_axes_raise(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        with pytest.raises(ValueError):
            sweep_grid("threshold", [0.006], "threshold", [0.015], recordings_dir=corpus)


# ---------------------------------------------------------------------------
# _parse_value_list
# ---------------------------------------------------------------------------


class TestParseValueList:
    def test_parses_floats(self):
        assert _parse_value_list("0.006,0.015,0.02", float) == [0.006, 0.015, 0.02]

    def test_parses_ints(self):
        assert _parse_value_list("512,1024,2048", int) == [512, 1024, 2048]

    def test_strips_whitespace_and_skips_blanks(self):
        assert _parse_value_list(" 1.0 , , 2.0 ,", float) == [1.0, 2.0]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _parse_value_list("   ,  ,", float)

    def test_bad_token_raises(self):
        with pytest.raises(ValueError):
            _parse_value_list("1.0,notanumber", float)


# ---------------------------------------------------------------------------
# CLI — main() with --sweep
# ---------------------------------------------------------------------------


class TestSweepCli:
    def test_sweep_human_table(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--sweep", "threshold", "--sweep-values", "0.006,0.015", "--dir", str(corpus)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "VAD sweep" in out
        assert "threshold=0.006" in out
        assert "threshold=0.015" in out

    def test_sweep_json(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--sweep", "gain", "--sweep-values", "1.0,2.0", "--dir", str(corpus), "--json"])
        assert rc == 0
        import json as _json

        payload = _json.loads(capsys.readouterr().out)
        assert len(payload) == 2
        assert payload[0]["params"]["gain"] == 1.0
        assert payload[1]["params"]["gain"] == 2.0
        assert "min_onsets" in payload[0]
        assert "max_onsets" in payload[0]
        assert payload[0]["max_onsets"] >= payload[0]["min_onsets"]

    def test_sweep_unknown_field_errors(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--sweep", "bogus", "--sweep-values", "1", "--dir", str(corpus)])
        assert rc == 2
        assert "unknown --sweep field" in capsys.readouterr().out

    def test_sweep_missing_values_errors(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--sweep", "threshold", "--dir", str(corpus)])
        assert rc == 2
        assert "requires --sweep-values" in capsys.readouterr().out

    def test_sweep_bad_values_errors(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--sweep", "threshold", "--sweep-values", "0.006,bad", "--dir", str(corpus)])
        assert rc == 2
        assert "bad --sweep-values" in capsys.readouterr().out

    def test_sweep_empty_corpus_reports_none(self, tmp_path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        rc = main(["--sweep", "threshold", "--sweep-values", "0.006", "--dir", str(empty)])
        assert rc == 1
        assert "No recordings found" in capsys.readouterr().out

    def test_frame_size_sweep_casts_int(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--sweep", "frame_size", "--sweep-values", "512,1024", "--dir", str(corpus), "--json"])
        assert rc == 0
        import json as _json

        payload = _json.loads(capsys.readouterr().out)
        assert payload[0]["params"]["frame_size"] == 512
        assert isinstance(payload[0]["params"]["frame_size"], int)

    def test_preroll_ms_flag_reported_in_header(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--preroll-ms", "192", "--dir", str(corpus)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "preroll=192.0ms" in out

    def test_preroll_sweep_moves_first_onset_earlier(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--sweep", "preroll_ms", "--sweep-values", "0,256", "--dir", str(corpus)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "preroll_ms=0" in out
        assert "preroll_ms=256" in out

    def test_sweep_reports_onset_timing_column(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--sweep", "debounce_ms", "--sweep-values", "100,200", "--dir", str(corpus)])
        assert rc == 0
        out = capsys.readouterr().out
        # The onset-timing aggregate is surfaced in the human table.
        assert "onset1=" in out
        # ...as is its worst-case ceiling.
        assert "onset1_max=" in out
        # ...and its best-case floor.
        assert "onset1_min=" in out
        # ...and its consistency (spread).
        assert "onset1_std=" in out
        # ...and the onset-count floor and over-split ceiling.
        assert "min_onsets=" in out
        assert "max_onsets=" in out
        # ...and the over-merge ceiling (longest committed segment).
        assert "max_seg=" in out

    def test_sweep_json_includes_mean_first_onset(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--sweep", "threshold", "--sweep-values", "0.006", "--dir", str(corpus), "--json"])
        assert rc == 0
        import json as _json

        payload = _json.loads(capsys.readouterr().out)
        assert "mean_first_onset_ms" in payload[0]
        assert payload[0]["mean_first_onset_ms"] > 0.0

    def test_sweep_json_includes_max_first_onset(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--sweep", "threshold", "--sweep-values", "0.006", "--dir", str(corpus), "--json"])
        assert rc == 0
        import json as _json

        payload = _json.loads(capsys.readouterr().out)
        assert "max_first_onset_ms" in payload[0]
        # The ceiling is never below the mean over the same detected set.
        assert payload[0]["max_first_onset_ms"] >= payload[0]["mean_first_onset_ms"]

    def test_sweep_json_includes_min_first_onset(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--sweep", "threshold", "--sweep-values", "0.006", "--dir", str(corpus), "--json"])
        assert rc == 0
        import json as _json

        payload = _json.loads(capsys.readouterr().out)
        assert "min_first_onset_ms" in payload[0]
        assert payload[0]["min_first_onset_ms"] > 0.0
        # The floor bounds the spread from below: min <= mean <= max.
        assert payload[0]["min_first_onset_ms"] <= payload[0]["mean_first_onset_ms"]
        assert payload[0]["min_first_onset_ms"] <= payload[0]["max_first_onset_ms"]

    def test_sweep_json_includes_std_first_onset(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--sweep", "threshold", "--sweep-values", "0.006", "--dir", str(corpus), "--json"])
        assert rc == 0
        import json as _json

        payload = _json.loads(capsys.readouterr().out)
        assert "std_first_onset_ms" in payload[0]
        # The spread is non-negative and, with a multi-recording corpus whose
        # onsets differ, bounded above by the full min→max range.
        assert payload[0]["std_first_onset_ms"] >= 0.0
        rng = payload[0]["max_first_onset_ms"] - payload[0]["min_first_onset_ms"]
        assert payload[0]["std_first_onset_ms"] <= rng + 1e-6

    def test_sweep_json_includes_max_segment(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--sweep", "threshold", "--sweep-values", "0.006", "--dir", str(corpus), "--json"])
        assert rc == 0
        import json as _json

        payload = _json.loads(capsys.readouterr().out)
        assert "max_segment_ms" in payload[0]
        # The over-merge ceiling is a real committed segment, so at least the
        # min_speech gate long when anything detected.
        assert payload[0]["max_segment_ms"] >= 0.0


# ---------------------------------------------------------------------------
# CLI — main() with --grid (2-D grid sweep)
# ---------------------------------------------------------------------------


class TestGridCli:
    def test_grid_human_table(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main([
            "--grid", "threshold,gain",
            "--grid-values-a", "0.006,0.015",
            "--grid-values-b", "1.0,2.0",
            "--dir", str(corpus),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "VAD grid" in out
        assert "threshold × gain" in out
        assert "2×2 cells" in out
        # Each cell labels both axes.
        assert out.count("threshold=") == 4
        assert out.count("gain=") == 4
        # The grid table carries all four onset-timing aggregates per cell.
        assert out.count("onset1=") == 4
        assert out.count("onset1_max=") == 4
        assert out.count("onset1_min=") == 4
        assert out.count("onset1_std=") == 4
        # ...and the onset-count floor + over-split ceiling per cell.
        assert out.count("min_onsets=") == 4
        assert out.count("max_onsets=") == 4
        # ...and the over-merge ceiling per cell.
        assert out.count("max_seg=") == 4

    def test_grid_json_row_major(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main([
            "--grid", "threshold,gain",
            "--grid-values-a", "0.006,0.015",
            "--grid-values-b", "1.0,2.0",
            "--dir", str(corpus), "--json",
        ])
        assert rc == 0
        import json as _json

        payload = _json.loads(capsys.readouterr().out)
        assert len(payload) == 4
        cells = [(c["params"]["threshold"], c["params"]["gain"]) for c in payload]
        assert cells == [(0.006, 1.0), (0.006, 2.0), (0.015, 1.0), (0.015, 2.0)]

    def test_grid_frame_size_axis_casts_int(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main([
            "--grid", "frame_size,gain",
            "--grid-values-a", "512,1024",
            "--grid-values-b", "1.0",
            "--dir", str(corpus), "--json",
        ])
        assert rc == 0
        import json as _json

        payload = _json.loads(capsys.readouterr().out)
        assert isinstance(payload[0]["params"]["frame_size"], int)
        assert payload[0]["params"]["frame_size"] == 512

    def test_grid_wrong_axis_count_errors(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--grid", "threshold", "--grid-values-a", "0.006",
                   "--grid-values-b", "1.0", "--dir", str(corpus)])
        assert rc == 2
        assert "exactly two" in capsys.readouterr().out

    def test_grid_unknown_field_errors(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--grid", "threshold,bogus", "--grid-values-a", "0.006",
                   "--grid-values-b", "1.0", "--dir", str(corpus)])
        assert rc == 2
        assert "unknown --grid field" in capsys.readouterr().out

    def test_grid_identical_axes_errors(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--grid", "threshold,threshold", "--grid-values-a", "0.006",
                   "--grid-values-b", "0.015", "--dir", str(corpus)])
        assert rc == 2
        assert "must differ" in capsys.readouterr().out

    def test_grid_missing_values_errors(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--grid", "threshold,gain", "--grid-values-a", "0.006",
                   "--dir", str(corpus)])
        assert rc == 2
        assert "requires --grid-values" in capsys.readouterr().out

    def test_grid_bad_values_errors(self, tmp_path, capsys):
        corpus = _make_corpus(tmp_path)
        rc = main(["--grid", "threshold,gain", "--grid-values-a", "0.006,bad",
                   "--grid-values-b", "1.0", "--dir", str(corpus)])
        assert rc == 2
        assert "bad --grid values" in capsys.readouterr().out

    def test_grid_empty_corpus_reports_none(self, tmp_path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        rc = main(["--grid", "threshold,gain", "--grid-values-a", "0.006",
                   "--grid-values-b", "1.0", "--dir", str(empty)])
        assert rc == 1
        assert "No recordings found" in capsys.readouterr().out
