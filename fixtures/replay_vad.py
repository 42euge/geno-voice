"""iter-189 — Headless VAD replay harness over real user recordings.

Loads each full-session recording WAV from ``fixtures/recordings/`` and
replays it through a Python port of the ``ContinuousListener`` VAD state
machine in ``client/voice-capture.js`` (RMS threshold + speech-onset
debounce + silence timeout). No microphone, no GUI, no browser — pure
ground-truth simulation that answers a single question for a given
parameter set:

    *Would the live client have detected the speech that this recording
    proves was present?*

The recordings are ground truth captured from the actual desktop app.
Their sibling ``.json`` files carry ``click_to_capture_ms``, ``peak_rms``,
``frames``, and ``sample_rate``. Each WAV is a regression fixture: the
more the user talks to the app, the more land here, and this harness
turns every one into a measurable detection check.

Why this exists (the latency finding):
    The live ``ContinuousListener`` only starts capturing *after*
    getUserMedia + AudioWorklet cold-start completes — measured at
    3.1–5.1s of ``click_to_capture_ms``. Users speak into that dead
    window, so live VAD sees only the quiet tail while the full-session
    recording shows loud speech. This harness replays the *whole*
    recording, so it measures what the VAD *could* recover if capture
    started on time. Comparing whole-recording detection against the
    live tail quantifies how much speech the latency window costs.

Usage (CLI):
    python fixtures/replay_vad.py                 # all recordings, default params
    python fixtures/replay_vad.py --threshold 0.006 --gain 1.0
    python fixtures/replay_vad.py --json          # machine-readable

Usage (library):
    from fixtures.replay_vad import VadParams, replay_recording
    result = replay_recording(path, VadParams(threshold=0.006))
"""

from __future__ import annotations

import argparse
import json
import wave
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"


# ---------------------------------------------------------------------------
# Parameters — mirror the ContinuousListener constructor knobs so a tuning
# experiment here ports straight to client/voice-capture.js.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VadParams:
    """One parameter set for the VAD state machine.

    ``threshold``     — RMS gate (client ``silenceThreshold``). Real speech
                        peaks ~0.037 RMS, silence maxes ~0.0003, so the
                        separation is wide. Upstream default was 0.015; the
                        desktop client lowered it to 0.006.
    ``debounce_ms``   — speech-onset debounce (client hard-codes 200ms): how
                        long RMS must stay over threshold before we commit to
                        "speaking" and stop discarding the candidate buffer.
    ``silence_ms``    — silence timeout (client ``silenceDurationMs``, 800ms):
                        how long RMS stays under threshold before a speech
                        segment is considered ended.
    ``min_speech_ms`` — minimum segment length (client ``minSpeechMs``, 500ms):
                        segments shorter than this are dropped as noise.
    ``gain``          — linear pre-amplification applied to samples before RMS.
                        Models a software gain stage (1.0 = no change).
    ``frame_size``    — samples per analysis frame. The client's worklet/
                        scriptProcessor delivers frames; we re-frame the WAV
                        the same way (~1024 samples by default).
    """

    threshold: float = 0.006
    debounce_ms: float = 200.0
    silence_ms: float = 800.0
    min_speech_ms: float = 500.0
    gain: float = 1.0
    frame_size: int = 1024


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """A detected speech segment (one onset → silence end)."""

    onset_frame: int
    end_frame: int
    onset_ms: float
    end_ms: float
    frames: int

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.onset_ms


@dataclass
class ReplayResult:
    """Outcome of replaying one recording through one parameter set."""

    name: str
    sample_rate: int
    duration_s: float
    total_frames: int
    frame_dur_ms: float
    # Detection
    onsets: int
    speaking_frames: int
    frames_over_threshold: int
    pct_over_threshold: float
    speaking_fraction: float
    segments: List[Segment] = field(default_factory=list)
    # RMS stats (post-gain)
    peak_rms: float = 0.0
    mean_rms: float = 0.0
    median_rms: float = 0.0
    # Ground-truth metadata (from sibling .json), if present
    meta_peak_rms: Optional[float] = None
    meta_click_to_capture_ms: Optional[float] = None
    # Verdict: did the known speech (meta peak_rms) clear the gate AND
    # did the state machine commit at least one onset?
    known_speech_would_trigger: bool = False

    def summary_line(self) -> str:
        trig = "TRIGGER" if self.known_speech_would_trigger else "MISS   "
        return (
            f"{self.name:<32} {trig}  onsets={self.onsets:<2} "
            f"speak_frames={self.speaking_frames:<5} "
            f"over={self.pct_over_threshold:5.1f}%  "
            f"peakRMS={self.peak_rms:.4f} meanRMS={self.mean_rms:.4f}"
        )


# ---------------------------------------------------------------------------
# Core: framing + RMS + state machine
# ---------------------------------------------------------------------------


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV as a float32 mono array in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        channels = wf.getnchannels()
        raw = wf.readframes(n_frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def frame_rms(samples: np.ndarray, frame_size: int, gain: float = 1.0) -> np.ndarray:
    """Per-frame RMS over non-overlapping ``frame_size`` windows.

    Mirrors ``ContinuousListener._handleFrame``: ``sqrt(mean(s^2))`` per
    frame. A trailing partial frame is included (the client processes
    whatever the last buffer holds). ``gain`` pre-amplifies samples.
    """
    if frame_size <= 0:
        raise ValueError("frame_size must be positive")
    if samples.size == 0:
        return np.zeros(0, dtype=np.float64)

    if gain != 1.0:
        samples = samples * gain

    n_full = samples.size // frame_size
    rms_vals: List[float] = []
    if n_full:
        full = samples[: n_full * frame_size].reshape(n_full, frame_size)
        rms_full = np.sqrt(np.mean(np.square(full, dtype=np.float64), axis=1))
        rms_vals.extend(rms_full.tolist())
    remainder = samples[n_full * frame_size :]
    if remainder.size:
        rms_vals.append(float(np.sqrt(np.mean(np.square(remainder, dtype=np.float64)))))
    return np.asarray(rms_vals, dtype=np.float64)


def simulate_vad(rms: np.ndarray, frame_dur_ms: float, params: VadParams) -> tuple[List[Segment], int]:
    """Replay the ContinuousListener state machine over per-frame RMS.

    Returns ``(segments, speaking_frames)`` where ``segments`` are the
    committed-and-accepted speech segments (passed the ``min_speech_ms``
    gate) and ``speaking_frames`` counts every frame spent in the
    committed-speaking state across all segments (including ones later
    dropped for being too short).

    The state machine, faithful to the JS:
      * Below-threshold frames clear any pending onset candidate.
      * An onset candidate must hold over-threshold for > ``debounce_ms``
        of *consecutive* frames before committing to speaking.
      * While speaking, each below-threshold frame advances a silence
        clock; once silence reaches ``silence_ms`` the segment ends.
      * A segment whose committed duration is < ``min_speech_ms`` is
        dropped (state returns to listening but no segment is emitted).
    """
    segments: List[Segment] = []
    speaking_frames = 0

    speaking = False
    candidate_start_ms: Optional[float] = None  # sim clock at first over-thresh
    candidate_frames = 0
    seg_onset_frame = 0
    seg_onset_ms = 0.0
    silence_ms_accum = 0.0
    clock_ms = 0.0

    def end_segment(end_frame: int, end_ms: float) -> None:
        nonlocal speaking
        speaking = False
        duration = end_ms - seg_onset_ms
        if duration >= params.min_speech_ms:
            segments.append(
                Segment(
                    onset_frame=seg_onset_frame,
                    end_frame=end_frame,
                    onset_ms=seg_onset_ms,
                    end_ms=end_ms,
                    frames=end_frame - seg_onset_frame,
                )
            )

    for i, value in enumerate(rms):
        over = value > params.threshold
        if over:
            if not speaking:
                if candidate_start_ms is None:
                    candidate_start_ms = clock_ms
                    candidate_frames = 1
                else:
                    candidate_frames += 1
                    # JS commits once the candidate has held longer than
                    # debounce_ms (strictly greater).
                    if clock_ms - candidate_start_ms > params.debounce_ms:
                        speaking = True
                        seg_onset_frame = i - candidate_frames + 1
                        seg_onset_ms = candidate_start_ms
                        silence_ms_accum = 0.0
                        candidate_start_ms = None
                        candidate_frames = 0
            else:
                silence_ms_accum = 0.0
        else:
            candidate_start_ms = None
            candidate_frames = 0
            if speaking:
                silence_ms_accum += frame_dur_ms
                if silence_ms_accum >= params.silence_ms:
                    end_segment(i + 1, clock_ms + frame_dur_ms)

        if speaking:
            speaking_frames += 1

        clock_ms += frame_dur_ms

    # Recording ended while still speaking — close the open segment at EOF.
    if speaking:
        end_segment(len(rms), clock_ms)

    return segments, speaking_frames


# ---------------------------------------------------------------------------
# Top-level: replay one recording
# ---------------------------------------------------------------------------


def _load_meta(wav_path: Path) -> dict:
    json_path = wav_path.with_suffix(".json")
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text())
    except (ValueError, OSError):
        return {}


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def replay_recording(wav_path: Path, params: VadParams) -> ReplayResult:
    """Replay one recording WAV through ``params`` and return the result."""
    samples, sample_rate = load_wav_mono(wav_path)
    rms = frame_rms(samples, params.frame_size, params.gain)
    frame_dur_ms = (params.frame_size / sample_rate) * 1000.0
    segments, speaking_frames = simulate_vad(rms, frame_dur_ms, params)

    frames_over = int(np.count_nonzero(rms > params.threshold)) if rms.size else 0
    total_frames = int(rms.size)
    pct_over = (100.0 * frames_over / total_frames) if total_frames else 0.0
    speaking_fraction = (speaking_frames / total_frames) if total_frames else 0.0

    meta = _load_meta(wav_path)
    meta_peak_rms = _as_float(meta.get("peak_rms"))
    meta_latency = _as_float(meta.get("click_to_capture_ms"))

    # The known speech (proven by the recording's metadata peak_rms) would
    # trigger the live client iff that peak clears the gate AND the state
    # machine actually committed an onset on the replay.
    gate_cleared = meta_peak_rms is not None and (meta_peak_rms * params.gain) > params.threshold
    known_speech_would_trigger = bool(len(segments) >= 1 and (gate_cleared or meta_peak_rms is None))

    return ReplayResult(
        name=wav_path.name,
        sample_rate=sample_rate,
        duration_s=(samples.size / sample_rate) if sample_rate else 0.0,
        total_frames=total_frames,
        frame_dur_ms=frame_dur_ms,
        onsets=len(segments),
        speaking_frames=speaking_frames,
        frames_over_threshold=frames_over,
        pct_over_threshold=pct_over,
        speaking_fraction=speaking_fraction,
        segments=segments,
        peak_rms=float(np.max(rms)) if rms.size else 0.0,
        mean_rms=float(np.mean(rms)) if rms.size else 0.0,
        median_rms=float(np.median(rms)) if rms.size else 0.0,
        meta_peak_rms=meta_peak_rms,
        meta_click_to_capture_ms=meta_latency,
        known_speech_would_trigger=known_speech_would_trigger,
    )


def replay_all(recordings_dir: Path = RECORDINGS_DIR, params: Optional[VadParams] = None) -> List[ReplayResult]:
    """Replay every ``*.wav`` in ``recordings_dir`` through ``params``."""
    params = params or VadParams()
    return [replay_recording(p, params) for p in sorted(recordings_dir.glob("*.wav"))]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _result_to_dict(r: ReplayResult) -> dict:
    d = asdict(r)
    d["segments"] = [asdict(s) for s in r.segments]
    return d


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Replay recordings through the VAD state machine.")
    parser.add_argument("--threshold", type=float, default=VadParams.threshold)
    parser.add_argument("--debounce-ms", type=float, default=VadParams.debounce_ms)
    parser.add_argument("--silence-ms", type=float, default=VadParams.silence_ms)
    parser.add_argument("--min-speech-ms", type=float, default=VadParams.min_speech_ms)
    parser.add_argument("--gain", type=float, default=VadParams.gain)
    parser.add_argument("--frame-size", type=int, default=VadParams.frame_size)
    parser.add_argument("--dir", type=Path, default=RECORDINGS_DIR)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    params = VadParams(
        threshold=args.threshold,
        debounce_ms=args.debounce_ms,
        silence_ms=args.silence_ms,
        min_speech_ms=args.min_speech_ms,
        gain=args.gain,
        frame_size=args.frame_size,
    )

    results = replay_all(args.dir, params)

    if args.json:
        print(json.dumps([_result_to_dict(r) for r in results], indent=2))
        return 0

    if not results:
        print(f"No recordings found in {args.dir}")
        return 1

    print(
        f"VAD replay — threshold={params.threshold} gain={params.gain} "
        f"debounce={params.debounce_ms}ms silence={params.silence_ms}ms "
        f"min_speech={params.min_speech_ms}ms frame={params.frame_size}"
    )
    print("-" * 100)
    triggered = 0
    for r in results:
        print(r.summary_line())
        triggered += int(r.known_speech_would_trigger)
    print("-" * 100)
    print(f"{triggered}/{len(results)} recordings would trigger detection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
