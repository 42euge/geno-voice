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
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Dict, List, Optional

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
    ``preroll_ms``    — rolling pre-onset buffer (backlog item 2). The live
                        client discards every sub-threshold frame before a
                        speech onset, clipping the soft attack of an utterance
                        (the quiet ramp-up below the RMS gate). A pre-roll
                        buffer keeps the last ``preroll_ms`` of pre-onset audio
                        and prepends it to the committed segment, so the
                        emitted ``onset_ms`` moves *earlier* by up to this much
                        — clamped to the recording start and the previous
                        segment's end (no overlap). ``0.0`` (the default)
                        reproduces today's clip-the-opening behaviour.
    """

    threshold: float = 0.006
    debounce_ms: float = 200.0
    silence_ms: float = 800.0
    min_speech_ms: float = 500.0
    gain: float = 1.0
    frame_size: int = 1024
    preroll_ms: float = 0.0


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
      * If ``preroll_ms`` > 0, an accepted segment's emitted onset is pulled
        back by up to that many ms of pre-onset audio (recovering the clipped
        utterance opening), clamped to the recording start and the previous
        segment's end. The pre-roll padding is cosmetic to the onset only —
        the ``min_speech_ms`` gate still measures the *committed* speech.
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

    # Pre-roll: how many whole frames to reach back before a committed onset.
    preroll_frames = (
        int(round(params.preroll_ms / frame_dur_ms))
        if params.preroll_ms > 0 and frame_dur_ms > 0
        else 0
    )

    def end_segment(end_frame: int, end_ms: float) -> None:
        nonlocal speaking
        speaking = False
        # The min_speech gate measures committed speech, not pre-roll padding.
        duration = end_ms - seg_onset_ms
        if duration >= params.min_speech_ms:
            onset_frame = seg_onset_frame
            if preroll_frames:
                # Clamp the pull-back to the recording start and the previous
                # segment's end so segments never overlap.
                floor_frame = segments[-1].end_frame if segments else 0
                onset_frame = max(floor_frame, seg_onset_frame - preroll_frames)
            onset_ms = onset_frame * frame_dur_ms
            segments.append(
                Segment(
                    onset_frame=onset_frame,
                    end_frame=end_frame,
                    onset_ms=onset_ms,
                    end_ms=end_ms,
                    frames=end_frame - onset_frame,
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
# Parameter sweep — run a grid of parameter values across the whole corpus
# and aggregate detection so a tuning experiment (backlog items 3–6 in
# docs/research/voice-capture-tuning.md) produces one comparison table
# instead of N hand-run single-param invocations.
# ---------------------------------------------------------------------------


@dataclass
class SweepPoint:
    """Corpus-aggregate detection for one parameter set in a sweep.

    ``params``         — the VadParams this point used.
    ``recordings``     — how many recordings were replayed.
    ``triggered``      — how many of them ``known_speech_would_trigger``.
    ``total_onsets``   — sum of detected onsets across the corpus.
    ``total_speaking_frames`` — sum of committed speaking frames.
    ``mean_pct_over``  — mean %-of-frames-over-threshold across recordings.
    ``min_onsets``     — fewest onsets any single recording got (the
                         worst-case recording for this parameter set; a
                         sweep wants to maximize this floor, not just the
                         total — one missed recording is a real miss).
    ``max_onsets``     — most onsets any single recording got (the over-split
                         ceiling). This is to ``total_onsets`` what
                         ``max_first_onset_ms`` is to ``mean_first_onset_ms``:
                         the symmetric companion to ``min_onsets``. It exists for
                         the silence-timeout backlog item — a too-short
                         ``silence_ms`` shatters one continuous utterance into
                         many short segments, so its signature is a single
                         recording's onset count climbing well above the others
                         while the corpus total barely moves. ``min_onsets`` (the
                         floor) catches a recording dropping to a *miss*;
                         ``max_onsets`` (the ceiling) catches the opposite failure
                         — a recording *fragmenting* — so a ``silence_ms`` sweep
                         can read the over-split end and the merge end in one
                         pass. ``0`` for an empty corpus.
    ``mean_first_onset_ms`` — mean of each recording's *first* segment
                         ``onset_ms`` (the emitted speech-start), averaged
                         only over recordings that detected at least one
                         segment. This is the onset-*timing* aggregate
                         (backlog item 5 follow-up): a smaller value means
                         speech is captured earlier. It complements the
                         onset-*count* aggregates so a debounce/pre-roll
                         sweep can show timing moving earlier in one pass,
                         without hand-inspecting each recording's ``--json``.
                         Recordings with no segments are excluded (a missed
                         recording has no onset time; counting it as 0ms
                         would falsely read as "earliest possible").
    ``max_first_onset_ms`` — the *latest* first-segment ``onset_ms`` any
                         single detected recording got (the worst-case
                         recording for onset *timing*). This is to
                         ``mean_first_onset_ms`` what ``min_onsets`` is to
                         ``total_onsets``: the mean can look great while one
                         recording is captured far too late, so a tuning sweep
                         wants to minimize this ceiling, not just the mean —
                         one badly-late onset is a real regression. Excludes
                         missed recordings on the same reasoning as the mean
                         (no segment → no onset time); ``0.0`` when nothing
                         detected.
    ``min_first_onset_ms`` — the *earliest* first-segment ``onset_ms`` any
                         single detected recording got (the best-case recording
                         for onset *timing*). Together with
                         ``mean_first_onset_ms`` and ``max_first_onset_ms`` it
                         gives the full best/typical/worst onset-timing shape in
                         one sweep pass, so a debounce/pre-roll sweep shows not
                         just whether timing improved on average but how the
                         whole spread moved. It also exposes the *irreducible
                         floor*: an onset-shaping knob (debounce) can't pull a
                         recording earlier than this best case, so when the min
                         stops moving the knob has saturated. Excludes missed
                         recordings on the same reasoning as the mean and max
                         (no segment → no onset time); ``0.0`` when nothing
                         detected.
    ``std_first_onset_ms`` — the population standard deviation of the
                         first-segment ``onset_ms`` across detected recordings
                         (the *consistency* of onset timing). ``min``/``mean``/
                         ``max`` give the envelope and center of the timing
                         distribution; this gives its *spread* — how tightly the
                         recordings cluster around the mean. Two parameter sets
                         can share a mean onset while one opens at a consistent
                         time every recording and the other swings between very
                         early and very late; the std is the only aggregate that
                         tells them apart, so a grid sweep can pick the cell that
                         opens early *and* consistently rather than early on
                         average with a wild tail. Population (not sample) std —
                         a single detected recording has zero spread (``0.0``),
                         which reads correctly as "perfectly consistent given one
                         data point" rather than the undefined sample std. Uses
                         the same detected-recordings subset as the mean (no
                         segment → no onset time); ``0.0`` when nothing detected.
    ``max_segment_ms`` — the longest single committed segment's ``duration_ms``
                         across the whole corpus (the over-*merge* ceiling). This
                         is the silence-timeout backlog item's missing half. The
                         onset-*count* aggregates read fragmentation: a too-short
                         ``silence_ms`` shatters one utterance into many short
                         segments, which ``max_onsets`` (iter-201) catches as a
                         climbing per-recording count. The opposite failure — a
                         too-*long* ``silence_ms`` fusing two real turns into one
                         run-on segment — leaves the onset count flat (or even
                         lower) and is invisible to every count aggregate; its
                         signature is a single segment's *duration* ballooning.
                         ``max_segment_ms`` reads exactly that: paired with
                         ``max_onsets`` it brackets both ends of the silence lever
                         — the ceiling on *count* catches over-splitting, the
                         ceiling on *duration* catches over-merging — so a
                         ``--sweep silence_ms`` reads the merge end and the split
                         end in one pass. It is the ``max`` of each emitted
                         segment's ``duration_ms`` (onset→end); since pre-roll
                         pulls the emitted onset earlier it can add at most that
                         window, small (≤512ms) next to the multi-second swing a
                         real over-merge produces, and a ``silence_ms`` sweep
                         holds pre-roll fixed anyway. ``0.0`` when nothing
                         detected.
    """

    params: VadParams
    recordings: int
    triggered: int
    total_onsets: int
    total_speaking_frames: int
    mean_pct_over: float
    min_onsets: int
    max_onsets: int
    mean_first_onset_ms: float
    max_first_onset_ms: float
    min_first_onset_ms: float
    std_first_onset_ms: float
    max_segment_ms: float

    def summary_line(self, label_key: str = "threshold") -> str:
        value = getattr(self.params, label_key)
        return (
            f"{label_key}={value:<8} "
            f"trig={self.triggered}/{self.recordings}  "
            f"min_onsets={self.min_onsets:<2} "
            f"max_onsets={self.max_onsets:<2} "
            f"onsets={self.total_onsets:<3} "
            f"speak_frames={self.total_speaking_frames:<5} "
            f"mean_over={self.mean_pct_over:5.1f}% "
            f"onset1_min={self.min_first_onset_ms:6.1f}ms "
            f"onset1={self.mean_first_onset_ms:6.1f}ms "
            f"onset1_max={self.max_first_onset_ms:6.1f}ms "
            f"onset1_std={self.std_first_onset_ms:6.1f}ms "
            f"max_seg={self.max_segment_ms:7.1f}ms"
        )


def aggregate_results(params: VadParams, results: List[ReplayResult]) -> SweepPoint:
    """Fold a list of per-recording results into one corpus-aggregate point."""
    n = len(results)
    triggered = sum(int(r.known_speech_would_trigger) for r in results)
    total_onsets = sum(r.onsets for r in results)
    total_speaking = sum(r.speaking_frames for r in results)
    mean_over = (sum(r.pct_over_threshold for r in results) / n) if n else 0.0
    min_onsets = min((r.onsets for r in results), default=0)
    # Over-split ceiling: the most onsets any single recording got. Paired with
    # min_onsets (the miss floor) it brackets the per-recording onset count, so a
    # silence_ms sweep reads a recording fragmenting (high max) as readily as one
    # dropping to a miss (zero min), without hand-inspecting each --json.
    max_onsets = max((r.onsets for r in results), default=0)
    # Onset *timing*: average each recording's first emitted onset, but only
    # over recordings that actually detected speech. A missed recording has
    # no onset time; folding in a 0.0 would falsely pull the mean toward
    # "earliest possible", masking the miss as great timing.
    first_onsets = [r.segments[0].onset_ms for r in results if r.segments]
    mean_first_onset = (sum(first_onsets) / len(first_onsets)) if first_onsets else 0.0
    # Worst-case onset timing: the latest first onset across detected
    # recordings. The mean can hide a single very-late capture; this ceiling
    # exposes it, mirroring how min_onsets exposes the worst-case count.
    max_first_onset = max(first_onsets) if first_onsets else 0.0
    # Best-case onset timing: the earliest first onset across detected
    # recordings. Paired with the mean and max it gives the full timing spread,
    # and marks the irreducible floor an onset-shaping knob can't push past.
    min_first_onset = min(first_onsets) if first_onsets else 0.0
    # Onset-timing consistency: population std of the first onsets over the same
    # detected subset. min/mean/max give the envelope; this gives the spread, so
    # a sweep can tell apart two cells with the same mean — one consistent, one
    # swinging. Population (ddof=0) so a single detected recording reads as 0.0
    # (perfectly consistent given one point) rather than an undefined sample std.
    std_first_onset = float(np.std(first_onsets)) if first_onsets else 0.0
    # Over-merge ceiling: the longest single committed segment across the corpus.
    # The symmetric companion to max_onsets on the *duration* axis — a too-long
    # silence_ms fuses two turns into one run-on segment, leaving the onset count
    # flat (invisible to every count aggregate) while a single segment's duration
    # balloons. Measures duration_ms (onset→end) over every detected segment.
    seg_durations = [s.duration_ms for r in results for s in r.segments]
    max_segment = max(seg_durations) if seg_durations else 0.0
    return SweepPoint(
        params=params,
        recordings=n,
        triggered=triggered,
        total_onsets=total_onsets,
        total_speaking_frames=total_speaking,
        mean_pct_over=mean_over,
        min_onsets=min_onsets,
        max_onsets=max_onsets,
        mean_first_onset_ms=mean_first_onset,
        max_first_onset_ms=max_first_onset,
        min_first_onset_ms=min_first_onset,
        std_first_onset_ms=std_first_onset,
        max_segment_ms=max_segment,
    )


def sweep_param(
    param_name: str,
    values: List[float],
    base: Optional[VadParams] = None,
    recordings_dir: Path = RECORDINGS_DIR,
) -> List[SweepPoint]:
    """Replay the whole corpus once per value of a single ``VadParams`` field.

    ``param_name`` is a field of ``VadParams`` (e.g. ``"threshold"``,
    ``"gain"``, ``"debounce_ms"``); each value in ``values`` is substituted
    into a copy of ``base`` (default ``VadParams()``) and the corpus is
    replayed. Returns one ``SweepPoint`` per value, in input order — the
    machine-readable comparison table the tuning backlog asks for.
    """
    base = base or VadParams()
    if param_name not in {f for f in VadParams.__dataclass_fields__}:
        raise ValueError(f"unknown VadParams field: {param_name!r}")
    points: List[SweepPoint] = []
    for value in values:
        params = replace(base, **{param_name: value})
        results = replay_all(recordings_dir, params)
        points.append(aggregate_results(params, results))
    return points


def sweep_grid(
    param_a: str,
    values_a: List[float],
    param_b: str,
    values_b: List[float],
    base: Optional[VadParams] = None,
    recordings_dir: Path = RECORDINGS_DIR,
) -> List[SweepPoint]:
    """Replay the corpus once per cell of a 2-D ``param_a`` × ``param_b`` grid.

    Backlog item 4 (``docs/research/voice-capture-tuning.md``): a single-axis
    sweep can be too coarse for picking a *joint* operating point — e.g.
    threshold and gain interact (more gain lifts quiet speech over the gate,
    so the best threshold shifts). This folds both axes into one corpus pass
    per cell and returns one ``SweepPoint`` per cell, in row-major order
    (``param_a`` outer, ``param_b`` inner) so the result reads as a table with
    ``param_a`` rows and ``param_b`` columns.

    Both names must be ``VadParams`` fields and must differ (a 2-D grid over
    one axis is just ``sweep_param``). The other flags ride along from
    ``base`` (default ``VadParams()``) unchanged.
    """
    base = base or VadParams()
    fields = set(VadParams.__dataclass_fields__)
    if param_a not in fields:
        raise ValueError(f"unknown VadParams field: {param_a!r}")
    if param_b not in fields:
        raise ValueError(f"unknown VadParams field: {param_b!r}")
    if param_a == param_b:
        raise ValueError(f"grid axes must differ; both are {param_a!r}")
    points: List[SweepPoint] = []
    for va in values_a:
        for vb in values_b:
            params = replace(base, **{param_a: va, param_b: vb})
            results = replay_all(recordings_dir, params)
            points.append(aggregate_results(params, results))
    return points


def _parse_value_list(raw: str, cast) -> List[float]:
    """Parse a comma-separated CLI value list, casting each entry."""
    out: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(cast(token))
    if not out:
        raise ValueError("empty value list")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _result_to_dict(r: ReplayResult) -> dict:
    d = asdict(r)
    d["segments"] = [asdict(s) for s in r.segments]
    return d


def _sweep_point_to_dict(p: SweepPoint) -> dict:
    d = asdict(p)
    d["params"] = asdict(p.params)
    return d


# Which VadParams fields take an int vs. float on the CLI value list.
_INT_FIELDS = {"frame_size"}


def _run_sweep(args, base: VadParams) -> int:
    """Handle ``--sweep PARAM --sweep-values V1,V2,...`` and print the grid."""
    if args.sweep not in VadParams.__dataclass_fields__:
        print(
            f"unknown --sweep field {args.sweep!r}; choose one of: "
            f"{', '.join(VadParams.__dataclass_fields__)}"
        )
        return 2
    if not args.sweep_values:
        print("--sweep requires --sweep-values (comma-separated)")
        return 2

    cast = int if args.sweep in _INT_FIELDS else float
    try:
        values = _parse_value_list(args.sweep_values, cast)
    except ValueError as exc:
        print(f"bad --sweep-values: {exc}")
        return 2

    points = sweep_param(args.sweep, values, base=base, recordings_dir=args.dir)

    if args.json:
        print(json.dumps([_sweep_point_to_dict(p) for p in points], indent=2))
        return 0

    if not points or points[0].recordings == 0:
        print(f"No recordings found in {args.dir}")
        return 1

    print(
        f"VAD sweep — {args.sweep} over {args.sweep_values} "
        f"({points[0].recordings} recordings)"
    )
    print("-" * 100)
    for p in points:
        print(p.summary_line(args.sweep))
    return 0


def _grid_summary_line(p: SweepPoint, param_a: str, param_b: str) -> str:
    """One-line summary of a grid cell, labelled by both swept axes."""
    va = getattr(p.params, param_a)
    vb = getattr(p.params, param_b)
    return (
        f"{param_a}={va:<8} {param_b}={vb:<8} "
        f"trig={p.triggered}/{p.recordings}  "
        f"min_onsets={p.min_onsets:<2} "
        f"max_onsets={p.max_onsets:<2} "
        f"onsets={p.total_onsets:<3} "
        f"speak_frames={p.total_speaking_frames:<5} "
        f"mean_over={p.mean_pct_over:5.1f}% "
        f"onset1_min={p.min_first_onset_ms:6.1f}ms "
        f"onset1={p.mean_first_onset_ms:6.1f}ms "
        f"onset1_max={p.max_first_onset_ms:6.1f}ms "
        f"onset1_std={p.std_first_onset_ms:6.1f}ms "
        f"max_seg={p.max_segment_ms:7.1f}ms"
    )


def _run_grid(args, base: VadParams) -> int:
    """Handle ``--grid A,B --grid-values-a ... --grid-values-b ...``."""
    fields = VadParams.__dataclass_fields__
    parts = [p.strip() for p in args.grid.split(",") if p.strip()]
    if len(parts) != 2:
        print("--grid takes exactly two comma-separated fields, e.g. threshold,gain")
        return 2
    param_a, param_b = parts
    for name in (param_a, param_b):
        if name not in fields:
            print(
                f"unknown --grid field {name!r}; choose from: {', '.join(fields)}"
            )
            return 2
    if param_a == param_b:
        print(f"--grid axes must differ; both are {param_a!r}")
        return 2
    if not args.grid_values_a or not args.grid_values_b:
        print("--grid requires --grid-values-a and --grid-values-b (comma-separated)")
        return 2

    try:
        values_a = _parse_value_list(args.grid_values_a, int if param_a in _INT_FIELDS else float)
        values_b = _parse_value_list(args.grid_values_b, int if param_b in _INT_FIELDS else float)
    except ValueError as exc:
        print(f"bad --grid values: {exc}")
        return 2

    points = sweep_grid(
        param_a, values_a, param_b, values_b, base=base, recordings_dir=args.dir
    )

    if args.json:
        print(json.dumps([_sweep_point_to_dict(p) for p in points], indent=2))
        return 0

    if not points or points[0].recordings == 0:
        print(f"No recordings found in {args.dir}")
        return 1

    print(
        f"VAD grid — {param_a} × {param_b} "
        f"({len(values_a)}×{len(values_b)} cells, {points[0].recordings} recordings)"
    )
    print("-" * 100)
    for p in points:
        print(_grid_summary_line(p, param_a, param_b))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Replay recordings through the VAD state machine.")
    parser.add_argument("--threshold", type=float, default=VadParams.threshold)
    parser.add_argument("--debounce-ms", type=float, default=VadParams.debounce_ms)
    parser.add_argument("--silence-ms", type=float, default=VadParams.silence_ms)
    parser.add_argument("--min-speech-ms", type=float, default=VadParams.min_speech_ms)
    parser.add_argument("--gain", type=float, default=VadParams.gain)
    parser.add_argument("--frame-size", type=int, default=VadParams.frame_size)
    parser.add_argument("--preroll-ms", type=float, default=VadParams.preroll_ms)
    parser.add_argument("--dir", type=Path, default=RECORDINGS_DIR)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--sweep",
        metavar="PARAM",
        help="sweep one VadParams field (threshold, gain, debounce_ms, "
        "silence_ms, min_speech_ms, frame_size, preroll_ms) across "
        "--sweep-values, holding the other flags fixed as the base",
    )
    parser.add_argument(
        "--sweep-values",
        metavar="V1,V2,...",
        help="comma-separated values for --sweep (e.g. 0.004,0.006,0.015)",
    )
    parser.add_argument(
        "--grid",
        metavar="A,B",
        help="2-D sweep over two VadParams fields (e.g. threshold,gain) — one "
        "corpus pass per cell of --grid-values-a × --grid-values-b, holding "
        "the other flags fixed as the base",
    )
    parser.add_argument(
        "--grid-values-a",
        metavar="V1,V2,...",
        help="comma-separated values for the first --grid axis",
    )
    parser.add_argument(
        "--grid-values-b",
        metavar="V1,V2,...",
        help="comma-separated values for the second --grid axis",
    )
    args = parser.parse_args(argv)

    params = VadParams(
        threshold=args.threshold,
        debounce_ms=args.debounce_ms,
        silence_ms=args.silence_ms,
        min_speech_ms=args.min_speech_ms,
        gain=args.gain,
        frame_size=args.frame_size,
        preroll_ms=args.preroll_ms,
    )

    if args.sweep:
        return _run_sweep(args, params)

    if args.grid:
        return _run_grid(args, params)

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
        f"min_speech={params.min_speech_ms}ms frame={params.frame_size} "
        f"preroll={params.preroll_ms}ms"
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
