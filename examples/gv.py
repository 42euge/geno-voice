#!/usr/bin/env python3
"""geno-voice CLI.

Usage:
    gv bench              # batch mode — wait for silence, transcribe, show timing
    gv stream             # streaming mode — live progressive transcription
    gv talk               # talk mode — STT → NLP → canned response → TTS
    gv chat               # chat mode — STT → LLM (litellm) → TTS
    gv simulate-mirror …  # offline WPM-mirror trajectory / grid-sweep simulator
    gv calibrate-base-wpm … # offline base_wpm calibration (--verdict for an adopt/keep call)
    gv vad recording.wav  # offline Silero VAD — segment a WAV into speech regions
    gv vad recording.wav --json # machine-readable segmentation (SileroResult.to_dict shape)
    gv vad-diff recording.wav --threshold-a 0.5 --threshold-b 0.7  # compare two P(speech) gates
    gv vad-sweep recording.wav --thresholds 0.3,0.5,0.7,0.9  # sweep N gates, tabulate the elbow
    gv vad-sweep recording.wav --min-silences 200,400,800,1600  # sweep the hangover instead
    gv vad-sweep recording.wav --min-speeches 50,100,200,400  # sweep the min-speech floor instead
    gv <cmd> --model ...  # override STT model
"""

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

# TTS speech-rate multiplier bounds (the kokoro `speed` parameter). The engine
# treats this as a wall-clock multiplier on synthesis, so values <= 0 are
# nonsensical (zero / negative-rate speech) and very large values produce
# unintelligible output. The accepted window matches kokoro's documented
# practical range; the parser rejects anything outside it with the usual
# argparse SystemExit(2) instead of forwarding garbage to the TTS engine.
SPEED_MIN = 0.5
SPEED_MAX = 2.0


def speed_type(raw):
    """Argparse ``type`` for ``--speed``: parse to float and bound-check.

    Pure and side-effect-free so it can be unit-tested directly. Raises
    :class:`argparse.ArgumentTypeError` (which argparse renders as
    ``SystemExit(2)``) when ``raw`` is not a number or falls outside
    ``[SPEED_MIN, SPEED_MAX]``. NaN is rejected explicitly — it compares
    false against both bounds, so without the guard the range message would
    misleadingly name the bounds rather than the real problem.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"speed must be a number, got {raw!r}")
    if value != value:  # NaN is unordered; name it directly.
        raise argparse.ArgumentTypeError("speed must be a number, got nan")
    if not (SPEED_MIN <= value <= SPEED_MAX):
        raise argparse.ArgumentTypeError(
            f"speed must be between {SPEED_MIN} and {SPEED_MAX}, got {value}"
        )
    return value


# Kokoro voice-id grammar: a single-letter language code, a one-letter gender
# (`f`/`m`), an underscore, then a lowercase a-z name. The curated American set
# lives in `tts/kokoro_engine.py:VOICES` (af_*/am_*), but kokoro also ships
# other-language packs (e.g. `bf_emma`, British female), so a strict membership
# check would wrongly reject legitimate voices. We validate the *format*
# instead — the same "close the garbage-in path at the parser" goal as
# `speed_type` (iter-182) — rejecting empty/whitespace/malformed ids before
# they reach the TTS engine, while staying permissive about the name itself.
VOICE_RE = re.compile(r"^[a-z][fm]_[a-z]+$")


def voice_type(raw):
    """Argparse ``type`` for ``--voice``: validate the kokoro id format.

    Pure and side-effect-free so it can be unit-tested directly. Raises
    :class:`argparse.ArgumentTypeError` (rendered by argparse as
    ``SystemExit(2)``) when ``raw`` does not match the kokoro voice-id
    grammar ``<lang><gender>_<name>`` (e.g. ``af_heart``, ``bf_emma``). This
    catches empty strings, leading/trailing whitespace, and obvious typos
    before they reach the engine, without pinning to the American-only
    curated list.
    """
    if not isinstance(raw, str) or not VOICE_RE.match(raw):
        raise argparse.ArgumentTypeError(
            "voice must look like '<lang><gender>_<name>' "
            f"(e.g. af_heart, bf_emma), got {raw!r}"
        )
    return raw


def model_type(raw):
    """Argparse ``type`` for ``--model``: reject empty / whitespace ids.

    The model knob is the broadest of the three CLI string inputs: it
    accepts short aliases (``tiny``, ``large-v3``), full HF repo ids
    (``mlx-community/whisper-large-v3-turbo``), and local filesystem
    paths. There is no single grammar to validate against — unlike
    ``--voice`` (iter-183) and ``--speed`` (iter-182) — so a strict
    format check would wrongly reject legitimate inputs. We validate
    only the one thing every valid form agrees on: a model id is a
    non-empty token with no surrounding or embedded whitespace.

    This catches the obvious garbage — an empty string, a whitespace-
    only value, or an accidental ``"  tiny"`` / ``"large v3"`` — at the
    parser with the usual ``SystemExit(2)`` (via
    :class:`argparse.ArgumentTypeError`), instead of forwarding it to
    the STT engine where it surfaces as a confusing load failure deep
    in the synthesis stack. Pure and side-effect-free for direct unit
    testing. Anything well-formed (alias, repo id, or path) passes
    through unchanged — model *existence* is still resolved lazily at
    load time, as before.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise argparse.ArgumentTypeError(
            f"model must be a non-empty id, got {raw!r}"
        )
    if raw != raw.strip() or any(c.isspace() for c in raw):
        raise argparse.ArgumentTypeError(
            f"model must not contain whitespace, got {raw!r}"
        )
    return raw


def unit_interval_type(raw):
    """Argparse ``type`` for ``gv vad --threshold``: a P(speech) gate in [0, 1].

    Silero emits a per-window speech probability in ``[0, 1]`` and the
    ``threshold`` knob gates on it (default 0.5, the model's own operating
    point). This is NOT an energy threshold — it is the neural confidence, so
    values outside ``[0, 1]`` are meaningless. Pure and side-effect-free for
    direct unit testing; raises :class:`argparse.ArgumentTypeError` (rendered by
    argparse as ``SystemExit(2)``) on a non-number, NaN, or out-of-range value.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"threshold must be a number, got {raw!r}")
    if value != value:  # NaN is unordered.
        raise argparse.ArgumentTypeError("threshold must be a number, got nan")
    if not (0.0 <= value <= 1.0):
        raise argparse.ArgumentTypeError(
            f"threshold must be between 0 and 1, got {value}"
        )
    return value


def unit_interval_list_type(raw):
    """Argparse ``type`` for ``gv vad-sweep --thresholds``: a list of gates.

    iter-235 ``gv vad-diff`` compares the P(speech) gate at exactly two points;
    iter-236 generalises that to a sweep over N thresholds, so this parses a
    comma-separated list (e.g. ``"0.3,0.5,0.7,0.9"``) into ``[0.3, 0.5, 0.7,
    0.9]`` at the parser. Each token must be a valid :func:`unit_interval_type`
    value (a number in ``[0, 1]``, not NaN); duplicates and unsorted input are
    preserved as given (the operator may want a specific column order). Rejects
    an empty list. Pure and side-effect-free for direct unit testing.
    """
    if not isinstance(raw, str):
        raise argparse.ArgumentTypeError(f"thresholds must be a string, got {raw!r}")
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError(
            f"thresholds must be a non-empty comma-separated list, got {raw!r}"
        )
    return [unit_interval_type(tok) for tok in tokens]


def nonneg_float_list_type(raw):
    """Argparse ``type`` for ``gv vad-sweep`` millisecond sweep axes: a list.

    iter-236 swept the P(speech) gate (``--thresholds``); iter-238 added a second
    sweep axis — the trailing-silence hangover ``min_silence_ms`` (``--min-
    silences``); iter-239 adds a third — the minimum-speech floor
    ``min_speech_ms`` (``--min-speeches``). Both millisecond axes share this
    validator: it parses a comma-separated list (e.g. ``"400,600,800,1000"``)
    into ``[400.0, 600.0, 800.0, 1000.0]`` at the parser. Each token must be a
    valid :func:`nonneg_float_type` value (a number ``>= 0``, not NaN — ``0`` is
    legitimate, it disables the minimum); duplicates and unsorted input are
    preserved as given (the operator may want a specific column order). Rejects
    an empty list. Pure and side-effect-free for direct unit testing — the
    millisecond twin of :func:`unit_interval_list_type`.
    """
    if not isinstance(raw, str):
        raise argparse.ArgumentTypeError(
            f"a millisecond list must be a string, got {raw!r}"
        )
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError(
            f"a millisecond list must be non-empty and comma-separated, got {raw!r}"
        )
    return [nonneg_float_type(tok) for tok in tokens]


def nonneg_float_type(raw):
    """Argparse ``type`` for the ``gv vad`` millisecond knobs (``--min-speech-ms``
    / ``--min-silence-ms`` / ``--speech-pad-ms``).

    Each is a duration in milliseconds passed straight to ``SileroParams``; a
    negative duration is nonsensical, while ``0`` is legitimate (disable the
    minimum / no padding). Pure and side-effect-free for direct unit testing;
    raises :class:`argparse.ArgumentTypeError` on a non-number, NaN, or negative
    value.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"value must be a number, got {raw!r}")
    if value != value:  # NaN is unordered.
        raise argparse.ArgumentTypeError("value must be a number, got nan")
    if value < 0:
        raise argparse.ArgumentTypeError(f"value must be >= 0, got {value}")
    return value


def nonneg_int_type(raw):
    """Argparse ``type`` for ``gv vad-grid --target``: a target segment count.

    The number of speech regions an operator wants the segmenter to land on for
    a given recording (e.g. one segment per spoken sentence). A count is a
    non-negative integer — ``0`` is legitimate ("expect silence"), negatives and
    fractional values are nonsensical. Pure and side-effect-free for direct unit
    testing; raises :class:`argparse.ArgumentTypeError` on a non-integer or
    negative value. The integer twin of :func:`nonneg_float_type`.
    """
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"target must be a whole number, got {raw!r}"
        )
    if value < 0:
        raise argparse.ArgumentTypeError(f"target must be >= 0, got {value}")
    return value


def pos_int_type(raw):
    """Argparse ``type`` for ``gv vad-grid --top``: a shortlist length.

    The number of closest grid cells to surface as a ranked shortlist beside
    the single ``best:`` pick. A shortlist of ``0`` cells is meaningless and a
    fractional/negative count is nonsensical, so this requires a whole number
    ``>= 1`` — the difference from :func:`nonneg_int_type` (where ``0`` is a
    legitimate "expect silence" target). Pure and side-effect-free for direct
    unit testing; raises :class:`argparse.ArgumentTypeError` otherwise.
    """
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"top must be a whole number, got {raw!r}"
        )
    if value < 1:
        raise argparse.ArgumentTypeError(f"top must be >= 1, got {value}")
    return value


def max_speech_type(raw):
    """Argparse ``type`` for ``gv vad --max-speech-s``: force-split bound.

    Silero's ``max_speech_duration_s`` splits any region longer than this; the
    :class:`SileroParams` default is ``inf`` (never force-split). Accepts a
    positive float OR a sentinel (``inf`` / ``none`` / ``off``) meaning "never
    split". ``float('inf')`` parses natively, so the sentinels are a nicety;
    ``0`` and negatives are rejected (a zero-length cap would split forever).
    Pure and side-effect-free for direct unit testing.
    """
    if isinstance(raw, str) and raw.strip().lower() in ("none", "off"):
        return float("inf")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"max-speech-s must be a number or 'inf', got {raw!r}"
        )
    if value != value:  # NaN is unordered.
        raise argparse.ArgumentTypeError("max-speech-s must be a number, got nan")
    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"max-speech-s must be positive (or 'inf'), got {value}"
        )
    return value


def wpm_list_type(raw):
    """Argparse ``type`` for ``--wpms``: parse a comma-separated WPM arc.

    The ``simulate-mirror`` subcommand replays a per-turn ``user_wpm`` sequence
    (e.g. a slow → fast → slow conversation) through the offline mirror. This
    parses ``"120,140,200,140,120"`` into ``[120.0, 140.0, 200.0, 140.0,
    120.0]`` at the parser so the handler receives a clean list of floats.

    Pure and side-effect-free for direct unit testing. Raises
    :class:`argparse.ArgumentTypeError` (rendered by argparse as
    ``SystemExit(2)``) on an empty list or any non-numeric / NaN token. A
    ``<= 0`` value is *allowed* — it is the iter-064 "no measurement that turn"
    marker the simulator replays faithfully (the speed holds), so an operator
    can model a silent / one-word turn in the arc.
    """
    if not isinstance(raw, str):
        raise argparse.ArgumentTypeError(f"wpms must be a string, got {raw!r}")
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError(
            f"wpms must be a non-empty comma-separated list, got {raw!r}"
        )
    values = []
    for tok in tokens:
        try:
            value = float(tok)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(
                f"wpms entries must be numbers, got {tok!r}"
            )
        if value != value:  # NaN is unordered.
            raise argparse.ArgumentTypeError("wpms entries must be numbers, got nan")
        values.append(value)
    return values


def positive_floats_type(raw):
    """Argparse ``type`` for ``--base-wpms`` / ``--strengths`` grid axes.

    Parses a comma-separated list of positive floats. Used for the two grid
    axes of ``simulate-mirror --grid``. Rejects empty lists and non-positive /
    non-numeric / NaN tokens — every grid axis value must be a usable tunable
    (``base_wpm`` must be positive; a ``strength`` of ``0`` is degenerate but
    still in ``WpmMirrorConfig``'s ``[0, 1]`` band, so the lower bound is left
    to the config validator and only ``<= 0`` base/strength tokens that would
    raise there are caught here as malformed axis input).

    Pure and side-effect-free for direct unit testing.
    """
    if not isinstance(raw, str):
        raise argparse.ArgumentTypeError(f"value must be a string, got {raw!r}")
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError(
            f"must be a non-empty comma-separated list, got {raw!r}"
        )
    values = []
    for tok in tokens:
        try:
            value = float(tok)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(f"entries must be numbers, got {tok!r}")
        if value != value:  # NaN is unordered.
            raise argparse.ArgumentTypeError("entries must be numbers, got nan")
        if value <= 0:
            raise argparse.ArgumentTypeError(
                f"entries must be positive, got {value}"
            )
        values.append(value)
    return values


def calibration_sample_type(raw):
    """Argparse ``type`` for one ``calibrate-base-wpm --samples`` triple.

    Parses ``"words:audio_seconds[:speed]"`` into a ``(words, audio_seconds,
    speed)`` tuple of floats. Each triple is one rendered TTS sample: a
    known-length script of ``words`` words synthesized into ``audio_seconds`` of
    audio at the Kokoro ``speed`` knob (default the ``1.0`` calibration point).
    The handler builds a :class:`CalibrationSample` from each tuple and folds
    them with ``calibrate_base_wpm``.

    Pure and side-effect-free for direct unit testing. Raises
    :class:`argparse.ArgumentTypeError` (rendered by argparse as
    ``SystemExit(2)``) on a malformed shape (wrong field count), a non-numeric /
    NaN field, or a non-positive field — every field must be a usable
    measurement, and the engine's own ``CalibrationSample.__post_init__`` rejects
    ``<= 0`` loudly, so the parser catches the same malformed input early with a
    clean CLI error instead of forwarding garbage.
    """
    if not isinstance(raw, str):
        raise argparse.ArgumentTypeError(f"sample must be a string, got {raw!r}")
    fields = [f.strip() for f in raw.split(":")]
    if len(fields) not in (2, 3) or any(not f for f in fields):
        raise argparse.ArgumentTypeError(
            "sample must be 'words:audio_seconds[:speed]', got "
            f"{raw!r}"
        )
    names = ("words", "audio_seconds", "speed")
    values = []
    for name, tok in zip(names, fields):
        try:
            value = float(tok)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(
                f"{name} must be a number, got {tok!r}"
            )
        if value != value:  # NaN is unordered.
            raise argparse.ArgumentTypeError(f"{name} must be a number, got nan")
        if value <= 0:
            raise argparse.ArgumentTypeError(
                f"{name} must be positive, got {value}"
            )
        values.append(value)
    if len(values) == 2:
        values.append(1.0)  # default speed = the 1.0 calibration point.
    return tuple(values)


def render_calibration(calib):
    """Render a ``BaseWpmCalibration`` verdict as plain-text report lines.

    Pure: returns a list of strings (no I/O, no ANSI) so it is testable in
    isolation — the handler joins and prints them. ``calib`` of ``None`` (no
    samples) yields a single "no samples" line.
    """
    if calib is None:
        return ["base_wpm calibration: no samples (nothing to calibrate from)"]
    lines = [
        "base_wpm calibration from rendered samples",
        f"  samples:          {calib.n_samples}",
        f"  implied base_wpm: {calib.implied_base_wpm:.1f} (median; set DEFAULT_BASE_WPM to this)",
        f"  range:            {calib.min_base_wpm:.1f} – {calib.max_base_wpm:.1f}",
        f"  spread:           {calib.spread:.1f} (renders disagree if large)",
        f"  nominal:          {calib.default_base_wpm:.1f}",
        f"  drift:            {calib.drift:+.1f} (implied − nominal; + ⇒ voice faster than nominal)",
    ]
    return lines


def render_calibration_verdict(verdict):
    """Render a ``CalibrationVerdict`` (iter-222) as plain-text report lines.

    Pure: returns a list of strings (no I/O, no ANSI) so it is testable in
    isolation — the handler joins and prints them. ``verdict`` of ``None`` (no
    samples ⇒ nothing to decide) yields a single "no verdict" line, mirroring
    :func:`render_calibration`'s empty contract. This is the iter-218/221
    CLI-later surface for the iter-222 verdict engine: it turns the raw
    spread/drift numbers into an adopt/keep DECISION the operator can act on.
    """
    if verdict is None:
        return ["base_wpm verdict: no samples (nothing to decide)"]
    decision = (
        f"re-seed base_wpm to {verdict.implied_base_wpm:.1f}"
        if verdict.recommend
        else "keep the current nominal"
    )
    lines = [
        "base_wpm calibration verdict",
        f"  decision: {decision}",
        f"  reason:   {verdict.reason}",
        f"  gates:    spread<={verdict.spread_max:.1f}, "
        f"|drift|>={verdict.drift_min:.1f}, samples>={verdict.min_samples}",
    ]
    return lines


def render_trajectory(traj, *, wpms=None):
    """Render a :class:`SpeedTrajectory` as plain-text report lines.

    Pure: returns a list of strings (no I/O, no ANSI) so it is testable in
    isolation — the handler joins and prints them. Follows the GENO.md
    convention of keeping presentation separate from the engine.
    """
    lines = [
        "WPM-mirror trajectory simulation",
        f"  initial speed:   {traj.initial_speed:.3f}",
        f"  final speed:     {traj.final_speed:.3f}",
    ]
    if traj.ideal_final_speed is None:
        lines.append("  ideal (target):  n/a (mirroring disabled / no measurable turn)")
    else:
        lines.append(f"  ideal (target):  {traj.ideal_final_speed:.3f}")
    if traj.final_gap is None:
        lines.append("  final gap:       n/a")
    else:
        lines.append(f"  final gap:       {traj.final_gap:+.3f} (residual to target)")
    lines.append(f"  max step:        {traj.max_step:.3f} (largest single-turn lurch)")
    lines.append(f"  moves:           {traj.moves} of {len(traj.speeds)} turns changed speed")
    if traj.speeds:
        per_turn = ", ".join(f"{s:.3f}" for s in traj.speeds)
        lines.append(f"  per-turn speeds: {per_turn}")
    return lines


def render_grid(points, best):
    """Render a grid sweep (``MirrorGridPoint`` list) + the picked best cell.

    Pure: returns a list of strings (no I/O). Each row shows a cell's tunables
    and its convergence / lurch / churn diagnostics; the trailing line names
    the data-driven pick (or notes that no cell was scorable).
    """
    lines = ["WPM-mirror grid sweep (base_wpm × strength)"]
    header = "  base_wpm  strength  final  gap     step   moves  score"
    lines.append(header)
    for p in points:
        gap = "  n/a " if p.final_gap is None else f"{p.final_gap:+.3f}"
        score = p.score()
        score_s = " n/a " if score is None else f"{score:.3f}"
        lines.append(
            f"  {p.base_wpm:7.1f}  {p.strength:7.2f}  "
            f"{p.final_speed:5.3f}  {gap}  {p.max_step:5.3f}  "
            f"{p.moves:4d}   {score_s}"
        )
    if best is None:
        lines.append("  best: none (no scorable cell — no measurable turn in the arc)")
    else:
        lines.append(
            f"  best: base_wpm={best.base_wpm:.1f} strength={best.strength:.2f} "
            f"(score {best.score():.3f})"
        )
    return lines


def render_vad_segments(result, *, threshold=None):
    """Render a Silero ``SileroResult`` (iter-231) as plain-text report lines.

    Pure: returns a list of strings (no I/O, no ANSI) so it is testable in
    isolation — the handler joins and prints them, mirroring the other
    ``render_*`` helpers. ``result`` of ``None`` (no segmenter available) yields
    a single explanatory line so the handler can degrade cleanly on a host
    without ``silero-vad`` installed.
    """
    if result is None:
        return [
            "silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        ]
    lines = [
        f"silero VAD segmentation — {result.name}",
        f"  sample rate:  {result.sample_rate} Hz",
        f"  duration:     {result.duration_s:.1f}s",
        f"  segments:     {result.num_segments}",
        f"  speech total: {result.speech_s:.1f}s",
    ]
    if threshold is not None:
        lines.append(f"  threshold:    {threshold:.2f} (P(speech) gate)")
    if result.num_segments == 0:
        lines.append("  (no speech regions detected)")
    else:
        for i, seg in enumerate(result.segments, start=1):
            lines.append(
                f"  [{i:>2}] {seg.start_s:7.2f}s – {seg.end_s:7.2f}s "
                f"({seg.duration_s:5.2f}s)"
            )
    return lines


def render_vad_json(result, *, threshold=None):
    """Render a Silero ``SileroResult`` (iter-231) as a JSON string.

    The machine-readable twin of :func:`render_vad_segments`, mirroring
    ``fixtures/replay_silero.py --json`` / ``SileroResult.to_dict`` so the
    segmentation can feed scripts and tooling, not just human eyes. Pure:
    returns a single JSON string (no I/O) built from the result's attributes —
    it does NOT call ``result.to_dict()``, so it works on any object exposing
    the ``SileroResult`` shape (lets tests drive it without importing torch).

    ``result`` of ``None`` (no segmenter available) yields an object with
    ``{"available": false}`` plus the install hint, so a consumer can detect
    the degraded path from the JSON itself rather than parsing prose. When
    ``threshold`` is supplied it is echoed into the payload alongside the
    segmentation, matching the human report's threshold line.
    """
    if result is None:
        return json.dumps(
            {
                "available": False,
                "hint": (
                    "install 'silero-vad' (pulls torch + torchaudio) to enable "
                    "offline neural segmentation"
                ),
            },
            indent=2,
        )
    payload = {
        "available": True,
        "name": result.name,
        "sample_rate": result.sample_rate,
        "duration_s": round(result.duration_s, 3),
        "num_segments": result.num_segments,
        "speech_s": round(result.speech_s, 3),
        "segments": [
            {
                "start_s": round(seg.start_s, 3),
                "end_s": round(seg.end_s, 3),
                "duration_s": round(seg.duration_s, 3),
            }
            for seg in result.segments
        ],
    }
    if threshold is not None:
        payload["threshold"] = threshold
    return json.dumps(payload, indent=2)


def vad_segmentation_delta(result_a, result_b):
    """Compute the delta between two Silero segmentations of the same WAV.

    iter-234 shipped the machine-readable ``gv vad --json`` surface; this is
    its first consumer — the pure core of ``gv vad-diff``, which segments one
    recording under two ``--threshold`` values (or any two SileroResult-shaped
    objects) and quantifies how the segmentation shifts. Bigger thresholds gate
    out marginal speech, so the higher-threshold run is typically a *subset* of
    the lower one (fewer regions, less total speech).

    Pure: takes two objects exposing the ``SileroResult`` shape
    (``num_segments`` / ``speech_s``) and returns a plain ``dict`` of both
    sides plus their signed deltas (b minus a). No I/O, no rounding loss beyond
    3 places (matching :func:`render_vad_json`), so it is testable without
    importing torch.
    """
    a_segs = result_a.num_segments
    b_segs = result_b.num_segments
    a_speech = round(result_a.speech_s, 3)
    b_speech = round(result_b.speech_s, 3)
    return {
        "num_segments_a": a_segs,
        "num_segments_b": b_segs,
        "num_segments_delta": b_segs - a_segs,
        "speech_s_a": a_speech,
        "speech_s_b": b_speech,
        "speech_s_delta": round(b_speech - a_speech, 3),
    }


def render_vad_diff(result_a, result_b, *, label_a, label_b):
    """Render a two-threshold segmentation comparison as plain-text lines.

    The human-readable twin of :func:`render_vad_diff_json`. ``label_a`` /
    ``label_b`` are the two ``--threshold`` values being compared. Either
    result of ``None`` (segmenter unavailable) yields the shared install hint,
    matching :func:`render_vad_segments`. Pure: returns a list of strings.
    """
    if result_a is None or result_b is None:
        return [
            "silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        ]
    d = vad_segmentation_delta(result_a, result_b)
    name = result_a.name
    return [
        f"silero VAD diff — {name}",
        f"  threshold A:  {label_a:.2f}",
        f"  threshold B:  {label_b:.2f}",
        f"  segments:     {d['num_segments_a']} → {d['num_segments_b']} "
        f"({_signed(d['num_segments_delta'])})",
        f"  speech total: {d['speech_s_a']:.1f}s → {d['speech_s_b']:.1f}s "
        f"({_signed_float(d['speech_s_delta'])}s)",
    ]


def render_vad_diff_json(result_a, result_b, *, label_a, label_b):
    """Render a two-threshold segmentation comparison as a JSON string.

    Machine-readable twin of :func:`render_vad_diff`, so a sweep harness can
    consume the delta directly. Either result ``None`` →
    ``{"available": false}`` + install hint, mirroring :func:`render_vad_json`.
    Pure: returns a single JSON string built from the results' attributes.
    """
    if result_a is None or result_b is None:
        return json.dumps(
            {
                "available": False,
                "hint": (
                    "install 'silero-vad' (pulls torch + torchaudio) to enable "
                    "offline neural segmentation"
                ),
            },
            indent=2,
        )
    payload = {
        "available": True,
        "name": result_a.name,
        "threshold_a": label_a,
        "threshold_b": label_b,
        **vad_segmentation_delta(result_a, result_b),
    }
    return json.dumps(payload, indent=2)


# gv vad-sweep axis metadata. iter-236 swept only the P(speech) gate; iter-238
# adds the trailing-silence hangover as a second axis; iter-239 adds the
# minimum-speech floor as a third. Each entry gives the human-table column label
# (right-justified to 9, matching the original "threshold" width) and the
# per-value display formatter — a gate prints with 2 decimals (0.30), a
# millisecond knob (hangover or speech floor) as a bare integer (800). The dict
# key is also the row key emitted by vad_segmentation_sweep and the CSV/JSON
# column name, so a consumer reads which dimension was swept straight off the
# data. Both millisecond axes format identically, so they're grouped below.
_SWEEP_MS_AXES = ("min_silence_ms", "min_speech_ms")
_SWEEP_AXIS_LABEL = {
    "threshold": "threshold",
    "min_silence_ms": "min_silence",
    "min_speech_ms": "min_speech",
}


def _format_sweep_axis_value(axis, value):
    """Format one swept-axis value for the human table (gate vs. ms knob)."""
    if axis in _SWEEP_MS_AXES:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _render_pick_block(cells, target, top, tie_break, *, format_axes, empty_noun):
    """Render the shared ``best:`` / ``top N:`` pick block for a sweep or grid.

    iter-241→244 grew a data-driven pick block — a ``best:`` line naming the
    cell whose ``num_segments`` is closest to ``target``, optionally followed by
    a ``top N (closest to target N):`` shortlist — and bolted an identical copy
    onto both :func:`render_vad_sweep` (1-D) and :func:`render_vad_grid` (2-D).
    The two copies differed ONLY in how many axis labels precede the count and
    the empty-table noun, so they were one format drift away from disagreeing.
    iter-245 factors that block out here so the two surfaces can never drift.

    ``cells`` is the flat row/cell list (a sweep row or a grid cell — the
    pickers read only the shared ``num_segments`` / ``speech_s`` keys, so either
    shape works). ``format_axes(cell)`` is a caller-supplied callable returning
    the per-cell axis section string (``"threshold=0.70"`` for a sweep,
    ``"threshold=0.70 min_silence=800"`` for a grid); the helper owns everything
    around it. ``empty_noun`` names the table for the empty-input message
    (``"sweep"`` or ``"grid"``). Returns the list of lines to append (empty when
    ``target is None``), so the caller stays ``lines.extend(...)``. Pure — reads
    nothing, mutates nothing.
    """
    if target is None:
        return []
    best = pick_best_grid_cell(cells, target, tie_break)
    if best is None:
        return [f"  best: none (empty {empty_noun}; target {target} segments)"]
    lines = [
        f"  best: {format_axes(best)} "
        f"({best['num_segments']} segments, "
        f"|Δ|={grid_cell_distance(best, target)} from target {target})"
    ]
    if top is not None:
        ranked = pick_top_grid_cells(cells, target, top, tie_break)
        lines.append(f"  top {len(ranked)} (closest to target {target}):")
        for rank, cell in enumerate(ranked, start=1):
            lines.append(
                f"    {rank}. {format_axes(cell)}  "
                f"{cell['num_segments']} segments  "
                f"|Δ|={grid_cell_distance(cell, target)}"
            )
    return lines


def vad_segmentation_sweep(values, results, *, axis="threshold"):
    """Pair each swept-axis value with its segmentation summary for a table.

    iter-235 ``gv vad-diff`` quantifies the segmentation delta between exactly
    two thresholds; iter-236 generalised that to a sweep over N P(speech) gates
    so the gate's elbow is visible at a glance; iter-238 parameterises the swept
    dimension via ``axis`` so the same machinery sweeps the trailing-silence
    hangover (``axis="min_silence_ms"``) instead — the analogue of
    ``fixtures/replay_silero.py``'s ``--min-silence-ms`` knob, but tabulated
    across N values.

    Pure: takes the parallel ``values`` list (one per swept-axis point) and
    ``results`` list (each a ``SileroResult``-shaped object segmented at the
    matching value) and returns a list of rows ``{axis, "num_segments",
    "speech_s"}``, ``speech_s`` rounded to 3 places like
    :func:`vad_segmentation_delta`. The row's axis key IS ``axis`` (default
    ``"threshold"`` keeps the iter-236 callers unchanged). No I/O, no torch
    import, so it is testable in isolation. Raises :class:`ValueError` if the
    two lists differ in length.
    """
    if len(values) != len(results):
        raise ValueError(
            f"values ({len(values)}) and results ({len(results)}) "
            "must be the same length"
        )
    return [
        {
            axis: v,
            "num_segments": r.num_segments,
            "speech_s": round(r.speech_s, 3),
        }
        for v, r in zip(values, results)
    ]


def render_vad_sweep(
    values, results, *, name, axis="threshold", target=None, top=None,
    tie_break="row-major",
):
    """Render a sweep as a plain-text table.

    The human-readable twin of :func:`render_vad_sweep_json`, mirroring the
    other ``render_*`` helpers. ``name`` is the WAV being swept; ``axis`` names
    the swept dimension (``"threshold"`` gate or ``"min_silence_ms"`` hangover),
    which sets the column label. Any ``None`` in ``results`` (segmenter
    unavailable) yields the shared install hint. Pure: returns a list of
    strings. Reading down the threshold table the segment count / speech total
    are typically non-increasing (higher gates admit less speech); a longer
    hangover merges adjacent regions, so the silence sweep tends to FEWER
    segments as the value rises — the elbow marks the knob getting too strict.

    iter-244 brings the iter-241→243 grid pick machinery to the 1-D sweep,
    closing the sweep↔grid feature gap. When ``target`` (a desired segment
    count) is given, a trailing ``best:`` line names the data-driven pick — the
    swept value whose recovered segment count is closest to ``target`` (via the
    shared :func:`pick_best_grid_cell`, since a sweep row carries the same
    ``num_segments`` / ``speech_s`` keys a grid cell does). ``target=None`` (the
    default) omits the line, keeping the iter-236 output unchanged. When ``top``
    (a positive shortlist length) is ALSO given, a ``top N:`` block follows
    listing the ``top`` values closest to ``target``, ranked nearest-first
    (:func:`pick_top_grid_cells`); its head is always the ``best:`` value.
    ``top`` is ignored without a ``target``. ``tie_break`` selects how
    equal-distance rows order: ``"row-major"`` (the default) keeps sweep order;
    ``"speech"`` breaks ties on recovered speech (most first) — the same seam as
    the grid.
    """
    if any(r is None for r in results):
        return [
            "silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        ]
    rows = vad_segmentation_sweep(values, results, axis=axis)
    label = _SWEEP_AXIS_LABEL.get(axis, axis)
    lines = [
        f"silero VAD sweep — {name}",
        f"  {label:>9}  segments  speech",
    ]
    for row in rows:
        lines.append(
            f"  {_format_sweep_axis_value(axis, row[axis]):>9}  "
            f"{row['num_segments']:>8}  {row['speech_s']:>5.1f}s"
        )
    lines.extend(
        _render_pick_block(
            rows,
            target,
            top,
            tie_break,
            format_axes=lambda row: (
                f"{label}={_format_sweep_axis_value(axis, row[axis])}"
            ),
            empty_noun="sweep",
        )
    )
    return lines


def render_vad_sweep_json(
    values, results, *, name, axis="threshold", target=None, top=None,
    tie_break="row-major",
):
    """Render a sweep as a JSON string.

    Machine-readable twin of :func:`render_vad_sweep`, so the sweep can feed a
    plotting/tuning script. The payload carries the swept ``axis`` name so a
    consumer knows which dimension the rows vary (the rows are keyed by that
    same name). Any ``None`` in ``results`` → ``{"available": false}`` + install
    hint, mirroring :func:`render_vad_json`. Pure: returns a single JSON string
    built from the results' attributes.

    iter-244: when ``target`` (a desired segment count) is given the payload
    gains a ``"target"`` int, a ``"tie_break"`` string, and a ``"best"`` row —
    :func:`pick_best_grid_cell`'s pick over the sweep rows, augmented with a
    ``"distance"`` key (``|num_segments - target|``). When ``top`` is ALSO given
    the payload gains a ``"top"`` list of the closest ``top`` rows
    (:func:`pick_top_grid_cells`), each with the same ``"distance"`` key — its
    head equals ``"best"``. ``target=None`` (the default) omits all of them,
    keeping the iter-236 payload unchanged; ``top`` is ignored without a target.
    """
    if any(r is None for r in results):
        return json.dumps(
            {
                "available": False,
                "hint": (
                    "install 'silero-vad' (pulls torch + torchaudio) to enable "
                    "offline neural segmentation"
                ),
            },
            indent=2,
        )
    rows = vad_segmentation_sweep(values, results, axis=axis)
    payload = {
        "available": True,
        "name": name,
        "axis": axis,
        "sweep": rows,
    }
    if target is not None:
        payload["target"] = target
        payload["tie_break"] = tie_break
        best = pick_best_grid_cell(rows, target, tie_break)
        payload["best"] = (
            None
            if best is None
            else {**best, "distance": grid_cell_distance(best, target)}
        )
        if top is not None:
            payload["top"] = [
                {**row, "distance": grid_cell_distance(row, target)}
                for row in pick_top_grid_cells(rows, target, top, tie_break)
            ]
    return json.dumps(payload, indent=2)


def render_vad_sweep_csv(values, results, *, name, axis="threshold"):
    """Render a sweep as CSV text (no trailing newline).

    The spreadsheet/plot-friendly twin of :func:`render_vad_sweep_json`: where
    JSON nests the rows under a ``sweep`` key for programmatic consumers, CSV
    emits a flat ``<axis>,num_segments,speech_s`` table that pipes straight into
    a spreadsheet or a plotting script (gnuplot, matplotlib's ``loadtxt``,
    pandas ``read_csv``) without a JSON-parsing step. The first column header is
    the swept ``axis`` name (``threshold``, ``min_silence_ms``, or
    ``min_speech_ms``) so the grid is self-describing. ``name`` is accepted for signature parity with the other
    ``render_vad_sweep_*`` twins but is not part of the tabular body — a CSV is
    a pure data grid, so the WAV name would only appear as an awkward repeated
    column. Any ``None`` in ``results`` (segmenter unavailable) yields a single
    ``# silero VAD unavailable: ...`` comment line so a degraded run is still
    self-describing rather than silently empty. Pure: returns a single string
    built with the stdlib :mod:`csv` writer (RFC-4180 quoting, ``\\r\\n`` row
    terminators) with the trailing terminator stripped.
    """
    if any(r is None for r in results):
        return (
            "# silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([axis, "num_segments", "speech_s"])
    for row in vad_segmentation_sweep(values, results, axis=axis):
        writer.writerow([row[axis], row["num_segments"], row["speech_s"]])
    return buf.getvalue().rstrip("\r\n")


def vad_segmentation_grid(
    row_values, col_values, results, *, row_axis="threshold", col_axis="min_silence_ms"
):
    """Pair each (row, col) axis-value cell with its segmentation summary.

    iter-240's 2-D analogue of :func:`vad_segmentation_sweep`: where the 1-D
    sweep tabulates ONE knob across N values, the grid tabulates the cartesian
    product of TWO knobs (the analogue of ``simulate-mirror --grid``'s
    base_wpm × strength), so an operator can read the elbow in two dimensions
    at once instead of running N separate 1-D sweeps. ``results`` is the
    flattened cell list in ROW-MAJOR order (row 0's whole row of columns first,
    then row 1's, …), length ``len(row_values) * len(col_values)``.

    Pure: returns a flat list of cell dicts ``{row_axis, col_axis,
    "num_segments", "speech_s"}`` in that same row-major order, ``speech_s``
    rounded to 3 places like :func:`vad_segmentation_sweep`. No I/O, no torch
    import, so it is testable in isolation. Raises :class:`ValueError` if
    ``results`` length differs from the row×col product.
    """
    expected = len(row_values) * len(col_values)
    if len(results) != expected:
        raise ValueError(
            f"results ({len(results)}) must equal row_values × col_values "
            f"({len(row_values)} × {len(col_values)} = {expected})"
        )
    cells = []
    i = 0
    for rv in row_values:
        for cv in col_values:
            r = results[i]
            i += 1
            cells.append(
                {
                    row_axis: rv,
                    col_axis: cv,
                    "num_segments": r.num_segments,
                    "speech_s": round(r.speech_s, 3),
                }
            )
    return cells


def grid_cell_distance(cell, target):
    """``|num_segments - target|`` for one grid cell — lower is better.

    The VAD-grid scoring analogue of :meth:`MirrorGridPoint.score`: where the
    WPM grid folds convergence + lurch into a lower-is-better number, a VAD
    grid cell is scored purely by how far its recovered segment count sits from
    the operator's ``target`` (e.g. one segment per spoken sentence). Pure —
    reads only ``cell["num_segments"]``, an int, so it never touches torch.
    """
    return abs(cell["num_segments"] - target)


def grid_cell_sort_key(cell, target, tie_break="row-major"):
    """Ranking sort key for one grid cell against ``target`` — lower sorts first.

    iter-243's tie-break seam for the data-driven pickers. The PRIMARY key is
    always :func:`grid_cell_distance` (``|num_segments - target|``); ``tie_break``
    decides how cells at EQUAL distance order:

    - ``"row-major"`` (the default) adds no secondary key, so a stable sort
      leaves equal-distance cells in their original row-major order — the
      iter-241/242 earliest-tie rule, unchanged byte-for-byte.
    - ``"speech"`` breaks the tie on recovered speech seconds, MOST speech
      first (a tied cell that clips less of the talker is the more defensible
      pick than merely the earlier one in the grid), via a ``-speech_s``
      secondary key.

    Pure — reads only ``cell["num_segments"]`` and, for ``"speech"`` ties,
    ``cell["speech_s"]``. Never touches torch.
    """
    distance = grid_cell_distance(cell, target)
    if tie_break == "speech":
        return (distance, -cell["speech_s"])
    return (distance,)


def pick_best_grid_cell(cells, target, tie_break="row-major"):
    """Pick the grid cell whose ``num_segments`` is closest to ``target``.

    iter-240 shipped ``gv vad-grid`` tabulating the gate × ms-knob cartesian
    product; this is iter-241's data-driven picker over that table — the VAD
    counterpart of :func:`pick_best_mirror_config` (``simulate-mirror --grid``).
    Scores every cell with :func:`grid_cell_sort_key` and returns the one with
    the smallest key — the ``(threshold, ms)`` pair that segments the recording
    into closest to the number of regions the operator expects.

    Earliest-tie rule (matching :func:`pick_best_mirror_config` and the VAD
    sweep): with the default ``tie_break="row-major"``, on an exact distance tie
    the earlier cell in row-major order wins, so a stable grid ordering yields a
    stable pick. iter-243's ``tie_break="speech"`` instead breaks distance ties
    on recovered speech (most first). ``None`` when ``cells`` is empty. Pure —
    reads nothing, mutates nothing.
    """
    if not cells:
        return None
    return min(cells, key=lambda c: grid_cell_sort_key(c, target, tie_break))


def pick_top_grid_cells(cells, target, k, tie_break="row-major"):
    """Rank the ``k`` grid cells closest to ``target`` (a shortlist, not one pick).

    iter-242's generalisation of :func:`pick_best_grid_cell`: where the single
    picker names the closest cell, this returns the closest ``k`` as a ranked
    list (nearest first), so an operator sees the runners-up — useful when the
    very best cell sits at a knob extreme they distrust, or when two cells tie
    on segment count and they want to break the tie on another axis by eye.

    Stable ordering: sorts by :func:`grid_cell_sort_key`, and Python's sort is
    stable. With the default ``tie_break="row-major"`` the key carries no
    secondary, so cells at equal distance keep their row-major order; iter-243's
    ``tie_break="speech"`` breaks those ties on recovered speech (most first).
    Either way the shortlist's head is identical to :func:`pick_best_grid_cell`'s
    pick under the same ``tie_break`` (both honour the same ordering). ``k`` is
    clamped to the grid size, so a shortlist longer than the grid simply returns
    every cell ranked. Returns ``[]`` for an empty grid. Pure — reads nothing,
    mutates nothing (sorts a shallow copy).
    """
    ranked = sorted(cells, key=lambda c: grid_cell_sort_key(c, target, tie_break))
    return ranked[:k]


def render_vad_grid(
    row_values,
    col_values,
    results,
    *,
    name,
    row_axis="threshold",
    col_axis="min_silence_ms",
    target=None,
    top=None,
    tie_break="row-major",
):
    """Render a 2-D grid sweep as a plain-text table.

    The human-readable twin of :func:`render_vad_grid_json`, the 2-D analogue
    of :func:`render_vad_sweep` and the direct counterpart of
    :func:`render_grid` (``simulate-mirror --grid``): a FLAT one-row-per-cell
    table (not a matrix) so each cell's two metrics — segment count and speech
    seconds — stay unambiguous. ``name`` is the WAV being swept; ``row_axis`` /
    ``col_axis`` name the two swept dimensions, which set the two leading column
    labels and value formats (a gate prints ``0.40``, a millisecond knob a bare
    ``800``). Any ``None`` in ``results`` (segmenter unavailable) yields the
    shared install hint.

    When ``target`` (a desired segment count) is given, a trailing ``best:``
    line names the data-driven pick — the cell whose segment count is closest
    to ``target`` (the :func:`pick_best_grid_cell` analogue of
    :func:`render_grid`'s best line). ``target=None`` (the default) omits the
    line, keeping the iter-240 output unchanged.

    When ``top`` (a positive shortlist length) is ALSO given, a ``top N:`` block
    follows the ``best:`` line listing the ``top`` cells closest to ``target``,
    ranked nearest-first (:func:`pick_top_grid_cells`), so the operator sees the
    runners-up — the head of the shortlist is always the ``best:`` cell. ``top``
    is ignored without a ``target`` (there is no distance to rank by).

    iter-243's ``tie_break`` selects how equal-distance cells order within the
    pick and the shortlist: ``"row-major"`` (the default) keeps the original
    grid order (the iter-241/242 behaviour, unchanged); ``"speech"`` breaks
    distance ties on recovered speech (most first). Pure: returns a list of
    strings.
    """
    if any(r is None for r in results):
        return [
            "silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        ]
    cells = vad_segmentation_grid(
        row_values, col_values, results, row_axis=row_axis, col_axis=col_axis
    )
    row_label = _SWEEP_AXIS_LABEL.get(row_axis, row_axis)
    col_label = _SWEEP_AXIS_LABEL.get(col_axis, col_axis)
    lines = [
        f"silero VAD grid — {name} ({row_label} × {col_label})",
        f"  {row_label:>11}  {col_label:>11}  segments  speech",
    ]
    for cell in cells:
        lines.append(
            f"  {_format_sweep_axis_value(row_axis, cell[row_axis]):>11}  "
            f"{_format_sweep_axis_value(col_axis, cell[col_axis]):>11}  "
            f"{cell['num_segments']:>8}  {cell['speech_s']:>5.1f}s"
        )
    lines.extend(
        _render_pick_block(
            cells,
            target,
            top,
            tie_break,
            format_axes=lambda cell: (
                f"{row_label}="
                f"{_format_sweep_axis_value(row_axis, cell[row_axis])} "
                f"{col_label}="
                f"{_format_sweep_axis_value(col_axis, cell[col_axis])}"
            ),
            empty_noun="grid",
        )
    )
    return lines


def render_vad_grid_json(
    row_values,
    col_values,
    results,
    *,
    name,
    row_axis="threshold",
    col_axis="min_silence_ms",
    target=None,
    top=None,
    tie_break="row-major",
):
    """Render a 2-D grid sweep as a JSON string.

    Machine-readable twin of :func:`render_vad_grid`, so the grid can feed a
    plotting/tuning script. The payload carries both swept axis names
    (``row_axis`` / ``col_axis``) so a consumer knows which two dimensions the
    cells vary (the cells are keyed by those same names). Any ``None`` in
    ``results`` → ``{"available": false}`` + install hint, mirroring
    :func:`render_vad_sweep_json`.

    When ``target`` (a desired segment count) is given, the payload gains a
    ``"target"`` int and a ``"best"`` cell — :func:`pick_best_grid_cell`'s pick,
    the cell whose segment count is closest to the target, augmented with a
    ``"distance"`` key (``|num_segments - target|``). ``target=None`` (the
    default) omits both keys, keeping the iter-240 payload unchanged.

    When ``top`` (a positive shortlist length) is ALSO given, the payload gains
    a ``"top"`` list of the closest ``top`` cells (:func:`pick_top_grid_cells`),
    ranked nearest-first, each augmented with the same ``"distance"`` key — its
    head equals ``"best"``. ``top`` is ignored without a ``target``.

    iter-243: when ``target`` is given the payload also carries a
    ``"tie_break"`` string naming how equal-distance cells were ordered
    (``"row-major"`` — the default — or ``"speech"``), so a consumer knows which
    tie-break produced the ``best`` / ``top`` ordering. ``target=None`` omits it
    along with the other pick keys. Pure: returns a single JSON string.
    """
    if any(r is None for r in results):
        return json.dumps(
            {
                "available": False,
                "hint": (
                    "install 'silero-vad' (pulls torch + torchaudio) to enable "
                    "offline neural segmentation"
                ),
            },
            indent=2,
        )
    cells = vad_segmentation_grid(
        row_values, col_values, results, row_axis=row_axis, col_axis=col_axis
    )
    payload = {
        "available": True,
        "name": name,
        "row_axis": row_axis,
        "col_axis": col_axis,
        "grid": cells,
    }
    if target is not None:
        payload["target"] = target
        payload["tie_break"] = tie_break
        best = pick_best_grid_cell(cells, target, tie_break)
        payload["best"] = (
            None
            if best is None
            else {**best, "distance": grid_cell_distance(best, target)}
        )
        if top is not None:
            payload["top"] = [
                {**cell, "distance": grid_cell_distance(cell, target)}
                for cell in pick_top_grid_cells(cells, target, top, tie_break)
            ]
    return json.dumps(payload, indent=2)


def render_vad_grid_csv(
    row_values,
    col_values,
    results,
    *,
    name,
    row_axis="threshold",
    col_axis="min_silence_ms",
):
    """Render a 2-D grid sweep as CSV text (no trailing newline).

    The spreadsheet/plot-friendly twin of :func:`render_vad_grid_json`: a flat
    ``<row_axis>,<col_axis>,num_segments,speech_s`` table (one row per cell, in
    row-major order) that pivots straight into a spreadsheet or a plotting
    script without a JSON-parsing step. The first two column headers are the
    swept axis names so the grid is self-describing. ``name`` is accepted for
    signature parity with the other ``render_vad_grid_*`` twins but is not part
    of the tabular body (a CSV is a pure data grid). Any ``None`` in ``results``
    (segmenter unavailable) yields a single ``# silero VAD unavailable: ...``
    comment line so a degraded run is still self-describing rather than silently
    empty. Pure: returns a single string built with the stdlib :mod:`csv`
    writer (RFC-4180 quoting), trailing terminator stripped.
    """
    if any(r is None for r in results):
        return (
            "# silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([row_axis, col_axis, "num_segments", "speech_s"])
    for cell in vad_segmentation_grid(
        row_values, col_values, results, row_axis=row_axis, col_axis=col_axis
    ):
        writer.writerow(
            [cell[row_axis], cell[col_axis], cell["num_segments"], cell["speech_s"]]
        )
    return buf.getvalue().rstrip("\r\n")


def _signed(n):
    """Format an int delta with an explicit sign (``+3`` / ``0`` / ``-2``)."""
    return f"+{n}" if n > 0 else str(n)


def _signed_float(x):
    """Format a float delta with an explicit sign, 1 decimal place."""
    return f"+{x:.1f}" if x > 0 else f"{x:.1f}"


def _load_wpm_mirror():
    """Load ``session/wpm_mirror.py`` directly by file path.

    ``session/__init__.py`` eagerly imports pipecat-dependent modules
    (``session.compute``) that aren't installable everywhere, but the mirror is
    pure stdlib. Loading it by path keeps this offline simulator importable on
    any platform — the same bypass the ``wpm_mirror`` unit tests use.
    """
    import importlib.util

    if "_gv_wpm_mirror" in sys.modules:
        return sys.modules["_gv_wpm_mirror"]

    path = Path(__file__).resolve().parent.parent / "session" / "wpm_mirror.py"
    spec = importlib.util.spec_from_file_location("_gv_wpm_mirror", path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass()'s ``cls.__module__`` lookup resolves
    # (frozen dataclasses in the module read sys.modules at class-creation).
    sys.modules["_gv_wpm_mirror"] = module
    spec.loader.exec_module(module)
    return module


def cmd_simulate_mirror(args, *, log=print):
    """Run the offline WPM-mirror simulator and print the report.

    Single-config trajectory mode by default; grid-sweep mode with ``--grid``.
    The engine (``session/wpm_mirror.py``) is loaded lazily by file path so the
    parser stays audio-free and importable on any platform. ``log`` is
    injectable for tests.
    """
    wm = _load_wpm_mirror()
    WpmMirrorConfig = wm.WpmMirrorConfig
    simulate_speed_trajectory = wm.simulate_speed_trajectory
    sweep_mirror_grid = wm.sweep_mirror_grid
    pick_best_mirror_config = wm.pick_best_mirror_config

    if args.grid:
        points = sweep_mirror_grid(
            args.wpms,
            args.base_wpms,
            args.strengths,
            initial_speed=args.initial_speed,
        )
        best = pick_best_mirror_config(points)
        for line in render_grid(points, best):
            log(line)
        return

    config = WpmMirrorConfig(
        enabled=True,
        base_wpm=args.base_wpm,
        strength=args.strength,
    )
    traj = simulate_speed_trajectory(
        args.wpms,
        initial_speed=args.initial_speed,
        config=config,
    )
    for line in render_trajectory(traj, wpms=args.wpms):
        log(line)


def cmd_calibrate_base_wpm(args, *, log=print):
    """Fold rendered TTS samples into a measured ``base_wpm`` and print it.

    The iter-220 calibration core (``CalibrationSample`` / ``calibrate_base_wpm``)
    is the audio-free arithmetic; this CLI handler is its iter-218-style
    CLI-later twin. Each ``--samples`` triple (``words:audio_seconds[:speed]``)
    is one render measured offline; the handler builds the samples and reports
    the robust median ``implied_base_wpm`` so a deployment can set
    ``DEFAULT_BASE_WPM`` from its own voice instead of the 165 nominal.

    The engine is loaded lazily by file path so the parser stays audio-free and
    importable on any platform. ``log`` is injectable for tests.
    """
    wm = _load_wpm_mirror()
    CalibrationSample = wm.CalibrationSample
    calibrate_base_wpm = wm.calibrate_base_wpm

    samples = [
        CalibrationSample(words=int(words), audio_seconds=audio_seconds, speed=speed)
        for (words, audio_seconds, speed) in args.samples
    ]
    calib = calibrate_base_wpm(samples, default_base_wpm=args.nominal)
    for line in render_calibration(calib):
        log(line)
    if args.verdict:
        # iter-223: fold the raw calibration into the iter-222 adopt/keep
        # verdict so the operator sees a DECISION, not just spread/drift.
        verdict = wm.calibration_verdict(
            calib,
            spread_max=args.spread_max,
            drift_min=args.drift_min,
            min_samples=args.min_samples,
        )
        for line in render_calibration_verdict(verdict):
            log(line)


def cmd_vad(args, *, log=print, segmenter=None, availability=None):
    """Segment a WAV file offline through Silero VAD and print the regions.

    The iter-231 Silero batch segmenter was reachable only via the :5111
    HTTP endpoint (``POST /vad/silero``) and the ``fixtures/replay_silero.py``
    script. This brings it to the gv CLI: ``gv vad recording.wav`` prints the
    detected speech regions (start/end/duration) for any 16-bit PCM WAV, no
    server required — the headless offline analogue of the live mic path.

    Dependencies are injected for testability (mirrors ``dispatch``'s handler
    injection): ``segmenter`` is the ``segment_recording`` callable and
    ``availability`` is the ``silero_available`` probe. Both default to the real
    :mod:`vad.silero` functions, imported lazily here so the parser stays
    audio/torch-free and importable on any platform. When ``silero-vad`` is
    absent the handler prints a clean install hint and returns rather than
    crashing — the same degrade-don't-die contract the server's 503 follows.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)

    if not availability():
        if as_json:
            log(render_vad_json(None))
        else:
            for line in render_vad_segments(None):
                log(line)
        return

    from vad.silero import SileroParams

    params = SileroParams(
        threshold=args.threshold,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
        max_speech_s=args.max_speech_s,
    )
    result = segmenter(args.wav, params=params)
    if as_json:
        log(render_vad_json(result, threshold=args.threshold))
    else:
        for line in render_vad_segments(result, threshold=args.threshold):
            log(line)


def cmd_vad_diff(args, *, log=print, segmenter=None, availability=None):
    """Segment one WAV under two thresholds and report how the result shifts.

    The first consumer of the iter-234 ``gv vad --json`` surface: instead of
    eyeballing two separate ``gv vad`` reports, ``gv vad-diff recording.wav``
    runs Silero twice (``--threshold-a`` then ``--threshold-b``, all other
    knobs shared) and prints the signed segment-count / speech-seconds delta.
    Useful for tuning the P(speech) gate against the recording corpus.

    Same injected-dependency contract as :func:`cmd_vad`: ``segmenter`` /
    ``availability`` default to the real :mod:`vad.silero` functions, imported
    lazily so the parser stays torch-free. When ``silero-vad`` is absent the
    handler prints the install hint and returns, never crashing.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)

    if not availability():
        if as_json:
            log(render_vad_diff_json(None, None, label_a=args.threshold_a,
                                     label_b=args.threshold_b))
        else:
            for line in render_vad_diff(None, None, label_a=args.threshold_a,
                                        label_b=args.threshold_b):
                log(line)
        return

    from vad.silero import SileroParams

    def _seg(threshold):
        params = SileroParams(
            threshold=threshold,
            min_speech_ms=args.min_speech_ms,
            min_silence_ms=args.min_silence_ms,
            speech_pad_ms=args.speech_pad_ms,
            max_speech_s=args.max_speech_s,
        )
        return segmenter(args.wav, params=params)

    result_a = _seg(args.threshold_a)
    result_b = _seg(args.threshold_b)
    if as_json:
        log(render_vad_diff_json(result_a, result_b, label_a=args.threshold_a,
                                 label_b=args.threshold_b))
    else:
        for line in render_vad_diff(result_a, result_b, label_a=args.threshold_a,
                                    label_b=args.threshold_b):
            log(line)


def cmd_vad_sweep(args, *, log=print, segmenter=None, availability=None):
    """Segment one WAV across a swept knob and print a sweep table.

    Generalises the iter-235 two-point ``gv vad-diff`` to a sweep over N values
    of ONE knob (the analogue of ``fixtures/replay_silero.py``'s sweep over the
    corpus). The default axis is the P(speech) gate:
    ``gv vad-sweep recording.wav --thresholds 0.3,0.5,0.7,0.9`` segments once per
    threshold (all other knobs shared) and tabulates segment-count / speech
    seconds vs threshold, so the gate's elbow is visible at a glance.

    iter-238 adds a SECOND axis — the trailing-silence hangover. Passing
    ``--min-silences 400,600,800,1000`` switches the swept dimension to
    ``min_silence_ms`` (the gate is then held fixed at scalar ``--threshold``),
    so an operator can find the hangover elbow where adjacent speech regions
    start merging. iter-239 adds a THIRD axis — the minimum-speech floor.
    Passing ``--min-speeches 50,100,200,400`` sweeps ``min_speech_ms`` (the gate
    again held at scalar ``--threshold``), so an operator can find the floor
    where short speech regions start getting dropped. The three axes are
    mutually exclusive (the parser rejects passing more than one); exactly one
    knob varies per run.

    Same injected-dependency contract as :func:`cmd_vad` / :func:`cmd_vad_diff`:
    ``segmenter`` / ``availability`` default to the real :mod:`vad.silero`
    functions, imported lazily so the parser stays torch-free. When
    ``silero-vad`` is absent the handler prints the install hint and returns,
    never crashing.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)
    as_csv = getattr(args, "csv", False)
    # iter-244: --target (a desired segment count) drives the data-driven
    # best-value pick, mirroring vad-grid's iter-241→243 machinery; None (the
    # default) leaves the iter-236 output unchanged. --top extends it to a
    # ranked shortlist of the N closest values; --tie-break selects how
    # equal-distance values order ("row-major" keeps sweep order, "speech"
    # prefers the value that recovered most speech). All three are derived
    # views, so the CSV emitter (a pure data grid) ignores them.
    target = getattr(args, "target", None)
    top = getattr(args, "top", None)
    tie_break = getattr(args, "tie_break", "row-major")

    # Pick the swept axis: --min-silences sweeps the hangover, --min-speeches the
    # minimum-speech floor (both with the gate held at scalar --threshold);
    # otherwise sweep --thresholds (the default iter-236 behaviour) with the
    # millisecond knobs held at their scalars. The parser guarantees at most one
    # of the three is set.
    min_silences = getattr(args, "min_silences", None)
    min_speeches = getattr(args, "min_speeches", None)
    if min_silences is not None:
        axis = "min_silence_ms"
        values = min_silences
    elif min_speeches is not None:
        axis = "min_speech_ms"
        values = min_speeches
    else:
        axis = "threshold"
        values = args.thresholds

    if not availability():
        if as_json:
            log(render_vad_sweep_json(
                [], [None], name=args.wav, axis=axis, target=target, top=top,
                tie_break=tie_break,
            ))
        elif as_csv:
            log(render_vad_sweep_csv([], [None], name=args.wav, axis=axis))
        else:
            for line in render_vad_sweep(
                [], [None], name=args.wav, axis=axis, target=target, top=top,
                tie_break=tie_break,
            ):
                log(line)
        return

    from vad.silero import SileroParams

    def _seg(value):
        # The swept axis takes ``value``; every other dimension is held at its
        # scalar knob. Every non-swept knob is shared across all runs.
        threshold = value if axis == "threshold" else args.threshold
        min_silence_ms = value if axis == "min_silence_ms" else args.min_silence_ms
        min_speech_ms = value if axis == "min_speech_ms" else args.min_speech_ms
        params = SileroParams(
            threshold=threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=args.speech_pad_ms,
            max_speech_s=args.max_speech_s,
        )
        return segmenter(args.wav, params=params)

    results = [_seg(v) for v in values]
    # Use the segmenter's own name (basename) so the sweep matches `gv vad`'s
    # report; fall back to the raw path only if the sweep is empty.
    name = results[0].name if results else args.wav
    if as_json:
        log(render_vad_sweep_json(
            values, results, name=name, axis=axis, target=target, top=top,
            tie_break=tie_break,
        ))
    elif as_csv:
        log(render_vad_sweep_csv(values, results, name=name, axis=axis))
    else:
        for line in render_vad_sweep(
            values, results, name=name, axis=axis, target=target, top=top,
            tie_break=tie_break,
        ):
            log(line)


def cmd_vad_grid(args, *, log=print, segmenter=None, availability=None):
    """Segment one WAV across a 2-D knob grid and print a grid table.

    The 2-D analogue of :func:`cmd_vad_sweep` (and the direct counterpart of
    ``simulate-mirror --grid``): where ``gv vad-sweep`` tabulates ONE knob
    across N values, ``gv vad-grid`` tabulates the cartesian product of TWO
    knobs so the elbow is readable in two dimensions at once.

    The ROW axis is always the P(speech) gate (``--thresholds``). The COLUMN
    axis is a millisecond knob: ``--min-silences`` (the trailing-silence
    hangover, the default) or ``--min-speeches`` (the minimum-speech floor) —
    mutually exclusive, exactly one column axis per run. Whichever ms knob is
    NOT the column axis is held fixed at its scalar (``--min-silence-ms`` /
    ``--min-speech-ms``), and every other knob is shared across all cells.

    Same injected-dependency contract as :func:`cmd_vad_sweep`: ``segmenter`` /
    ``availability`` default to the real :mod:`vad.silero` functions, imported
    lazily so the parser stays torch-free. When ``silero-vad`` is absent the
    handler prints the install hint and returns, never crashing.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)
    as_csv = getattr(args, "csv", False)
    # --target (a desired segment count) drives the data-driven best-cell pick;
    # None (the default) leaves the iter-240 output unchanged. The CSV emitter is
    # a pure data grid, so it ignores the target (the pick is a derived scalar,
    # not a per-cell column). --top extends that pick into a ranked shortlist of
    # the N closest cells; it rides along with --target (no target = no distance
    # to rank by) and is likewise CSV-irrelevant.
    target = getattr(args, "target", None)
    top = getattr(args, "top", None)
    # iter-243: --tie-break selects how equal-distance cells order within the
    # pick / shortlist. "row-major" (the default) keeps the original grid order
    # (iter-241/242 behaviour, byte-for-byte); "speech" breaks ties on recovered
    # speech (most first). A derived ordering, so CSV ignores it too.
    tie_break = getattr(args, "tie_break", "row-major")

    # Rows are always the gate; the column axis is whichever ms list was passed
    # (--min-speeches → floor; else --min-silences → hangover, the default). The
    # parser's mutex guarantees at most one ms list is set.
    row_axis = "threshold"
    row_values = args.thresholds
    min_speeches = getattr(args, "min_speeches", None)
    if min_speeches is not None:
        col_axis = "min_speech_ms"
        col_values = min_speeches
    else:
        col_axis = "min_silence_ms"
        col_values = args.min_silences

    if not availability():
        unavailable = [None]
        if as_json:
            log(
                render_vad_grid_json(
                    [], [], unavailable, name=args.wav,
                    row_axis=row_axis, col_axis=col_axis, target=target, top=top,
                    tie_break=tie_break,
                )
            )
        elif as_csv:
            log(
                render_vad_grid_csv(
                    [], [], unavailable, name=args.wav,
                    row_axis=row_axis, col_axis=col_axis,
                )
            )
        else:
            for line in render_vad_grid(
                [], [], unavailable, name=args.wav,
                row_axis=row_axis, col_axis=col_axis, target=target, top=top,
                tie_break=tie_break,
            ):
                log(line)
        return

    from vad.silero import SileroParams

    def _seg(row_value, col_value):
        # The two grid axes take (row_value, col_value); every other dimension
        # is held at its scalar knob. The non-column ms knob is held fixed.
        min_silence_ms = (
            col_value if col_axis == "min_silence_ms" else args.min_silence_ms
        )
        min_speech_ms = (
            col_value if col_axis == "min_speech_ms" else args.min_speech_ms
        )
        params = SileroParams(
            threshold=row_value,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=args.speech_pad_ms,
            max_speech_s=args.max_speech_s,
        )
        return segmenter(args.wav, params=params)

    # Row-major: row 0's whole row of columns first, then row 1's, … — the same
    # order vad_segmentation_grid flattens into.
    results = [_seg(rv, cv) for rv in row_values for cv in col_values]
    name = results[0].name if results else args.wav
    if as_json:
        log(
            render_vad_grid_json(
                row_values, col_values, results, name=name,
                row_axis=row_axis, col_axis=col_axis, target=target, top=top,
                tie_break=tie_break,
            )
        )
    elif as_csv:
        log(
            render_vad_grid_csv(
                row_values, col_values, results, name=name,
                row_axis=row_axis, col_axis=col_axis,
            )
        )
    else:
        for line in render_vad_grid(
            row_values, col_values, results, name=name,
            row_axis=row_axis, col_axis=col_axis, target=target, top=top,
            tie_break=tie_break,
        ):
            log(line)


def cmd_bench(args):
    # bench is a legacy argv-driven entrypoint: it parses its own sys.argv
    # rather than taking kwargs, so we rebuild argv here. Only forward
    # --model when it differs from the default so the bench parser keeps
    # using its own default otherwise.
    from mic_bench import main as bench_main
    sys.argv = ["gv bench"]
    if args.model != DEFAULT_MODEL:
        sys.argv.extend(["--model", args.model])
    bench_main()


def cmd_stream(args):
    from mic_stream import run_stream
    run_stream(model_repo=args.model)


def cmd_talk(args):
    from mic_talk import run_talk
    run_talk(model_repo=args.model, voice=args.voice, speed=args.speed)


def cmd_chat(args):
    from mic_chat import run_chat
    run_chat(model_repo=args.model, voice=args.voice, speed=args.speed)


# Command-name → handler. Injectable so dispatch() can be unit-tested with
# stub handlers instead of importing the audio modules.
DEFAULT_HANDLERS = {
    "bench": cmd_bench,
    "stream": cmd_stream,
    "talk": cmd_talk,
    "chat": cmd_chat,
    "simulate-mirror": cmd_simulate_mirror,
    "calibrate-base-wpm": cmd_calibrate_base_wpm,
    "vad": cmd_vad,
    "vad-diff": cmd_vad_diff,
    "vad-sweep": cmd_vad_sweep,
    "vad-grid": cmd_vad_grid,
}

# Seed mirror tunables, mirrored as the CLI defaults so the simulator's
# fixed-rate report matches the live SpeedController's seed config. Imported
# lazily inside build_parser so the module stays importable even if the
# session package were unavailable; falls back to the documented constants.
_MIRROR_DEFAULT_BASE_WPM = 165.0
_MIRROR_DEFAULT_STRENGTH = 0.5
_MIRROR_DEFAULT_CALIB_SPREAD_MAX = 10.0
_MIRROR_DEFAULT_CALIB_DRIFT_MIN = 5.0
_MIRROR_DEFAULT_CALIB_MIN_SAMPLES = 3


def build_parser():
    """Construct the gv argument parser.

    Pure: no I/O, no audio imports. The returned parser is safe to
    exercise from tests with ``parse_args([...])``.
    """
    parser = argparse.ArgumentParser(prog="gv", description="geno-voice CLI")
    sub = parser.add_subparsers(dest="command")

    # Source the mirror seed defaults from the engine so the simulator's
    # fixed-rate report matches the live SpeedController; fall back to the
    # documented constants if the engine can't be loaded.
    try:
        wm = _load_wpm_mirror()
        base_wpm_default = wm.DEFAULT_BASE_WPM
        strength_default = wm.DEFAULT_STRENGTH
        calib_spread_max_default = wm.DEFAULT_CALIB_SPREAD_MAX
        calib_drift_min_default = wm.DEFAULT_CALIB_DRIFT_MIN
        calib_min_samples_default = wm.DEFAULT_CALIB_MIN_SAMPLES
    except Exception:  # pragma: no cover - defensive fallback
        base_wpm_default = _MIRROR_DEFAULT_BASE_WPM
        strength_default = _MIRROR_DEFAULT_STRENGTH
        calib_spread_max_default = _MIRROR_DEFAULT_CALIB_SPREAD_MAX
        calib_drift_min_default = _MIRROR_DEFAULT_CALIB_DRIFT_MIN
        calib_min_samples_default = _MIRROR_DEFAULT_CALIB_MIN_SAMPLES

    bench = sub.add_parser("bench", help="Batch mode — transcribe after silence")
    bench.add_argument("--model", type=model_type, default=DEFAULT_MODEL)

    stream = sub.add_parser("stream", help="Streaming mode — live progressive transcription")
    stream.add_argument("--model", type=model_type, default=DEFAULT_MODEL)

    talk = sub.add_parser("talk", help="Talk mode — STT → NLP → canned response → TTS")
    talk.add_argument("--model", type=model_type, default=DEFAULT_MODEL)
    talk.add_argument(
        "--voice",
        type=voice_type,
        default="af_heart",
        help="TTS voice id, e.g. af_heart / bf_emma (default: af_heart)",
    )
    talk.add_argument(
        "--speed",
        type=speed_type,
        default=1.0,
        help=f"TTS speed in [{SPEED_MIN}, {SPEED_MAX}] (default: 1.0)",
    )

    chat = sub.add_parser("chat", help="Chat mode — STT → LLM (litellm) → TTS")
    chat.add_argument("--model", type=model_type, default=DEFAULT_MODEL)
    chat.add_argument(
        "--voice",
        type=voice_type,
        default="af_heart",
        help="TTS voice id, e.g. af_heart / bf_emma (default: af_heart)",
    )
    chat.add_argument(
        "--speed",
        type=speed_type,
        default=1.0,
        help=f"TTS speed in [{SPEED_MIN}, {SPEED_MAX}] (default: 1.0)",
    )

    sim = sub.add_parser(
        "simulate-mirror",
        help="Offline WPM-mirror simulator — replay a user-WPM arc through the "
        "mirror (no audio); --grid sweeps base_wpm × strength",
    )
    sim.add_argument(
        "--wpms",
        type=wpm_list_type,
        required=True,
        help="Comma-separated per-turn user WPM arc, e.g. 120,140,200,140,120 "
        "(a <=0 entry models a silent / no-measurement turn)",
    )
    sim.add_argument(
        "--initial-speed",
        type=speed_type,
        default=1.0,
        dest="initial_speed",
        help=f"Speed before any turn, in [{SPEED_MIN}, {SPEED_MAX}] (default: 1.0)",
    )
    sim.add_argument(
        "--base-wpm",
        type=float,
        default=base_wpm_default,
        dest="base_wpm",
        help=f"Single-config base_wpm calibration (default: {base_wpm_default})",
    )
    sim.add_argument(
        "--strength",
        type=float,
        default=strength_default,
        help=f"Single-config damping strength in [0, 1] "
        f"(default: {strength_default})",
    )
    sim.add_argument(
        "--grid",
        action="store_true",
        help="Grid-sweep mode: score every --base-wpms × --strengths cell and "
        "print the data-driven best pick",
    )
    sim.add_argument(
        "--base-wpms",
        type=positive_floats_type,
        default=[150.0, 165.0, 180.0],
        dest="base_wpms",
        help="Grid base_wpm axis (comma-separated; default: 150,165,180)",
    )
    sim.add_argument(
        "--strengths",
        type=positive_floats_type,
        default=[0.3, 0.5, 0.7],
        help="Grid strength axis (comma-separated; default: 0.3,0.5,0.7)",
    )

    calib = sub.add_parser(
        "calibrate-base-wpm",
        help="Offline base_wpm calibration — fold rendered TTS samples "
        "(words:audio_seconds[:speed]) into a measured base_wpm (no audio)",
    )
    calib.add_argument(
        "--samples",
        type=calibration_sample_type,
        nargs="+",
        required=True,
        metavar="WORDS:SECONDS[:SPEED]",
        help="One or more rendered samples as 'words:audio_seconds[:speed]', "
        "e.g. 50:18.2 50:9.1:2.0 (speed defaults to the 1.0 calibration point)",
    )
    calib.add_argument(
        "--nominal",
        type=float,
        default=base_wpm_default,
        help=f"Nominal base_wpm to report drift against (default: {base_wpm_default})",
    )
    calib.add_argument(
        "--verdict",
        action="store_true",
        help="Also print the iter-222 adopt/keep verdict (re-seed vs keep nominal) "
        "instead of just the raw spread/drift numbers",
    )
    calib.add_argument(
        "--spread-max",
        type=float,
        default=calib_spread_max_default,
        dest="spread_max",
        help="Verdict gate: max trusted per-sample spread in WPM "
        f"(default: {calib_spread_max_default})",
    )
    calib.add_argument(
        "--drift-min",
        type=float,
        default=calib_drift_min_default,
        dest="drift_min",
        help="Verdict gate: min absolute drift worth re-seeding for in WPM "
        f"(default: {calib_drift_min_default})",
    )
    calib.add_argument(
        "--min-samples",
        type=int,
        default=calib_min_samples_default,
        dest="min_samples",
        help="Verdict gate: min sample count for a robust median "
        f"(default: {calib_min_samples_default})",
    )

    # gv vad — offline Silero segmentation of a WAV file. Defaults mirror
    # SileroParams (sourced from the engine so the CLI tracks the real knobs);
    # fall back to the documented constants if vad.silero can't be imported
    # (keeps the parser construction audio/torch-free on any host).
    try:
        from vad.silero import SileroParams as _SP

        _sp = _SP()
        vad_threshold_default = _sp.threshold
        vad_min_speech_default = _sp.min_speech_ms
        vad_min_silence_default = _sp.min_silence_ms
        vad_speech_pad_default = _sp.speech_pad_ms
        vad_max_speech_default = _sp.max_speech_s
    except Exception:  # pragma: no cover - defensive fallback
        vad_threshold_default = 0.5
        vad_min_speech_default = 250.0
        vad_min_silence_default = 800.0
        vad_speech_pad_default = 30.0
        vad_max_speech_default = float("inf")

    vad = sub.add_parser(
        "vad",
        help="Offline Silero VAD — segment a WAV file into speech regions "
        "(no server / no mic; the headless analogue of the live mic path)",
    )
    vad.add_argument(
        "wav",
        help="Path to a 16-bit PCM WAV file to segment",
    )
    vad.add_argument(
        "--threshold",
        type=unit_interval_type,
        default=vad_threshold_default,
        help=f"P(speech) gate in [0, 1] (default: {vad_threshold_default})",
    )
    vad.add_argument(
        "--min-speech-ms",
        type=nonneg_float_type,
        default=vad_min_speech_default,
        dest="min_speech_ms",
        help="Drop speech regions shorter than this, in ms "
        f"(default: {vad_min_speech_default})",
    )
    vad.add_argument(
        "--min-silence-ms",
        type=nonneg_float_type,
        default=vad_min_silence_default,
        dest="min_silence_ms",
        help="Trailing silence before a region ends, in ms — matches the "
        f"pipecat stop_secs=0.8 live default (default: {vad_min_silence_default})",
    )
    vad.add_argument(
        "--speech-pad-ms",
        type=nonneg_float_type,
        default=vad_speech_pad_default,
        dest="speech_pad_ms",
        help="Symmetric padding added to each region, in ms "
        f"(default: {vad_speech_pad_default})",
    )
    vad.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds; 'inf'/'none' "
        "never splits (default: inf)",
    )
    vad.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable report "
        "(mirrors fixtures/replay_silero.py --json / SileroResult.to_dict)",
    )

    # gv vad-diff — segment one WAV under two thresholds and report the delta.
    # The first consumer of gv vad --json: compares the P(speech) gate at two
    # settings without eyeballing two separate reports. All knobs but threshold
    # are shared between the two runs.
    vad_diff = sub.add_parser(
        "vad-diff",
        help="Offline Silero VAD — segment a WAV at two thresholds and report "
        "the segment-count / speech-seconds delta (tune the P(speech) gate)",
    )
    vad_diff.add_argument(
        "wav",
        help="Path to a 16-bit PCM WAV file to segment under both thresholds",
    )
    vad_diff.add_argument(
        "--threshold-a",
        type=unit_interval_type,
        default=vad_threshold_default,
        dest="threshold_a",
        help=f"First P(speech) gate in [0, 1] (default: {vad_threshold_default})",
    )
    vad_diff.add_argument(
        "--threshold-b",
        type=unit_interval_type,
        default=0.7,
        dest="threshold_b",
        help="Second P(speech) gate in [0, 1] (default: 0.7)",
    )
    vad_diff.add_argument(
        "--min-speech-ms",
        type=nonneg_float_type,
        default=vad_min_speech_default,
        dest="min_speech_ms",
        help="Drop speech regions shorter than this, in ms — shared by both "
        f"runs (default: {vad_min_speech_default})",
    )
    vad_diff.add_argument(
        "--min-silence-ms",
        type=nonneg_float_type,
        default=vad_min_silence_default,
        dest="min_silence_ms",
        help="Trailing silence before a region ends, in ms — shared by both "
        f"runs (default: {vad_min_silence_default})",
    )
    vad_diff.add_argument(
        "--speech-pad-ms",
        type=nonneg_float_type,
        default=vad_speech_pad_default,
        dest="speech_pad_ms",
        help="Symmetric padding added to each region, in ms — shared by both "
        f"runs (default: {vad_speech_pad_default})",
    )
    vad_diff.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds — shared by "
        "both runs; 'inf'/'none' never splits (default: inf)",
    )
    vad_diff.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable report",
    )

    # gv vad-sweep — segment one WAV across a swept knob and tabulate the result.
    # Generalises iter-235's two-point vad-diff to a sweep so the knob's elbow is
    # visible at a glance. iter-236 swept the P(speech) gate (--thresholds);
    # iter-238 adds a second axis, the trailing-silence hangover (--min-silences);
    # iter-239 adds a third, the minimum-speech floor (--min-speeches). The three
    # axes are mutually exclusive. Every non-swept knob is shared across all runs.
    vad_sweep = sub.add_parser(
        "vad-sweep",
        help="Offline Silero VAD — segment a WAV across a swept knob "
        "(--thresholds gate, --min-silences hangover, or --min-speeches floor) "
        "and tabulate segment-count / speech-seconds (find the knob's elbow)",
    )
    vad_sweep.add_argument(
        "wav",
        help="Path to a 16-bit PCM WAV file to segment at each swept value",
    )
    # The swept axis: --thresholds (default) OR --min-silences OR --min-speeches,
    # never more than one. The default list lives on --thresholds so a bare
    # `vad-sweep rec.wav` keeps the iter-236 behaviour; a group default isn't
    # "provided", so the mutex only fires when two are passed explicitly.
    vad_sweep_axis = vad_sweep.add_mutually_exclusive_group()
    vad_sweep_axis.add_argument(
        "--thresholds",
        type=unit_interval_list_type,
        default=[0.3, 0.5, 0.7, 0.9],
        help="Comma-separated P(speech) gates in [0, 1] to sweep "
        "(default: 0.3,0.5,0.7,0.9; mutually exclusive with the ms axes)",
    )
    vad_sweep_axis.add_argument(
        "--min-silences",
        type=nonneg_float_list_type,
        default=None,
        dest="min_silences",
        help="Comma-separated trailing-silence hangovers in ms to sweep "
        "instead of the gate (e.g. 400,600,800,1000); the gate is held at the "
        "scalar --threshold (mutually exclusive with the other axes)",
    )
    vad_sweep_axis.add_argument(
        "--min-speeches",
        type=nonneg_float_list_type,
        default=None,
        dest="min_speeches",
        help="Comma-separated minimum-speech floors in ms to sweep instead of "
        "the gate (e.g. 50,100,200,400); the gate is held at the scalar "
        "--threshold (mutually exclusive with the other axes)",
    )
    vad_sweep.add_argument(
        "--threshold",
        type=unit_interval_type,
        default=vad_threshold_default,
        help="Scalar P(speech) gate held fixed when sweeping --min-silences or "
        "--min-speeches, in [0, 1]; ignored when sweeping --thresholds "
        f"(default: {vad_threshold_default})",
    )
    vad_sweep.add_argument(
        "--min-speech-ms",
        type=nonneg_float_type,
        default=vad_min_speech_default,
        dest="min_speech_ms",
        help="Drop speech regions shorter than this, in ms — shared by all "
        "runs when sweeping --thresholds; ignored when sweeping --min-speeches "
        f"(default: {vad_min_speech_default})",
    )
    vad_sweep.add_argument(
        "--min-silence-ms",
        type=nonneg_float_type,
        default=vad_min_silence_default,
        dest="min_silence_ms",
        help="Trailing silence before a region ends, in ms — shared by all "
        "runs when sweeping --thresholds; ignored when sweeping --min-silences "
        f"(default: {vad_min_silence_default})",
    )
    vad_sweep.add_argument(
        "--speech-pad-ms",
        type=nonneg_float_type,
        default=vad_speech_pad_default,
        dest="speech_pad_ms",
        help="Symmetric padding added to each region, in ms — shared by all "
        f"runs (default: {vad_speech_pad_default})",
    )
    vad_sweep.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds — shared by "
        "all runs; 'inf'/'none' never splits (default: inf)",
    )
    vad_sweep.add_argument(
        "--target",
        type=nonneg_int_type,
        default=None,
        dest="target",
        help="Desired segment count — when given, a data-driven 'best:' pick "
        "names the swept value whose recovered segment count is closest to it "
        "(the same machinery as vad-grid's --target); omit for just the table",
    )
    vad_sweep.add_argument(
        "--top",
        type=pos_int_type,
        default=None,
        dest="top",
        help="With --target, also list the N swept values closest to the target "
        "as a ranked shortlist (nearest first) so the runners-up are visible, "
        "not just the single 'best:' pick; ignored without --target",
    )
    vad_sweep.add_argument(
        "--tie-break",
        choices=("row-major", "speech"),
        default="row-major",
        dest="tie_break",
        help="How to break ties between values equally close to --target: "
        "'row-major' (default) keeps the earlier swept value; 'speech' prefers "
        "the value that recovered the most speech (clips the talker least); "
        "ignored without --target",
    )
    vad_sweep_fmt = vad_sweep.add_mutually_exclusive_group()
    vad_sweep_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable table",
    )
    vad_sweep_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit a flat <axis>,num_segments,speech_s CSV table for "
        "spreadsheets/plots (mutually exclusive with --json)",
    )

    # gv vad-grid — segment one WAV across a 2-D knob grid and tabulate the
    # result. The 2-D analogue of iter-236's vad-sweep (and the direct
    # counterpart of simulate-mirror --grid): rows are always the P(speech)
    # gate (--thresholds); the column axis is a millisecond knob, either the
    # trailing-silence hangover (--min-silences, default) or the minimum-speech
    # floor (--min-speeches), mutually exclusive. The non-column ms knob is held
    # at its scalar; every other knob is shared across all cells.
    vad_grid = sub.add_parser(
        "vad-grid",
        help="Offline Silero VAD — segment a WAV across a 2-D grid (gate "
        "--thresholds × an ms column axis: --min-silences hangover or "
        "--min-speeches floor) and tabulate segment-count / speech-seconds "
        "per cell (read the elbow in two dimensions at once)",
    )
    vad_grid.add_argument(
        "wav",
        help="Path to a 16-bit PCM WAV file to segment at each grid cell",
    )
    vad_grid.add_argument(
        "--thresholds",
        type=unit_interval_list_type,
        default=[0.3, 0.5, 0.7, 0.9],
        help="Comma-separated P(speech) gates in [0, 1] — the grid ROW axis "
        "(default: 0.3,0.5,0.7,0.9)",
    )
    # The column axis: --min-silences (default) OR --min-speeches, never both.
    # The default list lives on --min-silences so a bare `vad-grid rec.wav`
    # sweeps gate × hangover; a group default isn't "provided", so the mutex
    # only fires when both are passed explicitly.
    vad_grid_col = vad_grid.add_mutually_exclusive_group()
    vad_grid_col.add_argument(
        "--min-silences",
        type=nonneg_float_list_type,
        default=[400.0, 600.0, 800.0, 1000.0],
        dest="min_silences",
        help="Comma-separated trailing-silence hangovers in ms — the grid "
        "COLUMN axis (default: 400,600,800,1000; mutually exclusive with "
        "--min-speeches)",
    )
    vad_grid_col.add_argument(
        "--min-speeches",
        type=nonneg_float_list_type,
        default=None,
        dest="min_speeches",
        help="Comma-separated minimum-speech floors in ms to use as the grid "
        "COLUMN axis instead of the hangover (e.g. 50,100,200,400); the "
        "non-column ms knob is held at its scalar (mutually exclusive with "
        "--min-silences)",
    )
    vad_grid.add_argument(
        "--min-speech-ms",
        type=nonneg_float_type,
        default=vad_min_speech_default,
        dest="min_speech_ms",
        help="Drop speech regions shorter than this, in ms — held fixed across "
        "all cells when the column axis is --min-silences; ignored when "
        f"sweeping --min-speeches (default: {vad_min_speech_default})",
    )
    vad_grid.add_argument(
        "--min-silence-ms",
        type=nonneg_float_type,
        default=vad_min_silence_default,
        dest="min_silence_ms",
        help="Trailing silence before a region ends, in ms — held fixed across "
        "all cells when the column axis is --min-speeches; ignored when "
        f"sweeping --min-silences (default: {vad_min_silence_default})",
    )
    vad_grid.add_argument(
        "--speech-pad-ms",
        type=nonneg_float_type,
        default=vad_speech_pad_default,
        dest="speech_pad_ms",
        help="Symmetric padding added to each region, in ms — shared by all "
        f"cells (default: {vad_speech_pad_default})",
    )
    vad_grid.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds — shared by "
        "all cells; 'inf'/'none' never splits (default: inf)",
    )
    vad_grid.add_argument(
        "--target",
        type=nonneg_int_type,
        default=None,
        dest="target",
        help="Desired segment count — when given, a data-driven 'best:' pick "
        "names the cell whose recovered segment count is closest to it (the "
        "vad-grid analogue of simulate-mirror --grid's best pick); omit for "
        "just the table",
    )
    vad_grid.add_argument(
        "--top",
        type=pos_int_type,
        default=None,
        dest="top",
        help="With --target, also list the N cells closest to the target as a "
        "ranked shortlist (nearest first) so the runners-up are visible, not "
        "just the single 'best:' pick; ignored without --target",
    )
    vad_grid.add_argument(
        "--tie-break",
        choices=("row-major", "speech"),
        default="row-major",
        dest="tie_break",
        help="How to break ties between cells equally close to --target: "
        "'row-major' (default) keeps the earlier grid cell; 'speech' prefers "
        "the cell that recovered the most speech (clips the talker least); "
        "ignored without --target",
    )
    vad_grid_fmt = vad_grid.add_mutually_exclusive_group()
    vad_grid_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable table",
    )
    vad_grid_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit a flat <row_axis>,<col_axis>,num_segments,speech_s CSV "
        "table for spreadsheets/plots (mutually exclusive with --json)",
    )

    return parser


def dispatch(args, parser, *, handlers=None):
    """Route parsed args to the matching command handler.

    Returns the process exit code: 0 on a dispatched command, 1 when no
    (or an unknown) command was given. Handlers are injectable for
    testing; the default map wires the real audio entrypoints.
    """
    handlers = DEFAULT_HANDLERS if handlers is None else handlers

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    handler(args)
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return dispatch(args, parser)


if __name__ == "__main__":
    sys.exit(main())
