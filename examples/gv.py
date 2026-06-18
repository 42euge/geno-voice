#!/usr/bin/env python3
"""geno-voice CLI.

Usage:
    gv bench              # batch mode — wait for silence, transcribe, show timing
    gv stream             # streaming mode — live progressive transcription
    gv talk               # talk mode — STT → NLP → canned response → TTS
    gv chat               # chat mode — STT → LLM (litellm) → TTS
    gv simulate-mirror …  # offline WPM-mirror trajectory / grid-sweep simulator
    gv calibrate-base-wpm … # offline base_wpm calibration (--verdict for an adopt/keep call)
    gv <cmd> --model ...  # override STT model
"""

import argparse
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
