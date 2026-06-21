#!/usr/bin/env python3
"""geno-voice CLI.

Usage:
    gv bench              # batch mode — wait for silence, transcribe, show timing
    gv stream             # streaming mode — live progressive transcription
    gv talk               # talk mode — STT → NLP → canned response → TTS
    gv chat               # chat mode — STT → LLM (litellm) → TTS
    gv simulate-mirror …  # offline WPM-mirror trajectory / grid-sweep simulator
    gv calibrate-base-wpm … # offline base_wpm calibration (--verdict for an adopt/keep call; --json/--csv for per-sample data)
    gv vad recording.wav  # offline Silero VAD — segment a WAV into speech regions
    gv vad recording.wav --json # machine-readable segmentation (SileroResult.to_dict shape)
    gv vad-gaps recording.wav  # report the silence gaps BETWEEN speech regions (tune --min-silence-ms)
    gv vad-gaps recording.wav --json # machine-readable gap stats + per-gap list
    gv vad-gap-percentiles recording.wav --percentiles 50,90,99  # robust pause percentiles (outlier-proof vs min/mean/max)
    gv vad-gap-cdf recording.wav --cuts-ms 200,400,800,1600  # merge-CDF — what fraction of pauses each --min-silence-ms cut would merge
    gv vad-gap-cost recording.wav --cuts-ms 200,400,800,1600  # merge cost curve — marginal pauses merged per +100ms (derivative of the CDF; zero-rate bands are valleys)
    gv vad-gap-peak recording.wav --cuts-ms 200,400,800,1600  # verdict — names the costliest band (densest pause cluster / steepest CDF; where NOT to raise --min-silence-ms)
    gv vad-gap-recommend recording.wav  # verdict — recommends a --min-silence-ms in the valley between short/long pauses
    gv vad-gap-confidence recording.wav  # grade how trustworthy the recommendation is (strong/moderate/weak by valley dominance)
    gv vad-gap-hist recording.wav --bin-width-s 0.5  # histogram the silence-gap durations (see the distribution shape)
    gv vad-gap-sweep recording.wav --thresholds 0.3,0.5,0.7,0.9  # sweep N gates, tabulate how the min gap moves
    gv vad-gap-grid recording.wav --thresholds 0.3,0.5,0.7 --min-silences 400,800  # gate × hangover gap grid
    gv vad-diff recording.wav --threshold-a 0.5 --threshold-b 0.7  # compare two P(speech) gates
    gv vad-gap-diff recording.wav --threshold-a 0.5 --threshold-b 0.7  # how the silence-gap distribution shifts
    gv vad-sweep recording.wav --thresholds 0.3,0.5,0.7,0.9  # sweep N gates, tabulate the elbow
    gv vad-sweep recording.wav --min-silences 200,400,800,1600  # sweep the hangover instead
    gv vad-sweep recording.wav --min-speeches 50,100,200,400  # sweep the min-speech floor instead
    gv vad-sweep recording.wav --max-speeches 5,10,20,inf  # sweep the force-split ceiling (seconds)
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


def percentile_type(raw):
    """Argparse ``type`` for a single percentile in the range ``(0, 100]``.

    The scalar twin of :func:`percentile_list_type`, modelled on
    :func:`unit_interval_type`. Used by ``gv vad-gap-peak --min-rate-pct``
    (iter-357), which derives a rate floor from a percentile of the observed
    band rates instead of an absolute number. A percentile must be a number in
    the OPEN-AT-zero, CLOSED-at-100 range ``(0, 100]`` (``0`` would name the
    trivial minimum — keep every band, already the default; ``>100`` is
    meaningless), not NaN. Pure and side-effect-free for direct unit testing;
    raises :class:`argparse.ArgumentTypeError` on a non-number, NaN, or
    out-of-range value.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"percentile must be a number, got {raw!r}")
    if value != value:  # NaN is unordered.
        raise argparse.ArgumentTypeError("percentile must be a number, got nan")
    if not (0.0 < value <= 100.0):
        raise argparse.ArgumentTypeError(
            f"percentile must be in (0, 100], got {value}"
        )
    return value


def percentile_list_type(raw):
    """Argparse ``type`` for ``gv vad-gap-percentiles --percentiles``: a list.

    iter-338's ``gv vad-gap-percentiles`` reports robust order statistics of the
    inter-segment silence distribution (p50/p90/p99 by default) — unlike the
    min/mean/max ``gv vad-gaps`` reports, a percentile is unmoved by a single
    outlier pause, so it is the stable signal for choosing the end-of-turn
    hangover. This parses a comma-separated list (e.g. ``"50,90,99"``) into
    ``[50.0, 90.0, 99.0]`` at the parser. Each token must be a number in the
    OPEN-AT-zero, CLOSED-at-100 range ``(0, 100]`` (0 would be the trivial
    minimum, already covered by ``gv vad-gaps``; >100 is meaningless), not NaN.
    Duplicates and unsorted input are preserved as given (the operator may want
    a specific column order). Rejects an empty list. Pure and side-effect-free
    for direct unit testing — the percentile twin of
    :func:`unit_interval_list_type`.
    """
    if not isinstance(raw, str):
        raise argparse.ArgumentTypeError(f"percentiles must be a string, got {raw!r}")
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError(
            f"percentiles must be a non-empty comma-separated list, got {raw!r}"
        )
    return [percentile_type(tok) for tok in tokens]


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


def cut_ms_list_type(raw):
    """Argparse ``type`` for ``gv vad-gap-cdf --cuts-ms``: a millisecond list.

    iter-346's ``gv vad-gap-cdf`` evaluates the empirical CDF of the silence gaps
    at candidate end-of-turn hangover cuts — for each cut it reports the fraction
    of pauses that hangover would MERGE (pauses shorter than the cut). This parses
    a comma-separated list (e.g. ``"200,400,800,1600"``) into
    ``[200.0, 400.0, 800.0, 1600.0]`` at the parser. Each token must be a valid
    :func:`nonneg_float_type` value (a number ``>= 0``, not NaN — ``0`` is
    legitimate: a zero hangover merges nothing); duplicates and unsorted input are
    preserved as given (the operator may want a specific column order). Rejects an
    empty list. A thin millisecond-semantics alias of
    :func:`nonneg_float_list_type` kept distinct so the ``--cuts-ms`` error text
    reads in cut terms. Pure and side-effect-free for direct unit testing.
    """
    if not isinstance(raw, str):
        raise argparse.ArgumentTypeError(
            f"cuts must be a string, got {raw!r}"
        )
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError(
            f"cuts must be a non-empty comma-separated list, got {raw!r}"
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


def positive_float_type(raw):
    """Argparse ``type`` for ``simulate-mirror``'s ``--min-speed`` / ``--max-speed``
    intelligibility-band overrides (iter-319).

    Each is a TTS speed multiplier floor/ceiling handed to ``WpmMirrorConfig``,
    where ``WpmMirrorConfig.__post_init__`` requires ``min_speed > 0`` and
    ``max_speed >= min_speed``. A non-positive band edge is rejected here so the
    operator gets the usual argparse ``SystemExit(2)`` instead of a config
    ``ValueError`` deep in the handler. The cross-edge ordering
    (``max >= min``) is still left to the config validator since it spans two
    args. The positive-scalar twin of :func:`nonneg_float_type`.

    Pure and side-effect-free for direct unit testing.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"value must be a number, got {raw!r}")
    if value != value:  # NaN is unordered.
        raise argparse.ArgumentTypeError("value must be a number, got nan")
    if value <= 0:
        raise argparse.ArgumentTypeError(f"value must be positive, got {value}")
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


def positive_int_type(raw):
    """Argparse ``type`` for ``gv vad-gap-peak``'s ``--top-n`` (iter-354): a count >= 1.

    The number of costliest cost-curve bands to name. ``vad-gap-peak`` names the
    single steepest band by default; ``--top-n N`` asks for the N steepest. A
    count of zero would name nothing (pointless), and negatives / fractionals are
    nonsensical — so the floor is 1, not 0. The strictly-positive integer twin of
    :func:`nonneg_int_type`; raises :class:`argparse.ArgumentTypeError` on a
    non-integer or a value below 1. Pure and side-effect-free for direct testing.
    """
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"value must be a whole number, got {raw!r}"
        )
    if value < 1:
        raise argparse.ArgumentTypeError(f"value must be >= 1, got {value}")
    return value


def nonneg_penalty_type(raw):
    """Parse a weighted-set ``:penalty`` weight — a non-negative number (iter-251).

    iter-250's weighted-set penalty was a non-negative WHOLE number
    (:func:`nonneg_int_type`); a count could only be "1 segment worse", "2 worse",
    etc. iter-251 widens it to a non-negative FLOAT so an operator can dial the
    preference strength FINELY — ``3,5:1.5`` means "the accepted count 5 is 1.5
    segments worse than the preferred 3", landing the override threshold between
    whole-number penalties. The penalty stays ADDITIVE on the element's distance
    (:func:`grid_cell_distance`), so a fractional penalty interpolates the
    "preferred count wins at a larger raw distance" boundary an integer penalty
    could only step across.

    An INTEGRAL float collapses back to an ``int`` (``5:2`` → penalty ``2``,
    byte-for-byte the iter-250 result), so every prior integer-penalty
    parse/render/distance/JSON case is unchanged; only a genuinely fractional
    value stays a ``float`` (``5:1.5`` → ``1.5``). ``0`` is legitimate (an
    explicit unweighted element); negatives, NaN, and ``inf`` are rejected (a
    negative penalty would make a count BETTER than its raw distance — that is
    what the OTHER element's penalty already expresses — and an infinite penalty
    is a degenerate "never pick this" that should just drop the element). Pure
    and side-effect-free for direct unit testing; raises
    :class:`argparse.ArgumentTypeError` on a non-number, NaN, inf, or negative.
    The fractional twin of :func:`nonneg_int_type` for the weight slot.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"penalty must be a number, got {raw!r}"
        )
    if value != value:  # NaN is unordered.
        raise argparse.ArgumentTypeError("penalty must be a number, got nan")
    if value == float("inf"):
        raise argparse.ArgumentTypeError("penalty must be finite, got inf")
    if value < 0:
        raise argparse.ArgumentTypeError(f"penalty must be >= 0, got {value}")
    # An integral float collapses to an int so iter-250's integer-penalty output
    # (parse value, _format_target, JSON) is byte-for-byte unchanged; only a
    # genuinely fractional value stays a float.
    if value == int(value):
        return int(value)
    return value


def scale_factor_type(raw):
    """Parse a scaled-set ``*factor`` multiplier — a number ``>= 1`` (iter-252).

    iter-250/251's ``:penalty`` weight is ADDITIVE: it adds a constant to one
    element's distance (``distance + penalty``), so a less-preferred count is
    "N segments worse" no matter how far the cell drifts. iter-252 adds the
    MULTIPLICATIVE twin — a ``*factor`` weight that SCALES the element's distance
    (``distance * factor``), for the operator who thinks proportionally ("count 5
    is acceptable, but every segment I drift past it hurts 1.5× as much"). The two
    differ in a way worth stating: an additive penalty bites even an EXACT hit
    (``0 + penalty = penalty``), while a multiplicative factor leaves an exact hit
    free (``0 * factor = 0``) and only grows the cost as the cell count moves AWAY
    from the element — preference that scales with distance rather than offsets it.

    The factor is a number ``>= 1``: ``1`` is neutral (an element with no ``*``,
    scored unweighted) and a factor ``< 1`` is rejected — a factor below 1 would
    DISCOUNT an element's distance, making it MORE preferred than neutral, which
    is exactly what the OTHER elements' larger factors already express (the same
    symmetry as :func:`nonneg_penalty_type`, where ``0`` is neutral and negatives
    are rejected). NaN and ``inf`` are rejected (an infinite factor is a
    degenerate "never pick this off-exact"). An INTEGRAL float collapses to an
    ``int`` (``5*2.0`` → factor ``2``), so a whole-number factor renders and
    serialises cleanly; only a genuinely fractional value stays a ``float``
    (``5*1.5`` → ``1.5``). Pure and side-effect-free for direct unit testing;
    raises :class:`argparse.ArgumentTypeError` on a non-number, NaN, inf, or a
    value below 1. The multiplicative twin of :func:`nonneg_penalty_type`.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"factor must be a number, got {raw!r}"
        )
    if value != value:  # NaN is unordered.
        raise argparse.ArgumentTypeError("factor must be a number, got nan")
    if value == float("inf"):
        raise argparse.ArgumentTypeError("factor must be finite, got inf")
    if value < 1:
        raise argparse.ArgumentTypeError(f"factor must be >= 1, got {value}")
    # An integral float collapses to an int so a whole-number factor renders and
    # serialises as a plain int; only a genuinely fractional value stays a float.
    if value == int(value):
        return int(value)
    return value


def target_type(raw):
    """Argparse ``type`` for ``gv vad-grid``/``vad-sweep`` ``--target``.

    iter-246 generalises the scalar segment-count target into an optional
    tolerance BAND. The single-count form (``--target 3``) still parses to an
    ``int`` — byte-for-byte the iter-241→245 behaviour, so a scalar target's
    distance and output are unchanged. The closed range form (``--target 3-5``)
    parses to a ``(lo, hi)`` int tuple meaning "anywhere from 3 to 5 segments is
    perfect": :func:`grid_cell_distance` scores 0 for any count inside the
    inclusive band and the gap to the nearest edge otherwise, so an operator who
    wants "between 3 and 5 regions" no longer has to eyeball the table.

    iter-247 adds the OPEN-ended forms: ``--target 3-`` ("at least 3", the HI
    edge is open) parses to ``(3, None)``, and ``--target -5`` ("at most 5", the
    LO edge is open) parses to ``(None, 5)``. An open edge means "no bound on
    that side" — :func:`grid_cell_distance` scores 0 for any count on the
    satisfied side and the gap to the closed edge on the other. They reuse the
    same ``(lo, hi)`` tuple shape (with ``None`` marking the open edge) so the
    distance/pick/render machinery flows through with no parallel path.

    iter-248 adds the comma-separated SET form: ``--target 3,5,7`` means "3 OR 5
    OR 7 segments, but nothing between" and parses to a ``list`` of elements,
    each itself a scalar or a band (so ``3,5-7`` = "3 OR anywhere from 5 to 7"
    composes for free). :func:`grid_cell_distance` scores a set as the MIN
    distance to any listed element, so an operator with two acceptable counts no
    longer has to eyeball the table. A set is deduped preserving first-seen order
    and a degenerate set whose elements all collapse to one (``3,3``) reduces to
    that bare element, keeping scalar/band output byte-for-byte unchanged.

    iter-249 adds the ``>``-separated PREFERENCE-ORDER form: ``--target 3>5>7``
    means "prefer 3 segments, accept 5, tolerate 7 as a last resort". Like the
    set it accepts ANY listed count (distance is the MIN to any element, so all
    listed counts score 0), but UNLIKE the set it carries a precedence so a
    distance TIE breaks toward the earlier-listed (more preferred) element — the
    operator who would rather have 3 regions but settles for 5 finally has a way
    to say which wins when both are reachable. It parses to a ``{"prefer":
    [...]}`` dict (each element a scalar or band, so ``3>5-7`` composes), deduped
    preserving order, and a single-element preference (``3>3``) collapses to the
    bare element so scalar/band output stays byte-for-byte unchanged. Mixing
    ``,`` and ``>`` in one target is rejected — they are different composition
    operators (a flat OR vs a ranked OR) and stacking them is ambiguous.

    iter-250 adds the ``:penalty`` WEIGHTED-SET form on a comma set:
    ``--target 3,5:2`` means "prefer 3, accept 5 but treat it as 2 segments
    WORSE than it is". Where iter-249's preference breaks only EXACT distance
    ties, the weight folds preference INTO the distance — the penalty is ADDED to
    that element's distance — so a more-preferred count can win even at a
    slightly LARGER raw distance (e.g. count 3 at distance 1 beats count 5 at
    distance 0 once 5 carries a +2 penalty). It parses to a ``{"weighted":
    [(element, penalty), ...]}`` dict (each element a scalar or band, so
    ``3,5-7:2`` composes), deduped on the element preserving first-seen order,
    and a weighted set that collapses to one element reduces to the bare element
    (a lone penalty is a constant offset that cannot change a pick). A ``:``
    weight requires a ``,`` set (it is meaningless on a single element) and
    cannot be combined with ``>`` (preference) — both express preference, so
    stacking them is ambiguous. Each penalty is a non-negative number.

    iter-251 widens the ``:penalty`` weight from a whole number to a non-negative
    FLOAT (``--target 3,5:1.5``), so an operator can dial how strongly preference
    outweighs distance BETWEEN whole-number steps — ``5:1.5`` places the
    "preferred count wins at a larger raw distance" boundary halfway. An integral
    float collapses back to an ``int`` (``5:2`` → ``2``), so every iter-250
    integer-penalty parse/render/distance/JSON result is byte-for-byte unchanged.

    iter-252 adds the ``*factor`` MULTIPLICATIVE-weight form on a comma set:
    ``--target 3,5*1.5`` means "prefer 3, accept 5 but every segment I drift PAST
    5 hurts 1.5× as much". Where iter-250/251's ``:penalty`` is ADDITIVE (the
    weight OFFSETS the distance — it bites even an exact hit, ``0 + penalty``), the
    ``*factor`` weight SCALES the distance (``distance * factor``) — an exact hit
    stays free (``0 * factor = 0``) and the cost grows only as the cell count
    drifts away, for the operator who thinks proportionally rather than by a fixed
    offset. It parses to a ``{"scaled": [(element, factor), ...]}`` dict (each
    element a scalar or band, so ``3,5-7*1.5`` composes), deduped on the element
    preserving first-seen order, and a scaled set that collapses to one element
    reduces to the bare element (a lone factor scales every cell's distance
    uniformly and cannot change a pick). Each factor is a number ``>= 1`` (``1`` is
    neutral); ``*`` requires a ``,`` set, and cannot be combined with ``>``
    (preference), ``:`` (the additive weight — a set is additively OR
    multiplicatively weighted, not both at once). The ``{"scaled": ...}`` dict
    rides the same min-over-elements distance / pick / render / JSON machinery as
    ``{"weighted": ...}``, only with ``*`` instead of ``+`` folding the weight in.

    iter-287 lets the ``:penalty`` and ``*factor`` weights CO-OCCUR on one set —
    the AFFINE form: ``--target 3,5*1.5:2`` means "prefer 3, accept 5 but every
    segment past it costs 1.5× AND it starts 2 worse". When BOTH operators appear
    somewhere in the set it parses to a ``{"affine": [(element, factor, penalty),
    ...]}`` dict scoring ``distance * factor + penalty`` per element (the factor
    scales, the penalty then offsets) — composing iter-252's multiplicative factor
    and iter-250's additive penalty. Per element both weights are optional and
    order-free (``5*1.5:2`` == ``5:2*1.5``), defaulting to factor ``1`` / penalty
    ``0``. A set carrying ONLY ``:`` is still the iter-250 weighted dict and ONLY
    ``*`` still the iter-252 scaled dict — affine activates only on the mix — so
    every prior weighted/scaled parse is byte-for-byte unchanged. ``*``/``:`` still
    require a ``,`` set and still cannot stack with ``>`` (preference).

    Each present band edge is a non-negative whole number (reusing
    :func:`nonneg_int_type`'s rules — ``0`` is legitimate, negatives/fractionals
    are rejected), the band separator is a single ``-``, and for a CLOSED band
    ``lo <= hi`` (an inverted band is a typo, not a degenerate window). Exactly
    one edge may be empty (the open forms); a bare ``-`` with both edges empty
    is meaningless and rejected. A bare count with no ``-`` is the scalar form.
    An empty set/preference element (``3,``, ``3,,5``, ``3>``, ``>5``) is
    rejected. Pure and side-effect-free for direct unit testing; raises
    :class:`argparse.ArgumentTypeError` on any malformed input.
    """
    text = raw.strip() if isinstance(raw, str) else raw
    has_set = isinstance(text, str) and "," in text
    has_pref = isinstance(text, str) and ">" in text
    has_weight = isinstance(text, str) and ":" in text
    has_scale = isinstance(text, str) and "*" in text
    # iter-249: ',' (a flat set) and '>' (a ranked preference) are different
    # composition operators; stacking them in one target is ambiguous, so reject.
    if has_set and has_pref:
        raise argparse.ArgumentTypeError(
            f"target cannot mix ',' (set) and '>' (preference), got {raw!r}"
        )
    # iter-250: a ':penalty' weight (a WEIGHTED set) folds preference INTO the
    # distance, where the '>' preference only breaks exact ties — they are
    # different intent expressions, so stacking them is ambiguous. And a weight
    # is meaningless on a single element (a constant offset that cannot change
    # the pick), so it requires the comma-set context.
    if has_weight and has_pref:
        raise argparse.ArgumentTypeError(
            f"target cannot mix ':' (weight) and '>' (preference), got {raw!r}"
        )
    if has_weight and not has_set:
        raise argparse.ArgumentTypeError(
            f"target ':' weight requires a ',' set (e.g. 3,5:2), got {raw!r}"
        )
    # iter-252: a '*factor' weight (a SCALED set) folds preference into the
    # distance MULTIPLICATIVELY. Like ':penalty' it expresses preference, so it
    # cannot stack with '>' (preference), and a factor is meaningless on a single
    # element (it scales every cell's distance uniformly and cannot change a
    # pick), so it requires the comma-set context. iter-287: a ':penalty' and a
    # '*factor' MAY now co-occur — the AFFINE set (see below), where each element
    # is scaled THEN offset (distance*factor + penalty). The prior
    # "cannot mix ':' and '*'" rejection is gone.
    if has_scale and has_pref:
        raise argparse.ArgumentTypeError(
            f"target cannot mix '*' (factor) and '>' (preference), got {raw!r}"
        )
    if has_scale and not has_set:
        raise argparse.ArgumentTypeError(
            f"target '*' factor requires a ',' set (e.g. 3,5*1.5), got {raw!r}"
        )
    # iter-249: '>' separates a PREFERENCE order — prefer the earlier-listed
    # count, accept the later ones; the precedence breaks distance ties (see
    # grid_cell_sort_key). Parses to a {"prefer": [...]} dict so it is distinct
    # from the flat-set list. A single-element preference collapses to the bare
    # element.
    if has_pref:
        elements = _parse_target_collection(text, raw, ">", "preference")
        return elements[0] if len(elements) == 1 else {"prefer": elements}
    # iter-287: a comma-set whose elements carry BOTH a '*factor' and a ':penalty'
    # (anywhere in the set) is an AFFINE set — each element is scaled THEN offset
    # (distance*factor + penalty), the multiplicative and additive weights composed
    # on one element. It generalises both prior forms (factor 1 -> weighted,
    # penalty 0 -> scaled), so it only activates when BOTH operators are present;
    # a set with only ':' stays a {"weighted": ...} dict and only '*' stays a
    # {"scaled": ...} dict, byte-for-byte unchanged. Parses to a distinct
    # {"affine": [(element, factor, penalty), ...]} dict.
    if has_set and has_weight and has_scale:
        return _parse_affine_set(text, raw)
    # iter-250: a comma-set whose elements carry a ':penalty' is a WEIGHTED set —
    # the penalty is ADDED to that element's distance, so a more-preferred count
    # can win even at a slightly larger raw distance (unlike the iter-249
    # preference, which only breaks exact ties). Parses to a {"weighted": [...]}
    # dict distinct from both the flat-set list and the prefer dict.
    if has_set and has_weight:
        return _parse_weighted_set(text, raw)
    # iter-252: a comma-set whose elements carry a '*factor' is a SCALED set — the
    # factor MULTIPLIES that element's distance, so a more-preferred count can win
    # even at a slightly larger raw distance, and the cost grows with distance
    # (unlike the iter-250 additive penalty, which is a fixed offset). Parses to a
    # {"scaled": [...]} dict distinct from the flat-set list, the prefer dict, and
    # the weighted dict.
    if has_set and has_scale:
        return _parse_scaled_set(text, raw)
    # iter-248: a comma separates a SET of acceptable targets, each element
    # itself a scalar or a band. The set scores as the MIN distance to any
    # element (see grid_cell_distance), so '3,5' accepts 3 OR 5 but nothing
    # between. A single-element set collapses to that bare element.
    if has_set:
        elements = _parse_target_collection(text, raw, ",", "set")
        return elements[0] if len(elements) == 1 else elements
    return _parse_single_target(text)


def _parse_target_collection(text, raw, sep, kind):
    """Split, validate, parse, and dedupe a ``--target`` set/preference collection.

    Factored out of :func:`target_type` in iter-249 so the comma-separated SET
    (iter-248) and ``>``-separated PREFERENCE (iter-249) forms share one
    splitter: both split on ``sep``, reject any empty element (a leading/
    trailing/doubled separator is a typo), parse each element via
    :func:`_parse_single_target` (so each may itself be a scalar or band), and
    dedupe preserving first-seen order. Returns the element ``list`` (never
    collapsed — the caller decides how a single-element collection reduces).
    ``kind`` ("set"/"preference") names the form in the error message. Pure;
    raises :class:`argparse.ArgumentTypeError` on an empty or malformed element.
    """
    parts = [p.strip() for p in text.split(sep)]
    if any(p == "" for p in parts):
        raise argparse.ArgumentTypeError(
            f"target {kind} must be {sep!r}-separated targets, got {raw!r}"
        )
    elements = []
    for part in parts:
        element = _parse_single_target(part)
        if element not in elements:  # dedupe, preserve first-seen order
            elements.append(element)
    return elements


def _parse_weighted_set(text, raw):
    """Parse a comma-set whose elements may carry a ``:penalty`` weight (iter-250).

    The WEIGHTED set generalises the iter-248 flat set: an element written
    ``count:penalty`` (e.g. ``5:2``) adds ``penalty`` to that element's distance
    in :func:`grid_cell_distance`, so a less-penalised (more-preferred) count can
    win even at a slightly LARGER raw distance — the gap iter-249's preference
    left, where the precedence breaks only EXACT distance ties. An element with
    no ``:`` carries penalty ``0`` (the iter-248 element, unweighted). The base of
    each element is parsed by :func:`_parse_single_target`, so a band may be
    weighted too (``3,5-7:2`` = "prefer 3, accept the 5-7 band but 2 worse").

    Returns a ``{"weighted": [(element, penalty), ...]}`` dict (distinct from the
    flat-set ``list`` and the prefer ``dict``), deduped on the element preserving
    first-seen order (so ``3:1,3:2`` keeps the first penalty). A weighted set that
    collapses to a single element reduces to the BARE element — a lone penalty is
    a constant offset that cannot change any pick, so it is dropped to keep
    scalar/band output byte-for-byte unchanged. The penalty is a non-negative
    number (iter-251: :func:`nonneg_penalty_type`, a FLOAT — ``3,5:1.5`` dials
    the preference strength between whole-number steps; an integral float
    collapses to an ``int`` so iter-250's integer-penalty output is unchanged);
    ``0`` is legitimate but redundant. Pure; raises
    :class:`argparse.ArgumentTypeError` on an empty or malformed element/penalty.
    """
    parts = [p.strip() for p in text.split(",")]
    if any(p == "" for p in parts):
        raise argparse.ArgumentTypeError(
            f"target weighted set must be ','-separated targets, got {raw!r}"
        )
    weighted = []
    seen = []
    for part in parts:
        if ":" in part:
            base_text, _, penalty_text = part.partition(":")
            element = _parse_single_target(base_text.strip())
            penalty = nonneg_penalty_type(penalty_text.strip())
        else:
            element = _parse_single_target(part)
            penalty = 0
        if element not in seen:  # dedupe on element, first-seen penalty wins
            seen.append(element)
            weighted.append((element, penalty))
    if len(weighted) == 1:
        return weighted[0][0]
    return {"weighted": weighted}


def _parse_scaled_set(text, raw):
    """Parse a comma-set whose elements may carry a ``*factor`` weight (iter-252).

    The MULTIPLICATIVE twin of :func:`_parse_weighted_set`. An element written
    ``count*factor`` (e.g. ``5*1.5``) MULTIPLIES that element's distance by
    ``factor`` in :func:`grid_cell_distance` (``distance * factor``), so a
    less-scaled (more-preferred) count can win even at a slightly LARGER raw
    distance, and — unlike the iter-250 additive penalty (a fixed offset that bites
    even an exact hit) — the cost grows with distance and an exact hit stays free
    (``0 * factor = 0``). An element with no ``*`` carries factor ``1`` (neutral,
    the iter-248 element unweighted). The base of each element is parsed by
    :func:`_parse_single_target`, so a band may be scaled too (``3,5-7*1.5`` =
    "prefer 3, accept the 5-7 band but drift past it costs 1.5×").

    Returns a ``{"scaled": [(element, factor), ...]}`` dict (distinct from the
    flat-set ``list``, the prefer ``dict``, and the weighted ``dict``), deduped on
    the element preserving first-seen order (so ``3*2,3*3`` keeps the first
    factor). A scaled set that collapses to a single element reduces to the BARE
    element — a lone factor scales every cell's distance uniformly and cannot
    change a pick, so it is dropped to keep scalar/band output byte-for-byte
    unchanged. The factor is a number ``>= 1`` (:func:`scale_factor_type`; ``1`` is
    neutral, an integral float collapses to an ``int``). Pure; raises
    :class:`argparse.ArgumentTypeError` on an empty or malformed element/factor.
    """
    parts = [p.strip() for p in text.split(",")]
    if any(p == "" for p in parts):
        raise argparse.ArgumentTypeError(
            f"target scaled set must be ','-separated targets, got {raw!r}"
        )
    scaled = []
    seen = []
    for part in parts:
        if "*" in part:
            base_text, _, factor_text = part.partition("*")
            element = _parse_single_target(base_text.strip())
            factor = scale_factor_type(factor_text.strip())
        else:
            element = _parse_single_target(part)
            factor = 1
        if element not in seen:  # dedupe on element, first-seen factor wins
            seen.append(element)
            scaled.append((element, factor))
    if len(scaled) == 1:
        return scaled[0][0]
    return {"scaled": scaled}


def _parse_affine_part(part, raw):
    """Parse one AFFINE element ``count[*factor][:penalty]`` (iter-287).

    Both the ``*factor`` (multiplicative, :func:`scale_factor_type`) and
    ``:penalty`` (additive, :func:`nonneg_penalty_type`) weights are OPTIONAL and
    may appear in EITHER order — ``5*1.5:2`` and ``5:2*1.5`` parse identically. The
    base (a scalar or band, parsed by :func:`_parse_single_target`) is the text
    before the first weight operator, so a band composes (``5-7*1.5:2``). A factor
    defaults to ``1`` (neutral) and a penalty to ``0`` (none) when absent, so an
    element with neither is the plain iter-248 element. Returns
    ``(element, factor, penalty)``. Pure; raises
    :class:`argparse.ArgumentTypeError` on a duplicated operator or a malformed
    base/weight.
    """
    if part.count("*") > 1 or part.count(":") > 1:
        raise argparse.ArgumentTypeError(
            f"target affine element has a repeated weight operator, got {raw!r}"
        )
    star = part.find("*")
    colon = part.find(":")
    ops = [p for p in (star, colon) if p != -1]
    base_text = part if not ops else part[: min(ops)]
    element = _parse_single_target(base_text.strip())
    factor = 1
    penalty = 0
    if star != -1:
        # The factor token runs from just after '*' to the next operator (the
        # ':' if it follows '*') or the end of the element.
        end = colon if (colon != -1 and colon > star) else len(part)
        factor = scale_factor_type(part[star + 1 : end].strip())
    if colon != -1:
        end = star if (star != -1 and star > colon) else len(part)
        penalty = nonneg_penalty_type(part[colon + 1 : end].strip())
    return element, factor, penalty


def _parse_affine_set(text, raw):
    """Parse a comma-set whose elements carry BOTH ``*factor`` and ``:penalty`` (iter-287).

    The AFFINE set composes the iter-252 MULTIPLICATIVE factor and the iter-250
    ADDITIVE penalty on one set: each element contributes
    ``grid_cell_distance * factor + penalty`` (see :func:`grid_cell_distance`), the
    factor scaling the distance and the penalty offsetting the scaled result. It
    activates only when the set carries BOTH a ``*`` and a ``:`` SOMEWHERE — a set
    with only ``:`` is the iter-250 weighted set and only ``*`` the iter-252 scaled
    set, both byte-for-byte unchanged. Per element BOTH weights are optional and
    order-free (``5*1.5:2`` == ``5:2*1.5``), defaulting to factor ``1`` / penalty
    ``0``; the base may itself be a scalar or band (``3,5-7*1.5:2`` composes).

    Returns a ``{"affine": [(element, factor, penalty), ...]}`` dict (distinct
    from the flat-set ``list``, the prefer/weighted/scaled dicts), deduped on the
    element preserving first-seen order (so the first weights for a repeated
    element win). An affine set that collapses to a single element reduces to the
    BARE element — a lone factor scales every cell's distance uniformly and a lone
    penalty is a constant offset, so neither can change a pick, keeping
    scalar/band output byte-for-byte unchanged. Pure; raises
    :class:`argparse.ArgumentTypeError` on an empty or malformed element/weight.
    """
    parts = [p.strip() for p in text.split(",")]
    if any(p == "" for p in parts):
        raise argparse.ArgumentTypeError(
            f"target affine set must be ','-separated targets, got {raw!r}"
        )
    affine = []
    seen = []
    for part in parts:
        element, factor, penalty = _parse_affine_part(part, raw)
        if element not in seen:  # dedupe on element, first-seen weights win
            seen.append(element)
            affine.append((element, factor, penalty))
    if len(affine) == 1:
        return affine[0][0]
    return {"affine": affine}


def _parse_single_target(text):
    """Parse one ``--target`` element: a scalar count or a (possibly open) band.

    Factored out of :func:`target_type` in iter-248 so the comma-separated set
    form can parse each element with the same scalar/band/open-band rules. A
    bare count with no ``-`` is the scalar form (a bare ``int``); a ``LO-HI``
    form is a closed/open ``(lo, hi)`` tuple. Pure; raises
    :class:`argparse.ArgumentTypeError` on malformed input.
    """
    # A single '-' separates the band edges; no '-' is the scalar form. (Edges
    # are non-negative, so '-' is never a sign — it is always the separator.)
    if isinstance(text, str) and "-" in text:
        parts = text.split("-")
        if len(parts) != 2 or (parts[0] == "" and parts[1] == ""):
            raise argparse.ArgumentTypeError(
                f"target band must be LO-HI with whole numbers, got {text!r}"
            )
        # An empty edge is the open form ('3-' = at least 3 → (3, None);
        # '-5' = at most 5 → (None, 5)); None marks "no bound on that side".
        lo = nonneg_int_type(parts[0]) if parts[0] != "" else None
        hi = nonneg_int_type(parts[1]) if parts[1] != "" else None
        if lo is not None and hi is not None and lo > hi:
            raise argparse.ArgumentTypeError(
                f"target band LO-HI must have LO <= HI, got {text!r}"
            )
        return (lo, hi)
    return nonneg_int_type(text)


def _format_target(target):
    """Human-readable rendering of a ``--target`` value (scalar int or band).

    A scalar target renders as its bare count (``3``); an iter-246 closed
    ``(lo, hi)`` band renders as ``lo-hi`` (``3-5``). iter-247's open bands keep
    the empty edge empty so they read back exactly as typed: ``(3, None)`` →
    ``3-`` ("at least 3"), ``(None, 5)`` → ``-5`` ("at most 5"). iter-248's set
    form renders each element comma-joined so it reads back as typed:
    ``[3, 5, 7]`` → ``3,5,7``, ``[3, (5, 7)]`` → ``3,5-7``. iter-249's
    preference form (a ``{"prefer": [...]}`` dict) renders ``>``-joined so it too
    reads back as typed: ``{"prefer": [3, 5, 7]}`` → ``3>5>7``. iter-250's
    weighted-set form (a ``{"weighted": [(element, penalty), ...]}`` dict)
    renders comma-joined with each non-zero penalty appended as ``:penalty``:
    ``{"weighted": [(3, 0), (5, 2)]}`` → ``3,5:2``. iter-252's scaled-set form (a
    ``{"scaled": [(element, factor), ...]}`` dict) renders comma-joined with each
    non-neutral factor appended as ``*factor``: ``{"scaled": [(3, 1), (5, 1.5)]}``
    → ``3,5*1.5``. Used by
    :func:`_render_pick_block` so the ``best:`` / ``top N:`` lines read naturally
    for any form without each call site re-deriving the format. Pure.
    """
    if isinstance(target, dict) and "weighted" in target:
        # iter-250: a weighted set renders comma-joined, each element appending
        # ':penalty' only when non-zero so an unweighted element reads as a plain
        # count and the whole thing reads back exactly as typed.
        return ",".join(
            _format_target(element) + (f":{penalty}" if penalty else "")
            for element, penalty in target["weighted"]
        )
    if isinstance(target, dict) and "scaled" in target:
        # iter-252: a scaled set renders comma-joined, each element appending
        # '*factor' only when not the neutral 1 so an unweighted element reads as a
        # plain count and the whole thing reads back exactly as typed.
        return ",".join(
            _format_target(element) + (f"*{factor}" if factor != 1 else "")
            for element, factor in target["scaled"]
        )
    if isinstance(target, dict) and "affine" in target:
        # iter-287: an affine set renders comma-joined, each element appending
        # '*factor' (only when non-neutral) THEN ':penalty' (only when non-zero) —
        # the canonical '*' before ':' order — so a neither-weighted element reads
        # as a plain count and the whole thing reads back parseable as typed.
        return ",".join(
            _format_target(element)
            + (f"*{factor}" if factor != 1 else "")
            + (f":{penalty}" if penalty else "")
            for element, factor, penalty in target["affine"]
        )
    if isinstance(target, dict):
        return ">".join(_format_target(element) for element in target["prefer"])
    if isinstance(target, list):
        return ",".join(_format_target(element) for element in target)
    if isinstance(target, tuple):
        lo, hi = target
        lo_text = "" if lo is None else str(lo)
        hi_text = "" if hi is None else str(hi)
        return f"{lo_text}-{hi_text}"
    return str(target)


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


def max_speech_list_type(raw):
    """Argparse ``type`` for ``gv vad-grid --max-speeches``: a seconds list.

    iter-255 adds the force-split ceiling ``max_speech_s`` as a fourth vad-grid
    column axis (after the hangover, the speech floor, and the region padding —
    all millisecond knobs). Unlike those, this is a SECONDS knob, so it gets its
    own list validator rather than sharing :func:`nonneg_float_list_type`: it
    parses a comma-separated list (e.g. ``"5,10,inf"``) into ``[5.0, 10.0,
    inf]`` at the parser, each token validated by :func:`max_speech_type` so the
    ``inf``/``none``/``off`` "never split" sentinels and the positive-only rule
    (a ``0`` second cap would split forever) carry through per element.
    Duplicates and unsorted input are preserved as given (the operator may want
    a specific column order). Rejects an empty list. Pure and side-effect-free
    for direct unit testing — the seconds twin of :func:`nonneg_float_list_type`.
    """
    if not isinstance(raw, str):
        raise argparse.ArgumentTypeError(
            f"a seconds list must be a string, got {raw!r}"
        )
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError(
            f"a seconds list must be non-empty and comma-separated, got {raw!r}"
        )
    return [max_speech_type(tok) for tok in tokens]


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


def render_calibration_csv(samples, calib):
    """Render a ``calibrate-base-wpm`` calibration as CSV text (no trailing newline).

    The spreadsheet/plot-friendly twin of :func:`render_calibration`, the
    iter-316 next-item that closes the last machine-readable gap among the gv
    analysis surfaces — every VAD surface (``gv vad`` / ``vad-diff`` /
    ``vad-sweep`` / ``vad-grid``) carries the full human / ``--json`` / ``--csv``
    trio, and ``simulate-mirror`` gained ``--csv`` in iter-315, leaving
    ``calibrate-base-wpm`` as the lone surface with only a human report.

    A calibration is a SET of per-render samples folded to ONE verdict, so the
    natural CSV unit is **one row per sample** — the shape a plotter wants (an
    implied-base-wpm-per-sample scatter to eyeball the spread) and a spreadsheet
    wants (one render per line):
    ``sample,words,audio_seconds,speed,bot_wpm,implied_base_wpm``. ``sample`` is
    1-based, matching how an operator numbers their renders. ``bot_wpm`` is each
    render's measured rate and ``implied_base_wpm`` normalizes it back to the
    ``speed=1.0`` calibration point (the per-sample values the median is taken
    over), so the consumer sees both the raw measurement and the comparable
    normalized rate.

    The aggregate verdict (median ``implied_base_wpm``, range, spread, nominal,
    drift) is a single record describing the whole SET, not a per-sample fact, so
    duplicating it into every row would bloat the grid (the same reasoning
    :func:`render_trajectory_csv` uses to keep arc-level scalars out of its
    per-turn rows). Instead it trails as ``#`` comment lines — self-describing
    metadata a plotting/spreadsheet tool skips by default (pandas
    ``read_csv(comment="#")``), matching the ``#``-comment precedent
    :func:`render_vad_sweep_csv` uses for its own non-tabular metadata — so the
    per-sample rows stay a pure, parseable data grid while the calibration's
    bottom line remains visible in the same file. ``calib`` of ``None`` (no
    samples ⇒ nothing to calibrate) yields the header alone, mirroring
    :func:`render_trajectory_csv`'s empty-arc contract. Floats are rounded to 3
    places, matching :func:`render_grid_csv` / :func:`render_trajectory_csv`.
    Pure: returns a single string built with the stdlib :mod:`csv` writer
    (RFC-4180 quoting, ``\\r\\n`` row terminators) with the trailing terminator
    stripped.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["sample", "words", "audio_seconds", "speed", "bot_wpm", "implied_base_wpm"]
    )
    for i, s in enumerate(samples, start=1):
        writer.writerow(
            [
                i,
                s.words,
                round(float(s.audio_seconds), 3),
                round(float(s.speed), 3),
                round(s.bot_wpm, 3),
                round(s.implied_base_wpm, 3),
            ]
        )
    body = buf.getvalue().rstrip("\r\n")
    if calib is None:
        return body
    summary = [
        f"# implied_base_wpm (median): {round(calib.implied_base_wpm, 3)}",
        f"# range: {round(calib.min_base_wpm, 3)} - {round(calib.max_base_wpm, 3)}",
        f"# spread: {round(calib.spread, 3)}",
        f"# nominal: {round(calib.default_base_wpm, 3)}",
        f"# drift: {round(calib.drift, 3)}",
    ]
    return body + "\n" + "\n".join(summary)


def render_calibration_json(samples, calib):
    """Render a ``calibrate-base-wpm`` calibration as a JSON string.

    The nested/programmatic twin of :func:`render_calibration` /
    :func:`render_calibration_csv` — the iter-317 next-item that brings the
    calibration surface the same human / ``--json`` / ``--csv`` trio every
    VAD-analysis surface already carries. Where the CSV splits the calibration
    into a per-sample data grid plus a trailing ``#``-comment summary block (so a
    spreadsheet's rows stay pure), the JSON nests BOTH in one object: a
    ``samples`` list (one object per render — the per-sample data) AND a
    ``calibration`` object (the aggregate verdict — median / range / spread /
    nominal / drift). A nested consumer gets the whole record in one parse
    instead of having to skip comment lines.

    Each sample object is ``{"sample": 1-based int, "words": int,
    "audio_seconds": float, "speed": float, "bot_wpm": float,
    "implied_base_wpm": float}`` — the raw measurement (``bot_wpm``) and the
    normalized rate (``implied_base_wpm``) the median is taken over, the same
    fields the CSV row carries. ``calibration`` is ``null`` when there were no
    samples (nothing to calibrate), mirroring :func:`calibrate_base_wpm`'s empty
    contract and the CSV's header-only output. Like the CSV, the JSON omits the
    adopt/keep verdict — that DECISION is the ``--verdict`` human surface, not a
    data record; a consumer scripts the re-seed off the ``drift`` field. Floats
    round to 3 places. Pure: returns a single JSON string (no I/O).
    """
    sample_objs = [
        {
            "sample": i,
            "words": s.words,
            "audio_seconds": round(float(s.audio_seconds), 3),
            "speed": round(float(s.speed), 3),
            "bot_wpm": round(s.bot_wpm, 3),
            "implied_base_wpm": round(s.implied_base_wpm, 3),
        }
        for i, s in enumerate(samples, start=1)
    ]
    payload = {
        "samples": sample_objs,
        "calibration": (
            None
            if calib is None
            else {
                "implied_base_wpm": round(calib.implied_base_wpm, 3),
                "n_samples": calib.n_samples,
                "min_base_wpm": round(calib.min_base_wpm, 3),
                "max_base_wpm": round(calib.max_base_wpm, 3),
                "spread": round(calib.spread, 3),
                "nominal": round(calib.default_base_wpm, 3),
                "drift": round(calib.drift, 3),
            }
        ),
    }
    return json.dumps(payload, indent=2)


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


def render_grid(points, best, *, lurch_weight=None):
    """Render a grid sweep (``MirrorGridPoint`` list) + the picked best cell.

    Pure: returns a list of strings (no I/O). Each row shows a cell's tunables
    and its convergence / lurch / churn diagnostics; the trailing line names
    the data-driven pick (or notes that no cell was scorable).

    ``lurch_weight`` (iter-318) is the weight on the lurch term in
    :meth:`MirrorGridPoint.score` — the same value the handler passes to
    :func:`pick_best_mirror_config`, so the displayed ``score`` column always
    matches the score the pick was made on. ``None`` defers to the engine's
    :data:`DEFAULT_LURCH_WEIGHT` (the score's own default), preserving the
    pre-iter-318 behaviour for callers that don't supply one.
    """
    lines = ["WPM-mirror grid sweep (base_wpm × strength)"]
    header = "  base_wpm  strength  final  gap     step   moves  score"
    lines.append(header)
    for p in points:
        gap = "  n/a " if p.final_gap is None else f"{p.final_gap:+.3f}"
        score = p.score() if lurch_weight is None else p.score(lurch_weight)
        score_s = " n/a " if score is None else f"{score:.3f}"
        lines.append(
            f"  {p.base_wpm:7.1f}  {p.strength:7.2f}  "
            f"{p.final_speed:5.3f}  {gap}  {p.max_step:5.3f}  "
            f"{p.moves:4d}   {score_s}"
        )
    if best is None:
        lines.append("  best: none (no scorable cell — no measurable turn in the arc)")
    else:
        best_score = (
            best.score() if lurch_weight is None else best.score(lurch_weight)
        )
        lines.append(
            f"  best: base_wpm={best.base_wpm:.1f} strength={best.strength:.2f} "
            f"(score {best_score:.3f})"
        )
    return lines


def render_grid_csv(points, best, *, lurch_weight=None):
    """Render a ``simulate-mirror --grid`` sweep as CSV text (no trailing newline).

    The spreadsheet/plot-friendly twin of :func:`render_grid`, the first
    machine-readable surface on the ``simulate-mirror`` command — the iter-314
    next-item that closes the human / ``--csv`` gap on the WPM-mirror simulator,
    matching the trio the VAD-analysis surfaces already carry (``gv vad`` /
    ``vad-diff`` / ``vad-sweep`` / ``vad-grid``). Where :func:`render_grid`
    prints a padded human table plus a prose "best" footer, this emits a flat
    ``base_wpm,strength,final_speed,final_gap,max_step,moves,score,is_best``
    table — one row per scored cell, in the engine's sweep order — that pipes
    straight into a spreadsheet or plotting script (pandas ``read_csv``,
    matplotlib's ``loadtxt``) without a JSON-parsing step.

    The data-driven pick is folded into the table as an ``is_best`` boolean
    column (``1`` for the cell :func:`pick_best_mirror_config` chose, ``0``
    otherwise) rather than split into a separate prose footer — a CSV is a pure
    data grid, so the verdict belongs in a column the consumer can filter on,
    not in trailing English. ``best`` of ``None`` (no scorable cell) simply
    leaves every ``is_best`` at ``0``. Unscorable cells (no measurable turn)
    carry an empty ``final_gap``/``score`` field (matching the human table's
    ``n/a``) so a reader distinguishes "0.0 gap" from "no gap to compute".
    ``final_speed`` / ``final_gap`` / ``max_step`` are rounded to 3 places and
    ``score`` to 3 places, matching the human report's precision. Pure: returns
    a single string built with the stdlib :mod:`csv` writer (RFC-4180 quoting,
    ``\\r\\n`` row terminators) with the trailing terminator stripped.

    ``lurch_weight`` (iter-318) is the score weight the handler also passes to
    :func:`pick_best_mirror_config`, so the ``score`` column matches the score
    the ``is_best`` flag was decided on. ``None`` defers to the engine default.
    """
    # Identify the picked cell by value identity on its tunables so the flag
    # tracks pick_best_mirror_config's verdict without re-implementing scoring.
    best_key = None if best is None else (best.base_wpm, best.strength)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "base_wpm",
            "strength",
            "final_speed",
            "final_gap",
            "max_step",
            "moves",
            "score",
            "is_best",
        ]
    )
    for p in points:
        score = p.score() if lurch_weight is None else p.score(lurch_weight)
        is_best = 1 if best_key is not None and (p.base_wpm, p.strength) == best_key else 0
        writer.writerow(
            [
                p.base_wpm,
                p.strength,
                round(p.final_speed, 3),
                "" if p.final_gap is None else round(p.final_gap, 3),
                round(p.max_step, 3),
                p.moves,
                "" if score is None else round(score, 3),
                is_best,
            ]
        )
    return buf.getvalue().rstrip("\r\n")


def render_trajectory_csv(traj, *, wpms=None):
    """Render a ``simulate-mirror`` single-config trajectory as CSV text.

    The spreadsheet/plot-friendly twin of :func:`render_trajectory`, the
    trajectory-mode counterpart to :func:`render_grid_csv`. Where the grid CSV
    emits one row per *swept cell*, a single trajectory is one fold over the arc,
    so its natural CSV unit is one row per *turn*:
    ``turn,user_wpm,speed``. That is the machine-readable expansion of the human
    report's ``per-turn speeds:`` line — precisely the shape a plotter wants
    (speed-vs-turn curve, one point per row) and a spreadsheet wants (one turn
    per line). ``turn`` is 1-based, matching how an operator counts the arc.

    ``wpms`` (the per-turn input arc) is paired alongside each output speed when
    supplied so the consumer can plot the user's pace and the bot's response on
    the same axis; when its length matches ``traj.speeds`` each row carries its
    driving ``user_wpm``, otherwise the ``user_wpm`` field is left empty (the
    speeds still emit). The scalar convergence diagnostics the human report
    summarises (final gap, max step, moves) are intentionally NOT duplicated
    into every row — they are arc-level, derivable from the speed column, and
    would only bloat a per-turn grid; a consumer that wants them reads the
    ``--json`` surface or the human report. An empty arc (mirroring disabled /
    no turns) yields the header alone. Speeds and WPMs are rounded to 3 places,
    matching :func:`render_trajectory`. Pure: returns a single string built with
    the stdlib :mod:`csv` writer (RFC-4180 quoting), trailing terminator
    stripped.
    """
    paired = wpms is not None and len(wpms) == len(traj.speeds)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["turn", "user_wpm", "speed"])
    for i, speed in enumerate(traj.speeds, start=1):
        user_wpm = round(float(wpms[i - 1]), 3) if paired else ""
        writer.writerow([i, user_wpm, round(speed, 3)])
    return buf.getvalue().rstrip("\r\n")


def render_grid_json(points, best, *, lurch_weight=None):
    """Render a ``simulate-mirror --grid`` sweep as a JSON string.

    The nested/programmatic twin of :func:`render_grid` / :func:`render_grid_csv`
    — the iter-317 next-item that brings the WPM-mirror simulator's grid mode the
    same human / ``--json`` / ``--csv`` trio every VAD-analysis surface (``gv
    vad`` / ``vad-diff`` / ``vad-sweep`` / ``vad-grid``) already carries. Where
    the CSV emits a flat one-row-per-cell table, the JSON nests the sweep as a
    ``cells`` list of objects plus a top-level ``best`` object — the shape a
    consumer that wants the picked cell as a structured record (not an
    ``is_best`` column to filter on) reaches for.

    Each cell carries its tunables and convergence / lurch / churn diagnostics
    (``base_wpm``, ``strength``, ``final_speed``, ``final_gap``, ``max_step``,
    ``moves``, ``score``). ``final_gap`` / ``score`` are ``null`` for an
    unscorable cell (no measurable turn) — JSON ``null`` distinguishes "no value"
    from "0.0", the same distinction the CSV's empty field and the human table's
    ``n/a`` make. ``best`` is :func:`pick_best_mirror_config`'s pick rendered as
    the same cell shape, or ``null`` when no cell was scorable. Floats round to 3
    places, matching the human / CSV reports. Pure: returns a single JSON string
    (no I/O), built from the points' attributes so it is testable without audio.

    ``lurch_weight`` (iter-318) is the score weight the handler also passes to
    :func:`pick_best_mirror_config`, so each cell's ``score`` and the ``best``
    object's ``score`` are computed on the same weight the pick used. ``None``
    defers to the engine default.
    """
    def _cell(p):
        score = p.score() if lurch_weight is None else p.score(lurch_weight)
        return {
            "base_wpm": p.base_wpm,
            "strength": p.strength,
            "final_speed": round(p.final_speed, 3),
            "final_gap": None if p.final_gap is None else round(p.final_gap, 3),
            "max_step": round(p.max_step, 3),
            "moves": p.moves,
            "score": None if score is None else round(score, 3),
        }

    payload = {
        "mode": "grid",
        "cells": [_cell(p) for p in points],
        "best": None if best is None else _cell(best),
    }
    return json.dumps(payload, indent=2)


def render_trajectory_json(traj, *, wpms=None):
    """Render a ``simulate-mirror`` single-config trajectory as a JSON string.

    The nested/programmatic twin of :func:`render_trajectory` /
    :func:`render_trajectory_csv`, the trajectory-mode counterpart to
    :func:`render_grid_json`. Where the CSV flattens the arc to one row per turn
    (``turn,user_wpm,speed``) and intentionally drops the arc-level scalars, the
    JSON carries BOTH: a ``turns`` list (one object per turn, the per-turn curve)
    AND the convergence diagnostics (``initial_speed`` / ``final_speed`` /
    ``ideal_final_speed`` / ``final_gap`` / ``max_step`` / ``moves``) as
    top-level keys — exactly the human report's fields, but structured. A nested
    consumer gets the whole record in one object instead of having to re-derive
    the scalars from the speed column the way a CSV reader must.

    Each turn object is ``{"turn": 1-based int, "user_wpm": float|null, "speed":
    float}``. ``user_wpm`` pairs the driving input arc when ``wpms`` length
    matches ``traj.speeds`` (the same pairing rule as the CSV), else ``null``.
    ``ideal_final_speed`` / ``final_gap`` are ``null`` when the arc had no
    measurable turn (mirroring disabled / all-silent), matching the human
    report's ``n/a``. An empty arc yields an empty ``turns`` list with the
    scalars still present. Floats round to 3 places. Pure: returns a single JSON
    string built from the trajectory's attributes (no I/O, no audio).
    """
    paired = wpms is not None and len(wpms) == len(traj.speeds)
    turns = [
        {
            "turn": i,
            "user_wpm": round(float(wpms[i - 1]), 3) if paired else None,
            "speed": round(speed, 3),
        }
        for i, speed in enumerate(traj.speeds, start=1)
    ]
    payload = {
        "mode": "trajectory",
        "initial_speed": round(traj.initial_speed, 3),
        "final_speed": round(traj.final_speed, 3),
        "ideal_final_speed": (
            None
            if traj.ideal_final_speed is None
            else round(traj.ideal_final_speed, 3)
        ),
        "final_gap": None if traj.final_gap is None else round(traj.final_gap, 3),
        "max_step": round(traj.max_step, 3),
        "moves": traj.moves,
        "turns": turns,
    }
    return json.dumps(payload, indent=2)


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


def render_vad_csv(result, *, threshold=None):
    """Render a Silero ``SileroResult`` (iter-231) as a per-segment CSV table.

    The spreadsheet/plot-friendly twin of :func:`render_vad_segments` /
    :func:`render_vad_json`, completing the human / ``--json`` / ``--csv`` trio
    on the foundational ``gv vad`` surface — the trio that ``gv vad-sweep`` /
    ``gv vad-diff`` / ``gv vad-grid`` already carry (iter-237 / iter-251 /
    iter-313). Unlike those sweep/diff/grid CSVs (one row per *swept config*,
    aggregating each run to count + speech-seconds), a single ``gv vad`` run has
    exactly one segmentation, so the natural CSV unit is one row per detected
    *segment*: ``index,start_s,end_s,duration_s``. That is precisely the shape a
    plotter wants (draw a span per row) and a spreadsheet wants (one region per
    line), and it is the machine-readable expansion of the human report's
    ``[ n]  start – end (dur)`` lines.

    Seconds are rounded to 3 places, matching :func:`render_vad_json` so the two
    machine surfaces agree to the digit. ``index`` is 1-based, matching the
    human report's ``[ n]`` numbering. A result with zero segments yields the
    header alone (a valid, empty-bodied table — a consumer reads "no speech"
    from the absent rows rather than from prose). ``threshold`` is accepted for
    signature parity with the sibling renderers but is intentionally NOT emitted
    as a column: every row of a single run shares the one gate, so a per-row
    threshold column would be pure redundancy (contrast ``vad-sweep``/``-diff``,
    where the threshold genuinely *varies* row to row and so is a column there).

    ``result`` of ``None`` (segmenter unavailable) yields a single
    ``# silero VAD unavailable: ...`` comment line, matching
    :func:`render_vad_sweep_csv` / :func:`render_vad_diff_csv` so a degraded run
    is self-describing rather than silently empty. Pure: returns a single string
    built with the stdlib :mod:`csv` writer (RFC-4180 quoting, ``\\r\\n`` row
    terminators) with the trailing terminator stripped.
    """
    if result is None:
        return (
            "# silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["index", "start_s", "end_s", "duration_s"])
    for i, seg in enumerate(result.segments, start=1):
        writer.writerow(
            [
                i,
                round(seg.start_s, 3),
                round(seg.end_s, 3),
                round(seg.duration_s, 3),
            ]
        )
    return buf.getvalue().rstrip("\r\n")


def vad_silence_gaps(result):
    """Compute the inter-segment silence gaps of one Silero segmentation (iter-328).

    Every prior ``gv vad*`` surface reports where the *speech* is — segment
    count, per-segment spans, total speech-seconds. None reports where the
    *silence* is: the pauses BETWEEN consecutive speech regions. That gap
    distribution is the direct signal for tuning the end-of-turn hangover (the
    ``--min-silence-ms`` knob here / the live ``chat.vad.silence_duration``,
    iter-020): the SHORTEST real pause in a recording is the floor above which
    raising the hangover starts merging two genuine turns into one, and the
    spread tells the operator how much headroom they have. This is the
    silence-side complement of :func:`render_vad_segments`.

    Pure: takes any object exposing the ``SileroResult`` shape (a ``segments``
    list whose elements have ``start_s`` / ``end_s``) and returns a plain
    ``dict`` — no I/O, no torch import, so it is testable with lightweight
    stand-ins. Segments are sorted by start before differencing so out-of-order
    input is handled robustly, and a negative raw difference (touching /
    overlapping regions, which padding can produce) clamps to ``0.0`` — an
    overlap is not silence. Each gap records the 1-based index of the segment it
    FOLLOWS and that segment's end time, so a consumer can locate the pause in
    the recording. Gap seconds round to 3 places, matching the sibling VAD
    renderers. A result with fewer than 2 segments has no gaps: the ``gaps``
    list is empty and the min/max/mean are ``None`` (no pause to summarise),
    distinct from a ``0.0`` gap.
    """
    segs = sorted(result.segments, key=lambda s: s.start_s)
    gaps = []
    after_segment = []
    after_segment_end_s = []
    for i in range(1, len(segs)):
        prev_end = segs[i - 1].end_s
        raw = segs[i].start_s - prev_end
        # A negative gap means the regions touch/overlap (padding can do this) —
        # that is not silence, so clamp to 0.0.
        gaps.append(round(max(0.0, raw), 3))
        after_segment.append(i)  # 1-based index of the segment the gap follows
        after_segment_end_s.append(round(prev_end, 3))
    n_gaps = len(gaps)
    return {
        "num_segments": len(segs),
        "num_gaps": n_gaps,
        "gaps": gaps,
        "after_segment": after_segment,
        "after_segment_end_s": after_segment_end_s,
        "min_gap_s": min(gaps) if gaps else None,
        "max_gap_s": max(gaps) if gaps else None,
        "mean_gap_s": round(sum(gaps) / n_gaps, 3) if gaps else None,
        "total_silence_s": round(sum(gaps), 3),
    }


def render_vad_gaps(result):
    """Render the inter-segment silence gaps as plain-text report lines (iter-328).

    The human-readable face of :func:`vad_silence_gaps`, the silence-side twin
    of :func:`render_vad_segments`. ``result`` of ``None`` (segmenter
    unavailable) yields the shared install hint, matching the sibling
    renderers. A result with fewer than 2 segments has no gaps to report, so it
    prints a short explanatory line WITHOUT the ``--min-silence-ms`` advice
    (there is no shortest-pause floor to tune against). Otherwise it summarises
    the distribution (min/mean/max, total silence) — naming the actionable knob
    on the min-gap line — then lists each gap with the segment it follows. Pure:
    returns a list of strings (no I/O, no ANSI).
    """
    if result is None:
        return [
            "silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        ]
    d = vad_silence_gaps(result)
    lines = [
        f"silero VAD silence gaps — {result.name}",
        f"  segments:     {d['num_segments']}",
        f"  gaps:         {d['num_gaps']} (pauses between consecutive speech regions)",
    ]
    if d["num_gaps"] == 0:
        lines.append("  (fewer than 2 segments — no inter-segment pause to measure)")
        return lines
    lines.append(
        f"  min gap:      {d['min_gap_s']:.3f}s "
        "(shortest real pause — keep --min-silence-ms below this to avoid "
        "merging turns)"
    )
    lines.append(f"  mean gap:     {d['mean_gap_s']:.3f}s")
    lines.append(f"  max gap:      {d['max_gap_s']:.3f}s")
    lines.append(f"  total silence:{d['total_silence_s']:8.3f}s")
    for i, (gap, after, end) in enumerate(
        zip(d["gaps"], d["after_segment"], d["after_segment_end_s"]), start=1
    ):
        lines.append(
            f"  [{i:>2}] {gap:6.3f}s  after seg {after} (ends {end:.2f}s)"
        )
    return lines


def render_vad_gaps_json(result):
    """Render the inter-segment silence gaps as a JSON string (iter-328).

    The machine-readable twin of :func:`render_vad_gaps`, mirroring the
    degrade-to-``{"available": false}`` contract the other VAD JSON renderers
    use. Carries the aggregate stats (``num_segments`` / ``num_gaps`` /
    ``min_gap_s`` / ``max_gap_s`` / ``mean_gap_s`` / ``total_silence_s``) plus a
    ``gaps`` list of per-gap objects (``index`` 1-based, ``after_segment``,
    ``after_segment_end_s``, ``gap_s``). The min/max/mean are ``null`` when
    there are fewer than 2 segments (no pause to summarise) — JSON ``null``
    distinguishing "no gap" from a ``0.0`` gap, the same distinction the human
    report's omission and the CSV's empty body make. Pure: built from the
    result's attributes, so it works on any ``SileroResult``-shaped object.
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
    d = vad_silence_gaps(result)
    payload = {
        "available": True,
        "name": result.name,
        "num_segments": d["num_segments"],
        "num_gaps": d["num_gaps"],
        "min_gap_s": d["min_gap_s"],
        "max_gap_s": d["max_gap_s"],
        "mean_gap_s": d["mean_gap_s"],
        "total_silence_s": d["total_silence_s"],
        "gaps": [
            {
                "index": i,
                "after_segment": after,
                "after_segment_end_s": end,
                "gap_s": gap,
            }
            for i, (gap, after, end) in enumerate(
                zip(d["gaps"], d["after_segment"], d["after_segment_end_s"]), start=1
            )
        ],
    }
    return json.dumps(payload, indent=2)


def render_vad_gaps_csv(result):
    """Render the inter-segment silence gaps as a per-gap CSV table (iter-328).

    The spreadsheet/plot-friendly twin of :func:`render_vad_gaps` /
    :func:`render_vad_gaps_json`, completing the human / ``--json`` / ``--csv``
    trio every VAD-analysis surface carries. The natural CSV unit is one row per
    gap: ``index,after_segment,after_segment_end_s,gap_s`` — the machine
    expansion of the human report's per-gap lines, the shape a plotter wants
    (gap-vs-position) and a spreadsheet wants (one pause per line). The
    aggregate stats (min/mean/max/total) are derivable from the ``gap_s`` column
    so they are NOT duplicated into a wide row, matching
    :func:`render_vad_diff_csv`'s reasoning. A result with fewer than 2 segments
    yields the header alone (a valid empty-bodied table). ``result`` of ``None``
    (segmenter unavailable) yields a single ``# silero VAD unavailable: ...``
    comment line, matching the sibling CSV renderers. Pure: built with the
    stdlib :mod:`csv` writer, trailing terminator stripped.
    """
    if result is None:
        return (
            "# silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        )
    d = vad_silence_gaps(result)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["index", "after_segment", "after_segment_end_s", "gap_s"])
    for i, (gap, after, end) in enumerate(
        zip(d["gaps"], d["after_segment"], d["after_segment_end_s"]), start=1
    ):
        writer.writerow([i, after, end, gap])
    return buf.getvalue().rstrip("\r\n")


# Default percentiles for the gap-percentile surface: the median (the typical
# pause), p90 (a high-but-not-extreme pause), and p99 (near the longest). A
# percentile is unmoved by a single outlier the way min/mean/max are not, so it
# is the stable signal for choosing the end-of-turn hangover.
DEFAULT_GAP_PERCENTILES = (50.0, 90.0, 99.0)

# Default percentiles for the band-rate distribution that `gv vad-gap-peak`
# reports alongside the verdict (iter-358). Where DEFAULT_GAP_PERCENTILES
# summarises the inter-segment SILENCE distribution, these summarise the
# observed cost-band RATE distribution — the exact sample `--min-rate-pct`
# (iter-357) interpolates against. The quartile-leaning set (p50/p75/p90/p99)
# lets an operator see where a chosen percentile floor will land before
# committing to it: `--min-rate-pct 75` keeps the bands at or above the p75
# value printed here.
DEFAULT_BAND_RATE_PCTS = (50.0, 75.0, 90.0, 99.0)


def _percentile_of_sorted(sorted_values, p):
    """Linear-interpolated percentile ``p`` of an already-SORTED, non-empty list.

    The shared primitive behind :func:`vad_gap_percentiles` (iter-338) and
    iter-357's ``gv vad-gap-peak --min-rate-pct`` (the percentile-derived rate
    floor). Uses the numpy default ``"linear"`` / R-7 convention: for ``n``
    samples and percentile ``p`` the fractional rank is ``(p / 100) * (n - 1)``
    and the value interpolates between the samples at the floor and ceil of that
    rank. A single sample yields that sample for every percentile. The caller is
    responsible for sorting and for the empty-list / range / NaN guards (kept out
    of this hot primitive so it stays a pure two-line interpolation). Returns the
    raw (unrounded) value — the caller rounds to its own convention.
    """
    n = len(sorted_values)
    rank = (p / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def _band_rate_distribution(bands, percentiles=DEFAULT_BAND_RATE_PCTS):
    """Summarise the observed non-empty cost-band RATE distribution (iter-358).

    The companion of :func:`vad_gap_peak`'s ``--min-rate-pct`` (iter-357): that
    knob derives a rate FLOOR from the Pth percentile of the observed non-empty
    band rates, but until now an operator had no way to SEE that distribution —
    they had to guess where a chosen ``P`` would land. This helper exposes it.

    ``bands`` is the list of cost bands from :func:`vad_gap_cost` (each carrying
    a ``rate_per_100ms``). Empty valleys (rate 0) are not cost peaks and would
    skew the distribution toward zero, so they are excluded — exactly the sample
    ``--min-rate-pct`` ranks over. Returns a dict with ``count`` (number of
    non-empty bands), ``min`` / ``mean`` / ``max`` of their rates, and a
    ``percentiles`` list of ``{p, rate}`` objects (one per requested percentile,
    in the order given), each ``rate`` the linear / R-7 interpolated value over
    the sorted non-empty rates — the same convention :func:`vad_gap_percentiles`
    and the ``--min-rate-pct`` floor use, so the printed p75 equals the floor
    ``--min-rate-pct 75`` would apply. All rates round to 3 places, matching the
    band rates they are drawn from. When there are no non-empty bands (every band
    is a valley) ``count`` is ``0``, the aggregates are ``None`` and
    ``percentiles`` is empty — the same "no distribution to summarise" spelling
    the sibling gap surfaces use. Pure and side-effect-free for direct testing.
    """
    rates = sorted(b["rate_per_100ms"] for b in bands if b["rate_per_100ms"] > 0)
    if not rates:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "max": None,
            "percentiles": [],
        }
    return {
        "count": len(rates),
        "min": round(rates[0], 3),
        "mean": round(sum(rates) / len(rates), 3),
        "max": round(rates[-1], 3),
        "percentiles": [
            {"p": p, "rate": round(_percentile_of_sorted(rates, p), 3)}
            for p in percentiles
        ],
    }


def vad_gap_percentiles(result, *, percentiles=DEFAULT_GAP_PERCENTILES):
    """Compute robust order statistics of the inter-segment silence gaps (iter-338).

    The min/mean/max aggregates :func:`vad_silence_gaps` reports are each
    fragile to a single outlier pause: one unusually long between-paragraph
    silence drags the max (and the mean) up, hiding where the bulk of the pauses
    actually sit. A percentile is robust — p50 is the typical pause, p90 a
    high-but-not-extreme pause, p99 near the longest — so the percentile spread
    is the stable signal an operator reads to choose the end-of-turn hangover
    (``--min-silence-ms`` / the live ``chat.vad.silence_duration``): set the
    hangover comfortably below p50 to never merge a typical turn, and read p90 /
    p99 to know how much of the long tail you are willing to wait through. This
    is the order-statistic complement of :func:`vad_silence_gaps`.

    Pure: anchors to :func:`vad_silence_gaps` for the gap list + aggregates (so
    the totals always agree with ``gv vad-gaps``) and adds a ``percentiles``
    list of ``{p, value_s}`` objects, one per requested percentile in the order
    given. Each value is computed by linear interpolation between the two
    closest ranks of the SORTED gaps (numpy's default ``"linear"`` method): for
    ``n`` gaps and percentile ``p``, the fractional rank is
    ``(p / 100) * (n - 1)`` and the value interpolates between the gaps at the
    floor and ceil of that rank. Values round to 3 places, matching the sibling
    gap surfaces. A single gap yields that gap for every percentile. A result
    with fewer than 2 segments has no gaps, so ``percentiles`` is empty (no
    distribution to summarise) and the aggregates are ``None`` — the same
    distinction the other gap surfaces make. Raises :class:`ValueError` if
    ``percentiles`` is empty or any entry is not in ``(0, 100]`` / is NaN.
    """
    pcts = list(percentiles)
    if not pcts:
        raise ValueError("percentiles must be a non-empty sequence")
    for p in pcts:
        if p != p:  # NaN is unordered.
            raise ValueError("percentile must be a number, got nan")
        if not (0.0 < p <= 100.0):
            raise ValueError(f"percentile must be in (0, 100], got {p}")
    d = vad_silence_gaps(result)
    gaps = sorted(d["gaps"])
    out = []
    if gaps:
        for p in pcts:
            # Linear interpolation between the two bracketing samples (the numpy
            # "linear" / R-7 convention), shared with the iter-357 rate-floor.
            out.append({"p": p, "value_s": round(_percentile_of_sorted(gaps, p), 3)})
    return {
        "num_segments": d["num_segments"],
        "num_gaps": d["num_gaps"],
        "min_gap_s": d["min_gap_s"],
        "max_gap_s": d["max_gap_s"],
        "mean_gap_s": d["mean_gap_s"],
        "total_silence_s": d["total_silence_s"],
        "percentiles": out,
    }


def _format_percentile_label(p):
    """Render a percentile number compactly: ``50`` not ``50.0``, ``99.5`` kept.

    The default percentiles (50/90/99) and most operator inputs are whole
    numbers, so a ``p50`` label reads better than ``p50.0``; a fractional
    percentile (``99.5``) keeps its decimals. Pure helper shared by the human
    and CSV renderers so the two agree on the label spelling.
    """
    return f"{p:g}"


def render_vad_gap_percentiles(result, *, percentiles=DEFAULT_GAP_PERCENTILES):
    """Render the silence-gap percentiles as plain-text report lines (iter-338).

    The human-readable face of :func:`vad_gap_percentiles`, the order-statistic
    twin of :func:`render_vad_gaps`. ``result`` of ``None`` (segmenter
    unavailable) yields the shared install hint. A result with fewer than 2
    segments has no gaps, so it prints the same short explanatory line
    :func:`render_vad_gaps` uses (no distribution to summarise). Otherwise it
    prints the aggregate header (min/mean/max, total silence) then one line per
    requested percentile (``pNN: value`` aligned), naming the actionable
    ``--min-silence-ms`` knob on the median line — set the hangover below the
    median to never merge a typical turn. Pure: returns a list of strings (no
    I/O, no ANSI).
    """
    if result is None:
        return [
            "silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        ]
    s = vad_gap_percentiles(result, percentiles=percentiles)
    lines = [
        f"silero VAD gap percentiles — {result.name}",
        f"  segments:     {s['num_segments']}",
        f"  gaps:         {s['num_gaps']} (pauses between consecutive speech regions)",
    ]
    if s["num_gaps"] == 0:
        lines.append("  (fewer than 2 segments — no inter-segment pause to measure)")
        return lines
    lines.append(f"  min gap:      {s['min_gap_s']:.3f}s")
    lines.append(f"  mean gap:     {s['mean_gap_s']:.3f}s")
    lines.append(f"  max gap:      {s['max_gap_s']:.3f}s")
    lines.append(f"  total silence:{s['total_silence_s']:8.3f}s")
    for entry in s["percentiles"]:
        label = f"p{_format_percentile_label(entry['p'])}"
        suffix = (
            "  (typical pause — keep --min-silence-ms below this to avoid "
            "merging turns)"
            if entry["p"] == 50.0
            else ""
        )
        lines.append(f"  {label:<5} {entry['value_s']:7.3f}s{suffix}")
    return lines


def render_vad_gap_percentiles_json(result, *, percentiles=DEFAULT_GAP_PERCENTILES):
    """Render the silence-gap percentiles as a JSON string (iter-338).

    Machine-readable twin of :func:`render_vad_gap_percentiles`, mirroring the
    degrade-to-``{"available": false}`` contract the other VAD JSON renderers
    use. Carries the aggregate stats plus a ``percentiles`` list of
    ``{p, value_s}`` objects (empty for a <2-segment result, the same JSON
    spelling of "no distribution" the other gap surfaces use). Pure: built from
    :func:`vad_gap_percentiles`, so it works on any ``SileroResult``-shaped
    object.
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
    s = vad_gap_percentiles(result, percentiles=percentiles)
    payload = {
        "available": True,
        "name": result.name,
        "num_segments": s["num_segments"],
        "num_gaps": s["num_gaps"],
        "min_gap_s": s["min_gap_s"],
        "max_gap_s": s["max_gap_s"],
        "mean_gap_s": s["mean_gap_s"],
        "total_silence_s": s["total_silence_s"],
        "percentiles": s["percentiles"],
    }
    return json.dumps(payload, indent=2)


def render_vad_gap_percentiles_csv(result, *, percentiles=DEFAULT_GAP_PERCENTILES):
    """Render the silence-gap percentiles as a per-percentile CSV table (iter-338).

    The spreadsheet/plot-friendly twin of :func:`render_vad_gap_percentiles` /
    :func:`render_vad_gap_percentiles_json`, completing the human / ``--json`` /
    ``--csv`` trio every VAD-analysis surface carries. The natural CSV unit is
    one row per requested percentile: ``percentile,value_s`` — the shape a
    plotter wants (an empirical CDF) and a spreadsheet wants (one percentile per
    line). The aggregate stats are derivable from the per-gap data so they are
    NOT duplicated into the table, matching :func:`render_vad_gaps_csv`'s
    reasoning. A result with fewer than 2 segments yields the header alone (a
    valid empty-bodied table). ``result`` of ``None`` (segmenter unavailable)
    yields a single ``# silero VAD unavailable: ...`` comment line. Pure: built
    with the stdlib :mod:`csv` writer, trailing terminator stripped.
    """
    if result is None:
        return (
            "# silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        )
    s = vad_gap_percentiles(result, percentiles=percentiles)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["percentile", "value_s"])
    for entry in s["percentiles"]:
        writer.writerow([_format_percentile_label(entry["p"]), entry["value_s"]])
    return buf.getvalue().rstrip("\r\n")


# Default candidate hangover cuts (in milliseconds) for the gap merge-CDF
# surface: 200 / 400 / 800 / 1600 ms. They bracket the live default
# `--min-silence-ms` (800 ms, pipecat stop_secs=0.8) one octave either side, so
# the table shows how the merge fraction climbs as the hangover lengthens around
# the operating point. Mirrors the `gv vad-sweep --min-silences 200,400,800,1600`
# example axis.
DEFAULT_GAP_CDF_CUTS_MS = (200.0, 400.0, 800.0, 1600.0)


def vad_gap_cdf(result, *, cuts_ms=DEFAULT_GAP_CDF_CUTS_MS):
    """Evaluate the empirical CDF of the silence gaps at candidate hangover cuts (iter-346).

    The order-statistic INVERSE of :func:`vad_gap_percentiles`. Percentiles
    answer "what pause length sits at the p90?" (fraction → value); this answers
    the operationally-direct opposite — "if I set the end-of-turn hangover
    (``--min-silence-ms`` / the live ``chat.vad.silence_duration``) to candidate
    cut ``c``, what FRACTION of the inter-segment pauses are shorter than ``c``
    and would therefore be MERGED (swallowed as within-turn silence rather than
    ending a turn)?" (value → fraction). That is the empirical CDF of the gap
    distribution sampled at the operator's candidate cuts, turning the percentile
    table into a direct "this hangover merges X% of your pauses" answer.

    The merge rule follows the segmenter's own convention: a region ends once the
    trailing silence REACHES the hangover, so a pause ``>= c`` ends the turn
    (kept as a boundary) while a pause STRICTLY ``< c`` is too short to trigger an
    end and merges the two regions into one. ``merged`` counts gaps ``< cut_s``;
    ``merge_fraction`` is ``merged / num_gaps``. Keeping the hangover below the
    min gap merges nothing (``merge_fraction == 0``); raising it past the max gap
    merges everything (``merge_fraction == 1``).

    Pure: anchors to :func:`vad_silence_gaps` for the gap list + aggregates (so
    the totals always agree with ``gv vad-gaps``) and adds a ``cuts`` list of
    ``{cut_ms, cut_s, merged, kept, merge_fraction, keep_fraction}`` objects, one
    per requested cut in the order given. ``cut_s`` is ``cut_ms / 1000`` rounded
    to 3 places; the fractions round to 3 places, matching the sibling gap
    surfaces. Cuts are NOT sorted or de-duplicated (the operator may want a
    specific column order). A result with fewer than 2 segments has no gaps, so
    ``cuts`` is empty (no distribution to sample) and the aggregates are ``None``
    — the same distinction the other gap surfaces make. Raises :class:`ValueError`
    if ``cuts_ms`` is empty or any entry is negative / NaN.
    """
    cuts = list(cuts_ms)
    if not cuts:
        raise ValueError("cuts_ms must be a non-empty sequence")
    for c in cuts:
        if c != c:  # NaN is unordered.
            raise ValueError("cut must be a number, got nan")
        if c < 0:
            raise ValueError(f"cut must be >= 0, got {c}")
    d = vad_silence_gaps(result)
    gaps = d["gaps"]
    n_gaps = d["num_gaps"]
    out = []
    if gaps:
        for c in cuts:
            cut_s = c / 1000.0
            # A pause shorter than the hangover is too short to end a turn, so it
            # merges; a pause that reaches the hangover ends the turn (kept).
            merged = sum(1 for g in gaps if g < cut_s)
            kept = n_gaps - merged
            out.append(
                {
                    "cut_ms": c,
                    "cut_s": round(cut_s, 3),
                    "merged": merged,
                    "kept": kept,
                    "merge_fraction": round(merged / n_gaps, 3),
                    "keep_fraction": round(kept / n_gaps, 3),
                }
            )
    return {
        "num_segments": d["num_segments"],
        "num_gaps": n_gaps,
        "min_gap_s": d["min_gap_s"],
        "max_gap_s": d["max_gap_s"],
        "mean_gap_s": d["mean_gap_s"],
        "total_silence_s": d["total_silence_s"],
        "cuts": out,
    }


def _format_cut_label(cut_ms):
    """Render a candidate-cut millisecond value compactly: ``800`` not ``800.0``.

    The default cuts (200/400/800/1600) and most operator inputs are whole
    millisecond counts, so an ``800`` label reads better than ``800.0``; a
    fractional cut (``750.5``) keeps its decimals. Pure helper shared by the
    human and CSV renderers so the two agree on the label spelling — the cut twin
    of :func:`_format_percentile_label`.
    """
    return f"{cut_ms:g}"


def render_vad_gap_cdf(result, *, cuts_ms=DEFAULT_GAP_CDF_CUTS_MS):
    """Render the silence-gap merge-CDF as plain-text report lines (iter-346).

    The human-readable face of :func:`vad_gap_cdf`, the inverse-CDF twin of
    :func:`render_vad_gap_percentiles`. ``result`` of ``None`` (segmenter
    unavailable) yields the shared install hint. A result with fewer than 2
    segments has no gaps, so it prints the same short explanatory line
    :func:`render_vad_gaps` uses (no distribution to sample). Otherwise it prints
    the aggregate header (min/mean/max, total silence — naming the actionable
    ``--min-silence-ms`` knob on the min-gap line) then a small table: one row per
    candidate cut giving the cut in ms and seconds, the ``merged/num_gaps`` count,
    and the merge percentage. The merge fraction is the fraction of pauses shorter
    than the cut (which that hangover would swallow as within-turn silence). Pure:
    returns a list of strings (no I/O, no ANSI).
    """
    if result is None:
        return [
            "silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        ]
    c = vad_gap_cdf(result, cuts_ms=cuts_ms)
    lines = [
        f"silero VAD gap merge-CDF — {result.name}",
        f"  segments:     {c['num_segments']}",
        f"  gaps:         {c['num_gaps']} (pauses between consecutive speech regions)",
    ]
    if c["num_gaps"] == 0:
        lines.append("  (fewer than 2 segments — no inter-segment pause to measure)")
        return lines
    lines.append(
        f"  min gap:      {c['min_gap_s']:.3f}s "
        "(shortest real pause — keep --min-silence-ms below this to avoid "
        "merging turns)"
    )
    lines.append(f"  mean gap:     {c['mean_gap_s']:.3f}s")
    lines.append(f"  max gap:      {c['max_gap_s']:.3f}s")
    lines.append(f"  total silence:{c['total_silence_s']:8.3f}s")
    lines.append("  cut (ms)  cut (s)    merged   merge%")
    for entry in c["cuts"]:
        label = _format_cut_label(entry["cut_ms"])
        merged_col = f"{entry['merged']}/{c['num_gaps']}"
        pct = entry["merge_fraction"] * 100.0
        lines.append(
            f"  {label:>8}  {entry['cut_s']:7.3f}  {merged_col:>8}  {pct:>6.1f}%"
        )
    return lines


def render_vad_gap_cdf_json(result, *, cuts_ms=DEFAULT_GAP_CDF_CUTS_MS):
    """Render the silence-gap merge-CDF as a JSON string (iter-346).

    Machine-readable twin of :func:`render_vad_gap_cdf`, mirroring the
    degrade-to-``{"available": false}`` contract the other VAD JSON renderers
    use. Carries the aggregate stats plus a ``cuts`` list of
    ``{cut_ms, cut_s, merged, kept, merge_fraction, keep_fraction}`` objects
    (empty for a <2-segment result, the same JSON spelling of "no distribution"
    the other gap surfaces use). Pure: built from :func:`vad_gap_cdf`, so it works
    on any ``SileroResult``-shaped object.
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
    c = vad_gap_cdf(result, cuts_ms=cuts_ms)
    payload = {
        "available": True,
        "name": result.name,
        "num_segments": c["num_segments"],
        "num_gaps": c["num_gaps"],
        "min_gap_s": c["min_gap_s"],
        "max_gap_s": c["max_gap_s"],
        "mean_gap_s": c["mean_gap_s"],
        "total_silence_s": c["total_silence_s"],
        "cuts": c["cuts"],
    }
    return json.dumps(payload, indent=2)


def render_vad_gap_cdf_csv(result, *, cuts_ms=DEFAULT_GAP_CDF_CUTS_MS):
    """Render the silence-gap merge-CDF as a per-cut CSV table (iter-346).

    The spreadsheet/plot-friendly twin of :func:`render_vad_gap_cdf` /
    :func:`render_vad_gap_cdf_json`, completing the human / ``--json`` / ``--csv``
    trio every VAD-analysis surface carries. The natural CSV unit is one row per
    candidate cut: ``cut_ms,cut_s,merged,merge_fraction`` — the shape a plotter
    wants (the empirical CDF curve, merge-fraction vs cut) and a spreadsheet wants
    (one cut per line). The aggregate stats and the derivable ``kept`` /
    ``keep_fraction`` columns are NOT duplicated into the table, matching
    :func:`render_vad_gap_percentiles_csv`'s reasoning. A result with fewer than 2
    segments yields the header alone (a valid empty-bodied table). ``result`` of
    ``None`` (segmenter unavailable) yields a single ``# silero VAD unavailable:
    ...`` comment line. Pure: built with the stdlib :mod:`csv` writer, trailing
    terminator stripped.
    """
    if result is None:
        return (
            "# silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        )
    c = vad_gap_cdf(result, cuts_ms=cuts_ms)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["cut_ms", "cut_s", "merged", "merge_fraction"])
    for entry in c["cuts"]:
        writer.writerow(
            [
                _format_cut_label(entry["cut_ms"]),
                entry["cut_s"],
                entry["merged"],
                entry["merge_fraction"],
            ]
        )
    return buf.getvalue().rstrip("\r\n")


def vad_gap_cost(result, *, cuts_ms=DEFAULT_GAP_CDF_CUTS_MS):
    """Compute the silence-gap MERGE COST CURVE — the derivative of the CDF (iter-349).

    iter-346's :func:`vad_gap_cdf` reports the empirical CDF: at candidate
    hangover cut ``c``, what FRACTION of the inter-segment pauses are shorter than
    ``c`` (and so would MERGE). That is the *cumulative* picture. This surface is
    its DERIVATIVE: between two consecutive cuts, how many ADDITIONAL pauses get
    swallowed, and at what rate per +100 ms of hangover? Where the CDF answers
    "how much have I merged by cut ``c``?", the cost curve answers "what does the
    NEXT +100 ms of hangover cost me in merged turns?" — the marginal view an
    operator tuning ``--min-silence-ms`` (the live ``chat.vad.silence_duration``)
    actually reasons in.

    The key reading: a band with a HIGH rate sits inside a cluster of pauses
    (every extra millisecond of hangover swallows more real turn boundaries — an
    expensive place to raise the hangover); a band with rate ZERO is an EMPTY band
    in the distribution — a valley where raising the hangover costs nothing. That
    zero-cost valley is exactly where :func:`vad_gap_recommend` points, so the
    cost curve and the recommendation agree by construction: the recommended
    hangover sits in the flattest (cheapest) band.

    The cost is taken between consecutive cuts, so the cuts are SORTED and
    DE-DUPLICATED here (unlike :func:`vad_gap_cdf`, which preserves the operator's
    column order — a derivative needs a monotone axis). ``N`` distinct cuts yield
    ``N - 1`` bands. Each band records its ``from_ms`` / ``to_ms`` endpoints (and
    ``from_s`` / ``to_s``), ``width_ms``, the ``merged_added`` count (pauses with
    ``from_s <= gap < to_s``), the ``merged_cumulative`` count at the band top
    (which equals exactly what :func:`vad_gap_cdf` reports at ``to_ms``, so the
    two surfaces agree), and ``rate_per_100ms`` (``merged_added / width_ms * 100``
    — additional pauses merged per +100 ms of hangover). The merge rule follows
    the segmenter's own convention: a pause STRICTLY ``< cut`` is too short to end
    a turn and merges, ``>= cut`` is kept.

    Pure: anchors to :func:`vad_silence_gaps` for the gap list + aggregates (so
    the totals always agree with ``gv vad-gaps``) and adds a ``bands`` list. A
    result with fewer than 2 segments has no gaps, so ``bands`` is empty (no
    distribution to differentiate) and the aggregates are ``None`` — the same
    distinction the other gap surfaces make. A single distinct cut forms no band,
    so ``bands`` is empty (a degenerate axis, the same "nothing to show" spelling
    a <2-segment result uses). ``from_s`` / ``to_s`` round to 3 places and
    ``rate_per_100ms`` to 3, matching the sibling gap surfaces. Raises
    :class:`ValueError` if ``cuts_ms`` is empty or any entry is negative / NaN.
    """
    cuts = list(cuts_ms)
    if not cuts:
        raise ValueError("cuts_ms must be a non-empty sequence")
    for c in cuts:
        if c != c:  # NaN is unordered.
            raise ValueError("cut must be a number, got nan")
        if c < 0:
            raise ValueError(f"cut must be >= 0, got {c}")
    d = vad_silence_gaps(result)
    gaps = d["gaps"]
    n_gaps = d["num_gaps"]
    bands = []
    if gaps:
        # A derivative needs a monotone, de-duplicated axis (vad_gap_cdf keeps the
        # operator's column order; the cost between consecutive cuts does not).
        ordered = sorted(set(cuts))
        for prev, cur in zip(ordered, ordered[1:]):
            lo_s = prev / 1000.0
            hi_s = cur / 1000.0
            # Cumulative-at-top equals vad_gap_cdf's merged count at this cut, so
            # the two surfaces agree; the band's marginal cost is the difference.
            merged_lo = sum(1 for g in gaps if g < lo_s)
            merged_hi = sum(1 for g in gaps if g < hi_s)
            added = merged_hi - merged_lo
            width_ms = cur - prev
            bands.append(
                {
                    "from_ms": prev,
                    "to_ms": cur,
                    "from_s": round(lo_s, 3),
                    "to_s": round(hi_s, 3),
                    "width_ms": width_ms,
                    "merged_added": added,
                    "merged_cumulative": merged_hi,
                    "rate_per_100ms": round(added / width_ms * 100.0, 3),
                }
            )
    return {
        "num_segments": d["num_segments"],
        "num_gaps": n_gaps,
        "min_gap_s": d["min_gap_s"],
        "max_gap_s": d["max_gap_s"],
        "mean_gap_s": d["mean_gap_s"],
        "total_silence_s": d["total_silence_s"],
        "bands": bands,
    }


def render_vad_gap_cost(result, *, cuts_ms=DEFAULT_GAP_CDF_CUTS_MS):
    """Render the silence-gap merge cost curve as plain-text report lines (iter-349).

    The human-readable face of :func:`vad_gap_cost`, the derivative twin of
    :func:`render_vad_gap_cdf`. ``result`` of ``None`` (segmenter unavailable)
    yields the shared install hint. A result with fewer than 2 segments has no
    gaps, so it prints the same short explanatory line :func:`render_vad_gaps`
    uses (no distribution to differentiate). Otherwise it prints the aggregate
    header (min/mean/max, total silence — naming the actionable
    ``--min-silence-ms`` knob on the min-gap line) then a small table: one row per
    band between consecutive cuts giving the band's ms range, its width, the
    additional pauses it merges, and the marginal rate per +100 ms. A zero-rate
    band is a valley (raising the hangover there costs nothing — where
    ``gv vad-gap-recommend`` points). Pure: returns a list of strings (no I/O, no
    ANSI).
    """
    if result is None:
        return [
            "silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        ]
    c = vad_gap_cost(result, cuts_ms=cuts_ms)
    lines = [
        f"silero VAD gap merge cost curve — {result.name}",
        f"  segments:     {c['num_segments']}",
        f"  gaps:         {c['num_gaps']} (pauses between consecutive speech regions)",
    ]
    if c["num_gaps"] == 0:
        lines.append("  (fewer than 2 segments — no inter-segment pause to measure)")
        return lines
    lines.append(
        f"  min gap:      {c['min_gap_s']:.3f}s "
        "(shortest real pause — keep --min-silence-ms below this to avoid "
        "merging turns)"
    )
    lines.append(f"  mean gap:     {c['mean_gap_s']:.3f}s")
    lines.append(f"  max gap:      {c['max_gap_s']:.3f}s")
    lines.append(f"  total silence:{c['total_silence_s']:8.3f}s")
    if not c["bands"]:
        lines.append(
            "  (need at least 2 distinct cuts to form a cost band — none to show)"
        )
        return lines
    lines.append("  band (ms)        width   merged   per +100ms")
    for b in c["bands"]:
        lo = _format_cut_label(b["from_ms"])
        hi = _format_cut_label(b["to_ms"])
        band_col = f"{lo}-{hi}"
        width_col = f"{_format_cut_label(b['width_ms'])}ms"
        merged_col = f"+{b['merged_added']}"
        lines.append(
            f"  {band_col:<14}  {width_col:>6}  {merged_col:>6}  "
            f"{b['rate_per_100ms']:>9.3f}"
        )
    return lines


def render_vad_gap_cost_json(result, *, cuts_ms=DEFAULT_GAP_CDF_CUTS_MS):
    """Render the silence-gap merge cost curve as a JSON string (iter-349).

    Machine-readable twin of :func:`render_vad_gap_cost`, mirroring the
    degrade-to-``{"available": false}`` contract the other VAD JSON renderers
    use. Carries the aggregate stats plus a ``bands`` list of
    ``{from_ms, to_ms, from_s, to_s, width_ms, merged_added, merged_cumulative,
    rate_per_100ms}`` objects (empty for a <2-segment result or a single distinct
    cut, the same JSON spelling of "no distribution" the other gap surfaces use).
    Pure: built from :func:`vad_gap_cost`, so it works on any
    ``SileroResult``-shaped object.
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
    c = vad_gap_cost(result, cuts_ms=cuts_ms)
    payload = {
        "available": True,
        "name": result.name,
        "num_segments": c["num_segments"],
        "num_gaps": c["num_gaps"],
        "min_gap_s": c["min_gap_s"],
        "max_gap_s": c["max_gap_s"],
        "mean_gap_s": c["mean_gap_s"],
        "total_silence_s": c["total_silence_s"],
        "bands": c["bands"],
    }
    return json.dumps(payload, indent=2)


def render_vad_gap_cost_csv(result, *, cuts_ms=DEFAULT_GAP_CDF_CUTS_MS):
    """Render the silence-gap merge cost curve as a per-band CSV table (iter-349).

    The spreadsheet/plot-friendly twin of :func:`render_vad_gap_cost` /
    :func:`render_vad_gap_cost_json`, completing the human / ``--json`` / ``--csv``
    trio every VAD-analysis surface carries. The natural CSV unit is one row per
    band: ``from_ms,to_ms,width_ms,merged_added,merged_cumulative,rate_per_100ms``
    — the shape a plotter wants (the derivative curve, marginal merge-rate vs
    hangover) and a spreadsheet wants (one band per line). The aggregate stats and
    derivable ``from_s`` / ``to_s`` columns are NOT duplicated into the table,
    matching :func:`render_vad_gap_cdf_csv`'s reasoning. A result with fewer than 2
    segments — or a single distinct cut — yields the header alone (a valid
    empty-bodied table). ``result`` of ``None`` (segmenter unavailable) yields a
    single ``# silero VAD unavailable: ...`` comment line. Pure: built with the
    stdlib :mod:`csv` writer, trailing terminator stripped.
    """
    if result is None:
        return (
            "# silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        )
    c = vad_gap_cost(result, cuts_ms=cuts_ms)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "from_ms",
            "to_ms",
            "width_ms",
            "merged_added",
            "merged_cumulative",
            "rate_per_100ms",
        ]
    )
    for b in c["bands"]:
        writer.writerow(
            [
                _format_cut_label(b["from_ms"]),
                _format_cut_label(b["to_ms"]),
                _format_cut_label(b["width_ms"]),
                b["merged_added"],
                b["merged_cumulative"],
                b["rate_per_100ms"],
            ]
        )
    return buf.getvalue().rstrip("\r\n")


def vad_gap_peak(
    result,
    *,
    cuts_ms=DEFAULT_GAP_CDF_CUTS_MS,
    top_n=1,
    min_rate=0.0,
    min_rate_pct=None,
    rate_pcts=DEFAULT_BAND_RATE_PCTS,
):
    """Name the COSTLIEST band(s) of the merge cost curve — the densest pause cluster (iter-350; ``top_n`` iter-354; ``min_rate`` iter-355; ``min_rate_pct`` iter-357; ``band_rate_dist`` iter-358).

    The verdict companion of iter-349's :func:`vad_gap_cost`. The cost curve gives
    the marginal rate per +100 ms of hangover for every band between consecutive
    cuts; this surface reads that curve and names a single answer: which band is
    the STEEPEST — the place where raising ``--min-silence-ms`` (the live
    ``chat.vad.silence_duration``) swallows the most real turn boundaries per
    millisecond. That steepest band is the densest cluster of pauses, the steepest
    part of the CDF — the most EXPENSIVE place to push the hangover through.

    It is the mirror image of :func:`vad_gap_recommend`, which points at the
    cheapest (zero-rate) valley between clusters: where ``vad-gap-recommend`` says
    "set the hangover HERE — it costs nothing", ``vad-gap-peak`` says "do NOT raise
    the hangover THROUGH here — this is where it costs the most". An operator
    reading both gets the valley to aim for and the cluster to avoid cutting into.

    The peak is the band with the highest ``rate_per_100ms`` over the sorted,
    de-duplicated cuts (the same monotone axis :func:`vad_gap_cost` builds). The
    first band wins on a tie (earliest-tie, matching the rest of the family). A
    band's rate is zero exactly when it is an empty valley, so a peak rate of zero
    means EVERY band is a valley — there is no pause cluster anywhere in the
    scanned range; ``peak_found`` is then ``False`` and the peak fields are
    ``None`` (the same "no structure to name" spelling :func:`vad_gap_recommend`
    uses when ``split_found`` is ``False``).

    ``top_n`` (iter-354) names the N STEEPEST bands instead of just the single
    peak — for an operator who wants to see the whole cluster ranking, not only
    the worst offender. The result always carries a ``peaks`` list of up to
    ``top_n`` band entries sorted by descending rate (earliest band first on a
    tie), each carrying the same ``from_ms`` / ``to_ms`` / ``from_s`` / ``to_s`` /
    ``width_ms`` / ``merged_added`` / ``rate_per_100ms`` fields plus a 1-based
    ``rank`` (iter-356) that makes the descending-rate order explicit in the entry
    itself (``rank`` 1 is the steepest band; it always equals the entry's position
    in the list + 1). Only NON-empty
    bands (rate > 0) are listed — an empty valley is not a "cost peak" — so the
    list may hold FEWER than ``top_n`` entries (or be empty when every band is a
    valley). The legacy scalar ``peak_*`` fields are kept verbatim and always echo
    the FIRST entry of ``peaks`` (the single steepest band), so existing callers
    and the default ``top_n=1`` behaviour are byte-for-byte unchanged. ``top_n``
    must be ``>= 1`` (raises :class:`ValueError` otherwise).

    ``min_rate`` (iter-355) is a rate FLOOR: bands whose ``rate_per_100ms`` is
    strictly below it are dropped from the ranking before ``top_n`` truncation,
    so the named list holds only the bands worth worrying about — the ones at
    least ``min_rate`` pauses-merged per +100 ms of hangover. It pairs naturally
    with ``--top-n``: "give me the up-to-N steepest bands, but only those
    costing at least X". The default ``0.0`` keeps every non-empty band (a band's
    rate is ``> 0`` exactly when it merges at least one pause), so it is
    byte-for-byte the iter-354 behaviour. When the floor filters out EVERY band
    (none meets it) there is no peak to name: ``peak_found`` is ``False``, the
    scalar ``peak_*`` fields are ``None`` and ``peaks`` is empty — the same "no
    structure to name" spelling the all-valley case uses. The applied floor is
    echoed as ``min_rate``. Must be ``>= 0`` (raises :class:`ValueError`
    otherwise); a negative floor is nonsensical (every rate is non-negative).

    ``min_rate_pct`` (iter-357) is an ADAPTIVE alternative to the absolute
    ``min_rate``: instead of "drop bands below 0.08 per +100 ms", it says "drop
    bands below the Pth percentile of the OBSERVED non-empty band rates". An
    absolute floor must be re-tuned per recording — a quiet conversation and a
    rapid-fire one have wholly different rate scales — whereas a percentile floor
    adapts to the recording's own cost distribution: ``--min-rate-pct 75`` always
    names the top quartile of cost peaks, whatever the absolute numbers. The
    percentile is computed (linear / R-7 interpolation, the same convention as
    :func:`vad_gap_percentiles`) over the rates of the non-empty bands only —
    empty valleys are not cost peaks and would skew the distribution toward zero.
    The interpolated rate becomes the effective floor and is surfaced as
    ``effective_min_rate`` (also set when only the absolute ``min_rate`` is in
    play, where it simply equals ``min_rate``). ``min_rate_pct`` must be in
    ``(0, 100]`` (raises :class:`ValueError` otherwise) and is mutually exclusive
    with a positive ``min_rate`` (raises :class:`ValueError` if both are given) —
    an absolute floor and a percentile-derived floor are two ways to set the same
    knob. The default ``None`` leaves the absolute-``min_rate`` behaviour
    byte-for-byte unchanged. When there are no non-empty bands the percentile has
    nothing to rank over, so the effective floor is ``0.0`` and the result is the
    usual all-valley no-peak verdict.

    ``band_rate_dist`` (iter-358) is the companion VIEW of the percentile floor:
    the result always carries a ``band_rate_dist`` summary of the OBSERVED
    non-empty band rates — ``count`` / ``min`` / ``mean`` / ``max`` plus a
    ``percentiles`` list over ``rate_pcts`` (default p50/p75/p90/p99). It is the
    exact sample ``--min-rate-pct`` interpolates against, so an operator can SEE
    where a chosen ``P`` floor will land before committing to it: the ``rate`` at
    ``p75`` here equals the floor ``--min-rate-pct 75`` would apply. It is purely
    descriptive — it never changes which band is named — and is computed over all
    non-empty bands regardless of the ``min_rate`` / ``min_rate_pct`` floor (so it
    shows the full distribution, including the bands a floor drops). When every
    band is an empty valley it reports ``count`` ``0`` with ``None`` aggregates.

    Pure: anchors to :func:`vad_gap_cost` for the bands + aggregates (so the totals
    and per-band numbers always agree with ``gv vad-gaps`` / ``gv vad-gap-cost``)
    and adds the peak fields. A result with fewer than 2 segments has no gaps, so
    there are no bands and nothing to name (``peak_found`` ``False``, aggregates
    ``None``); a single distinct cut forms no band, the same degenerate-axis case
    :func:`vad_gap_cost` handles. ``peak_from_s`` / ``peak_to_s`` round to 3 places
    and ``peak_rate_per_100ms`` is carried through from the band (already rounded
    to 3). Raises :class:`ValueError` if ``cuts_ms`` is empty or any entry is
    negative / NaN (delegated to :func:`vad_gap_cost`).
    """
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")
    if min_rate < 0:
        raise ValueError(f"min_rate must be >= 0, got {min_rate}")
    if min_rate_pct is not None:
        if min_rate_pct != min_rate_pct:  # NaN is unordered.
            raise ValueError("min_rate_pct must be a number, got nan")
        if not (0.0 < min_rate_pct <= 100.0):
            raise ValueError(
                f"min_rate_pct must be in (0, 100], got {min_rate_pct}"
            )
        if min_rate > 0:
            # An absolute floor and a percentile-derived floor set the SAME knob
            # two different ways — accepting both would be ambiguous.
            raise ValueError(
                "min_rate and min_rate_pct are mutually exclusive (both set "
                f"the rate floor): got min_rate={min_rate}, "
                f"min_rate_pct={min_rate_pct}"
            )
    c = vad_gap_cost(result, cuts_ms=cuts_ms)
    bands = c["bands"]
    peak = {
        "num_segments": c["num_segments"],
        "num_gaps": c["num_gaps"],
        "min_gap_s": c["min_gap_s"],
        "max_gap_s": c["max_gap_s"],
        "mean_gap_s": c["mean_gap_s"],
        "total_silence_s": c["total_silence_s"],
        "num_bands": len(bands),
        "top_n": top_n,
        "min_rate": min_rate,
        "min_rate_pct": min_rate_pct,
        "effective_min_rate": min_rate,
        # iter-358: the observed non-empty band-rate distribution — the sample
        # --min-rate-pct interpolates against. Empty (count 0) when there are no
        # bands at all; recomputed below over the actual bands when they exist.
        "band_rate_dist": _band_rate_distribution([], percentiles=rate_pcts),
        "peak_found": False,
        "peak_from_ms": None,
        "peak_to_ms": None,
        "peak_from_s": None,
        "peak_to_s": None,
        "peak_width_ms": None,
        "peak_merged_added": None,
        "peak_rate_per_100ms": None,
        "peaks": [],
    }
    if not bands:
        return peak
    # iter-358: describe the observed non-empty band-rate distribution over the
    # actual bands (the empty placeholder above is replaced). Computed before any
    # floor is applied so it shows the FULL distribution — including bands a
    # min_rate / min_rate_pct floor will drop — letting the operator see where a
    # chosen percentile floor lands.
    peak["band_rate_dist"] = _band_rate_distribution(bands, percentiles=rate_pcts)
    # iter-357: when a percentile floor is requested, the effective absolute floor
    # is the Pth percentile of the OBSERVED non-empty band rates — so the cutoff
    # adapts to this recording's own cost scale instead of needing a per-recording
    # absolute number. Empty valleys (rate 0) are not cost peaks and would skew the
    # distribution toward zero, so they are excluded from the percentile sample. No
    # non-empty band means nothing to rank over: the floor stays 0.0 and the
    # all-valley no-peak verdict falls out below. Rounded to 3 places to match the
    # band rates it is compared against (which vad_gap_cost already rounds to 3).
    effective_min_rate = min_rate
    if min_rate_pct is not None:
        nonempty_rates = sorted(
            b["rate_per_100ms"] for b in bands if b["rate_per_100ms"] > 0
        )
        if nonempty_rates:
            effective_min_rate = round(
                _percentile_of_sorted(nonempty_rates, min_rate_pct), 3
            )
        peak["effective_min_rate"] = effective_min_rate
    # Rank the cost bands by descending marginal merge rate; only non-empty bands
    # (rate > 0) are cost peaks, and iter-355's min_rate floor (or iter-357's
    # percentile-derived effective_min_rate) further drops any band cheaper than
    # the threshold (effective_min_rate=0.0 keeps every non-empty band, so the
    # default is unchanged — a non-empty band always has rate > 0 >= 0).
    # Python's sort is stable, so sorting the bands — already in ascending-cut
    # order — by -rate keeps the EARLIER band first on a tie (earliest-tie,
    # matching vad_gap_recommend's widest-jump rule).
    ranked = sorted(
        (
            b
            for b in bands
            if b["rate_per_100ms"] > 0 and b["rate_per_100ms"] >= effective_min_rate
        ),
        key=lambda b: -b["rate_per_100ms"],
    )
    if not ranked:
        # Every band is an empty valley, or the min_rate floor filtered them all
        # out — no cost peak worth naming in the scanned range (mirrors
        # split_found=False).
        return peak
    peaks = [
        {
            # iter-356: 1-based rank making the descending-rate order explicit in
            # the entry itself. The list is already sorted, so rank == position+1;
            # carrying it as a field lets the machine faces (JSON/CSV) name the
            # ordering without relying on array/row position alone.
            "rank": i,
            "from_ms": b["from_ms"],
            "to_ms": b["to_ms"],
            "from_s": b["from_s"],
            "to_s": b["to_s"],
            "width_ms": b["width_ms"],
            "merged_added": b["merged_added"],
            "rate_per_100ms": b["rate_per_100ms"],
        }
        for i, b in enumerate(ranked[:top_n], start=1)
    ]
    peak["peaks"] = peaks
    # The legacy scalar peak_* fields always echo the single steepest band — the
    # first peaks entry — so default top_n=1 callers see no change.
    best = peaks[0]
    peak["peak_found"] = True
    peak["peak_from_ms"] = best["from_ms"]
    peak["peak_to_ms"] = best["to_ms"]
    peak["peak_from_s"] = best["from_s"]
    peak["peak_to_s"] = best["to_s"]
    peak["peak_width_ms"] = best["width_ms"]
    peak["peak_merged_added"] = best["merged_added"]
    peak["peak_rate_per_100ms"] = best["rate_per_100ms"]
    return peak


def render_vad_gap_peak(
    result,
    *,
    cuts_ms=DEFAULT_GAP_CDF_CUTS_MS,
    top_n=1,
    min_rate=0.0,
    min_rate_pct=None,
    show_rate_dist=False,
    rate_pcts=DEFAULT_BAND_RATE_PCTS,
):
    """Render the costliest-band verdict as plain-text report lines (iter-350; ``top_n`` iter-354; ``min_rate`` iter-355; ``min_rate_pct`` iter-357; ``show_rate_dist`` iter-358; floor-mark iter-360; unlisted-floor hint iter-362).

    The human-readable face of :func:`vad_gap_peak`, the verdict twin of
    :func:`render_vad_gap_cost`. ``result`` of ``None`` (segmenter unavailable)
    yields the shared install hint. A result with fewer than 2 segments has no
    gaps, so it prints the same short explanatory line :func:`render_vad_gaps`
    uses (nothing to name). Otherwise it prints the aggregate header (min/mean/max,
    total silence) then the verdict: the costliest band's ms range, the additional
    pauses it merges, and its marginal rate per +100 ms — the densest pause
    cluster, the place NOT to push the hangover through. When every band is an
    empty valley (no cluster in the scanned range), it says so.

    With ``top_n > 1`` (iter-354) it ranks the N steepest bands, printing one
    numbered ``#k costliest band`` line per peak (descending rate, earliest band
    first on a tie). ``top_n == 1`` is byte-for-byte the original single-peak
    block. With ``min_rate > 0`` (iter-355) it prints a ``rate floor`` note and
    drops any band cheaper than the floor; if the floor filters out every band it
    says so. ``min_rate == 0.0`` is unchanged. With ``min_rate_pct`` (iter-357)
    the floor note instead names the requested percentile and the effective rate
    it resolved to over the observed band rates. With ``show_rate_dist`` (iter-358)
    it appends a block summarising the observed non-empty band-rate distribution
    (count / min / mean / max + the ``rate_pcts`` percentiles) — the sample
    ``--min-rate-pct`` reads against, so the operator sees where a chosen
    percentile floor would land. The default ``False`` leaves the verdict face
    byte-for-byte unchanged. When ``show_rate_dist`` AND ``min_rate_pct`` are both
    set and the floor's percentile is one of the displayed ``rate_pcts`` quantiles,
    that pNN row is marked with ``<-- --min-rate-pct floor`` (iter-360) so the
    operator sees the cutoff in context; an unlisted floor percentile leaves every
    row unmarked but appends an explicit hint (iter-362) naming the unlisted floor
    percentile and telling the operator to add it to ``--rate-pcts`` to mark its
    row — so the missing marker never reads as "no floor". Pure: returns a list of
    strings (no I/O, no ANSI).
    """
    if result is None:
        return [
            "silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        ]
    p = vad_gap_peak(
        result,
        cuts_ms=cuts_ms,
        top_n=top_n,
        min_rate=min_rate,
        min_rate_pct=min_rate_pct,
        rate_pcts=rate_pcts,
    )
    eff = p["effective_min_rate"]
    floor_active = min_rate_pct is not None or min_rate > 0
    lines = [
        f"silero VAD gap cost peak — {result.name}",
        f"  segments:     {p['num_segments']}",
        f"  gaps:         {p['num_gaps']} (pauses between consecutive speech regions)",
    ]
    if p["num_gaps"] == 0:
        lines.append("  (fewer than 2 segments — no inter-segment pause to measure)")
        return lines
    lines.append(f"  min gap:      {p['min_gap_s']:.3f}s")
    lines.append(f"  mean gap:     {p['mean_gap_s']:.3f}s")
    lines.append(f"  max gap:      {p['max_gap_s']:.3f}s")
    lines.append(f"  total silence:{p['total_silence_s']:8.3f}s")
    if min_rate_pct is not None:
        # iter-357: a percentile-derived floor — name the requested percentile and
        # the absolute rate it resolved to over this recording's observed band
        # rates, so the operator sees both the adaptive knob and its concrete cut.
        lines.append(
            f"  rate floor:   p{_format_percentile_label(min_rate_pct)} of observed "
            f"band rates = {eff:.3f} per +100ms (only bands at or above this are "
            "named) (iter-357)"
        )
    elif min_rate > 0:
        # iter-355: note the active rate floor so the operator knows the ranking
        # below already excludes bands cheaper than this. Only emitted when a
        # floor is set, so the default (min_rate=0.0) face is unchanged.
        lines.append(
            f"  rate floor:   {min_rate:.3f} per +100ms (only bands at or above "
            "this are named) (iter-355)"
        )
    if show_rate_dist:
        # iter-358: show the observed non-empty band-rate distribution — the
        # sample --min-rate-pct interpolates against — so the operator sees where
        # a chosen percentile floor lands before committing. Only emitted when
        # requested, so the default verdict face is unchanged.
        dist = p["band_rate_dist"]
        if dist["count"] == 0:
            lines.append(
                "  band-rate dist: (no non-empty bands — no rate distribution to "
                "summarise) (iter-358)"
            )
        else:
            lines.append(
                f"  band-rate dist: {dist['count']} non-empty bands, "
                f"min {dist['min']:.3f} / mean {dist['mean']:.3f} / "
                f"max {dist['max']:.3f} per +100ms (the --min-rate-pct sample) "
                "(iter-358)"
            )
            floor_marked = False
            for entry in dist["percentiles"]:
                line = (
                    f"    p{_format_percentile_label(entry['p'])}: "
                    f"{entry['rate']:.3f} per +100ms"
                )
                # iter-360: when an adaptive percentile floor is active AND its
                # percentile is one of the displayed quantiles, mark that line so
                # the operator sees the cutoff IN CONTEXT — exactly which pNN row
                # the --min-rate-pct floor lands on, and thus which bands above it
                # survive. Only the matching row is marked; an unlisted floor
                # percentile (e.g. --min-rate-pct 80 with default p50/75/90/99)
                # leaves every row unmarked, so add it to --rate-pcts to see it.
                if min_rate_pct is not None and entry["p"] == min_rate_pct:
                    line += "  <-- --min-rate-pct floor (iter-360)"
                    floor_marked = True
                lines.append(line)
            # iter-362: when an adaptive percentile floor IS active but its
            # percentile is NOT one of the displayed --rate-pcts quantiles, no
            # row got the iter-360 marker — and silence reads as "no floor". Emit
            # an explicit hint naming the unlisted floor percentile so the
            # operator knows to add it to --rate-pcts to see the cutoff row in
            # context. The human-face complement of iter-361's
            # floor_percentile_listed JSON flag (which is False in exactly this
            # case). Suppressed when the floor row WAS marked, when no percentile
            # floor is active, or for an absolute --min-rate floor (not a
            # percentile, so no row could match).
            if min_rate_pct is not None and not floor_marked:
                lines.append(
                    f"    (the --min-rate-pct p"
                    f"{_format_percentile_label(min_rate_pct)} cutoff is not among "
                    "the shown --rate-pcts quantiles — add it to --rate-pcts to "
                    "mark its row) (iter-362)"
                )
    if not p["peak_found"]:
        if p["num_bands"] == 0:
            lines.append(
                "  (need at least 2 distinct cuts to form a cost band — none to show)"
            )
        elif floor_active:
            lines.append(
                "  (no cost peak meets the rate floor — every band is either an "
                "empty valley or cheaper than the floor; lower --min-rate/"
                "--min-rate-pct to name the shallower clusters)"
            )
        else:
            lines.append(
                "  (no cost peak — every band is an empty valley; no pause cluster "
                "in the scanned cut range, so raising the hangover costs nothing "
                "anywhere here)"
            )
        return lines
    if top_n == 1:
        # The original single-peak block — kept verbatim so top_n=1 is unchanged.
        lo = _format_cut_label(p["peak_from_ms"])
        hi = _format_cut_label(p["peak_to_ms"])
        width = _format_cut_label(p["peak_width_ms"])
        lines.append(
            f"  costliest band: {lo}-{hi}ms (width {width}ms) — the densest pause "
            "cluster / steepest part of the CDF"
        )
        lines.append(
            f"  cost:         merges +{p['peak_merged_added']} pauses, "
            f"{p['peak_rate_per_100ms']:.3f} per +100ms (most expensive place to "
            "raise --min-silence-ms — don't cut through here) (iter-350)"
        )
        return lines
    # top_n > 1: rank the N steepest bands, one numbered line per peak. Fewer than
    # top_n lines appear when the scanned range holds fewer non-empty bands.
    lines.append(
        f"  top {top_n} costliest bands (steepest first — the densest pause "
        "clusters / steepest parts of the CDF; don't raise --min-silence-ms "
        "through these) (iter-354):"
    )
    for i, pk in enumerate(p["peaks"], start=1):
        lo = _format_cut_label(pk["from_ms"])
        hi = _format_cut_label(pk["to_ms"])
        width = _format_cut_label(pk["width_ms"])
        lines.append(
            f"    #{i}: {lo}-{hi}ms (width {width}ms) — merges "
            f"+{pk['merged_added']} pauses, {pk['rate_per_100ms']:.3f} per +100ms"
        )
    return lines


def render_vad_gap_peak_json(
    result,
    *,
    cuts_ms=DEFAULT_GAP_CDF_CUTS_MS,
    top_n=1,
    min_rate=0.0,
    min_rate_pct=None,
    rate_pcts=DEFAULT_BAND_RATE_PCTS,
):
    """Render the costliest-band verdict as a JSON string (iter-350; ``top_n`` iter-354; ``min_rate`` iter-355; ``min_rate_pct`` iter-357; ``band_rate_dist`` iter-358; ``floor_percentile_listed`` iter-361).

    Machine-readable twin of :func:`render_vad_gap_peak`, mirroring the
    degrade-to-``{"available": false}`` contract the other VAD JSON renderers use.
    Carries the aggregate stats plus the peak fields (``peak_found`` /
    ``peak_from_ms`` / ``peak_to_ms`` / ``peak_from_s`` / ``peak_to_s`` /
    ``peak_width_ms`` / ``peak_merged_added`` / ``peak_rate_per_100ms`` /
    ``num_bands``). The peak fields are ``null`` / ``false`` for a <2-segment
    result, a single distinct cut, or an all-valley range (nothing to name) — the
    same JSON spelling of "no structure" the other gap surfaces use.

    iter-354 adds ``top_n`` (the requested count) and a ``peaks`` list of up to
    ``top_n`` ranked band objects (descending rate, earliest first on a tie), each
    carrying a 1-based ``rank`` (iter-356) that names its position in the ordering.
    The scalar ``peak_*`` fields are unchanged and echo ``peaks[0]``, so a default
    ``top_n=1`` payload is a strict superset of the iter-350 shape. iter-355 adds
    ``min_rate`` (the applied rate floor; bands cheaper than it are excluded from
    ``peaks``), defaulting to ``0.0`` (every non-empty band kept — the iter-354
    payload). iter-357 adds ``min_rate_pct`` (the requested percentile floor, or
    ``null`` when an absolute / no floor is used) and ``effective_min_rate`` (the
    absolute rate the floor resolved to — equal to ``min_rate`` when no percentile
    is given), so a machine consumer can read the concrete cutoff regardless of
    which knob set it. iter-358 adds ``band_rate_dist`` — a summary of the
    observed non-empty band-rate distribution (``count`` / ``min`` / ``mean`` /
    ``max`` + a ``percentiles`` list over ``rate_pcts``), the exact sample
    ``--min-rate-pct`` interpolates against, always present so a machine consumer
    can read where a chosen percentile floor would land. iter-361 adds
    ``floor_percentile_listed`` — the machine-readable twin of the iter-360 human
    floor-marker: ``True`` when an adaptive ``min_rate_pct`` floor is active AND
    its percentile is one of the displayed ``band_rate_dist`` quantiles (i.e. the
    human face would mark a ``pNN`` row with ``<-- --min-rate-pct floor``),
    ``False`` otherwise (no percentile floor, an absolute ``--min-rate`` floor, the
    floor percentile not among ``rate_pcts``, or an all-valley result with no
    quantile rows). A consumer can replicate the human marker turnkey without
    re-deriving it from ``min_rate_pct`` + ``rate_pcts``. Pure: built from
    :func:`vad_gap_peak`, so it works on any ``SileroResult``-shaped object.
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
    p = vad_gap_peak(
        result,
        cuts_ms=cuts_ms,
        top_n=top_n,
        min_rate=min_rate,
        min_rate_pct=min_rate_pct,
        rate_pcts=rate_pcts,
    )
    # iter-361: the machine-readable twin of the iter-360 human floor-marker.
    # True iff an adaptive percentile floor is active AND its percentile is one
    # of the band_rate_dist quantiles the human face would mark — derived purely
    # from the same two inputs the human renderer uses, so the faces never
    # disagree (the all-valley case has an empty percentiles list, so no floor
    # percentile can match — False, matching the human "no rows to mark" face).
    floor_percentile_listed = min_rate_pct is not None and any(
        entry["p"] == min_rate_pct for entry in p["band_rate_dist"]["percentiles"]
    )
    payload = {
        "available": True,
        "name": result.name,
        "num_segments": p["num_segments"],
        "num_gaps": p["num_gaps"],
        "min_gap_s": p["min_gap_s"],
        "max_gap_s": p["max_gap_s"],
        "mean_gap_s": p["mean_gap_s"],
        "total_silence_s": p["total_silence_s"],
        "num_bands": p["num_bands"],
        "top_n": p["top_n"],
        "min_rate": p["min_rate"],
        "min_rate_pct": p["min_rate_pct"],
        "effective_min_rate": p["effective_min_rate"],
        "band_rate_dist": p["band_rate_dist"],
        "floor_percentile_listed": floor_percentile_listed,
        "peak_found": p["peak_found"],
        "peak_from_ms": p["peak_from_ms"],
        "peak_to_ms": p["peak_to_ms"],
        "peak_from_s": p["peak_from_s"],
        "peak_to_s": p["peak_to_s"],
        "peak_width_ms": p["peak_width_ms"],
        "peak_merged_added": p["peak_merged_added"],
        "peak_rate_per_100ms": p["peak_rate_per_100ms"],
        "peaks": p["peaks"],
    }
    return json.dumps(payload, indent=2)


def render_vad_gap_peak_csv(
    result,
    *,
    cuts_ms=DEFAULT_GAP_CDF_CUTS_MS,
    top_n=1,
    min_rate=0.0,
    min_rate_pct=None,
    rate_pcts=DEFAULT_BAND_RATE_PCTS,
):
    """Render the costliest-band verdict as a CSV table (iter-350; ``top_n`` iter-354; ``min_rate`` iter-355; ``min_rate_pct`` iter-357; floor-listed comment iter-363).

    The spreadsheet-friendly twin of :func:`render_vad_gap_peak` /
    :func:`render_vad_gap_peak_json`, completing the human / ``--json`` / ``--csv``
    trio every VAD-analysis surface carries. Like the other verdict surfaces, the
    peak is a SINGLE result, so the natural CSV is one summary row:
    ``peak_found,peak_from_ms,peak_to_ms,peak_width_ms,peak_merged_added,
    peak_rate_per_100ms``. The derivable aggregates are NOT duplicated into the
    row, matching :func:`render_vad_gap_recommend_csv`'s reasoning. A result with
    fewer than 2 segments — or a single distinct cut — yields the header alone (a
    valid empty-bodied table — nothing to name). An all-valley range emits the row
    with ``peak_found`` ``False`` and blanks for the peak measures. ``result`` of
    ``None`` (segmenter unavailable) yields a single ``# silero VAD unavailable:
    ...`` comment line.

    iter-354's ``top_n`` emits one row PER ranked peak (descending rate, earliest
    first on a tie). iter-356 prepends an explicit ``rank`` column (1-based) so the
    descending order is named in the data, not implied by row position alone — the
    seven columns are ``rank,peak_found,peak_from_ms,peak_to_ms,peak_width_ms,
    peak_merged_added,peak_rate_per_100ms``. ``rank`` is ``1`` for the single-peak
    (``top_n == 1``) table and counts up across the ranked rows; it is blank on the
    no-peak ``peak_found=False`` row. iter-355's ``min_rate`` floor simply changes
    which bands are ranked (the columns are unchanged); when the floor leaves no
    peak but bands exist, the single ``rank``-blank ``peak_found=False`` blank row
    is emitted (same as the all-valley case). iter-357's ``min_rate_pct``
    likewise only changes which bands rank — the column schema is unchanged, so
    the percentile and absolute floors yield CSVs with an identical shape.

    iter-363 completes the floor-info signal across all three faces. The human
    face marks the floor's ``band_rate_dist`` row (iter-360) or hints when it is
    unlisted (iter-362); the JSON face carries a top-level
    ``floor_percentile_listed`` boolean (iter-361). The CSV's seven-column
    verdict-row schema has no place for a distribution or a per-row flag, so this
    surface trails the same fact as a single ``# floor_percentile_listed: ...``
    comment line — self-describing metadata a spreadsheet/plotter skips by default
    (pandas ``read_csv(comment="#")``), matching the ``#``-comment precedent
    :func:`render_calibration_csv` / :func:`render_vad_sweep_csv` use. It is
    emitted ONLY when an adaptive ``min_rate_pct`` floor is active (the case the
    flag describes); no comment trails an absolute ``--min-rate`` floor or the
    default no-floor table, so those CSVs are byte-for-byte unchanged. The boolean
    equals iter-361's JSON ``floor_percentile_listed`` exactly (derived from the
    same ``min_rate_pct`` + ``rate_pcts`` inputs), so the three faces never
    disagree. Pure: built with the stdlib :mod:`csv` writer, trailing terminator
    stripped.
    """
    if result is None:
        return (
            "# silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        )
    p = vad_gap_peak(
        result,
        cuts_ms=cuts_ms,
        top_n=top_n,
        min_rate=min_rate,
        min_rate_pct=min_rate_pct,
        rate_pcts=rate_pcts,
    )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "rank",
            "peak_found",
            "peak_from_ms",
            "peak_to_ms",
            "peak_width_ms",
            "peak_merged_added",
            "peak_rate_per_100ms",
        ]
    )
    if p["peaks"]:
        # One row per ranked peak; the explicit rank column (iter-356) names the
        # descending-rate order. peak_found is True for every named peak (only
        # non-empty bands reach the list).
        for pk in p["peaks"]:
            writer.writerow(
                [
                    pk["rank"],
                    True,
                    _format_cut_label(pk["from_ms"]),
                    _format_cut_label(pk["to_ms"]),
                    _format_cut_label(pk["width_ms"]),
                    pk["merged_added"],
                    pk["rate_per_100ms"],
                ]
            )
    elif p["num_bands"] > 0:
        # Bands exist but none is a cost peak (all-valley range): the single
        # peak_found=False row with blank measures (rank blank too — no peak).
        writer.writerow(["", p["peak_found"], "", "", "", "", ""])
    body = buf.getvalue().rstrip("\r\n")
    # iter-363: when an adaptive percentile floor is active, trail the
    # floor_percentile_listed fact (iter-361's JSON flag) as a #-comment line so
    # the CSV consumer learns — without re-deriving it — whether the floor's
    # percentile is one of the band_rate_dist quantiles (i.e. whether the human
    # face would mark a pNN row). Derived from the same min_rate_pct + rate_pcts
    # inputs as the JSON flag, so the faces agree exactly. No comment for an
    # absolute --min-rate floor or the default no-floor table — those CSVs are
    # unchanged.
    if min_rate_pct is not None:
        floor_percentile_listed = any(
            entry["p"] == min_rate_pct for entry in p["band_rate_dist"]["percentiles"]
        )
        body += f"\n# floor_percentile_listed: {floor_percentile_listed}"
    return body


# How far across the valley (or, in the no-valley fallback, across the gap up to
# the shortest pause) the recommended hangover sits, per --bias (iter-351). The
# valley is the EMPTY band between the short within-turn cluster and the long
# between-turn cluster, so ANY interior point splits the pauses the same way —
# the fraction only shifts the named number, never the merge accounting. "short"
# hugs the top of the short cluster (a smaller, eager hangover that ends turns
# fast — but risks clipping a slow talker); "long" hugs the bottom of the long
# cluster (a larger, patient hangover that tolerates mid-turn pauses — but adds
# end-of-turn latency); "balanced" is the midpoint, the iter-347 default. 0.5 for
# balanced reproduces the original ``(best_lo + best_hi) / 2`` midpoint EXACTLY.
GAP_RECOMMEND_BIAS_FRACTIONS = {"short": 0.25, "balanced": 0.5, "long": 0.75}
DEFAULT_GAP_RECOMMEND_BIAS = "balanced"


def vad_gap_recommend(result, *, bias=DEFAULT_GAP_RECOMMEND_BIAS):
    """Recommend an end-of-turn hangover by finding the valley in the gap distribution (iter-347).

    The natural CONSUMER of the gap-analysis family. ``gv vad-gaps`` /
    ``gv vad-gap-hist`` / ``gv vad-gap-cdf`` / ``gv vad-gap-percentiles`` all
    SHOW the operator the inter-segment pause distribution and leave the
    "so what number do I set ``--min-silence-ms`` to?" judgement to them. This
    surface answers that question directly: it reads the gap distribution, finds
    the valley between the short within-turn pauses (which should MERGE) and the
    long between-turn pauses (which should END a turn), and names a single
    recommended hangover sitting in that valley — the histogram's valley, turned
    into a number.

    The split is found by the largest-gap (1-D Jenks) rule on a single feature:
    sort the gaps, then take the WIDEST jump between two consecutive sorted gaps.
    That widest jump is the empty band separating the short cluster from the long
    cluster, and the recommended hangover sits inside it — below it sit the
    within-turn pauses (which a hangover there would merge), at or above it sit
    the between-turn pauses (which it would keep as boundaries). The merge
    accounting follows the segmenter's own convention (a pause STRICTLY ``< cut``
    merges; ``>= cut`` is kept), so ``below`` / ``at_or_above`` are exactly what
    :func:`vad_gap_cdf` would report at the recommended cut.

    ``bias`` (iter-351) chooses WHERE in that empty valley the number sits:
    ``"balanced"`` (default) is the midpoint, the iter-347 behaviour exactly;
    ``"short"`` biases a quarter of the way up from the short cluster's top (a
    smaller, eager hangover — ends turns faster, but risks clipping a slow
    talker); ``"long"`` biases three quarters of the way up, hugging the long
    cluster's bottom (a larger, patient hangover — tolerates mid-turn pauses, but
    adds end-of-turn latency). Because every interior point of the EMPTY valley
    splits the pauses identically, ``below`` / ``at_or_above`` are INVARIANT
    across biases — only ``recommended_ms`` / ``recommended_s`` shift. The chosen
    bias is echoed back in the ``bias`` field. Raises :class:`ValueError` for an
    unknown bias.

    Pure: anchors to :func:`vad_silence_gaps` for the gap list + aggregates (so
    the totals always agree with ``gv vad-gaps``) and adds the recommendation
    fields. A result with fewer than 2 segments has no gaps, so there is nothing
    to recommend: ``recommended_ms`` / ``recommended_s`` are ``None`` and the
    aggregates are ``None`` — the same distinction the other gap surfaces make.
    A single pause, or several pauses all the same length, has no valley
    (``split_found`` is ``False``); there is no short/long cluster split to make,
    so it recommends below the shortest pause (``min_gap * frac``, ``frac < 1``)
    — every real pause is then kept as a turn boundary, the conservative default,
    and ``bias`` only nudges HOW far below. ``cut_s`` / the gap endpoints round to
    3 places and ``recommended_ms`` to 1 place, matching the sibling gap surfaces.
    """
    try:
        frac = GAP_RECOMMEND_BIAS_FRACTIONS[bias]
    except KeyError:
        raise ValueError(
            f"unknown bias {bias!r}: expected one of "
            f"{sorted(GAP_RECOMMEND_BIAS_FRACTIONS)}"
        ) from None
    d = vad_silence_gaps(result)
    gaps = sorted(d["gaps"])
    n = d["num_gaps"]
    rec = {
        "num_segments": d["num_segments"],
        "num_gaps": n,
        "min_gap_s": d["min_gap_s"],
        "max_gap_s": d["max_gap_s"],
        "mean_gap_s": d["mean_gap_s"],
        "total_silence_s": d["total_silence_s"],
        "bias": bias,
        "recommended_s": None,
        "recommended_ms": None,
        "split_found": False,
        "below": 0,
        "at_or_above": 0,
        "gap_below_s": None,
        "gap_above_s": None,
        "valley_width_s": None,
    }
    if not gaps:
        return rec
    # The widest jump between consecutive SORTED gaps is the empty band that
    # separates the short within-turn cluster from the long between-turn cluster
    # (1-D Jenks / largest-gap split). Duplicates give a zero-width jump so they
    # never win; the first widest jump wins on a tie (earliest-tie, matching the
    # rest of the family).
    best_width = 0.0
    best_lo = best_hi = None
    for i in range(1, n):
        width = gaps[i] - gaps[i - 1]
        if width > best_width:
            best_width = width
            best_lo = gaps[i - 1]
            best_hi = gaps[i]
    if best_width > 0:
        rec["split_found"] = True
        rec["gap_below_s"] = round(best_lo, 3)
        rec["gap_above_s"] = round(best_hi, 3)
        rec["valley_width_s"] = round(best_width, 3)
        # frac of the way across the empty valley. balanced (0.5) is the original
        # midpoint exactly; short/long slide toward the short/long cluster.
        cut_s = best_lo + frac * (best_hi - best_lo)
    else:
        # No valley: one pause, or every pause the same length. There is no
        # short/long split to make, so recommend below the single cluster (a
        # fraction of the shortest pause) — every real pause is kept as a
        # boundary. balanced (0.5) is the original min_gap / 2.
        cut_s = gaps[0] * frac
    cut_s = round(cut_s, 3)
    # Count against the rounded recommendation so below / at_or_above are exactly
    # what setting --min-silence-ms to the reported number would produce.
    below = sum(1 for g in gaps if g < cut_s)
    rec["recommended_s"] = cut_s
    rec["recommended_ms"] = round(cut_s * 1000.0, 1)
    rec["below"] = below
    rec["at_or_above"] = n - below
    return rec


def render_vad_gap_recommend(result, *, bias=DEFAULT_GAP_RECOMMEND_BIAS):
    """Render the recommended-hangover verdict as plain-text report lines (iter-347).

    The human-readable face of :func:`vad_gap_recommend`. ``result`` of ``None``
    (segmenter unavailable) yields the shared install hint. A result with fewer
    than 2 segments has no gaps, so it prints the same short explanatory line
    :func:`render_vad_gaps` uses (nothing to recommend). Otherwise it prints the
    aggregate header (min/mean/max, total silence) then the verdict: the
    recommended ``--min-silence-ms`` number (annotated with the chosen ``bias``),
    the valley it sits in (or a note that no valley was found), and the effect —
    how many pauses that hangover would merge vs keep. ``bias`` (iter-351) chooses
    where in the valley the number sits — ``short``/``balanced``/``long`` — and is
    echoed on the recommended line. Pure: returns a list of strings (no I/O, no
    ANSI).
    """
    if result is None:
        return [
            "silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        ]
    r = vad_gap_recommend(result, bias=bias)
    lines = [
        f"silero VAD recommended hangover — {result.name}",
        f"  segments:     {r['num_segments']}",
        f"  gaps:         {r['num_gaps']} (pauses between consecutive speech regions)",
    ]
    if r["num_gaps"] == 0:
        lines.append("  (fewer than 2 segments — no inter-segment pause to measure)")
        return lines
    lines.append(f"  min gap:      {r['min_gap_s']:.3f}s")
    lines.append(f"  mean gap:     {r['mean_gap_s']:.3f}s")
    lines.append(f"  max gap:      {r['max_gap_s']:.3f}s")
    lines.append(f"  total silence:{r['total_silence_s']:8.3f}s")
    label = _format_cut_label(r["recommended_ms"])
    lines.append(
        f"  recommended --min-silence-ms: {label} ({r['recommended_s']:.3f}s) "
        f"[bias: {r['bias']}]"
    )
    if r["split_found"]:
        lines.append(
            f"  valley:       between {r['gap_below_s']:.3f}s (top of short "
            f"pauses) and {r['gap_above_s']:.3f}s (bottom of long pauses), "
            f"width {r['valley_width_s']:.3f}s"
        )
    else:
        lines.append(
            "  (no valley — pauses don't separate into short/long clusters; "
            "recommending just below the shortest pause so every pause is kept)"
        )
    lines.append(
        f"  effect:       merges {r['below']}/{r['num_gaps']} within-turn "
        f"pauses, keeps {r['at_or_above']}/{r['num_gaps']} as turn boundaries "
        "(iter-347)"
    )
    return lines


def render_vad_gap_recommend_json(result, *, bias=DEFAULT_GAP_RECOMMEND_BIAS):
    """Render the recommended-hangover verdict as a JSON string (iter-347).

    Machine-readable twin of :func:`render_vad_gap_recommend`, mirroring the
    degrade-to-``{"available": false}`` contract the other VAD JSON renderers
    use. Carries the aggregate stats plus the recommendation fields
    (``bias`` / ``recommended_ms`` / ``recommended_s`` / ``split_found`` /
    ``below`` / ``at_or_above`` / the valley endpoints). The recommendation fields
    are ``null`` / ``0`` / ``false`` for a <2-segment result (nothing to
    recommend), the same JSON spelling of "no distribution" the other gap surfaces
    use; ``bias`` (iter-351) is always present, echoing the chosen short/balanced/
    long bias. Pure: built from :func:`vad_gap_recommend`, so it works on any
    ``SileroResult``-shaped object.
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
    r = vad_gap_recommend(result, bias=bias)
    payload = {
        "available": True,
        "name": result.name,
        "num_segments": r["num_segments"],
        "num_gaps": r["num_gaps"],
        "min_gap_s": r["min_gap_s"],
        "max_gap_s": r["max_gap_s"],
        "mean_gap_s": r["mean_gap_s"],
        "total_silence_s": r["total_silence_s"],
        "bias": r["bias"],
        "recommended_ms": r["recommended_ms"],
        "recommended_s": r["recommended_s"],
        "split_found": r["split_found"],
        "below": r["below"],
        "at_or_above": r["at_or_above"],
        "gap_below_s": r["gap_below_s"],
        "gap_above_s": r["gap_above_s"],
        "valley_width_s": r["valley_width_s"],
    }
    return json.dumps(payload, indent=2)


def render_vad_gap_recommend_csv(result, *, bias=DEFAULT_GAP_RECOMMEND_BIAS):
    """Render the recommended-hangover verdict as a one-row CSV table (iter-347).

    The spreadsheet-friendly twin of :func:`render_vad_gap_recommend` /
    :func:`render_vad_gap_recommend_json`, completing the human / ``--json`` /
    ``--csv`` trio every VAD-analysis surface carries. Unlike the per-gap /
    per-cut surfaces, the verdict is a SINGLE recommendation, so the natural CSV
    is one summary row:
    ``bias,recommended_ms,recommended_s,split_found,below,at_or_above,num_gaps``.
    The derivable aggregates and valley endpoints are NOT duplicated into the row,
    matching :func:`render_vad_gap_cdf_csv`'s reasoning; the chosen ``bias``
    (iter-351) IS carried so a sweep across biases is self-describing per row. A
    result with fewer than 2 segments yields the header alone (a valid
    empty-bodied table — nothing to recommend). ``result`` of ``None`` (segmenter
    unavailable) yields a single ``# silero VAD unavailable: ...`` comment line.
    Pure: built with the stdlib :mod:`csv` writer, trailing terminator stripped.
    """
    if result is None:
        return (
            "# silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        )
    r = vad_gap_recommend(result, bias=bias)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "bias",
            "recommended_ms",
            "recommended_s",
            "split_found",
            "below",
            "at_or_above",
            "num_gaps",
        ]
    )
    if r["num_gaps"] > 0:
        writer.writerow(
            [
                r["bias"],
                _format_cut_label(r["recommended_ms"]),
                r["recommended_s"],
                r["split_found"],
                r["below"],
                r["at_or_above"],
                r["num_gaps"],
            ]
        )
    return buf.getvalue().rstrip("\r\n")


# Bias order for the bias-sweep surface (iter-352): short..balanced..long, the
# natural shortest-to-longest hangover reading. The recommended number is
# monotone non-decreasing across this order (short <= balanced <= long), so a
# reader scanning top-to-bottom sees the spread widen toward longer hangovers.
GAP_RECOMMEND_BIAS_ORDER = ("short", "balanced", "long")


def vad_gap_recommend_sweep(result):
    """Emit the short/balanced/long recommendations side by side (iter-352).

    The natural companion of iter-351's ``--bias`` knob. ``gv vad-gap-recommend
    --bias {short,balanced,long}`` names ONE defensible hangover; this surface
    names ALL THREE at once, so the operator sees the whole spread of defensible
    numbers without re-running the command per bias. It is the
    ``vad-gap-recommend`` analogue of how ``gv vad-gap-sweep`` shows a whole
    ``--min-silence-ms`` sweep instead of one ``gv vad-gaps`` snapshot.

    The valley is the EMPTY band the largest-gap split finds, so every interior
    point splits the pauses identically — ``split_found`` / ``below`` /
    ``at_or_above`` / the valley endpoints are INVARIANT across biases (only the
    named number shifts). This surface therefore reports those shared fields
    ONCE at the top level and a compact per-bias row carrying just the shifting
    ``recommended_ms`` / ``recommended_s``. The ``spread_ms`` / ``spread_s``
    fields name the short..long gap — how much the choice of bias moves the
    number, the same width iter-348's confidence surface grades the quality of.

    iter-353 folds that confidence grade into the shared block: ``grade`` /
    ``dominance`` / ``separation_ratio`` answer the question the bias spread
    raises ("which of the three should I pick?") with a prior one — "is the
    underlying valley even trustworthy?". The grade is a property of the VALLEY,
    not of where in it the number sits, so it too is invariant across biases and
    is reported ONCE alongside the other shared fields. The fields are anchored
    to :func:`vad_gap_confidence`, so they agree EXACTLY with
    ``gv vad-gap-confidence``.

    Pure: built from :func:`vad_gap_recommend` (one call per bias) and
    :func:`vad_gap_confidence` (once), so the aggregates, valley, merge
    accounting, and grade agree EXACTLY with ``gv vad-gap-recommend`` at each
    bias and ``gv vad-gap-confidence``. A result with fewer than 2 segments has
    no gaps (nothing to recommend): the per-bias ``recommended_ms`` /
    ``recommended_s`` are ``None``, ``spread`` is ``None``, and ``grade`` is
    ``None`` (the same "no distribution" spelling the sibling gap surfaces use).
    """
    rows = [vad_gap_recommend(result, bias=b) for b in GAP_RECOMMEND_BIAS_ORDER]
    base = rows[0]
    conf = vad_gap_confidence(result)
    out = {
        "num_segments": base["num_segments"],
        "num_gaps": base["num_gaps"],
        "min_gap_s": base["min_gap_s"],
        "max_gap_s": base["max_gap_s"],
        "mean_gap_s": base["mean_gap_s"],
        "total_silence_s": base["total_silence_s"],
        # Valley + merge accounting are invariant across biases; report once.
        "split_found": base["split_found"],
        "below": base["below"],
        "at_or_above": base["at_or_above"],
        "gap_below_s": base["gap_below_s"],
        "gap_above_s": base["gap_above_s"],
        "valley_width_s": base["valley_width_s"],
        # Confidence grade is a valley property, invariant across biases (iter-353).
        "grade": conf["grade"],
        "dominance": conf["dominance"],
        "separation_ratio": conf["separation_ratio"],
        "biases": [
            {
                "bias": r["bias"],
                "recommended_ms": r["recommended_ms"],
                "recommended_s": r["recommended_s"],
            }
            for r in rows
        ],
        "spread_ms": None,
        "spread_s": None,
    }
    if base["num_gaps"] > 0:
        lo = rows[0]["recommended_s"]
        hi = rows[-1]["recommended_s"]
        out["spread_s"] = round(hi - lo, 3)
        out["spread_ms"] = round((hi - lo) * 1000.0, 1)
    return out


def render_vad_gap_recommend_sweep(result):
    """Render the bias sweep as plain-text report lines (iter-352).

    The human-readable face of :func:`vad_gap_recommend_sweep`. ``result`` of
    ``None`` (segmenter unavailable) yields the shared install hint. A result
    with fewer than 2 segments prints the same short explanatory line the other
    gap surfaces use (nothing to recommend). Otherwise it prints the aggregate
    header, the shared valley + merge effect (invariant across biases, so stated
    once), then one line per bias naming its recommended ``--min-silence-ms``,
    and finally the short..long spread. Pure: returns a list of strings.
    """
    if result is None:
        return [
            "silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        ]
    s = vad_gap_recommend_sweep(result)
    lines = [
        f"silero VAD recommended-hangover bias sweep — {result.name}",
        f"  segments:     {s['num_segments']}",
        f"  gaps:         {s['num_gaps']} (pauses between consecutive speech regions)",
    ]
    if s["num_gaps"] == 0:
        lines.append("  (fewer than 2 segments — no inter-segment pause to measure)")
        return lines
    lines.append(f"  min gap:      {s['min_gap_s']:.3f}s")
    lines.append(f"  mean gap:     {s['mean_gap_s']:.3f}s")
    lines.append(f"  max gap:      {s['max_gap_s']:.3f}s")
    lines.append(f"  total silence:{s['total_silence_s']:8.3f}s")
    if s["split_found"]:
        lines.append(
            f"  valley:       between {s['gap_below_s']:.3f}s (top of short "
            f"pauses) and {s['gap_above_s']:.3f}s (bottom of long pauses), "
            f"width {s['valley_width_s']:.3f}s"
        )
    else:
        lines.append(
            "  (no valley — pauses don't separate into short/long clusters; "
            "recommending just below the shortest pause so every pause is kept)"
        )
    lines.append("  recommended --min-silence-ms by bias:")
    for row in s["biases"]:
        label = _format_cut_label(row["recommended_ms"])
        lines.append(
            f"    {row['bias']:<8} {label} ({row['recommended_s']:.3f}s)"
        )
    spread_label = _format_cut_label(s["spread_ms"])
    lines.append(
        f"  spread:       {spread_label}ms ({s['spread_s']:.3f}s) "
        "short→long (how much the bias choice moves the number)"
    )
    if s["grade"] == "none" or s["grade"] is None:
        lines.append(
            "  confidence:   none (no valley — the spread above is a "
            "conservative fallback, not a confident split)"
        )
    else:
        sep = (
            "n/a (only one jump / clean split)"
            if s["separation_ratio"] is None
            else f"{s['separation_ratio']:.3f}x the next-widest jump"
        )
        lines.append(
            f"  confidence:   {s['grade']} (valley is "
            f"{s['dominance'] * 100.0:.1f}% of the gap spread; {sep})"
        )
    lines.append(f"  suggestion:   {_gap_confidence_summary(s['grade'])} (iter-348)")
    lines.append(
        f"  effect:       merges {s['below']}/{s['num_gaps']} within-turn "
        f"pauses, keeps {s['at_or_above']}/{s['num_gaps']} as turn boundaries "
        "(invariant across biases, iter-352)"
    )
    return lines


def render_vad_gap_recommend_sweep_json(result):
    """Render the bias sweep as a JSON string (iter-352).

    Machine-readable twin of :func:`render_vad_gap_recommend_sweep`, mirroring
    the degrade-to-``{"available": false}`` contract the other VAD JSON renderers
    use. Carries the aggregate stats, the shared valley + merge accounting
    (invariant across biases), the per-bias ``biases`` list (each with
    ``bias`` / ``recommended_ms`` / ``recommended_s``), and the short..long
    ``spread_ms`` / ``spread_s``. The recommendation fields are ``null`` for a
    <2-segment result (nothing to recommend). Pure: built from
    :func:`vad_gap_recommend_sweep`.
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
    s = vad_gap_recommend_sweep(result)
    payload = {
        "available": True,
        "name": result.name,
        "num_segments": s["num_segments"],
        "num_gaps": s["num_gaps"],
        "min_gap_s": s["min_gap_s"],
        "max_gap_s": s["max_gap_s"],
        "mean_gap_s": s["mean_gap_s"],
        "total_silence_s": s["total_silence_s"],
        "split_found": s["split_found"],
        "below": s["below"],
        "at_or_above": s["at_or_above"],
        "gap_below_s": s["gap_below_s"],
        "gap_above_s": s["gap_above_s"],
        "valley_width_s": s["valley_width_s"],
        "grade": s["grade"],
        "dominance": s["dominance"],
        "separation_ratio": s["separation_ratio"],
        "biases": s["biases"],
        "spread_ms": s["spread_ms"],
        "spread_s": s["spread_s"],
    }
    return json.dumps(payload, indent=2)


def render_vad_gap_recommend_sweep_csv(result):
    """Render the bias sweep as a CSV table (iter-352).

    The spreadsheet-friendly twin of :func:`render_vad_gap_recommend_sweep` /
    :func:`render_vad_gap_recommend_sweep_json`, completing the human / ``--json``
    / ``--csv`` trio. Unlike the single-row ``gv vad-gap-recommend --csv``, the
    sweep is naturally one row PER bias:
    ``bias,recommended_ms,recommended_s,split_found,below,at_or_above,num_gaps``
    — the same columns as ``gv vad-gap-recommend --csv``, but three rows so a
    reader sees the whole spread at once (and the per-bias rows union cleanly
    with the single-bias surface's output). The shared valley endpoints and
    spread are NOT duplicated into every row, matching the single-bias surface's
    reasoning. The iter-353 confidence grade is likewise a shared scalar (one
    per valley, invariant across biases), so it is NOT folded into the per-bias
    rows either — duplicating it 3× would break the clean union with
    ``gv vad-gap-recommend --csv``; the grade lives on the human / ``--json``
    faces (and on ``gv vad-gap-confidence --csv``). A result with fewer than 2
    segments yields the header alone (a valid empty-bodied table). ``result`` of
    ``None`` yields a single ``# silero VAD unavailable: ...`` comment line.
    Pure: built with the stdlib :mod:`csv` writer, trailing terminator stripped.
    """
    if result is None:
        return (
            "# silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        )
    s = vad_gap_recommend_sweep(result)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "bias",
            "recommended_ms",
            "recommended_s",
            "split_found",
            "below",
            "at_or_above",
            "num_gaps",
        ]
    )
    if s["num_gaps"] > 0:
        for row in s["biases"]:
            writer.writerow(
                [
                    row["bias"],
                    _format_cut_label(row["recommended_ms"]),
                    row["recommended_s"],
                    s["split_found"],
                    s["below"],
                    s["at_or_above"],
                    s["num_gaps"],
                ]
            )
    return buf.getvalue().rstrip("\r\n")


# Dominance thresholds grading how clean the recommendation's valley is. The
# dominance is the fraction of the total gap SPREAD (max_gap - min_gap) taken up
# by the single widest jump (the valley). A clean bimodal distribution puts most
# of its spread into that one empty band; a smear of similar pauses spreads the
# range across many small jumps. >= 0.5 means the valley alone accounts for at
# least half the entire spread (strong two-cluster separation); >= 0.25 is a
# discernible valley (moderate); below that the clusters bleed together (weak).
GAP_CONFIDENCE_STRONG_DOMINANCE = 0.5
GAP_CONFIDENCE_MODERATE_DOMINANCE = 0.25


def vad_gap_confidence(result):
    """Grade how trustworthy the :func:`vad_gap_recommend` verdict is (iter-348).

    iter-347's ``gv vad-gap-recommend`` always names a number, but the number is
    only as good as the valley it sits in: a clean bimodal distribution (short
    within-turn pauses well separated from long between-turn pauses) gives a
    confident recommendation, whereas a smear of similar pauses gives a number
    that is barely better than a guess. This surface is the CONFIDENCE note
    iter-347's own next-item asked for — it reads the same gap distribution and
    grades how dominant the recommendation's valley is, so the operator knows
    whether to trust the number or fall back to manual tuning.

    The grade is driven by two derived measures of the widest jump (the valley
    the recommendation sits in), found the same way :func:`vad_gap_recommend`
    finds it (largest jump between consecutive SORTED gaps):

    - ``dominance`` — the valley width as a FRACTION of the total gap spread
      (``max_gap - min_gap``, i.e. the sum of every consecutive jump). A clean
      two-cluster distribution puts most of its spread into that one empty band
      (dominance near 1); a uniform smear spreads the range across many small
      jumps (dominance near 0). This drives the ``grade``:
      ``>= 0.5`` → ``"strong"``, ``>= 0.25`` → ``"moderate"``, else ``"weak"``.
    - ``separation_ratio`` — the widest jump divided by the SECOND-widest jump.
      A big ratio means the valley stands clearly above the next-biggest band
      (the recommendation is unambiguous); a ratio near 1 means a rival valley
      is almost as wide (the recommendation could plausibly have landed
      elsewhere). ``None`` when there is only one jump, or the runner-up is a
      zero-width jump (a perfectly clean separation — the valley is infinitely
      dominant over its rivals).

    Pure: anchors to :func:`vad_gap_recommend` (so the recommendation and valley
    fields agree EXACTLY with ``gv vad-gap-recommend``) and adds the confidence
    fields. A result with fewer than 2 segments has no gaps, so there is nothing
    to grade: ``grade`` is ``None`` and the measures are ``None``. A single
    pause, or several pauses all the same length, has no valley
    (``split_found`` is ``False``), so it cannot be graded either: ``grade`` is
    ``"none"`` and the measures are ``None`` (the recommendation falls back to
    just below the shortest pause — a conservative default, not a confident
    split). ``dominance`` / ``separation_ratio`` round to 3 places, matching the
    sibling gap surfaces.
    """
    r = vad_gap_recommend(result)
    conf = dict(r)
    conf["spread_s"] = None
    conf["runner_up_width_s"] = None
    conf["dominance"] = None
    conf["separation_ratio"] = None
    conf["grade"] = None
    n = r["num_gaps"]
    if n == 0:
        # Fewer than 2 segments — no gaps, nothing to grade. grade stays None.
        return conf
    if not r["split_found"]:
        # One pause, or every pause the same length: no short/long split exists,
        # so the recommendation is the conservative fallback, not a confident
        # valley. grade is the explicit "none" rather than a numeric grade.
        conf["grade"] = "none"
        return conf
    # Re-derive the sorted jumps the same way vad_gap_recommend does, so the
    # widest jump here is exactly the valley it recommended.
    gaps = sorted(vad_silence_gaps(result)["gaps"])
    jumps = [gaps[i] - gaps[i - 1] for i in range(1, n)]
    spread = sum(jumps)  # == max_gap - min_gap
    best = max(jumps)
    # The runner-up is the widest jump OTHER than the valley itself (drop one
    # instance of the max, then take the new max). With a single jump there is
    # no runner-up.
    rest = list(jumps)
    rest.remove(best)
    runner_up = max(rest) if rest else None
    conf["spread_s"] = round(spread, 3)
    conf["runner_up_width_s"] = None if runner_up is None else round(runner_up, 3)
    # spread > 0 here because split_found means best > 0 and best <= spread.
    dominance = best / spread
    conf["dominance"] = round(dominance, 3)
    if runner_up is not None and runner_up > 0:
        conf["separation_ratio"] = round(best / runner_up, 3)
    else:
        # Only one jump, or every rival jump is zero-width: the valley is
        # infinitely dominant over its rivals. Spelled as None ("no finite
        # rival"), matching how the rest of the family spells "not measurable".
        conf["separation_ratio"] = None
    if dominance >= GAP_CONFIDENCE_STRONG_DOMINANCE:
        conf["grade"] = "strong"
    elif dominance >= GAP_CONFIDENCE_MODERATE_DOMINANCE:
        conf["grade"] = "moderate"
    else:
        conf["grade"] = "weak"
    return conf


def _gap_confidence_summary(grade):
    """Map a confidence ``grade`` to a one-line operator suggestion (iter-348).

    Per-value mapping inside the helper (the session-summary diversity-check
    convention applied to this surface): each grade gets text saying what to do
    about it, with a defensive fallback for an unexpected value so a future grade
    never drops the signal silently.
    """
    if grade == "strong":
        return "trust the recommendation — the valley is well separated"
    if grade == "moderate":
        return (
            "the recommendation is usable but the valley is shallow — "
            "sanity-check it against gv vad-gap-hist"
        )
    if grade == "weak":
        return (
            "the pauses don't cluster cleanly — treat the recommendation as a "
            "starting point and tune --min-silence-ms by ear"
        )
    if grade == "none":
        return (
            "no valley to grade — the pauses are uniform, so the recommendation "
            "is a conservative fallback, not a confident split"
        )
    return "unrecognized confidence grade — inspect the distribution directly"


def render_vad_gap_confidence(result):
    """Render the recommendation-confidence grade as plain-text report lines (iter-348).

    The human-readable face of :func:`vad_gap_confidence`. ``result`` of ``None``
    (segmenter unavailable) yields the shared install hint. A result with fewer
    than 2 segments has no gaps, so it prints the same short explanatory line
    :func:`render_vad_gaps` uses (nothing to grade). Otherwise it prints the
    aggregate header (min/mean/max, total silence) then the verdict: the
    recommended ``--min-silence-ms`` number it is grading, the grade itself with
    its dominance / separation measures (or a no-valley note), and the one-line
    suggestion for that grade. Pure: returns a list of strings (no I/O, no ANSI).
    """
    if result is None:
        return [
            "silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        ]
    c = vad_gap_confidence(result)
    lines = [
        f"silero VAD recommendation confidence — {result.name}",
        f"  segments:     {c['num_segments']}",
        f"  gaps:         {c['num_gaps']} (pauses between consecutive speech regions)",
    ]
    if c["num_gaps"] == 0:
        lines.append("  (fewer than 2 segments — no inter-segment pause to measure)")
        return lines
    lines.append(f"  min gap:      {c['min_gap_s']:.3f}s")
    lines.append(f"  mean gap:     {c['mean_gap_s']:.3f}s")
    lines.append(f"  max gap:      {c['max_gap_s']:.3f}s")
    lines.append(f"  total silence:{c['total_silence_s']:8.3f}s")
    label = _format_cut_label(c["recommended_ms"])
    lines.append(
        f"  recommended --min-silence-ms: {label} ({c['recommended_s']:.3f}s)"
    )
    if c["grade"] == "none":
        lines.append(
            "  confidence:   none (no valley — pauses don't separate into "
            "short/long clusters)"
        )
    else:
        sep = (
            "n/a (only one jump / clean split)"
            if c["separation_ratio"] is None
            else f"{c['separation_ratio']:.3f}x the next-widest jump"
        )
        lines.append(
            f"  confidence:   {c['grade']} (valley {c['valley_width_s']:.3f}s is "
            f"{c['dominance'] * 100.0:.1f}% of the {c['spread_s']:.3f}s gap "
            f"spread; {sep})"
        )
    lines.append(f"  suggestion:   {_gap_confidence_summary(c['grade'])} (iter-348)")
    return lines


def render_vad_gap_confidence_json(result):
    """Render the recommendation-confidence grade as a JSON string (iter-348).

    Machine-readable twin of :func:`render_vad_gap_confidence`, mirroring the
    degrade-to-``{"available": false}`` contract the other VAD JSON renderers
    use. Carries the aggregate stats plus the recommendation fields AND the
    confidence fields (``grade`` / ``dominance`` / ``separation_ratio`` /
    ``spread_s`` / ``runner_up_width_s``). The confidence fields are ``null``
    (and ``grade`` ``null`` for a <2-segment result, ``"none"`` for a no-valley
    result), the same JSON spelling of "not measurable" the other gap surfaces
    use. Pure: built from :func:`vad_gap_confidence`, so it works on any
    ``SileroResult``-shaped object.
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
    c = vad_gap_confidence(result)
    payload = {
        "available": True,
        "name": result.name,
        "num_segments": c["num_segments"],
        "num_gaps": c["num_gaps"],
        "min_gap_s": c["min_gap_s"],
        "max_gap_s": c["max_gap_s"],
        "mean_gap_s": c["mean_gap_s"],
        "total_silence_s": c["total_silence_s"],
        "recommended_ms": c["recommended_ms"],
        "recommended_s": c["recommended_s"],
        "split_found": c["split_found"],
        "grade": c["grade"],
        "dominance": c["dominance"],
        "separation_ratio": c["separation_ratio"],
        "spread_s": c["spread_s"],
        "valley_width_s": c["valley_width_s"],
        "runner_up_width_s": c["runner_up_width_s"],
    }
    return json.dumps(payload, indent=2)


def render_vad_gap_confidence_csv(result):
    """Render the recommendation-confidence grade as a one-row CSV table (iter-348).

    The spreadsheet-friendly twin of :func:`render_vad_gap_confidence` /
    :func:`render_vad_gap_confidence_json`, completing the human / ``--json`` /
    ``--csv`` trio every VAD-analysis surface carries. Like the verdict it grades,
    the confidence is a SINGLE result, so the natural CSV is one summary row:
    ``recommended_ms,grade,dominance,separation_ratio,valley_width_s,spread_s``.
    The derivable aggregates are NOT duplicated into the row, matching
    :func:`render_vad_gap_recommend_csv`'s reasoning. A result with fewer than 2
    segments yields the header alone (a valid empty-bodied table — nothing to
    grade). ``result`` of ``None`` (segmenter unavailable) yields a single
    ``# silero VAD unavailable: ...`` comment line. Pure: built with the stdlib
    :mod:`csv` writer, trailing terminator stripped.
    """
    if result is None:
        return (
            "# silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        )
    c = vad_gap_confidence(result)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "recommended_ms",
            "grade",
            "dominance",
            "separation_ratio",
            "valley_width_s",
            "spread_s",
        ]
    )
    if c["num_gaps"] > 0:
        writer.writerow(
            [
                _format_cut_label(c["recommended_ms"]),
                c["grade"],
                "" if c["dominance"] is None else c["dominance"],
                "" if c["separation_ratio"] is None else c["separation_ratio"],
                "" if c["valley_width_s"] is None else c["valley_width_s"],
                "" if c["spread_s"] is None else c["spread_s"],
            ]
        )
    return buf.getvalue().rstrip("\r\n")


def vad_gap_histogram(result, *, bin_width_s=0.5):
    """Bucket the inter-segment silence gaps into a fixed-width histogram (iter-336).

    The min/mean/max aggregates :func:`vad_silence_gaps` reports collapse the
    silence distribution to three numbers; they cannot tell a *bimodal* pause
    pattern (a cluster of short within-turn pauses plus a cluster of long
    between-turn pauses) from a uniform spread with the same min/max. That shape
    is exactly what an operator tuning the end-of-turn hangover
    (``--min-silence-ms`` / the live ``chat.vad.silence_duration``) needs: a
    clear valley between a short-pause mode and a long-pause mode is the safe
    place to set the hangover, and a histogram shows that valley where the
    aggregates hide it. This is the distribution-shape complement of
    :func:`vad_silence_gaps`.

    Pure: takes any object exposing the ``SileroResult`` shape and a positive
    ``bin_width_s`` and returns a plain ``dict`` — it anchors to
    :func:`vad_silence_gaps` for the gap list + aggregates (so the totals always
    agree with ``gv vad-gaps``) and adds a ``bins`` list. Bins are half-open
    ``[lo, hi)`` intervals of width ``bin_width_s`` starting at ``0.0``; a gap of
    exactly ``g`` falls in bin ``floor(g / bin_width_s)`` (a gap on a boundary
    goes to the UPPER bin, standard half-open convention), and the bin span
    covers from ``0`` up to and including the max gap. Each bin records its
    ``lo_s`` / ``hi_s`` (rounded to 3 places, matching the sibling gap surfaces)
    and integer ``count``; the counts sum to ``num_gaps``. A result with fewer
    than 2 segments has no gaps, so ``bins`` is empty (no distribution to shape)
    and the aggregates are ``None``, the same distinction the other gap surfaces
    make. Raises :class:`ValueError` on a non-positive / NaN ``bin_width_s``.
    """
    if not (bin_width_s > 0):  # also rejects NaN (NaN > 0 is False)
        raise ValueError(f"bin_width_s must be positive, got {bin_width_s!r}")
    d = vad_silence_gaps(result)
    gaps = d["gaps"]
    bins = []
    if gaps:
        # Number of bins spans 0 up to and including the max gap. floor(max/bw)
        # is the index of the bin the max gap lands in; +1 makes it a count.
        n_bins = int(d["max_gap_s"] / bin_width_s) + 1
        counts = [0] * n_bins
        for gap in gaps:
            idx = int(gap / bin_width_s)
            # Clamp defensively: floating-point can nudge a boundary gap one bin
            # past the top (e.g. 1.5 / 0.5 == 2.9999...), which would index out.
            if idx >= n_bins:
                idx = n_bins - 1
            counts[idx] += 1
        for i, count in enumerate(counts):
            bins.append(
                {
                    "lo_s": round(i * bin_width_s, 3),
                    "hi_s": round((i + 1) * bin_width_s, 3),
                    "count": count,
                }
            )
    return {
        "num_segments": d["num_segments"],
        "num_gaps": d["num_gaps"],
        "bin_width_s": bin_width_s,
        "min_gap_s": d["min_gap_s"],
        "max_gap_s": d["max_gap_s"],
        "mean_gap_s": d["mean_gap_s"],
        "total_silence_s": d["total_silence_s"],
        "bins": bins,
    }


def render_vad_gap_histogram(result, *, bin_width_s=0.5):
    """Render the silence-gap histogram as plain-text report lines (iter-336).

    The human-readable face of :func:`vad_gap_histogram`, the distribution-shape
    twin of :func:`render_vad_gaps`. ``result`` of ``None`` (segmenter
    unavailable) yields the shared install hint. A result with fewer than 2
    segments has no gaps, so it prints the same short explanatory line
    :func:`render_vad_gaps` uses (no distribution to shape). Otherwise it prints
    the aggregate header (min/mean/max, total silence — naming the actionable
    ``--min-silence-ms`` knob on the min-gap line) then one line per bin: the
    half-open ``[lo, hi)`` range, the count, and an ASCII bar scaled to the
    busiest bin so the distribution shape (and any valley between a short-pause
    and a long-pause mode) is visible at a glance. Pure: returns a list of
    strings (no I/O, no ANSI).
    """
    if result is None:
        return [
            "silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        ]
    h = vad_gap_histogram(result, bin_width_s=bin_width_s)
    lines = [
        f"silero VAD gap histogram — {result.name}",
        f"  segments:     {h['num_segments']}",
        f"  gaps:         {h['num_gaps']} (pauses between consecutive speech regions)",
        f"  bin width:    {bin_width_s:.3f}s",
    ]
    if h["num_gaps"] == 0:
        lines.append("  (fewer than 2 segments — no inter-segment pause to measure)")
        return lines
    lines.append(
        f"  min gap:      {h['min_gap_s']:.3f}s "
        "(shortest real pause — keep --min-silence-ms below this to avoid "
        "merging turns)"
    )
    lines.append(f"  mean gap:     {h['mean_gap_s']:.3f}s")
    lines.append(f"  max gap:      {h['max_gap_s']:.3f}s")
    lines.append(f"  total silence:{h['total_silence_s']:8.3f}s")
    # Scale the ASCII bar to the busiest bin so the tallest is a fixed width.
    max_count = max(b["count"] for b in h["bins"])
    bar_width = 40
    for b in h["bins"]:
        filled = (
            0 if max_count == 0 else round(b["count"] / max_count * bar_width)
        )
        bar = "#" * filled
        lines.append(
            f"  [{b['lo_s']:6.3f}, {b['hi_s']:6.3f})  {b['count']:>4}  {bar}"
        )
    return lines


def render_vad_gap_histogram_json(result, *, bin_width_s=0.5):
    """Render the silence-gap histogram as a JSON string (iter-336).

    Machine-readable twin of :func:`render_vad_gap_histogram`, mirroring the
    degrade-to-``{"available": false}`` contract the other VAD JSON renderers
    use. Carries the aggregate stats plus the ``bin_width_s`` and a ``bins`` list
    of ``{lo_s, hi_s, count}`` objects (empty for a <2-segment result, the same
    JSON spelling of "no distribution" the other gap surfaces use). Pure: built
    from :func:`vad_gap_histogram`, so it works on any ``SileroResult``-shaped
    object.
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
    h = vad_gap_histogram(result, bin_width_s=bin_width_s)
    payload = {
        "available": True,
        "name": result.name,
        "num_segments": h["num_segments"],
        "num_gaps": h["num_gaps"],
        "bin_width_s": h["bin_width_s"],
        "min_gap_s": h["min_gap_s"],
        "max_gap_s": h["max_gap_s"],
        "mean_gap_s": h["mean_gap_s"],
        "total_silence_s": h["total_silence_s"],
        "bins": h["bins"],
    }
    return json.dumps(payload, indent=2)


def render_vad_gap_histogram_csv(result, *, bin_width_s=0.5):
    """Render the silence-gap histogram as a per-bin CSV table (iter-336).

    The spreadsheet/plot-friendly twin of :func:`render_vad_gap_histogram` /
    :func:`render_vad_gap_histogram_json`, completing the human / ``--json`` /
    ``--csv`` trio every VAD-analysis surface carries. The natural CSV unit is
    one row per bin: ``bin_index,lo_s,hi_s,count`` — the shape a plotter wants
    (count-vs-bin, a bar chart) and a spreadsheet wants (one bin per line). The
    aggregate stats are derivable from the per-gap data so they are NOT
    duplicated into the table, matching :func:`render_vad_gaps_csv`'s reasoning.
    A result with fewer than 2 segments yields the header alone (a valid
    empty-bodied table). ``result`` of ``None`` (segmenter unavailable) yields a
    single ``# silero VAD unavailable: ...`` comment line. Pure: built with the
    stdlib :mod:`csv` writer, trailing terminator stripped.
    """
    if result is None:
        return (
            "# silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        )
    h = vad_gap_histogram(result, bin_width_s=bin_width_s)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["bin_index", "lo_s", "hi_s", "count"])
    for i, b in enumerate(h["bins"], start=1):
        writer.writerow([i, b["lo_s"], b["hi_s"], b["count"]])
    return buf.getvalue().rstrip("\r\n")


def vad_gap_sweep(values, results, *, axis="threshold"):
    """Pair each swept-axis value with its inter-segment silence-gap summary.

    iter-330's gap-side analogue of :func:`vad_segmentation_sweep`. Where the
    segment sweep (iter-236+) tabulates segment-count / speech-seconds vs a
    swept knob, the gap sweep tabulates the SILENCE-gap distribution — for each
    swept value it runs :func:`vad_silence_gaps` over that value's segmentation
    and records ``num_gaps`` plus the min/mean/max/total aggregates. The
    headline column is ``min_gap_s``: the shortest real pause is the floor above
    which raising the end-of-turn hangover (``--min-silence-ms`` / the live
    ``chat.vad.silence_duration``) starts merging two genuine turns into one, so
    watching how that floor MOVES as the gate tightens tells an operator how
    much merge headroom each knob setting buys (a stricter ``--threshold`` or a
    longer ``--min-silence-ms`` gates out marginal speech, dropping or merging
    regions, which typically lengthens the shortest surviving pause).

    Pure: takes the parallel ``values`` list (one per swept-axis point) and
    ``results`` list (each a ``SileroResult``-shaped object segmented at the
    matching value) and returns a list of rows ``{axis, "num_segments",
    "num_gaps", "min_gap_s", "mean_gap_s", "max_gap_s", "total_silence_s"}``.
    The aggregates are ``None`` for a row whose segmentation has <2 segments (no
    pause to summarise), exactly as :func:`vad_silence_gaps` returns them. The
    row's swept-axis key IS ``axis`` (default ``"threshold"``). No I/O, no torch
    import, so it is testable in isolation. Raises :class:`ValueError` if the
    two lists differ in length.
    """
    if len(values) != len(results):
        raise ValueError(
            f"values ({len(values)}) and results ({len(results)}) "
            "must be the same length"
        )
    rows = []
    for v, r in zip(values, results):
        d = vad_silence_gaps(r)
        rows.append(
            {
                axis: v,
                "num_segments": d["num_segments"],
                "num_gaps": d["num_gaps"],
                "min_gap_s": d["min_gap_s"],
                "mean_gap_s": d["mean_gap_s"],
                "max_gap_s": d["max_gap_s"],
                "total_silence_s": d["total_silence_s"],
            }
        )
    return rows


def render_vad_gap_sweep(values, results, *, name, axis="threshold"):
    """Render a gap sweep as a plain-text table.

    The human-readable twin of :func:`render_vad_gap_sweep_json`, the gap-side
    analogue of :func:`render_vad_sweep`. ``name`` is the WAV being swept;
    ``axis`` names the swept dimension (sets the column label via
    :data:`_SWEEP_AXIS_LABEL`). Any ``None`` in ``results`` (segmenter
    unavailable) yields the shared install hint. Pure: returns a list of
    strings. Each row prints the swept value, the segment count, the gap count,
    and the min/mean/max gap; a row with <2 segments has no pause to summarise
    and prints ``-`` in the gap columns (distinct from a ``0.000`` gap). Reading
    down the table the min gap typically GROWS as the gate tightens (marginal
    speech is gated out, so adjacent regions merge and the shortest surviving
    pause lengthens) — the value that lifts ``min_gap`` above a target hangover
    is the one that buys merge headroom.
    """
    if any(r is None for r in results):
        return [
            "silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        ]
    rows = vad_gap_sweep(values, results, axis=axis)
    label = _SWEEP_AXIS_LABEL.get(axis, axis)
    lines = [
        f"silero VAD gap sweep — {name}",
        f"  {label:>9}  segments  gaps  min_gap  mean_gap   max_gap",
    ]
    for row in rows:
        if row["num_gaps"] == 0:
            gap_cols = f"{'-':>7}  {'-':>8}  {'-':>8}"
        else:
            gap_cols = (
                f"{row['min_gap_s']:>7.3f}  {row['mean_gap_s']:>8.3f}  "
                f"{row['max_gap_s']:>8.3f}"
            )
        lines.append(
            f"  {_format_sweep_axis_value(axis, row[axis]):>9}  "
            f"{row['num_segments']:>8}  {row['num_gaps']:>4}  {gap_cols}"
        )
    return lines


def render_vad_gap_sweep_json(values, results, *, name, axis="threshold"):
    """Render a gap sweep as a JSON string.

    Machine-readable twin of :func:`render_vad_gap_sweep`, so the sweep can feed
    a plotting/tuning script. The payload carries the swept ``axis`` name (the
    rows are keyed by that same name) and a ``sweep`` list of per-value rows,
    each with ``num_segments`` / ``num_gaps`` / ``min_gap_s`` / ``mean_gap_s`` /
    ``max_gap_s`` / ``total_silence_s`` (the aggregates ``null`` for a <2-segment
    row, the same JSON ``null`` distinction :func:`render_vad_gaps_json` makes).
    Any ``None`` in ``results`` → ``{"available": false}`` + install hint,
    mirroring :func:`render_vad_sweep_json`. Pure: returns a single JSON string.
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
    rows = vad_gap_sweep(values, results, axis=axis)
    payload = {
        "available": True,
        "name": name,
        "axis": axis,
        "sweep": rows,
    }
    return json.dumps(payload, indent=2)


def render_vad_gap_sweep_csv(values, results, *, name, axis="threshold"):
    """Render a gap sweep as CSV text (no trailing newline).

    The spreadsheet/plot-friendly twin of :func:`render_vad_gap_sweep_json`:
    where JSON nests the rows under a ``sweep`` key, CSV emits a flat
    ``<axis>,num_segments,num_gaps,min_gap_s,mean_gap_s,max_gap_s,
    total_silence_s`` table that pipes straight into a spreadsheet or plotter.
    The first column header is the swept ``axis`` name so the grid is
    self-describing. ``name`` is accepted for signature parity with the other
    ``render_vad_gap_sweep_*`` twins but is not part of the tabular body (a CSV
    is a pure data grid), matching :func:`render_vad_sweep_csv`. A <2-segment
    row emits empty cells in the aggregate columns (the CSV spelling of JSON
    ``null`` / the human table's ``-``). Any ``None`` in ``results`` (segmenter
    unavailable) yields a single ``# silero VAD unavailable: ...`` comment line.
    Pure: returns a single string built with the stdlib :mod:`csv` writer,
    trailing terminator stripped.
    """
    if any(r is None for r in results):
        return (
            "# silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            axis,
            "num_segments",
            "num_gaps",
            "min_gap_s",
            "mean_gap_s",
            "max_gap_s",
            "total_silence_s",
        ]
    )
    for row in vad_gap_sweep(values, results, axis=axis):
        writer.writerow(
            [
                row[axis],
                row["num_segments"],
                row["num_gaps"],
                # None → "" (empty cell), the CSV spelling of JSON null.
                "" if row["min_gap_s"] is None else row["min_gap_s"],
                "" if row["mean_gap_s"] is None else row["mean_gap_s"],
                "" if row["max_gap_s"] is None else row["max_gap_s"],
                row["total_silence_s"],
            ]
        )
    return buf.getvalue().rstrip("\r\n")


def vad_gap_peak_sweep(values, results, *, cuts_ms=DEFAULT_GAP_CDF_CUTS_MS, axis="threshold"):
    """Pair each swept-axis value with its COSTLIEST cost-curve band (iter-364).

    The peak-side analogue of :func:`vad_gap_sweep`. Where ``vad_gap_sweep``
    tabulates the silence-gap distribution (min/mean/max gap) across a swept
    knob, this surface tabulates the iter-350 cost PEAK — for each swept value it
    runs :func:`vad_gap_peak` over that value's segmentation and records the
    steepest band: its ms range, the pauses it merges, and its marginal rate per
    +100 ms of hangover. The headline column is ``peak_rate_per_100ms``: it shows
    how the cost of the densest pause cluster MOVES as a SEGMENTER knob (e.g. the
    ``--min-speech-ms`` floor) tightens. A stricter floor gates out marginal
    speech, which merges adjacent regions and reshapes the pause clusters — so the
    steepest band, and how expensive it is to push the hangover through it, drift
    with the knob. Where ``vad-gap-sweep`` watches the cheapest valley move, this
    watches the costliest cluster move; the two are the sweep-side twins of
    ``vad-gap-recommend`` vs ``vad-gap-peak``.

    Pure: takes the parallel ``values`` list (one per swept-axis point) and
    ``results`` list (each a ``SileroResult``-shaped object segmented at the
    matching value) and returns a list of rows ``{axis, "num_segments",
    "num_gaps", "peak_found", "peak_from_ms", "peak_to_ms", "peak_width_ms",
    "peak_merged_added", "peak_rate_per_100ms"}``. The peak fields are ``None`` /
    ``False`` for a row whose segmentation names no cost peak (a <2-segment
    result with no gaps, or an all-valley range with no pause cluster), exactly
    the "no structure to name" spelling :func:`vad_gap_peak` returns. The row's
    swept-axis key IS ``axis`` (default ``"threshold"``). Anchors to
    :func:`vad_gap_peak` (single peak — ``top_n=1``) so the per-row numbers agree
    EXACTLY with ``gv vad-gap-peak`` at each swept value. No I/O, no torch import,
    so it is testable in isolation. Raises :class:`ValueError` if the two lists
    differ in length (or if ``cuts_ms`` is empty / negative, delegated to
    :func:`vad_gap_peak`).
    """
    if len(values) != len(results):
        raise ValueError(
            f"values ({len(values)}) and results ({len(results)}) "
            "must be the same length"
        )
    rows = []
    for v, r in zip(values, results):
        p = vad_gap_peak(r, cuts_ms=cuts_ms)
        rows.append(
            {
                axis: v,
                "num_segments": p["num_segments"],
                "num_gaps": p["num_gaps"],
                "peak_found": p["peak_found"],
                "peak_from_ms": p["peak_from_ms"],
                "peak_to_ms": p["peak_to_ms"],
                "peak_width_ms": p["peak_width_ms"],
                "peak_merged_added": p["peak_merged_added"],
                "peak_rate_per_100ms": p["peak_rate_per_100ms"],
            }
        )
    return rows


def render_vad_gap_peak_sweep(values, results, *, name, cuts_ms=DEFAULT_GAP_CDF_CUTS_MS, axis="threshold"):
    """Render a cost-peak sweep as a plain-text table (iter-364).

    The human-readable twin of :func:`render_vad_gap_peak_sweep_json`, the
    peak-side analogue of :func:`render_vad_gap_sweep`. ``name`` is the WAV being
    swept; ``axis`` names the swept dimension (sets the column label via
    :data:`_SWEEP_AXIS_LABEL`). Any ``None`` in ``results`` (segmenter
    unavailable) yields the shared install hint. Pure: returns a list of strings.
    Each row prints the swept value, the segment count, the gap count, and the
    costliest band's ms range / merged-pauses / rate per +100 ms; a row that
    names no cost peak (no gaps, or an all-valley range) prints ``-`` in the peak
    columns (distinct from a band with a ``0.000`` rate, which never happens — a
    named peak always has rate > 0). Reading down the table the peak rate shows
    how expensive the densest pause cluster is at each knob setting — where it
    SHRINKS, raising the hangover through the steepest band costs less.
    """
    if any(r is None for r in results):
        return [
            "silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        ]
    rows = vad_gap_peak_sweep(values, results, cuts_ms=cuts_ms, axis=axis)
    label = _SWEEP_AXIS_LABEL.get(axis, axis)
    lines = [
        f"silero VAD gap cost-peak sweep — {name}",
        f"  {label:>9}  segments  gaps  peak_band_ms  merged  rate/100ms",
    ]
    for row in rows:
        if not row["peak_found"]:
            peak_cols = f"{'-':>12}  {'-':>6}  {'-':>10}"
        else:
            band = (
                f"{_format_cut_label(row['peak_from_ms'])}-"
                f"{_format_cut_label(row['peak_to_ms'])}"
            )
            peak_cols = (
                f"{band:>12}  {row['peak_merged_added']:>6}  "
                f"{row['peak_rate_per_100ms']:>10.3f}"
            )
        lines.append(
            f"  {_format_sweep_axis_value(axis, row[axis]):>9}  "
            f"{row['num_segments']:>8}  {row['num_gaps']:>4}  {peak_cols}"
        )
    return lines


def render_vad_gap_peak_sweep_json(values, results, *, name, cuts_ms=DEFAULT_GAP_CDF_CUTS_MS, axis="threshold"):
    """Render a cost-peak sweep as a JSON string (iter-364).

    Machine-readable twin of :func:`render_vad_gap_peak_sweep`, so the sweep can
    feed a plotting/tuning script. The payload carries the swept ``axis`` name
    (the rows are keyed by that same name), the ``cuts_ms`` axis the bands were
    scanned over, and a ``sweep`` list of per-value rows, each with
    ``num_segments`` / ``num_gaps`` / ``peak_found`` / ``peak_from_ms`` /
    ``peak_to_ms`` / ``peak_width_ms`` / ``peak_merged_added`` /
    ``peak_rate_per_100ms`` (the peak fields ``null`` / ``false`` for a row that
    names no cost peak, the same JSON spelling of "no structure" the other peak
    surfaces use). Any ``None`` in ``results`` → ``{"available": false}`` +
    install hint, mirroring :func:`render_vad_gap_sweep_json`. Pure: returns a
    single JSON string.
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
    rows = vad_gap_peak_sweep(values, results, cuts_ms=cuts_ms, axis=axis)
    payload = {
        "available": True,
        "name": name,
        "axis": axis,
        "cuts_ms": list(cuts_ms),
        "sweep": rows,
    }
    return json.dumps(payload, indent=2)


def render_vad_gap_peak_sweep_csv(values, results, *, name, cuts_ms=DEFAULT_GAP_CDF_CUTS_MS, axis="threshold"):
    """Render a cost-peak sweep as CSV text (no trailing newline) (iter-364).

    The spreadsheet/plot-friendly twin of :func:`render_vad_gap_peak_sweep_json`:
    where JSON nests the rows under a ``sweep`` key, CSV emits a flat
    ``<axis>,num_segments,num_gaps,peak_found,peak_from_ms,peak_to_ms,
    peak_width_ms,peak_merged_added,peak_rate_per_100ms`` table that pipes
    straight into a spreadsheet or plotter. The first column header is the swept
    ``axis`` name so the grid is self-describing. ``name`` is accepted for
    signature parity with the other ``render_vad_gap_peak_sweep_*`` twins but is
    not part of the tabular body (a CSV is a pure data grid), matching
    :func:`render_vad_gap_sweep_csv`. A row that names no cost peak emits
    ``peak_found`` ``False`` and empty cells in the peak-measure columns (the CSV
    spelling of JSON ``null`` / the human table's ``-``). Any ``None`` in
    ``results`` (segmenter unavailable) yields a single ``# silero VAD
    unavailable: ...`` comment line. Pure: returns a single string built with the
    stdlib :mod:`csv` writer, trailing terminator stripped.
    """
    if any(r is None for r in results):
        return (
            "# silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            axis,
            "num_segments",
            "num_gaps",
            "peak_found",
            "peak_from_ms",
            "peak_to_ms",
            "peak_width_ms",
            "peak_merged_added",
            "peak_rate_per_100ms",
        ]
    )
    for row in vad_gap_peak_sweep(values, results, cuts_ms=cuts_ms, axis=axis):
        if row["peak_found"]:
            writer.writerow(
                [
                    row[axis],
                    row["num_segments"],
                    row["num_gaps"],
                    row["peak_found"],
                    _format_cut_label(row["peak_from_ms"]),
                    _format_cut_label(row["peak_to_ms"]),
                    _format_cut_label(row["peak_width_ms"]),
                    row["peak_merged_added"],
                    row["peak_rate_per_100ms"],
                ]
            )
        else:
            # None → "" (empty cell), the CSV spelling of JSON null / human "-".
            writer.writerow(
                [
                    row[axis],
                    row["num_segments"],
                    row["num_gaps"],
                    row["peak_found"],
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
    return buf.getvalue().rstrip("\r\n")


def vad_gap_grid(
    row_values, col_values, results, *, row_axis="threshold", col_axis="min_silence_ms"
):
    """Pair each (row, col) axis-value cell with its inter-segment silence-gap summary.

    iter-332's 2-D analogue of :func:`vad_gap_sweep`, and the gap-side twin of
    :func:`vad_segmentation_grid`. Where the 1-D gap sweep (iter-330) tabulates
    the silence-gap distribution across ONE knob, the gap grid tabulates it
    across the cartesian product of TWO knobs (the gate × a column knob), so an
    operator can read how the shortest-pause floor MOVES in two dimensions at
    once instead of running N separate 1-D gap sweeps. ``vad-grid`` is to
    ``vad-sweep`` what this is to ``vad-gap-sweep``: same row-major flattening,
    same headline (``min_gap_s`` — the floor above which a longer end-of-turn
    hangover starts merging two genuine turns), but the cells now span a grid.

    ``results`` is the flattened cell list in ROW-MAJOR order (row 0's whole row
    of columns first, then row 1's, …), length
    ``len(row_values) * len(col_values)`` — exactly the order
    :func:`vad_segmentation_grid` consumes. For each cell it runs
    :func:`vad_silence_gaps` over that cell's segmentation and records
    ``num_segments`` / ``num_gaps`` plus the min/mean/max/total aggregates.

    Pure: returns a flat list of cell dicts ``{row_axis, col_axis,
    "num_segments", "num_gaps", "min_gap_s", "mean_gap_s", "max_gap_s",
    "total_silence_s"}`` in that same row-major order. The aggregates are
    ``None`` for a cell whose segmentation has <2 segments (no pause to
    summarise), exactly as :func:`vad_silence_gaps` returns them. No I/O, no
    torch import, so it is testable in isolation. Raises :class:`ValueError` if
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
            d = vad_silence_gaps(r)
            cells.append(
                {
                    row_axis: rv,
                    col_axis: cv,
                    "num_segments": d["num_segments"],
                    "num_gaps": d["num_gaps"],
                    "min_gap_s": d["min_gap_s"],
                    "mean_gap_s": d["mean_gap_s"],
                    "max_gap_s": d["max_gap_s"],
                    "total_silence_s": d["total_silence_s"],
                }
            )
    return cells


def render_vad_gap_grid(
    row_values,
    col_values,
    results,
    *,
    name,
    row_axis="threshold",
    col_axis="min_silence_ms",
):
    """Render a 2-D gap grid as a plain-text table.

    The human-readable twin of :func:`render_vad_gap_grid_json`, the gap-side
    analogue of :func:`render_vad_grid` and the 2-D analogue of
    :func:`render_vad_gap_sweep`: a FLAT one-row-per-cell table (not a matrix)
    so each cell's two swept values plus its gap aggregates stay unambiguous.
    ``name`` is the WAV being swept; ``row_axis`` / ``col_axis`` name the two
    swept dimensions, which set the two leading column labels and value formats
    (a gate prints ``0.40``, a millisecond knob a bare ``800``, the seconds
    ceiling compactly via ``%g``). Any ``None`` in ``results`` (segmenter
    unavailable) yields the shared install hint.

    Each row prints the two swept values, the segment count, the gap count, and
    the min/mean/max gap; a cell with <2 segments has no pause to summarise and
    prints ``-`` in the gap columns (distinct from a ``0.000`` gap). Like
    :func:`render_vad_gap_sweep` there is no ``best:`` pick block — the gap
    surface's signal is the distribution, not a segment-count target. Pure:
    returns a list of strings.
    """
    if any(r is None for r in results):
        return [
            "silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        ]
    cells = vad_gap_grid(
        row_values, col_values, results, row_axis=row_axis, col_axis=col_axis
    )
    row_label = _SWEEP_AXIS_LABEL.get(row_axis, row_axis)
    col_label = _SWEEP_AXIS_LABEL.get(col_axis, col_axis)
    lines = [
        f"silero VAD gap grid — {name} ({row_label} × {col_label})",
        f"  {row_label:>11}  {col_label:>11}  segments  gaps  "
        "min_gap  mean_gap   max_gap",
    ]
    for cell in cells:
        if cell["num_gaps"] == 0:
            gap_cols = f"{'-':>7}  {'-':>8}  {'-':>8}"
        else:
            gap_cols = (
                f"{cell['min_gap_s']:>7.3f}  {cell['mean_gap_s']:>8.3f}  "
                f"{cell['max_gap_s']:>8.3f}"
            )
        lines.append(
            f"  {_format_sweep_axis_value(row_axis, cell[row_axis]):>11}  "
            f"{_format_sweep_axis_value(col_axis, cell[col_axis]):>11}  "
            f"{cell['num_segments']:>8}  {cell['num_gaps']:>4}  {gap_cols}"
        )
    return lines


def render_vad_gap_grid_json(
    row_values,
    col_values,
    results,
    *,
    name,
    row_axis="threshold",
    col_axis="min_silence_ms",
):
    """Render a 2-D gap grid as a JSON string.

    Machine-readable twin of :func:`render_vad_gap_grid`, so the grid can feed a
    plotting/tuning script. The payload carries both swept axis names
    (``row_axis`` / ``col_axis``) so a consumer knows which two dimensions the
    cells vary (the cells are keyed by those same names) and a ``grid`` list of
    per-cell rows, each with ``num_segments`` / ``num_gaps`` / ``min_gap_s`` /
    ``mean_gap_s`` / ``max_gap_s`` / ``total_silence_s`` (the aggregates
    ``null`` for a <2-segment cell, the same JSON ``null`` distinction
    :func:`render_vad_gaps_json` and :func:`render_vad_gap_sweep_json` make).
    Any ``None`` in ``results`` → ``{"available": false}`` + install hint,
    mirroring :func:`render_vad_grid_json`. Like the gap sweep there is no
    ``target`` / ``best`` pick (the gap surface does not headline a
    segment-count target). Pure: returns a single JSON string.
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
    cells = vad_gap_grid(
        row_values, col_values, results, row_axis=row_axis, col_axis=col_axis
    )
    payload = {
        "available": True,
        "name": name,
        "row_axis": row_axis,
        "col_axis": col_axis,
        "grid": cells,
    }
    return json.dumps(payload, indent=2)


def render_vad_gap_grid_csv(
    row_values,
    col_values,
    results,
    *,
    name,
    row_axis="threshold",
    col_axis="min_silence_ms",
):
    """Render a 2-D gap grid as CSV text (no trailing newline).

    The spreadsheet/plot-friendly twin of :func:`render_vad_gap_grid_json`:
    where JSON nests the cells under a ``grid`` key, CSV emits a flat
    ``<row_axis>,<col_axis>,num_segments,num_gaps,min_gap_s,mean_gap_s,
    max_gap_s,total_silence_s`` table (one row per cell, in row-major order)
    that pivots straight into a spreadsheet or a plotting script. The first two
    column headers are the swept axis names so the grid is self-describing.
    ``name`` is accepted for signature parity with the other
    ``render_vad_gap_grid_*`` twins but is not part of the tabular body (a CSV
    is a pure data grid), matching :func:`render_vad_grid_csv`. A <2-segment
    cell emits empty cells in the aggregate columns (the CSV spelling of JSON
    ``null`` / the human table's ``-``). Any ``None`` in ``results`` (segmenter
    unavailable) yields a single ``# silero VAD unavailable: ...`` comment line.
    Pure: returns a single string built with the stdlib :mod:`csv` writer,
    trailing terminator stripped.
    """
    if any(r is None for r in results):
        return (
            "# silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            row_axis,
            col_axis,
            "num_segments",
            "num_gaps",
            "min_gap_s",
            "mean_gap_s",
            "max_gap_s",
            "total_silence_s",
        ]
    )
    for cell in vad_gap_grid(
        row_values, col_values, results, row_axis=row_axis, col_axis=col_axis
    ):
        writer.writerow(
            [
                cell[row_axis],
                cell[col_axis],
                cell["num_segments"],
                cell["num_gaps"],
                # None → "" (empty cell), the CSV spelling of JSON null.
                "" if cell["min_gap_s"] is None else cell["min_gap_s"],
                "" if cell["mean_gap_s"] is None else cell["mean_gap_s"],
                "" if cell["max_gap_s"] is None else cell["max_gap_s"],
                cell["total_silence_s"],
            ]
        )
    return buf.getvalue().rstrip("\r\n")


def vad_gap_delta(result_a, result_b):
    """Compute how the inter-segment silence-gap distribution shifts between two
    segmentations of the same WAV (iter-334).

    The gap-side analogue of :func:`vad_segmentation_delta`, and the pure core of
    ``gv vad-gap-diff``: where ``vad-diff`` quantifies how the segment-count /
    speech-seconds shift between two ``--threshold`` settings, this quantifies
    how the SILENCE-gap distribution shifts — the min/mean/max gap and the total
    silence. The headline is ``min_gap_s_delta``: the shortest real pause is the
    floor above which raising the end-of-turn hangover (``--min-silence-ms`` /
    the live ``chat.vad.silence_duration``) starts merging two genuine turns into
    one, so watching how that floor MOVES between two gates tells an operator
    whether a stricter gate buys merge headroom (gating out marginal speech
    typically drops or merges adjacent regions, lengthening the shortest
    surviving pause).

    Pure: runs :func:`vad_silence_gaps` over each result and returns a plain
    ``dict`` of both sides plus their signed deltas (b minus a). The
    always-present counts (``num_segments`` / ``num_gaps``) carry integer deltas.
    The gap aggregates (``min_gap_s`` / ``mean_gap_s`` / ``max_gap_s``) are
    ``None`` for a side with fewer than 2 segments (no pause to summarise), so a
    delta is ``None`` whenever EITHER side is ``None`` — a missing pause cannot
    be differenced, the same JSON ``null`` distinction the gap renderers make.
    ``total_silence_s`` is always a float (``0.0`` when there are no gaps), so its
    delta is always defined. Gap deltas round to 3 places, matching the sibling
    gap surfaces. No I/O, no torch import, so it is testable without importing
    torch.
    """
    a = vad_silence_gaps(result_a)
    b = vad_silence_gaps(result_b)

    def _delta(x, y):
        # A missing pause (None on either side) cannot be differenced.
        if x is None or y is None:
            return None
        return round(y - x, 3)

    return {
        "num_segments_a": a["num_segments"],
        "num_segments_b": b["num_segments"],
        "num_segments_delta": b["num_segments"] - a["num_segments"],
        "num_gaps_a": a["num_gaps"],
        "num_gaps_b": b["num_gaps"],
        "num_gaps_delta": b["num_gaps"] - a["num_gaps"],
        "min_gap_s_a": a["min_gap_s"],
        "min_gap_s_b": b["min_gap_s"],
        "min_gap_s_delta": _delta(a["min_gap_s"], b["min_gap_s"]),
        "mean_gap_s_a": a["mean_gap_s"],
        "mean_gap_s_b": b["mean_gap_s"],
        "mean_gap_s_delta": _delta(a["mean_gap_s"], b["mean_gap_s"]),
        "max_gap_s_a": a["max_gap_s"],
        "max_gap_s_b": b["max_gap_s"],
        "max_gap_s_delta": _delta(a["max_gap_s"], b["max_gap_s"]),
        "total_silence_s_a": a["total_silence_s"],
        "total_silence_s_b": b["total_silence_s"],
        "total_silence_s_delta": _delta(a["total_silence_s"], b["total_silence_s"]),
    }


def render_vad_gap_diff(result_a, result_b, *, label_a, label_b):
    """Render a two-threshold silence-gap comparison as plain-text lines (iter-334).

    The human-readable twin of :func:`render_vad_gap_diff_json`, the gap-side
    analogue of :func:`render_vad_diff`. ``label_a`` / ``label_b`` are the two
    ``--threshold`` values being compared. Either result of ``None`` (segmenter
    unavailable) yields the shared install hint, matching :func:`render_vad_diff`.
    Each aggregate row prints ``A → B (Δ)``; a side with fewer than 2 segments
    has no pause to summarise and prints ``-`` for that side's gap value, and the
    delta prints ``n/a`` when EITHER side is missing (a missing pause cannot be
    differenced, distinct from a ``0.000s`` change). The min-gap row names the
    actionable knob (``--min-silence-ms``). Pure: returns a list of strings.
    """
    if result_a is None or result_b is None:
        return [
            "silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        ]
    d = vad_gap_delta(result_a, result_b)
    name = result_a.name

    def _gap(value):
        # A missing pause (None) prints "-" — distinct from a 0.000s gap.
        return "-" if value is None else f"{value:.3f}s"

    def _gap_delta(value):
        # A delta is "n/a" when either side had no pause to difference.
        if value is None:
            return "n/a"
        return f"{_signed_float3(value)}s"

    return [
        f"silero VAD gap diff — {name}",
        f"  threshold A:  {label_a:.2f}",
        f"  threshold B:  {label_b:.2f}",
        f"  segments:     {d['num_segments_a']} → {d['num_segments_b']} "
        f"({_signed(d['num_segments_delta'])})",
        f"  gaps:         {d['num_gaps_a']} → {d['num_gaps_b']} "
        f"({_signed(d['num_gaps_delta'])})",
        f"  min gap:      {_gap(d['min_gap_s_a'])} → {_gap(d['min_gap_s_b'])} "
        f"({_gap_delta(d['min_gap_s_delta'])}) — keep --min-silence-ms below "
        "this to avoid merging turns",
        f"  mean gap:     {_gap(d['mean_gap_s_a'])} → {_gap(d['mean_gap_s_b'])} "
        f"({_gap_delta(d['mean_gap_s_delta'])})",
        f"  max gap:      {_gap(d['max_gap_s_a'])} → {_gap(d['max_gap_s_b'])} "
        f"({_gap_delta(d['max_gap_s_delta'])})",
        f"  total silence:{_gap(d['total_silence_s_a'])} → "
        f"{_gap(d['total_silence_s_b'])} "
        f"({_gap_delta(d['total_silence_s_delta'])})",
    ]


def render_vad_gap_diff_json(result_a, result_b, *, label_a, label_b):
    """Render a two-threshold silence-gap comparison as a JSON string (iter-334).

    Machine-readable twin of :func:`render_vad_gap_diff`, so a tuning script can
    consume the gap shift directly. Carries the two ``--threshold`` labels plus
    every key :func:`vad_gap_delta` returns (both sides + signed deltas). The gap
    aggregate deltas are ``null`` when either side has fewer than 2 segments (no
    pause to difference), the same JSON ``null`` distinction
    :func:`render_vad_gaps_json` makes. Either result ``None`` →
    ``{"available": false}`` + install hint, mirroring :func:`render_vad_diff_json`.
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
        **vad_gap_delta(result_a, result_b),
    }
    return json.dumps(payload, indent=2)


def render_vad_gap_diff_csv(result_a, result_b, *, label_a, label_b):
    """Render a two-threshold silence-gap comparison as CSV text (iter-334).

    The spreadsheet/plot-friendly twin of :func:`render_vad_gap_diff_json`,
    completing the human / ``--json`` / ``--csv`` trio. A gap diff IS the
    two-point degenerate of a gap sweep, so this emits the SAME flat
    ``threshold,num_segments,num_gaps,min_gap_s,mean_gap_s,max_gap_s,
    total_silence_s`` schema as :func:`render_vad_gap_sweep_csv` — one row for
    threshold A, one for threshold B — which means a two-value
    ``gv vad-gap-sweep --csv`` and a ``gv vad-gap-diff --csv`` over the same pair
    produce byte-identical tables, exactly as ``gv vad-diff --csv`` matches a
    two-value ``gv vad-sweep --csv`` (iter-313). The signed deltas the human/JSON
    twins surface are trivially derivable from the two rows (b minus a), so they
    are left out rather than duplicated into a wide row. A side with <2 segments
    emits empty cells in the aggregate columns (the CSV spelling of JSON
    ``null``). Either result ``None`` (segmenter unavailable) yields a single
    ``# silero VAD unavailable: ...`` comment line. Pure: returns a single string
    built with the stdlib :mod:`csv` writer, trailing terminator stripped.
    """
    if result_a is None or result_b is None:
        return (
            "# silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "threshold",
            "num_segments",
            "num_gaps",
            "min_gap_s",
            "mean_gap_s",
            "max_gap_s",
            "total_silence_s",
        ]
    )
    for label, result in ((label_a, result_a), (label_b, result_b)):
        g = vad_silence_gaps(result)
        writer.writerow(
            [
                label,
                g["num_segments"],
                g["num_gaps"],
                # None → "" (empty cell), the CSV spelling of JSON null.
                "" if g["min_gap_s"] is None else g["min_gap_s"],
                "" if g["mean_gap_s"] is None else g["mean_gap_s"],
                "" if g["max_gap_s"] is None else g["max_gap_s"],
                g["total_silence_s"],
            ]
        )
    return buf.getvalue().rstrip("\r\n")


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


def render_vad_diff_csv(result_a, result_b, *, label_a, label_b):
    """Render a two-threshold segmentation comparison as CSV text.

    The spreadsheet/plot-friendly twin of :func:`render_vad_diff_json`,
    completing the human / ``--json`` / ``--csv`` trio that ``gv vad-sweep`` and
    ``gv vad-grid`` already carry (iter-237 / iter-251) but ``gv vad-diff`` was
    missing. A diff IS the two-point degenerate of a threshold sweep, so this
    emits the SAME flat ``threshold,num_segments,speech_s`` schema as
    :func:`render_vad_sweep_csv` — one row for threshold A, one for threshold B —
    which means a two-value ``gv vad-sweep --csv`` and a ``gv vad-diff --csv``
    over the same pair produce byte-identical tables, and a consumer can
    ``pandas.concat`` diffs and sweeps without reconciling columns. The signed
    deltas the human/JSON twins surface are trivially derivable from the two
    rows (b minus a), so they are left out rather than duplicated into an
    awkward wide row. Either result ``None`` (segmenter unavailable) yields a
    single ``# silero VAD unavailable: ...`` comment line, matching
    :func:`render_vad_sweep_csv` so a degraded run is self-describing rather than
    silently empty. Pure: returns a single string built with the stdlib
    :mod:`csv` writer (RFC-4180 quoting, ``\\r\\n`` row terminators) with the
    trailing terminator stripped.
    """
    if result_a is None or result_b is None:
        return (
            "# silero VAD unavailable: install 'silero-vad' (pulls torch + "
            "torchaudio) to enable offline neural segmentation"
        )
    d = vad_segmentation_delta(result_a, result_b)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["threshold", "num_segments", "speech_s"])
    writer.writerow([label_a, d["num_segments_a"], d["speech_s_a"]])
    writer.writerow([label_b, d["num_segments_b"], d["speech_s_b"]])
    return buf.getvalue().rstrip("\r\n")


# gv vad-sweep axis metadata. iter-236 swept only the P(speech) gate; iter-238
# adds the trailing-silence hangover as a second axis; iter-239 adds the
# minimum-speech floor as a third; iter-253 adds the symmetric region padding
# (speech_pad_ms) as a fourth. iter-254 added speech_pad_ms as a vad-grid column
# axis; iter-255 adds the force-split ceiling (max_speech_s) as a vad-grid column
# axis — the first SECONDS axis (every prior axis is the gate or a millisecond
# knob). Each entry gives the human-table column label (right-justified to 9,
# matching the original "threshold" width) and the per-value display formatter —
# a gate prints with 2 decimals (0.30), a millisecond knob (hangover, speech
# floor, or pad) as a bare integer (800), and the seconds ceiling compactly
# (5, 12.5, inf) via %g. The dict key is also the row key emitted by
# vad_segmentation_sweep and the CSV/JSON column name, so a consumer reads which
# dimension was swept straight off the data. Millisecond axes format identically,
# so they're grouped below; the seconds axis is its own group.
_SWEEP_MS_AXES = ("min_silence_ms", "min_speech_ms", "speech_pad_ms")
_SWEEP_SECONDS_AXES = ("max_speech_s",)
_SWEEP_AXIS_LABEL = {
    "threshold": "threshold",
    "min_silence_ms": "min_silence",
    "min_speech_ms": "min_speech",
    "speech_pad_ms": "speech_pad",
    "max_speech_s": "max_speech",
}


def _format_sweep_axis_value(axis, value):
    """Format one swept-axis value for the human table (gate / ms / seconds knob)."""
    if axis in _SWEEP_MS_AXES:
        return f"{value:.0f}"
    if axis in _SWEEP_SECONDS_AXES:
        # %g gives a compact seconds value (5, 12.5) and renders inf as "inf"
        # (the never-force-split sentinel), so an operator can include the
        # baseline cap in the sweep without a special-cased branch.
        return f"{value:g}"
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
    # iter-246: a scalar target renders as its bare count ("3"), a (lo, hi) band
    # as "lo-hi" ("3-5"); _format_target owns the distinction so the lines read
    # naturally for either form.
    target_text = _format_target(target)
    best = pick_best_grid_cell(cells, target, tie_break)
    if best is None:
        return [f"  best: none (empty {empty_noun}; target {target_text} segments)"]
    lines = [
        f"  best: {format_axes(best)} "
        f"({best['num_segments']} segments, "
        f"|Δ|={grid_cell_distance(best, target)} from target {target_text})"
    ]
    if top is not None:
        ranked = pick_top_grid_cells(cells, target, top, tie_break)
        lines.append(f"  top {len(ranked)} (closest to target {target_text}):")
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
    """Distance from one grid cell's segment count to ``target`` — lower better.

    The VAD-grid scoring analogue of :meth:`MirrorGridPoint.score`: where the
    WPM grid folds convergence + lurch into a lower-is-better number, a VAD
    grid cell is scored purely by how far its recovered segment count sits from
    the operator's ``target`` (e.g. one segment per spoken sentence).

    ``target`` is either a scalar count or an iter-246 ``(lo, hi)`` tolerance
    band:

    - **scalar** — the original ``|num_segments - target|`` (e.g. target 5,
      count 7 → 2).
    - **closed band ``(lo, hi)``** — ``0`` for any count INSIDE the inclusive
      ``[lo, hi]`` window (every count in the band is equally perfect), else the
      gap to the nearest edge (count below ``lo`` → ``lo - count``; above ``hi``
      → ``count - hi``). A degenerate band ``(n, n)`` reduces to the scalar
      distance to ``n``.
    - **open band (iter-247)** — one edge is ``None`` ("no bound on that
      side"): ``(lo, None)`` ("at least ``lo``") scores ``0`` for any count
      ``>= lo`` and ``lo - count`` below it; ``(None, hi)`` ("at most ``hi``")
      scores ``0`` for any count ``<= hi`` and ``count - hi`` above it. The open
      side simply skips its bound check, so the closed-edge gap is the only
      distance that can be non-zero.
    - **set (iter-248)** — a ``list`` of elements (each a scalar or a band):
      the MIN distance to any element, so a count that satisfies ANY listed
      target scores ``0`` and otherwise scores the gap to the nearest one.
    - **preference (iter-249)** — a ``{"prefer": [...]}`` dict: distance is the
      MIN over its elements, IDENTICAL to the set — the precedence affects only
      tie-breaking (:func:`grid_cell_sort_key`), never the distance, so two
      counts that both satisfy the preference are equidistant here and the
      preference order decides between them at the sort-key layer.
    - **weighted set (iter-250)** — a ``{"weighted": [(element, penalty), ...]}``
      dict: the MIN over each element's distance PLUS its penalty, so a
      lower-penalty (more-preferred) count can score below a higher-penalty one
      even at a larger raw distance. Unlike the preference, the weight enters the
      distance itself, so it can override a distance gap (not just break a tie).
    - **scaled set (iter-252)** — a ``{"scaled": [(element, factor), ...]}``
      dict: the MIN over each element's distance TIMES its factor, the
      MULTIPLICATIVE twin of the weighted set. An exact hit stays free
      (``0 * factor = 0``) and the cost grows with distance, so preference scales
      with how far the cell drifts rather than offsetting by a fixed amount.
    - **affine set (iter-287)** — a ``{"affine": [(element, factor, penalty),
      ...]}`` dict: the MIN over each element's distance SCALED by its factor THEN
      OFFSET by its penalty (``distance * factor + penalty``), composing the
      multiplicative factor and additive penalty on one element. Generalises both
      weighted (factor ``1``) and scaled (penalty ``0``); the penalty bites even
      an exact hit while the factor grows the cost with distance.

    Pure — reads only ``cell["num_segments"]``, an int, so it never touches
    torch.
    """
    count = cell["num_segments"]
    if isinstance(target, dict) and "weighted" in target:
        # iter-250: each element's distance is its raw distance PLUS its penalty,
        # and the set scores as the MIN over those penalised distances — so a
        # cheaper (lower-penalty) element can win even at a larger raw distance.
        return min(
            grid_cell_distance(cell, element) + penalty
            for element, penalty in target["weighted"]
        )
    if isinstance(target, dict) and "scaled" in target:
        # iter-252: each element's distance is its raw distance TIMES its factor,
        # and the set scores as the MIN over those scaled distances — the
        # multiplicative twin of the weighted set. An exact hit stays free and the
        # cost grows with distance.
        return min(
            grid_cell_distance(cell, element) * factor
            for element, factor in target["scaled"]
        )
    if isinstance(target, dict) and "affine" in target:
        # iter-287: each element's distance is its raw distance SCALED by its factor
        # THEN OFFSET by its penalty (distance*factor + penalty), composing the
        # iter-252 multiplicative and iter-250 additive weights; the set scores as
        # the MIN over those affine distances. The penalty bites even an exact hit
        # (0*factor + penalty = penalty) and the factor grows the cost with distance.
        return min(
            grid_cell_distance(cell, element) * factor + penalty
            for element, factor, penalty in target["affine"]
        )
    if isinstance(target, dict):
        return min(grid_cell_distance(cell, element) for element in target["prefer"])
    if isinstance(target, list):
        return min(grid_cell_distance(cell, element) for element in target)
    if isinstance(target, tuple):
        lo, hi = target
        if lo is not None and count < lo:
            return lo - count
        if hi is not None and count > hi:
            return count - hi
        return 0
    return abs(count - target)


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

    iter-249: when ``target`` is a ``{"prefer": [...]}`` PREFERENCE form, the
    operator's stated precedence is the FIRST tie-break (stronger intent than
    grid position or recovered speech), inserted as a secondary key right after
    distance and before the ``tie_break`` key — so among cells at equal distance,
    the one nearest a more-preferred element (lower :func:`_preference_rank`)
    wins, and the ``tie_break`` (row-major/speech) only decides cells that ALSO
    tie on preference rank. For a non-preference target the key is byte-for-byte
    the iter-243 shape (no preference key inserted).

    iter-250/252/287: a ``{"weighted": ...}``, ``{"scaled": ...}``, or
    ``{"affine": ...}`` set inserts NO secondary key — its preference is already
    baked into :func:`grid_cell_distance` (the additive penalty / multiplicative
    factor / both composed), so cells at equal weighted distance are a genuine tie
    that the ``tie_break`` decides. Only the ``{"prefer": ...}`` dict gets the
    preference rank key.

    Pure — reads only ``cell["num_segments"]`` and, for ``"speech"`` ties,
    ``cell["speech_s"]``. Never touches torch.
    """
    distance = grid_cell_distance(cell, target)
    keys = [distance]
    if isinstance(target, dict) and "prefer" in target:
        keys.append(_preference_rank(cell, target["prefer"]))
    # iter-250: a {"weighted": ...} set needs NO secondary key — its preference is
    # already baked into the distance (the penalty), so equal penalised distance
    # is a genuine tie that the tie_break (row-major/speech) decides.
    if tie_break == "speech":
        keys.append(-cell["speech_s"])
    return tuple(keys)


def _preference_rank(cell, prefer):
    """Index of the ``prefer`` element a cell's count sits nearest — lower better.

    iter-249's tie-break input for the ``{"prefer": [...]}`` PREFERENCE target.
    Returns the index of the EARLIEST preference element achieving the minimum
    distance to the cell's ``num_segments``: a cell that satisfies element 0
    (distance 0) ranks ``0`` and beats one that only satisfies element 1
    (rank ``1``), so :func:`grid_cell_sort_key` leans the pick toward the
    operator's more-preferred count when several are equally close. Earliest-wins
    on a distance tie among elements (``list.index`` of the min), matching the
    earliest-tie rule everywhere else. Pure — reads only ``cell["num_segments"]``
    via :func:`grid_cell_distance`.
    """
    distances = [grid_cell_distance(cell, element) for element in prefer]
    return distances.index(min(distances))


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


def _signed_float3(x):
    """Format a float delta with an explicit sign, 3 decimal places.

    The gap-delta analogue of :func:`_signed_float` — the silence-gap surfaces
    (iter-328+) round to 3 places, so ``gv vad-gap-diff`` signs its deltas at the
    same precision (``+0.250`` / ``0.000`` / ``-0.125``).
    """
    return f"+{x:.3f}" if x > 0 else f"{x:.3f}"


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

    iter-315 added ``--csv``, the first machine-readable surface on this command:
    in ``--grid`` mode it emits the flat per-cell sweep table
    (:func:`render_grid_csv`) and in trajectory mode the per-turn speed curve
    (:func:`render_trajectory_csv`). iter-317 adds the nested ``--json`` twin
    (:func:`render_grid_json` / :func:`render_trajectory_json`), completing the
    human / ``--json`` / ``--csv`` trio the VAD-analysis surfaces already carry.
    ``--json`` and ``--csv`` are mutually exclusive at the parser.

    iter-318 adds ``--lurch-weight``, the first scoring knob on the grid sweep:
    it threads the parser value into :func:`pick_best_mirror_config` AND all
    three grid renderers so the displayed ``score`` column always reflects the
    weight the best pick was decided on. Trajectory mode has no score, so the
    knob is inert there.

    iter-319 adds ``--min-speed`` / ``--max-speed`` / ``--min-delta``, the
    intelligibility-band overrides. They thread into BOTH modes: in ``--grid``
    mode as the ``template`` whose band every cell clones (so a sweep can run
    against a non-seed band, e.g. a wider window for a faster voice), and in
    trajectory mode directly on the single config. The cross-edge ordering
    (``max_speed >= min_speed``) is validated by ``WpmMirrorConfig``; a bad pair
    is reported as a clean ``error:`` line rather than a traceback.
    """
    wm = _load_wpm_mirror()
    WpmMirrorConfig = wm.WpmMirrorConfig
    simulate_speed_trajectory = wm.simulate_speed_trajectory
    sweep_mirror_grid = wm.sweep_mirror_grid
    pick_best_mirror_config = wm.pick_best_mirror_config

    as_csv = getattr(args, "csv", False)
    as_json = getattr(args, "json", False)
    lurch_weight = getattr(args, "lurch_weight", None)

    # Band overrides (iter-319). getattr with the engine defaults keeps the
    # handler runnable from tests that build args without the new attributes.
    min_speed = getattr(args, "min_speed", wm.DEFAULT_MIN_SPEED)
    max_speed = getattr(args, "max_speed", wm.DEFAULT_MAX_SPEED)
    min_delta = getattr(args, "min_delta", wm.DEFAULT_MIN_DELTA)

    if args.grid:
        # The template carries only the shared band; sweep_mirror_grid overrides
        # base_wpm/strength per cell. A bad band (max < min) raises here.
        try:
            template = WpmMirrorConfig(
                min_speed=min_speed,
                max_speed=max_speed,
                min_delta=min_delta,
            )
        except ValueError as exc:
            log(f"error: {exc}")
            return
        points = sweep_mirror_grid(
            args.wpms,
            args.base_wpms,
            args.strengths,
            initial_speed=args.initial_speed,
            template=template,
        )
        if lurch_weight is None:
            best = pick_best_mirror_config(points)
        else:
            best = pick_best_mirror_config(points, lurch_weight)
        if as_json:
            log(render_grid_json(points, best, lurch_weight=lurch_weight))
        elif as_csv:
            log(render_grid_csv(points, best, lurch_weight=lurch_weight))
        else:
            for line in render_grid(points, best, lurch_weight=lurch_weight):
                log(line)
        return

    try:
        config = WpmMirrorConfig(
            enabled=True,
            base_wpm=args.base_wpm,
            strength=args.strength,
            min_speed=min_speed,
            max_speed=max_speed,
            min_delta=min_delta,
        )
    except ValueError as exc:
        log(f"error: {exc}")
        return
    traj = simulate_speed_trajectory(
        args.wpms,
        initial_speed=args.initial_speed,
        config=config,
    )
    if as_json:
        log(render_trajectory_json(traj, wpms=args.wpms))
    elif as_csv:
        log(render_trajectory_csv(traj, wpms=args.wpms))
    else:
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

    iter-316 added ``--csv``, the machine-readable twin: it emits the per-sample
    ``sample,words,audio_seconds,speed,bot_wpm,implied_base_wpm`` grid with the
    aggregate calibration trailing as ``#`` comment lines
    (:func:`render_calibration_csv`). iter-317 adds the nested ``--json`` twin
    (:func:`render_calibration_json`), completing the human / ``--json`` /
    ``--csv`` trio the VAD-analysis surfaces already carry. ``--json`` and
    ``--csv`` are mutually exclusive at the parser; either is the whole output in
    that mode — the ``--verdict`` adopt/keep DECISION is human prose, not a data
    record, so it is suppressed under both (a consumer scripts the re-seed call
    off the drift field itself).
    """
    wm = _load_wpm_mirror()
    CalibrationSample = wm.CalibrationSample
    calibrate_base_wpm = wm.calibrate_base_wpm

    samples = [
        CalibrationSample(words=int(words), audio_seconds=audio_seconds, speed=speed)
        for (words, audio_seconds, speed) in args.samples
    ]
    calib = calibrate_base_wpm(samples, default_base_wpm=args.nominal)
    if getattr(args, "json", False):
        log(render_calibration_json(samples, calib))
        return
    if getattr(args, "csv", False):
        log(render_calibration_csv(samples, calib))
        return
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

    iter-314 adds ``--csv`` (mutually exclusive with ``--json``), the
    per-segment spreadsheet/plot-friendly twin completing the human /
    ``--json`` / ``--csv`` trio that ``gv vad-sweep`` / ``gv vad-diff`` /
    ``gv vad-grid`` already carry.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)
    as_csv = getattr(args, "csv", False)

    if not availability():
        if as_json:
            log(render_vad_json(None))
        elif as_csv:
            log(render_vad_csv(None))
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
    elif as_csv:
        log(render_vad_csv(result, threshold=args.threshold))
    else:
        for line in render_vad_segments(result, threshold=args.threshold):
            log(line)


def cmd_vad_gaps(args, *, log=print, segmenter=None, availability=None):
    """Segment a WAV offline and report the inter-segment silence gaps.

    The silence-side complement of :func:`cmd_vad`: where ``gv vad`` reports the
    speech regions, ``gv vad-gaps recording.wav`` reports the pauses BETWEEN
    them — the gap distribution an operator reads to choose the end-of-turn
    hangover (``--min-silence-ms`` / the live ``chat.vad.silence_duration``).
    The shortest real pause is the floor above which raising the hangover starts
    merging two genuine turns into one.

    Same injected-dependency contract as :func:`cmd_vad` / :func:`cmd_vad_diff`:
    ``segmenter`` / ``availability`` default to the real :mod:`vad.silero`
    functions, imported lazily so the parser stays torch-free. The segmenter
    knobs are shared with ``gv vad`` so the gaps are measured against the same
    segmentation. ``--csv`` is mutually exclusive with ``--json``; when
    ``silero-vad`` is absent the handler prints the install hint and returns,
    never crashing.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)
    as_csv = getattr(args, "csv", False)

    if not availability():
        if as_json:
            log(render_vad_gaps_json(None))
        elif as_csv:
            log(render_vad_gaps_csv(None))
        else:
            for line in render_vad_gaps(None):
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
        log(render_vad_gaps_json(result))
    elif as_csv:
        log(render_vad_gaps_csv(result))
    else:
        for line in render_vad_gaps(result):
            log(line)


def cmd_vad_gap_percentiles(args, *, log=print, segmenter=None, availability=None):
    """Segment one WAV offline and report the inter-segment silence-gap PERCENTILES.

    iter-338's order-statistic complement of :func:`cmd_vad_gaps`. Where ``gv
    vad-gaps`` reports min/mean/max — each fragile to a single outlier pause —
    ``gv vad-gap-percentiles recording.wav`` reports robust percentiles
    (p50/p90/p99 by default, ``--percentiles`` to choose) of the pause
    distribution. The median is the typical pause; set the end-of-turn hangover
    (``--min-silence-ms`` / the live ``chat.vad.silence_duration``) comfortably
    below it to never merge a typical turn, and read p90 / p99 to size the long
    tail you are willing to wait through.

    Same injected-dependency contract as :func:`cmd_vad_gaps`: ``segmenter`` /
    ``availability`` default to the real :mod:`vad.silero` functions, imported
    lazily so the parser stays torch-free. The segmenter knobs are shared with
    ``gv vad`` so the gaps are measured against the same segmentation. ``--csv``
    is mutually exclusive with ``--json``; when ``silero-vad`` is absent the
    handler prints the install hint and returns, never crashing.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)
    as_csv = getattr(args, "csv", False)
    percentiles = args.percentiles

    if not availability():
        if as_json:
            log(render_vad_gap_percentiles_json(None, percentiles=percentiles))
        elif as_csv:
            log(render_vad_gap_percentiles_csv(None, percentiles=percentiles))
        else:
            for line in render_vad_gap_percentiles(None, percentiles=percentiles):
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
        log(render_vad_gap_percentiles_json(result, percentiles=percentiles))
    elif as_csv:
        log(render_vad_gap_percentiles_csv(result, percentiles=percentiles))
    else:
        for line in render_vad_gap_percentiles(result, percentiles=percentiles):
            log(line)


def cmd_vad_gap_cdf(args, *, log=print, segmenter=None, availability=None):
    """Segment one WAV offline and report the silence-gap merge-CDF at candidate cuts.

    iter-346's inverse-CDF complement of :func:`cmd_vad_gap_percentiles`. Where
    ``gv vad-gap-percentiles`` answers "what pause length is the p90?" (fraction →
    value), ``gv vad-gap-cdf recording.wav`` answers the operationally-direct
    opposite — "if I set the end-of-turn hangover (``--min-silence-ms`` / the live
    ``chat.vad.silence_duration``) to candidate cut ``c``, what FRACTION of the
    pauses would it MERGE?" (value → fraction). For each ``--cuts-ms`` value it
    reports how many gaps are shorter than that cut (and so would be swallowed as
    within-turn silence), turning the percentile table into a direct "this
    hangover merges X% of your pauses" answer.

    Same injected-dependency contract as :func:`cmd_vad_gap_percentiles`:
    ``segmenter`` / ``availability`` default to the real :mod:`vad.silero`
    functions, imported lazily so the parser stays torch-free. The segmenter knobs
    are shared with ``gv vad`` so the gaps are measured against the same
    segmentation. ``--csv`` is mutually exclusive with ``--json``; when
    ``silero-vad`` is absent the handler prints the install hint and returns,
    never crashing.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)
    as_csv = getattr(args, "csv", False)
    cuts_ms = args.cuts_ms

    if not availability():
        if as_json:
            log(render_vad_gap_cdf_json(None, cuts_ms=cuts_ms))
        elif as_csv:
            log(render_vad_gap_cdf_csv(None, cuts_ms=cuts_ms))
        else:
            for line in render_vad_gap_cdf(None, cuts_ms=cuts_ms):
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
        log(render_vad_gap_cdf_json(result, cuts_ms=cuts_ms))
    elif as_csv:
        log(render_vad_gap_cdf_csv(result, cuts_ms=cuts_ms))
    else:
        for line in render_vad_gap_cdf(result, cuts_ms=cuts_ms):
            log(line)


def cmd_vad_gap_recommend(args, *, log=print, segmenter=None, availability=None):
    """Segment one WAV offline and RECOMMEND an end-of-turn hangover number.

    iter-347's verdict surface — the natural consumer of the gap-analysis family.
    Where ``gv vad-gaps`` / ``gv vad-gap-hist`` / ``gv vad-gap-cdf`` /
    ``gv vad-gap-percentiles`` SHOW the operator the pause distribution and leave
    the "so what do I set ``--min-silence-ms`` to?" judgement to them,
    ``gv vad-gap-recommend recording.wav`` answers it directly: it finds the
    valley between the short within-turn pauses and the long between-turn pauses
    (the widest jump in the sorted gap distribution) and names a single
    recommended hangover sitting in that valley.

    Same injected-dependency contract as :func:`cmd_vad_gap_cdf`: ``segmenter`` /
    ``availability`` default to the real :mod:`vad.silero` functions, imported
    lazily so the parser stays torch-free. The segmenter knobs are shared with
    ``gv vad`` so the gaps are measured against the same segmentation. ``--bias``
    (iter-351) chooses where in the valley the recommended number sits
    (short/balanced/long); it defaults to ``balanced`` (the iter-347 behaviour).
    ``--csv`` is mutually exclusive with ``--json``; when ``silero-vad`` is absent
    the handler prints the install hint and returns, never crashing.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)
    as_csv = getattr(args, "csv", False)
    bias = getattr(args, "bias", DEFAULT_GAP_RECOMMEND_BIAS)

    if not availability():
        if as_json:
            log(render_vad_gap_recommend_json(None, bias=bias))
        elif as_csv:
            log(render_vad_gap_recommend_csv(None, bias=bias))
        else:
            for line in render_vad_gap_recommend(None, bias=bias):
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
        log(render_vad_gap_recommend_json(result, bias=bias))
    elif as_csv:
        log(render_vad_gap_recommend_csv(result, bias=bias))
    else:
        for line in render_vad_gap_recommend(result, bias=bias):
            log(line)


def cmd_vad_gap_recommend_sweep(args, *, log=print, segmenter=None, availability=None):
    """Segment one WAV offline and SWEEP the recommended hangover across biases.

    iter-352's companion of :func:`cmd_vad_gap_recommend`. Where
    ``gv vad-gap-recommend --bias {short,balanced,long}`` names ONE defensible
    end-of-turn hangover, ``gv vad-gap-recommend-sweep recording.wav`` names ALL
    THREE side by side, so the operator sees the whole spread of defensible
    numbers — and the short→long width of that spread — in one shot, without
    re-running the command per bias. The valley + merge accounting are invariant
    across biases, so they are reported once.

    Same injected-dependency contract as :func:`cmd_vad_gap_recommend`:
    ``segmenter`` / ``availability`` default to the real :mod:`vad.silero`
    functions, imported lazily so the parser stays torch-free. The segmenter
    knobs are shared with ``gv vad`` so the gaps are measured against the same
    segmentation. ``--csv`` is mutually exclusive with ``--json``; when
    ``silero-vad`` is absent the handler prints the install hint and returns,
    never crashing.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)
    as_csv = getattr(args, "csv", False)

    if not availability():
        if as_json:
            log(render_vad_gap_recommend_sweep_json(None))
        elif as_csv:
            log(render_vad_gap_recommend_sweep_csv(None))
        else:
            for line in render_vad_gap_recommend_sweep(None):
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
        log(render_vad_gap_recommend_sweep_json(result))
    elif as_csv:
        log(render_vad_gap_recommend_sweep_csv(result))
    else:
        for line in render_vad_gap_recommend_sweep(result):
            log(line)


def cmd_vad_gap_confidence(args, *, log=print, segmenter=None, availability=None):
    """Segment one WAV offline and GRADE the recommended-hangover confidence.

    iter-348's confidence surface — the companion of :func:`cmd_vad_gap_recommend`.
    Where ``gv vad-gap-recommend`` always names a number,
    ``gv vad-gap-confidence recording.wav`` grades how trustworthy that number is:
    it measures how dominant the recommendation's valley is (the widest jump in
    the sorted gap distribution) versus the total gap spread and the next-widest
    jump, and reports a ``strong`` / ``moderate`` / ``weak`` grade (or ``none``
    when the pauses are uniform and there is no valley to grade). A clean bimodal
    distribution grades strong; a smear of similar pauses grades weak — telling
    the operator whether to trust the recommendation or tune ``--min-silence-ms``
    by ear.

    Same injected-dependency contract as :func:`cmd_vad_gap_recommend`:
    ``segmenter`` / ``availability`` default to the real :mod:`vad.silero`
    functions, imported lazily so the parser stays torch-free. The segmenter knobs
    are shared with ``gv vad`` so the gaps are measured against the same
    segmentation. ``--csv`` is mutually exclusive with ``--json``; when
    ``silero-vad`` is absent the handler prints the install hint and returns,
    never crashing.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)
    as_csv = getattr(args, "csv", False)

    if not availability():
        if as_json:
            log(render_vad_gap_confidence_json(None))
        elif as_csv:
            log(render_vad_gap_confidence_csv(None))
        else:
            for line in render_vad_gap_confidence(None):
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
        log(render_vad_gap_confidence_json(result))
    elif as_csv:
        log(render_vad_gap_confidence_csv(result))
    else:
        for line in render_vad_gap_confidence(result):
            log(line)


def cmd_vad_gap_cost(args, *, log=print, segmenter=None, availability=None):
    """Segment one WAV offline and report the silence-gap MERGE COST CURVE.

    iter-349's derivative complement of :func:`cmd_vad_gap_cdf`. Where
    ``gv vad-gap-cdf`` reports the cumulative empirical CDF (at cut ``c``, what
    fraction of pauses have merged), ``gv vad-gap-cost recording.wav`` reports its
    DERIVATIVE — between consecutive ``--cuts-ms`` values, how many ADDITIONAL
    pauses get swallowed and at what rate per +100 ms of hangover. A high-rate
    band sits inside a pause cluster (expensive to raise the hangover there); a
    zero-rate band is an empty valley where raising the hangover costs nothing —
    exactly where ``gv vad-gap-recommend`` points. It is the marginal "what does
    the next +100 ms cost me?" view an operator tuning ``--min-silence-ms`` (the
    live ``chat.vad.silence_duration``) reasons in.

    Same injected-dependency contract as :func:`cmd_vad_gap_cdf`: ``segmenter`` /
    ``availability`` default to the real :mod:`vad.silero` functions, imported
    lazily so the parser stays torch-free. The segmenter knobs are shared with
    ``gv vad`` so the gaps are measured against the same segmentation. ``--csv``
    is mutually exclusive with ``--json``; when ``silero-vad`` is absent the
    handler prints the install hint and returns, never crashing.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)
    as_csv = getattr(args, "csv", False)
    cuts_ms = args.cuts_ms

    if not availability():
        if as_json:
            log(render_vad_gap_cost_json(None, cuts_ms=cuts_ms))
        elif as_csv:
            log(render_vad_gap_cost_csv(None, cuts_ms=cuts_ms))
        else:
            for line in render_vad_gap_cost(None, cuts_ms=cuts_ms):
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
        log(render_vad_gap_cost_json(result, cuts_ms=cuts_ms))
    elif as_csv:
        log(render_vad_gap_cost_csv(result, cuts_ms=cuts_ms))
    else:
        for line in render_vad_gap_cost(result, cuts_ms=cuts_ms):
            log(line)


def cmd_vad_gap_peak(args, *, log=print, segmenter=None, availability=None):
    """Segment one WAV offline and name the COSTLIEST band of the merge cost curve.

    iter-350's verdict complement of :func:`cmd_vad_gap_cost`. Where
    ``gv vad-gap-cost`` reports the full marginal cost curve (every band's rate per
    +100 ms of hangover), ``gv vad-gap-peak recording.wav`` names the single
    STEEPEST band — the densest pause cluster, the steepest part of the CDF, the
    most expensive place to raise ``--min-silence-ms`` (the live
    ``chat.vad.silence_duration``). It is the mirror of ``gv vad-gap-recommend``,
    which points at the cheapest valley: peak says where NOT to cut, recommend says
    where TO cut.

    Same injected-dependency contract as :func:`cmd_vad_gap_cost`: ``segmenter`` /
    ``availability`` default to the real :mod:`vad.silero` functions, imported
    lazily so the parser stays torch-free. The segmenter knobs are shared with
    ``gv vad`` so the gaps are measured against the same segmentation. ``--csv``
    is mutually exclusive with ``--json``; when ``silero-vad`` is absent the
    handler prints the install hint and returns, never crashing.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)
    as_csv = getattr(args, "csv", False)
    cuts_ms = args.cuts_ms
    top_n = getattr(args, "top_n", 1)
    min_rate = getattr(args, "min_rate", 0.0)
    min_rate_pct = getattr(args, "min_rate_pct", None)
    show_rate_dist = getattr(args, "show_rate_dist", False)
    rate_pcts = getattr(args, "rate_pcts", DEFAULT_BAND_RATE_PCTS)
    kw = dict(cuts_ms=cuts_ms, top_n=top_n, min_rate=min_rate, min_rate_pct=min_rate_pct)
    # iter-358: --show-rate-dist gates ONLY the human face (the JSON always carries
    # band_rate_dist for machine consumers; the CSV's verdict-row schema has no
    # place for the distribution and is unchanged).
    # iter-359: --rate-pcts drives the band_rate_dist percentile set for the human
    # AND json faces.
    # iter-363: the CSV face now also reads --rate-pcts — only to decide whether a
    # custom percentile set lists the active --min-rate-pct floor for its trailing
    # floor_percentile_listed #-comment (the row columns are still unchanged).
    human_kw = dict(kw, show_rate_dist=show_rate_dist, rate_pcts=rate_pcts)
    json_kw = dict(kw, rate_pcts=rate_pcts)
    csv_kw = dict(kw, rate_pcts=rate_pcts)

    if not availability():
        if as_json:
            log(render_vad_gap_peak_json(None, **json_kw))
        elif as_csv:
            log(render_vad_gap_peak_csv(None, **csv_kw))
        else:
            for line in render_vad_gap_peak(None, **human_kw):
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
        log(render_vad_gap_peak_json(result, **json_kw))
    elif as_csv:
        log(render_vad_gap_peak_csv(result, **csv_kw))
    else:
        for line in render_vad_gap_peak(result, **human_kw):
            log(line)


def cmd_vad_gap_histogram(args, *, log=print, segmenter=None, availability=None):
    """Segment one WAV offline and report the inter-segment silence-gap HISTOGRAM.

    iter-336's distribution-shape complement of :func:`cmd_vad_gaps`. Where
    ``gv vad-gaps`` reports the min/mean/max gap aggregates, ``gv vad-gap-hist
    recording.wav`` buckets the pauses into fixed-width bins (``--bin-width-s``)
    so the operator can SEE the distribution shape — a bimodal pattern (a
    short-pause mode plus a long-pause mode with a valley between) is the signal
    that there is a safe place to set the end-of-turn hangover
    (``--min-silence-ms`` / the live ``chat.vad.silence_duration``), and that
    shape is invisible in the three aggregate numbers alone.

    Same injected-dependency contract as :func:`cmd_vad_gaps`: ``segmenter`` /
    ``availability`` default to the real :mod:`vad.silero` functions, imported
    lazily so the parser stays torch-free. The segmenter knobs are shared with
    ``gv vad`` so the gaps are measured against the same segmentation. ``--csv``
    is mutually exclusive with ``--json``; when ``silero-vad`` is absent the
    handler prints the install hint and returns, never crashing.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)
    as_csv = getattr(args, "csv", False)
    bin_width_s = args.bin_width_s

    if not availability():
        if as_json:
            log(render_vad_gap_histogram_json(None, bin_width_s=bin_width_s))
        elif as_csv:
            log(render_vad_gap_histogram_csv(None, bin_width_s=bin_width_s))
        else:
            for line in render_vad_gap_histogram(None, bin_width_s=bin_width_s):
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
        log(render_vad_gap_histogram_json(result, bin_width_s=bin_width_s))
    elif as_csv:
        log(render_vad_gap_histogram_csv(result, bin_width_s=bin_width_s))
    else:
        for line in render_vad_gap_histogram(result, bin_width_s=bin_width_s):
            log(line)


def cmd_vad_gap_sweep(args, *, log=print, segmenter=None, availability=None):
    """Segment one WAV across a swept knob and tabulate the SILENCE-gap distribution.

    iter-330's gap-side analogue of :func:`cmd_vad_sweep`. Where ``gv vad-sweep``
    tabulates segment-count / speech-seconds vs a swept knob, ``gv
    vad-gap-sweep`` tabulates the inter-segment silence-gap distribution
    (segment count, gap count, min/mean/max gap) — the silence-side surface that
    completes the iter-328 ``gv vad-gaps`` family the way ``vad-sweep`` completes
    ``gv vad``. The headline is how the MIN gap moves as the knob tightens: the
    shortest real pause is the floor above which raising the end-of-turn
    hangover (``--min-silence-ms`` / the live ``chat.vad.silence_duration``)
    starts merging two genuine turns into one, so the value that lifts the min
    gap clear of a target hangover is the one that buys merge headroom.

    Shares the iter-256 five-axis sweep machinery of :func:`cmd_vad_sweep`: the
    default axis is the P(speech) gate (``--thresholds``); ``--min-silences`` /
    ``--min-speeches`` / ``--speech-pads`` / ``--max-speeches`` switch the swept
    dimension to the hangover / minimum-speech floor / region padding /
    force-split ceiling (seconds), the gate then held at the scalar
    ``--threshold``. The five axes are mutually exclusive; exactly one knob
    varies per run, every non-swept knob is shared across all runs. Unlike
    ``vad-sweep`` there is no ``--target`` pick block: the pick machinery scores
    on segment count, which the gap surface does not headline (its signal is the
    gap distribution, not a segment-count target).

    Same injected-dependency contract as :func:`cmd_vad_sweep`: ``segmenter`` /
    ``availability`` default to the real :mod:`vad.silero` functions, imported
    lazily so the parser stays torch-free. ``--csv`` is mutually exclusive with
    ``--json``; when ``silero-vad`` is absent the handler prints the install hint
    and returns, never crashing.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)
    as_csv = getattr(args, "csv", False)

    # Pick the swept axis, mirroring cmd_vad_sweep: --min-silences sweeps the
    # hangover, --min-speeches the minimum-speech floor, --speech-pads the
    # region padding, --max-speeches the force-split ceiling (seconds) — all
    # with the gate held at scalar --threshold; otherwise sweep --thresholds
    # (the default) with the other knobs held at their scalars. The parser
    # guarantees at most one of the five is set.
    min_silences = getattr(args, "min_silences", None)
    min_speeches = getattr(args, "min_speeches", None)
    speech_pads = getattr(args, "speech_pads", None)
    max_speeches = getattr(args, "max_speeches", None)
    if min_silences is not None:
        axis = "min_silence_ms"
        values = min_silences
    elif min_speeches is not None:
        axis = "min_speech_ms"
        values = min_speeches
    elif speech_pads is not None:
        axis = "speech_pad_ms"
        values = speech_pads
    elif max_speeches is not None:
        axis = "max_speech_s"
        values = max_speeches
    else:
        axis = "threshold"
        values = args.thresholds

    if not availability():
        if as_json:
            log(render_vad_gap_sweep_json([], [None], name=args.wav, axis=axis))
        elif as_csv:
            log(render_vad_gap_sweep_csv([], [None], name=args.wav, axis=axis))
        else:
            for line in render_vad_gap_sweep([], [None], name=args.wav, axis=axis):
                log(line)
        return

    from vad.silero import SileroParams

    def _seg(value):
        # The swept axis takes ``value``; every other dimension is held at its
        # scalar knob. Every non-swept knob is shared across all runs.
        threshold = value if axis == "threshold" else args.threshold
        min_silence_ms = value if axis == "min_silence_ms" else args.min_silence_ms
        min_speech_ms = value if axis == "min_speech_ms" else args.min_speech_ms
        speech_pad_ms = value if axis == "speech_pad_ms" else args.speech_pad_ms
        max_speech_s = value if axis == "max_speech_s" else args.max_speech_s
        params = SileroParams(
            threshold=threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_s=max_speech_s,
        )
        return segmenter(args.wav, params=params)

    results = [_seg(v) for v in values]
    # Use the segmenter's own name (basename) so the sweep matches `gv vad`'s
    # report; fall back to the raw path only if the sweep is empty.
    name = results[0].name if results else args.wav
    if as_json:
        log(render_vad_gap_sweep_json(values, results, name=name, axis=axis))
    elif as_csv:
        log(render_vad_gap_sweep_csv(values, results, name=name, axis=axis))
    else:
        for line in render_vad_gap_sweep(values, results, name=name, axis=axis):
            log(line)


def cmd_vad_gap_peak_sweep(args, *, log=print, segmenter=None, availability=None):
    """Segment one WAV across a swept knob and tabulate the COSTLIEST cost band.

    iter-364's peak-side analogue of :func:`cmd_vad_gap_sweep`. Where ``gv
    vad-gap-sweep`` tabulates the inter-segment silence-gap distribution (min/
    mean/max gap) vs a swept knob, ``gv vad-gap-peak-sweep`` tabulates the
    iter-350 cost PEAK — for each swept value it names the steepest cost-curve
    band (the densest pause cluster, the most expensive place to raise the
    end-of-turn hangover) and reports its ms range, merged pauses, and rate per
    +100 ms. The headline is how the peak rate MOVES as a SEGMENTER knob (e.g.
    the ``--min-speech-ms`` floor) tightens: a stricter floor gates marginal
    speech, merges adjacent regions, and reshapes the pause clusters, so the
    steepest band and its cost drift with the knob. Where ``vad-gap-sweep``
    watches the cheapest valley move (where it's safe to set the hangover), this
    watches the costliest cluster move (where NOT to push it through).

    Shares the iter-256 five-axis sweep machinery of :func:`cmd_vad_gap_sweep`:
    the default axis is the P(speech) gate (``--thresholds``); ``--min-silences``
    / ``--min-speeches`` / ``--speech-pads`` / ``--max-speeches`` switch the swept
    dimension to the hangover / minimum-speech floor / region padding /
    force-split ceiling (seconds), the gate then held at the scalar
    ``--threshold``. The five axes are mutually exclusive; exactly one knob varies
    per run, every non-swept knob is shared across all runs. ``--cuts-ms`` (the
    candidate hangover cuts defining the cost bands) is shared across every swept
    value so the bands are scanned on the same axis at each point.

    Same injected-dependency contract as :func:`cmd_vad_gap_sweep`: ``segmenter``
    / ``availability`` default to the real :mod:`vad.silero` functions, imported
    lazily so the parser stays torch-free. ``--csv`` is mutually exclusive with
    ``--json``; when ``silero-vad`` is absent the handler prints the install hint
    and returns, never crashing.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)
    as_csv = getattr(args, "csv", False)
    cuts_ms = args.cuts_ms

    # Pick the swept axis, mirroring cmd_vad_gap_sweep: --min-silences sweeps the
    # hangover, --min-speeches the minimum-speech floor, --speech-pads the region
    # padding, --max-speeches the force-split ceiling (seconds) — all with the
    # gate held at scalar --threshold; otherwise sweep --thresholds (the default).
    # The parser guarantees at most one of the five is set.
    min_silences = getattr(args, "min_silences", None)
    min_speeches = getattr(args, "min_speeches", None)
    speech_pads = getattr(args, "speech_pads", None)
    max_speeches = getattr(args, "max_speeches", None)
    if min_silences is not None:
        axis = "min_silence_ms"
        values = min_silences
    elif min_speeches is not None:
        axis = "min_speech_ms"
        values = min_speeches
    elif speech_pads is not None:
        axis = "speech_pad_ms"
        values = speech_pads
    elif max_speeches is not None:
        axis = "max_speech_s"
        values = max_speeches
    else:
        axis = "threshold"
        values = args.thresholds

    if not availability():
        if as_json:
            log(render_vad_gap_peak_sweep_json([], [None], name=args.wav, cuts_ms=cuts_ms, axis=axis))
        elif as_csv:
            log(render_vad_gap_peak_sweep_csv([], [None], name=args.wav, cuts_ms=cuts_ms, axis=axis))
        else:
            for line in render_vad_gap_peak_sweep([], [None], name=args.wav, cuts_ms=cuts_ms, axis=axis):
                log(line)
        return

    from vad.silero import SileroParams

    def _seg(value):
        # The swept axis takes ``value``; every other dimension is held at its
        # scalar knob. Every non-swept knob is shared across all runs.
        threshold = value if axis == "threshold" else args.threshold
        min_silence_ms = value if axis == "min_silence_ms" else args.min_silence_ms
        min_speech_ms = value if axis == "min_speech_ms" else args.min_speech_ms
        speech_pad_ms = value if axis == "speech_pad_ms" else args.speech_pad_ms
        max_speech_s = value if axis == "max_speech_s" else args.max_speech_s
        params = SileroParams(
            threshold=threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_s=max_speech_s,
        )
        return segmenter(args.wav, params=params)

    results = [_seg(v) for v in values]
    name = results[0].name if results else args.wav
    if as_json:
        log(render_vad_gap_peak_sweep_json(values, results, name=name, cuts_ms=cuts_ms, axis=axis))
    elif as_csv:
        log(render_vad_gap_peak_sweep_csv(values, results, name=name, cuts_ms=cuts_ms, axis=axis))
    else:
        for line in render_vad_gap_peak_sweep(values, results, name=name, cuts_ms=cuts_ms, axis=axis):
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

    iter-313 adds ``--csv`` (mutually exclusive with ``--json``), the
    spreadsheet/plot-friendly twin completing the human / ``--json`` / ``--csv``
    trio that ``gv vad-sweep`` / ``gv vad-grid`` already carry.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)
    as_csv = getattr(args, "csv", False)

    if not availability():
        if as_json:
            log(render_vad_diff_json(None, None, label_a=args.threshold_a,
                                     label_b=args.threshold_b))
        elif as_csv:
            log(render_vad_diff_csv(None, None, label_a=args.threshold_a,
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
    elif as_csv:
        log(render_vad_diff_csv(result_a, result_b, label_a=args.threshold_a,
                                label_b=args.threshold_b))
    else:
        for line in render_vad_diff(result_a, result_b, label_a=args.threshold_a,
                                    label_b=args.threshold_b):
            log(line)


def cmd_vad_gap_diff(args, *, log=print, segmenter=None, availability=None):
    """Segment one WAV under two thresholds and report how the SILENCE gaps shift.

    iter-334's gap-side analogue of :func:`cmd_vad_diff`, and the two-point
    degenerate of :func:`cmd_vad_gap_sweep`. Where ``gv vad-diff`` reports the
    segment-count / speech-seconds delta between two ``--threshold`` settings,
    ``gv vad-gap-diff recording.wav`` runs Silero twice (``--threshold-a`` then
    ``--threshold-b``, all other knobs shared) and reports how the inter-segment
    silence-gap distribution shifts — the min/mean/max gap and total silence.
    The headline is how the MIN gap moves: the shortest real pause is the floor
    above which raising the end-of-turn hangover (``--min-silence-ms`` / the live
    ``chat.vad.silence_duration``) starts merging two genuine turns into one, so
    a stricter gate that lifts the min gap clear of a target hangover is the one
    that buys merge headroom.

    Same injected-dependency contract as :func:`cmd_vad_diff`: ``segmenter`` /
    ``availability`` default to the real :mod:`vad.silero` functions, imported
    lazily so the parser stays torch-free. ``--csv`` is mutually exclusive with
    ``--json``; when ``silero-vad`` is absent the handler prints the install hint
    and returns, never crashing.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)
    as_csv = getattr(args, "csv", False)

    if not availability():
        if as_json:
            log(render_vad_gap_diff_json(None, None, label_a=args.threshold_a,
                                         label_b=args.threshold_b))
        elif as_csv:
            log(render_vad_gap_diff_csv(None, None, label_a=args.threshold_a,
                                        label_b=args.threshold_b))
        else:
            for line in render_vad_gap_diff(None, None, label_a=args.threshold_a,
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
        log(render_vad_gap_diff_json(result_a, result_b, label_a=args.threshold_a,
                                     label_b=args.threshold_b))
    elif as_csv:
        log(render_vad_gap_diff_csv(result_a, result_b, label_a=args.threshold_a,
                                    label_b=args.threshold_b))
    else:
        for line in render_vad_gap_diff(result_a, result_b, label_a=args.threshold_a,
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
    where short speech regions start getting dropped. iter-253 adds a FOURTH
    axis — the symmetric region padding. Passing ``--speech-pads 0,20,40,60``
    sweeps ``speech_pad_ms`` (the gate again held at scalar ``--threshold``), so
    an operator can find how much padding stops clipping the talker's onsets/
    tails before adjacent regions start merging. iter-256 adds a FIFTH axis —
    the force-split ceiling. Passing ``--max-speeches 5,10,20,inf`` sweeps
    ``max_speech_s`` (the gate again held at scalar ``--threshold``), so an
    operator can find where a long monologue starts getting chopped into more
    segments as the cap tightens (``inf`` anchors the no-cap baseline). Unlike
    the other four, this axis is measured in SECONDS, not ms — it reuses the
    iter-255 ``max_speech_list_type`` validator and seconds formatter. The five
    axes are mutually exclusive (the parser rejects passing more than one);
    exactly one knob varies per run.

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
    # minimum-speech floor, --speech-pads the symmetric region padding,
    # --max-speeches the force-split ceiling (seconds) — all with the gate held
    # at scalar --threshold; otherwise sweep --thresholds (the default iter-236
    # behaviour) with the other knobs held at their scalars. The parser
    # guarantees at most one of the five is set.
    min_silences = getattr(args, "min_silences", None)
    min_speeches = getattr(args, "min_speeches", None)
    speech_pads = getattr(args, "speech_pads", None)
    max_speeches = getattr(args, "max_speeches", None)
    if min_silences is not None:
        axis = "min_silence_ms"
        values = min_silences
    elif min_speeches is not None:
        axis = "min_speech_ms"
        values = min_speeches
    elif speech_pads is not None:
        axis = "speech_pad_ms"
        values = speech_pads
    elif max_speeches is not None:
        axis = "max_speech_s"
        values = max_speeches
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
        speech_pad_ms = value if axis == "speech_pad_ms" else args.speech_pad_ms
        max_speech_s = value if axis == "max_speech_s" else args.max_speech_s
        params = SileroParams(
            threshold=threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_s=max_speech_s,
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
    axis is ``--min-silences`` (the trailing-silence hangover, the default),
    ``--min-speeches`` (the minimum-speech floor), ``--speech-pads`` (the
    symmetric region padding, iter-254) — all millisecond knobs — or
    ``--max-speeches`` (the force-split ceiling, in SECONDS, iter-255) —
    mutually exclusive, exactly one column axis per run. Whichever knob is NOT
    the column axis is held fixed at its scalar (``--min-silence-ms`` /
    ``--min-speech-ms`` / ``--speech-pad-ms`` / ``--max-speech-s``), and every
    other knob is shared across all cells.

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

    # Rows are always the gate; the column axis is whichever list was passed
    # (--min-speeches → floor; --speech-pads → region padding; --max-speeches →
    # force-split ceiling, seconds; else --min-silences → hangover, the
    # default). The parser's mutex guarantees at most one column list is set.
    row_axis = "threshold"
    row_values = args.thresholds
    min_speeches = getattr(args, "min_speeches", None)
    speech_pads = getattr(args, "speech_pads", None)
    max_speeches = getattr(args, "max_speeches", None)
    if min_speeches is not None:
        col_axis = "min_speech_ms"
        col_values = min_speeches
    elif speech_pads is not None:
        col_axis = "speech_pad_ms"
        col_values = speech_pads
    elif max_speeches is not None:
        col_axis = "max_speech_s"
        col_values = max_speeches
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
        # is held at its scalar knob. Whichever ms knob (or the seconds ceiling)
        # is NOT the column axis is held fixed at its scalar.
        min_silence_ms = (
            col_value if col_axis == "min_silence_ms" else args.min_silence_ms
        )
        min_speech_ms = (
            col_value if col_axis == "min_speech_ms" else args.min_speech_ms
        )
        speech_pad_ms = (
            col_value if col_axis == "speech_pad_ms" else args.speech_pad_ms
        )
        max_speech_s = (
            col_value if col_axis == "max_speech_s" else args.max_speech_s
        )
        params = SileroParams(
            threshold=row_value,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_s=max_speech_s,
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


def cmd_vad_gap_grid(args, *, log=print, segmenter=None, availability=None):
    """Segment one WAV across a 2-D knob grid and tabulate the SILENCE-gap distribution.

    iter-332's gap-side analogue of :func:`cmd_vad_grid`, and the 2-D analogue
    of :func:`cmd_vad_gap_sweep`. Where ``gv vad-grid`` tabulates segment-count
    / speech-seconds per cell of a gate × column-knob grid, ``gv vad-gap-grid``
    tabulates the inter-segment silence-gap distribution (segment count, gap
    count, min/mean/max gap) per cell — the silence-side surface that completes
    the iter-330 ``gv vad-gap-sweep`` family the way ``vad-grid`` completes
    ``vad-sweep``. The headline is how the MIN gap moves across two knobs at
    once: the shortest real pause is the floor above which raising the
    end-of-turn hangover (``--min-silence-ms`` / the live
    ``chat.vad.silence_duration``) starts merging two genuine turns into one, so
    the cell that lifts the min gap clear of a target hangover buys merge
    headroom — read in two dimensions instead of one 1-D sweep at a time.

    Mirrors :func:`cmd_vad_grid`'s axis layout: the ROW axis is always the
    P(speech) gate (``--thresholds``); the COLUMN axis is ``--min-silences``
    (the trailing-silence hangover, the default), ``--min-speeches`` (the
    minimum-speech floor), ``--speech-pads`` (the symmetric region padding), or
    ``--max-speeches`` (the force-split ceiling, in SECONDS) — mutually
    exclusive, exactly one column axis per run. Whichever knob is NOT the column
    axis is held fixed at its scalar; every other knob is shared across all
    cells. Unlike ``vad-grid`` there is no ``--target`` pick block (the pick
    scores on segment count, which the gap surface does not headline).

    Same injected-dependency contract as :func:`cmd_vad_grid`: ``segmenter`` /
    ``availability`` default to the real :mod:`vad.silero` functions, imported
    lazily so the parser stays torch-free. ``--csv`` is mutually exclusive with
    ``--json``; when ``silero-vad`` is absent the handler prints the install
    hint and returns, never crashing.
    """
    if segmenter is None or availability is None:
        from vad.silero import segment_recording, silero_available

        segmenter = segment_recording if segmenter is None else segmenter
        availability = silero_available if availability is None else availability

    as_json = getattr(args, "json", False)
    as_csv = getattr(args, "csv", False)

    # Rows are always the gate; the column axis is whichever list was passed
    # (--min-speeches → floor; --speech-pads → region padding; --max-speeches →
    # force-split ceiling, seconds; else --min-silences → hangover, the
    # default). The parser's mutex guarantees at most one column list is set.
    row_axis = "threshold"
    row_values = args.thresholds
    min_speeches = getattr(args, "min_speeches", None)
    speech_pads = getattr(args, "speech_pads", None)
    max_speeches = getattr(args, "max_speeches", None)
    if min_speeches is not None:
        col_axis = "min_speech_ms"
        col_values = min_speeches
    elif speech_pads is not None:
        col_axis = "speech_pad_ms"
        col_values = speech_pads
    elif max_speeches is not None:
        col_axis = "max_speech_s"
        col_values = max_speeches
    else:
        col_axis = "min_silence_ms"
        col_values = args.min_silences

    if not availability():
        unavailable = [None]
        if as_json:
            log(
                render_vad_gap_grid_json(
                    [], [], unavailable, name=args.wav,
                    row_axis=row_axis, col_axis=col_axis,
                )
            )
        elif as_csv:
            log(
                render_vad_gap_grid_csv(
                    [], [], unavailable, name=args.wav,
                    row_axis=row_axis, col_axis=col_axis,
                )
            )
        else:
            for line in render_vad_gap_grid(
                [], [], unavailable, name=args.wav,
                row_axis=row_axis, col_axis=col_axis,
            ):
                log(line)
        return

    from vad.silero import SileroParams

    def _seg(row_value, col_value):
        # The two grid axes take (row_value, col_value); every other dimension
        # is held at its scalar knob. Whichever ms knob (or the seconds ceiling)
        # is NOT the column axis is held fixed at its scalar.
        min_silence_ms = (
            col_value if col_axis == "min_silence_ms" else args.min_silence_ms
        )
        min_speech_ms = (
            col_value if col_axis == "min_speech_ms" else args.min_speech_ms
        )
        speech_pad_ms = (
            col_value if col_axis == "speech_pad_ms" else args.speech_pad_ms
        )
        max_speech_s = (
            col_value if col_axis == "max_speech_s" else args.max_speech_s
        )
        params = SileroParams(
            threshold=row_value,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_s=max_speech_s,
        )
        return segmenter(args.wav, params=params)

    # Row-major: row 0's whole row of columns first, then row 1's, … — the same
    # order vad_gap_grid flattens into.
    results = [_seg(rv, cv) for rv in row_values for cv in col_values]
    name = results[0].name if results else args.wav
    if as_json:
        log(
            render_vad_gap_grid_json(
                row_values, col_values, results, name=name,
                row_axis=row_axis, col_axis=col_axis,
            )
        )
    elif as_csv:
        log(
            render_vad_gap_grid_csv(
                row_values, col_values, results, name=name,
                row_axis=row_axis, col_axis=col_axis,
            )
        )
    else:
        for line in render_vad_gap_grid(
            row_values, col_values, results, name=name,
            row_axis=row_axis, col_axis=col_axis,
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
    "vad-gaps": cmd_vad_gaps,
    "vad-gap-percentiles": cmd_vad_gap_percentiles,
    "vad-gap-cdf": cmd_vad_gap_cdf,
    "vad-gap-recommend": cmd_vad_gap_recommend,
    "vad-gap-recommend-sweep": cmd_vad_gap_recommend_sweep,
    "vad-gap-confidence": cmd_vad_gap_confidence,
    "vad-gap-cost": cmd_vad_gap_cost,
    "vad-gap-peak": cmd_vad_gap_peak,
    "vad-gap-hist": cmd_vad_gap_histogram,
    "vad-gap-sweep": cmd_vad_gap_sweep,
    "vad-gap-peak-sweep": cmd_vad_gap_peak_sweep,
    "vad-diff": cmd_vad_diff,
    "vad-gap-diff": cmd_vad_gap_diff,
    "vad-sweep": cmd_vad_sweep,
    "vad-grid": cmd_vad_grid,
    "vad-gap-grid": cmd_vad_gap_grid,
}

# Seed mirror tunables, mirrored as the CLI defaults so the simulator's
# fixed-rate report matches the live SpeedController's seed config. Imported
# lazily inside build_parser so the module stays importable even if the
# session package were unavailable; falls back to the documented constants.
_MIRROR_DEFAULT_BASE_WPM = 165.0
_MIRROR_DEFAULT_STRENGTH = 0.5
_MIRROR_DEFAULT_LURCH_WEIGHT = 0.5
_MIRROR_DEFAULT_MIN_SPEED = 0.8
_MIRROR_DEFAULT_MAX_SPEED = 1.3
_MIRROR_DEFAULT_MIN_DELTA = 0.05
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
        lurch_weight_default = wm.DEFAULT_LURCH_WEIGHT
        min_speed_default = wm.DEFAULT_MIN_SPEED
        max_speed_default = wm.DEFAULT_MAX_SPEED
        min_delta_default = wm.DEFAULT_MIN_DELTA
        calib_spread_max_default = wm.DEFAULT_CALIB_SPREAD_MAX
        calib_drift_min_default = wm.DEFAULT_CALIB_DRIFT_MIN
        calib_min_samples_default = wm.DEFAULT_CALIB_MIN_SAMPLES
    except Exception:  # pragma: no cover - defensive fallback
        base_wpm_default = _MIRROR_DEFAULT_BASE_WPM
        strength_default = _MIRROR_DEFAULT_STRENGTH
        lurch_weight_default = _MIRROR_DEFAULT_LURCH_WEIGHT
        min_speed_default = _MIRROR_DEFAULT_MIN_SPEED
        max_speed_default = _MIRROR_DEFAULT_MAX_SPEED
        min_delta_default = _MIRROR_DEFAULT_MIN_DELTA
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
    sim.add_argument(
        "--min-speed",
        type=positive_float_type,
        default=min_speed_default,
        dest="min_speed",
        help="Intelligibility-band floor on the resulting speed multiplier; the "
        "mirror never goes below this no matter how slow the user "
        f"(default: {min_speed_default}). Applies to BOTH trajectory and --grid "
        "mode (every cell shares the band)",
    )
    sim.add_argument(
        "--max-speed",
        type=positive_float_type,
        default=max_speed_default,
        dest="max_speed",
        help="Intelligibility-band ceiling on the resulting speed multiplier; the "
        "mirror never exceeds this no matter how fast the user "
        f"(default: {max_speed_default}). Must be >= --min-speed. Applies to BOTH "
        "trajectory and --grid mode",
    )
    sim.add_argument(
        "--min-delta",
        type=nonneg_float_type,
        default=min_delta_default,
        dest="min_delta",
        help="Deadband on the per-turn speed CHANGE: sub-threshold nudges are "
        f"dropped so the rate doesn't churn (default: {min_delta_default}; 0 "
        "disables). Applies to BOTH trajectory and --grid mode",
    )
    sim.add_argument(
        "--lurch-weight",
        type=float,
        default=lurch_weight_default,
        dest="lurch_weight",
        help="Grid score weight on the lurch term (max single-turn step) "
        "relative to the convergence term (|final_gap|); higher penalizes "
        f"jumpy speed changes more (default: {lurch_weight_default}). "
        "Affects --grid scoring and the best pick only; ignored in trajectory "
        "mode (no score)",
    )
    sim_fmt = sim.add_mutually_exclusive_group()
    sim_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit nested machine-readable JSON instead of the human report — a "
        "cells+best object in --grid mode, a turns+diagnostics object in "
        "trajectory mode; mutually exclusive with --csv",
    )
    sim_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit a flat CSV table instead of the human report — per-cell in "
        "--grid mode, per-turn (speed curve) in trajectory mode; mutually "
        "exclusive with --json",
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
    calib_fmt = calib.add_mutually_exclusive_group()
    calib_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit nested JSON (samples list + calibration object) instead of "
        "the human report, for programmatic consumers; mutually exclusive with "
        "--csv (both suppress the human report and --verdict)",
    )
    calib_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit per-sample CSV (sample,words,audio_seconds,speed,bot_wpm,"
        "implied_base_wpm) with the aggregate calibration trailing as # comment "
        "lines, for spreadsheets/plots (suppresses the human report and --verdict)",
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
    vad_fmt = vad.add_mutually_exclusive_group()
    vad_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable report "
        "(mirrors fixtures/replay_silero.py --json / SileroResult.to_dict)",
    )
    vad_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit a flat index,start_s,end_s,duration_s CSV table (one row per "
        "detected segment) for spreadsheets/plots; mutually exclusive with --json",
    )

    # gv vad-gaps — segment one WAV and report the inter-segment SILENCE gaps.
    # The silence-side complement of `gv vad`: where vad reports the speech
    # regions, vad-gaps reports the pauses between them — the gap distribution
    # an operator reads to choose the end-of-turn hangover (--min-silence-ms).
    # Shares all the segmenter knobs with `gv vad` so the gaps are measured
    # against the same segmentation.
    vad_gaps = sub.add_parser(
        "vad-gaps",
        help="Offline Silero VAD — segment a WAV and report the silence gaps "
        "between speech regions (tune the end-of-turn hangover / min-silence)",
    )
    vad_gaps.add_argument(
        "wav",
        help="Path to a 16-bit PCM WAV file to segment",
    )
    vad_gaps.add_argument(
        "--threshold",
        type=unit_interval_type,
        default=vad_threshold_default,
        help=f"P(speech) gate in [0, 1] (default: {vad_threshold_default})",
    )
    vad_gaps.add_argument(
        "--min-speech-ms",
        type=nonneg_float_type,
        default=vad_min_speech_default,
        dest="min_speech_ms",
        help="Drop speech regions shorter than this, in ms "
        f"(default: {vad_min_speech_default})",
    )
    vad_gaps.add_argument(
        "--min-silence-ms",
        type=nonneg_float_type,
        default=vad_min_silence_default,
        dest="min_silence_ms",
        help="Trailing silence before a region ends, in ms — matches the "
        f"pipecat stop_secs=0.8 live default (default: {vad_min_silence_default})",
    )
    vad_gaps.add_argument(
        "--speech-pad-ms",
        type=nonneg_float_type,
        default=vad_speech_pad_default,
        dest="speech_pad_ms",
        help="Symmetric padding added to each region, in ms "
        f"(default: {vad_speech_pad_default})",
    )
    vad_gaps.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds; 'inf'/'none' "
        "never splits (default: inf)",
    )
    vad_gaps_fmt = vad_gaps.add_mutually_exclusive_group()
    vad_gaps_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON (aggregate gap stats + per-gap list) "
        "instead of the human-readable report",
    )
    vad_gaps_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit a flat index,after_segment,after_segment_end_s,gap_s CSV "
        "table (one row per gap) for spreadsheets/plots; mutually exclusive "
        "with --json",
    )

    # gv vad-gap-percentiles — segment one WAV and report ROBUST percentiles of
    # the inter-segment silence gaps (iter-338). The order-statistic complement
    # of vad-gaps: where vad-gaps reports min/mean/max (each fragile to a single
    # outlier pause), vad-gap-percentiles reports p50/p90/p99 (--percentiles to
    # choose) — the median is the typical pause, so set the end-of-turn hangover
    # (--min-silence-ms) below it to never merge a typical turn. Shares all
    # segmenter knobs with `gv vad`.
    vad_gap_pct = sub.add_parser(
        "vad-gap-percentiles",
        help="Offline Silero VAD — segment a WAV and report robust percentiles "
        "(p50/p90/p99) of the silence gaps between speech regions "
        "(outlier-robust pause stats, unlike min/mean/max)",
    )
    vad_gap_pct.add_argument(
        "wav",
        help="Path to a 16-bit PCM WAV file to segment",
    )
    vad_gap_pct.add_argument(
        "--percentiles",
        type=percentile_list_type,
        default=list(DEFAULT_GAP_PERCENTILES),
        help="Comma-separated percentiles in (0, 100] to report, e.g. "
        "'50,90,99' (default: 50,90,99)",
    )
    vad_gap_pct.add_argument(
        "--threshold",
        type=unit_interval_type,
        default=vad_threshold_default,
        help=f"P(speech) gate in [0, 1] (default: {vad_threshold_default})",
    )
    vad_gap_pct.add_argument(
        "--min-speech-ms",
        type=nonneg_float_type,
        default=vad_min_speech_default,
        dest="min_speech_ms",
        help="Drop speech regions shorter than this, in ms "
        f"(default: {vad_min_speech_default})",
    )
    vad_gap_pct.add_argument(
        "--min-silence-ms",
        type=nonneg_float_type,
        default=vad_min_silence_default,
        dest="min_silence_ms",
        help="Trailing silence before a region ends, in ms — matches the "
        f"pipecat stop_secs=0.8 live default (default: {vad_min_silence_default})",
    )
    vad_gap_pct.add_argument(
        "--speech-pad-ms",
        type=nonneg_float_type,
        default=vad_speech_pad_default,
        dest="speech_pad_ms",
        help="Symmetric padding added to each region, in ms "
        f"(default: {vad_speech_pad_default})",
    )
    vad_gap_pct.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds; 'inf'/'none' "
        "never splits (default: inf)",
    )
    vad_gap_pct_fmt = vad_gap_pct.add_mutually_exclusive_group()
    vad_gap_pct_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON (aggregate stats + per-percentile "
        "list) instead of the human-readable report",
    )
    vad_gap_pct_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit a flat percentile,value_s CSV table (one row per percentile) "
        "for spreadsheets/plots; mutually exclusive with --json",
    )

    # gv vad-gap-cdf — segment one WAV and evaluate the empirical CDF of the
    # inter-segment silence gaps at candidate end-of-turn hangover cuts (iter-346).
    # The INVERSE of vad-gap-percentiles: percentiles answer "what pause length is
    # the p90?" (fraction → value), the merge-CDF answers "if I set --min-silence-ms
    # to cut c, what fraction of pauses would it MERGE (pauses shorter than c)?"
    # (value → fraction) — a direct "this hangover merges X% of your pauses"
    # answer. Shares all segmenter knobs with `gv vad`.
    vad_gap_cdf = sub.add_parser(
        "vad-gap-cdf",
        help="Offline Silero VAD — segment a WAV and evaluate the empirical CDF "
        "of the silence gaps at candidate --min-silence-ms cuts (what fraction "
        "of pauses each hangover would merge)",
    )
    vad_gap_cdf.add_argument(
        "wav",
        help="Path to a 16-bit PCM WAV file to segment",
    )
    vad_gap_cdf.add_argument(
        "--cuts-ms",
        type=cut_ms_list_type,
        default=list(DEFAULT_GAP_CDF_CUTS_MS),
        dest="cuts_ms",
        help="Comma-separated candidate hangover cuts in ms to evaluate, e.g. "
        "'200,400,800,1600' (default: 200,400,800,1600)",
    )
    vad_gap_cdf.add_argument(
        "--threshold",
        type=unit_interval_type,
        default=vad_threshold_default,
        help=f"P(speech) gate in [0, 1] (default: {vad_threshold_default})",
    )
    vad_gap_cdf.add_argument(
        "--min-speech-ms",
        type=nonneg_float_type,
        default=vad_min_speech_default,
        dest="min_speech_ms",
        help="Drop speech regions shorter than this, in ms "
        f"(default: {vad_min_speech_default})",
    )
    vad_gap_cdf.add_argument(
        "--min-silence-ms",
        type=nonneg_float_type,
        default=vad_min_silence_default,
        dest="min_silence_ms",
        help="Trailing silence before a region ends, in ms — matches the "
        f"pipecat stop_secs=0.8 live default (default: {vad_min_silence_default})",
    )
    vad_gap_cdf.add_argument(
        "--speech-pad-ms",
        type=nonneg_float_type,
        default=vad_speech_pad_default,
        dest="speech_pad_ms",
        help="Symmetric padding added to each region, in ms "
        f"(default: {vad_speech_pad_default})",
    )
    vad_gap_cdf.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds; 'inf'/'none' "
        "never splits (default: inf)",
    )
    vad_gap_cdf_fmt = vad_gap_cdf.add_mutually_exclusive_group()
    vad_gap_cdf_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON (aggregate stats + per-cut merge "
        "fractions) instead of the human-readable report",
    )
    vad_gap_cdf_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit a flat cut_ms,cut_s,merged,merge_fraction CSV table (one row "
        "per cut) for spreadsheets/plots; mutually exclusive with --json",
    )

    # gv vad-gap-cost — segment one WAV and report the merge COST CURVE (iter-349).
    # The DERIVATIVE of vad-gap-cdf: where the CDF gives the cumulative fraction of
    # pauses merged by each cut, the cost curve gives the marginal view — between
    # consecutive --cuts-ms values, how many ADDITIONAL pauses merge and at what
    # rate per +100 ms of hangover. A zero-rate band is an empty valley (raising
    # the hangover there costs nothing — where vad-gap-recommend points); a
    # high-rate band sits inside a pause cluster. Shares all segmenter knobs.
    vad_gap_cost = sub.add_parser(
        "vad-gap-cost",
        help="Offline Silero VAD — segment a WAV and report the merge cost curve "
        "(the derivative of vad-gap-cdf): how many additional pauses each band "
        "between --cuts-ms merges, and the rate per +100 ms of hangover",
    )
    vad_gap_cost.add_argument(
        "wav",
        help="Path to a 16-bit PCM WAV file to segment",
    )
    vad_gap_cost.add_argument(
        "--cuts-ms",
        type=cut_ms_list_type,
        default=list(DEFAULT_GAP_CDF_CUTS_MS),
        dest="cuts_ms",
        help="Comma-separated candidate hangover cuts in ms defining the cost "
        "bands, e.g. '200,400,800,1600' (sorted + de-duplicated; default: "
        "200,400,800,1600)",
    )
    vad_gap_cost.add_argument(
        "--threshold",
        type=unit_interval_type,
        default=vad_threshold_default,
        help=f"P(speech) gate in [0, 1] (default: {vad_threshold_default})",
    )
    vad_gap_cost.add_argument(
        "--min-speech-ms",
        type=nonneg_float_type,
        default=vad_min_speech_default,
        dest="min_speech_ms",
        help="Drop speech regions shorter than this, in ms "
        f"(default: {vad_min_speech_default})",
    )
    vad_gap_cost.add_argument(
        "--min-silence-ms",
        type=nonneg_float_type,
        default=vad_min_silence_default,
        dest="min_silence_ms",
        help="Trailing silence before a region ends, in ms — matches the "
        f"pipecat stop_secs=0.8 live default (default: {vad_min_silence_default})",
    )
    vad_gap_cost.add_argument(
        "--speech-pad-ms",
        type=nonneg_float_type,
        default=vad_speech_pad_default,
        dest="speech_pad_ms",
        help="Symmetric padding added to each region, in ms "
        f"(default: {vad_speech_pad_default})",
    )
    vad_gap_cost.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds; 'inf'/'none' "
        "never splits (default: inf)",
    )
    vad_gap_cost_fmt = vad_gap_cost.add_mutually_exclusive_group()
    vad_gap_cost_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON (aggregate stats + per-band marginal "
        "merge counts and rates) instead of the human-readable report",
    )
    vad_gap_cost_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit a flat from_ms,to_ms,width_ms,merged_added,merged_cumulative,"
        "rate_per_100ms CSV table (one row per band) for spreadsheets/plots; "
        "mutually exclusive with --json",
    )

    # gv vad-gap-peak — segment one WAV and NAME the costliest band of the merge
    # cost curve (iter-350). The verdict companion of vad-gap-cost: where the cost
    # curve gives the marginal rate per +100 ms for every band, vad-gap-peak names
    # the single STEEPEST band — the densest pause cluster, the steepest part of
    # the CDF, the most expensive place to raise --min-silence-ms. It is the mirror
    # of vad-gap-recommend (which points at the cheapest zero-rate valley): peak
    # says where NOT to cut, recommend says where TO cut. Shares all `gv vad` knobs.
    vad_gap_peak = sub.add_parser(
        "vad-gap-peak",
        help="Offline Silero VAD — segment a WAV and name the costliest band of "
        "the merge cost curve (the densest pause cluster / steepest part of the "
        "CDF — the most expensive place to raise --min-silence-ms)",
    )
    vad_gap_peak.add_argument(
        "wav",
        help="Path to a 16-bit PCM WAV file to segment",
    )
    vad_gap_peak.add_argument(
        "--cuts-ms",
        type=cut_ms_list_type,
        default=list(DEFAULT_GAP_CDF_CUTS_MS),
        dest="cuts_ms",
        help="Comma-separated candidate hangover cuts in ms defining the cost "
        "bands scanned for the peak, e.g. '200,400,800,1600' (sorted + "
        "de-duplicated; default: 200,400,800,1600)",
    )
    vad_gap_peak.add_argument(
        "--top-n",
        type=positive_int_type,
        default=1,
        dest="top_n",
        help="Name the N steepest cost bands instead of just the single peak "
        "(ranked by descending rate, earliest band first on a tie; only "
        "non-empty bands count, so fewer than N may appear). Default: 1",
    )
    # The rate floor can be set two ways — an ABSOLUTE rate (--min-rate, iter-355)
    # or a PERCENTILE of the observed band rates (--min-rate-pct, iter-357, which
    # adapts to the recording's own cost scale). They set the same knob, so they
    # are mutually exclusive (the core also raises ValueError if both are given).
    vad_gap_peak_floor = vad_gap_peak.add_mutually_exclusive_group()
    vad_gap_peak_floor.add_argument(
        "--min-rate",
        type=nonneg_float_type,
        default=0.0,
        dest="min_rate",
        help="Drop cost bands cheaper than this ABSOLUTE marginal rate (pauses "
        "merged per +100ms of hangover) before ranking — only the bands worth "
        "worrying about are named. Pairs with --top-n; mutually exclusive with "
        "--min-rate-pct. Default: 0.0 (keep every non-empty band)",
    )
    vad_gap_peak_floor.add_argument(
        "--min-rate-pct",
        type=percentile_type,
        default=None,
        dest="min_rate_pct",
        help="Drop cost bands cheaper than the Pth PERCENTILE of the observed "
        "non-empty band rates before ranking — an adaptive floor that scales to "
        "this recording's cost distribution (e.g. 75 names the top quartile of "
        "cost peaks). In (0, 100]; mutually exclusive with --min-rate. Default: "
        "unset (use --min-rate)",
    )
    vad_gap_peak.add_argument(
        "--show-rate-dist",
        action="store_true",
        dest="show_rate_dist",
        help="Append the observed non-empty band-rate distribution (count, "
        "min/mean/max, and the p50/p75/p90/p99 percentiles) to the human report "
        "— the exact sample --min-rate-pct reads against, so you can see where a "
        "chosen percentile floor will land. The --json face always carries this "
        "as 'band_rate_dist'. Default: off",
    )
    vad_gap_peak.add_argument(
        "--rate-pcts",
        type=percentile_list_type,
        default=list(DEFAULT_BAND_RATE_PCTS),
        dest="rate_pcts",
        help="Comma-separated percentiles to summarise the observed band-rate "
        "distribution at, e.g. '50,90,99' (each in (0, 100]; order preserved). "
        "Drives both the --show-rate-dist block and the --json 'band_rate_dist' "
        "percentiles, so an operator can ask for arbitrary quantiles instead of "
        "the default 50,75,90,99",
    )
    vad_gap_peak.add_argument(
        "--threshold",
        type=unit_interval_type,
        default=vad_threshold_default,
        help=f"P(speech) gate in [0, 1] (default: {vad_threshold_default})",
    )
    vad_gap_peak.add_argument(
        "--min-speech-ms",
        type=nonneg_float_type,
        default=vad_min_speech_default,
        dest="min_speech_ms",
        help="Drop speech regions shorter than this, in ms "
        f"(default: {vad_min_speech_default})",
    )
    vad_gap_peak.add_argument(
        "--min-silence-ms",
        type=nonneg_float_type,
        default=vad_min_silence_default,
        dest="min_silence_ms",
        help="Trailing silence before a region ends, in ms — matches the "
        f"pipecat stop_secs=0.8 live default (default: {vad_min_silence_default})",
    )
    vad_gap_peak.add_argument(
        "--speech-pad-ms",
        type=nonneg_float_type,
        default=vad_speech_pad_default,
        dest="speech_pad_ms",
        help="Symmetric padding added to each region, in ms "
        f"(default: {vad_speech_pad_default})",
    )
    vad_gap_peak.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds; 'inf'/'none' "
        "never splits (default: inf)",
    )
    vad_gap_peak_fmt = vad_gap_peak.add_mutually_exclusive_group()
    vad_gap_peak_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON (aggregate stats + peak-band fields) "
        "instead of the human-readable verdict",
    )
    vad_gap_peak_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit a rank,peak_found,peak_from_ms,peak_to_ms,peak_width_ms,"
        "peak_merged_added,peak_rate_per_100ms CSV summary (one row per ranked "
        "peak when --top-n > 1); mutually exclusive with --json",
    )

    # gv vad-gap-recommend — segment one WAV and RECOMMEND an end-of-turn hangover
    # number (iter-347). The verdict surface / natural consumer of the gap-analysis
    # family: where vad-gaps/vad-gap-hist/vad-gap-cdf/vad-gap-percentiles SHOW the
    # pause distribution, vad-gap-recommend finds the valley between short
    # within-turn pauses and long between-turn pauses (the widest jump in the
    # sorted gaps) and names a single recommended --min-silence-ms sitting in it.
    # Shares all segmenter knobs with `gv vad`.
    vad_gap_rec = sub.add_parser(
        "vad-gap-recommend",
        help="Offline Silero VAD — segment a WAV and recommend an end-of-turn "
        "--min-silence-ms by finding the valley between short within-turn "
        "pauses and long between-turn pauses (names the number, not just the "
        "distribution)",
    )
    vad_gap_rec.add_argument(
        "wav",
        help="Path to a 16-bit PCM WAV file to segment",
    )
    vad_gap_rec.add_argument(
        "--threshold",
        type=unit_interval_type,
        default=vad_threshold_default,
        help=f"P(speech) gate in [0, 1] (default: {vad_threshold_default})",
    )
    vad_gap_rec.add_argument(
        "--min-speech-ms",
        type=nonneg_float_type,
        default=vad_min_speech_default,
        dest="min_speech_ms",
        help="Drop speech regions shorter than this, in ms "
        f"(default: {vad_min_speech_default})",
    )
    vad_gap_rec.add_argument(
        "--min-silence-ms",
        type=nonneg_float_type,
        default=vad_min_silence_default,
        dest="min_silence_ms",
        help="Trailing silence before a region ends, in ms — matches the "
        f"pipecat stop_secs=0.8 live default (default: {vad_min_silence_default})",
    )
    vad_gap_rec.add_argument(
        "--speech-pad-ms",
        type=nonneg_float_type,
        default=vad_speech_pad_default,
        dest="speech_pad_ms",
        help="Symmetric padding added to each region, in ms "
        f"(default: {vad_speech_pad_default})",
    )
    vad_gap_rec.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds; 'inf'/'none' "
        "never splits (default: inf)",
    )
    vad_gap_rec.add_argument(
        "--bias",
        choices=sorted(GAP_RECOMMEND_BIAS_FRACTIONS),
        default=DEFAULT_GAP_RECOMMEND_BIAS,
        help="Where in the valley the recommended hangover sits: 'short' biases "
        "toward the short cluster (smaller, eager hangover — ends turns faster), "
        "'long' toward the long cluster (larger, patient hangover — tolerates "
        "mid-turn pauses), 'balanced' is the midpoint "
        f"(default: {DEFAULT_GAP_RECOMMEND_BIAS}). The below/keeps split is "
        "unchanged by the bias — only the named number shifts",
    )
    vad_gap_rec_fmt = vad_gap_rec.add_mutually_exclusive_group()
    vad_gap_rec_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON (aggregate stats + recommendation "
        "fields) instead of the human-readable verdict",
    )
    vad_gap_rec_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit a one-row bias,recommended_ms,recommended_s,split_found,below,"
        "at_or_above,num_gaps CSV summary; mutually exclusive with --json",
    )

    # gv vad-gap-recommend-sweep — segment one WAV and SWEEP the recommended
    # hangover across all three biases at once (iter-352). The companion of
    # iter-351's `--bias` knob: where `gv vad-gap-recommend --bias X` names ONE
    # defensible number, this names short/balanced/long side by side plus the
    # short→long spread, so the operator sees the whole range of defensible
    # numbers in one shot. The valley + merge accounting are invariant across
    # biases (only the named number shifts), so they are reported once. Shares
    # all `gv vad` segmenter knobs.
    vad_gap_rec_sweep = sub.add_parser(
        "vad-gap-recommend-sweep",
        help="Offline Silero VAD — segment a WAV and sweep the recommended "
        "end-of-turn --min-silence-ms across all three biases (short/balanced/"
        "long) side by side, with the short→long spread (the whole range of "
        "defensible numbers at once)",
    )
    vad_gap_rec_sweep.add_argument(
        "wav",
        help="Path to a 16-bit PCM WAV file to segment",
    )
    vad_gap_rec_sweep.add_argument(
        "--threshold",
        type=unit_interval_type,
        default=vad_threshold_default,
        help=f"P(speech) gate in [0, 1] (default: {vad_threshold_default})",
    )
    vad_gap_rec_sweep.add_argument(
        "--min-speech-ms",
        type=nonneg_float_type,
        default=vad_min_speech_default,
        dest="min_speech_ms",
        help="Drop speech regions shorter than this, in ms "
        f"(default: {vad_min_speech_default})",
    )
    vad_gap_rec_sweep.add_argument(
        "--min-silence-ms",
        type=nonneg_float_type,
        default=vad_min_silence_default,
        dest="min_silence_ms",
        help="Trailing silence before a region ends, in ms — matches the "
        f"pipecat stop_secs=0.8 live default (default: {vad_min_silence_default})",
    )
    vad_gap_rec_sweep.add_argument(
        "--speech-pad-ms",
        type=nonneg_float_type,
        default=vad_speech_pad_default,
        dest="speech_pad_ms",
        help="Symmetric padding added to each region, in ms "
        f"(default: {vad_speech_pad_default})",
    )
    vad_gap_rec_sweep.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds; 'inf'/'none' "
        "never splits (default: inf)",
    )
    vad_gap_rec_sweep_fmt = vad_gap_rec_sweep.add_mutually_exclusive_group()
    vad_gap_rec_sweep_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON (aggregate stats + shared valley + "
        "per-bias recommendations + spread) instead of the human-readable sweep",
    )
    vad_gap_rec_sweep_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit a three-row (one per bias) bias,recommended_ms,recommended_s,"
        "split_found,below,at_or_above,num_gaps CSV; mutually exclusive with --json",
    )

    # gv vad-gap-confidence — segment one WAV and GRADE how trustworthy the
    # vad-gap-recommend verdict is (iter-348). The companion of vad-gap-recommend:
    # the verdict always names a number, but the number is only as good as the
    # valley it sits in. This grades how dominant that valley is (its width vs the
    # total gap spread and the next-widest jump) into strong/moderate/weak — a
    # clean bimodal distribution grades strong, a smear of similar pauses grades
    # weak — so the operator knows whether to trust it. Shares all `gv vad` knobs.
    vad_gap_conf = sub.add_parser(
        "vad-gap-confidence",
        help="Offline Silero VAD — segment a WAV and grade how trustworthy the "
        "vad-gap-recommend hangover is (strong/moderate/weak by how dominant "
        "the valley is vs the total gap spread)",
    )
    vad_gap_conf.add_argument(
        "wav",
        help="Path to a 16-bit PCM WAV file to segment",
    )
    vad_gap_conf.add_argument(
        "--threshold",
        type=unit_interval_type,
        default=vad_threshold_default,
        help=f"P(speech) gate in [0, 1] (default: {vad_threshold_default})",
    )
    vad_gap_conf.add_argument(
        "--min-speech-ms",
        type=nonneg_float_type,
        default=vad_min_speech_default,
        dest="min_speech_ms",
        help="Drop speech regions shorter than this, in ms "
        f"(default: {vad_min_speech_default})",
    )
    vad_gap_conf.add_argument(
        "--min-silence-ms",
        type=nonneg_float_type,
        default=vad_min_silence_default,
        dest="min_silence_ms",
        help="Trailing silence before a region ends, in ms — matches the "
        f"pipecat stop_secs=0.8 live default (default: {vad_min_silence_default})",
    )
    vad_gap_conf.add_argument(
        "--speech-pad-ms",
        type=nonneg_float_type,
        default=vad_speech_pad_default,
        dest="speech_pad_ms",
        help="Symmetric padding added to each region, in ms "
        f"(default: {vad_speech_pad_default})",
    )
    vad_gap_conf.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds; 'inf'/'none' "
        "never splits (default: inf)",
    )
    vad_gap_conf_fmt = vad_gap_conf.add_mutually_exclusive_group()
    vad_gap_conf_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON (aggregate stats + recommendation + "
        "confidence fields) instead of the human-readable verdict",
    )
    vad_gap_conf_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit a one-row recommended_ms,grade,dominance,separation_ratio,"
        "valley_width_s,spread_s CSV summary; mutually exclusive with --json",
    )

    # gv vad-gap-hist — segment one WAV and HISTOGRAM the inter-segment silence
    # gaps into fixed-width bins (iter-336). The distribution-shape complement of
    # vad-gaps: where vad-gaps reports the min/mean/max aggregates, vad-gap-hist
    # shows the full pause-length distribution, so a bimodal pattern (short
    # within-turn pauses + long between-turn pauses, with a valley between) is
    # visible — that valley is the safe place to set the end-of-turn hangover
    # (--min-silence-ms). Shares all segmenter knobs with `gv vad`.
    vad_gap_hist = sub.add_parser(
        "vad-gap-hist",
        help="Offline Silero VAD — segment a WAV and histogram the silence gaps "
        "between speech regions into fixed-width bins (see the pause-length "
        "distribution shape, not just min/mean/max)",
    )
    vad_gap_hist.add_argument(
        "wav",
        help="Path to a 16-bit PCM WAV file to segment",
    )
    vad_gap_hist.add_argument(
        "--bin-width-s",
        type=positive_float_type,
        default=0.5,
        dest="bin_width_s",
        help="Histogram bin width in seconds (default: 0.5)",
    )
    vad_gap_hist.add_argument(
        "--threshold",
        type=unit_interval_type,
        default=vad_threshold_default,
        help=f"P(speech) gate in [0, 1] (default: {vad_threshold_default})",
    )
    vad_gap_hist.add_argument(
        "--min-speech-ms",
        type=nonneg_float_type,
        default=vad_min_speech_default,
        dest="min_speech_ms",
        help="Drop speech regions shorter than this, in ms "
        f"(default: {vad_min_speech_default})",
    )
    vad_gap_hist.add_argument(
        "--min-silence-ms",
        type=nonneg_float_type,
        default=vad_min_silence_default,
        dest="min_silence_ms",
        help="Trailing silence before a region ends, in ms — matches the "
        f"pipecat stop_secs=0.8 live default (default: {vad_min_silence_default})",
    )
    vad_gap_hist.add_argument(
        "--speech-pad-ms",
        type=nonneg_float_type,
        default=vad_speech_pad_default,
        dest="speech_pad_ms",
        help="Symmetric padding added to each region, in ms "
        f"(default: {vad_speech_pad_default})",
    )
    vad_gap_hist.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds; 'inf'/'none' "
        "never splits (default: inf)",
    )
    vad_gap_hist_fmt = vad_gap_hist.add_mutually_exclusive_group()
    vad_gap_hist_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON (aggregate stats + per-bin list) "
        "instead of the human-readable report",
    )
    vad_gap_hist_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit a flat bin_index,lo_s,hi_s,count CSV table (one row per bin) "
        "for spreadsheets/plots; mutually exclusive with --json",
    )

    # gv vad-gap-sweep — segment one WAV across a swept knob and tabulate the
    # inter-segment SILENCE-gap distribution (iter-330). The gap-side analogue
    # of vad-sweep: where vad-sweep tabulates segment-count / speech-seconds vs
    # the swept knob, vad-gap-sweep tabulates min/mean/max gap so an operator
    # can watch the shortest-pause floor move as the gate tightens — the value
    # that lifts the min gap clear of a target hangover (--min-silence-ms) is
    # the one that buys merge headroom. Shares the iter-256 five-axis mutex
    # (--thresholds default / --min-silences / --min-speeches / --speech-pads /
    # --max-speeches). No --target pick block (the pick scores on segment count,
    # which the gap surface does not headline).
    vad_gap_sweep = sub.add_parser(
        "vad-gap-sweep",
        help="Offline Silero VAD — segment a WAV across a swept knob "
        "(--thresholds gate, --min-silences hangover, --min-speeches floor, "
        "--speech-pads region padding, or --max-speeches force-split ceiling) "
        "and tabulate the inter-segment silence-gap distribution "
        "(min/mean/max gap — find where the shortest pause buys merge headroom)",
    )
    vad_gap_sweep.add_argument(
        "wav",
        help="Path to a 16-bit PCM WAV file to segment at each swept value",
    )
    # The swept axis: --thresholds (default) OR one of the four ms/seconds
    # knobs, never more than one — same shape as vad-sweep's axis mutex.
    vad_gap_sweep_axis = vad_gap_sweep.add_mutually_exclusive_group()
    vad_gap_sweep_axis.add_argument(
        "--thresholds",
        type=unit_interval_list_type,
        default=[0.3, 0.5, 0.7, 0.9],
        help="Comma-separated P(speech) gates in [0, 1] to sweep "
        "(default: 0.3,0.5,0.7,0.9; mutually exclusive with the ms/seconds axes)",
    )
    vad_gap_sweep_axis.add_argument(
        "--min-silences",
        type=nonneg_float_list_type,
        default=None,
        dest="min_silences",
        help="Comma-separated trailing-silence hangovers in ms to sweep "
        "instead of the gate (e.g. 400,600,800,1000); the gate is held at the "
        "scalar --threshold (mutually exclusive with the other axes)",
    )
    vad_gap_sweep_axis.add_argument(
        "--min-speeches",
        type=nonneg_float_list_type,
        default=None,
        dest="min_speeches",
        help="Comma-separated minimum-speech floors in ms to sweep instead of "
        "the gate (e.g. 50,100,200,400); the gate is held at the scalar "
        "--threshold (mutually exclusive with the other axes)",
    )
    vad_gap_sweep_axis.add_argument(
        "--speech-pads",
        type=nonneg_float_list_type,
        default=None,
        dest="speech_pads",
        help="Comma-separated symmetric region paddings in ms to sweep instead "
        "of the gate (e.g. 0,20,40,60); the gate is held at the scalar "
        "--threshold (mutually exclusive with the other axes)",
    )
    vad_gap_sweep_axis.add_argument(
        "--max-speeches",
        type=max_speech_list_type,
        default=None,
        dest="max_speeches",
        help="Comma-separated force-split ceilings in SECONDS to sweep instead "
        "of the gate (e.g. 5,10,20,inf); 'inf'/'none'/'off' anchors the no-cap "
        "baseline; the gate is held at the scalar --threshold (mutually "
        "exclusive with the other axes)",
    )
    vad_gap_sweep.add_argument(
        "--threshold",
        type=unit_interval_type,
        default=vad_threshold_default,
        help="Scalar P(speech) gate held fixed when sweeping --min-silences, "
        "--min-speeches, --speech-pads, or --max-speeches, in [0, 1]; ignored "
        "when sweeping --thresholds "
        f"(default: {vad_threshold_default})",
    )
    vad_gap_sweep.add_argument(
        "--min-speech-ms",
        type=nonneg_float_type,
        default=vad_min_speech_default,
        dest="min_speech_ms",
        help="Drop speech regions shorter than this, in ms — shared by all "
        "runs when sweeping --thresholds; ignored when sweeping --min-speeches "
        f"(default: {vad_min_speech_default})",
    )
    vad_gap_sweep.add_argument(
        "--min-silence-ms",
        type=nonneg_float_type,
        default=vad_min_silence_default,
        dest="min_silence_ms",
        help="Trailing silence before a region ends, in ms — shared by all "
        "runs when sweeping --thresholds; ignored when sweeping --min-silences "
        f"(default: {vad_min_silence_default})",
    )
    vad_gap_sweep.add_argument(
        "--speech-pad-ms",
        type=nonneg_float_type,
        default=vad_speech_pad_default,
        dest="speech_pad_ms",
        help="Symmetric padding added to each region, in ms — shared by all "
        "runs when sweeping --thresholds; ignored when sweeping --speech-pads "
        f"(default: {vad_speech_pad_default})",
    )
    vad_gap_sweep.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds — shared by "
        "all runs when sweeping --thresholds; ignored when sweeping "
        "--max-speeches; 'inf'/'none' never splits (default: inf)",
    )
    vad_gap_sweep_fmt = vad_gap_sweep.add_mutually_exclusive_group()
    vad_gap_sweep_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable table",
    )
    vad_gap_sweep_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit a flat <axis>,num_segments,num_gaps,min_gap_s,mean_gap_s,"
        "max_gap_s,total_silence_s CSV table for spreadsheets/plots "
        "(mutually exclusive with --json)",
    )

    # gv vad-gap-peak-sweep — segment one WAV across a swept knob and tabulate
    # the COSTLIEST cost-curve band at each value (iter-364). The peak-side
    # analogue of vad-gap-sweep: where vad-gap-sweep tabulates min/mean/max gap
    # vs the swept knob, vad-gap-peak-sweep tabulates the steepest band (densest
    # pause cluster) so an operator can watch how expensive it is to raise the
    # hangover through that cluster MOVE as a segmenter knob (e.g. --min-speeches)
    # tightens. Shares the iter-256 five-axis mutex (--thresholds default /
    # --min-silences / --min-speeches / --speech-pads / --max-speeches) and the
    # --cuts-ms cost-band axis with vad-gap-peak.
    vad_gap_peak_sweep = sub.add_parser(
        "vad-gap-peak-sweep",
        help="Offline Silero VAD — segment a WAV across a swept knob "
        "(--thresholds gate, --min-silences hangover, --min-speeches floor, "
        "--speech-pads region padding, or --max-speeches force-split ceiling) "
        "and tabulate the costliest cost-curve band at each value (the densest "
        "pause cluster — watch how the cost of raising --min-silence-ms through "
        "it moves)",
    )
    vad_gap_peak_sweep.add_argument(
        "wav",
        help="Path to a 16-bit PCM WAV file to segment at each swept value",
    )
    # The swept axis: --thresholds (default) OR one of the four ms/seconds knobs,
    # never more than one — same shape as vad-gap-sweep's axis mutex.
    vad_gap_peak_sweep_axis = vad_gap_peak_sweep.add_mutually_exclusive_group()
    vad_gap_peak_sweep_axis.add_argument(
        "--thresholds",
        type=unit_interval_list_type,
        default=[0.3, 0.5, 0.7, 0.9],
        help="Comma-separated P(speech) gates in [0, 1] to sweep "
        "(default: 0.3,0.5,0.7,0.9; mutually exclusive with the ms/seconds axes)",
    )
    vad_gap_peak_sweep_axis.add_argument(
        "--min-silences",
        type=nonneg_float_list_type,
        default=None,
        dest="min_silences",
        help="Comma-separated trailing-silence hangovers in ms to sweep "
        "instead of the gate (e.g. 400,600,800,1000); the gate is held at the "
        "scalar --threshold (mutually exclusive with the other axes)",
    )
    vad_gap_peak_sweep_axis.add_argument(
        "--min-speeches",
        type=nonneg_float_list_type,
        default=None,
        dest="min_speeches",
        help="Comma-separated minimum-speech floors in ms to sweep instead of "
        "the gate (e.g. 50,100,200,400); the gate is held at the scalar "
        "--threshold (mutually exclusive with the other axes)",
    )
    vad_gap_peak_sweep_axis.add_argument(
        "--speech-pads",
        type=nonneg_float_list_type,
        default=None,
        dest="speech_pads",
        help="Comma-separated symmetric region paddings in ms to sweep instead "
        "of the gate (e.g. 0,20,40,60); the gate is held at the scalar "
        "--threshold (mutually exclusive with the other axes)",
    )
    vad_gap_peak_sweep_axis.add_argument(
        "--max-speeches",
        type=max_speech_list_type,
        default=None,
        dest="max_speeches",
        help="Comma-separated force-split ceilings in SECONDS to sweep instead "
        "of the gate (e.g. 5,10,20,inf); 'inf'/'none'/'off' anchors the no-cap "
        "baseline; the gate is held at the scalar --threshold (mutually "
        "exclusive with the other axes)",
    )
    vad_gap_peak_sweep.add_argument(
        "--cuts-ms",
        type=cut_ms_list_type,
        default=list(DEFAULT_GAP_CDF_CUTS_MS),
        dest="cuts_ms",
        help="Comma-separated candidate hangover cuts in ms defining the cost "
        "bands scanned for the peak at each swept value, e.g. '200,400,800,1600' "
        "(sorted + de-duplicated; default: 200,400,800,1600)",
    )
    vad_gap_peak_sweep.add_argument(
        "--threshold",
        type=unit_interval_type,
        default=vad_threshold_default,
        help="Scalar P(speech) gate held fixed when sweeping --min-silences, "
        "--min-speeches, --speech-pads, or --max-speeches, in [0, 1]; ignored "
        "when sweeping --thresholds "
        f"(default: {vad_threshold_default})",
    )
    vad_gap_peak_sweep.add_argument(
        "--min-speech-ms",
        type=nonneg_float_type,
        default=vad_min_speech_default,
        dest="min_speech_ms",
        help="Drop speech regions shorter than this, in ms — shared by all "
        "runs when sweeping --thresholds; ignored when sweeping --min-speeches "
        f"(default: {vad_min_speech_default})",
    )
    vad_gap_peak_sweep.add_argument(
        "--min-silence-ms",
        type=nonneg_float_type,
        default=vad_min_silence_default,
        dest="min_silence_ms",
        help="Trailing silence before a region ends, in ms — shared by all "
        "runs when sweeping --thresholds; ignored when sweeping --min-silences "
        f"(default: {vad_min_silence_default})",
    )
    vad_gap_peak_sweep.add_argument(
        "--speech-pad-ms",
        type=nonneg_float_type,
        default=vad_speech_pad_default,
        dest="speech_pad_ms",
        help="Symmetric padding added to each region, in ms — shared by all "
        "runs when sweeping --thresholds; ignored when sweeping --speech-pads "
        f"(default: {vad_speech_pad_default})",
    )
    vad_gap_peak_sweep.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds — shared by "
        "all runs when sweeping --thresholds; ignored when sweeping "
        "--max-speeches; 'inf'/'none' never splits (default: inf)",
    )
    vad_gap_peak_sweep_fmt = vad_gap_peak_sweep.add_mutually_exclusive_group()
    vad_gap_peak_sweep_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable table",
    )
    vad_gap_peak_sweep_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit a flat <axis>,num_segments,num_gaps,peak_found,peak_from_ms,"
        "peak_to_ms,peak_width_ms,peak_merged_added,peak_rate_per_100ms CSV table "
        "for spreadsheets/plots (mutually exclusive with --json)",
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
    vad_diff_fmt = vad_diff.add_mutually_exclusive_group()
    vad_diff_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable report",
    )
    vad_diff_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit a flat threshold,num_segments,speech_s CSV table (one row "
        "per threshold) for spreadsheets/plots — byte-identical to a two-value "
        "vad-sweep --csv; mutually exclusive with --json",
    )

    # gv vad-gap-diff — segment one WAV under two thresholds and report how the
    # SILENCE-gap distribution shifts (iter-334). The gap-side analogue of
    # vad-diff and the two-point degenerate of vad-gap-sweep: where vad-diff
    # reports the segment-count / speech-seconds delta, vad-gap-diff reports the
    # min/mean/max gap delta so an operator can watch whether a stricter gate
    # lifts the shortest pause clear of a target hangover (--min-silence-ms),
    # buying merge headroom. All knobs but threshold are shared between the runs.
    vad_gap_diff = sub.add_parser(
        "vad-gap-diff",
        help="Offline Silero VAD — segment a WAV at two thresholds and report "
        "the inter-segment silence-gap delta (min/mean/max gap — watch the "
        "shortest pause move as the P(speech) gate tightens)",
    )
    vad_gap_diff.add_argument(
        "wav",
        help="Path to a 16-bit PCM WAV file to segment under both thresholds",
    )
    vad_gap_diff.add_argument(
        "--threshold-a",
        type=unit_interval_type,
        default=vad_threshold_default,
        dest="threshold_a",
        help=f"First P(speech) gate in [0, 1] (default: {vad_threshold_default})",
    )
    vad_gap_diff.add_argument(
        "--threshold-b",
        type=unit_interval_type,
        default=0.7,
        dest="threshold_b",
        help="Second P(speech) gate in [0, 1] (default: 0.7)",
    )
    vad_gap_diff.add_argument(
        "--min-speech-ms",
        type=nonneg_float_type,
        default=vad_min_speech_default,
        dest="min_speech_ms",
        help="Drop speech regions shorter than this, in ms — shared by both "
        f"runs (default: {vad_min_speech_default})",
    )
    vad_gap_diff.add_argument(
        "--min-silence-ms",
        type=nonneg_float_type,
        default=vad_min_silence_default,
        dest="min_silence_ms",
        help="Trailing silence before a region ends, in ms — shared by both "
        f"runs (default: {vad_min_silence_default})",
    )
    vad_gap_diff.add_argument(
        "--speech-pad-ms",
        type=nonneg_float_type,
        default=vad_speech_pad_default,
        dest="speech_pad_ms",
        help="Symmetric padding added to each region, in ms — shared by both "
        f"runs (default: {vad_speech_pad_default})",
    )
    vad_gap_diff.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds — shared by "
        "both runs; 'inf'/'none' never splits (default: inf)",
    )
    vad_gap_diff_fmt = vad_gap_diff.add_mutually_exclusive_group()
    vad_gap_diff_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable report",
    )
    vad_gap_diff_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit a flat threshold,num_segments,num_gaps,min_gap_s,mean_gap_s,"
        "max_gap_s,total_silence_s CSV table (one row per threshold) for "
        "spreadsheets/plots — byte-identical to a two-value vad-gap-sweep --csv; "
        "mutually exclusive with --json",
    )

    # gv vad-sweep — segment one WAV across a swept knob and tabulate the result.
    # Generalises iter-235's two-point vad-diff to a sweep so the knob's elbow is
    # visible at a glance. iter-236 swept the P(speech) gate (--thresholds);
    # iter-238 adds a second axis, the trailing-silence hangover (--min-silences);
    # iter-239 adds a third, the minimum-speech floor (--min-speeches); iter-253 a
    # fourth, the region padding (--speech-pads); iter-256 a fifth, the
    # force-split ceiling (--max-speeches, SECONDS). The five axes are mutually
    # exclusive. Every non-swept knob is shared across all runs.
    vad_sweep = sub.add_parser(
        "vad-sweep",
        help="Offline Silero VAD — segment a WAV across a swept knob "
        "(--thresholds gate, --min-silences hangover, --min-speeches floor, "
        "--speech-pads region padding, or --max-speeches force-split ceiling) "
        "and tabulate segment-count / speech-seconds (find the knob's elbow)",
    )
    vad_sweep.add_argument(
        "wav",
        help="Path to a 16-bit PCM WAV file to segment at each swept value",
    )
    # The swept axis: --thresholds (default) OR --min-silences OR --min-speeches
    # OR --speech-pads OR --max-speeches, never more than one. The default list lives on
    # --thresholds so a bare `vad-sweep rec.wav` keeps the iter-236 behaviour; a
    # group default isn't "provided", so the mutex only fires when two are passed
    # explicitly.
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
    vad_sweep_axis.add_argument(
        "--speech-pads",
        type=nonneg_float_list_type,
        default=None,
        dest="speech_pads",
        help="Comma-separated symmetric region paddings in ms to sweep instead "
        "of the gate (e.g. 0,20,40,60); the gate is held at the scalar "
        "--threshold (mutually exclusive with the other axes)",
    )
    vad_sweep_axis.add_argument(
        "--max-speeches",
        type=max_speech_list_type,
        default=None,
        dest="max_speeches",
        help="Comma-separated force-split ceilings in SECONDS to sweep instead "
        "of the gate (e.g. 5,10,20,inf); 'inf'/'none'/'off' anchors the no-cap "
        "baseline; the gate is held at the scalar --threshold (mutually "
        "exclusive with the other axes)",
    )
    vad_sweep.add_argument(
        "--threshold",
        type=unit_interval_type,
        default=vad_threshold_default,
        help="Scalar P(speech) gate held fixed when sweeping --min-silences, "
        "--min-speeches, --speech-pads, or --max-speeches, in [0, 1]; ignored "
        "when sweeping --thresholds "
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
        "runs when sweeping --thresholds; ignored when sweeping --speech-pads "
        f"(default: {vad_speech_pad_default})",
    )
    vad_sweep.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds — shared by "
        "all runs when sweeping --thresholds; ignored when sweeping "
        "--max-speeches; 'inf'/'none' never splits (default: inf)",
    )
    vad_sweep.add_argument(
        "--target",
        type=target_type,
        default=None,
        dest="target",
        help="Desired segment count (e.g. 3), closed band (e.g. 3-5), "
        "open band (3- = at least 3, -5 = at most 5), set (3,5,7 = 3 OR 5 "
        "OR 7), preference (3>5>7 = prefer 3, accept 5, then 7), weighted set "
        "(3,5:2 = prefer 3, accept 5 but 2 segments worse; the weight may be "
        "fractional, 3,5:1.5), scaled set (3,5*1.5 = prefer 3, accept 5 but "
        "drift past it costs 1.5x), or affine set (3,5*1.5:2 = both: scale by 1.5 "
        "then add 2) — when given, "
        "a data-driven 'best:' pick names the swept value whose recovered "
        "segment count is closest to it (a band/set/preference scores 0 anywhere "
        "it is satisfied; a preference breaks ties toward the earlier-listed "
        "count; a :weight adds to a count's distance and a *factor multiplies it "
        "so a preferred count can win at a larger distance); the same machinery "
        "as vad-grid's --target; omit "
        "for just the table",
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
    # gate (--thresholds); the column axis is the trailing-silence hangover
    # (--min-silences, default, ms), the minimum-speech floor (--min-speeches,
    # ms), the region padding (--speech-pads, iter-254, ms), or the force-split
    # ceiling (--max-speeches, iter-255, SECONDS), mutually exclusive. The
    # non-column knob is held at its scalar; every other knob is shared across
    # all cells.
    vad_grid = sub.add_parser(
        "vad-grid",
        help="Offline Silero VAD — segment a WAV across a 2-D grid (gate "
        "--thresholds × a column axis: --min-silences hangover, "
        "--min-speeches floor, --speech-pads region padding (ms), or "
        "--max-speeches force-split ceiling (seconds)) and tabulate "
        "segment-count / speech-seconds per cell (read the elbow in two "
        "dimensions at once)",
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
    # The column axis: --min-silences (default) OR --min-speeches OR
    # --speech-pads OR --max-speeches, never more than one. The default list
    # lives on --min-silences so a bare `vad-grid rec.wav` sweeps gate ×
    # hangover; a group default isn't "provided", so the mutex only fires when
    # two are passed explicitly.
    vad_grid_col = vad_grid.add_mutually_exclusive_group()
    vad_grid_col.add_argument(
        "--min-silences",
        type=nonneg_float_list_type,
        default=[400.0, 600.0, 800.0, 1000.0],
        dest="min_silences",
        help="Comma-separated trailing-silence hangovers in ms — the grid "
        "COLUMN axis (default: 400,600,800,1000; mutually exclusive with "
        "--min-speeches / --speech-pads / --max-speeches)",
    )
    vad_grid_col.add_argument(
        "--min-speeches",
        type=nonneg_float_list_type,
        default=None,
        dest="min_speeches",
        help="Comma-separated minimum-speech floors in ms to use as the grid "
        "COLUMN axis instead of the hangover (e.g. 50,100,200,400); the "
        "non-column knob is held at its scalar (mutually exclusive with "
        "--min-silences / --speech-pads / --max-speeches)",
    )
    vad_grid_col.add_argument(
        "--speech-pads",
        type=nonneg_float_list_type,
        default=None,
        dest="speech_pads",
        help="Comma-separated symmetric region paddings in ms to use as the "
        "grid COLUMN axis instead of the hangover (e.g. 0,20,40,80); the "
        "non-column knob is held at its scalar (mutually exclusive with "
        "--min-silences / --min-speeches / --max-speeches)",
    )
    vad_grid_col.add_argument(
        "--max-speeches",
        type=max_speech_list_type,
        default=None,
        dest="max_speeches",
        help="Comma-separated force-split ceilings in SECONDS to use as the "
        "grid COLUMN axis instead of the hangover (e.g. 5,10,20,inf); 'inf'/"
        "'none'/'off' never splits, so include it for the no-cap baseline; the "
        "non-column knob is held at its scalar (mutually exclusive with "
        "--min-silences / --min-speeches / --speech-pads)",
    )
    vad_grid.add_argument(
        "--min-speech-ms",
        type=nonneg_float_type,
        default=vad_min_speech_default,
        dest="min_speech_ms",
        help="Drop speech regions shorter than this, in ms — held fixed across "
        "all cells when the column axis is --min-silences/--speech-pads/"
        "--max-speeches; ignored when sweeping --min-speeches (default: "
        f"{vad_min_speech_default})",
    )
    vad_grid.add_argument(
        "--min-silence-ms",
        type=nonneg_float_type,
        default=vad_min_silence_default,
        dest="min_silence_ms",
        help="Trailing silence before a region ends, in ms — held fixed across "
        "all cells when the column axis is --min-speeches/--speech-pads/"
        "--max-speeches; ignored when sweeping --min-silences (default: "
        f"{vad_min_silence_default})",
    )
    vad_grid.add_argument(
        "--speech-pad-ms",
        type=nonneg_float_type,
        default=vad_speech_pad_default,
        dest="speech_pad_ms",
        help="Symmetric padding added to each region, in ms — held fixed across "
        "all cells when the column axis is --min-silences/--min-speeches/"
        "--max-speeches; ignored when sweeping --speech-pads (default: "
        f"{vad_speech_pad_default})",
    )
    vad_grid.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds — held fixed "
        "across all cells when the column axis is --min-silences/--min-speeches/"
        "--speech-pads; ignored when sweeping --max-speeches; 'inf'/'none' "
        "never splits (default: inf)",
    )
    vad_grid.add_argument(
        "--target",
        type=target_type,
        default=None,
        dest="target",
        help="Desired segment count (e.g. 3), closed band (e.g. 3-5), "
        "open band (3- = at least 3, -5 = at most 5), set (3,5,7 = 3 OR 5 "
        "OR 7), preference (3>5>7 = prefer 3, accept 5, then 7), weighted set "
        "(3,5:2 = prefer 3, accept 5 but 2 segments worse; the weight may be "
        "fractional, 3,5:1.5), scaled set (3,5*1.5 = drift past 5 costs 1.5x), or "
        "affine set (3,5*1.5:2 = both: scale by 1.5 then add 2) — when given, "
        "a data-driven 'best:' pick names the cell whose recovered segment count "
        "is closest to it (a band/set/preference scores 0 anywhere it is "
        "satisfied; a preference breaks ties toward the earlier-listed count; a "
        ":weight adds to a count's distance and a *factor multiplies it so a "
        "preferred count can win at a larger distance); the vad-grid analogue of "
        "simulate-mirror --grid's "
        "best pick; omit for just the table",
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

    # gv vad-gap-grid — segment one WAV across a 2-D knob grid and tabulate the
    # inter-segment SILENCE-gap distribution per cell (iter-332). The gap-side
    # analogue of vad-grid (and the 2-D analogue of vad-gap-sweep): rows are
    # always the P(speech) gate (--thresholds); the column axis is the
    # trailing-silence hangover (--min-silences, default, ms), the
    # minimum-speech floor (--min-speeches, ms), the region padding
    # (--speech-pads, ms), or the force-split ceiling (--max-speeches, SECONDS),
    # mutually exclusive. Each cell reports min/mean/max gap so an operator can
    # read where the shortest pause buys merge headroom across two knobs at
    # once. No --target pick block (the pick scores on segment count, which the
    # gap surface does not headline).
    vad_gap_grid = sub.add_parser(
        "vad-gap-grid",
        help="Offline Silero VAD — segment a WAV across a 2-D grid (gate "
        "--thresholds × a column axis: --min-silences hangover, "
        "--min-speeches floor, --speech-pads region padding (ms), or "
        "--max-speeches force-split ceiling (seconds)) and tabulate the "
        "inter-segment silence-gap distribution (min/mean/max gap) per cell "
        "— find where the shortest pause buys merge headroom in two dimensions",
    )
    vad_gap_grid.add_argument(
        "wav",
        help="Path to a 16-bit PCM WAV file to segment at each grid cell",
    )
    vad_gap_grid.add_argument(
        "--thresholds",
        type=unit_interval_list_type,
        default=[0.3, 0.5, 0.7, 0.9],
        help="Comma-separated P(speech) gates in [0, 1] — the grid ROW axis "
        "(default: 0.3,0.5,0.7,0.9)",
    )
    # The column axis: --min-silences (default) OR --min-speeches OR
    # --speech-pads OR --max-speeches, never more than one. The default list
    # lives on --min-silences so a bare `vad-gap-grid rec.wav` sweeps gate ×
    # hangover; a group default isn't "provided", so the mutex only fires when
    # two are passed explicitly. Mirrors vad-grid's column mutex exactly.
    vad_gap_grid_col = vad_gap_grid.add_mutually_exclusive_group()
    vad_gap_grid_col.add_argument(
        "--min-silences",
        type=nonneg_float_list_type,
        default=[400.0, 600.0, 800.0, 1000.0],
        dest="min_silences",
        help="Comma-separated trailing-silence hangovers in ms — the grid "
        "COLUMN axis (default: 400,600,800,1000; mutually exclusive with "
        "--min-speeches / --speech-pads / --max-speeches)",
    )
    vad_gap_grid_col.add_argument(
        "--min-speeches",
        type=nonneg_float_list_type,
        default=None,
        dest="min_speeches",
        help="Comma-separated minimum-speech floors in ms to use as the grid "
        "COLUMN axis instead of the hangover (e.g. 50,100,200,400); the "
        "non-column knob is held at its scalar (mutually exclusive with "
        "--min-silences / --speech-pads / --max-speeches)",
    )
    vad_gap_grid_col.add_argument(
        "--speech-pads",
        type=nonneg_float_list_type,
        default=None,
        dest="speech_pads",
        help="Comma-separated symmetric region paddings in ms to use as the "
        "grid COLUMN axis instead of the hangover (e.g. 0,20,40,80); the "
        "non-column knob is held at its scalar (mutually exclusive with "
        "--min-silences / --min-speeches / --max-speeches)",
    )
    vad_gap_grid_col.add_argument(
        "--max-speeches",
        type=max_speech_list_type,
        default=None,
        dest="max_speeches",
        help="Comma-separated force-split ceilings in SECONDS to use as the "
        "grid COLUMN axis instead of the hangover (e.g. 5,10,20,inf); 'inf'/"
        "'none'/'off' never splits, so include it for the no-cap baseline; the "
        "non-column knob is held at its scalar (mutually exclusive with "
        "--min-silences / --min-speeches / --speech-pads)",
    )
    vad_gap_grid.add_argument(
        "--min-speech-ms",
        type=nonneg_float_type,
        default=vad_min_speech_default,
        dest="min_speech_ms",
        help="Drop speech regions shorter than this, in ms — held fixed across "
        "all cells when the column axis is --min-silences/--speech-pads/"
        "--max-speeches; ignored when sweeping --min-speeches (default: "
        f"{vad_min_speech_default})",
    )
    vad_gap_grid.add_argument(
        "--min-silence-ms",
        type=nonneg_float_type,
        default=vad_min_silence_default,
        dest="min_silence_ms",
        help="Trailing silence before a region ends, in ms — held fixed across "
        "all cells when the column axis is --min-speeches/--speech-pads/"
        "--max-speeches; ignored when sweeping --min-silences (default: "
        f"{vad_min_silence_default})",
    )
    vad_gap_grid.add_argument(
        "--speech-pad-ms",
        type=nonneg_float_type,
        default=vad_speech_pad_default,
        dest="speech_pad_ms",
        help="Symmetric padding added to each region, in ms — held fixed across "
        "all cells when the column axis is --min-silences/--min-speeches/"
        "--max-speeches; ignored when sweeping --speech-pads (default: "
        f"{vad_speech_pad_default})",
    )
    vad_gap_grid.add_argument(
        "--max-speech-s",
        type=max_speech_type,
        default=vad_max_speech_default,
        dest="max_speech_s",
        help="Force-split regions longer than this, in seconds — held fixed "
        "across all cells when the column axis is --min-silences/--min-speeches/"
        "--speech-pads; ignored when sweeping --max-speeches; 'inf'/'none' "
        "never splits (default: inf)",
    )
    vad_gap_grid_fmt = vad_gap_grid.add_mutually_exclusive_group()
    vad_gap_grid_fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable table",
    )
    vad_gap_grid_fmt.add_argument(
        "--csv",
        action="store_true",
        help="Emit a flat <row_axis>,<col_axis>,num_segments,num_gaps,min_gap_s,"
        "mean_gap_s,max_gap_s,total_silence_s CSV table for spreadsheets/plots "
        "(mutually exclusive with --json)",
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
