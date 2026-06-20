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


def render_grid_csv(points, best):
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
        score = p.score()
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

    iter-315 adds ``--csv``, the first machine-readable surface on this command:
    in ``--grid`` mode it emits the flat per-cell sweep table
    (:func:`render_grid_csv`) and in trajectory mode the per-turn speed curve
    (:func:`render_trajectory_csv`), bringing the human / ``--csv`` pairing the
    VAD-analysis surfaces already carry to the WPM-mirror simulator.
    """
    wm = _load_wpm_mirror()
    WpmMirrorConfig = wm.WpmMirrorConfig
    simulate_speed_trajectory = wm.simulate_speed_trajectory
    sweep_mirror_grid = wm.sweep_mirror_grid
    pick_best_mirror_config = wm.pick_best_mirror_config

    as_csv = getattr(args, "csv", False)

    if args.grid:
        points = sweep_mirror_grid(
            args.wpms,
            args.base_wpms,
            args.strengths,
            initial_speed=args.initial_speed,
        )
        best = pick_best_mirror_config(points)
        if as_csv:
            log(render_grid_csv(points, best))
        else:
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
    if as_csv:
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
    sim.add_argument(
        "--csv",
        action="store_true",
        help="Emit a flat CSV table instead of the human report — per-cell in "
        "--grid mode, per-turn (speed curve) in trajectory mode",
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
