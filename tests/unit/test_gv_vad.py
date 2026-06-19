"""Tests for iter-233 — the ``gv vad`` subcommand (examples/gv.py).

iter-231 shipped the Silero batch segmenter (``vad/silero.py``), reachable
only via the :5111 HTTP endpoint (``POST /vad/silero``) and the
``fixtures/replay_silero.py`` script. iter-233 brings it to the gv CLI:
``gv vad recording.wav`` segments any 16-bit PCM WAV into speech regions
offline — no server, no mic — the headless analogue of the live mic path.

These tests exercise the new parser arg-type validators, the pure
``render_vad_segments`` helper, and the ``cmd_vad`` handler. The handler takes
injected ``segmenter`` / ``availability`` / ``log`` dependencies (mirroring
``dispatch``'s handler injection), so every test runs WITHOUT importing torch /
silero-vad and without touching real audio — fast and deterministic on the
x86_64 Linux runner.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples import gv  # noqa: E402


# ---- lightweight stand-ins for the SileroResult / SpeechSegment shapes ----
# We don't import vad.silero (it pulls torch); these mirror just the attributes
# render_vad_segments / cmd_vad read.


@dataclass
class _Seg:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class _Result:
    name: str
    sample_rate: int
    duration_s: float
    segments: List[_Seg] = field(default_factory=list)

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    @property
    def speech_s(self) -> float:
        return sum(s.duration_s for s in self.segments)


# ---- parser: registration & defaults -----------------------------------


def test_vad_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad"] is gv.cmd_vad


def test_vad_defaults_mirror_silero_params():
    args = gv.build_parser().parse_args(["vad", "rec.wav"])
    assert args.command == "vad"
    assert args.wav == "rec.wav"
    # Defaults track SileroParams (iter-231) — the live pipecat stop_secs=0.8.
    assert args.threshold == 0.5
    assert args.min_speech_ms == 250.0
    assert args.min_silence_ms == 800.0
    assert args.speech_pad_ms == 30.0
    assert args.max_speech_s == float("inf")


def test_vad_requires_wav_positional():
    # The WAV path is required — argparse exits 2 when it is missing.
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(["vad"])
    assert exc.value.code == 2


def test_vad_overrides_parse():
    args = gv.build_parser().parse_args(
        [
            "vad",
            "clip.wav",
            "--threshold",
            "0.7",
            "--min-speech-ms",
            "100",
            "--min-silence-ms",
            "500",
            "--speech-pad-ms",
            "0",
            "--max-speech-s",
            "12",
        ]
    )
    assert args.threshold == 0.7
    assert args.min_speech_ms == 100.0
    assert args.min_silence_ms == 500.0
    assert args.speech_pad_ms == 0.0
    assert args.max_speech_s == 12.0


# ---- unit_interval_type: the --threshold validator ---------------------


@pytest.mark.parametrize("raw", ["0", "0.5", "1", "0.999"])
def test_unit_interval_accepts_in_range(raw):
    value = gv.unit_interval_type(raw)
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("raw", ["-0.1", "1.1", "2", "100"])
def test_unit_interval_rejects_out_of_range(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.unit_interval_type(raw)


@pytest.mark.parametrize("raw", ["high", "", "0.5x"])
def test_unit_interval_rejects_non_numbers(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.unit_interval_type(raw)


def test_unit_interval_rejects_nan():
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        gv.unit_interval_type("nan")
    assert "nan" in str(exc.value)


def test_parser_rejects_out_of_range_threshold_via_systemexit():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(["vad", "x.wav", "--threshold", "1.5"])
    assert exc.value.code == 2


# ---- nonneg_float_type: the millisecond knobs --------------------------


@pytest.mark.parametrize("raw", ["0", "0.0", "250", "800.5"])
def test_nonneg_float_accepts_zero_and_positive(raw):
    value = gv.nonneg_float_type(raw)
    assert isinstance(value, float)
    assert value >= 0


@pytest.mark.parametrize("raw", ["-1", "-0.5"])
def test_nonneg_float_rejects_negative(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.nonneg_float_type(raw)


@pytest.mark.parametrize("raw", ["lots", "", "12ms"])
def test_nonneg_float_rejects_non_numbers(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.nonneg_float_type(raw)


def test_nonneg_float_rejects_nan():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.nonneg_float_type("nan")


# ---- nonneg_int_type: the vad-grid target segment count ----------------


@pytest.mark.parametrize("raw", ["0", "1", "5", "42"])
def test_nonneg_int_accepts_zero_and_positive(raw):
    value = gv.nonneg_int_type(raw)
    assert isinstance(value, int)
    assert value >= 0


@pytest.mark.parametrize("raw", ["-1", "-5"])
def test_nonneg_int_rejects_negative(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.nonneg_int_type(raw)


@pytest.mark.parametrize("raw", ["lots", "", "1.5", "3ms"])
def test_nonneg_int_rejects_non_integers(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.nonneg_int_type(raw)


# ---- pos_int_type: the vad-grid --top shortlist length -----------------


@pytest.mark.parametrize("raw", ["1", "3", "42"])
def test_pos_int_accepts_positive(raw):
    value = gv.pos_int_type(raw)
    assert isinstance(value, int)
    assert value >= 1


@pytest.mark.parametrize("raw", ["0", "-1", "-5"])
def test_pos_int_rejects_zero_and_negative(raw):
    # Unlike nonneg_int_type, zero is rejected — a 0-cell shortlist is useless.
    with pytest.raises(argparse.ArgumentTypeError):
        gv.pos_int_type(raw)


@pytest.mark.parametrize("raw", ["many", "", "1.5", "2x"])
def test_pos_int_rejects_non_integers(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.pos_int_type(raw)


# ---- target_type: scalar count OR (lo, hi) tolerance band (iter-246) ---


@pytest.mark.parametrize("raw,expected", [("0", 0), ("1", 1), ("5", 5), ("42", 42)])
def test_target_type_scalar_parses_to_int(raw, expected):
    value = gv.target_type(raw)
    assert value == expected
    assert isinstance(value, int)


@pytest.mark.parametrize(
    "raw,expected",
    [("3-5", (3, 5)), ("0-2", (0, 2)), ("4-4", (4, 4)), (" 3-5 ", (3, 5))],
)
def test_target_type_band_parses_to_tuple(raw, expected):
    value = gv.target_type(raw)
    assert value == expected
    assert isinstance(value, tuple)


@pytest.mark.parametrize("raw", ["5-3", "9-0"])
def test_target_type_rejects_inverted_band(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type(raw)


@pytest.mark.parametrize("raw", ["3-5-7", "", "a-b", "1.5-2", "-", " - ", "3.0-"])
def test_target_type_rejects_malformed(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type(raw)


@pytest.mark.parametrize(
    "raw,expected",
    [("3-", (3, None)), ("-5", (None, 5)), ("0-", (0, None)), ("-0", (None, 0)),
     (" 3- ", (3, None)), (" -5 ", (None, 5))],
)
def test_target_type_open_band_parses_to_tuple(raw, expected):
    # iter-247: an empty edge is the open form — '3-' = at least 3 → (3, None);
    # '-5' = at most 5 → (None, 5). One edge None, the other a non-negative int.
    value = gv.target_type(raw)
    assert value == expected
    assert isinstance(value, tuple)


# ---- _format_target: scalar vs band display ----------------------------


def test_format_target_scalar():
    assert gv._format_target(3) == "3"
    assert gv._format_target(0) == "0"


def test_format_target_band():
    assert gv._format_target((3, 5)) == "3-5"
    assert gv._format_target((0, 2)) == "0-2"


def test_format_target_open_band():
    # iter-247: an open edge stays empty so it reads back exactly as typed.
    assert gv._format_target((3, None)) == "3-"
    assert gv._format_target((None, 5)) == "-5"
    assert gv._format_target((0, None)) == "0-"
    assert gv._format_target((None, 0)) == "-0"


# ---- target_type: comma-separated SET form (iter-248) ------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("3,5,7", [3, 5, 7]),
        ("0,2", [0, 2]),
        (" 3 , 5 , 7 ", [3, 5, 7]),  # whitespace around each element trimmed
        ("3,5-7", [3, (5, 7)]),  # an element may itself be a band
        ("3,5-,-7", [3, (5, None), (None, 7)]),  # ...or an open band
    ],
)
def test_target_type_set_parses_to_list(raw, expected):
    # iter-248: a comma joins a SET of acceptable targets, each element a scalar
    # or a band; the whole thing parses to a list preserving first-seen order.
    value = gv.target_type(raw)
    assert value == expected
    assert isinstance(value, list)


def test_target_type_set_dedupes_preserving_order():
    # iter-248: a repeated element is collapsed, first-seen order preserved.
    assert gv.target_type("5,3,5,3") == [5, 3]


def test_target_type_single_element_set_collapses_to_bare_element():
    # iter-248: a set whose elements collapse to one reduces to that bare element
    # so scalar/band output stays byte-for-byte unchanged.
    value = gv.target_type("3,3")
    assert value == 3
    assert isinstance(value, int)
    band = gv.target_type("3-5,3-5")
    assert band == (3, 5)
    assert isinstance(band, tuple)


@pytest.mark.parametrize("raw", ["3,", ",5", "3,,5", " 3 , , 5 ", ","])
def test_target_type_rejects_empty_set_element(raw):
    # iter-248: an empty element (trailing/leading/doubled comma) is a typo.
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type(raw)


def test_target_type_rejects_malformed_set_element():
    # iter-248: each element must itself parse — a bad element fails the whole set.
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type("3,a,5")
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type("3,5-3")  # inverted band element


def test_format_target_set():
    # iter-248: a set renders comma-joined, each element via _format_target, so it
    # reads back exactly as typed.
    assert gv._format_target([3, 5, 7]) == "3,5,7"
    assert gv._format_target([3, (5, 7)]) == "3,5-7"
    assert gv._format_target([3, (5, None), (None, 7)]) == "3,5-,-7"


# ---- target_type: '>'-separated PREFERENCE-ORDER form (iter-249) -------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("3>5>7", {"prefer": [3, 5, 7]}),
        ("0>2", {"prefer": [0, 2]}),
        (" 3 > 5 > 7 ", {"prefer": [3, 5, 7]}),  # whitespace around each trimmed
        ("3>5-7", {"prefer": [3, (5, 7)]}),  # an element may itself be a band
        ("3>5->-7", {"prefer": [3, (5, None), (None, 7)]}),  # ...or an open band
    ],
)
def test_target_type_preference_parses_to_prefer_dict(raw, expected):
    # iter-249: '>' joins a PREFERENCE order, each element a scalar or band; the
    # whole thing parses to a {"prefer": [...]} dict preserving listed order.
    value = gv.target_type(raw)
    assert value == expected
    assert isinstance(value, dict)


def test_target_type_preference_dedupes_preserving_order():
    # iter-249: a repeated preference element is collapsed, first-seen order kept.
    assert gv.target_type("5>3>5>3") == {"prefer": [5, 3]}


def test_target_type_single_element_preference_collapses_to_bare_element():
    # iter-249: a preference whose elements collapse to one reduces to that bare
    # element so scalar/band output stays byte-for-byte unchanged.
    value = gv.target_type("3>3")
    assert value == 3
    assert isinstance(value, int)
    band = gv.target_type("3-5>3-5")
    assert band == (3, 5)
    assert isinstance(band, tuple)


@pytest.mark.parametrize("raw", ["3>", ">5", "3>>5", " 3 > > 5 ", ">"])
def test_target_type_rejects_empty_preference_element(raw):
    # iter-249: an empty element (trailing/leading/doubled '>') is a typo.
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type(raw)


def test_target_type_rejects_malformed_preference_element():
    # iter-249: each element must itself parse — a bad element fails the whole one.
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type("3>a>5")
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type("3>5-3")  # inverted band element


@pytest.mark.parametrize("raw", ["3,5>7", "3>5,7", "3,5-7>9"])
def test_target_type_rejects_mixing_set_and_preference(raw):
    # iter-249: ',' (flat set) and '>' (ranked preference) are different
    # composition operators; stacking them in one target is ambiguous.
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type(raw)


def test_format_target_preference():
    # iter-249: a preference renders '>'-joined, each element via _format_target,
    # so it reads back exactly as typed.
    assert gv._format_target({"prefer": [3, 5, 7]}) == "3>5>7"
    assert gv._format_target({"prefer": [3, (5, 7)]}) == "3>5-7"
    assert gv._format_target({"prefer": [3, (5, None), (None, 7)]}) == "3>5->-7"


# ---- target_type: ':penalty' WEIGHTED-SET form (iter-250) --------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("3,5:2", {"weighted": [(3, 0), (5, 2)]}),
        ("3:1,5:2", {"weighted": [(3, 1), (5, 2)]}),
        (" 3 , 5 : 2 ", {"weighted": [(3, 0), (5, 2)]}),  # whitespace trimmed
        ("3,5-7:2", {"weighted": [(3, 0), ((5, 7), 2)]}),  # an element may be a band
        ("3,5-:2", {"weighted": [(3, 0), ((5, None), 2)]}),  # ...or an open band
        ("3:0,5:2", {"weighted": [(3, 0), (5, 2)]}),  # an explicit 0 penalty is fine
    ],
)
def test_target_type_weighted_parses_to_weighted_dict(raw, expected):
    # iter-250: a ':penalty' on a comma-set element parses to a {"weighted":
    # [(element, penalty), ...]} dict, an element with no ':' carrying penalty 0.
    value = gv.target_type(raw)
    assert value == expected
    assert isinstance(value, dict)
    assert "weighted" in value


def test_target_type_weighted_dedupes_on_element_first_penalty_wins():
    # iter-250: a repeated element collapses, the first-seen penalty kept.
    assert gv.target_type("3:1,3:2,5:4") == {"weighted": [(3, 1), (5, 4)]}


def test_target_type_single_element_weighted_collapses_to_bare_element():
    # iter-250: a weighted set collapsing to one element drops the (now useless)
    # penalty and reduces to the bare element, keeping scalar output unchanged.
    value = gv.target_type("3:2,3:5")
    assert value == 3
    assert isinstance(value, int)


@pytest.mark.parametrize("raw", ["3:2,", ",5:2", "3:2,,5", "3:2, ,5"])
def test_target_type_rejects_empty_weighted_element(raw):
    # iter-250: an empty element (trailing/leading/doubled ',') is a typo.
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type(raw)


@pytest.mark.parametrize("raw", ["3,5:a", "3,5:-1", "3,a:2", "3,5-3:2"])
def test_target_type_rejects_malformed_weighted_element(raw):
    # iter-250: a non-int/negative penalty, a bad base, or an inverted band element
    # fails the whole target.
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type(raw)


def test_target_type_rejects_weight_without_set():
    # iter-250: a ':' weight is meaningless on a single element (a constant offset
    # that cannot change a pick), so it requires the ',' set context.
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type("3:2")
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type("3-5:2")  # a lone band with a weight is still single


@pytest.mark.parametrize("raw", ["3,5:2>7", "3>5:2", "3:2,5>7"])
def test_target_type_rejects_mixing_weight_and_preference(raw):
    # iter-250: ':' (weight) and '>' (preference) both express preference; stacking
    # them in one target is ambiguous.
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type(raw)


def test_format_target_weighted():
    # iter-250: a weighted set renders comma-joined, each non-zero penalty appended
    # as ':penalty', so it reads back exactly as typed (a zero penalty stays bare).
    assert gv._format_target({"weighted": [(3, 0), (5, 2)]}) == "3,5:2"
    assert gv._format_target({"weighted": [(3, 1), (5, 2)]}) == "3:1,5:2"
    assert gv._format_target({"weighted": [(3, 0), ((5, 7), 2)]}) == "3,5-7:2"


# ---- nonneg_penalty_type / fractional weighted-set weight (iter-251) ----


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0", 0),
        ("2", 2),
        ("2.0", 2),  # an integral float collapses to int
        ("1.5", 1.5),
        ("0.25", 0.25),
    ],
)
def test_nonneg_penalty_parses_number(raw, expected):
    # iter-251: the weight slot accepts a non-negative float; an integral float
    # collapses to an int so iter-250's integer-penalty output is unchanged.
    value = gv.nonneg_penalty_type(raw)
    assert value == expected
    assert type(value) is type(expected)


@pytest.mark.parametrize("raw", ["-1", "-0.5", "nan", "inf", "abc", ""])
def test_nonneg_penalty_rejects_bad(raw):
    # iter-251: a negative, NaN, infinite, or non-numeric penalty is nonsensical.
    with pytest.raises(argparse.ArgumentTypeError):
        gv.nonneg_penalty_type(raw)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("3,5:1.5", {"weighted": [(3, 0), (5, 1.5)]}),
        ("3:0.5,5:1.5", {"weighted": [(3, 0.5), (5, 1.5)]}),
        ("3,5-7:1.5", {"weighted": [(3, 0), ((5, 7), 1.5)]}),  # a band may be fractionally weighted
        ("3,5:2.0", {"weighted": [(3, 0), (5, 2)]}),  # integral float collapses to int
    ],
)
def test_target_type_weighted_accepts_fractional_penalty(raw, expected):
    # iter-251: a ':penalty' weight may be fractional so preference strength dials
    # finely; an integral float collapses to int (byte-for-byte iter-250 output).
    value = gv.target_type(raw)
    assert value == expected
    # the genuinely fractional penalty stays a float; the collapsed one is an int.
    assert isinstance(value["weighted"][1][1], type(expected["weighted"][1][1]))


def test_format_target_weighted_fractional_reads_back_as_typed():
    # iter-251: a fractional penalty renders with its decimal, an integral one bare.
    assert gv._format_target({"weighted": [(3, 0), (5, 1.5)]}) == "3,5:1.5"
    assert gv._format_target({"weighted": [(3, 0.5), (5, 1.5)]}) == "3:0.5,5:1.5"


def test_grid_cell_distance_fractional_penalty_interpolates_threshold():
    # iter-251: THE point of the fractional weight. Preferred 3 (penalty 0),
    # accepted 6 (penalty 1.5). Count 6 lands exactly on the accepted element
    # (penalised 1.5); count 4 is raw dist 1 from preferred 3 (penalised 1) and
    # count 5 is raw dist 2 (penalised 2). The 1.5 penalty sits BETWEEN whole
    # steps: count 4 (1.0) beats count 6 (1.5) beats count 5 (2.0) — an integer
    # penalty of 1 or 2 could not place the boundary here.
    target = {"weighted": [(3, 0), (6, 1.5)]}
    assert gv.grid_cell_distance({"num_segments": 6}, target) == 1.5
    assert gv.grid_cell_distance({"num_segments": 4}, target) == 1
    assert gv.grid_cell_distance({"num_segments": 5}, target) == 2


def test_pick_best_grid_cell_fractional_penalty_orders_between_steps():
    # iter-251: with accepted-6 penalty 1.5, count 4 (penalised 1.0) wins over the
    # exact-accepted count 6 (penalised 1.5). A penalty of 2 would also flip it,
    # but 1.5 also keeps count 6 BELOW count 5 (penalised 2.0) — the fractional
    # value distinguishes an ordering an integer penalty cannot.
    cells = _seg_speech_cells([(6, 9.0), (4, 1.0), (5, 5.0)])
    best = gv.pick_best_grid_cell(cells, {"weighted": [(3, 0), (6, 1.5)]})
    assert best["num_segments"] == 4
    ranked = gv.pick_top_grid_cells(cells, {"weighted": [(3, 0), (6, 1.5)]}, 3)
    assert [c["num_segments"] for c in ranked] == [4, 6, 5]


def test_render_grid_json_carries_fractional_penalty():
    # iter-251: a fractional penalty serialises straight through as a JSON number.
    results = [_cell_result(n) for n in (4, 6)]
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [400.0, 800.0], results, name="rec.wav",
            target={"weighted": [(3, 0), (6, 1.5)]},
        )
    )
    assert payload["target"] == {"weighted": [[3, 0], [6, 1.5]]}
    assert payload["best"]["num_segments"] == 4
    assert payload["best"]["distance"] == 1


def test_render_grid_fractional_penalty_line_reads_back():
    # iter-251: the best: line renders the fractional weight as typed and shows the
    # penalised |Δ| (here count 4 → 1) with no dict repr leaking.
    results = [_cell_result(n) for n in (6, 4)]
    lines = gv.render_vad_grid(
        [0.3], [400.0, 800.0], results, name="rec.wav",
        target={"weighted": [(3, 0), (6, 1.5)]},
    )
    best_line = lines[-1]
    assert "4 segments" in best_line
    assert "|Δ|=1" in best_line
    assert "target 3,6:1.5" in best_line
    assert "weighted" not in best_line


# ---- max_speech_type: the force-split bound ----------------------------


@pytest.mark.parametrize("raw", ["inf", "none", "off", "NONE", "Off"])
def test_max_speech_sentinels_mean_infinity(raw):
    assert gv.max_speech_type(raw) == float("inf")


def test_max_speech_accepts_positive_float():
    assert gv.max_speech_type("12.5") == 12.5


@pytest.mark.parametrize("raw", ["0", "-5"])
def test_max_speech_rejects_zero_and_negative(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.max_speech_type(raw)


def test_max_speech_rejects_non_number():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.max_speech_type("forever")


def test_max_speech_rejects_nan():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.max_speech_type("nan")


# ---- max_speech_list_type: the --max-speeches seconds list (iter-255) ----


def test_max_speech_list_parses_seconds():
    assert gv.max_speech_list_type("5,10,20") == [5.0, 10.0, 20.0]


def test_max_speech_list_accepts_inf_sentinels_per_token():
    # Each token runs through max_speech_type, so the never-split sentinels
    # carry through per element — an operator can sweep the no-cap baseline in.
    assert gv.max_speech_list_type("5,inf,none,off") == [
        5.0, float("inf"), float("inf"), float("inf"),
    ]


def test_max_speech_list_strips_blanks_and_trailing_comma():
    assert gv.max_speech_list_type(" 5 , 10 ,") == [5.0, 10.0]


def test_max_speech_list_preserves_order_and_dupes():
    assert gv.max_speech_list_type("10,5,10") == [10.0, 5.0, 10.0]


def test_max_speech_list_rejects_zero_member():
    # A 0-second cap would force-split forever — rejected per token like the
    # scalar max_speech_type.
    with pytest.raises(argparse.ArgumentTypeError):
        gv.max_speech_list_type("5,0")


def test_max_speech_list_rejects_negative_member():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.max_speech_list_type("5,-1")


@pytest.mark.parametrize("raw", ["", " ", ",", " , "])
def test_max_speech_list_rejects_empty(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.max_speech_list_type(raw)


def test_max_speech_list_rejects_non_string():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.max_speech_list_type(5.0)


# ---- _format_sweep_axis_value: the seconds axis (iter-255) ---------------


def test_format_sweep_axis_value_seconds_is_compact():
    # %g: bare integers stay bare (no ".00"), fractionals keep the decimal.
    assert gv._format_sweep_axis_value("max_speech_s", 5.0) == "5"
    assert gv._format_sweep_axis_value("max_speech_s", 12.5) == "12.5"


def test_format_sweep_axis_value_seconds_renders_inf():
    assert gv._format_sweep_axis_value("max_speech_s", float("inf")) == "inf"


def test_max_speech_axis_label():
    assert gv._SWEEP_AXIS_LABEL["max_speech_s"] == "max_speech"


# ---- render_vad_segments: pure presentation ----------------------------


def test_render_none_is_install_hint():
    lines = gv.render_vad_segments(None)
    assert len(lines) == 1
    assert "silero-vad" in lines[0]


def test_render_empty_segments_notes_no_speech():
    result = _Result(name="quiet.wav", sample_rate=16000, duration_s=4.0)
    lines = gv.render_vad_segments(result, threshold=0.5)
    text = "\n".join(lines)
    assert "quiet.wav" in text
    assert "segments:     0" in text
    assert "no speech regions detected" in text


def test_render_lists_each_segment():
    result = _Result(
        name="rec.wav",
        sample_rate=48000,
        duration_s=31.3,
        segments=[_Seg(1.6, 2.1), _Seg(10.7, 18.5)],
    )
    lines = gv.render_vad_segments(result, threshold=0.5)
    text = "\n".join(lines)
    assert "rec.wav" in text
    assert "48000 Hz" in text
    assert "segments:     2" in text
    # speech total = 0.5 + 7.8 = 8.3s
    assert "8.3s" in text
    # both regions rendered with their durations
    assert "[ 1]" in text and "[ 2]" in text
    assert "1.60s" in text and "18.50s" in text


def test_render_omits_threshold_line_when_none():
    result = _Result(name="r.wav", sample_rate=16000, duration_s=1.0)
    lines = gv.render_vad_segments(result)  # no threshold kwarg
    assert not any("threshold" in ln for ln in lines)


# ---- cmd_vad: handler with injected deps -------------------------------


def _args(**over):
    base = dict(
        wav="rec.wav",
        threshold=0.5,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        json=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_cmd_vad_unavailable_prints_install_hint():
    lines: List[str] = []
    called = {"segmenter": False}

    def seg(*a, **k):  # should NOT be called when unavailable
        called["segmenter"] = True
        raise AssertionError("segmenter must not run when silero unavailable")

    gv.cmd_vad(
        _args(),
        log=lines.append,
        segmenter=seg,
        availability=lambda: False,
    )
    assert called["segmenter"] is False
    assert len(lines) == 1
    assert "silero-vad" in lines[0]


def test_cmd_vad_segments_and_renders():
    lines: List[str] = []
    captured = {}

    def seg(wav, params=None):
        captured["wav"] = wav
        captured["params"] = params
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=5.0,
            segments=[_Seg(0.5, 1.5), _Seg(2.0, 4.0)],
        )

    gv.cmd_vad(
        _args(threshold=0.6, min_silence_ms=500.0),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    # The WAV path was forwarded to the segmenter.
    assert captured["wav"] == "rec.wav"
    # SileroParams carried the CLI knobs through.
    params = captured["params"]
    assert params.threshold == 0.6
    assert params.min_silence_ms == 500.0
    # The rendered report names the file and both segments.
    text = "\n".join(lines)
    assert "rec.wav" in text
    assert "[ 1]" in text and "[ 2]" in text
    assert "threshold:    0.60" in text


def test_cmd_vad_builds_real_silero_params():
    # The params object the handler builds must be a genuine vad.silero
    # SileroParams (not a duck) so the field names stay in lock-step with the
    # engine. This is the one test that imports the engine; skip if absent.
    silero = pytest.importorskip("vad.silero")
    captured = {}

    def seg(wav, params=None):
        captured["params"] = params
        return _Result(name="r.wav", sample_rate=16000, duration_s=1.0)

    gv.cmd_vad(
        _args(threshold=0.4, min_speech_ms=120.0, max_speech_s=20.0),
        log=lambda *_: None,
        segmenter=seg,
        availability=lambda: True,
    )
    params = captured["params"]
    assert isinstance(params, silero.SileroParams)
    assert params.threshold == 0.4
    assert params.min_speech_ms == 120.0
    assert params.max_speech_s == 20.0


# ---- parser: the --json flag -------------------------------------------


def test_vad_json_defaults_false():
    args = gv.build_parser().parse_args(["vad", "rec.wav"])
    assert args.json is False


def test_vad_json_flag_sets_true():
    args = gv.build_parser().parse_args(["vad", "rec.wav", "--json"])
    assert args.json is True


# ---- render_vad_json: pure machine-readable presentation ---------------


def test_render_json_none_marks_unavailable():
    payload = json.loads(gv.render_vad_json(None))
    assert payload["available"] is False
    assert "silero-vad" in payload["hint"]
    # No segmentation keys leak onto the degraded payload.
    assert "segments" not in payload


def test_render_json_empty_segments():
    result = _Result(name="quiet.wav", sample_rate=16000, duration_s=4.0)
    payload = json.loads(gv.render_vad_json(result, threshold=0.5))
    assert payload["available"] is True
    assert payload["name"] == "quiet.wav"
    assert payload["sample_rate"] == 16000
    assert payload["num_segments"] == 0
    assert payload["speech_s"] == 0.0
    assert payload["segments"] == []
    assert payload["threshold"] == 0.5


def test_render_json_lists_each_segment():
    result = _Result(
        name="rec.wav",
        sample_rate=48000,
        duration_s=31.3,
        segments=[_Seg(1.6, 2.1), _Seg(10.7, 18.5)],
    )
    payload = json.loads(gv.render_vad_json(result, threshold=0.6))
    assert payload["num_segments"] == 2
    # speech total = 0.5 + 7.8 = 8.3s, rounded to 3 places
    assert payload["speech_s"] == 8.3
    segs = payload["segments"]
    assert len(segs) == 2
    assert segs[0] == {"start_s": 1.6, "end_s": 2.1, "duration_s": 0.5}
    assert segs[1]["start_s"] == 10.7 and segs[1]["end_s"] == 18.5
    assert payload["threshold"] == 0.6


def test_render_json_omits_threshold_when_none():
    result = _Result(name="r.wav", sample_rate=16000, duration_s=1.0)
    payload = json.loads(gv.render_vad_json(result))  # no threshold kwarg
    assert "threshold" not in payload


def test_render_json_rounds_to_three_places():
    # Sub-millisecond Silero boundaries must round to 3 places like to_dict().
    result = _Result(
        name="r.wav",
        sample_rate=16000,
        duration_s=1.23456,
        segments=[_Seg(0.123456, 0.987654)],
    )
    payload = json.loads(gv.render_vad_json(result))
    assert payload["duration_s"] == 1.235
    assert payload["segments"][0]["start_s"] == 0.123
    assert payload["segments"][0]["end_s"] == 0.988
    assert payload["segments"][0]["duration_s"] == 0.864


def test_render_json_matches_silero_to_dict_shape():
    # The hand-built payload must carry the same segmentation keys as the
    # engine's SileroResult.to_dict() so consumers can treat gv output and
    # the replay/server output interchangeably. Skip if the engine is absent.
    silero = pytest.importorskip("vad.silero")
    result = silero.SileroResult(
        name="rec.wav",
        sample_rate=16000,
        duration_s=5.0,
        segments=[silero.SpeechSegment(0.5, 1.5)],
    )
    via_to_dict = result.to_dict()
    via_render = json.loads(gv.render_vad_json(result))
    # Every key to_dict() emits is present in the render with equal values.
    for key, value in via_to_dict.items():
        assert via_render[key] == value


# ---- cmd_vad: the --json branch ----------------------------------------


def test_cmd_vad_json_unavailable_emits_unavailable_payload():
    lines: List[str] = []
    gv.cmd_vad(
        _args(json=True),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not segment when unavailable")
        ),
        availability=lambda: False,
    )
    # One JSON document, parseable, marking the degraded path.
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["available"] is False


def test_cmd_vad_json_emits_segmentation_payload():
    lines: List[str] = []

    def seg(wav, params=None):
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=5.0,
            segments=[_Seg(0.5, 1.5), _Seg(2.0, 4.0)],
        )

    gv.cmd_vad(
        _args(json=True, threshold=0.6),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["available"] is True
    assert payload["num_segments"] == 2
    assert payload["threshold"] == 0.6
    assert payload["segments"][0]["start_s"] == 0.5


def test_cmd_vad_without_json_stays_human_readable():
    # Regression guard: omitting --json keeps the multi-line text report, not
    # a single JSON blob.
    lines: List[str] = []

    def seg(wav, params=None):
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=5.0,
            segments=[_Seg(0.5, 1.5)],
        )

    gv.cmd_vad(_args(json=False), log=lines.append, segmenter=seg, availability=lambda: True)
    # Multiple human-readable lines, and the first is NOT valid JSON.
    assert len(lines) > 1
    with pytest.raises(json.JSONDecodeError):
        json.loads(lines[0])


# ====================================================================
# iter-235 — gv vad-diff: compare two thresholds (first gv vad --json consumer)
# ====================================================================


def _diff_args(**over):
    base = dict(
        wav="rec.wav",
        threshold_a=0.5,
        threshold_b=0.7,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        json=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


# ---- parser: registration & defaults -----------------------------------


def test_vad_diff_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-diff"] is gv.cmd_vad_diff


def test_vad_diff_defaults():
    args = gv.build_parser().parse_args(["vad-diff", "rec.wav"])
    assert args.command == "vad-diff"
    assert args.wav == "rec.wav"
    assert args.threshold_a == 0.5
    assert args.threshold_b == 0.7
    # Shared knobs default to the same SileroParams values as `gv vad`.
    assert args.min_speech_ms == 250.0
    assert args.min_silence_ms == 800.0
    assert args.speech_pad_ms == 30.0
    assert args.json is False


def test_vad_diff_overrides_thresholds():
    args = gv.build_parser().parse_args(
        ["vad-diff", "rec.wav", "--threshold-a", "0.3", "--threshold-b", "0.9"]
    )
    assert args.threshold_a == 0.3
    assert args.threshold_b == 0.9


def test_vad_diff_rejects_out_of_range_threshold():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-diff", "rec.wav", "--threshold-a", "1.5"])
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-diff", "rec.wav", "--threshold-b", "-0.1"])


def test_vad_diff_json_flag():
    args = gv.build_parser().parse_args(["vad-diff", "rec.wav", "--json"])
    assert args.json is True


# ---- vad_segmentation_delta: pure delta core ---------------------------


def test_delta_fewer_segments_at_higher_threshold():
    # A higher gate is typically a subset: fewer regions, less speech.
    a = _Result(
        name="r.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0), _Seg(5.0, 6.0)],
    )
    b = _Result(
        name="r.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0)],
    )
    d = gv.vad_segmentation_delta(a, b)
    assert d["num_segments_a"] == 3
    assert d["num_segments_b"] == 1
    assert d["num_segments_delta"] == -2
    assert d["speech_s_a"] == 3.0
    assert d["speech_s_b"] == 1.0
    assert d["speech_s_delta"] == -2.0


def test_delta_identical_segmentations_are_zero():
    segs = [_Seg(0.0, 1.0), _Seg(2.0, 3.5)]
    a = _Result(name="r.wav", sample_rate=16000, duration_s=5.0, segments=list(segs))
    b = _Result(name="r.wav", sample_rate=16000, duration_s=5.0, segments=list(segs))
    d = gv.vad_segmentation_delta(a, b)
    assert d["num_segments_delta"] == 0
    assert d["speech_s_delta"] == 0.0


def test_delta_positive_when_b_has_more():
    a = _Result(name="r.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    b = _Result(
        name="r.wav",
        sample_rate=16000,
        duration_s=5.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    d = gv.vad_segmentation_delta(a, b)
    assert d["num_segments_delta"] == 1
    assert d["speech_s_delta"] == 1.0


def test_delta_rounds_to_three_places():
    a = _Result(
        name="r.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 0.123456)]
    )
    b = _Result(
        name="r.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 0.987654)]
    )
    d = gv.vad_segmentation_delta(a, b)
    assert d["speech_s_a"] == 0.123
    assert d["speech_s_b"] == 0.988
    assert d["speech_s_delta"] == 0.865


# ---- render_vad_diff: human-readable -----------------------------------


def test_render_diff_none_marks_unavailable():
    lines = gv.render_vad_diff(None, None, label_a=0.5, label_b=0.7)
    assert len(lines) == 1
    assert "silero-vad" in lines[0]


def test_render_diff_one_none_marks_unavailable():
    r = _Result(name="r.wav", sample_rate=16000, duration_s=5.0)
    assert "silero-vad" in gv.render_vad_diff(r, None, label_a=0.5, label_b=0.7)[0]


def test_render_diff_shows_signed_deltas():
    a = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0), _Seg(5.0, 6.0)],
    )
    b = _Result(name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)])
    text = "\n".join(gv.render_vad_diff(a, b, label_a=0.5, label_b=0.7))
    assert "rec.wav" in text
    assert "0.50" in text and "0.70" in text
    assert "3 → 1" in text
    assert "(-2)" in text
    assert "(-2.0s)" in text


def test_render_diff_positive_delta_carries_plus_sign():
    a = _Result(name="rec.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    b = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=5.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    text = "\n".join(gv.render_vad_diff(a, b, label_a=0.3, label_b=0.5))
    assert "1 → 2" in text
    assert "(+1)" in text
    assert "(+1.0s)" in text


# ---- render_vad_diff_json: machine-readable ----------------------------


def test_render_diff_json_none_marks_unavailable():
    payload = json.loads(gv.render_vad_diff_json(None, None, label_a=0.5, label_b=0.7))
    assert payload["available"] is False
    assert "silero-vad" in payload["hint"]
    assert "num_segments_delta" not in payload


def test_render_diff_json_carries_both_sides_and_deltas():
    a = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0), _Seg(5.0, 6.0)],
    )
    b = _Result(name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)])
    payload = json.loads(gv.render_vad_diff_json(a, b, label_a=0.5, label_b=0.7))
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["threshold_a"] == 0.5
    assert payload["threshold_b"] == 0.7
    assert payload["num_segments_a"] == 3
    assert payload["num_segments_b"] == 1
    assert payload["num_segments_delta"] == -2
    assert payload["speech_s_delta"] == -2.0


# ---- cmd_vad_diff: the handler -----------------------------------------


def test_cmd_vad_diff_unavailable_emits_hint():
    lines: List[str] = []
    gv.cmd_vad_diff(
        _diff_args(),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not segment when unavailable")
        ),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert "silero-vad" in lines[0]


def test_cmd_vad_diff_unavailable_json():
    lines: List[str] = []
    gv.cmd_vad_diff(
        _diff_args(json=True),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert json.loads(lines[0])["available"] is False


def test_cmd_vad_diff_runs_both_thresholds():
    # The handler must segment twice — once per threshold — forwarding the
    # shared knobs both times. We capture the threshold of each call.
    seen = []

    def seg(wav, params=None):
        seen.append(params.threshold)
        # Higher threshold → fewer segments (subset behaviour).
        n = 3 if params.threshold < 0.6 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_diff(
        _diff_args(threshold_a=0.5, threshold_b=0.7),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert seen == [0.5, 0.7]  # A first, then B
    text = "\n".join(lines)
    assert "3 → 1" in text
    assert "(-2)" in text


def test_cmd_vad_diff_json_branch():
    def seg(wav, params=None):
        n = 3 if params.threshold < 0.6 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_diff(
        _diff_args(threshold_a=0.5, threshold_b=0.7, json=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["num_segments_delta"] == -2
    assert payload["threshold_a"] == 0.5
    assert payload["threshold_b"] == 0.7


def test_cmd_vad_diff_shares_knobs_across_both_runs():
    # Both runs must carry the SAME min_speech_ms / max_speech_s — only the
    # threshold differs. Build genuine SileroParams to lock field names.
    pytest.importorskip("vad.silero")
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        return _Result(name="rec.wav", sample_rate=16000, duration_s=1.0)

    gv.cmd_vad_diff(
        _diff_args(threshold_a=0.4, threshold_b=0.8, min_speech_ms=120.0, max_speech_s=20.0),
        log=lambda *_: None,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(captured) == 2
    assert [p.threshold for p in captured] == [0.4, 0.8]
    # Shared knobs identical across both runs.
    assert captured[0].min_speech_ms == captured[1].min_speech_ms == 120.0
    assert captured[0].max_speech_s == captured[1].max_speech_s == 20.0


# ====================================================================
# iter-236 — gv vad-sweep: tabulate segmentation over N thresholds
# ====================================================================


def _sweep_args(**over):
    base = dict(
        wav="rec.wav",
        thresholds=[0.3, 0.5, 0.7, 0.9],
        min_silences=None,
        min_speeches=None,
        speech_pads=None,
        max_speeches=None,
        threshold=0.5,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        target=None,
        top=None,
        tie_break="row-major",
        json=False,
        csv=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


# ---- unit_interval_list_type: the --thresholds validator ---------------


def test_threshold_list_parses_comma_separated():
    values = gv.unit_interval_list_type("0.3,0.5,0.7,0.9")
    assert values == [0.3, 0.5, 0.7, 0.9]
    assert all(isinstance(v, float) for v in values)


def test_threshold_list_strips_whitespace_and_blanks():
    assert gv.unit_interval_list_type(" 0.2 , 0.8 ,") == [0.2, 0.8]


def test_threshold_list_preserves_order_and_duplicates():
    # The operator picks the column order; we don't sort or dedupe.
    assert gv.unit_interval_list_type("0.9,0.1,0.9") == [0.9, 0.1, 0.9]


@pytest.mark.parametrize("raw", ["", "  ", ",", " , "])
def test_threshold_list_rejects_empty(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.unit_interval_list_type(raw)


@pytest.mark.parametrize("raw", ["0.3,1.5", "0.3,-0.1", "2"])
def test_threshold_list_rejects_out_of_range_member(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.unit_interval_list_type(raw)


@pytest.mark.parametrize("raw", ["0.3,high", "x,0.5", "0.3,nan"])
def test_threshold_list_rejects_non_number_member(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.unit_interval_list_type(raw)


def test_threshold_list_rejects_non_string():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.unit_interval_list_type(0.5)


# ---- parser: registration & defaults -----------------------------------


def test_vad_sweep_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-sweep"] is gv.cmd_vad_sweep


def test_vad_sweep_defaults():
    args = gv.build_parser().parse_args(["vad-sweep", "rec.wav"])
    assert args.command == "vad-sweep"
    assert args.wav == "rec.wav"
    assert args.thresholds == [0.3, 0.5, 0.7, 0.9]
    # Shared knobs default to the same SileroParams values as `gv vad`.
    assert args.min_speech_ms == 250.0
    assert args.min_silence_ms == 800.0
    assert args.speech_pad_ms == 30.0
    assert args.max_speech_s == float("inf")
    assert args.json is False
    assert args.csv is False


def test_vad_sweep_overrides_thresholds():
    args = gv.build_parser().parse_args(
        ["vad-sweep", "rec.wav", "--thresholds", "0.1,0.6,0.95"]
    )
    assert args.thresholds == [0.1, 0.6, 0.95]


def test_vad_sweep_rejects_out_of_range_threshold():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-sweep", "rec.wav", "--thresholds", "0.5,1.5"])


def test_vad_sweep_json_flag():
    args = gv.build_parser().parse_args(["vad-sweep", "rec.wav", "--json"])
    assert args.json is True


def test_vad_sweep_csv_flag():
    args = gv.build_parser().parse_args(["vad-sweep", "rec.wav", "--csv"])
    assert args.csv is True
    assert args.json is False


def test_vad_sweep_json_and_csv_are_mutually_exclusive():
    # Two output formats can't both win; argparse rejects the combination with
    # the usual SystemExit(2) rather than silently picking one.
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-sweep", "rec.wav", "--json", "--csv"])


# ---- vad_segmentation_sweep: pure core ---------------------------------


def test_sweep_pairs_each_threshold_with_summary():
    r_lo = _Result(
        name="r.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0), _Seg(5.0, 6.0)],
    )
    r_hi = _Result(
        name="r.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    rows = gv.vad_segmentation_sweep([0.3, 0.9], [r_lo, r_hi])
    assert rows == [
        {"threshold": 0.3, "num_segments": 3, "speech_s": 3.0},
        {"threshold": 0.9, "num_segments": 1, "speech_s": 1.0},
    ]


def test_sweep_rounds_speech_to_three_places():
    r = _Result(
        name="r.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 0.123456)]
    )
    rows = gv.vad_segmentation_sweep([0.5], [r])
    assert rows[0]["speech_s"] == 0.123


def test_sweep_length_mismatch_raises():
    r = _Result(name="r.wav", sample_rate=16000, duration_s=5.0)
    with pytest.raises(ValueError):
        gv.vad_segmentation_sweep([0.3, 0.5], [r])


# ---- render_vad_sweep: human-readable ----------------------------------


def test_render_sweep_none_marks_unavailable():
    lines = gv.render_vad_sweep([], [None], name="rec.wav")
    assert len(lines) == 1
    assert "silero-vad" in lines[0]


def test_render_sweep_any_none_marks_unavailable():
    r = _Result(name="r.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    lines = gv.render_vad_sweep([0.3, 0.9], [r, None], name="rec.wav")
    assert "silero-vad" in lines[0]


def test_render_sweep_tabulates_each_threshold():
    r_lo = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0), _Seg(5.0, 6.0)],
    )
    r_hi = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    lines = gv.render_vad_sweep([0.3, 0.9], [r_lo, r_hi], name="rec.wav")
    text = "\n".join(lines)
    assert "rec.wav" in text
    assert "threshold" in text and "segments" in text and "speech" in text
    # one header line, one column-label line, one row per threshold
    assert len(lines) == 4
    assert "0.30" in text and "0.90" in text
    assert "3" in text and "1" in text


# ---- render_vad_sweep_json: machine-readable ---------------------------


def test_render_sweep_json_none_marks_unavailable():
    payload = json.loads(gv.render_vad_sweep_json([], [None], name="rec.wav"))
    assert payload["available"] is False
    assert "silero-vad" in payload["hint"]
    assert "sweep" not in payload


def test_render_sweep_json_carries_rows():
    r_lo = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0), _Seg(5.0, 6.0)],
    )
    r_hi = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    payload = json.loads(gv.render_vad_sweep_json([0.3, 0.9], [r_lo, r_hi], name="rec.wav"))
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["sweep"] == [
        {"threshold": 0.3, "num_segments": 3, "speech_s": 3.0},
        {"threshold": 0.9, "num_segments": 1, "speech_s": 1.0},
    ]


# ---- cmd_vad_sweep: the handler ----------------------------------------


def test_cmd_vad_sweep_unavailable_emits_hint():
    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not segment when unavailable")
        ),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert "silero-vad" in lines[0]


def test_cmd_vad_sweep_unavailable_json():
    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(json=True),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert json.loads(lines[0])["available"] is False


def test_cmd_vad_sweep_runs_every_threshold_in_order():
    seen = []

    def seg(wav, params=None):
        seen.append(params.threshold)
        # Higher threshold → fewer segments (subset behaviour).
        n = 3 if params.threshold < 0.6 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.2, 0.5, 0.8]),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert seen == [0.2, 0.5, 0.8]  # swept in order
    text = "\n".join(lines)
    assert "0.20" in text and "0.50" in text and "0.80" in text


def test_cmd_vad_sweep_json_branch():
    def seg(wav, params=None):
        n = 3 if params.threshold < 0.6 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.3, 0.9], json=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert [row["threshold"] for row in payload["sweep"]] == [0.3, 0.9]
    assert payload["sweep"][0]["num_segments"] == 3
    assert payload["sweep"][1]["num_segments"] == 1


def test_cmd_vad_sweep_shares_knobs_across_all_runs():
    # Every run must carry the SAME min_speech_ms / max_speech_s — only the
    # threshold differs. Build genuine SileroParams to lock field names.
    pytest.importorskip("vad.silero")
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        return _Result(name="rec.wav", sample_rate=16000, duration_s=1.0)

    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.2, 0.6, 0.95], min_speech_ms=120.0, max_speech_s=20.0),
        log=lambda *_: None,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(captured) == 3
    assert [p.threshold for p in captured] == [0.2, 0.6, 0.95]
    assert {p.min_speech_ms for p in captured} == {120.0}
    assert {p.max_speech_s for p in captured} == {20.0}


# ====================================================================
# iter-237 — gv vad-sweep --csv: flat spreadsheet/plot-friendly table
# ====================================================================


# ---- render_vad_sweep_csv: machine-readable CSV ------------------------


def test_render_sweep_csv_none_marks_unavailable():
    text = gv.render_vad_sweep_csv([], [None], name="rec.wav")
    # A degraded run is a single self-describing comment, not empty output.
    assert text.startswith("#")
    assert "silero-vad" in text


def test_render_sweep_csv_any_none_marks_unavailable():
    r = _Result(name="r.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    text = gv.render_vad_sweep_csv([0.3, 0.9], [r, None], name="rec.wav")
    assert text.startswith("#")
    assert "silero-vad" in text


def test_render_sweep_csv_header_and_rows():
    r_lo = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0), _Seg(5.0, 6.0)],
    )
    r_hi = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    text = gv.render_vad_sweep_csv([0.3, 0.9], [r_lo, r_hi], name="rec.wav")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["threshold", "num_segments", "speech_s"]
    assert rows[1] == ["0.3", "3", "3.0"]
    assert rows[2] == ["0.9", "1", "1.0"]
    # exactly the header + one row per threshold, nothing else.
    assert len(rows) == 3


def test_render_sweep_csv_no_trailing_newline():
    r = _Result(name="rec.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    text = gv.render_vad_sweep_csv([0.5], [r], name="rec.wav")
    # The renderer is pure text the caller logs; it must not carry a trailing
    # blank line (csv.writer's \r\n terminator stripped).
    assert not text.endswith("\n")
    assert not text.endswith("\r")


def test_render_sweep_csv_round_trips_to_sweep_rows():
    # The CSV body must describe the SAME segmentation as the JSON twin, so a
    # consumer reading either surface sees identical numbers.
    r_lo = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    r_hi = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    thresholds = [0.3, 0.9]
    results = [r_lo, r_hi]
    csv_text = gv.render_vad_sweep_csv(thresholds, results, name="rec.wav")
    json_rows = json.loads(
        gv.render_vad_sweep_json(thresholds, results, name="rec.wav")
    )["sweep"]
    csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert [
        {
            "threshold": float(row["threshold"]),
            "num_segments": int(row["num_segments"]),
            "speech_s": float(row["speech_s"]),
        }
        for row in csv_rows
    ] == json_rows


# ---- 1-D sweep CSV ↔ JSON round-trip on a non-default MS axis (iter-268) -
#
# iter-267 backlog item #4: the cross-surface round-trip above
# (test_render_sweep_csv_round_trips_to_sweep_rows) proves the CSV body
# describes the SAME segmentation as the JSON twin — but ONLY on the default
# `threshold` axis (it omits the `axis=` kwarg). The seconds axis got its own
# round-trip (test_render_sweep_csv_max_speech_round_trips_with_inf), but that
# one compares the CSV against vad_segmentation_sweep cells, NOT the JSON twin,
# and the millisecond axes (min_silence_ms / min_speech_ms / speech_pad_ms) had
# only single-row HEADER tests (test_render_sweep_csv_header_is_*_axis_name) —
# no multi-row CSV↔JSON cross-surface round-trip on a non-default ms axis.
# render_vad_sweep_csv and render_vad_sweep_json are axis-agnostic (each
# stringifies whichever value the `axis` kwarg names), so a regression that let
# the CSV's first column drift from the JSON row key, truncated a later ms-axis
# value, or disagreed on num_segments/speech_s across rows would have shipped
# green while the threshold-only round-trip and the single-row header tests
# stayed passing. These two close that hole on the 1-D sweep: a multi-row sweep
# on min_speech_ms and on speech_pad_ms, each asserting the CSV DictReader rows
# parse back to the EXACT JSON `sweep` payload. No production code changed (the
# wiring was already correct — proved by a pre-test smoke run).


def test_render_sweep_csv_min_speech_axis_round_trips_to_json_twin():
    # A multi-row min_speech_ms sweep: the CSV body must describe the SAME
    # segmentation as the JSON twin on this NON-default axis, so a consumer
    # reading either surface recovers identical numbers across every row.
    r_lo = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0), _Seg(4.0, 5.0)],
    )
    r_mid = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    r_hi = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0,
        segments=[_Seg(0.0, 1.0)],
    )
    floors = [200.0, 400.0, 800.0]
    results = [r_lo, r_mid, r_hi]
    csv_text = gv.render_vad_sweep_csv(
        floors, results, name="rec.wav", axis="min_speech_ms"
    )
    json_rows = json.loads(
        gv.render_vad_sweep_json(floors, results, name="rec.wav", axis="min_speech_ms")
    )["sweep"]
    csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
    # The first column header IS the swept axis name (self-describing CSV).
    assert csv_rows[0].keys() >= {"min_speech_ms", "num_segments", "speech_s"}
    assert [
        {
            "min_speech_ms": float(row["min_speech_ms"]),
            "num_segments": int(row["num_segments"]),
            "speech_s": float(row["speech_s"]),
        }
        for row in csv_rows
    ] == json_rows


def test_render_sweep_csv_speech_pad_axis_round_trips_to_json_twin():
    # The speech_pad_ms twin of the above: a different ms axis name keys both
    # surfaces, and the multi-row CSV↔JSON agreement must hold there too.
    r_lo = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    r_hi = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0,
        segments=[_Seg(0.0, 1.5)],
    )
    pads = [30.0, 90.0]
    results = [r_lo, r_hi]
    csv_text = gv.render_vad_sweep_csv(
        pads, results, name="rec.wav", axis="speech_pad_ms"
    )
    json_rows = json.loads(
        gv.render_vad_sweep_json(pads, results, name="rec.wav", axis="speech_pad_ms")
    )["sweep"]
    csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert csv_rows[0].keys() >= {"speech_pad_ms", "num_segments", "speech_s"}
    assert [
        {
            "speech_pad_ms": float(row["speech_pad_ms"]),
            "num_segments": int(row["num_segments"]),
            "speech_s": float(row["speech_s"]),
        }
        for row in csv_rows
    ] == json_rows


def test_render_sweep_csv_min_silence_axis_round_trips_to_json_twin():
    # iter-269: the third and LAST ms axis. iter-268 left min_silence_ms as the
    # only untested 1-D CSV↔JSON cross-surface round-trip (it shipped the
    # min_speech_ms and speech_pad_ms twins above). min_silence_ms is the most
    # common non-default sweep axis (it tunes the trailing-silence gate that
    # decides where one utterance ends and the next begins), yet its only prior
    # coverage was the single-row header test
    # (test_render_sweep_csv_header_is_axis_name). Closing it completes the
    # axis-agnostic round-trip matrix: threshold (default), seconds (max_speech_s),
    # and all three ms axes now prove the CSV body and JSON twin describe the
    # SAME segmentation across every row.
    r_lo = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0), _Seg(4.0, 5.0)],
    )
    r_mid = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    r_hi = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0,
        segments=[_Seg(0.0, 1.0)],
    )
    floors = [100.0, 300.0, 700.0]
    results = [r_lo, r_mid, r_hi]
    csv_text = gv.render_vad_sweep_csv(
        floors, results, name="rec.wav", axis="min_silence_ms"
    )
    json_rows = json.loads(
        gv.render_vad_sweep_json(floors, results, name="rec.wav", axis="min_silence_ms")
    )["sweep"]
    csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert csv_rows[0].keys() >= {"min_silence_ms", "num_segments", "speech_s"}
    assert [
        {
            "min_silence_ms": float(row["min_silence_ms"]),
            "num_segments": int(row["num_segments"]),
            "speech_s": float(row["speech_s"]),
        }
        for row in csv_rows
    ] == json_rows


# ---- cmd_vad_sweep --csv: the handler ----------------------------------


def test_cmd_vad_sweep_csv_unavailable_emits_comment():
    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(csv=True),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert lines[0].startswith("#")
    assert "silero-vad" in lines[0]


def test_cmd_vad_sweep_csv_branch():
    def seg(wav, params=None):
        n = 3 if params.threshold < 0.6 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.3, 0.9], csv=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    # One CSV blob logged in a single call (not line-by-line like the table).
    assert len(lines) == 1
    rows = list(csv.reader(io.StringIO(lines[0])))
    assert rows[0] == ["threshold", "num_segments", "speech_s"]
    assert [row[0] for row in rows[1:]] == ["0.3", "0.9"]
    assert rows[1][1] == "3"
    assert rows[2][1] == "1"


def test_cmd_vad_sweep_csv_uses_segmenter_name():
    # CSV body is a pure data grid — the segmenter's basename name does not leak
    # into the rows (signature parity only). Verify the table is exactly
    # header + data, no name column.
    def seg(wav, params=None):
        return _Result(
            name="basename.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(0.0, 1.0)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.5], csv=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert "basename.wav" not in lines[0]
    rows = list(csv.reader(io.StringIO(lines[0])))
    assert len(rows) == 2  # header + single threshold row


# ====================================================================
# iter-238 — gv vad-sweep --min-silences: a second sweep axis (hangover)
# ====================================================================


# ---- nonneg_float_list_type: the --min-silences validator --------------


def test_min_silences_list_parses_comma_separated():
    assert gv.nonneg_float_list_type("400,600,800,1000") == [400.0, 600.0, 800.0, 1000.0]


def test_min_silences_list_strips_whitespace_and_blanks():
    assert gv.nonneg_float_list_type(" 400 , 800 ,") == [400.0, 800.0]


def test_min_silences_list_preserves_order_and_duplicates():
    assert gv.nonneg_float_list_type("800,400,800") == [800.0, 400.0, 800.0]


def test_min_silences_list_allows_zero():
    # 0 ms is legitimate (disable the minimum hangover), unlike thresholds which
    # are bounded to [0, 1] — here only negatives are rejected.
    assert gv.nonneg_float_list_type("0,400") == [0.0, 400.0]


@pytest.mark.parametrize("raw", ["", "  ", ",", " , "])
def test_min_silences_list_rejects_empty(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.nonneg_float_list_type(raw)


@pytest.mark.parametrize("raw", ["400,-1", "-50"])
def test_min_silences_list_rejects_negative_member(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.nonneg_float_list_type(raw)


@pytest.mark.parametrize("raw", ["400,abc", "x"])
def test_min_silences_list_rejects_non_number_member(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        gv.nonneg_float_list_type(raw)


def test_min_silences_list_rejects_non_string():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.nonneg_float_list_type(800.0)


# ---- parser wiring: --min-silences axis --------------------------------


def test_vad_sweep_min_silences_parses():
    args = gv.build_parser().parse_args(
        ["vad-sweep", "rec.wav", "--min-silences", "400,600,800"]
    )
    assert args.min_silences == [400.0, 600.0, 800.0]
    # The threshold list keeps its default; the handler picks the silence axis.
    assert args.thresholds == [0.3, 0.5, 0.7, 0.9]


def test_vad_sweep_min_silences_default_is_none():
    # Without --min-silences the silence axis is off (None), so the handler
    # sweeps --thresholds (the iter-236 default).
    args = gv.build_parser().parse_args(["vad-sweep", "rec.wav"])
    assert args.min_silences is None


def test_vad_sweep_scalar_threshold_default_and_override():
    args = gv.build_parser().parse_args(["vad-sweep", "rec.wav"])
    assert args.threshold == 0.5
    args = gv.build_parser().parse_args(
        ["vad-sweep", "rec.wav", "--min-silences", "400,800", "--threshold", "0.7"]
    )
    assert args.threshold == 0.7


def test_vad_sweep_thresholds_and_min_silences_mutually_exclusive():
    # The two sweep axes can't both win; argparse rejects the combination with
    # SystemExit(2) rather than silently picking one.
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--thresholds", "0.3,0.5", "--min-silences", "400,800"]
        )


def test_vad_sweep_rejects_negative_min_silence_member():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--min-silences", "400,-1"]
        )


# ---- vad_segmentation_sweep: axis parameter ----------------------------


def test_sweep_axis_keys_rows_by_axis_name():
    r_lo = _Result(
        name="r.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    r_hi = _Result(
        name="r.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    rows = gv.vad_segmentation_sweep([400.0, 800.0], [r_lo, r_hi], axis="min_silence_ms")
    assert rows == [
        {"min_silence_ms": 400.0, "num_segments": 2, "speech_s": 2.0},
        {"min_silence_ms": 800.0, "num_segments": 1, "speech_s": 1.0},
    ]


def test_sweep_axis_defaults_to_threshold():
    # Omitting axis keeps the iter-236 row shape so old callers are unchanged.
    r = _Result(name="r.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    rows = gv.vad_segmentation_sweep([0.5], [r])
    assert rows == [{"threshold": 0.5, "num_segments": 1, "speech_s": 1.0}]


# ---- renderers: silence axis -------------------------------------------


def test_render_sweep_silence_axis_labels_column():
    r_lo = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    r_hi = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    lines = gv.render_vad_sweep(
        [400.0, 800.0], [r_lo, r_hi], name="rec.wav", axis="min_silence_ms"
    )
    text = "\n".join(lines)
    assert "min_silence" in text
    assert "threshold" not in text
    # Hangover values print as bare integers (400, 800), not 0.40 gates.
    assert "400" in text and "800" in text
    assert "0.40" not in text


def test_render_sweep_json_carries_axis():
    r = _Result(name="rec.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    payload = json.loads(
        gv.render_vad_sweep_json([400.0], [r], name="rec.wav", axis="min_silence_ms")
    )
    assert payload["axis"] == "min_silence_ms"
    assert payload["sweep"] == [
        {"min_silence_ms": 400.0, "num_segments": 1, "speech_s": 1.0}
    ]


def test_render_sweep_json_axis_defaults_to_threshold():
    r = _Result(name="rec.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    payload = json.loads(gv.render_vad_sweep_json([0.5], [r], name="rec.wav"))
    assert payload["axis"] == "threshold"


def test_render_sweep_csv_header_is_axis_name():
    r = _Result(name="rec.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    text = gv.render_vad_sweep_csv([400.0], [r], name="rec.wav", axis="min_silence_ms")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["min_silence_ms", "num_segments", "speech_s"]
    assert rows[1] == ["400.0", "1", "1.0"]


# ---- cmd_vad_sweep: silence axis end-to-end ----------------------------


def test_cmd_vad_sweep_silence_axis_sweeps_hangover():
    # When --min-silences is set, the segmenter sees the SWEPT min_silence_ms and
    # the gate held at scalar --threshold; --min-silence-ms is then ignored.
    pytest.importorskip("vad.silero")
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        # Longer hangover merges regions → fewer segments.
        n = 3 if params.min_silence_ms < 600 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(min_silences=[400.0, 800.0], threshold=0.7, min_silence_ms=999.0),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert [p.min_silence_ms for p in captured] == [400.0, 800.0]
    # Gate held at scalar --threshold for every run; the shared --min-silence-ms
    # scalar (999) is NOT used as a swept value.
    assert {p.threshold for p in captured} == {0.7}
    text = "\n".join(lines)
    assert "min_silence" in text
    assert "400" in text and "800" in text


def test_cmd_vad_sweep_silence_axis_json_branch():
    def seg(wav, params=None):
        n = 3 if params.min_silence_ms < 600 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(min_silences=[400.0, 800.0], json=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["axis"] == "min_silence_ms"
    assert [row["min_silence_ms"] for row in payload["sweep"]] == [400.0, 800.0]
    assert payload["sweep"][0]["num_segments"] == 3
    assert payload["sweep"][1]["num_segments"] == 1


def test_cmd_vad_sweep_silence_axis_csv_branch():
    def seg(wav, params=None):
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(0.0, 1.0)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(min_silences=[400.0, 800.0], csv=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    rows = list(csv.reader(io.StringIO(lines[0])))
    assert rows[0] == ["min_silence_ms", "num_segments", "speech_s"]
    assert [row[0] for row in rows[1:]] == ["400.0", "800.0"]


def test_cmd_vad_sweep_silence_axis_unavailable_uses_axis_label():
    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(min_silences=[400.0, 800.0], json=True),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert json.loads(lines[0])["available"] is False


# ====================================================================
# iter-239 — gv vad-sweep --min-speeches: a third sweep axis (floor)
# ====================================================================


# ---- parser wiring: --min-speeches axis --------------------------------


def test_vad_sweep_min_speeches_parses():
    args = gv.build_parser().parse_args(
        ["vad-sweep", "rec.wav", "--min-speeches", "50,100,200"]
    )
    assert args.min_speeches == [50.0, 100.0, 200.0]
    # The threshold list keeps its default; the handler picks the speech axis.
    assert args.thresholds == [0.3, 0.5, 0.7, 0.9]


def test_vad_sweep_min_speeches_default_is_none():
    # Without --min-speeches the speech axis is off (None), so the handler
    # sweeps --thresholds (the iter-236 default).
    args = gv.build_parser().parse_args(["vad-sweep", "rec.wav"])
    assert args.min_speeches is None


def test_vad_sweep_min_speeches_allows_zero():
    # 0 ms is legitimate (disable the floor — keep every region).
    args = gv.build_parser().parse_args(
        ["vad-sweep", "rec.wav", "--min-speeches", "0,100"]
    )
    assert args.min_speeches == [0.0, 100.0]


def test_vad_sweep_thresholds_and_min_speeches_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--thresholds", "0.3,0.5", "--min-speeches", "50,100"]
        )


def test_vad_sweep_min_silences_and_min_speeches_mutually_exclusive():
    # The two ms axes are also mutually exclusive — only one knob varies per run.
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--min-silences", "400,800", "--min-speeches", "50,100"]
        )


def test_vad_sweep_rejects_negative_min_speech_member():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--min-speeches", "100,-1"]
        )


# ---- vad_segmentation_sweep / renderers: speech axis -------------------


def test_sweep_axis_keys_rows_by_speech_axis_name():
    r_lo = _Result(
        name="r.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    r_hi = _Result(
        name="r.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    rows = gv.vad_segmentation_sweep([50.0, 400.0], [r_lo, r_hi], axis="min_speech_ms")
    assert rows == [
        {"min_speech_ms": 50.0, "num_segments": 2, "speech_s": 2.0},
        {"min_speech_ms": 400.0, "num_segments": 1, "speech_s": 1.0},
    ]


def test_render_sweep_speech_axis_labels_column():
    r_lo = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    r_hi = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    lines = gv.render_vad_sweep(
        [50.0, 400.0], [r_lo, r_hi], name="rec.wav", axis="min_speech_ms"
    )
    text = "\n".join(lines)
    assert "min_speech" in text
    assert "min_silence" not in text
    # Floor values print as bare integers (50, 400), not 0.50 gates.
    assert "50" in text and "400" in text
    assert "0.50" not in text


def test_render_sweep_json_carries_speech_axis():
    r = _Result(name="rec.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    payload = json.loads(
        gv.render_vad_sweep_json([100.0], [r], name="rec.wav", axis="min_speech_ms")
    )
    assert payload["axis"] == "min_speech_ms"
    assert payload["sweep"] == [
        {"min_speech_ms": 100.0, "num_segments": 1, "speech_s": 1.0}
    ]


def test_render_sweep_csv_header_is_speech_axis_name():
    r = _Result(name="rec.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    text = gv.render_vad_sweep_csv([100.0], [r], name="rec.wav", axis="min_speech_ms")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["min_speech_ms", "num_segments", "speech_s"]
    assert rows[1] == ["100.0", "1", "1.0"]


# ---- cmd_vad_sweep: speech axis end-to-end -----------------------------


def test_cmd_vad_sweep_speech_axis_sweeps_floor():
    # When --min-speeches is set, the segmenter sees the SWEPT min_speech_ms and
    # the gate held at scalar --threshold; the scalar --min-speech-ms is ignored.
    pytest.importorskip("vad.silero")
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        # A higher floor drops more short regions → fewer segments.
        n = 3 if params.min_speech_ms < 200 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(min_speeches=[50.0, 400.0], threshold=0.7, min_speech_ms=999.0),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert [p.min_speech_ms for p in captured] == [50.0, 400.0]
    # Gate held at scalar --threshold for every run; the shared --min-speech-ms
    # scalar (999) is NOT used as a swept value.
    assert {p.threshold for p in captured} == {0.7}
    text = "\n".join(lines)
    assert "min_speech" in text
    assert "50" in text and "400" in text


def test_cmd_vad_sweep_speech_axis_holds_silence_scalar():
    # The non-swept ms knob (--min-silence-ms) is shared across every run.
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        return _Result(
            name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
        )

    gv.cmd_vad_sweep(
        _sweep_args(min_speeches=[50.0, 400.0], min_silence_ms=750.0),
        log=lambda *a: None,
        segmenter=seg,
        availability=lambda: True,
    )
    assert {p.min_silence_ms for p in captured} == {750.0}


def test_cmd_vad_sweep_speech_axis_json_branch():
    def seg(wav, params=None):
        n = 3 if params.min_speech_ms < 200 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(min_speeches=[50.0, 400.0], json=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["axis"] == "min_speech_ms"
    assert [row["min_speech_ms"] for row in payload["sweep"]] == [50.0, 400.0]
    assert payload["sweep"][0]["num_segments"] == 3
    assert payload["sweep"][1]["num_segments"] == 1


def test_cmd_vad_sweep_speech_axis_csv_branch():
    def seg(wav, params=None):
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(0.0, 1.0)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(min_speeches=[50.0, 400.0], csv=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    rows = list(csv.reader(io.StringIO(lines[0])))
    assert rows[0] == ["min_speech_ms", "num_segments", "speech_s"]
    assert [row[0] for row in rows[1:]] == ["50.0", "400.0"]


def test_cmd_vad_sweep_speech_axis_unavailable():
    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(min_speeches=[50.0, 400.0], json=True),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert json.loads(lines[0])["available"] is False


# ====================================================================
# iter-253 — gv vad-sweep --speech-pads: a fourth sweep axis (padding)
# ====================================================================
# The symmetric region padding (speech_pad_ms) was held fixed across all runs;
# iter-253 promotes it to a fourth sweepable axis alongside the gate, hangover,
# and floor, mirroring the iter-238/239 axis-addition pattern. Padding values
# are non-negative floats; the gate is held at scalar --threshold while it
# sweeps; the four axes are mutually exclusive.


# ---- parser wiring: --speech-pads axis ---------------------------------


def test_vad_sweep_speech_pads_parses():
    args = gv.build_parser().parse_args(
        ["vad-sweep", "rec.wav", "--speech-pads", "0,20,40,60"]
    )
    assert args.speech_pads == [0.0, 20.0, 40.0, 60.0]
    # The threshold list keeps its default; the handler picks the pad axis.
    assert args.thresholds == [0.3, 0.5, 0.7, 0.9]


def test_vad_sweep_speech_pads_default_is_none():
    # Without --speech-pads the pad axis is off (None), so the handler sweeps
    # --thresholds (the iter-236 default).
    args = gv.build_parser().parse_args(["vad-sweep", "rec.wav"])
    assert args.speech_pads is None


def test_vad_sweep_speech_pads_allows_zero():
    # 0 ms is legitimate (no padding — the unpadded region boundaries).
    args = gv.build_parser().parse_args(
        ["vad-sweep", "rec.wav", "--speech-pads", "0,40"]
    )
    assert args.speech_pads == [0.0, 40.0]


def test_vad_sweep_thresholds_and_speech_pads_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--thresholds", "0.3,0.5", "--speech-pads", "0,40"]
        )


def test_vad_sweep_min_silences_and_speech_pads_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--min-silences", "400,800", "--speech-pads", "0,40"]
        )


def test_vad_sweep_min_speeches_and_speech_pads_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--min-speeches", "50,100", "--speech-pads", "0,40"]
        )


def test_vad_sweep_rejects_negative_speech_pad_member():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--speech-pads", "40,-1"]
        )


# ---- vad_segmentation_sweep / renderers: pad axis ----------------------


def test_sweep_axis_keys_rows_by_pad_axis_name():
    r_lo = _Result(
        name="r.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    r_hi = _Result(
        name="r.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    rows = gv.vad_segmentation_sweep([0.0, 60.0], [r_lo, r_hi], axis="speech_pad_ms")
    assert rows == [
        {"speech_pad_ms": 0.0, "num_segments": 2, "speech_s": 2.0},
        {"speech_pad_ms": 60.0, "num_segments": 1, "speech_s": 1.0},
    ]


def test_render_sweep_pad_axis_labels_column():
    r_lo = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(0.0, 1.0), _Seg(2.0, 3.0)],
    )
    r_hi = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
    )
    lines = gv.render_vad_sweep(
        [0.0, 60.0], [r_lo, r_hi], name="rec.wav", axis="speech_pad_ms"
    )
    text = "\n".join(lines)
    assert "speech_pad" in text
    assert "min_silence" not in text
    # Pad values print as bare integers (0, 60), not 0.00 gates.
    assert "60" in text
    assert "0.00" not in text


def test_render_sweep_json_carries_pad_axis():
    r = _Result(name="rec.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    payload = json.loads(
        gv.render_vad_sweep_json([40.0], [r], name="rec.wav", axis="speech_pad_ms")
    )
    assert payload["axis"] == "speech_pad_ms"
    assert payload["sweep"] == [
        {"speech_pad_ms": 40.0, "num_segments": 1, "speech_s": 1.0}
    ]


def test_render_sweep_csv_header_is_pad_axis_name():
    r = _Result(name="rec.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    text = gv.render_vad_sweep_csv([40.0], [r], name="rec.wav", axis="speech_pad_ms")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["speech_pad_ms", "num_segments", "speech_s"]
    assert rows[1] == ["40.0", "1", "1.0"]


# ---- cmd_vad_sweep: pad axis end-to-end --------------------------------


def test_cmd_vad_sweep_pad_axis_sweeps_padding():
    # When --speech-pads is set, the segmenter sees the SWEPT speech_pad_ms and
    # the gate held at scalar --threshold; the scalar --speech-pad-ms is ignored.
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        # More padding merges adjacent regions → fewer segments.
        n = 3 if params.speech_pad_ms < 40 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(speech_pads=[0.0, 60.0], threshold=0.7, speech_pad_ms=999.0),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert [p.speech_pad_ms for p in captured] == [0.0, 60.0]
    # Gate held at scalar --threshold for every run; the shared --speech-pad-ms
    # scalar (999) is NOT used as a swept value.
    assert {p.threshold for p in captured} == {0.7}
    text = "\n".join(lines)
    assert "speech_pad" in text
    assert "60" in text


def test_cmd_vad_sweep_pad_axis_holds_silence_scalar():
    # The non-swept ms knob (--min-silence-ms) is shared across every run.
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        return _Result(
            name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
        )

    gv.cmd_vad_sweep(
        _sweep_args(speech_pads=[0.0, 60.0], min_silence_ms=750.0),
        log=lambda *a: None,
        segmenter=seg,
        availability=lambda: True,
    )
    assert {p.min_silence_ms for p in captured} == {750.0}


def test_cmd_vad_sweep_pad_axis_json_branch():
    def seg(wav, params=None):
        n = 3 if params.speech_pad_ms < 40 else 1
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(speech_pads=[0.0, 60.0], json=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["axis"] == "speech_pad_ms"
    assert [row["speech_pad_ms"] for row in payload["sweep"]] == [0.0, 60.0]
    assert payload["sweep"][0]["num_segments"] == 3
    assert payload["sweep"][1]["num_segments"] == 1


def test_cmd_vad_sweep_pad_axis_csv_branch():
    def seg(wav, params=None):
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=10.0,
            segments=[_Seg(0.0, 1.0)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(speech_pads=[0.0, 60.0], csv=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    rows = list(csv.reader(io.StringIO(lines[0])))
    assert rows[0] == ["speech_pad_ms", "num_segments", "speech_s"]
    assert [row[0] for row in rows[1:]] == ["0.0", "60.0"]


def test_cmd_vad_sweep_pad_axis_unavailable():
    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(speech_pads=[0.0, 60.0], json=True),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert json.loads(lines[0])["available"] is False


# ====================================================================
# iter-256 — gv vad-sweep --max-speeches: a fifth sweep axis (ceiling)
# ====================================================================
# The force-split ceiling (max_speech_s) was held fixed across all runs;
# iter-256 promotes it to a fifth sweepable axis alongside the gate, hangover,
# floor, and padding — completing grid/sweep symmetry (every vad-grid column
# axis is now also a 1-D sweep axis). Unlike the other four, this axis is
# measured in SECONDS, not ms: it reuses the iter-255 max_speech_list_type
# validator (so 'inf'/'none'/'off' anchor the no-cap baseline) and the seconds
# formatter. The gate is held at scalar --threshold while it sweeps; the five
# axes are mutually exclusive.


# ---- parser wiring: --max-speeches axis --------------------------------


def test_vad_sweep_max_speeches_parses():
    args = gv.build_parser().parse_args(
        ["vad-sweep", "rec.wav", "--max-speeches", "5,10,20"]
    )
    assert args.max_speeches == [5.0, 10.0, 20.0]
    # The threshold list keeps its default; the handler picks the ceiling axis.
    assert args.thresholds == [0.3, 0.5, 0.7, 0.9]


def test_vad_sweep_max_speeches_default_is_none():
    # Without --max-speeches the ceiling axis is off (None), so the handler
    # sweeps --thresholds (the iter-236 default).
    args = gv.build_parser().parse_args(["vad-sweep", "rec.wav"])
    assert args.max_speeches is None


def test_vad_sweep_max_speeches_accepts_inf_sentinels():
    # 'inf'/'none'/'off' all anchor the no-cap baseline in the sweep, so the
    # operator can include the never-force-split point alongside finite caps.
    args = gv.build_parser().parse_args(
        ["vad-sweep", "rec.wav", "--max-speeches", "5,inf,none,off"]
    )
    assert args.max_speeches == [
        5.0,
        float("inf"),
        float("inf"),
        float("inf"),
    ]


def test_vad_sweep_thresholds_and_max_speeches_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--thresholds", "0.3,0.5", "--max-speeches", "5,10"]
        )


def test_vad_sweep_min_silences_and_max_speeches_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--min-silences", "400,800", "--max-speeches", "5,10"]
        )


def test_vad_sweep_min_speeches_and_max_speeches_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--min-speeches", "50,100", "--max-speeches", "5,10"]
        )


def test_vad_sweep_speech_pads_and_max_speeches_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--speech-pads", "0,40", "--max-speeches", "5,10"]
        )


def test_vad_sweep_rejects_zero_max_speech_member():
    # A 0s cap would force-split forever — rejected per token by the seconds
    # validator (positive-only, like the iter-255 vad-grid column axis).
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--max-speeches", "5,0"]
        )


def test_vad_sweep_rejects_negative_max_speech_member():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--max-speeches", "5,-1"]
        )


# ---- vad_segmentation_sweep / renderers: ceiling axis ------------------


def test_sweep_axis_keys_rows_by_max_speech_axis_name():
    # A tighter cap force-splits the long region into more segments.
    r_loose = _Result(
        name="r.wav",
        sample_rate=16000,
        duration_s=20.0,
        segments=[_Seg(0.0, 12.0)],
    )
    r_tight = _Result(
        name="r.wav",
        sample_rate=16000,
        duration_s=20.0,
        segments=[_Seg(0.0, 5.0), _Seg(5.0, 10.0)],
    )
    rows = gv.vad_segmentation_sweep(
        [float("inf"), 5.0], [r_loose, r_tight], axis="max_speech_s"
    )
    assert rows == [
        {"max_speech_s": float("inf"), "num_segments": 1, "speech_s": 12.0},
        {"max_speech_s": 5.0, "num_segments": 2, "speech_s": 10.0},
    ]


def test_render_sweep_max_speech_axis_labels_column():
    r_loose = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=20.0,
        segments=[_Seg(0.0, 12.0)],
    )
    r_tight = _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=20.0,
        segments=[_Seg(0.0, 5.0), _Seg(5.0, 10.0)],
    )
    lines = gv.render_vad_sweep(
        [float("inf"), 5.0], [r_loose, r_tight], name="rec.wav", axis="max_speech_s"
    )
    text = "\n".join(lines)
    assert "max_speech" in text
    assert "min_silence" not in text
    # Seconds print compactly (5, inf) via %g — no 5.00 gate leak, no ms-style
    # integer truncation, and the no-cap sentinel renders as inf.
    assert "inf" in text
    assert "5.00" not in text


def test_render_sweep_target_on_max_speech_axis_formats_seconds():
    # iter-257: the --target pick block ranks by num_segments (axis-agnostic), so
    # it already works on the seconds max_speech_s sweep. This guards that the
    # best: line names the swept SECONDS value via %g — no gate-style 0.00 leak,
    # the no-cap sentinel rendered as inf — rather than the .2f used for the gate.
    # Caps inf, 10, 5; segment counts 1, 2, 4; target 2 → the 10s value (2 segs).
    r_inf = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 12.0)],
    )
    r_ten = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 5.0), _Seg(5.0, 10.0)],
    )
    r_five = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 2.5), _Seg(2.5, 5.0), _Seg(5.0, 7.5), _Seg(7.5, 10.0)],
    )
    lines = gv.render_vad_sweep(
        [float("inf"), 10.0, 5.0], [r_inf, r_ten, r_five],
        name="rec.wav", axis="max_speech_s", target=2,
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=10" in best
    assert "2 segments" in best
    assert "|Δ|=0" in best
    assert "target 2" in best
    # Compact %g, no gate-style trailing zeros anywhere on the line.
    assert "10.00" not in best
    assert "max_speech=10.0" not in best


def test_render_sweep_target_best_can_name_inf_max_speech():
    # iter-257: when the no-cap baseline (inf) is the closest cell, the best: line
    # must render the sentinel as "inf", not "inf.00" or a float repr.
    r_inf = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 12.0)],
    )
    r_tight = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 2.5), _Seg(2.5, 5.0), _Seg(5.0, 7.5), _Seg(7.5, 10.0)],
    )
    lines = gv.render_vad_sweep(
        [float("inf"), 5.0], [r_inf, r_tight],
        name="rec.wav", axis="max_speech_s", target=1,
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=inf" in best
    assert "1 segments" in best
    assert "inf.00" not in best


def test_render_grid_target_on_max_speech_col_axis_formats_seconds():
    # iter-257: the seconds force-split ceiling is also a vad-grid COLUMN axis;
    # the --target best: line must format the col value via %g too. 1×2 grid over
    # max_speech_s caps inf, 5; counts 1, 4; target 3 → the 5s cell (|Δ|=1 vs 2).
    r_inf = _cell_result(1)
    r_tight = _cell_result(4)
    lines = gv.render_vad_grid(
        [0.3], [float("inf"), 5.0], [r_inf, r_tight],
        name="rec.wav", col_axis="max_speech_s", target=3,
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=5" in best
    assert "threshold=0.30" in best
    assert "4 segments" in best
    assert "target 3" in best
    # Seconds col formats compactly — no 5.00 leak, ints not truncated oddly.
    assert "max_speech=5.00" not in best


# ---- --target `top N:` HUMAN shortlist on the seconds max_speech_s axis ----
#
# iter-257 pinned the HUMAN best: line %g formatting on the seconds force-split
# ceiling axis, but covered ONLY the single best: line — NOT the `top N:`
# shortlist lines, which route through the SAME format_axes closure (and thus
# the same _format_sweep_axis_value %g path). iter-258 (JSON) and iter-259 (CSV)
# closed the machine surfaces. These close the last seconds-axis hole: the
# human `top N:` rows must render each picked SECONDS value via %g — compact
# (10, 5), the no-cap baseline as "inf" — with no gate-style 0.00 / inf.00 leak,
# on both the 1-D sweep and the 2-D grid column axis.


def test_render_sweep_target_top_list_formats_seconds():
    # Caps inf, 10, 5; segment counts 1, 2, 4; target 2, top 3. The shortlist
    # ranks nearest-first by num_segments (10s @ |Δ|=0, then inf @ 1 and 5s @ 2,
    # ties broken row-major). Every top row must name its SECONDS cap via %g.
    r_inf = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 12.0)],
    )
    r_ten = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 5.0), _Seg(5.0, 10.0)],
    )
    r_five = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 2.5), _Seg(2.5, 5.0), _Seg(5.0, 7.5), _Seg(7.5, 10.0)],
    )
    lines = gv.render_vad_sweep(
        [float("inf"), 10.0, 5.0], [r_inf, r_ten, r_five],
        name="rec.wav", axis="max_speech_s", target=2, top=3,
    )
    # The shortlist header names the count it ranks toward.
    assert any("top 3 (closest to target 2)" in ln for ln in lines)
    top_rows = [ln for ln in lines if ln.lstrip().startswith(("1.", "2.", "3."))]
    assert len(top_rows) == 3
    joined = "\n".join(top_rows)
    # Each swept cap appears compactly: 10, 5, and the no-cap baseline inf.
    assert "max_speech=10" in joined
    assert "max_speech=5" in joined
    assert "max_speech=inf" in joined
    # The nearest cell (10s, |Δ|=0) heads the shortlist.
    assert top_rows[0].lstrip().startswith("1.")
    assert "max_speech=10" in top_rows[0]
    assert "|Δ|=0" in top_rows[0]
    # No gate-style trailing zeros and no inf.00 leak anywhere in the shortlist.
    assert "10.00" not in joined
    assert "max_speech=10.0" not in joined
    assert "5.00" not in joined
    assert "inf.00" not in joined


def test_render_grid_target_top_list_on_max_speech_col_axis_formats_seconds():
    # The seconds force-split ceiling is also a grid COLUMN axis; the `top N:`
    # rows must format the col value via %g too. 1×2 grid over caps inf, 5 with
    # counts 1, 4; target 3, top 2 → the 5s cell (|Δ|=1) heads, inf (|Δ|=2) next.
    r_inf = _cell_result(1)
    r_tight = _cell_result(4)
    lines = gv.render_vad_grid(
        [0.3], [float("inf"), 5.0], [r_inf, r_tight],
        name="rec.wav", col_axis="max_speech_s", target=3, top=2,
    )
    assert any("top 2 (closest to target 3)" in ln for ln in lines)
    top_rows = [ln for ln in lines if ln.lstrip().startswith(("1.", "2."))]
    assert len(top_rows) == 2
    joined = "\n".join(top_rows)
    # Both the seconds col value and the held threshold row render compactly.
    assert "max_speech=5" in joined
    assert "max_speech=inf" in joined
    assert "threshold=0.30" in joined
    # The 5s cell (4 segs, |Δ|=1) heads the shortlist.
    assert top_rows[0].lstrip().startswith("1.")
    assert "max_speech=5" in top_rows[0]
    assert "|Δ|=1" in top_rows[0]
    # No gate-style 5.00 / inf.00 leak on the seconds column.
    assert "max_speech=5.00" not in joined
    assert "inf.00" not in joined


# ---- --target BANDED (lo-hi) form on the seconds max_speech_s axis ---------
#
# iter-246 shipped the banded `--target lo-hi` form (a [lo, hi] count window
# scoring distance 0 for any cell inside the band, else distance to the nearer
# edge); iter-247 added the open-edge forms (`lo-` "at least", `-hi" "at most").
# iter-255/257 added the seconds `max_speech_s` force-split-ceiling axis. Every
# seconds-axis target test from iter-257→261 used a SCALAR target, and every
# band-target render test (test_render_*_band_*) swept the threshold/min_silence
# axis — so the band-scoring path and the `%g` seconds formatting had no JOINT
# coverage. The band pick block ranks by num_segments (axis-agnostic), so it
# already works on the seconds sweep — these pin that the best: line names the
# band-chosen SECONDS cap via %g (compact 10/5, the no-cap baseline as "inf"),
# renders the band as "lo-hi"/"lo-"/"-hi" (not a tuple repr), and never leaks a
# gate-style 0.00 / inf.00, on both the 1-D sweep and the 2-D grid column axis.


def _max_speech_cells():
    # Caps inf, 10, 5 over a 20s clip; segment counts 1, 4, 2.
    r_inf = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 12.0)],
    )
    r_ten = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 5.0), _Seg(5.0, 10.0), _Seg(10.0, 13.0), _Seg(13.0, 16.0)],
    )
    r_five = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 5.0), _Seg(5.0, 10.0)],
    )
    return r_inf, r_ten, r_five


def test_render_sweep_band_target_on_max_speech_axis_formats_seconds():
    # Caps inf, 10, 5; counts 1, 4, 2. Band 3-5 puts the 10s cap (4 segs) inside
    # the band (|Δ|=0), winning over inf (short by 2) and 5s (5s lands at 2, short
    # by 1). The best: line names the SECONDS cap via %g and renders the band.
    r_inf, r_ten, r_five = _max_speech_cells()
    lines = gv.render_vad_sweep(
        [float("inf"), 10.0, 5.0], [r_inf, r_ten, r_five],
        name="rec.wav", axis="max_speech_s", target=(3, 5),
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=10" in best
    assert "4 segments" in best
    assert "|Δ|=0" in best
    # The band renders as "3-5", not a tuple repr.
    assert "target 3-5" in best
    assert "(3, 5)" not in best
    # Compact %g — no gate-style trailing zeros on the seconds cap.
    assert "10.00" not in best
    assert "max_speech=10.0" not in best


def test_render_sweep_band_target_best_can_name_inf_max_speech():
    # An "at most" band (-2) is satisfied only by the no-cap baseline (inf, 1 seg);
    # the 10s cap (4) is over by 2 and the 5s cap (2) lands exactly on the upper
    # edge — wait, 2 ≤ 2 so 5s ALSO satisfies. Use caps inf, 5 (counts 1, 2) and
    # band -1 so only inf (1 seg) lands inside; it must render the sentinel "inf".
    r_inf, _r_ten, r_five = _max_speech_cells()
    lines = gv.render_vad_sweep(
        [float("inf"), 5.0], [r_inf, r_five],
        name="rec.wav", axis="max_speech_s", target=(None, 1),
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=inf" in best
    assert "1 segments" in best
    assert "|Δ|=0" in best
    # The open "at most" band renders as "-1", with no None leak or inf.00.
    assert "target -1" in best
    assert "None" not in best
    assert "inf.00" not in best


def test_render_sweep_open_band_at_least_on_max_speech_axis_formats_seconds():
    # An "at least 3" band (3-) is satisfied by the 10s cap (4 segs, |Δ|=0); the
    # inf baseline (1) and 5s cap (2) both fall short. The best: line names the
    # 10s cap via %g and renders the open band as "3-" with no None leak.
    r_inf, r_ten, r_five = _max_speech_cells()
    lines = gv.render_vad_sweep(
        [float("inf"), 10.0, 5.0], [r_inf, r_ten, r_five],
        name="rec.wav", axis="max_speech_s", target=(3, None),
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=10" in best
    assert "4 segments" in best
    assert "|Δ|=0" in best
    assert "target 3-" in best
    assert "None" not in best
    assert "10.00" not in best


def test_render_grid_band_target_on_max_speech_col_axis_formats_seconds():
    # The seconds force-split ceiling is also a vad-grid COLUMN axis; the banded
    # best: line must format the col value via %g too. 1×2 grid over caps inf, 5
    # (counts 1, 4); band 3-5 puts the 5s cell (4 segs) inside the band (|Δ|=0),
    # winning over the inf baseline (short by 2).
    lines = gv.render_vad_grid(
        [0.3], [float("inf"), 5.0], [_cell_result(1), _cell_result(4)],
        name="rec.wav", col_axis="max_speech_s", target=(3, 5),
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=5" in best
    assert "threshold=0.30" in best
    assert "4 segments" in best
    assert "|Δ|=0" in best
    assert "target 3-5" in best
    assert "(3, 5)" not in best
    # Seconds col formats compactly — no 5.00 leak.
    assert "max_speech=5.00" not in best


def test_render_grid_json_band_target_carries_seconds_max_speech():
    # The grid JSON surface must carry a band target on the seconds col axis as a
    # [lo, hi] array AND emit the chosen cap as a finite seconds number (5.0), with
    # distance 0 when the count lands inside the band.
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [float("inf"), 5.0], [_cell_result(1), _cell_result(4)],
            name="rec.wav", col_axis="max_speech_s", target=(3, 5),
        )
    )
    assert payload["target"] == [3, 5]
    assert payload["best"]["num_segments"] == 4
    assert payload["best"]["distance"] == 0
    assert payload["best"]["max_speech_s"] == 5.0


# ---- --target comma SET form on the seconds max_speech_s axis --------------
#
# iter-248 shipped the comma SET form (`--target 3,5,7`): a cell scores its
# distance to the NEAREST set element, so any cell landing on a listed count
# scores 0. iter-255/257 added the seconds `max_speech_s` force-split-ceiling
# axis. Every seconds-axis target test from iter-257→262 used a SCALAR or BANDED
# target, and every SET-target render test (test_render_grid_*_set_*) swept the
# threshold/min_silence axis — so the set-scoring path and the `%g` seconds
# formatting had no JOINT coverage. The pick block ranks by num_segments
# (axis-agnostic), so it already works on the seconds sweep — these pin that the
# best: line names the set-chosen SECONDS cap via %g (compact 10/5, the no-cap
# baseline as "inf"), renders the set as "2,4,6" (not a list repr), never leaks
# a gate-style 0.00 / inf.00, and that the grid JSON carries the set as a JSON
# array with a finite-seconds best cap — on both the 1-D sweep and 2-D grid.


def test_render_sweep_set_target_on_max_speech_axis_formats_seconds():
    # Caps inf, 10, 5; counts 1, 4, 2. Set 2,4,6 lands exactly on the 10s cap
    # (4 segs, |Δ|=0), winning over inf (nearest element 2, off by 1) and 5s
    # (lands on element 2, |Δ|=0 too — but row order puts 10s first). The best:
    # line names the SECONDS cap via %g and renders the set as "2,4,6".
    r_inf, r_ten, r_five = _max_speech_cells()
    lines = gv.render_vad_sweep(
        [float("inf"), 10.0, 5.0], [r_inf, r_ten, r_five],
        name="rec.wav", axis="max_speech_s", target=[2, 4, 6],
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=10" in best
    assert "4 segments" in best
    assert "|Δ|=0" in best
    # The set renders as "2,4,6", not a list repr.
    assert "target 2,4,6" in best
    assert "[2, 4, 6]" not in best
    # Compact %g — no gate-style trailing zeros on the seconds cap.
    assert "10.00" not in best
    assert "max_speech=10.0" not in best


def test_render_sweep_set_target_best_can_name_inf_max_speech():
    # When the no-cap baseline (inf) lands on a set element and the swept caps
    # don't, the best: line must render the sentinel as "inf", not "inf.00".
    # Caps inf, 5 (counts 1, 2); set 1,7 lands on inf (1 seg, |Δ|=0); 5s (2) is
    # off by 1 from the nearer element 1.
    r_inf, _r_ten, r_five = _max_speech_cells()
    lines = gv.render_vad_sweep(
        [float("inf"), 5.0], [r_inf, r_five],
        name="rec.wav", axis="max_speech_s", target=[1, 7],
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=inf" in best
    assert "1 segments" in best
    assert "|Δ|=0" in best
    assert "target 1,7" in best
    assert "[1, 7]" not in best
    assert "inf.00" not in best


def test_render_grid_set_target_on_max_speech_col_axis_formats_seconds():
    # The seconds force-split ceiling is also a vad-grid COLUMN axis; the set
    # best: line must format the col value via %g too. 1×2 grid over caps inf, 5
    # (counts 1, 4); set 2,4,6 lands on the 5s cell (4 segs, |Δ|=0), winning over
    # the inf baseline (1 seg, off by 1 from element 2).
    lines = gv.render_vad_grid(
        [0.3], [float("inf"), 5.0], [_cell_result(1), _cell_result(4)],
        name="rec.wav", col_axis="max_speech_s", target=[2, 4, 6],
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=5" in best
    assert "threshold=0.30" in best
    assert "4 segments" in best
    assert "|Δ|=0" in best
    assert "target 2,4,6" in best
    assert "[2, 4, 6]" not in best
    # Seconds col formats compactly — no 5.00 leak.
    assert "max_speech=5.00" not in best


def test_render_grid_json_set_target_carries_seconds_max_speech():
    # The grid JSON surface must carry a set target on the seconds col axis as a
    # JSON array AND emit the chosen cap as a finite seconds number (5.0), with
    # distance 0 when the count lands on a set element.
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [float("inf"), 5.0], [_cell_result(1), _cell_result(4)],
            name="rec.wav", col_axis="max_speech_s", target=[2, 4, 6],
        )
    )
    assert payload["target"] == [2, 4, 6]
    assert payload["best"]["num_segments"] == 4
    assert payload["best"]["distance"] == 0
    assert payload["best"]["max_speech_s"] == 5.0


# ---- --target preference (a>b) form on the seconds max_speech_s axis ---------
#
# iter-249 shipped the ranked PREFERENCE form (`--target 4>2`): a `{"prefer":
# [...]}` dict whose DISTANCE is the MIN over its elements (IDENTICAL to the flat
# set), but whose precedence breaks EXACT distance ties toward the earlier-listed
# (more-preferred) element via `_preference_rank` — inserted as a secondary sort
# key before the row-major/speech tie-break. iter-255/257 added the seconds
# `max_speech_s` force-split-ceiling axis. Every seconds-axis target test from
# iter-257→263 covered scalar, banded, and the flat SET form, but never the
# PREFERENCE form — and every preference render test swept the threshold axis. So
# the preference tie-break path and the `%g` seconds formatting had no JOINT
# coverage: a regression that broke EITHER (a preference rank that mishandled the
# seconds axis, or a `%g`→`.2f` drift on the preference-CHOSEN cap, or the JSON
# carrying the `{"prefer": [...]}` dict) would have shipped green while the
# threshold-axis preference tests stayed passing. These pin that the preference
# OVERRIDES row order to pick the more-preferred seconds cap (the discriminating
# behaviour vs the flat set), names it via %g (compact 10/5, the no-cap baseline
# as "inf"), renders the preference as "4>2" (not a `{"prefer": ...}` dict repr),
# never leaks a gate-style 0.00 / inf.00, and that the grid JSON carries the
# preference as its `{"prefer": [...]}` dict with a finite-seconds best cap — on
# both the 1-D sweep and the 2-D grid. No production code changed (the wiring was
# already correct — proved by a pre-test smoke run).


def test_render_sweep_prefer_target_on_max_speech_axis_formats_seconds():
    # Caps inf, 5, 10 (counts 1, 2, 4) in row order. Preference 4>2: both the 10s
    # cap (4 segs) and the 5s cap (2 segs) score |Δ|=0, but the preference ranks
    # the 4 ahead of the 2, so the 10s cap WINS even though the 5s cap is the
    # earlier row — the discriminating behaviour vs the flat set [4,2], which
    # would pick the earlier 5s cap on the row-major tie. The best: line names
    # the SECONDS cap via %g and renders the preference as "4>2".
    r_inf, r_ten, r_five = _max_speech_cells()
    lines = gv.render_vad_sweep(
        [float("inf"), 5.0, 10.0], [r_inf, r_five, r_ten],
        name="rec.wav", axis="max_speech_s", target={"prefer": [4, 2]},
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=10" in best
    assert "4 segments" in best
    assert "|Δ|=0" in best
    # The preference renders as "4>2", not a {"prefer": ...} dict repr.
    assert "target 4>2" in best
    assert "prefer" not in best
    # Compact %g — no gate-style trailing zeros on the seconds cap.
    assert "10.00" not in best
    assert "max_speech=10.0" not in best


def test_render_sweep_prefer_target_best_can_name_inf_max_speech():
    # When the no-cap baseline (inf) lands on the most-preferred element and a
    # swept cap also satisfies a less-preferred one, the preference picks inf and
    # the best: line must render the sentinel as "inf", not "inf.00".
    # Caps inf, 5 (counts 1, 2); preference 1>2: inf (1 seg) ranks 0, 5s (2 segs)
    # ranks 1 — both |Δ|=0, inf wins on preference.
    r_inf, _r_ten, r_five = _max_speech_cells()
    lines = gv.render_vad_sweep(
        [float("inf"), 5.0], [r_inf, r_five],
        name="rec.wav", axis="max_speech_s", target={"prefer": [1, 2]},
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=inf" in best
    assert "1 segments" in best
    assert "|Δ|=0" in best
    assert "target 1>2" in best
    assert "prefer" not in best
    assert "inf.00" not in best


def test_render_grid_prefer_target_on_max_speech_col_axis_formats_seconds():
    # The seconds force-split ceiling is also a vad-grid COLUMN axis; the
    # preference best: line must format the col value via %g too AND honour the
    # preference tie-break. 1×2 grid over caps inf, 5 (counts 2, 4); preference
    # 4>2 ranks the 5s cell (4 segs) ahead of the inf baseline (2 segs) — both
    # |Δ|=0, so preference picks the 5s cap, OVERRIDING the earlier inf row (the
    # flat set [4,2] would pick inf instead).
    lines = gv.render_vad_grid(
        [0.3], [float("inf"), 5.0], [_cell_result(2), _cell_result(4)],
        name="rec.wav", col_axis="max_speech_s", target={"prefer": [4, 2]},
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=5" in best
    assert "threshold=0.30" in best
    assert "4 segments" in best
    assert "|Δ|=0" in best
    assert "target 4>2" in best
    assert "prefer" not in best
    # Seconds col formats compactly — no 5.00 leak.
    assert "max_speech=5.00" not in best


def test_render_grid_json_prefer_target_carries_seconds_max_speech():
    # The grid JSON surface must carry a preference target on the seconds col axis
    # as its {"prefer": [...]} dict (distinct from a flat-set array) AND emit the
    # preference-chosen cap as a finite seconds number (5.0), with distance 0 when
    # the count satisfies the preferred element.
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [float("inf"), 5.0], [_cell_result(2), _cell_result(4)],
            name="rec.wav", col_axis="max_speech_s", target={"prefer": [4, 2]},
        )
    )
    assert payload["target"] == {"prefer": [4, 2]}
    assert payload["best"]["num_segments"] == 4
    assert payload["best"]["distance"] == 0
    assert payload["best"]["max_speech_s"] == 5.0


# ---- --target weighted set (a,b:p) form on the seconds max_speech_s axis -----
#
# iter-250 shipped the additive-penalty WEIGHTED set (`--target 3,6:2`): a
# `{"weighted": [(element, penalty), ...]}` dict whose DISTANCE is the MIN over
# each element's RAW distance PLUS its penalty. Unlike the iter-249 preference
# (which folds intent only into the tie-break, never the distance), the weight
# enters the distance itself, so a cheaper (lower-penalty) element can win even
# at a LARGER raw distance — it overrides a distance GAP, not merely an exact
# tie. iter-255/257 added the seconds `max_speech_s` force-split-ceiling axis.
# Every seconds-axis target test from iter-257→264 covered scalar, banded, the
# flat SET, and the ranked PREFERENCE form, but never the WEIGHTED form — and
# every weighted render test (iter-250) swept the threshold/min-silence axis. So
# the penalised-distance path and the `%g` seconds formatting had no JOINT
# coverage: a regression that broke EITHER (a weighted distance that mishandled
# the seconds axis, a `%g`→`.2f` drift on the weighted-CHOSEN cap, the `:penalty`
# rendering, or the JSON carrying the `{"weighted": [...]}` dict) would have
# shipped green while the threshold-axis weighted tests stayed passing. These pin
# that the penalty OVERRIDES the raw-distance gap to pick the lower-penalty
# seconds cap (the discriminating behaviour vs the flat set, which picks the
# nearest raw cap), names it via %g (compact 10/5, the no-cap baseline as "inf"),
# renders the weighted set as "3,6:2" (not a `{"weighted": ...}` dict repr),
# never leaks a gate-style 0.00 / inf.00, and that the grid JSON carries the
# weighted set as its `{"weighted": [...]}` dict with a finite-seconds best cap —
# on both the 1-D sweep and the 2-D grid. No production code changed (the wiring
# was already correct — proved by a pre-test smoke run).


def test_render_sweep_weighted_target_on_max_speech_axis_formats_seconds():
    # Caps inf, 10, 5 (counts 1, 4, 6) in row order. Weighted 3,6:2: the 5s cap
    # (count 6) lands exactly on the accepted element (penalised 2); the 10s cap
    # (count 4) is raw dist 1 from the free element 3 (penalised 1), so the FLAT
    # set [3,6] would pick the 10s cap (raw dist 1 < the 5s cap's raw dist 0-to-6
    # ... ). The weighted +2 penalty on the 6 flips the pick: 10s scores
    # min(|4-3|+0, |4-6|+2)=1, 5s scores min(|6-3|+0, 0+2)=2, so the weighted set
    # picks the 10s cap at |Δ|=1 — the discriminating behaviour: with the penalty
    # the nearer-to-the-free-element cap wins over the one sitting on the costly
    # accepted element. The best: line names the SECONDS cap via %g and renders
    # the weighted set as "3,6:2".
    r_inf = _result_n_speech(1, 1.0)
    r_ten = _result_n_speech(4, 4.0)
    r_five = _result_n_speech(6, 6.0)
    lines = gv.render_vad_sweep(
        [float("inf"), 10.0, 5.0], [r_inf, r_ten, r_five],
        name="rec.wav", axis="max_speech_s", target={"weighted": [(3, 0), (6, 2)]},
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=10" in best
    assert "4 segments" in best
    assert "|Δ|=1" in best
    # The weighted set renders as "3,6:2", not a {"weighted": ...} dict repr.
    assert "target 3,6:2" in best
    assert "weighted" not in best
    # Compact %g — no gate-style trailing zeros on the seconds cap.
    assert "10.00" not in best
    assert "max_speech=10.0" not in best


def test_render_sweep_weighted_target_overrides_gap_not_just_tie():
    # The weighted set's defining property vs the flat set: the penalty folds into
    # the DISTANCE, so it overrides a raw-distance GAP, not just an exact tie.
    # Caps inf, 5 (counts 1, 6). Weighted 3,6:9 — a huge penalty on the accepted 6:
    # inf (count 1) scores min(|1-3|+0, |1-6|+9)=2; 5s (count 6) scores
    # min(|6-3|+0, 0+9)=3, so the inf baseline WINS at |Δ|=2 even though the 5s cap
    # sits exactly on a listed element — the +9 penalty makes the free element 3
    # the cheaper route. The flat set [3,6] would instead pick the 5s cap (raw
    # dist 0). Proves the penalty changes the WINNER, not merely a tie order.
    r_inf = _result_n_speech(1, 1.0)
    r_five = _result_n_speech(6, 6.0)
    lines = gv.render_vad_sweep(
        [float("inf"), 5.0], [r_inf, r_five],
        name="rec.wav", axis="max_speech_s", target={"weighted": [(3, 0), (6, 9)]},
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=inf" in best
    assert "1 segments" in best
    assert "|Δ|=2" in best
    assert "target 3,6:9" in best
    assert "weighted" not in best
    # The no-cap baseline names the sentinel as "inf", not "inf.00".
    assert "inf.00" not in best


def test_render_grid_weighted_target_on_max_speech_col_axis_formats_seconds():
    # The seconds force-split ceiling is also a vad-grid COLUMN axis; the weighted
    # best: line must format the col value via %g too AND honour the penalised
    # distance. 1×2 grid over caps inf, 5 (counts 6, 4). Weighted 3,6:2: the inf
    # baseline (count 6) sits on the accepted element (penalised 2); the 5s cap
    # (count 4) is raw dist 1 from the free 3 (penalised 1), so the penalty flips
    # the pick to the 5s cap at |Δ|=1 — OVERRIDING the earlier inf row (the flat
    # set [3,6] would pick inf, count 6, raw dist 0).
    lines = gv.render_vad_grid(
        [0.3], [float("inf"), 5.0], [_cell_result(6), _cell_result(4)],
        name="rec.wav", col_axis="max_speech_s", target={"weighted": [(3, 0), (6, 2)]},
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=5" in best
    assert "threshold=0.30" in best
    assert "4 segments" in best
    assert "|Δ|=1" in best
    assert "target 3,6:2" in best
    assert "weighted" not in best
    # Seconds col formats compactly — no 5.00 leak.
    assert "max_speech=5.00" not in best


def test_render_grid_json_weighted_target_carries_seconds_max_speech():
    # The grid JSON surface must carry a weighted target on the seconds col axis as
    # its {"weighted": [[element, penalty], ...]} dict (each pair a 2-element array,
    # distinct from a flat-set array of scalars) AND emit the weighted-chosen cap as
    # a finite seconds number (5.0), with the penalised distance (1).
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [float("inf"), 5.0], [_cell_result(6), _cell_result(4)],
            name="rec.wav", col_axis="max_speech_s", target={"weighted": [(3, 0), (6, 2)]},
        )
    )
    assert payload["target"] == {"weighted": [[3, 0], [6, 2]]}
    assert payload["best"]["num_segments"] == 4
    assert payload["best"]["distance"] == 1
    assert payload["best"]["max_speech_s"] == 5.0


# ---- --target scaled set (a,b*f) form on the seconds max_speech_s axis -------
#
# iter-252 shipped the multiplicative-factor SCALED set (`--target 3,8*2`): a
# `{"scaled": [(element, factor), ...]}` dict whose DISTANCE is the MIN over each
# element's RAW distance TIMES its factor — the multiplicative twin of the
# iter-250 additive WEIGHTED set. The defining contrast with BOTH the flat set
# and the additive weighted form: an exact hit stays FREE (raw 0 × any factor =
# 0), unlike an additive penalty which bites even an exact hit, while the cost
# GROWS with distance (drifting one count past a factor-2 element costs 2, not
# the constant offset the weighted form adds). iter-255/257 added the seconds
# `max_speech_s` force-split-ceiling axis. Every seconds-axis target test from
# iter-257→265 covered scalar, banded, the flat SET, the ranked PREFERENCE, and
# the additive WEIGHTED form, but never the SCALED form — and every scaled render
# test (iter-252) swept the threshold/min-silence axis. So the scaled-distance
# path and the `%g` seconds formatting had no JOINT coverage: a regression that
# broke EITHER (a scaled distance that mishandled the seconds axis, a `%g`→`.2f`
# drift on the scaled-CHOSEN cap, the `*factor` rendering, or the JSON carrying
# the `{"scaled": [...]}` dict) would have shipped green while the threshold-axis
# scaled tests stayed passing. These pin that the factor AMPLIFIES the off-element
# cost to pick the lower-factor seconds cap (the discriminating behaviour vs the
# flat set, which picks the nearest raw cap), that an exact hit on a high-factor
# element stays free (the discriminating behaviour vs the additive weighted set,
# which penalises it), names the cap via %g (compact 10/5, the no-cap baseline as
# "inf"), renders the scaled set as "3,8*2" (not a `{"scaled": ...}` dict repr),
# never leaks a gate-style 0.00 / inf.00, and that the grid JSON carries the
# scaled set as its `{"scaled": [...]}` dict with a finite-seconds best cap — on
# both the 1-D sweep and the 2-D grid. No production code changed (the wiring was
# already correct — proved by a pre-test smoke run).


def test_render_sweep_scaled_target_on_max_speech_axis_formats_seconds():
    # Caps inf, 10 (counts 7, 4) in row order. Scaled 3,8*2: preferred 3 (factor
    # 1), accepted 8 (factor 2). The inf baseline (count 7) is raw dist 4 from the
    # free 3 and raw dist 1 from the factor-2 8 → min(4*1, 1*2)=2; the 10s cap
    # (count 4) is raw dist 1 from the free 3 and raw dist 4 from the 8 →
    # min(1*1, 4*2)=1. So the scaled set picks the 10s cap at |Δ|=1. The FLAT set
    # [3,8] instead scores inf as min(4,1)=1 and 10s as min(1,4)=1 — a tie the
    # earlier inf row wins. The factor AMPLIFYING the off-8 distance is what flips
    # the pick to the finite 10s cap (the discriminating behaviour vs the flat
    # set). The best: line names the SECONDS cap via %g and renders "3,8*2".
    r_inf = _result_n_speech(7, 7.0)
    r_ten = _result_n_speech(4, 4.0)
    lines = gv.render_vad_sweep(
        [float("inf"), 10.0], [r_inf, r_ten],
        name="rec.wav", axis="max_speech_s", target={"scaled": [(3, 1), (8, 2)]},
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=10" in best
    assert "4 segments" in best
    assert "|Δ|=1" in best
    # The scaled set renders as "3,8*2", not a {"scaled": ...} dict repr.
    assert "target 3,8*2" in best
    assert "scaled" not in best
    # Compact %g — no gate-style trailing zeros on the seconds cap.
    assert "10.00" not in best
    assert "max_speech=10.0" not in best


def test_render_sweep_scaled_target_keeps_exact_hit_free():
    # The scaled set's defining property vs the ADDITIVE weighted set: an exact
    # hit stays FREE (raw 0 × factor = 0), where an additive penalty would bite
    # it. Caps inf, 5 (counts 4, 8). Scaled 3,8*2: inf (count 4) scores
    # min(1*1, 4*2)=1; the 5s cap (count 8) sits EXACTLY on the factor-2 element
    # 8 → min(5*1, 0*2)=0, so it WINS at |Δ|=0 despite the high factor — the
    # factor never bites the exact hit. (The additive weighted 3,8:2 would instead
    # penalise that hit to 2 and pick the inf baseline.) Proves the multiplicative
    # form leaves on-target caps free.
    r_inf = _result_n_speech(4, 4.0)
    r_five = _result_n_speech(8, 8.0)
    lines = gv.render_vad_sweep(
        [float("inf"), 5.0], [r_inf, r_five],
        name="rec.wav", axis="max_speech_s", target={"scaled": [(3, 1), (8, 2)]},
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=5" in best
    assert "8 segments" in best
    assert "|Δ|=0" in best
    assert "target 3,8*2" in best
    assert "scaled" not in best
    # Finite seconds cap formats compactly — no 5.00 leak.
    assert "max_speech=5.00" not in best


def test_render_grid_scaled_target_on_max_speech_col_axis_formats_seconds():
    # The seconds force-split ceiling is also a vad-grid COLUMN axis; the scaled
    # best: line must format the col value via %g too AND honour the scaled
    # distance. 1×2 grid over caps inf, 5 (counts 7, 4). Scaled 3,8*2: the inf
    # baseline (count 7) scores min(4*1, 1*2)=2; the 5s cap (count 4) scores
    # min(1*1, 4*2)=1, so the factor flips the pick to the 5s cap at |Δ|=1 —
    # OVERRIDING the earlier inf row (the flat set [3,8] would tie at 1 and keep
    # the earlier inf, count 7).
    lines = gv.render_vad_grid(
        [0.3], [float("inf"), 5.0], [_cell_result(7), _cell_result(4)],
        name="rec.wav", col_axis="max_speech_s", target={"scaled": [(3, 1), (8, 2)]},
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=5" in best
    assert "threshold=0.30" in best
    assert "4 segments" in best
    assert "|Δ|=1" in best
    assert "target 3,8*2" in best
    assert "scaled" not in best
    # Seconds col formats compactly — no 5.00 leak.
    assert "max_speech=5.00" not in best


def test_render_grid_json_scaled_target_carries_seconds_max_speech():
    # The grid JSON surface must carry a scaled target on the seconds col axis as
    # its {"scaled": [[element, factor], ...]} dict (each pair a 2-element array,
    # distinct from a flat-set array of scalars and from a {"weighted": ...} dict)
    # AND emit the scaled-chosen cap as a finite seconds number (5.0), with the
    # scaled distance (1).
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [float("inf"), 5.0], [_cell_result(7), _cell_result(4)],
            name="rec.wav", col_axis="max_speech_s", target={"scaled": [(3, 1), (8, 2)]},
        )
    )
    assert payload["target"] == {"scaled": [[3, 1], [8, 2]]}
    assert payload["best"]["num_segments"] == 4
    assert payload["best"]["distance"] == 1
    assert payload["best"]["max_speech_s"] == 5.0


def test_render_sweep_json_carries_max_speech_axis():
    r = _Result(name="rec.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    payload = json.loads(
        gv.render_vad_sweep_json([10.0], [r], name="rec.wav", axis="max_speech_s")
    )
    assert payload["axis"] == "max_speech_s"
    assert payload["sweep"] == [
        {"max_speech_s": 10.0, "num_segments": 1, "speech_s": 1.0}
    ]


def test_render_sweep_csv_header_is_max_speech_axis_name():
    r = _Result(name="rec.wav", sample_rate=16000, duration_s=5.0, segments=[_Seg(0.0, 1.0)])
    text = gv.render_vad_sweep_csv([10.0], [r], name="rec.wav", axis="max_speech_s")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["max_speech_s", "num_segments", "speech_s"]
    assert rows[1] == ["10.0", "1", "1.0"]


# ---- CSV seconds (max_speech_s) inf-baseline coverage (iter-259) -------
#
# iter-257 pinned the HUMAN best: line %g formatting and iter-258 the JSON
# best/top serialization on the seconds force-split-ceiling axis. The third
# machine surface — the CSV emitter (render_vad_sweep_csv / render_vad_grid_csv)
# — writes the raw axis value into the first column via the stdlib csv writer,
# which renders a finite cap as "10.0" and the never-force-split baseline as
# "inf" (str(float('inf'))), NOT the JSON-style "Infinity" token and NOT a blank.
# iter-257/258 left that distinction untested; these close the seconds-axis
# CSV hole on both the 1-D sweep and the 2-D grid column axis.


def test_render_sweep_csv_max_speech_inf_baseline_writes_inf():
    # The no-cap baseline (inf) must serialize as the bare token "inf" in the
    # first column — not "Infinity" (JSON), not blank, not a float repr.
    r_inf = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 12.0)],
    )
    r_ten = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 5.0), _Seg(5.0, 10.0)],
    )
    text = gv.render_vad_sweep_csv(
        [float("inf"), 10.0], [r_inf, r_ten], name="rec.wav", axis="max_speech_s"
    )
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["max_speech_s", "num_segments", "speech_s"]
    # Baseline row: the cap column is exactly "inf" (csv writer stringifies
    # float('inf') -> "inf"), and the cell parses back to math.inf.
    assert rows[1][0] == "inf"
    assert math.isinf(float(rows[1][0]))
    assert rows[1][1] == "1"
    # Finite cap stays a plain decimal, no gate-style truncation.
    assert rows[2][0] == "10.0"
    assert rows[2][1] == "2"
    # Guard against the JSON serialization leaking into the CSV surface.
    assert "Infinity" not in text


def test_render_sweep_csv_max_speech_round_trips_with_inf():
    # Every CSV cap cell parses back to its sweep value, inf included, so a
    # downstream loadtxt/read_csv consumer recovers the seconds axis losslessly.
    caps = [float("inf"), 10.0, 5.0]
    results = [
        _Result(name="rec.wav", sample_rate=16000, duration_s=20.0, segments=[_Seg(0.0, 12.0)]),
        _Result(name="rec.wav", sample_rate=16000, duration_s=20.0,
                 segments=[_Seg(0.0, 5.0), _Seg(5.0, 10.0)]),
        _Result(name="rec.wav", sample_rate=16000, duration_s=20.0,
                 segments=[_Seg(0.0, 2.5), _Seg(2.5, 5.0), _Seg(5.0, 7.5), _Seg(7.5, 10.0)]),
    ]
    text = gv.render_vad_sweep_csv(caps, results, name="rec.wav", axis="max_speech_s")
    cells = gv.vad_segmentation_sweep(caps, results, axis="max_speech_s")
    rows = list(csv.reader(io.StringIO(text)))
    for csv_row, cell in zip(rows[1:], cells):
        assert float(csv_row[0]) == cell["max_speech_s"]
        assert int(csv_row[1]) == cell["num_segments"]


def test_render_grid_csv_max_speech_col_axis_inf_baseline_writes_inf():
    # Same guarantee on the 2-D grid COLUMN axis: the inf baseline column writes
    # "inf", the finite cap writes "5.0", the held threshold row stays "0.3".
    r_inf = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 12.0)],
    )
    r_five = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 2.5), _Seg(2.5, 5.0)],
    )
    text = gv.render_vad_grid_csv(
        [0.3], [float("inf"), 5.0], [r_inf, r_five], name="rec.wav",
        row_axis="threshold", col_axis="max_speech_s",
    )
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["threshold", "max_speech_s", "num_segments", "speech_s"]
    assert rows[1][:3] == ["0.3", "inf", "1"]
    assert math.isinf(float(rows[1][1]))
    assert rows[2][:3] == ["0.3", "5.0", "2"]
    assert "Infinity" not in text


def test_render_grid_csv_max_speech_col_axis_multi_row_round_trips_with_inf():
    # iter-267: the iter-259 grid-CSV seconds proof above is a single 1×2 grid
    # (one threshold row). The MULTI-row case — several threshold rows each
    # crossed with the seconds force-split caps — was never exercised on the
    # column-axis CSV: the iter-259 round-trip test (test_render_grid_csv_
    # round_trips_to_grid_cells) sweeps min_silence_ms, NOT the seconds axis,
    # and the col-axis inf test holds threshold fixed at a single value. So the
    # row-major emission of an inf baseline ONCE PER ROW, and the lossless
    # float() round-trip of every seconds cell across rows, had no joint
    # coverage. A regression that emitted the inf token only on the first row,
    # or that let a later row's seconds cell drift to "Infinity"/blank/truncated,
    # would have shipped green while the single-row col-axis test stayed passing.
    # 2 thresholds × 3 caps (inf, 10, 5), row-major counts 1/2/4 then 1/3/6.
    caps = [float("inf"), 10.0, 5.0]
    results = [
        _Result(name="rec.wav", sample_rate=16000, duration_s=20.0,
                 segments=[_Seg(0.0, 12.0)]),
        _Result(name="rec.wav", sample_rate=16000, duration_s=20.0,
                 segments=[_Seg(0.0, 5.0), _Seg(5.0, 10.0)]),
        _Result(name="rec.wav", sample_rate=16000, duration_s=20.0,
                 segments=[_Seg(0.0, 2.5), _Seg(2.5, 5.0), _Seg(5.0, 7.5), _Seg(7.5, 10.0)]),
        _Result(name="rec.wav", sample_rate=16000, duration_s=20.0,
                 segments=[_Seg(0.0, 12.0)]),
        _Result(name="rec.wav", sample_rate=16000, duration_s=20.0,
                 segments=[_Seg(0.0, 4.0), _Seg(4.0, 8.0), _Seg(8.0, 12.0)]),
        _Result(name="rec.wav", sample_rate=16000, duration_s=20.0,
                 segments=[_Seg(float(i) * 2, float(i) * 2 + 1.0) for i in range(6)]),
    ]
    text = gv.render_vad_grid_csv(
        [0.3, 0.5], caps, results, name="rec.wav",
        row_axis="threshold", col_axis="max_speech_s",
    )
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["threshold", "max_speech_s", "num_segments", "speech_s"]
    # The inf baseline is the FIRST column cell of EVERY threshold row, not just
    # the first — proves the seconds-axis sentinel survives row-major repetition.
    assert rows[1][:2] == ["0.3", "inf"]
    assert rows[4][:2] == ["0.5", "inf"]
    assert all(math.isinf(float(rows[r][1])) for r in (1, 4))
    # Every cell parses losslessly back to its grid value, inf included, so a
    # downstream loadtxt/read_csv consumer recovers BOTH grid axes across rows.
    cells = gv.vad_segmentation_grid(
        [0.3, 0.5], caps, results, row_axis="threshold", col_axis="max_speech_s",
    )
    assert len(rows) - 1 == len(cells) == 6
    for csv_row, cell in zip(rows[1:], cells):
        assert float(csv_row[0]) == cell["threshold"]
        assert float(csv_row[1]) == cell["max_speech_s"]
        assert int(csv_row[2]) == cell["num_segments"]
        assert float(csv_row[3]) == cell["speech_s"]
    # Guard against the JSON Infinity token leaking into any row of the surface.
    assert "Infinity" not in text


def test_render_grid_csv_max_speech_row_axis_inf_baseline_writes_inf():
    # iter-267: the seconds force-split cap is a COLUMN axis in cmd_vad_grid, but
    # render_vad_grid_csv is axis-agnostic (it stringifies whichever value the
    # row_axis/col_axis name). Pinning the seconds axis on the ROW position proves
    # the inf sentinel writes "inf" in the FIRST column too (the column the
    # round-trip consumers key on), not only when it rides the second column.
    # 2 caps (inf, 5) × 1 hangover column.
    results = [
        _Result(name="rec.wav", sample_rate=16000, duration_s=20.0,
                 segments=[_Seg(0.0, 12.0)]),
        _Result(name="rec.wav", sample_rate=16000, duration_s=20.0,
                 segments=[_Seg(0.0, 2.5), _Seg(2.5, 5.0)]),
    ]
    text = gv.render_vad_grid_csv(
        [float("inf"), 5.0], [400.0], results, name="rec.wav",
        row_axis="max_speech_s", col_axis="min_silence_ms",
    )
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["max_speech_s", "min_silence_ms", "num_segments", "speech_s"]
    assert rows[1][:3] == ["inf", "400.0", "1"]
    assert math.isinf(float(rows[1][0]))
    assert rows[2][:3] == ["5.0", "400.0", "2"]
    assert "Infinity" not in text


# ---- --target JSON `best`/`top` on the seconds max_speech_s axis -------
#
# iter-257 pinned the HUMAN best: line %g formatting on the seconds axis;
# iter-258 closes the parallel hole on the MACHINE surface — the JSON `best`
# (and `top`) cell must carry the picked SECONDS max_speech_s value, and the
# `inf` no-cap baseline must survive the JSON round-trip (Python's json emits
# the `Infinity` token, which json.loads reads back as float('inf')).


def test_render_sweep_json_target_best_carries_seconds_max_speech():
    # Caps inf, 10, 5; segment counts 1, 2, 4; target 2 → the 10s cell (2 segs).
    # The best cell's max_speech_s key must be the bare float 10.0 (not a
    # gate-style string, not truncated) with distance 0.
    r_inf = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 12.0)],
    )
    r_ten = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 5.0), _Seg(5.0, 10.0)],
    )
    r_five = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 2.5), _Seg(2.5, 5.0), _Seg(5.0, 7.5), _Seg(7.5, 10.0)],
    )
    payload = json.loads(
        gv.render_vad_sweep_json(
            [float("inf"), 10.0, 5.0], [r_inf, r_ten, r_five],
            name="rec.wav", axis="max_speech_s", target=2,
        )
    )
    assert payload["axis"] == "max_speech_s"
    assert payload["target"] == 2
    assert payload["best"]["max_speech_s"] == 10.0
    assert payload["best"]["num_segments"] == 2
    assert payload["best"]["distance"] == 0


def test_render_sweep_json_best_inf_max_speech_survives_round_trip():
    # When the no-cap baseline (inf) is the closest cell, the JSON best value
    # must round-trip back to float('inf') — the seconds axis carries the
    # sentinel through json.dumps/json.loads unchanged.
    r_inf = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 12.0)],
    )
    r_tight = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 2.5), _Seg(2.5, 5.0), _Seg(5.0, 7.5), _Seg(7.5, 10.0)],
    )
    payload = json.loads(
        gv.render_vad_sweep_json(
            [float("inf"), 5.0], [r_inf, r_tight],
            name="rec.wav", axis="max_speech_s", target=1,
        )
    )
    best = payload["best"]["max_speech_s"]
    assert math.isinf(best) and best > 0
    assert payload["best"]["num_segments"] == 1
    assert payload["best"]["distance"] == 0


def test_render_sweep_json_top_list_carries_seconds_max_speech():
    # The `top` shortlist on the seconds axis names each cell by its
    # max_speech_s value; the head equals `best`, distances non-decreasing, and
    # the inf baseline survives the round-trip inside the list too.
    r_inf = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 12.0)],
    )
    r_ten = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 5.0), _Seg(5.0, 10.0)],
    )
    r_five = _Result(
        name="rec.wav", sample_rate=16000, duration_s=20.0,
        segments=[_Seg(0.0, 2.5), _Seg(2.5, 5.0), _Seg(5.0, 7.5), _Seg(7.5, 10.0)],
    )
    payload = json.loads(
        gv.render_vad_sweep_json(
            [float("inf"), 10.0, 5.0], [r_inf, r_ten, r_five],
            name="rec.wav", axis="max_speech_s", target=2, top=3,
        )
    )
    top = payload["top"]
    assert [c["max_speech_s"] for c in top][0] == payload["best"]["max_speech_s"]
    dists = [c["distance"] for c in top]
    assert dists == sorted(dists)
    # Every cell carries a numeric seconds value; the inf baseline is present.
    caps = [c["max_speech_s"] for c in top]
    assert any(math.isinf(c) for c in caps)
    assert 10.0 in caps and 5.0 in caps


def test_render_grid_json_target_best_carries_seconds_col_axis():
    # The seconds force-split ceiling is also a vad-grid COLUMN axis; the JSON
    # best cell must carry the picked col value (10.0) plus the held row gate.
    # 1×3 grid over caps inf, 10, 5; counts 1, 2, 4; target 2 → the 10s cell.
    results = [_cell_result(n) for n in (1, 2, 4)]
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [float("inf"), 10.0, 5.0], results,
            name="rec.wav", col_axis="max_speech_s", target=2,
        )
    )
    assert payload["col_axis"] == "max_speech_s"
    assert payload["best"]["max_speech_s"] == 10.0
    assert payload["best"]["threshold"] == 0.3
    assert payload["best"]["num_segments"] == 2
    assert payload["best"]["distance"] == 0


def test_render_grid_json_best_inf_col_axis_survives_round_trip():
    # The no-cap baseline as the best grid cell round-trips to float('inf') on
    # the column axis, same guarantee as the 1-D sweep.
    results = [_cell_result(n) for n in (1, 4)]
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [float("inf"), 5.0], results,
            name="rec.wav", col_axis="max_speech_s", target=1,
        )
    )
    best = payload["best"]["max_speech_s"]
    assert math.isinf(best) and best > 0
    assert payload["best"]["num_segments"] == 1
    assert payload["best"]["distance"] == 0


# ---- cmd_vad_sweep: ceiling axis end-to-end ----------------------------


def test_cmd_vad_sweep_max_speech_axis_sweeps_ceiling():
    # When --max-speeches is set, the segmenter sees the SWEPT max_speech_s and
    # the gate held at scalar --threshold; the scalar --max-speech-s is ignored.
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        # A tighter cap force-splits the monologue into more segments.
        n = 1 if params.max_speech_s == float("inf") else 3
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=20.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(
            max_speeches=[float("inf"), 5.0], threshold=0.7, max_speech_s=999.0
        ),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert [p.max_speech_s for p in captured] == [float("inf"), 5.0]
    # Gate held at scalar --threshold for every run; the shared --max-speech-s
    # scalar (999) is NOT used as a swept value.
    assert {p.threshold for p in captured} == {0.7}
    text = "\n".join(lines)
    assert "max_speech" in text
    assert "inf" in text


def test_cmd_vad_sweep_max_speech_axis_holds_silence_scalar():
    # The non-swept ms knob (--min-silence-ms) is shared across every run.
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        return _Result(
            name="rec.wav", sample_rate=16000, duration_s=10.0, segments=[_Seg(0.0, 1.0)]
        )

    gv.cmd_vad_sweep(
        _sweep_args(max_speeches=[5.0, 10.0], min_silence_ms=750.0),
        log=lambda *a: None,
        segmenter=seg,
        availability=lambda: True,
    )
    assert {p.min_silence_ms for p in captured} == {750.0}


def test_cmd_vad_sweep_max_speech_axis_json_branch():
    def seg(wav, params=None):
        n = 1 if params.max_speech_s == float("inf") else 3
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=20.0,
            segments=[_Seg(float(i), i + 0.5) for i in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(max_speeches=[float("inf"), 5.0], json=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["axis"] == "max_speech_s"
    # inf round-trips through JSON as the string "Infinity" (Python json), so
    # assert on the recovered counts and that the first value is non-finite.
    assert payload["sweep"][0]["num_segments"] == 1
    assert payload["sweep"][1]["num_segments"] == 3
    assert payload["sweep"][1]["max_speech_s"] == 5.0


def test_cmd_vad_sweep_max_speech_axis_csv_branch():
    def seg(wav, params=None):
        return _Result(
            name="rec.wav",
            sample_rate=16000,
            duration_s=20.0,
            segments=[_Seg(0.0, 1.0)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(max_speeches=[5.0, 10.0], csv=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    rows = list(csv.reader(io.StringIO(lines[0])))
    assert rows[0] == ["max_speech_s", "num_segments", "speech_s"]
    assert [row[0] for row in rows[1:]] == ["5.0", "10.0"]


def test_cmd_vad_sweep_max_speech_axis_unavailable():
    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(max_speeches=[5.0, 10.0], json=True),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert json.loads(lines[0])["available"] is False


# ====================================================================
# iter-240 — gv vad-grid: tabulate segmentation over a 2-D knob grid
# ====================================================================
# The 2-D analogue of vad-sweep (and the counterpart of simulate-mirror --grid):
# rows are always the P(speech) gate (--thresholds); the column axis is an ms
# knob — --min-silences (hangover, default) or --min-speeches (floor), mutually
# exclusive. Cells are flattened ROW-MAJOR.


def _grid_args(**over):
    base = dict(
        wav="rec.wav",
        thresholds=[0.3, 0.5],
        min_silences=[400.0, 800.0],
        min_speeches=None,
        speech_pads=None,
        max_speeches=None,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        target=None,
        top=None,
        tie_break="row-major",
        json=False,
        csv=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


# ---- parser: registration & wiring -------------------------------------


def test_vad_grid_in_handler_map():
    assert gv.DEFAULT_HANDLERS["vad-grid"] is gv.cmd_vad_grid


def test_vad_grid_requires_wav_positional():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-grid"])


def test_vad_grid_defaults():
    args = gv.build_parser().parse_args(["vad-grid", "rec.wav"])
    assert args.command == "vad-grid"
    assert args.thresholds == [0.3, 0.5, 0.7, 0.9]
    assert args.min_silences == [400.0, 600.0, 800.0, 1000.0]
    assert args.min_speeches is None
    assert args.json is False
    assert args.csv is False


def test_vad_grid_overrides_axes():
    args = gv.build_parser().parse_args(
        ["vad-grid", "rec.wav", "--thresholds", "0.4,0.6", "--min-silences", "500,900"]
    )
    assert args.thresholds == [0.4, 0.6]
    assert args.min_silences == [500.0, 900.0]


def test_vad_grid_min_speeches_column_axis():
    args = gv.build_parser().parse_args(
        ["vad-grid", "rec.wav", "--min-speeches", "50,100,200"]
    )
    assert args.min_speeches == [50.0, 100.0, 200.0]
    # The hangover list keeps its default; the handler picks the speech column.
    assert args.min_silences == [400.0, 600.0, 800.0, 1000.0]


def test_vad_grid_min_silences_and_min_speeches_mutually_exclusive():
    # Only one column axis may win; argparse rejects the combination.
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-grid", "rec.wav", "--min-silences", "400,800", "--min-speeches", "50,100"]
        )


# iter-254 — gv vad-grid --speech-pads: a third column-axis option (padding)


def test_vad_grid_speech_pads_column_axis():
    args = gv.build_parser().parse_args(
        ["vad-grid", "rec.wav", "--speech-pads", "0,20,40,80"]
    )
    assert args.speech_pads == [0.0, 20.0, 40.0, 80.0]
    # The hangover list keeps its default; the handler picks the pad column.
    assert args.min_silences == [400.0, 600.0, 800.0, 1000.0]
    assert args.min_speeches is None


def test_vad_grid_speech_pads_default_is_none():
    # Without --speech-pads the pad axis is off (None), so the default column
    # axis stays the hangover.
    args = gv.build_parser().parse_args(["vad-grid", "rec.wav"])
    assert args.speech_pads is None


def test_vad_grid_speech_pads_allows_zero():
    args = gv.build_parser().parse_args(
        ["vad-grid", "rec.wav", "--speech-pads", "0,40"]
    )
    assert args.speech_pads == [0.0, 40.0]


def test_vad_grid_speech_pads_rejects_negative_member():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-grid", "rec.wav", "--speech-pads", "20,-1"])


def test_vad_grid_silences_and_speech_pads_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-grid", "rec.wav", "--min-silences", "400,800", "--speech-pads", "0,40"]
        )


def test_vad_grid_speeches_and_speech_pads_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-grid", "rec.wav", "--min-speeches", "50,100", "--speech-pads", "0,40"]
        )


# iter-255 — gv vad-grid --max-speeches: a fourth column-axis option, the
# force-split ceiling (SECONDS, not ms); the four column axes are mutually
# exclusive.


def test_vad_grid_max_speeches_column_axis():
    args = gv.build_parser().parse_args(
        ["vad-grid", "rec.wav", "--max-speeches", "5,10,inf"]
    )
    assert args.max_speeches == [5.0, 10.0, float("inf")]
    # The hangover list keeps its default; the handler picks the ceiling column.
    assert args.min_silences == [400.0, 600.0, 800.0, 1000.0]
    assert args.min_speeches is None
    assert args.speech_pads is None


def test_vad_grid_max_speeches_default_is_none():
    # Without --max-speeches the ceiling axis is off (None), so the default
    # column axis stays the hangover.
    args = gv.build_parser().parse_args(["vad-grid", "rec.wav"])
    assert args.max_speeches is None


def test_vad_grid_max_speeches_rejects_zero_member():
    # A 0-second cap would split forever — rejected like the scalar knob.
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-grid", "rec.wav", "--max-speeches", "5,0"])


def test_vad_grid_max_speeches_rejects_negative_member():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-grid", "rec.wav", "--max-speeches", "5,-1"])


def test_vad_grid_silences_and_max_speeches_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-grid", "rec.wav", "--min-silences", "400,800", "--max-speeches", "5,10"]
        )


def test_vad_grid_speeches_and_max_speeches_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-grid", "rec.wav", "--min-speeches", "50,100", "--max-speeches", "5,10"]
        )


def test_vad_grid_pads_and_max_speeches_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-grid", "rec.wav", "--speech-pads", "0,40", "--max-speeches", "5,10"]
        )


def test_cmd_vad_grid_max_speech_column_sweeps_ceiling():
    # The segmenter sees the SWEPT max_speech_s with the gate; the three ms
    # knobs are held at their scalars (incl. the --max-speech-s scalar, which is
    # the swept axis now so it's never used as a held value).
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        return _cell_result(1)

    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], max_speeches=[5.0, 10.0],
            min_silence_ms=777.0, min_speech_ms=333.0, speech_pad_ms=44.0,
            max_speech_s=999.0,
        ),
        log=lambda *a: None,
        segmenter=seg,
        availability=lambda: True,
    )
    assert [(p.threshold, p.max_speech_s) for p in captured] == [
        (0.3, 5.0), (0.3, 10.0), (0.5, 5.0), (0.5, 10.0),
    ]
    assert {p.min_silence_ms for p in captured} == {777.0}
    assert {p.min_speech_ms for p in captured} == {333.0}
    assert {p.speech_pad_ms for p in captured} == {44.0}
    # The shared --max-speech-s scalar (999) is the swept axis now, so it is
    # never used as a held value.
    assert 999.0 not in {p.max_speech_s for p in captured}


def test_cmd_vad_grid_max_speech_column_sweeps_inf():
    # 'inf' is a legitimate swept value (the no-cap baseline) and must flow
    # through to the segmenter unchanged.
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        return _cell_result(1)

    gv.cmd_vad_grid(
        _grid_args(thresholds=[0.3], max_speeches=[5.0, float("inf")]),
        log=lambda *a: None,
        segmenter=seg,
        availability=lambda: True,
    )
    assert [p.max_speech_s for p in captured] == [5.0, float("inf")]


def test_cmd_vad_grid_max_speech_column_json_axis():
    # The JSON payload names the swept column axis so a consumer knows the grid
    # crossed gate × ceiling.
    def seg(wav, params=None):
        return _cell_result(1)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(thresholds=[0.3, 0.5], max_speeches=[5.0, 10.0], json=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    payload = json.loads(lines[0])
    assert payload["row_axis"] == "threshold"
    assert payload["col_axis"] == "max_speech_s"
    assert [c["max_speech_s"] for c in payload["grid"]] == [5.0, 10.0, 5.0, 10.0]


def test_cmd_vad_grid_max_speech_column_human_label():
    # The human table labels the ceiling column "max_speech" and formats
    # compact seconds via %g (no gate-style 0.00 leak, inf shown as "inf").
    def seg(wav, params=None):
        return _cell_result(1)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(thresholds=[0.3, 0.5], max_speeches=[5.0, float("inf")]),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "max_speech" in text
    # Compact-seconds formatting, not "5.00", and the no-cap baseline as "inf".
    assert "5.00" not in text
    assert "inf" in text


def test_vad_grid_rejects_out_of_range_threshold():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-grid", "rec.wav", "--thresholds", "0.5,1.5"])


def test_vad_grid_rejects_negative_min_silence_member():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-grid", "rec.wav", "--min-silences", "400,-1"])


def test_vad_grid_json_and_csv_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-grid", "rec.wav", "--json", "--csv"])


def test_vad_grid_target_defaults_none():
    args = gv.build_parser().parse_args(["vad-grid", "rec.wav"])
    assert args.target is None


def test_vad_grid_target_parses_int():
    args = gv.build_parser().parse_args(["vad-grid", "rec.wav", "--target", "5"])
    assert args.target == 5
    assert isinstance(args.target, int)


def test_vad_grid_target_zero_allowed():
    args = gv.build_parser().parse_args(["vad-grid", "rec.wav", "--target", "0"])
    assert args.target == 0


def test_vad_grid_target_dash_n_is_open_band_at_most():
    # iter-247: '-1' is no longer a (rejected) negative scalar — it is the open
    # band "at most 1" → (None, 1). A bare negative count is no longer
    # expressible, which is fine: nobody targets a negative segment count.
    args = gv.build_parser().parse_args(["vad-grid", "rec.wav", "--target", "-1"])
    assert args.target == (None, 1)


def test_vad_grid_rejects_fractional_target():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-grid", "rec.wav", "--target", "1.5"])


def test_vad_grid_target_parses_set():
    # iter-248: a comma-separated set parses to a list of acceptable targets.
    args = gv.build_parser().parse_args(
        ["vad-grid", "rec.wav", "--target", "3,5,7"]
    )
    assert args.target == [3, 5, 7]
    assert isinstance(args.target, list)


def test_vad_grid_top_defaults_none():
    args = gv.build_parser().parse_args(["vad-grid", "rec.wav"])
    assert args.top is None


def test_vad_grid_top_parses_int():
    args = gv.build_parser().parse_args(
        ["vad-grid", "rec.wav", "--target", "3", "--top", "5"]
    )
    assert args.top == 5
    assert isinstance(args.top, int)


def test_vad_grid_rejects_zero_top():
    # A 0-cell shortlist is meaningless — pos_int_type rejects it.
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-grid", "rec.wav", "--top", "0"])


def test_vad_grid_rejects_negative_top():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-grid", "rec.wav", "--top", "-2"])


def test_vad_grid_tie_break_defaults_row_major():
    args = gv.build_parser().parse_args(["vad-grid", "rec.wav"])
    assert args.tie_break == "row-major"


def test_vad_grid_tie_break_parses_speech():
    args = gv.build_parser().parse_args(
        ["vad-grid", "rec.wav", "--target", "3", "--tie-break", "speech"]
    )
    assert args.tie_break == "speech"


def test_vad_grid_rejects_unknown_tie_break():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-grid", "rec.wav", "--tie-break", "alphabetical"]
        )


# ---- vad_segmentation_grid: pure core ----------------------------------


def _cell_result(n):
    return _Result(
        name="rec.wav",
        sample_rate=16000,
        duration_s=10.0,
        segments=[_Seg(float(i), i + 0.5) for i in range(n)],
    )


def test_grid_cells_keyed_by_both_axes_row_major():
    # 2 thresholds × 2 hangovers = 4 cells, row-major: (t0,c0),(t0,c1),(t1,c0),(t1,c1).
    results = [_cell_result(n) for n in (4, 3, 2, 1)]
    cells = gv.vad_segmentation_grid(
        [0.3, 0.5], [400.0, 800.0], results,
        row_axis="threshold", col_axis="min_silence_ms",
    )
    assert [(c["threshold"], c["min_silence_ms"]) for c in cells] == [
        (0.3, 400.0), (0.3, 800.0), (0.5, 400.0), (0.5, 800.0),
    ]
    assert [c["num_segments"] for c in cells] == [4, 3, 2, 1]


def test_grid_speech_rounded_to_three_places():
    r = _Result(
        name="rec.wav", sample_rate=16000, duration_s=10.0,
        segments=[_Seg(0.0, 1.23456)],
    )
    cells = gv.vad_segmentation_grid([0.5], [400.0], [r])
    assert cells[0]["speech_s"] == 1.235


def test_grid_length_mismatch_raises():
    with pytest.raises(ValueError):
        gv.vad_segmentation_grid([0.3, 0.5], [400.0, 800.0], [_cell_result(1)])


def test_grid_defaults_to_threshold_x_min_silence():
    cells = gv.vad_segmentation_grid([0.5], [400.0], [_cell_result(2)])
    assert "threshold" in cells[0] and "min_silence_ms" in cells[0]


# ---- grid_cell_distance / pick_best_grid_cell: the data-driven pick -----


def _grid_cells(seg_counts, row_values=None, col_values=None):
    """Build a row-major cell list with the given segment counts."""
    row_values = row_values if row_values is not None else [0.3, 0.5]
    col_values = col_values if col_values is not None else [400.0, 800.0]
    results = [_cell_result(n) for n in seg_counts]
    return gv.vad_segmentation_grid(row_values, col_values, results)


def test_grid_cell_distance_is_abs_segment_gap():
    cell = {"num_segments": 7}
    assert gv.grid_cell_distance(cell, 5) == 2
    assert gv.grid_cell_distance(cell, 9) == 2
    assert gv.grid_cell_distance(cell, 7) == 0


def test_grid_cell_distance_band_zero_inside():
    # iter-246: any count inside the inclusive [3, 5] band scores 0.
    for n in (3, 4, 5):
        assert gv.grid_cell_distance({"num_segments": n}, (3, 5)) == 0


def test_grid_cell_distance_band_gap_to_nearest_edge():
    # Below the band → gap to lo; above → gap to hi.
    assert gv.grid_cell_distance({"num_segments": 1}, (3, 5)) == 2  # 3 - 1
    assert gv.grid_cell_distance({"num_segments": 8}, (3, 5)) == 3  # 8 - 5
    assert gv.grid_cell_distance({"num_segments": 2}, (3, 5)) == 1  # just below
    assert gv.grid_cell_distance({"num_segments": 6}, (3, 5)) == 1  # just above


def test_grid_cell_distance_degenerate_band_equals_scalar():
    # A (n, n) band is exactly the scalar distance to n.
    for n in (0, 3, 7):
        cell = {"num_segments": n + 2}
        assert gv.grid_cell_distance(cell, (n, n)) == gv.grid_cell_distance(cell, n)


def test_grid_cell_distance_open_band_at_least():
    # iter-247: (3, None) = "at least 3" — 0 for any count >= 3, gap below.
    for n in (3, 4, 9):
        assert gv.grid_cell_distance({"num_segments": n}, (3, None)) == 0
    assert gv.grid_cell_distance({"num_segments": 1}, (3, None)) == 2  # 3 - 1
    assert gv.grid_cell_distance({"num_segments": 0}, (3, None)) == 3  # 3 - 0


def test_grid_cell_distance_open_band_at_most():
    # iter-247: (None, 5) = "at most 5" — 0 for any count <= 5, gap above.
    for n in (0, 3, 5):
        assert gv.grid_cell_distance({"num_segments": n}, (None, 5)) == 0
    assert gv.grid_cell_distance({"num_segments": 7}, (None, 5)) == 2  # 7 - 5
    assert gv.grid_cell_distance({"num_segments": 6}, (None, 5)) == 1  # just above


def test_grid_cell_distance_set_is_min_over_elements():
    # iter-248: a set [3, 5, 7] scores 0 for any listed count and the gap to the
    # nearest listed count otherwise (min distance over the elements).
    for n in (3, 5, 7):
        assert gv.grid_cell_distance({"num_segments": n}, [3, 5, 7]) == 0
    assert gv.grid_cell_distance({"num_segments": 4}, [3, 5, 7]) == 1  # nearest 3/5
    assert gv.grid_cell_distance({"num_segments": 6}, [3, 5, 7]) == 1  # nearest 5/7
    assert gv.grid_cell_distance({"num_segments": 9}, [3, 5, 7]) == 2  # gap to 7
    assert gv.grid_cell_distance({"num_segments": 0}, [3, 5, 7]) == 3  # gap to 3


def test_grid_cell_distance_set_of_bands_is_min_over_elements():
    # iter-248: set elements may be bands — [(2, 3), (7, 8)] scores 0 inside
    # either band, the gap to the nearest band edge in the gap between them.
    target = [(2, 3), (7, 8)]
    for n in (2, 3, 7, 8):
        assert gv.grid_cell_distance({"num_segments": n}, target) == 0
    assert gv.grid_cell_distance({"num_segments": 5}, target) == 2  # 7 - 5 vs 5 - 3
    assert gv.grid_cell_distance({"num_segments": 4}, target) == 1  # 4 - 3
    assert gv.grid_cell_distance({"num_segments": 6}, target) == 1  # 7 - 6


def test_grid_cell_distance_preference_is_min_over_elements():
    # iter-249: a preference scores IDENTICALLY to a set — the MIN distance to any
    # listed element. The precedence only affects tie-breaking, never distance.
    target = {"prefer": [3, 5, 7]}
    for n in (3, 5, 7):
        assert gv.grid_cell_distance({"num_segments": n}, target) == 0
    assert gv.grid_cell_distance({"num_segments": 4}, target) == 1
    assert gv.grid_cell_distance({"num_segments": 9}, target) == 2
    # Equal to the same set, element-for-element.
    for n in range(0, 10):
        assert gv.grid_cell_distance(
            {"num_segments": n}, target
        ) == gv.grid_cell_distance({"num_segments": n}, [3, 5, 7])


def test_preference_rank_is_index_of_nearest_element():
    # iter-249: the rank is the index of the earliest preference element at the
    # minimum distance — count 3 ranks 0 (satisfies element 0), count 5 ranks 1.
    prefer = [3, 5, 7]
    assert gv._preference_rank({"num_segments": 3}, prefer) == 0
    assert gv._preference_rank({"num_segments": 5}, prefer) == 1
    assert gv._preference_rank({"num_segments": 7}, prefer) == 2
    # A count between two elements ranks toward the nearest; ties go earliest.
    assert gv._preference_rank({"num_segments": 4}, prefer) == 0  # |4-3|=1 vs |4-5|=1
    assert gv._preference_rank({"num_segments": 6}, prefer) == 1  # |6-5|=1 vs |6-7|=1


def test_pick_best_grid_cell_closest_to_target():
    # Counts 4,3,2,1 over the 2×2 grid; target 3 picks the second cell exactly.
    cells = _grid_cells([4, 3, 2, 1])
    best = gv.pick_best_grid_cell(cells, 3)
    assert best["num_segments"] == 3
    assert (best["threshold"], best["min_silence_ms"]) == (0.3, 800.0)


def test_pick_best_grid_cell_earliest_tie_wins():
    # Two cells are equidistant from target 3 (counts 2 and 4 both |Δ|=1); the
    # earlier one in row-major order wins, matching pick_best_mirror_config.
    cells = _grid_cells([2, 4, 2, 4])
    best = gv.pick_best_grid_cell(cells, 3)
    # First cell (count 2, |Δ|=1) wins over the later count-4 cell (also |Δ|=1).
    assert best["num_segments"] == 2
    assert (best["threshold"], best["min_silence_ms"]) == (0.3, 400.0)


def test_pick_best_grid_cell_empty_is_none():
    assert gv.pick_best_grid_cell([], 3) is None


def test_pick_best_grid_cell_exact_zero_target():
    cells = _grid_cells([3, 0, 2, 1])
    best = gv.pick_best_grid_cell(cells, 0)
    assert best["num_segments"] == 0


# ---- pick_top_grid_cells: the ranked shortlist -------------------------


def test_pick_top_grid_cells_ranked_nearest_first():
    # Counts 4,3,2,1 over the 2×2 grid; target 1 ranks 1,2,3 (distances 0,1,2).
    cells = _grid_cells([4, 3, 2, 1])
    top = gv.pick_top_grid_cells(cells, 1, 3)
    assert [c["num_segments"] for c in top] == [1, 2, 3]


def test_pick_top_grid_cells_head_is_the_best_pick():
    # The shortlist head must equal pick_best_grid_cell's single pick.
    cells = _grid_cells([4, 3, 2, 1])
    top = gv.pick_top_grid_cells(cells, 2, 3)
    assert top[0] == gv.pick_best_grid_cell(cells, 2)


def test_pick_top_grid_cells_stable_on_ties():
    # Counts 2,4,2,4; target 3 → all |Δ|=1. Stable sort keeps row-major order.
    cells = _grid_cells([2, 4, 2, 4])
    top = gv.pick_top_grid_cells(cells, 3, 4)
    assert [(c["threshold"], c["min_silence_ms"]) for c in top] == [
        (0.3, 400.0), (0.3, 800.0), (0.5, 400.0), (0.5, 800.0),
    ]


def test_pick_top_grid_cells_clamps_k_to_grid_size():
    # k larger than the grid simply returns every cell ranked.
    cells = _grid_cells([4, 1], row_values=[0.3], col_values=[400.0, 800.0])
    top = gv.pick_top_grid_cells(cells, 1, 10)
    assert len(top) == 2


def test_pick_top_grid_cells_empty_is_empty_list():
    assert gv.pick_top_grid_cells([], 3, 5) == []


def test_pick_top_grid_cells_does_not_mutate_input():
    cells = _grid_cells([4, 3, 2, 1])
    before = [c["num_segments"] for c in cells]
    gv.pick_top_grid_cells(cells, 1, 2)
    assert [c["num_segments"] for c in cells] == before


# ---- iter-243: grid_cell_sort_key + --tie-break ------------------------


def _seg_speech_cells(pairs):
    """Build minimal cell dicts from (num_segments, speech_s) pairs.

    The pickers read only those two keys, so a hand-built dict suffices to test
    tie ordering with EQUAL segment counts but DIFFERENT recovered speech —
    which ``_grid_cells`` (speech coupled to count) cannot express.
    """
    return [
        {
            "threshold": 0.3 + 0.1 * i,
            "min_silence_ms": 400.0,
            "num_segments": n,
            "speech_s": s,
        }
        for i, (n, s) in enumerate(pairs)
    ]


def test_grid_cell_sort_key_row_major_is_distance_only():
    cell = {"num_segments": 7, "speech_s": 4.0}
    assert gv.grid_cell_sort_key(cell, 5) == (2,)
    assert gv.grid_cell_sort_key(cell, 5, "row-major") == (2,)


def test_grid_cell_sort_key_speech_adds_negated_speech():
    cell = {"num_segments": 7, "speech_s": 4.0}
    assert gv.grid_cell_sort_key(cell, 5, "speech") == (2, -4.0)


def test_grid_cell_sort_key_preference_inserts_rank_after_distance():
    # iter-249: a preference target adds the preference rank as the secondary key,
    # right after distance. Two satisfying counts are distance-0 but rank differs.
    target = {"prefer": [3, 5]}
    assert gv.grid_cell_sort_key({"num_segments": 3, "speech_s": 1.0}, target) == (0, 0)
    assert gv.grid_cell_sort_key({"num_segments": 5, "speech_s": 9.0}, target) == (0, 1)


def test_grid_cell_sort_key_preference_then_speech_orders_keys():
    # iter-249: with both a preference target and the speech tie-break, the key is
    # (distance, preference_rank, -speech_s) — preference outranks speech.
    target = {"prefer": [3, 5]}
    cell = {"num_segments": 5, "speech_s": 4.0}
    assert gv.grid_cell_sort_key(cell, target, "speech") == (0, 1, -4.0)


def test_pick_best_grid_cell_preference_breaks_tie_toward_preferred():
    # iter-249: counts 5 and 3 both satisfy preference 3>5 (distance 0). The
    # preference picks the 3-count cell even though the 5-count one is EARLIER in
    # row-major order and recovered MORE speech — preference outranks both.
    cells = _seg_speech_cells([(5, 9.0), (3, 1.0)])
    best = gv.pick_best_grid_cell(cells, {"prefer": [3, 5]})
    assert best["num_segments"] == 3


def test_pick_best_grid_cell_preference_does_not_override_distance():
    # iter-249: a closer cell still wins regardless of preference order — distance
    # remains the primary key. Count 4 (distance 0 to element 1 '4') beats count 9
    # even though 9 satisfies the MORE-preferred element 0 only at distance 3... so
    # use a case where the preferred element is far: prefer 8>4, counts 9 and 4.
    cells = _seg_speech_cells([(9, 1.0), (4, 1.0)])
    best = gv.pick_best_grid_cell(cells, {"prefer": [8, 4]})
    # count 9 → dist 1 (to 8); count 4 → dist 0 (to 4). Distance wins: pick 4.
    assert best["num_segments"] == 4


def test_pick_top_grid_cells_preference_orders_runners_up():
    # iter-249: three cells all satisfy preference 3>5>7 (distance 0); the shortlist
    # ranks them by preference order, most-preferred first.
    cells = _seg_speech_cells([(7, 9.0), (3, 1.0), (5, 2.0)])
    top = gv.pick_top_grid_cells(cells, {"prefer": [3, 5, 7]}, 3)
    assert [c["num_segments"] for c in top] == [3, 5, 7]


# ---- weighted set: penalty folds preference into distance (iter-250) ---


def test_grid_cell_distance_weighted_adds_penalty_min_over_elements():
    # iter-250: each element's distance is its raw distance PLUS its penalty; the
    # set scores as the MIN over those penalised distances.
    target = {"weighted": [(3, 0), (5, 2)]}
    assert gv.grid_cell_distance({"num_segments": 3}, target) == 0  # 0+0 vs 2+2
    assert gv.grid_cell_distance({"num_segments": 5}, target) == 2  # 2+0 vs 0+2
    assert gv.grid_cell_distance({"num_segments": 4}, target) == 1  # 1+0 vs 1+2


def test_grid_cell_distance_weighted_lets_preferred_win_at_larger_raw_distance():
    # iter-250: THE point of the form. Preferred 3 (penalty 0), accepted 5
    # (penalty 2). Count 5 lands EXACTLY on the accepted element (raw distance 0)
    # but pays its +2 → 2. Count 4 is raw distance 1 from the preferred 3 → 1, so
    # it BEATS the exact-accepted count 5. The penalty overrides a raw-distance
    # gap — unlike iter-249's preference, which only breaks exact distance ties.
    target = {"weighted": [(3, 0), (5, 2)]}
    assert gv.grid_cell_distance({"num_segments": 5}, target) == 2
    assert gv.grid_cell_distance({"num_segments": 4}, target) == 1


def test_grid_cell_sort_key_weighted_inserts_no_secondary_key():
    # iter-250: the weighted set's preference is already baked into the distance,
    # so the sort key carries NO secondary preference-rank component — equal
    # penalised distance is a genuine tie left to the tie_break.
    target = {"weighted": [(3, 0), (5, 2)]}
    assert gv.grid_cell_sort_key({"num_segments": 3, "speech_s": 1.0}, target) == (0,)
    assert gv.grid_cell_sort_key(
        {"num_segments": 5, "speech_s": 1.0}, target, "speech"
    ) == (2, -1.0)


def test_pick_best_grid_cell_weighted_overrides_distance_gap():
    # iter-250: count 5 lands EXACTLY on accepted element 5 (raw dist 0, penalised
    # 2); count 4 is raw dist 1 from preferred 3 (penalised 1). The penalty flips
    # the pick to count 4 — beating the raw-distance gap (unlike iter-249).
    cells = _seg_speech_cells([(5, 9.0), (4, 1.0)])
    best = gv.pick_best_grid_cell(cells, {"weighted": [(3, 0), (5, 2)]})
    assert best["num_segments"] == 4


# ---- scaled set: factor folds preference into distance MULTIPLICATIVELY (iter-252) ---


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", 1),
        ("2", 2),
        ("2.0", 2),  # an integral float collapses to int
        ("1.5", 1.5),
        ("3.25", 3.25),
    ],
)
def test_scale_factor_parses_number(raw, expected):
    # iter-252: the factor slot accepts a number >= 1; an integral float collapses
    # to an int so a whole-number factor renders/serialises as a plain int.
    value = gv.scale_factor_type(raw)
    assert value == expected
    assert type(value) is type(expected)


@pytest.mark.parametrize("raw", ["0", "0.5", "-1", "-0.5", "nan", "inf", "abc", ""])
def test_scale_factor_rejects_bad(raw):
    # iter-252: a factor below 1 would DISCOUNT an element's distance (the other
    # elements' larger factors already express that), and NaN/inf/non-numeric are
    # nonsensical.
    with pytest.raises(argparse.ArgumentTypeError):
        gv.scale_factor_type(raw)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("3,5*1.5", {"scaled": [(3, 1), (5, 1.5)]}),
        ("3*2,5*1.5", {"scaled": [(3, 2), (5, 1.5)]}),
        ("3,5-7*1.5", {"scaled": [(3, 1), ((5, 7), 1.5)]}),  # a band may be scaled
        ("3,5*2.0", {"scaled": [(3, 1), (5, 2)]}),  # integral float collapses to int
    ],
)
def test_target_type_scaled_accepts_factor(raw, expected):
    # iter-252: a '*factor' weight parses to a {"scaled": [...]} dict; a fractional
    # factor stays a float, an integral one collapses to int.
    value = gv.target_type(raw)
    assert value == expected
    assert isinstance(value["scaled"][1][1], type(expected["scaled"][1][1]))


def test_target_type_scaled_dedupes_first_factor_wins():
    # iter-252: dedupe on the element preserving first-seen order — the first factor
    # for a repeated element wins.
    assert gv.target_type("3*2,5*1.5,3*9") == {"scaled": [(3, 2), (5, 1.5)]}


def test_target_type_scaled_single_element_collapses():
    # iter-252: a scaled set that reduces to one element drops the factor (a lone
    # factor scales every cell uniformly and cannot change a pick), so scalar
    # output is byte-for-byte unchanged.
    assert gv.target_type("5*2,5*3") == 5


def test_target_type_rejects_factor_without_set():
    # iter-252: a '*' factor is meaningless on a single element, so it requires the
    # ',' set context.
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type("3*1.5")
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type("3-5*1.5")  # a lone band with a factor is still single


@pytest.mark.parametrize("raw", ["3,5*1.5>7", "3*1.5>7", "3,5*1.5,7>9"])
def test_target_type_rejects_mixing_factor_and_preference(raw):
    # iter-252: '*' (factor) and '>' (preference) both express preference; stacking
    # them is ambiguous.
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type(raw)


@pytest.mark.parametrize("raw", ["3,5:2*1.5", "3*1.5,5:2", "3,5*1.5:2"])
def test_target_type_rejects_mixing_factor_and_penalty(raw):
    # iter-252: ':' (additive penalty) and '*' (multiplicative factor) are two ways
    # to weight one set; a set is one OR the other, not both at once.
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type(raw)


@pytest.mark.parametrize("raw", ["3,5*a", "3,5*0.5", "3,a*2", "3,5-3*2"])
def test_target_type_rejects_malformed_scaled_element(raw):
    # iter-252: a non-number/below-1 factor, a bad base, or an inverted band element
    # fails the whole target.
    with pytest.raises(argparse.ArgumentTypeError):
        gv.target_type(raw)


def test_format_target_scaled():
    # iter-252: a scaled set renders comma-joined, each non-neutral factor appended
    # as '*factor' (a neutral factor 1 stays bare), so it reads back as typed.
    assert gv._format_target({"scaled": [(3, 1), (5, 1.5)]}) == "3,5*1.5"
    assert gv._format_target({"scaled": [(3, 2), (5, 1.5)]}) == "3*2,5*1.5"
    assert gv._format_target({"scaled": [(3, 1), ((5, 7), 1.5)]}) == "3,5-7*1.5"


def test_grid_cell_distance_scaled_multiplies_min_over_elements():
    # iter-252: each element's distance is its raw distance TIMES its factor; the
    # set scores as the MIN over those scaled distances. An exact hit stays free.
    target = {"scaled": [(3, 1), (5, 1.5)]}
    assert gv.grid_cell_distance({"num_segments": 3}, target) == 0  # 0*1 vs 2*1.5
    assert gv.grid_cell_distance({"num_segments": 5}, target) == 0  # 2*1 vs 0*1.5
    assert gv.grid_cell_distance({"num_segments": 7}, target) == 3  # 4*1 vs 2*1.5=3


def test_grid_cell_distance_scaled_grows_cost_with_distance():
    # iter-252: THE point of the form vs the additive penalty. Preferred 3
    # (factor 1), accepted 5 (factor 2). The exact-accepted count 5 stays FREE
    # (0*2=0) — unlike an additive penalty, which bites even an exact hit. But
    # drifting one past 5 to count 6 costs 1*2=2, double the raw drift, so the
    # cost scales with distance rather than offsetting by a constant.
    target = {"scaled": [(3, 1), (5, 2)]}
    assert gv.grid_cell_distance({"num_segments": 5}, target) == 0  # exact hit free
    assert gv.grid_cell_distance({"num_segments": 6}, target) == 2  # 3*1=3 vs 1*2=2
    assert gv.grid_cell_distance({"num_segments": 4}, target) == 1  # 1*1=1 vs 1*2=2


def test_grid_cell_sort_key_scaled_inserts_no_secondary_key():
    # iter-252: the scaled set's preference is already baked into the distance (the
    # factor), so the sort key carries NO secondary preference-rank component.
    target = {"scaled": [(3, 1), (5, 2)]}
    assert gv.grid_cell_sort_key({"num_segments": 3, "speech_s": 1.0}, target) == (0,)
    assert gv.grid_cell_sort_key(
        {"num_segments": 7, "speech_s": 1.0}, target, "speech"
    ) == (4, -1.0)


def test_pick_best_grid_cell_scaled_grows_cost_with_distance():
    # iter-252: preferred 3 (factor 1), accepted 5 (factor 2). Count 6 (one past the
    # accepted 5) costs 1*2=2; count 4 (one past the preferred 3) costs 1*1=1, so
    # the pick leans toward the lower-factor element as cells drift away.
    cells = _seg_speech_cells([(6, 9.0), (4, 1.0)])
    best = gv.pick_best_grid_cell(cells, {"scaled": [(3, 1), (5, 2)]})
    assert best["num_segments"] == 4


def test_render_grid_json_carries_scaled_target():
    # iter-252: a scaled target serialises as {"scaled": [...]} — each element a
    # nested array, the factor a JSON number (a band element nests its own [lo,hi]).
    results = [_cell_result(n) for n in (4, 7)]
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [400.0, 800.0], results, name="rec.wav",
            target={"scaled": [(3, 1), ((5, 7), 1.5)]},
        )
    )
    assert payload["target"] == {"scaled": [[3, 1], [[5, 7], 1.5]]}


def test_render_grid_scaled_target_line_reads_back():
    # iter-252: the best: line renders the scaled weight as typed and shows the
    # scaled |Δ| with no dict repr leaking.
    results = [_cell_result(n) for n in (6, 4)]
    lines = gv.render_vad_grid(
        [0.3], [400.0, 800.0], results, name="rec.wav",
        target={"scaled": [(3, 1), (5, 2)]},
    )
    best_line = lines[-1]
    assert "4 segments" in best_line
    assert "target 3,5*2" in best_line
    assert "scaled" not in best_line


def test_pick_best_grid_cell_speech_tie_break_prefers_most_speech():
    # Both cells are |Δ|=1 from target 3; speech tie-break picks the 5.0s cell
    # even though the 2.0s cell is earlier in row-major order.
    cells = _seg_speech_cells([(4, 2.0), (4, 5.0)])
    best = gv.pick_best_grid_cell(cells, 3, "speech")
    assert best["speech_s"] == 5.0


def test_pick_best_grid_cell_row_major_keeps_earlier_on_tie():
    # Same tie, but the default tie-break keeps the earlier (2.0s) cell.
    cells = _seg_speech_cells([(4, 2.0), (4, 5.0)])
    best = gv.pick_best_grid_cell(cells, 3)
    assert best["speech_s"] == 2.0


def test_pick_best_grid_cell_speech_does_not_override_distance():
    # A closer cell wins regardless of speech: distance is the primary key.
    cells = _seg_speech_cells([(4, 9.0), (3, 0.1)])
    best = gv.pick_best_grid_cell(cells, 3, "speech")
    assert best["num_segments"] == 3


def test_pick_top_grid_cells_speech_tie_break_orders_runners_up():
    # Three cells all |Δ|=1 from target 3; speech tie-break ranks them by
    # recovered speech, most first.
    cells = _seg_speech_cells([(4, 1.0), (2, 3.0), (4, 2.0)])
    top = gv.pick_top_grid_cells(cells, 3, 3, "speech")
    assert [c["speech_s"] for c in top] == [3.0, 2.0, 1.0]


def test_pick_top_grid_cells_speech_head_is_best_pick():
    cells = _seg_speech_cells([(4, 1.0), (4, 5.0)])
    top = gv.pick_top_grid_cells(cells, 3, 2, "speech")
    assert top[0] == gv.pick_best_grid_cell(cells, 3, "speech")


def test_pick_best_grid_cell_empty_is_none_with_tie_break():
    assert gv.pick_best_grid_cell([], 3, "speech") is None


# ---- renderers ----------------------------------------------------------


def test_render_grid_none_is_install_hint():
    lines = gv.render_vad_grid([0.5], [400.0], [None], name="rec.wav")
    assert len(lines) == 1
    assert "silero VAD unavailable" in lines[0]


def test_render_grid_labels_both_axes():
    results = [_cell_result(n) for n in (3, 1)]
    lines = gv.render_vad_grid(
        [0.3], [400.0, 800.0], results, name="rec.wav",
        row_axis="threshold", col_axis="min_silence_ms",
    )
    assert "threshold × min_silence" in lines[0]
    assert "rec.wav" in lines[0]
    # Column header carries both axis labels.
    assert "threshold" in lines[1] and "min_silence" in lines[1]
    # A gate prints with 2 decimals, a ms knob as a bare integer.
    body = "\n".join(lines[2:])
    assert "0.30" in body
    assert "400" in body and "800" in body


def test_render_grid_speech_axis_column_label():
    results = [_cell_result(2)]
    lines = gv.render_vad_grid(
        [0.5], [100.0], results, name="rec.wav",
        row_axis="threshold", col_axis="min_speech_ms",
    )
    assert "threshold × min_speech" in lines[0]


def test_render_grid_json_carries_both_axes():
    results = [_cell_result(n) for n in (3, 1)]
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [400.0, 800.0], results, name="rec.wav",
            row_axis="threshold", col_axis="min_silence_ms",
        )
    )
    assert payload["available"] is True
    assert payload["name"] == "rec.wav"
    assert payload["row_axis"] == "threshold"
    assert payload["col_axis"] == "min_silence_ms"
    assert [(c["threshold"], c["min_silence_ms"]) for c in payload["grid"]] == [
        (0.3, 400.0), (0.3, 800.0),
    ]
    assert [c["num_segments"] for c in payload["grid"]] == [3, 1]


def test_render_grid_json_none_marks_unavailable():
    payload = json.loads(gv.render_vad_grid_json([0.5], [400.0], [None], name="rec.wav"))
    assert payload["available"] is False
    assert "hint" in payload


# ---- the --target best-cell pick in the renderers ----------------------


def test_render_grid_omits_best_line_without_target():
    results = [_cell_result(n) for n in (4, 1)]
    lines = gv.render_vad_grid([0.3], [400.0, 800.0], results, name="rec.wav")
    assert not any("best:" in ln for ln in lines)


def test_render_grid_appends_best_line_with_target():
    # 1×2 grid, counts 4 and 1; target 2 picks the count-1 cell (|Δ|=1 vs 2).
    results = [_cell_result(n) for n in (4, 1)]
    lines = gv.render_vad_grid(
        [0.3], [400.0, 800.0], results, name="rec.wav", target=2,
    )
    best_line = lines[-1]
    assert best_line.lstrip().startswith("best:")
    assert "min_silence=800" in best_line
    assert "threshold=0.30" in best_line
    assert "1 segments" in best_line
    assert "target 2" in best_line


def test_render_grid_best_line_empty_grid():
    lines = gv.render_vad_grid([], [], [], name="rec.wav", target=3)
    assert lines[-1].lstrip().startswith("best: none")


def test_render_grid_band_target_picks_in_band_cell():
    # iter-246: 1×2 grid counts 4 and 1; band 3-5 puts 4 inside the band (|Δ|=0),
    # so it wins over the count-1 cell (|Δ|=2 below lo).
    results = [_cell_result(n) for n in (4, 1)]
    lines = gv.render_vad_grid(
        [0.3], [400.0, 800.0], results, name="rec.wav", target=(3, 5),
    )
    best_line = lines[-1]
    assert best_line.lstrip().startswith("best:")
    assert "4 segments" in best_line
    assert "|Δ|=0" in best_line
    # The band renders as "3-5", not a tuple repr.
    assert "target 3-5" in best_line
    assert "(3, 5)" not in best_line


def test_render_grid_band_empty_grid_renders_band_text():
    lines = gv.render_vad_grid([], [], [], name="rec.wav", target=(3, 5))
    assert "target 3-5 segments" in lines[-1]


def test_render_grid_open_band_at_least_picks_satisfying_cell():
    # iter-247: counts 4 and 1; band 3- ("at least 3") satisfies the 4-count cell
    # (|Δ|=0) and the 1-count cell is short by 2, so the 4-count cell wins.
    results = [_cell_result(n) for n in (4, 1)]
    lines = gv.render_vad_grid(
        [0.3], [400.0, 800.0], results, name="rec.wav", target=(3, None),
    )
    best_line = lines[-1]
    assert best_line.lstrip().startswith("best:")
    assert "4 segments" in best_line
    assert "|Δ|=0" in best_line
    # The open band renders as "3-", not a tuple repr with None.
    assert "target 3-" in best_line
    assert "None" not in best_line


def test_render_grid_open_band_at_most_picks_satisfying_cell():
    # iter-247: counts 4 and 8; band -5 ("at most 5") satisfies the 4-count cell
    # (|Δ|=0) and the 8-count cell is over by 3, so the 4-count cell wins.
    results = [_cell_result(n) for n in (4, 8)]
    lines = gv.render_vad_grid(
        [0.3], [400.0, 800.0], results, name="rec.wav", target=(None, 5),
    )
    best_line = lines[-1]
    assert "4 segments" in best_line
    assert "|Δ|=0" in best_line
    assert "target -5" in best_line
    assert "None" not in best_line


def test_render_grid_json_omits_best_and_target_without_target():
    results = [_cell_result(n) for n in (4, 1)]
    payload = json.loads(
        gv.render_vad_grid_json([0.3], [400.0, 800.0], results, name="rec.wav")
    )
    assert "best" not in payload
    assert "target" not in payload


def test_render_grid_json_carries_best_and_target():
    results = [_cell_result(n) for n in (4, 1)]
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [400.0, 800.0], results, name="rec.wav", target=2,
        )
    )
    assert payload["target"] == 2
    assert payload["best"]["num_segments"] == 1
    assert payload["best"]["distance"] == 1
    assert payload["best"]["min_silence_ms"] == 800.0


def test_render_grid_json_carries_band_target():
    # iter-246: a band target serialises as a [lo, hi] JSON array, and the best
    # cell's distance is 0 when its count lands inside the band.
    results = [_cell_result(n) for n in (4, 1)]
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [400.0, 800.0], results, name="rec.wav", target=(3, 5),
        )
    )
    assert payload["target"] == [3, 5]
    assert payload["best"]["num_segments"] == 4
    assert payload["best"]["distance"] == 0


def test_render_grid_json_carries_open_band_target():
    # iter-247: an open band serialises with null for the open edge — (3, None)
    # → [3, null], (None, 5) → [null, 5]. The satisfied cell scores distance 0.
    results = [_cell_result(n) for n in (4, 1)]
    at_least = json.loads(
        gv.render_vad_grid_json(
            [0.3], [400.0, 800.0], results, name="rec.wav", target=(3, None),
        )
    )
    assert at_least["target"] == [3, None]
    assert at_least["best"]["num_segments"] == 4
    assert at_least["best"]["distance"] == 0

    at_most = json.loads(
        gv.render_vad_grid_json(
            [0.3], [400.0, 800.0], [_cell_result(n) for n in (4, 8)],
            name="rec.wav", target=(None, 5),
        )
    )
    assert at_most["target"] == [None, 5]
    assert at_most["best"]["num_segments"] == 4
    assert at_most["best"]["distance"] == 0


def test_render_grid_set_target_picks_satisfying_cell():
    # iter-248: counts 4 and 5; set 3,5,7 satisfies the 5-count cell (|Δ|=0) and
    # the 4-count cell is off by 1, so the 5-count cell wins.
    results = [_cell_result(n) for n in (4, 5)]
    lines = gv.render_vad_grid(
        [0.3], [400.0, 800.0], results, name="rec.wav", target=[3, 5, 7],
    )
    best_line = lines[-1]
    assert best_line.lstrip().startswith("best:")
    assert "5 segments" in best_line
    assert "|Δ|=0" in best_line
    # The set renders as "3,5,7", not a list repr.
    assert "target 3,5,7" in best_line
    assert "[3, 5, 7]" not in best_line


def test_render_grid_json_carries_set_target():
    # iter-248: a set target serialises as a JSON array of its elements (a band
    # element nests as its own [lo, hi] array). The satisfied cell scores 0.
    results = [_cell_result(n) for n in (4, 5)]
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [400.0, 800.0], results, name="rec.wav", target=[3, 5, 7],
        )
    )
    assert payload["target"] == [3, 5, 7]
    assert payload["best"]["num_segments"] == 5
    assert payload["best"]["distance"] == 0


def test_render_grid_preference_target_breaks_tie_toward_preferred():
    # iter-249: counts 3 and 5 both satisfy preference 5>3 (distance 0); the
    # preference picks the 5-count cell even though the 3-count cell is earlier in
    # row-major order. The preference renders as "5>3", not a dict repr.
    results = [_cell_result(n) for n in (3, 5)]
    lines = gv.render_vad_grid(
        [0.3], [400.0, 800.0], results, name="rec.wav", target={"prefer": [5, 3]},
    )
    best_line = lines[-1]
    assert best_line.lstrip().startswith("best:")
    assert "5 segments" in best_line
    assert "|Δ|=0" in best_line
    assert "target 5>3" in best_line
    assert "prefer" not in best_line


def test_render_grid_json_carries_preference_target():
    # iter-249: a preference target serialises as {"prefer": [...]} — a JSON object
    # carrying the listed order (a band element nests as its own [lo, hi] array).
    results = [_cell_result(n) for n in (4, 5)]
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [400.0, 800.0], results, name="rec.wav",
            target={"prefer": [3, (5, 7)]},
        )
    )
    assert payload["target"] == {"prefer": [3, [5, 7]]}
    assert payload["best"]["num_segments"] == 5
    assert payload["best"]["distance"] == 0


def test_render_grid_weighted_target_overrides_distance_gap():
    # iter-250: preferred 3 (penalty 0), accepted 6 (penalty 2). Count 6 lands
    # exactly on the accepted element (penalised 2); count 4 is raw dist 1 from the
    # preferred 3 (penalised 1), so the +2 penalty flips the pick to the 4-count
    # cell. The weighted set renders as "3,6:2" with no dict repr; |Δ| is penalised.
    results = [_cell_result(n) for n in (6, 4)]
    lines = gv.render_vad_grid(
        [0.3], [400.0, 800.0], results, name="rec.wav",
        target={"weighted": [(3, 0), (6, 2)]},
    )
    best_line = lines[-1]
    assert best_line.lstrip().startswith("best:")
    assert "4 segments" in best_line
    assert "|Δ|=1" in best_line
    assert "target 3,6:2" in best_line
    assert "weighted" not in best_line


def test_render_grid_json_carries_weighted_target():
    # iter-250: a weighted target serialises as {"weighted": [...]} — each element
    # a [element, penalty] pair (a band element nests as its own [lo, hi] array).
    results = [_cell_result(n) for n in (4, 5)]
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [400.0, 800.0], results, name="rec.wav",
            target={"weighted": [(3, 0), ((5, 7), 2)]},
        )
    )
    assert payload["target"] == {"weighted": [[3, 0], [[5, 7], 2]]}
    # count 4 → min(|4-3|+0, dist-to-band+2) = min(1, 0+2) = 1; count 5 → min(2, 2).
    assert payload["best"]["num_segments"] == 4
    assert payload["best"]["distance"] == 1


def test_render_grid_json_best_is_none_for_empty_grid():
    payload = json.loads(
        gv.render_vad_grid_json([], [], [], name="rec.wav", target=3)
    )
    assert payload["target"] == 3
    assert payload["best"] is None


# ---- the --top ranked shortlist in the renderers -----------------------


def test_render_grid_omits_top_block_without_top():
    # --target alone keeps just the best line; no "top N" block.
    results = [_cell_result(n) for n in (4, 3, 2, 1)]
    lines = gv.render_vad_grid(
        [0.3, 0.5], [400.0, 800.0], results, name="rec.wav", target=1,
    )
    assert not any("top " in ln for ln in lines)


def test_render_grid_top_requires_target():
    # top without target → no distance to rank by, so no shortlist appears.
    results = [_cell_result(n) for n in (4, 3, 2, 1)]
    lines = gv.render_vad_grid(
        [0.3, 0.5], [400.0, 800.0], results, name="rec.wav", top=3,
    )
    assert not any("top " in ln for ln in lines)
    assert not any("best:" in ln for ln in lines)


def test_render_grid_appends_top_block_with_target_and_top():
    # 2×2 grid counts 4,3,2,1; target 1, top 3 → ranks 1,2,3 (distances 0,1,2).
    results = [_cell_result(n) for n in (4, 3, 2, 1)]
    lines = gv.render_vad_grid(
        [0.3, 0.5], [400.0, 800.0], results, name="rec.wav", target=1, top=3,
    )
    text = "\n".join(lines)
    assert "top 3 (closest to target 1):" in text
    # Three ranked rows, nearest first.
    assert "1. " in text and "2. " in text and "3. " in text
    rank1 = next(ln for ln in lines if ln.lstrip().startswith("1."))
    assert "1 segments" in rank1 and "|Δ|=0" in rank1


def test_render_grid_top_block_clamps_to_grid_size():
    # top 10 over a 2-cell grid lists only the 2 real cells.
    results = [_cell_result(n) for n in (4, 1)]
    lines = gv.render_vad_grid(
        [0.3], [400.0, 800.0], results, name="rec.wav", target=1, top=10,
    )
    assert "top 2 (closest to target 1):" in "\n".join(lines)


def test_render_grid_top_block_omitted_for_empty_grid():
    # Empty grid → best: none, and no shortlist block.
    lines = gv.render_vad_grid([], [], [], name="rec.wav", target=3, top=5)
    assert lines[-1].lstrip().startswith("best: none")
    assert not any("top " in ln for ln in lines)


def test_render_grid_json_omits_top_without_top():
    results = [_cell_result(n) for n in (4, 1)]
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [400.0, 800.0], results, name="rec.wav", target=2,
        )
    )
    assert "top" not in payload


def test_render_grid_json_carries_top_list():
    results = [_cell_result(n) for n in (4, 3, 2, 1)]
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3, 0.5], [400.0, 800.0], results, name="rec.wav",
            target=1, top=3,
        )
    )
    assert [c["num_segments"] for c in payload["top"]] == [1, 2, 3]
    assert [c["distance"] for c in payload["top"]] == [0, 1, 2]
    # Head of the shortlist equals best.
    assert payload["top"][0] == payload["best"]


def test_render_grid_json_top_ignored_without_target():
    results = [_cell_result(n) for n in (4, 1)]
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [400.0, 800.0], results, name="rec.wav", top=3,
        )
    )
    assert "top" not in payload
    assert "best" not in payload


def _result_n_speech(n, speech_s):
    """A _Result with ``n`` segments whose durations sum to ``speech_s``.

    Lets a renderer test build cells with EQUAL segment counts but DIFFERENT
    recovered speech — needed to exercise the speech tie-break end-to-end.
    """
    if n == 0:
        return _Result(name="rec.wav", sample_rate=16000, duration_s=10.0)
    per = speech_s / n
    segs = [_Seg(2.0 * i, 2.0 * i + per) for i in range(n)]
    return _Result(name="rec.wav", sample_rate=16000, duration_s=100.0, segments=segs)


def test_render_grid_speech_tie_break_picks_most_speech():
    # 1×2 grid, both cells 4 segments (|Δ|=1 from target 3) but col0 recovers
    # 2.0s, col1 5.0s. Speech tie-break names the 5.0s cell (min_silence=800),
    # not the earlier col0.
    results = [_result_n_speech(4, 2.0), _result_n_speech(4, 5.0)]
    lines = gv.render_vad_grid(
        [0.3], [400.0, 800.0], results, name="rec.wav",
        target=3, tie_break="speech",
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "min_silence=800" in best


def test_render_grid_row_major_default_keeps_earlier_tie():
    # Same grid, default tie-break keeps the earlier (min_silence=400) cell.
    results = [_result_n_speech(4, 2.0), _result_n_speech(4, 5.0)]
    lines = gv.render_vad_grid(
        [0.3], [400.0, 800.0], results, name="rec.wav", target=3,
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "min_silence=400" in best


def test_render_grid_json_carries_tie_break_key():
    results = [_cell_result(n) for n in (4, 1)]
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [400.0, 800.0], results, name="rec.wav",
            target=2, tie_break="speech",
        )
    )
    assert payload["tie_break"] == "speech"


def test_render_grid_json_tie_break_defaults_row_major():
    results = [_cell_result(n) for n in (4, 1)]
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [400.0, 800.0], results, name="rec.wav", target=2,
        )
    )
    assert payload["tie_break"] == "row-major"


def test_render_grid_json_tie_break_absent_without_target():
    # No target → no pick keys at all, tie_break included.
    results = [_cell_result(n) for n in (4, 1)]
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [400.0, 800.0], results, name="rec.wav",
        )
    )
    assert "tie_break" not in payload


def test_render_grid_json_speech_tie_break_orders_top_list():
    # Two 4-segment cells (|Δ|=1) with different speech; speech tie-break puts
    # the higher-speech cell first in the top list.
    results = [_result_n_speech(4, 2.0), _result_n_speech(4, 5.0)]
    payload = json.loads(
        gv.render_vad_grid_json(
            [0.3], [400.0, 800.0], results, name="rec.wav",
            target=3, top=2, tie_break="speech",
        )
    )
    assert payload["top"][0]["speech_s"] == 5.0
    assert payload["top"][0] == payload["best"]


def test_render_grid_csv_header_and_rows():
    results = [_cell_result(n) for n in (3, 1)]
    text = gv.render_vad_grid_csv(
        [0.3], [400.0, 800.0], results, name="rec.wav",
        row_axis="threshold", col_axis="min_silence_ms",
    )
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["threshold", "min_silence_ms", "num_segments", "speech_s"]
    assert rows[1][:3] == ["0.3", "400.0", "3"]
    assert rows[2][:3] == ["0.3", "800.0", "1"]


def test_render_grid_csv_no_trailing_newline():
    text = gv.render_vad_grid_csv([0.5], [400.0], [_cell_result(1)], name="rec.wav")
    assert not text.endswith("\n")
    assert not text.endswith("\r")


def test_render_grid_csv_none_marks_unavailable():
    text = gv.render_vad_grid_csv([0.5], [400.0], [None], name="rec.wav")
    assert text.startswith("# silero VAD unavailable")


def test_render_grid_csv_round_trips_to_grid_cells():
    results = [_cell_result(n) for n in (4, 3, 2, 1)]
    text = gv.render_vad_grid_csv(
        [0.3, 0.5], [400.0, 800.0], results, name="rec.wav",
        row_axis="threshold", col_axis="min_silence_ms",
    )
    cells = gv.vad_segmentation_grid(
        [0.3, 0.5], [400.0, 800.0], results,
        row_axis="threshold", col_axis="min_silence_ms",
    )
    rows = list(csv.reader(io.StringIO(text)))
    for csv_row, cell in zip(rows[1:], cells):
        assert float(csv_row[0]) == cell["threshold"]
        assert float(csv_row[1]) == cell["min_silence_ms"]
        assert int(csv_row[2]) == cell["num_segments"]
        assert float(csv_row[3]) == cell["speech_s"]


def test_render_grid_csv_round_trips_to_json_twin_on_nondefault_axes():
    # iter-270: the 2-D analogue of the iter-268/269 1-D CSV↔JSON twins. The
    # grid round-trip above (test_render_grid_csv_round_trips_to_grid_cells)
    # compares the CSV body against the SHARED data layer (vad_segmentation_grid
    # cells) on the DEFAULT axis pair (threshold × min_silence_ms). But the two
    # machine surfaces — render_vad_grid_csv and render_vad_grid_json — were
    # never round-tripped DIRECTLY against each other, and never on a NON-default
    # row_axis/col_axis. Both emitters are axis-agnostic (each stringifies
    # whichever value the row_axis/col_axis kwarg names), so a regression that
    # let the CSV's first two columns drift from the JSON cell keys, reordered
    # the row-major cell emission on one surface but not the other, or truncated
    # a later cell's value on just one surface, would have shipped green while
    # the default-axes grid-cells round-trip and the 1-D ms-axis twins stayed
    # passing. Pin the CSV body and the JSON `grid` payload to describe the SAME
    # segmentation, cell for cell, on a fully non-default axis pair
    # (min_speech_ms rows × speech_pad_ms columns).
    row_values = [200.0, 400.0]
    col_values = [30.0, 90.0, 150.0]
    # 2 rows × 3 cols = 6 cells, row-major counts 6/4/2 then 5/3/1.
    results = [_cell_result(n) for n in (6, 4, 2, 5, 3, 1)]
    csv_text = gv.render_vad_grid_csv(
        row_values, col_values, results, name="rec.wav",
        row_axis="min_speech_ms", col_axis="speech_pad_ms",
    )
    grid = json.loads(
        gv.render_vad_grid_json(
            row_values, col_values, results, name="rec.wav",
            row_axis="min_speech_ms", col_axis="speech_pad_ms",
        )
    )["grid"]
    csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
    # The first two CSV column headers ARE the swept axis names (self-describing
    # CSV), keyed identically to the JSON cells.
    assert csv_rows[0].keys() >= {
        "min_speech_ms", "speech_pad_ms", "num_segments", "speech_s"
    }
    assert [
        {
            "min_speech_ms": float(row["min_speech_ms"]),
            "speech_pad_ms": float(row["speech_pad_ms"]),
            "num_segments": int(row["num_segments"]),
            "speech_s": float(row["speech_s"]),
        }
        for row in csv_rows
    ] == grid


# ---- cmd_vad_grid: end-to-end ------------------------------------------


def test_cmd_vad_grid_unavailable_emits_hint():
    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no seg")),
        availability=lambda: False,
    )
    text = "\n".join(lines)
    assert "silero VAD unavailable" in text


def test_cmd_vad_grid_unavailable_json():
    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(json=True),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no seg")),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert json.loads(lines[0])["available"] is False


def test_cmd_vad_grid_sweeps_full_cartesian_product():
    # 2 thresholds × 2 hangovers = 4 cells, row-major over (threshold, hangover).
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        return _cell_result(1)

    gv.cmd_vad_grid(
        _grid_args(thresholds=[0.3, 0.5], min_silences=[400.0, 800.0]),
        log=lambda *a: None,
        segmenter=seg,
        availability=lambda: True,
    )
    assert [(p.threshold, p.min_silence_ms) for p in captured] == [
        (0.3, 400.0), (0.3, 800.0), (0.5, 400.0), (0.5, 800.0),
    ]


def test_cmd_vad_grid_holds_nonaxis_knobs_fixed():
    # When the column axis is the hangover, the scalar --min-speech-ms is shared
    # by every cell (it is NOT swept); speech_pad / max_speech too.
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        return _cell_result(1)

    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], min_silences=[400.0, 800.0],
            min_speech_ms=333.0, speech_pad_ms=42.0,
        ),
        log=lambda *a: None,
        segmenter=seg,
        availability=lambda: True,
    )
    assert {p.min_speech_ms for p in captured} == {333.0}
    assert {p.speech_pad_ms for p in captured} == {42.0}


def test_cmd_vad_grid_speech_column_holds_silence_scalar():
    # When the column axis is --min-speeches, min_speech_ms is SWEPT and the
    # scalar --min-silence-ms is held fixed across every cell.
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        return _cell_result(1)

    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], min_silences=[400.0, 800.0],
            min_speeches=[50.0, 100.0], min_silence_ms=777.0,
        ),
        log=lambda *a: None,
        segmenter=seg,
        availability=lambda: True,
    )
    assert [(p.threshold, p.min_speech_ms) for p in captured] == [
        (0.3, 50.0), (0.3, 100.0), (0.5, 50.0), (0.5, 100.0),
    ]
    assert {p.min_silence_ms for p in captured} == {777.0}


def test_cmd_vad_grid_speech_pad_column_sweeps_pad_holds_others():
    # iter-254: when the column axis is --speech-pads, speech_pad_ms is SWEPT
    # and the scalar --min-silence-ms / --min-speech-ms are held fixed across
    # every cell (the shared --speech-pad-ms scalar is NOT used as a value).
    captured = []

    def seg(wav, params=None):
        captured.append(params)
        return _cell_result(1)

    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], speech_pads=[0.0, 40.0],
            min_silence_ms=777.0, min_speech_ms=333.0, speech_pad_ms=999.0,
        ),
        log=lambda *a: None,
        segmenter=seg,
        availability=lambda: True,
    )
    assert [(p.threshold, p.speech_pad_ms) for p in captured] == [
        (0.3, 0.0), (0.3, 40.0), (0.5, 0.0), (0.5, 40.0),
    ]
    assert {p.min_silence_ms for p in captured} == {777.0}
    assert {p.min_speech_ms for p in captured} == {333.0}
    # The shared --speech-pad-ms scalar (999) is the swept axis now, so it is
    # never used as a held value.
    assert 999.0 not in {p.speech_pad_ms for p in captured}


def test_cmd_vad_grid_speech_pad_column_json_axis():
    # The JSON payload names the swept column axis so a consumer knows the grid
    # crossed gate × padding.
    def seg(wav, params=None):
        return _cell_result(1)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(thresholds=[0.3, 0.5], speech_pads=[0.0, 40.0], json=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    payload = json.loads(lines[0])
    assert payload["row_axis"] == "threshold"
    assert payload["col_axis"] == "speech_pad_ms"
    assert [c["speech_pad_ms"] for c in payload["grid"]] == [0.0, 40.0, 0.0, 40.0]


def test_cmd_vad_grid_speech_pad_column_human_label():
    # The human table labels the pad column "speech_pad" and formats bare
    # integers (no gate-style 0.00 leak).
    def seg(wav, params=None):
        return _cell_result(1)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(thresholds=[0.3, 0.5], speech_pads=[0.0, 40.0]),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "speech_pad" in text
    # Bare-integer formatting for the ms axis, not "0.00"/"40.00".
    assert "40" in text
    assert "40.00" not in text


def test_cmd_vad_grid_json_branch():
    def seg(wav, params=None):
        # Fewer segments at a higher gate.
        n = 3 if params.threshold < 0.5 else 1
        return _cell_result(n)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(thresholds=[0.3, 0.5], min_silences=[400.0, 800.0], json=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["row_axis"] == "threshold"
    assert payload["col_axis"] == "min_silence_ms"
    assert len(payload["grid"]) == 4
    assert [c["num_segments"] for c in payload["grid"]] == [3, 3, 1, 1]


def test_cmd_vad_grid_csv_branch():
    def seg(wav, params=None):
        return _cell_result(1)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(thresholds=[0.3, 0.5], min_silences=[400.0, 800.0], csv=True),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert len(lines) == 1
    rows = list(csv.reader(io.StringIO(lines[0])))
    assert rows[0] == ["threshold", "min_silence_ms", "num_segments", "speech_s"]
    assert len(rows) == 5  # header + 4 cells


def test_cmd_vad_grid_uses_segmenter_name():
    def seg(wav, params=None):
        return _Result(
            name="actual-basename.wav", sample_rate=16000, duration_s=5.0,
            segments=[_Seg(0.0, 1.0)],
        )

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(thresholds=[0.5], min_silences=[400.0]),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert "actual-basename.wav" in "\n".join(lines)


def test_cmd_vad_grid_target_emits_best_line():
    # Fewer segments at a higher gate; target 1 should pick a high-gate cell.
    def seg(wav, params=None):
        n = 3 if params.threshold < 0.5 else 1
        return _cell_result(n)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(thresholds=[0.3, 0.5], min_silences=[400.0, 800.0], target=1),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "best:" in text
    assert "1 segments" in text


def test_cmd_vad_grid_target_in_json_branch():
    def seg(wav, params=None):
        n = 3 if params.threshold < 0.5 else 1
        return _cell_result(n)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], min_silences=[400.0, 800.0],
            target=1, json=True,
        ),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    payload = json.loads(lines[0])
    assert payload["target"] == 1
    assert payload["best"]["num_segments"] == 1
    assert payload["best"]["distance"] == 0


def test_cmd_vad_grid_band_target_picks_in_band_cell():
    # iter-246: low gate → 4 segments, high gate → 1. Band 3-5 lands the 4-count
    # cell inside the band (|Δ|=0), so it wins.
    def seg(wav, params=None):
        n = 4 if params.threshold < 0.5 else 1
        return _cell_result(n)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], min_silences=[400.0, 800.0], target=(3, 5),
        ),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "best:" in text
    assert "4 segments" in text
    assert "target 3-5" in text


def test_cmd_vad_sweep_band_target_picks_in_band_value():
    # iter-246, sweep form: low gate → 4 segments, high gate → 1. Band 3-5 makes
    # the 4-count threshold the best (|Δ|=0 inside the band).
    def seg(wav, params=None):
        n = 4 if params.threshold < 0.5 else 1
        return _cell_result(n)

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.3, 0.7], target=(3, 5)),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "best:" in text
    assert "4 segments" in text
    assert "target 3-5" in text


def test_cmd_vad_grid_open_band_target_picks_satisfying_cell():
    # iter-247: low gate → 4 segments, high gate → 1. Band 3- ("at least 3")
    # satisfies the 4-count cell (|Δ|=0), so it wins over the short 1-count cell.
    def seg(wav, params=None):
        n = 4 if params.threshold < 0.5 else 1
        return _cell_result(n)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], min_silences=[400.0, 800.0], target=(3, None),
        ),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "best:" in text
    assert "4 segments" in text
    assert "target 3-" in text
    assert "None" not in text


def test_cmd_vad_sweep_open_band_target_picks_satisfying_value():
    # iter-247, sweep form: low gate → 4 segments, high gate → 8. Band -5
    # ("at most 5") satisfies the 4-count threshold (|Δ|=0), so it wins.
    def seg(wav, params=None):
        n = 4 if params.threshold < 0.5 else 8
        return _cell_result(n)

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.3, 0.7], target=(None, 5)),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "best:" in text
    assert "4 segments" in text
    assert "target -5" in text
    assert "None" not in text


def test_cmd_vad_grid_set_target_picks_satisfying_cell():
    # iter-248: low gate → 4 segments, high gate → 5. Set 3,5,7 satisfies the
    # 5-count cell (|Δ|=0), so it wins over the off-by-one 4-count cell.
    def seg(wav, params=None):
        n = 4 if params.threshold < 0.5 else 5
        return _cell_result(n)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], min_silences=[400.0, 800.0], target=[3, 5, 7],
        ),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "best:" in text
    assert "5 segments" in text
    assert "target 3,5,7" in text
    assert "[3, 5, 7]" not in text


def test_cmd_vad_sweep_set_target_picks_satisfying_value():
    # iter-248, sweep form: low gate → 4 segments, high gate → 7. Set 3,5,7
    # satisfies the 7-count threshold (|Δ|=0), so it wins over the 4-count one.
    def seg(wav, params=None):
        n = 4 if params.threshold < 0.5 else 7
        return _cell_result(n)

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.3, 0.7], target=[3, 5, 7]),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "best:" in text
    assert "7 segments" in text
    assert "target 3,5,7" in text
    assert "[3, 5, 7]" not in text


def test_cmd_vad_grid_preference_target_breaks_tie_toward_preferred():
    # iter-249, grid form: low gate → 3 segments, high gate → 5. Both satisfy
    # preference 5>3 (distance 0), so the preference picks the 5-count cell even
    # though the 3-count cell is earlier in row-major order.
    def seg(wav, params=None):
        n = 3 if params.threshold < 0.5 else 5
        return _cell_result(n)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], min_silences=[400.0],
            target={"prefer": [5, 3]},
        ),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "best:" in text
    assert "5 segments" in text
    assert "target 5>3" in text
    assert "prefer" not in text


def test_cmd_vad_sweep_preference_target_breaks_tie_toward_preferred():
    # iter-249, sweep form: low gate → 7 segments, high gate → 3. Both satisfy
    # preference 3>7 (distance 0), so the preference picks the 3-count value even
    # though the 7-count value is swept earlier.
    def seg(wav, params=None):
        n = 7 if params.threshold < 0.5 else 3
        return _cell_result(n)

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.3, 0.7], target={"prefer": [3, 7]}),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "best:" in text
    assert "3 segments" in text
    assert "target 3>7" in text
    assert "prefer" not in text


def test_cmd_vad_grid_weighted_target_overrides_distance_gap():
    # iter-250, grid form: preferred 3 (penalty 0), accepted 8 (penalty 3). Low
    # gate → 8 segments (lands on the accepted element, penalised 3); high gate → 4
    # (raw dist 1 from preferred 3, penalised 1). The +3 penalty flips the pick to
    # the 4-count cell even though 8 matches an accepted count exactly.
    def seg(wav, params=None):
        n = 8 if params.threshold < 0.5 else 4
        return _cell_result(n)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], min_silences=[400.0],
            target={"weighted": [(3, 0), (8, 3)]},
        ),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "best:" in text
    assert "4 segments" in text
    assert "target 3,8:3" in text
    assert "weighted" not in text


def test_cmd_vad_sweep_weighted_target_overrides_distance_gap():
    # iter-250, sweep form: preferred 3 (penalty 0), accepted 8 (penalty 3). Low
    # gate → 8 segments (penalised 3); high gate → 5 (raw dist 2 from preferred 3,
    # penalised 2). The +3 penalty flips the pick to the 5-count value (2 < 3).
    def seg(wav, params=None):
        n = 8 if params.threshold < 0.5 else 5
        return _cell_result(n)

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.3, 0.7], target={"weighted": [(3, 0), (8, 3)]}),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "best:" in text
    assert "5 segments" in text
    assert "target 3,8:3" in text
    assert "weighted" not in text


def test_cmd_vad_grid_scaled_target_grows_cost_with_distance():
    # iter-252, grid form: preferred 3 (factor 1), accepted 8 (factor 2). Low gate →
    # 10 segments (raw dist 2 past accepted 8, scaled 2*2=4); high gate → 5 (raw
    # dist 2 from preferred 3, scaled 2*1=2). The factor grows the off-8 cost so the
    # pick lands on the 5-count cell (2 < 4).
    def seg(wav, params=None):
        n = 10 if params.threshold < 0.5 else 5
        return _cell_result(n)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], min_silences=[400.0],
            target={"scaled": [(3, 1), (8, 2)]},
        ),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "best:" in text
    assert "5 segments" in text
    assert "target 3,8*2" in text
    assert "scaled" not in text


def test_cmd_vad_sweep_scaled_target_grows_cost_with_distance():
    # iter-252, sweep form: preferred 3 (factor 1), accepted 8 (factor 2). Low gate →
    # 10 segments (scaled 2*2=4); high gate → 5 (scaled 2*1=2). The factor flips the
    # pick to the 5-count value (2 < 4).
    def seg(wav, params=None):
        n = 10 if params.threshold < 0.5 else 5
        return _cell_result(n)

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.3, 0.7], target={"scaled": [(3, 1), (8, 2)]}),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "best:" in text
    assert "5 segments" in text
    assert "target 3,8*2" in text
    assert "scaled" not in text


def test_cmd_vad_grid_csv_ignores_target():
    # CSV is a pure data grid — --target adds no best row/column.
    def seg(wav, params=None):
        return _cell_result(2)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], min_silences=[400.0, 800.0],
            target=1, csv=True,
        ),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    rows = list(csv.reader(io.StringIO(lines[0])))
    assert rows[0] == ["threshold", "min_silence_ms", "num_segments", "speech_s"]
    assert len(rows) == 5  # header + 4 cells, no best row
    assert "best" not in lines[0]


def test_cmd_vad_grid_no_target_omits_best_line():
    def seg(wav, params=None):
        return _cell_result(2)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(thresholds=[0.3], min_silences=[400.0]),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    assert "best:" not in "\n".join(lines)


def test_cmd_vad_grid_top_emits_shortlist_block():
    # Fewer segments at a higher gate; target 1, top 2 lists the two closest.
    def seg(wav, params=None):
        n = 3 if params.threshold < 0.5 else 1
        return _cell_result(n)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], min_silences=[400.0, 800.0],
            target=1, top=2,
        ),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "best:" in text
    assert "top 2 (closest to target 1):" in text


def test_cmd_vad_grid_top_in_json_branch():
    def seg(wav, params=None):
        n = 3 if params.threshold < 0.5 else 1
        return _cell_result(n)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], min_silences=[400.0, 800.0],
            target=1, top=2, json=True,
        ),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    payload = json.loads(lines[0])
    assert len(payload["top"]) == 2
    # Closest first; the head equals best.
    assert payload["top"][0] == payload["best"]
    assert payload["top"][0]["distance"] <= payload["top"][1]["distance"]


def test_cmd_vad_grid_csv_ignores_top():
    # CSV is a pure data grid — --top adds no shortlist rows.
    def seg(wav, params=None):
        return _cell_result(2)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], min_silences=[400.0, 800.0],
            target=1, top=2, csv=True,
        ),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    rows = list(csv.reader(io.StringIO(lines[0])))
    assert len(rows) == 5  # header + 4 cells, no shortlist rows
    assert "top" not in lines[0]


def test_cmd_vad_grid_top_without_target_omits_shortlist():
    # --top rides along with --target; alone it adds nothing.
    def seg(wav, params=None):
        return _cell_result(2)

    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(thresholds=[0.3, 0.5], min_silences=[400.0, 800.0], top=2),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    text = "\n".join(lines)
    assert "top " not in text
    assert "best:" not in text


def test_cmd_vad_grid_top_unavailable_branch_no_crash():
    # silero absent → install hint, no shortlist, no crash even with --top.
    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(target=1, top=3),
        log=lines.append,
        segmenter=lambda wav, params=None: _cell_result(1),
        availability=lambda: False,
    )
    assert any("silero VAD unavailable" in ln for ln in lines)


def _tied_count_varied_speech_seg(wav, params=None):
    """Every cell has 4 segments, but a longer hangover recovers more speech.

    Lets a cmd-level test exercise the speech tie-break: all cells are equally
    distant from a target of 3, so only the tie-break decides the pick.
    """
    speech_s = 5.0 if params.min_silence_ms >= 800.0 else 2.0
    per = speech_s / 4
    segs = [_Seg(2.0 * i, 2.0 * i + per) for i in range(4)]
    return _Result(name="rec.wav", sample_rate=16000, duration_s=100.0, segments=segs)


def test_cmd_vad_grid_speech_tie_break_picks_most_speech():
    # All cells |Δ|=1 from target 3; speech tie-break names the high-speech
    # (min_silence=800) cell, not the earlier row-major one.
    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], min_silences=[400.0, 800.0],
            target=3, tie_break="speech",
        ),
        log=lines.append,
        segmenter=_tied_count_varied_speech_seg,
        availability=lambda: True,
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "min_silence=800" in best


def test_cmd_vad_grid_default_tie_break_keeps_earlier_cell():
    # Same grid, default row-major tie-break keeps the earlier (400) cell.
    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], min_silences=[400.0, 800.0], target=3,
        ),
        log=lines.append,
        segmenter=_tied_count_varied_speech_seg,
        availability=lambda: True,
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "min_silence=400" in best


def test_cmd_vad_grid_tie_break_in_json_payload():
    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], min_silences=[400.0, 800.0],
            target=3, tie_break="speech", json=True,
        ),
        log=lines.append,
        segmenter=_tied_count_varied_speech_seg,
        availability=lambda: True,
    )
    payload = json.loads(lines[0])
    assert payload["tie_break"] == "speech"
    # The pick is the high-speech cell.
    assert payload["best"]["min_silence_ms"] == 800.0


def test_cmd_vad_grid_csv_ignores_tie_break():
    # CSV is a pure data grid — --tie-break adds no column / changes no rows.
    lines: List[str] = []
    gv.cmd_vad_grid(
        _grid_args(
            thresholds=[0.3, 0.5], min_silences=[400.0, 800.0],
            target=3, tie_break="speech", csv=True,
        ),
        log=lines.append,
        segmenter=_tied_count_varied_speech_seg,
        availability=lambda: True,
    )
    rows = list(csv.reader(io.StringIO(lines[0])))
    assert rows[0] == ["threshold", "min_silence_ms", "num_segments", "speech_s"]
    assert len(rows) == 5  # header + 4 cells
    assert "tie_break" not in lines[0]


# ====================================================================
# iter-244 — gv vad-sweep --target / --top / --tie-break:
# bring the iter-241→243 grid pick machinery to the 1-D sweep.
# A sweep row {axis, num_segments, speech_s} carries the same keys a
# grid cell does, so the shared pickers (pick_best_grid_cell /
# pick_top_grid_cells / grid_cell_distance) apply unchanged.
# ====================================================================


# ---- parser: --target / --top / --tie-break flags ----------------------


def test_vad_sweep_target_default_none():
    args = gv.build_parser().parse_args(["vad-sweep", "rec.wav"])
    assert args.target is None
    assert args.top is None
    assert args.tie_break == "row-major"


def test_vad_sweep_target_parses_int():
    args = gv.build_parser().parse_args(["vad-sweep", "rec.wav", "--target", "3"])
    assert args.target == 3


def test_vad_sweep_target_accepts_zero():
    # nonneg_int_type accepts 0 (a degenerate but legal target).
    args = gv.build_parser().parse_args(["vad-sweep", "rec.wav", "--target", "0"])
    assert args.target == 0


@pytest.mark.parametrize("raw", ["2.5", "high", "5-3"])
def test_vad_sweep_target_rejects_bad(raw):
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-sweep", "rec.wav", "--target", raw])


def test_vad_sweep_target_dash_n_is_open_band_at_most():
    # iter-247: '-1' parses as the open band "at most 1" → (None, 1), not a
    # rejected negative scalar.
    args = gv.build_parser().parse_args(["vad-sweep", "rec.wav", "--target", "-1"])
    assert args.target == (None, 1)


def test_vad_sweep_top_parses_int():
    args = gv.build_parser().parse_args(
        ["vad-sweep", "rec.wav", "--target", "3", "--top", "2"]
    )
    assert args.top == 2


@pytest.mark.parametrize("raw", ["0", "-1", "1.5", "x"])
def test_vad_sweep_top_rejects_bad(raw):
    # pos_int_type rejects 0 (a 0-length shortlist is meaningless), negatives,
    # fractionals and non-ints.
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(["vad-sweep", "rec.wav", "--top", raw])


def test_vad_sweep_tie_break_parses_speech():
    args = gv.build_parser().parse_args(
        ["vad-sweep", "rec.wav", "--target", "3", "--tie-break", "speech"]
    )
    assert args.tie_break == "speech"


def test_vad_sweep_tie_break_rejects_unknown():
    with pytest.raises(SystemExit):
        gv.build_parser().parse_args(
            ["vad-sweep", "rec.wav", "--tie-break", "loudest"]
        )


# ---- render_vad_sweep: best / top block --------------------------------


def _sweep_results(counts):
    """One _Result per swept value, each with `n` zero-padded-length segments.

    Speech seconds scale with index so tie-breaks have something to bite on.
    """
    out = []
    for i, n in enumerate(counts):
        per = 0.5 + 0.1 * i  # later values recover slightly more speech per seg
        segs = [_Seg(2.0 * j, 2.0 * j + per) for j in range(n)]
        out.append(
            _Result(name="rec.wav", sample_rate=16000, duration_s=100.0, segments=segs)
        )
    return out


def test_render_sweep_no_target_omits_best_line():
    results = _sweep_results([3, 1])
    lines = gv.render_vad_sweep([0.3, 0.9], results, name="rec.wav")
    assert not any("best:" in ln for ln in lines)


def test_render_sweep_best_line_names_closest_value():
    # counts 5,3,1 at thresholds 0.3,0.5,0.7; target 3 → the 0.50 value (3 segs).
    results = _sweep_results([5, 3, 1])
    lines = gv.render_vad_sweep(
        [0.3, 0.5, 0.7], results, name="rec.wav", target=3
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "threshold=0.50" in best
    assert "3 segments" in best
    assert "|Δ|=0" in best
    assert "target 3" in best


def test_render_sweep_best_line_min_silence_axis_label():
    results = _sweep_results([5, 3])
    lines = gv.render_vad_sweep(
        [400.0, 800.0], results, name="rec.wav", axis="min_silence_ms", target=3
    )
    best = next(ln for ln in lines if "best:" in ln)
    # The label tracks the swept axis, and ms knobs format as bare ints.
    assert "min_silence=800" in best


def test_render_sweep_empty_with_target_reports_none():
    lines = gv.render_vad_sweep([], [], name="rec.wav", target=3)
    best = next(ln for ln in lines if "best:" in ln)
    assert "none" in best
    assert "empty sweep" in best


def test_render_sweep_top_block_lists_closest_values():
    # counts 5,4,3,1 at 0.3,0.5,0.7,0.9; target 3, top 2 → 0.70 (Δ0), then 0.50 (Δ1).
    results = _sweep_results([5, 4, 3, 1])
    lines = gv.render_vad_sweep(
        [0.3, 0.5, 0.7, 0.9], results, name="rec.wav", target=3, top=2
    )
    top_idx = next(i for i, ln in enumerate(lines) if "top 2" in ln)
    block = lines[top_idx + 1 : top_idx + 3]
    assert "1." in block[0] and "threshold=0.70" in block[0] and "|Δ|=0" in block[0]
    assert "2." in block[1] and "threshold=0.50" in block[1] and "|Δ|=1" in block[1]


def test_render_sweep_top_ignored_without_target():
    results = _sweep_results([5, 3, 1])
    lines = gv.render_vad_sweep([0.3, 0.5, 0.7], results, name="rec.wav", top=2)
    assert not any("top 2" in ln for ln in lines)
    assert not any("best:" in ln for ln in lines)


def test_render_sweep_top_head_equals_best():
    results = _sweep_results([5, 4, 3, 1])
    lines = gv.render_vad_sweep(
        [0.3, 0.5, 0.7, 0.9], results, name="rec.wav", target=3, top=3
    )
    best = next(ln for ln in lines if "best:" in ln)
    first_ranked = next(ln for ln in lines if ln.strip().startswith("1."))
    # Both name the same swept value.
    assert "threshold=0.70" in best
    assert "threshold=0.70" in first_ranked


# ---- render_vad_sweep_json: target / top / tie_break -------------------


def test_render_sweep_json_no_target_omits_pick_keys():
    results = _sweep_results([3, 1])
    payload = json.loads(
        gv.render_vad_sweep_json([0.3, 0.9], results, name="rec.wav")
    )
    assert "target" not in payload
    assert "best" not in payload
    assert "top" not in payload
    assert "tie_break" not in payload


def test_render_sweep_json_target_adds_best_and_distance():
    results = _sweep_results([5, 3, 1])
    payload = json.loads(
        gv.render_vad_sweep_json([0.3, 0.5, 0.7], results, name="rec.wav", target=3)
    )
    assert payload["target"] == 3
    assert payload["tie_break"] == "row-major"
    assert payload["best"]["threshold"] == 0.5
    assert payload["best"]["num_segments"] == 3
    assert payload["best"]["distance"] == 0


def test_render_sweep_json_top_list_ranked_head_equals_best():
    results = _sweep_results([5, 4, 3, 1])
    payload = json.loads(
        gv.render_vad_sweep_json(
            [0.3, 0.5, 0.7, 0.9], results, name="rec.wav", target=3, top=3
        )
    )
    top = payload["top"]
    # Distances non-decreasing, head == best.
    dists = [c["distance"] for c in top]
    assert dists == sorted(dists)
    assert top[0]["threshold"] == payload["best"]["threshold"]


def test_render_sweep_json_top_ignored_without_target():
    results = _sweep_results([5, 3, 1])
    payload = json.loads(
        gv.render_vad_sweep_json([0.3, 0.5, 0.7], results, name="rec.wav", top=2)
    )
    assert "top" not in payload


def test_render_sweep_json_empty_with_target_best_none():
    payload = json.loads(
        gv.render_vad_sweep_json([], [], name="rec.wav", target=3)
    )
    assert payload["target"] == 3
    assert payload["best"] is None


# ---- tie-break: speech vs row-major over a sweep -----------------------


def _sweep_tied_count_varied_speech():
    """Two swept values, both 4 segments, the second recovering more speech.

    Both tie at |Δ|=1 from target 3, so only the tie-break decides the pick.
    """
    a = _Result(
        name="rec.wav", sample_rate=16000, duration_s=100.0,
        segments=[_Seg(2.0 * j, 2.0 * j + 0.5) for j in range(4)],  # 2.0s
    )
    b = _Result(
        name="rec.wav", sample_rate=16000, duration_s=100.0,
        segments=[_Seg(2.0 * j, 2.0 * j + 1.25) for j in range(4)],  # 5.0s
    )
    return a, b


def test_render_sweep_speech_tie_break_prefers_most_speech():
    a, b = _sweep_tied_count_varied_speech()
    lines = gv.render_vad_sweep(
        [0.3, 0.7], [a, b], name="rec.wav", target=3, tie_break="speech"
    )
    best = next(ln for ln in lines if "best:" in ln)
    # Higher-speech (later, 0.70) value wins the tie, not the earlier one.
    assert "threshold=0.70" in best


def test_render_sweep_default_tie_break_keeps_earlier_value():
    a, b = _sweep_tied_count_varied_speech()
    lines = gv.render_vad_sweep([0.3, 0.7], [a, b], name="rec.wav", target=3)
    best = next(ln for ln in lines if "best:" in ln)
    assert "threshold=0.30" in best


def test_render_sweep_json_speech_tie_break_in_payload():
    a, b = _sweep_tied_count_varied_speech()
    payload = json.loads(
        gv.render_vad_sweep_json(
            [0.3, 0.7], [a, b], name="rec.wav", target=3, tie_break="speech"
        )
    )
    assert payload["tie_break"] == "speech"
    assert payload["best"]["threshold"] == 0.7


# ---- tie-break: speech tie-break on the SECONDS max_speech_s axis -------
#
# iter-243's speech tie-break and iter-255/257's max_speech_s seconds axis are
# orthogonal seams that had never been exercised TOGETHER on a render surface:
# every speech-tie-break render test (above, and the grid twin) sweeps the
# threshold/min_silence axis, so the %g seconds formatting and the -speech_s
# secondary key had no joint coverage. A regression that broke EITHER (a
# tie-break that ignored speech on the seconds axis, or a %g→.2f drift on the
# tie-break-chosen cap) would have shipped green. These pin both at once: the
# tie-break picks the most-speech cap AND that cap renders compactly.


def test_render_sweep_speech_tie_break_on_max_speech_seconds_axis():
    # Two caps (10s, never-split inf), both 4 segments (|Δ|=1 from target 3);
    # the inf baseline recovers MORE speech (5.0s vs 2.0s). The speech tie-break
    # must name the inf cap, and it must render as the compact "max_speech=inf"
    # — not a gate-style "inf.00", and the finite cap must not leak "10.00".
    a, b = _sweep_tied_count_varied_speech()
    lines = gv.render_vad_sweep(
        [10.0, float("inf")], [a, b], name="rec.wav",
        axis="max_speech_s", target=3, tie_break="speech",
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=inf" in best
    text = "\n".join(lines)
    assert "max_speech=inf.00" not in text
    assert "max_speech=10.00" not in text


def test_render_sweep_default_tie_break_keeps_earlier_max_speech_cap():
    # Same caps and tie, default row-major tie-break keeps the EARLIER finite
    # 10s cap (rendered compactly), proving the seconds axis honours the
    # earliest-tie rule just like the threshold axis.
    a, b = _sweep_tied_count_varied_speech()
    lines = gv.render_vad_sweep(
        [10.0, float("inf")], [a, b], name="rec.wav",
        axis="max_speech_s", target=3,
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=10 " in best
    assert "max_speech=10.00" not in "\n".join(lines)


def test_render_grid_speech_tie_break_on_max_speech_col_axis_seconds():
    # The 2-D grid twin on the COLUMN seconds axis: a 1×2 grid over caps
    # (5s, inf), both 4 segments (|Δ|=1 from target 3), the inf baseline
    # recovering more speech. Speech tie-break names the inf cap, rendered as
    # the compact "max_speech=inf"; the held threshold row stays "threshold=0.30"
    # and no gate-style "inf.00" / "5.00" leak appears.
    results = [_result_n_speech(4, 2.0), _result_n_speech(4, 5.0)]
    lines = gv.render_vad_grid(
        [0.3], [5.0, float("inf")], results, name="rec.wav",
        col_axis="max_speech_s", target=3, tie_break="speech",
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "max_speech=inf" in best
    assert "threshold=0.30" in best
    text = "\n".join(lines)
    assert "max_speech=inf.00" not in text
    assert "max_speech=5.00" not in text


def test_render_grid_speech_tie_break_on_max_speech_col_axis_orders_top_list():
    # Same grid, --top 2: the speech tie-break puts the higher-speech inf cap
    # FIRST in the shortlist, and both shortlist rows render the seconds caps
    # compactly (max_speech=inf / max_speech=5), never a .2f gate-style leak.
    results = [_result_n_speech(4, 2.0), _result_n_speech(4, 5.0)]
    lines = gv.render_vad_grid(
        [0.3], [5.0, float("inf")], results, name="rec.wav",
        col_axis="max_speech_s", target=3, top=2, tie_break="speech",
    )
    shortlist = [ln for ln in lines if ln.lstrip().startswith(("1.", "2."))]
    assert len(shortlist) == 2
    assert "max_speech=inf" in shortlist[0]  # most-speech cap heads the list
    assert "max_speech=5" in shortlist[1]
    text = "\n".join(shortlist)
    assert "max_speech=inf.00" not in text
    assert "max_speech=5.00" not in text


# ---- cmd_vad_sweep: end-to-end threading -------------------------------


def _count_by_threshold_seg(wav, params=None):
    # 5,3,1 segments at gates 0.3,0.5,0.7 (higher gate → fewer segments).
    n = 5 if params.threshold < 0.4 else (3 if params.threshold < 0.6 else 1)
    return _Result(
        name="rec.wav", sample_rate=16000, duration_s=100.0,
        segments=[_Seg(2.0 * j, 2.0 * j + 0.5) for j in range(n)],
    )


def test_cmd_vad_sweep_target_emits_best_line():
    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.3, 0.5, 0.7], target=3),
        log=lines.append,
        segmenter=_count_by_threshold_seg,
        availability=lambda: True,
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "threshold=0.50" in best and "3 segments" in best


def test_cmd_vad_sweep_no_target_no_best_line():
    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.3, 0.5, 0.7]),
        log=lines.append,
        segmenter=_count_by_threshold_seg,
        availability=lambda: True,
    )
    assert not any("best:" in ln for ln in lines)


def test_cmd_vad_sweep_json_carries_target_and_best():
    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.3, 0.5, 0.7], target=3, top=2, json=True),
        log=lines.append,
        segmenter=_count_by_threshold_seg,
        availability=lambda: True,
    )
    payload = json.loads(lines[0])
    assert payload["target"] == 3
    assert payload["best"]["threshold"] == 0.5
    assert payload["top"][0]["threshold"] == 0.5
    assert [c["distance"] for c in payload["top"]] == sorted(
        c["distance"] for c in payload["top"]
    )


def test_cmd_vad_sweep_csv_ignores_target_top_tiebreak():
    # CSV stays a pure data grid — none of the pick flags add a column/line.
    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(
            thresholds=[0.3, 0.5, 0.7], target=3, top=2,
            tie_break="speech", csv=True,
        ),
        log=lines.append,
        segmenter=_count_by_threshold_seg,
        availability=lambda: True,
    )
    rows = list(csv.reader(io.StringIO(lines[0])))
    assert rows[0] == ["threshold", "num_segments", "speech_s"]
    assert len(rows) == 4  # header + 3 swept values
    assert "best" not in lines[0] and "target" not in lines[0]


def test_cmd_vad_sweep_speech_tie_break_picks_most_speech():
    # Two gates, both 4 segments; the higher gate recovers more speech here.
    def seg(wav, params=None):
        speech = 5.0 if params.threshold >= 0.6 else 2.0
        per = speech / 4
        return _Result(
            name="rec.wav", sample_rate=16000, duration_s=100.0,
            segments=[_Seg(2.0 * j, 2.0 * j + per) for j in range(4)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.3, 0.7], target=3, tie_break="speech"),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "threshold=0.70" in best


def test_cmd_vad_sweep_unavailable_target_no_crash():
    # The unavailable branch threads target/top/tie_break and still just hints.
    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(target=3, top=2, json=True),
        log=lines.append,
        segmenter=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")),
        availability=lambda: False,
    )
    assert len(lines) == 1
    assert json.loads(lines[0])["available"] is False


def test_cmd_vad_sweep_target_on_min_silence_axis():
    # --target works on the hangover axis too (label tracks the swept axis).
    def seg(wav, params=None):
        n = 5 if params.min_silence_ms < 600.0 else 3
        return _Result(
            name="rec.wav", sample_rate=16000, duration_s=100.0,
            segments=[_Seg(2.0 * j, 2.0 * j + 0.5) for j in range(n)],
        )

    lines: List[str] = []
    gv.cmd_vad_sweep(
        _sweep_args(thresholds=[0.3, 0.5, 0.7, 0.9], min_silences=[400.0, 800.0], target=3),
        log=lines.append,
        segmenter=seg,
        availability=lambda: True,
    )
    best = next(ln for ln in lines if "best:" in ln)
    assert "min_silence=800" in best and "3 segments" in best


# ---- _render_pick_block: the shared sweep/grid pick block (iter-245) ----
# iter-245 factored the duplicated best:/top N: block out of render_vad_sweep
# and render_vad_grid into one helper. These exercise the helper in isolation;
# the existing render_vad_sweep/render_vad_grid tests above pin that the two
# callers still emit byte-identical output through it.

def _pick_cells():
    # Three rows keyed like a sweep row (the helper only reads num_segments /
    # speech_s, so a bare dict with an axis key is enough).
    return [
        {"threshold": 0.3, "num_segments": 5, "speech_s": 12.0},
        {"threshold": 0.5, "num_segments": 3, "speech_s": 9.0},
        {"threshold": 0.7, "num_segments": 1, "speech_s": 4.0},
    ]


def _axes(cell):
    return f"threshold={cell['threshold']:.2f}"


def test_render_pick_block_no_target_returns_empty():
    # target=None → no pick block at all (callers stay byte-for-byte identical).
    out = gv._render_pick_block(
        _pick_cells(), None, None, "row-major",
        format_axes=_axes, empty_noun="sweep",
    )
    assert out == []


def test_render_pick_block_empty_cells_reports_none_with_noun():
    # Empty input → a single best: none line carrying the caller's noun.
    out = gv._render_pick_block(
        [], 3, None, "row-major", format_axes=_axes, empty_noun="grid",
    )
    assert out == ["  best: none (empty grid; target 3 segments)"]


def test_render_pick_block_best_line_uses_format_axes_and_distance():
    # best: line names the closest cell via format_axes and reports |Δ|.
    out = gv._render_pick_block(
        _pick_cells(), 3, None, "row-major",
        format_axes=_axes, empty_noun="sweep",
    )
    assert out == [
        "  best: threshold=0.50 (3 segments, |Δ|=0 from target 3)"
    ]


def test_render_pick_block_top_block_ranked_nearest_first():
    # top=2 → a ranked shortlist after best:, head == best, |Δ| ascending.
    out = gv._render_pick_block(
        _pick_cells(), 3, 2, "row-major",
        format_axes=_axes, empty_noun="sweep",
    )
    assert out[0] == "  best: threshold=0.50 (3 segments, |Δ|=0 from target 3)"
    assert out[1] == "  top 2 (closest to target 3):"
    assert out[2] == "    1. threshold=0.50  3 segments  |Δ|=0"
    # Two cells tie at |Δ|=2 (0.3→5, 0.7→1); row-major keeps 0.3 first.
    assert out[3] == "    2. threshold=0.30  5 segments  |Δ|=2"


def test_render_pick_block_top_ignored_without_target_is_moot():
    # top is only consulted when target is set; with target=None the whole
    # block is empty regardless of top.
    out = gv._render_pick_block(
        _pick_cells(), None, 5, "row-major",
        format_axes=_axes, empty_noun="sweep",
    )
    assert out == []


def test_render_pick_block_speech_tie_break_breaks_on_speech():
    # Two cells tie at |Δ|=2; speech tie-break ranks the higher-speech one
    # first (0.3 has 12.0s vs 0.7's 4.0s) — distinct from row-major order.
    out = gv._render_pick_block(
        _pick_cells(), 3, 3, "speech",
        format_axes=_axes, empty_noun="sweep",
    )
    # rank 2 (first runner-up) should be the most-speech tied cell.
    assert out[3] == "    2. threshold=0.30  5 segments  |Δ|=2"
    assert out[4] == "    3. threshold=0.70  1 segments  |Δ|=2"


def test_render_pick_block_top_clamped_to_cell_count():
    # A shortlist longer than the input simply lists every cell.
    out = gv._render_pick_block(
        _pick_cells(), 3, 99, "row-major",
        format_axes=_axes, empty_noun="sweep",
    )
    assert out[1] == "  top 3 (closest to target 3):"
    assert len([ln for ln in out if ln.lstrip().startswith(tuple("123"))]) == 3


def test_render_pick_block_sweep_grid_share_one_implementation():
    # The drift-prevention guarantee: sweep and grid differ ONLY in the
    # format_axes callable + empty noun. Feed the SAME cells with a 1-axis and
    # a 2-axis formatter; everything around the axis section is identical.
    cells = [{"threshold": 0.5, "min_silence_ms": 800.0,
              "num_segments": 3, "speech_s": 9.0}]
    one = gv._render_pick_block(
        cells, 3, 1, "row-major",
        format_axes=lambda c: f"threshold={c['threshold']:.2f}",
        empty_noun="sweep",
    )
    two = gv._render_pick_block(
        cells, 3, 1, "row-major",
        format_axes=lambda c: (
            f"threshold={c['threshold']:.2f} "
            f"min_silence={c['min_silence_ms']:.0f}"
        ),
        empty_noun="grid",
    )
    # The fixed scaffolding (best:/top labels, segment counts, |Δ|, ranking)
    # is byte-identical; only the axis section the callable produced differs.
    assert one[0].replace("threshold=0.50", "AX") == \
        two[0].replace("threshold=0.50 min_silence=800", "AX")
    assert one[1] == two[1]
    assert one[2].replace("threshold=0.50", "AX") == \
        two[2].replace("threshold=0.50 min_silence=800", "AX")
