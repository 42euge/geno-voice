"""Tests for iter-409 — the ``gv stt-rtf-batch`` subcommand (examples/gv.py).

iter-405 shipped the single-engine STT real-time-factor profiling core and
iter-406/407/408 surfaced its profile + recommend/keep verdict on the CLI. All
of those stop at ONE engine. iter-409 generalises the core to a CORPUS
(``profile_stt_rtf_batch`` in ``stt/rtf_profile.py``) and exposes it on the
``gv`` CLI as ``stt-rtf-batch`` — the STT-side twin of
``gv calibrate-base-wpm-batch`` (iter-397) — so an operator choosing a
transcriber for the host sees which engines keep up with realtime in one table.

These tests exercise the ``--engine`` arg shape, the pure
``render_stt_rtf_batch`` helper, and the handler (driven with an injected
``log`` so no real I/O happens). The profiling core is pure stdlib loaded by
file path, so the handler runs on this x86_64 Linux runner without torch or
faster-whisper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples import gv  # noqa: E402


# ---- parser: wiring -----------------------------------------------------


def test_stt_rtf_batch_in_handler_map():
    assert gv.DEFAULT_HANDLERS["stt-rtf-batch"] is gv.cmd_stt_rtf_batch


def test_batch_parses_engine_groups():
    args = gv.build_parser().parse_args(
        [
            "stt-rtf-batch",
            "--engine",
            "mlx",
            "10.0:1.2",
            "5.0:0.8",
            "--engine",
            "fw",
            "10.0:6.0",
        ]
    )
    assert args.command == "stt-rtf-batch"
    assert args.engine == [["mlx", "10.0:1.2", "5.0:0.8"], ["fw", "10.0:6.0"]]


def test_batch_requires_engine():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(["stt-rtf-batch"])
    assert exc.value.code == 2


def test_batch_gate_defaults_match_core():
    args = gv.build_parser().parse_args(["stt-rtf-batch", "--engine", "a", "10.0:1.0"])
    srp = gv._load_stt_rtf_profile()
    assert args.rel_spread_max == srp.DEFAULT_STT_RTF_REL_SPREAD_MAX
    assert args.min_samples == srp.DEFAULT_STT_RTF_MIN_SAMPLES


def test_batch_gate_overrides_parse():
    args = gv.build_parser().parse_args(
        [
            "stt-rtf-batch",
            "--engine",
            "a",
            "10.0:1.0",
            "--rel-spread-max",
            "0.05",
            "--min-samples",
            "5",
        ]
    )
    assert args.rel_spread_max == 0.05
    assert args.min_samples == 5


# ---- render -------------------------------------------------------------


def _batch(*engines, **kw):
    srp = gv._load_stt_rtf_profile()
    built = [
        (
            label,
            [
                srp.TranscriptionSample(audio_seconds=a, transcribe_seconds=t)
                for (a, t) in pairs
            ],
        )
        for (label, pairs) in engines
    ]
    return srp.profile_stt_rtf_batch(built, **kw)


def test_render_header_counts_engines():
    batch = _batch(("a", [(10.0, 1.0)]), ("empty", []))
    lines = gv.render_stt_rtf_batch(batch)
    assert lines[0] == "STT real-time-factor batch (2 engines, 1 profiled)"
    assert lines[1] == "  gates: relative_spread<=0.15, samples>=3"


def test_render_lists_each_engine_with_grade_and_delta():
    batch = _batch(("fast", [(10.0, 1.0)]), ("rt", [(10.0, 8.0)]))
    lines = gv.render_stt_rtf_batch(batch)
    body = "\n".join(lines)
    assert "fast: 0.100 RTF (fast, Δmedian" in body
    assert "rt: 0.800 RTF (realtime, Δmedian" in body


def test_render_marks_lighten_for_recommended_engine():
    batch = _batch(
        ("heavy", [(10.0, 15.0), (10.0, 15.2), (10.0, 14.8)]),
        ("light", [(10.0, 1.0), (10.0, 1.1), (10.0, 0.9)]),
    )
    lines = gv.render_stt_rtf_batch(batch)
    heavy = [l for l in lines if l.strip().startswith("heavy:")][0]
    light = [l for l in lines if l.strip().startswith("light:")][0]
    assert heavy.endswith("← lighten")
    assert "← lighten" not in light


def test_render_corpus_summary_line():
    batch = _batch(("fast", [(10.0, 1.0)]), ("slow", [(10.0, 15.0)]))
    lines = gv.render_stt_rtf_batch(batch)
    corpus = [l for l in lines if l.strip().startswith("corpus:")][0]
    assert "median" in corpus
    assert "keep up with realtime" in corpus


def test_render_grades_histogram():
    batch = _batch(
        ("fast", [(10.0, 1.0)]),
        ("rt", [(10.0, 8.0)]),
        ("slow", [(10.0, 15.0)]),
        ("empty", []),
    )
    lines = gv.render_stt_rtf_batch(batch)
    grades = [l for l in lines if l.strip().startswith("grades:")][0]
    assert grades == "  grades: 1 fast, 1 realtime, 1 slow, 1 unprofiled"


def test_render_unprofiled_engine_marked():
    batch = _batch(("a", [(10.0, 1.0)]), ("empty", []))
    lines = gv.render_stt_rtf_batch(batch)
    assert "  empty: - (unprofiled — no samples)" in lines


def test_render_all_unprofiled_corpus_note():
    batch = _batch(("a", []), ("b", []))
    lines = gv.render_stt_rtf_batch(batch)
    assert "  corpus: (no engine profiled — nothing to summarise)" in lines


def test_render_grade_order_from_engine():
    # The histogram bucket order is read from the engine constant so the two
    # never drift.
    assert gv._stt_rtf_batch_grade_order() == ("fast", "realtime", "slow", "unprofiled")


# ---- handler ------------------------------------------------------------


def test_handler_emits_report():
    lines = []
    args = gv.build_parser().parse_args(
        ["stt-rtf-batch", "--engine", "a", "10.0:1.0", "--engine", "b", "10.0:8.0"]
    )
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    assert lines[0] == "STT real-time-factor batch (2 engines, 2 profiled)"
    assert any("a: 0.100 RTF" in l for l in lines)
    assert any("b: 0.800 RTF" in l for l in lines)


def test_handler_matches_render_directly():
    args = gv.build_parser().parse_args(
        ["stt-rtf-batch", "--engine", "a", "10.0:1.0", "--engine", "b", "10.0:8.0"]
    )
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    batch = _batch(("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    assert lines == gv.render_stt_rtf_batch(batch)


def test_handler_malformed_pair_is_clean_error():
    # A malformed pair is rejected by the SAME stt_rtf_sample_type validator the
    # single-engine --samples uses, surfacing as ArgumentTypeError rather than a
    # forwarded-garbage traceback (it raises in the handler since a --engine
    # group's tokens can't be an argparse type=).
    args = gv.build_parser().parse_args(
        ["stt-rtf-batch", "--engine", "a", "10.0:notanumber"]
    )
    with pytest.raises(argparse.ArgumentTypeError):
        gv.cmd_stt_rtf_batch(args, log=lambda *_: None)


def test_handler_engine_with_only_label_is_unprofiled():
    args = gv.build_parser().parse_args(["stt-rtf-batch", "--engine", "empty"])
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    assert "  empty: - (unprofiled — no samples)" in lines


def test_handler_honors_min_samples_override():
    # A min_samples floor of 5 keeps an otherwise-recommended slow engine, so the
    # ← lighten marker is absent — the gate reached the per-engine verdict.
    args = gv.build_parser().parse_args(
        [
            "stt-rtf-batch",
            "--engine",
            "heavy",
            "10.0:15.0",
            "10.0:15.2",
            "10.0:14.8",
            "--min-samples",
            "5",
        ]
    )
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    assert not any(l.endswith("← lighten") for l in lines)


def test_handler_default_log_is_print(capsys):
    args = gv.build_parser().parse_args(["stt-rtf-batch", "--engine", "a", "10.0:1.0"])
    gv.cmd_stt_rtf_batch(args)
    out = capsys.readouterr().out
    assert "STT real-time-factor batch (1 engines, 1 profiled)" in out


# ---- --json / --csv parser wiring (iter-410) ----------------------------


def test_batch_json_flag_defaults_off():
    args = gv.build_parser().parse_args(["stt-rtf-batch", "--engine", "a", "10.0:1.0"])
    assert args.json is False
    assert args.csv is False


def test_batch_json_flag_sets_true():
    args = gv.build_parser().parse_args(
        ["stt-rtf-batch", "--engine", "a", "10.0:1.0", "--json"]
    )
    assert args.json is True


def test_batch_csv_flag_sets_true():
    args = gv.build_parser().parse_args(
        ["stt-rtf-batch", "--engine", "a", "10.0:1.0", "--csv"]
    )
    assert args.csv is True


def test_batch_json_and_csv_mutually_exclusive():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(
            ["stt-rtf-batch", "--engine", "a", "10.0:1.0", "--json", "--csv"]
        )
    assert exc.value.code == 2


# ---- render_stt_rtf_batch_json (iter-410) -------------------------------


def test_render_json_corpus_aggregates():
    import json

    batch = _batch(("fast", [(10.0, 1.0)]), ("slow", [(10.0, 15.0)]))
    payload = json.loads(gv.render_stt_rtf_batch_json(batch))
    assert payload["num_engines"] == 2
    assert payload["num_profiled"] == 2
    assert payload["corpus_median_rtf"] == 0.8  # median of 0.1 and 1.5
    assert payload["corpus_min_rtf"] == 0.1
    assert payload["corpus_max_rtf"] == 1.5
    assert payload["corpus_spread"] == 1.4
    assert payload["rel_spread_max"] == 0.15
    assert payload["min_samples"] == 3
    assert payload["grade_counts"] == {
        "fast": 1,
        "realtime": 0,
        "slow": 1,
        "unprofiled": 0,
    }
    assert [r["engine"] for r in payload["rows"]] == ["fast", "slow"]


def test_render_json_row_matches_single_engine_json():
    # A batch row's nested profile object equals the single-engine --json profile
    # on the same samples EXACTLY (the cross-surface agreement contract).
    import json

    srp = gv._load_stt_rtf_profile()
    samples = [srp.TranscriptionSample(audio_seconds=10.0, transcribe_seconds=1.2)]
    single = json.loads(gv.render_stt_rtf_json(samples, srp.profile_stt_rtf(samples)))
    batch = _batch(("only", [(10.0, 1.2)]))
    row = json.loads(gv.render_stt_rtf_batch_json(batch))["rows"][0]
    assert row["profile"] == single["profile"]


def test_render_json_unprofiled_engine_is_null():
    import json

    batch = _batch(("a", [(10.0, 1.0)]), ("empty", []))
    payload = json.loads(gv.render_stt_rtf_batch_json(batch))
    empty = [r for r in payload["rows"] if r["engine"] == "empty"][0]
    assert empty["profile"] is None
    assert empty["delta_from_median_rtf"] is None
    assert empty["recommend"] is None


def test_render_json_empty_corpus_aggregates_null():
    import json

    batch = _batch(("a", []), ("b", []))
    payload = json.loads(gv.render_stt_rtf_batch_json(batch))
    assert payload["corpus_median_rtf"] is None
    assert payload["corpus_min_rtf"] is None
    assert payload["corpus_max_rtf"] is None
    assert payload["corpus_spread"] is None
    assert payload["num_profiled"] == 0
    assert payload["num_engines"] == 2


def test_render_json_recommend_flag():
    import json

    batch = _batch(
        ("heavy", [(10.0, 15.0), (10.0, 15.2), (10.0, 14.8)]),
        ("light", [(10.0, 1.0), (10.0, 1.1), (10.0, 0.9)]),
    )
    rows = {r["engine"]: r for r in json.loads(gv.render_stt_rtf_batch_json(batch))["rows"]}
    assert rows["heavy"]["recommend"] is True
    assert rows["light"]["recommend"] is False


def test_render_json_speed_margin_null_for_slow():
    import json

    batch = _batch(("slow", [(10.0, 15.0)]))
    row = json.loads(gv.render_stt_rtf_batch_json(batch))["rows"][0]
    assert row["profile"]["speed_grade"] == "slow"
    assert row["profile"]["speed_margin"] is None


def test_render_json_echoes_gate_overrides():
    import json

    batch = _batch(("a", [(10.0, 1.0)]), rel_spread_max=0.05, min_samples=5)
    payload = json.loads(gv.render_stt_rtf_batch_json(batch))
    assert payload["rel_spread_max"] == 0.05
    assert payload["min_samples"] == 5


# ---- render_stt_rtf_batch_csv (iter-410) --------------------------------


def _csv_data_rows(text):
    return [l for l in text.splitlines() if l and not l.startswith("#")]


def test_render_csv_header_and_per_engine_rows():
    batch = _batch(("fast", [(10.0, 1.0)]), ("slow", [(10.0, 15.0)]))
    text = gv.render_stt_rtf_batch_csv(batch)
    rows = _csv_data_rows(text)
    assert rows[0] == (
        "engine,median_rtf,n_samples,min_rtf,max_rtf,spread,relative_spread,"
        "speed_grade,speed_margin,delta_from_median_rtf,recommend,flyer"
    )
    assert rows[1].startswith("fast,0.1,1,")
    assert rows[2].startswith("slow,1.5,1,")
    # one header + one row per engine
    assert len(rows) == 3


def test_render_csv_unprofiled_engine_blank_cells():
    batch = _batch(("a", [(10.0, 1.0)]), ("empty", []))
    rows = _csv_data_rows(gv.render_stt_rtf_batch_csv(batch))
    empty = [r for r in rows if r.startswith("empty,")][0]
    assert empty == "empty,,,,,,,,,,,"


def test_render_csv_corpus_aggregates_as_comments():
    batch = _batch(("fast", [(10.0, 1.0)]), ("slow", [(10.0, 15.0)]))
    text = gv.render_stt_rtf_batch_csv(batch)
    assert "# num_engines: 2" in text
    assert "# num_profiled: 2" in text
    assert "# corpus_median_rtf: 0.8" in text
    assert "# range: 0.1 - 1.5" in text
    assert "# corpus_spread: 1.4" in text
    assert "# num_keep_up: 1" in text
    assert "# num_recommend: 0" in text
    assert "# grades: 1 fast, 0 realtime, 1 slow, 0 unprofiled" in text


def test_render_csv_recommend_cell():
    batch = _batch(
        ("heavy", [(10.0, 15.0), (10.0, 15.2), (10.0, 14.8)]),
        ("light", [(10.0, 1.0), (10.0, 1.1), (10.0, 0.9)]),
    )
    rows = _csv_data_rows(gv.render_stt_rtf_batch_csv(batch))
    heavy = [r for r in rows if r.startswith("heavy,")][0]
    light = [r for r in rows if r.startswith("light,")][0]
    # ...,recommend,flyer — neither engine is a flyer in this 2-engine corpus.
    assert heavy.endswith(",true,false")
    assert light.endswith(",false,false")


def test_render_csv_empty_corpus_blank_aggregates():
    batch = _batch(("a", []), ("b", []))
    text = gv.render_stt_rtf_batch_csv(batch)
    assert "# corpus_median_rtf: " in text
    assert "# range:  - " in text
    assert "# num_profiled: 0" in text


# ---- handler --json / --csv dispatch (iter-410) -------------------------


def test_handler_json_emits_json_payload():
    import json

    args = gv.build_parser().parse_args(
        ["stt-rtf-batch", "--engine", "a", "10.0:1.0", "--engine", "b", "10.0:8.0", "--json"]
    )
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["num_engines"] == 2
    assert [r["engine"] for r in payload["rows"]] == ["a", "b"]


def test_handler_json_matches_render_directly():
    args = gv.build_parser().parse_args(
        ["stt-rtf-batch", "--engine", "a", "10.0:1.0", "--engine", "b", "10.0:8.0", "--json"]
    )
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    batch = _batch(("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    assert lines == [gv.render_stt_rtf_batch_json(batch)]


def test_handler_csv_matches_render_directly():
    args = gv.build_parser().parse_args(
        ["stt-rtf-batch", "--engine", "a", "10.0:1.0", "--engine", "b", "10.0:8.0", "--csv"]
    )
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    batch = _batch(("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    assert lines == [gv.render_stt_rtf_batch_csv(batch)]


def test_handler_csv_emits_header_row():
    args = gv.build_parser().parse_args(
        ["stt-rtf-batch", "--engine", "a", "10.0:1.0", "--csv"]
    )
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    assert len(lines) == 1
    assert lines[0].splitlines()[0].startswith("engine,median_rtf,n_samples,")


# ---- iter-411: --sort-by / --top-n -------------------------------------


def _engine_order(lines):
    """Pull the per-engine label order out of a human-render line list."""
    out = []
    for l in lines:
        s = l.strip()
        if s.startswith(("STT real-time", "gates:", "corpus:", "grades:", "flyers:")):
            continue
        if ":" in s:
            out.append(s.split(":", 1)[0])
    return out


# parser wiring -----------------------------------------------------------


def test_sort_by_defaults_none():
    args = gv.build_parser().parse_args(["stt-rtf-batch", "--engine", "a", "10.0:1.0"])
    assert args.sort_by is None


def test_top_n_defaults_none():
    args = gv.build_parser().parse_args(["stt-rtf-batch", "--engine", "a", "10.0:1.0"])
    assert args.top_n is None


@pytest.mark.parametrize("key", ["median_rtf", "grade", "delta"])
def test_sort_by_parses_each_key(key):
    args = gv.build_parser().parse_args(
        ["stt-rtf-batch", "--engine", "a", "10.0:1.0", "--sort-by", key]
    )
    assert args.sort_by == key


def test_sort_by_rejects_unknown_key():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(
            ["stt-rtf-batch", "--engine", "a", "10.0:1.0", "--sort-by", "nope"]
        )
    assert exc.value.code == 2


def test_top_n_parses_positive():
    args = gv.build_parser().parse_args(
        ["stt-rtf-batch", "--engine", "a", "10.0:1.0", "--top-n", "2"]
    )
    assert args.top_n == 2


def test_top_n_rejects_zero():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(
            ["stt-rtf-batch", "--engine", "a", "10.0:1.0", "--top-n", "0"]
        )
    assert exc.value.code == 2


# sort type validator -----------------------------------------------------


def test_sort_type_normalizes_case_and_whitespace():
    assert gv.stt_rtf_batch_sort_type("  Median_RTF ") == "median_rtf"


def test_sort_type_rejects_empty():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.stt_rtf_batch_sort_type("")


def test_sort_type_rejects_non_string():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.stt_rtf_batch_sort_type(3)


# sort primitive ----------------------------------------------------------


def test_sort_none_preserves_input_order():
    batch = _batch(("c", [(10.0, 15.0)]), ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    rows = gv._sort_stt_rtf_batch_rows(batch.rows, None)
    assert [r["engine"] for r in rows] == ["c", "a", "b"]
    # A copy, not the source tuple.
    assert rows is not batch.rows


def test_sort_by_median_rtf_ascending():
    batch = _batch(("c", [(10.0, 15.0)]), ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    rows = gv._sort_stt_rtf_batch_rows(batch.rows, "median_rtf")
    assert [r["engine"] for r in rows] == ["a", "b", "c"]


def test_sort_by_grade_descending():
    batch = _batch(("slow", [(10.0, 15.0)]), ("fast", [(10.0, 1.0)]), ("rt", [(10.0, 8.0)]))
    rows = gv._sort_stt_rtf_batch_rows(batch.rows, "grade")
    assert [r["engine"] for r in rows] == ["fast", "rt", "slow"]


def test_sort_by_delta_descending_abs():
    # corpus median of [0.1, 0.8, 1.5] = 0.8; |delta|: a=0.7, c=0.7, b=0.0.
    batch = _batch(("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]), ("c", [(10.0, 15.0)]))
    rows = gv._sort_stt_rtf_batch_rows(batch.rows, "delta")
    # a and c tie at 0.7 -> stable input order keeps a before c; b (0.0) last.
    assert [r["engine"] for r in rows] == ["a", "c", "b"]


def test_sort_unprofiled_sorts_last_under_every_key():
    for key in ("median_rtf", "grade", "delta"):
        batch = _batch(("empty", []), ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
        rows = gv._sort_stt_rtf_batch_rows(batch.rows, key)
        assert rows[-1]["engine"] == "empty", key


def test_sort_unrecognised_key_preserves_order():
    batch = _batch(("c", [(10.0, 15.0)]), ("a", [(10.0, 1.0)]))
    rows = gv._sort_stt_rtf_batch_rows(batch.rows, "bogus")
    assert [r["engine"] for r in rows] == ["c", "a"]


# human render ------------------------------------------------------------


def test_human_render_sort_by_reorders_and_echoes_header():
    batch = _batch(("c", [(10.0, 15.0)]), ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    lines = gv.render_stt_rtf_batch(batch, sort_by="median_rtf")
    assert lines[0].endswith("(sorted by median_rtf)")
    assert _engine_order(lines) == ["a", "b", "c"]


def test_human_render_default_unchanged():
    batch = _batch(("c", [(10.0, 15.0)]), ("a", [(10.0, 1.0)]))
    assert gv.render_stt_rtf_batch(batch) == gv.render_stt_rtf_batch(
        batch, sort_by=None, top_n=None
    )
    # No reshaping markers in the header by default.
    assert "(sorted by" not in gv.render_stt_rtf_batch(batch)[0]
    assert "(top " not in gv.render_stt_rtf_batch(batch)[0]


def test_human_render_top_n_truncates_and_echoes_header():
    batch = _batch(("c", [(10.0, 15.0)]), ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    lines = gv.render_stt_rtf_batch(batch, sort_by="median_rtf", top_n=2)
    assert "(top 2 of 3)" in lines[0]
    assert _engine_order(lines) == ["a", "b"]


def test_human_render_top_n_ge_count_no_marker():
    batch = _batch(("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    lines = gv.render_stt_rtf_batch(batch, top_n=5)
    assert "(top " not in lines[0]
    assert _engine_order(lines) == ["a", "b"]


def test_human_render_corpus_unaffected_by_top_n():
    # Aggregates describe the WHOLE corpus even when rows are truncated.
    batch = _batch(("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]), ("c", [(10.0, 15.0)]))
    full = [l for l in gv.render_stt_rtf_batch(batch) if l.strip().startswith(("corpus:", "grades:"))]
    capped = [
        l
        for l in gv.render_stt_rtf_batch(batch, sort_by="median_rtf", top_n=1)
        if l.strip().startswith(("corpus:", "grades:"))
    ]
    assert full == capped


# json render -------------------------------------------------------------


def test_json_sort_by_reorders_rows_and_adds_key():
    import json

    batch = _batch(("c", [(10.0, 15.0)]), ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    payload = json.loads(gv.render_stt_rtf_batch_json(batch, sort_by="median_rtf"))
    assert payload["sort_by"] == "median_rtf"
    assert [r["engine"] for r in payload["rows"]] == ["a", "b", "c"]


def test_json_top_n_truncates_rows_and_adds_key():
    import json

    batch = _batch(("c", [(10.0, 15.0)]), ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    payload = json.loads(
        gv.render_stt_rtf_batch_json(batch, sort_by="median_rtf", top_n=2)
    )
    assert payload["top_n"] == 2
    assert [r["engine"] for r in payload["rows"]] == ["a", "b"]
    # Corpus aggregates still describe the whole corpus.
    assert payload["num_engines"] == 3
    assert payload["num_profiled"] == 3


def test_json_default_has_no_sort_or_topn_keys():
    import json

    batch = _batch(("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    payload = json.loads(gv.render_stt_rtf_batch_json(batch))
    assert "sort_by" not in payload
    assert "top_n" not in payload


# csv render --------------------------------------------------------------


def _csv_data_engines(text):
    out = []
    for line in text.splitlines():
        if line.startswith("#") or line.startswith("engine,"):
            continue
        out.append(line.split(",", 1)[0])
    return out


def test_csv_sort_by_reorders_rows_and_comments():
    batch = _batch(("c", [(10.0, 15.0)]), ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    text = gv.render_stt_rtf_batch_csv(batch, sort_by="median_rtf")
    assert "# sort_by: median_rtf" in text.splitlines()
    assert _csv_data_engines(text) == ["a", "b", "c"]


def test_csv_top_n_truncates_rows_and_comments():
    batch = _batch(("c", [(10.0, 15.0)]), ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    text = gv.render_stt_rtf_batch_csv(batch, sort_by="median_rtf", top_n=2)
    comment_lines = [l for l in text.splitlines() if l.startswith("#")]
    assert "# sort_by: median_rtf" in comment_lines
    assert "# top_n: 2" in comment_lines
    # sort_by reads before top_n.
    assert comment_lines.index("# sort_by: median_rtf") < comment_lines.index("# top_n: 2")
    assert _csv_data_engines(text) == ["a", "b"]
    # Corpus comment still names all engines.
    assert "# num_engines: 3" in comment_lines


def test_csv_default_has_no_sort_or_topn_comments():
    batch = _batch(("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    text = gv.render_stt_rtf_batch_csv(batch)
    assert "# sort_by:" not in text
    assert "# top_n:" not in text


# handler threading -------------------------------------------------------


def test_handler_threads_sort_and_topn_to_human():
    args = gv.build_parser().parse_args(
        [
            "stt-rtf-batch",
            "--engine", "c", "10.0:15.0",
            "--engine", "a", "10.0:1.0",
            "--engine", "b", "10.0:8.0",
            "--sort-by", "median_rtf",
            "--top-n", "2",
        ]
    )
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    batch = _batch(("c", [(10.0, 15.0)]), ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    assert lines == gv.render_stt_rtf_batch(batch, sort_by="median_rtf", top_n=2)


def test_handler_threads_sort_and_topn_to_json():
    args = gv.build_parser().parse_args(
        [
            "stt-rtf-batch",
            "--engine", "c", "10.0:15.0",
            "--engine", "a", "10.0:1.0",
            "--json",
            "--sort-by", "median_rtf",
        ]
    )
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    batch = _batch(("c", [(10.0, 15.0)]), ("a", [(10.0, 1.0)]))
    assert lines == [gv.render_stt_rtf_batch_json(batch, sort_by="median_rtf", top_n=None)]


def test_handler_threads_sort_and_topn_to_csv():
    args = gv.build_parser().parse_args(
        [
            "stt-rtf-batch",
            "--engine", "c", "10.0:15.0",
            "--engine", "a", "10.0:1.0",
            "--csv",
            "--sort-by", "grade",
            "--top-n", "1",
        ]
    )
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    batch = _batch(("c", [(10.0, 15.0)]), ("a", [(10.0, 1.0)]))
    assert lines == [gv.render_stt_rtf_batch_csv(batch, sort_by="grade", top_n=1)]


# ---- iter-412: --min-grade render-only speed-grade floor ----------------
#
# A speed-grade floor drops every engine below it, leaving only the engines
# that keep up well enough. The grades produced by the fixtures: 0.1 RTF (10:1)
# -> "fast", 0.8 RTF (10:8) -> "realtime", 1.5 RTF (10:15) -> "slow".


# parser wiring -----------------------------------------------------------


def test_min_grade_defaults_none():
    args = gv.build_parser().parse_args(["stt-rtf-batch", "--engine", "a", "10.0:1.0"])
    assert args.min_grade is None


@pytest.mark.parametrize("grade", ["slow", "realtime", "fast"])
def test_min_grade_parses_each_grade(grade):
    args = gv.build_parser().parse_args(
        ["stt-rtf-batch", "--engine", "a", "10.0:1.0", "--min-grade", grade]
    )
    assert args.min_grade == grade


def test_min_grade_rejects_unknown():
    with pytest.raises(SystemExit) as exc:
        gv.build_parser().parse_args(
            ["stt-rtf-batch", "--engine", "a", "10.0:1.0", "--min-grade", "turbo"]
        )
    assert exc.value.code == 2


# min-grade type validator ------------------------------------------------


def test_min_grade_type_accepts_each_grade():
    for g in ("slow", "realtime", "fast"):
        assert gv.stt_rtf_batch_min_grade_type(g) == g


def test_min_grade_type_normalizes_case_and_whitespace():
    assert gv.stt_rtf_batch_min_grade_type("  FAST ") == "fast"


def test_min_grade_type_rejects_empty():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.stt_rtf_batch_min_grade_type("")


def test_min_grade_type_rejects_unknown():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.stt_rtf_batch_min_grade_type("unprofiled")


def test_min_grade_type_rejects_non_string():
    with pytest.raises(argparse.ArgumentTypeError):
        gv.stt_rtf_batch_min_grade_type(2)


# filter primitive --------------------------------------------------------


def test_filter_none_keeps_every_engine():
    batch = _batch(("fast", [(10.0, 1.0)]), ("rt", [(10.0, 8.0)]), ("empty", []))
    rows = gv._filter_stt_rtf_batch_rows_by_grade(batch.rows, None)
    assert [r["engine"] for r in rows] == ["fast", "rt", "empty"]
    # A copy, not the source tuple.
    assert rows is not batch.rows


def test_filter_slow_keeps_all_profiled_drops_unprofiled():
    batch = _batch(
        ("fast", [(10.0, 1.0)]),
        ("rt", [(10.0, 8.0)]),
        ("slow", [(10.0, 15.0)]),
        ("empty", []),
    )
    rows = gv._filter_stt_rtf_batch_rows_by_grade(batch.rows, "slow")
    assert [r["engine"] for r in rows] == ["fast", "rt", "slow"]


def test_filter_realtime_drops_slow_and_unprofiled():
    batch = _batch(
        ("fast", [(10.0, 1.0)]),
        ("rt", [(10.0, 8.0)]),
        ("slow", [(10.0, 15.0)]),
        ("empty", []),
    )
    rows = gv._filter_stt_rtf_batch_rows_by_grade(batch.rows, "realtime")
    assert [r["engine"] for r in rows] == ["fast", "rt"]


def test_filter_fast_keeps_only_fastest():
    batch = _batch(
        ("fast", [(10.0, 1.0)]),
        ("rt", [(10.0, 8.0)]),
        ("slow", [(10.0, 15.0)]),
    )
    rows = gv._filter_stt_rtf_batch_rows_by_grade(batch.rows, "fast")
    assert [r["engine"] for r in rows] == ["fast"]


def test_grade_meets_min_total_order():
    # None / unrecognised rank below every floor; the floor itself always passes.
    assert gv._stt_rtf_grade_meets_min("fast", "slow")
    assert gv._stt_rtf_grade_meets_min("slow", "slow")
    assert not gv._stt_rtf_grade_meets_min("slow", "realtime")
    assert not gv._stt_rtf_grade_meets_min(None, "slow")
    assert not gv._stt_rtf_grade_meets_min("bogus", "slow")
    # None min_grade passes everything (no filter requested).
    assert gv._stt_rtf_grade_meets_min(None, None)


# human render ------------------------------------------------------------


def test_human_render_min_grade_filters_and_echoes_header():
    batch = _batch(
        ("fast", [(10.0, 1.0)]), ("rt", [(10.0, 8.0)]), ("slow", [(10.0, 15.0)])
    )
    lines = gv.render_stt_rtf_batch(batch, min_grade="realtime")
    assert "(min grade realtime)" in lines[0]
    assert _engine_order(lines) == ["fast", "rt"]


def test_human_render_min_grade_applied_before_sort_and_top_n():
    # Floor keeps fast+rt; sort by median_rtf asc; top-n 1 -> fast only.
    batch = _batch(
        ("rt", [(10.0, 8.0)]), ("fast", [(10.0, 1.0)]), ("slow", [(10.0, 15.0)])
    )
    lines = gv.render_stt_rtf_batch(
        batch, min_grade="realtime", sort_by="median_rtf", top_n=1
    )
    assert "(min grade realtime)" in lines[0]
    assert "(sorted by median_rtf)" in lines[0]
    assert _engine_order(lines) == ["fast"]


def test_human_render_min_grade_removes_every_engine_emits_note():
    batch = _batch(("slow", [(10.0, 15.0)]), ("empty", []))
    lines = gv.render_stt_rtf_batch(batch, min_grade="fast")
    text = "\n".join(lines)
    assert "(no engine profiled to grade 'fast' or better)" in text
    assert _engine_order(lines) == []


def test_human_render_min_grade_does_not_change_corpus_summary():
    batch = _batch(
        ("fast", [(10.0, 1.0)]), ("rt", [(10.0, 8.0)]), ("slow", [(10.0, 15.0)])
    )
    full = [
        l
        for l in gv.render_stt_rtf_batch(batch)
        if l.strip().startswith(("corpus:", "grades:"))
    ]
    floored = [
        l
        for l in gv.render_stt_rtf_batch(batch, min_grade="fast")
        if l.strip().startswith(("corpus:", "grades:"))
    ]
    assert full == floored


def test_human_render_default_has_no_min_grade_marker():
    batch = _batch(("fast", [(10.0, 1.0)]), ("rt", [(10.0, 8.0)]))
    assert "(min grade" not in gv.render_stt_rtf_batch(batch)[0]


# json render -------------------------------------------------------------


def test_json_min_grade_filters_rows_and_names_key():
    import json

    batch = _batch(
        ("fast", [(10.0, 1.0)]), ("rt", [(10.0, 8.0)]), ("slow", [(10.0, 15.0)])
    )
    payload = json.loads(gv.render_stt_rtf_batch_json(batch, min_grade="realtime"))
    assert payload["min_grade"] == "realtime"
    assert [r["engine"] for r in payload["rows"]] == ["fast", "rt"]
    # Corpus aggregates still describe the whole corpus.
    assert payload["num_engines"] == 3
    assert payload["num_profiled"] == 3


def test_json_min_grade_none_omits_key():
    import json

    batch = _batch(("fast", [(10.0, 1.0)]), ("rt", [(10.0, 8.0)]))
    payload = json.loads(gv.render_stt_rtf_batch_json(batch))
    assert "min_grade" not in payload


def test_json_min_grade_combines_with_sort_and_topn():
    import json

    batch = _batch(
        ("rt", [(10.0, 8.0)]), ("fast", [(10.0, 1.0)]), ("slow", [(10.0, 15.0)])
    )
    payload = json.loads(
        gv.render_stt_rtf_batch_json(
            batch, min_grade="realtime", sort_by="median_rtf", top_n=1
        )
    )
    assert payload["min_grade"] == "realtime"
    assert payload["sort_by"] == "median_rtf"
    assert payload["top_n"] == 1
    assert [r["engine"] for r in payload["rows"]] == ["fast"]


# csv render --------------------------------------------------------------


def test_csv_min_grade_filters_rows_and_comments_key():
    batch = _batch(
        ("fast", [(10.0, 1.0)]), ("rt", [(10.0, 8.0)]), ("slow", [(10.0, 15.0)])
    )
    text = gv.render_stt_rtf_batch_csv(batch, min_grade="realtime")
    assert "# min_grade: realtime" in text.splitlines()
    assert _csv_data_engines(text) == ["fast", "rt"]
    # Corpus comment still names all engines.
    assert "# num_engines: 3" in text.splitlines()


def test_csv_min_grade_none_omits_comment():
    batch = _batch(("fast", [(10.0, 1.0)]), ("rt", [(10.0, 8.0)]))
    text = gv.render_stt_rtf_batch_csv(batch)
    assert "# min_grade:" not in text


def test_csv_min_grade_reads_before_sort_and_topn():
    batch = _batch(
        ("rt", [(10.0, 8.0)]), ("fast", [(10.0, 1.0)]), ("slow", [(10.0, 15.0)])
    )
    text = gv.render_stt_rtf_batch_csv(
        batch, min_grade="realtime", sort_by="median_rtf", top_n=1
    )
    comment_lines = [l for l in text.splitlines() if l.startswith("#")]
    assert "# min_grade: realtime" in comment_lines
    assert "# sort_by: median_rtf" in comment_lines
    assert "# top_n: 1" in comment_lines
    # min_grade reads before sort_by reads before top_n.
    assert comment_lines.index("# min_grade: realtime") < comment_lines.index(
        "# sort_by: median_rtf"
    )
    assert comment_lines.index("# sort_by: median_rtf") < comment_lines.index(
        "# top_n: 1"
    )
    assert _csv_data_engines(text) == ["fast"]


# handler threading -------------------------------------------------------


def test_handler_threads_min_grade_to_human():
    args = gv.build_parser().parse_args(
        [
            "stt-rtf-batch",
            "--engine", "fast", "10.0:1.0",
            "--engine", "rt", "10.0:8.0",
            "--engine", "slow", "10.0:15.0",
            "--min-grade", "realtime",
        ]
    )
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    batch = _batch(
        ("fast", [(10.0, 1.0)]), ("rt", [(10.0, 8.0)]), ("slow", [(10.0, 15.0)])
    )
    assert lines == gv.render_stt_rtf_batch(batch, min_grade="realtime")


def test_handler_threads_min_grade_to_json():
    args = gv.build_parser().parse_args(
        [
            "stt-rtf-batch",
            "--engine", "fast", "10.0:1.0",
            "--engine", "slow", "10.0:15.0",
            "--json",
            "--min-grade", "fast",
        ]
    )
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    batch = _batch(("fast", [(10.0, 1.0)]), ("slow", [(10.0, 15.0)]))
    assert lines == [
        gv.render_stt_rtf_batch_json(batch, min_grade="fast", sort_by=None, top_n=None)
    ]


def test_handler_threads_min_grade_to_csv():
    args = gv.build_parser().parse_args(
        [
            "stt-rtf-batch",
            "--engine", "fast", "10.0:1.0",
            "--engine", "slow", "10.0:15.0",
            "--csv",
            "--min-grade", "realtime",
            "--sort-by", "grade",
        ]
    )
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    batch = _batch(("fast", [(10.0, 1.0)]), ("slow", [(10.0, 15.0)]))
    assert lines == [
        gv.render_stt_rtf_batch_csv(
            batch, min_grade="realtime", sort_by="grade", top_n=None
        )
    ]


# =========================================================================
# iter-413 — --summary: name the single most-representative engine
# =========================================================================

# parser wiring -----------------------------------------------------------


def test_summary_defaults_false():
    args = gv.build_parser().parse_args(["stt-rtf-batch", "--engine", "a", "10.0:1.0"])
    assert args.summary is False


def test_summary_flag_parses_true():
    args = gv.build_parser().parse_args(
        ["stt-rtf-batch", "--engine", "a", "10.0:1.0", "--summary"]
    )
    assert args.summary is True


# selection primitive -----------------------------------------------------


def test_best_row_picks_nearest_corpus_median():
    # Medians 0.1 / 0.8 / 1.5 -> corpus median 0.8, so the 0.8 engine (delta 0) wins.
    batch = _batch(
        ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]), ("c", [(10.0, 15.0)])
    )
    best = gv._best_stt_rtf_batch_row(batch.rows)
    assert best["engine"] == "b"
    assert best["delta_from_median_rtf"] == 0.0


def test_best_row_none_when_no_engine_profiled():
    batch = _batch(("a", []), ("b", []))
    assert gv._best_stt_rtf_batch_row(batch.rows) is None


def test_best_row_ignores_unprofiled_engine():
    # The unprofiled engine has delta None and must never be picked.
    batch = _batch(("a", [(10.0, 1.0)]), ("empty", []))
    best = gv._best_stt_rtf_batch_row(batch.rows)
    assert best["engine"] == "a"


def test_best_row_tie_breaks_to_higher_grade():
    # Two engines equidistant from the corpus median; the faster grade wins.
    # Medians 0.4 / 0.6 / 1.0 -> corpus median 0.6. With two engines tied on |Δ|
    # we instead use a symmetric pair around the median: 0.3 (fast) and 0.9 are
    # 0.3 from median 0.6, but 0.3 grades "fast" and 0.9 grades "realtime".
    batch = _batch(
        ("hi", [(10.0, 3.0)]), ("mid", [(10.0, 6.0)]), ("lo", [(10.0, 9.0)])
    )
    # Drop the exact-median engine so the two flankers tie on |Δ| = 0.3.
    rows = [r for r in batch.rows if r["engine"] != "mid"]
    best = gv._best_stt_rtf_batch_row(rows)
    assert best["engine"] == "hi"  # "fast" beats "realtime" on the grade tie


def test_best_row_tie_breaks_to_earliest_position():
    # Two engines identical in every key keep the earlier-listed one.
    batch = _batch(
        ("first", [(10.0, 6.0)]),
        ("second", [(10.0, 6.0)]),
        ("anchor", [(10.0, 6.0)]),
    )
    best = gv._best_stt_rtf_batch_row(batch.rows)
    assert best["engine"] == "first"


# human render ------------------------------------------------------------


def test_human_summary_names_representative_engine():
    batch = _batch(
        ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]), ("c", [(10.0, 15.0)])
    )
    lines = gv.render_stt_rtf_batch(batch, summary=True)
    rep = [l for l in lines if l.strip().startswith("representative:")][0]
    assert "representative: b → 0.800 RTF (grade realtime, Δmedian 0.000)" in rep
    # No per-engine table rows beyond the verdict (the only ":"-bearing body line
    # is the representative verdict itself).
    assert [e for e in _engine_order(lines) if e != "representative"] == []


def test_human_summary_independent_of_sort_and_top_n():
    batch = _batch(
        ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]), ("c", [(10.0, 15.0)])
    )
    plain = gv.render_stt_rtf_batch(batch, summary=True)
    reshaped = gv.render_stt_rtf_batch(
        batch, summary=True, sort_by="delta", top_n=1
    )
    assert plain == reshaped


def test_human_summary_respects_min_grade():
    # Floor "fast" keeps only the 0.1 engine, so it represents the corpus.
    batch = _batch(
        ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]), ("c", [(10.0, 15.0)])
    )
    lines = gv.render_stt_rtf_batch(batch, summary=True, min_grade="fast")
    rep = [l for l in lines if l.strip().startswith("representative:")][0]
    assert "representative: a →" in rep
    assert "(min grade fast)" in lines[0]


def test_human_summary_empty_corpus_note():
    batch = _batch(("a", []), ("b", []))
    lines = gv.render_stt_rtf_batch(batch, summary=True)
    text = "\n".join(lines)
    assert "nothing to summarise" in text
    assert not any(l.strip().startswith("representative:") for l in lines)


def test_human_summary_min_grade_empties_pick_note():
    batch = _batch(("slow", [(10.0, 15.0)]))
    lines = gv.render_stt_rtf_batch(batch, summary=True, min_grade="fast")
    text = "\n".join(lines)
    assert "no engine profiled to grade 'fast' or better carries a median RTF" in text


def test_human_summary_still_emits_whole_corpus_lines():
    batch = _batch(
        ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]), ("c", [(10.0, 15.0)])
    )
    full = [
        l
        for l in gv.render_stt_rtf_batch(batch)
        if l.strip().startswith(("corpus:", "grades:"))
    ]
    summ = [
        l
        for l in gv.render_stt_rtf_batch(batch, summary=True)
        if l.strip().startswith(("corpus:", "grades:"))
    ]
    assert full == summ


# json render -------------------------------------------------------------


def test_json_summary_replaces_rows_with_best():
    import json

    batch = _batch(
        ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]), ("c", [(10.0, 15.0)])
    )
    payload = json.loads(gv.render_stt_rtf_batch_json(batch, summary=True))
    assert payload["summary"] is True
    assert "rows" not in payload
    assert payload["best"]["engine"] == "b"
    assert payload["best"]["delta_from_median_rtf"] == 0.0
    # Whole-corpus aggregates unchanged.
    assert payload["num_engines"] == 3
    assert payload["num_profiled"] == 3


def test_json_summary_best_null_when_empty():
    import json

    batch = _batch(("a", []), ("b", []))
    payload = json.loads(gv.render_stt_rtf_batch_json(batch, summary=True))
    assert payload["summary"] is True
    assert payload["best"] is None


def test_json_summary_omits_sort_and_top_n_keys():
    import json

    batch = _batch(("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    payload = json.loads(
        gv.render_stt_rtf_batch_json(
            batch, summary=True, sort_by="delta", top_n=1
        )
    )
    assert "sort_by" not in payload
    assert "top_n" not in payload


def test_json_summary_respects_min_grade():
    import json

    batch = _batch(
        ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]), ("c", [(10.0, 15.0)])
    )
    payload = json.loads(
        gv.render_stt_rtf_batch_json(batch, summary=True, min_grade="fast")
    )
    assert payload["min_grade"] == "fast"
    assert payload["best"]["engine"] == "a"


def test_json_default_omits_summary_key():
    import json

    batch = _batch(("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    payload = json.loads(gv.render_stt_rtf_batch_json(batch))
    assert "summary" not in payload
    assert "rows" in payload


# csv render --------------------------------------------------------------


def test_csv_summary_single_data_row_and_comment():
    batch = _batch(
        ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]), ("c", [(10.0, 15.0)])
    )
    text = gv.render_stt_rtf_batch_csv(batch, summary=True)
    assert "# summary: true" in text.splitlines()
    assert _csv_data_engines(text) == ["b"]
    # Corpus comment still describes the whole corpus.
    assert "# num_engines: 3" in text.splitlines()


def test_csv_summary_header_only_when_empty():
    batch = _batch(("a", []), ("b", []))
    text = gv.render_stt_rtf_batch_csv(batch, summary=True)
    assert _csv_data_engines(text) == []
    assert "# summary: true" in text.splitlines()


def test_csv_summary_omits_sort_and_top_n_comments():
    batch = _batch(("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    text = gv.render_stt_rtf_batch_csv(
        batch, summary=True, sort_by="delta", top_n=1
    )
    assert "# sort_by:" not in text
    assert "# top_n:" not in text


def test_csv_summary_respects_min_grade():
    batch = _batch(
        ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]), ("c", [(10.0, 15.0)])
    )
    text = gv.render_stt_rtf_batch_csv(batch, summary=True, min_grade="fast")
    comment_lines = [l for l in text.splitlines() if l.startswith("#")]
    assert "# summary: true" in comment_lines
    assert "# min_grade: fast" in comment_lines
    # summary reads before min_grade.
    assert comment_lines.index("# summary: true") < comment_lines.index(
        "# min_grade: fast"
    )
    assert _csv_data_engines(text) == ["a"]


def test_csv_default_omits_summary_comment():
    batch = _batch(("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    text = gv.render_stt_rtf_batch_csv(batch)
    assert "# summary:" not in text


# handler threading -------------------------------------------------------


def test_handler_threads_summary_to_human():
    args = gv.build_parser().parse_args(
        [
            "stt-rtf-batch",
            "--engine", "a", "10.0:1.0",
            "--engine", "b", "10.0:8.0",
            "--engine", "c", "10.0:15.0",
            "--summary",
        ]
    )
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    batch = _batch(
        ("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]), ("c", [(10.0, 15.0)])
    )
    assert lines == gv.render_stt_rtf_batch(batch, summary=True)


def test_handler_threads_summary_to_json():
    args = gv.build_parser().parse_args(
        [
            "stt-rtf-batch",
            "--engine", "a", "10.0:1.0",
            "--engine", "b", "10.0:8.0",
            "--json",
            "--summary",
        ]
    )
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    batch = _batch(("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    assert lines == [
        gv.render_stt_rtf_batch_json(
            batch, min_grade=None, sort_by=None, top_n=None, summary=True
        )
    ]


def test_handler_threads_summary_to_csv():
    args = gv.build_parser().parse_args(
        [
            "stt-rtf-batch",
            "--engine", "a", "10.0:1.0",
            "--engine", "b", "10.0:8.0",
            "--csv",
            "--summary",
            "--min-grade", "realtime",
        ]
    )
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    batch = _batch(("a", [(10.0, 1.0)]), ("b", [(10.0, 8.0)]))
    assert lines == [
        gv.render_stt_rtf_batch_csv(
            batch, min_grade="realtime", sort_by=None, top_n=None, summary=True
        )
    ]


# ---- iter-414: per-engine flyer flag + corpus flyers line ----------------


def _flyer_batch():
    # Four tightly-clustered engines (rtf ~0.1) plus one wild flyer (rtf 5.0).
    # sorted medians: [0.1, 0.11, 0.12, 0.13, 5.0] => Q1 0.11, Q3 0.13, IQR 0.02,
    # fence [0.080, 0.160]; only ``slow`` falls outside it.
    return _batch(
        ("a", [(10.0, 1.0)]),  # 0.1
        ("b", [(10.0, 1.1)]),  # 0.11
        ("c", [(10.0, 1.2)]),  # 0.12
        ("d", [(10.0, 1.3)]),  # 0.13
        ("slow", [(10.0, 50.0)]),  # 5.0 — flyer
    )


def test_render_marks_flyer_row_and_names_corpus_flyers():
    text = "\n".join(gv.render_stt_rtf_batch(_flyer_batch()))
    # The flyer engine's row carries the inline ← flyer marker...
    slow_line = next(l for l in text.splitlines() if l.strip().startswith("slow:"))
    assert "← flyer" in slow_line
    a_line = next(l for l in text.splitlines() if l.strip().startswith("a:"))
    assert "← flyer" not in a_line
    # ...and the corpus flyers: line names the outlier and the fence bounds.
    assert "flyers: 1 (slow) outside [0.080, 0.160] RTF" in text


def test_render_flyers_none_when_corpus_agrees():
    text = "\n".join(
        gv.render_stt_rtf_batch(
            _batch(
                ("a", [(10.0, 1.0)]),
                ("b", [(10.0, 1.1)]),
                ("c", [(10.0, 1.2)]),
            )
        )
    )
    assert "flyers: none" in text
    assert "← flyer" not in text


def test_render_flyers_line_describes_whole_corpus_under_top_n():
    # --top-n truncates the displayed rows, but the flyers: line still names the
    # outlier over the WHOLE corpus even when the flyer row itself is elided.
    text = "\n".join(
        gv.render_stt_rtf_batch(_flyer_batch(), sort_by="median_rtf", top_n=2)
    )
    # median_rtf ascending keeps the two fastest (a, b) — the flyer row is dropped...
    assert "slow:" not in text
    # ...but the corpus flyers: line still names it.
    assert "flyers: 1 (slow) outside [0.080, 0.160] RTF" in text


def test_render_flyers_line_present_under_summary():
    # --summary collapses to one verdict row, but the corpus flyers: line still
    # describes the whole corpus.
    text = "\n".join(gv.render_stt_rtf_batch(_flyer_batch(), summary=True))
    assert "flyers: 1 (slow) outside [0.080, 0.160] RTF" in text


def test_render_empty_corpus_has_no_flyers_line():
    text = "\n".join(gv.render_stt_rtf_batch(_batch(("a", []), ("b", []))))
    assert "flyers:" not in text


def test_render_json_carries_fence_and_per_row_flyer():
    import json

    payload = json.loads(gv.render_stt_rtf_batch_json(_flyer_batch()))
    # The IQR / Tukey-fence corpus keys ride alongside the existing aggregates.
    assert round(payload["corpus_q1_rtf"], 3) == 0.11
    assert round(payload["corpus_q3_rtf"], 3) == 0.13
    assert round(payload["corpus_iqr_rtf"], 3) == 0.02
    assert round(payload["corpus_fence_lo_rtf"], 3) == 0.08
    assert round(payload["corpus_fence_hi_rtf"], 3) == 0.16
    assert payload["num_flyers"] == 1
    flags = {r["engine"]: r["flyer"] for r in payload["rows"]}
    assert flags["slow"] is True
    assert flags["a"] is False


def test_render_json_empty_corpus_fence_null_and_no_flyers():
    import json

    payload = json.loads(gv.render_stt_rtf_batch_json(_batch(("a", []), ("b", []))))
    assert payload["corpus_q1_rtf"] is None
    assert payload["corpus_q3_rtf"] is None
    assert payload["corpus_iqr_rtf"] is None
    assert payload["corpus_fence_lo_rtf"] is None
    assert payload["corpus_fence_hi_rtf"] is None
    assert payload["num_flyers"] == 0
    # An unprofiled engine's row flyer is null.
    assert all(r["flyer"] is None for r in payload["rows"])


def test_render_json_summary_best_carries_flyer():
    import json

    payload = json.loads(gv.render_stt_rtf_batch_json(_flyer_batch(), summary=True))
    # The best (most-representative) engine is not the flyer.
    assert payload["best"]["flyer"] is False
    # num_flyers still describes the whole corpus in summary mode.
    assert payload["num_flyers"] == 1


def test_render_csv_carries_fence_comment_and_per_row_flyer():
    import csv as _csv
    import io as _io

    text = gv.render_stt_rtf_batch_csv(_flyer_batch())
    # The fence + num_flyers ride in the trailing # comment block...
    assert "# corpus_iqr_rtf: 0.02" in text
    assert "# corpus_fence_rtf: 0.08 - 0.16" in text
    assert "# num_flyers: 1" in text
    # ...and the flyer column carries true/false per engine.
    data = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
    reader = {r["engine"]: r for r in _csv.DictReader(_io.StringIO(data))}
    assert reader["slow"]["flyer"] == "true"
    assert reader["a"]["flyer"] == "false"
    # The flyer column is the last header column.
    header = data.splitlines()[0]
    assert header.endswith(",flyer")


def test_render_csv_unprofiled_engine_flyer_cell_blank():
    batch = _batch(("a", [(10.0, 1.0)]), ("empty", []))
    text = gv.render_stt_rtf_batch_csv(batch)
    data = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
    empty = [l for l in data.splitlines() if l.startswith("empty,")][0]
    # engine label then every numeric cell blank, including the trailing flyer.
    assert empty == "empty,,,,,,,,,,,"


# ---- iter-416: --flyers-only (show ONLY the corpus outlier engines) ------


def test_filter_flyers_only_false_returns_copy_of_all_rows():
    # The default (flyers_only=False) keeps every row and returns a fresh list,
    # never the source object — byte-identical to the pre-filter rows.
    batch = _flyer_batch()
    out = gv._filter_stt_rtf_batch_rows_flyers_only(batch.rows, False)
    assert out == list(batch.rows)
    assert out is not batch.rows


def test_filter_flyers_only_keeps_only_outlier_rows():
    batch = _flyer_batch()
    out = gv._filter_stt_rtf_batch_rows_flyers_only(batch.rows, True)
    assert [r["engine"] for r in out] == ["slow"]


def test_filter_flyers_only_drops_unprofiled_engine():
    # An unprofiled engine carries flyer=None (no median RTF to be an outlier) and
    # is never kept by the flyers-only filter.
    batch = _batch(
        ("a", [(10.0, 1.0)]),
        ("b", [(10.0, 1.1)]),
        ("c", [(10.0, 1.2)]),
        ("d", [(10.0, 1.3)]),
        ("slow", [(10.0, 50.0)]),
        ("empty", []),
    )
    out = gv._filter_stt_rtf_batch_rows_flyers_only(batch.rows, True)
    assert [r["engine"] for r in out] == ["slow"]


def test_filter_flyers_only_does_not_mutate_source():
    batch = _flyer_batch()
    before = list(batch.rows)
    gv._filter_stt_rtf_batch_rows_flyers_only(batch.rows, True)
    assert list(batch.rows) == before


def test_empty_filter_note_min_grade_only_unchanged():
    # With only a grade floor the note is byte-identical to the pre-iter-416 wording.
    assert (
        gv._stt_rtf_batch_empty_filter_note("fast", False)
        == "no engine profiled to grade 'fast' or better"
    )


def test_empty_filter_note_flyers_only():
    assert (
        gv._stt_rtf_batch_empty_filter_note(None, True)
        == "no engine is a corpus flyer"
    )


def test_empty_filter_note_both_clauses_joined():
    assert (
        gv._stt_rtf_batch_empty_filter_note("fast", True)
        == "no engine profiled to grade 'fast' or better and is a corpus flyer"
    )


def test_parser_flyers_only_default_false():
    args = gv.build_parser().parse_args(
        ["stt-rtf-batch", "--engine", "a", "10.0:1.0"]
    )
    assert args.flyers_only is False


def test_parser_flyers_only_sets_true():
    args = gv.build_parser().parse_args(
        ["stt-rtf-batch", "--engine", "a", "10.0:1.0", "--flyers-only"]
    )
    assert args.flyers_only is True


def test_render_flyers_only_shows_just_the_outlier_rows():
    lines = gv.render_stt_rtf_batch(_flyer_batch(), flyers_only=True)
    text = "\n".join(lines)
    # The header echoes the mode...
    assert "(flyers only)" in lines[0]
    # ...only the outlier engine row is shown (still marked ← flyer)...
    assert _engine_order(lines) == ["slow"]
    assert "← flyer" in text


def test_render_flyers_only_default_unchanged():
    batch = _flyer_batch()
    assert gv.render_stt_rtf_batch(batch, flyers_only=False) == (
        gv.render_stt_rtf_batch(batch)
    )


def test_render_flyers_only_empty_when_corpus_agrees_emits_note():
    # A corpus with no outlier yields an empty flyers-only table + a note.
    batch = _batch(
        ("a", [(10.0, 1.0)]),
        ("b", [(10.0, 1.1)]),
        ("c", [(10.0, 1.2)]),
    )
    lines = gv.render_stt_rtf_batch(batch, flyers_only=True)
    text = "\n".join(lines)
    assert "(no engine is a corpus flyer)" in text
    # The corpus / grades / flyers lines still describe the whole corpus.
    assert any(l.strip().startswith("corpus:") for l in lines)
    assert "flyers: none" in text


def test_render_flyers_only_composes_with_min_grade_note():
    # A grade floor that empties the table names BOTH clauses in the note.
    batch = _batch(
        ("a", [(10.0, 1.0)]),
        ("b", [(10.0, 1.1)]),
        ("c", [(10.0, 1.2)]),
    )
    text = "\n".join(
        gv.render_stt_rtf_batch(batch, min_grade="fast", flyers_only=True)
    )
    assert (
        "no engine profiled to grade 'fast' or better and is a corpus flyer"
        in text
    )


def test_render_flyers_only_does_not_change_corpus_aggregates():
    batch = _flyer_batch()
    only = "\n".join(gv.render_stt_rtf_batch(batch, flyers_only=True))
    full = "\n".join(gv.render_stt_rtf_batch(batch))
    corpus_only = next(l for l in only.splitlines() if l.strip().startswith("corpus:"))
    corpus_full = next(l for l in full.splitlines() if l.strip().startswith("corpus:"))
    assert corpus_only == corpus_full
    assert "flyers: 1 (slow) outside [0.080, 0.160] RTF" in only


def test_render_flyers_only_composes_with_summary_picks_among_flyers():
    # --summary + --flyers-only: the representative is chosen among the outliers only.
    text = "\n".join(
        gv.render_stt_rtf_batch(_flyer_batch(), summary=True, flyers_only=True)
    )
    # Only one engine is a flyer, so it is the representative.
    assert "slow" in text
    assert "(flyers only)" in text


def test_json_flyers_only_filters_rows_and_names_key():
    import json as _json

    obj = _json.loads(
        gv.render_stt_rtf_batch_json(_flyer_batch(), flyers_only=True)
    )
    assert obj["flyers_only"] is True
    assert [r["engine"] for r in obj["rows"]] == ["slow"]
    # The corpus aggregates still describe the whole corpus.
    assert obj["num_engines"] == 5
    assert obj["num_flyers"] == 1


def test_json_flyers_only_none_omits_key():
    import json as _json

    obj = _json.loads(gv.render_stt_rtf_batch_json(_flyer_batch()))
    assert "flyers_only" not in obj


def test_json_flyers_only_summary_best_among_flyers():
    import json as _json

    obj = _json.loads(
        gv.render_stt_rtf_batch_json(
            _flyer_batch(), summary=True, flyers_only=True
        )
    )
    assert obj["flyers_only"] is True
    assert obj["best"]["engine"] == "slow"


def test_csv_flyers_only_filters_rows_and_comments_key():
    import csv as _csv
    import io as _io

    text = gv.render_stt_rtf_batch_csv(_flyer_batch(), flyers_only=True)
    assert "# flyers_only: true" in text
    data = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
    rows = list(_csv.DictReader(_io.StringIO(data)))
    assert [r["engine"] for r in rows] == ["slow"]
    # The whole-corpus num_flyers comment is unchanged.
    assert "# num_flyers: 1" in text


def test_csv_flyers_only_none_omits_comment():
    text = gv.render_stt_rtf_batch_csv(_flyer_batch())
    assert "# flyers_only" not in text


def test_csv_flyers_only_reads_after_min_grade_comment():
    # Comment order is min_grade -> flyers_only (the order the filters apply).
    text = gv.render_stt_rtf_batch_csv(
        _flyer_batch(), min_grade="slow", flyers_only=True
    )
    lines = text.splitlines()
    mg = next(i for i, ln in enumerate(lines) if ln.startswith("# min_grade:"))
    fo = next(i for i, ln in enumerate(lines) if ln.startswith("# flyers_only:"))
    assert mg < fo


def test_handler_flyers_only_threads_to_human_render():
    args = gv.build_parser().parse_args(
        [
            "stt-rtf-batch",
            "--engine", "a", "10.0:1.0",
            "--engine", "b", "10.0:1.1",
            "--engine", "c", "10.0:1.2",
            "--engine", "d", "10.0:1.3",
            "--engine", "slow", "10.0:50.0",
            "--flyers-only",
        ]
    )
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    text = "\n".join(lines)
    assert "(flyers only)" in text
    assert _engine_order(lines) == ["slow"]


def test_handler_flyers_only_matches_render_directly():
    argv = [
        "stt-rtf-batch",
        "--engine", "a", "10.0:1.0",
        "--engine", "b", "10.0:1.1",
        "--engine", "c", "10.0:1.2",
        "--engine", "d", "10.0:1.3",
        "--engine", "slow", "10.0:50.0",
        "--flyers-only",
    ]
    # human
    args = gv.build_parser().parse_args(argv)
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    batch = _flyer_batch()
    assert lines == gv.render_stt_rtf_batch(batch, flyers_only=True)
    # json
    args = gv.build_parser().parse_args(argv + ["--json"])
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    assert lines == [gv.render_stt_rtf_batch_json(batch, flyers_only=True)]
    # csv
    args = gv.build_parser().parse_args(argv + ["--csv"])
    lines = []
    gv.cmd_stt_rtf_batch(args, log=lines.append)
    assert lines == [gv.render_stt_rtf_batch_csv(batch, flyers_only=True)]
