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
        "speed_grade,speed_margin,delta_from_median_rtf,recommend"
    )
    assert rows[1].startswith("fast,0.1,1,")
    assert rows[2].startswith("slow,1.5,1,")
    # one header + one row per engine
    assert len(rows) == 3


def test_render_csv_unprofiled_engine_blank_cells():
    batch = _batch(("a", [(10.0, 1.0)]), ("empty", []))
    rows = _csv_data_rows(gv.render_stt_rtf_batch_csv(batch))
    empty = [r for r in rows if r.startswith("empty,")][0]
    assert empty == "empty,,,,,,,,,,"


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
    assert heavy.endswith(",true")
    assert light.endswith(",false")


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
