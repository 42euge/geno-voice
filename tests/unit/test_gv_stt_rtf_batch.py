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
