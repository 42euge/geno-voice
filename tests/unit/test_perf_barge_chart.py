"""Tests for iter-042 — barge-in latency in perf scenario schema
+ generator chart rendering.

The perf-suite scenario itself runs in tests/performance/ (not
unit). These tests cover the generator side: given a snapshot
with a barge-in row, the page renders barge-latency charts. With
no barge measurements, the charts are suppressed (don't show
all-zero bars).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "generate_iteration_reports.py"

spec = importlib.util.spec_from_file_location("gen_reports", SCRIPT_PATH)
gen_reports = importlib.util.module_from_spec(spec)
sys.modules["gen_reports"] = gen_reports
spec.loader.exec_module(gen_reports)


def _scenario(**overrides):
    base = {
        "name": "default",
        "description": "default scenario",
        "ttfs_ms": 100.0,
        "stt_ms": 50.0,
        "tts_ms": 30.0,
        "playback_ms": 25.0,
        "llm_first_token_ms": 80.0,
        "llm_first_sentence_ms": 90.0,
        "llm_total_ms": 200.0,
        "speech_duration_ms": 1000.0,
        "sentences_spoken": 1,
        "sentences_cancelled": 0,
        "wall_ms": 250.0,
        "barge_in": False,
        "barge_in_latency_ms": 0.0,
        "mic_stale_frames": 0,
    }
    base.update(overrides)
    return base


def _payload(scenarios, iteration="042"):
    return {
        "captured_at": f"2026-05-24T12:00:00",
        "iteration": iteration,
        "scenarios": scenarios,
    }


# ---- Latest-snapshot chart -------------------------------------------------


class TestLatestSnapshotBargeChart:
    def test_no_barge_data_omits_chart(self):
        # All scenarios have barge_in_latency_ms=0 → chart suppressed.
        out = gen_reports.render_performance_page(
            _payload([_scenario(name="s1"), _scenario(name="s2")])
        )
        assert "Barge-in latency" not in out

    def test_at_least_one_barge_measurement_emits_chart(self):
        out = gen_reports.render_performance_page(
            _payload([
                _scenario(name="short_short"),  # no barge
                _scenario(
                    name="barge_in",
                    barge_in=True,
                    barge_in_latency_ms=42.0,
                    sentences_cancelled=1,
                ),
            ])
        )
        # Header rendered once in latest snapshot section.
        assert "Barge-in latency" in out
        # The numeric value shows up in the bar chart label.
        assert "42ms" in out

    def test_chart_color_is_yellow_palette(self):
        # Sanity: the iter-042 helper renders barge bars in #f7c453
        # (the yellow palette color). Quick check for the color.
        out = gen_reports.render_performance_page(
            _payload([_scenario(barge_in=True, barge_in_latency_ms=10.0)])
        )
        assert "#f7c453" in out


# ---- Time-series across iterations -----------------------------------------


class TestHistoryBargeChart:
    def test_history_with_no_barge_omits_chart(self, tmp_path):
        history = [
            _payload([_scenario(name="a")], iteration="040"),
            _payload([_scenario(name="a")], iteration="041"),
        ]
        out = gen_reports.render_performance_page(history[-1], history)
        # No barge-latency time-series since no row had a measurement.
        # (The latest-snapshot section also drops it; so 0 occurrences.)
        assert "Barge-in latency" not in out

    def test_history_with_barge_emits_time_series(self, tmp_path):
        # iter-042 has a real barge measurement; iter-041 has none.
        # The history view should include a barge chart with one
        # data point (only iter-042).
        history = [
            _payload([_scenario(name="a")], iteration="041"),
            _payload(
                [
                    _scenario(name="a"),
                    _scenario(
                        name="barge_in",
                        barge_in=True,
                        barge_in_latency_ms=15.0,
                    ),
                ],
                iteration="042",
            ),
        ]
        out = gen_reports.render_performance_page(history[-1], history)
        # Latest-snapshot AND history both render barge charts —
        # >= 2 occurrences.
        assert out.count("Barge-in latency") >= 2

    def test_history_legend_includes_barge_scenario(self):
        history = [
            _payload(
                [
                    _scenario(name="short"),
                    _scenario(
                        name="barge_in",
                        barge_in=True,
                        barge_in_latency_ms=20.0,
                    ),
                ],
                iteration="042",
            ),
            _payload(
                [
                    _scenario(name="short"),
                    _scenario(
                        name="barge_in",
                        barge_in=True,
                        barge_in_latency_ms=18.0,
                    ),
                ],
                iteration="043",
            ),
        ]
        out = gen_reports.render_performance_page(history[-1], history)
        # Legend label "barge_in" appears in the time-series chart.
        assert ">barge_in<" in out


# ---- ScenarioResult schema —  all new fields surfaced -------------------


class TestScenarioSchemaSurfacedInTable:
    """The scenario description table on performance.html shows
    sentence count + barge_in flag. After iter-042, the ``barge_in``
    column should reflect the new ``barge_in`` boolean.
    """

    def test_table_shows_yes_when_barge_landed(self):
        out = gen_reports.render_performance_page(
            _payload([
                _scenario(
                    name="barge_in",
                    barge_in=True,
                    barge_in_latency_ms=10.0,
                ),
            ])
        )
        # Table cell with 'yes'.
        assert "<td>yes</td>" in out

    def test_table_shows_no_when_no_barge(self):
        out = gen_reports.render_performance_page(
            _payload([_scenario(name="short")])
        )
        assert "<td>no</td>" in out
