"""Tests for iter-036 — performance report generation pieces.

Covers the new parts of generate_iteration_reports.py:
  - _svg_horizontal_bars helper.
  - _load_perf_results.
  - render_performance_page composition (with + without data).
  - Performance link wired into iter pages, index, and testing page.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "generate_iteration_reports.py"

spec = importlib.util.spec_from_file_location("gen_reports", SCRIPT_PATH)
gen_reports = importlib.util.module_from_spec(spec)
sys.modules["gen_reports"] = gen_reports
spec.loader.exec_module(gen_reports)


# ---- Horizontal bar chart ---------------------------------------------------


class TestHorizontalBars:
    def test_empty_renders_placeholder(self):
        out = gen_reports._svg_horizontal_bars(
            [], title="Empty", x_label="ms"
        )
        assert "no data" in out
        assert "<svg" not in out

    def test_single_row(self):
        out = gen_reports._svg_horizontal_bars(
            [("scenario_a", 42.0)], title="One", x_label="ms"
        )
        assert "<svg" in out
        assert "scenario_a" in out
        assert "42ms" in out

    def test_multiple_rows(self):
        out = gen_reports._svg_horizontal_bars(
            [("a", 10.0), ("b", 50.0), ("c", 30.0)],
            title="Multi", x_label="ms",
        )
        # One <rect> per row.
        assert out.count("<rect") == 3
        # All labels emitted.
        assert "a" in out and "b" in out and "c" in out

    def test_label_html_escaped(self):
        out = gen_reports._svg_horizontal_bars(
            [("<script>", 1.0)], title="Esc", x_label="ms"
        )
        assert "&lt;script&gt;" in out
        # Tag never appears as raw content (the `>` from
        # `<svg viewBox=...>` etc are fine — check only the label
        # area). A simple check: no '>scenario_with_<script>' pattern.
        assert "<script>" not in out


# ---- Loader -----------------------------------------------------------------


class TestLoadPerfResults:
    def test_missing_file_returns_none(self, tmp_path):
        out = gen_reports._load_perf_results(tmp_path / "missing.json")
        assert out is None

    def test_valid_json_loads(self, tmp_path):
        path = tmp_path / "perf.json"
        payload = {"captured_at": "2026-05-24", "scenarios": []}
        path.write_text(json.dumps(payload))
        out = gen_reports._load_perf_results(path)
        assert out == payload

    def test_invalid_json_returns_none(self, tmp_path):
        path = tmp_path / "perf.json"
        path.write_text("not json {")
        out = gen_reports._load_perf_results(path)
        assert out is None


# ---- Performance page composition -------------------------------------------


def _payload(scenarios):
    return {"captured_at": "2026-05-24T12:00:00", "scenarios": scenarios}


def _scenario(**overrides):
    base = {
        "name": "default",
        "description": "default scenario",
        "ttfs_ms": 100.0,
        "stt_ms": 50.0,
        "tts_ms": 30.0,
        "playback_ms": 25.0,
        "llm_first_token_ms": 80.0,
        "llm_total_ms": 200.0,
        "speech_duration_ms": 1000.0,
        "sentences_spoken": 1,
        "wall_ms": 250.0,
        "barge_in": False,
    }
    base.update(overrides)
    return base


class TestRenderPerformancePage:
    def test_none_payload_renders_placeholder(self):
        out = gen_reports.render_performance_page(None)
        # Has page chrome.
        assert "<h1>Performance</h1>" in out
        # Tells the user how to populate.
        assert "tests/performance/" in out
        # No charts.
        assert "<svg" not in out

    def test_empty_scenarios_renders_placeholder(self):
        out = gen_reports.render_performance_page(_payload([]))
        assert "<h1>Performance</h1>" in out
        assert "<svg" not in out

    def test_with_scenarios_renders_five_charts(self):
        out = gen_reports.render_performance_page(_payload([
            _scenario(name="s1"),
            _scenario(name="s2", ttfs_ms=200.0),
        ]))
        # TTFS, STT, TTS, LLM 1st token, wall — five charts.
        assert out.count("<svg") == 5
        # Both scenario names appear in label position.
        assert "s1" in out and "s2" in out

    def test_scenario_table_rendered(self):
        out = gen_reports.render_performance_page(_payload([
            _scenario(name="a", description="alpha desc", sentences_spoken=3),
            _scenario(name="b", description="beta desc", barge_in=True),
        ]))
        assert "<table" in out
        assert "alpha desc" in out
        assert "beta desc" in out
        # Barge-in column reflects the flag.
        assert "yes" in out
        assert "no" in out

    def test_navigation_links_index_and_testing(self):
        out = gen_reports.render_performance_page(_payload([_scenario()]))
        assert 'href="index.html"' in out
        assert 'href="testing.html"' in out

    def test_captured_at_displayed(self):
        out = gen_reports.render_performance_page(_payload([_scenario()]))
        assert "2026-05-24T12:00:00" in out


# ---- Nav link wired into siblings -------------------------------------------


def _make_iter(**kw):
    defaults = dict(
        number="001", title="t", branch="b", commit="c",
        date="2026-01-01", body_md="body", tests_added=5,
        total_tests=100, test_runtime_s=10.0,
        next_id="", prev_id="",
    )
    defaults.update(kw)
    return gen_reports.Iteration(**defaults)


class TestPerformanceLinkWired:
    def test_iter_page_includes_performance_nav(self):
        out = gen_reports.render_iteration(_make_iter(number="042"))
        # Top nav + bottom nav, so >=2 occurrences.
        assert out.count('href="performance.html"') >= 2

    def test_index_includes_performance_nav(self):
        out = gen_reports.render_index([_make_iter()])
        assert 'href="performance.html"' in out

    def test_testing_page_links_to_performance(self, tmp_path):
        out = gen_reports.render_testing_page([_make_iter()], tmp_path)
        assert 'href="performance.html"' in out
