"""Tests for iter-039 — per-iteration perf snapshots + time-series.

Covers:
  - _resolve_iter_number from tests/performance/.
  - _load_perf_history walks perf-iter-NNN.json, sorts by iteration.
  - _svg_multi_line_chart renders one polyline per series.
  - render_performance_page splits into "Latest snapshot" + "Across
    iterations" sections.
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


# ---- _load_perf_history ----------------------------------------------------


class TestLoadPerfHistory:
    def test_no_dir_returns_empty(self, tmp_path):
        assert gen_reports._load_perf_history(tmp_path / "missing") == []

    def test_empty_dir_returns_empty(self, tmp_path):
        assert gen_reports._load_perf_history(tmp_path) == []

    def test_loads_one_file(self, tmp_path):
        payload = {"iteration": "001", "scenarios": []}
        (tmp_path / "perf-iter-001.json").write_text(json.dumps(payload))
        out = gen_reports._load_perf_history(tmp_path)
        assert len(out) == 1
        assert out[0]["iteration"] == "001"

    def test_sorts_by_iteration(self, tmp_path):
        for n in ["010", "002", "005"]:
            (tmp_path / f"perf-iter-{n}.json").write_text(
                json.dumps({"iteration": n, "scenarios": []})
            )
        out = gen_reports._load_perf_history(tmp_path)
        assert [p["iteration"] for p in out] == ["002", "005", "010"]

    def test_skips_malformed_json(self, tmp_path):
        (tmp_path / "perf-iter-001.json").write_text("not json {")
        (tmp_path / "perf-iter-002.json").write_text(
            json.dumps({"iteration": "002", "scenarios": []})
        )
        out = gen_reports._load_perf_history(tmp_path)
        assert len(out) == 1
        assert out[0]["iteration"] == "002"

    def test_ignores_non_perf_iter_files(self, tmp_path):
        # Other files in the directory shouldn't be picked up.
        (tmp_path / "perf-iter-001.json").write_text(
            json.dumps({"iteration": "001", "scenarios": []})
        )
        (tmp_path / "perf-results.json").write_text(
            json.dumps({"scenarios": []})
        )
        (tmp_path / "iter-001.html").write_text("<html/>")
        out = gen_reports._load_perf_history(tmp_path)
        assert len(out) == 1

    def test_uses_filename_iteration_when_json_missing_field(self, tmp_path):
        (tmp_path / "perf-iter-042.json").write_text(
            json.dumps({"scenarios": []})  # no 'iteration' key
        )
        out = gen_reports._load_perf_history(tmp_path)
        assert out[0]["iteration"] == "042"


# ---- _svg_multi_line_chart -------------------------------------------------


class TestMultiLineChart:
    def test_empty_dict_renders_placeholder(self):
        out = gen_reports._svg_multi_line_chart(
            {}, title="Empty", y_label="ms"
        )
        assert "no data" in out
        assert "<svg" not in out

    def test_all_empty_series_renders_placeholder(self):
        out = gen_reports._svg_multi_line_chart(
            {"a": [], "b": []}, title="Empty", y_label="ms"
        )
        assert "no data" in out

    def test_single_point_per_series(self):
        out = gen_reports._svg_multi_line_chart(
            {"scn1": [(1.0, 100.0)], "scn2": [(1.0, 200.0)]},
            title="Single", y_label="ms",
        )
        assert "<svg" in out
        # Single points → circles, no path.
        assert out.count("<circle") == 2

    def test_multi_point_renders_path(self):
        out = gen_reports._svg_multi_line_chart(
            {"scn1": [(1.0, 100.0), (2.0, 110.0), (3.0, 90.0)]},
            title="Multi", y_label="ms",
        )
        assert "<path" in out
        # Three dots for the three points.
        assert out.count("<circle") == 3

    def test_legend_includes_all_labels(self):
        out = gen_reports._svg_multi_line_chart(
            {"alpha": [(1, 1)], "beta": [(1, 2)], "gamma": [(1, 3)]},
            title="L", y_label="ms",
        )
        assert ">alpha<" in out
        assert ">beta<" in out
        assert ">gamma<" in out

    def test_title_html_escaped(self):
        out = gen_reports._svg_multi_line_chart(
            {"a": [(1, 1)]}, title="<script>", y_label="ms",
        )
        assert "&lt;script&gt;" in out


# ---- render_performance_page (history portion) -----------------------------


def _scenario(name="s1", ttfs=100.0, wall=200.0, tts=20.0, stt=10.0):
    return {
        "name": name,
        "description": f"{name} desc",
        "ttfs_ms": ttfs,
        "stt_ms": stt,
        "tts_ms": tts,
        "playback_ms": 0.0,
        "llm_first_token_ms": 0.0,
        "llm_total_ms": 0.0,
        "speech_duration_ms": 1000.0,
        "sentences_spoken": 1,
        "wall_ms": wall,
        "barge_in": False,
    }


def _snapshot(iteration: str, scenarios):
    return {
        "captured_at": f"2026-05-24T{iteration}:00:00",
        "iteration": iteration,
        "scenarios": scenarios,
    }


class TestPerformancePageHistory:
    def test_no_history_no_section(self):
        snap = _snapshot("038", [_scenario()])
        out = gen_reports.render_performance_page(snap, [])
        assert "Across iterations" not in out

    def test_single_iteration_shows_soft_note(self):
        snap = _snapshot("038", [_scenario()])
        history = [snap]
        out = gen_reports.render_performance_page(snap, history)
        assert "Across iterations" in out
        assert "Only one iteration captured so far" in out
        # No multi-line chart.
        # (Latest snapshot has 5 charts already; the history section
        # should NOT add more when only one iter exists.)
        # We can spot-check: count h3 entries in history block = 0.

    def test_multi_iteration_renders_charts(self):
        history = [
            _snapshot("037", [_scenario("a", ttfs=120), _scenario("b", ttfs=80)]),
            _snapshot("038", [_scenario("a", ttfs=100), _scenario("b", ttfs=90)]),
            _snapshot("039", [_scenario("a", ttfs=90),  _scenario("b", ttfs=95)]),
        ]
        out = gen_reports.render_performance_page(history[-1], history)
        assert "Across iterations" in out
        # Both scenarios show up in the history charts (each as a
        # legend entry).
        assert ">a<" in out
        assert ">b<" in out
        # 5 latest-snapshot charts + 4 history charts = 9 SVGs.
        assert out.count("<svg") == 9

    def test_history_section_includes_iteration_count(self):
        history = [_snapshot(str(i).zfill(3), [_scenario()]) for i in range(35, 40)]
        out = gen_reports.render_performance_page(history[-1], history)
        # 5 captured iterations.
        assert "5" in out and "captured iterations" in out

    def test_non_numeric_iteration_skipped(self):
        # An accidentally-corrupt iteration key like "foo" would
        # break int() — should be skipped without crashing.
        history = [
            _snapshot("037", [_scenario("a", ttfs=100)]),
            {"iteration": "foo", "scenarios": [_scenario("a", ttfs=999)]},
            _snapshot("039", [_scenario("a", ttfs=100)]),
        ]
        out = gen_reports.render_performance_page(history[-1], history)
        # Doesn't crash; the bad row is skipped from the time-series.
        assert "Across iterations" in out


# ---- iter-number resolution from tests/performance/ ------------------------


class TestResolveIterNumber:
    """The perf test module computes its own iteration number from
    ITERATION_LOG.md. Re-import the module fresh per test (since
    LOG_PATH is module-level) and validate the helper.
    """

    def _load_module(self, root: Path):
        # Fresh importlib so each test gets a clean ROOT.
        import importlib.util
        path = root / "tests" / "performance" / "test_pipeline_perf.py"
        if not path.exists():
            pytest.skip("perf test module not present")
        # Use a unique name; register in sys.modules BEFORE exec
        # so @dataclass can look up cls.__module__ during class
        # construction (otherwise dataclasses raises AttributeError
        # on a None entry).
        mod_name = "perf_mod_for_test"
        s = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(s)
        sys.modules[mod_name] = mod
        try:
            s.loader.exec_module(mod)
        finally:
            sys.modules.pop(mod_name, None)
        return mod

    def test_real_repo_resolves_to_a_3digit_string(self):
        mod = self._load_module(ROOT)
        n = mod._resolve_iter_number()
        assert isinstance(n, str)
        assert len(n) == 3
        assert n.isdigit()
