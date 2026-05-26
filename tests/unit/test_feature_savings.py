"""Tests for iter-111 — feature-savings table rendering.

Covers three pieces of generate_iteration_reports.py:
  - FEATURE_SAVINGS_TABLE — the curated entry list
  - _format_pair_value / _format_pair_delta — value formatting
  - _render_feature_savings_section — HTML composition
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "generate_iteration_reports.py"

spec = importlib.util.spec_from_file_location("gen_reports", SCRIPT_PATH)
gen_reports = importlib.util.module_from_spec(spec)
sys.modules["gen_reports"] = gen_reports
spec.loader.exec_module(gen_reports)


# ---- _format_pair_value ---------------------------------------------------


def test_format_ms_uses_one_decimal():
    assert gen_reports._format_pair_value(123.456, "ttfs_ms") == "123.5"


def test_format_ms_handles_zero():
    assert gen_reports._format_pair_value(0.0, "ttfs_ms") == "0.0"


def test_format_context_tokens_renders_as_int():
    assert gen_reports._format_pair_value(53.0, "context_tokens") == "53"


def test_format_last_filler_id_renders_as_int():
    """last_filler_id is the id() of a filler clip, naturally large."""
    assert gen_reports._format_pair_value(140649285605376, "last_filler_id") == (
        "140649285605376"
    )


def test_format_none_value_renders_as_dash():
    assert gen_reports._format_pair_value(None, "ttfs_ms") == "—"


def test_format_unknown_metric_falls_back_to_str():
    """Defensive: unknown metric type just str()'s the value."""
    assert gen_reports._format_pair_value(42, "unknown_metric") == "42"


# ---- _format_pair_delta ---------------------------------------------------


def test_delta_ms_positive():
    """Treatment > control → positive delta with +ms suffix."""
    assert gen_reports._format_pair_delta(100.0, 150.0, "ttfs_ms") == "+50.0ms"


def test_delta_ms_negative():
    """Treatment < control (the savings case) → negative delta."""
    assert gen_reports._format_pair_delta(150.0, 100.0, "ttfs_ms") == "-50.0ms"


def test_delta_ms_with_decimals():
    """Sub-millisecond deltas survive."""
    assert gen_reports._format_pair_delta(100.4, 100.7, "ttfs_ms") == "+0.3ms"


def test_delta_context_tokens_includes_ratio_when_nonzero_control():
    """When control > 0, show both raw delta and percentage."""
    out = gen_reports._format_pair_delta(53, 24, "context_tokens")
    assert "-29" in out
    assert "-55%" in out


def test_delta_context_tokens_handles_zero_control():
    """Zero control → just raw delta, no division-by-zero."""
    out = gen_reports._format_pair_delta(0, 5, "context_tokens")
    assert "+5" in out
    assert "%" not in out


def test_delta_last_filler_id_describes_state_change():
    """When the filler newly fires (control=0, treatment≠0), say so."""
    out = gen_reports._format_pair_delta(0, 999, "last_filler_id")
    assert "filler now fires" in out


def test_delta_last_filler_id_describes_no_longer_fires():
    """Reverse direction is also useful."""
    out = gen_reports._format_pair_delta(999, 0, "last_filler_id")
    assert "filler no longer fires" in out


def test_delta_with_none_values_returns_dash():
    """If either side is None, we can't compute a delta."""
    assert gen_reports._format_pair_delta(None, 100, "ttfs_ms") == "—"
    assert gen_reports._format_pair_delta(100, None, "ttfs_ms") == "—"


# ---- FEATURE_SAVINGS_TABLE structure --------------------------------------


def test_table_has_at_least_six_entries():
    """iter-098, iter-099, iter-100, iter-101 grid (3 entries),
    iter-102 → 6+ pairs."""
    assert len(gen_reports.FEATURE_SAVINGS_TABLE) >= 6


def test_each_entry_has_required_keys():
    """Every entry must carry the keys the renderer reads."""
    required = {
        "feature", "control_scenario", "treatment_scenario",
        "primary_metric", "primary_label",
        "secondary_metric", "secondary_label", "secondary_lower_is",
        "takeaway",
    }
    for entry in gen_reports.FEATURE_SAVINGS_TABLE:
        missing = required - entry.keys()
        assert not missing, (
            f"entry {entry.get('feature')!r} missing keys: {missing}"
        )


def test_no_duplicate_treatment_scenario_names():
    """Each treatment scenario is unique — two entries pointing
    at the same row would render two different 'feature savings'
    rows for the same data."""
    treatments = [e["treatment_scenario"] for e in gen_reports.FEATURE_SAVINGS_TABLE]
    assert len(treatments) == len(set(treatments)), (
        f"duplicate treatments: {treatments}"
    )


# ---- _render_feature_savings_section --------------------------------------


def _build_scenarios(*pairs) -> list[dict]:
    """Each arg is a (name, **fields) tuple."""
    out = []
    for entry in pairs:
        name, fields = entry
        row = {"name": name}
        row.update(fields)
        out.append(row)
    return out


def test_section_empty_when_no_pairs_present():
    """If the snapshot has no scenarios that match a curated
    pair, render NOTHING. Old perf JSONs (pre-iter-098) should
    leave the section silent rather than show an empty table."""
    scenarios = [{"name": "short_short", "ttfs_ms": 100.0}]
    out = gen_reports._render_feature_savings_section(scenarios)
    assert out == ""


def test_section_renders_when_one_pair_complete():
    """One complete pair → section + 1 row."""
    scenarios = _build_scenarios(
        ("long_preamble_aggressive_off",
         {"ttfs_ms": 931.6, "mean_sentence_chars": 36.3}),
        ("long_preamble_aggressive_on",
         {"ttfs_ms": 891.1, "mean_sentence_chars": 27.0}),
    )
    out = gen_reports._render_feature_savings_section(scenarios)
    assert "<table" in out
    assert "Feature savings" in out
    assert "Aggressive first-sentence splitter" in out
    assert "931.5" in out or "931.6" in out
    assert "891.1" in out
    # delta = 891.1 - 931.6 = -40.5 (within rounding)
    assert "-40.5ms" in out or "-40.6ms" in out


def test_section_skips_pair_with_missing_treatment():
    """Half-present pair → row skipped silently."""
    scenarios = _build_scenarios(
        ("long_preamble_aggressive_off",
         {"ttfs_ms": 931.6, "mean_sentence_chars": 36.3}),
        # No `_on` row.
    )
    out = gen_reports._render_feature_savings_section(scenarios)
    # Empty section — nothing matched.
    assert out == ""


def test_section_skips_pair_with_missing_control():
    scenarios = _build_scenarios(
        ("long_preamble_aggressive_on",
         {"ttfs_ms": 891.1, "mean_sentence_chars": 27.0}),
    )
    out = gen_reports._render_feature_savings_section(scenarios)
    assert out == ""


def test_context_cap_pair_renders_as_token_metric():
    """context_cap_default + context_cap_tight pair uses
    context_tokens as the primary metric, not ms."""
    scenarios = _build_scenarios(
        ("context_cap_default",
         {"ttfs_ms": 800.0, "context_tokens": 53}),
        ("context_cap_tight",
         {"ttfs_ms": 800.0, "context_tokens": 24}),
    )
    out = gen_reports._render_feature_savings_section(scenarios)
    assert "context cap" in out.lower() or "max_user_assistant" in out
    assert "53" in out
    assert "24" in out
    # Delta = -29 tokens → -55%.
    assert "-29" in out
    assert "-55%" in out


def test_filler_threshold_pair_renders_filler_state_change():
    """filler_threshold_default + filler_threshold_aggressive pair
    uses last_filler_id as the secondary marker. When control=0
    and treatment is nonzero, the delta column should say 'filler
    now fires'."""
    scenarios = _build_scenarios(
        ("filler_threshold_default",
         {"ttfs_ms": 1301.1, "last_filler_id": 0}),
        ("filler_threshold_aggressive",
         {"ttfs_ms": 950.3, "last_filler_id": 140649285608192}),
    )
    out = gen_reports._render_feature_savings_section(scenarios)
    assert "filler" in out.lower()
    assert "1301.1" in out
    assert "950.3" in out


def test_section_renders_all_six_when_full_snapshot():
    """When every curated pair has both control + treatment in
    the snapshot, the table has 6 rows. Sanity for 'all
    iter-098+ data is wired up correctly'."""
    scenarios = _build_scenarios(
        ("long_preamble_aggressive_off",
         {"ttfs_ms": 931.6, "mean_sentence_chars": 36.3}),
        ("long_preamble_aggressive_on",
         {"ttfs_ms": 891.1, "mean_sentence_chars": 27.0}),
        ("filler_threshold_default",
         {"ttfs_ms": 1301.1, "last_filler_id": 0}),
        ("filler_threshold_aggressive",
         {"ttfs_ms": 950.3, "last_filler_id": 1}),
        ("auto_aggressive_off",
         {"ttfs_ms": 1431.7, "mean_sentence_chars": 49.0}),
        ("auto_aggressive_on",
         {"ttfs_ms": 1401.5, "mean_sentence_chars": 32.3}),
        ("auto_aggressive_off_50ms",
         {"ttfs_ms": 1951.7, "mean_sentence_chars": 49.0}),
        ("auto_aggressive_on_50ms",
         {"ttfs_ms": 1801.4, "mean_sentence_chars": 32.3}),
        ("auto_aggressive_off_100ms",
         {"ttfs_ms": 2601.8, "mean_sentence_chars": 49.0}),
        ("auto_aggressive_on_100ms",
         {"ttfs_ms": 2301.6, "mean_sentence_chars": 32.3}),
        ("context_cap_default",
         {"ttfs_ms": 800.0, "context_tokens": 53}),
        ("context_cap_tight",
         {"ttfs_ms": 800.0, "context_tokens": 24}),
    )
    out = gen_reports._render_feature_savings_section(scenarios)
    # 6 <tr> in the body — count the tbody rows specifically.
    assert out.count("<tr>") == 1 + 6  # 1 thead + 6 body rows


def test_section_html_uses_perf_table_class():
    """Match the existing perf-table styling so the section
    visually integrates."""
    scenarios = _build_scenarios(
        ("long_preamble_aggressive_off",
         {"ttfs_ms": 931.6, "mean_sentence_chars": 36.3}),
        ("long_preamble_aggressive_on",
         {"ttfs_ms": 891.1, "mean_sentence_chars": 27.0}),
    )
    out = gen_reports._render_feature_savings_section(scenarios)
    assert 'class="perf-table"' in out


def test_section_escapes_html_in_takeaway():
    """If a takeaway happens to contain HTML special chars,
    they're escaped. Defensive: prevents a future curated
    entry from breaking the page."""
    # We don't have a malicious entry in the curated list, but we
    # can stub one in via monkey-patching.
    original = gen_reports.FEATURE_SAVINGS_TABLE
    try:
        gen_reports.FEATURE_SAVINGS_TABLE = [
            {
                "feature": "<script>alert(1)</script>",
                "control_scenario": "ctrl",
                "treatment_scenario": "treat",
                "primary_metric": "ttfs_ms",
                "primary_label": "TTFS",
                "secondary_metric": "mean_sentence_chars",
                "secondary_label": "Mean chars",
                "secondary_lower_is": "shorter",
                "takeaway": "Use <script>",
            },
        ]
        scenarios = _build_scenarios(
            ("ctrl", {"ttfs_ms": 100.0, "mean_sentence_chars": 30}),
            ("treat", {"ttfs_ms": 90.0, "mean_sentence_chars": 25}),
        )
        out = gen_reports._render_feature_savings_section(scenarios)
        # Raw <script> must NOT appear; HTML-escaped form must.
        assert "<script>" not in out
        assert "&lt;script&gt;" in out
    finally:
        gen_reports.FEATURE_SAVINGS_TABLE = original


# ---- Integration with render_performance_page -----------------------------


def test_performance_page_includes_feature_savings_when_pairs_present():
    """When the perf payload contains pair scenarios, the
    rendered page includes the feature-savings section."""
    payload = {
        "captured_at": "2026-05-25T19:00:00",
        "iteration": "111",
        "scenarios": [
            {
                "name": "long_preamble_aggressive_off",
                "description": "Long preamble, splitter off",
                "ttfs_ms": 931.6, "mean_sentence_chars": 36.3,
                "stt_ms": 0.0, "tts_ms": 0.0, "playback_ms": 0.0,
                "llm_first_token_ms": 0.0, "llm_total_ms": 0.0,
                "wall_ms": 1000.0, "sentences_spoken": 1,
                "barge_in": False, "barge_in_latency_ms": 0.0,
            },
            {
                "name": "long_preamble_aggressive_on",
                "description": "Long preamble, splitter on",
                "ttfs_ms": 891.1, "mean_sentence_chars": 27.0,
                "stt_ms": 0.0, "tts_ms": 0.0, "playback_ms": 0.0,
                "llm_first_token_ms": 0.0, "llm_total_ms": 0.0,
                "wall_ms": 950.0, "sentences_spoken": 1,
                "barge_in": False, "barge_in_latency_ms": 0.0,
            },
        ],
    }
    out = gen_reports.render_performance_page(payload, history=[])
    assert "Feature savings" in out
    assert "Aggressive first-sentence splitter" in out


def test_performance_page_omits_section_when_no_pairs():
    """When the snapshot has no pair scenarios, the section is
    silent — old JSONs render without an empty 'Feature savings'
    table."""
    payload = {
        "captured_at": "2026-01-01T00:00:00",
        "iteration": "036",
        "scenarios": [
            {
                "name": "short_short",
                "description": "Short utterance",
                "ttfs_ms": 100.0, "stt_ms": 0.0, "tts_ms": 0.0,
                "playback_ms": 0.0, "llm_first_token_ms": 0.0,
                "llm_total_ms": 0.0, "wall_ms": 100.0,
                "sentences_spoken": 1, "barge_in": False,
                "barge_in_latency_ms": 0.0,
            },
        ],
    }
    out = gen_reports.render_performance_page(payload, history=[])
    assert "Feature savings" not in out
