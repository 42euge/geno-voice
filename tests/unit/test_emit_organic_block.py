"""Tests for iter-154 — organic-turn-taking naturalness metrics
(backlog #8 in docs/research/organic-turn-taking.md).

Covers the per-turn TurnMetrics fields (false_endpoint /
continuers_detected), the OrganicStats dataclass, the
_emit_organic_block session-summary helper, and the
print_session_summary wiring — including the central guarantee that a
half-duplex session (both counters zero) prints byte-for-byte the same
summary it did before iter-154.

Mirrors iter-090's _emit_barge_block test pattern: extract a multi-line
session-summary block into a helper, test it directly with synthetic
inputs.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    OrganicStats,
    TurnMetrics,
    _emit_organic_block,
    print_session_summary,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _capture():
    lines: list[str] = []

    def emit(line: str = "") -> None:
        lines.append(_strip_ansi(line))

    return emit, lines


# ---- TurnMetrics field defaults ---------------------------------


class TestTurnMetricsDefaults:
    def test_false_endpoint_defaults_false(self):
        assert TurnMetrics().false_endpoint is False

    def test_continuers_detected_defaults_zero(self):
        assert TurnMetrics().continuers_detected == 0


# ---- OrganicStats defaults --------------------------------------


class TestOrganicStatsDefaults:
    def test_all_zero(self):
        s = OrganicStats()
        assert s.false_endpoints == 0
        assert s.continuers_total == 0
        assert s.n == 0


# ---- No-data path (the half-duplex default) ---------------------


class TestNoData:
    def test_both_zero_omits_everything(self):
        # The whole point: half-duplex turns populate neither field,
        # so the block contributes nothing to the summary.
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats(n=10))
        assert lines == []

    def test_both_zero_with_no_turns_omits_everything(self):
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats())
        assert lines == []


# ---- False endpoints --------------------------------------------


class TestFalseEndpoints:
    def test_emits_header_and_rate(self):
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats(false_endpoints=1, n=10))
        assert any("Organic turn-taking:" in ln for ln in lines)
        assert any(
            "False endpoints:  1/10 turns (10%)" in ln for ln in lines
        )

    def test_low_rate_omits_suggestion(self):
        # 10% is below the 20% threshold — no "too eager" suggestion.
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats(false_endpoints=1, n=10))
        assert not any("too eager" in ln for ln in lines)

    def test_high_rate_appends_suggestion(self):
        # 3/10 = 30% > 20% → suggestion fires.
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats(false_endpoints=3, n=10))
        assert any(
            "EOU too eager; raise silence_duration" in ln for ln in lines
        )

    def test_exactly_twenty_percent_omits_suggestion(self):
        # 2/10 = 20%, NOT > 20% — boundary is exclusive.
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats(false_endpoints=2, n=10))
        assert any(
            "False endpoints:  2/10 turns (20%)" in ln for ln in lines
        )
        assert not any("too eager" in ln for ln in lines)

    def test_just_above_twenty_percent_appends_suggestion(self):
        # 3/10 = 30% > 20%.
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats(false_endpoints=3, n=10))
        assert any("too eager" in ln for ln in lines)

    def test_unknown_n_still_shows_count(self):
        # n == 0 but false_endpoints > 0: keep the count visible
        # rather than dropping it (no division-by-zero, no rate).
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats(false_endpoints=2, n=0))
        assert any("False endpoints:  2" in ln for ln in lines)
        assert not any("turns" in ln for ln in lines)


# ---- Continuers --------------------------------------------------


class TestContinuers:
    def test_emits_continuers_held(self):
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats(continuers_total=4, n=10))
        assert any("Organic turn-taking:" in ln for ln in lines)
        assert any(
            "Continuers held:  4 (backchannels kept the floor)" in ln
            for ln in lines
        )

    def test_continuers_alone_does_not_emit_false_endpoint_line(self):
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats(continuers_total=4, n=10))
        assert not any("False endpoints" in ln for ln in lines)


# ---- Both signals together --------------------------------------


class TestBothSignals:
    def test_emits_both_lines_under_one_header(self):
        emit, lines = _capture()
        _emit_organic_block(
            emit,
            OrganicStats(false_endpoints=1, continuers_total=3, n=10),
        )
        headers = [ln for ln in lines if "Organic turn-taking:" in ln]
        assert len(headers) == 1
        assert any("False endpoints:  1/10 turns" in ln for ln in lines)
        assert any("Continuers held:  3" in ln for ln in lines)


# ---- print_session_summary wiring -------------------------------


def _summary(metrics):
    buf = io.StringIO()
    print_session_summary(metrics, {"model": "test"}, file=buf)
    return _strip_ansi(buf.getvalue())


class TestSummaryWiring:
    def test_half_duplex_summary_has_no_organic_block(self):
        # Turns that never touch the organic fields → no organic block.
        metrics = [TurnMetrics(ttfs=0.5, transcript="hi", response="yo")]
        out = _summary(metrics)
        assert "Organic turn-taking" not in out
        assert "False endpoints" not in out
        assert "Continuers held" not in out

    def test_false_endpoint_turn_surfaces_in_summary(self):
        metrics = [
            TurnMetrics(ttfs=0.5, false_endpoint=True),
            TurnMetrics(ttfs=0.5),
        ]
        out = _summary(metrics)
        assert "Organic turn-taking:" in out
        # 1 of 2 turns = 50% > 20% → suggestion present.
        assert "False endpoints:  1/2 turns (50%)" in out
        assert "EOU too eager" in out

    def test_continuers_surface_in_summary(self):
        metrics = [
            TurnMetrics(ttfs=0.5, continuers_detected=2),
            TurnMetrics(ttfs=0.5, continuers_detected=1),
        ]
        out = _summary(metrics)
        assert "Organic turn-taking:" in out
        # Summed across turns: 2 + 1 = 3.
        assert "Continuers held:  3" in out

    def test_per_turn_print_emits_false_endpoint(self, capsys):
        TurnMetrics(ttfs=0.5, false_endpoint=True).print(1)
        out = _strip_ansi(capsys.readouterr().out)
        assert "False endpoint: yes" in out

    def test_per_turn_print_emits_continuers(self, capsys):
        TurnMetrics(ttfs=0.5, continuers_detected=2).print(1)
        out = _strip_ansi(capsys.readouterr().out)
        assert "Continuers:" in out
        assert "2" in out

    def test_per_turn_print_clean_turn_has_no_organic_lines(self, capsys):
        TurnMetrics(ttfs=0.5, transcript="hi").print(1)
        out = _strip_ansi(capsys.readouterr().out)
        assert "False endpoint" not in out
        assert "Continuers:" not in out
