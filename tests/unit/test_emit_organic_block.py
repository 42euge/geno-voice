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
    SessionMeta,
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

    def test_merge_capped_defaults_false(self):
        assert TurnMetrics().merge_capped is False


# ---- OrganicStats defaults --------------------------------------


class TestOrganicStatsDefaults:
    def test_all_zero(self):
        s = OrganicStats()
        assert s.false_endpoints == 0
        assert s.continuers_total == 0
        assert s.n == 0
        assert s.utterances_held == 0
        assert s.merges_capped == 0
        assert s.backchannels_emitted == 0


# ---- Utterances held (iter-161) ---------------------------------


class TestUtterancesHeld:
    def test_held_alone_emits_block_and_line(self):
        # A session that only held utterances (no false endpoints, no
        # continuers) still surfaces the held count under the header.
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats(utterances_held=3, n=5))
        assert any("Organic turn-taking:" in ln for ln in lines)
        assert any(
            "Utterances held:  3 (mid-thought, buffered for merge "
            "— not VAD false triggers)" in ln
            for ln in lines
        )

    def test_held_zero_omits_line(self):
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats(false_endpoints=1, n=5))
        assert not any("Utterances held" in ln for ln in lines)

    def test_held_alone_does_not_emit_false_endpoint_line(self):
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats(utterances_held=2))
        assert not any("False endpoints" in ln for ln in lines)
        assert not any("Continuers held" in ln for ln in lines)

    def test_held_with_other_signals_under_one_header(self):
        emit, lines = _capture()
        _emit_organic_block(
            emit,
            OrganicStats(
                false_endpoints=1, continuers_total=2, utterances_held=3, n=10
            ),
        )
        headers = [ln for ln in lines if "Organic turn-taking:" in ln]
        assert len(headers) == 1
        assert any("False endpoints:  1/10 turns" in ln for ln in lines)
        assert any("Continuers held:  2" in ln for ln in lines)
        assert any("Utterances held:  3" in ln for ln in lines)


# ---- Merges capped (iter-163) -----------------------------------


class TestMergesCapped:
    def test_capped_alone_emits_block_and_line(self):
        # A session that only hit the merge cap still surfaces it under the
        # header (the cap firing must never be silent).
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats(merges_capped=2, n=5))
        assert any("Organic turn-taking:" in ln for ln in lines)
        assert any(
            "Merges capped:    2 (hit max_merge_depth still mid-thought "
            "— retune merge window/EOU; iter-157 cap)" in ln
            for ln in lines
        )

    def test_capped_zero_omits_line(self):
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats(false_endpoints=1, n=5))
        assert not any("Merges capped" in ln for ln in lines)

    def test_capped_alone_does_not_emit_other_lines(self):
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats(merges_capped=1))
        assert not any("False endpoints" in ln for ln in lines)
        assert not any("Continuers held" in ln for ln in lines)
        assert not any("Utterances held" in ln for ln in lines)

    def test_capped_with_other_signals_under_one_header(self):
        emit, lines = _capture()
        _emit_organic_block(
            emit,
            OrganicStats(
                false_endpoints=2, continuers_total=1, utterances_held=3,
                merges_capped=1, n=10,
            ),
        )
        headers = [ln for ln in lines if "Organic turn-taking:" in ln]
        assert len(headers) == 1
        assert any("False endpoints:  2/10 turns" in ln for ln in lines)
        assert any("Continuers held:  1" in ln for ln in lines)
        assert any("Utterances held:  3" in ln for ln in lines)
        assert any("Merges capped:    1" in ln for ln in lines)


# ---- Backchannels emitted (iter-175) ----------------------------


class TestBackchannelsEmitted:
    def test_emitted_alone_emits_block_and_line(self):
        # A session that only emitted agent backchannels (no false
        # endpoints, no continuers, no holds/caps) still surfaces the
        # count under the header.
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats(backchannels_emitted=4, n=5))
        assert any("Organic turn-taking:" in ln for ln in lines)
        assert any(
            "Backchannels:     4 (agent mid-speech cues — active listening)"
            in ln
            for ln in lines
        )

    def test_emitted_zero_omits_line(self):
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats(false_endpoints=1, n=5))
        assert not any("Backchannels:" in ln for ln in lines)

    def test_emitted_alone_does_not_emit_other_lines(self):
        emit, lines = _capture()
        _emit_organic_block(emit, OrganicStats(backchannels_emitted=2))
        assert not any("False endpoints" in ln for ln in lines)
        assert not any("Continuers held" in ln for ln in lines)
        assert not any("Utterances held" in ln for ln in lines)
        assert not any("Merges capped" in ln for ln in lines)

    def test_emitted_with_other_signals_under_one_header(self):
        emit, lines = _capture()
        _emit_organic_block(
            emit,
            OrganicStats(
                false_endpoints=1,
                continuers_total=2,
                utterances_held=3,
                merges_capped=1,
                backchannels_emitted=5,
                n=10,
            ),
        )
        headers = [ln for ln in lines if "Organic turn-taking:" in ln]
        assert len(headers) == 1
        assert any("False endpoints:  1/10 turns" in ln for ln in lines)
        assert any("Continuers held:  2" in ln for ln in lines)
        assert any("Utterances held:  3" in ln for ln in lines)
        assert any("Merges capped:    1" in ln for ln in lines)
        assert any("Backchannels:     5" in ln for ln in lines)


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

    def test_merge_capped_turn_surfaces_in_summary(self):
        # A capped turn (always also false_endpoint) surfaces BOTH the
        # false-endpoint rate and the distinct "Merges capped" line.
        metrics = [
            TurnMetrics(ttfs=0.5, false_endpoint=True, merge_capped=True),
            TurnMetrics(ttfs=0.5),
        ]
        out = _summary(metrics)
        assert "Organic turn-taking:" in out
        assert "Merges capped:    1" in out
        assert "False endpoints:  1/2 turns" in out

    def test_no_capped_turn_omits_capped_line(self):
        # A false endpoint that was a clean repair (not capped) shows the
        # false-endpoint line but NOT the merges-capped line.
        metrics = [
            TurnMetrics(ttfs=0.5, false_endpoint=True, merge_capped=False),
            TurnMetrics(ttfs=0.5),
        ]
        out = _summary(metrics)
        assert "False endpoints:  1/2 turns" in out
        assert "Merges capped" not in out


# ---- print_session_summary wiring of utterances_held (iter-161) -


def _summary_meta(metrics, meta):
    buf = io.StringIO()
    print_session_summary(metrics, {"model": "test"}, file=buf, meta=meta)
    return _strip_ansi(buf.getvalue())


class TestHeldSummaryWiring:
    def test_held_via_meta_surfaces_in_summary(self):
        # A completed turn plus a SessionMeta carrying utterances_held
        # surfaces the held line under the organic block.
        out = _summary_meta(
            [TurnMetrics(ttfs=0.5, transcript="hi")],
            SessionMeta(utterances_held=2),
        )
        assert "Organic turn-taking:" in out
        assert "Utterances held:  2" in out
        assert "not VAD false triggers" in out

    def test_held_zero_via_meta_omits_block(self):
        # Default meta (held=0) on a plain turn → no organic block.
        out = _summary_meta(
            [TurnMetrics(ttfs=0.5, transcript="hi")],
            SessionMeta(),
        )
        assert "Utterances held" not in out
        assert "Organic turn-taking" not in out

    def test_held_surfaces_on_zero_turn_early_return(self):
        # A session that held a fragment but completed zero turns still
        # shows the held count on the no-completed-turns early-return path.
        out = _summary_meta([], SessionMeta(utterances_held=1))
        assert "Session ended (no completed turns)" in out
        assert "Utterances held:  1" in out

    def test_held_not_counted_as_vad_false_trigger(self):
        # The headline fix: a held utterance must NOT appear in the VAD
        # false-trigger line. With held=2 and false_triggers=0, there is
        # no VAD false-trig line, but the held line is present.
        out = _summary_meta(
            [TurnMetrics(ttfs=0.5, transcript="hi")],
            SessionMeta(utterances_held=2, false_triggers=0),
        )
        assert "VAD false-trig" not in out
        assert "Utterances held:  2" in out


# ---- print_session_summary wiring of backchannels (iter-175) ----


class TestBackchannelsSummaryWiring:
    def test_default_meta_omits_backchannel_line(self):
        # SessionMeta() defaults backchannels_emitted=0 → no line, no block.
        out = _summary_meta(
            [TurnMetrics(ttfs=0.5, transcript="hi")],
            SessionMeta(),
        )
        assert "Backchannels:" not in out
        assert "Organic turn-taking" not in out

    def test_backchannels_via_meta_surfaces_in_summary(self):
        out = _summary_meta(
            [TurnMetrics(ttfs=0.5, transcript="hi")],
            SessionMeta(backchannels_emitted=3),
        )
        assert "Organic turn-taking:" in out
        assert "Backchannels:     3 (agent mid-speech cues — active listening)" in out

    def test_backchannels_surface_on_zero_turn_early_return(self):
        # A session that emitted agent backchannels but completed zero
        # turns still shows the count on the no-completed-turns path.
        out = _summary_meta([], SessionMeta(backchannels_emitted=2))
        assert "Session ended (no completed turns)" in out
        assert "Backchannels:     2" in out

    def test_backchannels_alongside_continuers_both_surface(self):
        # The agent-side count and the user-side continuer count are
        # distinct lines under one header (mirror signals).
        out = _summary_meta(
            [TurnMetrics(ttfs=0.5, continuers_detected=4)],
            SessionMeta(backchannels_emitted=3),
        )
        headers = [ln for ln in out.splitlines() if "Organic turn-taking:" in ln]
        assert len(headers) == 1
        assert "Continuers held:  4" in out
        assert "Backchannels:     3" in out
