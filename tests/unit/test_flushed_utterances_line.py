"""iter-167 — flushed-utterances session-summary line (backlog #9 wiring hop 2).

When the organic ``UtteranceAggregator`` holds a mid-thought fragment ("I was
thinking about the") and the user then trails off into a long inter-turn
silence with no continuation, iter-165's recorder idle timeout + iter-166's
``idle_timed_out`` flag let ``run_session`` notice it, and iter-164's
``decide_silence_flush`` (wired through an injected decider) chooses to FLUSH
the fragment to the engine mid-session rather than leave it held until a new
thought displaces it (iter-162) or shutdown flushes it (iter-160). The flushed
text is recorded on ``state.flushed_utterances``, surfaced here — the
mid-session-idle analog of iter-160's shutdown ``stranded_utterance`` and
iter-162's displaced fragments.

These tests cover the helper in isolation (suppression + single/multi
formatting) and its wiring through ``print_session_summary`` on both the normal
and the no-completed-turns early-return path.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    SessionMeta,
    TurnMetrics,
    _emit_flushed_utterances_line,
    print_session_summary,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _collect():
    lines: list[str] = []
    return lines, lines.append


# ---- helper in isolation: suppression rules --------------------------------


class TestSuppression:
    def test_none_emits_nothing(self):
        lines, emit = _collect()
        _emit_flushed_utterances_line(emit, None)
        assert lines == []

    def test_empty_list_emits_nothing(self):
        lines, emit = _collect()
        _emit_flushed_utterances_line(emit, [])
        assert lines == []

    def test_all_blank_fragments_emit_nothing(self):
        lines, emit = _collect()
        _emit_flushed_utterances_line(emit, ["", "   ", "\t"])
        assert lines == []


# ---- helper in isolation: single-fragment emission -------------------------


class TestSingleFragment:
    def test_fragment_emitted_quoted(self):
        lines, emit = _collect()
        _emit_flushed_utterances_line(emit, ["I was thinking about the"])
        assert len(lines) == 1
        assert "'I was thinking about the'" in lines[0]

    def test_fragment_is_stripped(self):
        lines, emit = _collect()
        _emit_flushed_utterances_line(emit, ["  trailing  "])
        assert "'trailing'" in lines[0]

    def test_line_names_iteration_for_grep(self):
        lines, emit = _collect()
        _emit_flushed_utterances_line(emit, ["hmm"])
        assert "iter-167" in lines[0]

    def test_line_explains_what_happened(self):
        lines, emit = _collect()
        _emit_flushed_utterances_line(emit, ["hmm"])
        assert "idle" in lines[0].lower()
        assert "flushed" in lines[0].lower()


# ---- helper in isolation: multi-fragment emission --------------------------


class TestMultiFragment:
    def test_multiple_fragments_each_on_own_line(self):
        lines, emit = _collect()
        _emit_flushed_utterances_line(emit, ["one", "two", "three"])
        # 1 header + 3 fragment lines.
        assert len(lines) == 4
        assert "3 fragments" in lines[0]
        assert "iter-167" in lines[0]
        assert "'one'" in lines[1]
        assert "'two'" in lines[2]
        assert "'three'" in lines[3]

    def test_blank_fragments_dropped_from_count(self):
        lines, emit = _collect()
        # Two non-blank + a blank → counted as 1 (single-fragment form).
        _emit_flushed_utterances_line(emit, ["only", "   "])
        assert len(lines) == 1
        assert "'only'" in lines[0]
        # Single-fragment form, not the "N fragments" header.
        assert "fragments" not in lines[0]


# ---- integration through print_session_summary -----------------------------


def _summary(metrics_list, *, flushed):
    buf = io.StringIO()
    print_session_summary(
        metrics_list,
        {"model": "test-model"},
        file=buf,
        meta=SessionMeta(flushed_utterances=flushed),
    )
    return _strip_ansi(buf.getvalue())


def _one_metric() -> TurnMetrics:
    return TurnMetrics(
        stt_time=0.05,
        llm_first_token=0.1,
        tts_time=0.2,
        ttfs=0.3,
        model="test-model",
    )


class TestSummaryIntegration:
    def test_flushed_surfaced_with_completed_turns(self):
        out = _summary([_one_metric()], flushed=["I was thinking about the"])
        assert "I was thinking about the" in out
        assert "Flushed uttr." in out

    def test_no_flushed_line_when_empty(self):
        out = _summary([_one_metric()], flushed=[])
        assert "Flushed uttr." not in out

    def test_flushed_surfaced_on_no_completed_turns_path(self):
        # The early-return path (zero metrics) must still surface fragments —
        # a session can flush a fragment yet land no completed turns.
        out = _summary([], flushed=["wait actually"])
        assert "no completed turns" in out
        assert "wait actually" in out

    def test_no_completed_turns_no_flushed_is_clean(self):
        out = _summary([], flushed=[])
        assert "no completed turns" in out
        assert "Flushed uttr." not in out

    def test_multiple_flushed_all_surfaced(self):
        out = _summary(
            [_one_metric()], flushed=["frag one", "frag two"]
        )
        assert "frag one" in out
        assert "frag two" in out
        assert "2 fragments" in out
