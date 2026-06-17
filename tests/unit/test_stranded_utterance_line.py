"""iter-160 — stranded-utterance session-summary line (backlog #9 flush).

The organic ``UtteranceAggregator`` (iter-156/158) holds a mid-thought
utterance, waiting for a quick continuation to merge on. Mid-session that
pending always resolves: the next utterance's measured gap forces a NEW release
inside ``offer``. The one case ``offer`` can never reach is *shutdown* — the
user trailed off after a fragment and quit. iter-160 flushes the aggregator on
exit (in ``run_session``) and surfaces the released text in the session summary
via ``_emit_stranded_utterance_line`` so the dropped fragment is visible rather
than silently lost.

These tests cover the helper in isolation (suppression rules + formatting) and
its wiring through ``print_session_summary`` on both the normal and the
no-completed-turns early-return path.
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
    _emit_stranded_utterance_line,
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
        _emit_stranded_utterance_line(emit, None)
        assert lines == []

    def test_empty_string_emits_nothing(self):
        lines, emit = _collect()
        _emit_stranded_utterance_line(emit, "")
        assert lines == []

    def test_whitespace_only_emits_nothing(self):
        lines, emit = _collect()
        _emit_stranded_utterance_line(emit, "   \t  ")
        assert lines == []


# ---- helper in isolation: emission + formatting ----------------------------


class TestEmission:
    def test_fragment_is_emitted_quoted(self):
        lines, emit = _collect()
        _emit_stranded_utterance_line(emit, "I was going to say")
        assert len(lines) == 1
        assert "'I was going to say'" in lines[0]

    def test_fragment_is_stripped(self):
        lines, emit = _collect()
        _emit_stranded_utterance_line(emit, "  trailing thought  ")
        assert "'trailing thought'" in lines[0]

    def test_line_names_iteration_for_grep(self):
        lines, emit = _collect()
        _emit_stranded_utterance_line(emit, "hmm")
        assert "iter-160" in lines[0]

    def test_line_explains_what_happened(self):
        lines, emit = _collect()
        _emit_stranded_utterance_line(emit, "hmm")
        assert "mid-thought" in lines[0].lower()


# ---- integration through print_session_summary -----------------------------


def _summary(metrics_list, *, stranded):
    buf = io.StringIO()
    print_session_summary(
        metrics_list,
        {"model": "test-model"},
        file=buf,
        meta=SessionMeta(stranded_utterance=stranded),
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
    def test_stranded_surfaced_with_completed_turns(self):
        out = _summary([_one_metric()], stranded="I think maybe")
        assert "I think maybe" in out
        assert "Stranded uttr." in out

    def test_no_stranded_line_when_none(self):
        out = _summary([_one_metric()], stranded=None)
        assert "Stranded uttr." not in out

    def test_stranded_surfaced_on_no_completed_turns_path(self):
        # The early-return path (zero metrics) must still surface a
        # fragment — the user spoke one unfinished utterance then quit.
        out = _summary([], stranded="wait actually")
        assert "no completed turns" in out
        assert "wait actually" in out

    def test_no_completed_turns_no_stranded_is_clean(self):
        out = _summary([], stranded=None)
        assert "no completed turns" in out
        assert "Stranded uttr." not in out
