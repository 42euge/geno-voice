"""iter-162 — displaced-utterances session-summary line (backlog #9 fix).

When the organic ``UtteranceAggregator`` holds a mid-thought fragment ("I was
thinking about the") and the next thing it hears is NOT a quick continuation
but a long silence followed by a genuinely new utterance ("What time is it?"),
the buffer releases the abandoned fragment as its own ``NEW`` turn *and* the
new utterance in a single ``offer`` — two distinct turns. iter-159's
``resolve_turn`` used to space-glue them into one garbled LLM input. iter-162
responds to the new utterance only and routes the abandoned fragment(s) to
``state.utterances_displaced``, surfaced here — the mid-session analog of
iter-160's shutdown ``stranded_utterance``.

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
    _emit_displaced_utterances_line,
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
        _emit_displaced_utterances_line(emit, None)
        assert lines == []

    def test_empty_list_emits_nothing(self):
        lines, emit = _collect()
        _emit_displaced_utterances_line(emit, [])
        assert lines == []

    def test_all_blank_fragments_emit_nothing(self):
        lines, emit = _collect()
        _emit_displaced_utterances_line(emit, ["", "   ", "\t"])
        assert lines == []


# ---- helper in isolation: single-fragment emission -------------------------


class TestSingleFragment:
    def test_fragment_emitted_quoted(self):
        lines, emit = _collect()
        _emit_displaced_utterances_line(emit, ["I was thinking about the"])
        assert len(lines) == 1
        assert "'I was thinking about the'" in lines[0]

    def test_fragment_is_stripped(self):
        lines, emit = _collect()
        _emit_displaced_utterances_line(emit, ["  trailing  "])
        assert "'trailing'" in lines[0]

    def test_line_names_iteration_for_grep(self):
        lines, emit = _collect()
        _emit_displaced_utterances_line(emit, ["hmm"])
        assert "iter-162" in lines[0]

    def test_line_explains_what_happened(self):
        lines, emit = _collect()
        _emit_displaced_utterances_line(emit, ["hmm"])
        assert "mid-thought" in lines[0].lower()
        assert "abandoned" in lines[0].lower()


# ---- helper in isolation: multi-fragment emission --------------------------


class TestMultiFragment:
    def test_multiple_fragments_each_on_own_line(self):
        lines, emit = _collect()
        _emit_displaced_utterances_line(emit, ["one", "two", "three"])
        # 1 header + 3 fragment lines.
        assert len(lines) == 4
        assert "3 fragments" in lines[0]
        assert "iter-162" in lines[0]
        assert "'one'" in lines[1]
        assert "'two'" in lines[2]
        assert "'three'" in lines[3]

    def test_blank_fragments_dropped_from_count(self):
        lines, emit = _collect()
        # Two non-blank + a blank → counted as 1 (single-fragment form).
        _emit_displaced_utterances_line(emit, ["only", "   "])
        assert len(lines) == 1
        assert "'only'" in lines[0]
        # Single-fragment form, not the "N fragments" header.
        assert "fragments" not in lines[0]


# ---- integration through print_session_summary -----------------------------


def _summary(metrics_list, *, displaced):
    buf = io.StringIO()
    print_session_summary(
        metrics_list,
        {"model": "test-model"},
        file=buf,
        meta=SessionMeta(utterances_displaced=displaced),
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
    def test_displaced_surfaced_with_completed_turns(self):
        out = _summary([_one_metric()], displaced=["I was thinking about the"])
        assert "I was thinking about the" in out
        assert "Displaced uttr." in out

    def test_no_displaced_line_when_empty(self):
        out = _summary([_one_metric()], displaced=[])
        assert "Displaced uttr." not in out

    def test_displaced_surfaced_on_no_completed_turns_path(self):
        # The early-return path (zero metrics) must still surface fragments —
        # a session can displace a fragment yet land no completed turns.
        out = _summary([], displaced=["wait actually"])
        assert "no completed turns" in out
        assert "wait actually" in out

    def test_no_completed_turns_no_displaced_is_clean(self):
        out = _summary([], displaced=[])
        assert "no completed turns" in out
        assert "Displaced uttr." not in out

    def test_multiple_displaced_all_surfaced(self):
        out = _summary(
            [_one_metric()], displaced=["frag one", "frag two"]
        )
        assert "frag one" in out
        assert "frag two" in out
        assert "2 fragments" in out
