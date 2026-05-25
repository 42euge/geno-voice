"""Tests for iter-078 — trim event rate.

Metric 2.24 from docs/perf-metrics-taxonomy.md.

Counts how many times across a session ``trim_history`` actually
evicted at least one message, plus the cumulative count of evicted
messages. Pairs with iter-077: validates the ``max_user_assistant``
threshold is calibrated.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_helpers import trim_history  # noqa: E402
from examples._chat_loop import ChatLoop  # noqa: E402
from examples._chat_metrics import (  # noqa: E402
    TurnMetrics,
    print_session_summary,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _summary(metrics_list, **kwargs):
    out = io.StringIO()
    print_session_summary(metrics_list, {"model": "stub"}, file=out, **kwargs)
    return _strip_ansi(out.getvalue())


# ---- print_session_summary kwargs --------------------------------


class TestSummaryKwargs:
    def test_default_omits_line(self):
        plain = _summary([TurnMetrics(ttfs=0.5)])
        assert "Trim events" not in plain

    def test_zero_events_omits(self):
        # Explicit zero is the same as default.
        plain = _summary(
            [TurnMetrics(ttfs=0.5)],
            trim_events=0,
            trim_messages_evicted=0,
        )
        assert "Trim events" not in plain

    def test_one_event_emits(self):
        plain = _summary(
            [TurnMetrics(ttfs=0.5)],
            trim_events=1,
            trim_messages_evicted=2,
        )
        assert "Trim events:      1 (2 evicted, 2.0/event)" in plain

    def test_steady_state_one_per_event(self):
        # 5 events evicting 5 messages → 1.0/event (the calibrated case).
        plain = _summary(
            [TurnMetrics(ttfs=0.5)],
            trim_events=5,
            trim_messages_evicted=5,
        )
        assert "Trim events:      5 (5 evicted, 1.0/event)" in plain

    def test_high_ratio(self):
        # 2 events evicting 8 messages → 4.0/event (catch-up trim).
        plain = _summary(
            [TurnMetrics(ttfs=0.5)],
            trim_events=2,
            trim_messages_evicted=8,
        )
        assert "Trim events:      2 (8 evicted, 4.0/event)" in plain


# ---- trim_history return-length contract --------------------------


class TestTrimDiffSemantics:
    """The mic_chat caller diffs lengths around trim_history. These
    tests verify that contract directly: a trim that evicts produces
    a length delta; a no-op produces zero.
    """

    def test_no_op_below_threshold(self):
        msgs = [{"role": "system", "content": "S"}] + [
            {"role": "user", "content": f"u{i}"} for i in range(5)
        ]
        before = len(msgs)
        trimmed = trim_history(msgs, max_user_assistant=20)
        assert len(trimmed) == before
        assert before - len(trimmed) == 0

    def test_evicts_above_threshold(self):
        msgs = [{"role": "system", "content": "S"}] + [
            {"role": "user", "content": f"u{i}"} for i in range(25)
        ]
        before = len(msgs)
        trimmed = trim_history(msgs, max_user_assistant=20)
        # 1 system + 25 → trim retains 1 system + last 20 = 21.
        assert len(trimmed) == 21
        assert before - len(trimmed) == 5

    def test_empty(self):
        assert trim_history([]) == []
        assert len(trim_history([])) == 0


# ---- ChatLoop.trim_messages passthrough --------------------------


class TestChatLoopTrimMessages:
    def test_forwards_evict(self):
        # Ensure ChatLoop.trim_messages preserves the same semantics
        # the mic_chat counter relies on.
        msgs = [{"role": "system", "content": "S"}] + [
            {"role": "user", "content": f"m{i}"} for i in range(15)
        ]
        before = len(msgs)
        out = ChatLoop.trim_messages(msgs, max_user_assistant=10)
        evicted = before - len(out)
        # 1 + 15 in, 1 + 10 out → evicted 5.
        assert evicted == 5

    def test_forwards_no_op(self):
        msgs = [{"role": "system", "content": "S"}] + [
            {"role": "user", "content": f"m{i}"} for i in range(3)
        ]
        before = len(msgs)
        out = ChatLoop.trim_messages(msgs, max_user_assistant=10)
        assert before - len(out) == 0


# ---- Co-emission with iter-077 context-tokens --------------------


class TestCoEmission:
    def test_both_lines_present(self):
        # Long enough session to surface both metrics.
        metrics = [
            TurnMetrics(ttfs=0.5, llm_total=0.5, context_tokens=20),
            TurnMetrics(ttfs=0.5, llm_total=0.5, context_tokens=40),
            TurnMetrics(ttfs=0.5, llm_total=0.5, context_tokens=80),
        ]
        plain = _summary(metrics, trim_events=2, trim_messages_evicted=2)
        assert "Context tokens" in plain
        assert "Context growth" in plain
        assert "Trim events" in plain
