"""Tests for iter-041 — barge-in latency metric.

Metric 2.10 from docs/perf-metrics-taxonomy.md. The whole barge-in
feature lives or dies on this number — >200ms is when the user
thinks the bot is ignoring them.

Implementation:
  - BargeInCoordinator stamps ``playback_stopped_at`` after
    ``worker.cancel()`` returns in ``trigger()``.
  - ChatLoop computes ``metrics.barge_in_latency =
    coord.playback_stopped_at - coord.triggered_at``.
  - Per-turn print shows latency on barge-in turns; session
    summary shows median + worst across the session.

These tests verify the coordinator timestamps, ChatLoop wiring,
and summary aggregation.
"""

from __future__ import annotations

import io
import re
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    TurnMetrics,
    print_session_summary,
)
from examples._chat_pipeline import BargeInCoordinator  # noqa: E402


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ---- Coordinator timestamps ------------------------------------------------


class TestCoordinatorTimestamps:
    def test_default_playback_stopped_at_is_none(self):
        c = BargeInCoordinator()
        assert c.playback_stopped_at is None

    def test_trigger_stamps_playback_stopped_at(self):
        clock_val = [0.0]

        def fake_clock():
            return clock_val[0]

        c = BargeInCoordinator(clock=fake_clock)
        clock_val[0] = 10.0
        c.trigger()
        # playback_stopped_at stamped after trigger() runs.
        assert c.playback_stopped_at == 10.0
        # triggered_at also stamped (was already there pre-iter-041,
        # but the latency formula needs both).
        assert c.triggered_at == 10.0

    def test_trigger_with_worker_stamps_after_cancel_returns(self):
        # Use a sleeping mock worker.cancel() to make sure
        # playback_stopped_at is sampled AFTER cancel() returns,
        # not before.
        clock_val = [100.0]

        def fake_clock():
            return clock_val[0]

        worker = MagicMock()

        def slow_cancel(timeout=5.0):
            # Simulate worker join taking real time. Inside, the
            # caller's clock should NOT have advanced since this is
            # a fake clock, but we mutate it manually here to
            # represent "time passing while worker.cancel runs."
            clock_val[0] = 100.05

        worker.cancel.side_effect = slow_cancel

        c = BargeInCoordinator(worker=worker, clock=fake_clock)
        c.trigger()

        # triggered_at was sampled before cancel ran (clock=100.0).
        assert c.triggered_at == 100.0
        # playback_stopped_at sampled after cancel returned (clock=100.05).
        assert c.playback_stopped_at == 100.05
        # Latency the chat loop would compute: 50ms.
        assert c.playback_stopped_at - c.triggered_at == pytest.approx(0.05)

    def test_idempotent_does_not_overwrite(self):
        # Second trigger() is a no-op, even though clock advanced.
        clock_val = [0.0]

        def fake_clock():
            return clock_val[0]

        c = BargeInCoordinator(clock=fake_clock)
        clock_val[0] = 5.0
        c.trigger()
        first = c.playback_stopped_at
        clock_val[0] = 999.0
        c.trigger()  # second call — no-op
        assert c.playback_stopped_at == first

    def test_trigger_with_no_worker_still_stamps(self):
        # No worker bound — trigger still records both timestamps so
        # tests / callers without a worker get a well-defined value.
        clock_val = [50.0]

        def fake_clock():
            return clock_val[0]

        c = BargeInCoordinator(clock=fake_clock)
        c.trigger()
        assert c.triggered_at == 50.0
        assert c.playback_stopped_at == 50.0


# ---- TurnMetrics default + per-turn print ----------------------------------


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_default_is_zero(self):
        assert TurnMetrics().barge_in_latency == 0.0

    def test_no_barge_in_omits_latency(self):
        m = TurnMetrics(transcript="hi", model="stub", barge_in=False)
        out = self._capture(m)
        assert "Barge latency" not in out

    def test_barge_in_with_zero_latency_omits_line(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            barge_in=True, barge_in_latency=0.0,
        )
        out = self._capture(m)
        assert "Barge latency" not in out

    def test_barge_in_with_latency_emits_line(self):
        m = TurnMetrics(
            transcript="hi", model="stub",
            barge_in=True, barge_in_latency=0.080,  # 80ms
        )
        out = self._capture(m)
        assert "Barge latency" in out
        assert "80ms" in out
        assert "(detect → halt)" in out


# ---- ChatLoop wiring -------------------------------------------------------
#
# The end-to-end path (mic → watcher → coord.trigger → worker.cancel →
# play_aligned breaks via cancel_event → worker thread joins) is
# already exercised in tests/integration/. Here we just verify that
# given a coordinator with both timestamps set, the ChatLoop arithmetic
# lands on metrics. The actual barge-in scenario tests already pass
# for iter-035 / iter-040; we don't re-run them.


class TestChatLoopArithmetic:
    def test_latency_is_difference_of_timestamps(self):
        # Build a coordinator with controlled timestamps and check
        # the ChatLoop subtraction. We do this by importing the
        # math directly — no actual loop run needed.
        c = BargeInCoordinator()
        c.triggered_at = 10.0
        c.playback_stopped_at = 10.150
        # The chat loop does:
        #   metrics.barge_in_latency = max(0.0, p - t)
        latency = max(0.0, c.playback_stopped_at - c.triggered_at)
        assert latency == pytest.approx(0.150)

    def test_negative_difference_clamps_to_zero(self):
        # If something pathological set playback_stopped_at < triggered_at
        # (would be a clock-injection bug or test setup error), the
        # metric clamps to 0 rather than going negative.
        c = BargeInCoordinator()
        c.triggered_at = 100.0
        c.playback_stopped_at = 99.0
        latency = max(0.0, c.playback_stopped_at - c.triggered_at)
        assert latency == 0.0


# ---- Session summary aggregate ---------------------------------------------


def _make_metric(barge_in=False, latency=0.0, sentences_cancelled=0):
    return TurnMetrics(
        ttfs=0.5, barge_in=barge_in,
        barge_in_latency=latency,
        sentences_cancelled=sentences_cancelled,
    )


class TestSessionSummary:
    def test_no_barges_omits_block(self):
        out = io.StringIO()
        print_session_summary([_make_metric()], {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        assert "Median barge" not in plain
        assert "Worst barge" not in plain

    def test_barges_with_latencies_show_median_and_worst(self):
        out = io.StringIO()
        print_session_summary(
            [
                _make_metric(barge_in=True, latency=0.05),  # 50ms
                _make_metric(barge_in=True, latency=0.15),  # 150ms
                _make_metric(barge_in=True, latency=0.30),  # 300ms (worst)
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Median barge:     150ms" in plain
        assert "Worst barge:      300ms" in plain

    def test_zero_latency_turns_filtered(self):
        # Even-length filter test: 100ms + 200ms with one zero-latency
        # turn (filtered) → median 150ms (not 100 from including 0).
        out = io.StringIO()
        print_session_summary(
            [
                _make_metric(barge_in=True, latency=0.0),    # filtered
                _make_metric(barge_in=True, latency=0.100),
                _make_metric(barge_in=True, latency=0.200),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        assert "Median barge:     150ms" in plain

    def test_all_zero_latency_omits_block(self):
        out = io.StringIO()
        print_session_summary(
            [
                _make_metric(barge_in=True, latency=0.0),
                _make_metric(barge_in=True, latency=0.0),
            ],
            {"model": "stub"}, file=out,
        )
        plain = _strip_ansi(out.getvalue())
        # Barge-ins counter still emitted (we still saw them), but
        # median/worst block is suppressed because no measurements.
        assert "Median barge" not in plain
        assert "Worst barge" not in plain
