"""Tests for iter-031 — filter zero-TTFS turns from session summary.

Pre-iter-031: ``print_session_summary`` aggregated TTFS over every
turn in ``metrics_list``. But TTFS stays at the 0.0 default for any
turn that ended without playing audio:
  - worker errored before first audio
  - user barged in before any audio
  - LLM yielded no tokens (rare but possible)

Including those zeros made the median sag and produced
"Best TTFS: 0ms", which reads like a great result and is actually
"this turn never produced audio." iter-031 filters TTFS to the turns
where ``ttfs > 0``, and emits "n/a" if every turn was zero.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import TurnMetrics, print_session_summary  # noqa: E402


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _make(*, ttfs: float = 0.0, **kwargs) -> TurnMetrics:
    return TurnMetrics(
        stt_time=kwargs.get("stt_time", 0.05),
        llm_first_token=kwargs.get("llm_first_token", 0.1),
        tts_time=kwargs.get("tts_time", 0.2),
        ttfs=ttfs,
        fillers_played=kwargs.get("fillers_played", 0),
        barge_in=kwargs.get("barge_in", False),
    )


class TestZeroTtfsTurnsExcluded:
    def test_zero_ttfs_excluded_from_median(self):
        # Three turns: 0.5s, 0.0s (no audio), 1.5s. Median should be
        # 1.0s ((0.5 + 1.5) / 2), not 0.5s ((0.0 + 1.0) / 2 / ...).
        metrics = [_make(ttfs=0.5), _make(ttfs=0.0), _make(ttfs=1.5)]
        out = io.StringIO()
        print_session_summary(metrics, {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        # With filtering: median of [0.5, 1.5] = 1.0 → 1000ms
        assert "Median TTFS:      1000ms" in plain
        # Best TTFS is the smallest non-zero — 500ms, NOT 0ms.
        assert "Best TTFS:        500ms" in plain
        assert "Best TTFS:        0ms" not in plain

    def test_zero_ttfs_excluded_from_best(self):
        # Two turns where one is zero. Best should be the non-zero one.
        metrics = [_make(ttfs=0.0), _make(ttfs=0.42)]
        out = io.StringIO()
        print_session_summary(metrics, {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        assert "Best TTFS:        420ms" in plain

    def test_all_zero_ttfs_renders_na(self):
        # Every turn ended without audio (e.g. continuous LLM errors
        # or barge-ins). We can't compute a meaningful TTFS — emit
        # "n/a" rather than "0ms".
        metrics = [_make(ttfs=0.0), _make(ttfs=0.0)]
        out = io.StringIO()
        print_session_summary(metrics, {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        assert "Median TTFS:      n/a" in plain
        assert "Best TTFS:        n/a" in plain
        # Critically: no misleading "0ms".
        assert "Median TTFS:      0ms" not in plain
        assert "Best TTFS:        0ms" not in plain

    def test_single_zero_ttfs_session_renders_na(self):
        metrics = [_make(ttfs=0.0)]
        out = io.StringIO()
        print_session_summary(metrics, {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        assert "n/a" in plain

    def test_all_nonzero_ttfs_unchanged(self):
        # Regression: pre-iter-031 behavior on a happy-path session
        # is identical (no zero filtering removes any data).
        metrics = [_make(ttfs=0.3), _make(ttfs=0.5), _make(ttfs=0.7)]
        out = io.StringIO()
        print_session_summary(metrics, {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        # Median of [0.3, 0.5, 0.7] = 0.5 → 500ms.
        assert "Median TTFS:      500ms" in plain
        # Best (min) = 300ms.
        assert "Best TTFS:        300ms" in plain

    def test_other_aggregates_not_filtered(self):
        # iter-031 is scoped to TTFS only — STT/LLM/TTS aggregates
        # should still include all turns. (A turn with ttfs=0 is
        # likely the worker-errored case; STT and LLM-1st may still
        # be valid measurements for those turns.)
        metrics = [
            _make(ttfs=0.0, stt_time=0.1, llm_first_token=0.2, tts_time=0.0),
            _make(ttfs=0.5, stt_time=0.3, llm_first_token=0.4, tts_time=0.6),
        ]
        out = io.StringIO()
        print_session_summary(metrics, {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        # Median STT over both turns: (100 + 300) / 2 = 200ms.
        assert "Median STT:       200ms" in plain
        # Median LLM 1st: (200 + 400) / 2 = 300ms.
        assert "Median LLM 1st:   300ms" in plain

    def test_barge_in_count_unaffected(self):
        # A turn can be a barge-in AND have ttfs=0 (barged in before
        # any audio played). The barge-in counter still includes it.
        metrics = [_make(ttfs=0.0, barge_in=True), _make(ttfs=0.4)]
        out = io.StringIO()
        print_session_summary(metrics, {"model": "stub"}, file=out)
        plain = _strip_ansi(out.getvalue())
        assert "Barge-ins:        1" in plain
        # And TTFS aggregates only count the non-zero turn.
        assert "Best TTFS:        400ms" in plain
