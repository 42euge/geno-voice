"""Tests for iter-086 — SessionMeta dataclass refactor.

print_session_summary historically grew kwargs as new session-
level signals landed (false_triggers, llm_errors, trim_events,
...). iter-086 wraps these in a SessionMeta dataclass so future
additions extend the dataclass instead of the function signature.
This test suite verifies:
  - SessionMeta defaults are 0/0.0/etc.
  - The legacy kwargs path produces identical output.
  - The meta path produces identical output.
  - Mixed-mode (some via meta, some via kwargs) merges correctly.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    SessionMeta,
    TurnMetrics,
    print_session_summary,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _summary(metrics_list, **kwargs):
    out = io.StringIO()
    print_session_summary(metrics_list, {"model": "stub"}, file=out, **kwargs)
    return _strip_ansi(out.getvalue())


def _normal_metrics():
    return [TurnMetrics(ttfs=0.5), TurnMetrics(ttfs=0.6)]


# ---- SessionMeta defaults --------------------------------------


class TestDefaults:
    def test_all_defaults_zero(self):
        m = SessionMeta()
        assert m.false_triggers == 0
        assert m.session_seconds == 0.0
        assert m.llm_errors == 0
        assert m.trim_events == 0
        assert m.trim_messages_evicted == 0

    def test_can_be_constructed_with_args(self):
        m = SessionMeta(
            false_triggers=2,
            session_seconds=120.0,
            llm_errors=1,
            trim_events=3,
            trim_messages_evicted=4,
        )
        assert m.false_triggers == 2
        assert m.session_seconds == 120.0
        assert m.llm_errors == 1
        assert m.trim_events == 3
        assert m.trim_messages_evicted == 4


# ---- Legacy kwargs path ----------------------------------------


class TestLegacyKwargs:
    def test_legacy_path_works(self):
        plain = _summary(
            _normal_metrics(),
            false_triggers=2,
            session_seconds=120.0,
            llm_errors=1,
            trim_events=3,
            trim_messages_evicted=4,
        )
        # All session-level lines render.
        assert "120.0" not in plain  # session_seconds rendered as integer minutes
        assert "VAD false-trig" in plain  # gated on false_triggers > 0
        assert "Errors:" in plain  # gated on llm_errors > 0
        assert "Trim events:" in plain  # gated on trim_events > 0


# ---- meta path -------------------------------------------------


class TestMetaPath:
    def test_meta_path_produces_same_output(self):
        # Compare legacy and meta-path output character-for-character.
        legacy = _summary(
            _normal_metrics(),
            false_triggers=2,
            session_seconds=120.0,
            llm_errors=1,
            trim_events=3,
            trim_messages_evicted=4,
        )
        via_meta = _summary(
            _normal_metrics(),
            meta=SessionMeta(
                false_triggers=2,
                session_seconds=120.0,
                llm_errors=1,
                trim_events=3,
                trim_messages_evicted=4,
            ),
        )
        assert legacy == via_meta

    def test_meta_with_no_args_omits_lines(self):
        # SessionMeta() with all defaults → no session-level lines emit.
        plain = _summary(_normal_metrics(), meta=SessionMeta())
        assert "VAD false-trig" not in plain
        assert "Errors:" not in plain
        assert "Trim events:" not in plain

    def test_partial_meta_emits_only_set_fields(self):
        # Only false_triggers populated via meta.
        plain = _summary(
            _normal_metrics(),
            meta=SessionMeta(false_triggers=3),
        )
        assert "VAD false-trig" in plain
        assert "Errors:" not in plain
        assert "Trim events:" not in plain


# ---- Mixed-mode (legacy + meta) --------------------------------


class TestMixedMode:
    def test_meta_takes_precedence(self):
        # Meta provides false_triggers=5; legacy kwarg false_triggers=99
        # should be ignored because meta wins for fields it covers.
        plain = _summary(
            _normal_metrics(),
            meta=SessionMeta(false_triggers=5),
            false_triggers=99,  # ignored
        )
        # The metric should reflect 5, not 99.
        assert "5/" in plain  # "5/N (X%)" form
        assert "99" not in plain

    def test_legacy_fills_gaps_when_meta_partial(self):
        # Meta only sets llm_errors; trim_events comes via legacy kwarg.
        plain = _summary(
            _normal_metrics(),
            meta=SessionMeta(llm_errors=2),
            trim_events=3,
            trim_messages_evicted=3,
        )
        # Both kinds of session-level lines emit.
        assert "Errors:" in plain
        assert "Trim events:" in plain
