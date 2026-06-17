"""Tests for iter-158 — the cross-turn utterance aggregator.

``session/utterance_aggregator.py`` is the second half of backlog #9's
live-loop driver. iter-156's ``UtteranceBuffer`` owns the hold-and-merge
*state*, but its ``offer(text, gap_secs)`` requires the inter-utterance silence
gap, which the buffer deliberately does not measure (no clock reads). The
aggregator owns the *one* scalar the buffer can't — the previous utterance's
endpoint timestamp — and derives ``gap_secs`` from injected speech timestamps,
keeping the eventual entrypoint a thin driver.

The half-duplex invariant flows through end-to-end: with a default
``FullDuplexConfig()`` the underlying buffer is a transparent passthrough, so
the aggregator emits every utterance immediately and never holds — the gap is
measured and reported but never changes the output.

Like its sibling seams, ``utterance_aggregator`` imports ``session.*`` at
module scope, but ``session/__init__.py`` eagerly imports pipecat-dependent
modules (absent on the x86_64 Linux runner). So we stand up a stub ``session``
namespace package and load the pure modules into it by file path.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SESSION_DIR = Path(__file__).resolve().parents[2] / "session"


def _load_by_path(name, filename, package=None):
    spec = importlib.util.spec_from_file_location(name, _SESSION_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    if package is not None:
        mod.__package__ = package
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if "session" not in sys.modules:
    _pkg = types.ModuleType("session")
    _pkg.__path__ = [str(_SESSION_DIR)]
    sys.modules["session"] = _pkg
for _name, _file in (
    ("session.text_eou", "text_eou.py"),
    ("session.full_duplex", "full_duplex.py"),
    ("session.utterance_merging", "utterance_merging.py"),
    ("session.utterance_buffer", "utterance_buffer.py"),
):
    if _name not in sys.modules:
        _load_by_path(_name, _file, package="session")

_agg = _load_by_path(
    "session.utterance_aggregator", "utterance_aggregator.py", package="session"
)
_ub = sys.modules["session.utterance_buffer"]
_fd = sys.modules["session.full_duplex"]
_eou = sys.modules["session.text_eou"]

UtteranceAggregator = _agg.UtteranceAggregator
AggregatedResult = _agg.AggregatedResult
UtteranceBuffer = _ub.UtteranceBuffer
EmittedTurn = _ub.EmittedTurn
FullDuplexConfig = _fd.FullDuplexConfig
TextEOUConfig = _eou.TextEOUConfig

# A config with utterance merging on (master enabled).
ORGANIC = FullDuplexConfig(enabled=True)

# Reference completeness values (from text_eou, pinned for clarity):
INCOMPLETE = "I was thinking about the"  # 0.3 — dangling article
COMPLETE = "That is my whole point."     # 1.0


# ---- the half-duplex invariant (default config) ------------------------------


def test_default_config_emits_immediately():
    agg = UtteranceAggregator()
    res = agg.offer(INCOMPLETE, speech_start_at=1.0, speech_end_at=2.0)
    assert res.turns == [EmittedTurn(INCOMPLETE, False)]
    assert res.held is None
    assert res.merged is False
    assert agg.pending is None
    assert agg.active is False


def test_default_config_never_holds_even_unfinished_quick():
    # The exact merge corner (unfinished + quick gap) — still passthrough off.
    agg = UtteranceAggregator()
    r1 = agg.offer(INCOMPLETE, speech_start_at=1.0, speech_end_at=2.0)
    assert r1.held is None and r1.turns[0].false_endpoint is False
    # Quick 0.2s gap after the prior endpoint at 2.0.
    r2 = agg.offer("the deadline.", speech_start_at=2.2, speech_end_at=3.0)
    assert r2.turns == [EmittedTurn("the deadline.", False)]
    assert r2.merged is False
    assert r2.gap_secs == pytest.approx(0.2)


def test_default_config_flush_always_empty():
    agg = UtteranceAggregator()
    agg.offer(INCOMPLETE, speech_start_at=1.0, speech_end_at=2.0)
    res = agg.flush()
    assert res.turns == []
    assert res.held is None
    assert agg.prev_end_at is None


def test_explicit_default_full_duplex_is_passthrough():
    agg = UtteranceAggregator(config=FullDuplexConfig(enabled=False))
    res = agg.offer(INCOMPLETE, speech_start_at=0.0, speech_end_at=1.0)
    assert res.turns == [EmittedTurn(INCOMPLETE, False)]
    assert res.held is None
    assert agg.active is False


# ---- gap computation ---------------------------------------------------------


def test_first_offer_gap_is_infinite():
    agg = UtteranceAggregator(config=ORGANIC)
    res = agg.offer(INCOMPLETE, speech_start_at=5.0, speech_end_at=6.0)
    assert res.gap_secs == float("inf")
    # prev_end_at now tracks this utterance's endpoint.
    assert agg.prev_end_at == pytest.approx(6.0)


def test_gap_is_start_minus_prev_end():
    agg = UtteranceAggregator(config=ORGANIC)
    # First utterance ends at 2.0 (held — unfinished + first gap is inf,
    # so it's a fresh candidate; INCOMPLETE looks unfinished so it's held).
    agg.offer(INCOMPLETE, speech_start_at=1.0, speech_end_at=2.0)
    # Second starts at 2.5 → gap = 2.5 - 2.0 = 0.5.
    res = agg.offer("the deadline.", speech_start_at=2.5, speech_end_at=3.0)
    assert res.gap_secs == pytest.approx(0.5)


def test_gap_uses_prev_speech_end_not_prev_start():
    # The gap must subtract the *endpoint* of the prior utterance, not its
    # start — silence is measured from when the prior speech stopped.
    agg = UtteranceAggregator(config=ORGANIC)
    agg.offer(INCOMPLETE, speech_start_at=0.0, speech_end_at=4.0)
    res = agg.offer("the deadline.", speech_start_at=4.3, speech_end_at=5.0)
    # gap is 4.3 - 4.0 = 0.3, NOT 4.3 - 0.0.
    assert res.gap_secs == pytest.approx(0.3)


def test_negative_raw_gap_clamped_to_zero():
    # Clock-skew: next utterance stamped slightly before prior endpoint.
    agg = UtteranceAggregator(config=ORGANIC)
    agg.offer(INCOMPLETE, speech_start_at=0.0, speech_end_at=5.0)
    res = agg.offer("the deadline.", speech_start_at=4.9, speech_end_at=6.0)
    assert res.gap_secs == 0.0


def test_prev_end_at_updates_each_offer():
    agg = UtteranceAggregator(config=ORGANIC)
    agg.offer(COMPLETE, speech_start_at=0.0, speech_end_at=1.0)
    assert agg.prev_end_at == pytest.approx(1.0)
    agg.offer(COMPLETE, speech_start_at=3.0, speech_end_at=4.5)
    assert agg.prev_end_at == pytest.approx(4.5)


# ---- organic merge driven by the measured gap --------------------------------


def test_quick_continuation_merges():
    agg = UtteranceAggregator(config=ORGANIC)
    # Unfinished prior — held (gap inf doesn't matter, held on completeness).
    r1 = agg.offer(INCOMPLETE, speech_start_at=0.0, speech_end_at=2.0)
    assert r1.turns == []
    assert r1.held == INCOMPLETE
    assert agg.pending == INCOMPLETE
    # Quick continuation 0.3s later → merge into one complete turn.
    r2 = agg.offer("the deadline.", speech_start_at=2.3, speech_end_at=3.0)
    assert r2.gap_secs == pytest.approx(0.3)
    assert r2.turns == [EmittedTurn("I was thinking about the the deadline.", True)]
    assert r2.merged is True
    assert agg.pending is None


def test_long_gap_does_not_merge_releases_both():
    agg = UtteranceAggregator(config=ORGANIC)
    r1 = agg.offer(INCOMPLETE, speech_start_at=0.0, speech_end_at=2.0)
    assert r1.held == INCOMPLETE
    # 5s gap → too long, NOT a continuation. The held prior is released, and
    # the new utterance starts fresh.
    r2 = agg.offer("Something else.", speech_start_at=7.0, speech_end_at=8.0)
    assert r2.gap_secs == pytest.approx(5.0)
    assert EmittedTurn(INCOMPLETE, False) in r2.turns
    assert r2.merged is False


def test_complete_prior_emits_immediately_no_hold():
    agg = UtteranceAggregator(config=ORGANIC)
    res = agg.offer(COMPLETE, speech_start_at=0.0, speech_end_at=1.0)
    assert res.turns == [EmittedTurn(COMPLETE, False)]
    assert res.held is None
    assert agg.pending is None


def test_flush_releases_held_pending():
    agg = UtteranceAggregator(config=ORGANIC)
    agg.offer(INCOMPLETE, speech_start_at=0.0, speech_end_at=2.0)
    res = agg.flush()
    assert res.turns == [EmittedTurn(INCOMPLETE, False)]
    assert res.held is None
    assert agg.pending is None
    assert agg.prev_end_at is None


def test_flush_resets_so_next_gap_is_infinite():
    agg = UtteranceAggregator(config=ORGANIC)
    agg.offer(INCOMPLETE, speech_start_at=0.0, speech_end_at=2.0)
    agg.flush()
    # After flush, the next utterance has no prior endpoint → gap inf.
    res = agg.offer("New thought.", speech_start_at=2.1, speech_end_at=3.0)
    assert res.gap_secs == float("inf")


def test_flush_gap_secs_is_infinite():
    agg = UtteranceAggregator(config=ORGANIC)
    res = agg.flush()
    assert res.gap_secs == float("inf")
    assert res.turns == []


# ---- chained merges accumulate across turns ----------------------------------


def test_chained_quick_continuations_accumulate():
    agg = UtteranceAggregator(config=ORGANIC)
    agg.offer("I was thinking about the", speech_start_at=0.0, speech_end_at=1.0)
    agg.offer("big and", speech_start_at=1.2, speech_end_at=2.0)
    # Running text still unfinished (trailing conjunction) — held and merged.
    assert agg.pending == "I was thinking about the big and"
    res = agg.offer("the deadline.", speech_start_at=2.2, speech_end_at=3.0)
    assert res.turns == [
        EmittedTurn("I was thinking about the big and the deadline.", True)
    ]
    assert res.merged is True


# ---- tuning threads through to the buffer ------------------------------------


def test_max_gap_secs_threads_through():
    # A tight max_gap (0.1) means a 0.5s gap is NOT quick → no merge.
    agg = UtteranceAggregator(config=ORGANIC, max_gap_secs=0.1)
    agg.offer(INCOMPLETE, speech_start_at=0.0, speech_end_at=2.0)
    res = agg.offer("the deadline.", speech_start_at=2.5, speech_end_at=3.0)
    assert res.gap_secs == pytest.approx(0.5)
    # Gap exceeds the tight ceiling → prior released as its own NEW turn.
    assert EmittedTurn(INCOMPLETE, False) in res.turns
    assert res.merged is False


def test_max_merge_depth_threads_through():
    # cap=1: the first merge force-emits even though the running text is still
    # unfinished, instead of holding again. Exercises the cap path proper.
    agg = UtteranceAggregator(config=ORGANIC, max_merge_depth=1)
    agg.offer("I was thinking about the", speech_start_at=0.0, speech_end_at=1.0)
    res = agg.offer("big and", speech_start_at=1.2, speech_end_at=2.0)
    # "I was thinking about the big and" is still unfinished (trailing
    # conjunction) but the cap force-emits it, still flagged false_endpoint.
    assert res.turns == [EmittedTurn("I was thinking about the big and", True)]
    assert res.merged is True
    assert agg.pending is None


def test_eou_config_threads_through():
    # A custom completeness config the buffer must honor. Use a tiny
    # ceiling so even INCOMPLETE scores above it → no hold.
    agg = UtteranceAggregator(config=ORGANIC, incomplete_ceiling=0.0)
    res = agg.offer(INCOMPLETE, speech_start_at=0.0, speech_end_at=1.0)
    # Nothing is unfinished enough to hold at ceiling 0.0.
    assert res.turns == [EmittedTurn(INCOMPLETE, False)]
    assert res.held is None


# ---- injected buffer ---------------------------------------------------------


def test_injected_buffer_is_used():
    buf = UtteranceBuffer(config=ORGANIC)
    agg = UtteranceAggregator(buffer=buf)
    assert agg.buffer is buf
    assert agg.active is True
    agg.offer(INCOMPLETE, speech_start_at=0.0, speech_end_at=2.0)
    # State lands on the injected buffer.
    assert buf.pending == INCOMPLETE


def test_injected_buffer_with_config_rejected():
    buf = UtteranceBuffer(config=ORGANIC)
    with pytest.raises(ValueError, match="either a pre-built"):
        UtteranceAggregator(buffer=buf, config=ORGANIC)


def test_injected_buffer_with_eou_config_rejected():
    buf = UtteranceBuffer(config=ORGANIC)
    with pytest.raises(ValueError, match="either a pre-built"):
        UtteranceAggregator(buffer=buf, eou_config=TextEOUConfig())


# ---- AggregatedResult contract -----------------------------------------------


def test_aggregated_result_defaults():
    res = AggregatedResult()
    assert res.turns == []
    assert res.held is None
    assert res.gap_secs == float("inf")
    assert res.merged is False


def test_aggregated_result_is_frozen():
    res = AggregatedResult()
    with pytest.raises(Exception):
        res.turns = [EmittedTurn("x")]  # type: ignore[misc]


def test_aggregated_result_merged_reflects_false_endpoint():
    res = AggregatedResult(turns=[EmittedTurn("x", True)])
    assert res.merged is True
    res2 = AggregatedResult(turns=[EmittedTurn("x", False)])
    assert res2.merged is False


# ---- purity: independent aggregators don't share state -----------------------


def test_independent_aggregators_isolated():
    a = UtteranceAggregator(config=ORGANIC)
    b = UtteranceAggregator(config=ORGANIC)
    a.offer(INCOMPLETE, speech_start_at=0.0, speech_end_at=2.0)
    assert a.pending == INCOMPLETE
    assert b.pending is None
    assert b.prev_end_at is None


def test_empty_text_handling():
    # An empty/blank chunk (noise) — buffer drops it; aggregator still
    # records the endpoint and reports the gap.
    agg = UtteranceAggregator(config=ORGANIC)
    res = agg.offer("", speech_start_at=0.0, speech_end_at=1.0)
    assert res.turns == []
    assert res.held is None
    assert agg.prev_end_at == pytest.approx(1.0)
