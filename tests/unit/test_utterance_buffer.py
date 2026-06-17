"""Tests for iter-156 — the stateful utterance buffer-merge coordinator.

``session/utterance_buffer.py`` is the live-loop driver for backlog #9: it
wraps the stateless ``decide_utterance_continuation`` seam (iter-155) in the
hold-and-merge *state* a live STT loop needs — a pending held utterance, the
running merged text, and a ``false_endpoint`` flag that travels with each
released turn so iter-154's metric populates from the live path.

The whole point is the **half-duplex invariant**: with a default
``FullDuplexConfig()`` the buffer is a transparent passthrough — every ``offer``
emits its text immediately, nothing is ever held, ``flush`` is always empty.
Only with utterance merging explicitly on does the hold-and-merge machinery
engage, and even then only an *unfinished-looking* utterance is held.

Like its sibling seams, ``utterance_buffer`` imports ``session.*`` at module
scope, but ``session/__init__.py`` eagerly imports pipecat-dependent modules
(absent on the x86_64 Linux runner). So we stand up a stub ``session``
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
):
    if _name not in sys.modules:
        _load_by_path(_name, _file, package="session")

_ub = _load_by_path(
    "session.utterance_buffer", "utterance_buffer.py", package="session"
)
_fd = sys.modules["session.full_duplex"]
_eou = sys.modules["session.text_eou"]

UtteranceBuffer = _ub.UtteranceBuffer
EmittedTurn = _ub.EmittedTurn
BufferResult = _ub.BufferResult
DEFAULT_MAX_MERGE_DEPTH = _ub.DEFAULT_MAX_MERGE_DEPTH
FullDuplexConfig = _fd.FullDuplexConfig
TextEOUConfig = _eou.TextEOUConfig

# A config with utterance merging on (master enabled).
ORGANIC = FullDuplexConfig(enabled=True)

# Reference completeness values (from text_eou, pinned here for clarity):
INCOMPLETE = "I was thinking about the"  # 0.3 — dangling article
CONJ = "I think that and"                # 0.2 — trailing conjunction
COMMA = "Let me say,"                     # 0.6 — trailing comma (== ceiling)
COMPLETE = "That is my whole point."      # 1.0


# ---- the half-duplex invariant (default config) ------------------------------


def test_default_config_is_passthrough_emit_immediately():
    buf = UtteranceBuffer()
    res = buf.offer(INCOMPLETE, gap_secs=0.5)
    assert res.turns == [EmittedTurn(INCOMPLETE, False)]
    assert res.held is None
    assert res.merged is False
    assert buf.pending is None
    assert buf.active is False


def test_default_config_never_holds_even_unfinished_quick():
    # The exact merge corner (unfinished + quick) — still passthrough off.
    buf = UtteranceBuffer()
    r1 = buf.offer(INCOMPLETE, gap_secs=0.2)
    assert r1.held is None and r1.turns[0].false_endpoint is False
    r2 = buf.offer("the deadline.", gap_secs=0.2)
    assert r2.turns == [EmittedTurn("the deadline.", False)]
    assert r2.merged is False


def test_default_config_flush_always_empty():
    buf = UtteranceBuffer()
    buf.offer(INCOMPLETE, gap_secs=0.1)
    res = buf.flush()
    assert res.turns == []
    assert res.held is None


def test_explicit_default_full_duplex_is_passthrough():
    buf = UtteranceBuffer(config=FullDuplexConfig(enabled=False))
    res = buf.offer(CONJ, gap_secs=0.1)
    assert res.turns == [EmittedTurn(CONJ, False)]
    assert buf.active is False


def test_master_on_but_merging_held_back_is_passthrough():
    cfg = FullDuplexConfig(enabled=True, utterance_merging=False)
    buf = UtteranceBuffer(config=cfg)
    res = buf.offer(INCOMPLETE, gap_secs=0.2)
    assert res.turns == [EmittedTurn(INCOMPLETE, False)]
    assert res.held is None
    assert buf.active is False


# ---- organic mode: holding an unfinished prior -------------------------------


def test_organic_holds_unfinished_utterance():
    buf = UtteranceBuffer(config=ORGANIC)
    res = buf.offer(INCOMPLETE, gap_secs=0.5)
    assert res.turns == []           # nothing emitted yet — held back
    assert res.held == INCOMPLETE
    assert buf.pending == INCOMPLETE
    assert buf.active is True


def test_organic_emits_complete_utterance_immediately():
    buf = UtteranceBuffer(config=ORGANIC)
    res = buf.offer(COMPLETE, gap_secs=0.5)
    assert res.turns == [EmittedTurn(COMPLETE, False)]
    assert res.held is None
    assert buf.pending is None


# ---- organic mode: the merge (false endpoint repair) -------------------------


def test_merge_glues_quick_continuation_onto_unfinished_prior():
    buf = UtteranceBuffer(config=ORGANIC)
    buf.offer(INCOMPLETE, gap_secs=0.5)             # held
    res = buf.offer("deadline today.", gap_secs=1.0)
    assert res.turns == [EmittedTurn("I was thinking about the deadline today.", True)]
    assert res.merged is True
    assert res.held is None
    assert buf.pending is None


def test_merge_sets_false_endpoint_flag():
    buf = UtteranceBuffer(config=ORGANIC)
    buf.offer(CONJ, gap_secs=0.3)
    res = buf.offer("we should ship it.", gap_secs=0.8)
    assert len(res.turns) == 1
    assert res.turns[0].false_endpoint is True
    assert res.turns[0].text == "I think that and we should ship it."


def test_merged_text_join_is_single_spaced():
    buf = UtteranceBuffer(config=ORGANIC)
    buf.offer("  I was thinking about the  ", gap_secs=0.2)
    res = buf.offer("   deadline.   ", gap_secs=0.2)
    assert res.turns[0].text == "I was thinking about the deadline."


def test_chained_merges_accumulate_and_stay_false_endpoint():
    # Two consecutive unfinished continuations before a complete one.
    buf = UtteranceBuffer(config=ORGANIC)
    r1 = buf.offer(INCOMPLETE, gap_secs=0.2)                 # held (0.3)
    assert r1.turns == []
    r2 = buf.offer("the upcoming and", gap_secs=0.3)         # merge → "...and" (0.2), held again
    assert r2.turns == []
    assert r2.held == "I was thinking about the the upcoming and"
    r3 = buf.offer("the launch date.", gap_secs=0.3)         # merge → complete, emit
    assert len(r3.turns) == 1
    assert r3.turns[0].false_endpoint is True
    assert r3.turns[0].text == (
        "I was thinking about the the upcoming and the launch date."
    )


# ---- organic mode: NEW turns (no merge) --------------------------------------


def test_long_gap_after_unfinished_releases_prior_as_new_turn():
    buf = UtteranceBuffer(config=ORGANIC)
    buf.offer(INCOMPLETE, gap_secs=0.2)             # held
    res = buf.offer(COMPLETE, gap_secs=5.0)         # gap too long ⇒ NEW
    # Pending released (not a merge ⇒ false_endpoint False), new complete emits.
    assert res.turns == [
        EmittedTurn(INCOMPLETE, False),
        EmittedTurn(COMPLETE, False),
    ]
    assert res.merged is False
    assert buf.pending is None


def test_complete_prior_then_quick_new_both_emit_separately():
    buf = UtteranceBuffer(config=ORGANIC)
    r1 = buf.offer(COMPLETE, gap_secs=0.2)          # complete ⇒ emit immediately
    assert r1.turns == [EmittedTurn(COMPLETE, False)]
    r2 = buf.offer("And another point.", gap_secs=0.2)
    assert r2.turns == [EmittedTurn("And another point.", False)]
    assert r2.merged is False


def test_new_after_unfinished_then_held_again_when_new_is_unfinished():
    buf = UtteranceBuffer(config=ORGANIC)
    buf.offer(INCOMPLETE, gap_secs=0.2)             # held
    # Long gap ⇒ NEW: release prior; the new chunk is itself unfinished ⇒ held.
    res = buf.offer(CONJ, gap_secs=5.0)
    assert res.turns == [EmittedTurn(INCOMPLETE, False)]
    assert res.held == CONJ
    assert buf.pending == CONJ


# ---- flush -------------------------------------------------------------------


def test_flush_releases_held_unfinished_pending():
    buf = UtteranceBuffer(config=ORGANIC)
    buf.offer(INCOMPLETE, gap_secs=0.2)
    res = buf.flush()
    assert res.turns == [EmittedTurn(INCOMPLETE, False)]
    assert res.held is None
    assert buf.pending is None


def test_flush_preserves_merged_flag_on_held_pending():
    # A merge that keeps the running text unfinished stays held; flush must
    # surface the false_endpoint flag accumulated so far.
    buf = UtteranceBuffer(config=ORGANIC)
    buf.offer(INCOMPLETE, gap_secs=0.2)             # held (0.3)
    r = buf.offer("the upcoming and", gap_secs=0.3)  # merge, still unfinished ⇒ held
    assert r.turns == []
    res = buf.flush()
    assert len(res.turns) == 1
    assert res.turns[0].false_endpoint is True
    assert res.turns[0].text == "I was thinking about the the upcoming and"


def test_flush_empty_when_nothing_pending():
    buf = UtteranceBuffer(config=ORGANIC)
    buf.offer(COMPLETE, gap_secs=0.2)               # emitted, nothing held
    assert buf.flush().turns == []


# ---- boundaries & edge inputs ------------------------------------------------


def test_gap_at_max_is_inclusive_merge():
    buf = UtteranceBuffer(config=ORGANIC)
    buf.offer(INCOMPLETE, gap_secs=0.2)
    res = buf.offer("the end.", gap_secs=2.0)       # exactly max ⇒ merge
    assert res.merged is True


def test_gap_just_above_max_is_new():
    buf = UtteranceBuffer(config=ORGANIC)
    buf.offer(INCOMPLETE, gap_secs=0.2)
    res = buf.offer(COMPLETE, gap_secs=2.0001)      # above max ⇒ NEW
    assert res.merged is False
    assert res.turns[0].text == INCOMPLETE


def test_completeness_at_ceiling_holds_then_merges():
    # COMMA == 0.6 == DEFAULT_INCOMPLETE_CEILING (inclusive) ⇒ held & mergeable.
    buf = UtteranceBuffer(config=ORGANIC)
    r1 = buf.offer(COMMA, gap_secs=0.2)
    assert r1.held == COMMA
    r2 = buf.offer("here is the rest.", gap_secs=0.2)
    assert r2.merged is True
    assert r2.turns[0].text == "Let me say, here is the rest."


def test_empty_offer_in_passthrough_emits_empty_turn():
    buf = UtteranceBuffer()
    res = buf.offer("", gap_secs=0.1)
    assert res.turns == [EmittedTurn("", False)]


def test_empty_offer_while_holding_keeps_pending():
    # An empty/noise chunk shouldn't merge or clobber the held pending.
    buf = UtteranceBuffer(config=ORGANIC)
    buf.offer(INCOMPLETE, gap_secs=0.2)             # held
    res = buf.offer("   ", gap_secs=0.3)
    # decide returns NEW for blank next ⇒ pending released, blank not held.
    assert res.turns == [EmittedTurn(INCOMPLETE, False)]
    assert res.held is None
    assert buf.pending is None


def test_none_text_treated_as_empty_in_passthrough():
    buf = UtteranceBuffer()
    res = buf.offer(None, gap_secs=0.1)
    assert res.turns == [EmittedTurn("", False)]


# ---- custom thresholds & configs ---------------------------------------------


def test_custom_max_gap_secs_threads_through():
    buf = UtteranceBuffer(config=ORGANIC, max_gap_secs=0.5)
    buf.offer(INCOMPLETE, gap_secs=0.2)
    res = buf.offer(COMPLETE, gap_secs=1.0)         # > custom max ⇒ NEW
    assert res.merged is False


def test_custom_incomplete_ceiling_threads_through():
    # Lower ceiling than COMMA's 0.6 ⇒ COMMA now looks complete ⇒ emit, not held.
    buf = UtteranceBuffer(config=ORGANIC, incomplete_ceiling=0.5)
    res = buf.offer(COMMA, gap_secs=0.2)
    assert res.turns == [EmittedTurn(COMMA, False)]
    assert res.held is None


def test_custom_eou_config_threads_through():
    # A custom TextEOUConfig with a different comma weight changes whether COMMA
    # is held. Set comma weight high (complete) ⇒ emit immediately.
    cfg = TextEOUConfig(comma_completeness=0.9)
    buf = UtteranceBuffer(config=ORGANIC, eou_config=cfg)
    res = buf.offer(COMMA, gap_secs=0.2)
    assert res.turns == [EmittedTurn(COMMA, False)]


# ---- BufferResult / EmittedTurn contracts ------------------------------------


def test_emitted_turn_defaults_false_endpoint_false():
    assert EmittedTurn("hi").false_endpoint is False


def test_buffer_result_defaults_empty():
    r = BufferResult()
    assert r.turns == []
    assert r.held is None
    assert r.merged is False


def test_buffer_result_merged_reflects_any_false_endpoint():
    r = BufferResult(turns=[EmittedTurn("a", False), EmittedTurn("b", True)])
    assert r.merged is True


def test_emitted_turn_is_frozen():
    t = EmittedTurn("x")
    with pytest.raises(Exception):
        t.text = "y"  # type: ignore[misc]


# ---- purity / no cross-call leakage ------------------------------------------


def test_independent_buffers_do_not_share_state():
    a = UtteranceBuffer(config=ORGANIC)
    b = UtteranceBuffer(config=ORGANIC)
    a.offer(INCOMPLETE, gap_secs=0.2)
    assert a.pending == INCOMPLETE
    assert b.pending is None


def test_active_property_reflects_config():
    assert UtteranceBuffer().active is False
    assert UtteranceBuffer(config=ORGANIC).active is True
    assert UtteranceBuffer(
        config=FullDuplexConfig(enabled=True, utterance_merging=False)
    ).active is False


# ---- iter-157: merge-depth safety cap ----------------------------------------

# A continuation that, when glued onto an unfinished prior, still ends in a
# trailing function word (here "the"), so the running text stays *incomplete*
# and keeps being held — the exact pathological shape the cap guards against.
CONT_FRAG = "and the"


def test_default_max_merge_depth_is_eight():
    # The documented default — high above any realistic conversation, so the
    # cap is a backstop that never fires in practice.
    assert DEFAULT_MAX_MERGE_DEPTH == 8


def test_merge_count_starts_at_zero():
    buf = UtteranceBuffer(config=ORGANIC)
    assert buf.merge_count == 0


def test_merge_count_tracks_each_merge():
    buf = UtteranceBuffer(config=ORGANIC, max_merge_depth=8)
    buf.offer(INCOMPLETE, gap_secs=0.2)                 # held, no merge yet
    assert buf.merge_count == 0
    buf.offer(CONT_FRAG, gap_secs=0.3)                  # 1st merge, still held
    assert buf.merge_count == 1
    buf.offer(CONT_FRAG, gap_secs=0.3)                  # 2nd merge, still held
    assert buf.merge_count == 2


def test_cap_force_emits_after_max_merges():
    # cap=2 ⇒ the 2nd merge force-emits the still-unfinished running text
    # instead of holding it a third time.
    buf = UtteranceBuffer(config=ORGANIC, max_merge_depth=2)
    r1 = buf.offer(INCOMPLETE, gap_secs=0.2)            # held
    assert r1.turns == [] and r1.held is not None
    r2 = buf.offer(CONT_FRAG, gap_secs=0.3)             # 1st merge → held again
    assert r2.turns == [] and r2.held is not None
    r3 = buf.offer(CONT_FRAG, gap_secs=0.3)             # 2nd merge → cap → emit
    assert len(r3.turns) == 1
    assert r3.held is None
    assert buf.pending is None
    assert buf.merge_count == 0                         # reset after release


def test_force_emitted_turn_keeps_false_endpoint_flag():
    # The force-emitted turn absorbed real merges on the way up, so it must
    # still report the repaired false endpoint.
    buf = UtteranceBuffer(config=ORGANIC, max_merge_depth=2)
    buf.offer(INCOMPLETE, gap_secs=0.2)
    buf.offer(CONT_FRAG, gap_secs=0.3)
    res = buf.offer(CONT_FRAG, gap_secs=0.3)
    assert res.turns[0].false_endpoint is True
    assert res.merged is True
    assert res.turns[0].text == (
        "I was thinking about the and the and the"
    )


def test_cap_of_one_force_emits_on_first_merge():
    # cap=1 ⇒ the initial unfinished utterance is still held (holding is not a
    # merge), but the very first merge force-emits rather than holding again.
    buf = UtteranceBuffer(config=ORGANIC, max_merge_depth=1)
    r1 = buf.offer(INCOMPLETE, gap_secs=0.2)            # held (no merge)
    assert r1.turns == [] and r1.held == INCOMPLETE
    r2 = buf.offer(CONT_FRAG, gap_secs=0.3)             # 1st merge → cap → emit
    assert len(r2.turns) == 1
    assert r2.turns[0].false_endpoint is True
    assert r2.turns[0].text == "I was thinking about the and the"
    assert r2.held is None
    assert buf.pending is None


def test_below_cap_behaves_like_iter156():
    # With a generous cap, a realistic 2-merge chain ending in a complete
    # sentence behaves byte-for-byte as iter-156 (no force-emit).
    buf = UtteranceBuffer(config=ORGANIC, max_merge_depth=8)
    buf.offer(INCOMPLETE, gap_secs=0.2)
    buf.offer(CONT_FRAG, gap_secs=0.3)
    res = buf.offer("launch date.", gap_secs=0.3)       # completes ⇒ natural emit
    assert len(res.turns) == 1
    assert res.held is None
    assert res.turns[0].false_endpoint is True
    assert res.turns[0].text == (
        "I was thinking about the and the launch date."
    )


def test_merge_count_resets_on_new_turn():
    # A genuine NEW release (long gap) resets the merge counter so the next
    # candidate starts its own cap budget.
    buf = UtteranceBuffer(config=ORGANIC, max_merge_depth=4)
    buf.offer(INCOMPLETE, gap_secs=0.2)
    buf.offer(CONT_FRAG, gap_secs=0.3)                  # merge_count → 1
    assert buf.merge_count == 1
    res = buf.offer(INCOMPLETE, gap_secs=5.0)           # long gap ⇒ NEW release
    assert res.turns[0].text == "I was thinking about the and the"
    assert res.turns[0].false_endpoint is True          # released pending merged
    assert buf.merge_count == 0                          # fresh candidate
    assert buf.pending == INCOMPLETE


def test_merge_count_resets_on_flush():
    buf = UtteranceBuffer(config=ORGANIC, max_merge_depth=4)
    buf.offer(INCOMPLETE, gap_secs=0.2)
    buf.offer(CONT_FRAG, gap_secs=0.3)
    assert buf.merge_count == 1
    buf.flush()
    assert buf.merge_count == 0
    assert buf.pending is None


def test_continued_offers_after_force_emit_start_fresh():
    # After a force-emit, the buffer is empty and a new unfinished utterance is
    # held under a fresh cap budget — the cap doesn't permanently disable
    # holding.
    buf = UtteranceBuffer(config=ORGANIC, max_merge_depth=1)
    buf.offer(INCOMPLETE, gap_secs=0.2)
    buf.offer(CONT_FRAG, gap_secs=0.3)                  # force-emit
    assert buf.pending is None
    r = buf.offer(CONJ, gap_secs=0.2)                   # new unfinished ⇒ held
    assert r.turns == []
    assert r.held == CONJ
    assert buf.merge_count == 0


def test_max_merge_depth_below_one_raises():
    with pytest.raises(ValueError, match="max_merge_depth"):
        UtteranceBuffer(config=ORGANIC, max_merge_depth=0)
    with pytest.raises(ValueError, match="max_merge_depth"):
        UtteranceBuffer(max_merge_depth=-1)


def test_cap_irrelevant_in_half_duplex_passthrough():
    # The cap only governs the hold-and-merge path; half-duplex never holds, so
    # even a tiny cap is a no-op there.
    buf = UtteranceBuffer(max_merge_depth=1)            # default half-duplex
    r1 = buf.offer(INCOMPLETE, gap_secs=0.2)
    r2 = buf.offer(CONT_FRAG, gap_secs=0.3)
    assert r1.turns == [EmittedTurn(INCOMPLETE, False)]
    assert r2.turns == [EmittedTurn(CONT_FRAG, False)]
    assert buf.merge_count == 0
