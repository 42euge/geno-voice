"""Tests for iter-146 — the streaming stabilization seam (examples/mic_stream.py).

``mic_stream.py`` is the live progressive-transcription entrypoint (``gv
stream``). Its core logic — the iter-008 streaming-overlap design that
promotes a transcript prefix from speculative (dim) to stable (bright)
once it holds unchanged across consecutive inference passes — used to live
inline inside ``run_stream``'s mic loop and had zero test coverage.

iter-146 extracts that per-pass step into the pure ``stabilize_pass``
function (no I/O, no clock reads — wall-clock side-effects stay at the
caller) plus the ``_longest_common_prefix`` helper. These tests drive a
full multi-pass convergence without pyaudio or mlx_whisper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples import mic_stream as ms  # noqa: E402


# ---- _longest_common_prefix --------------------------------------------


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("hello world", "hello there", "hello "),
        ("", "anything", ""),
        ("anything", "", ""),
        ("same", "same", "same"),
        ("abc", "abcdef", "abc"),
        ("abcdef", "abc", "abc"),
        ("xyz", "abc", ""),
    ],
)
def test_longest_common_prefix(a, b, expected):
    assert ms._longest_common_prefix(a, b) == expected


def test_longest_common_prefix_returns_prefix_of_first_arg():
    # The result is sliced from ``a`` — confirm it is genuinely a prefix.
    a, b = "the quick brown", "the quiet"
    out = ms._longest_common_prefix(a, b)
    assert a.startswith(out)
    assert out == "the qui"


# ---- stabilize_pass: single-pass shape ---------------------------------


def _fresh(text, prev=""):
    """First-pass call against an empty stable/candidate state."""
    return ms.stabilize_pass(
        text=text,
        prev_full_text=prev,
        stable="",
        stable_candidate="",
        stable_count=0,
        passes=1,
        settled_at_pass=0,
    )


def test_first_pass_is_all_speculative():
    step = _fresh("the cat")
    assert step.stable == ""
    assert step.speculative == "the cat"
    assert not step.promoted
    assert step.prev_full_text == "the cat"


def test_first_pass_records_text_as_prev():
    step = _fresh("hello")
    assert step.prev_full_text == "hello"


def test_empty_candidate_matches_empty_common_no_change():
    # common("x", "") == "" which equals the empty candidate → not changed,
    # count increments rather than resets.
    step = ms.stabilize_pass(
        text="x", prev_full_text="", stable="", stable_candidate="",
        stable_count=0, passes=1, settled_at_pass=0,
    )
    assert step.changed is False
    assert step.stable_count == 1


def test_candidate_change_resets_count_and_flags_changed():
    # prev text shares prefix "the " → common="the ", differs from the
    # current candidate "the cat" → changed, count resets to 1.
    step = ms.stabilize_pass(
        text="the dog", prev_full_text="the dax",
        stable="", stable_candidate="the cat",
        stable_count=2, passes=3, settled_at_pass=0,
    )
    assert step.changed is True
    assert step.stable_candidate == "the d"
    assert step.stable_count == 1
    assert step.promoted is False


def test_stable_candidate_holds_increments_count():
    # common("the cat sat", "the cat") == "the cat" == candidate → held.
    step = ms.stabilize_pass(
        text="the cat sat", prev_full_text="the cat",
        stable="", stable_candidate="the cat",
        stable_count=1, passes=2, settled_at_pass=0,
    )
    assert step.changed is False
    assert step.stable_count == 2


# ---- stabilize_pass: promotion -----------------------------------------


def test_promotes_when_count_reaches_threshold():
    # candidate "the cat" held for the 2nd pass (default STABILITY_PASSES=2)
    # and is longer than current stable "" → promote.
    step = ms.stabilize_pass(
        text="the cat sat", prev_full_text="the cat",
        stable="", stable_candidate="the cat",
        stable_count=1, passes=5, settled_at_pass=0,
    )
    assert step.promoted is True
    assert step.stable == "the cat"
    assert step.settled_at_pass == 5
    assert step.speculative == " sat"


def test_no_promotion_when_candidate_not_longer_than_stable():
    # Already-stable "the cat"; candidate equal length → no re-promotion.
    step = ms.stabilize_pass(
        text="the cat", prev_full_text="the cat",
        stable="the cat", stable_candidate="the cat",
        stable_count=3, passes=9, settled_at_pass=4,
    )
    assert step.promoted is False
    assert step.stable == "the cat"
    assert step.settled_at_pass == 4  # unchanged
    assert step.speculative == ""


def test_custom_stability_passes_threshold():
    # With stability_passes=3, a count that just reached 3 promotes.
    step = ms.stabilize_pass(
        text="hello world", prev_full_text="hello world",
        stable="", stable_candidate="hello world",
        stable_count=2, passes=4, settled_at_pass=0,
        stability_passes=3,
    )
    assert step.promoted is True
    assert step.stable == "hello world"


def test_higher_threshold_defers_promotion():
    step = ms.stabilize_pass(
        text="hello world", prev_full_text="hello world",
        stable="", stable_candidate="hello world",
        stable_count=1, passes=2, settled_at_pass=0,
        stability_passes=3,
    )
    assert step.promoted is False
    assert step.stable == ""


# ---- stabilize_pass: speculative tail ----------------------------------


def test_speculative_is_tail_past_stable():
    step = ms.stabilize_pass(
        text="the cat sat down", prev_full_text="the cat sat",
        stable="the cat", stable_candidate="the cat",
        stable_count=2, passes=6, settled_at_pass=3,
    )
    assert step.speculative == " sat down"


def test_speculative_is_full_text_when_no_stable_prefix():
    # A re-recognition that does NOT start with the stable prefix → the
    # whole text becomes speculative (defensive branch).
    step = ms.stabilize_pass(
        text="entirely different", prev_full_text="the cat",
        stable="the cat", stable_candidate="the cat",
        stable_count=2, passes=6, settled_at_pass=3,
    )
    assert step.speculative == "entirely different"
    assert step.stable == "the cat"


# ---- stabilize_pass: full convergence sequence -------------------------


def test_multi_pass_convergence_promotes_growing_prefix():
    """Drive several passes the way run_stream's loop does and assert the
    stable prefix grows monotonically as the transcript settles."""
    # Mirror the entrypoint locals.
    stable = ""
    stable_candidate = ""
    stable_count = 0
    settled_at_pass = 0
    prev = ""

    # Whisper re-emits increasingly complete hypotheses, then the prefix
    # holds steady once the utterance ends — the held prefix is what gets
    # promoted (a prefix that keeps growing never settles).
    hypotheses = [
        "the",
        "the cat",
        "the cat sat",
        "the cat sat",
        "the cat sat",
    ]

    promotions = []
    for i, text in enumerate(hypotheses, start=1):
        step = ms.stabilize_pass(
            text, prev, stable, stable_candidate, stable_count,
            i, settled_at_pass,
        )
        stable = step.stable
        stable_candidate = step.stable_candidate
        stable_count = step.stable_count
        settled_at_pass = step.settled_at_pass
        prev = step.prev_full_text
        if step.promoted:
            promotions.append((i, stable))

    # The held prefix settled, and the final stable is a prefix of the text.
    assert promotions, "expected at least one promotion across the sequence"
    assert stable == "the cat sat"
    assert "the cat sat".startswith(stable)
    # The full text reconstructs from stable + speculative.
    assert stable + step.speculative == "the cat sat"


def test_unstable_stream_never_promotes():
    """If every pass disagrees with the last, the common prefix keeps
    resetting and nothing is ever promoted to stable."""
    stable = ""
    stable_candidate = ""
    stable_count = 0
    settled_at_pass = 0
    prev = ""

    # Each hypothesis diverges immediately (different first word).
    hypotheses = ["alpha one", "bravo two", "charlie three", "delta four"]

    for i, text in enumerate(hypotheses, start=1):
        step = ms.stabilize_pass(
            text, prev, stable, stable_candidate, stable_count,
            i, settled_at_pass,
        )
        stable = step.stable
        stable_candidate = step.stable_candidate
        stable_count = step.stable_count
        settled_at_pass = step.settled_at_pass
        prev = step.prev_full_text
        assert not step.promoted

    assert stable == ""
    assert step.speculative == "delta four"
