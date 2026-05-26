"""Tests for iter-105 — compute_wer() primitive.

Validates the standard WER formula (S+D+I)/N against canonical
test cases. The implementation uses word-level Levenshtein DP,
so all three error types (substitution, deletion, insertion)
are exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_wer import compute_wer, _tokenize  # noqa: E402


# ---- Tokenization ----------------------------------------------------------


def test_tokenize_basic():
    """Lowercase + strip punctuation + split on whitespace."""
    assert _tokenize("Hello, World!") == ["hello", "world"]


def test_tokenize_keeps_apostrophes():
    """Contractions stay intact — "don't" stays as one word.
    Splitting on apostrophe inflates the denominator and
    over-penalizes contraction errors."""
    assert _tokenize("Don't worry") == ["don't", "worry"]


def test_tokenize_collapses_whitespace():
    """Multiple spaces, tabs, and newlines all collapse."""
    assert _tokenize("hello   world\tfoo\nbar") == [
        "hello", "world", "foo", "bar",
    ]


def test_tokenize_empty_string():
    assert _tokenize("") == []


def test_tokenize_punctuation_only():
    """A string of pure punctuation tokenizes to empty list."""
    assert _tokenize("!!! ??? ...") == []


# ---- Perfect transcription -------------------------------------------------


def test_perfect_match_zero_wer():
    """Identical strings → 0.0."""
    assert compute_wer("hello world", "hello world") == 0.0


def test_perfect_match_case_insensitive():
    """Case difference alone shouldn't count as error."""
    assert compute_wer("Hello World", "hello world") == 0.0


def test_perfect_match_with_punctuation():
    """Punctuation differences alone shouldn't count as error."""
    assert compute_wer("Hello, world!", "hello world") == 0.0


# ---- Single-edit cases (each error type) -----------------------------------


def test_one_substitution():
    """One word replaced — 1/3 = 0.333..."""
    wer = compute_wer("the quick fox", "the slow fox")
    assert wer == pytest.approx(1 / 3)


def test_one_deletion():
    """One word missing from hypothesis — 1/3."""
    wer = compute_wer("the quick fox", "the fox")
    assert wer == pytest.approx(1 / 3)


def test_one_insertion():
    """One extra word in hypothesis — 1 insertion / 3 ref words
    = 1/3. Denominator is always reference length, never max."""
    wer = compute_wer("the fox runs", "the brown fox runs")
    assert wer == pytest.approx(1 / 3)


# ---- Multi-edit cases ------------------------------------------------------


def test_completely_wrong_hypothesis():
    """Hypothesis bears no relation to reference. All 4 ref
    words are substituted → 4/4 = 1.0. (Plus 0 ins/del because
    hyp also has 4 words.)"""
    wer = compute_wer("the quick brown fox", "lorem ipsum dolor sit")
    assert wer == 1.0


def test_two_substitutions():
    """2 subs / 4 ref words = 0.5."""
    wer = compute_wer(
        "the quick brown fox",
        "the slow brown dog",
    )
    assert wer == pytest.approx(0.5)


def test_one_sub_one_del():
    """1 sub + 1 del / 4 ref = 0.5."""
    wer = compute_wer(
        "the quick brown fox",
        "the slow fox",
    )
    assert wer == pytest.approx(0.5)


# ---- Empty input edge cases -----------------------------------------------


def test_both_empty():
    """No reference, no hypothesis → no errors."""
    assert compute_wer("", "") == 0.0


def test_empty_hypothesis_full_reference():
    """All ref words deleted → 1.0."""
    assert compute_wer("hello world", "") == 1.0


def test_empty_reference_with_hypothesis():
    """Convention chosen: every hyp word is an insertion against
    empty ref. WER = len(hyp). (Some libs return inf.)"""
    assert compute_wer("", "spurious words") == 2.0


def test_empty_reference_empty_hypothesis():
    """Vacuously correct."""
    assert compute_wer("", "") == 0.0


# ---- Return type ----------------------------------------------------------


def test_returns_float():
    """Always a float, never int. Important for downstream
    median/mean math."""
    assert isinstance(compute_wer("a b c", "a b c"), float)
    assert isinstance(compute_wer("", ""), float)
    assert isinstance(compute_wer("a b", "x y"), float)


# ---- Realistic STT scenarios ---------------------------------------------


def test_realistic_clean_transcription():
    """Whisper-large quality on a clean utterance — should be 0
    or very low."""
    ref = "what is the weather today"
    hyp = "what is the weather today"
    assert compute_wer(ref, hyp) == 0.0


def test_realistic_one_word_misheard():
    """Common STT error: one word substituted. 1/5 = 0.2."""
    ref = "what is the weather today"
    hyp = "what is the whether today"
    assert compute_wer(ref, hyp) == pytest.approx(1 / 5)


def test_realistic_dropped_filler():
    """STT often drops "um" or "uh" — counts as deletion."""
    ref = "um what is the weather"
    hyp = "what is the weather"
    assert compute_wer(ref, hyp) == pytest.approx(1 / 5)


# ---- Ordering invariance ---------------------------------------------------


def test_argument_order_matters():
    """compute_wer(ref, hyp) and compute_wer(hyp, ref) are NOT
    the same — N is always the REFERENCE length. Insertions and
    deletions swap.
    """
    a = compute_wer("the fox", "the brown fox")
    b = compute_wer("the brown fox", "the fox")
    # a: ref=2 words, 1 insertion → 0.5
    # b: ref=3 words, 1 deletion → 0.333
    assert a == pytest.approx(0.5)
    assert b == pytest.approx(1 / 3)
    assert a != b
