"""Tests for iter-148 — the backchannel / continuer classifier.

``session/backchannel.py`` is backlog item #1 of the organic turn-taking
track (``docs/research/organic-turn-taking.md``). It recognizes short,
low-energy, closed-class "keep going" utterances (continuers) as a distinct
signal from substantive speech — the backchannels that ``filter_noise``
currently discards as noise.

``classify_backchannel`` is a pure function (no I/O, no clock), so these
tests drive it directly with text and an optional injected energy value.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# ``session/__init__.py`` eagerly imports pipecat-dependent modules
# (session.compute), which aren't installable on this x86_64 Linux runner.
# ``backchannel`` is pure stdlib, so load it directly by file path to bypass
# the package ``__init__`` — mirrors how the mic_* tests keep platform deps
# (pyaudio) out of the unit path.
_BC_PATH = Path(__file__).resolve().parents[2] / "session" / "backchannel.py"
_spec = importlib.util.spec_from_file_location("_bc_under_test", _BC_PATH)
_bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bc)

Backchannel = _bc.Backchannel
classify_backchannel = _bc.classify_backchannel
is_continuer = _bc.is_continuer
CONTINUER_LEXICON = _bc.CONTINUER_LEXICON
DEFAULT_ENERGY_CEILING = _bc.DEFAULT_ENERGY_CEILING
DEFAULT_MAX_CONTINUER_WORDS = _bc.DEFAULT_MAX_CONTINUER_WORDS


# ---- NOT_SPEECH --------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\t\n", "...", "!!", "  ,  "])
def test_empty_or_punct_only_is_not_speech(text):
    """Whitespace- or punctuation-only chunks carry no words."""
    assert classify_backchannel(text) is Backchannel.NOT_SPEECH


# ---- CONTINUER (single token) ------------------------------------------


@pytest.mark.parametrize(
    "text", ["mhmm", "yeah", "right", "okay", "ok", "hmm", "sure", "oh", "wow", "yep"]
)
def test_single_token_continuers(text):
    assert classify_backchannel(text) is Backchannel.CONTINUER


def test_continuer_case_insensitive():
    assert classify_backchannel("YEAH") is Backchannel.CONTINUER
    assert classify_backchannel("Mhmm") is Backchannel.CONTINUER


@pytest.mark.parametrize("text", ["yeah.", "right!", "mm-hmm.", "uh-huh!", "  okay  "])
def test_continuer_punct_and_whitespace_stripped(text):
    """Trailing punctuation, hyphens, and surrounding whitespace normalize."""
    assert classify_backchannel(text) is Backchannel.CONTINUER


# ---- CONTINUER (multi-word phrases & repeats) --------------------------


@pytest.mark.parametrize(
    "text",
    ["i see", "go on", "uh huh", "mm hmm", "got it", "yeah yeah", "mhmm mhmm"],
)
def test_two_word_continuers(text):
    """Phrase-lexicon entries and two repeated continuer tokens both pass."""
    assert classify_backchannel(text) is Backchannel.CONTINUER


def test_hyphenated_phrase_normalizes_to_phrase_key():
    """'uh-huh' → 'uh huh' (a phrase lexicon entry), not two unknown tokens."""
    assert classify_backchannel("uh-huh") is Backchannel.CONTINUER


# ---- SUBSTANTIVE -------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "what do you think about this",
        "I don't know what to do anymore",
        "the deadline is tomorrow",
        "tell me a story",
    ],
)
def test_long_utterances_are_substantive(text):
    assert classify_backchannel(text) is Backchannel.SUBSTANTIVE


def test_over_word_limit_short_circuits_before_lexicon():
    """Three+ words is substantive even if every word is a continuer."""
    assert classify_backchannel("yeah yeah yeah") is Backchannel.SUBSTANTIVE


def test_short_but_unknown_words_are_substantive():
    assert classify_backchannel("absolutely not") is Backchannel.SUBSTANTIVE
    assert classify_backchannel("stop") is Backchannel.SUBSTANTIVE


def test_mixed_continuer_and_content_word_is_substantive():
    """'yeah but' — one continuer, one content word ⇒ not a pure continuer."""
    assert classify_backchannel("yeah but") is Backchannel.SUBSTANTIVE


# ---- energy gate -------------------------------------------------------


def test_low_energy_continuer_stays_continuer():
    assert classify_backchannel("yeah", energy=0.1) is Backchannel.CONTINUER


def test_energy_at_ceiling_is_still_continuer():
    """Strict '>' — energy exactly at the ceiling is not loud enough to flip."""
    assert (
        classify_backchannel("yeah", energy=DEFAULT_ENERGY_CEILING)
        is Backchannel.CONTINUER
    )


def test_loud_continuer_becomes_substantive():
    """An emphatic 'YEAH!' is taking the floor, not backchanneling."""
    assert classify_backchannel("yeah", energy=0.9) is Backchannel.SUBSTANTIVE


def test_energy_gate_skipped_when_none():
    """No energy supplied ⇒ lexicon + length decide (text-only callers)."""
    assert classify_backchannel("mhmm", energy=None) is Backchannel.CONTINUER


def test_loud_substantive_unaffected_by_gate():
    """Energy never *upgrades* substantive speech to a continuer."""
    assert (
        classify_backchannel("what do you think", energy=0.05)
        is Backchannel.SUBSTANTIVE
    )


def test_custom_energy_ceiling():
    assert (
        classify_backchannel("yeah", energy=0.2, energy_ceiling=0.1)
        is Backchannel.SUBSTANTIVE
    )
    assert (
        classify_backchannel("yeah", energy=0.2, energy_ceiling=0.5)
        is Backchannel.CONTINUER
    )


# ---- custom word limit -------------------------------------------------


def test_custom_max_words_allows_longer_continuer_phrases():
    assert (
        classify_backchannel("yeah yeah yeah", max_words=3) is Backchannel.CONTINUER
    )


def test_custom_max_words_can_tighten():
    assert classify_backchannel("i see", max_words=1) is Backchannel.SUBSTANTIVE


# ---- is_continuer convenience ------------------------------------------


def test_is_continuer_true_and_false():
    assert is_continuer("mhmm") is True
    assert is_continuer("what should i do") is False
    assert is_continuer("") is False


def test_is_continuer_forwards_energy_and_kwargs():
    assert is_continuer("yeah", energy=0.9) is False
    assert is_continuer("yeah yeah yeah", max_words=3) is True


# ---- lexicon / constants sanity ----------------------------------------


def test_defaults_are_sane():
    assert DEFAULT_MAX_CONTINUER_WORDS == 2
    assert 0.0 < DEFAULT_ENERGY_CEILING < 1.0


def test_lexicon_is_lowercase_and_nonempty():
    assert CONTINUER_LEXICON
    assert all(w == w.lower() for w in CONTINUER_LEXICON)


def test_classifier_return_type_is_enum():
    assert isinstance(classify_backchannel("yeah"), Backchannel)
