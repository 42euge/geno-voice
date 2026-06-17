"""Tests for iter-150 — the rule-based text EOU precursor (backlog #4).

``session/text_eou.py`` lowers turn-end confidence when a transcript trails
off on a conjunction / dangling preposition / filler / ellipsis — the cheap,
dependency-free precursor to a learned text turn-detector (LiveKit-style). Its
output multiplies the silence-derived confidence from the iter-149 seam, so it
can only *dampen* confidence on evidence of incompleteness, never raise it.

``utterance_completeness`` / ``is_utterance_complete`` / ``TextAwareTurnDecider``
are pure (no I/O, no clock), so these tests drive them directly with injected
text + silence.

``text_eou`` does ``from session.turn_decider import ...`` at module scope.
Importing ``session`` normally runs ``session/__init__`` which eagerly pulls
pipecat (absent on the x86_64 Linux runner). So we build a minimal ``session``
namespace package and load ``turn_decider`` + ``text_eou`` into it by file path
— the same trick test_turn_decider.py's engine-integration tests use, and how
the mic_* / backchannel tests keep platform deps out of the unit path.
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
    _pkg.__path__ = []  # mark as a package
    sys.modules["session"] = _pkg
if "session.turn_decider" not in sys.modules:
    _load_by_path("session.turn_decider", "turn_decider.py", package="session")
_eou = _load_by_path("session.text_eou", "text_eou.py", package="session")

TextEOUConfig = _eou.TextEOUConfig
utterance_completeness = _eou.utterance_completeness
is_utterance_complete = _eou.is_utterance_complete
TextAwareTurnDecider = _eou.TextAwareTurnDecider
CONJUNCTION_MARKERS = _eou.CONJUNCTION_MARKERS
DANGLING_MARKERS = _eou.DANGLING_MARKERS
FILLER_MARKERS = _eou.FILLER_MARKERS

_silence = sys.modules["session.turn_decider"].silence_confidence
TurnDeciderConfig = sys.modules["session.turn_decider"].TurnDeciderConfig


# ---------------------------------------------------------------------------
# utterance_completeness — complete utterances score 1.0
# ---------------------------------------------------------------------------

class TestCompleteUtterances:
    def test_plain_sentence_is_complete(self):
        assert utterance_completeness("I think that's it.") == 1.0

    def test_sentence_without_terminal_punctuation_is_complete(self):
        assert utterance_completeness("that is what I decided") == 1.0

    def test_question_is_complete(self):
        assert utterance_completeness("what do you think?") == 1.0

    def test_exclamation_is_complete(self):
        assert utterance_completeness("that's amazing!") == 1.0

    def test_empty_is_complete_no_text_evidence(self):
        # No text ⇒ no dampening; silence alone decides.
        assert utterance_completeness("") == 1.0

    def test_whitespace_only_is_complete(self):
        assert utterance_completeness("   \n\t ") == 1.0

    def test_word_containing_marker_substring_is_complete(self):
        # "android" ends in "...id" but the whole token isn't a marker.
        assert utterance_completeness("I bought an android") == 1.0

    def test_noun_after_article_is_complete(self):
        assert utterance_completeness("I'll take the deadline") == 1.0


# ---------------------------------------------------------------------------
# utterance_completeness — incomplete: conjunctions (strongest)
# ---------------------------------------------------------------------------

class TestConjunctionEndings:
    def test_trailing_and(self):
        c = utterance_completeness("I went to the store and")
        assert c == TextEOUConfig().conjunction_completeness

    def test_trailing_because(self):
        assert utterance_completeness("I left because") == TextEOUConfig().conjunction_completeness

    def test_trailing_but(self):
        assert utterance_completeness("I wanted to but") == TextEOUConfig().conjunction_completeness

    def test_case_insensitive(self):
        assert utterance_completeness("And then we left AND") == TextEOUConfig().conjunction_completeness

    def test_trailing_conjunction_with_comma(self):
        # Comma after the conjunction still resolves to the conjunction word.
        assert utterance_completeness("I tried, and,") == TextEOUConfig().conjunction_completeness

    def test_every_conjunction_marker_dampens(self):
        cfg = TextEOUConfig()
        for w in CONJUNCTION_MARKERS:
            assert utterance_completeness(f"something {w}") == cfg.conjunction_completeness


# ---------------------------------------------------------------------------
# utterance_completeness — incomplete: dangling function words
# ---------------------------------------------------------------------------

class TestDanglingEndings:
    def test_trailing_to(self):
        assert utterance_completeness("I want to") == TextEOUConfig().dangling_completeness

    def test_trailing_the(self):
        assert utterance_completeness("can you pass the") == TextEOUConfig().dangling_completeness

    def test_trailing_with_my(self):
        assert utterance_completeness("I'm going with my") == TextEOUConfig().dangling_completeness

    def test_every_dangling_marker_dampens(self):
        cfg = TextEOUConfig()
        for w in DANGLING_MARKERS:
            # Skip words that are also conjunctions (resolved earlier) — none
            # in DANGLING are, but guard anyway for clarity.
            if w in CONJUNCTION_MARKERS:
                continue
            assert utterance_completeness(f"give it {w}") == cfg.dangling_completeness


# ---------------------------------------------------------------------------
# utterance_completeness — incomplete: fillers
# ---------------------------------------------------------------------------

class TestFillerEndings:
    def test_trailing_um(self):
        assert utterance_completeness("I was thinking um") == TextEOUConfig().filler_completeness

    def test_trailing_like(self):
        assert utterance_completeness("it was like") == TextEOUConfig().filler_completeness

    def test_filler_only_dampens(self):
        assert utterance_completeness("uh") == TextEOUConfig().filler_completeness


# ---------------------------------------------------------------------------
# utterance_completeness — ellipsis + comma
# ---------------------------------------------------------------------------

class TestEllipsisAndComma:
    def test_three_dot_ellipsis(self):
        assert utterance_completeness("I don't know...") == TextEOUConfig().ellipsis_completeness

    def test_unicode_ellipsis(self):
        assert utterance_completeness("well…") == TextEOUConfig().ellipsis_completeness

    def test_ellipsis_with_trailing_space(self):
        assert utterance_completeness("hmm ... ") == TextEOUConfig().ellipsis_completeness

    def test_single_period_is_complete(self):
        # One period is a normal sentence end, not an ellipsis.
        assert utterance_completeness("that's it.") == 1.0

    def test_trailing_comma(self):
        assert utterance_completeness("first I did this,") == TextEOUConfig().comma_completeness


# ---------------------------------------------------------------------------
# utterance_completeness — precedence between overlapping classes
# ---------------------------------------------------------------------------

class TestPrecedence:
    def test_so_is_conjunction_not_filler(self):
        # "so" is in both CONJUNCTION and FILLER; conjunction is checked first.
        assert "so" in CONJUNCTION_MARKERS and "so" in FILLER_MARKERS
        assert utterance_completeness("I was thinking so") == TextEOUConfig().conjunction_completeness

    def test_ellipsis_beats_trailing_word(self):
        # Ellipsis is checked before the last-word class; "and..." ⇒ ellipsis.
        assert utterance_completeness("I went and...") == TextEOUConfig().ellipsis_completeness

    def test_word_marker_beats_comma(self):
        assert utterance_completeness("I went to,") == TextEOUConfig().dangling_completeness


# ---------------------------------------------------------------------------
# is_utterance_complete — boolean convenience
# ---------------------------------------------------------------------------

class TestIsUtteranceComplete:
    def test_complete_sentence_true(self):
        assert is_utterance_complete("I think that's it.") is True

    def test_trailing_conjunction_false(self):
        assert is_utterance_complete("I went to the store and") is False

    def test_comma_at_default_threshold_is_complete(self):
        # comma_completeness (0.6) == default threshold (0.6) ⇒ complete (>=).
        assert is_utterance_complete("first I did this,") is True

    def test_empty_is_complete(self):
        assert is_utterance_complete("") is True

    def test_custom_threshold_flips_comma(self):
        cfg = TextEOUConfig(complete_threshold=0.7)
        assert is_utterance_complete("first I did this,", cfg) is False


# ---------------------------------------------------------------------------
# TextEOUConfig — validation + custom values
# ---------------------------------------------------------------------------

class TestTextEOUConfig:
    def test_is_frozen(self):
        cfg = TextEOUConfig()
        with pytest.raises(Exception):
            cfg.conjunction_completeness = 0.9  # type: ignore[misc]

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            TextEOUConfig(conjunction_completeness=1.5)
        with pytest.raises(ValueError):
            TextEOUConfig(dangling_completeness=-0.1)

    def test_custom_completeness_used(self):
        cfg = TextEOUConfig(conjunction_completeness=0.05)
        assert utterance_completeness("I left and", cfg) == 0.05

    def test_default_strength_ordering(self):
        # Documented ordering: conjunction strongest ... comma weakest.
        cfg = TextEOUConfig()
        assert (
            cfg.conjunction_completeness
            < cfg.dangling_completeness
            < cfg.filler_completeness
            < cfg.ellipsis_completeness
            <= cfg.comma_completeness
        )


# ---------------------------------------------------------------------------
# TextAwareTurnDecider — combines silence + text, same interface
# ---------------------------------------------------------------------------

class TestTextAwareTurnDecider:
    def test_complete_text_matches_silence_only(self):
        d = TextAwareTurnDecider()
        s = 4.5
        assert d.confidence(
            silence_duration_secs=s, transcript_chunk="that's my decision."
        ) == _silence(s)

    def test_no_transcript_matches_silence_only(self):
        d = TextAwareTurnDecider()
        for s in (0.0, 3.0, 4.5, 6.0, 12.0):
            assert d.confidence(silence_duration_secs=s) == _silence(s)

    def test_empty_transcript_matches_silence_only(self):
        d = TextAwareTurnDecider()
        assert d.confidence(silence_duration_secs=4.5, transcript_chunk="") == _silence(4.5)

    def test_incomplete_text_dampens_below_silence_only(self):
        d = TextAwareTurnDecider()
        s = 4.5
        full = _silence(s)
        damped = d.confidence(silence_duration_secs=s, transcript_chunk="I went to the store and")
        assert damped < full
        assert damped == pytest.approx(full * TextEOUConfig().conjunction_completeness)

    def test_dampening_never_exceeds_silence(self):
        d = TextAwareTurnDecider()
        for s in (2.5, 3.0, 4.0, 4.5, 5.5):
            for text in ("done.", "I want to", "um", "well...", "list this,"):
                assert d.confidence(silence_duration_secs=s, transcript_chunk=text) <= _silence(s)

    def test_zero_silence_stays_zero_regardless_of_text(self):
        # Below the silence floor confidence is 0.0; dampening can't go negative.
        d = TextAwareTurnDecider()
        assert d.confidence(silence_duration_secs=0.0, transcript_chunk="and because") == 0.0

    def test_confidence_is_keyword_only(self):
        d = TextAwareTurnDecider()
        with pytest.raises(TypeError):
            d.confidence(4.0)  # type: ignore[misc]

    def test_uses_injected_silence_config(self):
        d = TextAwareTurnDecider(
            silence_config=TurnDeciderConfig(silence_floor_secs=0.0, silence_ceiling_secs=2.0)
        )
        # silence_confidence(1.0) == 0.5 with this band; complete text ⇒ no damp.
        assert d.confidence(
            silence_duration_secs=1.0, transcript_chunk="done."
        ) == pytest.approx(0.5)

    def test_uses_injected_text_config(self):
        d = TextAwareTurnDecider(text_config=TextEOUConfig(conjunction_completeness=0.0))
        # Conjunction ending fully zeroes the confidence with this text config.
        assert d.confidence(silence_duration_secs=4.5, transcript_chunk="I left and") == 0.0

    def test_matches_silence_turn_decider_interface(self):
        # Same keyword-only signature as SilenceTurnDecider — drop-in swap.
        from session.turn_decider import SilenceTurnDecider
        td = SilenceTurnDecider()
        ta = TextAwareTurnDecider()
        # With complete text the two agree exactly.
        s = 4.5
        assert ta.confidence(
            silence_duration_secs=s, transcript_chunk="all done."
        ) == td.confidence(silence_duration_secs=s, transcript_chunk="all done.")


# ---------------------------------------------------------------------------
# Engine integration: a trailing-off transcript keeps the engine silent where
# the silence-only confidence would have fired a cue.
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    @classmethod
    def _load_engine(cls):
        try:
            if "session.triggers" not in sys.modules:
                _load_by_path("session.triggers", "triggers.py", package="session")
            return _load_by_path("session.turn_taking", "turn_taking.py", package="session")
        except Exception:
            return None

    def test_trailing_conjunction_suppresses_cue_that_silence_alone_fires(self):
        tt = self._load_engine()
        if tt is None:
            pytest.skip("turn_taking unavailable on this runner")

        engine = tt.TurnTakingEngine()
        engine.state.session_start = tt.time.time() - 300
        engine.update_state(user_spoke_secs=30)

        silence = 4.5  # inside [backchannel_min=4, response_min=6)
        decider = TextAwareTurnDecider()

        # Complete utterance: confidence clears backchannel threshold ⇒ PLAY_CUE.
        complete_conf = decider.confidence(
            silence_duration_secs=silence, transcript_chunk="that's my whole point."
        )
        assert engine.decide(silence, complete_conf).action == tt.Action.PLAY_CUE

        # Same silence, but trailing off on a conjunction: dampened below the
        # backchannel threshold ⇒ STAY_SILENT (don't barge into a mid-thought).
        engine2 = tt.TurnTakingEngine()
        engine2.state.session_start = tt.time.time() - 300
        engine2.update_state(user_spoke_secs=30)
        incomplete_conf = decider.confidence(
            silence_duration_secs=silence, transcript_chunk="I was going to say that and"
        )
        assert incomplete_conf < complete_conf
        assert engine2.decide(silence, incomplete_conf).action == tt.Action.STAY_SILENT
