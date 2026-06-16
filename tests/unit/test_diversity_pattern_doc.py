"""Tests for iter-129 — keep the diversity-check pattern doc in
sync with its actual instances.

GENO.md's "Session-summary diversity-check pattern" section
lists the helpers that follow the template. If a 5th instance
lands without doc update, the doc-vs-code drift test fails.
Same shape as iter-117's audio-fixture sanity tests: structural
invariants that always run.

The opposite drift (a helper is documented but missing from the
codebase) also fails — protects against accidental rename
without doc-update.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples import _chat_metrics  # noqa: E402

GENO_MD = ROOT / "GENO.md"


# Names of the helpers that ARE diversity-check instances. This
# list is the source of truth — when a 6th iteration lands,
# update this list AND the GENO.md template count + bullet list
# atomically. If they fall out of sync, the doc-sync tests fire.
_DIVERSITY_HELPERS = (
    "_emit_filler_diversity_line",         # iter-114
    "_emit_naturalness_consistency_line",  # iter-115 / iter-126
    "_emit_barge_phase_consistency_line",  # iter-120
    "_emit_sentence_length_consistency_line",  # iter-128
    "_emit_stt_rtf_consistency_line",      # iter-140
    "_emit_tts_rtf_consistency_line",      # iter-141
)


def _read_doc() -> str:
    return GENO_MD.read_text()


# ---- Doc presence -------------------------------------------------------


def test_geno_md_has_diversity_pattern_section():
    """The promotion iteration only succeeded if the section is
    actually in GENO.md."""
    doc = _read_doc()
    assert "Session-summary diversity-check pattern" in doc


def test_doc_references_run_finder_helper():
    """The pattern depends on iter-116's
    `_longest_consecutive_run`. Mentioning it explicitly lets a
    future reader connect the pattern to its primitive."""
    doc = _read_doc()
    assert "_longest_consecutive_run" in doc


# ---- Each helper exists in the codebase --------------------------------


def test_each_documented_helper_exists():
    """Every helper named in `_DIVERSITY_HELPERS` exists as an
    importable function in `_chat_metrics`. Catches a rename
    that didn't update this list."""
    missing = [
        name for name in _DIVERSITY_HELPERS
        if not hasattr(_chat_metrics, name)
    ]
    assert not missing, (
        f"helpers listed in _DIVERSITY_HELPERS but missing from "
        f"_chat_metrics: {missing}"
    )


def test_each_documented_helper_is_callable():
    """Sanity that each is a callable (not just any attribute)."""
    for name in _DIVERSITY_HELPERS:
        attr = getattr(_chat_metrics, name)
        assert callable(attr), f"{name} is not callable"


# ---- Doc references each helper iter ----------------------------------


def test_doc_references_each_responsible_iteration():
    """The pattern bullet list mentions each iteration that
    contributed an instance OR a meaningful fix. Not strict
    full-name-match — just verifying the iter numbers appear
    so a reader can find them in ITERATION_LOG.md.

    iter-131: iter-126 added — it's the fix-iter for iter-115's
    documented limitation, and the doc references it via
    "iter-115/126 naturalness". The attribution should survive
    future doc edits.
    """
    doc = _read_doc()
    expected_iters = [
        "iter-114",  # filler diversity
        "iter-115",  # naturalness (initial)
        "iter-120",  # barge-phase
        "iter-126",  # naturalness (filter fix; iter-131 added to list)
        "iter-128",  # sentence-length (continuous-metric instance)
        "iter-140",  # stt-rtf (2nd continuous-metric instance)
        "iter-141",  # tts-rtf (3rd continuous-metric instance)
    ]
    for it in expected_iters:
        assert it in doc, f"missing {it} attribution in GENO.md"


def test_doc_template_lists_consistent_threshold_conventions():
    """The threshold-convention bullet calls out 3 / 4 / 5 with
    each pinned to its responsible iter. Simple substring
    check; if the bullet is rewritten without preserving the
    iter mappings, this fires."""
    doc = _read_doc()
    # Match the canonical threshold/iter patterns.
    assert re.search(r"3\s*\(iter-114", doc), (
        "doc no longer pins threshold=3 to iter-114 (filler)"
    )
    assert re.search(r"4\s*\(iter-120", doc), (
        "doc no longer pins threshold=4 to iter-120 (barge-phase)"
    )
    assert re.search(r"5\s*\(iter-115.*iter-128", doc, re.DOTALL), (
        "doc no longer pins threshold=5 to iter-115 + iter-128"
    )


# ---- Pattern count -----------------------------------------------------


def test_doc_template_claims_match_actual_instance_count():
    """The doc says 'four instances confirm it'. If the
    `_DIVERSITY_HELPERS` list grows without doc update, this
    test fires — forcing the doc to be amended atomically with
    the code change."""
    doc = _read_doc()
    # Map count → expected English numeral (extend if a 5th
    # instance lands).
    counts = {
        4: ("four instances", "Four"),
        5: ("five instances", "Five"),
        6: ("six instances", "Six"),
    }
    actual = len(_DIVERSITY_HELPERS)
    if actual not in counts:
        # A future iteration may bump us past 5; the test should
        # gently nudge the contributor to expand this map.
        assert False, (
            f"add a 'four/five/...' entry to the counts dict in "
            f"this test — current count {actual} is not handled"
        )
    expected_phrase, _ = counts[actual]
    assert expected_phrase in doc.lower(), (
        f"GENO.md says different number of instances than "
        f"_DIVERSITY_HELPERS lists ({actual}): expected "
        f"{expected_phrase!r}, got doc that doesn't mention it"
    )


# ---- All instances are tested ----------------------------------------


def test_doc_references_each_callable_name():
    """iter-131: backport from iter-130's extraction-pattern
    sentinel. Each documented helper is name-dropped somewhere
    in the doc — readers can search for the helper name. If a
    future contributor renames a helper without updating the
    doc, this fires."""
    doc = _read_doc()
    for name in _DIVERSITY_HELPERS:
        assert name in doc, f"missing reference to {name} in GENO.md"


def test_doc_lists_8_numbered_rules():
    """iter-131: backport from iter-130. The pattern section
    enumerates exactly 8 rules. If someone adds a 9th rule,
    this nudges them to confirm the rule generalizes across
    all instances (and to update this test)."""
    doc = _read_doc()
    section_start = doc.index("### Session-summary diversity-check pattern")
    # The diversity section is the LAST pattern section in
    # GENO.md — it ends at the next "## " (top-level header).
    # Search forward from the section start for the next
    # top-level heading.
    section_text = doc[section_start:]
    next_top = section_text.find("\n## ")
    if next_top != -1:
        section_text = section_text[:next_top]
    numbered_lines = re.findall(
        r"^\d+\.\s+\*\*", section_text, re.MULTILINE,
    )
    assert len(numbered_lines) == 8, (
        f"expected 8 numbered rules, got {len(numbered_lines)}: "
        f"{numbered_lines!r}"
    )


def test_each_helper_has_a_corresponding_test_file():
    """For each documented helper, a matching test file should
    exist in tests/unit/. Catches an instance landing without
    its own test coverage. The naming convention so far is
    `tests/unit/test_<helper_name>.py`.
    """
    test_dir = ROOT / "tests" / "unit"
    missing = []
    for name in _DIVERSITY_HELPERS:
        candidate = test_dir / f"test{name}.py"
        if not candidate.exists():
            missing.append(str(candidate.name))
    assert not missing, (
        f"helpers without dedicated test files: {missing}"
    )
