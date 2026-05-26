"""Tests for iter-130 — keep the mic_chat extraction pattern doc
in sync with its actual instances.

iter-129 introduced the drift-sentinel test pattern (catches
when documented helper lists diverge from real code) and
applied it to GENO.md's "Session-summary diversity-check
pattern" section. iter-130 applies the same shape to the
sibling section, "mic_chat.py extraction pattern."

The drift this iteration FIXES:
  iter-110's `run_session` extraction follows the same 5-rule
  shape as iter-107/108/109 but was never added to the doc.
  iter-130 updates the count to 4 and adds the test that
  prevents future drift.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GENO_MD = ROOT / "GENO.md"


# Source of truth for the extraction-pattern instances. Each
# entry: (iter_number, module_path, public_callable_name).
# Adding a 5th instance is a one-line append here PLUS updating
# the GENO.md count + iter list. The tests below verify both
# happen atomically.
_EXTRACTION_INSTANCES = (
    ("iter-107", "examples._chat_fillers",   "prerender_fillers"),
    ("iter-108", "examples._chat_engines",   "load_engines"),
    ("iter-109", "examples._chat_audio_io",  "build_audio_io"),
    ("iter-110", "examples._chat_session",   "run_session"),
)


def _read_doc() -> str:
    return GENO_MD.read_text()


# ---- Doc presence -----------------------------------------------------


def test_geno_md_has_extraction_pattern_section():
    doc = _read_doc()
    assert "mic_chat.py extraction pattern" in doc


def test_doc_mentions_run_chat():
    """The pattern documents extractions FROM run_chat
    specifically. Section text should reference it."""
    doc = _read_doc()
    assert "run_chat" in doc


# ---- Each instance exists in the codebase ----------------------------


def test_each_documented_instance_module_imports():
    """Every module named in `_EXTRACTION_INSTANCES` is
    importable. Catches a module rename that didn't update the
    list."""
    failed = []
    for iter_num, module_path, _ in _EXTRACTION_INSTANCES:
        try:
            importlib.import_module(module_path)
        except Exception as e:
            failed.append(f"{iter_num} {module_path}: {e!r}")
    assert not failed, "\n".join(failed)


def test_each_documented_instance_has_callable():
    """Beyond import, the public callable named in each entry
    must exist + be callable."""
    failed = []
    for iter_num, module_path, callable_name in _EXTRACTION_INSTANCES:
        mod = importlib.import_module(module_path)
        attr = getattr(mod, callable_name, None)
        if attr is None:
            failed.append(
                f"{iter_num}: {module_path}.{callable_name} not found"
            )
            continue
        if not callable(attr):
            failed.append(
                f"{iter_num}: {module_path}.{callable_name} is not callable"
            )
    assert not failed, "\n".join(failed)


# ---- Doc references each iter ---------------------------------------


def test_doc_references_each_iter_in_section():
    """The pattern bullet list in GENO.md mentions every
    iter that contributed an instance."""
    doc = _read_doc()
    for iter_num, _, _ in _EXTRACTION_INSTANCES:
        assert iter_num in doc, (
            f"GENO.md missing {iter_num} attribution"
        )


def test_doc_references_each_callable_name():
    """Each callable is name-dropped (in backticks) somewhere
    in the doc — readers can search for the helper name."""
    doc = _read_doc()
    for _, _, name in _EXTRACTION_INSTANCES:
        # Allow either backticked or plain mention — the
        # specific format isn't load-bearing as long as the
        # name appears.
        assert name in doc, (
            f"GENO.md missing reference to {name}"
        )


# ---- Pattern count ---------------------------------------------------


def test_doc_template_claims_match_actual_instance_count():
    """The doc says 'four instances confirm it' (post-iter-130).
    If `_EXTRACTION_INSTANCES` grows without doc update, this
    fires.
    """
    doc = _read_doc()
    counts = {
        3: ("three instances", "Three"),
        4: ("four instances", "Four"),
        5: ("five instances", "Five"),
        6: ("six instances", "Six"),
    }
    actual = len(_EXTRACTION_INSTANCES)
    if actual not in counts:
        assert False, (
            f"add a numeral entry to the counts dict — current "
            f"count {actual} not handled"
        )
    expected_phrase, _ = counts[actual]
    # Look for the lowercase phrase in lowercased doc — the doc
    # could capitalize it ("Four instances...") or use lowercase
    # ("four instances confirm it").
    assert expected_phrase in doc.lower(), (
        f"GENO.md says different number of instances than "
        f"_EXTRACTION_INSTANCES lists ({actual}): expected "
        f"{expected_phrase!r}, got doc that doesn't mention it"
    )


# ---- Rules-vs-instances cross-check --------------------------------


def test_doc_lists_5_numbered_rules():
    """The pattern section enumerates exactly 5 rules. If
    someone adds a 6th rule, this nudges them to update the
    test (and confirm the rule generalizes across all
    instances)."""
    doc = _read_doc()
    section_start = doc.index("### mic_chat.py extraction pattern")
    next_section = doc.index("### Session-summary diversity-check pattern")
    section_text = doc[section_start:next_section]
    # Match lines like "1. **...**" at top level. Numbered
    # rules in this section start with "1." through "5.".
    numbered_lines = re.findall(r"^\d+\.\s+\*\*", section_text, re.MULTILINE)
    assert len(numbered_lines) == 5, (
        f"expected 5 numbered rules, got {len(numbered_lines)}: "
        f"{numbered_lines!r}"
    )


# ---- Tests-for-instances cross-check ----------------------------


def test_each_instance_has_a_corresponding_test_file():
    """For each documented helper, a matching test file should
    exist in tests/unit/. Catches an extraction landing without
    its own test coverage. Naming convention: helper module
    name (drop the underscore prefix) → test file."""
    test_dir = ROOT / "tests" / "unit"
    missing = []
    for iter_num, module_path, _ in _EXTRACTION_INSTANCES:
        # Module is e.g. "examples._chat_fillers". Test file
        # candidates: test_chat_fillers.py.
        bare = module_path.split(".")[-1].lstrip("_")
        candidate = test_dir / f"test_{bare}.py"
        if not candidate.exists():
            missing.append(
                f"{iter_num} {module_path}: expected {candidate.name}"
            )
    assert not missing, "\n".join(missing)
