"""Tests for iter-144 — keep the README's "Contributing patterns"
section in sync with the GENO.md pattern docs it points at.

iter-136 carried a next-direction forward across many laps:
"Document the GENO.md diversity-check + extraction patterns in the
README so contributors discover them without reading GENO.md." This
lap lands it. The README now has a "Contributing patterns" section
that links the two GENO.md pattern sections and their guarding
test files.

Same drift-sentinel shape as iter-129/130/136:
- The README must name both GENO.md pattern sections (so a contributor
  can find them) — and those sections must actually exist in GENO.md.
- The README must name both guarding test files — and those files
  must actually exist.

The README is contributor-facing. A pointer to a section that was
renamed, or to a test file that was deleted, is worse than no pointer:
it sends a reader looking for something that isn't there.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
GENO_MD = ROOT / "GENO.md"
TEST_DIR = ROOT / "tests" / "unit"

# The two GENO.md pattern sections the README points contributors at.
# Each entry: (section_heading_in_geno_md, guarding_test_filename).
# Source of truth — if a third pattern section earns a README pointer,
# append here and update the README atomically.
_PATTERN_SECTIONS = (
    ("mic_chat.py extraction pattern", "test_extraction_pattern_doc.py"),
    (
        "Session-summary diversity-check pattern",
        "test_diversity_pattern_doc.py",
    ),
)


def _read_readme() -> str:
    return README.read_text()


def _read_geno_md() -> str:
    return GENO_MD.read_text()


# ---- Section presence -------------------------------------------------


def test_readme_has_contributing_patterns_section():
    """The promotion iteration only succeeded if the section is
    actually in the README."""
    doc = _read_readme()
    assert "## Contributing patterns" in doc


def test_readme_links_to_geno_md():
    """The section's whole point is to send readers to GENO.md for
    the full pattern text. The link must be present."""
    doc = _read_readme()
    assert "GENO.md" in doc


# ---- Each named section exists in GENO.md ----------------------------


def test_readme_names_each_pattern_section():
    """The README must name each GENO.md pattern section so a
    contributor can search for it."""
    doc = _read_readme()
    missing = [
        section for section, _ in _PATTERN_SECTIONS if section not in doc
    ]
    assert not missing, (
        f"README's Contributing patterns section doesn't name: {missing}"
    )


def test_each_named_section_exists_in_geno_md():
    """Every section the README points at must actually be a heading
    in GENO.md. Catches a GENO.md rename that didn't update the
    README pointer."""
    geno = _read_geno_md()
    missing = [
        section
        for section, _ in _PATTERN_SECTIONS
        if f"### {section}" not in geno
    ]
    assert not missing, (
        f"README points at GENO.md sections that no longer exist as "
        f"headings: {missing}"
    )


# ---- Each guarding test file is named + exists -----------------------


def test_readme_names_each_guarding_test_file():
    """The README claims both pattern sections are guarded by
    drift-sentinel tests. It must name each guarding test file so a
    reader can find it."""
    doc = _read_readme()
    missing = [
        test_file
        for _, test_file in _PATTERN_SECTIONS
        if test_file not in doc
    ]
    assert not missing, (
        f"README doesn't name guarding test files: {missing}"
    )


def test_each_guarding_test_file_exists():
    """Every test file the README references as a guarding sentinel
    must actually exist. Catches a test rename/delete that left the
    README pointing at nothing."""
    missing = [
        test_file
        for _, test_file in _PATTERN_SECTIONS
        if not (TEST_DIR / test_file).exists()
    ]
    assert not missing, (
        f"README names guarding test files that don't exist: {missing}"
    )


# ---- Section count -----------------------------------------------------


def test_readme_documents_both_pattern_sections():
    """Sanity: there are exactly two GENO.md pattern sections today.
    If a third is added (and earns a README pointer), this nudges the
    contributor to extend `_PATTERN_SECTIONS` and the README together.
    """
    assert len(_PATTERN_SECTIONS) == 2, (
        "extend the README's Contributing patterns section and this "
        "test together if the pattern-section count changes"
    )
