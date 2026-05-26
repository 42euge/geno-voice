"""Tests for iter-136 — keep README's benchmark section in
sync with the actual CLI.

iter-129 + iter-130 introduced the drift-sentinel pattern for
GENO.md pattern docs. iter-136 applies the same shape to the
README's "Evaluating a new STT engine" section: if the CLI
flags / format choices / corpus fixtures change without updating
the README, this test fires.

The README is operator-facing documentation. Stale CLI examples
are worse than no examples — operators copy what they see.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
SCRIPT_PATH = ROOT / "scripts" / "run_stt_benchmark.py"
CORPUS_PATH = ROOT / "tests" / "fixtures" / "wer" / "corpus.json"

sys.path.insert(0, str(ROOT))


def _read_readme() -> str:
    return README.read_text()


def _read_script_source() -> str:
    """Read the script as text. Avoids importing it (which
    would conflict with `@dataclass` lookups when the module is
    loaded under a non-canonical name)."""
    return SCRIPT_PATH.read_text()


# ---- Section presence -----------------------------------------------


def test_readme_has_benchmark_section():
    """The promoted documentation iter only succeeded if the
    section is in the README."""
    doc = _read_readme()
    assert "Evaluating a new STT engine" in doc


def test_readme_references_benchmark_script_path():
    """Operators following the docs must find a working script
    path. If the file is moved/renamed, this fires."""
    doc = _read_readme()
    assert "scripts/run_stt_benchmark.py" in doc


def test_referenced_script_exists():
    """The README points at scripts/run_stt_benchmark.py — it
    must be a real file."""
    assert SCRIPT_PATH.exists(), f"{SCRIPT_PATH} missing"


# ---- CLI flags documented ------------------------------------------


def test_readme_documents_each_supported_format():
    """The text/json/csv format choices are all mentioned in
    the README's Output formats list."""
    doc = _read_readme()
    for fmt in ["text", "json", "csv"]:
        assert fmt in doc, f"format {fmt!r} not mentioned in README"


def test_readme_format_choices_match_argparse_choices():
    """Whatever choices the script accepts via --format must
    each appear in the README. Catches a future iteration that
    adds a new format (e.g., html, markdown) without doc update.
    """
    src = _read_script_source()
    import re as _re
    match = _re.search(r'choices=\[([^\]]+)\]', src)
    assert match, "could not find argparse choices in script"
    choices_text = match.group(1)
    choices = _re.findall(r'"([^"]+)"', choices_text)
    assert choices, "could not parse choices list"

    doc = _read_readme()
    for choice in choices:
        assert choice in doc, (
            f"--format choice {choice!r} accepted by script but "
            f"not documented in README"
        )


def test_readme_documents_diff_flag():
    """--diff is the headline workflow for iter-134/135. The
    CLI section in the README must reference it."""
    doc = _read_readme()
    assert "--diff" in doc


def test_readme_documents_engine_flag():
    """--engine is the canonical entry point. Not optional."""
    doc = _read_readme()
    assert "--engine" in doc


# ---- Corpus references ---------------------------------------------


def test_readme_mentions_corpus_path():
    """The corpus location should be discoverable from the
    README."""
    doc = _read_readme()
    assert "tests/fixtures/wer" in doc


def test_referenced_corpus_exists():
    assert CORPUS_PATH.exists(), f"{CORPUS_PATH} missing"


def test_readme_corpus_count_matches_actual():
    """The README states the corpus has '5 audio fixtures'.
    If a fixture is added/removed from corpus.json, the README
    must be updated. Catches stale counts."""
    doc = _read_readme()
    with CORPUS_PATH.open() as f:
        corpus = json.load(f)
    actual_count = len(corpus.get("audio_fixtures", []))

    # Map English numerals to actual count. Update this dict
    # if the corpus grows past 9 fixtures.
    numerals = {
        3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    }
    if actual_count not in numerals:
        assert False, (
            f"corpus has {actual_count} fixtures — extend the "
            f"numerals dict in this test"
        )
    expected_phrase = f"{numerals[actual_count]} audio fixtures"
    assert expected_phrase in doc, (
        f"README states a different corpus count than corpus.json "
        f"has ({actual_count}): expected mention of "
        f"{expected_phrase!r}"
    )


# ---- Iteration attribution -----------------------------------------


def test_referenced_corpus_has_audio_fixtures_section():
    """The README's CI integration claim depends on the corpus
    having audio fixtures (not just text). If the section
    disappears, the documented workflow stops working."""
    with CORPUS_PATH.open() as f:
        corpus = json.load(f)
    assert "audio_fixtures" in corpus, (
        "corpus.json missing audio_fixtures section — README's "
        "benchmark documentation will produce no output"
    )
    assert len(corpus["audio_fixtures"]) > 0


# ---- Code block sanity ---------------------------------------------


def test_readme_code_blocks_use_real_script_invocation():
    """Every code block that shows a `python scripts/...` line
    must reference the actual script path. Catches a typo'd
    script name like `python scripts/run_benchmark.py` (which
    would silently fail to find the script)."""
    doc = _read_readme()
    import re as _re
    # Find all `python scripts/<something>` invocations in code
    # blocks and bare lines.
    invocations = _re.findall(
        r'python\s+(scripts/[^\s\\]+)',
        doc,
    )
    # The intended script path:
    expected = "scripts/run_stt_benchmark.py"
    for path in invocations:
        # Strip trailing punctuation / args.
        bare = path.rstrip(",.")
        assert bare == expected, (
            f"README references {bare!r}; expected {expected!r}"
        )
