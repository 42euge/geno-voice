"""Tests for iter-322 — keep ``docs/research/organic-turn-taking.md`` honest
about the artifacts it references.

This is the third research-doc drift sentinel, but a *lighter* shape than the
iter-320 (``tts-pacing-mirror.md``) / iter-321 (``voice-capture-tuning.md``)
command-parse sentinels — and deliberately so. Those two docs are CLI-bearing:
they show dozens of copy-pasteable ``gv`` examples, so the right guard parses
each example through the real ``build_parser``. The organic-turn-taking doc is
different: it is the largest research doc (~1860 lines) but it is
**design/narrative** — its body is a SOTA landscape, a pipeline map, a
prioritized backlog, and a per-lap findings log. It shows essentially **no**
runnable ``gv`` examples (iter-321's next-item observed this directly).

So its drift risk is **structural, not syntactic**. Across ~30 organic laps
(iters 148→180) the findings log accreted dense references to the modules,
test files, and entrypoints each lap shipped — ``session/backchannel.py``,
``examples/_chat_loop.py``, ``tests/unit/test_utterance_buffer.py``, and so on.
A reader following the narrative clicks through to those files. When a module is
renamed or a test file deleted, those references rot **silently** — the doc
still reads fine, but every path it names is a dead link.

This sentinel **extracts every path-qualified ``*.py`` reference** the doc makes
(``dir/.../file.py`` form, the shape that is unambiguously a repo path rather
than a bare basename) and asserts each one **still exists** relative to the repo
root. A renamed/removed module turns this red. It also pins the doc's structural
anchors (the backlog + findings-log sections, the nav entry) so a heading rename
or a dropped nav entry surfaces here rather than as a silently-orphaned doc.

Why path-qualified only: the doc also mentions bare basenames in prose
(``mic_chat.py``, ``turn_decider.py``) which are ambiguous (which directory?)
and already covered by their path-qualified mentions elsewhere — keying on the
``dir/file.py`` form keeps the extractor unambiguous and the check exact.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "research" / "organic-turn-taking.md"
MKDOCS = ROOT / "mkdocs.yml"

# A path-qualified Python reference: at least one ``dir/`` segment before the
# ``file.py``. This is the form that is unambiguously a repo path (as opposed to
# a bare ``foo.py`` basename mentioned in prose). Optionally wrapped in
# backticks in the Markdown source.
_PY_PATH_RE = re.compile(r"(?<![\w/])((?:[A-Za-z0-9_]+/)+[A-Za-z0-9_]+\.py)")


def _read_doc() -> str:
    return DOC.read_text()


def _py_path_refs(text: str) -> list[str]:
    """Every distinct path-qualified ``dir/.../file.py`` reference in the doc,
    in first-seen order."""
    seen: dict[str, None] = {}
    for m in _PY_PATH_RE.finditer(text):
        seen.setdefault(m.group(1), None)
    return list(seen)


# ---- the doc exists and is wired into the site -----------------------


def test_doc_file_exists():
    assert DOC.is_file(), "the organic-turn-taking research doc must exist"


def test_doc_in_mkdocs_nav():
    """A doc not in the nav is invisible on the published site."""
    nav = MKDOCS.read_text()
    assert "research/organic-turn-taking.md" in nav


# ---- structural anchors ----------------------------------------------


def test_doc_has_backlog_and_findings_sections():
    """The two living sections the per-lap rhythm depends on: the backlog the
    track pulls its next item from, and the findings log laps append to. A
    heading rename surfaces here."""
    doc = _read_doc()
    assert "## Organic-voice backlog" in doc
    assert "## Findings log (append per lap)" in doc


def test_doc_names_the_two_half_duplex_entrypoints():
    """The 'Why this track' framing rests on the two half-duplex entrypoints;
    name them so the framing can't drift off the real code."""
    doc = _read_doc()
    assert "examples/mic_talk.py" in doc
    assert "pipecat_server.py" in doc


# ---- the strong check: every path-qualified reference still exists ----


def test_doc_has_py_path_references():
    """Guard against the extraction regex silently matching nothing (which
    would make the existence check vacuously pass). This doc is large and dense
    with module references, so the floor is generous."""
    refs = _py_path_refs(_read_doc())
    assert len(refs) >= 30, f"expected many dir/file.py refs, found {len(refs)}"


def test_all_documented_py_paths_exist():
    """The core sentinel: every ``dir/.../file.py`` the doc names must still
    exist relative to the repo root. A renamed/removed module turns this red
    instead of leaving a silently-dead reference in the narrative."""
    missing = [
        ref for ref in _py_path_refs(_read_doc()) if not (ROOT / ref).is_file()
    ]
    assert not missing, f"doc references nonexistent files: {missing}"


def test_referenced_session_modules_are_real():
    """A focused subset: the ``session/*.py`` organic-stack modules are the
    spine of the track. Assert each one the doc names exists, so a module
    rename in the organic stack is caught even if the broad check above were
    ever loosened."""
    refs = [r for r in _py_path_refs(_read_doc()) if r.startswith("session/")]
    assert refs, "doc must reference the session/ organic modules"
    for ref in refs:
        assert (ROOT / ref).is_file(), f"{ref} referenced by doc but missing"


def test_referenced_test_files_are_real():
    """The findings log cites the guarding test file each lap shipped. A
    deleted/renamed test file leaves a dead citation; catch it here."""
    refs = [
        r
        for r in _py_path_refs(_read_doc())
        if r.startswith("tests/unit/test_")
    ]
    assert refs, "doc must cite the guarding test files"
    for ref in refs:
        assert (ROOT / ref).is_file(), f"{ref} cited by doc but missing"
