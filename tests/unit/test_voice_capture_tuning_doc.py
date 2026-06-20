"""Tests for iter-321 — keep ``docs/research/voice-capture-tuning.md`` in sync
with the real ``gv`` CLI it documents.

This backports the iter-320 command-parse **drift sentinel** (which guards
``docs/research/tts-pacing-mirror.md``) onto the older, much larger VAD-analysis
research doc. The VAD doc grew across ~30 laps (iters 189→287+: the
``replay_vad.py`` harness, then ``gv vad`` / ``vad-diff`` / ``vad-sweep`` /
``vad-grid`` and their ``--target`` grammar) and shows dozens of copy-pasteable
``gv`` examples — exactly the surface that rots silently when a flag is renamed.

A doc that shows command examples is worse than no doc once the CLI moves on:
a reader copies a flag that no longer parses and hits ``SystemExit(2)``. So
this test **extracts every ``gv`` command in the doc's fenced code blocks and
parses it through the real ``build_parser``**. If a documented flag is renamed
or removed, the example stops parsing and this test goes red.

Unlike the mirror doc (which is ``gv``-only), this doc *also* shows
``python fixtures/replay_vad.py`` / ``replay_silero.py`` examples — the
extractor filters on the leading ``gv `` token so those non-``gv`` lines are
skipped, matching how the iter-320 extractor already filters.

It also pins the structural pointers (all four ``vad*`` subcommands named, the
``--json`` / ``--csv`` machine surfaces described as mutually exclusive, the
replay harnesses + ``vad/silero.py`` named, the nav entry present) so a future
rename of a subcommand or the doc file surfaces here rather than as a silently
dead link.
"""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples import gv  # noqa: E402

DOC = ROOT / "docs" / "research" / "voice-capture-tuning.md"
MKDOCS = ROOT / "mkdocs.yml"

# The CLI subcommands this doc documents.
VAD_SUBCOMMANDS = (
    "vad",
    "vad-gaps",
    "vad-gap-sweep",
    "vad-diff",
    "vad-sweep",
    "vad-grid",
    "vad-gap-grid",
)


def _read_doc() -> str:
    return DOC.read_text()


def _fenced_blocks(text: str) -> list[str]:
    """Return the contents of every ``` ``` fenced code block."""
    return re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.DOTALL)


def _gv_commands(text: str) -> list[str]:
    """Every ``gv ...`` command line in the doc's fenced code blocks.

    Joins backslash-continued lines and strips trailing ``# ...`` comments so
    the residue is exactly what an operator would type. Returns the arg list
    *after* the leading ``gv`` token. Non-``gv`` lines (e.g.
    ``python fixtures/replay_vad.py``) are skipped — same filter the iter-320
    extractor uses.
    """
    commands: list[str] = []
    for block in _fenced_blocks(text):
        # Re-join shell line continuations (``\`` at EOL) into one logical line.
        joined = re.sub(r"\\\n\s*", " ", block)
        for line in joined.splitlines():
            line = line.strip()
            if not line.startswith("gv "):
                continue
            # Drop an inline ``# explanatory comment`` tail.
            line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
            commands.append(line[len("gv ") :])
    return commands


# ---- the doc exists and is wired into the site -----------------------


def test_doc_file_exists():
    assert DOC.is_file(), "the voice-capture tuning research doc must exist"


def test_doc_in_mkdocs_nav():
    """A doc not in the nav is invisible on the published site."""
    nav = MKDOCS.read_text()
    assert "research/voice-capture-tuning.md" in nav


# ---- structural pointers ---------------------------------------------


def test_doc_names_all_vad_subcommands():
    doc = _read_doc()
    for cmd in VAD_SUBCOMMANDS:
        assert f"gv {cmd}" in doc, f"doc must document `gv {cmd}`"


def test_doc_describes_machine_surfaces_as_mutually_exclusive():
    doc = _read_doc()
    for flag in ("--json", "--csv"):
        assert flag in doc
    # The two machine surfaces are called mutually exclusive somewhere.
    assert "mutually exclusive" in doc


def test_doc_names_the_replay_harnesses():
    """The methodology leans on the headless replay harnesses; name the files
    so a reader can open them, and verify they still exist."""
    doc = _read_doc()
    for fixture in ("fixtures/replay_vad.py", "fixtures/replay_silero.py"):
        assert fixture in doc, f"doc must name {fixture}"
        assert (ROOT / fixture).is_file(), f"{fixture} must exist"


def test_doc_names_the_silero_segmenter_module():
    """The primary VAD path is ``vad/silero.py``; name it and verify it
    exists."""
    doc = _read_doc()
    assert "vad/silero.py" in doc
    assert (ROOT / "vad" / "silero.py").is_file()


# ---- the strong check: every documented command actually parses -------


def test_doc_has_command_examples():
    """Guard against the extraction regex silently matching nothing (which
    would make the parse tests vacuously pass). This doc is large and shows
    dozens of examples, so the floor is generous."""
    cmds = _gv_commands(_read_doc())
    assert len(cmds) >= 30, f"expected many gv examples, found {len(cmds)}"


def test_every_documented_vad_subcommand_appears_in_examples():
    """Each of the four subcommands the doc prose names must also appear as a
    runnable example — not just in prose."""
    cmds = _gv_commands(_read_doc())
    leading = {c.split()[0] for c in cmds if c.split()}
    for sub in VAD_SUBCOMMANDS:
        assert sub in leading, f"doc must show a runnable `gv {sub}` example"


def test_all_documented_gv_commands_parse():
    """The core sentinel: no documented `gv` line — whatever the subcommand —
    may fail to parse. A renamed/removed flag turns this red."""
    parser = gv.build_parser()
    for cmd in _gv_commands(_read_doc()):
        try:
            args = parser.parse_args(shlex.split(cmd))
        except SystemExit:  # argparse exits 2 on a bad flag
            pytest.fail(f"documented command does not parse: gv {cmd}")
        # The parsed subcommand must be the one the line leads with.
        assert args.command == cmd.split()[0], cmd


def test_documented_gv_commands_route_to_known_subcommands():
    """Every documented example targets one of the four known subcommands —
    catches a doc drifting onto a subcommand this test doesn't cover."""
    for cmd in _gv_commands(_read_doc()):
        head = cmd.split()[0]
        assert head in VAD_SUBCOMMANDS, f"unexpected subcommand in doc: gv {cmd}"
