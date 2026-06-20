"""Tests for iter-320 — keep ``docs/research/tts-pacing-mirror.md`` in sync
with the real ``gv`` CLI it documents.

The ``simulate-mirror`` / ``calibrate-base-wpm`` surface grew five laps
(iters 215–319: the engine, the two subcommands, then ``--csv`` / ``--json``,
``--lurch-weight``, and the band overrides) without a prose home, while every
VAD-analysis surface had a rich research doc. iter-320 lands the doc.

A doc that shows command examples is worse than no doc once the CLI moves on:
a reader copies a flag that no longer parses and hits ``SystemExit(2)``. So
this is a **drift sentinel** in the iter-129/130/144 mold — but instead of
checking that named sections/files exist, it does the stronger thing the
subject matter allows: it **extracts every ``gv`` command in the doc's fenced
code blocks and parses it through the real ``build_parser``**. If a documented
flag is renamed or removed, the example stops parsing and this test goes red.

It also pins the structural pointers (both subcommands named, the format trio
described, the sibling VAD doc cross-referenced, the nav entry present) so a
future rename of a subcommand or the doc file surfaces here rather than as a
silently dead link.
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

DOC = ROOT / "docs" / "research" / "tts-pacing-mirror.md"
MKDOCS = ROOT / "mkdocs.yml"


def _read_doc() -> str:
    return DOC.read_text()


def _fenced_blocks(text: str) -> list[str]:
    """Return the contents of every ``` ``` fenced code block."""
    return re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.DOTALL)


def _gv_commands(text: str) -> list[str]:
    """Every ``gv ...`` command line in the doc's fenced code blocks.

    Joins backslash-continued lines and strips trailing ``# ...`` comments so
    the residue is exactly what an operator would type. Returns the arg list
    *after* the leading ``gv`` token.
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
    assert DOC.is_file(), "the TTS-pacing mirror research doc must exist"


def test_doc_in_mkdocs_nav():
    """A doc not in the nav is invisible on the published site."""
    nav = MKDOCS.read_text()
    assert "research/tts-pacing-mirror.md" in nav


# ---- structural pointers ---------------------------------------------


def test_doc_names_both_subcommands():
    doc = _read_doc()
    for cmd in ("simulate-mirror", "calibrate-base-wpm"):
        assert f"gv {cmd}" in doc, f"doc must document `gv {cmd}`"


def test_doc_describes_the_format_trio():
    doc = _read_doc()
    for flag in ("--json", "--csv"):
        assert flag in doc
    # Both machine surfaces are called mutually exclusive somewhere.
    assert "mutually exclusive" in doc


def test_doc_cross_references_sibling_vad_doc():
    """The trio claim leans on the VAD-analysis surfaces; name that doc so a
    reader can find it."""
    doc = _read_doc()
    assert "Voice-capture tuning" in doc


def test_doc_names_the_engine_module():
    """The methodology section claims a pure-stdlib engine; name the file so a
    reader can open it."""
    doc = _read_doc()
    assert "session/wpm_mirror.py" in doc
    assert (ROOT / "session" / "wpm_mirror.py").is_file()


# ---- the strong check: every documented command actually parses -------


def test_doc_has_command_examples():
    """Guard against the extraction regex silently matching nothing (which
    would make the parse test vacuously pass)."""
    cmds = _gv_commands(_read_doc())
    assert len(cmds) >= 8, f"expected several gv examples, found {len(cmds)}"


def test_every_documented_simulate_mirror_command_parses():
    cmds = [c for c in _gv_commands(_read_doc()) if c.startswith("simulate-mirror")]
    assert cmds, "doc must show simulate-mirror examples"
    parser = gv.build_parser()
    for cmd in cmds:
        args = parser.parse_args(shlex.split(cmd))
        assert args.command == "simulate-mirror", cmd


def test_every_documented_calibrate_command_parses():
    cmds = [
        c for c in _gv_commands(_read_doc()) if c.startswith("calibrate-base-wpm")
    ]
    assert cmds, "doc must show calibrate-base-wpm examples"
    parser = gv.build_parser()
    for cmd in cmds:
        args = parser.parse_args(shlex.split(cmd))
        assert args.command == "calibrate-base-wpm", cmd


def test_all_documented_gv_commands_parse():
    """Belt-and-suspenders: no documented `gv` line — whatever the subcommand —
    may fail to parse. A renamed/removed flag turns this red."""
    parser = gv.build_parser()
    for cmd in _gv_commands(_read_doc()):
        try:
            parser.parse_args(shlex.split(cmd))
        except SystemExit:  # argparse exits 2 on a bad flag
            pytest.fail(f"documented command does not parse: gv {cmd}")


# ---- the documented flags/defaults match the real parser --------------


def test_documented_simulate_mirror_flags_exist():
    """Each flag the doc prose names must be a real simulate-mirror option."""
    parser = gv.build_parser()
    base = parser.parse_args(["simulate-mirror", "--wpms", "120,200,120"])
    for dest in (
        "wpms",
        "initial_speed",
        "base_wpm",
        "strength",
        "grid",
        "base_wpms",
        "strengths",
        "lurch_weight",
        "min_speed",
        "max_speed",
        "min_delta",
        "json",
        "csv",
    ):
        assert hasattr(base, dest), f"simulate-mirror missing --{dest}"


def test_documented_calibrate_flags_exist():
    parser = gv.build_parser()
    base = parser.parse_args(["calibrate-base-wpm", "--samples", "50:18.2"])
    for dest in (
        "samples",
        "nominal",
        "verdict",
        "spread_max",
        "drift_min",
        "min_samples",
        "json",
        "csv",
    ):
        assert hasattr(base, dest), f"calibrate-base-wpm missing --{dest}"


def test_documented_band_defaults_match_engine():
    """The doc states the band/strength/base_wpm seed defaults as fact; pin
    them so a future seed change forces a doc edit."""
    doc = _read_doc()
    parser = gv.build_parser()
    args = parser.parse_args(["simulate-mirror", "--wpms", "120,200,120"])
    # Defaults the prose asserts in parentheses.
    assert args.base_wpm == 165.0 and "165.0" in doc
    assert args.strength == 0.5 and "0.5" in doc
    assert args.min_speed == 0.8 and "0.8" in doc
    assert args.max_speed == 1.3 and "1.3" in doc
    assert args.min_delta == 0.05 and "0.05" in doc
    assert args.lurch_weight == 0.5
