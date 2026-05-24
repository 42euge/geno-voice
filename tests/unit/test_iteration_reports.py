"""Tests for scripts/generate_iteration_reports.py.

The generator parses ITERATION_LOG.md into Iteration objects then
renders each to HTML. These tests cover the parser and the
markdown→HTML helpers — the rendering is deterministic from the
parsed data, so well-tested pieces compose into trustworthy output.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "generate_iteration_reports.py"

# Load the script as a module so we can import its helpers.
spec = importlib.util.spec_from_file_location("gen_reports", SCRIPT_PATH)
gen_reports = importlib.util.module_from_spec(spec)
sys.modules["gen_reports"] = gen_reports
spec.loader.exec_module(gen_reports)


# ---- Inline markdown helpers ------------------------------------------------


class TestRenderInline:
    def test_plain_text_is_html_escaped(self):
        out = gen_reports._render_inline("a < b > c & d")
        assert "&lt;" in out
        assert "&gt;" in out
        assert "&amp;" in out

    def test_bold_renders_strong(self):
        out = gen_reports._render_inline("this is **bold** text")
        assert "<strong>bold</strong>" in out

    def test_inline_code_renders_code_tag(self):
        out = gen_reports._render_inline("call `func()` directly")
        assert "<code>func()</code>" in out

    def test_inline_code_content_is_escaped(self):
        out = gen_reports._render_inline("escape `<>&` chars")
        # The content INSIDE code is escaped because it's HTML.
        assert "<code>&lt;&gt;&amp;</code>" in out

    def test_bold_and_code_compose(self):
        out = gen_reports._render_inline("**bold** and `code` together")
        assert "<strong>bold</strong>" in out
        assert "<code>code</code>" in out

    def test_italic_underscore(self):
        out = gen_reports._render_inline("uses _italics_ here")
        assert "<em>italics</em>" in out

    def test_no_italic_in_word_with_underscore(self):
        # "user_role" should NOT be rendered as "user<em>role</em>"
        out = gen_reports._render_inline("variable user_role and item")
        assert "<em>" not in out


# ---- Block-level markdown ---------------------------------------------------


class TestMdToHtml:
    def test_h2_header(self):
        out = gen_reports.md_to_html("## My Header")
        assert "<h2>My Header</h2>" in out

    def test_h3_header(self):
        out = gen_reports.md_to_html("### Subhead")
        assert "<h3>Subhead</h3>" in out

    def test_paragraph(self):
        out = gen_reports.md_to_html("This is a paragraph.")
        assert "<p>This is a paragraph.</p>" in out

    def test_horizontal_rule(self):
        out = gen_reports.md_to_html("---")
        assert "<hr>" in out

    def test_fenced_code_block(self):
        md = "```\nx = 1\nprint(x)\n```"
        out = gen_reports.md_to_html(md)
        assert '<pre class="code-block">' in out
        assert "x = 1" in out
        assert "print(x)" in out

    def test_fenced_code_escapes_html(self):
        md = "```\n<html>\n```"
        out = gen_reports.md_to_html(md)
        assert "&lt;html&gt;" in out
        assert "<html>" not in out.replace('<pre class="code-block">', "").replace("<code>", "")

    def test_bullet_list(self):
        md = "- one\n- two\n- three"
        out = gen_reports.md_to_html(md)
        assert "<ul>" in out
        assert "<li>one</li>" in out
        assert "<li>two</li>" in out
        assert "<li>three</li>" in out

    def test_table_renders_as_html(self):
        md = "| col1 | col2 |\n|------|------|\n| a    | b    |\n| c    | d    |"
        out = gen_reports.md_to_html(md)
        assert "<table" in out
        assert "<th>col1</th>" in out
        assert "<th>col2</th>" in out
        assert "<td>a</td>" in out
        assert "<td>d</td>" in out

    def test_blockquote(self):
        md = "> a quoted line"
        out = gen_reports.md_to_html(md)
        assert "<blockquote>" in out
        assert "a quoted line" in out


# ---- Parser -----------------------------------------------------------------


class TestParseIterations:
    def test_empty_log_returns_empty(self):
        out = gen_reports.parse_iterations("")
        assert out == []

    def test_log_without_iter_headers_returns_empty(self):
        out = gen_reports.parse_iterations("# Status\n\nNo iters yet.")
        assert out == []

    def test_single_iteration_parsed(self):
        log = """\
## iter-001 — first iteration

**Branch:** `iter-001-foo` (merged ff to main, commit `abc1234`)
**Date:** 2026-01-01

Body content here.

Verification: `python -m pytest tests/unit/` → **22 passed in 0.02s** (0 existing + 22 new).
"""
        iters = gen_reports.parse_iterations(log)
        assert len(iters) == 1
        it = iters[0]
        assert it.number == "001"
        assert it.title == "first iteration"
        assert it.branch == "iter-001-foo"
        assert it.commit == "abc1234"
        assert it.date == "2026-01-01"
        assert it.total_tests == 22
        assert it.tests_added == 22

    def test_multiple_iterations_with_navigation(self):
        log = """\
## iter-001 — first

**Branch:** `b1` (merged ff to main, commit `c1`)
**Date:** 2026-01-01

Body 1.

---

## iter-002 — second

**Branch:** `b2` (merged ff to main, commit `c2`)
**Date:** 2026-01-02

Body 2.
"""
        iters = gen_reports.parse_iterations(log)
        assert len(iters) == 2
        assert iters[0].number == "001"
        assert iters[1].number == "002"
        # Navigation wired up.
        assert iters[0].next_id == "002"
        assert iters[0].prev_id == ""
        assert iters[1].prev_id == "001"
        assert iters[1].next_id == ""

    def test_status_block_between_iterations_is_skipped(self):
        # The real ITERATION_LOG.md has "# Status" recap blocks
        # between batches that aren't iterations.
        log = """\
## iter-001 — first

**Branch:** `b1` (merged ff to main, commit `c1`)
**Date:** 2026-01-01

Body 1.

# Status

Some recap here that's not part of any iteration.

---

## iter-002 — second

**Branch:** `b2` (merged ff to main, commit `c2`)
**Date:** 2026-01-02

Body 2.
"""
        iters = gen_reports.parse_iterations(log)
        # Two iterations, status block stays out.
        assert len(iters) == 2
        # iter-001's body should NOT contain the status text.
        assert "recap here" not in iters[0].body_md

    def test_iteration_without_branch_metadata_still_parses(self):
        # Some early iterations might lack the standard metadata block.
        log = """\
## iter-001 — early

Just body text, no metadata.
"""
        iters = gen_reports.parse_iterations(log)
        assert len(iters) == 1
        assert iters[0].number == "001"
        assert iters[0].title == "early"
        assert iters[0].branch == ""
        assert iters[0].commit == ""


# ---- Rendering --------------------------------------------------------------


class TestRendering:
    def _make_iter(self, **kwargs):
        return gen_reports.Iteration(
            number=kwargs.get("number", "001"),
            title=kwargs.get("title", "test iter"),
            branch=kwargs.get("branch", "test-branch"),
            commit=kwargs.get("commit", "abc1234"),
            date=kwargs.get("date", "2026-01-01"),
            body_md=kwargs.get("body_md", "Body content."),
            tests_added=kwargs.get("tests_added", 5),
            total_tests=kwargs.get("total_tests", 100),
            next_id=kwargs.get("next_id", ""),
            prev_id=kwargs.get("prev_id", ""),
        )

    def test_render_iteration_includes_title(self):
        it = self._make_iter(title="my fix")
        html = gen_reports.render_iteration(it)
        assert "iter-001 — my fix" in html

    def test_render_iteration_includes_branch_and_commit(self):
        it = self._make_iter(branch="iter-001-foo", commit="deadbeef")
        html = gen_reports.render_iteration(it)
        assert "iter-001-foo" in html
        assert "deadbeef" in html

    def test_render_iteration_includes_test_counts(self):
        it = self._make_iter(tests_added=12, total_tests=358)
        html = gen_reports.render_iteration(it)
        assert "+12" in html
        assert "358" in html

    def test_render_iteration_navigation_when_only_first(self):
        it = self._make_iter(prev_id="", next_id="002")
        html = gen_reports.render_iteration(it)
        assert "iter-002.html" in html
        # No prev link.
        assert "iter-000" not in html

    def test_render_iteration_navigation_with_both(self):
        it = self._make_iter(number="002", prev_id="001", next_id="003")
        html = gen_reports.render_iteration(it)
        assert "iter-001.html" in html
        assert "iter-003.html" in html

    def test_render_index_lists_iterations(self):
        iters = [
            self._make_iter(number="001", title="a"),
            self._make_iter(number="002", title="b"),
        ]
        html = gen_reports.render_index(iters)
        assert "iter-001 — a" in html
        assert "iter-002 — b" in html
        assert 'href="iter-001.html"' in html
        assert 'href="iter-002.html"' in html

    def test_render_iteration_escapes_title_html(self):
        it = self._make_iter(title="fix <script> bug")
        html = gen_reports.render_iteration(it)
        # Title is HTML-escaped in the page.
        assert "&lt;script&gt;" in html
        # The actual <script> tag does NOT appear in user content.
        # (The page itself has <script> tags from CSS embed... actually
        # it doesn't — we use inline <style>. So bare <script> would
        # only appear via injection. Make sure none does.)
        assert "<script>" not in html


# ---- Real-log integration ---------------------------------------------------


class TestRealIterationLog:
    def test_real_log_parses_into_28_plus_iterations(self):
        log_path = ROOT / "ITERATION_LOG.md"
        if not log_path.exists():
            pytest.skip("ITERATION_LOG.md not present")
        text = log_path.read_text()
        iters = gen_reports.parse_iterations(text)
        # Sanity: project has at least 28 iterations as of iter-029.
        assert len(iters) >= 28
        # First and last are recognizable.
        assert iters[0].number == "001"
        # Numbers strictly ascending.
        nums = [int(it.number) for it in iters]
        assert nums == sorted(nums)
