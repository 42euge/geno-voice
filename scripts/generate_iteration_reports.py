"""Generate browsable HTML reports from ITERATION_LOG.md.

Produces one HTML file per iteration plus an index.html under
``iter-reports/``. No external dependencies — uses a small
markdown-subset renderer that handles the constructs ITERATION_LOG.md
actually uses (headers, bold, italic, code, fenced code blocks,
bullet lists, tables, blockquotes).

Usage::

    python scripts/generate_iteration_reports.py

Run from the repo root after appending a new iter-NNN section to
ITERATION_LOG.md. iter-029 added this script and made
"regenerate reports" a standard step in the iteration workflow.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "ITERATION_LOG.md"
OUT_DIR = REPO_ROOT / "iter-reports"


@dataclass
class Iteration:
    """One ## iter-NNN — title section parsed from the log."""
    number: str  # "001"
    title: str
    branch: str = ""
    commit: str = ""
    date: str = ""
    body_md: str = ""  # full markdown body, headers + paragraphs
    tests_added: int = 0
    total_tests: int = 0
    next_id: str = ""  # "002" for navigation
    prev_id: str = ""

    @property
    def slug(self) -> str:
        return f"iter-{self.number}"

    @property
    def filename(self) -> str:
        return f"{self.slug}.html"


# ---- Parser -----------------------------------------------------------------


_ITER_HEADER_RE = re.compile(
    r"^## iter-(?P<num>\d{3}) — (?P<title>.+?)$"
)
_BRANCH_RE = re.compile(
    r"\*\*Branch:\*\*\s+`(?P<branch>[^`]+)`.*commit\s+`(?P<commit>[^`]+)`"
)
_DATE_RE = re.compile(r"\*\*Date:\*\*\s+(?P<date>\S+)")
_TESTS_RE = re.compile(
    r"\*\*(?P<total>\d+)\s+passed.*\((?P<existing>\d+)\s+existing\s+\+\s+(?P<new>\d+)\s+new\)"
)


def parse_iterations(log_text: str) -> list[Iteration]:
    """Walk through ITERATION_LOG.md, splitting at ``## iter-NNN —``
    headers and accumulating the body of each section.

    Sections that appear between iterations but aren't iter headers
    (like ``# Status`` summary blocks) are skipped — they belong to
    the log itself, not a particular iteration.
    """
    lines = log_text.splitlines()
    iterations: list[Iteration] = []
    current: Iteration | None = None
    body: list[str] = []

    def _finish():
        nonlocal current, body
        if current is not None:
            current.body_md = "\n".join(body).strip()
            _populate_metadata(current)
            iterations.append(current)
        current = None
        body = []

    for line in lines:
        m = _ITER_HEADER_RE.match(line)
        if m:
            _finish()
            current = Iteration(number=m.group("num"), title=m.group("title").strip())
            continue
        # End the current iteration when a top-level "# Status" section
        # appears (those are recap blocks the log emits between batches).
        if line.startswith("# Status") and current is not None:
            _finish()
            continue
        if current is not None:
            body.append(line)

    _finish()

    # Wire up prev/next navigation.
    for i, it in enumerate(iterations):
        if i > 0:
            it.prev_id = iterations[i - 1].number
        if i + 1 < len(iterations):
            it.next_id = iterations[i + 1].number

    return iterations


def _populate_metadata(it: Iteration) -> None:
    body = it.body_md
    m = _BRANCH_RE.search(body)
    if m:
        it.branch = m.group("branch")
        it.commit = m.group("commit")
    m = _DATE_RE.search(body)
    if m:
        it.date = m.group("date")
    m = _TESTS_RE.search(body)
    if m:
        it.total_tests = int(m.group("total"))
        it.tests_added = int(m.group("new"))


# ---- Markdown → HTML --------------------------------------------------------


_INLINE_CODE_RE = re.compile(r"`([^`]+?)`")
_BOLD_RE = re.compile(r"\*\*([^*]+?)\*\*")
_ITALIC_RE = re.compile(r"(?<![*\w])\*(?!\s)([^*]+?)(?<!\s)\*(?!\w)")
_ITALIC_UNDER_RE = re.compile(r"(?<![\w_])_(?!\s)([^_]+?)(?<!\s)_(?![\w_])")


def _render_inline(text: str) -> str:
    """Apply bold / italic / inline-code to a line of text. Order
    matters: code first (so its content is escaped raw), then bold
    + italic on the remaining text.
    """
    # Pull code spans out into placeholders so their content isn't
    # touched by the bold/italic regexes.
    code_spans: list[str] = []

    def _stash(m: re.Match) -> str:
        code_spans.append(m.group(1))
        return f"\x00CODE{len(code_spans) - 1}\x00"

    text = _INLINE_CODE_RE.sub(_stash, text)
    text = html.escape(text)
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = _ITALIC_RE.sub(r"<em>\1</em>", text)
    text = _ITALIC_UNDER_RE.sub(r"<em>\1</em>", text)
    # Restore code spans (escaped now).
    for i, span in enumerate(code_spans):
        text = text.replace(
            f"\x00CODE{i}\x00",
            f"<code>{html.escape(span)}</code>",
        )
    return text


def md_to_html(md: str) -> str:
    """Render a markdown subset to HTML.

    Supports: ``###`` / ``####`` headers, fenced code blocks
    (```), bullet lists (``- `` and ``  - `` for nested), tables
    using ``|`` syntax, blockquotes (``> ``), horizontal rules
    (``---``), inline bold/italic/code.
    """
    out: list[str] = []
    lines = md.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Fenced code block
        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # consume closing ```
            code_html = html.escape("\n".join(code_lines))
            out.append(f'<pre class="code-block"><code>{code_html}</code></pre>')
            continue

        # Horizontal rule
        if stripped == "---":
            out.append("<hr>")
            i += 1
            continue

        # Headers
        if stripped.startswith("#### "):
            out.append(f"<h4>{_render_inline(stripped[5:])}</h4>")
            i += 1
            continue
        if stripped.startswith("### "):
            out.append(f"<h3>{_render_inline(stripped[4:])}</h3>")
            i += 1
            continue
        if stripped.startswith("## "):
            out.append(f"<h2>{_render_inline(stripped[3:])}</h2>")
            i += 1
            continue

        # Table
        if "|" in line and i + 1 < len(lines) and re.match(r"^\s*\|?[\s|:-]+\|?\s*$", lines[i + 1]):
            # Parse table block.
            tbl_lines: list[str] = [line]
            i += 1
            sep = lines[i]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            out.append(_render_table(tbl_lines, sep))
            continue

        # Bullet list (with optional one level of nesting)
        if re.match(r"^(\s*)- ", line):
            list_lines: list[str] = []
            while i < len(lines) and (
                re.match(r"^(\s*)- ", lines[i]) or
                (lines[i].startswith("  ") and lines[i].strip())
            ):
                list_lines.append(lines[i])
                i += 1
            out.append(_render_bullet_list(list_lines))
            continue

        # Blockquote
        if stripped.startswith("> "):
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip()[1:].strip())
                i += 1
            out.append(
                f'<blockquote>{_render_inline(" ".join(quote_lines))}</blockquote>'
            )
            continue

        # Blank line — paragraph separator (no-op; out has its own
        # block-level boundaries).
        if not stripped:
            i += 1
            continue

        # Paragraph: collect contiguous non-blank lines.
        para_lines = [stripped]
        j = i + 1
        while j < len(lines) and lines[j].strip() and not _is_block_starter(lines[j]):
            para_lines.append(lines[j].strip())
            j += 1
        out.append(f"<p>{_render_inline(' '.join(para_lines))}</p>")
        i = j

    return "\n".join(out)


def _is_block_starter(line: str) -> bool:
    s = line.strip()
    return (
        s.startswith("```")
        or s == "---"
        or s.startswith("#")
        or s.startswith("- ")
        or s.startswith("> ")
        or "|" in line  # rough — table detection is fragile
    )


def _render_bullet_list(lines: list[str]) -> str:
    """Render a (possibly nested) bullet list."""
    items: list[str] = []
    sub_buffer: list[str] = []

    def _flush_sub():
        nonlocal sub_buffer
        if not sub_buffer:
            return
        rendered = _render_bullet_list(sub_buffer)
        if items:
            # Nested under the most recent item.
            items[-1] = items[-1].rstrip("</li>") + rendered + "</li>"
        else:
            # No parent item — emit the nested list at the top level.
            # This happens when ITERATION_LOG.md has indented bullets
            # that aren't actually nested under anything.
            items.append(rendered)
        sub_buffer = []

    for line in lines:
        m = re.match(r"^(\s*)- (.*)$", line)
        if m:
            indent = len(m.group(1))
            content = m.group(2)
            if indent == 0:
                _flush_sub()
                items.append(f"<li>{_render_inline(content)}</li>")
            else:
                sub_buffer.append(line[2:])  # strip 2 leading spaces
        elif line.startswith("  ") and items:
            # Continuation of the last item.
            items[-1] = items[-1].replace(
                "</li>",
                f" {_render_inline(line.strip())}</li>",
            )

    _flush_sub()
    return "<ul>" + "".join(items) + "</ul>"


def _render_table(rows: list[str], sep_line: str) -> str:
    """Render a markdown table to HTML."""
    def _split(line: str) -> list[str]:
        cells = line.strip()
        if cells.startswith("|"):
            cells = cells[1:]
        if cells.endswith("|"):
            cells = cells[:-1]
        return [c.strip() for c in cells.split("|")]

    if not rows:
        return ""
    header = _split(rows[0])
    body_rows = [_split(r) for r in rows[1:]]
    th = "".join(f"<th>{_render_inline(c)}</th>" for c in header)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{_render_inline(c)}</td>" for c in row) + "</tr>"
        for row in body_rows
    )
    return (
        f'<table class="iter-table"><thead><tr>{th}</tr></thead>'
        f"<tbody>{body_html}</tbody></table>"
    )


# ---- HTML rendering ---------------------------------------------------------


_PAGE_CSS = """\
:root {
  --bg: #0f1115;
  --panel: #161922;
  --text: #e1e4eb;
  --muted: #8a92a3;
  --accent: #7aa2f7;
  --accent-2: #9ece6a;
  --code-bg: #1d2030;
  --border: #2a2f3e;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
               'Helvetica Neue', Arial, sans-serif;
  line-height: 1.55;
}
.container { max-width: 880px; margin: 0 auto; padding: 24px 28px 64px; }
header { border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
header h1 { margin: 0 0 4px; font-size: 24px; }
header .meta { color: var(--muted); font-size: 13px; }
header .meta code { background: none; color: var(--muted); padding: 0; }
nav { margin-bottom: 16px; font-size: 13px; }
nav a { color: var(--accent); text-decoration: none; margin-right: 12px; }
nav a:hover { text-decoration: underline; }
h2, h3, h4 { line-height: 1.2; margin-top: 28px; }
h2 { font-size: 20px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
h3 { font-size: 17px; }
p, li { color: var(--text); }
strong { color: #fff; }
a { color: var(--accent); }
code {
  background: var(--code-bg);
  padding: 2px 6px;
  border-radius: 3px;
  font-family: 'JetBrains Mono', 'Fira Code', Menlo, Consolas, monospace;
  font-size: 0.9em;
}
pre.code-block {
  background: var(--code-bg);
  padding: 14px 16px;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 0.88em;
  border: 1px solid var(--border);
}
pre.code-block code { background: none; padding: 0; border-radius: 0; font-size: 1em; }
ul { padding-left: 22px; }
ul li { margin-bottom: 4px; }
ul ul { margin-top: 4px; }
blockquote {
  border-left: 3px solid var(--accent);
  margin: 0;
  padding: 4px 14px;
  color: var(--muted);
  background: var(--panel);
  border-radius: 0 4px 4px 0;
}
hr { border: none; border-top: 1px solid var(--border); margin: 28px 0; }
table.iter-table {
  width: 100%;
  border-collapse: collapse;
  margin: 16px 0;
  font-size: 0.92em;
}
table.iter-table th, table.iter-table td {
  border: 1px solid var(--border);
  padding: 6px 10px;
  text-align: left;
  vertical-align: top;
}
table.iter-table th { background: var(--panel); }
.iter-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 14px 16px;
  margin-bottom: 12px;
}
.iter-card a { color: var(--text); text-decoration: none; display: block; }
.iter-card a:hover .title { color: var(--accent); }
.iter-card .title { font-weight: 600; font-size: 15px; }
.iter-card .meta { color: var(--muted); font-size: 12px; margin-top: 4px; }
.iter-card .badge {
  display: inline-block;
  background: var(--code-bg);
  color: var(--accent-2);
  padding: 1px 8px;
  border-radius: 10px;
  font-size: 11px;
  margin-left: 8px;
}
"""


def _page_template(title: str, body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
{_PAGE_CSS}
</style>
</head>
<body>
<div class="container">
{body_html}
</div>
</body>
</html>
"""


def render_iteration(it: Iteration) -> str:
    nav_parts: list[str] = ['<a href="index.html">← Index</a>']
    if it.prev_id:
        nav_parts.append(f'<a href="iter-{it.prev_id}.html">← iter-{it.prev_id}</a>')
    if it.next_id:
        nav_parts.append(f'<a href="iter-{it.next_id}.html">iter-{it.next_id} →</a>')
    nav_html = "<nav>" + "".join(nav_parts) + "</nav>"

    meta_bits: list[str] = []
    if it.date:
        meta_bits.append(f"<span>{html.escape(it.date)}</span>")
    if it.branch:
        meta_bits.append(
            f"<span>branch: <code>{html.escape(it.branch)}</code></span>"
        )
    if it.commit:
        meta_bits.append(
            f"<span>commit: <code>{html.escape(it.commit)}</code></span>"
        )
    if it.tests_added:
        meta_bits.append(
            f"<span>tests: <code>+{it.tests_added}</code> "
            f"(total {it.total_tests})</span>"
        )

    header = f"""<header>
<h1>iter-{it.number} — {html.escape(it.title)}</h1>
<div class="meta">{' · '.join(meta_bits)}</div>
</header>"""

    body_html = nav_html + header + md_to_html(it.body_md) + nav_html
    return _page_template(f"iter-{it.number} — {it.title}", body_html)


def render_index(iterations: list[Iteration]) -> str:
    cards: list[str] = []
    for it in iterations:
        meta_bits = []
        if it.date:
            meta_bits.append(html.escape(it.date))
        if it.tests_added:
            meta_bits.append(f"+{it.tests_added} tests")
        if it.commit:
            meta_bits.append(f"commit {it.commit[:7]}")
        meta = " · ".join(meta_bits)
        badge = (
            f'<span class="badge">{it.total_tests} total</span>'
            if it.total_tests
            else ""
        )
        cards.append(f"""<div class="iter-card">
<a href="{it.filename}">
<div class="title">iter-{it.number} — {html.escape(it.title)}{badge}</div>
<div class="meta">{meta}</div>
</a>
</div>""")

    summary = (
        f"<p>{len(iterations)} iterations shipped"
        + (
            f" · most recent: <strong>iter-{iterations[-1].number}</strong>"
            f" ({html.escape(iterations[-1].title)}) "
            f"with <strong>{iterations[-1].total_tests}</strong> tests passing"
            if iterations and iterations[-1].total_tests
            else ""
        )
        + ".</p>"
    )

    body_html = (
        '<header><h1>geno-voice iteration log</h1>'
        '<div class="meta">Generated from ITERATION_LOG.md</div></header>'
        + summary
        + "".join(cards)
    )
    return _page_template("geno-voice iteration log", body_html)


# ---- Entry point ------------------------------------------------------------


def main() -> int:
    if not LOG_PATH.exists():
        print(f"ITERATION_LOG.md not found at {LOG_PATH}")
        return 1

    log_text = LOG_PATH.read_text()
    iterations = parse_iterations(log_text)
    if not iterations:
        print("No iterations parsed from ITERATION_LOG.md")
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    for it in iterations:
        path = OUT_DIR / it.filename
        path.write_text(render_iteration(it))

    index_path = OUT_DIR / "index.html"
    index_path.write_text(render_index(iterations))

    print(
        f"Wrote {len(iterations)} iteration reports + index to {OUT_DIR}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
