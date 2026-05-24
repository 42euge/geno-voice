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
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "ITERATION_LOG.md"
OUT_DIR = REPO_ROOT / "iter-reports"
# iter-036: performance results dumped by tests/performance/.
# Optional — performance.html is only rendered when this exists.
PERF_RESULTS_PATH = OUT_DIR / "perf-results.json"


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
    # iter-035: pulled from the verification line. Used for the
    # testing.html runtime plot. 0 means "not parseable" (some
    # early iterations had no timing in their verification line).
    test_runtime_s: float = 0.0
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
# iter-035: pull the runtime portion of the verification line —
# patterns like ``**413 passed in 17.9s**`` or ``**358 passed in 18s**``.
# Optional (some early iters omitted it).
_RUNTIME_RE = re.compile(
    r"\*\*\d+\s+passed\s+in\s+(?P<seconds>\d+(?:\.\d+)?)s\*\*"
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
    m = _RUNTIME_RE.search(body)
    if m:
        it.test_runtime_s = float(m.group("seconds"))


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
/* iter-035: testing report — stat grid + SVG chart styles. */
.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  margin: 16px 0 24px;
}
.stat {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 14px 16px;
}
.stat-num {
  font-size: 24px;
  font-weight: 600;
  color: var(--accent);
}
.stat-label {
  font-size: 12px;
  color: var(--muted);
  margin-top: 4px;
}
svg.chart {
  display: block;
  width: 100%;
  max-width: 720px;
  height: auto;
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  margin: 8px 0 24px;
}
svg.chart .grid { stroke: var(--border); stroke-width: 0.5; }
svg.chart .axis { fill: var(--muted); font-size: 11px; }
svg.chart .chart-title { fill: var(--text); font-size: 13px; font-weight: 600; }
svg.chart .chart-axis-label { fill: var(--muted); font-size: 11px; }
.chart-empty {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 14px;
  color: var(--muted);
  font-size: 13px;
}
/* iter-036: horizontal bar chart + perf scenario table. */
svg.chart.hbar { max-width: 720px; }
svg.chart.hbar .axis-label { fill: var(--text); font-size: 12px; }
svg.chart.hbar .value { fill: var(--muted); font-size: 11px; }
svg.chart .legend { fill: var(--text); font-size: 11px; }
table.perf-table {
  border-collapse: collapse;
  width: 100%;
  margin: 16px 0 24px;
  font-size: 13px;
}
table.perf-table th, table.perf-table td {
  border: 1px solid var(--border);
  padding: 6px 10px;
  text-align: left;
}
table.perf-table th { background: var(--card-bg); color: var(--muted); }
table.perf-table td.num { text-align: right; }
table.perf-table code { background: var(--code-bg); padding: 1px 4px; border-radius: 3px; }
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
    # iter-035: testing report link, present on every iter page.
    nav_parts.append('<a href="testing.html">Testing →</a>')
    # iter-036: performance report link.
    nav_parts.append('<a href="performance.html">Performance →</a>')
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

    # iter-035: top-level link to the testing posture page.
    # iter-036: + performance page link.
    nav_html = (
        '<nav>'
        '<a href="testing.html">Testing →</a>'
        '<a href="performance.html">Performance →</a>'
        '</nav>'
    )

    body_html = (
        '<header><h1>geno-voice iteration log</h1>'
        '<div class="meta">Generated from ITERATION_LOG.md</div></header>'
        + nav_html
        + summary
        + "".join(cards)
    )
    return _page_template("geno-voice iteration log", body_html)


# ---- iter-035: testing report (SVG plots) ------------------------------------


def _count_test_files(repo_root: Path) -> tuple[int, int]:
    """Count test files under tests/unit/ and tests/integration/. The
    counts shown on testing.html sit alongside the per-iteration test
    totals, so a reader can see both "how many test FILES we have"
    and "how many test CASES are running."

    Returns (unit_files, integration_files). Either may be 0.
    """
    unit_dir = repo_root / "tests" / "unit"
    int_dir = repo_root / "tests" / "integration"
    unit_n = len(list(unit_dir.glob("test_*.py"))) if unit_dir.exists() else 0
    int_n = len(list(int_dir.glob("test_*.py"))) if int_dir.exists() else 0
    return unit_n, int_n


def _svg_line_chart(
    series: list[tuple[float, float]],
    *,
    title: str,
    y_label: str,
    width: int = 720,
    height: int = 280,
    color: str = "#7aa2f7",
) -> str:
    """Render a simple SVG line chart with axes, gridlines, and dots
    at each data point. Pure-Python, no dependencies — the goal is
    a single self-contained iter-reports/ tree that opens fine on a
    plain HTTP server.

    `series` is a list of (x, y) tuples, where x is the iteration
    number and y is the metric. x is plotted left-to-right.
    """
    if not series:
        return f'<div class="chart-empty">{html.escape(title)}: no data</div>'

    pad_l, pad_r, pad_t, pad_b = 56, 16, 30, 36
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b

    xs = [p[0] for p in series]
    ys = [p[1] for p in series]
    x_min, x_max = min(xs), max(xs)
    y_min = 0  # always anchor at zero so growth is visually honest
    y_max = max(max(ys), 1)
    # Add 8% headroom on top so the highest point doesn't touch the frame.
    y_max *= 1.08

    def sx(x: float) -> float:
        if x_max == x_min:
            return pad_l + inner_w / 2
        return pad_l + (x - x_min) / (x_max - x_min) * inner_w

    def sy(y: float) -> float:
        if y_max == y_min:
            return pad_t + inner_h / 2
        return pad_t + inner_h - (y - y_min) / (y_max - y_min) * inner_h

    # Polyline path
    path_d = " ".join(
        ("M" if i == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}"
        for i, (x, y) in enumerate(series)
    )

    # Y-axis ticks: 0, 25%, 50%, 75%, 100% of y_max
    y_ticks = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = y_min + frac * (y_max - y_min)
        y = sy(v)
        y_ticks.append(
            f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y:.1f}" y2="{y:.1f}" '
            f'class="grid"/>'
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" '
            f'text-anchor="end" class="axis">{int(round(v))}</text>'
        )

    # X-axis ticks: first, mid, last
    x_ticks = []
    for x in (x_min, (x_min + x_max) / 2, x_max):
        xi = sx(x)
        x_ticks.append(
            f'<text x="{xi:.1f}" y="{height - pad_b + 18}" '
            f'text-anchor="middle" class="axis">iter-{int(round(x)):03d}</text>'
        )

    # Data points
    dots = "".join(
        f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{color}"/>'
        for x, y in series
    )

    return f"""<svg viewBox="0 0 {width} {height}" class="chart"
xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{html.escape(title)}">
<text x="{pad_l}" y="18" class="chart-title">{html.escape(title)}</text>
<text x="{pad_l}" y="{height - 4}" class="chart-axis-label">iteration</text>
<text x="14" y="{pad_t + inner_h / 2}" class="chart-axis-label" transform="rotate(-90 14,{pad_t + inner_h / 2})" text-anchor="middle">{html.escape(y_label)}</text>
{''.join(y_ticks)}
{''.join(x_ticks)}
<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2"/>
{dots}
</svg>"""


def _svg_bar_chart(
    series: list[tuple[float, float]],
    *,
    title: str,
    y_label: str,
    width: int = 720,
    height: int = 280,
    color: str = "#9ece6a",
) -> str:
    """Render a simple SVG bar chart. ``series`` is (x, y) tuples;
    bars are placed at evenly-spaced columns along x.
    """
    if not series:
        return f'<div class="chart-empty">{html.escape(title)}: no data</div>'

    pad_l, pad_r, pad_t, pad_b = 56, 16, 30, 36
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b

    n = len(series)
    bar_gap = 2
    bar_w = max(2.0, inner_w / n - bar_gap)

    ys = [p[1] for p in series]
    y_min = 0
    y_max = max(max(ys), 1) * 1.08

    def sy(y: float) -> float:
        return pad_t + inner_h - (y - y_min) / (y_max - y_min) * inner_h

    bars = []
    for i, (x, y) in enumerate(series):
        bx = pad_l + i * (inner_w / n)
        by = sy(y)
        bh = pad_t + inner_h - by
        bars.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" '
            f'height="{bh:.1f}" fill="{color}"/>'
        )

    y_ticks = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = y_min + frac * (y_max - y_min)
        y = sy(v)
        y_ticks.append(
            f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y:.1f}" y2="{y:.1f}" '
            f'class="grid"/>'
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" '
            f'text-anchor="end" class="axis">{int(round(v))}</text>'
        )

    # X-axis ticks at first, mid, last iteration.
    if n >= 1:
        first_idx, mid_idx, last_idx = 0, n // 2, n - 1
        x_ticks = []
        for idx in {first_idx, mid_idx, last_idx}:
            xi = pad_l + idx * (inner_w / n) + bar_w / 2
            iter_num = int(round(series[idx][0]))
            x_ticks.append(
                f'<text x="{xi:.1f}" y="{height - pad_b + 18}" '
                f'text-anchor="middle" class="axis">iter-{iter_num:03d}</text>'
            )
    else:
        x_ticks = []

    return f"""<svg viewBox="0 0 {width} {height}" class="chart"
xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{html.escape(title)}">
<text x="{pad_l}" y="18" class="chart-title">{html.escape(title)}</text>
<text x="{pad_l}" y="{height - 4}" class="chart-axis-label">iteration</text>
<text x="14" y="{pad_t + inner_h / 2}" class="chart-axis-label" transform="rotate(-90 14,{pad_t + inner_h / 2})" text-anchor="middle">{html.escape(y_label)}</text>
{''.join(y_ticks)}
{''.join(x_ticks)}
{''.join(bars)}
</svg>"""


def render_testing_page(iterations: list[Iteration], repo_root: Path) -> str:
    """Build iter-reports/testing.html — one page summarizing the
    project's test posture over iterations.

    Plots:
      - Total tests over time (line chart, cumulative).
      - Tests added per iteration (bar chart).
      - Test runtime over time (line chart, where measured).

    Plus a current breakdown of test files by suite (unit /
    integration) so the reader can tell the suites apart at a
    glance.
    """
    nums = [int(it.number) for it in iterations]

    # Series 1: cumulative total tests. Use the parsed ``total_tests``
    # for iters where it was recorded; carry the previous value
    # forward when an iter didn't update the count (rare — but it's
    # safer than zero).
    total_series: list[tuple[float, float]] = []
    last_total = 0
    for it in iterations:
        if it.total_tests > 0:
            last_total = it.total_tests
        total_series.append((float(it.number), float(last_total)))

    # Series 2: tests added per iteration (the parsed `+N new` value).
    added_series = [
        (float(it.number), float(it.tests_added)) for it in iterations
    ]

    # Series 3: runtime over iterations — only include iters where
    # we parsed a number (early iters had no timing in the
    # verification line).
    runtime_series = [
        (float(it.number), it.test_runtime_s)
        for it in iterations
        if it.test_runtime_s > 0
    ]

    chart_total = _svg_line_chart(
        total_series,
        title="Total tests passing",
        y_label="tests",
    )
    chart_added = _svg_bar_chart(
        added_series,
        title="Tests added per iteration",
        y_label="tests added",
    )
    chart_runtime = _svg_line_chart(
        runtime_series,
        title="Test runtime (seconds)",
        y_label="seconds",
        color="#bb9af7",
    )

    unit_n, int_n = _count_test_files(repo_root)
    latest = iterations[-1] if iterations else None
    latest_total = latest.total_tests if latest else 0
    latest_runtime = latest.test_runtime_s if latest else 0
    median_added = (
        sorted(it.tests_added for it in iterations)[len(iterations) // 2]
        if iterations
        else 0
    )

    summary_html = f"""<div class="stat-grid">
<div class="stat"><div class="stat-num">{latest_total}</div><div class="stat-label">tests passing</div></div>
<div class="stat"><div class="stat-num">{unit_n}</div><div class="stat-label">unit test files</div></div>
<div class="stat"><div class="stat-num">{int_n}</div><div class="stat-label">integration test files</div></div>
<div class="stat"><div class="stat-num">{latest_runtime:.1f}s</div><div class="stat-label">latest runtime</div></div>
<div class="stat"><div class="stat-num">{median_added}</div><div class="stat-label">median added / iter</div></div>
</div>"""

    # Run-it-yourself instructions
    run_block = """<h2>Run the tests yourself</h2>
<pre class="code-block"><code># Unit suite — fast, no I/O dependencies
python -m pytest tests/unit/

# Integration suite — drives ChatLoop end-to-end with virtual audio
python -m pytest tests/integration/

# Both
python -m pytest tests/unit/ tests/integration/
</code></pre>"""

    # iter-036: testing page links sideways to the perf page.
    nav_html = (
        '<nav>'
        '<a href="index.html">← Index</a>'
        '<a href="performance.html">Performance →</a>'
        '</nav>'
    )

    body_html = (
        nav_html
        + "<header><h1>Testing</h1>"
        '<div class="meta">Test posture across iterations · '
        "Generated from ITERATION_LOG.md + repo scan</div></header>"
        + summary_html
        + "<h2>Total tests passing</h2>" + chart_total
        + "<h2>Tests added per iteration</h2>" + chart_added
        + "<h2>Test runtime</h2>"
        '<p class="meta">Pulled from each iteration\'s verification '
        'line (e.g. <code>457 passed in 18.3s</code>). Some early '
        'iterations omitted the seconds suffix and are excluded.</p>'
        + chart_runtime
        + run_block
        + nav_html
    )
    return _page_template("Testing — geno-voice", body_html)


# ---- iter-036: performance report (per-scenario bar charts) -----------------


def _svg_horizontal_bars(
    rows: list[tuple[str, float]],
    *,
    title: str,
    x_label: str,
    width: int = 720,
    bar_height: int = 22,
    color: str = "#7aa2f7",
    units: str = "ms",
) -> str:
    """Render a horizontal bar chart with one row per (label, value).
    Used for per-scenario performance comparisons.

    Each row's label sits left of the bar; the numeric value renders
    at the bar's right edge.
    """
    if not rows:
        return f'<div class="chart-empty">{html.escape(title)}: no data</div>'

    pad_l, pad_r, pad_t, pad_b = 200, 80, 30, 24
    inner_w = width - pad_l - pad_r
    height = pad_t + pad_b + len(rows) * (bar_height + 6)

    values = [v for _, v in rows]
    v_max = max(max(values), 1) * 1.08

    bars: list[str] = []
    for i, (label, v) in enumerate(rows):
        y = pad_t + i * (bar_height + 6)
        bw = (v / v_max) * inner_w if v_max > 0 else 0
        bars.append(
            f'<rect x="{pad_l}" y="{y}" width="{bw:.1f}" height="{bar_height}" '
            f'fill="{color}"/>'
        )
        bars.append(
            f'<text x="{pad_l - 8}" y="{y + bar_height * 0.7:.1f}" '
            f'text-anchor="end" class="axis-label">'
            f'{html.escape(label)}</text>'
        )
        # Value at end of bar.
        bars.append(
            f'<text x="{pad_l + bw + 6:.1f}" y="{y + bar_height * 0.7:.1f}" '
            f'class="value">'
            f'{v:.0f}{html.escape(units)}</text>'
        )

    # X-axis tick labels: 0, mid, max.
    ticks = []
    for frac in (0.0, 0.5, 1.0):
        x = pad_l + frac * inner_w
        v = frac * v_max
        ticks.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t - 4}" '
            f'y2="{height - pad_b}" class="grid"/>'
            f'<text x="{x:.1f}" y="{height - pad_b + 12}" '
            f'text-anchor="middle" class="axis">{int(v)}</text>'
        )

    return f"""<svg viewBox="0 0 {width} {height}" class="chart hbar"
xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{html.escape(title)}">
<text x="14" y="18" class="chart-title">{html.escape(title)}</text>
<text x="{pad_l + inner_w / 2}" y="{height - 4}" text-anchor="middle"
  class="chart-axis-label">{html.escape(x_label)}</text>
{''.join(ticks)}
{''.join(bars)}
</svg>"""


def _load_perf_results(path: Path) -> dict | None:
    """Load the JSON dumped by tests/performance/. Return None if
    the file doesn't exist or fails to parse.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _load_perf_history(reports_dir: Path) -> list[dict]:
    """iter-039: load all per-iteration perf snapshots from
    ``perf-iter-NNN.json`` files in ``reports_dir``, sorted by
    iteration. Returns a list of payloads. Empty list if none.
    """
    out: list[dict] = []
    if not reports_dir.exists():
        return out
    for path in sorted(reports_dir.glob("perf-iter-*.json")):
        m = re.match(r"perf-iter-(\d{3})\.json$", path.name)
        if not m:
            continue
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        payload.setdefault("iteration", m.group(1))
        out.append(payload)
    out.sort(key=lambda p: p.get("iteration", ""))
    return out


def _svg_multi_line_chart(
    series_by_label: dict,
    *,
    title: str,
    y_label: str,
    width: int = 720,
    height: int = 320,
) -> str:
    """iter-039: SVG line chart with one polyline per series — used
    for "metric over iterations, one line per scenario". Each series
    gets a color from a fixed palette so the same scenario looks the
    same across charts.

    `series_by_label` maps scenario_name → list of (iteration_number,
    value) tuples.
    """
    if not series_by_label or not any(s for s in series_by_label.values()):
        return f'<div class="chart-empty">{html.escape(title)}: no data</div>'

    palette = ["#7aa2f7", "#9ece6a", "#bb9af7", "#e0af68", "#f7768e",
               "#7dcfff", "#f7c453", "#c0caf5"]
    pad_l, pad_r, pad_t, pad_b = 56, 200, 30, 36
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b

    all_xs: list[float] = []
    all_ys: list[float] = []
    for pts in series_by_label.values():
        for x, y in pts:
            all_xs.append(x)
            all_ys.append(y)
    if not all_xs:
        return f'<div class="chart-empty">{html.escape(title)}: no data</div>'
    x_min, x_max = min(all_xs), max(all_xs)
    y_min = 0
    y_max = max(max(all_ys), 1) * 1.08

    def sx(x: float) -> float:
        if x_max == x_min:
            return pad_l + inner_w / 2
        return pad_l + (x - x_min) / (x_max - x_min) * inner_w

    def sy(y: float) -> float:
        return pad_t + inner_h - (y - y_min) / (y_max - y_min) * inner_h

    y_ticks = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = y_min + frac * (y_max - y_min)
        y = sy(v)
        y_ticks.append(
            f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y:.1f}" y2="{y:.1f}" '
            f'class="grid"/>'
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" '
            f'text-anchor="end" class="axis">{int(round(v))}</text>'
        )

    x_ticks = []
    if x_max == x_min:
        x_ticks.append(
            f'<text x="{sx(x_min):.1f}" y="{height - pad_b + 18}" '
            f'text-anchor="middle" class="axis">'
            f'iter-{int(round(x_min)):03d}</text>'
        )
    else:
        for x in (x_min, (x_min + x_max) / 2, x_max):
            x_ticks.append(
                f'<text x="{sx(x):.1f}" y="{height - pad_b + 18}" '
                f'text-anchor="middle" class="axis">'
                f'iter-{int(round(x)):03d}</text>'
            )

    series_html: list[str] = []
    legend_html: list[str] = []
    for i, (label, pts) in enumerate(series_by_label.items()):
        if not pts:
            continue
        color = palette[i % len(palette)]
        pts_sorted = sorted(pts, key=lambda p: p[0])
        if len(pts_sorted) == 1:
            x, y = pts_sorted[0]
            series_html.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" '
                f'fill="{color}"/>'
            )
        else:
            d = " ".join(
                ("M" if j == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}"
                for j, (x, y) in enumerate(pts_sorted)
            )
            series_html.append(
                f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
            for x, y in pts_sorted:
                series_html.append(
                    f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" '
                    f'fill="{color}"/>'
                )
        ly = pad_t + 4 + i * 18
        lx = width - pad_r + 12
        legend_html.append(
            f'<rect x="{lx}" y="{ly}" width="14" height="3" fill="{color}"/>'
            f'<text x="{lx + 20}" y="{ly + 5}" class="legend">'
            f'{html.escape(label)}</text>'
        )

    return f"""<svg viewBox="0 0 {width} {height}" class="chart"
xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{html.escape(title)}">
<text x="{pad_l}" y="18" class="chart-title">{html.escape(title)}</text>
<text x="{pad_l}" y="{height - 4}" class="chart-axis-label">iteration</text>
<text x="14" y="{pad_t + inner_h / 2}" class="chart-axis-label" transform="rotate(-90 14,{pad_t + inner_h / 2})" text-anchor="middle">{html.escape(y_label)}</text>
{''.join(y_ticks)}
{''.join(x_ticks)}
{''.join(series_html)}
{''.join(legend_html)}
</svg>"""


def render_performance_page(perf_payload: dict | None, history: list[dict] | None = None) -> str:
    """Build iter-reports/performance.html — per-scenario bar charts
    of TTFS, STT, TTS, LLM-1st-token, and wall time.

    If ``perf_payload`` is None (no JSON yet), render a placeholder
    page that explains how to populate it.
    """
    nav_html = (
        '<nav>'
        '<a href="index.html">← Index</a>'
        '<a href="testing.html">← Testing</a>'
        '</nav>'
    )

    if not perf_payload or not perf_payload.get("scenarios"):
        body_html = (
            nav_html
            + '<header><h1>Performance</h1>'
            '<div class="meta">No perf-results.json yet</div></header>'
            '<p>Run the performance suite to populate this page:</p>'
            '<pre class="code-block"><code>python -m pytest tests/performance/</code></pre>'
            '<p>The suite drives <code>ChatLoop</code> across simulated '
            'scenarios (short/long utterance, slow LLM, slow TTS, fillers '
            'enabled, etc.) and dumps timings to '
            '<code>iter-reports/perf-results.json</code>. This page '
            "renders bar charts of those timings per scenario.</p>"
            + nav_html
        )
        return _page_template("Performance — geno-voice", body_html)

    scenarios = perf_payload["scenarios"]
    captured_at = perf_payload.get("captured_at", "unknown")

    def _rows(metric_key: str) -> list[tuple[str, float]]:
        return [(s["name"], s.get(metric_key, 0.0)) for s in scenarios]

    chart_ttfs = _svg_horizontal_bars(
        _rows("ttfs_ms"),
        title="TTFS by scenario",
        x_label="ms (lower is better)",
        color="#7aa2f7",
    )
    chart_stt = _svg_horizontal_bars(
        _rows("stt_ms"),
        title="STT time by scenario",
        x_label="ms",
        color="#9ece6a",
    )
    chart_tts = _svg_horizontal_bars(
        _rows("tts_ms"),
        title="TTS time by scenario",
        x_label="ms (cumulative across sentences)",
        color="#bb9af7",
    )
    chart_llm = _svg_horizontal_bars(
        _rows("llm_first_token_ms"),
        title="LLM first-token by scenario",
        x_label="ms",
        color="#e0af68",
    )
    chart_wall = _svg_horizontal_bars(
        _rows("wall_ms"),
        title="Wall-clock time by scenario",
        x_label="ms",
        color="#f7768e",
    )

    # Scenario description table for context.
    rows_html: list[str] = []
    for s in scenarios:
        rows_html.append(
            f"<tr>"
            f"<td><code>{html.escape(s['name'])}</code></td>"
            f"<td>{html.escape(s['description'])}</td>"
            f"<td class='num'>{int(s['sentences_spoken'])}</td>"
            f"<td>{'yes' if s.get('barge_in') else 'no'}</td>"
            f"</tr>"
        )
    table_html = (
        '<table class="perf-table"><thead><tr>'
        '<th>Scenario</th><th>Description</th>'
        '<th>Sentences</th><th>Barge-in</th>'
        '</tr></thead><tbody>'
        + "".join(rows_html)
        + '</tbody></table>'
    )

    body_html = (
        nav_html
        + '<header><h1>Performance</h1>'
        f'<div class="meta">Captured at <code>{html.escape(captured_at)}</code> '
        f'· {len(scenarios)} scenario{"" if len(scenarios) == 1 else "s"} '
        f'· stub LLM + stub TTS</div></header>'
        '<p>Each scenario drives one full <code>ChatLoop.run_one_turn</code> '
        'on virtual audio. Stubs are used for STT / LLM / TTS so the numbers '
        'reflect <em>pipeline overhead</em>, not neural-net latency. '
        'Real-engine perf testing belongs in a separate live suite.</p>'
        + table_html
        + '<h2>Latest snapshot</h2>'
        '<p class="meta">Most recent run; one bar per scenario per metric.</p>'
        '<h3>Time-to-first-speech (TTFS)</h3>'
        + chart_ttfs
        + '<h3>STT time</h3>' + chart_stt
        + '<h3>TTS time</h3>' + chart_tts
        + '<h3>LLM first-token</h3>' + chart_llm
        + '<h3>Wall-clock turn time</h3>' + chart_wall
        + _render_perf_history_section(history or [])
        + '<h2>Refresh the data</h2>'
        '<pre class="code-block"><code>python -m pytest tests/performance/'
        '\npython scripts/generate_iteration_reports.py'
        '</code></pre>'
        + nav_html
    )
    return _page_template("Performance — geno-voice", body_html)


def _render_perf_history_section(history: list[dict]) -> str:
    """iter-039: render a "metric over iterations" section using the
    multi-line chart helper. One chart per metric; one line per
    scenario; x-axis = iteration number.

    If only one iteration of history exists, render a soft note
    rather than a sparse chart — multi-line charts need at least
    two iterations to convey trend.
    """
    if not history:
        return ""
    if len(history) < 2:
        only = history[0].get("iteration", "?")
        return (
            '<h2>Across iterations</h2>'
            f'<p class="meta">Only one iteration captured so far '
            f'(<code>iter-{html.escape(only)}</code>). The time-series '
            'view will populate as more snapshots are collected — '
            'each iteration\'s perf run writes <code>iter-reports/'
            'perf-iter-NNN.json</code>.</p>'
        )

    # Build {scenario_name: [(iter_num, value), ...]} for each metric.
    def _build(metric: str) -> dict:
        out: dict = {}
        for snap in history:
            try:
                it_num = float(int(snap.get("iteration", "0")))
            except (ValueError, TypeError):
                continue
            for s in snap.get("scenarios", []):
                name = s.get("name")
                if not name:
                    continue
                v = s.get(metric)
                if not isinstance(v, (int, float)):
                    continue
                out.setdefault(name, []).append((it_num, float(v)))
        return out

    chart_ttfs = _svg_multi_line_chart(
        _build("ttfs_ms"), title="TTFS over iterations", y_label="ms",
    )
    chart_wall = _svg_multi_line_chart(
        _build("wall_ms"), title="Wall over iterations", y_label="ms",
    )
    chart_tts = _svg_multi_line_chart(
        _build("tts_ms"), title="TTS over iterations", y_label="ms",
    )
    chart_stt = _svg_multi_line_chart(
        _build("stt_ms"), title="STT over iterations", y_label="ms",
    )

    return (
        '<h2>Across iterations</h2>'
        f'<p class="meta">Time-series across '
        f'<strong>{len(history)}</strong> captured iterations. '
        'One line per scenario.</p>'
        '<h3>TTFS</h3>' + chart_ttfs
        + '<h3>Wall-clock turn time</h3>' + chart_wall
        + '<h3>TTS</h3>' + chart_tts
        + '<h3>STT</h3>' + chart_stt
    )


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

    # iter-035: testing posture page, with SVG plots.
    testing_path = OUT_DIR / "testing.html"
    testing_path.write_text(render_testing_page(iterations, REPO_ROOT))

    # iter-036: performance page (per-scenario bar charts). Always
    # written — when no JSON exists it shows the "run the suite"
    # placeholder so the link from testing.html / iter pages
    # doesn't 404.
    perf_payload = _load_perf_results(PERF_RESULTS_PATH)
    perf_history = _load_perf_history(OUT_DIR)
    perf_path = OUT_DIR / "performance.html"
    perf_path.write_text(render_performance_page(perf_payload, perf_history))

    print(
        f"Wrote {len(iterations)} iteration reports + "
        f"index + testing.html + performance.html to {OUT_DIR}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
