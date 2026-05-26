"""STT benchmark CLI — runs an STT engine through the 5-fixture
WER corpus and reports per-fixture pass/fail.

iter-132: Operators evaluating a new STTEngine subclass need a
quick way to see how it stacks up against the iter-117–127
audio corpus. This script is the answer:

    python scripts/run_stt_benchmark.py --engine faster_whisper
    python scripts/run_stt_benchmark.py --engine faster_whisper \\
        --model tiny --device cpu --compute int8

Each entry in `tests/fixtures/wer/corpus.json:audio_fixtures`
is transcribed through the chosen engine. The benchmark
computes WER against the reference and reports pass/fail
against the recorded `[expected_wer_min, expected_wer_max]`
band.

Design seams (mirrors iter-108 / iter-119):

  - ``run_benchmark`` is a pure function: takes a transcribe
    callable + corpus list + log callable. No engine or file I/O
    in the function itself. Tests pass stub callables.
  - The CLI parses args, builds the engine via ``stt.get_engine``,
    closes the loop with ``run_benchmark`` + a print log.
  - Per-fixture timing recorded so operators can compare engine
    speed alongside accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examples._chat_wer import compute_wer  # noqa: E402

CORPUS_PATH = ROOT / "tests" / "fixtures" / "wer" / "corpus.json"


@dataclass
class FixtureResult:
    """One fixture's benchmark row."""

    name: str
    reference: str
    hypothesis: str
    wer: float
    expected_min: float
    expected_max: float
    elapsed_seconds: float
    passed: bool


@dataclass
class BenchmarkSummary:
    """Aggregate of all fixture results."""

    results: list[FixtureResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passing(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failing(self) -> int:
        return self.total - self.passing

    @property
    def total_elapsed(self) -> float:
        return sum(r.elapsed_seconds for r in self.results)


def run_benchmark(
    transcribe: Callable[[str], str],
    fixtures: list[dict],
    fixture_dir: Path,
    *,
    log: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
    verbose: bool = True,
) -> BenchmarkSummary:
    """Run a benchmark over the given fixtures.

    Args:
        transcribe: callable taking an audio file path, returning
            transcript text. The CLI wraps an STTEngine; tests
            pass a deterministic stub.
        fixtures: list of fixture dicts (the contents of
            ``corpus.json:audio_fixtures``).
        fixture_dir: directory containing the audio files
            referenced by ``audio_path``.
        log: emit callable for the per-row report. Default
            ``print``. Ignored when ``verbose=False``.
        clock: monotonic-time source for per-fixture elapsed.
            Default ``time.monotonic``.
        verbose: emit per-row + summary text via ``log``. When
            False, run silently — used by the JSON/CSV output
            paths in iter-133 which format the summary
            post-run instead. Default True (preserves iter-132
            behavior).

    Returns:
        A ``BenchmarkSummary`` with one ``FixtureResult`` per
        fixture entry. Caller decides exit status (0 if all
        pass, 1 otherwise).

    Format of the per-row log line (verbose=True):

        clean_audio          PASS  WER 0.20  band [0.00, 0.40]  elapsed 0.85s

    A trailing summary line emits at the end:

        5/5 fixtures passed in 4.2s
    """
    summary = BenchmarkSummary()
    for f in fixtures:
        path = fixture_dir / f["audio_path"]
        t0 = clock()
        hypothesis = transcribe(str(path))
        elapsed = clock() - t0
        wer = compute_wer(f["reference"], hypothesis)
        passed = (
            f["expected_wer_min"] <= wer <= f["expected_wer_max"]
        )
        result = FixtureResult(
            name=f["name"],
            reference=f["reference"],
            hypothesis=hypothesis,
            wer=wer,
            expected_min=f["expected_wer_min"],
            expected_max=f["expected_wer_max"],
            elapsed_seconds=elapsed,
            passed=passed,
        )
        summary.results.append(result)
        if verbose:
            status = "PASS" if passed else "FAIL"
            log(
                f"{f['name']:25s} {status:4s}  "
                f"WER {wer:.2f}  "
                f"band [{f['expected_wer_min']:.2f}, {f['expected_wer_max']:.2f}]  "
                f"elapsed {elapsed:.2f}s"
            )

    if verbose:
        log(
            f"\n{summary.passing}/{summary.total} fixtures passed "
            f"in {summary.total_elapsed:.1f}s"
        )
    return summary


def format_summary_json(summary: BenchmarkSummary, *, indent: int = 2) -> str:
    """iter-133: serialize a ``BenchmarkSummary`` as JSON.

    Output shape:

    .. code-block:: json

        {
          "passing": 5,
          "failing": 0,
          "total": 5,
          "total_elapsed_seconds": 4.32,
          "results": [
            {
              "name": "clean_audio",
              "reference": "...",
              "hypothesis": "...",
              "wer": 0.20,
              "expected_min": 0.0,
              "expected_max": 0.4,
              "elapsed_seconds": 1.93,
              "passed": true
            },
            ...
          ]
        }

    ``indent=2`` for human-readable output; pass ``indent=None``
    for compact one-line JSON suitable for piping.
    """
    payload = {
        "passing": summary.passing,
        "failing": summary.failing,
        "total": summary.total,
        "total_elapsed_seconds": summary.total_elapsed,
        "results": [
            {
                "name": r.name,
                "reference": r.reference,
                "hypothesis": r.hypothesis,
                "wer": r.wer,
                "expected_min": r.expected_min,
                "expected_max": r.expected_max,
                "elapsed_seconds": r.elapsed_seconds,
                "passed": r.passed,
            }
            for r in summary.results
        ],
    }
    return json.dumps(payload, indent=indent)


def format_summary_csv(summary: BenchmarkSummary) -> str:
    """iter-133: serialize a ``BenchmarkSummary`` as CSV.

    Header row + one data row per fixture. No summary aggregate
    in the CSV — operators can compute it from the rows. Fields:

        name,passed,wer,expected_min,expected_max,elapsed_seconds,reference,hypothesis

    Strings are quoted and any embedded quotes are doubled
    (RFC 4180). Suitable for ``pandas.read_csv`` or spreadsheet
    import.
    """
    import csv as _csv
    import io as _io

    out = _io.StringIO()
    writer = _csv.writer(out, quoting=_csv.QUOTE_MINIMAL)
    writer.writerow([
        "name", "passed", "wer",
        "expected_min", "expected_max",
        "elapsed_seconds", "reference", "hypothesis",
    ])
    for r in summary.results:
        writer.writerow([
            r.name, r.passed, f"{r.wer:.4f}",
            f"{r.expected_min:.4f}", f"{r.expected_max:.4f}",
            f"{r.elapsed_seconds:.4f}",
            r.reference, r.hypothesis,
        ])
    return out.getvalue()


# iter-134: diff mode. Operators run the benchmark, save the JSON,
# make a change, re-run with --diff <saved.json> to see what
# changed. Fixture-level deltas + status flips highlight
# regressions and improvements without forcing a manual diff.


@dataclass
class FixtureDiff:
    """One fixture's diff between current and baseline runs.

    A diff entry covers four cases:
    - matched (both runs have the fixture): WER + status flip
      computed normally
    - new (current only): baseline_* fields are None
    - removed (baseline only): current_* fields are None
    - both missing: not represented (would be a no-op entry)
    """

    name: str
    current_wer: float | None
    baseline_wer: float | None
    current_passed: bool | None
    baseline_passed: bool | None

    @property
    def wer_delta(self) -> float | None:
        """Current minus baseline. None when either side is None."""
        if self.current_wer is None or self.baseline_wer is None:
            return None
        return self.current_wer - self.baseline_wer

    @property
    def status_change(self) -> str:
        """Categorize the change for display:
        - "new"        : current only
        - "removed"    : baseline only
        - "regressed"  : was PASS, now FAIL
        - "improved"   : was FAIL, now PASS
        - "unchanged"  : both have same passed status
        """
        if self.baseline_passed is None and self.current_passed is not None:
            return "new"
        if self.current_passed is None and self.baseline_passed is not None:
            return "removed"
        if self.baseline_passed and not self.current_passed:
            return "regressed"
        if not self.baseline_passed and self.current_passed:
            return "improved"
        return "unchanged"


@dataclass
class BenchmarkDiff:
    """Aggregate diff between a current benchmark run and a
    baseline."""

    fixture_diffs: list[FixtureDiff] = field(default_factory=list)
    current_passing: int = 0
    current_total: int = 0
    baseline_passing: int = 0
    baseline_total: int = 0

    @property
    def regressions(self) -> list[FixtureDiff]:
        return [d for d in self.fixture_diffs if d.status_change == "regressed"]

    @property
    def improvements(self) -> list[FixtureDiff]:
        return [d for d in self.fixture_diffs if d.status_change == "improved"]

    @property
    def new_fixtures(self) -> list[FixtureDiff]:
        return [d for d in self.fixture_diffs if d.status_change == "new"]

    @property
    def removed_fixtures(self) -> list[FixtureDiff]:
        return [d for d in self.fixture_diffs if d.status_change == "removed"]


def compute_diff(
    current: BenchmarkSummary, baseline: dict,
) -> BenchmarkDiff:
    """Compute a diff between a current ``BenchmarkSummary`` and
    a parsed baseline JSON (as returned by ``json.load`` on
    ``format_summary_json`` output).

    Output ordering:
    - All fixtures present in BOTH runs come first, in the order
      they appear in `current.results`.
    - "new" fixtures (current only) follow.
    - "removed" fixtures (baseline only) come last.

    This ordering puts the most actionable rows (matching
    fixtures) first, while still surfacing corpus changes.
    """
    diff = BenchmarkDiff()
    diff.current_passing = current.passing
    diff.current_total = current.total
    diff.baseline_passing = baseline.get("passing", 0)
    diff.baseline_total = baseline.get("total", 0)

    baseline_by_name = {
        r["name"]: r for r in baseline.get("results", [])
    }

    seen_names: set[str] = set()
    for r in current.results:
        b = baseline_by_name.get(r.name)
        if b is None:
            diff.fixture_diffs.append(FixtureDiff(
                name=r.name,
                current_wer=r.wer,
                baseline_wer=None,
                current_passed=r.passed,
                baseline_passed=None,
            ))
        else:
            diff.fixture_diffs.append(FixtureDiff(
                name=r.name,
                current_wer=r.wer,
                baseline_wer=b.get("wer"),
                current_passed=r.passed,
                baseline_passed=b.get("passed"),
            ))
        seen_names.add(r.name)

    # Removed fixtures: in baseline but not current.
    for r in baseline.get("results", []):
        if r["name"] not in seen_names:
            diff.fixture_diffs.append(FixtureDiff(
                name=r["name"],
                current_wer=None,
                baseline_wer=r.get("wer"),
                current_passed=None,
                baseline_passed=r.get("passed"),
            ))

    return diff


def format_diff_text(diff: BenchmarkDiff) -> str:
    """Render a ``BenchmarkDiff`` as human-readable text.

    Per-fixture rows show:

        clean_audio          PASS  WER 0.20 -> 0.20  Δ +0.000
        noisy_audio          PASS  WER 0.20 -> 0.25  Δ +0.050
        multispeaker_audio   FAIL  WER 0.80 -> 1.20  Δ +0.400 (regressed)

    Trailing summary:

        4/5 → 5/5 fixtures passing (+1)
        Improvements: noisy_audio
        Regressions: multispeaker_audio
        New fixtures: extra_audio
        Removed fixtures: legacy_audio
    """
    lines: list[str] = []
    for d in diff.fixture_diffs:
        status_change = d.status_change
        if d.current_passed is None:
            status = "REMOV"
            cur_wer_s = "—"
        elif d.current_passed:
            status = "PASS"
            cur_wer_s = f"{d.current_wer:.2f}"
        else:
            status = "FAIL"
            cur_wer_s = f"{d.current_wer:.2f}"

        if d.baseline_wer is None:
            base_wer_s = "—"
        else:
            base_wer_s = f"{d.baseline_wer:.2f}"

        if d.wer_delta is None:
            delta_s = "    "
        else:
            sign = "+" if d.wer_delta >= 0 else ""
            delta_s = f"Δ {sign}{d.wer_delta:.3f}"

        marker = ""
        if status_change in ("regressed", "improved", "new", "removed"):
            marker = f" ({status_change})"

        lines.append(
            f"{d.name:25s} {status:5s}  "
            f"WER {base_wer_s} -> {cur_wer_s}  "
            f"{delta_s}{marker}"
        )

    lines.append("")
    delta = diff.current_passing - diff.baseline_passing
    sign = "+" if delta >= 0 else ""
    lines.append(
        f"{diff.baseline_passing}/{diff.baseline_total} → "
        f"{diff.current_passing}/{diff.current_total} fixtures passing "
        f"({sign}{delta})"
    )

    if diff.improvements:
        names = ", ".join(d.name for d in diff.improvements)
        lines.append(f"Improvements: {names}")
    if diff.regressions:
        names = ", ".join(d.name for d in diff.regressions)
        lines.append(f"Regressions: {names}")
    if diff.new_fixtures:
        names = ", ".join(d.name for d in diff.new_fixtures)
        lines.append(f"New fixtures: {names}")
    if diff.removed_fixtures:
        names = ", ".join(d.name for d in diff.removed_fixtures)
        lines.append(f"Removed fixtures: {names}")

    return "\n".join(lines)


def format_diff_json(diff: BenchmarkDiff, *, indent: int = 2) -> str:
    """iter-135: serialize a ``BenchmarkDiff`` as JSON.

    Output shape (parallel to ``format_summary_json`` for the
    summary path):

    .. code-block:: json

        {
          "current_passing": 5, "current_total": 5,
          "baseline_passing": 4, "baseline_total": 5,
          "passing_delta": 1,
          "regression_count": 0,
          "improvement_count": 1,
          "new_count": 0,
          "removed_count": 0,
          "fixture_diffs": [
            {
              "name": "noisy_audio",
              "current_wer": 0.20, "baseline_wer": 0.30,
              "wer_delta": -0.10,
              "current_passed": true, "baseline_passed": false,
              "status_change": "improved"
            },
            ...
          ]
        }

    Top-level aggregates surface the headline numbers without
    the caller iterating fixture_diffs. ``passing_delta`` is
    ``current_passing - baseline_passing`` — positive means the
    benchmark improved overall.

    None values (for new/removed fixtures) serialize as JSON
    ``null``.
    """
    payload = {
        "current_passing": diff.current_passing,
        "current_total": diff.current_total,
        "baseline_passing": diff.baseline_passing,
        "baseline_total": diff.baseline_total,
        "passing_delta": diff.current_passing - diff.baseline_passing,
        "regression_count": len(diff.regressions),
        "improvement_count": len(diff.improvements),
        "new_count": len(diff.new_fixtures),
        "removed_count": len(diff.removed_fixtures),
        "fixture_diffs": [
            {
                "name": d.name,
                "current_wer": d.current_wer,
                "baseline_wer": d.baseline_wer,
                "wer_delta": d.wer_delta,
                "current_passed": d.current_passed,
                "baseline_passed": d.baseline_passed,
                "status_change": d.status_change,
            }
            for d in diff.fixture_diffs
        ],
    }
    return json.dumps(payload, indent=indent)


def format_diff_csv(diff: BenchmarkDiff) -> str:
    """iter-135: serialize a ``BenchmarkDiff`` as CSV. Header
    row + one row per fixture diff. Mirrors
    ``format_summary_csv``'s shape but with diff-specific
    columns:

        name,status_change,current_wer,baseline_wer,wer_delta,
        current_passed,baseline_passed

    None values render as empty strings (RFC-4180 idiomatic for
    "missing"). Numeric fields use 4 decimals — same precision
    as the summary CSV.
    """
    import csv as _csv
    import io as _io

    out = _io.StringIO()
    writer = _csv.writer(out, quoting=_csv.QUOTE_MINIMAL)
    writer.writerow([
        "name", "status_change",
        "current_wer", "baseline_wer", "wer_delta",
        "current_passed", "baseline_passed",
    ])

    def _fmt_optional(v):
        """None → empty string. Float → 4-decimal. Bool/str →
        verbatim."""
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return v

    for d in diff.fixture_diffs:
        writer.writerow([
            d.name, d.status_change,
            _fmt_optional(d.current_wer),
            _fmt_optional(d.baseline_wer),
            _fmt_optional(d.wer_delta),
            _fmt_optional(d.current_passed),
            _fmt_optional(d.baseline_passed),
        ])
    return out.getvalue()


def _build_transcribe_from_engine_args(
    engine: str, model: str, device: str, compute: str,
    *, beam_size: int = 1, temperature: float = 0.0,
) -> Callable[[str], str]:
    """Construct a transcribe closure from CLI args.

    iter-132 design choice: for ``faster_whisper`` specifically,
    bypass the ``FasterWhisperEngine`` wrapper and call the
    underlying ``WhisperModel.transcribe`` directly with greedy
    decoding (``beam_size=1, temperature=0``). This matches
    iter-125's "deterministic decoding for benchmark assertions"
    rule — the recorded WER bands assume greedy output, and
    using default beam-search makes multispeaker (and other
    edge-case fixtures) fail or pass non-deterministically.

    For other engines (``whisper``, ``gemma4``), the CLI uses
    the standard ``stt.get_engine`` factory + ``transcribe``
    path. If those engines have non-deterministic output, that's
    their concern; this CLI just runs them.
    """
    if engine == "faster_whisper":
        # Direct path: construct the model, transcribe with
        # greedy kwargs. Skip the engine wrapper entirely.
        from faster_whisper import WhisperModel
        from stt.faster_whisper_engine import _resolve_model_repo
        resolved = _resolve_model_repo(model or "tiny")
        m = WhisperModel(
            resolved, device=device, compute_type=compute,
        )

        def transcribe(audio_path: str) -> str:
            segments, _info = m.transcribe(
                audio_path, language="en",
                beam_size=beam_size, temperature=temperature,
            )
            return " ".join(s.text for s in segments).strip()

        return transcribe

    # Generic path for non-faster_whisper engines: use the
    # registered engine wrapper, accept whatever decoding it
    # provides.
    from stt import get_engine
    kwargs: dict = {}
    if model:
        kwargs["model_repo"] = model
    instance = get_engine(engine, **kwargs)

    def transcribe(audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            wav_bytes = f.read()
        result = instance.transcribe(wav_bytes)
        if isinstance(result, tuple):
            return result[0] or ""
        return result or ""

    return transcribe


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run STT benchmark against the WER fixture corpus.",
    )
    parser.add_argument(
        "--engine", default="faster_whisper",
        help="STT engine name (registered in stt.ENGINES). "
             "Default: faster_whisper.",
    )
    parser.add_argument(
        "--model", default="",
        help="Model repo / size string. Default: engine's own default.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device (faster_whisper only). Default: cpu.",
    )
    parser.add_argument(
        "--compute", default="int8",
        help="Compute type (faster_whisper only). Default: int8.",
    )
    parser.add_argument(
        "--format", default="text",
        choices=["text", "json", "csv"],
        help="Output format. text (default) — human-readable per-row. "
             "json — full result dump suitable for piping. "
             "csv — header + one row per fixture for spreadsheets.",
    )
    parser.add_argument(
        "--diff", default="",
        help="Path to a baseline JSON file (from a previous "
             "--format json run). When set, output shows a diff "
             "highlighting per-fixture WER changes + status flips. "
             "iter-135: --format chooses how the diff is rendered "
             "(text default, json, csv).",
    )
    args = parser.parse_args()

    with CORPUS_PATH.open() as f:
        corpus = json.load(f)
    fixtures = corpus.get("audio_fixtures", [])
    if not fixtures:
        print(
            "no audio_fixtures in corpus.json — nothing to benchmark",
            file=sys.stderr,
        )
        return 1

    try:
        transcribe = _build_transcribe_from_engine_args(
            args.engine, args.model, args.device, args.compute,
        )
    except Exception as e:
        print(f"engine construction failed: {e}", file=sys.stderr)
        return 2

    # iter-134: --diff loads a baseline JSON before running and
    # formats the result as a diff. Overrides --format (text-only).
    baseline = None
    if args.diff:
        try:
            with open(args.diff) as bf:
                baseline = json.load(bf)
        except FileNotFoundError:
            print(
                f"baseline JSON not found: {args.diff}",
                file=sys.stderr,
            )
            return 3
        except json.JSONDecodeError as e:
            print(
                f"baseline JSON parse error in {args.diff}: {e}",
                file=sys.stderr,
            )
            return 3

    # iter-133: --format controls output. For text (default),
    # run_benchmark emits per-row + summary inline. For json/csv,
    # silence run_benchmark and dump the formatted output after.
    # iter-134: when --diff is active, run silently and emit the
    # diff. iter-135: --format now dispatches the diff renderer
    # (text/json/csv) the same way it dispatches the summary.
    if baseline is not None:
        summary = run_benchmark(
            transcribe, fixtures, CORPUS_PATH.parent,
            verbose=False,
        )
        diff = compute_diff(summary, baseline)
        if args.format == "json":
            print(format_diff_json(diff))
        elif args.format == "csv":
            print(format_diff_csv(diff), end="")
        else:
            print(format_diff_text(diff))
    elif args.format == "text":
        summary = run_benchmark(
            transcribe, fixtures, CORPUS_PATH.parent,
        )
    else:
        summary = run_benchmark(
            transcribe, fixtures, CORPUS_PATH.parent,
            verbose=False,
        )
        if args.format == "json":
            print(format_summary_json(summary))
        elif args.format == "csv":
            # CSV ends in a newline already; print(..., end="")
            # to avoid a blank trailing line.
            print(format_summary_csv(summary), end="")

    return 0 if summary.failing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
