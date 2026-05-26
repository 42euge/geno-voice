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

    # iter-133: --format controls output. For text (default),
    # run_benchmark emits per-row + summary inline. For json/csv,
    # silence run_benchmark and dump the formatted output after.
    if args.format == "text":
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
