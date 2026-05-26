"""Tests for iter-132 — STT benchmark CLI.

Covers the pure ``run_benchmark`` function with stubbed
transcribe callables. Doesn't construct a real STT engine —
that's the CLI's job. Pure-function tests run regardless of
faster-whisper availability.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_stt_benchmark.py"

sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("run_stt_benchmark", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules["run_stt_benchmark"] = module
spec.loader.exec_module(module)


def _capture():
    lines: list[str] = []

    def log(line: str) -> None:
        lines.append(line)

    return log, lines


# ---- Fixture helpers ---------------------------------------------------


def _fix(name, ref, hyp_via_transcribe, lo, hi):
    return {
        "name": name,
        "reference": ref,
        "audio_path": f"{name}.wav",
        "expected_wer_min": lo,
        "expected_wer_max": hi,
    }


def _fixed_transcribe(reply: str | dict[str, str]):
    """Return a transcribe stub. If `reply` is a dict, look up
    by audio_path basename; else always return the string."""
    if isinstance(reply, dict):
        def _stub(path: str) -> str:
            return reply[Path(path).name]
        return _stub

    def _stub(path: str) -> str:
        return reply
    return _stub


def _fake_clock():
    """Monotonic clock with explicit step on each tick. Returns
    a closure that increments by 0.1s per call."""
    state = {"t": 0.0}

    def _now() -> float:
        state["t"] += 0.1
        return state["t"]

    return _now


# ---- Single-fixture happy path -------------------------------------


def test_single_perfect_fixture_passes():
    log, lines = _capture()
    fixtures = [_fix("clean", "hello world", "hello world", 0.0, 0.2)]
    summary = module.run_benchmark(
        _fixed_transcribe("hello world"),
        fixtures,
        Path("/tmp/anything"),
        log=log,
        clock=_fake_clock(),
    )
    assert summary.total == 1
    assert summary.passing == 1
    assert summary.failing == 0
    assert summary.results[0].wer == 0.0
    assert summary.results[0].passed is True


def test_single_fixture_below_band_fails():
    """If WER is BELOW expected_min, the fixture fails. Catches
    'engine got better but corpus didn't update' drift."""
    log, lines = _capture()
    fixtures = [_fix("clean", "hello world", "hello world", 0.10, 0.40)]
    summary = module.run_benchmark(
        _fixed_transcribe("hello world"),  # WER 0.0
        fixtures, Path("/tmp"), log=log, clock=_fake_clock(),
    )
    assert summary.failing == 1
    assert summary.results[0].passed is False


def test_single_fixture_above_band_fails():
    """The common failure direction: model regressed."""
    log, lines = _capture()
    fixtures = [_fix("clean", "hello world", None, 0.0, 0.10)]
    summary = module.run_benchmark(
        _fixed_transcribe("totally wrong words now"),  # high WER
        fixtures, Path("/tmp"), log=log, clock=_fake_clock(),
    )
    assert summary.failing == 1


# ---- Multi-fixture summary ---------------------------------------


def test_multi_fixture_partial_failure_summary():
    """3 fixtures, 2 pass, 1 fails. Summary reflects the mix."""
    log, lines = _capture()
    fixtures = [
        _fix("a", "one two", None, 0.0, 0.5),
        _fix("b", "three four", None, 0.0, 0.5),
        _fix("c", "five six", None, 0.0, 0.10),  # tight band → fail
    ]
    replies = {
        "a.wav": "one two",
        "b.wav": "three four",
        "c.wav": "totally different",  # fails the tight band
    }
    summary = module.run_benchmark(
        _fixed_transcribe(replies),
        fixtures, Path("/tmp"), log=log, clock=_fake_clock(),
    )
    assert summary.total == 3
    assert summary.passing == 2
    assert summary.failing == 1


def test_all_fixtures_pass_summary_line():
    """Final summary line reports 'N/N fixtures passed'."""
    log, lines = _capture()
    fixtures = [
        _fix("a", "x", "x", 0.0, 0.0),
        _fix("b", "y", "y", 0.0, 0.0),
    ]
    replies = {"a.wav": "x", "b.wav": "y"}
    module.run_benchmark(
        _fixed_transcribe(replies),
        fixtures, Path("/tmp"), log=log, clock=_fake_clock(),
    )
    final = lines[-1]
    assert "2/2 fixtures passed" in final


def test_failing_fixtures_in_summary_line():
    """When some fail, summary shows partial."""
    log, lines = _capture()
    fixtures = [
        _fix("a", "x", "x", 0.0, 0.0),
        _fix("b", "y", "y", 0.0, 0.0),
        _fix("c", "z", "z", 0.5, 1.0),  # band makes z=0 a fail
    ]
    replies = {"a.wav": "x", "b.wav": "y", "c.wav": "z"}
    module.run_benchmark(
        _fixed_transcribe(replies),
        fixtures, Path("/tmp"), log=log, clock=_fake_clock(),
    )
    final = lines[-1]
    assert "2/3 fixtures passed" in final


# ---- Per-row format ---------------------------------------------


def test_per_row_includes_name_status_wer_and_band():
    log, lines = _capture()
    fixtures = [_fix("clean_audio", "hi there", "hi there", 0.0, 0.2)]
    module.run_benchmark(
        _fixed_transcribe("hi there"),
        fixtures, Path("/tmp"), log=log, clock=_fake_clock(),
    )
    # First line is the per-row report.
    row = lines[0]
    assert "clean_audio" in row
    assert "PASS" in row
    assert "WER 0.00" in row
    assert "[0.00, 0.20]" in row
    assert "elapsed" in row


def test_failing_row_emits_FAIL_status():
    log, lines = _capture()
    fixtures = [_fix("noisy", "ref text", None, 0.0, 0.05)]
    module.run_benchmark(
        _fixed_transcribe("totally wrong"),
        fixtures, Path("/tmp"), log=log, clock=_fake_clock(),
    )
    row = lines[0]
    assert "FAIL" in row


# ---- Empty corpus ---------------------------------------------


def test_empty_corpus_returns_empty_summary():
    log, lines = _capture()
    summary = module.run_benchmark(
        _fixed_transcribe("any"),
        [], Path("/tmp"), log=log, clock=_fake_clock(),
    )
    assert summary.total == 0
    assert summary.passing == 0
    assert summary.failing == 0
    # Even empty, the summary line should emit "0/0 fixtures passed".
    assert any("0/0 fixtures passed" in l for l in lines)


# ---- Timing ---------------------------------------------------


def test_per_fixture_elapsed_is_positive():
    """The fake_clock increments by 0.1 per call, so each
    fixture's elapsed is at least 0.1 (clock called twice per
    transcription: before and after)."""
    log, _ = _capture()
    fixtures = [_fix("a", "x", "x", 0.0, 0.0)]
    summary = module.run_benchmark(
        _fixed_transcribe("x"),
        fixtures, Path("/tmp"), log=log, clock=_fake_clock(),
    )
    assert summary.results[0].elapsed_seconds >= 0.1


def test_total_elapsed_sums_per_fixture():
    """Summary's total_elapsed is the sum of per-fixture
    elapsed, not the wall-clock of the whole run."""
    log, _ = _capture()
    fixtures = [
        _fix("a", "x", "x", 0.0, 0.0),
        _fix("b", "y", "y", 0.0, 0.0),
    ]
    replies = {"a.wav": "x", "b.wav": "y"}
    summary = module.run_benchmark(
        _fixed_transcribe(replies),
        fixtures, Path("/tmp"), log=log, clock=_fake_clock(),
    )
    expected = sum(r.elapsed_seconds for r in summary.results)
    assert summary.total_elapsed == expected


# ---- Hypothesis stored on FixtureResult --------------------


def test_hypothesis_stored_for_debugging():
    """When a fixture FAILS, the operator wants to see what the
    engine actually said. The FixtureResult carries the
    hypothesis."""
    log, _ = _capture()
    fixtures = [_fix("a", "expected text", None, 0.0, 0.10)]
    summary = module.run_benchmark(
        _fixed_transcribe("totally different output"),
        fixtures, Path("/tmp"), log=log, clock=_fake_clock(),
    )
    assert summary.results[0].hypothesis == "totally different output"


# ---- Path construction ----------------------------------------


def test_transcribe_called_with_full_audio_path():
    """The transcribe callable receives `fixture_dir/audio_path`,
    not just the basename. Real engines need the full path to
    open the file."""
    received: list[str] = []

    def _recording_transcribe(path: str) -> str:
        received.append(path)
        return "any"

    log, _ = _capture()
    fixtures = [_fix("a", "x", None, 0.0, 1.0)]
    module.run_benchmark(
        _recording_transcribe,
        fixtures, Path("/some/dir"), log=log, clock=_fake_clock(),
    )
    assert received == ["/some/dir/a.wav"]


# ---- Default log/clock ------------------------------------


def test_default_log_is_print(capsys):
    """When `log` kwarg omitted, output flows through `print`."""
    fixtures = [_fix("a", "x", "x", 0.0, 0.0)]
    module.run_benchmark(
        _fixed_transcribe("x"),
        fixtures, Path("/tmp"),
    )
    captured = capsys.readouterr()
    assert "a" in captured.out
    assert "1/1 fixtures passed" in captured.out


# ---- iter-133: verbose=False suppresses output -------------------


def test_verbose_false_suppresses_per_row_output():
    """When verbose=False, the log callable receives no per-row
    or summary calls. Used by JSON/CSV CLI paths to avoid
    interleaved text + format output."""
    log, lines = _capture()
    fixtures = [_fix("a", "x", "x", 0.0, 0.0)]
    module.run_benchmark(
        _fixed_transcribe("x"),
        fixtures, Path("/tmp"),
        log=log, clock=_fake_clock(), verbose=False,
    )
    assert lines == []


def test_verbose_false_still_returns_summary():
    """Suppressing output doesn't suppress the return value —
    the caller still gets a populated BenchmarkSummary."""
    log, _ = _capture()
    fixtures = [_fix("a", "x", "x", 0.0, 0.0)]
    summary = module.run_benchmark(
        _fixed_transcribe("x"),
        fixtures, Path("/tmp"),
        log=log, clock=_fake_clock(), verbose=False,
    )
    assert summary.total == 1
    assert summary.passing == 1


def test_verbose_default_is_true():
    """Backward compat: omitting verbose preserves iter-132
    behavior (per-row + summary text emitted)."""
    log, lines = _capture()
    fixtures = [_fix("a", "x", "x", 0.0, 0.0)]
    module.run_benchmark(
        _fixed_transcribe("x"),
        fixtures, Path("/tmp"),
        log=log, clock=_fake_clock(),
    )
    assert len(lines) >= 2  # at least one row + summary


# ---- iter-133: format_summary_json -------------------------------


def _build_summary(*entries):
    """entries: (name, passed, wer, exp_min, exp_max, elapsed,
    ref, hyp). Returns a BenchmarkSummary with matching
    FixtureResults."""
    summary = module.BenchmarkSummary()
    for name, passed, wer, lo, hi, elapsed, ref, hyp in entries:
        summary.results.append(module.FixtureResult(
            name=name, reference=ref, hypothesis=hyp,
            wer=wer, expected_min=lo, expected_max=hi,
            elapsed_seconds=elapsed, passed=passed,
        ))
    return summary


def test_json_format_includes_aggregate_fields():
    """The JSON dump includes passing/failing/total/total_elapsed
    at the top level — aggregates that operators want without
    iterating the results."""
    import json as _json
    summary = _build_summary(
        ("a", True,  0.0, 0.0, 0.5, 0.1, "x", "x"),
        ("b", False, 0.9, 0.0, 0.5, 0.2, "y", "z"),
    )
    out = module.format_summary_json(summary)
    parsed = _json.loads(out)
    assert parsed["passing"] == 1
    assert parsed["failing"] == 1
    assert parsed["total"] == 2
    # Float compare with tolerance for accumulated FP error.
    assert abs(parsed["total_elapsed_seconds"] - 0.3) < 1e-9


def test_json_format_includes_per_fixture_records():
    """Each result entry has every FixtureResult field."""
    import json as _json
    summary = _build_summary(
        ("clean", True, 0.20, 0.0, 0.4, 0.85, "the ref", "the hyp"),
    )
    parsed = _json.loads(module.format_summary_json(summary))
    record = parsed["results"][0]
    assert record["name"] == "clean"
    assert record["reference"] == "the ref"
    assert record["hypothesis"] == "the hyp"
    assert record["wer"] == 0.2
    assert record["expected_min"] == 0.0
    assert record["expected_max"] == 0.4
    assert record["elapsed_seconds"] == 0.85
    assert record["passed"] is True


def test_json_format_handles_empty_summary():
    """Zero fixtures → valid JSON with empty results array."""
    import json as _json
    summary = module.BenchmarkSummary()
    out = module.format_summary_json(summary)
    parsed = _json.loads(out)
    assert parsed["total"] == 0
    assert parsed["passing"] == 0
    assert parsed["failing"] == 0
    assert parsed["results"] == []


def test_json_format_indent_kwarg_controls_pretty_printing():
    """Default indent=2 produces multi-line output. indent=None
    gives compact one-line."""
    summary = _build_summary(
        ("a", True, 0.0, 0.0, 0.0, 0.1, "x", "x"),
    )
    pretty = module.format_summary_json(summary, indent=2)
    compact = module.format_summary_json(summary, indent=None)
    assert "\n" in pretty
    assert "\n" not in compact


def test_json_format_is_valid_json():
    """No matter what's in the data, output parses cleanly.
    Defends against unescaped quotes / control chars in
    transcripts."""
    import json as _json
    summary = _build_summary(
        ("weird", True, 0.0, 0.0, 0.5, 0.1,
         'he said "hello"', "tab\there"),
    )
    out = module.format_summary_json(summary)
    parsed = _json.loads(out)
    assert parsed["results"][0]["reference"] == 'he said "hello"'
    assert parsed["results"][0]["hypothesis"] == "tab\there"


# ---- iter-133: format_summary_csv ------------------------------


def test_csv_format_starts_with_header_row():
    """First row is the column names. Operators can read into
    pandas without skiprows."""
    summary = _build_summary(
        ("a", True, 0.0, 0.0, 0.5, 0.1, "x", "x"),
    )
    out = module.format_summary_csv(summary)
    first_line = out.splitlines()[0]
    for col in [
        "name", "passed", "wer",
        "expected_min", "expected_max",
        "elapsed_seconds", "reference", "hypothesis",
    ]:
        assert col in first_line


def test_csv_format_one_data_row_per_fixture():
    """N fixtures → N+1 lines (header + N data rows)."""
    summary = _build_summary(
        ("a", True, 0.0, 0.0, 0.5, 0.1, "x", "x"),
        ("b", False, 0.9, 0.0, 0.5, 0.2, "y", "z"),
        ("c", True, 0.1, 0.0, 0.5, 0.3, "p", "q"),
    )
    out = module.format_summary_csv(summary)
    lines = [ln for ln in out.splitlines() if ln]
    assert len(lines) == 4  # header + 3


def test_csv_format_quotes_strings_with_commas():
    """RFC-4180 compliance: a transcript with a comma must be
    quoted so spreadsheet imports parse correctly."""
    summary = _build_summary(
        ("a", True, 0.0, 0.0, 0.5, 0.1,
         "hello, world", "okay, then"),
    )
    out = module.format_summary_csv(summary)
    assert '"hello, world"' in out
    assert '"okay, then"' in out


def test_csv_format_handles_empty_summary():
    """Zero fixtures → just the header row."""
    summary = module.BenchmarkSummary()
    out = module.format_summary_csv(summary)
    lines = [ln for ln in out.splitlines() if ln]
    assert len(lines) == 1  # header only


def test_csv_format_round_trips_through_csv_module():
    """The output is parseable by csv.DictReader. Catches
    obvious CSV-formatting bugs (unquoted quotes, missing
    fields, etc.)."""
    import csv as _csv
    import io as _io
    summary = _build_summary(
        ("clean", True, 0.20, 0.0, 0.4, 0.85, "ref", "hyp"),
    )
    out = module.format_summary_csv(summary)
    reader = _csv.DictReader(_io.StringIO(out))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["name"] == "clean"
    assert rows[0]["passed"] == "True"
    assert rows[0]["reference"] == "ref"
    assert rows[0]["hypothesis"] == "hyp"


def test_csv_format_fixed_decimal_precision():
    """Numeric fields are formatted to 4 decimals — consistent
    across rows, no scientific notation surprise."""
    summary = _build_summary(
        ("a", True, 0.123456789, 0.0, 0.5, 0.987654321, "x", "x"),
    )
    out = module.format_summary_csv(summary)
    # WER 0.123456789 → "0.1235" (rounded to 4 decimals).
    assert "0.1235" in out
    assert "0.9877" in out


# ---- iter-134: compute_diff ------------------------------------


def _baseline_payload(*entries):
    """Build a baseline JSON payload (parsed-dict shape).
    entries: (name, passed, wer)."""
    results = []
    passing = 0
    for name, passed, wer in entries:
        results.append({
            "name": name, "passed": passed, "wer": wer,
            "reference": "", "hypothesis": "",
            "expected_min": 0.0, "expected_max": 1.0,
            "elapsed_seconds": 0.0,
        })
        if passed:
            passing += 1
    return {
        "passing": passing,
        "failing": len(entries) - passing,
        "total": len(entries),
        "total_elapsed_seconds": 0.0,
        "results": results,
    }


def test_diff_unchanged_session():
    """Same fixtures, same WER, same status → all unchanged."""
    current = _build_summary(
        ("a", True, 0.20, 0.0, 0.5, 0.1, "ref", "hyp"),
        ("b", True, 0.10, 0.0, 0.5, 0.1, "ref", "hyp"),
    )
    baseline = _baseline_payload(
        ("a", True, 0.20),
        ("b", True, 0.10),
    )
    diff = module.compute_diff(current, baseline)
    assert len(diff.fixture_diffs) == 2
    assert all(d.status_change == "unchanged" for d in diff.fixture_diffs)
    assert diff.regressions == []
    assert diff.improvements == []


def test_diff_regression_detected():
    """A fixture that was PASS in baseline and FAIL in current
    is flagged as 'regressed'."""
    current = _build_summary(
        ("noisy", False, 0.60, 0.0, 0.5, 0.1, "ref", "hyp"),
    )
    baseline = _baseline_payload(("noisy", True, 0.30))
    diff = module.compute_diff(current, baseline)
    assert len(diff.regressions) == 1
    assert diff.regressions[0].name == "noisy"
    assert diff.improvements == []


def test_diff_improvement_detected():
    """FAIL → PASS counts as 'improved'."""
    current = _build_summary(
        ("fixed", True, 0.10, 0.0, 0.5, 0.1, "ref", "hyp"),
    )
    baseline = _baseline_payload(("fixed", False, 0.80))
    diff = module.compute_diff(current, baseline)
    assert len(diff.improvements) == 1
    assert diff.improvements[0].name == "fixed"
    assert diff.regressions == []


def test_diff_new_fixture_in_current():
    """Fixture in current but not baseline → 'new'."""
    current = _build_summary(
        ("a", True, 0.10, 0.0, 0.5, 0.1, "ref", "hyp"),
        ("new_one", True, 0.20, 0.0, 0.5, 0.1, "ref", "hyp"),
    )
    baseline = _baseline_payload(("a", True, 0.10))
    diff = module.compute_diff(current, baseline)
    assert len(diff.new_fixtures) == 1
    assert diff.new_fixtures[0].name == "new_one"


def test_diff_removed_fixture():
    """Fixture in baseline but not current → 'removed'."""
    current = _build_summary(
        ("a", True, 0.10, 0.0, 0.5, 0.1, "ref", "hyp"),
    )
    baseline = _baseline_payload(
        ("a", True, 0.10),
        ("legacy", True, 0.30),
    )
    diff = module.compute_diff(current, baseline)
    assert len(diff.removed_fixtures) == 1
    assert diff.removed_fixtures[0].name == "legacy"
    assert diff.removed_fixtures[0].current_wer is None
    assert diff.removed_fixtures[0].baseline_wer == 0.30


def test_diff_wer_delta():
    """The wer_delta property is current minus baseline."""
    current = _build_summary(
        ("a", True, 0.30, 0.0, 0.5, 0.1, "ref", "hyp"),
    )
    baseline = _baseline_payload(("a", True, 0.20))
    diff = module.compute_diff(current, baseline)
    fd = diff.fixture_diffs[0]
    assert abs(fd.wer_delta - 0.10) < 1e-9


def test_diff_wer_delta_none_for_new_or_removed():
    """When one side is missing, wer_delta is None."""
    current = _build_summary(
        ("only_in_current", True, 0.10, 0.0, 0.5, 0.1, "r", "h"),
    )
    baseline = _baseline_payload(("only_in_baseline", True, 0.20))
    diff = module.compute_diff(current, baseline)
    for fd in diff.fixture_diffs:
        assert fd.wer_delta is None


def test_diff_aggregates_match_summary_counts():
    """current_passing/total + baseline_passing/total reflect
    the source data accurately."""
    current = _build_summary(
        ("a", True,  0.0, 0.0, 0.5, 0.1, "r", "h"),
        ("b", True,  0.0, 0.0, 0.5, 0.1, "r", "h"),
        ("c", False, 0.9, 0.0, 0.5, 0.1, "r", "h"),
    )
    baseline = _baseline_payload(
        ("a", True, 0.0),
        ("b", False, 0.9),
        ("c", False, 0.9),
    )
    diff = module.compute_diff(current, baseline)
    assert diff.current_passing == 2
    assert diff.current_total == 3
    assert diff.baseline_passing == 1
    assert diff.baseline_total == 3


# ---- iter-134: format_diff_text ------------------------------


def test_format_diff_text_includes_per_fixture_rows():
    current = _build_summary(
        ("a", True, 0.20, 0.0, 0.5, 0.1, "ref", "hyp"),
    )
    baseline = _baseline_payload(("a", True, 0.10))
    out = module.format_diff_text(module.compute_diff(current, baseline))
    assert "a" in out
    assert "0.10 -> 0.20" in out
    assert "PASS" in out


def test_format_diff_text_marks_regression():
    current = _build_summary(
        ("noisy", False, 0.60, 0.0, 0.5, 0.1, "ref", "hyp"),
    )
    baseline = _baseline_payload(("noisy", True, 0.30))
    out = module.format_diff_text(module.compute_diff(current, baseline))
    assert "regressed" in out


def test_format_diff_text_marks_improvement():
    current = _build_summary(
        ("fixed", True, 0.10, 0.0, 0.5, 0.1, "ref", "hyp"),
    )
    baseline = _baseline_payload(("fixed", False, 0.80))
    out = module.format_diff_text(module.compute_diff(current, baseline))
    assert "improved" in out


def test_format_diff_text_summary_line():
    current = _build_summary(
        ("a", True,  0.0, 0.0, 0.5, 0.1, "r", "h"),
        ("b", True,  0.0, 0.0, 0.5, 0.1, "r", "h"),
        ("c", False, 0.9, 0.0, 0.5, 0.1, "r", "h"),
    )
    baseline = _baseline_payload(
        ("a", True, 0.0),
        ("b", False, 0.9),
        ("c", False, 0.9),
    )
    out = module.format_diff_text(module.compute_diff(current, baseline))
    assert "1/3 → 2/3" in out
    assert "(+1)" in out


def test_format_diff_text_lists_regressions_explicitly():
    current = _build_summary(
        ("a", True,  0.0, 0.0, 0.5, 0.1, "r", "h"),
        ("b", False, 0.9, 0.0, 0.5, 0.1, "r", "h"),
        ("c", False, 0.9, 0.0, 0.5, 0.1, "r", "h"),
    )
    baseline = _baseline_payload(
        ("a", True, 0.0),
        ("b", True, 0.0),
        ("c", True, 0.0),
    )
    out = module.format_diff_text(module.compute_diff(current, baseline))
    assert "Regressions: b, c" in out


def test_format_diff_text_lists_new_and_removed():
    current = _build_summary(
        ("kept",     True, 0.0, 0.0, 0.5, 0.1, "r", "h"),
        ("new_one",  True, 0.0, 0.0, 0.5, 0.1, "r", "h"),
    )
    baseline = _baseline_payload(
        ("kept", True, 0.0),
        ("legacy", True, 0.0),
    )
    out = module.format_diff_text(module.compute_diff(current, baseline))
    assert "New fixtures: new_one" in out
    assert "Removed fixtures: legacy" in out


def test_format_diff_text_handles_unchanged_session():
    current = _build_summary(
        ("a", True, 0.10, 0.0, 0.5, 0.1, "r", "h"),
    )
    baseline = _baseline_payload(("a", True, 0.10))
    out = module.format_diff_text(module.compute_diff(current, baseline))
    assert "Improvements:" not in out
    assert "Regressions:" not in out
    assert "1/1 → 1/1" in out
    assert "(+0)" in out


def test_format_diff_text_negative_delta_renders_correctly():
    current = _build_summary(
        ("a", True, 0.10, 0.0, 0.5, 0.1, "r", "h"),
    )
    baseline = _baseline_payload(("a", True, 0.30))
    out = module.format_diff_text(module.compute_diff(current, baseline))
    assert "-0.200" in out
