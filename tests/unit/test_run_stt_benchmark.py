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
