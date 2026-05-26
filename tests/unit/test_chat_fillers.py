"""Tests for iter-107 — prerender_fillers helper.

The function takes a synth_fn (any callable returning audio +
tokens) and an idle_threshold for logging. It must:
  - Render each text and accumulate non-empty results
  - Skip empty audio silently
  - Catch per-text exceptions and log without aborting
  - Log a final summary line ONLY when texts is non-empty
  - Return the rendered list intact
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_fillers import prerender_fillers  # noqa: E402


# ---- Fakes ----------------------------------------------------------------


class _FakeAudio:
    """Stands in for a numpy array — only `len()` is exercised."""

    def __init__(self, length: int):
        self._length = length

    def __len__(self) -> int:
        return self._length


def _ok_synth(text: str):
    """Always returns 2048-sample audio + empty tokens."""
    return _FakeAudio(2048), []


def _empty_synth(text: str):
    """Returns 0-length audio — should be silently skipped."""
    return _FakeAudio(0), []


def _make_failing_synth(failures: set[str]):
    """Synth that raises ValueError for any text in `failures`,
    returns ok audio for everything else."""
    def synth(text: str):
        if text in failures:
            raise ValueError(f"synth failed for {text}")
        return _FakeAudio(2048), []
    return synth


def _capture():
    lines: list[str] = []

    def log(line: str) -> None:
        lines.append(line)

    return log, lines


# ---- No-op edge cases -----------------------------------------------------


def test_empty_texts_returns_empty_list_no_log():
    """No filler texts → no work, no log line."""
    log, lines = _capture()
    result = prerender_fillers(_ok_synth, [], log=log)
    assert result == []
    assert lines == []


def test_empty_iterable_works_too():
    """Iterable (not list) input still works — the function
    converts internally."""
    log, lines = _capture()
    result = prerender_fillers(_ok_synth, iter([]), log=log)
    assert result == []
    assert lines == []


# ---- Happy path -----------------------------------------------------------


def test_all_synth_succeed_returns_all_clips():
    """3 texts, 3 successful synths → 3 clips returned."""
    log, lines = _capture()
    result = prerender_fillers(
        _ok_synth, ["um", "let me think", "well,"], log=log,
    )
    assert len(result) == 3
    # Each entry is (audio, tokens) — tokens is empty list in our stub.
    for audio, tokens in result:
        assert len(audio) == 2048
        assert tokens == []


def test_summary_log_emits_once_with_count_and_threshold():
    """Single summary line at the end, formatted with both the
    success ratio and the idle threshold."""
    log, lines = _capture()
    prerender_fillers(
        _ok_synth, ["a", "b"], idle_threshold=0.6, log=log,
    )
    # 2 texts, 0 failures → exactly 1 line (the summary).
    assert len(lines) == 1
    assert "Pre-rendered 2/2 fillers" in lines[0]
    assert "(idle threshold 0.60s)" in lines[0]


def test_idle_threshold_zero_still_renders_in_summary():
    """A 0.0 threshold is still printed (operator may have
    explicitly set it)."""
    log, lines = _capture()
    prerender_fillers(_ok_synth, ["a"], idle_threshold=0.0, log=log)
    assert "0.00s" in lines[0]


# ---- Failure handling -----------------------------------------------------


def test_one_synth_fails_others_proceed():
    """Per-text try/except — one failure doesn't kill the rest."""
    log, lines = _capture()
    synth = _make_failing_synth(failures={"bad"})
    result = prerender_fillers(
        synth, ["good1", "bad", "good2"], log=log,
    )
    # 2 successes.
    assert len(result) == 2
    # 1 failure log + 1 summary log.
    assert len(lines) == 2
    failure_lines = [ln for ln in lines if ln.startswith("filler synth failed")]
    assert len(failure_lines) == 1
    assert "'bad'" in failure_lines[0]
    # Summary reports 2/3.
    summary = [ln for ln in lines if "Pre-rendered" in ln]
    assert len(summary) == 1
    assert "Pre-rendered 2/3 fillers" in summary[0]


def test_all_synths_fail_returns_empty_list_with_summary():
    """Every text fails → empty result list, but summary still
    fires with 0/N."""
    log, lines = _capture()
    synth = _make_failing_synth(failures={"a", "b"})
    result = prerender_fillers(synth, ["a", "b"], log=log)
    assert result == []
    summary = [ln for ln in lines if "Pre-rendered" in ln]
    assert len(summary) == 1
    assert "Pre-rendered 0/2 fillers" in summary[0]


def test_failure_log_line_exposes_exception_text():
    """The error message from the exception is in the log line —
    operator can see WHY the synth failed."""
    log, lines = _capture()
    synth = _make_failing_synth(failures={"x"})
    prerender_fillers(synth, ["x"], log=log)
    failure = next(ln for ln in lines if ln.startswith("filler synth failed"))
    assert "synth failed for x" in failure


# ---- Empty-audio handling -------------------------------------------------


def test_empty_audio_silently_skipped():
    """A synth that returns 0-length audio is dropped without a
    log line — matches the original inline behavior, which only
    logged on exception, not on empty audio."""
    log, lines = _capture()
    result = prerender_fillers(_empty_synth, ["a"], log=log)
    assert result == []
    # Only the summary line — no per-text failure line.
    assert len(lines) == 1
    assert "Pre-rendered 0/1 fillers" in lines[0]


def test_mixed_empty_and_ok():
    """Some texts → ok audio, others → empty. Only ok ones land."""
    log, lines = _capture()

    def alternating(text: str):
        if text == "skip":
            return _FakeAudio(0), []
        return _FakeAudio(2048), []

    result = prerender_fillers(
        alternating, ["a", "skip", "b", "skip", "c"], log=log,
    )
    assert len(result) == 3   # only "a", "b", "c" landed
    summary = [ln for ln in lines if "Pre-rendered" in ln]
    assert "Pre-rendered 3/5 fillers" in summary[0]


# ---- Default log callable -------------------------------------------------


def test_default_log_is_print(capsys):
    """When no log kwarg passed, output goes to stdout via print."""
    prerender_fillers(_ok_synth, ["a"], idle_threshold=0.5)
    captured = capsys.readouterr()
    assert "Pre-rendered 1/1 fillers" in captured.out
    assert "0.50s" in captured.out


# ---- Caller-shape parity --------------------------------------------------


def test_synth_fn_receives_just_the_text():
    """The synth callable is invoked with a single positional
    arg — the text. Real callers wrap engine/voice/speed in a
    closure; tests don't need to know about those."""
    seen: list[str] = []

    def recording_synth(text: str):
        seen.append(text)
        return _FakeAudio(2048), []

    log, _ = _capture()
    prerender_fillers(
        recording_synth, ["hmm", "well,"], log=log,
    )
    assert seen == ["hmm", "well,"]
