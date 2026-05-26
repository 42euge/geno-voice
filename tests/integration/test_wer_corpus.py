"""iter-106 — WER fixture corpus integration test.

Exercises the full WER pipeline end-to-end:

    fixture (reference + simulated STT hypothesis)
      → compute_wer → TurnMetrics
      → print_session_summary
      → assert WER line emits with the expected aggregate

The corpus lives in `tests/fixtures/wer/corpus.json` — text-only
because we don't have a CPU-only STT runtime on x86_64 Linux
(mlx-whisper is Mac-only). When real audio fixtures land, the
integration runner will swap simulated hypotheses for STT output
on real .wav files and the same expected ranges should still hold.
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_metrics import (  # noqa: E402
    SessionMeta,
    TurnMetrics,
    print_session_summary,
)
from examples._chat_wer import compute_wer  # noqa: E402

CORPUS_PATH = ROOT / "tests" / "fixtures" / "wer" / "corpus.json"


@pytest.fixture(scope="module")
def corpus():
    with CORPUS_PATH.open() as f:
        data = json.load(f)
    return data["fixtures"]


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ---- Per-fixture WER bounds -----------------------------------------------


def test_corpus_loads_and_has_expected_size(corpus):
    """Sanity check: the corpus is the size iter-106 designed.
    If fixtures are added later, bump this assertion intentionally."""
    assert len(corpus) == 5
    names = {f["name"] for f in corpus}
    assert names == {
        "clean", "substitution", "dropped_filler",
        "noisy", "catastrophic",
    }


@pytest.mark.parametrize("idx", range(5))
def test_each_fixture_lands_in_expected_wer_band(idx, corpus):
    """For each fixture, compute WER and assert it falls inside
    the recorded [expected_wer_min, expected_wer_max] band. This
    is the regression sentinel for both the corpus and the
    compute_wer primitive — if either drifts, the test surfaces
    which fixture broke."""
    f = corpus[idx]
    wer = compute_wer(f["reference"], f["hypothesis"])
    assert f["expected_wer_min"] <= wer <= f["expected_wer_max"], (
        f"{f['name']}: WER {wer:.3f} outside "
        f"[{f['expected_wer_min']}, {f['expected_wer_max']}]"
    )


# ---- End-to-end through TurnMetrics + session summary --------------------


def _build_turn_with_wer(reference: str, hypothesis: str) -> TurnMetrics:
    """Helper: synthesize a TurnMetrics with WER populated.

    Real production code path (when audio fixtures land) is:
        text = stt_engine.transcribe(audio_bytes)
        wer = compute_wer(reference, text)
        metrics.wer = wer
        metrics.wer_measured = True

    We simulate the same flow without the audio + STT step so
    the WER plumbing is testable on x86_64 today.
    """
    m = TurnMetrics(transcript=hypothesis)
    m.wer = compute_wer(reference, hypothesis)
    m.wer_measured = True
    # Fill in a couple of fields the summary needs for n>0 math
    # so the WER line isn't suppressed by a "no successful
    # turns" branch elsewhere.
    m.ttfs = 0.5
    m.sentences_spoken = 1
    return m


def test_session_summary_emits_wer_line_when_corpus_drives_it(corpus):
    """Drive print_session_summary with all 5 corpus fixtures
    converted to TurnMetrics. The WER line MUST emit (every
    fixture has wer_measured=True), and median + max must match
    the corpus distribution."""
    metrics_list = [
        _build_turn_with_wer(f["reference"], f["hypothesis"])
        for f in corpus
    ]

    buf = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buf
    try:
        print_session_summary(
            metrics_list,
            llm_config={"model": "stub"},
            meta=SessionMeta(),
        )
    finally:
        sys.stdout = sys_stdout

    out = _strip_ansi(buf.getvalue())
    wer_lines = [ln for ln in out.splitlines() if "WER:" in ln]
    assert len(wer_lines) == 1, f"expected 1 WER line, got {wer_lines!r}"
    line = wer_lines[0]

    # Compute expected aggregate from the same fixtures so the
    # test stays in sync with the corpus.
    import statistics
    wers = [compute_wer(f["reference"], f["hypothesis"]) for f in corpus]
    expected_median = statistics.median(wers)
    expected_max = max(wers)

    assert f"{expected_median:.2f} median" in line, line
    assert f"{expected_max:.2f} max" in line, line
    assert f"({len(wers)} turns measured)" in line, line


def test_session_summary_omits_wer_line_when_no_turn_measured():
    """A session with no measured WER (all turns wer_measured=
    False, the default) MUST NOT emit a WER line. This is the
    regression sentinel for the wer_measured filter — without it,
    every session would emit '0.00 median' which is wrong/
    misleading."""
    metrics_list = [
        TurnMetrics(transcript="hi", ttfs=0.5, sentences_spoken=1),
        TurnMetrics(transcript="ok", ttfs=0.5, sentences_spoken=1),
    ]

    buf = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buf
    try:
        print_session_summary(
            metrics_list,
            llm_config={"model": "stub"},
            meta=SessionMeta(),
        )
    finally:
        sys.stdout = sys_stdout

    out = _strip_ansi(buf.getvalue())
    wer_lines = [ln for ln in out.splitlines() if "WER:" in ln]
    assert wer_lines == [], (
        f"WER line should not emit when no turn measured WER, got {wer_lines!r}"
    )


def test_partial_measurement_session(corpus):
    """Mixed session: some turns measured, others not. Only the
    measured turns should contribute to the WER aggregate."""
    measured = [
        _build_turn_with_wer(f["reference"], f["hypothesis"])
        for f in corpus[:2]   # clean + substitution only
    ]
    unmeasured = [
        TurnMetrics(transcript="hi", ttfs=0.5, sentences_spoken=1),
        TurnMetrics(transcript="ok", ttfs=0.5, sentences_spoken=1),
    ]
    metrics_list = measured + unmeasured

    buf = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buf
    try:
        print_session_summary(
            metrics_list,
            llm_config={"model": "stub"},
            meta=SessionMeta(),
        )
    finally:
        sys.stdout = sys_stdout

    out = _strip_ansi(buf.getvalue())
    wer_lines = [ln for ln in out.splitlines() if "WER:" in ln]
    assert len(wer_lines) == 1
    # Only the 2 measured turns count, not 4.
    assert "(2 turns measured)" in wer_lines[0]
