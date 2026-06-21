"""iter-339 — End-to-end ``gv vad-gap-percentiles`` over the real corpus.

The percentile-side companion to :mod:`test_gv_vad_gaps_cli` (iter-329),
:mod:`test_gv_vad_gap_sweep_cli` (iter-331), :mod:`test_gv_vad_gap_grid_cli`
(iter-333), :mod:`test_gv_vad_gap_diff_cli` (iter-335) and
:mod:`test_gv_vad_gap_histogram_cli` (iter-337). Those modules pin the point /
sweep / grid / diff / histogram gap surfaces against the REAL Silero segmenter;
this one does the same for ``gv vad-gap-percentiles`` (iter-338), the robust
order-statistic surface. Where ``gv vad-gaps`` answers "what are the
min/mean/max pauses at THIS setting?" — each fragile to a single outlier pause —
``gv vad-gap-percentiles`` answers "where do the pauses actually SIT?" via
p50/p90/p99, which a lone long between-paragraph silence cannot drag around. The
median is the typical pause; set the end-of-turn hangover (``--min-silence-ms`` /
the live ``chat.vad.silence_duration``) comfortably below p50 to never merge a
typical turn, and read p90 / p99 to size the long tail.

The anchoring property mirrors the family's: the percentiles surface's gap count
and aggregates must equal an independent ``gv vad-gaps --json`` run segmented at
the same gate (so the totals always agree with ``gv vad-gaps``). On top of that,
this module pins the percentile-specific invariants against a real
segmentation: the percentiles are monotonic non-decreasing in ``p``; every
percentile value lies within ``[min_gap, max_gap]``; ``p100`` equals the max gap
exactly; and the median is ROBUST — re-running at a setting that adds/removes one
extreme gap moves the max more than it moves p50. The 31s continuous recording
(which energy-VAD cannot split) is THE GATE here too: at the baseline 0.5 gate it
must yield ≥2 segments, hence ≥1 measurable gap to summarise.

Skips cleanly when the recordings (large binary captures, not committed) or
``silero-vad`` (+ torch deps) are absent — same contract as the sibling modules.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples import gv  # noqa: E402
from vad.silero import silero_available  # noqa: E402

RECORDINGS_DIR = ROOT / "fixtures" / "recordings"
CONTINUOUS_31S = "voice-20260618-110355.wav"


def _recordings() -> list[Path]:
    if not RECORDINGS_DIR.is_dir():
        return []
    return sorted(RECORDINGS_DIR.glob("*.wav"))


RECORDINGS = _recordings()

pytestmark = [
    pytest.mark.skipif(
        not RECORDINGS,
        reason=f"no ground-truth recordings in {RECORDINGS_DIR} (rsync'd onto the loop host)",
    ),
    pytest.mark.skipif(
        not silero_available(),
        reason="silero-vad not installed (pulls torch + torchaudio)",
    ),
]


def _gaps_args(wav: Path, **over):
    """Namespace for a single-setting ``gv vad-gaps`` run (the anchor)."""
    base = dict(
        wav=str(wav),
        threshold=0.5,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        json=False,
        csv=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _run_gaps(wav: Path, **over) -> list[str]:
    lines: list[str] = []
    gv.cmd_vad_gaps(_gaps_args(wav, **over), log=lines.append)
    return lines


def _pct_args(wav: Path, **over):
    """Namespace for a ``gv vad-gap-percentiles`` run mirroring the parser defaults."""
    base = dict(
        wav=str(wav),
        percentiles=[50.0, 90.0, 99.0],
        threshold=0.5,
        min_speech_ms=250.0,
        min_silence_ms=800.0,
        speech_pad_ms=30.0,
        max_speech_s=float("inf"),
        json=False,
        csv=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _run_pct(wav: Path, **over) -> list[str]:
    lines: list[str] = []
    gv.cmd_vad_gap_percentiles(_pct_args(wav, **over), log=lines.append)
    return lines


def test_gv_vad_gap_pct_31s_has_a_measurable_gap_at_baseline():
    """THE GATE: taking percentiles of the 31s continuous recording at the
    baseline 0.5 gate must yield ≥2 segments and ≥1 inter-segment gap (energy-VAD
    cannot split this clip; Silero must), so there is a distribution to summarise
    — a non-empty ``percentiles`` list."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    payload = json.loads(_run_pct(wav, threshold=0.5, json=True)[0])
    assert payload["available"] is True
    assert payload["name"] == CONTINUOUS_31S
    assert payload["num_segments"] >= 2
    assert payload["num_gaps"] == payload["num_segments"] - 1
    assert payload["num_gaps"] >= 1
    assert payload["percentiles"], "expected a non-empty percentile list for a recording with gaps"
    assert [e["p"] for e in payload["percentiles"]] == [50.0, 90.0, 99.0]


def test_gv_vad_gap_pct_matches_independent_vad_gaps():
    """The anchoring property: the percentiles surface's gap count and aggregates
    must equal an independent ``gv vad-gaps --json`` run segmented at the same
    gate — proving the percentiles summarise the SAME segmentation a standalone
    ``gv vad-gaps`` produces, not a re-derived one."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    pct = json.loads(_run_pct(wav, threshold=0.3, json=True)[0])
    anchor = json.loads(_run_gaps(wav, threshold=0.3, json=True)[0])

    assert pct["num_segments"] == anchor["num_segments"]
    assert pct["num_gaps"] == anchor["num_gaps"]
    assert pct["min_gap_s"] == anchor["min_gap_s"]
    assert pct["mean_gap_s"] == anchor["mean_gap_s"]
    assert pct["max_gap_s"] == anchor["max_gap_s"]
    assert pct["total_silence_s"] == anchor["total_silence_s"]


def test_gv_vad_gap_pct_monotonic_and_bounded():
    """On a real segmentation the percentile values must be monotonic
    non-decreasing in ``p`` (a higher percentile is never a smaller pause) and
    every value must lie within ``[min_gap_s, max_gap_s]`` — the percentiles are
    order statistics of the very gaps the aggregates summarise."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    payload = json.loads(
        _run_pct(wav, threshold=0.3, percentiles=[10.0, 25.0, 50.0, 75.0, 90.0, 99.0], json=True)[0]
    )
    if payload["num_gaps"] == 0:
        pytest.skip("no gaps to summarise")
    values = [e["value_s"] for e in payload["percentiles"]]
    assert values == sorted(values), f"percentiles not monotonic non-decreasing: {values}"
    for v in values:
        assert payload["min_gap_s"] <= v <= payload["max_gap_s"] + 1e-9


def test_gv_vad_gap_pct_p100_equals_max_gap():
    """``p100`` (the largest rank) must equal the aggregate ``max_gap_s`` exactly
    on a real segmentation — the top of the empirical CDF is the longest pause."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    payload = json.loads(_run_pct(wav, threshold=0.3, percentiles=[100.0], json=True)[0])
    if payload["num_gaps"] == 0:
        pytest.skip("no gaps to summarise")
    assert payload["percentiles"][0]["p"] == 100.0
    assert payload["percentiles"][0]["value_s"] == payload["max_gap_s"]


def test_gv_vad_gap_pct_median_is_robust_to_an_outlier():
    """The headline property: the median is ROBUST where the max is not. Two real
    segmentations of the same clip at adjacent gates yield two pause
    distributions; the one with the higher (more split) gate adds short gaps and
    can shift the extremes, but the typical pause (p50) should move LESS than the
    max does — the whole reason percentiles exist alongside min/mean/max. We only
    assert the robustness inequality when the two runs genuinely differ in their
    extreme; otherwise the property is vacuous and we skip."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    a = json.loads(_run_pct(wav, threshold=0.3, percentiles=[50.0], json=True)[0])
    b = json.loads(_run_pct(wav, threshold=0.5, percentiles=[50.0], json=True)[0])
    if a["num_gaps"] < 2 or b["num_gaps"] < 2:
        pytest.skip("need ≥2 gaps in both runs to compare distributions")
    max_shift = abs(a["max_gap_s"] - b["max_gap_s"])
    median_shift = abs(a["percentiles"][0]["value_s"] - b["percentiles"][0]["value_s"])
    if max_shift == 0:
        pytest.skip("the two gates produced the same max gap — robustness is vacuous here")
    assert median_shift <= max_shift, (
        f"median moved more than the max between gates "
        f"(median {median_shift:.3f}s vs max {max_shift:.3f}s) — percentiles not robust"
    )


def test_gv_vad_gap_pct_csv_matches_json():
    """``gv vad-gap-percentiles --csv`` describes the same percentiles as
    ``--json`` over THE GATE recording — same row count, same ``value_s`` per
    percentile — proving the two machine-readable surfaces agree."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")

    csv_lines = _run_pct(wav, threshold=0.3, csv=True)
    csv_rows = list(csv.DictReader(io.StringIO(csv_lines[0])))

    payload = json.loads(_run_pct(wav, threshold=0.3, json=True)[0])
    assert len(csv_rows) == len(payload["percentiles"])
    for csv_row, entry in zip(csv_rows, payload["percentiles"]):
        # The CSV label spelling is `50` not `50.0`; compare numerically.
        assert float(csv_row["percentile"]) == entry["p"]
        assert abs(float(csv_row["value_s"]) - entry["value_s"]) <= 1e-9


def test_gv_vad_gap_pct_human_report_shows_aggregates_and_percentiles():
    """The human report over THE GATE recording carries the title, the aggregate
    header, the actionable ``--min-silence-ms`` knob hint on the median line, and
    one ``pNN`` line per requested percentile — the distribution is summarised,
    not just aggregated."""
    wav = RECORDINGS_DIR / CONTINUOUS_31S
    if not wav.exists():
        pytest.skip(f"{CONTINUOUS_31S} not present")
    lines = _run_pct(wav, threshold=0.3)
    text = "\n".join(lines)
    assert "gap percentiles" in text
    assert CONTINUOUS_31S in text
    assert "--min-silence-ms" in text
    assert any(ln.strip().startswith("p50") for ln in lines)
    assert any(ln.strip().startswith("p90") for ln in lines)
    assert any(ln.strip().startswith("p99") for ln in lines)


@pytest.mark.parametrize("wav", RECORDINGS, ids=[p.name for p in RECORDINGS])
def test_gv_vad_gap_pct_emits_a_report_for_every_recording(wav: Path):
    """Every corpus recording produces a well-formed human report (title + the
    min-gap knob hint) and a parseable JSON payload whose gap count is one fewer
    than its segment count (or zero for a single region), with a percentile per
    request that is monotonic and bounded — and an empty ``percentiles`` list for
    a <2-segment clip."""
    lines = _run_pct(wav)
    text = "\n".join(lines)
    assert "gap percentiles" in text
    assert wav.name in text

    payload = json.loads(_run_pct(wav, json=True)[0])
    assert payload["available"] is True
    assert payload["name"] == wav.name
    if payload["num_segments"] >= 2:
        assert payload["num_gaps"] == payload["num_segments"] - 1
        assert payload["min_gap_s"] is not None
        assert [e["p"] for e in payload["percentiles"]] == [50.0, 90.0, 99.0]
        values = [e["value_s"] for e in payload["percentiles"]]
        assert values == sorted(values)
        for v in values:
            assert payload["min_gap_s"] <= v <= payload["max_gap_s"] + 1e-9
    else:
        assert payload["num_gaps"] == 0
        assert payload["min_gap_s"] is None
        assert payload["mean_gap_s"] is None
        assert payload["max_gap_s"] is None
        assert payload["percentiles"] == []
