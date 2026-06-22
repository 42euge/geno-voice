"""Tests for iter-409 — batch STT-RTF profiling over a CORPUS of engines.

Where iter-405's ``profile_stt_rtf`` folds ONE engine's transcription timings
into a single median RTF, ``profile_stt_rtf_batch`` generalises to N engines —
the STT-side twin of iter-397's ``calibrate_base_wpm_batch``. Each engine is
profiled independently (and folded through the iter-407 ``stt_rtf_verdict``
against shared gates so the per-engine recommendations are apples-to-apples),
and the per-engine ``median_rtf`` values are summarised by an outlier-robust
corpus median, so an operator choosing a transcriber for the host sees which
engines keep up with realtime and which are the bottleneck.

This module pins the pure engine (``SttRtfBatch`` / ``profile_stt_rtf_batch`` /
``STT_RTF_BATCH_GRADE_ORDER``). The gv-side human render is pinned in
``test_gv_stt_rtf_batch.py``.

Pure arithmetic over injected timings — no torch, no faster-whisper, no audio
I/O — so it runs in the unit gate regardless of platform, exactly like the
single-engine core.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stt.rtf_profile import (  # noqa: E402
    DEFAULT_STT_RTF_MIN_SAMPLES,
    DEFAULT_STT_RTF_REL_SPREAD_MAX,
    STT_RTF_BATCH_GRADE_ORDER,
    SttRtfBatch,
    TranscriptionSample,
    profile_stt_rtf,
    profile_stt_rtf_batch,
)


def _samples(*pairs):
    """Build TranscriptionSamples from (audio_seconds, transcribe_seconds) tuples."""
    return [
        TranscriptionSample(audio_seconds=a, transcribe_seconds=t) for (a, t) in pairs
    ]


# --------------------------------------------------------------------------
# shape / counting
# --------------------------------------------------------------------------


def test_empty_corpus_has_no_aggregates():
    batch = profile_stt_rtf_batch([])
    assert isinstance(batch, SttRtfBatch)
    assert batch.num_engines == 0
    assert batch.num_profiled == 0
    assert batch.corpus_median_rtf is None
    assert batch.corpus_min_rtf is None
    assert batch.corpus_max_rtf is None
    assert batch.corpus_spread is None
    assert batch.num_keep_up == 0
    assert batch.num_recommend == 0
    assert batch.rows == ()


def test_one_row_per_engine_in_input_order():
    batch = profile_stt_rtf_batch(
        [
            ("a", _samples((10.0, 1.0))),
            ("b", _samples((10.0, 2.0))),
            ("c", _samples((10.0, 3.0))),
        ]
    )
    assert batch.num_engines == 3
    assert [r["engine"] for r in batch.rows] == ["a", "b", "c"]


def test_each_row_carries_its_own_profile():
    samples = _samples((10.0, 1.2), (5.0, 0.8), (10.0, 1.0))
    batch = profile_stt_rtf_batch([("mlx", samples)])
    row = batch.rows[0]
    # The embedded profile must agree EXACTLY with profile_stt_rtf on the samples.
    assert row["profile"] == profile_stt_rtf(samples)


# --------------------------------------------------------------------------
# corpus aggregates
# --------------------------------------------------------------------------


def test_corpus_median_is_outlier_robust():
    # Two tight engines plus one pathological slow one: the median sits at the
    # middle engine, NOT dragged toward the outlier the way a mean would be.
    batch = profile_stt_rtf_batch(
        [
            ("a", _samples((10.0, 1.0))),  # rtf 0.1
            ("b", _samples((10.0, 2.0))),  # rtf 0.2
            ("c", _samples((10.0, 50.0))),  # rtf 5.0 — pathological
        ]
    )
    assert batch.corpus_median_rtf == 0.2
    assert batch.corpus_min_rtf == 0.1
    assert batch.corpus_max_rtf == 5.0
    assert batch.corpus_spread == 4.9


def test_single_engine_zero_spread():
    batch = profile_stt_rtf_batch([("only", _samples((10.0, 1.0)))])
    assert batch.num_profiled == 1
    assert batch.corpus_median_rtf == 0.1
    assert batch.corpus_spread == 0.0


def test_delta_from_median_signed():
    batch = profile_stt_rtf_batch(
        [
            ("fast", _samples((10.0, 1.0))),  # rtf 0.1
            ("mid", _samples((10.0, 3.0))),  # rtf 0.3 (median)
            ("slow", _samples((10.0, 5.0))),  # rtf 0.5
        ]
    )
    by = {r["engine"]: r for r in batch.rows}
    assert by["mid"]["delta_from_median_rtf"] == 0.0
    assert round(by["fast"]["delta_from_median_rtf"], 10) == -0.2
    assert round(by["slow"]["delta_from_median_rtf"], 10) == 0.2


# --------------------------------------------------------------------------
# unprofiled (no-sample) engines
# --------------------------------------------------------------------------


def test_unprofiled_engine_excluded_from_aggregates_but_listed():
    batch = profile_stt_rtf_batch(
        [
            ("a", _samples((10.0, 1.0))),
            ("empty", []),
        ]
    )
    assert batch.num_engines == 2
    assert batch.num_profiled == 1
    # The empty engine is listed with a None profile/verdict and no delta.
    empty = [r for r in batch.rows if r["engine"] == "empty"][0]
    assert empty["profile"] is None
    assert empty["verdict"] is None
    assert empty["delta_from_median_rtf"] is None
    # Excluded from the corpus aggregates (which describe only the profiled one).
    assert batch.corpus_median_rtf == 0.1


def test_all_unprofiled_corpus_is_empty():
    batch = profile_stt_rtf_batch([("a", []), ("b", [])])
    assert batch.num_engines == 2
    assert batch.num_profiled == 0
    assert batch.corpus_median_rtf is None
    assert batch.grade_counts["unprofiled"] == 2
    assert batch.num_keep_up == 0
    assert batch.num_recommend == 0


# --------------------------------------------------------------------------
# grade histogram
# --------------------------------------------------------------------------


def test_grade_order_canonical():
    assert STT_RTF_BATCH_GRADE_ORDER == ("fast", "realtime", "slow", "unprofiled")


def test_histogram_has_all_buckets_summing_to_num_engines():
    batch = profile_stt_rtf_batch(
        [
            ("fast", _samples((10.0, 1.0))),  # rtf 0.1 -> fast
            ("rt", _samples((10.0, 8.0))),  # rtf 0.8 -> realtime
            ("slow", _samples((10.0, 15.0))),  # rtf 1.5 -> slow
            ("empty", []),  # -> unprofiled
        ]
    )
    counts = batch.grade_counts
    assert set(counts) == set(STT_RTF_BATCH_GRADE_ORDER)
    assert counts == {"fast": 1, "realtime": 1, "slow": 1, "unprofiled": 1}
    assert sum(counts.values()) == batch.num_engines


def test_num_keep_up_counts_fast_and_realtime():
    batch = profile_stt_rtf_batch(
        [
            ("fast", _samples((10.0, 1.0))),  # fast
            ("rt", _samples((10.0, 8.0))),  # realtime
            ("slow", _samples((10.0, 15.0))),  # slow
        ]
    )
    assert batch.num_keep_up == 2  # fast + realtime, not slow


# --------------------------------------------------------------------------
# per-engine verdict + recommend count
# --------------------------------------------------------------------------


def test_row_verdict_recommends_slow_trustworthy_engine():
    # 3 tight, genuinely-slow samples => recommend lighten.
    batch = profile_stt_rtf_batch(
        [("heavy", _samples((10.0, 15.0), (10.0, 15.2), (10.0, 14.8)))]
    )
    row = batch.rows[0]
    assert row["verdict"] is not None
    assert row["verdict"].recommend is True
    assert batch.num_recommend == 1


def test_slow_but_too_few_samples_not_recommended():
    # A single slow timing grades "slow" but fails the sample gate, so the
    # verdict does NOT recommend — num_recommend stays below the slow count.
    batch = profile_stt_rtf_batch([("heavy", _samples((10.0, 15.0)))])
    assert batch.grade_counts["slow"] == 1
    assert batch.rows[0]["verdict"].recommend is False
    assert batch.num_recommend == 0


def test_num_recommend_le_slow_count():
    batch = profile_stt_rtf_batch(
        [
            ("trust_slow", _samples((10.0, 15.0), (10.0, 15.2), (10.0, 14.8))),
            ("noisy_slow", _samples((10.0, 15.0), (10.0, 40.0), (10.0, 12.0))),
            ("fast", _samples((10.0, 1.0), (10.0, 1.1), (10.0, 0.9))),
        ]
    )
    assert batch.num_recommend <= batch.grade_counts["slow"]
    assert batch.num_recommend == 1  # only the trustworthy slow engine


# --------------------------------------------------------------------------
# gate threading
# --------------------------------------------------------------------------


def test_gate_defaults_echoed():
    batch = profile_stt_rtf_batch([("a", _samples((10.0, 1.0)))])
    assert batch.rel_spread_max == DEFAULT_STT_RTF_REL_SPREAD_MAX
    assert batch.min_samples == DEFAULT_STT_RTF_MIN_SAMPLES


def test_gates_threaded_to_per_engine_verdict():
    samples = _samples((10.0, 15.0), (10.0, 15.2), (10.0, 14.8))
    # A tighter min_samples floor of 5 flips an otherwise-recommended slow engine
    # to keep — the gate must reach the per-engine verdict.
    batch = profile_stt_rtf_batch([("heavy", samples)], min_samples=5)
    assert batch.min_samples == 5
    assert batch.rows[0]["verdict"].recommend is False
    assert batch.num_recommend == 0


def test_rel_spread_gate_threaded():
    # Genuinely slow but disagreeing runs: a strict rel_spread_max keeps it.
    samples = _samples((10.0, 15.0), (10.0, 40.0), (10.0, 12.0))
    batch = profile_stt_rtf_batch([("noisy", samples)], rel_spread_max=0.01)
    assert batch.rel_spread_max == 0.01
    assert batch.rows[0]["verdict"].recommend is False
