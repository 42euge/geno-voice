"""
STT real-time-factor (RTF) profiling — the STT-side analogue of the TTS
``base_wpm`` calibration family (iter-220..404).

Every prior gv-family analysis lap lived on the TTS side: the
``calibrate_base_wpm`` core (iter-220) folds rendered-audio timings into a robust
median speaking rate with spread/grade/margin diagnostics, and the gv CLI grew a
rich surface around it (single + batch, three formats, sort/top-n/grade-floor/
summary, IQR, flyer flag). The STT side — how fast the transcriber turns audio
into text — was unexplored by that family. ``stt_rtf`` exists as a *per-turn live
metric* (iter-049, surfaced in the chat session summary and the iter-140
consistency sentinel), but there was no way to FOLD a handful of measured
transcriptions into one robust verdict the way ``calibrate_base_wpm`` does for
the bot's speaking rate.

This module is that verdict. It is the **pure measurement core**:

- :class:`TranscriptionSample` carries one transcription (``audio_seconds``,
  ``transcribe_seconds``) and derives its ``rtf`` (transcribe / audio — the
  iter-049 convention: < 1 means the STT keeps up with realtime, > 1 means it is
  the latency bottleneck).
- :func:`profile_stt_rtf` folds one-or-more samples into a robust **median** RTF
  (robust to a single mis-timed run, exactly like ``calibrate_base_wpm``'s
  median) plus spread, ``relative_spread``, a categorical ``speed_grade``
  (``"fast"`` / ``"realtime"`` / ``"slow"``), and a ``speed_margin`` (headroom to
  the next-worse grade knee).
- :func:`stt_rtf_verdict` folds an :class:`SttRtfProfile` into a data-driven
  recommend/keep call (iter-407) — the STT-side twin of iter-222's
  :func:`~session.wpm_mirror.calibration_verdict`. It recommends ACTION (a
  lighter model or streaming partials) only when the runs are ENOUGH
  (``n_samples``), AGREE (``relative_spread``), AND the median RTF is genuinely
  SLOW enough to be the bottleneck; otherwise it says keep the current engine.

Pure arithmetic over injected timings — no torch, no faster-whisper, no audio
I/O, no clock — so it imports and runs in the unit gate on any platform, the
same property that lets the ``calibrate_base_wpm`` core be tested without a TTS
engine. A caller (a future ``gv stt-rtf`` command, mirroring iter-221's
``gv calibrate-base-wpm``) wraps a real ``STTEngine`` to PRODUCE the samples; the
fold itself never touches hardware.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


#: ``rtf`` at or below which the transcriber is graded ``"fast"`` — it turns
#: audio into text in well under realtime, so end-of-turn STT can be invoked
#: inline with no perceptible stall. 0.5 means the transcription takes at most
#: half the audio's duration; mlx-whisper on Apple Silicon (the iter-049
#: reference point) lands ~0.1-0.3, comfortably inside this band.
RTF_FAST_MAX: float = 0.5

#: ``rtf`` at or below which the transcriber still keeps up with realtime
#: (``"realtime"`` grade); above it the STT is the latency bottleneck
#: (``"slow"``). 1.0 is the natural knee: an rtf of exactly 1.0 means the
#: transcription takes as long as the audio it transcribes, so anything slower
#: cannot keep pace with a live stream without falling behind.
RTF_REALTIME_MAX: float = 1.0


@dataclass(frozen=True)
class TranscriptionSample:
    """One measured transcription used to profile STT speed.

    A clip of known duration (``audio_seconds``) transcribed in a measured
    wall-clock time (``transcribe_seconds``). Derives the real-time factor
    (``rtf``) the iter-049 way: transcribe time per second of audio.

    Attributes:
      audio_seconds: duration of the transcribed clip (must be ``> 0``).
      transcribe_seconds: measured wall-clock transcription time (must be
        ``> 0``).

    Both are validated in ``__post_init__`` because a non-positive value is a
    measurement bug (an empty clip or an unmeasured timer) that would otherwise
    divide by zero or produce a nonsensical negative RTF — the same guard
    :class:`~session.wpm_mirror.CalibrationSample` applies to its render fields.
    """

    audio_seconds: float
    transcribe_seconds: float

    def __post_init__(self) -> None:
        if self.audio_seconds <= 0:
            raise ValueError(
                f"audio_seconds must be positive (got {self.audio_seconds})"
            )
        if self.transcribe_seconds <= 0:
            raise ValueError(
                f"transcribe_seconds must be positive "
                f"(got {self.transcribe_seconds})"
            )

    @property
    def rtf(self) -> float:
        """Real-time factor (iter-049): transcribe_seconds / audio_seconds.

        ``< 1`` ⇒ the STT runs faster than realtime (can be invoked inline at
        end-of-turn with no stall); ``> 1`` ⇒ the STT is the latency bottleneck
        (the engine/model is too heavy for the host, streaming partial
        transcription or a smaller model would help).
        """
        return self.transcribe_seconds / self.audio_seconds


@dataclass(frozen=True)
class SttRtfProfile:
    """Verdict of folding one-or-more :class:`TranscriptionSample` runs.

    The STT-side twin of :class:`~session.wpm_mirror.BaseWpmCalibration`.

    Attributes:
      median_rtf: the robust **median** of the samples' ``rtf`` — the
        representative real-time factor for this engine/model on this host.
        Median (not mean) so a single mis-timed run (a cold model load, a GC
        pause) does not skew it, mirroring ``calibrate_base_wpm``'s median.
      n_samples: how many samples were folded.
      min_rtf / max_rtf: the extremes of the per-sample ``rtf`` — the profile's
        range.
      spread: ``max_rtf - min_rtf`` — a large spread means the runs disagree
        (thermal throttling, contended host, variable clip difficulty), so the
        median is less trustworthy.
      relative_spread: ``spread / median_rtf`` — the spread normalized by the
        median, a dimensionless coefficient of dispersion (the iter-393
        convention). The absolute ``spread`` is rate-dependent (a 0.1 range is
        tight at an rtf of 2.0 but wide at an rtf of 0.15), so this companion
        lets an operator judge whether the runs AGREE independent of the
        engine's nominal speed. ``0.0`` when the runs agree exactly.
      speed_grade: a categorical speed grade bucketing ``median_rtf`` into
        ``"fast"`` / ``"realtime"`` / ``"slow"`` — the one-glance read of whether
        end-of-turn STT can run inline (``"fast"``), keeps pace with a live
        stream (``"realtime"``), or is the latency bottleneck (``"slow"``). See
        :func:`rtf_speed_grade` for the boundaries.
      speed_margin: how much ``median_rtf`` HEADROOM the profile has before its
        ``speed_grade`` would degrade to the next-worse grade — the calibration
        analogue of iter-396's ``dispersion_margin``. ``"slow"`` (the worst
        grade) has no worse grade to degrade into, so its margin is ``None``.
        See :func:`rtf_speed_margin`.
    """

    median_rtf: float
    n_samples: int
    min_rtf: float
    max_rtf: float
    spread: float
    relative_spread: float
    speed_grade: str
    speed_margin: float | None


def rtf_speed_grade(rtf: float) -> str:
    """Bucket an STT ``rtf`` into a categorical speed grade.

    The STT-side analogue of iter-394's :func:`dispersion_grade`: turning a raw
    real-time factor into a one-glance ``"fast"`` / ``"realtime"`` / ``"slow"``
    read of how well the transcriber keeps up. Shares the iter-140
    ``_stt_rtf_bucket`` semantics (the per-turn consistency sentinel) but uses
    the calibration family's inclusive-lower-band knee convention so a value
    landing exactly on a knee grades the more FAVOURABLE side.

    Boundaries:

    - ``rtf <= RTF_FAST_MAX`` (0.5) ⇒ ``"fast"`` — transcription takes at most
      half the audio's duration; end-of-turn STT runs inline with no stall.
    - ``<= RTF_REALTIME_MAX`` (1.0) ⇒ ``"realtime"`` — still keeps pace with a
      live stream, but with less headroom.
    - otherwise ⇒ ``"slow"`` — the STT takes longer than the audio it
      transcribes; it is the latency bottleneck (a smaller model or streaming
      partial transcription would help).

    Pure: a function of one float.
    """
    r = float(rtf)
    if r <= RTF_FAST_MAX:
        return "fast"
    if r <= RTF_REALTIME_MAX:
        return "realtime"
    return "slow"


def rtf_speed_margin(rtf: float) -> float | None:
    """How much ``rtf`` headroom is left before the speed grade degrades.

    :func:`rtf_speed_grade` answers *which* speed band an RTF falls in; this
    answers *how comfortably* — the distance from ``rtf`` up to the knee where it
    would tip into the next-WORSE grade. The STT-side twin of iter-396's
    :func:`~session.wpm_mirror.dispersion_margin`: a ``"fast"`` at rtf 0.49
    (margin 0.01) is one slow run from ``"realtime"``, while one at 0.15 (margin
    0.35) sits firmly inside the band.

    Returns, by grade:

    - ``"fast"`` ⇒ ``RTF_FAST_MAX - rtf`` — headroom to the fast/realtime knee
      (0.5).
    - ``"realtime"`` ⇒ ``RTF_REALTIME_MAX - rtf`` — headroom to the
      realtime/slow knee (1.0).
    - ``"slow"`` ⇒ ``None``. The worst grade has no worse grade to degrade into,
      so "headroom to the next-worse grade" is undefined — spelled ``None`` the
      same way the calibration/gap family spells "not measurable".

    A value sitting exactly on a knee grades the more favourable side (the
    inclusive-lower-band convention) with a ``0.0`` margin. Pure: a function of
    one float.
    """
    r = float(rtf)
    grade = rtf_speed_grade(r)
    if grade == "fast":
        return RTF_FAST_MAX - r
    if grade == "realtime":
        return RTF_REALTIME_MAX - r
    return None


def profile_stt_rtf(samples) -> SttRtfProfile | None:
    """Fold transcription-timing runs into a robust STT-RTF verdict.

    The STT-side twin of :func:`~session.wpm_mirror.calibrate_base_wpm`. Returns
    the **median** of the per-sample ``rtf`` as the representative real-time
    factor (robust to a single mis-timed run) plus spread, ``relative_spread``,
    a categorical ``speed_grade`` (``"fast"`` / ``"realtime"`` / ``"slow"``), and
    a ``speed_margin`` (headroom to the next-worse grade knee).

    Args:
      samples: iterable of :class:`TranscriptionSample`. Empty ⇒ ``None``
        (nothing to profile from).

    Pure — no I/O, no clock, no mutation of the input.
    """
    rtfs = [s.rtf for s in samples]
    if not rtfs:
        return None
    median = statistics.median(rtfs)
    lo = min(rtfs)
    hi = max(rtfs)
    spread = hi - lo
    # median is a positive rate (each rtf > 0 since audio_seconds and
    # transcribe_seconds are both positive), so the division is well-defined.
    relative_spread = spread / median
    return SttRtfProfile(
        median_rtf=median,
        n_samples=len(rtfs),
        min_rtf=lo,
        max_rtf=hi,
        spread=spread,
        relative_spread=relative_spread,
        speed_grade=rtf_speed_grade(median),
        speed_margin=rtf_speed_margin(median),
    )


# --------------------------------------------------------------------------
# iter-407 — data-driven verdict over an STT-RTF profile.
#
# iter-405 measured the median RTF and iter-406 surfaced it on the CLI, but both
# stop at raw numbers (median, spread, grade, margin) and leave the operator to
# eyeball whether the transcriber is actually too slow to act on. That is the
# same gap iter-222's ``calibration_verdict`` closed for ``base_wpm``: a verdict,
# not a bare reading. This is the STT-side twin. Acting on a slow STT (dropping
# to a lighter model, or wiring streaming partial transcription) is worth doing
# only when the runs AGREE (small relative_spread ⇒ the median is trustworthy),
# there are ENOUGH of them (a single timing is not a profile), AND the median
# RTF is genuinely SLOW enough to be the latency bottleneck (a "fast" or
# "realtime" engine keeps pace, so swapping it just churns config for no
# latency win). All three gates must pass to recommend action — otherwise keep
# the current engine. Pure arithmetic over an existing ``SttRtfProfile``; no
# I/O, no clock.
#
# NOTE the trust gate uses ``relative_spread`` (dimensionless), NOT the absolute
# ``spread`` iter-222 uses. RTF is rate-dependent — a 0.1 spread is tight at an
# rtf of 2.0 but wide at 0.15 — so the absolute range is not comparable across
# engines, whereas relative_spread is (the same reason iter-393 introduced it
# for the calibration family). The significance gate is the categorical
# ``speed_grade == "slow"`` rather than an absolute drift, because "slow" is the
# meaningful threshold for the STT side (the rtf crossed the realtime knee),
# just as a drift past ``drift_min`` is the meaningful threshold on the TTS side.
# --------------------------------------------------------------------------

#: ``relative_spread`` at or below which the profile's runs are trusted to AGREE
#: — the median RTF is a reliable read of the engine's speed. Above this the
#: timings disagree (thermal throttling, a contended host, variable clip
#: difficulty), so the median is not trustworthy and the right move is to
#: re-measure more consistently rather than act on a noisy verdict. 0.15 matches
#: :data:`~session.wpm_mirror.CALIB_LOOSE_REL_SPREAD` — the calibration family's
#: "loose but usable" knee — so the STT and TTS sides share one dimensionless
#: agreement bar.
DEFAULT_STT_RTF_REL_SPREAD_MAX: float = 0.15

#: Minimum number of samples for the median to be robust. A single transcription
#: is one timing, not a profile; a couple can still be a fluke. Mirrors
#: :data:`~session.wpm_mirror.DEFAULT_CALIB_MIN_SAMPLES`.
DEFAULT_STT_RTF_MIN_SAMPLES: int = 3


@dataclass(frozen=True)
class SttRtfVerdict:
    """Data-driven recommendation over an :class:`SttRtfProfile`.

    The STT-side twin of :class:`~session.wpm_mirror.CalibrationVerdict`
    (iter-222). Where that decides whether to re-seed ``base_wpm`` from a TTS
    calibration, this decides whether the transcriber is slow enough — and the
    measurement trustworthy enough — to be worth ACTING on (dropping to a
    lighter model, or wiring streaming partial transcription).

    Attributes:
      recommend: ``True`` iff all three gates pass — there are enough samples
        (``n_samples >= min_samples``), the runs agree
        (``relative_spread <= rel_spread_max``), and the median RTF is genuinely
        ``"slow"`` (it crossed the realtime knee, so the STT is the latency
        bottleneck). When ``True`` the operator should lighten the STT path;
        when ``False`` keep the current engine.
      reason: a short human-readable explanation of the decision — which gate
        failed, or that all passed.
      median_rtf: the profile's measured median RTF (echoed; the value the
        decision is about).
      speed_grade: the profile's ``"fast"`` / ``"realtime"`` / ``"slow"`` grade
        (echoed; the significance gate reads it).
      relative_spread: the profile's dimensionless coefficient of dispersion
        (echoed; the trust gate reads it).
      n_samples: how many samples backed the profile (echoed; the sample gate
        reads it).
      rel_spread_max / min_samples: the thresholds the verdict was computed
        against (echoed so the decision is self-describing).
    """

    recommend: bool
    reason: str
    median_rtf: float
    speed_grade: str
    relative_spread: float
    n_samples: int
    rel_spread_max: float
    min_samples: int


def stt_rtf_verdict(
    profile: "SttRtfProfile | None",
    *,
    rel_spread_max: float = DEFAULT_STT_RTF_REL_SPREAD_MAX,
    min_samples: int = DEFAULT_STT_RTF_MIN_SAMPLES,
) -> SttRtfVerdict | None:
    """Decide whether a slow STT-RTF profile is worth acting on.

    Folds the three trust/significance gates over an existing
    :class:`SttRtfProfile` (the iter-405 measurement). Acting on the STT path
    (a lighter model, streaming partials) is recommended only when **all** of:

    - **enough samples** — ``n_samples >= min_samples`` (a single transcription
      is one timing, not a profile);
    - **runs agree** — ``relative_spread <= rel_spread_max`` (a wide dispersion
      means the median is not trustworthy, so re-measure rather than act);
    - **genuinely slow** — ``speed_grade == "slow"`` (the median crossed the
      realtime knee; a ``"fast"`` or ``"realtime"`` engine already keeps pace,
      so swapping it just churns config for no latency win).

    The gates are checked in that order so ``reason`` names the *first* failure
    (sample count is the most fundamental, then trust, then significance) —
    mirroring :func:`~session.wpm_mirror.calibration_verdict`'s ordering.

    Args:
      profile: an :class:`SttRtfProfile`, or ``None`` (no samples ⇒ nothing to
        decide ⇒ this function returns ``None``, mirroring
        :func:`profile_stt_rtf`'s empty contract and
        :func:`~session.wpm_mirror.calibration_verdict`'s).
      rel_spread_max: max trusted dimensionless dispersion (defaults to
        :data:`DEFAULT_STT_RTF_REL_SPREAD_MAX`).
      min_samples: min sample count for a robust median (defaults to
        :data:`DEFAULT_STT_RTF_MIN_SAMPLES`).

    Pure — reads only the profile's fields, mutates nothing.
    """
    if profile is None:
        return None

    rel_spread_max = float(rel_spread_max)
    min_samples = int(min_samples)

    n = profile.n_samples
    rel_spread = profile.relative_spread
    grade = profile.speed_grade
    median = profile.median_rtf

    if n < min_samples:
        recommend = False
        reason = (
            f"only {n} sample(s) — need {min_samples}+ for a robust median; "
            "keep the current engine"
        )
    elif rel_spread > rel_spread_max:
        recommend = False
        reason = (
            f"runs disagree (relative spread {rel_spread:.2f} > "
            f"{rel_spread_max:.2f}) — the median RTF is not trustworthy; "
            "re-measure more consistently"
        )
    elif grade != "slow":
        recommend = False
        reason = (
            f"median RTF {median:.2f} grades {grade} — the STT keeps pace; "
            "keep the current engine"
        )
    else:
        recommend = True
        reason = (
            f"runs agree (relative spread {rel_spread:.2f} <= "
            f"{rel_spread_max:.2f}) over {n} samples and median RTF "
            f"{median:.2f} is slow — lighten the STT path (a smaller model or "
            "streaming partial transcription)"
        )

    return SttRtfVerdict(
        recommend=recommend,
        reason=reason,
        median_rtf=median,
        speed_grade=grade,
        relative_spread=rel_spread,
        n_samples=n,
        rel_spread_max=rel_spread_max,
        min_samples=min_samples,
    )


# --------------------------------------------------------------------------
# iter-409 — profile a CORPUS of STT engines/models in one fold.
#
# iter-405 measured one engine's median RTF, iter-406 surfaced it on the CLI,
# and iter-407/408 folded + surfaced the recommend/keep verdict. All three stop
# at a SINGLE engine. The deferred remaining step is the BATCH surface — profile
# N engines/models and tabulate which ones keep up with realtime — the STT-side
# twin of iter-397's ``calibrate_base_wpm_batch`` (which profiles N TTS voices'
# base rates). An operator choosing a transcriber for the host (mlx-whisper vs
# faster-whisper vs a heavier model) had no single surface ranking their RTFs;
# this builds it. Pure arithmetic over injected timings — no torch, no
# faster-whisper, no audio I/O — so it runs in the unit gate on any platform,
# exactly like the single-engine core above.
# --------------------------------------------------------------------------

#: Histogram bucket order for an STT-RTF batch (iter-409). The three real speed
#: grades plus an ``"unprofiled"`` bucket for an engine submitted with no
#: samples — kept DISTINCT from ``"slow"`` so a no-sample engine never silently
#: merges into the bottleneck count (the same ``uncalibrated``-vs-``scattered``
#: distinction iter-397's :data:`~session.wpm_mirror.CALIB_BATCH_GRADE_ORDER`
#: makes). A gv render reads this order so the two never drift on the bucket set,
#: and the histogram always shows all four buckets summing to ``num_engines``.
STT_RTF_BATCH_GRADE_ORDER = ("fast", "realtime", "slow", "unprofiled")


def _percentile_of_sorted(sorted_values, p: float) -> float:
    """Linear-interpolated percentile ``p`` of an already-SORTED, non-empty list.

    The R-7 / numpy-``"linear"`` convention (iter-414, the STT-RTF-batch twin of
    :func:`session.wpm_mirror._percentile_of_sorted`, kept local so this module
    stays self-contained and importable without the calibration pipeline): for
    ``n`` samples the fractional rank is ``(p / 100) * (n - 1)`` and the value
    interpolates between the samples at the floor and ceil of that rank. A single
    sample yields that sample for every percentile. Caller sorts and guards the
    empty list. Returns the raw (unrounded) value — the caller rounds.
    """
    n = len(sorted_values)
    rank = (p / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


@dataclass(frozen=True)
class SttRtfBatch:
    """Verdict of profiling a CORPUS of STT engines/models (iter-409).

    The batch analogue of :class:`SttRtfProfile`, and the STT-side twin of
    :class:`~session.wpm_mirror.BaseWpmCalibrationBatch` (iter-397): one
    :class:`SttRtfProfile` per engine (``rows``), summarised by the
    outlier-robust corpus median of the per-engine ``median_rtf``.

    Attributes:
      rows: one entry per engine, in input order, each a dict with keys
        ``engine`` (the label), ``profile`` (the engine's
        :class:`SttRtfProfile`, or ``None`` when the engine had no samples),
        ``verdict`` (the engine's :class:`SttRtfVerdict` — the iter-407
        recommend/keep call, or ``None`` for an unprofiled engine), and
        ``delta_from_median_rtf`` (the engine's ``median_rtf`` minus the corpus
        median, ``None`` for an unprofiled engine), and ``flyer`` (the iter-414
        boolean — ``True`` when the engine's ``median_rtf`` falls outside the
        Tukey fence ``[Q1 - 1.5*IQR, Q3 + 1.5*IQR]``, ``False`` when it sits
        inside, ``None`` for an unprofiled engine that carries no median). The
        per-engine grade / margin / spread live on the embedded ``profile`` so
        each row agrees EXACTLY with ``gv stt-rtf`` on that engine's samples.
      num_engines: how many engines were submitted (``len(rows)``).
      num_profiled: how many carried at least one sample (fed the corpus
        aggregates); an unprofiled engine contributes a row but not a number.
      corpus_median_rtf: the outlier-robust **median** of the profiled engines'
        ``median_rtf`` — the representative RTF across the fleet of engines
        (``None`` when none profiled). Median (not mean) so one pathological
        engine cannot drag the corpus centre.
      corpus_min_rtf / corpus_max_rtf: the extremes of the per-engine
        ``median_rtf`` — the fastest and slowest engine (``None`` when none
        profiled).
      corpus_spread: ``corpus_max_rtf - corpus_min_rtf`` — how far apart the
        engines clock (``None`` when none profiled). A large spread means the
        engines genuinely differ in speed on this host, so the choice of
        transcriber matters.
      corpus_q1_rtf / corpus_q3_rtf: the 25th / 75th percentiles of the
        per-engine ``median_rtf`` (R-7 interpolation), ``None`` when none
        profiled. The edges of the middle half of the corpus.
      corpus_iqr_rtf: ``corpus_q3_rtf - corpus_q1_rtf`` — the outlier-ROBUST
        spread (iter-414). Where ``corpus_spread`` (max - min) is a single pair
        of extremes — one pathological engine inflates it — the IQR measures the
        width of the MIDDLE HALF of the corpus, so a lone outlier engine cannot
        widen it. A tight IQR alongside a wide spread is the signature of "the
        engines agree on a typical speed, but one is a flyer". The STT-side twin
        of iter-403's ``implied_base_wpm_iqr``. ``None`` when none profiled.
      corpus_fence_lo_rtf / corpus_fence_hi_rtf: the standard Tukey boxplot fence
        ``[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`` over the per-engine ``median_rtf``
        (iter-414, the STT-side analogue of iter-404's
        ``implied_base_wpm_fence_lo``/``_hi``). An engine whose ``median_rtf``
        falls outside this band is a ``flyer`` — a transcriber clocking at a
        speed the rest of the corpus does not. ``None`` when none profiled.
      num_flyers: how many engines are flyers (``median_rtf`` outside the Tukey
        fence); ``0`` when none profiled. Reading high alongside a tight IQR
        means one engine is an outlier the corpus median already shrugged off.
      grade_counts: how many engines sit at each speed grade, keyed by
        :data:`STT_RTF_BATCH_GRADE_ORDER` (always all four buckets, summing to
        ``num_engines``) — at a glance, how many engines keep up with realtime
        (``fast`` + ``realtime``) versus are the bottleneck (``slow``).
      num_keep_up: how many engines keep pace with realtime — their
        ``speed_grade`` is ``"fast"`` or ``"realtime"`` (``median_rtf <= 1.0``).
        The one-number answer to "which engines keep up?" — the count, with the
        histogram giving the breakdown.
      num_recommend: how many engines the iter-407 verdict flags for action
        (``recommend`` is ``True`` — genuinely slow AND the measurement is
        trustworthy). ``<= grade_counts["slow"]``: a slow engine measured from
        too few or disagreeing samples is NOT recommended (the median is not yet
        trustworthy), so this counts only the engines worth acting on now.
      rel_spread_max / min_samples: the verdict gate thresholds every per-engine
        verdict was computed against (echoed so the batch is self-describing and
        the gv render can name the gates without re-deriving them).

    Pure / frozen; built from :func:`profile_stt_rtf` + :func:`stt_rtf_verdict`
    per engine.
    """

    rows: tuple
    num_engines: int
    num_profiled: int
    corpus_median_rtf: float | None
    corpus_min_rtf: float | None
    corpus_max_rtf: float | None
    corpus_spread: float | None
    corpus_q1_rtf: float | None
    corpus_q3_rtf: float | None
    corpus_iqr_rtf: float | None
    corpus_fence_lo_rtf: float | None
    corpus_fence_hi_rtf: float | None
    num_flyers: int
    grade_counts: dict
    num_keep_up: int
    num_recommend: int
    rel_spread_max: float
    min_samples: int


def profile_stt_rtf_batch(
    engines,
    *,
    rel_spread_max: float = DEFAULT_STT_RTF_REL_SPREAD_MAX,
    min_samples: int = DEFAULT_STT_RTF_MIN_SAMPLES,
) -> SttRtfBatch:
    """Profile a corpus of STT engines and tabulate which ones keep up (iter-409).

    ``engines`` is an iterable of ``(label, samples)`` pairs — one per engine,
    where ``samples`` is that engine's iterable of :class:`TranscriptionSample`
    timings (the same input :func:`profile_stt_rtf` takes). Each engine is
    profiled INDEPENDENTLY and its profile is folded through
    :func:`stt_rtf_verdict` (against the shared ``rel_spread_max`` /
    ``min_samples`` gates so the per-engine recommendations are apples-to-apples),
    and the per-engine ``median_rtf`` values are summarised by their
    outlier-robust median.

    An engine with no samples profiles to ``None`` (the :func:`profile_stt_rtf`
    empty contract): it contributes a row tagged ``unprofiled`` (its ``verdict``
    is ``None``) and is excluded from the corpus median / extremes / spread, but
    is still counted in ``num_engines``. This mirrors how iter-397's
    ``calibrate_base_wpm_batch`` keeps a no-sample voice in the table with a
    ``None`` calibration rather than dropping it silently.

    Args:
      engines: iterable of ``(label, samples)`` pairs.
      rel_spread_max: the verdict trust gate, threaded to every per-engine
        :func:`stt_rtf_verdict` (defaults to
        :data:`DEFAULT_STT_RTF_REL_SPREAD_MAX`).
      min_samples: the verdict sample gate, threaded likewise (defaults to
        :data:`DEFAULT_STT_RTF_MIN_SAMPLES`).

    Returns an :class:`SttRtfBatch`. Pure — no I/O, no clock, no mutation of the
    inputs.
    """
    rel_spread_max = float(rel_spread_max)
    min_samples = int(min_samples)

    rows = []
    medians = []
    for label, samples in engines:
        profile = profile_stt_rtf(samples)
        verdict = stt_rtf_verdict(
            profile, rel_spread_max=rel_spread_max, min_samples=min_samples
        )
        rows.append(
            {
                "engine": label,
                "profile": profile,
                "verdict": verdict,
                # delta + flyer filled in once the corpus median / fences are known.
                "delta_from_median_rtf": None,
                "flyer": None,
            }
        )
        if profile is not None:
            medians.append(profile.median_rtf)

    if medians:
        corpus_median = statistics.median(medians)
        corpus_min = min(medians)
        corpus_max = max(medians)
        corpus_spread = corpus_max - corpus_min
        # iter-414 outlier-ROBUST spread: the inter-quartile range (Q3 - Q1) of the
        # per-engine median RTFs. Where ``corpus_spread`` (max - min) is a single
        # pair of extremes — one pathological engine inflates it — the IQR measures
        # the width of the MIDDLE HALF of the corpus, so a lone outlier engine
        # cannot widen it. The STT-side twin of iter-403's IQR on
        # ``calibrate_base_wpm_batch``.
        srt = sorted(medians)
        corpus_q1 = _percentile_of_sorted(srt, 25)
        corpus_q3 = _percentile_of_sorted(srt, 75)
        corpus_iqr = corpus_q3 - corpus_q1
        # iter-414 outlier (flyer) flag, the STT-side analogue of iter-404's flyer
        # flag on ``calibrate_base_wpm_batch``. The standard Tukey boxplot fence: an
        # engine is a flyer when its median RTF falls below Q1 - 1.5*IQR or above
        # Q3 + 1.5*IQR, so an operator picking a transcriber sees WHICH engine
        # clocks at a speed the rest of the corpus does not without scanning the
        # Δmedian column. When IQR is 0 (a degenerate corpus whose middle half is a
        # single value) the fences collapse to [Q1, Q3] and only engines strictly
        # outside that band are flyers — a lone different engine among identical
        # ones is correctly named.
        fence_lo = corpus_q1 - 1.5 * corpus_iqr
        fence_hi = corpus_q3 + 1.5 * corpus_iqr
        for r in rows:
            profile = r["profile"]
            if profile is not None:
                r["delta_from_median_rtf"] = profile.median_rtf - corpus_median
                r["flyer"] = (
                    profile.median_rtf < fence_lo
                    or profile.median_rtf > fence_hi
                )
        num_flyers = sum(1 for r in rows if r["flyer"])
    else:
        corpus_median = corpus_min = corpus_max = corpus_spread = None
        corpus_q1 = corpus_q3 = corpus_iqr = None
        fence_lo = fence_hi = None
        num_flyers = 0

    counts = {g: 0 for g in STT_RTF_BATCH_GRADE_ORDER}
    for r in rows:
        profile = r["profile"]
        key = "unprofiled" if profile is None else profile.speed_grade
        if key in counts:
            counts[key] += 1
        else:
            # Defensive: an unrecognised future grade lands in unprofiled rather
            # than vanishing — the counts must still sum to num_engines.
            counts["unprofiled"] += 1

    # An engine "keeps up" when its median RTF stays at or under the realtime
    # knee — i.e. it grades "fast" or "realtime", the two non-bottleneck buckets.
    num_keep_up = counts["fast"] + counts["realtime"]
    # The verdict recommends action only for a genuinely-slow AND trustworthy
    # engine, so num_recommend <= grade_counts["slow"].
    num_recommend = sum(
        1 for r in rows if r["verdict"] is not None and r["verdict"].recommend
    )

    return SttRtfBatch(
        rows=tuple(rows),
        num_engines=len(rows),
        num_profiled=len(medians),
        corpus_median_rtf=corpus_median,
        corpus_min_rtf=corpus_min,
        corpus_max_rtf=corpus_max,
        corpus_spread=corpus_spread,
        corpus_q1_rtf=corpus_q1,
        corpus_q3_rtf=corpus_q3,
        corpus_iqr_rtf=corpus_iqr,
        corpus_fence_lo_rtf=fence_lo,
        corpus_fence_hi_rtf=fence_hi,
        num_flyers=num_flyers,
        grade_counts=counts,
        num_keep_up=num_keep_up,
        num_recommend=num_recommend,
        rel_spread_max=rel_spread_max,
        min_samples=min_samples,
    )
