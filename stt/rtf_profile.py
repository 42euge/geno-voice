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
