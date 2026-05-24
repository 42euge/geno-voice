"""Per-turn metrics struct + printer extracted from mic_chat.

Lives in its own module so tests can import ``TurnMetrics`` without
pulling in mic_chat's top-level ``import pyaudio`` (which is
unavailable on x86_64 Linux without ALSA dev headers, and on most
CI runners).

Same pattern as iter-006/007 — pull pure-Python primitives out of
the pyaudio-bound entry point.

Also hosts ``print_session_summary`` (iter-017): the
KeyboardInterrupt summary block previously inlined in
``mic_chat.run_chat``, now testable + using ``statistics.median``
for proper even-length handling.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

# ANSI codes — duplicated from mic_chat so this module remains a
# clean leaf with no dependency back on mic_chat itself.
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


@dataclass
class TurnMetrics:
    speech_duration: float = 0.0
    stt_time: float = 0.0
    llm_first_token: float = 0.0
    # iter-038: time from LLM start to the first complete sentence
    # reaching the TTS worker. Distinct from llm_first_token: the
    # LLM may stream chatty preamble for a while before a terminator
    # arrives. Metric 1.10 in the perf-metrics taxonomy. 0 means
    # no complete sentence ever emerged this turn (rare).
    llm_first_sentence: float = 0.0
    llm_total: float = 0.0
    tts_time: float = 0.0
    playback_time: float = 0.0
    ttfs: float = 0.0
    total_e2e: float = 0.0
    sentences_spoken: int = 0
    # iter-014: surface filler + barge-in counters that were added
    # to the worker / coordinator in iter-011 / iter-012 but never
    # made it into the per-turn summary.
    fillers_played: int = 0
    barge_in: bool = False
    # iter-037: count of mic frames flushed at start of turn (or
    # on the LLM-error path). Metric 2.19 from the perf-metrics
    # taxonomy. Many stale frames means the mic accumulated
    # unwanted audio between turns — bot voice leaking back via
    # OS loopback / Bluetooth duplex / acoustic echo. A reliable
    # signal that echo cancellation is needed in the user's setup.
    mic_stale_frames: int = 0
    transcript: str = ""
    response: str = ""
    model: str = ""

    def print(self, turn: int) -> None:
        print()
        print(f"  {_DIM}{'─' * 56}{_RESET}")
        print(f"  {_BOLD}Turn {turn}{_RESET}")
        print(f"  {_DIM}You:{_RESET} \"{self.transcript}\"")
        print()
        print(f"  {_DIM}┌─ PIPELINE{_RESET}")
        print(f"  {_DIM}│{_RESET}  Speech:        {self.speech_duration*1000:>7.0f}ms")
        print(f"  {_DIM}│{_RESET}  STT:           {self.stt_time*1000:>7.0f}ms")
        print(f"  {_DIM}│{_RESET}  LLM 1st tok:   {self.llm_first_token*1000:>7.0f}ms")
        # iter-038: TTFsent — time-to-first-sentence. Show the gap
        # between first-token and first-sentence in parens so the
        # user sees how much "preamble lag" the splitter waited
        # through. Skip the line entirely on the rare turn where
        # no complete sentence emerged.
        if self.llm_first_sentence > 0:
            preamble_gap = self.llm_first_sentence - self.llm_first_token
            print(
                f"  {_DIM}│{_RESET}  LLM 1st sent:  "
                f"{self.llm_first_sentence*1000:>7.0f}ms  "
                f"({_DIM}+{preamble_gap*1000:.0f}ms preamble{_RESET})"
            )
        print(f"  {_DIM}│{_RESET}  LLM total:     {self.llm_total*1000:>7.0f}ms  ({self.model})")
        tts_suffix = f"({self.sentences_spoken} sentences"
        if self.fillers_played > 0:
            tts_suffix += f" + {self.fillers_played} filler"
            if self.fillers_played > 1:
                tts_suffix += "s"
        tts_suffix += ")"
        print(f"  {_DIM}│{_RESET}  TTS:           {self.tts_time*1000:>7.0f}ms  {tts_suffix}")
        print(f"  {_DIM}│{_RESET}  Playback:      {self.playback_time*1000:>7.0f}ms")
        if self.barge_in:
            print(
                f"  {_DIM}│{_RESET}  {_YELLOW}Barge-in:      "
                f"yes (user interrupted){_RESET}"
            )
        # iter-037: only emit when non-zero — a clean turn shouldn't
        # spend pixels on a stale-frame counter that's almost always 0.
        # When >0 it's worth noticing — bot voice leaking back through
        # the OS mic is a real-world problem that points at acoustic
        # echo / Bluetooth duplex / loopback misconfiguration.
        if self.mic_stale_frames > 0:
            stale_seconds = self.mic_stale_frames / 16000  # RATE
            color = _YELLOW if stale_seconds > 0.5 else _DIM
            print(
                f"  {_DIM}│{_RESET}  {color}Mic stale:     "
                f"{self.mic_stale_frames:>5} frames ({stale_seconds:.1f}s){_RESET}"
            )
        print(f"  {_DIM}│{_RESET}")
        ttfs_color = _GREEN if self.ttfs < 3.0 else _YELLOW
        print(
            f"  {_DIM}├─{_RESET} {_BOLD}TTFS:{_RESET}            "
            f"{ttfs_color}{self.ttfs*1000:>7.0f}ms{_RESET}  "
            f"(speech stop → speaker)"
        )
        total_color = _GREEN if self.total_e2e < 6.0 else _YELLOW
        print(
            f"  {_DIM}└─{_RESET} {_BOLD}Total turn:{_RESET}      "
            f"{total_color}{self.total_e2e*1000:>7.0f}ms{_RESET}"
        )
        print(f"  {_DIM}{'─' * 56}{_RESET}")
        print()


def _median_ms(values: list[float]) -> float:
    """Return the median of `values` in milliseconds.

    Uses ``statistics.median`` so even-length lists return the
    average of the two middle elements (rather than the upper
    median that ``sorted[len//2]`` produces — see iter-017 for
    why that mattered).
    """
    if not values:
        return 0.0
    return statistics.median(values) * 1000


def print_session_summary(metrics_list: list[TurnMetrics], llm_config: dict, *, file=None) -> None:
    """Print a multi-line session summary on KeyboardInterrupt.

    Was inlined inside ``mic_chat.run_chat``'s KeyboardInterrupt
    handler with two issues iter-017 fixes:
      - ``sorted[len//2]`` reports the upper median for even-length
        lists, biasing 2-turn (and other small) sessions.
      - It was untestable without instantiating mic_chat.

    `file` defaults to ``sys.stdout`` (via ``print``); tests pass
    a ``StringIO`` to inspect the output.
    """
    def _emit(line: str = "") -> None:
        if file is None:
            print(line)
        else:
            file.write(line + "\n")

    _emit()
    _emit()
    _emit(f"{_DIM}{'─' * 56}{_RESET}")
    if not metrics_list:
        _emit(f"{_BOLD}  Session ended (no completed turns){_RESET}")
        _emit()
        return

    n = len(metrics_list)
    stt_times = [m.stt_time for m in metrics_list]
    llm_ft = [m.llm_first_token for m in metrics_list]
    # iter-038: median TTFsent over turns where a sentence actually
    # emerged. Filter out 0s (parallel to iter-031's TTFS-zero filter)
    # so a turn with no complete sentence doesn't bias the median.
    llm_fs = [m.llm_first_sentence for m in metrics_list if m.llm_first_sentence > 0]
    tts_times = [m.tts_time for m in metrics_list]
    # iter-031: a turn that ended without audio (worker error,
    # barge-in before first audio, LLM produced no tokens) leaves
    # ``metrics.ttfs`` at its 0.0 default. Including those zeros
    # in the aggregate biases the median down and makes "Best
    # TTFS: 0ms" appear, which is misleading — TTFS only has
    # meaning for turns that actually played audio. Filter.
    ttfs_times = [m.ttfs for m in metrics_list if m.ttfs > 0]
    fillers_total = sum(m.fillers_played for m in metrics_list)
    barges_total = sum(1 for m in metrics_list if m.barge_in)
    # iter-037: aggregate mic-stale-frame totals. Only surface when
    # something actually leaked — a clean session shouldn't be cluttered
    # with a "0 stale frames" line.
    stale_total = sum(m.mic_stale_frames for m in metrics_list)

    _emit(f"{_BOLD}  Session Summary ({n} turn{'' if n == 1 else 's'}){_RESET}")
    _emit(f"    Median STT:       {_median_ms(stt_times):.0f}ms")
    _emit(f"    Median LLM 1st:   {_median_ms(llm_ft):.0f}ms")
    if llm_fs:
        _emit(f"    Median LLM sent:  {_median_ms(llm_fs):.0f}ms")
    _emit(f"    Median TTS:       {_median_ms(tts_times):.0f}ms")
    if ttfs_times:
        _emit(
            f"    {_BOLD}Median TTFS:      {_median_ms(ttfs_times):.0f}ms{_RESET}"
        )
        _emit(f"    Best TTFS:        {min(ttfs_times) * 1000:.0f}ms")
    else:
        # All turns ended without audio. Emit a placeholder rather
        # than a misleading "0ms" so the user knows it isn't a
        # win, it's an absence of data.
        _emit(f"    {_BOLD}Median TTFS:      n/a{_RESET}")
        _emit(f"    Best TTFS:        n/a")
    if fillers_total:
        _emit(f"    Fillers played:   {fillers_total}")
    if barges_total:
        _emit(f"    Barge-ins:        {barges_total}")
    if stale_total:
        # iter-037: surface aggregate stale-frame total so a "session
        # had constant echo" pattern is visible at the end of the run.
        stale_seconds_total = stale_total / 16000
        _emit(
            f"    Mic stale:        {stale_total} frames "
            f"({stale_seconds_total:.1f}s) — check echo cancellation"
        )
    _emit(f"    Model:            {llm_config.get('model', 'unknown')}")
    _emit()
