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
    tts_times = [m.tts_time for m in metrics_list]
    ttfs_times = [m.ttfs for m in metrics_list]
    fillers_total = sum(m.fillers_played for m in metrics_list)
    barges_total = sum(1 for m in metrics_list if m.barge_in)

    _emit(f"{_BOLD}  Session Summary ({n} turn{'' if n == 1 else 's'}){_RESET}")
    _emit(f"    Median STT:       {_median_ms(stt_times):.0f}ms")
    _emit(f"    Median LLM 1st:   {_median_ms(llm_ft):.0f}ms")
    _emit(f"    Median TTS:       {_median_ms(tts_times):.0f}ms")
    _emit(
        f"    {_BOLD}Median TTFS:      {_median_ms(ttfs_times):.0f}ms{_RESET}"
    )
    _emit(f"    Best TTFS:        {min(ttfs_times) * 1000:.0f}ms")
    if fillers_total:
        _emit(f"    Fillers played:   {fillers_total}")
    if barges_total:
        _emit(f"    Barge-ins:        {barges_total}")
    _emit(f"    Model:            {llm_config.get('model', 'unknown')}")
    _emit()
