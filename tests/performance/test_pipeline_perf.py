"""Performance integration tests — drive the ChatLoop across
real-world-shaped scenarios and record TTFS / STT / TTS / playback
times. Output is dumped to ``iter-reports/perf-results.json`` so
the report generator can render charts.

These tests are wall-clock-dependent and therefore inherently
noisy. They MUST NOT be part of the regular ``tests/`` run — gate
them behind the ``perf`` marker so they only execute when
explicitly requested:

    python -m pytest tests/performance/ -v

Each test is a single scenario. The tests don't assert specific
millisecond bounds (those vary by hardware and would be either
flaky or so loose they're meaningless). Instead, each scenario
records its measurements and asserts only structural invariants
(metrics object exists, audio actually played, etc.). The report
generator reads the dumped JSON and visualizes the comparisons.

Each scenario uses STUB LLM and STUB TTS — same as the
integration suite. The point is to measure how the *pipeline
overhead* scales, not the underlying neural net latency. Real-LLM
/ real-TTS perf testing belongs in a separate "live" suite that
isn't in scope here.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_loop import ChatLoop  # noqa: E402
from examples._chat_recording import CHUNK, RATE  # noqa: E402
from examples.virtual_audio import (  # noqa: E402
    VirtualMicStream,
    VirtualSpeakerStream,
    concat,
    make_silence,
    make_tone_burst,
)


PERF_OUT = ROOT / "iter-reports" / "perf-results.json"
# iter-039: also save a per-iteration snapshot so the perf page can
# render time-series across iterations. Iteration number is
# discovered from the most recent ``## iter-NNN —`` heading in
# ITERATION_LOG.md (the same source the report generator reads),
# falling back to the git commit count if the log can't be parsed.
LOG_PATH = ROOT / "ITERATION_LOG.md"


def _resolve_iter_number() -> str:
    """Return the most recent iter-NNN from ITERATION_LOG.md, or
    fall back to a 3-digit commit count if parsing fails.

    Pulled out as a free function so it's testable in isolation.
    """
    if LOG_PATH.exists():
        # Walk lines from the END so we find the most recent header
        # without scanning the whole file.
        lines = LOG_PATH.read_text().splitlines()
        import re as _re
        pat = _re.compile(r"^## iter-(\d{3}) —")
        for line in reversed(lines):
            m = pat.match(line)
            if m:
                return m.group(1)
    # Fallback: use the commit count. Won't collide with a real
    # iter-NNN since no iter has had >999 commits.
    try:
        import subprocess as _sp
        n = _sp.check_output(
            ["git", "rev-list", "--count", "HEAD"], cwd=ROOT, text=True,
        ).strip()
        return n.zfill(3)
    except Exception:
        return "000"


# ---- Scenario plumbing ------------------------------------------------------


@dataclass
class ScenarioResult:
    """One scenario's measurements."""
    name: str
    description: str
    ttfs_ms: float = 0.0          # time from speech end to first audio
    # iter-053: naturalness bucket — "rushed" / "natural" / "slow" / "".
    naturalness_bucket: str = ""
    stt_ms: float = 0.0
    # iter-049: STT real-time factor.
    stt_rtf: float = 0.0
    # iter-072: STT preview-vs-final divergence (0..1).
    stt_preview_divergence: float = 0.0
    tts_ms: float = 0.0           # cumulative synth time
    # iter-050: TTS real-time factor.
    tts_rtf: float = 0.0
    playback_ms: float = 0.0      # cumulative speaker write time
    llm_first_token_ms: float = 0.0
    # iter-052: LLM tokens-per-second.
    llm_tps: float = 0.0
    # iter-038: time from LLM start to first complete sentence.
    llm_first_sentence_ms: float = 0.0
    llm_total_ms: float = 0.0
    speech_duration_ms: float = 0.0
    sentences_spoken: int = 0
    # iter-040: sentences cut mid-stream by cancel_event.
    sentences_cancelled: int = 0
    # iter-043: streaming overlap ratio (0.0–1.0).
    streaming_overlap_ratio: float = 0.0
    # iter-073: first-sentence overlap savings, ms.
    first_synth_overlap_ms: float = 0.0
    # iter-074: bargeable-time fraction (0..1).
    bargeable_fraction: float = 0.0
    # iter-076: TTFS attribution residual (synth + dispatch), ms.
    synth_dispatch_ms: float = 0.0
    # iter-077: approximate context-token count sent to the LLM.
    context_tokens: int = 0
    # iter-080: pre-empted words on barge turns.
    preempted_words: int = 0
    # iter-081: id() of the filler clip picked this turn.
    last_filler_id: int = 0
    # iter-082: TTC (cross-turn) — ms.
    time_to_comprehension_ms: float = 0.0
    # iter-083: first-token-to-audio gap, ms.
    first_token_to_audio_ms: float = 0.0
    # iter-085: max LLM inter-token gap, ms.
    max_token_gap_ms: float = 0.0
    # iter-044: cumulative between-sentence worker idle gap.
    worker_idle_gap_ms: float = 0.0
    # iter-045: mean character length of sentences submitted.
    mean_sentence_chars: float = 0.0
    # iter-070: per-turn min/max sentence lengths.
    min_sentence_chars: int = 0
    max_sentence_chars: int = 0
    # iter-071: token-reveal lag, ms.
    mean_token_reveal_lag_ms: float = 0.0
    max_token_reveal_lag_ms: float = 0.0
    # iter-059: sentence-split coverage (0.0–1.0).
    sentence_split_coverage: float = 0.0
    # iter-046: bot speaking rate (words per minute).
    bot_wpm: float = 0.0
    wall_ms: float = 0.0          # full run_one_turn wall-clock
    barge_in: bool = False
    # iter-041: time from barge-in detect to playback halt. 0 if
    # no barge-in fired this scenario.
    barge_in_latency_ms: float = 0.0
    # iter-060: time from coord.trigger() to llm_gen.close() returning.
    llm_cancel_to_close_ms: float = 0.0
    # iter-047: barge-in phase ("llm_stream", "playback", or "").
    barge_in_phase: str = ""
    # iter-057: primed-frames replay seconds carried into next turn.
    primed_frames_seconds: float = 0.0
    # iter-037: count of mic frames flushed at start of turn.
    mic_stale_frames: int = 0
    # iter-061: time inside speaker_factory() in the worker.
    speaker_open_ms: float = 0.0
    # iter-062: peak SentenceWorker queue depth observed.
    max_queue_depth: int = 0
    # iter-063: EoT detection latency (last-speech → DONE_OK), ms.
    eot_latency_ms: float = 0.0
    # iter-064: user speaking rate (words per minute).
    user_wpm: float = 0.0
    # iter-065: EoT overhead beyond silence_duration, ms.
    eot_overhead_ms: float = 0.0


_RESULTS: list[ScenarioResult] = []


def _record(result: ScenarioResult) -> None:
    """Append a result and re-write the JSON file. Re-writing on
    every record means partial runs still leave usable output if a
    later scenario crashes.

    iter-039: ALSO write a per-iteration snapshot to
    ``perf-iter-NNN.json``. The latest-snapshot file remains for
    backwards compat with iter-036 charts; the per-iter files are
    what the time-series view reads. Both files are kept in sync
    on every record, so a partial run still leaves usable history.
    """
    _RESULTS.append(result)
    PERF_OUT.parent.mkdir(parents=True, exist_ok=True)
    iter_num = _resolve_iter_number()
    payload = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "iteration": iter_num,
        "scenarios": [asdict(r) for r in _RESULTS],
    }
    PERF_OUT.write_text(json.dumps(payload, indent=2))
    # Per-iter file. Same payload — keeps the schema parallel.
    per_iter = PERF_OUT.parent / f"perf-iter-{iter_num}.json"
    per_iter.write_text(json.dumps(payload, indent=2))


@pytest.fixture(scope="session", autouse=True)
def _reset_results():
    """Clear the in-memory list at the start of a session so a fresh
    run doesn't accumulate stale data.
    """
    _RESULTS.clear()
    yield


# ---- Stubs (same shape as integration suite) --------------------------------


def _stt_engine(transcript="hello world", stt_delay=0.0):
    """STT stub. ``stt_delay`` simulates transcription latency by
    sleeping inside transcribe_fn; the recorder times the call and
    rolls it into ``stt_time``.
    """
    engine = SimpleNamespace(_last_text=None, model_repo="stub")

    def transcribe(wav):
        if not wav:
            return None
        if stt_delay > 0:
            time.sleep(stt_delay)
        return transcript

    return engine, transcribe


def _const_synth(samples=2048, synth_delay=0.0):
    """TTS stub. ``synth_delay`` simulates kokoro-shaped synthesis
    latency. Default samples ≈ 85ms at 24kHz.
    """
    def synth(sentence):
        if synth_delay > 0:
            time.sleep(synth_delay)
        return np.full(samples, 0.5, dtype=np.float32), []
    return synth


def _slow_play(speaker, audio, tokens, *, is_first_sentence=False, cancel_event=None):
    audio_int16 = (audio * 32767).astype(np.int16)
    chunk = 256
    written = 0
    while written < len(audio_int16):
        if cancel_event is not None and cancel_event.is_set():
            break
        end = min(written + chunk, len(audio_int16))
        speaker.write(audio_int16[written:end].tobytes())
        written = end
        time.sleep(0.005)
    return 0.0


def _yield_tokens(
    text,
    *,
    per_token_delay=0.0,
    stall_after=None,
    stall_seconds=0.0,
    context_factor=0.0,
):
    """Build a streaming-LLM stub.

    ``stall_after`` (optional): substring; the simulated stream
    pauses for ``stall_seconds`` AFTER yielding the first token
    that contains this substring. Used by iter-100 to trigger the
    iter-093 auto-aggressive-on-stall logic at a deterministic
    point in the response.

    ``context_factor`` (iter-112): seconds-per-input-character
    delay applied BEFORE the first token yields. Simulates real
    LLM TTFB scaling with input context size — production LLMs'
    first-token latency grows with prompt length (KV-cache fill
    dominates). 0.0 (default) = no scaling, matching pre-iter-112
    behavior. A factor of 0.005 ≈ 5ms/char, roughly tracking
    what production APIs report on small-context turns.
    """
    import re

    def factory(messages, config):
        if context_factor > 0:
            total_chars = sum(
                len(str(m.get("content", "")))
                for m in messages
            )
            time.sleep(total_chars * context_factor)
        parts = re.findall(r"\S+|\.|!|\?", text)
        for p in parts:
            if per_token_delay > 0:
                time.sleep(per_token_delay)
            yield p + " "
            if stall_after is not None and stall_after in p and stall_seconds > 0:
                time.sleep(stall_seconds)

    return factory


def _utterance(seconds: float, mic: VirtualMicStream) -> None:
    """Push a tone-burst utterance with the leading + trailing
    silence the VAD needs to fire DONE_OK.
    """
    mic.push(concat(
        make_silence(0.3, rate=RATE),
        make_tone_burst(seconds, rate=RATE, amp=0.3),
        make_silence(1.5, rate=RATE),
    ))


def _run_scenario(
    name: str,
    description: str,
    *,
    speech_seconds: float = 1.0,
    stt_delay: float = 0.0,
    synth_delay: float = 0.0,
    response: str = "Got it.",
    per_token_delay: float = 0.0,
    fillers: list | None = None,
    idle_threshold: float = 0.0,
    aggressive_first_sentence: bool = False,
    auto_aggressive_threshold: float = 0.0,
    stall_after: str | None = None,
    stall_seconds: float = 0.0,
) -> ScenarioResult:
    mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
    _utterance(speech_seconds, mic)
    engine, transcribe = _stt_engine(transcript="benchmark", stt_delay=stt_delay)

    loop = ChatLoop(
        mic=mic,
        speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
        stt_engine=engine,
        transcribe_fn=transcribe,
        llm_stream_fn=_yield_tokens(
            response,
            per_token_delay=per_token_delay,
            stall_after=stall_after,
            stall_seconds=stall_seconds,
        ),
        llm_config={"model": "stub"},
        synth_fn=_const_synth(synth_delay=synth_delay),
        play_fn=_slow_play,
        fillers=fillers,
        idle_threshold=idle_threshold,
        aggressive_first_sentence=aggressive_first_sentence,
        auto_aggressive_threshold=auto_aggressive_threshold,
    )

    t0 = time.monotonic()
    result = loop.run_one_turn([])
    wall = time.monotonic() - t0

    assert result.metrics is not None, f"{name}: scenario produced no metrics"
    m = result.metrics
    res = ScenarioResult(
        name=name,
        description=description,
        ttfs_ms=m.ttfs * 1000,
        naturalness_bucket=m.naturalness_bucket,
        stt_ms=m.stt_time * 1000,
        stt_rtf=m.stt_rtf,
        stt_preview_divergence=m.stt_preview_divergence,
        tts_rtf=m.tts_rtf,
        tts_ms=m.tts_time * 1000,
        playback_ms=m.playback_time * 1000,
        llm_first_token_ms=m.llm_first_token * 1000,
        llm_tps=m.llm_tps,
        # iter-042: also capture iter-038 + iter-040 + iter-041 +
        # iter-037 metrics on the perf-snapshot row so the
        # time-series charts can pick them up later.
        llm_first_sentence_ms=m.llm_first_sentence * 1000,
        llm_total_ms=m.llm_total * 1000,
        speech_duration_ms=m.speech_duration * 1000,
        sentences_spoken=m.sentences_spoken,
        sentences_cancelled=m.sentences_cancelled,
        streaming_overlap_ratio=m.streaming_overlap_ratio,
        first_synth_overlap_ms=m.first_synth_overlap_seconds * 1000,
        bargeable_fraction=m.bargeable_fraction,
        synth_dispatch_ms=m.synth_dispatch_seconds * 1000,
        context_tokens=m.context_tokens,
        preempted_words=m.preempted_words,
        last_filler_id=m.last_filler_id,
        time_to_comprehension_ms=m.time_to_comprehension * 1000,
        first_token_to_audio_ms=m.first_token_to_audio * 1000,
        max_token_gap_ms=m.max_token_gap * 1000,
        worker_idle_gap_ms=m.worker_idle_gap_total * 1000,
        mean_sentence_chars=m.mean_sentence_chars,
        sentence_split_coverage=m.sentence_split_coverage,
        bot_wpm=m.bot_wpm,
        wall_ms=wall * 1000,
        barge_in=m.barge_in,
        barge_in_latency_ms=m.barge_in_latency * 1000,
        llm_cancel_to_close_ms=m.llm_cancel_to_close * 1000,
        barge_in_phase=m.barge_in_phase,
        primed_frames_seconds=m.primed_frames_seconds,
        mic_stale_frames=m.mic_stale_frames,
        speaker_open_ms=m.speaker_open_seconds * 1000,
        max_queue_depth=m.max_queue_depth,
        eot_latency_ms=m.eot_latency * 1000,
        user_wpm=m.user_wpm,
        eot_overhead_ms=m.eot_overhead * 1000,
        min_sentence_chars=m.min_sentence_chars,
        max_sentence_chars=m.max_sentence_chars,
        mean_token_reveal_lag_ms=m.mean_token_reveal_lag * 1000,
        max_token_reveal_lag_ms=m.max_token_reveal_lag * 1000,
    )
    _record(res)
    return res


def _run_session_scenario(
    name: str,
    description: str,
    *,
    n_turns: int,
    max_user_assistant: int,
    response_per_turn: str = "Alright, that makes sense to me.",
    speech_seconds: float = 1.0,
    context_factor: float = 0.0,
) -> tuple[ScenarioResult, int]:
    """Run a multi-turn dialog and record the LAST turn's metrics.

    Returns ``(scenario_result, trim_events_total)`` so the caller
    can assert on session-level signals that don't fit into a
    single TurnMetrics row. iter-102 uses this to A/B the
    context-cap (max_user_assistant=5 vs 20) — context_tokens on
    the final turn is the dominant signal, but trim_events
    reflects how often the cap actually fired.
    """
    import threading as _th

    mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
    # Push utterance 1 up-front so turn 1's recorder has audio.
    # Subsequent utterances are pushed by a feeder thread that
    # waits for the previous turn's flush_pending_audio (run at
    # the start of each turn) to complete — same trick as the
    # iter-042 barge-in scenario. Without the delay, flush would
    # drain the pre-loaded buffer and the recorder would wait
    # forever for tone.
    _utterance(speech_seconds, mic)
    pending = _th.Semaphore(0)

    def _feeder():
        for _ in range(n_turns - 1):
            pending.acquire()  # blocks until next turn signals ready
            time.sleep(0.05)   # land after flush_pending_audio
            _utterance(speech_seconds, mic)

    feeder = _th.Thread(target=_feeder, daemon=True)
    feeder.start()

    engine, transcribe = _stt_engine(transcript="benchmark")
    loop = ChatLoop(
        mic=mic,
        speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
        stt_engine=engine,
        transcribe_fn=transcribe,
        llm_stream_fn=_yield_tokens(
            response_per_turn,
            per_token_delay=0.0,
            context_factor=context_factor,
        ),
        llm_config={"model": "stub"},
        synth_fn=_const_synth(),
        play_fn=_slow_play,
    )

    messages: list[dict] = [{"role": "system", "content": "You are concise."}]
    trim_events = 0
    last_metrics = None
    last_wall = 0.0
    for turn_idx in range(n_turns):
        if turn_idx > 0:
            # Tell the feeder to push the next utterance now.
            pending.release()
        t0 = time.monotonic()
        result = loop.run_one_turn(messages)
        last_wall = time.monotonic() - t0
        if result.metrics is None:
            continue
        last_metrics = result.metrics
        len_before = len(messages)
        messages = ChatLoop.trim_messages(
            messages, max_user_assistant=max_user_assistant,
        )
        if len_before - len(messages) > 0:
            trim_events += 1

    assert last_metrics is not None, f"{name}: no turns produced metrics"
    m = last_metrics
    res = ScenarioResult(
        name=name,
        description=description,
        ttfs_ms=m.ttfs * 1000,
        naturalness_bucket=m.naturalness_bucket,
        stt_ms=m.stt_time * 1000,
        stt_rtf=m.stt_rtf,
        stt_preview_divergence=m.stt_preview_divergence,
        tts_rtf=m.tts_rtf,
        tts_ms=m.tts_time * 1000,
        playback_ms=m.playback_time * 1000,
        llm_first_token_ms=m.llm_first_token * 1000,
        llm_tps=m.llm_tps,
        llm_first_sentence_ms=m.llm_first_sentence * 1000,
        llm_total_ms=m.llm_total * 1000,
        speech_duration_ms=m.speech_duration * 1000,
        sentences_spoken=m.sentences_spoken,
        sentences_cancelled=m.sentences_cancelled,
        streaming_overlap_ratio=m.streaming_overlap_ratio,
        first_synth_overlap_ms=m.first_synth_overlap_seconds * 1000,
        bargeable_fraction=m.bargeable_fraction,
        synth_dispatch_ms=m.synth_dispatch_seconds * 1000,
        context_tokens=m.context_tokens,
        preempted_words=m.preempted_words,
        last_filler_id=m.last_filler_id,
        time_to_comprehension_ms=m.time_to_comprehension * 1000,
        first_token_to_audio_ms=m.first_token_to_audio * 1000,
        max_token_gap_ms=m.max_token_gap * 1000,
        worker_idle_gap_ms=m.worker_idle_gap_total * 1000,
        mean_sentence_chars=m.mean_sentence_chars,
        sentence_split_coverage=m.sentence_split_coverage,
        bot_wpm=m.bot_wpm,
        wall_ms=last_wall * 1000,
        barge_in=m.barge_in,
        barge_in_latency_ms=m.barge_in_latency * 1000,
        llm_cancel_to_close_ms=m.llm_cancel_to_close * 1000,
        barge_in_phase=m.barge_in_phase,
        primed_frames_seconds=m.primed_frames_seconds,
        mic_stale_frames=m.mic_stale_frames,
        speaker_open_ms=m.speaker_open_seconds * 1000,
        max_queue_depth=m.max_queue_depth,
        eot_latency_ms=m.eot_latency * 1000,
        user_wpm=m.user_wpm,
        eot_overhead_ms=m.eot_overhead * 1000,
        min_sentence_chars=m.min_sentence_chars,
        max_sentence_chars=m.max_sentence_chars,
        mean_token_reveal_lag_ms=m.mean_token_reveal_lag * 1000,
        max_token_reveal_lag_ms=m.max_token_reveal_lag * 1000,
    )
    _record(res)
    return res, trim_events


# ---- Scenarios --------------------------------------------------------------


class TestPerfScenarios:
    """One method per scenario. Each records a row to perf-results.json."""

    def test_short_utterance_short_response(self):
        r = _run_scenario(
            "short_short",
            "1s utterance, 5-token response (best-case)",
            speech_seconds=1.0,
            response="Sure thing.",
        )
        # Sanity: full path completed.
        assert r.sentences_spoken >= 1
        assert r.ttfs_ms > 0

    def test_short_utterance_long_response(self):
        r = _run_scenario(
            "short_long",
            "1s utterance, 8-sentence response — exercises streaming overlap",
            speech_seconds=1.0,
            response=" ".join(f"sentence number {i}." for i in range(8)),
        )
        assert r.sentences_spoken >= 1

    def test_long_utterance_short_response(self):
        r = _run_scenario(
            "long_short",
            "3s utterance, short response — STT path-length scaling",
            speech_seconds=3.0,
            response="Quick reply.",
        )
        assert r.sentences_spoken >= 1

    def test_synth_latency_simulated(self):
        # Real kokoro synth runs ~50-100ms per sentence; simulate.
        r = _run_scenario(
            "tts_50ms_per_sentence",
            "TTS shaped like real kokoro (50ms / sentence) — TTFS impact",
            speech_seconds=1.0,
            response="Reply one. Reply two. Reply three.",
            synth_delay=0.05,
        )
        assert r.sentences_spoken >= 1

    def test_stt_latency_simulated(self):
        r = _run_scenario(
            "stt_100ms",
            "STT shaped like real whisper (100ms) — TTFS impact",
            speech_seconds=1.0,
            stt_delay=0.1,
            response="OK.",
        )
        assert r.sentences_spoken >= 1
        # STT delay should show up in stt_ms.
        assert r.stt_ms >= 100

    def test_slow_llm_first_token(self):
        # Real LLMs often have 200-500ms TTFT; simulate worst case.
        r = _run_scenario(
            "slow_llm_300ms",
            "LLM with 300ms first-token delay — dominant-cost case",
            speech_seconds=1.0,
            response="A delayed reply.",
            per_token_delay=0.05,
        )
        assert r.sentences_spoken >= 1

    def test_with_fillers(self):
        # Pre-rendered filler clip plays during LLM stall.
        filler = (np.full(2048, 0.3, dtype=np.float32), [])
        r = _run_scenario(
            "fillers_on",
            "Filler enabled, slow LLM — TTFS dominated by filler clip start",
            speech_seconds=1.0,
            response="The actual answer is here.",
            per_token_delay=0.1,  # slow so filler triggers
            fillers=[filler],
            idle_threshold=0.15,
        )
        assert r.sentences_spoken >= 1

    # iter-098: A/B for the aggressive first-sentence splitter
    # (iter-088). The same long-preamble response is run twice —
    # once with the splitter off, once on. The time-series chart
    # then shows the TTFS delta empirically rather than relying on
    # the synthetic helpers tests in tests/unit/test_chat_helpers.
    #
    # Long-preamble = a clause >20 chars with a comma BEFORE the
    # first period. Aggressive splitter slices on the comma; the
    # default splitter waits for the period.
    _LONG_PREAMBLE = (
        "Well let me think about this for a moment, "
        "the answer is twelve. "
        "And the reason is straightforward. "
        "It just is."
    )

    def test_long_preamble_aggressive_off(self):
        r = _run_scenario(
            "long_preamble_aggressive_off",
            "Long preamble, splitter off — TTFS waits for first period",
            speech_seconds=1.0,
            response=self._LONG_PREAMBLE,
            per_token_delay=0.01,
            aggressive_first_sentence=False,
        )
        assert r.sentences_spoken >= 1

    def test_long_preamble_aggressive_on(self):
        r = _run_scenario(
            "long_preamble_aggressive_on",
            "Long preamble, splitter on — slices on early comma (iter-088)",
            speech_seconds=1.0,
            response=self._LONG_PREAMBLE,
            per_token_delay=0.01,
            aggressive_first_sentence=True,
        )
        assert r.sentences_spoken >= 1

    # iter-099: A/B for the filler idle_threshold (iter-011 + iter-051).
    # Same slow-LLM setup driven twice — once with the operator-default
    # 0.6s threshold, once with the aggressive 0.15s. The time-series
    # chart then shows how much earlier the filler kicks in. Lower
    # threshold = earlier filler = lower TTFS-to-first-audio, at the
    # cost of more false-positive fillers on naturally fast turns
    # (the iter-051 sensitivity that iter-096 surfaces in the
    # session-summary recommendation).
    _FILLER_CLIP = (np.full(2048, 0.3, dtype=np.float32), [])

    def test_filler_threshold_default(self):
        r = _run_scenario(
            "filler_threshold_default",
            "Slow LLM, filler threshold 0.6s — operator default",
            speech_seconds=1.0,
            response="The actual answer is here.",
            per_token_delay=0.1,
            fillers=[self._FILLER_CLIP],
            idle_threshold=0.6,
        )
        assert r.sentences_spoken >= 1

    def test_filler_threshold_aggressive(self):
        r = _run_scenario(
            "filler_threshold_aggressive",
            "Slow LLM, filler threshold 0.15s — aggressive (iter-051)",
            speech_seconds=1.0,
            response="The actual answer is here.",
            per_token_delay=0.1,
            fillers=[self._FILLER_CLIP],
            idle_threshold=0.15,
        )
        assert r.sentences_spoken >= 1

    # iter-100: A/B for auto_aggressive_threshold (iter-093). The
    # same long-preamble response is streamed twice, with a 500ms
    # mid-stream stall right after the comma. With threshold=0.0,
    # the splitter stays strict and TTFS waits for the post-stall
    # period. With threshold=0.3, the stall trips the auto-flip;
    # the splitter goes aggressive on the next iteration and slices
    # at the comma already in the buffer — first audio out before
    # the stalled tokens finish arriving.
    _STALLED_PREAMBLE = (
        "Well let me think about this for a moment, "
        "the answer is twelve. "
        "And the reason is straightforward."
    )

    def test_auto_aggressive_off(self):
        r = _run_scenario(
            "auto_aggressive_off",
            "Mid-stream stall, auto-aggressive off — TTFS includes stall",
            speech_seconds=1.0,
            response=self._STALLED_PREAMBLE,
            per_token_delay=0.01,
            stall_after="moment,",
            stall_seconds=0.5,
            auto_aggressive_threshold=0.0,
        )
        assert r.sentences_spoken >= 1

    def test_auto_aggressive_on(self):
        r = _run_scenario(
            "auto_aggressive_on",
            "Mid-stream stall, auto-aggressive on (0.3s) — flip on stall",
            speech_seconds=1.0,
            response=self._STALLED_PREAMBLE,
            per_token_delay=0.01,
            stall_after="moment,",
            stall_seconds=0.5,
            auto_aggressive_threshold=0.3,
        )
        assert r.sentences_spoken >= 1

    # iter-101: extends the iter-100 pair into a 3-point grid by
    # varying per_token_delay — the dominant factor in how much
    # auto-aggressive saves. iter-100 only used 0.01 (artificially
    # fast); iter-052's "real LLM TPS" baseline puts production
    # somewhere in the 0.05-0.10s range. By pairing each delay
    # value (off vs on), the time-series chart now shows the
    # savings curve, not just a single delta.
    def test_auto_aggressive_off_50ms(self):
        r = _run_scenario(
            "auto_aggressive_off_50ms",
            "Mid-stream stall, off, 50ms/token — production-like TPS",
            speech_seconds=1.0,
            response=self._STALLED_PREAMBLE,
            per_token_delay=0.05,
            stall_after="moment,",
            stall_seconds=0.5,
            auto_aggressive_threshold=0.0,
        )
        assert r.sentences_spoken >= 1

    def test_auto_aggressive_on_50ms(self):
        r = _run_scenario(
            "auto_aggressive_on_50ms",
            "Mid-stream stall, on, 50ms/token — production-like TPS",
            speech_seconds=1.0,
            response=self._STALLED_PREAMBLE,
            per_token_delay=0.05,
            stall_after="moment,",
            stall_seconds=0.5,
            auto_aggressive_threshold=0.3,
        )
        assert r.sentences_spoken >= 1

    def test_auto_aggressive_off_100ms(self):
        r = _run_scenario(
            "auto_aggressive_off_100ms",
            "Mid-stream stall, off, 100ms/token — slow LLM",
            speech_seconds=1.0,
            response=self._STALLED_PREAMBLE,
            per_token_delay=0.1,
            stall_after="moment,",
            stall_seconds=0.5,
            auto_aggressive_threshold=0.0,
        )
        assert r.sentences_spoken >= 1

    def test_auto_aggressive_on_100ms(self):
        r = _run_scenario(
            "auto_aggressive_on_100ms",
            "Mid-stream stall, on, 100ms/token — slow LLM",
            speech_seconds=1.0,
            response=self._STALLED_PREAMBLE,
            per_token_delay=0.1,
            stall_after="moment,",
            stall_seconds=0.5,
            auto_aggressive_threshold=0.3,
        )
        assert r.sentences_spoken >= 1

    # iter-102: A/B for max_user_assistant context cap. The default
    # cap (20) is loose enough that an 8-turn session never trims;
    # a tight cap (5) starts evicting after turn 3, so the late-
    # session context_tokens is bounded. This is the first
    # pair-scenario that needs MULTI-TURN state — the cap only
    # matters AFTER history has accumulated. Single-turn perf
    # scenarios can't see this.
    _SESSION_TURNS = 8

    def test_context_cap_default(self):
        r, trim_events = _run_session_scenario(
            "context_cap_default",
            f"{self._SESSION_TURNS}-turn session, cap=20 (default — never trims)",
            n_turns=self._SESSION_TURNS,
            max_user_assistant=20,
        )
        assert r.sentences_spoken >= 1
        assert trim_events == 0, f"expected no trims at cap=20, got {trim_events}"

    def test_context_cap_tight(self):
        r, trim_events = _run_session_scenario(
            "context_cap_tight",
            f"{self._SESSION_TURNS}-turn session, cap=5 (tight — trims after turn 3)",
            n_turns=self._SESSION_TURNS,
            max_user_assistant=5,
        )
        assert r.sentences_spoken >= 1
        # Cap=5 means we keep the last 5 user+assistant messages.
        # 8 turns produce 16 such messages → at least one trim event.
        assert trim_events >= 1

    # iter-112: paired session scenarios that exercise the new
    # context_factor knob in _yield_tokens. With the stub LLM
    # delaying its first token proportionally to total input
    # chars, the cap=5 turn now BENEFITS visibly — trim keeps the
    # message list shorter, so first-token latency stays low. The
    # cap=20 turn pays the full context cost. iter-102 showed the
    # token-billing delta (-55%); this row shows the latency delta.
    # 2ms/char is a wall-time compromise: 5ms/char (production-
    # realistic KV-fill cost) made the perf suite 12s longer per
    # iteration; 2ms/char keeps the savings ratio identical and
    # cuts the extra wall time to ~5s.
    _CONTEXT_FACTOR_GRID = 0.002

    def test_context_cap_default_ctx2ms(self):
        r, trim_events = _run_session_scenario(
            "context_cap_default_ctx2ms",
            (
                f"{self._SESSION_TURNS}-turn session, cap=20 + context_factor=2ms/char "
                f"— full LLM TTFB cost on every turn"
            ),
            n_turns=self._SESSION_TURNS,
            max_user_assistant=20,
            context_factor=self._CONTEXT_FACTOR_GRID,
        )
        assert r.sentences_spoken >= 1
        assert trim_events == 0

    def test_context_cap_tight_ctx2ms(self):
        r, trim_events = _run_session_scenario(
            "context_cap_tight_ctx2ms",
            (
                f"{self._SESSION_TURNS}-turn session, cap=5 + context_factor=2ms/char "
                f"— trim bounds the LLM TTFB cost"
            ),
            n_turns=self._SESSION_TURNS,
            max_user_assistant=5,
            context_factor=self._CONTEXT_FACTOR_GRID,
        )
        assert r.sentences_spoken >= 1
        assert trim_events >= 1

    def test_barge_in_during_playback(self):
        # iter-042: deterministic barge-in scenario.
        #
        # The naive "push barge audio in the mic up front" approach
        # fails because iter-002's flush_pending_audio drains the mic
        # before phase 2 starts, eating the barge audio. We work
        # around it by pushing the barge from a thread that fires
        # AFTER the flush — by which time the watcher is active and
        # picks up the audio reliably.
        import threading as _th

        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        # Initial utterance — recorder consumes this.
        _utterance(1.0, mic)

        def _delayed_barge():
            # 50ms gives flush_pending_audio time to run + watcher
            # to start. Then push 0.6s of tone — well above the
            # min_speech_duration window, plenty for VAD ACTIVE.
            time.sleep(0.05)
            mic.push(concat(
                make_silence(0.05, rate=RATE),
                make_tone_burst(0.6, rate=RATE, amp=0.4),
                make_silence(0.5, rate=RATE),
            ))

        engine, transcribe = _stt_engine(transcript="hi")
        # Long bot response so playback runs long enough for the
        # delayed-push thread's burst to land mid-stream.
        long_response = " ".join(f"sentence {i}." for i in range(8))

        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=_yield_tokens(long_response, per_token_delay=0.015),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(samples=2048),
            play_fn=_slow_play,
        )

        _th.Thread(target=_delayed_barge, daemon=True).start()
        t0 = time.monotonic()
        result = loop.run_one_turn([])
        wall = time.monotonic() - t0

        assert result.metrics is not None
        m = result.metrics
        # Record the row even if the barge didn't trigger this run —
        # the perf charts can still show "0 barge_in_latency_ms" as
        # honest data. But mark the description so the operator
        # knows whether this run actually exercised the barge path.
        landed = m.barge_in
        res = ScenarioResult(
            name="barge_in",
            description=(
                "User barges in mid-playback (8-sentence response, "
                f"barge {'landed' if landed else 'did NOT land — timing-flaky'})"
            ),
            ttfs_ms=m.ttfs * 1000,
        naturalness_bucket=m.naturalness_bucket,
            stt_ms=m.stt_time * 1000,
        stt_rtf=m.stt_rtf,
        stt_preview_divergence=m.stt_preview_divergence,
        tts_rtf=m.tts_rtf,
            tts_ms=m.tts_time * 1000,
            playback_ms=m.playback_time * 1000,
            llm_first_token_ms=m.llm_first_token * 1000,
        llm_tps=m.llm_tps,
            llm_first_sentence_ms=m.llm_first_sentence * 1000,
            llm_total_ms=m.llm_total * 1000,
            speech_duration_ms=m.speech_duration * 1000,
            sentences_spoken=m.sentences_spoken,
            sentences_cancelled=m.sentences_cancelled,
        streaming_overlap_ratio=m.streaming_overlap_ratio,
        first_synth_overlap_ms=m.first_synth_overlap_seconds * 1000,
        bargeable_fraction=m.bargeable_fraction,
        synth_dispatch_ms=m.synth_dispatch_seconds * 1000,
        context_tokens=m.context_tokens,
        preempted_words=m.preempted_words,
        last_filler_id=m.last_filler_id,
        time_to_comprehension_ms=m.time_to_comprehension * 1000,
        first_token_to_audio_ms=m.first_token_to_audio * 1000,
        max_token_gap_ms=m.max_token_gap * 1000,
        worker_idle_gap_ms=m.worker_idle_gap_total * 1000,
        mean_sentence_chars=m.mean_sentence_chars,
        sentence_split_coverage=m.sentence_split_coverage,
        bot_wpm=m.bot_wpm,
            wall_ms=wall * 1000,
            barge_in=m.barge_in,
            barge_in_latency_ms=m.barge_in_latency * 1000,
        llm_cancel_to_close_ms=m.llm_cancel_to_close * 1000,
        barge_in_phase=m.barge_in_phase,
        primed_frames_seconds=m.primed_frames_seconds,
            mic_stale_frames=m.mic_stale_frames,
            speaker_open_ms=m.speaker_open_seconds * 1000,
            max_queue_depth=m.max_queue_depth,
            eot_latency_ms=m.eot_latency * 1000,
            user_wpm=m.user_wpm,
            eot_overhead_ms=m.eot_overhead * 1000,
            min_sentence_chars=m.min_sentence_chars,
            max_sentence_chars=m.max_sentence_chars,
            mean_token_reveal_lag_ms=m.mean_token_reveal_lag * 1000,
            max_token_reveal_lag_ms=m.max_token_reveal_lag * 1000,
        )
        _record(res)


# ---- Final emit -------------------------------------------------------------


def test_results_written():
    """Sanity test that runs LAST (alphabetical) and confirms the
    JSON file exists with a sensible payload. If any scenario above
    crashed we still want this to make assertions about what DID
    land.
    """
    assert PERF_OUT.exists(), f"perf-results.json was never written"
    payload = json.loads(PERF_OUT.read_text())
    assert "scenarios" in payload
    assert len(payload["scenarios"]) >= 1
    for s in payload["scenarios"]:
        # Each row has expected fields (canary against regressions
        # in the schema the report generator depends on).
        for key in (
            "name", "description", "ttfs_ms", "stt_ms", "tts_ms",
            "playback_ms", "llm_first_token_ms", "llm_total_ms",
            "wall_ms", "sentences_spoken",
        ):
            assert key in s, f"missing {key!r} in scenario {s.get('name')}"
