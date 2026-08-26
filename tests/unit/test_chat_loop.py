"""End-to-end ChatLoop tests.

This is the iter-009 / iter-010 / iter-012 dream: drive the actual
production per-turn code path with stub STT, stub LLM, and virtual
audio. No more "approximate the structure of run_chat" helper
functions — these tests run the real ChatLoop.run_one_turn.

Stubs we provide:
  - STT: a SimpleNamespace whose `_last_text` is set by the
    transcribe_fn we inject.
  - LLM stream: a generator function that yields synthetic tokens
    with an optional per-token delay (so a watcher thread has time
    to fire mid-stream for barge-in tests).
  - synth_fn: returns a constant audio for any text — lets us
    measure where audio went without depending on kokoro.
  - play_fn: writes to a VirtualSpeakerStream and respects
    cancel_event so barge-in tests can verify mid-stream stop.
  - mic / speaker: VirtualMicStream + VirtualSpeakerStream from
    iter-005.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_loop import ChatLoop, TurnResult  # noqa: E402
from examples._chat_recording import CHUNK, RATE  # noqa: E402
from examples.virtual_audio import (  # noqa: E402
    VirtualMicStream,
    VirtualSpeakerStream,
    concat,
    make_silence,
    make_tone_burst,
)


# ---- Helpers / doubles -------------------------------------------------------


def _stt_engine_stub(transcript: str = "hello"):
    """Returns (engine, transcribe_fn). The transcribe_fn always
    returns the canned transcript for any wav, and also
    sets engine._last_text — same protocol the real
    record_utterance_streaming uses.
    """
    engine = SimpleNamespace(_last_text=None, model_repo="stub")

    def transcribe(wav_bytes: bytes):
        if not wav_bytes:
            return None
        return transcript

    return engine, transcribe


def _llm_stream_yielding(text: str, *, per_token_delay: float = 0.0):
    """Build an llm_stream-shaped function that yields the given
    text as space-delimited tokens.
    """
    import re

    def factory(messages, config):
        # OpenAI-style splitting: words + punctuation as separate
        # tokens, each followed by a space (so the SENTENCE_END
        # regex finds boundaries).
        parts = re.findall(r"\S+|\.|!|\?", text)
        for p in parts:
            if per_token_delay > 0:
                time.sleep(per_token_delay)
            yield p + " "

    return factory


def _llm_stream_raising():
    """An llm_stream-shaped function that raises mid-stream after
    yielding one token. Used to test the error path.
    """
    def factory(messages, config):
        yield "First "
        raise RuntimeError("simulated LLM failure")

    return factory


def _const_synth(samples: int = 2048):
    """synth_fn that returns the same audio shape regardless of text."""
    def synth(text: str):
        return np.full(samples, 0.5, dtype=np.float32), []
    return synth


def _slow_play(speaker, audio_np, tokens, *, is_first_sentence=False, cancel_event=None):
    """play_fn that respects cancel_event between chunks. Sleeps
    briefly per chunk so a watcher thread has time to fire.
    """
    audio_int16 = (audio_np * 32767).astype(np.int16)
    chunk = 256
    written = 0
    t0 = time.monotonic()
    while written < len(audio_int16):
        if cancel_event is not None and cancel_event.is_set():
            break
        end = min(written + chunk, len(audio_int16))
        speaker.write(audio_int16[written:end].tobytes())
        written = end
        time.sleep(0.005)
    return time.monotonic() - t0


def _instant_play(speaker, audio_np, tokens, *, is_first_sentence=False, cancel_event=None):
    """play_fn that writes audio in one shot. Faster for happy-path tests."""
    speaker.write((audio_np * 32767).astype(np.int16).tobytes())
    return len(audio_np) / 24000.0


def _utterance_audio() -> np.ndarray:
    """Standard fixture: 0.3s lead silence + 1.0s tone + 1.2s
    trailing silence. Long enough to fire DONE_OK.
    """
    return concat(
        make_silence(0.3, rate=RATE),
        make_tone_burst(1.0, rate=RATE, amp=0.3),
        make_silence(1.2, rate=RATE),
    )


def _make_chat_loop(
    *,
    mic: VirtualMicStream,
    speaker_factory,
    transcript: str = "hello",
    llm_text: str = "Hi there. How are you?",
    llm_per_token_delay: float = 0.0,
    llm_stream_fn=None,
    play_fn=_instant_play,
    fillers=None,
    idle_threshold: float = 0.0,
    idle_timeout=None,
    barge_in_enabled: bool = True,
):
    engine, transcribe = _stt_engine_stub(transcript=transcript)
    if llm_stream_fn is None:
        llm_stream_fn = _llm_stream_yielding(llm_text, per_token_delay=llm_per_token_delay)
    return ChatLoop(
        mic=mic,
        speaker_factory=speaker_factory,
        rate=RATE,
        chunk=CHUNK,
        stt_engine=engine,
        transcribe_fn=transcribe,
        llm_stream_fn=llm_stream_fn,
        llm_config={"model": "stub-model"},
        synth_fn=_const_synth(samples=1024),
        play_fn=play_fn,
        fillers=fillers,
        idle_threshold=idle_threshold,
        idle_timeout=idle_timeout,
        barge_in_enabled=barge_in_enabled,
    )


# ---- Tests ------------------------------------------------------------------


class TestNoTranscription:
    def test_too_short_utterance_returns_none_metrics(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        # 0.05s tone is below the 0.3s min_speech_duration → DONE_TOO_SHORT.
        mic.push(concat(
            make_silence(0.3, rate=RATE),
            make_tone_burst(0.05, rate=RATE, amp=0.3),
            make_silence(1.5, rate=RATE),
        ))
        spk_holder = {"spk": None}

        def factory():
            spk_holder["spk"] = VirtualSpeakerStream(rate=24000)
            return spk_holder["spk"]

        loop = _make_chat_loop(
            mic=mic, speaker_factory=factory, transcript="x"  # short transcript
        )
        messages = [{"role": "system", "content": "be nice"}]
        result = loop.run_one_turn(messages)

        assert result.metrics is None
        assert result.next_primed_frames is None
        assert result.had_error is False
        # No user message added since recording produced empty wav.
        assert messages == [{"role": "system", "content": "be nice"}]
        # Speaker was never opened (worker never started).
        assert spk_holder["spk"] is None

    def test_empty_transcription_returns_none_metrics(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(_utterance_audio())
        # Stub STT that returns empty text — the "(no transcription)"
        # path inside ChatLoop should fire and we get None metrics.
        engine = SimpleNamespace(_last_text=None, model_repo="stub")
        loop = ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=lambda w: "",  # always empty
            llm_stream_fn=_llm_stream_yielding("never reached"),
            llm_config={"model": "stub-model"},
            synth_fn=_const_synth(),
            play_fn=_instant_play,
        )
        result = loop.run_one_turn([{"role": "system", "content": "x"}])
        assert result.metrics is None


class TestNormalTurn:
    def test_full_turn_produces_metrics_and_history(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(_utterance_audio())
        spk_holder = {"spk": None}

        def factory():
            spk_holder["spk"] = VirtualSpeakerStream(rate=24000)
            return spk_holder["spk"]

        loop = _make_chat_loop(
            mic=mic,
            speaker_factory=factory,
            transcript="how are you",
            llm_text="I am well. Thanks for asking.",
        )
        messages = [{"role": "system", "content": "be nice"}]
        result = loop.run_one_turn(messages)

        assert result.metrics is not None
        assert result.next_primed_frames is None
        assert result.had_error is False
        # Metrics populated.
        m = result.metrics
        assert m.transcript == "how are you"
        assert m.response.startswith("I am well")
        assert "Thanks for asking" in m.response
        assert m.sentences_spoken >= 1
        assert m.barge_in is False
        # llm_total > 0 (we did stream).
        assert m.llm_total > 0
        # History updated: user + assistant appended.
        assert len(messages) == 3
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "how are you"
        assert messages[2]["role"] == "assistant"
        # Speaker received audio.
        assert len(spk_holder["spk"].captured) > 0

    def test_metrics_count_sentences_correctly(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(_utterance_audio())

        loop = _make_chat_loop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            transcript="hi",
            llm_text="One. Two. Three.",
        )
        messages = []
        result = loop.run_one_turn(messages)
        assert result.metrics is not None
        # Three sentence-terminators → at least three sentences played.
        assert result.metrics.sentences_spoken >= 3


class TestBargeInDuringLlmStream:
    def test_user_speech_during_stream_cancels_and_returns_primed(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        # First, the recorded utterance.
        mic.push(_utterance_audio())

        # Schedule the user's barge-in audio to land mid-stream.
        def push_barge_in():
            time.sleep(0.05)
            mic.push(make_tone_burst(0.5, rate=RATE, amp=0.3))

        threading.Thread(target=push_barge_in, daemon=True).start()

        spk_holder = {"spk": None}

        def factory():
            spk_holder["spk"] = VirtualSpeakerStream(rate=24000)
            return spk_holder["spk"]

        loop = _make_chat_loop(
            mic=mic,
            speaker_factory=factory,
            transcript="initial",
            llm_text="One sentence. Two sentence. Three sentence. Four sentence.",
            llm_per_token_delay=0.02,  # slow enough for the barge-in to fire
            play_fn=_slow_play,  # respects cancel_event
        )
        messages = []
        result = loop.run_one_turn(messages)

        # We expect SOME of these to be true; the exact behavior
        # depends on race timing. The robust assertions are:
        # - had_error stays False (LLM didn't raise)
        # - metrics may exist with barge_in=True OR be None (if
        #   the watcher fired before any token arrived)
        assert result.had_error is False
        # Barge-in fired → next_primed_frames is non-None.
        assert result.next_primed_frames is not None
        assert len(result.next_primed_frames) > 0
        # If metrics exist, barge_in flag is set.
        if result.metrics is not None:
            assert result.metrics.barge_in is True

    def test_half_duplex_ignores_speech_during_reply(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(_utterance_audio())

        def push_during_reply():
            time.sleep(0.05)
            mic.push(make_tone_burst(0.5, rate=RATE, amp=0.3))

        threading.Thread(target=push_during_reply, daemon=True).start()

        loop = _make_chat_loop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            transcript="initial",
            llm_text="One sentence. Two sentence. Three sentence. Four sentence.",
            llm_per_token_delay=0.02,
            play_fn=_slow_play,
            barge_in_enabled=False,
        )
        result = loop.run_one_turn([])

        assert result.had_error is False
        assert result.next_primed_frames is None
        assert result.metrics is not None
        assert result.metrics.barge_in is False
        assert "Four sentence" in result.metrics.response


class TestLlmErrorPath:
    def test_llm_raises_returns_none_metrics_with_error(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(_utterance_audio())

        loop = _make_chat_loop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            transcript="hello",
            llm_stream_fn=_llm_stream_raising(),
        )
        messages = [{"role": "system", "content": "x"}]
        result = loop.run_one_turn(messages)

        assert result.had_error is True
        assert result.metrics is None
        # User message should have been popped (the LLM call failed).
        assert messages == [{"role": "system", "content": "x"}]

    def test_llm_error_no_barge_in_returns_none_primed(self):
        """Sanity check the error path without user audio in flight:
        no barge-in detected, no primed_frames carried forward.

        (The error-path frame-carry LOGIC itself is covered in
        tests/unit/test_hardening.py::TestErrorPathFrameCarryover —
        we don't try to recreate the race here. The race would
        require the LLM to raise BEFORE the watcher fires
        coord.trigger, which the for-loop's
        ``if coord.is_set(): break`` makes impossible to engineer
        deterministically: if the watcher fires first, the loop
        exits cleanly and the exception never reaches the except
        block.)
        """
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(_utterance_audio())

        loop = _make_chat_loop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            transcript="hi",
            llm_stream_fn=_llm_stream_raising(),
        )
        result = loop.run_one_turn([])

        assert result.had_error is True
        assert result.metrics is None
        assert result.next_primed_frames is None


class TestFillerIntegration:
    def test_filler_plays_when_llm_stalls_on_first_token(self):
        """LLM with a slow first token (>idle_threshold) → filler
        plays before the real first sentence. Verifies the
        iter-011 wiring through ChatLoop.
        """
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(_utterance_audio())
        spk_holder = {"spk": None}

        def factory():
            spk_holder["spk"] = VirtualSpeakerStream(rate=24000)
            return spk_holder["spk"]

        # Slow LLM: 200ms of silence before the first token.
        def slow_llm(messages, config):
            time.sleep(0.2)
            for tok in "Done. ".split():
                yield tok + " "

        # Pre-rendered filler (a short distinct audio).
        filler_audio = np.full(512, 0.3, dtype=np.float32)
        filler_clip = (filler_audio, [])

        loop = _make_chat_loop(
            mic=mic,
            speaker_factory=factory,
            transcript="hi",
            llm_stream_fn=slow_llm,
            fillers=[filler_clip],
            idle_threshold=0.05,
        )
        result = loop.run_one_turn([])

        assert result.metrics is not None
        # Filler played because idle threshold (50ms) was reached
        # before the LLM first token (200ms).
        assert result.metrics.fillers_played == 1
        # Sentences also played.
        assert result.metrics.sentences_spoken >= 1
        # Speaker captured filler+sentence audio.
        assert len(spk_holder["spk"].captured) > 0


# ---- iter-019: end-to-end ChatLoop with real kokoro --------------------------


def _kokoro_loadable() -> bool:
    try:
        from examples.virtual_audio import _import_kokoro_engine
        _import_kokoro_engine()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _kokoro_loadable(), reason="kokoro TTS not loadable")
class TestChatLoopWithRealKokoro:
    """The closest test we have to "the real production chat loop
    is working." Uses real kokoro for synth, real
    examples._chat_playback.play_aligned for play, virtual mic +
    speaker for I/O, stub STT + stub LLM. Validates that
    synthesize_with_alignment composes correctly with
    SentenceWorker, BargeInWatcher, and the rest of ChatLoop.

    ~3-6s per test on first run (kokoro model load), faster after.
    Skipped on hosts where kokoro doesn't load.
    """

    def test_full_turn_with_real_synth_produces_real_audio(self):
        from examples._chat_playback import (
            TTS_RATE as PLAYBACK_RATE,
            play_aligned as _play_core,
        )
        from examples._chat_tts import synthesize_with_alignment
        from examples.virtual_audio import _import_kokoro_engine

        # Real kokoro engine.
        tts_engine = _import_kokoro_engine()

        # User input: virtual mic with iter-005's standard tone-burst
        # utterance fixture. Stub STT will pretend it transcribed
        # to "hi".
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(_utterance_audio())

        spk_holder = {"spk": None}

        def factory():
            spk_holder["spk"] = VirtualSpeakerStream(rate=PLAYBACK_RATE)
            return spk_holder["spk"]

        def real_synth(sentence: str):
            return synthesize_with_alignment(
                tts_engine, sentence, "af_heart", 1.0,
            )

        def real_play(speaker, audio_np, tokens, *,
                      is_first_sentence=False, cancel_event=None):
            return _play_core(
                speaker, audio_np, tokens,
                is_first_sentence=is_first_sentence,
                cancel_event=cancel_event,
                rate=PLAYBACK_RATE,
            )

        engine, transcribe = _stt_engine_stub(transcript="hi")

        loop = ChatLoop(
            mic=mic,
            speaker_factory=factory,
            rate=RATE,
            chunk=CHUNK,
            stt_engine=engine,
            transcribe_fn=transcribe,
            llm_stream_fn=_llm_stream_yielding("Hello there."),
            llm_config={"model": "stub-model"},
            synth_fn=real_synth,
            play_fn=real_play,
        )
        result = loop.run_one_turn([])

        assert result.metrics is not None
        assert result.had_error is False
        assert result.metrics.transcript == "hi"
        assert "Hello" in result.metrics.response
        assert result.metrics.sentences_spoken >= 1

        # Speaker received real synthesized audio. At 24kHz mono int16,
        # a "Hello there." utterance is ~0.5-1.5s = 24k-72k samples =
        # 48k-144k bytes. Bound loosely.
        captured = spk_holder["spk"].captured
        assert 20_000 < len(captured) < 200_000
        # Verify the audio has actual signal (RMS > 0.001 in float32).
        decoded = spk_holder["spk"].captured_float32
        assert float(np.sqrt(np.mean(decoded ** 2))) > 0.001


# ---- iter-027: KeyboardInterrupt cleanup ------------------------------------


class TestKeyboardInterruptCleanup:
    """``except Exception`` in run_one_turn doesn't catch
    KeyboardInterrupt (which inherits from BaseException). Without
    iter-027's finally additions, KeyboardInterrupt during the
    for-token loop bypassed worker/watcher cleanup — the worker
    thread kept running with the speaker stream open, only dying
    when the daemon thread was killed at process exit.

    iter-027 adds idempotent stop calls to the finally so any
    exit path (normal, error, KeyboardInterrupt, anything else)
    cleans up before propagating.
    """

    def test_keyboardinterrupt_propagates_and_closes_speaker(self):
        """LLM raises KeyboardInterrupt mid-stream → exception
        propagates out of run_one_turn AND the speaker stream
        gets closed (which it can only do via worker._run's
        finally, which only runs when the worker is stopped).
        """
        spk_holder = {"spk": None}

        def factory():
            spk_holder["spk"] = VirtualSpeakerStream(rate=24000)
            return spk_holder["spk"]

        def bad_llm(messages, config):
            yield "First "
            time.sleep(0.05)  # let worker thread start playing
            raise KeyboardInterrupt

        loop = _make_chat_loop(
            mic=_mic_with_utterance(),
            speaker_factory=factory,
            transcript="hi",
            llm_stream_fn=bad_llm,
            play_fn=_slow_play,
        )

        with pytest.raises(KeyboardInterrupt):
            loop.run_one_turn([])

        # Speaker was closed via worker.stop in the finally block.
        assert spk_holder["spk"]._closed is True

    def test_keyboardinterrupt_stops_worker_thread(self):
        """The worker thread should be joined (not is_alive)
        after run_one_turn returns from KeyboardInterrupt.
        """
        # Capture worker reference via SentenceWorker construction hook.
        from examples._chat_pipeline import SentenceWorker
        captured = []

        original_init = SentenceWorker.__init__

        def hook_init(self, **kwargs):
            original_init(self, **kwargs)
            captured.append(self)

        def bad_llm(messages, config):
            yield "First "
            time.sleep(0.05)
            raise KeyboardInterrupt

        loop = _make_chat_loop(
            mic=_mic_with_utterance(),
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            transcript="hi",
            llm_stream_fn=bad_llm,
            play_fn=_slow_play,
        )

        SentenceWorker.__init__ = hook_init  # type: ignore[method-assign]
        try:
            with pytest.raises(KeyboardInterrupt):
                loop.run_one_turn([])
        finally:
            SentenceWorker.__init__ = original_init  # type: ignore[method-assign]

        assert len(captured) == 1
        worker = captured[0]
        # Thread should be joined within a short window after run_one_turn returns.
        if worker._thread is not None:
            worker._thread.join(timeout=2.0)
            assert worker._thread.is_alive() is False
        # And cancelled flag should be set (because stop sets it via
        # the regular path... actually stop doesn't set cancelled,
        # only cancel does. Just verify _stop_event was set.)
        assert worker._stop_event.is_set()

    def test_normal_completion_unaffected_by_finally_stop_calls(self):
        """The new finally additions are idempotent — calling
        stop on an already-completed worker/watcher should be a
        no-op. Verify by running a normal turn and confirming
        all the iter-015 expected outcomes still hold.
        """
        loop = _make_chat_loop(
            mic=_mic_with_utterance(),
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            transcript="hi",
            llm_text="Hello.",
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        assert result.had_error is False
        assert result.metrics.transcript == "hi"


def _mic_with_utterance():
    """Helper for the iter-027 tests below."""
    mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
    mic.push(_utterance_audio())
    return mic


class TestIdleTimeoutWiring:
    """iter-166 — wire iter-165's recorder ``idle_timeout`` through ChatLoop.

    iter-165 added the pre-speech ``idle_timeout`` mechanism to
    ``record_utterance_streaming`` but no call site passed it. This lap
    threads it through ``ChatLoop.__init__`` and surfaces the recorder's
    ``idle_timed_out`` side-band flag on ``TurnResult`` so the still-pending
    ``run_session`` consumer (the iter-164 ``decide_silence_flush`` driver)
    can tell a deliberate inter-turn idle timeout apart from a VAD false
    trigger.

    The timeout fires off the recorder's frame-aligned virtual clock
    (``now = t_origin + frame_idx * frame_dt``), so it is deterministic
    under the default ``time.monotonic`` clock too — a mic that only ever
    yields silence (VirtualMicStream pads with silence forever once drained)
    elapses ``idle_timeout`` virtual seconds within a bounded number of
    frame reads.
    """

    def test_default_no_idle_timeout_waits_for_speech(self):
        # idle_timeout defaults to None ⇒ the loop waits for speech and
        # records normally; idle_timed_out is never set. Byte-for-byte the
        # pre-iter-166 path.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(
            concat(
                make_silence(2.0, rate=RATE),  # long lead silence
                make_tone_burst(1.0, rate=RATE, amp=0.3),
                make_silence(1.2, rate=RATE),
            )
        )
        loop = _make_chat_loop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            transcript="how are you",
            llm_text="I am well.",
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        assert result.idle_timed_out is False

    def test_idle_timeout_fires_and_flags_turn_result(self):
        # Only silence on the mic ⇒ without a timeout the recorder would
        # block forever. With idle_timeout set, run_one_turn returns a
        # no-metrics TurnResult flagged idle_timed_out — and the LLM is
        # never reached (no transcript).
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(make_silence(0.5, rate=RATE))  # then padded silence forever
        spk_holder = {"spk": None}

        def factory():
            spk_holder["spk"] = VirtualSpeakerStream(rate=24000)
            return spk_holder["spk"]

        loop = _make_chat_loop(
            mic=mic,
            speaker_factory=factory,
            transcript="should not be used",
            idle_timeout=1.0,
        )
        messages = [{"role": "system", "content": "be nice"}]
        result = loop.run_one_turn(messages)

        assert result.metrics is None
        assert result.had_error is False
        assert result.held is False
        assert result.idle_timed_out is True
        assert result.next_primed_frames is None
        # No utterance ⇒ no user message appended, speaker never opened.
        assert messages == [{"role": "system", "content": "be nice"}]
        assert spk_holder["spk"] is None

    def test_speech_before_timeout_records_and_does_not_flag(self):
        # Speech starts within the timeout window ⇒ a normal turn with
        # metrics; idle_timed_out stays False because the recorder saw a
        # speech frame before the pre-speech window elapsed.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(_utterance_audio())  # 0.3s lead silence, well under 5s
        loop = _make_chat_loop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            transcript="hi",
            llm_text="Hello.",
            idle_timeout=5.0,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        assert result.metrics.transcript == "hi"
        assert result.idle_timed_out is False

    def test_false_trigger_without_idle_timeout_is_not_flagged(self):
        # A genuine VAD false trigger / too-short utterance (no idle_timeout
        # configured) returns no metrics but idle_timed_out is False — the
        # flag is reserved for the deliberate pre-speech timeout, so
        # run_session can distinguish the two no-metrics causes.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        mic.push(
            concat(
                make_silence(0.3, rate=RATE),
                make_tone_burst(0.05, rate=RATE, amp=0.3),  # < min_speech
                make_silence(1.5, rate=RATE),
            )
        )
        loop = _make_chat_loop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            transcript="x",
        )
        result = loop.run_one_turn([])
        assert result.metrics is None
        assert result.idle_timed_out is False


class TestRespondToText:
    """iter-168: respond_to_text — answer a flushed mid-thought fragment
    (backlog #9's spoken-response hop) as its own turn, with NO mic recording.

    These tests never push mic audio: respond_to_text bypasses Phase 1
    (record + STT) entirely and drives only the shared _stream_response half.
    """

    def _loop(self, *, speaker_factory, llm_text="Sure thing. Here you go."):
        # mic is never read by respond_to_text, but ChatLoop requires one.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        return _make_chat_loop(
            mic=mic,
            speaker_factory=speaker_factory,
            llm_text=llm_text,
        )

    def test_answers_text_and_updates_history(self):
        spk_holder = {"spk": None}

        def factory():
            spk_holder["spk"] = VirtualSpeakerStream(rate=24000)
            return spk_holder["spk"]

        loop = self._loop(
            speaker_factory=factory,
            llm_text="I am well. Thanks for asking.",
        )
        messages = [{"role": "system", "content": "be nice"}]
        result = loop.respond_to_text(messages, "how are you")

        assert result.metrics is not None
        assert result.had_error is False
        m = result.metrics
        # The flushed fragment IS the turn's transcript.
        assert m.transcript == "how are you"
        assert m.response.startswith("I am well")
        assert "Thanks for asking" in m.response
        assert m.sentences_spoken >= 1
        assert m.llm_total > 0
        # No recording happened, so STT timings are zeroed.
        assert m.stt_time == 0.0
        assert m.speech_duration == 0.0
        # History updated exactly like a recorded turn: user + assistant.
        assert len(messages) == 3
        assert messages[1] == {"role": "user", "content": "how are you"}
        assert messages[2]["role"] == "assistant"
        # Audio was actually spoken through the worker.
        assert len(spk_holder["spk"].captured) > 0

    def test_strips_whitespace_from_text(self):
        loop = self._loop(speaker_factory=lambda: VirtualSpeakerStream(rate=24000))
        messages = [{"role": "system", "content": "x"}]
        result = loop.respond_to_text(messages, "  the deadline is  ")
        assert result.metrics is not None
        assert result.metrics.transcript == "the deadline is"
        assert messages[1]["content"] == "the deadline is"

    def test_blank_text_is_a_no_op(self):
        spk_holder = {"spk": None}

        def factory():
            spk_holder["spk"] = VirtualSpeakerStream(rate=24000)
            return spk_holder["spk"]

        loop = self._loop(speaker_factory=factory)
        messages = [{"role": "system", "content": "x"}]
        result = loop.respond_to_text(messages, "   ")
        # No turn produced, no history mutation, no speaker opened.
        assert result.metrics is None
        assert result.next_primed_frames is None
        assert messages == [{"role": "system", "content": "x"}]
        assert spk_holder["spk"] is None

    def test_empty_text_is_a_no_op(self):
        loop = self._loop(speaker_factory=lambda: VirtualSpeakerStream(rate=24000))
        messages = []
        result = loop.respond_to_text(messages, "")
        assert result.metrics is None
        assert messages == []

    def test_displaced_is_always_empty(self):
        loop = self._loop(speaker_factory=lambda: VirtualSpeakerStream(rate=24000))
        result = loop.respond_to_text([], "answer this")
        # A flushed fragment IS the turn — nothing was displaced by it.
        assert result.displaced == ()

    def test_llm_error_pops_user_message(self):
        loop = ChatLoop(
            mic=VirtualMicStream(rate=RATE, chunk_size=CHUNK),
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            rate=RATE,
            chunk=CHUNK,
            stt_engine=SimpleNamespace(_last_text=None, model_repo="stub"),
            transcribe_fn=lambda w: "unused",
            llm_stream_fn=_llm_stream_raising(),
            llm_config={"model": "stub-model"},
            synth_fn=_const_synth(),
            play_fn=_instant_play,
        )
        messages = [{"role": "system", "content": "x"}]
        result = loop.respond_to_text(messages, "trigger the error")
        assert result.metrics is None
        assert result.had_error is True
        # The user message we appended was popped on error — history clean.
        assert messages == [{"role": "system", "content": "x"}]

    def test_model_stamped_from_config(self):
        loop = self._loop(speaker_factory=lambda: VirtualSpeakerStream(rate=24000))
        result = loop.respond_to_text([], "hello there")
        assert result.metrics is not None
        assert result.metrics.model == "stub-model"
