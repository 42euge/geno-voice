"""iter-159 — ChatLoop ⇄ UtteranceAggregator live wiring (backlog #9).

These tests drive the *real* production path: ``ChatLoop.run_one_turn`` with a
real ``UtteranceAggregator`` injected, virtual audio, stub STT, and a stub LLM.
They prove the two new branches the wiring adds:

  - HELD: an organic-mode aggregator that holds a mid-thought utterance makes
    ``run_one_turn`` return no-metrics (the loop re-listens, like a false
    trigger) without opening the LLM stream.
  - RELEASED + MERGED: a quick follow-on utterance merges with the held one,
    the LLM responds to the *joined* text, and ``TurnMetrics.false_endpoint``
    is set — populating iter-154's metric from the live path.

And the invariants:

  - aggregator=None (default) ⇒ byte-for-byte the pre-iter-159 path.
  - an injected aggregator with a default (half-duplex) config ⇒ transparent
    passthrough: every utterance responded to immediately, false_endpoint False.

The aggregator's gap math is driven by the recorder's ``speech_start_at`` and
the loop's ``speech_ended_at``, both off the injected clock. We inject a manual
clock so consecutive turns land within / outside ``max_gap_secs`` (2.0s)
deterministically rather than depending on wall time.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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

# ---- Load the pure session modules by path (dodge eager-pipecat import) -----

_SESSION_DIR = ROOT / "session"


def _load_session_modules():
    if "session" not in sys.modules:
        pkg = types.ModuleType("session")
        pkg.__path__ = [str(_SESSION_DIR)]
        sys.modules["session"] = pkg
    for name in (
        "full_duplex",
        "text_eou",
        "utterance_merging",
        "utterance_buffer",
        "utterance_aggregator",
    ):
        full = f"session.{name}"
        if full in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(
            full, _SESSION_DIR / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "session"
        sys.modules[full] = mod
        spec.loader.exec_module(mod)


_load_session_modules()
from session.full_duplex import FullDuplexConfig  # noqa: E402
from session.utterance_aggregator import UtteranceAggregator  # noqa: E402


# ---- Doubles ----------------------------------------------------------------


def _stt_engine_stub(transcript: str):
    engine = SimpleNamespace(_last_text=None, model_repo="stub")

    def transcribe(wav_bytes: bytes):
        if not wav_bytes:
            return None
        return transcript

    return engine, transcribe


def _llm_echo():
    """LLM that echoes the user's content as a single complete sentence —
    lets a test read back exactly what text the loop fed the engine.
    """

    def factory(messages, config):
        user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        yield f"You said {user}."

    return factory


def _const_synth(samples: int = 1024):
    def synth(text: str):
        return np.full(samples, 0.5, dtype=np.float32), []

    return synth


def _instant_play(speaker, audio_np, tokens, *, is_first_sentence=False, cancel_event=None):
    speaker.write((audio_np * 32767).astype(np.int16).tobytes())
    return len(audio_np) / 24000.0


def _utterance_audio() -> np.ndarray:
    return concat(
        make_silence(0.3, rate=RATE),
        make_tone_burst(1.0, rate=RATE, amp=0.3),
        make_silence(1.2, rate=RATE),
    )


class _ManualClock:
    """Monotonic clock advanced explicitly by the test. The recorder reads it
    once at t_origin per call; the loop reads it for speech_ended_at and
    turn_start. We bump it between turns to control the inter-utterance gap.
    """

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _make_loop(mic, *, transcript, aggregator=None, clock=None, llm_stream_fn=None):
    engine, transcribe = _stt_engine_stub(transcript)
    return ChatLoop(
        mic=mic,
        speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
        rate=RATE,
        chunk=CHUNK,
        stt_engine=engine,
        transcribe_fn=transcribe,
        llm_stream_fn=llm_stream_fn or _llm_echo(),
        llm_config={"model": "stub-model"},
        synth_fn=_const_synth(),
        play_fn=_instant_play,
        aggregator=aggregator,
        clock=clock or (lambda: 0.0),
    )


def _push_utterance(mic):
    mic.push(_utterance_audio())


# ---- aggregator=None: unchanged path ----------------------------------------


class TestNoAggregator:
    def test_default_path_unchanged(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_utterance(mic)
        loop = _make_loop(mic, transcript="how are you", aggregator=None)
        messages = [{"role": "system", "content": "x"}]
        result = loop.run_one_turn(messages)
        assert result.metrics is not None
        assert result.metrics.transcript == "how are you"
        assert result.metrics.false_endpoint is False
        # iter-161: a responded turn is never flagged held.
        assert result.held is False
        # iter-162: no aggregator ⇒ nothing displaced.
        assert result.displaced == ()
        # iter-163: no aggregator ⇒ never merge-capped.
        assert result.metrics.merge_capped is False


# ---- Half-duplex aggregator: transparent passthrough ------------------------


class TestHalfDuplexPassthrough:
    def test_injected_default_config_responds_immediately(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_utterance(mic)
        agg = UtteranceAggregator(config=FullDuplexConfig())  # half-duplex
        loop = _make_loop(mic, transcript="hello world", aggregator=agg)
        result = loop.run_one_turn([{"role": "system", "content": "x"}])
        assert result.metrics is not None
        assert result.metrics.transcript == "hello world"
        assert result.metrics.false_endpoint is False
        # Nothing held — the buffer is a passthrough.
        assert agg.pending is None
        # iter-161: half-duplex never holds, so the turn is never flagged.
        assert result.held is False
        # iter-162: half-duplex releases one turn at a time — never displaces.
        assert result.displaced == ()
        # iter-163: half-duplex never holds, so the cap can never fire.
        assert result.metrics.merge_capped is False


# ---- Organic aggregator: hold + merge ---------------------------------------


def _organic_aggregator():
    return UtteranceAggregator(
        config=FullDuplexConfig(enabled=True, utterance_merging=True)
    )


class TestOrganicHold:
    def test_midthought_utterance_is_held_no_metrics(self):
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_utterance(mic)
        agg = _organic_aggregator()
        clock = _ManualClock()
        loop = _make_loop(
            mic, transcript="I think that", aggregator=agg, clock=clock
        )
        messages = [{"role": "system", "content": "x"}]
        result = loop.run_one_turn(messages)
        # Held — no metrics, no LLM response, no user message appended.
        assert result.metrics is None
        assert result.had_error is False
        # iter-161: flagged ``held`` so run_session counts it as a held
        # utterance, not a VAD false trigger.
        assert result.held is True
        assert agg.pending == "I think that"
        assert messages == [{"role": "system", "content": "x"}]


class TestOrganicMerge:
    def test_quick_followon_merges_and_sets_false_endpoint(self):
        agg = _organic_aggregator()
        clock = _ManualClock()
        messages = [{"role": "system", "content": "x"}]

        # Turn 1: mid-thought, held.
        mic1 = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_utterance(mic1)
        loop1 = _make_loop(
            mic1, transcript="I think that", aggregator=agg, clock=clock
        )
        r1 = loop1.run_one_turn(messages)
        assert r1.metrics is None
        assert agg.pending == "I think that"

        # Small gap (< max_gap_secs=2.0) → the next utterance merges.
        clock.advance(0.5)

        # Turn 2: completes the thought → release the merged turn.
        mic2 = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_utterance(mic2)
        loop2 = _make_loop(
            mic2,
            transcript="the sky is blue.",
            aggregator=agg,
            clock=clock,
        )
        r2 = loop2.run_one_turn(messages)
        assert r2.metrics is not None
        # Responded to the JOINED text.
        assert r2.metrics.transcript == "I think that the sky is blue."
        # The metric iter-154 introduced, now populated from the live path.
        assert r2.metrics.false_endpoint is True
        # LLM saw the merged text.
        assert "I think that the sky is blue." in r2.metrics.response
        assert agg.pending is None
        # iter-162: a genuine merge releases a SINGLE turn — nothing displaced.
        assert r2.displaced == ()
        # iter-163: a natural merge (completed on a real sentence) is NOT the
        # cap firing.
        assert r2.metrics.merge_capped is False

    def test_long_silence_displaces_abandoned_fragment(self):
        # iter-162: the user trails off mid-thought ("I was thinking about
        # the"), held. Then — after a long silence (> max_gap_secs) that
        # proves it was NOT a false endpoint — a genuinely new complete
        # utterance arrives. The buffer releases the abandoned fragment as
        # its own NEW turn AND the new utterance in one offer. We must
        # respond to the NEW utterance only (not the glued garble) and carry
        # the abandoned fragment forward as ``displaced``.
        agg = _organic_aggregator()
        clock = _ManualClock()
        messages = [{"role": "system", "content": "x"}]

        # Turn 1: mid-thought, held.
        mic1 = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_utterance(mic1)
        loop1 = _make_loop(
            mic1, transcript="I was thinking about the",
            aggregator=agg, clock=clock,
        )
        r1 = loop1.run_one_turn(messages)
        assert r1.metrics is None
        assert r1.displaced == ()
        assert agg.pending == "I was thinking about the"

        # Long gap (> max_gap_secs=2.0) → NOT a continuation.
        clock.advance(5.0)

        # Turn 2: a complete, genuinely-new thought.
        mic2 = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_utterance(mic2)
        loop2 = _make_loop(
            mic2, transcript="What time is it?",
            aggregator=agg, clock=clock,
        )
        r2 = loop2.run_one_turn(messages)
        assert r2.metrics is not None
        # Respond to the NEW utterance — NOT "I was thinking about the What
        # time is it?" (the pre-iter-162 glued garble).
        assert r2.metrics.transcript == "What time is it?"
        assert r2.metrics.false_endpoint is False
        # The abandoned fragment rides out as displaced, not in the response.
        assert r2.displaced == ("I was thinking about the",)
        assert "I was thinking about the" not in r2.metrics.response
        assert agg.pending is None

    def test_complete_utterance_emits_immediately_no_false_endpoint(self):
        agg = _organic_aggregator()
        clock = _ManualClock()
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_utterance(mic)
        loop = _make_loop(
            mic,
            transcript="What time is it?",
            aggregator=agg,
            clock=clock,
        )
        result = loop.run_one_turn([{"role": "system", "content": "x"}])
        assert result.metrics is not None
        assert result.metrics.transcript == "What time is it?"
        assert result.metrics.false_endpoint is False
        assert agg.pending is None


class TestOrganicMergeCap:
    # iter-163: a pathological "unfinished forever" stream hits the
    # max_merge_depth cap; the buffer force-emits the still-mid-thought text.
    # That turn must stamp TurnMetrics.merge_capped so the operator sees the
    # backstop fire instead of it being silently counted as a clean merge.

    def _capped_aggregator(self):
        return UtteranceAggregator(
            config=FullDuplexConfig(enabled=True, utterance_merging=True),
            max_merge_depth=1,   # the very first merge force-emits
        )

    def test_cap_force_emit_sets_merge_capped(self):
        agg = self._capped_aggregator()
        clock = _ManualClock()
        messages = [{"role": "system", "content": "x"}]

        # Turn 1: mid-thought, held (holding is not a merge, so no cap yet).
        mic1 = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_utterance(mic1)
        loop1 = _make_loop(
            mic1, transcript="I was thinking about the",
            aggregator=agg, clock=clock,
        )
        r1 = loop1.run_one_turn(messages)
        assert r1.metrics is None
        assert agg.pending == "I was thinking about the"

        # Quick gap (< max_gap_secs) → would merge; but cap=1 force-emits the
        # still-unfinished running text on this first merge.
        clock.advance(0.5)
        mic2 = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_utterance(mic2)
        loop2 = _make_loop(
            mic2, transcript="and the",      # still trailing — unfinished
            aggregator=agg, clock=clock,
        )
        r2 = loop2.run_one_turn(messages)
        assert r2.metrics is not None
        assert r2.metrics.transcript == "I was thinking about the and the"
        # The cap fired: both flags set.
        assert r2.metrics.false_endpoint is True
        assert r2.metrics.merge_capped is True
        # Force-emit clears the buffer.
        assert agg.pending is None
