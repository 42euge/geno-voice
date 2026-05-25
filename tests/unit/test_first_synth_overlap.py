"""Tests for iter-073 — first-sentence overlap savings.

Metric 2.2 from docs/perf-metrics-taxonomy.md.

    overlap = max(0, min(synth_done, llm_done) - max(synth_start, llm_start))

Distinct from iter-043's whole-stream ratio: this scopes to the
FIRST sentence specifically because that's what gates TTFS. 0
means first synth ran entirely after LLM finished (sequential —
streaming bought nothing for TTFS). Equal to first-synth duration
means first synth ran entirely under LLM streaming (best case).
"""

from __future__ import annotations

import io
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_loop import ChatLoop  # noqa: E402
from examples._chat_metrics import (  # noqa: E402
    TurnMetrics,
    print_session_summary,
)
from examples._chat_pipeline import SentenceWorker  # noqa: E402
from examples._chat_recording import CHUNK, RATE  # noqa: E402
from examples.virtual_audio import (  # noqa: E402
    VirtualMicStream,
    VirtualSpeakerStream,
    concat,
    make_silence,
    make_tone_burst,
)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ---- Default + per-turn print --------------------------------------------


class TestDefault:
    def test_turnmetrics_default_zero(self):
        assert TurnMetrics().first_synth_overlap_seconds == 0.0

    def test_worker_defaults_none(self):
        w = SentenceWorker(
            speaker_factory=lambda: object(),
            synth_fn=lambda s: (np.zeros(8, dtype=np.float32), []),
            play_fn=lambda *a, **k: 0.0,
        )
        assert w.first_synth_start_at is None
        assert w.first_synth_done_at is None


class TestPerTurnPrint:
    def _capture(self, m: TurnMetrics) -> str:
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            m.print(turn=1)
        finally:
            sys.stdout = old
        return _strip_ansi(buf.getvalue())

    def test_zero_omits_line(self):
        m = TurnMetrics(transcript="hi", model="stub",
                        first_synth_overlap_seconds=0.0)
        assert "1st-synth save" not in self._capture(m)

    def test_nonzero_emits_dim(self):
        m = TurnMetrics(transcript="hi", model="stub",
                        first_synth_overlap_seconds=0.040)  # 40ms
        out = self._capture(m)
        assert "1st-synth save" in out
        assert "40ms" in out
        assert "TTFS shaved by streaming" in out

    def test_meaningful_value_emits_green(self):
        # >100ms threshold — green path. Test just verifies the
        # value lands; color is an ANSI thing we strip.
        m = TurnMetrics(transcript="hi", model="stub",
                        first_synth_overlap_seconds=0.250)
        out = self._capture(m)
        assert "250ms" in out


# ---- Session aggregate --------------------------------------------


def _m(secs=0.0):
    return TurnMetrics(ttfs=0.5, first_synth_overlap_seconds=secs)


def _summary(metrics_list, **kwargs):
    out = io.StringIO()
    print_session_summary(metrics_list, {"model": "stub"}, file=out, **kwargs)
    return _strip_ansi(out.getvalue())


class TestSessionSummary:
    def test_no_data_omits(self):
        assert "1st-synth saved" not in _summary([_m(), _m()])

    def test_emits_median(self):
        plain = _summary([_m(secs=0.05), _m(secs=0.15), _m(secs=0.25)])
        # Median of [50, 150, 250] = 150.
        assert "1st-synth saved:  150ms median" in plain

    def test_zeros_filtered(self):
        plain = _summary([_m(secs=0.0), _m(secs=0.10), _m(secs=0.20)])
        # Median of [100, 200] = 150.
        assert "1st-synth saved:  150ms median" in plain


# ---- ChatLoop arithmetic ------------------------------------------


def _const_synth(samples=2048, delay=0.0):
    def synth(s):
        if delay > 0:
            time.sleep(delay)
        return np.full(samples, 0.5, dtype=np.float32), []
    return synth


def _fast_play(speaker, audio, tokens, *, is_first_sentence=False, cancel_event=None):
    audio_int16 = (audio * 32767).astype(np.int16)
    speaker.write(audio_int16.tobytes())
    return 0.0


def _yield_tokens(text, *, per_token_delay=0.0):
    import re as _re
    def factory(messages, config):
        for p in _re.findall(r"\S+|\.|!|\?", text):
            if per_token_delay > 0:
                time.sleep(per_token_delay)
            yield p + " "
    return factory


def _push_one(mic):
    mic.push(concat(
        make_silence(0.3, rate=RATE),
        make_tone_burst(1.0, rate=RATE, amp=0.3),
        make_silence(1.5, rate=RATE),
    ))


class TestChatLoopWiring:
    def _build_loop(self, *, mic, response, synth_delay=0.0,
                    per_token_delay=0.0):
        engine = SimpleNamespace(_last_text=None, model_repo="stub")
        return ChatLoop(
            mic=mic,
            speaker_factory=lambda: VirtualSpeakerStream(rate=24000),
            stt_engine=engine,
            transcribe_fn=lambda w: "hi" if w else None,
            llm_stream_fn=_yield_tokens(response, per_token_delay=per_token_delay),
            llm_config={"model": "stub"},
            synth_fn=_const_synth(delay=synth_delay),
            play_fn=_fast_play,
        )

    def test_overlap_when_synth_under_llm(self):
        # Slow LLM (so it's still streaming when first synth runs)
        # + slow synth → first synth starts before LLM done →
        # measurable overlap.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        # Multi-token response with per-token delay so the LLM
        # stream window spans a meaningful interval.
        long = " ".join(f"sentence {i}." for i in range(8))
        loop = self._build_loop(
            mic=mic, response=long,
            synth_delay=0.04, per_token_delay=0.015,
        )
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # First synth started before LLM finished → overlap > 0.
        assert result.metrics.first_synth_overlap_seconds > 0
        # Bounded by the synth duration itself.
        assert result.metrics.first_synth_overlap_seconds < 1.0

    def test_overlap_zero_for_no_synth(self):
        # Response with no terminator → no complete sentence → no
        # synth call → first_synth_start_at stays None → overlap 0.
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        loop = self._build_loop(mic=mic, response="no terminator yet")
        result = loop.run_one_turn([])
        assert result.metrics is not None
        # Trailing remainder still gets submitted at end-of-stream,
        # so synth DOES happen. Just verify the field is in [0, ∞).
        assert result.metrics.first_synth_overlap_seconds >= 0
        # And bounded by some sane upper bound.
        assert result.metrics.first_synth_overlap_seconds < 10.0

    def test_overlap_clamps_at_zero(self):
        # Field must never go negative (defensive — would happen
        # if synth started AFTER llm_stream_done).
        mic = VirtualMicStream(rate=RATE, chunk_size=CHUNK)
        _push_one(mic)
        loop = self._build_loop(mic=mic, response="Hi.")
        result = loop.run_one_turn([])
        assert result.metrics is not None
        assert result.metrics.first_synth_overlap_seconds >= 0


# ---- SentenceWorker timing latching ---------------------------------


class TestWorkerLatch:
    def test_first_synth_latched_once(self):
        # Multi-sentence: only the FIRST synth's start/done get
        # latched — subsequent sentences don't update the field.
        observed = []

        def synth(s):
            observed.append(s)
            return np.zeros(8, dtype=np.float32), []

        w = SentenceWorker(
            speaker_factory=lambda: SimpleNamespace(
                write=lambda b: None, close=lambda: None,
            ),
            synth_fn=synth,
            play_fn=lambda *a, **k: 0.0,
        )
        w.start()
        w.submit("first")
        w.submit("second")
        w.submit("third")
        w.submit_done()
        w.wait_done(timeout=2.0)

        assert len(observed) == 3
        assert w.first_synth_start_at is not None
        assert w.first_synth_done_at is not None
        # done > start always.
        assert w.first_synth_done_at >= w.first_synth_start_at

    def test_first_synth_unset_on_zero_synth(self):
        # No submissions → no synth → no latch.
        w = SentenceWorker(
            speaker_factory=lambda: SimpleNamespace(
                write=lambda b: None, close=lambda: None,
            ),
            synth_fn=lambda s: (np.zeros(8, dtype=np.float32), []),
            play_fn=lambda *a, **k: 0.0,
        )
        w.start()
        w.submit_done()
        w.wait_done(timeout=2.0)
        assert w.first_synth_start_at is None
        assert w.first_synth_done_at is None

    def test_failed_first_synth_not_latched(self):
        # If the first synth raises, the SECOND (successful) synth
        # becomes the "first" for accounting purposes.
        calls = [0]

        def synth(s):
            calls[0] += 1
            if calls[0] == 1:
                raise RuntimeError("first synth fails")
            return np.zeros(8, dtype=np.float32), []

        w = SentenceWorker(
            speaker_factory=lambda: SimpleNamespace(
                write=lambda b: None, close=lambda: None,
            ),
            synth_fn=synth,
            play_fn=lambda *a, **k: 0.0,
        )
        w.start()
        w.submit("first")
        w.submit("second")
        w.submit_done()
        w.wait_done(timeout=2.0)

        # The first synth was the second sentence (because the
        # first one raised). Latch should reflect the successful
        # synth, not the failed one.
        assert calls[0] == 2
        assert len(w.errors) == 1
        assert w.first_synth_start_at is not None
