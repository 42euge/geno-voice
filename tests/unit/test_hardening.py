"""Hardening tests for iter-014.

These tests target small but real edge cases that surfaced from a
careful re-read of the production code:

1. ``rms()`` on an empty array used to emit
   ``RuntimeWarning: Mean of empty slice`` and return NaN. NaN
   silently breaks ``VadState.feed`` because ``NaN > threshold``
   is always False — every frame becomes IDLE forever. Now it
   returns 0.0 cleanly.

2. The mic_chat LLM-error path stops the watcher but throws away
   ``watcher.frames``. If the user happened to be barging in when
   the LLM call failed (network blip, DNS hiccup, 5xx), their
   audio gets dropped. The fix carries the frames forward as
   ``primed_frames`` for the next ``record_utterance_streaming``
   call.

3. ``TurnMetrics`` now exposes ``fillers_played`` and ``barge_in``
   so the per-turn summary can show them.
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_recording import (  # noqa: E402
    CHUNK,
    RATE,
    record_utterance_streaming,
    rms,
)
from examples.virtual_audio import (  # noqa: E402
    VirtualMicStream,
    concat,
    make_silence,
    make_tone_burst,
)


# ---- 1) rms edge cases -------------------------------------------------------


class TestRmsEdgeCases:
    def test_empty_array_returns_zero_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any RuntimeWarning becomes an exception
            result = rms(np.array([], dtype=np.float32))
        assert result == 0.0

    def test_none_returns_zero(self):
        # Belt and suspenders: rms(None) shouldn't crash either.
        assert rms(None) == 0.0

    def test_silence_returns_zero(self):
        result = rms(np.zeros(1024, dtype=np.float32))
        assert result == 0.0

    def test_constant_amplitude_matches_amplitude(self):
        # All samples equal +0.5 → RMS = 0.5.
        result = rms(np.full(1024, 0.5, dtype=np.float32))
        assert result == pytest.approx(0.5)

    def test_sine_rms_matches_theory(self):
        # 0.3 amplitude sine has theoretical RMS = 0.3 / sqrt(2) ≈ 0.2121.
        n = 4096
        t = np.arange(n) / RATE
        wave = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        result = rms(wave)
        assert 0.20 < result < 0.22


# ---- 2) record_utterance_streaming with empty mic reads ---------------------


class _EmptyMicStream:
    """A pyaudio-shape stream that returns alternating empty bytes
    and silence — exposes the rms-on-empty path that used to NaN.
    Will eventually deliver speech so the function returns.
    """

    def __init__(self, n_empty_reads: int = 3, *, rate: int = RATE,
                 chunk_size: int = CHUNK):
        self.rate = rate
        self.chunk_size = chunk_size
        self._n_empty = n_empty_reads
        self._reads = 0
        # After the empty reads, deliver a tone burst then silence so
        # the VAD can fire DONE_OK eventually.
        tail = concat(
            make_silence(0.3, rate=rate),
            make_tone_burst(0.6, rate=rate, amp=0.3),
            make_silence(1.5, rate=rate),
        )
        self._tail = bytearray(tail.astype(np.int16).tobytes())

    def get_read_available(self) -> int:
        return max(self.chunk_size, len(self._tail) // 2)

    def read(self, n_frames: int, exception_on_overflow: bool = False) -> bytes:
        self._reads += 1
        if self._reads <= self._n_empty:
            return b""
        n_bytes = n_frames * 2
        out = bytes(self._tail[:n_bytes])
        del self._tail[:n_bytes]
        # Pad so we never block.
        if len(out) < n_bytes:
            out = out + b"\x00" * (n_bytes - len(out))
        return out


class FrameClock:
    def __init__(self, chunk: int = CHUNK, rate: int = RATE):
        self._dt = chunk / rate
        self._t = 0.0

    def __call__(self) -> float:
        t = self._t
        self._t += self._dt
        return t


class TestRecordUtteranceWithEmptyReads:
    def test_empty_bytes_reads_dont_raise_or_warn(self):
        """Hardening regression: empty mic reads used to push NaN
        into the VAD via rms(empty_array). Now they're silently
        treated as silence (level 0.0), and recording proceeds.
        """
        engine = SimpleNamespace(_last_text=None, model_repo="stub")
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # surface any latent RuntimeWarning
            wav, dur, _ = record_utterance_streaming(
                _EmptyMicStream(n_empty_reads=3),
                engine,
                transcribe_fn=lambda w: "ok",
                clock=FrameClock(),
                output=io.StringIO(),
            )
        # We got a non-empty wav covering the speech that arrived
        # AFTER the empty reads.
        assert len(wav) > 0
        assert dur > 0.0


# ---- 3) TurnMetrics.print surfaces new fields --------------------------------


class TestTurnMetricsPrint:
    def _capture_print(self, **fields) -> str:
        from examples._chat_metrics import TurnMetrics
        m = TurnMetrics(**fields)
        # Capture stdout via redirect.
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.print(turn=1)
        return buf.getvalue()

    def test_fillers_played_zero_omitted_from_tts_line(self):
        out = self._capture_print(
            sentences_spoken=2, fillers_played=0,
        )
        assert "filler" not in out.lower()
        assert "2 sentences" in out

    def test_one_filler_appears_in_tts_line(self):
        out = self._capture_print(
            sentences_spoken=2, fillers_played=1,
        )
        assert "1 filler" in out
        # No plural for the count of one.
        assert "1 fillers" not in out

    def test_multiple_fillers_pluralized(self):
        out = self._capture_print(
            sentences_spoken=3, fillers_played=2,
        )
        assert "2 fillers" in out

    def test_no_barge_in_omits_barge_line(self):
        out = self._capture_print(barge_in=False)
        assert "Barge-in" not in out

    def test_barge_in_true_shows_barge_line(self):
        out = self._capture_print(barge_in=True)
        assert "Barge-in:" in out
        assert "user interrupted" in out

    def test_default_metrics_has_new_fields_at_zero(self):
        from examples._chat_metrics import TurnMetrics
        m = TurnMetrics()
        assert m.fillers_played == 0
        assert m.barge_in is False


# ---- 4) Error-path frame carryover (logic, no real LLM) ----------------------

class TestErrorPathFrameCarryover:
    """We can't easily exercise mic_chat.run_chat's error path end to
    end without a real LLM. But we can verify the small piece of
    logic in isolation: given a watcher whose `detected=True` and
    `frames=[...]`, the LLM-error path should grab those frames as
    `primed_frames`. This test mirrors that logic shape.
    """

    def test_watcher_detected_carries_frames_to_primed(self):
        # Simulate the error-path snippet in mic_chat.
        captured_frames = [b"\x01" * (CHUNK * 2) for _ in range(5)]
        watcher = SimpleNamespace(detected=True, frames=captured_frames)

        # The exact code shape from mic_chat.run_chat error path:
        primed_frames: list[bytes] | None = None
        if watcher.detected:
            primed_frames = list(watcher.frames)

        assert primed_frames == captured_frames
        # Defensive copy: not the same list object.
        assert primed_frames is not watcher.frames

    def test_watcher_not_detected_leaves_primed_none(self):
        watcher = SimpleNamespace(detected=False, frames=[])
        primed_frames: list[bytes] | None = None
        if watcher.detected:
            primed_frames = list(watcher.frames)
        assert primed_frames is None
