"""iter-232 — Fast, deterministic unit coverage for the streaming Silero path.

``vad/silero.py``'s ``SileroStream`` / ``stream_samples`` give live capture
*incremental* speech-start/​end decisions (the desktop ContinuousListener wants
a turn cut the instant Silero sees trailing silence, not after a whole-WAV
round-trip). The batch ``segment_*`` path needs the whole utterance buffered
first; this is its frame-by-frame analogue.

These tests stub ``silero_vad.VADIterator`` with a faithful re-implementation of
its real state machine (trigger on P>=threshold, close after
``min_silence_ms`` of P<threshold-0.15, with ``speech_pad`` applied), so the
buffering / event-pairing / flush logic is exercised deterministically WITHOUT
the model. The "does streaming reconstruct batch segmentation on real audio"
proof lives in ``tests/integration/test_silero_stream_recordings.py``.

Audio convention in these fakes: a sample value >= 0.5 marks a "loud" (speech)
sample; < 0.5 marks silence. The fake iterator reads only ``window[0]``.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from vad.silero import (
    SileroParams,
    SileroStream,
    SpeechSegment,
    StreamEvent,
    StreamProtocol,
    WINDOW_SAMPLES,
    _events_to_segments,
    decode_float32_le,
    stream_samples,
)


# ---------------------------------------------------------------------------
# Fake VADIterator — mirrors the real silero_vad.VADIterator state machine.
# ---------------------------------------------------------------------------


class _FakeVADIterator:
    """Re-implements the real VADIterator trigger/hangover logic deterministically.

    P(speech) is 1.0 when ``window[0] >= 0.5`` else 0.0. Everything else (the
    ``temp_end`` silence hangover, the speech_pad applied to start/end, the
    sample counter) matches the upstream ``__call__`` so our tests assert the
    real contract, not a toy.
    """

    last_init_kwargs = None

    def __init__(self, model, threshold=0.5, sampling_rate=16000,
                 min_silence_duration_ms=100, speech_pad_ms=30):
        _FakeVADIterator.last_init_kwargs = dict(
            threshold=threshold, sampling_rate=sampling_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, return_seconds=False, time_resolution=1):
        n = len(x[0]) if hasattr(x, "dim") and x.dim() == 2 else len(x)
        first = float(x[0])
        self.current_sample += n
        speech_prob = 1.0 if first >= 0.5 else 0.0

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0
        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            start = max(0, self.current_sample - self.speech_pad_samples - n)
            t = round(start / self.sampling_rate, time_resolution) if return_seconds else int(start)
            return {"start": t}
        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            end = self.temp_end + self.speech_pad_samples - n
            self.temp_end = 0
            self.triggered = False
            t = round(end / self.sampling_rate, time_resolution) if return_seconds else int(end)
            return {"end": t}
        return None


def _install_fake(monkeypatch):
    mod = type(sys)("silero_vad")
    mod.load_silero_vad = lambda *a, **k: "FAKE_MODEL"
    mod.VADIterator = _FakeVADIterator
    monkeypatch.setitem(sys.modules, "silero_vad", mod)
    # torch must be the real thing (SileroStream buffers with it); it is present.
    import vad.silero as sv
    monkeypatch.setattr(sv, "_MODEL", None)


def _loud(n):
    return np.full(n, 0.9, dtype=np.float32)


def _quiet(n):
    return np.zeros(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# StreamEvent / _events_to_segments — pure helpers, no model needed.
# ---------------------------------------------------------------------------


class TestStreamEvent:
    def test_to_dict_rounds_and_labels(self):
        assert StreamEvent("start", 1.23456).to_dict() == {"type": "start", "time_s": 1.235}
        assert StreamEvent("end", 4.0).to_dict() == {"type": "end", "time_s": 4.0}


class TestEventsToSegments:
    def test_pairs_start_end(self):
        evs = [StreamEvent("start", 1.0), StreamEvent("end", 2.0),
               StreamEvent("start", 5.0), StreamEvent("end", 6.5)]
        segs = _events_to_segments(evs)
        assert segs == [SpeechSegment(1.0, 2.0), SpeechSegment(5.0, 6.5)]

    def test_drops_dangling_start(self):
        # A leftover start with no end (caller forgot to flush) is dropped.
        segs = _events_to_segments([StreamEvent("start", 1.0)])
        assert segs == []

    def test_ignores_orphan_end(self):
        segs = _events_to_segments([StreamEvent("end", 2.0)])
        assert segs == []

    def test_empty(self):
        assert _events_to_segments([]) == []


# ---------------------------------------------------------------------------
# SileroStream construction + param plumbing.
# ---------------------------------------------------------------------------


class TestSileroStreamConstruction:
    def test_rejects_unsupported_sample_rate(self, monkeypatch):
        _install_fake(monkeypatch)
        with pytest.raises(ValueError):
            SileroStream(model="M", sample_rate=44100)

    def test_window_size_for_16k(self, monkeypatch):
        _install_fake(monkeypatch)
        s = SileroStream(model="M", sample_rate=16000)
        assert s.window == 512 == WINDOW_SAMPLES

    def test_window_size_for_8k(self, monkeypatch):
        _install_fake(monkeypatch)
        s = SileroStream(model="M", sample_rate=8000)
        assert s.window == 256

    def test_maps_params_to_vaditerator(self, monkeypatch):
        _install_fake(monkeypatch)
        SileroStream(
            params=SileroParams(threshold=0.7, min_silence_ms=900, speech_pad_ms=40),
            model="M",
        )
        kw = _FakeVADIterator.last_init_kwargs
        assert kw["threshold"] == 0.7
        assert kw["min_silence_duration_ms"] == 900
        assert kw["speech_pad_ms"] == 40
        assert kw["sampling_rate"] == 16000

    def test_lazy_loads_model_when_none(self, monkeypatch):
        _install_fake(monkeypatch)
        # _MODEL reset by _install_fake; constructing with no model triggers load.
        s = SileroStream(sample_rate=16000)
        assert s._iter.model == "FAKE_MODEL"


# ---------------------------------------------------------------------------
# push() — windowing, buffering, event emission.
# ---------------------------------------------------------------------------


class TestSileroStreamPush:
    def _stream(self, monkeypatch, **pkw):
        _install_fake(monkeypatch)
        return SileroStream(params=SileroParams(**pkw), model="M", sample_rate=16000)

    def test_silence_only_emits_nothing(self, monkeypatch):
        s = self._stream(monkeypatch)
        events = s.push(_quiet(WINDOW_SAMPLES * 5))
        assert events == []
        assert s.triggered is False

    def test_speech_onset_emits_start_and_sets_triggered(self, monkeypatch):
        s = self._stream(monkeypatch, speech_pad_ms=0.0)
        events = s.push(_loud(WINDOW_SAMPLES * 3))
        kinds = [e.kind for e in events]
        assert kinds == ["start"]
        assert s.triggered is True

    def test_speech_then_silence_closes_segment(self, monkeypatch):
        # min_silence 0 → first quiet window after speech closes the region.
        s = self._stream(monkeypatch, min_silence_ms=0.0, speech_pad_ms=0.0)
        s.push(_loud(WINDOW_SAMPLES * 2))
        events = s.push(_quiet(WINDOW_SAMPLES * 2))
        assert [e.kind for e in events] == ["end"]
        assert s.triggered is False

    def test_buffers_sub_window_remainder_across_pushes(self, monkeypatch):
        s = self._stream(monkeypatch, speech_pad_ms=0.0)
        # Push less than a full window: no window consumed yet → no event.
        assert s.push(_loud(WINDOW_SAMPLES - 10)) == []
        assert s.triggered is False
        # The remaining 10 samples complete the window on the next push.
        events = s.push(_loud(10))
        assert [e.kind for e in events] == ["start"]

    def test_chunk_size_independence(self, monkeypatch):
        # Same audio, fed in tiny vs large chunks, yields identical events.
        audio = np.concatenate([_loud(WINDOW_SAMPLES * 3),
                                 _quiet(WINDOW_SAMPLES * 3)])

        s1 = self._stream(monkeypatch, min_silence_ms=0.0, speech_pad_ms=0.0)
        ev_big = s1.push(audio)

        s2 = self._stream(monkeypatch, min_silence_ms=0.0, speech_pad_ms=0.0)
        ev_small = []
        for i in range(0, len(audio), 37):  # ragged, non-window-aligned chunks
            ev_small.extend(s2.push(audio[i:i + 37]))

        assert [(e.kind, round(e.time_s, 3)) for e in ev_big] == \
               [(e.kind, round(e.time_s, 3)) for e in ev_small]

    def test_empty_push_is_noop(self, monkeypatch):
        s = self._stream(monkeypatch)
        assert s.push(np.zeros(0, dtype=np.float32)) == []


# ---------------------------------------------------------------------------
# flush() — close a segment left open at end-of-stream.
# ---------------------------------------------------------------------------


class TestSileroStreamFlush:
    def _stream(self, monkeypatch, **pkw):
        _install_fake(monkeypatch)
        return SileroStream(params=SileroParams(**pkw), model="M", sample_rate=16000)

    def test_flush_closes_open_segment_at_total_elapsed(self, monkeypatch):
        s = self._stream(monkeypatch, speech_pad_ms=0.0)
        s.push(_loud(WINDOW_SAMPLES * 4))  # opens, never closes
        assert s.triggered is True
        events = s.flush()
        assert len(events) == 1
        assert events[0].kind == "end"
        # 4 windows * 512 / 16000 s elapsed.
        assert events[0].time_s == pytest.approx(WINDOW_SAMPLES * 4 / 16000, abs=1e-3)
        assert s.triggered is False

    def test_flush_is_noop_when_not_triggered(self, monkeypatch):
        s = self._stream(monkeypatch)
        s.push(_quiet(WINDOW_SAMPLES * 2))
        assert s.flush() == []

    def test_flush_is_idempotent(self, monkeypatch):
        s = self._stream(monkeypatch, speech_pad_ms=0.0)
        s.push(_loud(WINDOW_SAMPLES * 2))
        assert len(s.flush()) == 1
        assert s.flush() == []  # second flush closes nothing


# ---------------------------------------------------------------------------
# reset() — re-arm for a new utterance.
# ---------------------------------------------------------------------------


class TestSileroStreamReset:
    def test_reset_clears_state_and_buffer(self, monkeypatch):
        _install_fake(monkeypatch)
        s = SileroStream(params=SileroParams(speech_pad_ms=0.0), model="M", sample_rate=16000)
        s.push(_loud(WINDOW_SAMPLES - 5))  # leave a buffered remainder + would-trigger
        s.push(_loud(WINDOW_SAMPLES * 2))
        assert s.triggered is True
        s.reset()
        assert s.triggered is False
        assert s._total_samples == 0
        # After reset, the buffered remainder is gone: a fresh sub-window push
        # emits nothing.
        assert s.push(_quiet(WINDOW_SAMPLES - 1)) == []


# ---------------------------------------------------------------------------
# stream_samples() — end-to-end reconstruction into a SileroResult.
# ---------------------------------------------------------------------------


class TestStreamSamples:
    def test_empty_audio_returns_empty_result(self, monkeypatch):
        _install_fake(monkeypatch)
        r = stream_samples(np.zeros(0, dtype=np.float32), 16000, model="M")
        assert r.num_segments == 0
        assert r.duration_s == 0.0

    def test_reconstructs_two_segments_with_flush(self, monkeypatch):
        _install_fake(monkeypatch)
        # speech | silence(closes) | speech (stays open → flush closes it)
        audio = np.concatenate([
            _loud(WINDOW_SAMPLES * 3),
            _quiet(WINDOW_SAMPLES * 3),
            _loud(WINDOW_SAMPLES * 2),
        ])
        r = stream_samples(
            audio, 16000,
            params=SileroParams(min_silence_ms=0.0, speech_pad_ms=0.0),
            model="M",
        )
        assert r.num_segments == 2
        # Both segments have positive duration and are ordered.
        assert all(seg.duration_s > 0 for seg in r.segments)
        assert r.segments[1].start_s >= r.segments[0].end_s

    def test_duration_matches_audio_length(self, monkeypatch):
        _install_fake(monkeypatch)
        audio = _quiet(16000)  # 1s at 16k
        r = stream_samples(audio, 16000, model="M")
        assert r.duration_s == pytest.approx(1.0, abs=0.01)
        assert r.num_segments == 0


# ---------------------------------------------------------------------------
# decode_float32_le — wire-format decode for the WebSocket binary frames.
# ---------------------------------------------------------------------------


class TestDecodeFloat32LE:
    def test_roundtrips_float32_little_endian(self):
        arr = np.array([0.1, -0.2, 0.9, 0.0], dtype=np.float32)
        out = decode_float32_le(arr.tobytes())
        assert np.allclose(out, arr, atol=1e-6)
        assert len(out) == 4

    def test_empty_bytes(self):
        assert decode_float32_le(b"") == []

    def test_ignores_trailing_partial_sample(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        # 8 bytes for two floats + 2 stray bytes that don't complete a sample.
        out = decode_float32_le(arr.tobytes() + b"\x00\x01")
        assert len(out) == 2
        assert np.allclose(out, arr, atol=1e-6)


# ---------------------------------------------------------------------------
# StreamProtocol — the WebSocket message state machine (server is thin glue).
# ---------------------------------------------------------------------------


class _RecordingStream:
    """A fake SileroStream that records calls, so protocol dispatch is asserted
    without the model. Mirrors the SileroStream public surface the protocol uses.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.pushed = []
        self.flushed = 0
        self.reset_count = 0

    def push(self, samples):
        self.pushed.append(list(samples))
        # Emit a start the first time we see any sample, for assertion.
        return [StreamEvent("start", 0.0)] if len(samples) else []

    def flush(self):
        self.flushed += 1
        return [StreamEvent("end", 1.0)] if self.flushed == 1 else []

    def reset(self):
        self.reset_count += 1


class TestStreamProtocol:
    def _proto(self):
        made = []

        def factory(cfg):
            s = _RecordingStream(cfg)
            made.append(s)
            return s

        return StreamProtocol(factory), made

    def test_starts_unarmed(self):
        proto, _ = self._proto()
        assert proto.armed is False

    def test_config_message_arms_stream(self):
        proto, made = self._proto()
        reply = proto.handle_text({"threshold": 0.6, "sample_rate": 16000})
        assert reply == {"events": [], "armed": True}
        assert proto.armed is True
        assert made[0].cfg == {"threshold": 0.6, "sample_rate": 16000}

    def test_binary_before_config_arms_default_stream(self):
        proto, made = self._proto()
        arr = np.array([0.5, 0.5], dtype=np.float32)
        reply = proto.handle_binary(arr.tobytes())
        assert proto.armed is True
        assert made[0].cfg == {}  # default-armed
        assert reply["events"] == [{"type": "start", "time_s": 0.0}]
        assert made[0].pushed[0] == pytest.approx([0.5, 0.5])

    def test_flush_command_closes_segment(self):
        proto, made = self._proto()
        proto.handle_text({})  # arm
        reply = proto.handle_text({"cmd": "flush"})
        assert reply == {"events": [{"type": "end", "time_s": 1.0}], "flushed": True}
        assert made[0].flushed == 1

    def test_flush_before_arm_is_safe(self):
        proto, _ = self._proto()
        reply = proto.handle_text({"cmd": "flush"})
        assert reply == {"events": [], "flushed": True}

    def test_reset_command_rearms_same_stream(self):
        proto, made = self._proto()
        proto.handle_text({})  # arm
        reply = proto.handle_text({"cmd": "reset"})
        assert reply == {"events": [], "reset": True}
        assert made[0].reset_count == 1
        assert len(made) == 1  # reset does NOT build a new stream

    def test_reset_before_arm_is_safe(self):
        proto, _ = self._proto()
        reply = proto.handle_text({"cmd": "reset"})
        assert reply == {"events": [], "reset": True}

    def test_reconfig_builds_a_fresh_stream(self):
        proto, made = self._proto()
        proto.handle_text({"threshold": 0.5})
        proto.handle_text({"threshold": 0.9})
        assert len(made) == 2
        assert made[1].cfg == {"threshold": 0.9}

    def test_binary_frame_pushes_decoded_samples(self):
        proto, made = self._proto()
        proto.handle_text({})  # arm
        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        proto.handle_binary(arr.tobytes())
        assert made[0].pushed[-1] == pytest.approx([0.1, 0.2, 0.3])

    def test_empty_binary_frame_pushes_nothing_meaningful(self):
        proto, made = self._proto()
        proto.handle_text({})  # arm
        reply = proto.handle_binary(b"")
        # Empty decode → push([]) → no events.
        assert reply == {"events": []}
