"""Tests for iter-109 — build_audio_io factory.

The factory wires three callables (`speaker_factory`, `synth_fn`,
`play_fn`) to PyAudio + kokoro + _play_aligned_core. Tests pass:
  - A stub `pa` exposing `.open()` that records its kwargs
  - A stub `pyaudio_module` exposing `paInt16`
  - A stub `tts_engine` (passed straight through to
    synthesize_with_alignment, which is patched on the module)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._chat_audio_io import AudioIO, build_audio_io  # noqa: E402


# ---- Stubs ----------------------------------------------------------------


class _RecordingPyAudio:
    """Stand-in for `pyaudio.PyAudio()`. Captures every `open()`
    call in `opens` so tests can assert on the kwargs."""

    def __init__(self):
        self.opens: list[dict] = []

    def open(self, **kwargs):
        self.opens.append(kwargs)
        return f"speaker-{len(self.opens)}"


class _PyAudioModule:
    """Stub of the `pyaudio` module. Only `paInt16` is read."""

    paInt16 = "paInt16"


# ---- Return type ----------------------------------------------------------


def test_returns_audio_io_dataclass():
    """Result is an AudioIO with the three named callables."""
    pa = _RecordingPyAudio()
    audio_io = build_audio_io(
        pa, "tts_stub", "voice_a", 1.0, pyaudio_module=_PyAudioModule(),
    )
    assert isinstance(audio_io, AudioIO)
    assert callable(audio_io.speaker_factory)
    assert callable(audio_io.synth_fn)
    assert callable(audio_io.play_fn)


# ---- speaker_factory -------------------------------------------------------


def test_speaker_factory_passes_expected_kwargs_to_pa_open():
    """Each call to speaker_factory delegates to pa.open with
    the same kwargs the original inline closure used."""
    pa = _RecordingPyAudio()
    audio_io = build_audio_io(
        pa, "tts_stub", "voice_a", 1.0, pyaudio_module=_PyAudioModule(),
    )
    audio_io.speaker_factory()
    assert len(pa.opens) == 1
    kwargs = pa.opens[0]
    assert kwargs["format"] == "paInt16"
    assert kwargs["channels"] == 1
    assert kwargs["output"] is True
    assert kwargs["frames_per_buffer"] == 1024
    # rate defaults to TTS_RATE imported from _chat_playback.
    assert kwargs["rate"] > 0


def test_speaker_factory_returns_pa_open_result():
    """Whatever pa.open() returns, the factory returns. The
    SentenceWorker treats it as opaque."""
    pa = _RecordingPyAudio()
    audio_io = build_audio_io(
        pa, "tts_stub", "voice_a", 1.0, pyaudio_module=_PyAudioModule(),
    )
    result = audio_io.speaker_factory()
    # _RecordingPyAudio returns "speaker-1" on first open.
    assert result == "speaker-1"


def test_speaker_factory_can_be_called_multiple_times():
    """One factory call per sentence — must not get stuck on
    state from the first call."""
    pa = _RecordingPyAudio()
    audio_io = build_audio_io(
        pa, "tts_stub", "voice_a", 1.0, pyaudio_module=_PyAudioModule(),
    )
    audio_io.speaker_factory()
    audio_io.speaker_factory()
    audio_io.speaker_factory()
    assert len(pa.opens) == 3


def test_speaker_chunk_kwarg_overrides_default():
    """Operators tuning latency might want a smaller chunk."""
    pa = _RecordingPyAudio()
    audio_io = build_audio_io(
        pa, "tts_stub", "voice_a", 1.0,
        pyaudio_module=_PyAudioModule(),
        speaker_chunk=256,
    )
    audio_io.speaker_factory()
    assert pa.opens[0]["frames_per_buffer"] == 256


def test_rate_kwarg_overrides_default():
    """Ditto for sample rate — useful for non-kokoro TTS."""
    pa = _RecordingPyAudio()
    audio_io = build_audio_io(
        pa, "tts_stub", "voice_a", 1.0,
        pyaudio_module=_PyAudioModule(),
        rate=16000,
    )
    audio_io.speaker_factory()
    assert pa.opens[0]["rate"] == 16000


# ---- synth_fn -------------------------------------------------------------


def test_synth_fn_invokes_synthesize_with_alignment():
    """synth_fn delegates to synthesize_with_alignment with
    (engine, sentence, voice, speed) — the same wiring the
    original inline closure had."""
    pa = _RecordingPyAudio()
    audio_io = build_audio_io(
        pa, "tts_stub", "voice_a", 1.5, pyaudio_module=_PyAudioModule(),
    )
    with patch(
        "examples._chat_audio_io.synthesize_with_alignment",
    ) as mock_synth:
        mock_synth.return_value = ("audio_data", "tokens_data")
        result = audio_io.synth_fn("hello world")

    assert result == ("audio_data", "tokens_data")
    mock_synth.assert_called_once_with(
        "tts_stub", "hello world", "voice_a", 1.5,
    )


def test_synth_fn_speed_is_threaded_through():
    """Different speed value flows to synthesize_with_alignment."""
    pa = _RecordingPyAudio()
    audio_io = build_audio_io(
        pa, "tts_stub", "voice_a", 0.8, pyaudio_module=_PyAudioModule(),
    )
    with patch(
        "examples._chat_audio_io.synthesize_with_alignment",
    ) as mock_synth:
        mock_synth.return_value = (None, None)
        audio_io.synth_fn("test")
    args = mock_synth.call_args.args
    assert args[3] == 0.8


# ---- play_fn --------------------------------------------------------------


def test_play_fn_invokes_play_aligned_core():
    """play_fn delegates to _play_aligned_core with the same
    forward-arguments + the rate kwarg threaded through."""
    pa = _RecordingPyAudio()
    audio_io = build_audio_io(
        pa, "tts_stub", "voice_a", 1.0, pyaudio_module=_PyAudioModule(),
    )
    with patch(
        "examples._chat_audio_io._play_aligned_core",
    ) as mock_play:
        mock_play.return_value = 0.5
        result = audio_io.play_fn("speaker_obj", "audio_np", "tokens")

    assert result == 0.5
    args, kwargs = mock_play.call_args
    assert args == ("speaker_obj", "audio_np", "tokens")
    assert kwargs["is_first_sentence"] is False
    assert kwargs["cancel_event"] is None
    assert kwargs["lag_out"] is None


def test_play_fn_threads_through_optional_kwargs():
    """is_first_sentence / cancel_event / lag_out are forwarded
    intact — the SentenceWorker uses them for ttfs + barge-in
    + reveal-lag tracking."""
    pa = _RecordingPyAudio()
    audio_io = build_audio_io(
        pa, "tts_stub", "voice_a", 1.0, pyaudio_module=_PyAudioModule(),
    )
    cancel_evt = object()
    lag_collector = []
    with patch(
        "examples._chat_audio_io._play_aligned_core",
    ) as mock_play:
        mock_play.return_value = 0.0
        audio_io.play_fn(
            "speaker", "audio", "tokens",
            is_first_sentence=True,
            cancel_event=cancel_evt,
            lag_out=lag_collector,
        )
    kwargs = mock_play.call_args.kwargs
    assert kwargs["is_first_sentence"] is True
    assert kwargs["cancel_event"] is cancel_evt
    assert kwargs["lag_out"] is lag_collector


def test_play_fn_passes_rate_kwarg():
    """The rate kwarg from build_audio_io is captured in the
    closure and forwarded into _play_aligned_core."""
    pa = _RecordingPyAudio()
    audio_io = build_audio_io(
        pa, "tts_stub", "voice_a", 1.0,
        pyaudio_module=_PyAudioModule(),
        rate=16000,
    )
    with patch(
        "examples._chat_audio_io._play_aligned_core",
    ) as mock_play:
        mock_play.return_value = 0.0
        audio_io.play_fn("speaker", "audio", "tokens")
    assert mock_play.call_args.kwargs["rate"] == 16000


# ---- pyaudio_module fallback ---------------------------------------------


def test_no_pyaudio_module_kwarg_uses_lazy_import():
    """When pyaudio_module is omitted, the factory still
    constructs (the lazy import only fires when speaker_factory
    is CALLED, not when build_audio_io itself runs)."""
    pa = _RecordingPyAudio()
    # No exception even though pyaudio may not be installed in
    # CI — the import is inside speaker_factory.
    audio_io = build_audio_io(pa, "tts_stub", "voice_a", 1.0)
    assert isinstance(audio_io, AudioIO)


# ---- iter-214: callable speed (WPM-mirroring live wiring) -----------------


def test_synth_fn_accepts_callable_speed_and_resolves_per_call():
    """A zero-arg callable for `speed` is invoked fresh on every synth, so a
    speed updated between turns takes effect on the next sentence."""
    pa = _RecordingPyAudio()
    speeds = iter([1.0, 1.2, 0.9])
    audio_io = build_audio_io(
        pa, "tts_stub", "voice_a", lambda: next(speeds),
        pyaudio_module=_PyAudioModule(),
    )
    with patch(
        "examples._chat_audio_io.synthesize_with_alignment",
    ) as mock_synth:
        mock_synth.return_value = (None, None)
        audio_io.synth_fn("one")
        audio_io.synth_fn("two")
        audio_io.synth_fn("three")
    # The 4th positional arg (speed) tracks the callable's successive returns.
    seen = [call.args[3] for call in mock_synth.call_args_list]
    assert seen == [1.0, 1.2, 0.9]


def test_synth_fn_float_speed_is_constant_per_sentence():
    """A plain float for `speed` (the historical shape) is used unchanged on
    every sentence — the proven constant-rate path."""
    pa = _RecordingPyAudio()
    audio_io = build_audio_io(
        pa, "tts_stub", "voice_a", 1.1, pyaudio_module=_PyAudioModule(),
    )
    with patch(
        "examples._chat_audio_io.synthesize_with_alignment",
    ) as mock_synth:
        mock_synth.return_value = (None, None)
        audio_io.synth_fn("a")
        audio_io.synth_fn("b")
    seen = [call.args[3] for call in mock_synth.call_args_list]
    assert seen == [1.1, 1.1]


def test_callable_speed_reflects_live_mutation_via_controller():
    """End-to-end with the real SpeedController: mutating the controller
    between synth calls changes the speed the synth path sees."""
    from examples._chat_speed import SpeedController

    class _Mirror:
        def speed(self, *, user_wpm, current_speed):
            return 1.3

    controller = SpeedController(1.0, mirror=_Mirror())
    pa = _RecordingPyAudio()
    audio_io = build_audio_io(
        pa, "tts_stub", "voice_a", controller.current,
        pyaudio_module=_PyAudioModule(),
    )
    with patch(
        "examples._chat_audio_io.synthesize_with_alignment",
    ) as mock_synth:
        mock_synth.return_value = (None, None)
        audio_io.synth_fn("before")   # speed 1.0
        controller.observe(220.0)     # mirror bumps to 1.3
        audio_io.synth_fn("after")    # speed 1.3
    seen = [call.args[3] for call in mock_synth.call_args_list]
    assert seen == [1.0, 1.3]
