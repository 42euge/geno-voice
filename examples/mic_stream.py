"""Streaming transcription mode — live progressive output with confidence display.

Text appears dim while speculative, goes bright once it stabilizes across
consecutive inference passes. A debug timing strip shows pipeline stats.
"""

import io
import os
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stt import WhisperEngine

RATE = 16000
CHANNELS = 1
CHUNK = 1024
SILENCE_THRESHOLD = 0.02
SILENCE_DURATION = 1.5
INFERENCE_INTERVAL = 1.0
STABILITY_PASSES = 2

DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K"
UP = "\033[A"


@dataclass
class TranscriptState:
    stable: str = ""
    speculative: str = ""
    passes: int = 0
    last_change_at: float = 0.0
    collapse_times: list = field(default_factory=list)
    infer_times: list = field(default_factory=list)
    settled_at_pass: int = 0
    speech_start: float = 0.0

    @property
    def full_text(self):
        return self.stable + self.speculative


def _longest_common_prefix(a: str, b: str) -> str:
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return a[:i]


@dataclass
class StabilizeStep:
    """Result of one per-pass stabilization step.

    Carries the updated stabilization locals back to the caller. ``changed``
    and ``promoted`` are signals the caller acts on with its own clock —
    time side-effects (``state.last_change_at``, ``state.collapse_times``)
    stay at the entrypoint, mirroring the GENO.md extraction convention
    (keep wall-clock / presentation owned by the caller).
    """

    stable: str
    speculative: str
    stable_candidate: str
    stable_count: int
    settled_at_pass: int
    prev_full_text: str
    changed: bool
    promoted: bool


def stabilize_pass(
    text: str,
    prev_full_text: str,
    stable: str,
    stable_candidate: str,
    stable_count: int,
    passes: int,
    settled_at_pass: int,
    *,
    stability_passes: int = STABILITY_PASSES,
) -> StabilizeStep:
    """Pure per-pass stabilization step for streaming transcription.

    Implements the iter-008 streaming-overlap design: track the longest
    common prefix across consecutive inference passes; once a prefix has
    held unchanged for ``stability_passes`` passes (and is longer than the
    current stable text), promote it to ``stable``. Everything past the
    stable prefix is ``speculative`` (rendered dim until it settles).

    No I/O and no clock reads — the caller applies wall-clock side-effects
    when ``changed`` / ``promoted`` are set, so this is fully unit-testable.
    """
    common = _longest_common_prefix(text, prev_full_text)

    changed = common != stable_candidate
    if changed:
        stable_candidate = common
        stable_count = 1
    else:
        stable_count += 1

    promoted = False
    if stable_count >= stability_passes and len(stable_candidate) > len(stable):
        stable = stable_candidate
        settled_at_pass = passes
        promoted = True

    if text.startswith(stable):
        speculative = text[len(stable):]
    else:
        speculative = text

    return StabilizeStep(
        stable=stable,
        speculative=speculative,
        stable_candidate=stable_candidate,
        stable_count=stable_count,
        settled_at_pass=settled_at_pass,
        prev_full_text=text,
        changed=changed,
        promoted=promoted,
    )


def _transcribe(engine: WhisperEngine, wav_bytes: bytes) -> tuple[str | None, float]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp = f.name
    t0 = time.monotonic()
    try:
        import mlx_whisper
        result = mlx_whisper.transcribe(tmp, path_or_hf_repo=engine.model_repo)
        elapsed = time.monotonic() - t0
        return result["text"].strip(), elapsed
    except Exception:
        return None, time.monotonic() - t0
    finally:
        os.unlink(tmp)


def _buffer_to_wav(frames: list[bytes]) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


def _render(state: TranscriptState, debug: str):
    """Redraw transcript + debug strip in-place (2 lines)."""
    sys.stdout.write(f"\r{CLEAR_LINE}")
    sys.stdout.write(f"  {BOLD}{state.stable}{RESET}{DIM}{state.speculative}{RESET}")
    sys.stdout.write(f"\n{CLEAR_LINE}")
    sys.stdout.write(f"  {DIM}{debug}{RESET}")
    sys.stdout.write(f"{UP}\r")
    sys.stdout.flush()


def _render_collapse(state: TranscriptState):
    """Print final collapsed line and stats."""
    total_time = time.monotonic() - state.speech_start if state.speech_start else 0
    avg_infer = sum(state.infer_times) / len(state.infer_times) if state.infer_times else 0

    sys.stdout.write(f"\r{CLEAR_LINE}")
    sys.stdout.write(f"  {GREEN}{state.stable}{RESET}")
    sys.stdout.write(f"\n{CLEAR_LINE}")
    infer_str = "/".join(f"{t*1000:.0f}" for t in state.infer_times[-6:])
    sys.stdout.write(
        f"  {DIM}│ ✓ {total_time:.1f}s │ "
        f"{state.passes} passes │ "
        f"avg: {avg_infer*1000:.0f}ms │ "
        f"infer: [{infer_str}]ms │ "
        f"settled: pass {state.settled_at_pass} │{RESET}"
    )
    sys.stdout.write("\n\n")
    sys.stdout.flush()


def run_stream(model_repo: str):
    """Main streaming loop."""
    model_short = model_repo.split("/")[-1]
    print(f"\n{BOLD}gv stream{RESET}")
    print(f"{DIM}model: {model_short}{RESET}")
    print()

    t_load = time.monotonic()
    engine = WhisperEngine(model_repo=model_repo)
    engine._load()
    print(f"  Model loaded in {(time.monotonic() - t_load)*1000:.0f}ms")
    print(f"  {GREEN}Ready.{RESET} Speak into your mic. Ctrl+C to quit.\n")

    # Lazy import — keeps the module importable on hosts without pyaudio
    # (e.g. Linux x86_64) so the pure helpers can be unit-tested.
    import pyaudio

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=CHANNELS,
                     rate=RATE, input=True, frames_per_buffer=CHUNK)

    audio_frames = []
    state = TranscriptState()
    speaking = False
    silence_start = None
    prev_full_text = ""
    stable_candidate = ""
    stable_count = 0
    last_inference_at = 0.0
    inference_lock = threading.Lock()
    utterance_count = 0
    all_infer_times = []

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            level = float(np.sqrt(np.mean(audio ** 2)))

            if level > SILENCE_THRESHOLD:
                if not speaking:
                    speaking = True
                    state = TranscriptState(speech_start=time.monotonic())
                    prev_full_text = ""
                    stable_candidate = ""
                    stable_count = 0
                    last_inference_at = time.monotonic()
                    # Reserve 2 lines
                    sys.stdout.write("\n\n" + UP + UP)
                    sys.stdout.flush()
                silence_start = None
                audio_frames.append(data)

            elif speaking:
                audio_frames.append(data)
                if silence_start is None:
                    silence_start = time.monotonic()
                elif time.monotonic() - silence_start >= SILENCE_DURATION:
                    # Collapse — final pass
                    if audio_frames:
                        wav = _buffer_to_wav(audio_frames)
                        text, elapsed = _transcribe(engine, wav)
                        state.infer_times.append(elapsed)
                        if text:
                            state.stable = text
                            state.speculative = ""
                    _render_collapse(state)
                    all_infer_times.extend(state.infer_times)
                    utterance_count += 1
                    audio_frames = []
                    speaking = False
                    silence_start = None
                    continue

            # Periodic inference while speaking
            if speaking and (time.monotonic() - last_inference_at) >= INFERENCE_INTERVAL and audio_frames:
                last_inference_at = time.monotonic()
                wav = _buffer_to_wav(audio_frames)
                text, elapsed = _transcribe(engine, wav)
                state.infer_times.append(elapsed)
                state.passes += 1

                if text:
                    step = stabilize_pass(
                        text,
                        prev_full_text,
                        state.stable,
                        stable_candidate,
                        stable_count,
                        state.passes,
                        state.settled_at_pass,
                    )
                    stable_candidate = step.stable_candidate
                    stable_count = step.stable_count
                    state.stable = step.stable
                    state.speculative = step.speculative
                    state.settled_at_pass = step.settled_at_pass
                    prev_full_text = step.prev_full_text

                    # Wall-clock side-effects stay at the caller.
                    if step.changed:
                        state.last_change_at = time.monotonic()
                    if step.promoted:
                        state.collapse_times.append(time.monotonic())

                # Render
                since_change = time.monotonic() - state.last_change_at if state.last_change_at else 0
                debug = (
                    f"│ pass {state.passes} │ "
                    f"infer: {elapsed*1000:.0f}ms │ "
                    f"stable: {len(state.stable)}ch │ "
                    f"spec: {len(state.speculative)}ch │ "
                    f"last Δ: {since_change:.1f}s ago │"
                )
                _render(state, debug)

    except KeyboardInterrupt:
        # Move past the 2 reserved lines
        sys.stdout.write(f"\n\n")
        print(f"{DIM}{'─' * 50}{RESET}")
        if all_infer_times:
            sorted_times = sorted(all_infer_times)
            median = sorted_times[len(sorted_times) // 2]
            print(f"{BOLD}  Session Summary ({utterance_count} utterances){RESET}")
            print(f"    Median infer:  {median*1000:.0f}ms")
            print(f"    Min infer:     {min(all_infer_times)*1000:.0f}ms")
            print(f"    Max infer:     {max(all_infer_times)*1000:.0f}ms")
            print(f"    Total passes:  {len(all_infer_times)}")
            print(f"    Model:         {model_short}")
        print()
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
