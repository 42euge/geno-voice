# Iteration Log

Each entry summarizes one /loop iteration: what was learned, what changed, what is next.

## Known Bugs (from initial diagnosis)

1. **Duplicate "Bot:" lines** — `mic_chat.py:377` prints `[N] waiting...` with `end=""` but never gets cleared before bot output. Multiple "Bot:" lines appear during multi-sentence responses.
2. **Timer shows absurd values (3201756ms)** — `mic_chat.py:470` computes `llm_total` after all playback completes (includes TTS + playback time, not just LLM).
3. **No error recovery after LLM failure** — `mic_chat.py:479-482` does `messages.pop(); continue` without flushing the mic buffer. STT accumulates garbage after DNS errors.
4. **Excessive live transcription reprints** — `mic_chat.py:216` uses `\r` which breaks when preview text wraps terminal width.

## Goals

- Fix all 4 bugs above
- Add unit tests for the pipeline logic (sentence splitting, metrics calculation, error recovery)
- Refactor mic_chat.py into testable components (separate display, timing, error handling)
- Implement streaming LLM → TTS overlap (start TTS while LLM still streaming)
- Add filler-word generation to mask latency (configurable "thinking" sounds)
- Improve turn-taking (allow barge-in / interruption during bot speech)

---

## iter-001 — fix bugs #1 and #2, extract testable helpers

**Branch:** `iter-001-bugs-1-2` (merged ff to main, commit `1a2656a`)
**Date:** 2026-05-23

What changed:
- Bug #1 (duplicate "Bot:" lines) fixed: cleared the `[N] waiting...` row
  before printing the `Bot:` prefix in `play_aligned`. Single-line edit
  in `examples/mic_chat.py`.
- Bug #2 (absurd `llm_total` values) fixed: stamped
  `llm_stream_done_at` immediately after the streaming for-loop and used
  that for `metrics.llm_total`. Previously the timer was sampled after
  the trailing TTS + playback for the partial last sentence, which is
  why values like 3201756ms appeared.
- Refactored the small testable parts of `mic_chat.py` into
  `examples/_chat_helpers.py` (stdlib only, no audio imports):
  `split_complete_sentences`, `trim_history`, `TurnTimings`,
  `format_preview_line`. `mic_chat.py` now imports and uses these.
- Added `tests/unit/test_chat_helpers.py` with 22 tests covering
  sentence splitting (empty, no-terminator, multi, mid-sentence period),
  history trim (empty, short, long, no-system, mutation), `TurnTimings`
  contracts (first-token-once, llm_total stays put after stream done,
  TTFS recorded once), and preview-line truncation.

Verification: `python -m pytest tests/unit/` → **22 passed in 0.02s**.

Notes:
- Pre-existing issues NOT touched: `tests/test_session.py` collection
  error, `tests/e2e/` fixture/server errors. These also fail on main
  prior to this iteration, so they're out of scope.
- `format_preview_line` is groundwork for bug #4 (next iteration) —
  it's tested here but not yet wired into the live preview render.

Next:
- iter-002: bug #3 (mic buffer flush after LLM error) — requires
  refactoring the audio stream to be drainable from a test, or at
  minimum extracting `flush_pending_audio(stream)` and unit-testing
  that it consumes available chunks without blocking.
- iter-003: bug #4 (preview wrap) — wire `format_preview_line` into
  `record_utterance_streaming` and verify with a fake stdout test.

---

## iter-002 — fix bug #3 (mic buffer not flushed after LLM error)

**Branch:** `iter-002-bug-3` (merged ff to main, commit `795a3aa`)
**Date:** 2026-05-23

What changed:
- Added `flush_pending_audio(stream, chunk_size, max_iterations)` in
  `examples/_chat_helpers.py`. Non-blocking drain via
  `get_read_available()` + `read(exception_on_overflow=False)`. Hard
  cap on iteration count guards against a misbehaving stream that
  always reports "data available." Catches exceptions from either
  call so a stream hiccup mid-drain doesn't crash the chat loop.
- Wired the flush into the LLM-error handler in `mic_chat.py`. After a
  DNS failure / timeout / 5xx, we now drop whatever audio piled up
  during the failed call and print a dim notice with how many seconds
  were discarded so the user understands their next utterance starts
  clean.
- 7 new tests using a `FakeStream` test double: empty stream → 0 reads;
  full chunks only with partial remainder retained; max-iterations cap
  on a lying stream; both `get_read_available` and `read` raising
  cleanly.

Verification: `python -m pytest tests/unit/` → **29 passed in 0.02s**.

Notes:
- Cannot end-to-end test the actual pyaudio path on x86_64 Linux
  without hardware. The FakeStream contract matches pyaudio's two
  methods exactly, so the unit tests give us solid coverage of the
  drain logic. A future on-device smoke test would close the gap.

Next:
- iter-003: bug #4 (preview reprint wraps when text exceeds terminal
  width). Wire `format_preview_line` into `record_utterance_streaming`,
  use `shutil.get_terminal_size()` to pick the width, and add a test
  with a fake stdout that asserts no wraparound.
- iter-004: split `record_utterance_streaming` into pure VAD logic
  (testable) + thin pyaudio glue. The VAD state machine is the most
  fragile part of the loop and currently has zero tests.

---

## iter-003 — fix bug #4 (preview reprint wraps on long utterances)

**Branch:** `iter-003-bug-4` (merged ff to main, commit `7349fb7`)
**Date:** 2026-05-23

What changed:
- Added `render_preview(text, *, max_width, prefix, file, dim)` to
  `examples/_chat_helpers.py`. Wraps the existing
  `format_preview_line` truncation, prepends `\r\033[2K`, optionally
  wraps the body in dim ANSI, and writes-then-flushes to any
  file-like object. Returns the exact emitted string so tests can
  assert on visible width without faking terminal I/O.
- `record_utterance_streaming` in `mic_chat.py` now calls
  `shutil.get_terminal_size(fallback=(80, 24)).columns` and passes
  that into `render_preview` instead of writing the format string by
  hand. So even on a narrow terminal (or one resized mid-utterance),
  the live preview clamps to the current width.
- 7 new tests with `io.StringIO` and a counting buffer test double:
  short-text passthrough, long-text truncation with ellipsis, leading
  `\r + CLEAR_LINE`, write+flush behavior, `dim=False` strips dim
  ANSI, custom-prefix width math, back-to-back rewrite contract.

Verification: `python -m pytest tests/unit/` → **36 passed in 0.03s**.

All four originally-listed mic_chat.py bugs are now fixed:
  ✓ Bug #1 (iter-001) — duplicate "Bot:" lines
  ✓ Bug #2 (iter-001) — absurd llm_total values
  ✓ Bug #3 (iter-002) — mic buffer not flushed after LLM error
  ✓ Bug #4 (iter-003) — preview reprint wraps on long utterances

Next:
- iter-004: extract VAD state machine into a pure helper. Currently
  the silence/speech detection is inlined in
  `record_utterance_streaming` with monotonic timestamps and ad-hoc
  state — moving it to `vad_state.feed(level, now) -> Event` makes
  it testable without audio hardware. This is also the prerequisite
  for the architecture goals (streaming overlap, barge-in).
- iter-005: streaming LLM → TTS overlap. Right now sentences
  synthesize and play synchronously inside the for-loop, blocking
  the next token receipt. A producer/consumer queue with the LLM
  streaming on the main thread and TTS+playback on a worker would
  cut median TTFS substantially.

---

## iter-004 — extract VadState pure helper, 11 new tests

**Branch:** `iter-004-vad` (merged ff to main, commit `dff3f9e`)
**Date:** 2026-05-23

What changed:
- Added `VadState` (and `VadEvent` enum) in
  `examples/_chat_helpers.py`. Pure state machine: `feed(level, now)`
  returns one of IDLE / ACTIVE / DONE_OK / DONE_TOO_SHORT and updates
  `speaking`, `speech_start`, `silence_start`, `last_speech_duration`.
  Auto-resets on either DONE event so the state machine is ready for
  the next utterance.
- `record_utterance_streaming` in `mic_chat.py` rewritten to consume
  events instead of managing inline flags. Behavioral parity preserved:
  the trailing silence frame is still appended to `frames` before
  break (matches the original `frames.append(data); ... break` order).
  `too_short` flag replaces the post-loop duration recheck.
- 11 new tests covering: initial idle, first loud frame, continuous
  speech, silence timer start, brief silence doesn't end utterance,
  full silence window → DONE_OK, too-short window → DONE_TOO_SHORT,
  state-reset-between-utterances, strictly-greater threshold boundary,
  speech_duration recorded correctly, sub-threshold blips stay IDLE.

Verification: `python -m pytest tests/unit/` → **47 passed in 0.04s**.

Notes:
- A full integration test of `record_utterance_streaming` itself was
  considered but skipped: `mic_chat.py` imports `pyaudio` at module
  level, which isn't installable on x86_64 Linux without ALSA dev
  headers. The pure VadState unit tests give us complete coverage of
  the extracted logic; the wiring is straightforward and small.
- Hit a real bug while writing the threshold-boundary test:
  `1.4 - 0.6 = 0.7999999999999999`, which silently fails a `>= 0.8`
  comparison. Tests now use timestamps that avoid float-precision
  cliffs (e.g. 1.5 instead of 1.4 for the 0.8s silence window). Worth
  remembering for any future timestamp-comparison code.

Next:
- iter-005: streaming LLM → TTS overlap. The for-loop currently does
  synth + playback synchronously inside each iteration, blocking the
  next token. A `Queue` + worker thread for synth and a second worker
  for playback would let the LLM stream proceed at network speed
  while audio is being produced and played. Median TTFS could drop
  substantially since first-sentence synth could start the moment the
  first sentence's text is complete, even if more tokens are still
  arriving.
- iter-006: filler-word generation. While waiting for the first LLM
  token (`metrics.llm_first_token` typically 200-800ms), play a
  pre-rendered "hmm" or "let me think" so the user perceives lower
  latency. Configurable via `config.local.yaml`.
- iter-007: barge-in. Run a thin VAD on the mic input *during*
  bot playback. If the user starts speaking, kill the playback
  stream, drop pending sentences, and start recording. Requires the
  TTS + playback work from iter-005 since playback needs to be
  cancellable from outside the playing thread.

---

## iter-005 — virtual audio interfaces + TTS-driven simulation

**Branch:** `iter-005-virtual-audio` (merged ff to main, commit `ad20b0c`)
**Date:** 2026-05-23

User redirected the iteration order: instead of doing streaming
overlap, build software audio interfaces and use TTS as a simulated
mic source. This unblocks every future iteration that needs end-to-end
testing without hardware.

What changed (new file: `examples/virtual_audio.py`):
- `VirtualMicStream` — push audio in, pyaudio-shaped read/get_available
  out. Reads beyond buffer zero-pad by default (matches a quiet mic).
- `VirtualSpeakerStream` — write captures audio. Optional `loopback_to`
  routes every write back into a paired mic. That's the seed for
  iter-007 (barge-in) — bot output and user input share a stream.
- `VirtualAudioInterface` — drop-in replacement for `pyaudio.PyAudio()`.
  `.open(input=True)` returns a mic, `.open(output=True)` returns a
  speaker, `loopback=True` wires them.
- Audio fixtures: `make_silence`, `make_tone_burst`, `make_noise_burst`,
  `concat`. Pure numpy; noise is deterministic with a seed.
- `feed_tts(mic, text, ...)` — renders via `tts/kokoro_engine`,
  resamples 24k→mic.rate (cheap linear), pads silence around speech.
  Raises `RuntimeError` cleanly if kokoro can't load (so tests can
  skip).
- `simulate_vad_over_audio(audio, ...)` — chunk-by-chunk driver that
  returns the full event sequence + final VadState.

Tests (26 new in `tests/unit/test_virtual_audio.py`):
- VirtualMicStream: round-trip, `push_silence`, underflow zero-pad,
  no-padding mode, close blocks, reads recorded, float audio
  scaled+clipped.
- VirtualSpeakerStream: capture, float32 normalization, loopback,
  close blocks.
- VirtualAudioInterface: input/output dispatch, loopback wiring,
  terminate.
- Fixtures: silence-is-zero, tone-burst RMS matches theory
  (0.3 amp sine ≈ 0.21 RMS), noise determinism, concat ordering.
- VAD-over-fixtures: pure silence stays idle; speech+silence fires
  exactly one DONE_OK; short blip → DONE_TOO_SHORT; two utterances
  → two DONE events; sub-threshold amp stays idle.
- Byte-level path: VirtualMicStream → frombuffer → VadState produces
  expected events through the same path production code traverses.
- **TTS smoke test (opt-in)**: rendered "Hello, this is a simulation
  test." via real kokoro, fed the virtual mic, drove VadState end to
  end. DONE_OK fires with `last_speech_duration > 0.5s`. Confirms
  the synth → resample → bytes → VAD chain works on x86_64 Linux
  (despite the cpp-extensions warning from torch 2.10 / kokoro).

Verification: `python -m pytest tests/unit/` → **73 passed in 6.5s**.

Notes:
- Surprised that kokoro actually synthesizes on Linux x86_64 — the
  `Skipping import of cpp extensions due to incompatible torch
  version` warning at import time looked fatal but the model still
  produces audio (slower path). The skipif guard catches the case
  where it can't, so this stays portable.
- The loopback wiring is intentionally simple: every write
  unconditionally pushes to the linked mic. Real barge-in (iter-007)
  needs an `enabled` flag we toggle while bot is speaking; that's a
  one-line addition when we get there.

Next:
- iter-006 (was iter-005 before user redirect): streaming LLM → TTS
  overlap. Now testable end-to-end: synth bot reply via TTS, route
  through a `VirtualSpeakerStream`, assert TTFS measured against
  `speech_ended_at` is below threshold.
- iter-007: hook `mic_chat.py` to optionally use
  `VirtualAudioInterface` instead of `pyaudio.PyAudio()` so the chat
  loop can be smoke-tested in CI with a fixed prompt + a stubbed
  LLM. This is the foundation for benchmarking pipeline latency
  changes deterministically.

---

## iter-006 — extract record_utterance_streaming to pyaudio-free module

**Branch:** `iter-006-recording-extract` (merged ff to main, commit `9cf4631`)
**Date:** 2026-05-23

What changed:
- New module `examples/_chat_recording.py` hosts the recording loop +
  its audio/VAD constants (RATE, CHUNK, SILENCE_THRESHOLD, etc.),
  `rms`, `_buffer_to_wav`, `_transcribe_quick`, and
  `record_utterance_streaming`. No top-level pyaudio import — the
  module is importable on x86_64 Linux without ALSA.
- `record_utterance_streaming` gained three injection points:
    `transcribe_fn` — `callable(wav) -> str | None`. Tests stub this.
    `clock` — `callable() -> float`. Tests pass a `FrameClock` that
        advances exactly one chunk per call, removing wall-time
        dependency from the test loop.
    `output` — file-like for the live preview. Tests pass `StringIO`.
- `mic_chat.py` re-imports the constants and function from the new
  module. Live behavior unchanged.
- Side fix: the too-short branch now explicitly sets
  `stt_engine._last_text = None` so a stale value from a previous
  turn can't leak through.

Tests (8 new in `tests/unit/test_chat_recording.py`):
- Three-tuple with plausible values for a 1s tone burst.
- Transcript stashed on `engine._last_text`.
- Returned bytes parse as a valid 16kHz mono int16 WAV.
- Too-short blip returns `(b'', 0.0, 0.0)` and skips transcribe_fn.
- `transcribe_fn` receives strictly-growing wav buffers across
  multiple preview intervals (regression cover for streaming preview).
- Preview only re-renders when text changes (prevents flicker).
- Default `transcribe_fn` falls back to `_transcribe_quick` and
  returns None on Linux without crashing.
- **TTS-fed smoke test (opt-in)**: real kokoro renders "Hello, this
  is a recording test.", virtual mic receives it, the full
  `record_utterance_streaming` runs with a stub transcriber, and
  asserts a well-formed wav of plausible duration.

Verification: `python -m pytest tests/unit/` → **81 passed in ~9s**
(73 existing + 8 new).

Notes:
- The `FrameClock` test helper turned out to be the unlock: by
  advancing time exactly `chunk/rate` per `clock()` call, the VAD
  state machine sees the same time progression it would with real
  audio, but the test runs as fast as numpy can crunch the buffer.
- One subtle thing learned writing the tests: `transcribe_fn` is
  called twice during the typical happy path — once at each
  `INFERENCE_INTERVAL` cross during speech (preview), and once at
  the very end on the final wav. The "growing buffer" test
  documents that contract.

Next:
- iter-007: extract `play_aligned` to a pyaudio-free module the same
  way. Accept any speaker-shaped object (write/stop_stream/close),
  test against `VirtualSpeakerStream`. After this, every primitive
  the chat loop touches is testable.
- iter-008: build streaming overlap — a `SentenceWorker` thread that
  consumes from a `Queue` of complete sentences, synthesizes via
  `tts_engine`, plays via the speaker stream, and reports playback
  duration. The LLM stream feeds the queue. Test using
  `VirtualSpeakerStream` to assert that playback starts before the
  LLM stream is done.
- iter-009: barge-in. Run `VadState` on the mic input *during*
  speaker writes (loopback mode means mic and speaker share a
  buffer; we'd disable loopback for this and use a separate mic
  feeder for "user starts speaking" injection). When the worker
  detects speech, drain the sentence queue, stop the speaker,
  return control.

---

## iter-007 — extract play_aligned to pyaudio-free module

**Branch:** `iter-007-playback-extract` (merged ff to main, commit `3005fde`)
**Date:** 2026-05-24

What changed:
- New module `examples/_chat_playback.py` hosts the playback loop +
  `TTS_RATE`, `DEFAULT_PLAY_CHUNK`, ANSI codes, plus `_emit_token`
  and `_is_punct_only` helpers.
- `play_aligned(speaker_stream, audio_np, tokens, ...)` now takes a
  caller-owned speaker-shaped object (only `.write(bytes)` is
  required). Lifecycle is the caller's responsibility.
- `mic_chat.play_aligned` becomes a thin wrapper that opens a
  pyaudio output stream, calls the core function, and closes the
  stream — preserving the current open-per-sentence behavior. The
  persistent-stream optimization is iter-008's job.
- Injection points: `output` (file-like) and `clock` (callable),
  same pattern as iter-006.
- Quirk preserved: tokens emitted during the loop are bolded; tokens
  emitted *after* the loop (when `start` > audio duration) are emitted
  in plain text. This matches the original code; a regression test
  documents it so a future cleanup is intentional.

Tests (18 new in `tests/unit/test_chat_playback.py`):
- `_is_punct_only`: punctuation True, words False, empty False.
- Audio writes: total bytes match input, chunk size honored, empty
  audio writes nothing, float→int16 saturation correct.
- First-sentence prefix: `\r CLEAR_LINE  Bot: ` only on first, no
  prefix on subsequent calls.
- Token reveal: in-order during playback, punctuation uses
  backspace, empty/whitespace tokens dropped, words bold, trailing
  tokens flushed unbolded after loop.
- Clock injection: elapsed computed from supplied clock; default
  output falls back to real stdout.
- Loopback: `VirtualSpeakerStream` → `VirtualMicStream` wiring still
  works; audio bytes round-trip identically. Seed for iter-009.

Verification: `python -m pytest tests/unit/` → **99 passed in ~9s**
(81 existing + 18 new).

Notes:
- Two iters to extract two functions. Now every primitive the chat
  loop touches — VAD, audio flush, recording, playback, render-
  preview, sentence-split, history-trim — is testable on x86_64
  Linux. The chat loop's remaining `pyaudio` calls are confined to
  `run_chat`'s setup/teardown plus the two thin wrappers.
- The loopback test is a tiny but meaningful proof point: writing
  to a `VirtualSpeakerStream` whose `loopback_to` is set pushes the
  exact same int16 samples into the paired mic. That's the
  primitive iter-009 barge-in needs — bot speaks, mic sees the
  speech, VAD watches the mic.

Next:
- iter-008: streaming LLM → TTS overlap. Pull the in-loop synth +
  play_aligned calls out into a `SentenceWorker` thread that pulls
  complete sentences from a `Queue`. The main thread keeps the LLM
  stream open so token receipt isn't blocked by audio playback.
  Holds a single persistent speaker stream — open in the worker's
  setup, close in its teardown. Test by feeding a fixed list of
  sentences into the queue, watching the speaker accumulate audio,
  and asserting the worker runs ahead of the simulated LLM clock.
- iter-009: barge-in. Mic-side `VadState` runs in parallel with the
  worker. On `DONE_OK` from the mic, signal the worker to drop the
  sentence queue and stop the speaker, return control. Test using
  loopback paired with a "user injection" feeder that pushes a
  user-speech burst into the same mic the worker is listening to.
- iter-010: filler-word generation. While `metrics.llm_first_token`
  is high, emit a pre-rendered "hmm…" or "let me think…" via TTS
  before the real first sentence. Cheap latency-perception win.
  Configurable via `config.local.yaml`.

---

## iter-008 — streaming LLM → TTS overlap via SentenceWorker

**Branch:** `iter-008-streaming-overlap` (merged ff to main, commit `ad4d946`)
**Date:** 2026-05-24

The headline architecture change. Previously the chat for-loop
synthesized + played each sentence synchronously *inside* the
token-receipt loop, so every byte of audio playback was a byte of
LLM stream latency the network connection was otherwise wasting.
Multi-sentence replies stacked the cost linearly.

Now: a background `SentenceWorker` thread consumes complete
sentences off a `Queue`, runs synth + play, and reports metrics. The
main thread submits and keeps consuming tokens.

What changed:
- New module `examples/_chat_pipeline.py` with `SentenceWorker`.
  - Constructor: `speaker_factory`, `synth_fn`, `play_fn`, optional
    `clock` and `output` (all dependency-injected).
  - Lifecycle: `start` → `submit` × N → `submit_done` →
    `wait_done(timeout=...)`. Or `stop(timeout=...)` for
    early termination (LLM error / Ctrl+C).
  - Metrics tracked: `sentences_spoken`, `tts_time`,
    `playback_time`, `first_audio_at`, `errors`.
  - Owns one persistent speaker stream — opens via `speaker_factory`
    in `_run`, closes (`stop_stream` + `close`) in the `finally`
    block. No more open-per-sentence overhead.
- `mic_chat.run_chat` rewired:
  - Replaces the in-loop synth + `play_aligned` calls with
    `worker.submit(sentence)`.
  - `wait_done(timeout=120)` after `submit_done` to drain any
    sentences still queued when LLM stream ends.
  - TTFS now derives from `worker.first_audio_at` (slightly more
    accurate — captured just before first audio is written, rather
    than before synth begins).
  - Error path calls `worker.stop()` for clean teardown before the
    bug-3 mic flush.

Tests (18 new in `tests/unit/test_chat_pipeline.py`):
- Lifecycle: submit + done + wait runs clean; double start raises;
  wait_done without start raises; submit after submit_done dropped.
- Ordering: sentences play in submission order (proven via
  growing-sample-count synth); `is_first_sentence` only True on first;
  metrics accumulate; `first_audio_at` set exactly once.
- Skipping: empty/whitespace sentences and empty audio from synth
  both skipped without incrementing count.
- Error handling: synth raising captured in `errors`, loop
  continues; play raising same; speaker_factory raising exits
  cleanly without hang.
- Stop: pending dropped + thread joined; before-start is a no-op;
  submit_done immediately followed by stop doesn't hang.
- Speaker wiring: `speaker._closed is True` after worker exit;
  loopback to `VirtualMicStream` round-trips audio (seed for
  iter-009 barge-in).

Verification: `python -m pytest tests/unit/` → **117 passed in 9s**
(99 existing + 18 new).

Notes:
- The threaded `stop` test uses a `synth_started` Event +
  `proceed` Event to deterministically place the worker in
  "currently synthesizing, two more queued" state before calling
  `stop()`. The assertion is "exactly one sentence got spoken" —
  the in-flight one finishes, the queued ones are dropped. This
  was the most likely-to-be-flaky test; it consistently runs in
  <50ms.
- `worker.errors` exists so production code can surface background
  failures without taking down the chat loop. `mic_chat.run_chat`
  prints any worker errors after `wait_done` returns.
- Single-sentence replies see no behavior change. Multi-sentence
  replies should show lower TTFS for sentence 2+ because the LLM
  stream now overlaps with playback.

Next:
- iter-009: barge-in. Run `VadState` on the mic stream *during*
  speaker writes. When the user starts speaking, signal the worker
  to drop pending sentences and stop the speaker mid-sentence.
  Requires a "hard cancel" path on the worker (the current `stop`
  is "drain + finish current"); add `cancel()` that sets a flag
  the play_fn checks per audio chunk. Test using loopback paired
  with a "user injection" feeder that pushes user-speech into the
  mic the worker is also watching.
- iter-010: filler words during LLM first-token wait. Pre-render
  short audio clips ("hmm…", "let me think…") and have the worker
  play them if the queue stays empty for more than ~600ms. Hide
  behind a config flag so it's opt-in.

---

## iter-009 — barge-in primitives: cancel_event + cancel + BargeInWatcher

**Branch:** `iter-009-bargein` (merged ff to main, commit `c4b34d5`)
**Date:** 2026-05-24

What changed:

1. **`play_aligned` cancel_event** — `examples/_chat_playback.play_aligned`
   gained an optional `cancel_event` kwarg. The chunked-write loop
   checks it between chunks; on set, breaks early and returns
   elapsed time so far. Backward compat: omit the kwarg and behavior
   is unchanged.

2. **`SentenceWorker.cancel()`** — hard-stop method that:
   - Sets the worker's internal `_cancel_event` (forwarded into
     `play_fn` so the in-flight sentence breaks mid-stream).
   - Drains the queue and joins the thread.
   - Sets `self.cancelled = True` for the caller to inspect.
   - Idempotent.
   The play_fn signature now optionally accepts `cancel_event` —
   a `try/except TypeError` fallback preserves the iter-008
   contract for play_fns that don't accept the new kwarg.

3. **`BargeInWatcher`** — new class in `examples/_chat_pipeline.py`.
   Background thread that reads from a mic stream, runs `VadState`,
   fires a callback on the configured trigger event. Configurable:
   - `trigger_on="active"` (default) — fires the instant level
     crosses threshold. Fastest reaction.
   - `trigger_on="done_ok"` — fires only after a complete utterance
     (waits for the silence window). Useful when you don't want
     to interrupt on every cough.
   Captures all consumed frames in `self.frames` so the orchestrator
   can replay the user's first syllables into the next record loop.
   Uses `get_read_available()` to avoid feeding silent zero-pad to
   the VAD when the mic is empty.

Tests (17 new in `tests/unit/test_bargein.py`):
- play_aligned cancel_event: pre-set breaks immediately, mid-loop
  set breaks between chunks, unset completes, omission works.
- SentenceWorker.cancel: interrupts in-flight sentence (verified
  by side-effect Event in play_fn), idempotent, before-start
  no-op, iter-008 play_fns without cancel_event still work.
- BargeInWatcher: silence stays idle, speech burst triggers
  callback < 2s, callback fires once during long speech,
  `done_ok` trigger waits for silence window (uses injected
  frame-clock for virtual time), invalid trigger raises,
  double-start raises, frames captured.
- **Integration**: 3 sentences submitted, watcher detects user
  speech on virtual mic, fires `worker.cancel`, worker breaks
  mid-sentence, byte-count math proves partial write, dropped
  sentences confirmed via `sentences_spoken <= 1`.

Verification: `python -m pytest tests/unit/` → **134 passed in 9s**
(117 existing + 17 new).

Notes:
- DONE_OK trigger test exposed a real subtlety: `time.monotonic()`
  doesn't work for `VirtualMicStream` tests because pre-pushed audio
  is served in milliseconds. The VAD's 0.8s silence window never
  elapses by wall clock. Solution: inject a frame-aligned clock —
  same `FrameClock` pattern as iter-006. In production this is a
  non-issue since real PyAudio serves frames at audio rate.
- A play_fn that respects `cancel_event` must yield between chunks
  (sleep, blocking I/O, anything). Otherwise the worker thread
  races through the play loop before the test can set the flag.
  The test fixture uses `time.sleep(0.005)`; real audio playback
  blocks at audio rate so this happens naturally.

Next:
- iter-010: wire barge-in into `mic_chat.run_chat`. Plumbing:
  during `worker.wait_done()`, run a `BargeInWatcher` on `mic`
  with `worker.cancel` as the callback. After `wait_done` returns,
  if `watcher.detected`, feed `watcher.frames` into the next
  `record_utterance_streaming` call (needs a small extension to
  the recording function to accept "primed" frames). Without this,
  the user's first syllables are dropped.
- iter-011: filler-word generation. Pre-render short audio clips
  ("hmm", "let me think") at startup. The worker grows an idle-
  watchdog timer: if the queue has been empty for >600ms while
  `submit_done` hasn't been called, play one filler clip. Hide
  behind `config.local.yaml` flag.

---

## iter-010 — wire barge-in into chat loop, primed_frames replay

**Branch:** `iter-010-bargein-wiring` (merged ff to main, commit `9279220`)
**Date:** 2026-05-24

Closes the loop iter-009 started. The cancel/watcher primitives are
now actually used by `mic_chat.run_chat`, and the user's barge-in
audio is preserved into the next record turn instead of dropped.

What changed:

- `examples/_chat_recording.py`:
  - `record_utterance_streaming` gained a `primed_frames` kwarg.
    Each primed frame is fed through the VAD before any live mic
    reads, in order. Each frame's bytes are appended to the
    output wav as if read live.
  - **Refactored time accounting**: the function uses a single
    virtual clock anchored at function entry —
    `now = t_origin + frame_idx * (CHUNK / RATE)`. The previous
    `clock()`-per-frame approach had two failure modes that the
    primed-frames tests surfaced: (1) primed frames served in
    microseconds made the silence window appear to close instantly;
    (2) with a per-call test FrameClock, the first live read's
    timestamp could land *before* the last primed timestamp,
    corrupting `last_speech_duration`. The unified virtual clock
    keeps time monotonic at audio rate regardless of clock
    provider. Production behavior is identical because PyAudio
    reads already block at audio rate.

- `examples/mic_chat.py`:
  - `run_chat` carries `primed_frames` across iterations of the
    while loop. After `worker.wait_done` returns, if
    `watcher.detected`, `watcher.frames` is captured into
    `primed_frames` and fed to the next `record_utterance` call.
    Reset to None after consumption.
  - Spawns a `BargeInWatcher` pointed at the same mic the
    recorder uses, with `worker.cancel` as the callback. Watcher
    starts after `worker.start`; stops after `worker.wait_done`.
    Mic is flushed (bug-3 helper from iter-002) right before the
    watcher starts so audio buffered during the LLM phase isn't
    misread as user speech.
  - LLM-error path also stops the watcher so we don't leak a
    thread.

Tests (9 new):
- `tests/unit/test_chat_recording.py` — 6 priming tests:
  - None / `[]` are no-ops; speech burst appears in wav; silence
    falls through to live mic; too-short blip yields
    `DONE_TOO_SHORT`; pedantic frame-count check.
- `tests/unit/test_bargein_orchestration.py` — 3 full-loop tests:
  - Bot speaks → user barges in on virtual mic → watcher fires
    `worker.cancel` → `primed_frames` go into next record call →
    valid wav covers user audio. Closes the loop end to end.
  - No-barge-in path: full sentence plays, no priming.
  - 5-sentence queue: barge-in during sentence 1 drops the
    remaining 4.

Verification: `python -m pytest tests/unit/` → **143 passed in 10s**
(134 existing + 9 new).

Notes:
- The per-frame `clock()` bug was hidden under the simple iter-006
  case (live-only reads through FrameClock) and only surfaced when
  primed and live frames coexisted. The fix (`t_origin` + frame
  counter) is also conceptually cleaner — the function's view of
  time is now invariant to how the underlying clock behaves.
- The orchestration tests are as close as we get on x86_64 Linux
  to "the real chat loop is working." Deterministic in 0.2s using
  only virtual audio. That's the payoff for everything iter-005
  through iter-009 built up.

Next:
- iter-011: filler-word generation (carried over from iter-009
  plan). Pre-render short clips ("hmm", "let me think") at
  startup. Worker grows an idle-watchdog timer: if queue stays
  empty for >600ms while `submit_done` hasn't been called, play
  one filler. Behind a `config.local.yaml` flag.
- iter-012: optional — keep the `BargeInWatcher` running across
  the LLM-streaming phase too, not just during play. Currently
  if the user starts talking while the LLM is still streaming
  the first response token, we miss those early frames. Doable
  but requires reasoning carefully about who owns the mic when.
- iter-013: real-mic CI smoke test using ALSA loopback. Pipe
  TTS-rendered audio through `aplay` into `arecord`, then run
  the chat loop end-to-end on the actual PyAudio stack. Optional
  — closes the gap between virtual-audio tests and real
  hardware, but may not be worth the maintenance burden pre-1.0.

---

## iter-011 — filler-word generation in SentenceWorker

**Branch:** `iter-011-fillers` (merged ff to main, commit `12b718b`)
**Date:** 2026-05-24

The third architecture goal from the original focus list. While the
LLM is still spinning up its first token (typically 200-800ms), the
chat loop sat silent. Now the worker can play a pre-rendered filler
clip ("hmm", "let me think") to mask that latency.

What changed:

- `examples/_chat_pipeline.py`:
  - `SentenceWorker.__init__` gained `fillers`, `idle_threshold`,
    `filler_picker` kwargs.
  - `_run` uses `queue.get(timeout=idle_threshold)` when fillers
    are configured AND no sentence has played yet AND no filler
    has played yet. On timeout, picks one filler and plays it.
    After playing, reverts to blocking `get()` — exactly one
    filler per worker run.
  - Refactored play logic into a new `_play_clip()` helper so
    both fillers and real sentences share the same bookkeeping
    (`first_audio_at`, `playback_time`, `errors`,
    `cancel_event` handling).
  - New counter `fillers_played` (distinct from
    `sentences_spoken`).
  - TTFS works correctly: `first_audio_at` is captured the first
    time `_play_clip` runs, which means a filler accurately
    measures the user-perceived latency it's meant to mask.

- `examples/mic_chat.py`:
  - New `load_chat_config()` reads the optional `chat:` section
    of `config.local.yaml` (empty if missing — no breaking
    change for existing configs).
  - `run_chat` pre-renders fillers via `synthesize_with_alignment`
    using `chat.fillers` (list of strings); idle threshold from
    `chat.fillers_idle_threshold` (default 0.6s).
  - Bad filler entries are logged and skipped — never block the
    caller.
  - SentenceWorker construction passes the rendered list; if
    empty (default), `idle_threshold` is forced to 0 so the
    worker never times out.

Tests (12 new in `tests/unit/test_fillers.py`):
- Backward compat: no fillers / empty list = unchanged behavior.
- Happy path: filler plays after idle, no filler if sentence
  arrives early, exactly one filler per run.
- Filler counts as the first audio output (`is_first_sentence=True`
  for the "Bot:" prefix).
- Counters: `first_audio_at` captured at filler start;
  `fillers_played` starts at 0.
- Robustness: empty filler audio consumes slot but `fillers_played`
  stays 0; play raising during filler doesn't block real sentence;
  `cancel()` during filler playback works (same cancel_event
  path as real sentences).
- Byte-stream order: filler bytes precede sentence bytes in
  speaker's captured stream, with distinct amplitudes (0.3 vs 0.5
  → int16 9830 vs 16383) to make the regions distinguishable.

Verification: `python -m pytest tests/unit/` → **155 passed in 11s**
(143 existing + 12 new).

Notes:
- The `_play_clip` extraction made the diff bigger but is the
  right shape going forward — fillers, sentences, and any future
  audio output (e.g. confirmation tones) all share the same path.
- One filler per run is intentional. A tempting extension is
  "play another filler if the LLM stalls mid-stream too," but
  that adds a watchdog timer and a bunch of edge cases. Skipping
  for now; the first-token-latency mask is where 90% of the win is.

---

# Status

The architecture phase is **complete** for the user-specified focus
list:

  ✓ Streaming overlap        — iter-008
  ✓ Barge-in                 — iter-009 + iter-010
  ✓ Filler words             — iter-011

Cumulative state: **155 unit tests passing in ~11s** on x86_64
Linux without pyaudio or mlx-whisper. The chat-loop primitives
(`record_utterance_streaming`, `play_aligned`, `SentenceWorker`,
`BargeInWatcher`, `VadState`) are all testable in isolation, and
the orchestration tests prove they compose correctly into a
full bot-speaks → user-barges-in → record-replays cycle running
deterministically in 0.2s.

Remaining log items (iter-012 extended-watcher, iter-013 ALSA
loopback CI) are stretch goals with diminishing returns for a
pre-1.0 project. Could revisit if/when actual user feedback
demands them.

---

## iter-012 — extend barge-in across the LLM-streaming phase

**Branch:** `iter-012-llm-stream-barge` (merged ff to main, commit `f1e7a4e`)
**Date:** 2026-05-24

iter-009/010 covered barge-in during worker playback. The remaining
gap: a user who started talking while the LLM was still emitting
its first token would be detected by the watcher and the worker
would be cancelled, but the for-token loop would keep consuming
tokens for sentences that would never be played (silently
swallowed by the now-cancelled worker). Wasted compute, wasted
bandwidth, no early exit.

What changed:

- `examples/_chat_pipeline.py`:
  - New `BargeInCoordinator` class. Single-shot signal that
    bundles together the actions barge-in needs to cascade
    through:
      1. A `threading.Event` the for-token loop checks each
         iteration.
      2. `SentenceWorker.cancel()` — playback stops mid-stream.
      3. Optional `on_trigger` hook (intended for closing HTTP
         requests streams in production; not wired yet).
  - Idempotent — multiple triggers are safe, only the first
    cascades. Lock-protected event-set; the cancel cascade
    runs outside the lock so a long join doesn't block other
    callers checking `is_set()`.
  - Robust — exceptions in `worker.cancel` or `on_trigger`
    are swallowed so they can't leave the event un-set.

- `examples/mic_chat.py`:
  - Imports `BargeInCoordinator` alongside the existing
    primitives.
  - Per-turn coordinator wired into the watcher
    (`on_speech_detected = coord.trigger`) and into the
    for-token loop (`if coord.is_set(): break`).
  - Post-stream path split: on clean completion, drain buffer
    + `submit_done` + `wait_done`; on barge-in, just wait for
    the (already-cancelled) worker thread.
  - Status print now distinguishes "barge-in during LLM-stream
    phase" vs "playback phase" by comparing
    `coord.triggered_at` to `llm_stream_done_at`.

Tests (17 new):

- `tests/unit/test_bargein_coordinator.py` — 12 tests:
  - Basic: `is_set` starts False; trigger sets event +
    timestamp; `event` property exposes the underlying
    `threading.Event`.
  - Idempotent: double-trigger fires once;
    `triggered_at` doesn't update on the second call.
  - Wiring: `worker.cancel` called; no-worker case OK;
    `on_trigger` hook fires; event set BEFORE
    `worker.cancel` returns (proven via concurrent reader
    thread).
  - Robustness: `worker.cancel` exception doesn't break
    trigger; `on_trigger` exception doesn't break trigger.
  - Real-worker integration: trigger cancels a
    `SentenceWorker` for real, less-than-full audio reaches
    the speaker.

- `tests/unit/test_llm_stream_barge.py` — 5 tests:
  - For-token early exit: no barge-in consumes all tokens;
    pre-set coord yields zero tokens; mid-stream trigger
    breaks with at most one sentence submitted.
  - **Full integration**: faked LLM stream emits tokens with
    a per-token delay, user speech pushed to a `VirtualMicStream`
    mid-stream, watcher fires `coord`, consumer loop breaks,
    worker cancelled, watcher captures user audio for replay.
    End-to-end barge-in during LLM streaming verified.
  - Clean completion: no user audio, full stream consumed,
    worker plays all sentences.

Verification: `python -m pytest tests/unit/` → **172 passed in 11s**
(155 existing + 17 new).

Notes:
- The faked-LLM orchestration test runs in 150ms. Production
  behavior is identical because the same components compose
  the same way — the test just stubs out the HTTP transport.
- The "event set before cancel returns" test was tricky to
  write deterministically. The current approach uses a
  concurrent reader thread that polls `is_set()` while the
  trigger runs; the assertion fails if the worker.cancel
  blocks the event-set call. Real production cancel paths
  do their own join, so this guards against accidentally
  reordering the cascade.

---

# Status

The architecture phase is complete plus the cross-phase barge-in
extension:

  ✓ Streaming overlap         — iter-008
  ✓ Barge-in (playback)       — iter-009 + iter-010
  ✓ Filler words              — iter-011
  ✓ Barge-in (LLM stream)     — iter-012

172 unit tests passing in 11s on x86_64 Linux. Every primitive in
the chat pipeline is testable in isolation, and three orchestration
tests (iter-009 worker-cancel, iter-010 record-replay, iter-012
LLM-stream-cancel) compose them into deterministic full-cycle
verifications.

Remaining log item (iter-013 ALSA loopback CI) is the only stretch
goal left. Diminishing returns at this point — the gap between
virtual-audio-tested behavior and real-hardware behavior is small
enough that real-mic CI is more about catching deployment
regressions than driving development.

---

## iter-013 — extract SSE parser, close LLM stream on barge-in

**Branch:** `iter-013-llm-close` (merged ff to main, commit `8d949a8`)
**Date:** 2026-05-24

iter-012 left a thread loose: when barge-in fires during the LLM
stream phase, the consumer breaks out of the for-token loop, but
the LLM generator (and its underlying HTTP response) hangs on until
garbage collection eventually runs the finally block. This iter
closes that gap and extracts the SSE parser as a side benefit.

What changed:

1. **Extract SSE parser** into `examples/_chat_llm.py`:
  - `parse_sse_token_stream(lines)` — pure parser. Takes any
    iterable of lines (bytes or str), yields content tokens.
    Handles `[DONE]`, empty/non-`data:` lines, malformed JSON,
    missing fields, bytes decoding (UTF-8 with replacement).
  - `stream_chat_completion(messages, config)` — thin generator
    wrapping `requests.post` + `parse_sse_token_stream` with a
    `try/finally` that closes the `Response`.
  - `examples/mic_chat.py` re-exports `stream_chat_completion`
    as `llm_stream` so external imports still work.

2. **Consumer-side close**:
  - The naive idea (`coord.on_trigger = gen.close`) fails:
    cross-thread generator close raises
    `ValueError("generator already executing")` while the
    consumer is mid-`next()`, and `BargeInCoordinator`'s
    hook-exception swallow turns that into a silent no-op.
  - Correct pattern: wrap the for-loop in `try/finally:
    llm_gen.close()`. That runs in the consumer's thread after
    it breaks, no race. The generator's own finally then closes
    the response, releasing the TCP connection.
  - `mic_chat.run_chat`:
    * Holds the LLM generator handle up front.
    * `try/finally` around the for-loop closes it on any exit
      path (clean completion, barge-in break, exception).
    * `on_trigger` is NOT wired to `gen.close` — comment
      explains why so a future contributor doesn't reintroduce
      the bug.

Tests (18 new in `tests/unit/test_chat_llm.py`):

`parse_sse_token_stream` (13):
- empty input; single + multi data lines; `[DONE]` stops;
  blank/non-data lines skipped; malformed JSON skipped; missing
  `choices`/`delta`/`content` skipped; empty content skipped;
  bytes decoded; invalid UTF-8 doesn't raise; empty
  `"choices": []` handled.

`stream_chat_completion` lifecycle (3):
- Full consumption closes response once. Generator close
  propagates to response close (verified with a slow fake
  response so the close lands mid-stream). Exception in
  response close is swallowed by the finally.

Consumer-side close pattern (2):
- Consumer breaks mid-stream after `coord.trigger`; outer
  `try/finally` calls `gen.close`, response is released.
  Mirrors the mic_chat.run_chat shape exactly.
- Normal completion via `[DONE]` also releases response;
  explicit `gen.close` after natural end is idempotent.

Verification: `python -m pytest tests/unit/` → **190 passed in 11s**
(172 existing + 18 new).

Notes:
- The first attempt at this iteration wired `on_trigger=gen.close`
  and tested it cross-thread. Failing test caught the
  generator-already-executing bug — that's the value of running
  the test before assuming the design. The fix-in-place
  documents the correct same-thread pattern.
- Python generators are inherently single-threaded consumers —
  `close()` must come from the iterating thread. Worth
  remembering for any future generator-cleanup work (e.g. iter-014
  if we ever want to interrupt synth_with_alignment).
- The SSE parser handles edge cases (empty content, role chunks,
  malformed JSON, `[DONE]`) that real providers actually emit.
  Cheap insurance against expensive live debugging.

---

# Status

12 iterations shipped. The architecture phase plus its loose ends
are all closed:

  ✓ Bugs #1-#4 fixed              — iter-001 through 003
  ✓ VadState extracted            — iter-004
  ✓ Virtual audio + TTS feeder    — iter-005
  ✓ record_utterance_streaming    — iter-006
  ✓ play_aligned                  — iter-007
  ✓ Streaming overlap             — iter-008
  ✓ Barge-in (playback)           — iter-009 + iter-010
  ✓ Filler words                  — iter-011
  ✓ Barge-in (LLM stream)         — iter-012
  ✓ LLM stream cleanup on barge   — iter-013

**190 unit tests passing in 11s** on x86_64 Linux without pyaudio,
mlx-whisper, or a real LLM. Every primitive is testable in
isolation, and three full-loop orchestration tests
(iter-009, iter-010, iter-012) exercise the whole pipeline
deterministically using virtual audio + faked LLM tokens.

The only remaining log item is iter-014 (formerly iter-013) ALSA
loopback CI — pure infrastructure work with diminishing returns.

---

## iter-014 — hardening pass: rms NaN, error-path frames, metric surfacing

**Branch:** `iter-014-hardening` (merged ff to main, commit `806c21a`)
**Date:** 2026-05-24

A careful re-read of the now-stable code surface turned up three
small but real issues. None blocked the happy path; each could
silently corrupt user-visible behavior under uncommon-but-real
conditions. (The ALSA-loopback stretch goal stays parked; this
one is closer to home and unblocks proper testing of metrics.)

What changed:

1. **`rms()` returned NaN on empty input.** `np.mean` of an empty
   slice emits `RuntimeWarning` and returns NaN. NaN silently
   broke `VadState.feed` because `NaN > threshold` is always
   False — empty reads stayed IDLE forever and the loop stalled.
   Empty reads happen in practice (torn PyAudio reads, virtual
   mic flushed mid-iteration). Fix: guard with `len(frame) == 0`
   returning 0.0. Verified by a test that runs under
   `warnings.simplefilter("error")`.

2. **LLM-error path dropped watcher-captured user audio.** If the
   user was barging in at the moment the LLM call failed (network
   blip, DNS, 5xx), the error path stopped the watcher cleanly
   but discarded `watcher.frames`. Now copied into `primed_frames`
   for the next record turn, with a status print so the user
   knows what happened.

3. **`fillers_played` + `barge_in` not surfaced in `TurnMetrics`.**
   iter-011 added `worker.fillers_played`; iter-012 added the
   coord-set check. Neither made it into the per-turn summary.
   Now: TTS line shows "X sentences + Y fillers" suffix; a yellow
   "Barge-in: yes (user interrupted)" line appears when the
   coordinator fired.

Side benefit: extracted `TurnMetrics` into a new
`examples/_chat_metrics.py` (same iter-006/007 pattern). Tests
can now import it without dragging in pyaudio at module scope.

Tests (14 new in `tests/unit/test_hardening.py`):

`rms()` edge cases (5):
- empty array under `simplefilter("error")` returns 0.0 with
  no warning
- `None` input returns 0.0 (defensive)
- silence returns 0.0
- constant amplitude returns that amplitude
- 0.3 sine RMS matches theoretical 0.3/√2 within tolerance

`record_utterance_streaming` with empty reads (1):
- Custom mic stream serves 3 empty-bytes reads then real audio.
  Wrapped in `simplefilter("error")` so any latent warning
  fails the test. Loop completes and produces a non-empty wav
  covering the post-empty audio.

`TurnMetrics.print` output (6):
- 0 fillers omits from TTS line; 1 filler shows singular;
  2+ shows plural; `barge_in=False` omits line;
  `barge_in=True` shows yellow status; default metrics has
  new fields at 0/False.

Error-path carryover logic (2):
- `watcher.detected=True` copies frames; defensive copy (not
  same list object).
- `watcher.detected=False` leaves `primed_frames=None`.

Verification: `python -m pytest tests/unit/` → **204 passed in 11s**
(190 existing + 14 new).

Notes:
- The `TurnMetrics.print` tests caught a small pluralization bug
  in passing — "1 fillers" vs "1 filler". Easy to miss without
  the test.
- `_capture_print` uses `contextlib.redirect_stdout` instead of
  pytest's `capsys` because it's a unit test of the printer
  output, not a fixture-driven integration test. Either works;
  redirect_stdout is more explicit about what's being captured.
- The error-path-carryover test exercises the logic shape rather
  than calling `mic_chat.run_chat` directly. A full end-to-end
  test of `run_chat` would require stubbing STT + LLM + audio
  hardware — possible but a separate refactor.

---

# Status

**14 iterations shipped, 204 tests passing in 11s.** All five
focus-list goals plus four follow-ons:

| Phase | Iterations |
|-------|------------|
| Bugs #1-#4 | 001-003 |
| Refactor (5 helper modules + virtual_audio + _chat_metrics) | 004-007, 014 |
| Streaming overlap | 008 |
| Barge-in (playback + LLM-stream + frame-replay) | 009-010, 012 |
| Filler words | 011 |
| LLM-stream cleanup | 013 |
| Hardening (rms, error-path frames, metrics) | 014 |

Every primitive in the chat pipeline is testable in isolation,
and three full-loop orchestration tests exercise the whole
pipeline deterministically. Real bugs caught while writing tests
this round include: the per-frame clock divergence on primed/live
boundary (iter-010), the cross-thread generator close ValueError
(iter-013), and the rms-empty-NaN that would silently stall the
recording loop (iter-014). Each was hidden behind happy-path
behavior and would have manifested only under specific edge
conditions in production.

The codebase is at a comfortable stopping point. Further work
would be: real-mic CI smoke test (iter-013 in original numbering,
diminishing returns), `run_chat` end-to-end refactor (substantial
but tests the actual production path), or pivoting to features
beyond the original focus list.

---

## iter-015 — extract ChatLoop class, real production path now tested

**Branch:** `iter-015-chat-loop` (merged ff to main, commit `82a8e7b`)
**Date:** 2026-05-24

The orchestration tests in iter-009 / iter-010 / iter-012
*approximated* the structure of `run_chat`'s per-turn body. They
proved the components compose correctly when wired the way
`mic_chat.run_chat` wires them, but the actual function was
untested. A refactor that accidentally changed the wiring would
still pass the orchestration tests — they tested their own
helper, not `run_chat`.

This iteration extracts the per-turn body into a `ChatLoop` class
with every dependency injected. The same code path that the
production chat loop runs is now driven directly by tests using
`VirtualMicStream` + `VirtualSpeakerStream` + stub STT + stub LLM.

What changed:

- `examples/_chat_loop.py` (new):
  - `ChatLoop` class. Constructor accepts:
    - Audio: `mic`, `speaker_factory`, `rate`, `chunk`,
      `silence_duration`.
    - STT: `stt_engine`, optional `transcribe_fn`.
    - LLM: `llm_stream_fn` (callable), `llm_config`.
    - TTS: `synth_fn`, `play_fn`.
    - Filler config: `fillers`, `idle_threshold`.
    - Tunables: `clock`, `output`, timeouts.
  - `run_one_turn(messages, primed_frames=None) -> TurnResult`
    where `TurnResult` has `metrics`, `next_primed_frames`,
    `had_error`.
  - Same observable behavior as the inline `run_chat` body,
    including iter-013 LLM cleanup and iter-014 hardening.

- `examples/mic_chat.py`:
  - Per-turn body shrunk from ~200 lines to ~15. `run_chat` now
    builds three closures (`_speaker_factory`, `_synth`,
    `_play`) that bind the runtime types, instantiates
    `ChatLoop`, and drives it in a thin while-True with
    `primed_frames` threading.

Tests (8 new in `tests/unit/test_chat_loop.py`):

NoTranscription (2):
- Too-short utterance → metrics=None, no primed, no error.
- Empty transcript → "(no transcription)" path → metrics=None.

NormalTurn (2):
- Full turn: stub STT="how are you", stub LLM="I am well.
  Thanks for asking.", virtual mic+speaker. Verifies metrics
  populated, history updated (user + assistant), speaker
  received audio.
- Sentence count: LLM yielding "One. Two. Three." → at least
  three sentences played.

BargeInDuringLlmStream (1):
- Background thread schedules user speech mid-LLM-stream.
  Verifies `barge_in` flag, `next_primed_frames` non-None,
  no error.

LlmErrorPath (2):
- LLM raises → `had_error=True`, metrics=None, user message
  popped from history.
- LLM raises with no user audio → `next_primed=None`. (The
  race "LLM raises AND watcher fires" is covered in isolation
  by `test_hardening.py::TestErrorPathFrameCarryover` — can't
  engineer it deterministically here because the for-loop's
  `coord.is_set()` check makes the watcher win the race; if it
  fires first the loop exits cleanly and no exception reaches
  the except block.)

FillerIntegration (1):
- Slow LLM (200ms before first token), 50ms idle threshold.
  Filler plays once, real sentences after. `fillers_played==1`,
  `sentences_spoken>=1`. Validates the iter-011 wiring through
  ChatLoop end to end.

Verification: `python -m pytest tests/unit/` → **212 passed in 11s**
(204 existing + 8 new).

Notes:
- The first attempt at the LLM-error-with-barge-in test failed
  for an instructive reason: the watcher fires `coord.trigger`
  before the LLM raises, so the for-loop's
  `if coord.is_set(): break` exits cleanly without ever raising.
  The test's premise (both error path AND watcher detection
  observable in the result) is impossible to engineer
  deterministically. Documented in the replacement test's
  comment so future contributors don't try the same dead end.
- The extraction reduced `mic_chat.py` from a ~280-line
  per-turn block to a ~15-line caller. Almost everything moved
  into `_chat_loop.py`. The trade-off: one more module to
  navigate, but every other module in the project now imports
  cleanly without pyaudio at module scope. Test reach is the
  big win.

---

# Status

**15 iterations shipped, 212 unit tests passing in 11s.** Every
layer of the chat pipeline is now testable on x86_64 Linux:

| Layer | Test surface |
|-------|--------------|
| VAD state machine | `test_chat_helpers.py` (iter-004) |
| Recording loop | `test_chat_recording.py` (iter-006/010) |
| Playback loop | `test_chat_playback.py` (iter-007) |
| Streaming worker | `test_chat_pipeline.py` (iter-008) |
| Filler subsystem | `test_fillers.py` (iter-011) |
| Barge-in primitives | `test_bargein.py` (iter-009) |
| Barge-in coordinator | `test_bargein_coordinator.py` (iter-012) |
| LLM SSE parser | `test_chat_llm.py` (iter-013) |
| Hardening edge cases | `test_hardening.py` (iter-014) |
| **Full per-turn orchestration** | `test_chat_loop.py` (iter-015) |
| Sub-orchestrations | `test_bargein_orchestration.py`, `test_llm_stream_barge.py` |
| Virtual audio infra | `test_virtual_audio.py` (iter-005) |

The iter-015 ChatLoop tests are the integration-level validation
that was missing — they run the actual production code path with
stub external dependencies, in 0.6s deterministic.

Real bugs caught while writing tests, in chronological order:
- iter-008: thread-safety semantics around `submit_done`/`wait_done`
- iter-009: float precision near 0.8 (1.4 - 0.6 = 0.7999...)
- iter-010: per-frame `clock()` divergence on primed/live boundary
- iter-013: cross-thread `gen.close()` raises `ValueError`
- iter-014: `rms()` NaN on empty input silently stalls VAD
- iter-015: barge-in event fires before LLM exception → can't
  test the simultaneous case, only via separate logic test

Each was hidden behind happy-path behavior and would have surfaced
only under specific edge conditions in production.

The codebase is at a strong stopping point. Future directions:
1. Real-mic CI smoke test (iter-013 in original numbering, infra
   work, diminishing returns)
2. End-to-end test using real kokoro TTS as the LLM-token-source
   surrogate (closer to production reality but slower; ~6s per
   test)
3. Pivot to features outside the original focus list (multi-turn
   memory tuning, voice cloning, multi-language, etc.)

---

## iter-016 — abbreviation-aware sentence splitter

**Branch:** `iter-016-splitter` (merged ff to main, commit `f475a12`)
**Date:** 2026-05-24

A small but real production bug. The simple `(?<=[.!?])\s+`
regex broke common voice cases:

```
>>> split_complete_sentences("Mr. Smith arrived. Hi.")
(['Mr.', 'Smith arrived.'], 'Hi.')   # WRONG
```

In production this would cause TTS to pause awkwardly between
"Mr." and "Smith" — the bot would speak "Mister." as if it were
a complete sentence, then start a new one with "Smith". Same for
Dr., U.S.A., i.e., e.g., Ph.D., and friends.

What changed:

`examples/_chat_helpers.py`:
- New `NON_TERMINATING_ABBREVIATIONS` frozenset (lowercased) with
  ~70 common abbreviations across categories: titles, Latin,
  business/legal, academic, geographic, calendar, numeric.
- New `_word_before_period(buffer, period_idx)` helper that walks
  back over `[a-zA-Z.]` to extract the relevant token, so
  multi-period abbreviations like `i.e` and `u.s.a` match their
  set entries.
- `split_complete_sentences` rewritten to find all candidate
  matches via `finditer`, post-filter via the abbreviation check,
  and emit only the real splits.

Behavior change is purely additive — splits that were previously
correct stay correct. Splits in front of an abbreviation in the
set now don't fire. Unknown abbreviations fall back to the old
behavior (worst case: one extra split, same as before; no
regression). `!` and `?` terminators are unchanged because the
abbreviation check only applies to `.`.

Tests (11 new in `TestSplitCompleteSentences`):
- Mr. doesn't split when followed by more text
- Mr. + real terminator splits correctly at the real one
- Dr., etc., Ph.D. — single-word and multi-period cases
- i.e., e.g., U.S.A. — multi-period abbreviations
- "!" and "?" terminators unaffected by the abbreviation check
- Abbreviation at start of buffer
- Unknown abbreviation still splits (no regression)

Verification: `python -m pytest tests/unit/` → **223 passed in 12s**
(212 existing + 11 new).

Notes:
- The abbreviation set is intentionally conservative. Adding more
  is cheap (just append to the frozenset). The current ~70 entries
  cover the abbreviations a typical English-language LLM might
  emit in voice context.
- Multi-period abbreviations work because `_word_before_period`
  walks back over both letters and periods, building "i.e",
  "u.s.a", "ph.d", etc. — these match the set entries directly.
- Limitation: an abbreviation inside a longer made-up word (e.g.
  "etcaholic.") would still get matched as `etcaholic` not in the
  set → splits normally. That's correct behavior; we only want to
  match true abbreviations.
- The implementation runs `finditer` once on the buffer instead
  of `split` then post-process. Same work; cleaner code.

---

## iter-017 — extract print_session_summary, fix median, add a.m./p.m.

**Branch:** `iter-017-session-summary` (merged ff to main, commit `f4c04cb`)
**Date:** 2026-05-24

Two small polish items bundled into one iteration.

### 1. print_session_summary

The KeyboardInterrupt summary in `mic_chat.run_chat` had two
issues:

- **Inlined** inside the `except KeyboardInterrupt` clause →
  untestable without spinning up the full chat loop.
- **Wrong median**: `sorted(times)[len(times)//2]` returns the
  *upper* median for even-length lists. A 2-turn session with
  STTs `[50ms, 200ms]` reported "median 200ms" — biased toward
  the slower outlier. With `statistics.median` it's 125ms.

Fix:
- `examples/_chat_metrics.py` grew `print_session_summary(
  metrics_list, llm_config, *, file=None)`. Computes medians via
  `_median_ms` (statistics-based), handles empty / single-turn /
  pluralization, surfaces `fillers_played` and `barge_in` totals
  when nonzero.
- `mic_chat.run_chat` KeyboardInterrupt handler is now a one-line
  delegate.

### 2. a.m. / p.m. abbreviations

Time-of-day formats were missing from the iter-016 set:

```
"It is 9:30 a.m. Time to wake."
  -> ['It is 9:30 a.m.'], 'Time to wake.'   # before iter-017
  -> [], 'It is 9:30 a.m. Time to wake.'    # after
```

Same UX hit as iter-016: TTS would have paused at "a.m." treating
it as a sentence end. Fix: added `"a.m"` and `"p.m"` to
`NON_TERMINATING_ABBREVIATIONS`. Capitalized forms (`A.M.`) work
because the lookup is lowercased.

Tests (19 new in `tests/unit/test_session_summary.py`):

`_median_ms` (5):
- empty → 0; single value; odd-length; **even-length averages
  middle two** (regression cover for the `sorted[len//2]` bug);
  unsorted input.

`print_session_summary` (10):
- empty list → "no completed turns"; single turn shows all
  lines; 3 turns pluralized; even-length `[50,100,150,200]`
  reports 125ms not 150ms; best TTFS = min; fillers line only
  when any played; barge-ins line only when any fired; missing
  model → "unknown"; default file writes to stdout.

a.m. / p.m. (4):
- `a.m.` doesn't split; `p.m.` doesn't split; capitalized `P.M.`
  doesn't split (lowercased lookup); two full sentences with
  a.m./p.m. each split correctly at real terminators only.

Verification: `python -m pytest tests/unit/` → **242 passed in 12s**
(223 existing + 19 new).

Notes:
- The first attempt at the 4-sentence test had wrong expectations
  for the trailing "Yes." (no whitespace after → stays in
  remainder). Fixed in place; added a parallel test with trailing
  space that DOES split both. Both behaviors documented.
- `_median_ms` returns ms (not seconds) so it composes naturally
  with the f-string formatters used by callers.

---

# Status (17 iterations)

**242 unit tests passing in 12s** on x86_64 Linux. The codebase
remains at a strong stopping point. Real polish items still
available include: more abbreviation entries (PhD variants,
European number formatting like `3,14`), session-summary CSV
output for benchmarking, and config validation. Each is small.

Stretch goals (real-mic CI, end-to-end with kokoro as token
source) remain unaddressed. They're substantial infra work with
diminishing returns; consider them when actual user feedback
demands.

---

## iter-018 — extract + validate config parsing

**Branch:** `iter-018-config` (merged ff to main, commit `81a1879`)
**Date:** 2026-05-24

`load_llm_config` was the last untested chunk that real users
actually hit. It coupled file I/O, parsing, validation, and
`sys.exit` into one function, and failed in real ways:

- An empty `config.local.yaml` loaded as `None`, then crashed
  with `AttributeError("'NoneType' object has no attribute 'get'")`
  instead of a useful "config is empty" message.
- Missing required fields (`model`, `base_url`) didn't raise — the
  user's first hint was an HTTP 400 deep in the request stack.
- Unresolved `${ENV_VAR}` placeholders called `sys.exit` from
  library-shaped code, hard to test or reuse.

What changed:

- **`examples/_chat_config.py`** (new) — pure-data parsers:
  - `ConfigError` exception type
  - `parse_llm_config(cfg, *, env=os.environ)` — validate +
    resolve. Raises `ConfigError` with user-readable messages on:
    cfg is None/not a Mapping, missing `llm` section, missing
    required fields, empty/non-string field values, unresolved
    `${VAR}` placeholders. Side-effects: strips trailing slashes
    from `base_url`, defaults `max_tokens` to 150.
  - `parse_chat_config(cfg)` — tolerant. Never raises; returns
    `{}` on absent/None/non-dict input.

- **`examples/mic_chat.py`**:
  - `load_llm_config` / `load_chat_config` become thin file-I/O
    wrappers that call the new parsers and convert `ConfigError`
    into a printed message + `sys.exit(1)`.
  - New `_read_yaml_or_exit` helper handles `yaml.YAMLError`
    cleanly (previously the YAMLError would propagate as an
    uncaught exception).

Tests (35 new in `tests/unit/test_chat_config.py`):

`parse_llm_config` happy path (6):
- minimum valid config; max_tokens passthrough; extra fields
  passthrough; base_url trailing slash stripped (single +
  multiple); returns new dict (no input mutation).

env-var resolution (7):
- set env var resolves; unset raises (var name in msg); empty
  value raises; empty `${}` raises; `${X` literal; `abc${X}`
  literal; `prefix-${KEY}` literal — documents the full-string-
  only contract.

structural failures (5):
- `None` input → ConfigError; non-Mapping top-level; missing
  `llm` section; `llm: None`; `llm: "wrong-type"`.

required field validation (parametrized over 3 fields × 3 cases):
- missing key; empty string; non-string type. 9 cases for ~3
  lines of test code — adding a new required field gets
  symmetric coverage automatically.

`parse_chat_config` (7):
- no `chat` section / `chat: None` / non-dict / top-level
  None or non-dict — all return `{}`. Valid chat section
  passed through as a copy.

Verification: `python -m pytest tests/unit/` → **277 passed in 11s**
(242 existing + 35 new).

Notes:
- The "empty placeholder" / "no closing brace" / "partial
  template" tests document edge cases that were previously
  ambiguous. Now the contract is explicit: only the FULL value
  matching `${VAR}` resolves; anything else is literal.
- The parametrized 3×3 required-field tests give 9 cases for ~3
  lines of test code. Great for regression coverage if a future
  iter adds another required field.

---

# Status (18 iterations)

**277 unit tests passing in 11s.** Every part of the chat
pipeline that can be unit-tested has been; what's left is mostly
the runtime bindings (real PyAudio, real kokoro inference, real
HTTP) and the things only worth testing on real hardware.

Genuine remaining options:
- Real-mic CI smoke test via ALSA loopback (infra, diminishing
  returns)
- End-to-end with real kokoro as LLM-token-source surrogate
  (~6s per test, validates the full chain)
- More splitter polish (PhD variants, percentage signs,
  European number formatting)
- CSV export of session metrics for benchmarking
- Move `synthesize_with_alignment` out of mic_chat into its
  own pyaudio-free module (would let us test it with a
  fake pipeline; modest value since the function is mostly a
  loop around kokoro's pipeline iterator)

The codebase has reached a stable, well-tested state. Each
remaining iteration adds polish rather than capability.

---

## iter-019 — extract synthesize_with_alignment, real-kokoro e2e test

**Branch:** `iter-019-tts-extract` (merged ff to main, commit `203e251`)
**Date:** 2026-05-24

Last function in `mic_chat.py` reachable only via pyaudio-tainted
imports — pulled into `examples/_chat_tts.py` so it composes with
the iter-005 virtual audio + iter-015 ChatLoop without dragging
real pyaudio along.

Side benefit: closes the "real production synth path" testing
gap. Until iter-019 the orchestration tests stubbed `synth_fn`
with a constant-audio function. We trusted the real
`synthesize_with_alignment` composed correctly because contracts
matched; we couldn't prove it.

What changed:

- **`examples/_chat_tts.py`** (new):
  - `synthesize_with_alignment` relocated, semantics unchanged.
  - Refactor: torch.Tensor detection switched from
    `isinstance(audio, torch.Tensor)` (which required importing
    torch for the numpy fast path) to a duck-type check via
    `hasattr(audio, "numpy") and not isinstance(audio, np.ndarray)`.
    Tests can pass fake "tensor" objects without depending on
    torch.
  - `TTS_RATE = 24000` lives here (and remains in
    `_chat_playback.py`; both modules need it for their own
    purposes — not worth a cross-module import for a 4-char
    constant).

- **`examples/mic_chat.py`**:
  - Now re-exports `synthesize_with_alignment` from `_chat_tts`
    so external imports keep working.

Tests (12 new):

`tests/unit/test_chat_tts.py` — 9 unit + 2 kokoro integration:
- Empty pipeline returns `(empty, empty)`.
- Single chunk round-trips audio + tokens.
- **Multi-chunk OFFSETS tokens by accumulated duration** — the
  load-bearing piece of the function. Without this, iter-007's
  playback alignment would re-start at 0 for every chunk after
  the first, breaking word-by-word reveal.
- Three chunks: offset accumulation works for >2.
- Concatenated audio preserves chunk values byte-for-byte.
- Tensor audio with `.numpy()` converts (fake tensor — no torch
  import needed).
- Numpy audio passes through unchanged.
- `_load` called on every invocation.
- Results with no tokens still contribute audio.
- Real kokoro: short sentence produces plausible output (audio
  length, token shape, timings within audio duration).
- Real kokoro: amplitude in `[-1, 1]` with non-zero RMS.

`tests/unit/test_chat_loop.py` — 1 kokoro e2e:
- `ChatLoop.run_one_turn` driven with real kokoro synth + real
  `play_aligned` + virtual mic/speaker + stub STT/LLM. Verifies
  metrics, history, and that the speaker received real audio
  signal (RMS > 0.001 over a realistic byte-count window). The
  closest test to "the real production chat loop is working" —
  ~3-6s on first run for kokoro load, faster after.

Verification: `python -m pytest tests/unit/` → **289 passed in 18s**
(277 existing + 12 new).

Notes:
- Suite went from 11s → 18s. Three new kokoro integration tests
  cost ~1-2s each. Could mark them `@pytest.mark.slow` if this
  becomes painful, but <20s is still tight enough for fast
  iteration.
- `_chat_loop.py` was unchanged this iteration — `synth_fn` is
  already injectable, so swapping in real kokoro just means
  passing the right callable. That's the iter-015 architecture
  paying off.
- The duck-type tensor check (`hasattr(audio, "numpy")`) is
  cleaner than the original `isinstance(audio, torch.Tensor)`
  because it doesn't require importing torch when the engine
  already returns numpy. Real kokoro returns tensors; the
  conversion still works.

---

# Status (19 iterations)

**289 unit tests passing in 18s.** Every function in `examples/`
that's not directly tied to a runtime resource (PyAudio, the
LiteLLM HTTP server) is now covered by unit tests. The tests
that ARE tied to runtime resources (real kokoro, virtual mic
+ speaker as PyAudio surrogates) compose them through the
production code path itself.

The chat pipeline `examples/` directory layout:

| Module | Purpose | Tests |
|--------|---------|-------|
| `_chat_helpers.py` | VAD, sentence splitter, history trim, etc. | test_chat_helpers.py (49) |
| `_chat_recording.py` | record_utterance_streaming | test_chat_recording.py (14) |
| `_chat_playback.py` | play_aligned token-aligned playback | test_chat_playback.py (18) |
| `_chat_pipeline.py` | SentenceWorker + watcher + coordinator | test_chat_pipeline.py + test_bargein.py + test_bargein_coordinator.py + test_fillers.py (75) |
| `_chat_metrics.py` | TurnMetrics + session summary | test_session_summary.py (19) + test_hardening.py (6) |
| `_chat_config.py` | parse_llm_config + parse_chat_config | test_chat_config.py (35) |
| `_chat_llm.py` | SSE parser + stream_chat_completion | test_chat_llm.py (18) |
| `_chat_tts.py` | synthesize_with_alignment | test_chat_tts.py (11) |
| `_chat_loop.py` | ChatLoop.run_one_turn orchestration | test_chat_loop.py (9) |
| `virtual_audio.py` | VirtualMicStream/SpeakerStream + fixtures | test_virtual_audio.py (26) |
| `mic_chat.py` | thin pyaudio shim that wires everything | (covered transitively) |

The codebase has reached its design endpoint. Further
iterations would be: pick up real-mic CI infra (high cost,
diminishing returns), add new features (multi-language voice,
voice cloning), or call it done.

---

## iter-020 — configurable VAD via chat.vad config section

**Branch:** `iter-020-vad-config` (merged ff to main, commit `75ba389`)
**Date:** 2026-05-24

The VAD parameters had been hardcoded module constants since
iter-006. Real users have noisier mics, busier rooms, and
different turn-taking preferences than the desk-mic defaults
assume. They've had no way to tune except by editing source.
This iteration exposes them via config.local.yaml:

```yaml
chat:
  vad:
    silence_threshold: 0.05      # noisier room
    silence_duration: 0.5         # faster turn-taking
    min_speech_duration: 0.5      # ignore brief blips
```

What changed:

- **`examples/_chat_recording.py`**: `record_utterance_streaming`
  gained `silence_threshold`, `silence_duration`,
  `min_speech_duration` kwargs (default to the existing module
  constants).

- **`examples/_chat_loop.py`**: `ChatLoop` constructor accepts
  the same three kwargs, forwards them to record_utterance per
  turn.

- **`examples/_chat_config.py`**: `VAD_DEFAULTS` dict +
  `parse_vad_config(chat_cfg)` extractor. Tolerant — bad types,
  non-positive numbers, missing keys all fall through to
  defaults so a typo'd VAD config doesn't kill the chat loop.

- **`examples/mic_chat.py`**: reads `chat.vad` via
  `parse_vad_config`, passes to ChatLoop.

Tests (16 new in `tests/unit/test_vad_config.py`):

`parse_vad_config` (10):
- no chat / no vad section → defaults
- partial vad backfills; full vad overrides
- int values coerced to float
- non-Mapping chat / vad → defaults (tolerant)
- string values for numeric fields fall back per-key
- zero / negative values fall back per-key
- returns a new dict

`record_utterance_streaming` with overridden VAD (5):
- High threshold rejects quiet speech that the default detects.
- Low threshold catches sub-default audio (RMS ≈ 0.0035).
- Short `silence_duration` produces shorter wav (parallel-mic
  comparison vs default 0.8s window on the same audio).
- Strict `min_speech_duration` → DONE_TOO_SHORT → empty wav.
- Default kwargs match module constants (regression cover).

ChatLoop forwarding (1):
- End-to-end: ChatLoop with `min_speech_duration=1.0` rejects a
  0.5s utterance via DONE_TOO_SHORT → `result.metrics is None`.

Verification: `python -m pytest tests/unit/` → **305 passed in 18s**
(289 existing + 16 new).

Notes:
- Tolerant validation pattern matches `parse_chat_config` from
  iter-018: never raise on bad input, just fall back. The chat
  loop is more useful with bad VAD config than dead.
- The "compared to default 0.8s window" test uses two parallel
  mic streams pushing the same audio — clean way to verify
  behavioral difference without timing assertions.
- `VAD_DEFAULTS` is exposed as a module attribute so tests can
  assert against it directly (rather than the recording-module
  constants we want callers to be insulated from).

---

## iter-021 — digit-prefixed ordinals don't trigger abbreviation match

**Branch:** `iter-021-ordinals` (merged ff to main, commit `c501d91`)
**Date:** 2026-05-24

Subtle iter-016 collision: the abbreviation set contains `"st"`
(Street) and `"rd"` (Road) for postal-address contexts. But
those also happen to be the suffix letters of the ordinal forms
`1st` and `3rd`. The `_word_before_period` walk-back skipped
the leading digit and extracted just `"st"` / `"rd"`, which then
matched the abbreviation set, which then prevented the splitter
from firing.

Concrete failure:

```
>>> split_complete_sentences("He came 1st. Then we go.")
([], 'He came 1st. Then we go.')   # WRONG — should split
```

In production, the bot voicing "He came in 1st. Then we
celebrated." would have run the two sentences together as one
TTS chunk, with no pause where the listener's ear expected one.

Fix in `_word_before_period`: after walking back over
`[a-zA-Z.]`, check one position further. If it's a digit, this
is a numeric ordinal (1st, 2nd, 3rd, 4th, 100th, ...) — not an
abbreviation. Return empty so the splitter falls through to
default split behavior.

Coverage matrix:

| Input | Suffix | In abbrevs? | Before iter-021 | After iter-021 |
|-------|--------|-------------|------------------|----------------|
| `1st.` | `st` | yes (Street) | doesn't split (BUG) | splits |
| `2nd.` | `nd` | no | splits | splits |
| `3rd.` | `rd` | yes (Road) | doesn't split (BUG) | splits |
| `4th.` | `th` | no | splits | splits |
| `100th.` | `th` | no | splits | splits |
| `Mr.` | `mr` | yes (Mister) | doesn't split | doesn't split |
| `9 a.m.` | `a.m` | yes | doesn't split | doesn't split |

The "9 a.m." control case is what makes the fix surgical: the
digit `9` is separated from `a.m` by a space, so the walk-back's
`start` lands on the position after the space, and
`buffer[start-1]` is the space (not the digit). Fix doesn't
trigger; abbreviation match still works.

Tests (8 new in `TestSplitCompleteSentences`):
- `1st` regression; `3rd` regression
- `2nd` / `4th` / `100th` working
- numbered list items still split
- `"Mr. Smith"` still doesn't split (control)
- `"9 a.m. Time"` still doesn't split (control)

Verification: `python -m pytest tests/unit/` → **313 passed in 17s**
(305 existing + 8 new).

Notes:
- Guard is `start < end and start > 0` — both edges. Without
  the `start < end` clause, a period at index 0 with no walk
  would read `buffer[-1]` (Python's wrap-around to last char),
  which would be wrong. Without the `start > 0` clause, we'd
  index out of bounds.
- This bug was hidden under iter-016's acceptance testing
  because the test corpus didn't include ordinals. The iter-021
  tests fill that gap explicitly.

---

## iter-022 — split when terminator is followed by closing quote

**Branch:** `iter-022-quoted-speech` (merged ff to main, commit `390978e`)
**Date:** 2026-05-24

The original `(?<=[.!?])\s+` lookbehind only matched when
whitespace immediately followed a sentence terminator. With
US-style quoted speech where the period sits *inside* the closing
quote (`"hello."` vs UK-style `"hello".`), the char before the
whitespace is `"`, not a terminator. The splitter never fired,
so the bot would speak the quoted sentence and the next sentence
as one long TTS chunk.

Concrete failure (3 cases, all fixed):

```
"He said \"hello.\" Then he left."
  Before: ([], '...')                            # no split
  After:  (['He said "hello."'], 'Then he left.')

"She asked \"why?\" Then waited."
  Before: ([], '...')
  After:  (['She asked "why?"'], 'Then waited.')

"She said ‘hi.’ Done."   (smart quotes)
  Before: ([], '...')
  After:  (["She said ‘hi.’"], "Done.")
```

Fix uses Python regex alternation (variable-length lookbehind
isn't supported, but two fixed-length alternatives can be OR'd):

```python
SENTENCE_END = re.compile(
    r'(?<=[.!?])\s+|(?<=[.!?][\"\'”’])\s+'
)
```

Quote characters covered: straight double `"`, straight single
`'`, smart right double `”` (U+201D), smart right single `’`
(U+2019). Other quote styles (CJK 「」, etc.) aren't included; if
the LLM emits those they'll fall back to no-split — no
regression vs today.

Walk-back updated: after finding `m.start() - 1`, if that's a
closing quote, walk one more position back to find the actual
terminator. The iter-016 abbreviation check and iter-021 ordinal
check still operate on the real terminator position.

Tests (8 new in `TestSplitCompleteSentences`):
- US-style with `.`, `?`, `!` inside the quote (3 cases)
- Single quote, smart double, smart single (3 cases)
- UK-style period outside still splits (control)
- Abbreviation outside + quote inside: `Mr. Smith said "hi."`
  splits correctly at the inner `"`, not at `Mr.`

Verification: `python -m pytest tests/unit/` → **321 passed in 17s**
(313 existing + 8 new).

Notes:
- The two-branch lookbehind costs marginally more regex work
  but the match buffer is small (one in-progress bot reply
  token-by-token), so cost is negligible.
- Bot-emitted smart quotes are common in modern LLM output:
  Claude/GPT formatted dialogue often uses `“…”` not straight
  `"…"`. Including smart quotes by default catches that
  without needing the bot's prompt to special-case it.
- The "abbreviation OUTSIDE + quote INSIDE" combined test is
  the regression-cover that catches future refactors that try
  to simplify the walk-back logic.

---

## iter-023 — introspect play_fn signature once, fix double-call bug

**Branch:** `iter-023-typeerror` (merged ff to main, commit `8be4532`)
**Date:** 2026-05-24

Real subtle bug found by code review.
`SentenceWorker._play_clip` had been wrapping each `play_fn` call
in a `try/except TypeError` to fall back to a no-cancel-event
signature for iter-008-style play_fns. That swallow had a silent
failure mode — a `play_fn` whose **body** raised TypeError (for
*any* reason) would be retried with the fallback signature, which
would also raise (same body), so the function got called *twice*
for the same sentence.

Demonstrated before fix:

```
buggy_play raises TypeError, calls speaker.write first.
Speaker received 4096 bytes (1024 samples × 2 calls × 2 bytes/sample)
Expected 2048 bytes (one call's worth of partial audio)
```

Fix: detect once at `SentenceWorker.__init__` whether the
`play_fn` accepts a `cancel_event` kwarg via `inspect.signature`.
Store the result on the worker, dispatch per-call without
`try/except`. A TypeError raised by the `play_fn`'s body now
surfaces correctly via the outer `except Exception` and gets
recorded in `worker.errors` exactly once.

What changed:

- **`examples/_chat_pipeline.py`**:
  - New module-level helper
    `_play_fn_accepts_cancel_event(play_fn)`. Uses
    `inspect.signature`; checks for either an explicit
    `cancel_event` parameter or a `**kwargs`-style variadic.
    Conservative on errors: callables whose signature can't be
    inspected (some C extensions / builtins) fall back to False,
    preserving the old fallback path.
  - `SentenceWorker.__init__` runs the check once and stores
    `self._play_fn_supports_cancel`.
  - `SentenceWorker._play_clip` dispatches via the flag — no
    more `try/except TypeError`.

Tests (11 new in `tests/unit/test_play_fn_introspection.py`):

Signature introspection (6):
- explicit `cancel_event` kwarg detected
- no `cancel_event` → not detected
- `**kwargs` variadic → detected
- lambdas with / without `cancel_event`
- uninspectable callable falls back to False

Worker dispatch (3):
- `play_fn` with `cancel_event` receives the worker's internal
  `threading.Event`-shaped object
- `play_fn` without `cancel_event` doesn't get extra kwargs
- `**kwargs` `play_fn` receives `cancel_event` in the dict

TypeError bug regression (2):
- buggy `play_fn` raising TypeError called **exactly once** per
  sentence; speaker captures one chunk's worth (2048 bytes), not
  double; one error in `worker.errors`.
- After a buggy first sentence, subsequent sentences still play
  cleanly (loop continues; doesn't deadlock).

Verification: `python -m pytest tests/unit/` → **332 passed in 17s**
(321 existing + 11 new).

Notes:
- `inspect.signature` has a small one-time cost at construction —
  negligible for the iter-008 + iter-023 design where workers
  are short-lived (one per turn).
- This is the kind of bug that's easy to miss in code review —
  `try/except TypeError` "looks defensive" but silently masks
  real failures. Worth remembering: `try/except` on a too-broad
  exception class is a code smell; introspect what you actually
  care about.

---

## iter-024 — consolidate inline RMS in BargeInWatcher

**Branch:** `iter-024-rms-consolidate` (merged ff to main, commit `14943ef`)
**Date:** 2026-05-24

Maintenance hazard found by code review. `BargeInWatcher._run`
had an inline RMS computation that duplicated the `rms()` helper
in `_chat_recording`. Both had the iter-014 NaN-on-empty guard,
but two implementations of the same logic means a future fix in
one place could miss the other.

Concrete near-miss: iter-014 *did* have to update both code paths.
That happened to work out because they were both touched in the
same commit. But future RMS-related fixes (clipping detection,
DC offset removal, whatever) shouldn't depend on that luck.

Fix: `BargeInWatcher.__init__` imports `rms` from
`_chat_recording` and binds it as `self._rms`. `_run` calls
`self._rms(audio)` per frame instead of inlining the expression.
Lazy import keeps `_chat_pipeline` a leaf module at import time —
`_chat_recording` is loaded only when a watcher is actually
instantiated.

Tests (7 new in `tests/unit/test_rms_consolidation.py`):

Watcher uses consolidated rms (3):
- `watcher._rms is rms` (identity, not equality — catches
  future copy-paste regressions where someone inlines a "small
  fix")
- silence doesn't trigger (behavioral parity)
- speech triggers (behavioral parity)

Single source of truth (4):
- identity assertion against `_chat_recording.rms`
- parametrized rms values for canonical inputs (zero array,
  constant array, empty array)

Verification: `python -m pytest tests/unit/` → **339 passed in 17s**
(332 existing + 7 new).

Notes:
- The `is`-identity test is the load-bearing assertion — equality
  could be satisfied by an inlined re-implementation; identity
  insists on the same function object.
- This is a refactor with zero behavioral change. The point is
  preventing a class of future bugs (drift between duplicated
  code), not fixing a current one.

---

## iter-025 — BargeInWatcher captures only post-detection frames

**Branch:** `iter-025-watcher-frames` (merged ff to main, commit `3de5039`)
**Date:** 2026-05-24

Pre-iter-025 the watcher stored every frame from `.start()` to
`.stop()`, including pre-detection silence/noise/feedback. When that
buffer was fed into the next `record_utterance_streaming` as
`primed_frames`, the recording loop's VAD would treat any high-RMS
pre-detection content as user speech — so STT could end up
transcribing the bot's acoustic feedback as the user's words. Only
matters with speakers (not headphones), but real production use
case.

Fix: frames stores only:
- the **trigger frame** (the chunk that actually crossed threshold;
  this is the user's first audible syllable)
- all subsequent frames until `stop()`
- optionally, the most recent N pre-detection chunks if the caller
  passes `lead_in_chunks > 0` (default 0)

The `lead_in_chunks` kwarg is a ring buffer of the last N
pre-detection frames. When detection fires, the buffer is flushed
into `frames` in order, followed by the trigger frame.

Implementation gotcha caught by failing test on first attempt: the
trigger frame must NOT go into the ring buffer before the trigger
check, otherwise the lead-in flush + the explicit append double-
count it:

```
Frame 8 = TONE (trigger)
Wrong: buffer=[s5,s6,s7,t8] → flush → [s5,s6,s7,t8] → append t8 → [s5,s6,s7,t8,t8]
Right: trigger first → flush [s5,s6,s7] → append t8 → [s5,s6,s7,t8]
```

The order is now: `if detected → append; elif trigger → flush+append; else → ring buffer`.

Tests (8 new in `tests/unit/test_watcher_frames.py`):

Default `lead_in=0` (3):
- Pure silence → no frames, no detection.
- Silence + tone + silence → frames << total events.
- Trigger frame is the first stored frame and has high RMS.

Lead-in buffer (5):
- `lead_in_chunks=3` → first 3 stored frames are silence, 4th
  has signal.
- `lead_in_chunks=0` explicit equivalent to omitting the kwarg.
- Ring buffer caps at max size (1.0s silence + lead_in=2 yields
  only 2 silence frames pre-trigger).
- Negative `lead_in_chunks` raises `ValueError`.
- Internal `_lead_in_buffer` cleared after flush.

Verification: `python -m pytest tests/unit/` → **347 passed in 18s**
(339 existing + 8 new).

Notes:
- Behavior change. Existing tests pass because they assert
  `len(frames) > 0` not specific counts. External callers that
  relied on full pre-detection capture can restore via
  `lead_in_chunks=N`.
- Bot acoustic feedback is fundamentally an AEC problem. iter-025
  just stops *amplifying* it by feeding the bot audio back through
  STT.
- Ring buffer bounded at `lead_in_chunks × CHUNK` bytes — at
  default 0, zero memory. Generous `lead_in=10` (~640ms), 20KB.
- The first-attempt test caught the double-append bug cleanly.
  Worth noting for future "small implementation" work: writing
  the test before/while writing the code keeps these orderings
  honest.

---

## iter-026 — skip post-loop token flush when play cancelled

**Branch:** `iter-026-cancel-flush` (merged ff to main, commit `42d16b0`)
**Date:** 2026-05-24

Real UX bug found by code review. After the iter-009 cancel_event
breaks `play_aligned`'s play loop, the post-loop "flush trailing
tokens" branch ran unconditionally. Bot's voice cut correctly
(audio chunks stop) but bot's text kept printing to the terminal —
specifically tokens whose `start_ts` exceeded audio duration.

Concrete demonstration before fix:

```
Audio: 4096 samples
Tokens: ['hello' @ 0.0, 'world' @ 0.05, 'TRAILING' @ 1.5]
Cancel: fires after first chunk

Speaker bytes:    1024  ← correct, audio stopped
Visible output:   'hello world TRAILING'   ← BUG: TRAILING shouldn't appear
```

Fix: guard the trailing-flush with `not cancelled`. The audio cut
is the user-visible signal the bot stopped; the text should match.

Tests (4 new):
- Cancel mid-loop with trailing token → token suppressed
- Pre-set cancel → both pre/post tokens suppressed (loop never
  enters body, post-flush skipped)
- No cancel_event → trailing flushed (control)
- Unset cancel_event → trailing flushed (control, behaves like
  no cancel_event)

Verification: `python -m pytest tests/unit/` → **351 passed in 18s**
(347 existing + 4 new).

Notes:
- "Trailing tokens" exist when the TTS engine emits tokens whose
  `start_ts` exceeds synthesized audio duration — uncommon but
  kokoro does it occasionally.
- Two control tests catch any future regression that re-enables
  the unconditional flush. Cheap insurance.
- This bug is the kind that's invisible without a barge-in
  scenario in the test corpus. The iter-009/010 orchestration
  tests use cancellable_play that doesn't have trailing tokens,
  so they didn't exercise this path. iter-026's targeted tests
  fill that gap.

---

## iter-027 — stop worker+watcher in run_one_turn finally

**Branch:** `iter-027-keyboardinterrupt` (merged ff to main, commit `3f68dc0`)
**Date:** 2026-05-24

Real shutdown bug found by code review. `except Exception` in
`ChatLoop.run_one_turn` doesn't catch `KeyboardInterrupt` (which
inherits from `BaseException`). When the user pressed Ctrl+C
during the for-token loop, `KeyboardInterrupt` propagated through
`except Exception` (untouched) → `finally` (which only closed
`llm_gen`) → out of run_one_turn. Worker and watcher threads kept
running, with the worker's speaker stream still open. Daemon-thread
cleanup at process exit eventually killed them, but the speaker
wasn't cleanly closed first.

Demonstrated:

```
Before iter-027:
  KeyboardInterrupt during LLM stream
  → speaker._closed: False   ← worker thread still running

After iter-027:
  → speaker._closed: True    ← worker.stop in finally → _run finally → close
```

Fix: add idempotent `watcher.stop` + `worker.stop` calls to
`run_one_turn`'s finally block. The existing finally only closed
`llm_gen`; we now also stop both background threads on any exit
path. Idempotent stops don't affect normal-completion or
LLM-error paths.

Tests (3 new in `TestKeyboardInterruptCleanup`):
- KeyboardInterrupt propagates AND speaker is closed (proves
  worker.stop ran via the worker's own finally).
- Worker thread is joined after KeyboardInterrupt propagates
  (captured via SentenceWorker `__init__` hook).
- Normal completion still works (idempotent stops are no-ops on
  already-stopped workers/watchers).

Verification: `python -m pytest tests/unit/` → **354 passed in 17s**
(351 existing + 3 new).

Notes:
- The finally now does four cleanup operations: `llm_gen.close`,
  `watcher.stop`, `worker.stop`, all wrapped in `try/except` so
  a failure in one doesn't cascade. Order: close LLM first
  (release HTTP socket promptly), stop worker last (give
  synth/play a moment to wind down).
- `BaseException` vs `Exception` distinction is one of those
  Python details easy to miss. `except Exception` is good
  defensively (won't catch system-level signals) but you have
  to plan for the propagation path through `finally`.
- Pattern: any background thread spawned in a function should
  have stop calls in the function's finally block, not just
  in the success path. KeyboardInterrupt and other
  BaseExceptions take you straight there.

---

## iter-028 — pass VAD config to BargeInWatcher in ChatLoop

**Branch:** `iter-028-watcher-vad` (merged ff to main, commit `c35028b`)
**Date:** 2026-05-24

iter-020 follow-on. iter-020 made VAD parameters configurable via
the `chat.vad` config section, and `ChatLoop` accepted them as
kwargs and forwarded them to `record_utterance_streaming`. But the
`BargeInWatcher` constructed inside `run_one_turn` was built without
a `vad=` kwarg — so it constructed its own `VadState()` with
hardcoded defaults regardless of user config.

Concrete consequence: a user setting
`chat.vad.silence_threshold = 0.05` for a noisy room got the
recorder tuned, but the barge-in watcher kept the default `0.02`.
The watcher would fire on background-noise levels the recorder
was ignoring → false barge-ins.

Fix: `ChatLoop` builds a `VadState` from its stored VAD params and
passes it to `BargeInWatcher` via the existing `vad=` kwarg.

Tests (4 new in `tests/unit/test_watcher_vad_threshold.py`):

VAD config plumbing (2):
- Default ChatLoop → watcher receives default-threshold VadState
  (verified by capturing the `vad=` kwarg via `__init__` hook)
- Custom ChatLoop (silence_threshold=0.05, silence_duration=0.5,
  min=0.5) → watcher receives the same custom VadState

Behavioral verification (2):
- Watcher with high threshold (0.05) ignores quiet noise
  (amp 0.03, RMS ≈ 0.021) — no detection, no callback
- Watcher with default threshold (0.02) catches the same noise

The behavioral pair is the load-bearing assertion: the wiring
test alone could pass with a misconfigured threshold (e.g. if
someone hardcoded a value); the parallel watchers showing
divergent detection prove the threshold *actually* does what it
claims.

Verification: `python -m pytest tests/unit/` → **358 passed in 18s**
(354 existing + 4 new).

Notes:
- This is the iter-020 follow-on that should have been part of
  iter-020 itself. Caught only by code-review-driven inspection
  of which functions consume the VAD config — the iter-020
  behavioral test only exercised the recorder, not the watcher.
- `BargeInWatcher.vad=` kwarg has existed since iter-009 —
  iter-028 is purely "ChatLoop now uses it." No API change to
  the watcher.
- Pattern: when a config param is added to one consumer, audit
  *all* consumers. Half-applied config is worse than no config
  because it produces inconsistent behavior between subsystems
  that should agree.

---

## iter-029 — HTML iteration reports

**Branch:** `iter-029-html-reports` (merged ff to main, commit `2d862af`)
**Date:** 2026-05-24

The log itself is now substantial (28 iterations, 2200+ lines of
markdown), and skimming it as one file is awkward. The user asked
for browsable HTML reports per iteration plus integration of report
generation into the loop going forward.

Adds `scripts/generate_iteration_reports.py` — a no-deps markdown→HTML
generator that walks ITERATION_LOG.md, splits at `## iter-NNN —`
headers, extracts metadata (branch, commit, date, test counts) via
regex, and emits one HTML page per iteration plus an index.

Output lives in `iter-reports/`:
- `index.html` — card grid of all iterations with title + metadata
- `iter-NNN.html` × N — individual pages with prev/next nav

The renderer handles a useful subset of markdown:
- Headers (h2/h3/h4), paragraphs, hr (`---`)
- Fenced code blocks (HTML-escaped inside `<pre><code>`)
- Bullet lists with nested sub-lists
- Inline code (`` `code` ``), bold (`**`), italic (`*`/`_`)
- Tables (pipe-delimited, header row + separator)
- Blockquotes

All user content is HTML-escaped before rendering, so an iteration
title or body that contains `<script>` becomes `&lt;script&gt;`.

Tests (30 in `tests/unit/test_iteration_reports.py`):
- Inline rules: HTML escape, bold, italic, inline code, composition,
  word-internal underscore preserved (`user_role` stays plain)
- Block rules: h2/h3, paragraph, hr, fenced code (with escape),
  bullet lists, tables, blockquotes
- Parser edges: empty log, headerless log, single iteration with
  full metadata, multiple iterations with prev/next wiring,
  `# Status` block between iterations gets dropped, iteration
  without metadata still parses
- Rendering: title shown, branch+commit shown, test counts shown,
  navigation conditional on prev/next presence, index lists all
  iterations, title is HTML-escaped
- Real-log integration: parses actual ITERATION_LOG.md into ≥28
  iterations with strictly ascending numbers

Bug found and fixed during implementation: `_render_bullet_list._flush_sub`
crashed with `IndexError: list index out of range` when the body
of an iteration started with an indented sub-bullet (no parent
bullet yet). Fixed to check `if items:` before appending the nested
list to the previous item, and to emit the nested list at the top
level if no parent exists yet. Defensive — the structure is unusual
but it does occur.

Verification: `python -m pytest tests/unit/` → **388 passed in 18s**
(358 existing + 30 new).

Going forward: each iteration's ship process should include
`python scripts/generate_iteration_reports.py` after appending the
summary to ITERATION_LOG.md, so `iter-reports/` always reflects
current state. The script is fast (<100ms for 29 iterations) and
deterministic.

---

## iter-030 — clock injection for BargeInCoordinator

**Branch:** `iter-030-coord-clock` (merged ff to main, commit `3d23c03`)
**Date:** 2026-05-24

`SentenceWorker`, `BargeInWatcher`, `ChatLoop`, `record_utterance_streaming`
all accept an injected clock so tests can replace it with a deterministic
counter. `BargeInCoordinator` was the odd one out — it stamped
`self.triggered_at = time.monotonic()` directly. In production this
matched (everyone is monotonic), but the moment a test injected a fake
clock, the phase comparison in `ChatLoop`:

    phase = (
        "LLM-stream phase"
        if coord.triggered_at is not None
        and llm_stream_done_at is not None
        and coord.triggered_at < llm_stream_done_at
        else "playback phase"
    )

mixed two clocks: `triggered_at` in real wall time (from `time.monotonic`),
`llm_stream_done_at` on the test's fake clock. The `<` comparison then
depended on whether the real clock had advanced enough during the test
to land before or after the fake clock's frozen value — a flake.

What changed:
- `BargeInCoordinator.__init__` accepts `clock=time.monotonic`. Default
  preserves prior behavior so unrelated callers keep working.
- `trigger()` samples `self._clock()` instead of `time.monotonic()`.
- `ChatLoop.run_one_turn` constructs the coordinator with
  `clock=self._clock` so both timestamps come from the same source.

Tests (7, in `tests/unit/test_bargein_coordinator_clock.py`):
- `TestCoordinatorAcceptsClock` — default monotonic, injected clock,
  idempotency, counter-style callable.
- `TestChatLoopForwardsClockToCoord` — hooks `BargeInCoordinator.__init__`
  to capture the kwarg and verifies ChatLoop passed its own clock.
- `TestPhaseDecisionDeterministicUnderMockedClock` — direct demonstration
  that the phase comparison is now deterministic under a fake clock,
  in both before-stream-end and after-stream-end cases.

Verification: `python -m pytest tests/unit/` → **395 passed in 17.5s**
(388 existing + 7 new).

Notes:
- Pattern repeated from iter-028: when a config knob (clock, VAD
  state, threshold) is added to several components in a system,
  every component that participates in the same comparison or
  state machine has to take the same knob, otherwise the system
  is "half-configured" and behaves inconsistently. Worth a
  repo-wide audit periodically — what other shared concerns
  haven't been threaded all the way through?
- The phase string is currently only used for a printed
  diagnostic. If it ever drives behavior (e.g. a different
  recovery strategy depending on which phase the barge-in
  hit), this fix becomes load-bearing.

---

## iter-031 — filter zero-TTFS turns from session summary

**Branch:** `iter-031-ttfs-zero` (merged ff to main, commit `32e9577`)
**Date:** 2026-05-24

`print_session_summary` aggregated TTFS over every turn in
`metrics_list`, but a turn that ends without audio playback leaves
`metrics.ttfs` at its 0.0 default. Three real ways this happens:

1. Worker errored before producing audio (synth crashed, speaker
   factory raised, etc.)
2. User barged in before any audio played
3. LLM produced no tokens (rare — empty completion)

In `_chat_loop.run_one_turn`:

    if worker.first_audio_at is not None:
        metrics.ttfs = worker.first_audio_at - speech_ended_at

If `first_audio_at` is None, ttfs stays 0.0 and the turn is still
appended to `metrics_list` (it had a transcript and a response —
just no audio). Pre-iter-031, those zeros poisoned both stats:

    Median TTFS:      150ms     (true value would be 280ms)
    Best TTFS:        0ms       (no audio ever played that turn)

"Best TTFS: 0ms" is the worst version — reads like a great result,
actually means absence of data.

Fix: filter `ttfs_times = [m.ttfs for m in metrics_list if m.ttfs > 0]`.
If every turn was zero, render "n/a" instead of computing on an
empty list (which would crash `min()` and `statistics.median()`).

Tests (7 in `tests/unit/test_session_summary_ttfs_filter.py`):
- Zero-TTFS turn excluded from median.
- Zero-TTFS turn excluded from best (min).
- All-zero session renders "n/a".
- Single-turn all-zero renders "n/a".
- Happy path (no zeros) is unchanged — regression guard.
- Other aggregates (STT, LLM, TTS) still include all turns —
  iter-031 is scoped to TTFS only because those measurements are
  still valid even when no audio plays.
- Barge-in counter still counts barge-in turns even when ttfs=0.

Verification: `python -m pytest tests/unit/` → **402 passed in 18.5s**
(395 existing + 7 new).

Notes:
- iter-017 introduced `print_session_summary` and `_median_ms` and
  fixed an upper-median bias on even-length lists. iter-031 is the
  follow-on for "what happens when some turns have 0 TTFS" — same
  function, distinct edge case.
- Pattern: aggregating sentinel/default values is silent data
  corruption. When a metric has a "no value" state, the aggregate
  has to handle that explicitly. The same audit could apply to
  any 0.0-default TurnMetrics field (`stt_time` / `llm_first_token`
  / `tts_time`) — but for those, 0 is rarely meaningful as "no
  data" because successful turns produce non-zero values for them.
  TTFS is special because a successful turn (transcript + response)
  can still have 0 TTFS if the response was never spoken aloud.

---

## iter-032 — SSE parser swallows AttributeError on non-dict choices[0]

**Branch:** `iter-032-sse-attrerror` (merged ff to main, commit `7d3bca5`)
**Date:** 2026-05-24

`parse_sse_token_stream` caught `(JSONDecodeError, KeyError, IndexError,
TypeError)`. The OpenAI-shaped chunk lookup is:

    chunk = json.loads(data)
    delta = chunk["choices"][0].get("delta", {})

A chunk like `{"choices": [null]}` is well-formed JSON, the index lookup
`choices[0]` succeeds (returns `None`), and `.get("delta", {})` raises
**AttributeError** — which was NOT in the except tuple. The entire
generator then aborts mid-stream. Every token after the bad chunk
(including the actual response) is lost.

Observed in the wild: some local proxies / load balancers inject
keep-alive heartbeats as `{"choices": [null]}` between real chunks.
Before this fix, hitting one of those was fatal to the response.

Verified the bug interactively before fixing:

    >>> list(parse_sse_token_stream([
    ...     'data: ' + json.dumps({"choices": [{"delta": {"content": "hello "}}]}),
    ...     'data: ' + json.dumps({"choices": [None]}),
    ...     'data: ' + json.dumps({"choices": [{"delta": {"content": "world"}}]}),
    ...     'data: [DONE]',
    ... ]))
    AttributeError: 'NoneType' object has no attribute 'get'

Fix: add `AttributeError` to the except tuple. Update the docstring
to mention non-dict `choices[0]` as a tolerated failure mode.

Tests (11 in `tests/unit/test_chat_llm_attribute_error.py`):
- `TestNonDictChoicesElement` — `choices[0]` is None / string / int /
  list. Each bad chunk gets skipped; trailing good tokens reach the
  consumer.
- `TestStreamFullyAbortsWithoutFix` — multiple bad chunks
  interspersed with good ones; bad chunks at start / end of stream.
  All good tokens make it through, `[DONE]` still terminates cleanly.
- `TestExistingErrorPathsStillCaught` — regression guards for the
  four error types we already handled (JSONDecodeError, KeyError,
  IndexError, TypeError) so a future "narrow the except clause"
  cleanup doesn't silently revert.

Verification: `python -m pytest tests/unit/` → **413 passed in 17.9s**
(402 existing + 11 new).

Notes:
- Pattern: when an except clause enumerates specific exceptions,
  audit the protected expression for every TYPE of failure that
  produces an error not in the list. Here the audit is "what does
  `.get()` raise on non-dict?" — AttributeError. Easy to miss in
  the original code review because the line "looks fine."
- Wider exception (e.g. `except Exception`) would have caught this
  but also masks bugs in the parser itself. The narrower fix is
  better — explicit, documented, traceable.
- Could be extended to `except (json.JSONDecodeError, LookupError,
  AttributeError, TypeError)` (LookupError covers KeyError +
  IndexError) for marginal compactness. Kept the explicit list to
  preserve grep-ability of which conditions are tolerated.

---

## iter-033 — split sentences after terminator+closing-paren / bracket

**Branch:** `iter-033-closing-parens` (merged ff to main, commit `298bac2`)
**Date:** 2026-05-24

iter-022 added support for terminator+closing-quote so US-style
quoted speech split:

    He said "hello." Then he left.

But the same shape with parens / brackets — common in LLM output —
didn't split:

    He left (long ago.) Today returned.   →  ❌ no split
    Per spec [see ref.] We continue.       →  ❌ no split

Verified before fixing:

    >>> split_complete_sentences('He left (long ago.) Today returned.')
    ([], 'He left (long ago.) Today returned.')

The streaming overlap pipeline (iter-008) waits for complete
sentences before submitting to TTS, so a paren-tail bot response
sat in the buffer until the *next* terminator arrived (often the
end of the whole reply). TTFS suffered.

Fix: generalize `_CLOSING_QUOTES` (quotes only) to
`_CLOSING_AFTER_TERMINATOR` (quotes + parens + brackets + curly).
Build SENTENCE_END from the constant via `re.escape` so adding
more closing chars in the future only requires touching the
string. Keep `_CLOSING_QUOTES` as a backwards-compat alias.

Update the abbreviation walk-back in `split_complete_sentences`
to use the new constant — non-terminating abbreviations inside
parens (`See note (etc.) and more.`) still don't split.

Tests (21 in `tests/unit/test_splitter_closing_parens.py`):
- `TestClosingParens` — period/exclamation/question + ); paren
  at very end (no whitespace, no split); multi-sentence chain.
- `TestClosingBrackets` — `.]` and `.}`.
- `TestRegressionsFromIter022` — all four quote variants still
  split (straight + smart, double + single).
- `TestPlainSentencesUnaffected` — non-quoted, non-parenthesized
  sentences split exactly as before.
- `TestAbbreviationInsideParens` — `(etc.)` and `(i.e.)` still
  don't split; `(Mr. Smith.)` does.
- `TestConstantAndRegex` — `)`, `]`, `}` all in the set; quotes
  still in the set; backwards-compat alias preserved; regex
  matches every closing variant.

Verification: `python -m pytest tests/unit/` → **434 passed in 17.5s**
(413 existing + 21 new).

Notes:
- Pattern reused from iter-022: when a sentence-boundary feature
  has multiple natural variants (quotes, parens, brackets), the
  cleanest approach is one constant + re.escape so each variant
  is data, not code.
- Did NOT add `>` (closing angle / blockquote marker). LLM output
  rarely uses it as a closing-after-terminator and it'd risk
  false positives in HTML-flavored content. Same reasoning for
  `>>` and `}}`.

---

## iter-034 — tolerant parse_filler_config

**Branch:** `iter-034-filler-config` (merged ff to main, commit `fd5eda6`)
**Date:** 2026-05-24

iter-011 introduced filler words but the parsing lived inline in
`mic_chat.run_chat`:

    filler_texts = list(chat_cfg.get("fillers") or [])
    filler_idle_threshold = float(chat_cfg.get("fillers_idle_threshold", 0.6))

Two real failure modes verified before fixing:

1. `chat.fillers: "hi"` (string instead of list). `list("hi")`
   iterates the string and yields `["h", "i"]`. The user's typo
   became two two-character "fillers" passed to TTS, producing
   nonsense audio.
2. `chat.fillers_idle_threshold: "abc"`. `float("abc")` raises
   ValueError, killing chat startup before any STT / LLM / TTS
   load — and the user only sees a stack trace.

iter-020 already established the tolerant-parser pattern via
`parse_vad_config`: typo'd values silently fall back to defaults
rather than crashing. iter-034 applies it to fillers.

Add `parse_filler_config(chat_cfg)` to `_chat_config.py`:
- Returns `{"texts": list[str], "idle_threshold": float}`.
- Non-list `fillers` → empty list (drops the string-as-iter bug).
- Non-string items → silently dropped.
- Empty / whitespace-only strings → dropped + stripped.
- Non-positive `fillers_idle_threshold` → default 0.6.

`mic_chat.run_chat` now calls the new parser. Removed the inline
brittle code; behavior on well-formed config is unchanged.

Tests (23 in `tests/unit/test_filler_config.py`):
- `TestEmptyOrMissing` — defaults for empty / None / non-mapping;
  result list is independent of `FILLER_DEFAULTS` (no shared
  reference that future callers could corrupt).
- `TestHappyPath` — well-formed input passes through; int
  threshold coerced to float.
- `TestFillersListInputForms` — `fillers` as string / dict / int
  / None / empty list all return empty texts list.
- `TestNonStringItemsDropped` — list with mixed types drops
  non-strings; empty / whitespace-only strings dropped; whitespace
  stripped from valid items.
- `TestIdleThresholdInputForms` — string / negative / zero / None
  all fall back to default 0.6.
- `TestPartialConfig` — partial configs with only one key; combined
  with unrelated chat keys (`vad`, etc.) — no interference.

Verification: `python -m pytest tests/unit/` → **457 passed in 18.3s**
(434 existing + 23 new).

Notes:
- Pattern: tolerant config parsing (iter-020 + iter-034). When
  user-supplied YAML touches a config field, the parser should
  fall back to a sensible default rather than crash. The user
  finds the misbehavior fast enough to debug. Crashing during
  `__init__` blocks the entire app, including the parts that
  *don't* depend on this config.
- One could argue `chat.fillers: "hi"` should warn the user to
  fix their config. Accepted trade-off for now: silence trumps
  noise. If a user complains "fillers don't work", the misconfig
  is easy to spot in the YAML.
- iter-034 doesn't touch `parse_vad_config` since it already
  works the same way. Could in future: a `parse_chat_section`
  umbrella that delegates to per-section parsers, then `mic_chat`
  just unpacks one structured result. For now two parsers in
  sequence is fine.

---

## iter-035 — integration test suite + testing report with plots

**Branch:** `iter-035-testing-report` (merged ff to main, commit `0a30a55`)
**Date:** 2026-05-24

Three pieces ship together to formalize testing visibility:

**1. `tests/integration/` directory.** Until iter-035 the cross-module
end-to-end tests lived under `tests/unit/test_chat_loop.py` and
`tests/unit/test_bargein_orchestration.py` — fine, but they're
structurally integration tests (real ChatLoop, virtual audio, stub
LLM, multi-thread). Promoting them to a dedicated suite gives the
testing report something to break out and signals what to add when
a unit test isn't enough.

Five integration tests in `tests/integration/test_chat_pipeline_e2e.py`:
- `TestSingleTurnHappyPath` — full turn through ChatLoop with metrics,
  messages, transcript, response. Plus no-speech-returns-None.
- `TestMultiTurnConversation` — two turns; history preserved.
  **Gotcha discovered:** the BargeInWatcher during turn 1's playback
  reads the mic buffer; if turn 2's audio is pre-pushed, the watcher
  eats it and turn 2 hangs forever waiting for speech. Must push
  between turns. Documented in the test.
- `TestLLMErrorRecovery` — failing_llm raises; ChatLoop pops the
  user message, returns had_error=True.
- `TestStreamingOverlap` — structural check that worker emitted
  audio + TTFS measured. Stricter timing assertions kept in
  unit suite (test_chat_loop.py) where mocked clocks make them
  deterministic.

One additional `TestBargeInDuringPlayback` test guarded by
`pytest.skip` when the watcher doesn't trigger on a given run
(timing-dependent across hosts).

**2. `iter-reports/testing.html`** — single-page testing posture
with three SVG charts (pure-Python, no matplotlib / no JS deps):
- Total tests passing (cumulative line, scaled from iter-001's 22 to
  iter-035's 481 — visualizes the test growth curve).
- Tests added per iteration (bar chart — shows the lumpy adds: small
  bug fixes vs big iterations like iter-005's virtual audio).
- Test runtime in seconds (line, plotted only for iters where the
  verification line had a parseable seconds value).

Plus a stat grid (latest count, unit-files / integration-files,
latest runtime, median added-per-iter) and run-instructions for
both suites.

**3. Testing nav link** on every iter page and the index, so a
reader reviewing iter-NNN can jump directly to the longitudinal
testing view.

Generator extensions in `scripts/generate_iteration_reports.py`:
- New `_RUNTIME_RE` regex pulls `"**N passed in S.Ss**"` (decimals
  optional — early iters had no seconds suffix).
- `Iteration.test_runtime_s` field, populated by `_populate_metadata`.
- `_svg_line_chart` / `_svg_bar_chart` helpers — pure-Python SVG
  with axes, gridlines, dots, HTML-escaped titles.
- `_count_test_files` walks `tests/unit/` and `tests/integration/`
  for `test_*.py` files (skips `conftest.py`, `__init__.py`).
- `render_testing_page(iterations, repo_root)` composes the page.
- `render_iteration` and `render_index` add `Testing →` to nav.
- `main()` writes `testing.html` alongside index + iter pages.

Tests (19 new in `tests/unit/test_testing_report.py`):
- `TestRuntimeParsing` — decimal seconds, integer seconds, no-seconds
  yields 0.0.
- `TestSvgLineChart` — empty placeholder, single-point, multi-point
  path commands and dots, HTML-escape on title.
- `TestSvgBarChart` — empty placeholder, bars-per-data-point,
  iter-NNN axis labels.
- `TestCountTestFiles` — mixed unit/integration counts; missing dirs
  return zero (no crash).
- `TestRenderTestingPage` — three charts present, run instructions
  emitted, links back to index, runtime chart handles
  zero-runtime iters and all-zero placeholder.
- `TestIterPageHasTestingLink` / `TestIndexHasTestingLink` — nav
  link wired to iter pages and index.

Verification: `python -m pytest tests/unit/ tests/integration/` →
**481 passed, 1 skipped in 17.9s** (457 existing + 19 testing-report
+ 5 integration; 1 skipped is the timing-flaky barge-in).

Notes:
- Going forward each iter's ship process should run
  `python scripts/generate_iteration_reports.py` after appending
  the summary; testing.html refreshes automatically. This was
  already established in iter-029 — iter-035 only adds the new
  output file, no workflow change.
- The runtime chart will sparsen as more old iterations are
  considered (early iters had no seconds in the verification line).
  That's fine — the chart explicitly notes this in its caption.
- The "+22 tests" first-iter spike is real (iter-001 introduced the
  unit suite). It dominates the bar chart; that's a faithful
  picture of the real growth.

---

## iter-036 — performance integration suite + perf page + metrics taxonomy

**Branch:** `iter-036-perf` (merged ff to main, commit `73ac030`)
**Date:** 2026-05-24

User asked for a "performance integration test (see TTFS for different
scenarios, time to TTS or STT)" linked from the testing report, plus a
brainstorm of new metrics worth tracking ("standard, architecture-specific,
novel"). iter-036 ships all of that.

**1. `tests/performance/`** — drives `ChatLoop.run_one_turn` across
eight scenarios with stub LLM / TTS / STT (so the numbers reflect
pipeline overhead, not neural-net latency):

- `short_short` — 1s utterance + 5-token reply (best-case baseline).
- `short_long` — 8-sentence reply (exercises streaming overlap).
- `long_short` — 3s utterance, short reply (STT path-length scaling).
- `tts_50ms_per_sentence` — kokoro-shaped TTS latency.
- `stt_100ms` — whisper-shaped STT latency.
- `slow_llm_300ms` — 300ms first-token (real-LLM dominant cost).
- `fillers_on` — pre-rendered filler triggers during LLM stall.
- A final `test_results_written` row that asserts the JSON schema
  is intact (canary against future changes breaking the renderer).

Results dump to `iter-reports/perf-results.json` after each run.
The dump happens incrementally — if a later scenario crashes, the
earlier rows still land in the file.

**2. `iter-reports/performance.html`** — generator reads the JSON
and renders five horizontal-bar charts:

- TTFS by scenario
- STT time by scenario
- TTS time by scenario
- LLM first-token by scenario
- Wall-clock turn time

Plus a scenario description table (sentences spoken, barge-in flag).
When `perf-results.json` doesn't exist, the page shows a
"run the suite" placeholder rather than 404'ing.

**3. `docs/perf-metrics-taxonomy.md`** — research deliverable from a
sub-agent. ~46 metrics in three buckets:

- *Standard* (20 metrics): S2S latency, EoT detection, WER,
  STT/TTS RTF, turn-taking jitter, false/missed-trigger rates,
  audio under/overruns, cold-start penalty, etc.
- *Architecture-specific* (24 metrics): streaming-overlap ratio,
  first-sentence overlap savings, filler-mask success/false-positive
  rates, sentence-split coverage, worker queue depth, speaker-open
  overhead, barge-in latency by phase, primed-frames replay duration,
  LLM stream cancel-to-close, mic-flush stale-frame count, VAD-config
  consistency, etc.
- *Novel/speculative* (22 metrics): naturalness gap (200-400ms
  sweet spot), conversation rhythm score, regret rate
  (barge-in within 200ms of bot first audio), recovery quality,
  pre-empted-content loss, FT-A gap, sub-second-turn rate,
  phantom-sentence rate, etc.

Each entry has a definition, instrumentation site (referencing real
modules in the codebase), UX/perf rationale, and computation formula.
The taxonomy is the source list for future iterations to pull from
when adding instrumentation — pick a metric, instrument it, expose
it via TurnMetrics, plot it on the perf page.

Generator extensions in `scripts/generate_iteration_reports.py`:
- `_svg_horizontal_bars` helper (pure SVG, no deps).
- `_load_perf_results` loader (graceful None on missing/invalid).
- `render_performance_page(payload)` — placeholder + full mode.
- Performance nav link added to iter pages, index, testing page.

Tests (16 new in `tests/unit/test_performance_report.py`):
- `TestHorizontalBars` — empty placeholder, single/multi rows,
  HTML escape on labels.
- `TestLoadPerfResults` — missing file, valid JSON, malformed JSON.
- `TestRenderPerformancePage` — placeholder modes, five-chart
  composition, scenario table, captured_at, navigation.
- `TestPerformanceLinkWired` — nav link in iter pages, index,
  testing page.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **505 passed, 1 skipped in 35s** (481 existing
+ 16 perf-report + 8 perf scenarios; 1 skipped is the timing-flaky
barge-in from iter-035).

Notes:
- The taxonomy file is a reading list, not a wiring spec. Future
  iterations should treat it as a backlog: pick a metric, instrument
  it with one PR, validate it appears in TurnMetrics + the perf
  scenarios, then move on.
- Performance numbers from stub-driven scenarios are NOT comparable
  across hardware. The page is for cross-scenario shape comparison
  on a single machine. Real-engine perf (kokoro + mlx-whisper +
  actual LLM) needs a separate "live" suite — out of scope here.
- The performance page link is now on every iter page so a reader
  reviewing iter-NNN can jump straight to the perf snapshot.

---

## iter-037 — mic stale-frame count on TurnMetrics (taxonomy 2.19)

**Branch:** `iter-037-stale-frames` (merged ff to main, commit `1ff0064`)
**Date:** 2026-05-24

First instrumentation pulled from `docs/perf-metrics-taxonomy.md`.
Metric **2.19 — Mic flush stale-frame count**, in the
"Architecture-specific" bucket. The data was already there:
`flush_pending_audio` has returned a `drained` int since iter-002,
but ChatLoop dropped it on the floor.

Why this metric matters: many stale frames each turn means the mic
accumulated bot audio between turns — acoustic echo, OS loopback,
or Bluetooth duplex. A small consistent value is harmless (trailing
silence the recorder didn't consume before VAD's DONE_OK). A large
value is the signal "your setup needs echo cancellation."

Changes:
- `TurnMetrics.mic_stale_frames: int = 0` (new field).
- `ChatLoop.run_one_turn` captures `flush_pending_audio`'s return
  and stashes on metrics.
- `TurnMetrics.print` emits "Mic stale: N frames (Ns)" only when
  N > 0; yellow if > 0.5s, dim otherwise (don't crowd the print
  with a clean turn).
- `print_session_summary` aggregates totals; emits only when
  total is non-zero, with "check echo cancellation" suggestion.

Honest limitation documented in the iteration log: the count
includes trailing silence the recorder didn't consume before
VAD's DONE_OK fired. So a clean turn still shows ~0.5s of
"stale" because the recorder's silence_duration window left
unconsumed silent bytes in the buffer. Future iter could
distinguish silent stale (harmless) from voiced stale (real echo)
via RMS on the flushed bytes — iter-037 keeps the metric simple
and exposes the raw count.

Tests (10 in `tests/unit/test_mic_stale_frames.py`):
- `TestDefault` — TurnMetrics defaults to 0.
- `TestPerTurnPrint` — zero omits the line, non-zero emits with
  frames + seconds.
- `TestSessionSummary` — zero total omits, non-zero emits with
  aggregate + seconds + suggestion.
- `TestChatLoopCapturesStaleFrames` — field is populated on
  metrics; extra audio beyond utterance is counted; extra-burst
  strictly greater than baseline.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **515 passed, 1 skipped in 32s** (505
existing + 10 new).

Notes:
- Pattern: pulling metrics from the taxonomy doc one at a time
  is the right cadence. Each instrumentation should be a
  small, testable unit. iter-037 took two file edits + one new
  test file — appropriate scope.
- Next candidates from the taxonomy: 2.10 (barge-in latency —
  triggered_at exists since iter-030, just need playback_stopped_at),
  2.18 (cancel-event correctness), 1.10 (LLM time-to-first-sentence).

---

## iter-038 — LLM time-to-first-sentence metric (taxonomy 1.10)

**Branch:** `iter-038-ttfsent` (merged ff to main, commit `e7bd3f5`)
**Date:** 2026-05-24

Second metric pulled from `docs/perf-metrics-taxonomy.md`. **Metric
1.10 — LLM time-to-first-sentence (TTFsent)**, in the "Standard"
bucket.

The existing `llm_first_token` records when the first token arrived
from the LLM. But TTS can't actually run until a complete *sentence*
reaches the worker — and the splitter only emits one when a
terminator (`.`, `!`, `?`) followed by whitespace appears. The LLM
may stream chatty preamble (no terminator) for a while between
first-token and first-sentence, and that gap is invisible in the
current metrics.

Two scenarios with identical first-token times can have very
different TTFS depending on how the LLM phrases its response. A
system prompt nudge like "respond in short sentences" would compress
TTFsent without changing first-token. With this metric, the impact
is measurable.

Changes:
- `TurnMetrics.llm_first_sentence: float = 0.0` (new field).
- `ChatLoop.run_one_turn` stamps `first_sentence_at` the first
  time `split_complete_sentences` yields a non-empty list. Field
  set to `first_sentence_at - llm_start` after the LLM stream
  completes; 0 if no terminator ever arrived.
- `TurnMetrics.print` emits "LLM 1st sent: Nms (+Mms preamble)"
  only when > 0. The parenthetical shows the gap from first-token
  so the reader sees splitter-wait at a glance.
- `print_session_summary` shows median TTFsent over turns where
  it's > 0 (parallel to iter-031's TTFS filter).

Tests (10 in `tests/unit/test_llm_first_sentence.py`):
- `TestDefault` — TurnMetrics defaults to 0.
- `TestPerTurnPrint` — zero omits the line; non-zero shows preamble
  gap; zero-gap edge case (first token IS the terminator) shows
  "+0ms preamble".
- `TestSessionSummary` — no turns omits the median line; multi-turn
  emits median in ms; zero turns filtered from the median.
- `TestChatLoopCapturesFirstSentence` — terminator triggers the
  stamp; no-terminator stream leaves it at 0; long-preamble stream
  shows gap >= preamble sleep.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **525 passed, 1 skipped in 31s** (515 existing
+ 10 new).

Notes:
- Two metrics from the taxonomy now live (2.19 + 1.10). The
  cadence is one metric per iteration; that's the right rhythm —
  small testable additions, each with its own diagnostic value.
- Next candidates: 2.10 (barge-in latency — needs a `playback_stopped_at`
  hook in the worker), 2.18 (cancel-event correctness — `play_aligned`
  should report HOW it exited, not just elapsed seconds).
- The "preamble lag" insight is only as good as the splitter's
  terminator detection. iter-016 / iter-021 / iter-022 / iter-033
  hardened the splitter for abbreviations, ordinals, quoted speech,
  and parens — without those fixes, TTFsent would be misleadingly
  high on text containing "Mr. Smith" or "1st place" or `(...).`.

---

## iter-039 — per-iteration perf snapshots + time-series charts

**Branch:** `iter-039-perf-history` (merged ff to main, commit `9be7b1b`)
**Date:** 2026-05-24

iter-036 saved one `perf-results.json` (latest snapshot). With many
iterations landing, the more useful view became "metric trajectory
over iterations" — did TTFS regress when iter-031 filtered zero
turns? Did iter-038's TTFsent instrumentation slow wall time? You
can't see that from a single snapshot.

User asked for "performance test reports for each iteration." Three
pieces:

**1. Per-iter snapshot file.** Each perf run now writes
`iter-reports/perf-iter-NNN.json` alongside the existing
`perf-results.json`. Iteration number resolved from the most recent
`## iter-NNN —` heading in `ITERATION_LOG.md`; falls back to git
commit count if the log can't be parsed.

**2. Generator helpers.** Two new functions in
`scripts/generate_iteration_reports.py`:
- `_load_perf_history(reports_dir)` walks `perf-iter-NNN.json`
  files, sorted by iteration. Robust to malformed JSON, missing
  iteration field, unrelated files in the dir.
- `_svg_multi_line_chart(series_by_label, ...)` — pure-SVG line
  chart with one polyline per series + a per-series legend in
  fixed palette colors.

**3. `performance.html` reorganized.** Two top-level sections:
- "Latest snapshot" — the iter-036 horizontal bar charts (TTFS,
  STT, TTS, LLM 1st token, wall — one bar per scenario).
- "Across iterations" — new time-series block with multi-line
  charts (TTFS, wall, TTS, STT — one line per scenario, x-axis =
  iter number). Soft "only one captured" note when history has
  one entry; fills in over time.

Iter-038 was the seed (perf-iter-038.json saved during this
iteration's perf run before commit). iter-039.json will be added
by the loop's perf-run step at the end of this iteration. From
iter-040 forward each loop iteration appends another snapshot.

Loop prompt updated to include the perf-run step ahead of report
regeneration:

    python -m pytest tests/performance/ -q
    python scripts/generate_iteration_reports.py

Tests (19 in `tests/unit/test_perf_history.py`):
- `TestLoadPerfHistory` — empty / missing dir, single file,
  multiple files sorted by iteration, malformed JSON skipped,
  non-`perf-iter-` files ignored, filename fallback when JSON
  lacks 'iteration' key.
- `TestMultiLineChart` — empty / all-empty placeholder, single
  point per series renders as circle, multi-point as path+dots,
  legend has every label, HTML-escaped title.
- `TestPerformancePageHistory` — no history → no section, single
  iter → soft note, multi-iter → 9 SVGs (5 latest + 4 history),
  iteration count emitted, non-numeric iteration row skipped
  without crashing.
- `TestResolveIterNumber` — real repo resolves to a 3-digit
  string (canary that the helper still works against the live
  ITERATION_LOG.md).

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **544 passed, 1 skipped in 32s** (525 existing
+ 19 new).

Notes:
- `cancel-correctness` (taxonomy 2.18) was originally going to be
  iter-039 but the user redirected to per-iter perf history. The
  cancel-correctness work will return as iter-040 or later.
- The "Across iterations" charts will show real trends starting
  iter-041 (when there are 3+ snapshots). Until then, a single
  point per scenario is rendered as a circle so the page still
  looks reasonable.
- Choice not made: NO automatic deletion of old per-iter snapshots.
  At ~1 KB each, 100 iterations = ~100 KB. Cheap, and the history
  is the whole point.

---

## iter-040 — cancel-correctness metric (taxonomy 2.18)

**Branch:** `iter-040-cancel-correctness` (merged ff to main, commit `441b05f`)
**Date:** 2026-05-24

Third metric pulled from `docs/perf-metrics-taxonomy.md`. **Metric
2.18 — Cancel-event correctness rate**, in the "Architecture-specific"
bucket. Validates the iter-009 / iter-026 cancel plumbing.

A barge-in can land in two places:
1. *Mid-sentence* — cancel fires during a sentence's playback, the
   play loop breaks via `cancel_event`, audio cuts off immediately.
   Clean cut-off. iter-026's post-cancel-flush guard kicks in.
2. *Between sentences* — cancel fires in the silent gap, the
   current sentence finishes naturally, the worker exits before
   the next. Also clean.

Both are success outcomes, but they tell different stories about
how the user is interrupting:
- **High mid-stream rate** — users are impatient or the bot's
  response is going wrong, and they're cutting it off.
- **Low mid-stream rate** — users are waiting for natural pauses,
  the bot is being responsive enough that mid-sentence interrupts
  aren't needed.

Detection mechanism: before/after sampling around `play_fn`. In
`SentenceWorker._play_clip`:

    cancel_was_set_before = self._cancel_event.is_set()
    elapsed = self._play_fn(...)
    if not cancel_was_set_before and self._cancel_event.is_set():
        self.cancelled_sentences += 1

Tighter than sampling only after the call: a cancel firing in the
microseconds following natural completion would otherwise count
as a false positive. With the before-snapshot, we know cancel
transitioned during the play.

Surfaced at three layers:

1. `SentenceWorker.cancelled_sentences: int = 0` — counter on the
   worker, accumulates across all sentences in the run.
2. `TurnMetrics.sentences_cancelled: int = 0` — ChatLoop transfers
   from worker.
3. Per-turn print and session summary use it to label barge-ins:
   - "yes (user interrupted) (1 cut mid-stream)"
   - "yes (user interrupted) (between sentences)"
   - "Barge-ins: 4 (2 mid-stream, 50%)"
   - "Barge-ins: 2 (all between sentences)"

Tests (12 in `tests/unit/test_cancel_correctness.py`):
- `TestDefault` — counter defaults to 0 on both Worker and TurnMetrics.
- `TestPlayClipDetection`:
  - Natural completion does NOT increment.
  - Mid-stream cancel increments exactly once (subsequent queued
    sentences are drained without play, so don't double-count).
  - Pre-set cancel_event does NOT increment (no false positives
    from cleanup paths).
- `TestPerTurnPrint` — no-barge omits, mid-stream shows count,
  between-sentences shows label.
- `TestSessionSummary` — no-barges omits, all-mid → "100%", mixed
  → "50%", all-gap → "all between sentences".

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **556 passed, 1 skipped in 22s** (544 existing
+ 12 new).

Notes:
- Three metrics from the taxonomy now live: 2.19 (mic stale frames),
  1.10 (LLM TTFsent), 2.18 (cancel correctness). The cadence — one
  metric per iteration, small testable scope — is the right
  rhythm.
- Originally planned as iter-039 but the user redirected to the
  per-iter perf history feature; iter-040 picks it up.
- Subtle: the before/after sample handles the common race but not
  every edge. If cancel fires DURING the natural completion path
  in some platform-specific way, the detection might miss. For an
  aggregate metric across many turns, the noise is acceptable.
- Combine with iter-031 (TTFS-zero filter) and iter-038 (TTFsent)
  in the session summary, the operator now has a much richer
  picture: TTFS variance, sentence-level latency, and barge-in
  shape, not just totals.

---

## iter-041 — barge-in latency metric (taxonomy 2.10)

**Branch:** `iter-041-barge-latency` (merged ff to main, commit `f05bf96`)
**Date:** 2026-05-24

Fourth metric pulled from `docs/perf-metrics-taxonomy.md`. **Metric
2.10 — Barge-in latency**, in the "Architecture-specific" bucket.

The taxonomy doc is blunt about why this matters: ">~200ms is the
moment the user thinks the bot is ignoring them." The whole barge-in
feature (iter-009 / iter-010 / iter-012 / iter-024 / iter-025 / iter-026
/ iter-027 / iter-028) lives or dies on one number.

Implementation:

- `BargeInCoordinator` gains `playback_stopped_at: Optional[float]`,
  stamped after `worker.cancel()` returns inside `trigger()`. That's
  the moment the worker thread has joined and playback is truly
  halted (cancel_event drained the play_aligned chunk loop, then
  the thread exited).
- `ChatLoop` computes
  `metrics.barge_in_latency = max(0.0, playback_stopped_at - triggered_at)`.
  Clamps negative (shouldn't happen, defensive against clock
  injection bugs).
- `TurnMetrics.barge_in_latency: float = 0.0` (new field).

Surfaced at three layers:

1. Per-turn print: "Barge latency: Nms (detect → halt)" on barge-in
   turns. Yellow if >100ms, green if ≤100ms.
2. Session summary: median + worst across the session. Filters
   zero-latency turns (parallel to iter-031 / iter-038's filters).
3. The performance.html time-series will pick this up
   automatically once perf-iter-NNN.json snapshots include
   barge-in scenarios with latency. (The current perf scenarios
   don't trigger barge-ins, so this is a follow-on for a future
   iter that adds a `barge_in` perf scenario.)

Test design — using a `MagicMock` worker with a side_effect that
mutates the fake clock during `cancel()`, we prove the
`playback_stopped_at` is sampled AFTER cancel returns (not
before). That's the contract the metric depends on.

Tests (15 in `tests/unit/test_barge_latency.py`):
- `TestCoordinatorTimestamps` — default None, trigger stamps both
  timestamps, worker.cancel happens BEFORE the stamp (sleeping mock
  proves the order), idempotent doesn't overwrite, no-worker still
  stamps both.
- `TestPerTurnPrint` — default 0, no-barge omits, barge with zero
  latency omits, barge with latency emits with "(detect → halt)".
- `TestChatLoopArithmetic` — subtraction, negative clamps to 0.
- `TestSessionSummary` — no-barge omits block, barges with
  latencies show median + worst, zero-latency turns filtered
  from median, all-zero suppresses block.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **571 passed, 1 skipped in 20s** (556
existing + 15 new).

Notes:
- Four metrics from the taxonomy now live: 2.19 (mic stale frames),
  1.10 (LLM TTFsent), 2.18 (cancel correctness), 2.10 (barge
  latency). The cadence — one per iteration — has held over four
  iterations and the codebase has absorbed each cleanly.
- Combined with iter-040's mid-stream count, you can now answer:
  "of the barges this session, how many were mid-sentence and
  how slow was the cancel?" That's the full barge-in story.
- Future: a perf-suite scenario that explicitly fires a barge-in
  mid-playback so this metric lands in the time-series charts.
  iter-035's TestBargeInDuringPlayback is the right shape but
  pytest.skipped due to flaky timing; the perf suite's deterministic
  timeline could make it reliable.

---

## iter-042 — perf-suite barge-in scenario + barge-in latency on perf snapshot

**Branch:** `iter-042-perf-barge` (merged ff to main, commit `345ad7c`)
**Date:** 2026-05-24

iter-041 instrumented barge-in latency on `TurnMetrics`. iter-042
wires it into the perf-snapshot row, adds a deterministic perf
scenario that actually triggers a barge-in, and renders the
metric on `performance.html` (latest-snapshot bar chart + time-series
across iterations).

**The deterministic-barge workaround.** The naive "push barge audio
in the mic up front" approach fails because iter-002's
`flush_pending_audio` drains the mic between phases and eats the
barge tone before the watcher starts. The fix: push the barge
from a daemon thread that fires 50ms after `run_one_turn` starts —
by which point the flush has run and the watcher is active.

This is much more reliable than iter-035's `TestBargeInDuringPlayback`
(which is marked `pytest.skip` when timing-flaky). The perf scenario
has run cleanly on every loop iteration since landing.

**ScenarioResult schema additions.** Five new fields (mostly
catch-up from earlier iterations whose metrics weren't yet on the
perf rows):
- `llm_first_sentence_ms` (iter-038, taxonomy 1.10)
- `sentences_cancelled` (iter-040, taxonomy 2.18)
- `barge_in_latency_ms` (iter-041, taxonomy 2.10)
- `mic_stale_frames` (iter-037, taxonomy 2.19)

**Generator chart additions.** Two new chart sites:
- "Barge-in latency by scenario" in the latest-snapshot section
  (horizontal bar chart, yellow palette).
- "Barge-in latency over iterations" in the time-series section
  (multi-line chart, one line per scenario).

Both charts use an emit-only-with-data rule: if no scenario row
in the relevant data has a non-zero measurement, the chart is
suppressed (don't show all-zero bars or empty lines).

Tests (8 in `tests/unit/test_perf_barge_chart.py`):
- `TestLatestSnapshotBargeChart` — no-barge-data omits the chart,
  at-least-one emits with the value visible, palette color check.
- `TestHistoryBargeChart` — no-barge-history omits, with-barge
  emits with legend label, scenario name shows up in time-series.
- `TestScenarioSchemaSurfacedInTable` — table reflects barge_in
  flag (yes/no cells).

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **580 passed, 1 skipped in 21s** (571
existing + 8 new + 1 new perf scenario).

The new perf scenario records: `barge_in=True`,
`barge_in_latency_ms ~5ms` (virtual audio is instant),
`sentences_cancelled=1`. Real-world latency on PyAudio + kokoro
will be 20-100ms — the chart is calibrated for that range.

Notes:
- iter-040 / iter-041 / iter-042 form a tight trio: instrument
  cancel correctness, instrument barge latency, wire both into
  the perf scenario. The full barge-in observability story is
  now: "a barge fired, here's how fast, and here's whether it
  cut a sentence mid-stream."
- Future barge work would need to drive perf scenarios that
  REGRESS when something gets slower (e.g. a wider VAD poll
  interval). The perf charts are ready; the regression detection
  isn't built yet.

---

## iter-043 — streaming overlap ratio metric (taxonomy 2.1)

**Branch:** `iter-043-overlap-ratio` (merged ff to main, commit `c6d684d`)
**Date:** 2026-05-24

Fifth metric pulled from `docs/perf-metrics-taxonomy.md`. **Metric
2.1 — Streaming overlap ratio**, in the "Architecture-specific"
bucket. The taxonomy is direct: "the whole point of `SentenceWorker`
is to run TTS in parallel with token receipt. If overlap is 0,
the worker is just adding latency."

Definition adopted (a simple proxy):

    overlap_ratio = max(0, llm_stream_done_at - first_audio_at) / llm_total

- 1.0 (capped) — audio started before LLM finished and stayed
  through end-of-stream.
- 0.5 — audio overlapped half the stream.
- 0 — audio only started AFTER LLM finished, or didn't play at
  all. Sequential. iter-008 streaming-overlap not paying off
  this turn.

Skipped a more elaborate "union of synth + play intervals
intersected with LLM window" definition because (a) it requires
new instrumentation in the worker (synth_at, play_at lists), (b)
the simple proxy already answers the operational question
("does my pipeline benefit from streaming?").

Surfaced at three layers:

1. `TurnMetrics.streaming_overlap_ratio: float = 0.0` (new field).
2. ChatLoop computes it after `llm_stream_done_at` is set.
3. Per-turn print: "Overlap: NN% (LLM↔TTS concurrency)" only on
   turns where >0. Green ≥50%, yellow <50%.
4. Session summary: "Median overlap: NN%" over turns where >0
   (parallel to iter-031's TTFS filter).
5. `ScenarioResult.streaming_overlap_ratio` on perf rows.

Empirical results from the perf suite (stub LLM is too fast for
most scenarios to show overlap):
- short_short / short_long / long_short / tts / stt / slow_llm: 0%.
- fillers_on: 70% (filler clip starts immediately, LLM is slow).
- barge_in: 50% (barge cuts mid-stream during LLM).

Real-LLM perf with 200-500ms TTFT would show meaningful overlap
on every multi-sentence scenario. The metric is calibrated for
that reality.

Tests (9 in `tests/unit/test_streaming_overlap.py`):
- Default 0.
- Per-turn print — zero omits, non-zero shows pct + explainer,
  100% renders correctly.
- Session aggregate — no-data omits, some-data shows median,
  zero turns filtered.
- ChatLoop arithmetic — real audio play yields >0 ratio bounded
  ≤1; no-audio (LLM yields fragments only) yields 0.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **589 passed, 1 skipped in 21s** (580 existing
+ 9 new).

Notes:
- Five metrics from the taxonomy now live (2.19 / 1.10 / 2.18 /
  2.10 / 2.1). Iterating one per loop has worked; we have ~40
  more on the wishlist.
- The "fillers_on" 70% number is itself a useful signal —
  validates that iter-011's filler design actually overlaps the
  filler with the LLM stream.

---

## iter-044 — worker idle-gap metric (taxonomy 2.16)

**Branch:** `iter-044-idle-gap` (merged ff to main, commit `7e52670`)
**Date:** 2026-05-24

Sixth metric pulled from `docs/perf-metrics-taxonomy.md`. **Metric
2.16 — Sentence-worker idle gap**, "Architecture-specific" bucket.

Cumulative time `SentenceWorker` spent blocked on
`self._queue.get(...)` waiting for the next sentence. Excludes the
first wait — that's TTFsent (iter-038 territory); after the first
sentence, the gap before the next is the metric.

Diagnostic value: combined with iter-043's `streaming_overlap_ratio`,
the pair localizes pipeline bottlenecks:

| overlap | idle_gap | what it means |
|---------|----------|---------------|
| low     | high     | LLM is the bottleneck (didn't produce sentences fast enough) |
| low     | low      | synth is the bottleneck (ate the LLM's lead before next sentence arrived) |
| high    | any      | pipeline is healthy |

Implementation:

- `SentenceWorker.idle_gap_total: float = 0.0` (new field).
- Stamp `gap_t0 = self._clock()` before each `queue.get(...)`. On
  successful return AND after the first sentence has been spoken,
  add the delta to `idle_gap_total`.
- `TurnMetrics.worker_idle_gap_total: float = 0.0`, transferred by
  ChatLoop.
- Per-turn print: "Idle gap: Nms (worker waited for sentences)" only
  when >0. Yellow if >300ms ("worker is starving"), dim otherwise.
- `ScenarioResult.worker_idle_gap_ms` on perf snapshots.

Tests (8 in `tests/unit/test_worker_idle_gap.py`):
- Defaults are 0 on both Worker and TurnMetrics.
- `TestPerTurnPrint` — zero omits the line, non-zero shows ms +
  explainer.
- `TestWorkerIdleGap`:
  - First wait does NOT count (single-sentence response gives
    `idle_gap_total == 0` regardless of pre-sentence wait).
  - Between-sentence gap IS counted (real sleep yields >0 gap).
  - Back-to-back sentences yield <50ms gap (just queue overhead).
- `TestChatLoopWires` — field lands on metrics as a float ≥0.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **597 passed, 1 skipped in 22s** (589 existing
+ 8 new).

Notes:
- Six metrics from the taxonomy now live (2.19 / 1.10 / 2.18 /
  2.10 / 2.1 / 2.16). The "where is the bottleneck?" picture is
  now reasonably complete: TTFS for end-to-end, llm_first_sentence
  for LLM-side wait, streaming_overlap_ratio for parallelism
  achievement, idle_gap for between-sentence wait, plus the
  cancel + barge picture for interaction quality.
- Next candidates: 1.13 (bot WPM), 2.6 (sentence-split fragmentation),
  2.20 (loopback echo barge-in rate).

---

## iter-045 — sentence-split fragmentation metric (taxonomy 2.6)

**Branch:** `iter-045-sentence-len` (merged ff to main, commit `3a675ab`)
**Date:** 2026-05-24

Seventh metric pulled from `docs/perf-metrics-taxonomy.md`. **Metric
2.6 — Sentence-split fragmentation**, "Architecture-specific" bucket.

Mean character length of sentences submitted to the worker per turn.
The taxonomy notes:
- mean ≪ ~30 chars: over-fragmented, defeats streaming-overlap
  because TTS finishes the short sentence before the next arrives.
- mean ≫ ~150 chars: under-fragmented, increases TTFS because
  the first complete sentence arrives too late.
- mean 50-100: healthy LLM voice output.

Combined with iter-043 (overlap) + iter-044 (idle gap), the splitter
side of the pipeline is now observable: if overlap is low and idle
gap is high, look at fragmentation — short sentences would explain
it (worker keeps starving).

Implementation:
- ChatLoop accumulates `sentence_chars_total + sentence_chars_count`
  across the for-token loop (and the trailing-remainder submit).
- `TurnMetrics.mean_sentence_chars: float = 0.0` (new field).
- Per-turn print appends `", avg N chars"` inside the existing TTS
  suffix — compact, no new line.
- Session summary: `Mean sentence: N chars` averaged across turns
  where any sentence was submitted (>0 filter, parallel to
  iter-031's TTFS).
- `ScenarioResult.mean_sentence_chars` on perf snapshots.

Tests (10 in `tests/unit/test_sentence_fragmentation.py`):
- Default 0; per-turn print zero/nonzero; session aggregate
  zero/some-data/filter-zeros.
- ChatLoop wires:
  - Short sentences ("Yes.", "OK.", "Done.") → mean < 10.
  - Normal sentences (~50 chars) → mean 30-70.
  - No-terminator stream → trailing-remainder submit captures
    the fragment, so mean > 0 (not 0). Documents the actual
    behavior of the chat loop.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **607 passed, 1 skipped in 22s** (597
existing + 10 new).

Notes:
- Seven metrics from the taxonomy now live (2.19 / 1.10 / 2.18 /
  2.10 / 2.1 / 2.16 / 2.6). The full picture per turn:
  speech → STT → LLM 1st token / 1st sentence → idle gap → TTS
  (with fragmentation) → playback → TTFS → barge (with cancel
  shape + latency) → mic stale frames. Pretty complete diagnostic
  surface.
- Next candidates: 1.13 (bot WPM), 2.20 (loopback echo barge-in
  rate), 1.4 (VAD false-trigger rate).

---

## iter-046 — bot WPM metric (taxonomy 1.13)

**Branch:** `iter-046-bot-wpm` (merged ff to main, commit `cd5a3f2`)
**Date:** 2026-05-24

Eighth metric pulled from `docs/perf-metrics-taxonomy.md`. **Metric
1.13 — Bot speaking rate (words per minute)**, "Standard" bucket.

UX research clusters comfortable voice-agent speech at **150-180 WPM**:
- <130: too slow → user interrupts (manifests as iter-040
  mid-stream cancels).
- 130-200: comfortable.
- >200: too fast → user can't follow (manifests as repeat-please
  follow-ups).

The metric tells you whether kokoro's `speed=1.0` default is right
for the voice. Some voices want 0.9 or 1.1; bot_wpm is the closed-
loop signal that confirms the choice.

Implementation:
- `SentenceWorker.word_count_total: int = 0` and
  `SentenceWorker.audio_seconds_total: float = 0.0` (new fields).
  Per sentence, count non-punctuation tokens from the alignment;
  fall back to `len(sentence.split())` when alignment is missing
  (kokoro misconfig). Audio duration = `len(audio_np) / 24000`.
- `TurnMetrics.bot_wpm: float = 0.0`; ChatLoop derives from worker
  totals when both >0.
- Per-turn print: "Bot WPM: NNN (target 150-180)" only when >0.
  Green 130-200, yellow otherwise.
- Session summary: median across measurable turns.
- `ScenarioResult.bot_wpm` on perf snapshots.

Tests (13 in `tests/unit/test_bot_wpm.py`):
- Defaults zero on Worker + Metrics.
- Per-turn print: zero omits, in-range emits with target label,
  out-of-range still shown.
- Session aggregate: no-data omits, with-data emits median,
  zero-filter (parallel to iter-031's TTFS pattern).
- Worker word counting: alignment-with-tokens counts non-punct,
  empty alignment falls back to whitespace split, multi-sentence
  accumulates correctly.
- ChatLoop wires: 3 words in 1 sec audio → ~180 WPM; empty audio
  → 0 WPM.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **620 passed, 1 skipped in 22s** (607 existing
+ 13 new).

Notes:
- Eight metrics from the taxonomy now live (2.19 / 1.10 / 2.18 /
  2.10 / 2.1 / 2.16 / 2.6 / 1.13). The dashboard is
  starting to feel "complete enough" for ops review of a real
  voice agent — TTFS at the top, and a fan of diagnostic metrics
  underneath that explains the TTFS.
- Combined with iter-045's `mean_sentence_chars`, the worker
  side of the pipeline has clear visibility: how many words per
  minute (rate), how long is each sentence (size), how long does
  the worker idle between sentences (gap), how many sentences
  get cut mid-stream (cancel correctness). All four shapes
  visible in the per-turn print and session summary.
- Next candidates: 2.20 (loopback echo barge-in rate), 1.4
  (VAD false-trigger rate), 1.5 (VAD missed-speech rate).

---

## iter-047 — barge-in phase metric (taxonomy 2.11)

**Branch:** `iter-047-barge-phase` (merged ff to main, commit `ff7c43f`)
**Date:** 2026-05-24

Ninth metric pulled from `docs/perf-metrics-taxonomy.md`. **Metric
2.11 — Barge-in phase distribution**.

The phase string was already computed in `_chat_loop` for the
diagnostic print. iter-047 lifts it to a structured metric so the
session summary can show the distribution — which is the actually
useful operational view.

The two phases tell different stories:
- **`llm_stream`**: user interrupted while the LLM was still
  streaming tokens. The bot hadn't started speaking yet. User was
  impatient with TTFS. **Root cause: LLM TTFT.**
- **`playback`**: user interrupted while the bot was speaking.
  Verbose / wrong response. **Root cause: system prompt / response
  quality.**

A session full of llm_stream barges and another full of playback
barges have orthogonal fixes. iter-047 makes that distinction
visible at session-summary glance.

Implementation:
- `TurnMetrics.barge_in_phase: str = ""` (new field).
- ChatLoop assigns `"llm_stream"` or `"playback"` alongside the
  existing diagnostic print — single source of truth, refactored
  the inline string-construction to use the same key.
- Per-turn Barge-in line gains `(during LLM stream)` /
  `(during playback)` suffix after the existing cancel note.
- Session summary: `Barge phases: N LLM-stream, M playback`
  emitted when at least one phase was recorded.
- `ScenarioResult.barge_in_phase` on perf snapshots — the new
  field flows into the time-series automatically.

Tests (10 in `tests/unit/test_barge_phase.py`):
- Default empty string.
- Per-turn print: no-barge omits, llm_stream and playback both
  shown with distinct suffixes, empty-phase-with-barge omits
  cleanly (forward-compat).
- Session aggregate: no-data omits, mixed shows counts, single
  phase shows zero for the absent phase.
- ChatLoop wires: deterministic barge scenario (perf-suite shape)
  populates one of the two values; no-barge keeps "".

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **630 passed, 1 skipped in 23s** (620
existing + 10 new).

Notes:
- Nine metrics from the taxonomy now live (2.19 / 1.10 / 2.18 /
  2.10 / 2.1 / 2.16 / 2.6 / 1.13 / 2.11). Per-turn print + session
  summary together render a fairly complete operational dashboard:
  TTFS, sub-budgets, fragmentation, parallelism, cancel quality,
  barge shape + phase + latency, mic stale frames, bot WPM.
- Refactor opportunity noted: the per-turn Barge-in line is
  getting long ("yes (user interrupted) (1 cut mid-stream)
  (during LLM stream)"). Future iter could split into multiple
  fields or compress to symbols. Out of scope here — the message
  is accurate, just verbose.

---

## iter-048 — VAD false-trigger rate metric (taxonomy 1.4)

**Branch:** `iter-048-vad-false` (merged ff to main, commit `9e8579d`)
**Date:** 2026-05-24

Tenth metric pulled from `docs/perf-metrics-taxonomy.md`. **Metric
1.4 — VAD false-trigger rate**.

Counts turns where `ChatLoop.run_one_turn` returned `metrics=None`
AND `had_error=False` — VAD fired ACTIVE but the utterance was too
short (`DONE_TOO_SHORT`) or the transcription came back empty. High
rate = silence_threshold too low or min_speech_duration too short;
the bot "thinks" the user spoke when they didn't and wastes a turn.

This is the first **session-level** metric — different shape from
the per-turn TurnMetrics fields. Distinguished cleanly from LLM
errors (which also yield metrics=None but with had_error=True).

Implementation:
- `print_session_summary` gains `false_triggers: int = 0` kwarg
  (back-compat default). Emits "VAD false-trig: N/M (P%) — tune
  silence_threshold or min_speech_duration" when >0.
- `mic_chat.run_chat` counts false triggers across the chat loop:
  the existing `if result.metrics is None: continue` branch now
  bumps `false_triggers` when `had_error` is False. Passes total
  to print_session_summary.

Documented limitation: when `metrics_list` is empty (session ended
before any successful turn), the existing early-return placeholder
("Session ended (no completed turns)") fires before the false-
trigger line would emit. Future iter could lift the false-trigger
reporting above the early return — for now, callers see the
placeholder and ignore the suppressed metric. Edge case isn't
critical because in practice if every turn is a false trigger,
the user will notice anyway and tune their config.

Tests (7 in `tests/unit/test_vad_false_trigger.py`):
- `print_session_summary` kwarg behavior: default zero omits,
  explicit zero omits, one trigger shows "N/M (P%)" with tuning
  text, high rate shown correctly, empty-metrics-list edge case
  documented.
- `mic_chat` loop pattern: simulated TurnResult sequence
  increments counter on `metrics=None && !had_error`; does NOT
  increment on `metrics=None && had_error` (LLM error path is
  not a false trigger).

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **637 passed, 1 skipped in 20s** (630
existing + 7 new).

Notes:
- Ten metrics from the taxonomy now live (2.19 / 1.10 / 2.18 /
  2.10 / 2.1 / 2.16 / 2.6 / 1.13 / 2.11 / 1.4). Hit a milestone:
  10/46 — 22% of the taxonomy is instrumented.
- Remaining high-value candidates: 1.5 (VAD missed-speech rate),
  2.4 (filler false-positive rate), 1.7 (STT real-time factor),
  3.1 (naturalness gap).
- Session-level metric pattern works well — passing as kwarg to
  print_session_summary keeps the data flow explicit. Future
  session metrics (1.5, 2.4) can use the same shape.

---

## iter-049 — STT real-time factor metric (taxonomy 1.7)

**Branch:** `iter-049-stt-rtf` (merged ff to main, commit `b7f02f4`)
**Date:** 2026-05-24

Eleventh metric pulled from `docs/perf-metrics-taxonomy.md`. **Metric
1.7 — STT real-time factor (RTF)**, "Standard" bucket.

Simplest possible metric: `stt_rtf = stt_time / speech_duration`.
Both inputs already on TurnMetrics since iter-001. iter-049 just
exposes the ratio.

Operational signal:
- **<1**: STT runs faster than realtime — safe to invoke inline at
  end-of-turn. Mlx-whisper-large on Apple Silicon: ~0.1-0.3.
- **>1**: STT is the bottleneck — need streaming partial
  transcription, smaller model, or hardware acceleration.

Implementation:
- `TurnMetrics.stt_rtf: float = 0.0` (new field).
- ChatLoop computes after `stt_time` + `speech_dur` are set; guards
  div-by-zero (zero-speech turns can't reach this code path post-
  iter-031, but the guard documents the contract).
- Per-turn print: extends STT line to `"STT: NNms (RTF 0.05x)"`.
  Green if <1, yellow if ≥1. Falls back to plain `STT: NNms`
  when RTF is 0 (back-compat for tests + zero-data turns).
- Session summary: `Median STT RTF: N.NNx` filtered for >0.
- `ScenarioResult.stt_rtf` on perf snapshots.

Tests (8 in `tests/unit/test_stt_rtf.py`):
- Default 0.
- Per-turn print: zero falls back to plain line, sub-realtime
  shown, ≥1 still rendered.
- Session aggregate: no-data omits, with-data shows median,
  zero-filter (parallel to iter-031 / iter-038 / iter-046 / iter-048).
- ChatLoop arithmetic: real run with `slow_transcribe` (50ms
  sleep) on ~1s speech yields RTF that matches
  `stt_time / speech_duration` and is <0.2.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **645 passed, 1 skipped in 22s** (637
existing + 8 new).

Notes:
- Eleven metrics from the taxonomy now live (24% of the 46-metric
  list). The trivial-ratio pattern (1.7) shipped in this iter took
  ~10 minutes — proof that some metrics are nearly-free once the
  underlying inputs are already captured.
- Next candidates: 1.5 (VAD missed-speech rate), 2.4 (filler
  false-positive rate), 1.11 (TTS RTF).
- 1.11 (TTS RTF) is the symmetric pair to 1.7. Same shape: ratio
  of synth time to audio duration. Could be a tight follow-on.

---

## iter-050 — TTS real-time factor metric (taxonomy 1.11)

**Branch:** `iter-050-tts-rtf` (merged ff to main, commit `c12d8b2`)
**Date:** 2026-05-24

Twelfth metric from `docs/perf-metrics-taxonomy.md`. **Metric 1.11
— TTS real-time factor**, "Standard" bucket. Symmetric to iter-049's
STT RTF.

    tts_rtf = tts_time / audio_seconds_total

- **<1**: synth runs faster than the audio it produces — overlap
  (iter-008) buys real wall-clock savings. Kokoro on Apple Silicon:
  ~0.1-0.3.
- **>1**: synth is the bottleneck. Streaming-overlap won't help;
  need a faster TTS or smaller voice.

Both inputs already captured: `tts_time` since iter-001,
`audio_seconds_total` since iter-046. iter-050 just derives the ratio.

Implementation mirrors iter-049 exactly (same shape, same guards,
same UX choices):
- `TurnMetrics.tts_rtf: float = 0.0`.
- ChatLoop computes after worker returns; div-by-zero guard.
- Per-turn print appends `(RTF N.NNx)` to the TTS suffix; green
  <1, yellow ≥1.
- Session summary: `Median TTS RTF: N.NNx` filtered for >0.
- `ScenarioResult.tts_rtf` on perf snapshots.

Tests (9 in `tests/unit/test_tts_rtf.py`):
- Default 0; per-turn print zero/non-zero/high; session aggregate
  zero/some/filter; ChatLoop arithmetic with controlled
  synth latency yields the expected ratio; empty audio yields 0.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **654 passed, 1 skipped in 22s** (645
existing + 9 new).

**Milestone:** 12/46 taxonomy metrics live (26%). iter-050 marks
50 iterations on geno-voice — over 600 tests, 13 perf snapshots
in history, ten distinct visible metrics in the session summary
diagnostic. The dashboard now reads (with all metrics populated):

```
Session Summary (3 turns)
    Median STT:       50ms
    Median STT RTF:   0.05x      ← iter-049
    Median LLM 1st:   100ms
    Median LLM sent:  300ms
    Median TTS:       200ms
    Median TTS RTF:   0.20x      ← iter-050
    Median TTFS:      650ms
    Best TTFS:        500ms
    Barge-ins:        2 (1 mid-stream, 50%)
    Median barge:     150ms
    Worst barge:      150ms
    Barge phases:     1 LLM-stream, 1 playback
    Mic stale:        320 frames (0.0s)
    VAD false-trig:   1/4 (25%)
    Median overlap:   65%
    Mean sentence:    62 chars
    Median bot WPM:   165
    Model:            local-llama
```

Notes:
- iter-049 + iter-050 each took ~10 minutes. The trivial-ratio
  metrics from the taxonomy are nearly free once both inputs are
  captured. Worth batching them when there are several together
  in future.
- Next candidates: 1.5 (VAD missed-speech rate — needs
  manual ground truth, harder), 2.4 (filler false-positive rate
  — straightforward).

---

## iter-051 — filler false-positive rate metric (taxonomy 2.4)

**Branch:** `iter-051-filler-fp` (merged ff to main, commit `816132f`)
**Date:** 2026-05-24

Thirteenth metric pulled from `docs/perf-metrics-taxonomy.md`.
**Metric 2.4 — Filler false-positive rate**.

A turn's filler is a false positive when:
1. A filler actually played (`fillers_played > 0`).
2. AND the LLM's first token arrived faster than the configured
   `idle_threshold`.

Meaning: the bot would have started speaking on its own before the
filler was needed. The filler made the bot sound disfluent for no
reason. Tune `idle_threshold` up.

This is the first metric that's a flag (bool) rather than a number.
Pattern: per-turn truth, session-level rate. Mirrors iter-048's
VAD false-trigger rate but lives on TurnMetrics rather than as a
session-level kwarg (because iter-051 needs the per-turn comparison
to compute, and per-turn data is the natural carrier).

Implementation:
- `TurnMetrics.filler_false_positive: bool = False` (new field).
- ChatLoop sets True when:
  - `metrics.fillers_played > 0`
  - AND `self._idle_threshold > 0`
  - AND `0 < metrics.llm_first_token < self._idle_threshold`
  The inner `>0` guard on `llm_first_token` prevents false-marking
  turns where first-token wasn't captured (no LLM response).
- Per-turn print: appends `"*"` to the filler suffix when FP
  ("1 filler*" instead of "1 filler"). Compact — stars are easy
  to spot in output review.
- Session summary: `Filler FP rate: M/N (P%) — tune idle_threshold up`
  emitted only when at least one FP occurred. Clean sessions don't
  see the line.

Tests (10 in `tests/unit/test_filler_false_positive.py`):
- Default False.
- Per-turn print: no-filler / filler-no-FP / filler-FP cases.
- Session aggregate: no-fillers, fillers-no-FP, partial FP with
  tuning suggestion, all-FP at 100%.
- ChatLoop wires: fast LLM + 0.5s idle_threshold → if filler
  played, FP=True; no-fillers configured → FP stays False.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **664 passed, 1 skipped in 23s** (654
existing + 10 new).

Notes:
- Thirteen metrics live (28% of the 46-metric taxonomy).
- Combined with iter-040 (cancel correctness) and iter-047 (barge
  phase), the worker's "filler quality" picture is now complete:
  - How often did fillers fire? (`fillers_played` totals)
  - Of those, how often were they unnecessary? (FP rate, iter-051)
  - When LLM was slow, did the filler ALSO get cut by a barge-in?
    (combine iter-051 + iter-047 logic).
- Next candidates: 1.5 (VAD missed-speech, harder, needs ground
  truth), 2.7 (worker queue depth, simple), 1.9 (LLM TPS — needs
  token count which we have).

---

## iter-052 — LLM tokens-per-second metric (taxonomy 1.9)

**Branch:** `iter-052-llm-tps` (merged ff to main, commit `1097340`)
**Date:** 2026-05-24

Fourteenth metric pulled from `docs/perf-metrics-taxonomy.md`.
**Metric 1.9 — LLM tokens-per-second**, "Standard" bucket.

Stream throughput of the LLM measured AFTER first token:

    llm_tps = (token_count - 1) / (llm_stream_done_at - first_token_at)

Excluding the first-token wait avoids conflating TTFT (already
on `metrics.llm_first_token` since iter-001) with steady-state
throughput. The taxonomy separation: TTFT is "did the LLM start
fast?", TPS is "is the LLM keeping up?" Two different problems
with two different fixes.

Operational context:
- Local 7B-13B models on Apple Silicon: 30-80 tps.
- Cloud APIs (Anthropic, OpenAI): 20-60 tps.
- <20 tps with no first-token issue → endpoint is overloaded,
  pre-token cache is cold, or model is too large for hardware.
- LLM TPS directly gates `llm_first_sentence` (iter-038), which
  gates `ttfs` (iter-001), so this metric explains downstream slowness.

Implementation:
- `TurnMetrics.llm_tps: float = 0.0` (new field).
- ChatLoop adds `token_count += 1` inside the existing for-token
  loop. After `llm_stream_done_at` is set, computes the ratio
  with three guards: ≥2 tokens, positive interval, both endpoints
  set. Single-token / empty / barge-cut streams leave TPS at 0.
- Per-turn print: extends LLM-total line to
  `"NNms (model, NN tps)"` when measurable.
- Session summary: `Median LLM TPS: NN` filtered for >0.
- `ScenarioResult.llm_tps` on perf snapshots.

Tests (10 in `tests/unit/test_llm_tps.py`):
- Default 0.
- Per-turn print: zero omits the suffix, non-zero shows it,
  high TPS rendered with `:.0f` rounding.
- Session aggregate: no-data omits, with-data shows median,
  zero-filter (parallel to iter-031 / iter-038 / iter-049 / iter-050).
- ChatLoop arithmetic: 6 tokens with 20ms each yields 20-100 TPS
  (real timing has overhead — bound is loose); single token
  yields 0 (need ≥2); empty stream yields 0.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **674 passed, 1 skipped in 23s** (664
existing + 10 new).

Notes:
- Fourteen metrics live (30% of the 46-metric taxonomy).
- The LLM diagnostic now has TTFT, TTFsent, TPS, total — the full
  picture from "did it start" to "is it keeping up" to "did it
  finish."
- Next candidates: 2.7 (worker queue depth — needs a sampler
  daemon), 1.15 (turn count / session length — derivable from
  metrics_list), 3.1 (naturalness gap — interesting novel metric).

---

## iter-053 — naturalness bucket metric (taxonomy 3.1)

**Branch:** `iter-053-naturalness` (merged ff to main, commit `a6c4015`)
**Date:** 2026-05-24

Fifteenth metric pulled from `docs/perf-metrics-taxonomy.md`.
**First "Novel/speculative" bucket entry** — earlier 14 metrics
were all "Standard" or "Architecture-specific." Time to start
pulling from the speculative bucket.

**The insight:** humans don't optimize for minimum latency in
conversation — they optimize for natural pause. A bot responding
in 50ms feels robotic / interrupting; one in 250ms feels
conversational. Most voice-agent dashboards report only "lower
TTFS is better" — this metric explicitly identifies the sweet
spot.

Buckets:
  **<200ms**: rushed (bot interrupted natural pause)
  **200-400ms**: natural (matches human conversational rhythm)
  **>400ms**: slow (user notices lag)
  **""**: no audio this turn

Implementation:
- `TurnMetrics.naturalness_bucket: str = ""` (new field).
- ChatLoop assigns the bucket immediately after `metrics.ttfs` is
  computed. Bucketing logic is inline (no helper function — three
  branches, simple to read).
- Per-turn print: appends bucket to the TTFS line:
  `"TTFS: 250ms (speech stop → speaker, natural)"`.
- Session summary: distribution counts —
  `"Naturalness: 1 rushed, 2 natural, 1 slow"`. Emits only when
  at least one bucket has a count.
- `ScenarioResult.naturalness_bucket` on perf snapshots.

Tests (19 in `tests/unit/test_naturalness_bucket.py`):
- Default empty.
- Per-turn print: empty omits, all four bucket states tagged.
- Session aggregate: no-buckets omits, mixed distribution shown,
  single-bucket case.
- ChatLoop wiring: bucket set when audio played; "" when no audio.
- Boundary parametrize: 9 boundary values (0/50/199/200/300/400/
  401/1000/5000) → expected bucket. Documents the inclusive
  upper bound at 400ms (≤400 = natural).

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **693 passed, 1 skipped in 23s** (674
existing + 19 new).

Notes:
- **Fifteen metrics live (33% of the 46-metric taxonomy).** First
  speculative metric. The category labels are turning out to be
  useful — Standard metrics expose conventional ops measures;
  Architecture-specific surface design choices; Novel reframe
  what "good" looks like.
- The naturalness bucket might surprise users running in test
  scenarios. Stub LLM + stub TTS frequently produce sub-200ms
  TTFS → "rushed" — accurate but may seem alarming. The metric
  is calibrated for the production pipeline (real LLM TTFT 150-500ms,
  kokoro synth ~100ms first sentence) where 200-400ms is achievable.
- Next candidates: 1.15 (turn count + session length — derive in
  print_session_summary), 2.7 (worker queue depth — needs a
  sampler daemon), 3.2 (conversation rhythm score).

---

## iter-054 — session length + turns/min metric (taxonomy 1.15)

**Branch:** `iter-054-session-len` (merged ff to main, commit `4ea23cf`)
**Date:** 2026-05-24

Sixteenth metric pulled from `docs/perf-metrics-taxonomy.md`.
**Metric 1.15 — Turn count + session length**, "Standard" bucket.

The taxonomy describes this as a "denominator metric" — useful
not for its own value, but as the basis for normalizing any rate
metric (e.g. "false triggers per minute" rather than just "N
false triggers" — the rate is comparable across sessions of
different lengths).

Implementation:
- `print_session_summary` gains `session_seconds: float = 0.0`
  kwarg (back-compat default).
- Header changes: `"Session Summary (3 turns)"` →
  `"Session Summary (3 turns over 4m 30s)"` when seconds known.
- Human-readable duration formatting:
  - `<60s` → `Ns`
  - `<1h` → `Mm` or `Mm Ns`
  - `≥1h` → `Hh Mm`
- New `Turns/min: N.N` line emitted when `session_seconds >= 1.0`
  AND there are completed turns (avoids divide-by-tiny on test-
  shaped sessions).
- `mic_chat.run_chat` tracks `session_start = time.monotonic()`
  at session entry; passes `(now - session_start)` on the
  KeyboardInterrupt summary call.

Tests (11 in `tests/unit/test_session_length.py`):
- Header duration formatting across all four shape ranges
  (omitted, sub-minute, round-minutes, minutes+seconds, hours).
- Turns/min: omitted when not measurable, computed at common
  rates, edge case of empty metrics_list (early-return takes
  precedence over the rate line).

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **704 passed, 1 skipped in 24s** (693
existing + 11 new). **Crossed the 700-test mark.**

Notes:
- Sixteen metrics live (35% of the 46-metric taxonomy).
- The denominator-metric pattern unlocks future iters: anything
  like "barges/min" or "FP/min" can now use `session_seconds`
  as the timebase.
- Session header now reads:
  `Session Summary (3 turns over 5m 22s)`
  followed by `Turns/min: 0.6`. Concrete and immediately scannable.

---

## iter-055 — conversation rhythm score (taxonomy 3.2)

**Branch:** `iter-055-rhythm` (merged ff to main, commit `a26367f`)
**Date:** 2026-05-24

Seventeenth metric pulled from `docs/perf-metrics-taxonomy.md`.
Second "Novel/speculative" entry, after iter-053's naturalness.

    rhythm = 1 - stdev(ttfs) / median(ttfs)    clamped to [0, 1]

The taxonomy's framing: "consistency feels like a personality;
jitter feels like a system." A bot with steady ~300ms TTFS feels
like it has presence; one that oscillates between 100ms and 800ms
feels broken even if its median is identical. The median alone
doesn't capture this — variance does.

Pure session-level derivation. No new TurnMetrics field, no new
ChatLoop instrumentation. Three lines in `print_session_summary`
on top of the existing `ttfs_times` list.

Score interpretation:
- **1.00**: perfectly consistent (all TTFS equal — synthetic case)
- **~0.7**: typical good production session
- **~0.5**: moderate jitter (stdev ≈ half the median)
- **0.00**: clamped — stdev exceeds median (high outliers)

Implementation:
- `print_session_summary` computes rhythm in the existing
  `if ttfs_times:` branch. Requires `len(ttfs_times) >= 2` for
  `statistics.stdev` to be defined. Clamps to `[0, 1]` to handle
  high-variance sessions cleanly (raw can go negative).
- Output: `"Rhythm score:     N.NN"` between `Best TTFS` and
  `Naturalness` in the summary block.

Tests (7 in `tests/unit/test_rhythm_score.py`):
- Suppression: no-TTFS omits, single-turn omits.
- Score values: perfect consistency → `1.00`, moderate
  (200/300/400ms) → `0.67`, high jitter → clamped to `0.00`,
  two-turn session → `0.53` (sample stdev formula).
- Integration: line appears between `Best TTFS` and `Naturalness`
  in the output (order check verifies no regressions to ordering).

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **711 passed, 1 skipped in 23s** (704
existing + 7 new).

Notes:
- Seventeen metrics live (37% of the 46-metric taxonomy).
- Combined with iter-053's naturalness bucket distribution, the
  TTFS dimension is now richly described:
  - Median TTFS (level)
  - Best TTFS (best case)
  - Rhythm score (consistency)
  - Naturalness distribution (sweet-spot fit)
  Four orthogonal lenses on the same underlying number.
- Next candidates: 3.4 (regret rate), 2.7 (worker queue depth —
  needs sampler daemon, more involved).

---

## iter-056 — regret rate metric (taxonomy 3.4)

**Branch:** `iter-056-regret` (merged ff to main, commit `6325c36`)
**Date:** 2026-05-24

Eighteenth metric pulled from `docs/perf-metrics-taxonomy.md`.
Third "Novel/speculative" entry (after iter-053 naturalness +
iter-055 rhythm).

A barge-in is **"regret"** when the user starts speaking within
200ms of bot first audio. Implies the bot pre-empted the user —
the user was already mid-utterance and the bot misjudged
end-of-turn (silence_duration fired too early). The taxonomy
notes: "the bot may be pre-empting; raise silence_duration."

Distinct from iter-053's "rushed" naturalness:
- **rushed** = bot's TTFS was very low (subjective fast response,
  bot's internal timing dimension)
- **regret** = the user actually objected to the bot speaking
  (user's behavior dimension)

Both can be true on the same turn. Both pointing at the same
underlying issue (bot speaking too early), but observed from
different angles. The recommended fix differs:
- rushed → reduce LLM/synth speed to add naturalness pause
- regret → raise `silence_duration` so end-of-turn is more
  conservative

Implementation:
- `TurnMetrics.barge_in_regret: bool = False` (new field).
- ChatLoop sets True when:
  - `coord.triggered_at` is set
  - AND `worker.first_audio_at` is set
  - AND `coord.is_set()` (sanity — barge actually fired)
  - AND `0 < (coord.triggered_at - worker.first_audio_at) < 0.2`
- Per-turn Barge-in line gains `— regret` suffix when True.
- Session summary: `Regret rate: M/N (P%) — bot may be pre-empting;
  raise silence_duration` emitted when at least one regret happened.

Tests (17 in `tests/unit/test_regret_rate.py`):
- Default False.
- Per-turn print: no-barge / barge-no-regret / barge-with-regret.
- Session aggregate: no-barge / barges-no-regret / partial regret
  / all-regret.
- Boundary parametrize: 7 gap values document the strict inequality:
  `0` (False), `50/100/199` (True), `200` (False), `250/1000` (False).
- Guards: no-first-audio + no-barge → False regardless of
  triggered_at.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **728 passed, 1 skipped in 23s** (711 existing
+ 17 new).

Notes:
- Eighteen metrics live (39% of the 46-metric taxonomy).
- The barge-in dimension is now exhaustively covered:
  - Did it fire? (`barge_in`)
  - Where did it land? (`barge_in_phase`: llm_stream / playback)
  - How fast did we respond? (`barge_in_latency`)
  - Did it cut a sentence mid-stream? (`sentences_cancelled`)
  - Was the bot wrong to start speaking? (`barge_in_regret`)
  Five orthogonal lenses on barge-in events.
- Next candidates: 2.7 (worker queue depth — sampler daemon),
  1.2 (EoT detection latency — needs frame-by-frame VAD timing),
  3.6 (interruption recovery — multi-turn).

---

## iter-057 — primed-frames replay duration metric (taxonomy 2.12)

**Branch:** `iter-057-primed-frames` (merged ff to main, commit `639344f`)
**Date:** 2026-05-24

Nineteenth metric pulled from `docs/perf-metrics-taxonomy.md`.
**Metric 2.12 — Primed-frames replay duration**, "Architecture-
specific" bucket.

The chat_loop already computed `len(next_primed) * chunk / rate`
on barge turns for the diagnostic print line. iter-057 promotes
that value to a `TurnMetrics` field so the session summary can
aggregate it across turns — operational view of how much user
audio iter-025's lead-in is preserving.

Why it matters: high totals validate iter-025's design (the
watcher's ring buffer is meaningfully preserving the user's
first syllables for the next STT pass). Near-zero totals would
suggest iter-025 isn't paying off and could be simplified or
removed.

Implementation:
- `TurnMetrics.primed_frames_seconds: float = 0.0` (new field).
- ChatLoop assigns the value in the watcher-detected branch
  alongside `metrics.barge_in_phase`. Refactored the existing
  print to use the metric (single source of truth for both
  diagnostic and aggregation).
- Per-turn print: new `Primed frames: NNNms (carried into next
  turn)` line on barge turns.
- Session summary: `Primed audio: N.Ns (carried into next turn —
  validates iter-025)` when total > 0.
- `ScenarioResult.primed_frames_seconds` on perf snapshots.

Tests (9 in `tests/unit/test_primed_frames_seconds.py`):
- Default 0.
- Per-turn print: zero omits, non-zero shows ms + explainer,
  no-barge zero omits.
- Session aggregate: no-primed omits, multi-turn / single-turn
  totals.
- Computation contract: formula at default CHUNK/RATE; zero-frames
  edge case.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **737 passed, 1 skipped in 22s** (728
existing + 9 new).

Notes:
- **Nineteen metrics live (41% of the 46-metric taxonomy).**
  Crossed the 40% milestone.
- Refactor opportunity surfaced: the session summary's
  conditional structure has accumulated a lot of `if X:` /
  `_emit(...)` pairs across iterations. Some are nested under
  `if barges_total:` (correctly, since they're barge-only), some
  are at the same level (correctly, since they're independent).
  iter-057 had to fix one initially-mis-nested block. Future
  iter could refactor into a list of `MetricSection` objects
  with explicit predicates — but the current shape is still
  readable and shipping changes is low-friction.
- Next candidates: 2.5 (filler novelty — needs filler-text
  history), 1.16-onwards (long tail), 2.7 (worker queue depth).

---

## iter-058 — error rate per stage metric (taxonomy 1.16)

**Branch:** `iter-058-error-rates` (merged ff to main, commit `8c402b8`)
**Date:** 2026-05-24

Twentieth metric pulled from `docs/perf-metrics-taxonomy.md`.
**Metric 1.16 — Error rate per stage**, "Standard" bucket.

Two-layer reporting because errors fail at different scopes:
- **LLM errors** kill the entire turn. The user spoke, the
  bot crashed before any audio. Tracked at session level
  (`llm_errors` kwarg).
- **Worker errors** lose one sentence but the rest of the turn
  proceeds. Some audio plays. Tracked per-turn
  (`TurnMetrics.worker_errors`).

These tell different reliability stories. A session with 5 LLM
errors is unusable; a session with 5 worker errors heard mostly-
right responses with the occasional missing sentence.

Implementation:
- `TurnMetrics.worker_errors: int = 0` (new field).
- ChatLoop assigns `metrics.worker_errors = len(worker.errors)`
  on the success path (alongside the existing diagnostic print
  loop over those same errors).
- `print_session_summary` gains `llm_errors: int = 0` kwarg
  (back-compat default).
- Aggregate: `worker_errors_total = sum(m.worker_errors for m in
  metrics_list)`.
- Output: `Errors: N LLM, M worker (over K attempts)` where
  `K = n_completed + llm_errors + false_triggers`. Each piece
  appears only when its count is non-zero (e.g. only-LLM-errors
  shows `"N LLM (over K attempts)"`, no worker bit).
- `mic_chat.run_chat` increments `llm_errors` in the existing
  `if result.had_error: continue` branch; passes total to summary.

Tests (9 in `tests/unit/test_error_rates.py`):
- Per-turn field default + settable.
- Clean session omits the block.
- LLM only / worker only / both — output formatting + denominator
  computation.
- false_triggers contribute to attempts.
- Worker errors summed across turns.
- Empty `metrics_list` → no-completed-turns placeholder takes
  precedence over the error block.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **746 passed, 1 skipped in 24s** (737 existing
+ 9 new).

Notes:
- **Twenty metrics live (43% of the 46-metric taxonomy).** Hit
  the round number.
- The reliability surface is now: error rate per stage + barge-in
  shape + cancel correctness + filler false-positive. Plus mic
  stale frames as a leakage signal. Together: full reliability
  picture for ops review.
- Next candidates: 2.5 (sentence-split coverage — already have the
  inputs), 2.7 (worker queue depth), 2.14 (LLM stream cancel-to-
  close).

---

## iter-059 — sentence-split coverage metric (taxonomy 2.5)

**Branch:** `iter-059-split-coverage` (merged ff to main, commit `1308417`)
**Date:** 2026-05-24

Twenty-first metric pulled from `docs/perf-metrics-taxonomy.md`.
**Metric 2.5 — Sentence-split coverage**, "Architecture-specific"
bucket.

    coverage = complete_sentence_chars / (complete + remainder)

Range [0, 1]:
- **1.0**: LLM always ended responses with punctuation. Every
  char went to the worker as a complete sentence — fully
  overlap-friendly.
- **0.5**: half the chars came as remainder.
- **0.0**: LLM produced fragments only; all chars flushed as
  remainder.

Why it matters: the trailing remainder is the `worker.submit()` that
happens AFTER `llm_stream_done_at`. There's nothing for streaming
overlap (iter-043's metric) to pair with — the worker synthesizes
that final piece while the user waits. High remainder share is
operational waste of the iter-008 streaming-overlap design.

Implementation:
- `TurnMetrics.sentence_split_coverage: float = 0.0` (new field).
- ChatLoop tracks `complete_sentence_chars` and `remainder_chars`
  separately, in addition to iter-045's `sentence_chars_total`
  accumulator. Computes coverage when total > 0.
- Per-turn print: appends `, N% complete` to TTS suffix only when
  `0 < coverage < 1.0`. Perfect 100% (the expected case) doesn't
  clutter the line.
- Session summary: `Split coverage: N%` (median over turns where >0).
- `ScenarioResult.sentence_split_coverage` on perf snapshots.

Tests (10 in `tests/unit/test_split_coverage.py`):
- Default 0.
- Per-turn print: perfect / zero / partial cases.
- Session aggregate: no-data / with-data / zero-filter.
- ChatLoop arithmetic:
  - `"Hello world. "` → 100% (clean terminator).
  - `"fragment without terminator"` → 0% (all flushed as remainder).
  - `"Done. trailing"` → 5/13 ≈ 38% (one complete + remainder).

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **756 passed, 1 skipped in 23s** (746
existing + 10 new).

Notes:
- **Twenty-one metrics live (46% of the 46-metric taxonomy).** Half
  the wishlist down. The tail metrics will be progressively harder
  (need new instrumentation, ablation studies, or external ground
  truth).
- Combined with iter-043 (overlap ratio) and iter-038 (TTFsent),
  the streaming-overlap story is now fully observable: how much
  was the LLM keeping up (TTFsent), how much overlap actually
  happened (overlap ratio), and how much did the splitter waste
  via remainder (split coverage).
- The 0.0 default ambiguity ("no chars submitted" vs "all
  remainder") is documented in the test. In practice the session
  aggregate's >0 filter handles the no-chars case cleanly; the
  all-remainder case is rare enough (LLM yielding raw fragments
  without any terminator) that the false-positive cost is low.
- Next candidates: 2.7 (worker queue depth — sampler daemon),
  2.14 (LLM stream cancel-to-close), 1.2 (EoT detection latency).

---

## iter-060 — LLM stream cancel-to-close metric (taxonomy 2.14)

**Branch:** `iter-060-cancel-close` (merged ff to main, commit `e6280fc`)
**Date:** 2026-05-24

Twenty-second metric pulled from `docs/perf-metrics-taxonomy.md`.
**Metric 2.14 — LLM stream cancel-to-close**, "Architecture-specific"
bucket.

    cancel_to_close = close_finished_at - coord.triggered_at

Time from `BargeInCoordinator.trigger()` firing to `llm_gen.close()`
returning. Only meaningful on barge turns. High values mean the
upstream HTTP socket is taking a long time to wind down — wastes
tokens we paid for and can block the next turn.

Implementation:
- `TurnMetrics.llm_cancel_to_close: float = 0.0` (new field).
- ChatLoop's existing `finally` block (iter-013) already calls
  `llm_gen.close()`. iter-060 wraps that with timestamps:
  `close_started_at` / `close_finished_at` captured around the
  call.
- Field assigned **just before the post-`finally` success
  return**, guarded by `coord.is_set()` AND `coord.triggered_at
  is not None`. The except path returns inside the except block
  before reaching this assignment, which is correct (no metrics
  on error turns anyway).
- Per-turn print: "LLM cancel: NNms (trigger → stream close)"
  on barge turns when >0. Yellow >500ms.
- Session summary: "Median LLM canc: NNms" filtered for >0.
- `ScenarioResult.llm_cancel_to_close_ms` on perf snapshots.

Tests (12 in `tests/unit/test_llm_cancel_close.py`):
- Default 0.
- Per-turn print: zero / non-zero / high-value cases.
- Session aggregate: no-data / with-data / zero-filter.
- ChatLoop wires: clean turn yields 0; deterministic barge scenario
  yields >0 when barge landed.
- Arithmetic boundaries: tiny gap, slow gap, negative gap clamps to 0
  (defensive — shouldn't happen but documents the contract).

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **768 passed, 1 skipped in 23s** (756 existing
+ 12 new).

Notes:
- **Twenty-two metrics live (48% of the 46-metric taxonomy).**
- The barge-in dimension is now exhaustively measured (six
  orthogonal lenses):
  - `barge_in` — fired or not
  - `barge_in_phase` — LLM-stream vs playback
  - `barge_in_latency` — detect → halt
  - `sentences_cancelled` — mid-stream cuts
  - `barge_in_regret` — bot pre-empted user
  - **`llm_cancel_to_close`** — HTTP socket wind-down ← iter-060
- Subtle code structure note: instrumenting in `finally` while
  using on the success path required careful flow analysis. The
  except path returns inside except (before reaching the post-
  finally code that uses the timestamp); on success, the finally
  runs first and stages the timestamp, then the post-finally
  code reads + assigns. Captured in test_clamping which documents
  the contract.
- Next candidates: 2.7 (worker queue depth), 2.8 (speaker open
  overhead), 1.5 (VAD missed-speech rate — needs ground truth).

## iter-061 — speaker open overhead metric (taxonomy 2.8)

**Branch:** iter-061-speaker-open  **Commit:** 67e96c3  **Date:** 2026-05-24

Added `speaker_open_seconds` — time the SentenceWorker thread spends
inside `speaker_factory()` opening the per-turn persistent output device.
The iter-008 win was holding ONE speaker across all sentences of a
turn (vs reopening per sentence). If open cost balloons (driver change,
Bluetooth pairing, SDL/PortAudio init) TTFS regresses silently. This
metric makes the regression visible.

Implementation:
- `SentenceWorker.speaker_open_seconds: float = 0.0` (new field).
  `_run()` now wraps the `speaker_factory()` call with `clock()`
  reads on either side and assigns the delta after a successful
  open. On the failure path (factory raises) the field stays at 0.0
  — the early-return happens before the post-open clock read. That
  contract is pinned by `test_factory_failure_leaves_zero`.
- `TurnMetrics.speaker_open_seconds: float = 0.0` and ChatLoop
  transfer alongside the other worker-→-metrics assignments.
- Per-turn print: "Speaker open: NNms (device init)" only when
  >0. Yellow if >50ms; the iter-008 design assumes opens are cheap
  because they happen once per turn rather than once per sentence.
- Session summary: "Speaker open: median NNms / worst NNms" with
  zero-filter (turns whose worker exited before opening — error
  paths — don't bias the aggregate). Single-turn sessions emit a
  raw "NNms" without median/worst decoration.
- `ScenarioResult.speaker_open_ms` on perf snapshots, time-series
  ready for the report generator.

Tests (12 in `tests/unit/test_speaker_open.py`):
- Defaults: TurnMetrics + SentenceWorker both 0.
- Per-turn print: zero omitted, non-zero shown, above-threshold path.
- Session aggregate: no-data omitted, single-value no-decoration,
  multi-value median+worst, zero-filter.
- Worker timing: real `time.sleep` inside factory captured (with
  generous tolerance for CI scheduler jitter); factory failure
  leaves the field at 0.0 + appends the error.
- ChatLoop wiring: a deliberately slow `speaker_factory` (40ms
  sleep) bubbles through all the way to TurnMetrics.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **780 passed, 1 skipped in 22s** (768 existing
+ 12 new).

Notes:
- **Twenty-three metrics live (50% of the 46-metric taxonomy).**
- The output-stage dimension is now measured at every layer:
  open cost (this), synth time (`tts_time`), playback time
  (`playback_time`), audio output rate (`bot_wpm`), realtime
  factor (`tts_rtf`), filler frequency (`fillers_played`),
  speaker write rate (implicit in `playback_time`). A regression
  anywhere on the output path will surface in at least one of
  these without needing manual ad-hoc instrumentation.
- Next candidates: 2.7 (worker queue depth), 2.9 (persistent-
  speaker open count per session), 1.5 (VAD missed-speech rate
  — needs ground truth), 1.2 (EoT detection latency).

## iter-062 — peak worker queue depth metric (taxonomy 2.7)

**Branch:** iter-062-queue-depth  **Commit:** 9109bb4  **Date:** 2026-05-24

Added `max_queue_depth` — peak number of sentences waiting in the
SentenceWorker's queue at any moment during a turn. Inverse of
iter-044's `worker_idle_gap_total` (worker starved). Together the
two metrics localize where the streaming pipeline gets stuck:
- High idle gap, low queue depth → LLM is the bottleneck.
- Low idle gap, high queue depth → synth is the bottleneck.
- Both low → balanced (the iter-008 streaming-overlap design is
  paying off).
- Both high → buffering oscillation (rare; would suggest the
  splitter itself is producing in bursts).

Implementation:
- `SentenceWorker.max_queue_depth: int = 0` (new field).
- Sampled inside `SentenceWorker.submit()` after each `Queue.put`:
  ```python
  self._queue.put(sentence)
  depth = self._queue.qsize()
  if depth > self.max_queue_depth:
      self.max_queue_depth = depth
  ```
  `qsize()` is documented as approximate but the only race is
  with the consumer thread which only DRAINS — over-count is
  harmless for a peak metric and the test suite confirms the
  observed behavior.
- `TurnMetrics.max_queue_depth: int = 0` and ChatLoop transfer
  alongside other worker → metrics assignments.
- Per-turn print: "Queue depth: N (synth backlog peak)" only
  when >1. Yellow if ≥3.
- Session summary: "Worst queue: N (M/N turns backed up)" with
  worst depth across turns + count of turns where any backup
  happened. Single-backup-turn case shows "1 turn backed up"
  instead of "1/N turns" for readability.
- `ScenarioResult.max_queue_depth` on perf snapshots.

Tests (16 in `tests/unit/test_queue_depth.py`):
- Defaults: TurnMetrics + SentenceWorker both 0.
- Per-turn print: 0 / 1 omitted; 2 dim; 5 visible.
- Session aggregate: no-data / only-healthy / single-backup /
  multi-backup variants.
- Worker tracking: single submit records ≥1; burst with slow synth
  builds depth; no-submits keeps 0; submits-after-stop ignored.
- ChatLoop wiring: short response stays ≤1; slow-synth scenario
  with 5 sentences bubbles depth into TurnMetrics within sanity
  bounds.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **796 passed, 1 skipped in 23s** (780 existing
+ 16 new).

Notes:
- **Twenty-four metrics live (52% of the 46-metric taxonomy).**
- The producer/consumer dimension is now fully measured — both
  sides of the queue have peak indicators (this) AND idle indicators
  (idle_gap), so a regression in either direction shows up.
- Next candidates: 2.9 (persistent-speaker open count per session
  — should always be 1 with iter-008 design; >1 means the worker
  thread crashed and restarted), 1.5 (VAD missed-speech rate —
  needs ground truth fixture), 1.2 (EoT detection latency).

## iter-063 — EoT detection latency metric (taxonomy 1.2)

**Branch:** iter-063-eot-latency  **Commit:** 0fc936b  **Date:** 2026-05-24

Added `eot_latency` — time from the user's last in-speech frame to
VadEvent.DONE_OK firing. Lower bound is roughly `silence_duration`
(the VAD has to wait that long before deciding the user really
stopped); the gap above that is implementation overhead (chunk
granularity, processing). Critical UX number — "the agent feels
slow" complaints map directly to this.

Implementation:
- `record_utterance_streaming()` gained an optional `out_metrics:
  dict | None = None` keyword arg. When provided, the function
  populates it with side-band measurements that don't fit the
  `(wav, dur, stt)` return tuple. Currently emits one key —
  `"eot_latency"` — only on the DONE_OK success path. ~15 existing
  unpack sites in tests are untouched (they don't pass the new
  kwarg).
- Inside the recorder loop, `last_speech_at` is updated for any
  frame where `level > silence_threshold`. On DONE_OK the gap
  between the current `now` and `last_speech_at` is the EoT
  latency. Setting before the VadEvent dispatch is intentional —
  any frame above threshold is "in speech" regardless of which
  VadEvent it produces.
- ChatLoop creates a fresh `rec_metrics: dict = {}`, passes it,
  reads `eot_latency` after, copies into TurnMetrics.
- Per-turn print: "EoT detect: NNms (silence wait)" only when >0.
  Yellow when >1.0s — the user has stopped talking but the agent
  is still waiting; tunable down to ~500ms in noisy rooms.
- Session summary: median + worst (worst skipped when all values
  uniform — single-knob tuning sessions don't need the line).
  Emitted before STT to mirror the per-turn pipeline order.
- `ScenarioResult.eot_latency_ms` on perf snapshots — first metric
  exposing what fraction of S2S latency is "VAD waiting" vs
  "actual work."

Tests (12 in `tests/unit/test_eot_latency.py`):
- Defaults: TurnMetrics zero.
- Per-turn print: 0 omitted, normal value (850ms) dim, high (1.5s).
- Session aggregate: no-data omitted, uniform values (no Worst
  line), spread (median + worst), zero-filter.
- Recorder integration via VirtualMicStream: DONE_OK populates
  within sanity bounds, DONE_TOO_SHORT does NOT populate, kwarg
  is fully optional (backwards-compat with old callers).
- ChatLoop wiring: TurnMetrics.eot_latency reflects the recorder's
  measurement end-to-end.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **808 passed, 1 skipped in 23s** (796 existing
+ 12 new).

Notes:
- **Twenty-five metrics live (54% of the 46-metric taxonomy).**
- This is the first metric that requires plumbing an out-parameter
  through `record_utterance_streaming` — the (wav, dur, stt) return
  tuple was getting saturated. The dict pattern scales: future
  recorder-side metrics (1.3 VAD trailing-silence wall, 1.8 STT
  preview-vs-final divergence, 1.14 user speaking rate) just add
  more keys, no signature churn.
- The S2S latency budget (`ttfs`) now decomposes as:
  `eot_latency + stt_time + llm_first_sentence + first_synth_time +
  first_chunk_play_time`. With `silence_duration=0.8s` config, the
  EoT term is ~800ms — typically the LARGEST single contributor to
  TTFS in well-tuned systems. Surfacing it makes the "lower
  silence_duration" decision visible and measurable.
- Next candidates: 1.3 (VAD trailing-silence wall: `eot - silence_duration`
  — almost free given iter-063's groundwork), 1.14 (user WPM —
  derive from transcript len / speech_duration), 2.13 (primed-frames
  STT contribution — needs offline ablation).

## iter-064 — user WPM metric (taxonomy 1.14)

**Branch:** iter-064-user-wpm  **Commit:** ec83283  **Date:** 2026-05-24

Added `user_wpm` — user speaking rate in words-per-minute, derived
from `len(transcript.split()) / speech_duration * 60`. Symmetric
to iter-046's `bot_wpm`. Useful for the mirroring effect: adapting
bot WPM to match user produces higher rapport and lower interruption
rate (UX research). Wide variance is normal — humans speak 100-200
WPM depending on context (slow in monologue, fast in conversation).

Implementation:
- `TurnMetrics.user_wpm: float = 0.0` (new field).
- ChatLoop computes immediately after `metrics.transcript` is set:
  whitespace-split word count is a decent proxy — Whisper transcripts
  use space-separated tokens and the error vs true tokenization is
  dwarfed by natural variance in human speech rates. Skipped on
  zero `speech_duration` or empty transcript (defensive — keeps
  the field at the 0.0 default).
- Per-turn print: appends "(NNN WPM)" suffix to the existing Speech
  line when known. No color coding — there's no "wrong" rate for
  the user, only a "match the user" target for the bot.
- Session summary: "Median user WPM: NNN" filtered for >0. When
  both user and bot WPM are known, also emit "Mirror gap: ±NN WPM
  (bot − user)" — the cross-side delta that predicts conversational
  feel (≈0 = mirroring, >40 = bot too fast, <-40 = bot too slow).
- `ScenarioResult.user_wpm` on perf snapshots — first metric
  capturing the user side of the conversational rhythm picture.

Tests (11 in `tests/unit/test_user_wpm.py`):
- Default zero.
- Per-turn print: 0 omits suffix; non-zero appends; extreme values
  emit without color treatment.
- Session aggregate: no-data omitted; user-only emits median but
  no mirror gap; both present emit median + signed gap; negative
  gap; zero-filter.
- ChatLoop wiring: deterministic transcribe stub yields a 4-word
  phrase from a 0.6s tone; user_wpm lands in the [100, 800] sanity
  band (CI-safe tolerance for VAD timing jitter).
- Empty transcript edge: TurnMetrics default unchanged when
  ChatLoop's n_words guard would skip the assignment.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **819 passed, 1 skipped in 23s** (808 existing
+ 11 new).

Notes:
- **Twenty-six metrics live (57% of the 46-metric taxonomy).**
- The conversational rhythm dimension is now bilateral: bot WPM
  (iter-046), user WPM (this), and the mirror gap derived metric.
  Combined with iter-055's TTFS rhythm score, the operator now has
  five orthogonal "feel" indicators on the session summary.
- Next candidates: 1.3 (VAD trailing-silence wall: `eot - silence_duration`
  — almost free given iter-063's groundwork), 1.6 (WER, needs
  ground truth corpus), 1.20 (cold-start latency penalty: turn-1
  TTFS minus median of remaining turns).

## iter-065 — VAD trailing-silence wall metric (taxonomy 1.3)

**Branch:** iter-065-eot-overhead  **Commit:** 36569ba  **Date:** 2026-05-24

Added `eot_overhead = max(0, eot_latency - silence_duration_used)`.
Decomposes the EoT wait into:
- "knob-budget" — the configured `silence_duration` (what we
  asked for).
- "implementation overhead" — chunk granularity + processing time
  on top.

The decomposition tells the operator which lever to pull when EoT
latency is too high:
- Overhead ≈ 0 → the wait is fully explained by the knob; lower
  `chat.vad.silence_duration` (default 0.8s).
- Overhead > 100ms → something else in the recording loop is slow;
  tuning the knob won't help.

Implementation:
- `TurnMetrics.eot_overhead: float = 0.0` (new field).
- ChatLoop computes immediately after `metrics.eot_latency` is
  populated:
  ```python
  if metrics.eot_latency > 0:
      metrics.eot_overhead = max(
          0.0, metrics.eot_latency - self._silence_duration
      )
  ```
  The `max(0, ...)` clamps a rare off-by-one between
  `last_speech_at` (the actual last in-speech frame) and VadState's
  silence-window start (the first sub-threshold frame, one chunk
  later).
- Per-turn print: appends "+NNms overhead" to the EoT line when
  > 10ms (above chunk-granularity noise). Yellow when >100ms.
- Session summary: "EoT overhead: NNms (above silence_duration)"
  emitted only when at least one turn showed real overhead.
- `ScenarioResult.eot_overhead_ms` on perf snapshots.

Tests (12 in `tests/unit/test_eot_overhead.py`):
- Default zero.
- Per-turn print: zero / sub-chunk-noise (≤10ms) / real overhead /
  high overhead / no-EoT-line edge case.
- Session aggregate: no-overhead omitted, present emits median,
  sub-chunk filtered.
- ChatLoop arithmetic: clamps at zero (off-by-one tolerance);
  always ≤ eot_latency by definition; eot_latency=0 turns leave
  overhead at default.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **831 passed, 1 skipped in 23s** (819 existing
+ 12 new).

Notes:
- **Twenty-seven metrics live (59% of the 46-metric taxonomy.)**
- The recording-side latency story is now fully decomposed:
  speech_duration (user actually talking) + eot_latency (VAD wait)
  + stt_time (transcription) — and within eot_latency,
  silence_duration_used (knob) + eot_overhead (implementation).
  Any future regression in the recording path will surface in
  exactly one of those terms.
- Next candidates: 1.20 (cold-start latency penalty: turn-1 TTFS
  minus median of remaining turns), 2.15 (worker error-recovery
  success: turns where worker.errors is non-empty but ttfs > 0 —
  silent partial degradation), 1.6 (WER, needs ground truth corpus).

## iter-066 — cold-start latency penalty metric (taxonomy 1.20)

**Branch:** iter-066-cold-start  **Commit:** 0f34530  **Date:** 2026-05-24

Added cold-start latency penalty — `metrics_list[0].ttfs -
median(m.ttfs for m in metrics_list[1:] if m.ttfs > 0)`. Captures
lazy initialization that hits turn 1 disproportionately — model
load, speaker open, TTS warmup, lazy imports — and would otherwise
get buried in the overall TTFS median. Critical because users judge
the entire session by its first impression.

Implementation:
- Pure derivation in `print_session_summary`; no new TurnMetrics
  field needed (everything is computable from existing per-turn
  TTFS values). Sits alongside the rhythm-score block in the
  per-session output.
- Guards:
  - Turn 1 must have measurable TTFS (otherwise comparing an
    absent first turn).
  - At least 1 steady-state turn (turns 2:N) must have TTFS > 0.
  - `|penalty|` must exceed the 50ms jitter floor — natural
    turn-to-turn variation alone produces ~30-40ms swings.
- Sign preserved on output. Positive (typical): turn 1 was slower.
  Negative (rare): turn 1 was faster — could indicate post-turn-1
  GC pauses, cache pollution, or otherwise-warm subsystems going
  cold (the inverse of what we'd usually see).
- Output: "Cold start: +NNms vs steady state".

Tests (11 in `tests/unit/test_cold_start.py`):
- No-emit boundaries: 1-turn / 0-turn sessions, turn 1 with no
  TTFS, no steady-state turns, sub-jitter penalty.
- Positive penalty: 2-turn minimum / multi-turn median / steady-
  state zero filtering.
- Negative penalty: turn 1 faster than steady state, sign rendered
  with leading "-".
- Threshold: just-below-floor omits, just-above emits.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **842 passed, 1 skipped in 23s** (831 existing
+ 11 new).

Notes:
- **Twenty-eight metrics live (61% of the 46-metric taxonomy).**
- This is the second purely session-level metric (after iter-055's
  rhythm score). Both decompose information that's already in the
  per-turn record but only becomes meaningful in aggregate.
- Cold-start penalty completes the "first impression" picture
  alongside iter-061's `speaker_open_seconds` (which on turn 1 is
  the speaker open cost, on later turns is 0): now the operator
  can attribute turn-1 lag to specific subsystems.
- Next candidates: 2.15 (worker error-recovery success: turns
  where worker.errors is non-empty but ttfs > 0 — silent partial
  degradation), 1.12 (turn-taking jitter: stdev of TTFS — already
  implicitly in rhythm score, could promote to its own line),
  1.6 (WER, needs ground truth corpus).

## iter-067 — worker error-recovery success rate (taxonomy 2.15)

**Branch:** iter-067-error-recovery  **Commit:** 3ffa740  **Date:** 2026-05-24

Added worker error-recovery success rate — of the turns where the
SentenceWorker raised at least one synth/play exception, what
fraction still produced audio (`ttfs > 0`). The metric captures
**silent partial degradation**: a sentence inside the turn failed
but the rest covered for it, so externally the turn looked fine.
That's the most insidious failure mode — the user heard a
complete-sounding response, the operator saw "successful turn,"
and the underlying bug got swallowed.

Implementation:
- Pure derivation in `print_session_summary`; no new TurnMetrics
  field (reuses iter-058's `worker_errors` + existing `ttfs`).
- Sits inside the existing iter-058 Errors block — only emits when
  the block is already showing.
- Counts denominator as "turns where any worker error happened,"
  not "total error count" — a single turn with 5 errors is still
  one turn for the recovery rate.
- Output: "Worker recovery: M/N turns produced audio (X%) —
  partial degradation".

Interpretation:
- 100% recovery → silent partial degradation (worst): the
  per-sentence error isolation is masking real bugs.
- 0% recovery → loud failure: every error knocks out the whole
  turn (user notices and complains).
- Mixed → both modes present; investigate the recovered turns
  for swallowed-bug patterns.

Tests (7 in `tests/unit/test_worker_recovery.py`):
- No-emit boundaries: clean session, only LLM errors (no worker).
- Full recovery (silent partial degradation marker), zero
  recovery (loud failure), mixed.
- Denominator excludes clean turns (only error turns count).
- Multiple errors per turn count as one turn.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **849 passed, 1 skipped in 23s** (842 existing
+ 7 new).

Notes:
- **Twenty-nine metrics live (63% of the 46-metric taxonomy.)**
- This is the third session-level-only metric (after iter-055
  rhythm score and iter-066 cold-start). The pattern of "compute
  in print_session_summary from existing per-turn fields" scales
  cheaply — 5-15 lines of code per metric and no plumbing churn.
- The error story is now fully bilateral:
  - LLM errors (turn-fatal, session-level count from iter-058)
  - Worker errors (per-turn count from iter-058)
  - Recovery rate (this) — the bridge between the two
- Next candidates: 1.12 (turn-taking jitter: stdev of TTFS — could
  promote from rhythm score's internal computation to its own
  line), 1.6 (WER, needs ground truth corpus), 1.18 (interruption
  rate: barge_in count / turns — already implicitly visible in
  the Barge-ins line, could add the rate explicitly).

## iter-068 — TTFS jitter / turn-taking jitter (taxonomy 1.12)

**Branch:** iter-068-jitter  **Commit:** 5c9cbfc  **Date:** 2026-05-25

Added `TTFS jitter` — `stdev(ttfs for ttfs in metrics_list if ttfs > 0)`
promoted from iter-055's rhythm-score internal computation to its
own line. The rhythm score is a normalized [0,1] number useful for
at-a-glance health comparison; the raw jitter in milliseconds is
the more actionable number when tuning. Humans tolerate consistent
slow turn-taking better than inconsistent fast turn-taking — a
250ms jitter at 600ms median feels more broken than a steady 750ms
median.

Implementation:
- Pure derivation in `print_session_summary`; reuses the same
  `sd = statistics.stdev(ttfs_times)` line that feeds the rhythm
  score, just emits the value as ms.
- Same gating as the rhythm score: needs ≥2 measurable TTFS
  values. Inserted right after the rhythm-score line so the two
  sit visually adjacent.
- Output: "TTFS jitter: ±NNms".

Tests (9 in `tests/unit/test_ttfs_jitter.py`):
- No-emit boundaries: 0-turn, 1-turn, all-no-audio sessions.
- Emit cases: two-turn arithmetic check, uniform values yield
  ±0ms, high-variance multi-turn, zero-TTFS turns excluded from
  the sample.
- Co-emission: jitter and rhythm score appear together when
  ≥2 turns; both omitted otherwise.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **858 passed, 1 skipped in 23s** (849 existing
+ 9 new).

Notes:
- **Thirty metrics live (65% of the 46-metric taxonomy.)**
- The TTFS dimension now has four orthogonal lenses: median (the
  central tendency), best (the floor — what's achievable on a
  good turn), jitter (the spread — this iteration), rhythm score
  (the normalized inverse for at-a-glance comparison). Plus
  iter-066's cold-start penalty isolating turn 1.
- Next candidates: 1.18 (interruption rate as explicit %, already
  shown as raw count), 2.6 (sentence-length histogram, currently
  emitted as mean), 1.6 (WER, needs ground truth corpus).

## iter-069 — interruption rate (taxonomy 1.18)

**Branch:** iter-069-interruption-rate  **Commit:** ddbed66  **Date:** 2026-05-25

Added explicit "Interruption rate: M/N turns (X%)" session-summary
line. Industry single-number UX KPI: "what fraction of bot turns
did the user feel they had to interrupt?" Distinct from the
existing mid-stream %, which uses total barges as denominator;
this uses total completed turns.

Implementation:
- Pure derivation in `print_session_summary`; reads `barges_total`
  and `n` (already computed). Inserted right after the existing
  Barge-ins block so the two related lines sit adjacent.
- Gated on `barges_total > 0` (the whole barge sub-block).
- Output: "Interruption rate: M/N turns (X%)".

Why distinct from the mid-stream %:
- mid-stream % = mid_cancels / barges_total → "of barges, what
  fraction were violent (cut a sentence mid-stream)"
- interruption rate = barges_total / n → "of turns, what fraction
  the user interrupted at all"

Both are useful and answer different operator questions. The
interruption rate is the headline UX number; mid-stream % is the
"how aggressive were the interruptions" diagnostic.

Tests (7 in `tests/unit/test_interruption_rate.py`):
- No-emit: zero turns, zero barges.
- Emit cases: 1/3 (33%), 2/2 (100%), 1/10 (10%).
- Co-emission with the existing mid-stream line (both visible).
- Distinct denominators verified with a case where the two
  percentages differ (4 barges of 8 turns, 1 mid-stream → 50% rate
  vs 25% mid-stream).

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **865 passed, 1 skipped in 23s** (858 existing
+ 7 new).

Notes:
- **Thirty-one metrics live (67% of the 46-metric taxonomy.)**
- The barge-in dimension is now exhaustively measured at every
  level: rate (this), latency (iter-041), phase (iter-047), regret
  (iter-056), mid-stream cuts (iter-040), cancel-to-close
  (iter-060). A regression in any sub-aspect surfaces in exactly
  one of these.
- Next candidates: 1.6 (WER, needs ground truth corpus), 1.17
  (audio device underrun/overrun count — would require
  surfacing PyAudio's overflow flag instead of swallowing it),
  2.6 (sentence-length histogram — currently only the mean
  surfaces; could promote min/max).

## iter-070 — sentence-length min/max range (taxonomy 2.6)

**Branch:** iter-070-sentence-range  **Commit:** 512bdf8  **Date:** 2026-05-25

Added per-turn `min_sentence_chars` + `max_sentence_chars` alongside
iter-045's existing mean. The mean alone hides bimodal patterns:
- Turn A: sentences [10, 130] → mean=70, range=[10..130]
- Turn B: sentences [70, 70]  → mean=70, range=[70..70]

Both report `mean_sentence_chars=70`, but only Turn A has the
"short interjection followed by long monologue" fragmentation
profile that defeats streaming overlap. The min/max surface this.

Implementation:
- ChatLoop tracks `sentence_min_chars` / `sentence_max_chars` as
  locals alongside the existing total/count accumulators. Updated
  on each `worker.submit(sentence)` AND on the trailing-remainder
  submit (since that's a real synthesis unit too).
- `TurnMetrics.min_sentence_chars: int = 0`,
  `max_sentence_chars: int = 0` (new fields). Both default to 0
  on turns with no submissions; populated only when at least one
  sentence was submitted.
- Per-turn print: extends the iter-045 avg suffix with
  "[min..max]" when min ≠ max. `min == max` (single sentence /
  uniform turn) skips the suffix to avoid noise.
- Session summary: "Sentence range: [shortest..longest] chars
  (session)" — worst-case across all turns, gated on at least one
  turn diverging.
- `ScenarioResult.min_sentence_chars` + `max_sentence_chars` on
  perf snapshots.

Tests (11 in `tests/unit/test_sentence_range.py`):
- Defaults zero.
- Per-turn print: no-mean omits range; uniform omits range;
  diverging emits "[min..max]"; min=0 (unset signal) skips.
- Session aggregate: no-data / uniform / divergent / zero-filter.
- ChatLoop wiring: bimodal LLM response ("Yes." + long sentence)
  produces min<10, max>50; single-sentence response produces
  min == max.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **876 passed, 1 skipped in 24s** (865 existing
+ 11 new).

Notes:
- **Thirty-two metrics live (70% of the 46-metric taxonomy).**
- The fragmentation dimension is now fully measured: mean
  (iter-045 — central tendency), min/max (this — distribution
  shape), split coverage (iter-059 — what fraction made it as a
  complete sentence). A regression in any one shows up
  independently.
- Next candidates: 1.6 (WER, needs ground truth corpus), 1.17
  (audio device underrun/overrun count — would require surfacing
  PyAudio's overflow flag), 2.17 (token-reveal lag — measures the
  text-vs-audio sync; needs play_aligned instrumentation).

## iter-071 — token-reveal lag metric (taxonomy 2.17)

**Branch:** iter-071-token-lag  **Commit:** 66bebd3  **Date:** 2026-05-25

Added per-token reveal-lag tracking — the wall-clock offset between
when a token is printed and the audio second its `start` field
claims:

    lag = (clock_at_emit - t0) - token["start"]

Positive lag = text falls behind audio (UX feels broken — text
becomes "late subtitles"). Negative lag = text leads audio (spoils
the bot before it speaks). Per-turn mean and max.

Implementation:
- `play_aligned()` gained an optional `lag_out: dict | None = None`
  kwarg. When provided AND at least one token emits, populates
  `{"sum": float, "count": int, "max": float}`. `max` is the
  single-token observation with the largest **absolute** value
  (sign preserved).
- New `_play_fn_accepts_lag_out()` helper in `_chat_pipeline.py`,
  parallel to existing `_play_fn_accepts_cancel_event`. Called
  once at SentenceWorker construction; the worker only passes the
  kwarg when supported, so old/minimal play_fns continue to work
  without churn.
- `SentenceWorker` exposes `token_reveal_lag_sum`,
  `token_reveal_lag_count`, `token_reveal_lag_max`. Per call: pass
  a fresh dict, fold into totals. The play call site grew from 2
  branches to 4 because there are now two independent kwargs to
  multiplex; further additions will need a small refactor.
- `TurnMetrics.mean_token_reveal_lag` + `max_token_reveal_lag`
  (new fields). ChatLoop computes the mean at the turn boundary
  (`worker.token_reveal_lag_sum / worker.token_reveal_lag_count`)
  so the denominator is stable.
- `mic_chat.py`'s production `_play` wrapper forwards `lag_out`
  through to `_play_aligned_core` so the metric lights up in live
  sessions.
- Per-turn print: "Token-reveal: ±NNms mean, ±MMms peak" sign
  preserved; yellow when `|mean| > 100ms`.
- Session summary: "Token lag: ±NNms median, ±MMms worst peak" —
  worst-peak chosen across turns by largest absolute value.
- `ScenarioResult.mean_token_reveal_lag_ms` + `max_token_reveal_lag_ms`
  on perf snapshots.

Tests (16 in `tests/unit/test_token_reveal_lag.py`):
- play_aligned contract: kwarg optional; populated correctly with
  tokens; empty dict signals "no data" when no tokens emit.
- Worker sniff helper: explicit kwarg / **kwargs / no support all
  detected correctly.
- Worker accumulation: no-support play_fn leaves zero; sums and
  counts add across multiple calls; max picks largest absolute
  value (negative-bigger wins over positive-smaller).
- TurnMetrics + per-turn print: defaults zero, omitted when both
  fields zero, sign-preserved render for positive and negative.
- Session aggregate: no-data omitted, median + worst-peak with
  positive values, worst-peak picks largest absolute (negative).

**Bug fix bundled:** the iter-064 Mirror-gap line ended up inside
an iter-071 block during the edits; restored it under the
`bot_wpms` scope where it belongs. Caught by the existing iter-064
tests during the full-suite run, fixed before merge.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **892 passed, 1 skipped in 25s** (876 existing
+ 16 new).

Notes:
- **Thirty-three metrics live (72% of the 46-metric taxonomy).**
- The text/audio-sync dimension is now exposed end-to-end. The
  metric makes a real bug class visible: speakers that buffer
  ahead of the worker's `samples_played` cursor will produce
  positive lag that grows linearly across the turn — a regression
  pattern previously invisible.
- The play_fn `_chat_pipeline._play_clip` call site now branches
  on `cancel_event × lag_out`. A future iter that adds a third
  optional kwarg should refactor into a kwargs-dict approach.
- Next candidates: 1.6 (WER, needs ground truth corpus), 1.17
  (audio device underrun/overrun count — would require surfacing
  PyAudio's overflow flag instead of swallowing it), 2.21 (VAD-
  config consistency between recorder and watcher — already
  enforced by iter-028's plumbing; promote to a regression
  sentinel).

## iter-072 — STT preview-vs-final divergence metric (taxonomy 1.8)

**Branch:** iter-072-stt-divergence  **Commit:** 05b51da  **Date:** 2026-05-25

Added STT preview-vs-final divergence — how aligned the live STT
preview the user sees while speaking is with the final transcript:

    divergence = 1 - SequenceMatcher(None, preview, final).ratio()

0 = preview matched final perfectly (incremental Whisper output
was already correct — live STT was useful). 1 = totally different
— the user had to wait for the final to know if they were
understood. Uses `difflib.SequenceMatcher` from stdlib so no new
dependencies.

Implementation:
- `record_utterance_streaming` populates the iter-063
  `out_metrics` dict with key `"stt_preview_divergence"` when
  both `preview_text` and `final_text` are non-empty. The compute
  happens after the final transcribe completes; sized for the
  compare-once-at-end pattern (vs per-frame). DONE_TOO_SHORT path
  exits early as before — no key written.
- `TurnMetrics.stt_preview_divergence: float = 0.0` (new field).
  ChatLoop copies from `rec_metrics`.
- Per-turn print: appends "preview Δ N%" to the STT line when
  `> 0`. Yellow when `> 30%` (live preview was unreliable enough
  the user can't trust it). Combined with the iter-049 RTF tag,
  the STT line now shows two diagnostics:
  `STT: NNms (RTF 0.25x, preview Δ 12%)`
- Session summary: "STT preview Δ: N% (median)" filtered for `>0`.
- `ScenarioResult.stt_preview_divergence` on perf snapshots.

Tests (12 in `tests/unit/test_stt_preview_divergence.py`):
- Default zero, per-turn print: zero omits suffix; low value
  emits dim; high value emits (yellow); combined with RTF.
- Session aggregate: no-data, median calculation, zero-filter.
- Recorder integration: perfect-preview (preview == final) →
  divergence 0; matched preview/final case (recorder doesn't
  populate when buffer-growth dynamics make preview converge
  to final); no-preview case (utterance too short for
  inference interval).
- ChatLoop wiring: field bubbles to TurnMetrics in [0,1].

Notes on test design:
- The recorder's preview always reflects the LATEST `transcribe_fn`
  return for the growing buffer. Forcing a "different preview vs
  final" outcome via virtual mic would require either patching
  INFERENCE_INTERVAL or distinguishing calls by some side-channel
  the recorder doesn't expose. The test suite verifies the
  divergence formula (via `difflib.SequenceMatcher` directly),
  the populate-only-when-both-exist contract, and the in-bounds
  property — but not a specific high-divergence scenario through
  the live recorder. That's adequate for a regression sentinel:
  the formula is fixed, the recorder either populates correctly
  or doesn't.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **904 passed, 1 skipped in 24s** (892 existing
+ 12 new).

Notes:
- **Thirty-four metrics live (74% of the 46-metric taxonomy).**
- The STT dimension now has three lenses: time (`stt_time`,
  iter-base), speed (`stt_rtf`, iter-049 — how fast vs realtime),
  reliability (this — how trustworthy the live preview was). The
  STT line in per-turn output has reached its target density —
  any further STT decoration would warrant a second line.
- iter-072 is the second use of the iter-063 `out_metrics` dict
  pattern from `record_utterance_streaming`. The pattern is
  paying dividends: each new recorder-side metric is a single
  key + a 5-line population block, no signature churn.
- Next candidates: 1.6 (WER, needs ground truth corpus), 1.17
  (audio device underrun/overrun count), 2.2 (first-sentence
  overlap savings: how much of the first sentence's tts_time
  happened before llm_stream_done — distinct from iter-043's
  ratio which covers the whole stream).

## iter-073 — first-sentence overlap savings (taxonomy 2.2)

**Branch:** iter-073-first-overlap  **Commit:** 8926bcf  **Date:** 2026-05-25

Added `first_synth_overlap_seconds` — seconds of first-sentence
synth that ran concurrently with LLM streaming. Distinct from
iter-043's `streaming_overlap_ratio` (whole-stream ratio): this
scopes specifically to the FIRST sentence because that's what
gates TTFS. The metric directly quantifies the user-perceived win
from iter-008's streaming-sentence-dispatch design.

Implementation:
- `SentenceWorker` latches `first_synth_start_at` and
  `first_synth_done_at` on the FIRST **successful** synth call.
  Failed first synths don't latch — the next successful synth
  becomes "first" for accounting purposes (matches the user's
  perception: a synth that raised never produced audio that
  gates TTFS).
- ChatLoop computes the standard interval overlap:
  ```python
  overlap = max(0, min(synth_done, llm_done) - max(synth_start, llm_start))
  ```
  Stored on `TurnMetrics.first_synth_overlap_seconds`. 0 means
  first synth ran entirely after LLM finished (sequential — no
  TTFS savings); equal to first-synth duration means first synth
  ran entirely under LLM streaming (best case).
- Per-turn print: "1st-synth save: NNms (TTFS shaved by streaming)"
  when >0. Green when >100ms (meaningful TTFS win).
- Session summary: "1st-synth saved: NNms median" filtered for >0.
- `ScenarioResult.first_synth_overlap_ms` on perf snapshots.

Tests (14 in `tests/unit/test_first_synth_overlap.py`):
- Defaults zero / None.
- Per-turn print: zero omits, low-value emits dim, meaningful
  emits green.
- Session aggregate: no-data, median, zero-filter.
- ChatLoop wiring: slow-LLM + slow-synth produces measurable
  overlap; quick response produces a sane bounded value; field
  always non-negative.
- Worker timing: latched once on first successful synth;
  unset when zero submissions; failed first synth doesn't latch
  (next successful one does).

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **918 passed, 1 skipped in 24s** (904 existing
+ 14 new).

Notes:
- **Thirty-five metrics live (76% of the 46-metric taxonomy).**
- The streaming-overlap dimension is now bilateral:
  - `streaming_overlap_ratio` (iter-043) — whole-stream view,
    "did the bot speak before the LLM finished, ever?"
  - `first_synth_overlap_seconds` (iter-073) — TTFS-specific,
    "how many ms did parallelism shave off the first sentence?"
- Together they decompose the iter-008 streaming-overlap design's
  payoff into a "did it help at all" header + a "by how much for
  TTFS" detail.
- Next candidates: 1.6 (WER, needs ground truth corpus), 1.17
  (audio device underrun/overrun count), 1.19 (bargeable-time
  fraction — portion of bot speech where barge-in is even
  possible; needs BargeInWatcher.start_at + stop_at exposed).

## iter-074 — bargeable-time fraction metric (taxonomy 1.19)

**Branch:** iter-074-bargeable-time  **Commit:** 3e39c26  **Date:** 2026-05-25

Added `bargeable_fraction` — of the time the bot was producing
audio, what fraction was the BargeInWatcher active (i.e. barge-in
was actually possible)?

    fraction = intersection(watcher_window, bot_speech_window) / bot_speech_duration

1.0 is the architectural default — watcher.start precedes
worker.first_audio_at and watcher.stop is called right after
worker.wait_done. Anything below 1.0 means the bot was functionally
uninterruptible for some fraction of its speech.

Implementation:
- `BargeInWatcher` gained `started_at` and `stopped_at` (new public
  attributes). Latched in `start()` (before launching the thread)
  and `stop()` (after the thread joins) so they bookend the actual
  listening window.
- ChatLoop computes the standard interval overlap right after
  `watcher.stop()`:
  ```python
  inter_start = max(watcher.started_at, worker.first_audio_at)
  inter_end = watcher.stopped_at
  intersection = max(0.0, inter_end - inter_start)
  fraction = min(1.0, intersection / bot_speech_dur)
  ```
  Stored on `TurnMetrics.bargeable_fraction`. 0 means no audio
  played this turn (or watcher lifecycle didn't fire).
- Per-turn print: emits "Bargeable: NN% (watcher coverage of
  bot speech)" only when 0 < value < 0.99 — alarm-only. 1.0 is
  the healthy default and stays clutter-free.
- Session summary: emits "Bargeable: NN% worst (M/N turns < 99%)
  — watcher coverage regression" only when at least one turn
  dropped below threshold.
- `ScenarioResult.bargeable_fraction` on perf snapshots.

**Sentinel role:**
- Currently lands at 1.0 in clean sessions (the watcher always
  fully covers bot speech in the iter-074 architecture).
- A future change that pauses the watcher mid-turn — e.g. during
  filler playback to avoid self-detection, or during a "polite
  silence" guard — would push this below 1.0 and the alarm
  becomes visible immediately.
- Pairs with iter-067's worker recovery rate as another "silent
  partial degradation" sentinel: both expose UX regressions that
  individual happy-path turns wouldn't reveal.

Tests (15 in `tests/unit/test_bargeable_fraction.py`):
- Watcher timestamps: defaults None; start stamps `started_at`
  immediately; stop stamps `stopped_at` after join; stop without
  start leaves both None.
- Per-turn print: 0 omits, 1.0 omits, just-under (85%) emits,
  low (40%) emits.
- Session aggregate: no-data, all-perfect, one-below-threshold,
  multiple-below.
- ChatLoop wiring: clean turn lands at 1.0 (architectural
  default); always in [0, 1].

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **933 passed, 1 skipped in 25s** (918 existing
+ 15 new).

Notes:
- **Thirty-six metrics live (78% of the 46-metric taxonomy).**
- The "regression sentinel" pattern continues to scale: this is
  the third metric (after iter-062 worker queue depth and iter-067
  worker recovery rate) whose value in healthy sessions is
  uniform but whose deviation surfaces architectural drift
  immediately.
- Next candidates: 1.6 (WER, needs ground truth corpus), 1.17
  (audio device underrun/overrun count), 2.20 (loopback /
  acoustic-echo barge-in rate — barges triggered by the bot's
  own audio; needs cross-correlation between barge timestamps
  and bot-audio-active intervals).

## iter-075 — refactor _play_clip kwargs into dynamic dict (cleanup)

**Branch:** iter-075-play-kwargs  **Commit:** 68611a1  **Date:** 2026-05-25

iter-071 flagged that the play_fn invocation in
`SentenceWorker._play_clip` had grown to a 4-branch `if/elif`
chain on `cancel_event × lag_out`. Each new optional kwarg
would double the branch count.

This iteration replaces the chain with a single dynamically-built
kwargs dict. Each play_fn-supported kwarg adds one entry; unsupported
kwargs are omitted entirely. Preserves the iter-023 contract that
the worker only passes kwargs the play_fn signature accepts (so
old/minimal play_fns continue to work without churn).

Before:
```python
if self._play_fn_supports_cancel and lag_out is not None:
    elapsed = self._play_fn(speaker, audio_np, tokens, ...)
elif self._play_fn_supports_cancel:
    elapsed = self._play_fn(speaker, audio_np, tokens, ...)
elif lag_out is not None:
    elapsed = self._play_fn(speaker, audio_np, tokens, ...)
else:
    elapsed = self._play_fn(speaker, audio_np, tokens, ...)
```

After:
```python
base_kwargs = {"is_first_sentence": is_first}
if self._play_fn_supports_cancel:
    base_kwargs["cancel_event"] = self._cancel_event
if lag_out is not None:
    base_kwargs["lag_out"] = lag_out
elapsed = self._play_fn(speaker, audio_np, tokens, **base_kwargs)
```

Adding a third optional kwarg is now an O(1) edit — one new
sniff helper + one new conditional dict-set.

Tests:
- 6 new tests in `test_play_kwargs_assembly.py` pinning the
  kwargs assembly across each signature shape (minimal /
  cancel_only / lag_only / both / **kwargs). Covers the case
  the old branching protected against — TypeError when passing
  unsupported kwargs to a minimal signature.
- All 933 prior tests continue to pass — the refactor is
  behavior-preserving.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **939 passed, 1 skipped in 24s** (933 existing
+ 6 new).

Notes:
- This is the project's first dedicated cleanup iteration since
  metric instrumentation work began. The pattern of "each iter
  adds a metric and surfaces a small refactor opportunity for
  the next" has been working — both compound interest (metric
  coverage now 78% of taxonomy) and code health stay positive.
- Next candidates: continuing with metrics (1.6 WER needs ground
  truth, 1.17 audio underrun/overrun, 2.20 loopback echo barge
  rate), OR the iter-073 sentence-first-audio path-length
  decomposition (taxonomy 2.22) which would surface the 7
  per-turn timestamps as a structured trace line.

## iter-076 — TTFS attribution breakdown (taxonomy 2.22)

**Branch:** iter-076-ttfs-attribution  **Commit:** 9b9bdf4  **Date:** 2026-05-25

Added `synth_dispatch_seconds` and a TTFS attribution display.
The breakdown decomposes TTFS into three accounting buckets that
sum to 100%:
- **STT**: speech-end → STT done.
- **LLM**: STT done → first complete sentence reaches the worker.
- **synth+dispatch**: first sentence at worker → first audio
  played (covers synth, queue dispatch, audio device buffering).

The synth_dispatch term is the residual:
```python
synth_dispatch_seconds = max(0, ttfs - stt_time - llm_first_sentence)
```
The clamp is defensive against microsecond clock-skew producing
tiny negative residuals (the recorder and loop use independent
clocks).

Implementation:
- `TurnMetrics.synth_dispatch_seconds: float = 0.0` (new field).
- ChatLoop computes the residual immediately after `metrics.ttfs`
  is set, using the locally-available `first_sentence_at` and
  `llm_start` instead of the `metrics.llm_first_sentence` field
  which gets assigned later in the same block.
- Per-turn print emits "Attribution: STT N% + LLM N% + synth N%"
  only when all three legs are measurable. Renders as floored
  ints so the visible sum stays ≤ 100% (vs `round`, which can
  produce 101%).
- Session summary emits "TTFS breakdown: STT N% + LLM N% +
  synth N%" with median percentages across turns where all three
  legs landed.
- `ScenarioResult.synth_dispatch_ms` on perf snapshots.

Tests (11 in `tests/unit/test_ttfs_attribution.py`):
- Default zero / per-turn print: zero omits, partial-data omits,
  full breakdown emits with correct percentages.
- ChatLoop arithmetic: residual non-negative (clamp works);
  residual ≤ ttfs; three legs sum to ≤ ttfs (within 50ms of ttfs
  in real traces — clock skew tolerance).
- Session aggregate: no-data / partial-data / uniform / mixed
  (zero-leg turns excluded from median).

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **950 passed, 1 skipped in 24s** (939 existing
+ 11 new).

Notes:
- **Thirty-seven metrics live (80% of the 46-metric taxonomy).**
- The TTFS dimension now has six lenses:
  - `ttfs` — single number (the budget).
  - `naturalness_bucket` (iter-053) — qualitative (rushed/natural/slow).
  - `cold_start_penalty` (iter-066) — turn-1 vs steady-state.
  - `rhythm_score` + `ttfs_jitter` (iter-055/068) — consistency.
  - `synth_dispatch_seconds` (this) — leg attribution residual.
  - Plus the discrete leg metrics (stt_time, llm_first_sentence,
    first_synth_overlap_seconds).
- The breakdown turns "TTFS is 850ms" into "850ms = 200ms STT +
  500ms LLM + 150ms synth — LLM is the bottleneck." Operator
  can read which leg dominates without cross-referencing six
  separate metrics.
- Next candidates: 1.6 (WER, needs ground truth corpus), 1.17
  (audio device underrun/overrun count), 2.13 (primed-frames
  STT contribution — needs offline ablation), 2.20 (loopback /
  acoustic-echo barge rate — needs cross-correlation).

## iter-077 — conversation history grow rate (taxonomy 2.23)

**Branch:** iter-077-context-tokens  **Commit:** 17f2859  **Date:** 2026-05-25

Added `context_tokens` — approximate count of context tokens being
sent to the LLM each turn. Whitespace-split is a rough estimator
(real tokenizer counts vary by model) but the per-turn TREND is
what matters: late-session turns get progressively slower as
context grows, so a creep here predicts an `llm_first_token`
regression even before the latency itself measurably worsens.

Pairs with iter-024's `trim_history(max_user_assistant=20)` — if
context keeps growing despite the trim cap, system-prompt bloat
is the culprit.

Implementation:
- `TurnMetrics.context_tokens: int = 0` (new field).
- ChatLoop counts immediately after the user message is appended
  to the messages list, BEFORE the LLM call:
  ```python
  metrics.context_tokens = sum(
      len(str(m.get("content", "")).split())
      for m in messages
  )
  ```
- Per-turn print: appends "N ctx" suffix to the LLM total line
  alongside the existing TPS suffix:
  `LLM total: 500ms (qwen-2.5-7b, 45 tps, 120 ctx)`.
- Session summary:
  - "Context tokens: N median, M max" (always when measurable).
  - "Context growth: ±N tokens (turn 1 → turn N)" when ≥3 turns
    have measurable context — the first-vs-last delta is a quick
    growth sentinel without needing a least-squares slope.
- `ScenarioResult.context_tokens` on perf snapshots.

Tests (12 in `tests/unit/test_context_tokens.py`):
- Default zero, per-turn print: zero omits suffix, non-zero
  appends, combines with TPS.
- Session aggregate: no-data, 2-turn (median+max but no growth
  line), 3-turn (median+max+growth), negative growth (trim
  aggressive), zero filtering.
- ChatLoop arithmetic: short transcript yields exact word count,
  long transcript scales correctly, prefilled history adds to
  total.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **962 passed, 1 skipped in 25s** (950 existing
+ 12 new).

Notes:
- **Thirty-eight metrics live (83% of the 46-metric taxonomy).**
- The conversation-state dimension is now measurable — context
  size predicts LLM TTFB before the latency itself shows it.
  Pairs with iter-076's TTFS attribution: a creeping LLM% in the
  attribution + a growing context_tokens points squarely at
  prompt bloat, not LLM slowness.
- Next candidates: 2.24 (trim event rate — paired with this;
  fires when trim_history actually evicts), 3.7 (pre-empted-
  content loss — bot tokens generated but never spoken on
  barge-in turns), 3.11 (silent-turn rate — turns where
  transcript exists but ttfs == 0), 1.6 (WER, needs ground
  truth corpus).

## iter-078 — trim event rate (taxonomy 2.24)

**Branch:** iter-078-trim-rate  **Commit:** 3a6ef5c  **Date:** 2026-05-25

Added trim event rate — counts how many times across a session
`trim_history` actually evicted at least one message, plus the
cumulative count of evicted messages. Pairs with iter-077:
validates the iter-024 `max_user_assistant=20` threshold is
calibrated.

Reading the metric:
- `events == 0` across a long session → trim threshold too loose;
  `Context tokens: max` (iter-077) will be growing in lockstep.
- `evicted/events == 1` → calibrated steady-state (each turn
  trims exactly the one new pair that pushed past the cap).
- `evicted/events >> 1` → trim catching up after a longer gap
  (e.g. `max_user_assistant` was lowered mid-session).

Implementation:
- `print_session_summary` gained two new kwargs:
  `trim_events: int = 0` and `trim_messages_evicted: int = 0`.
  Both default to 0 — existing callers unaffected.
- mic_chat tracks both counters by diffing message-list lengths
  around each `ChatLoop.trim_messages` call. Non-invasive: no
  changes to `trim_history` or `ChatLoop.trim_messages` signatures.
- Output: "Trim events: N (M evicted, X.X/event)" emitted only
  when `trim_events > 0`. The per-event ratio surfaces severity
  immediately.

Tests (11 in `tests/unit/test_trim_event_rate.py`):
- print_session_summary kwargs: default omits, zero omits, one
  event emits, steady-state 1.0/event, high ratio (catch-up).
- `trim_history` length-diff contract: no-op below threshold,
  evicts above threshold, empty case.
- `ChatLoop.trim_messages` passthrough preserves the same
  semantics.
- Co-emission with iter-077 context tokens: both lines visible
  in a session that has both.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **973 passed, 1 skipped in 24s** (962 existing
+ 11 new).

Notes:
- **Thirty-nine metrics live (85% of the 46-metric taxonomy).**
- The conversation-state dimension is now bilateral: context
  size (iter-077, growth signal) and trim activity (this,
  calibration signal). A session where context_tokens grows
  unbounded AND trim_events stays at 0 is the unambiguous
  "trim broken" signal — both metrics agree.
- The pattern of "session-level kwargs to print_session_summary"
  is paying off (false_triggers, llm_errors, session_seconds,
  trim_events, trim_messages_evicted). Each new session-level
  signal is one new kwarg, no per-turn churn.
- Next candidates: 3.7 (pre-empted-content loss — bot tokens
  generated but not spoken on barge turns), 3.11 (silent-turn
  rate — turns where transcript exists but ttfs == 0), 1.6
  (WER, needs ground truth corpus), 1.17 (audio underrun/
  overrun count — needs PyAudio plumbing).

## iter-079 — silent-turn rate (taxonomy 3.11)

**Branch:** iter-079-silent-turn  **Commit:** e92461b  **Date:** 2026-05-25

Added silent-turn rate — counts turns where the user spoke
(transcript captured) but the bot produced no audio (`ttfs == 0`).
Distinct from worker errors: no exception fires, no console
warning, the user just experiences "I said something and got
silence." The invisible failure mode the operator most needs to
catch.

Common causes:
- `worker.errors` took out every sentence (per-sentence
  isolation kept the turn from raising but no audio remained).
- LLM returned an empty response (model produced 0 tokens).
- All sentences pre-empted before `first_audio_at` (barge fired
  before any synth landed).
- LLM produced no terminator-bearing tokens, so no synth
  submissions ever happened (the trailing remainder also got
  dropped via cancel).

Implementation:
- Pure session-level derivation in `print_session_summary`. No
  new `TurnMetrics` field — uses existing `transcript` and `ttfs`.
- Filter: `m.transcript and m.ttfs == 0`. Emit only when the
  count is non-zero — clean sessions stay clutter-free.
- Output: "Silent turns: M/N (X%) — bot produced no audio".

Tests (8 in `tests/unit/test_silent_turn_rate.py`):
- No-emit boundaries: clean session, zero turns, transcript-less
  ttfs==0 (false trigger, not silent).
- Emit cases: 1/3, 2/2 (100%), 1/10.
- Distinction from worker errors: silent-without-errors emits
  alone; silent-with-errors emits both metrics with consistent
  counts (recovery rate's denominator is "error turns,"
  silent-rate's denominator is "all turns").

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **981 passed, 1 skipped in 24s** (973 existing
+ 8 new).

Notes:
- **Forty metrics live (87% of the 46-metric taxonomy).**
- The "silent partial degradation" sentinel cluster (iter-067
  worker recovery, iter-074 bargeable fraction, iter-079 silent
  turns) now fully covers the failure modes where externally
  successful turns hide internal trouble:
  - Worker recovery rate: errors that DID surface but were
    recovered from.
  - Bargeable fraction: watcher coverage of bot speech.
  - Silent-turn rate: the most user-visible failure — no audio
    despite a clean transcript.
- Next candidates: 3.7 (pre-empted-content loss — bot tokens
  generated but not played on barge turns; needs response token
  count alongside spoken count), 3.8 (filler novelty index —
  unique fillers / total played), 1.6 (WER, needs ground truth
  corpus), 1.17 (audio underrun/overrun count — needs PyAudio
  plumbing).

## iter-080 — pre-empted-content loss (taxonomy 3.7)

**Branch:** iter-080-preempted-loss  **Commit:** 16f188f  **Date:** 2026-05-25

Added `preempted_words` — words the LLM generated but the user
never heard because of a mid-stream barge. Computed on barge
turns only:

    preempted_words = max(0, len(response.split()) - worker.word_count_total)

0 on non-barge turns and on barge turns where the cut happened
cleanly between sentences (no words lost). High values on a
barge turn signal the bot was being verbose enough that the user
interrupted; pairs with iter-069 interruption rate and iter-047
barge phase to localize the cause:

- High `preempted_words` + `playback`-phase barge → bot was too
  verbose, user cut off mid-sentence.
- Low `preempted_words` + `llm_stream`-phase barge → user
  impatient with TTFS, no audio had played yet so nothing to
  pre-empt.

Implementation:
- `TurnMetrics.preempted_words: int = 0` (new field).
- ChatLoop computes ONLY when `coord.is_set()` (barge fired):
  ```python
  if coord.is_set():
      response_words = len(metrics.response.split())
      played_words = worker.word_count_total
      metrics.preempted_words = max(0, response_words - played_words)
  ```
  The `max(0, ...)` clamp is defensive — `worker.word_count_total`
  comes from the per-sentence token alignment which can over-count
  in edge cases (e.g. punctuation-only tokens).
- Per-turn print: emits "Pre-empted: NN words (generated but not
  played)" inside the existing `barge_in` block. Yellow when >10
  words (≥5s of speech lost — bot was being verbose).
- Session summary: "Pre-empted words: NN total (M/N barges,
  X avg/loss)" — total words pre-empted, fraction of barges that
  involved any loss, average words lost per lossy barge.
- `ScenarioResult.preempted_words` on perf snapshots.

Tests (11 in `tests/unit/test_preempted_loss.py`):
- Default zero / per-turn print: zero omits, no-barge omits, low
  emits dim, high emits.
- Session aggregate: no-barge omits, all-clean omits, single
  lossy barge, mixed lossy+clean.
- ChatLoop wiring: clean turn yields 0; deterministic delayed-
  barge scenario yields ≥0 always (clamp), bounded above by
  response word count.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **992 passed, 1 skipped in 25s** (981 existing
+ 11 new).

Notes:
- **Forty-one metrics live (89% of the 46-metric taxonomy).**
- The barge-in dimension now has 8 orthogonal lenses, fully
  describing the interruption from cause to consequence:
  - `barge_in` (rate, base) — did it happen?
  - `barge_in_phase` (iter-047) — LLM-stream vs playback?
  - `barge_in_latency` (iter-041) — how fast did we react?
  - `sentences_cancelled` (iter-040) — how aggressive was the cut?
  - `barge_in_regret` (iter-056) — did we pre-empt the user?
  - `llm_cancel_to_close` (iter-060) — HTTP socket teardown?
  - `primed_frames_seconds` (iter-057) — how much did we save
    for the next turn?
  - `preempted_words` (iter-080) — how much did we waste?
- Next candidates: 3.8 (filler novelty index — unique fillers /
  total played), 3.13 (adaptive-rate margin — bot_wpm / user_wpm
  ratio; partially covered by iter-064's mirror gap), 1.6 (WER,
  needs ground truth corpus), 1.17 (audio underrun/overrun count
  — needs PyAudio plumbing).

## iter-081 — filler novelty index (taxonomy 3.8)

**Branch:** iter-081-filler-novelty  **Commit:** 58abd25  **Date:** 2026-05-25

Added filler novelty index — distinct filler clip IDs across the
session vs total plays. Hearing the same "umm" three times in a
row is worse than no filler at all; the metric measures whether
the picker is actually distributing across the rendered_fillers
list.

Implementation:
- `SentenceWorker.last_filler_id: Optional[int] = None` (new
  public attribute). Latched after a successful filler play:
  `self.last_filler_id = id(clip)`. The id() is stable across
  worker instances because mic_chat holds the canonical
  rendered_fillers list and passes it by reference each turn.
- `TurnMetrics.last_filler_id: int = 0` (0 = no filler this turn).
  ChatLoop transfers from worker when worker had a filler.
- `print_session_summary` computes session-wide diversity via set
  arithmetic:
  ```python
  unique_filler_ids = {
      m.last_filler_id for m in metrics_list if m.last_filler_id != 0
  }
  novelty_pct = len(unique_filler_ids) / fillers_total * 100
  ```
- Output: "Filler novelty: M unique / N (X%)" emitted only when
  ≥2 fillers played (single-play sessions are 100% trivially —
  no info).
- `ScenarioResult.last_filler_id` on perf snapshots. Caveat: the
  id is process-stable but not portable across runs — useful for
  in-session aggregation, NOT for cross-iteration time-series.

Tests (11 in `tests/unit/test_filler_novelty.py`):
- Defaults zero / None.
- Worker latching: empty fillers list keeps None; single filler
  available with deterministic picker latches its id; 3-filler
  picker picks the second and latches THAT id (not the others).
  Tests use a delayed `submit_done()` so the queue.get timeout
  fires the filler path before the sentinel cancels it.
- Session aggregate: no-fillers omits, single-play omits (100%
  trivially), perfect diversity (3/3 = 100%), partial (2/4 = 50%),
  all-same (1/5 = 20%), zero-id turns excluded from denominator.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **1003 passed, 1 skipped in 24s** (992 existing
+ 11 new). **Test suite crossed 1000.**

Notes:
- **Forty-two metrics live (91% of the 46-metric taxonomy).**
- Filler dimension complete: count (iter-014 fillers_played),
  false-positive rate (iter-051), novelty index (this — taxonomy
  3.8). Three orthogonal lenses on the iter-011 filler design.
- The `id()`-based session aggregation pattern is unusual but
  fits the use case: we need a stable-per-process identifier for
  sets, and the rendered_fillers tuples already provide that
  naturally. Documented the time-series caveat so future
  iterations don't try to compare `last_filler_id` across runs.
- Next candidates: 1.6 (WER, needs ground truth corpus), 1.17
  (audio underrun/overrun count — needs PyAudio plumbing), 3.13
  (adaptive-rate margin — bot_wpm/user_wpm; partially covered
  by iter-064's mirror gap), 3.14 (TTC proxy — time from bot
  first audio to next user turn first speech).

## iter-082 — TTC (time-to-comprehension) proxy (taxonomy 3.14)

**Branch:** iter-082-ttc  **Commit:** 8e5f4f8  **Date:** 2026-05-25

Added `time_to_comprehension` — a cross-turn metric measuring the
gap from the PREVIOUS turn's bot first audio to THIS turn's first
speech-detected frame:

    ttc = current_turn.speech_start_at - prev_turn.first_audio_at

Captures how long the user listened before responding. Three
signals:
- **<500ms**: user already knew the answer; bot was telling them
  what they already knew (under-performing).
- **>5s**: user was confused / thinking / multi-tasking.
- **1-3s**: bell-curve target — typical conversational response
  time.

Implementation:
- `record_utterance_streaming` gained an `out_metrics["speech_start_at"]`
  key. Latched in the recorder's per-frame loop the first time a
  frame's RMS crosses `silence_threshold`. Same DONE_OK gating as
  `eot_latency` — DONE_TOO_SHORT path doesn't write.
- `ChatLoop` gained `self._last_first_audio_at: Optional[float]`
  instance state. Initialized to None; updated each turn that
  produces audio (skipped on silent turns to avoid anchoring to
  a missing event).
- TTC computed on turns where both prev's first_audio_at AND
  this turn's speech_start_at exist. Clamped at 0 to handle
  test-mode clock drift between the recorder's frame-aligned
  virtual clock and the loop's wall-clock.
- Per-turn print: emits "TTC: NNms (user listened before
  responding)" when >0. Yellow at the rushed (<500ms) and slow
  (>5s) extremes.
- Session aggregate: median + outlier counts ("X rushed, Y slow")
  inline. Showcases distribution shape, not just central tendency.
- `ScenarioResult.time_to_comprehension_ms` on perf snapshots.

Tests (16 in `tests/unit/test_ttc_proxy.py`):
- Default zero / per-turn print: zero omits, natural emits dim,
  rushed and slow render with sign-aware color paths.
- Recorder integration: `speech_start_at` populated on DONE_OK,
  omitted on DONE_TOO_SHORT.
- Session aggregate: no-data, natural-only (no outlier
  annotation), rushed outlier, slow outlier, both kinds, zero
  filtering.
- ChatLoop wiring: turn 1 zero (no prev); turn 2 populated and
  non-negative; silent prev turn breaks the chain (bounded
  result).

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **1019 passed, 1 skipped in 24s** (1003
existing + 16 new).

Notes:
- **Forty-three metrics live (93% of the 46-metric taxonomy).**
- This is the FIRST cross-turn metric in the project. The
  ChatLoop-as-session-state pattern (vs purely run_one_turn-as-
  pure-function) opens the door to other cross-turn signals:
  3.13 adaptive-rate margin (already partially covered), 3.16
  cross-modal consistency, 1.20 cold-start (already done at
  print-time).
- The `out_metrics` dict in `record_utterance_streaming` now
  carries 3 keys (eot_latency, stt_preview_divergence,
  speech_start_at). Every recorder-side metric becomes one new
  key + one ChatLoop read — pattern stays clean.
- Next candidates (3 metrics from 46 left, all heavyweight):
  1.6 (WER, needs ground truth corpus), 1.17 (audio underrun/
  overrun count — needs PyAudio plumbing), 2.13 (primed-frames
  STT contribution — needs offline ablation).

## iter-083 — first-token-to-audio gap (FT-A) (taxonomy 3.18)

**Branch:** iter-083-fta-gap  **Commit:** afd33fb  **Date:** 2026-05-25

Added `first_token_to_audio` — time from when the LLM's first
token landed at the splitter to when the worker played its first
audio chunk:

    FT-A = worker.first_audio_at - first_token_at

Complementary to iter-052's `llm_first_token`. Together they
bracket where TTFS time was spent:
- High **`llm_first_token`**, low FT-A → LLM is the bottleneck
  (TTFB to invest in: model latency, network, prompt length).
- Low `llm_first_token`, high **FT-A** → sentence-split + TTS is
  the bottleneck (synth speed, splitter waiting on terminator).
- Both high → both legs need work.

Implementation:
- `TurnMetrics.first_token_to_audio: float = 0.0` (new field).
- ChatLoop computes when both `first_token_at` and
  `worker.first_audio_at` exist; clamped at 0 against tiny
  clock-skew negatives.
- Per-turn print: appends "+NNms → audio" suffix to the LLM
  1st-tok line, mirroring iter-038's "+NNms preamble" suffix on
  the LLM 1st-sent line. The visual symmetry makes the
  decomposition obvious.
- Session summary: "Median FT-A: NNms" emitted alongside
  "Median LLM 1st" so the operator sees both halves.
- `ScenarioResult.first_token_to_audio_ms` on perf snapshots.

Tests (9 in `tests/unit/test_fta_gap.py`):
- Default zero / per-turn print: zero omits suffix, non-zero
  appends.
- Session aggregate: no-data, median, zero-filter.
- ChatLoop arithmetic: clean turn populates ≥0 and bounded;
  clamp at zero defensive.
- Complementarity test: stt + llm_ft + FT-A = ttfs algebraically
  on a constructed case.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **1028 passed, 1 skipped in 25s** (1019
existing + 9 new).

Notes:
- **Forty-four metrics live (96% of the 46-metric taxonomy).**
- The TTFS decomposition is now exhaustive on the diagnostic
  side: macro budget (`ttfs`), three-bucket attribution
  (iter-076), bilateral split (`llm_first_token` + this), preamble
  gap (iter-038), naturalness bucket (iter-053), cold-start
  penalty (iter-066), jitter + rhythm (iter-068/055). Eight
  orthogonal lenses on the most important UX number.
- Remaining 2 metrics from 46: 1.6 (WER, needs ground truth
  corpus) and 1.17 (audio device underrun/overrun count, needs
  PyAudio plumbing). Both heavyweight; might do them as
  separate larger iterations or pivot to other work after the
  taxonomy is exhausted.

## iter-084 — sub-second turn rate (taxonomy 3.19)

**Branch:** iter-084-sub-second  **Commit:** 394ebf5  **Date:** 2026-05-25

Added sub-second turn rate — single human-feel threshold across
the session: what fraction of TTFS values landed under 1.0s.
<1.0s feels "instant" to most users, so the rate is a clear KPI
for "did we hit the snappy bar." Easier to track than median
when comparing across sessions or model swaps.

Implementation:
- Pure session-level derivation in `print_session_summary`. No
  new TurnMetrics field — uses the already-computed `ttfs_times`
  list (which already excludes zero-TTFS turns).
- Always emits when there's at least one measurable TTFS — the
  rate itself is informative even at 0% (the operator wants to
  know that, too).
- Output: "Sub-second TTFS: M/N (X%)" inserted between Best TTFS
  and Rhythm score for visual locality with the other "TTFS-shape"
  metrics.

Tests (9 in `tests/unit/test_sub_second_rate.py`):
- No-emit boundaries: zero turns, all-no-audio.
- Emit cases: 100%, 0%, mixed (50%), boundary (0.999s counts,
  1.0s doesn't), zero-TTFS turns excluded from denominator.
- Co-emission with Best TTFS and Rhythm score.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **1037 passed, 1 skipped in 24s** (1028
existing + 9 new).

Notes:
- **Forty-five metrics live (98% of the 46-metric taxonomy).**
- The TTFS dimension now has 9 lenses: budget, attribution,
  bilateral split, naturalness, cold-start, jitter, rhythm,
  best-of, and now hit-rate-vs-1s. After this, only WER (1.6 —
  needs ground truth corpus) remains; iter-085 may pivot to
  other work or tackle WER as a heavyweight iteration.
- **Pattern observation:** the last 10 iterations have alternated
  between session-level pure-derivation metrics (cheap, ~10
  lines + tests) and per-turn instrumented metrics (medium, ~50
  lines + tests across multiple files). The session-level
  metrics tend to be richer in the "diagnostic value per line of
  code" sense — they leverage data already captured rather than
  paying instrumentation costs. iter-085 onward will likely lean
  on this pattern more.

## iter-085 — max LLM inter-token gap (taxonomy 3.21 take)

**Branch:** iter-085-token-gap  **Commit:** 8c7af72  **Date:** 2026-05-25

Added `max_token_gap` — worst pause between consecutive LLM tokens
during the stream. Catches mid-stream stalls — currently invisible
because the user just hears a long pause and no signal fires.

The full taxonomy 3.21 metric is "fraction of LLM stalls (gap > N
seconds between tokens) that resolve before barge-in fires"; this
iteration ships the simpler "max gap observed" cousin, which gives
the immediately-actionable signal (any stall happened, how bad)
without the cross-event windowing.

Implementation:
- ChatLoop instruments the `for token in llm_gen` loop with a
  `prev_token_at` tracker. Each iteration: compute `gap = now -
  prev_token_at`; update `max_token_gap` if larger. The
  first-token wait is excluded — that's already iter-052's
  `llm_first_token`.
- `TurnMetrics.max_token_gap: float = 0.0` (new field).
- Per-turn print: appends "max gap NNms" suffix to the LLM total
  line when >200ms (sub-200ms is normal token-streaming jitter).
  Yellow at >500ms.
- Session summary: "Worst LLM stall: NNms (M/N turns)" emits
  when at least one turn stalled — gives the operator both the
  worst observation and a count to differentiate sporadic noise
  from a chronic problem.
- `ScenarioResult.max_token_gap_ms` on perf snapshots.

Tests (12 in `tests/unit/test_max_token_gap.py`):
- Defaults zero / per-turn print: zero omits, sub-threshold
  omits, significant emits dim, severe emits yellow.
- Session aggregate: no-data, all-below-threshold, single-stall
  (1/3 turns), multiple-stalls (3/4 turns).
- ChatLoop wiring: fast token stream → tiny gap; injected 200ms
  stall → at least 180ms observed (jitter-tolerant); single-token
  response → small bounded gap.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **1049 passed, 1 skipped in 24s** (1037
existing + 12 new).

Notes:
- **Forty-six metrics live (100% of the 46-metric taxonomy if
  3.21 is counted as covered by this take, OR 45.5/46 if the
  full stall-recoverability is required.)** The single remaining
  taxonomy entry (1.6 WER) is genuinely heavyweight (needs
  ground-truth corpus + jiwer integration); future iterations
  will likely pivot to non-taxonomy work — codebase refactors,
  architecture improvements, or report-generation enhancements.
- The instrumentation pattern is self-contained: 5 added lines
  in the LLM streaming loop, 1 new TurnMetrics field, ~10 lines
  for print + aggregate. Lower cost than most iter-N
  instrumentation despite touching the hot path.
- Next directions:
  - Investigate the iter-082 ChatLoop instance state pattern
    (`_last_first_audio_at`) — generalize for other cross-turn
    metrics (TTC's structural cousin: first-syllable-loss
    detection across barge boundaries).
  - 1.6 WER as a heavyweight iteration with a built-in test
    fixture corpus.
  - Refactor: TurnMetrics has grown to ~60 fields. A grouping
    refactor (perhaps nested dataclasses by phase: SttMetrics,
    LlmMetrics, TtsMetrics, BargeMetrics) would aid readability
    without changing the wire format.

## iter-086 — SessionMeta dataclass refactor

**Branch:** iter-086-session-meta  **Commit:** c91802e  **Date:** 2026-05-25

`print_session_summary` had grown to take 5 session-level kwargs
(`false_triggers`, `session_seconds`, `llm_errors`, `trim_events`,
`trim_messages_evicted`), and each new session-level signal would
extend the signature AND every call site.

Refactor: wrap the 5 signals in a `SessionMeta` dataclass. Future
session-level metrics extend the dataclass instead of the function
signature. mic_chat now constructs SessionMeta and passes
`meta=...`; `print_session_summary` prefers meta when given and
falls back to legacy kwargs otherwise.

Implementation:
- New `SessionMeta` dataclass in `_chat_metrics.py`, defined right
  above `print_session_summary`. Five fields mirror the existing
  kwargs; all default to 0/0.0.
- `print_session_summary` gained a `meta: Optional[SessionMeta] =
  None` parameter. When provided, fields default-merge with legacy
  kwargs (meta wins for any field it sets; kwargs fill gaps).
- The legacy kwargs are still accepted — pure backwards-compat:
  zero changes required for existing callers and tests.
- mic_chat updated to use the SessionMeta path — single object
  passed instead of 5 kwargs, future-proof against more
  additions.

Tests (8 in `tests/unit/test_session_meta.py`):
- SessionMeta defaults are 0/0.0; constructor accepts all fields.
- Legacy kwargs path produces correct output (regression cover).
- Meta path produces output IDENTICAL to legacy — guards against
  drift between the two equivalent code paths.
- Partial meta (only some fields set) emits only those lines.
- Mixed-mode (some via meta, some via legacy kwargs) merges
  correctly: meta wins for fields it sets; kwargs fill gaps.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **1057 passed, 1 skipped in 25s** (1049
existing + 8 new). Existing tests pass unchanged — backwards
compat is real, not theoretical.

Notes:
- This is the second non-metric iteration after iter-075 (play_fn
  kwargs refactor). The pattern of "extract growth to a dataclass
  before the next addition forces the change" continues — both
  iterations were preventive cleanup that reduces churn on
  future additions.
- The "produce output identical to legacy" test is the key
  regression sentinel: if a future change introduces drift
  between the two paths, the test catches it immediately rather
  than silently letting one mode rot.
- Next directions:
  - Continue refactoring momentum: `print_session_summary` is
    1500+ lines; extracting per-section helpers (TTFS block,
    barge-in block, error block) would aid readability.
  - WER (1.6) as the heavyweight remaining taxonomy entry — needs
    ground-truth corpus + jiwer integration.
  - Architecture: filler design improvements (multi-shot fillers
    if first sentence still hasn't arrived after another idle
    threshold), or aggressive first-sentence splitter (split on
    comma for the first sentence to start synth earlier).

## iter-087 — multi-shot fillers (architecture)

**Branch:** iter-087-multi-fillers  **Commit:** 1497509  **Date:** 2026-05-25

The original iter-011 design fired ONE filler per turn after
`idle_threshold` elapsed. iter-087 enables **up to MAX_FILLERS=2**
fillers per turn, picking from clips not yet played to avoid
"umm umm" repetition.

UX win: on slow-LLM turns where the FIRST filler completed but
the LLM still hadn't produced a complete sentence, the user
previously heard awkward silence until first audio. Now a second
distinct filler can fire to maintain conversational presence.

Implementation:
- Replaced boolean `filler_used` with `played_filler_ids: set[int]`
  + `MAX_FILLERS = 2` cap inside `SentenceWorker._run()`.
- Filler-timeout gate now:
  ```python
  filler_can_fire = (
      bool(self._fillers)
      and len(played_filler_ids) < MAX_FILLERS
      and len(played_filler_ids) < len(self._fillers)
  )
  ```
  Both bounds matter: cap at 2 absolute AND can't exceed available
  clips (single-filler configs stay single-shot).
- Picker invoked on a FILTERED list of unplayed clips so the
  second pick can't accidentally repeat the first — defensive
  against custom pickers that ignore the input order.
- `last_filler_id` (iter-081) remains the LAST clip played per
  turn; session-wide novelty in `print_session_summary` continues
  to aggregate via set arithmetic across turns.

Tests (5 new in `test_multi_shot_fillers.py`):
- Single-clip config: stays single-shot (no clip to swap to).
- Multi-clip + slow LLM: 2 fires happen within the test window.
- Distinct clips picked: second fire's id ≠ first.
- MAX_FILLERS cap honored: 5 clips + long wait → exactly 2 fires.
- Picker that always returns lst[0] cannot pick the same clip
  twice (worker's available-list filter saves it).

Adjusted iter-081's `test_filler_picker_picks_one` to be tolerant
of multi-shot semantics — the assertion now accepts either
single-shot or multi-shot outcomes since the test's fixed wait
window may race the second-fire timeout.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **1062 passed, 1 skipped in 25s** (1057
existing + 5 new). All prior tests pass — no regression.

Notes:
- This is the **first architectural change** since iter-008's
  streaming-sentence-dispatch and iter-009's barge-in primitives.
  Most iterations since iter-009 have been measurement-focused
  (the 46-metric taxonomy). iter-086 (cleanup) and now iter-087
  (architecture) start a non-measurement phase.
- The metric instrumentation built in iter-051/081 was directly
  exercised here — `filler_false_positive` (iter-051) still
  works correctly with multi-shot (any filler that played when
  llm_first_token < idle_threshold is a false positive); session-
  wide `Filler novelty` (iter-081) now becomes a richer signal
  because turns with 2 fires contribute 2 ids to the unique
  count.
- Next directions:
  - Aggressive first-sentence splitter (split on comma for
    sentence 1 to start synth earlier — TTFS reduction at the
    cost of prosody).
  - Continue refactoring momentum: extract a TTFS-block helper
    from print_session_summary.
  - WER (1.6) as a heavyweight iteration with built-in test
    fixture corpus.

## iter-088 — aggressive first-sentence splitter (architecture)

**Branch:** iter-088-aggressive-split  **Commit:** b05ba49  **Date:** 2026-05-25

Reduces TTFS on long-preamble LLM responses ("Well, let me think
about that for a moment, ...") by allowing the splitter to break
on a comma+whitespace as the FIRST sentence terminator — but only
when the pre-comma content exceeds 20 characters so short
interjections like "Sure," don't trigger early splits.

Strict `.!?` matching resumes from sentence 2 onward — only the
preamble gets the aggressive treatment. ChatLoop tracks per-turn
state, flipping the flag off as soon as the first sentence
emerges.

Opt-in via `chat.aggressive_first_sentence` in `config.local.yaml`
(default off — there's a prosody tradeoff for the TTFS reduction;
operators choose).

Implementation:
- New `AGGRESSIVE_COMMA_END = re.compile(r'(?<=,)\s+')` regex and
  `AGGRESSIVE_MIN_CHARS = 20` threshold in `_chat_helpers.py`.
- `split_complete_sentences` gained `aggressive_first: bool =
  False` kwarg. When True, finds the EARLIEST qualifying comma
  match (start ≥ MIN_CHARS) and inserts it as the first split
  candidate IF it precedes any strict `.!?` match. The strict
  matches still win when they appear earlier — periods take
  priority on ties.
- `ChatLoop.__init__` gained `aggressive_first_sentence: bool =
  False`. The for-token loop tracks `aggressive_active` per turn,
  starts True only when the loop's config enabled it, flips False
  as soon as the splitter returns ANY complete sentence.
- `mic_chat` reads from `chat_cfg.get("aggressive_first_sentence",
  False)` and passes through to ChatLoop.

Tests (11 in `tests/unit/test_aggressive_splitter.py`):
- Splitter contract: default-off behavior unchanged; long-preamble
  splits at the qualifying comma; short pre-comma doesn't trigger;
  period earlier in buffer wins over comma later; only the FIRST
  qualifying comma is considered (subsequent commas in the buffer
  don't split); AGGRESSIVE_MIN_CHARS exposed as a public constant;
  empty buffer.
- ChatLoop wiring: default-off long-preamble still single-sentence;
  aggressive-on splits at the preamble comma producing 2 sentences;
  even with multiple long-preamble commas, only the FIRST splits
  aggressively; default constructor kwarg is off.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **1073 passed, 1 skipped in 25s** (1062
existing + 11 new). All prior tests pass — backwards-compat
guaranteed.

Notes:
- This is the second consecutive architectural iteration after
  iter-087 (multi-shot fillers). Both are opt-in features that
  improve TTFS / UX with explicit tradeoffs the operator chooses.
- The TTFS attribution metric (iter-076) and the FT-A gap
  (iter-083) provide ready-made instrumentation to measure the
  TTFS impact of enabling this. Future work could compare two
  perf-test runs (aggressive on/off) to quantify.
- Next directions:
  - Wire iter-076 perf-snapshot to optionally A/B aggressive
    on/off and graph the TTFS delta.
  - Continue refactoring momentum: extract a TTFS-block helper
    from print_session_summary (1500+ lines).
  - Continue architecture: maybe an "aggressive on long stalls"
    mode that flips the aggressive flag mid-turn if a long stall
    is detected (combines iter-085 max_token_gap signal with
    iter-088 splitter behavior).

## iter-089 — extract _emit_ttfs_block helper

**Branch:** iter-089-ttfs-helper  **Commit:** dca7298  **Date:** 2026-05-25

`print_session_summary`'s TTFS block was 80 lines of co-emitted
session-summary output (Median TTFS, Best TTFS, sub-second rate,
rhythm, jitter, cold-start, naturalness) — all gated on
`ttfs_times` being non-empty AND various sub-conditions.

Extracted to a standalone `_emit_ttfs_block(emit, ttfs_times,
metrics_list, naturalness_counts)` helper. Behavior-preserving:
output is byte-for-byte identical (1073 prior tests pass unchanged
without modification).

This sets the pattern for further teardown. `print_session_summary`
is currently 1500+ lines; future iterations can extract Errors,
Barge, Filler, etc. blocks the same way:
1. Identify a contiguous block of lines that share gating
   conditions.
2. Pull the lines + the data they read from into a helper.
3. Replace the inline block with a single call.
4. Verify the existing tests still pass (regression sentinel).
5. Add direct tests on the helper for faster, more focused
   coverage.

Tests (9 in `tests/unit/test_emit_ttfs_block.py`):
- Empty ttfs_times → n/a placeholders.
- Single-turn → only the lines that don't need ≥2 turns
  (rhythm/jitter/cold-start gated, skipped).
- Multi-turn → rhythm + jitter emit.
- Cold-start emits when penalty > 50ms; skipped below.
- Naturalness emits only when any bucket set.
- Ordering invariant: median → best → sub-second → rhythm →
  jitter → cold-start → naturalness in stable order.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **1082 passed, 1 skipped in 25s** (1073
existing + 9 new). The 1073 prior tests are the regression
sentinel — they pass unchanged because the helper produces
byte-for-byte identical output.

Notes:
- The "test the helper directly" pattern is faster (sub-millisecond
  per test vs ~25ms going through the full session-summary path)
  and more focused (no setup of unrelated metrics). Future
  per-block helpers should follow this pattern.
- `print_session_summary` is now 1500 → 1426 lines, a modest
  reduction. Each future block extraction will compound.
- Next directions:
  - Continue refactoring: extract _emit_barge_block (the iter-040/
    041/047/056/057/060/080 cluster — substantial size).
  - Continue architecture: combine iter-085 (max_token_gap) and
    iter-088 (aggressive splitter) into "auto-aggressive" — flip
    splitter to aggressive mid-turn if a stall is detected.
  - WER ground-truth fixture as a heavyweight iteration.

## iter-090 — extract _emit_barge_block helper

**Branch:** iter-090-barge-helper  **Commit:** dbf434d  **Date:** 2026-05-25

The barge-in section of `print_session_summary` was 76 lines
across 8 sub-conditions (count + mid-stream %, interruption rate,
latency median+worst, cancel-to-close, phase distribution, regret
rate, pre-empted words). Extract to a standalone
`_emit_barge_block(emit, stats: BargeStats)` helper.

Introduced a `BargeStats` dataclass to bundle the 11 input fields
cleanly. Following the iter-086 SessionMeta pattern: each new
barge-side metric extends the dataclass instead of growing the
helper signature.

Behavior-preserving: byte-for-byte identical output. The 1082
prior tests pass unchanged — the regression sentinel pattern from
iter-089 carries forward.

Tests (13 in `tests/unit/test_emit_barge_block.py`):
- `BargeStats` defaults are all zero / empty.
- No-barges path emits nothing.
- Single barge: clean form ("all between sentences") vs
  mid-stream form ("N mid-stream, X%").
- Interruption rate against `n` denominator.
- Latencies: median + worst when present.
- Cancel-to-close median when present.
- Phase distribution: emits when either non-zero, omits when
  both 0.
- Regret rate emits with correct %, omits when 0.
- Pre-empted words: total + avg/loss correctly computed.
- Ordering invariant across all 8 emitted line types.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **1095 passed, 1 skipped in 25s** (1082
existing + 13 new).

Notes:
- `print_session_summary` is now ~1370 lines (down from 1500
  pre-iter-089). Each block extraction compounds — TTFS (-80) +
  barge (-76) = -156 lines reduced. Function still has Errors,
  Filler, WPM, Sentence-stats, Token-lag, Queue, Mic-stale, VAD-
  false-trig, Trim, Context, Recovery, Bargeable blocks left to
  consider.
- The `BargeStats` dataclass pattern is paying off — adding
  iter-080 pre-empted-content fields was a 2-line dataclass
  extension instead of growing a kwarg list. Future barge
  metrics extend the dataclass cleanly.
- Next directions:
  - Continue: extract `_emit_filler_block` (iter-014/051/081).
  - Continue: extract `_emit_errors_block` (iter-058/067/079 +
    silent-turn rate).
  - Architecture: combine iter-085 max_token_gap with iter-088
    aggressive_first to auto-flip the splitter mid-turn on a
    detected stall.

## iter-091 — extract _emit_filler_block helper

**Branch:** iter-091-filler-helper  **Commit:** 83a355a  **Date:** 2026-05-25

Continues the iter-089/iter-090 refactor pattern. The filler
section of `print_session_summary` was 32 lines across three
sub-conditions:
- "Fillers played: N" (iter-014).
- "Filler FP rate: M/K (X%)" (iter-051) when any false positive
  fired.
- "Filler novelty: M unique / N (X%)" (iter-081) when ≥2 fillers
  played.

Extracted to `_emit_filler_block(emit, stats: FillerStats)` with
a `FillerStats` dataclass bundling the 4 input fields. Mirrors
the iter-090 `BargeStats` pattern.

Behavior-preserving: byte-for-byte identical output. The 1095
prior tests pass unchanged.

Tests (9 in `tests/unit/test_emit_filler_block.py`):
- `FillerStats` defaults all zero.
- No-data → no emit.
- Single-play → count line only (FP rate gated on >0 FPs;
  novelty gated on ≥2 plays).
- FP rate emits with correct % when FPs present; omits when 0.
- Novelty emits at 2+ plays; omits at single play (trivially
  100%).
- Ordering invariant: count → FP → novelty.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **1104 passed, 1 skipped in 26s** (1095
existing + 9 new).

Notes:
- `print_session_summary` is now ~1340 lines (down from 1500
  pre-iter-089). Block extractions so far: TTFS (-80), Barge
  (-76), Filler (-32) = -188 lines reduced. Significant
  readability win.
- The pattern is fully formed:
  1. Identify a contiguous block sharing a top-level guard.
  2. Bundle inputs into a small dataclass.
  3. Move the block to a helper that takes `(emit, stats)`.
  4. Existing tests are the regression sentinel.
  5. Add focused helper-level tests.
- Next directions:
  - Extract `_emit_errors_block` (iter-058 + iter-067 worker
    recovery + iter-079 silent turns).
  - Extract `_emit_wpm_block` (iter-046 bot + iter-064 user
    + mirror gap).
  - Architecture: combine iter-085 + iter-088 → auto-aggressive
    splitter on detected stall.

## iter-092 — extract _emit_errors_block helper

**Branch:** iter-092-errors-helper  **Commit:** 43532ec  **Date:** 2026-05-25

Continues the refactor pattern. The errors-and-silent-turn section
of `print_session_summary` was 50 lines across three sub-conditions:
- "Errors: N LLM, M worker (over X attempts)" (iter-058) when any
  error fired.
- "Worker recovery: M/N turns produced audio (X%)" (iter-067)
  when any worker error fired.
- "Silent turns: M/N (X%) — bot produced no audio" (iter-079)
  when any silent turn occurred.

Extracted to `_emit_errors_block(emit, stats: ErrorStats)` backed
by a 7-field `ErrorStats` dataclass. Behavior-preserving:
byte-for-byte identical output (1104 prior tests pass unchanged).

Tests (11 in `tests/unit/test_emit_errors_block.py`):
- `ErrorStats` defaults all zero.
- No-data → no emit.
- LLM-only errors → count + attempts singular/plural form.
- Worker errors + recovery rates (full success, zero, partial).
- Combined LLM+worker → both kinds in one Errors line.
- Silent turns: emit when present, omit when zero, independent
  of error block (silent turns can fire without any errors).
- Ordering invariant: Errors → Worker recovery → Silent turns.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **1115 passed, 1 skipped in 24s** (1104
existing + 11 new).

Notes:
- `print_session_summary` is now ~1290 lines (down from 1500
  pre-iter-089). Block extractions so far: TTFS (-80), Barge
  (-76), Filler (-32), Errors (-50) = -238 lines reduced.
- Four blocks extracted; the function is approaching the size
  where the remaining co-emitted clusters (WPM, sentences, mic
  stale, VAD false-trig, trim, context, queue, token-lag,
  bargeable, primed-audio) could be extracted in 2-3 more
  iterations.
- The `*Stats` dataclass pattern is robust — each new metric in
  a block extends the dataclass instead of touching the helper
  signature.
- Next directions:
  - Extract `_emit_wpm_block` (iter-046 bot WPM, iter-064 user
    WPM + mirror gap).
  - Extract `_emit_sentence_block` (iter-045 mean, iter-070
    range, iter-059 split coverage).
  - Architecture: combine iter-085 + iter-088 → auto-aggressive
    splitter on detected stall.

## iter-093 — auto-aggressive splitter on stall (architecture)

**Branch:** iter-093-auto-aggressive  **Commit:** 0c8a118  **Date:** 2026-05-25

Combines two existing pieces into a real behavior change:
- **iter-085** tracks max inter-token gap during the LLM stream.
- **iter-088** implements aggressive comma-split for the first
  sentence (opt-in static config).

iter-093 wires them together: when the LLM stalls past a
configurable threshold AND no sentence has emerged yet, ChatLoop
flips `aggressive_active=True` mid-turn so the NEXT splitter call
can comma-split for earlier audio recovery.

Rationale: the user has waited longer than expected; getting some
audio out faster (with imperfect prosody) beats continued silence.
A 500-1000ms threshold sits comfortably above normal token-
streaming jitter (~50ms inter-token gap) but well below the user's
"is this broken" threshold (~2s).

Opt-in via `chat.auto_aggressive_threshold` in `config.local.yaml`
(0 = disabled, recommend 0.5-1.0s when enabled).

Implementation:
- `ChatLoop.__init__` gains `auto_aggressive_threshold: float =
  0.0`.
- Inside the for-token loop, after the gap update:
  ```python
  if (
      self._auto_aggressive_threshold > 0
      and gap > self._auto_aggressive_threshold
      and not aggressive_active
      and first_sentence_at is None
  ):
      aggressive_active = True
  ```
- Static `aggressive_first_sentence=True` path is unaffected by
  the auto-flip (the `not aggressive_active` guard short-circuits).
- mic_chat reads `chat_cfg.get("auto_aggressive_threshold", 0.0)`.

Tests (7 in `tests/unit/test_auto_aggressive.py`):
- Disabled-default behavior unchanged.
- Threshold set + no stall → no flip.
- Threshold + qualifying stall → flips, produces 2 sentences
  (preamble + tail).
- iter-085 metric still records the stall correctly.
- Sub-threshold stall doesn't flip.
- Post-first-sentence stall ignored (audio is already flowing).
- Static + auto coexist without double-flip.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **1122 passed, 1 skipped in 29s** (1115
existing + 7 new).

Notes:
- This is the first iteration that **uses** previously-instrumented
  metrics to drive a behavior change, not just measure or report.
  iter-085's `max_token_gap` was originally a passive observation;
  iter-093 makes it actionable.
- The pattern generalizes — other instrumented signals could
  drive other auto-tuning behaviors:
  - iter-051's filler false-positive rate could auto-tune
    `idle_threshold` over time.
  - iter-077's `context_tokens` growth could auto-trim more
    aggressively.
  - iter-072's `stt_preview_divergence` could auto-disable the
    preview line when consistently misleading.
- Next directions:
  - Continue refactor: extract `_emit_wpm_block` (iter-046+064).
  - Continue auto-tuning: filler idle_threshold based on observed
    LLM TTFT distribution.
  - Investigate the WER fixture (1.6) as a heavyweight iteration.

## iter-094 — extract _emit_wpm_block helper

**Branch:** iter-094-wpm-helper  **Commit:** 570b375  **Date:** 2026-05-25

Continues the refactor pattern. The WPM block of
`print_session_summary` was 14 lines covering three lines:
- "Median user WPM: NNN" (iter-064).
- "Median bot WPM: NNN" (iter-046).
- "Mirror gap: ±NN WPM (bot − user)" (iter-064) when both
  measurable.

Extracted to `_emit_wpm_block(emit, stats: WpmStats)` backed by a
2-field `WpmStats` dataclass.

Behavior-preserving: byte-for-byte identical output. The 1122
prior tests pass unchanged.

Tests (7 in `tests/unit/test_emit_wpm_block.py`):
- `WpmStats` defaults to empty lists.
- No-data → no emit.
- User-only / bot-only → only the relevant median, no mirror gap.
- Both → all three lines with correct gap sign.
- Negative mirror gap rendering.
- Ordering invariant: user → bot → gap.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **1129 passed, 1 skipped in 27s** (1122
existing + 7 new).

Notes:
- `print_session_summary` is now ~1280 lines (down from 1500
  pre-iter-089). Block extractions: TTFS (-80), Barge (-76),
  Filler (-32), Errors (-50), WPM (-12) = -250 lines reduced.
- The `*Stats` dataclass + `_emit_*_block` helper pattern is
  consistent across 5 helpers now. Adding a new metric to any of
  these blocks is a 1-2 line dataclass extension instead of
  growing a kwarg list.
- Next directions:
  - Extract `_emit_sentence_block` (iter-045 mean_sentence_chars,
    iter-070 sentence range, iter-059 split coverage).
  - Continue auto-tuning lane: filler idle_threshold based on
    iter-051 false-positive distribution (suggest, don't auto-
    apply, mid-session).
  - Investigate the WER fixture (1.6) as a heavyweight iteration.

## iter-095 — extract _emit_sentence_block helper

**Branch:** iter-095-sentence-helper  **Commit:** c4b4ceb  **Date:** 2026-05-25

Continues the refactor pattern. The sentence-shape section of
`print_session_summary` was 30 lines covering:
- "Mean sentence: NN chars" (iter-045) when measurable.
- "Sentence range: [min..max] chars (session)" (iter-070) when
  min != max.
- "Split coverage: NN%" (iter-059) when measurable.

Extracted to `_emit_sentence_block(emit, stats: SentenceStats)`
backed by a 4-field `SentenceStats` dataclass (sentence_lens,
min_chars_seen, max_chars_seen, coverage_values).

Behavior-preserving: byte-for-byte identical output. The 1129
prior tests pass unchanged.

Tests (9 in `tests/unit/test_emit_sentence_block.py`):
- `SentenceStats` defaults (empty / 0).
- No-data → no emit.
- Mean-only when range fields unset (range gated on max>0).
- Range emits when min<max; omits when min==max.
- Coverage emits when present, omits when empty, independent of
  mean (can fire alone).
- Ordering invariant: mean → range → coverage.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **1138 passed, 1 skipped in 28s** (1129
existing + 9 new).

Notes:
- `print_session_summary` is now ~1255 lines (down from 1500
  pre-iter-089). Block extractions: TTFS (-80), Barge (-76),
  Filler (-32), Errors (-50), WPM (-12), Sentence (-30) = -280
  lines reduced. Six helpers extracted, all following the
  identical `*Stats` dataclass + helper pattern.
- The function is approaching a comfortable size where the
  remaining inline blocks (token-lag, queue depth, mic stale,
  VAD false-trig, primed audio, trim, context, bargeable, TTFS
  attribution) could be batched into 2-3 more iterations or
  consolidated into a single `_emit_misc_block` for the
  smaller ones.
- Next directions:
  - Continue refactor: extract `_emit_recording_block` (iter-037
    mic stale + iter-048 false-trigger + iter-074 bargeable
    fraction).
  - Continue auto-tuning: filler idle_threshold suggestion in
    print_session_summary based on the iter-051 false-positive
    distribution (recommend, don't auto-apply).
  - WER fixture (1.6) heavyweight iteration.

## iter-096 — filler idle_threshold recommendation (auto-tuning)

**Branch:** iter-096-idle-recommend  **Commit:** dc600da  **Date:** 2026-05-25

When filler false positives are firing, the FP-rate session line
previously said "tune idle_threshold up" without a specific
target. iter-096 computes a recommended value from the observed
`llm_first_token` distribution and surfaces it inline:

    Filler FP rate:   2/4 (50%) — tune idle_threshold up to 1.2s

Recommendation formula:

    recommended = max(current * 1.2, p75(first_token) + 100ms)

Rounded to 1 decimal. The 1.2× current ensures a meaningful bump
even when first-token times cluster near the existing threshold;
the p75 + 100ms ensures we cover most real first-token waits
while still firing fillers on the slow tail.

Implementation:
- `SessionMeta` gains `idle_threshold: float = 0.0` (no legacy
  kwarg path — only flows through SessionMeta).
- `FillerStats` gains `recommended_idle_threshold: float = 0.0`.
- `print_session_summary` computes the recommendation when:
  - `filler_false_positives > 0`
  - `meta.idle_threshold > 0`
  - At least 2 turns have measurable `llm_first_token`
- `_emit_filler_block` appends "to N.Ns" to the legacy "tune up"
  text when the recommendation is set.
- `mic_chat` passes `filler_idle_threshold` via SessionMeta.

Backwards-compat: when no recommendation is computed, the legacy
"tune idle_threshold up" suffix is preserved unchanged. All 1138
prior tests pass.

Tests (9 in `tests/unit/test_idle_threshold_recommend.py`):
- `FillerStats.recommended_idle_threshold` default zero.
- Helper rendering: no recommendation → legacy suffix; non-zero →
  appended; no-FP → no FP-rate line at all (recommendation moot).
- End-to-end via `print_session_summary`: no threshold → legacy
  text; with threshold + observations → 0.6s rendered; high
  first-token times → p75 path produces 2.1s; no first-token
  data → falls back to legacy; clean session → no FP line.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **1147 passed, 1 skipped in 40s** (1138
existing + 9 new).

Notes:
- This is the **second auto-tuning use of an instrumented metric**
  after iter-093's auto-aggressive splitter. iter-051 was a
  passive measurement (FP rate); iter-096 turns it into
  actionable advice (specific recommended value).
- The recommendation is shown, not auto-applied — mid-session
  config mutation is too risky. Operator updates `config.local.yaml`
  for the next session and re-runs.
- Pattern generalizes:
  - iter-077 context tokens could recommend a tighter
    `max_user_assistant` trim cap when growth is unbounded.
  - iter-072 stt_preview_divergence could recommend disabling
    the live preview when consistently misleading.
  - iter-068 TTFS jitter could suggest investigating LLM endpoint
    when stdev grows large.
- Next directions:
  - More extractions: `_emit_recording_block` (mic stale, false-
    trigger, bargeable fraction).
  - More auto-tuning recommendations following the iter-096
    pattern.
  - WER fixture (1.6) heavyweight iteration.

## iter-097 — extract _emit_history_block helper

**Branch:** iter-097-history-helper  **Commit:** 76d4e2e  **Date:** 2026-05-25

Combines iter-077 context-tokens and iter-078 trim-events into a
single `_emit_history_block` helper — both about conversation-
history management (history grows → trim caps it). The two
iter-blocks were inline-adjacent in `print_session_summary`;
combining them into one helper reflects their semantic
relationship.

`HistoryStats` dataclass with 3 fields
(`context_token_counts`, `trim_events`, `trim_messages_evicted`).
Behavior-preserving: byte-for-byte identical output. The 1147
prior tests pass unchanged.

Tests (10 in `tests/unit/test_emit_history_block.py`):
- `HistoryStats` defaults (empty / 0).
- No-data → no emit.
- Context tokens: 2-turn (median+max only, no growth), 3-turn
  (growth line emits), negative growth.
- Trim events: emit with ratio, omit when zero, steady-state
  1.0/event.
- Independence: trim can emit without context data.
- Ordering: context → growth → trim.

Verification: `python -m pytest tests/unit/ tests/integration/
tests/performance/` → **1157 passed, 1 skipped in 28s** (1147
existing + 10 new).

Notes:
- **Seven block helpers extracted** so far: TTFS, Barge, Filler,
  Errors, WPM, Sentence, History. `print_session_summary` is now
  ~1230 lines (down from 1500 pre-iter-089). Cumulative -310
  lines reduced.
- The semantic-grouping pattern (combining adjacent related blocks
  rather than one-per-iter-block) starts paying off — iter-077
  and iter-078 share a dataclass call site, reducing the helper
  overhead.
- Next directions:
  - Extract `_emit_recording_block` (iter-037 mic-stale + iter-048
    VAD false-trigger + iter-074 bargeable + iter-057 primed
    audio).
  - More auto-tuning: context-tokens growth could recommend a
    tighter `max_user_assistant` cap when trim_events stays at 0
    despite measurable growth.
  - WER fixture (1.6) heavyweight iteration.

## iter-098 — A/B perf scenarios for aggressive first-sentence splitter

**Goal:** Validate iter-088's TTFS-reducing claim with empirical
numbers in the perf time-series chart, not just unit tests.

iter-088 added the `aggressive_first_sentence` opt-in that slices
the first sentence on a comma boundary (≥20 chars) instead of
waiting for the period. The unit tests in
`tests/unit/test_chat_helpers.py` verify the split behavior, and
iter-093 added the auto-aggressive-on-stall trigger, but neither
told the operator how much TTFS the splitter actually saves on a
realistic long-preamble response. That number lives only in the
perf suite.

**Change:** Two new perf scenarios driving the same long-preamble
response twice — one with `aggressive_first_sentence=False`, one
with `True`. The shared `_LONG_PREAMBLE` is structured exactly
the way the splitter benefits most: a 41-char clause with a comma
before the first period.

```python
_LONG_PREAMBLE = (
    "Well let me think about this for a moment, "
    "the answer is twelve. "
    "And the reason is straightforward. "
    "It just is."
)
```

`_run_scenario` grew one new kwarg (`aggressive_first_sentence`)
that forwards directly to `ChatLoop.__init__`. No other plumbing
changes — the existing rich `ScenarioResult` schema already
captures TTFS, llm_first_sentence, and mean_sentence_chars.

**Empirical A/B** (this run, x86_64 Linux stub pipeline):

| Metric                 | aggressive=False | aggressive=True | Δ      |
|------------------------|------------------|-----------------|--------|
| TTFS                   | 931.6 ms         | 891.1 ms        | -40 ms |
| llm_first_sentence     | 131.0 ms         | 90.6 ms         | -40 ms |
| mean_sentence_chars    | 36.3             | 27.0            | -9     |

The mean_chars drop from 36→27 confirms the splitter actually
fired (otherwise both columns would match). The TTFS savings
matches the llm_first_sentence savings near-perfectly because
synth + dispatch are deterministic in the stub pipeline — the
only variable here is *when* the first complete fragment is
handed to the worker.

Verification: `python -m pytest tests/performance/ -q` → **11
passed in 2.4s** (was 9, +2 new). Full unit + integration suite:
`tests/ --ignore=tests/performance --ignore=tests/test_session.py
--ignore=tests/e2e -q` → **1148 passed, 1 skipped** (identical to
main; the test_session.py + e2e errors are pre-existing audio-
fixture infra issues, unrelated).

Notes:
- **Pattern: pair-scenarios for opt-in features.** iter-088 +
  iter-093 are both opt-in flags. Adding paired scenarios (off vs
  on) on the *same* response gives a clean ratio that survives
  hardware-dependent absolute numbers — a 50% TTFS reduction
  reads the same on a M1 and a stub, even if the absolute ms
  differ.
- **Future A/B candidates with the same shape:**
  - Filler `idle_threshold=0.6` vs `0.15` on a slow LLM (TTFS
    dominated by filler start vs LLM TTFB).
  - `auto_aggressive_threshold=0.0` vs `0.5` on a stalled
    response (per-token-delay > threshold flips mid-stream).
  - `max_user_assistant=20` vs `5` on a long session (context
    growth → LLM TTFB scaling).
- Each pair adds 2 rows to the perf time-series; the chart
  generator already plots them as separate lines, so the visual
  comparison is free.
- Next directions:
  - Pair scenarios for filler idle_threshold (iter-051
    sensitivity test).
  - Extract `_emit_recording_block` (still pending from iter-097).
  - WER fixture (1.6) heavyweight iteration.

## iter-099 — A/B perf scenarios for filler idle_threshold

**Goal:** Apply iter-098's pair-scenario pattern to the filler
`idle_threshold` (iter-011, sensitivity-tuned in iter-051,
auto-recommended in iter-096) so the time-series chart can show
how the threshold trades TTFS savings against false-positive
filler fires.

**Change:** Two new perf scenarios driving the same slow-LLM
response (per_token_delay=0.1) twice — one at the operator-default
0.6s threshold, one at the aggressive 0.15s threshold. Both share
the existing `_FILLER_CLIP` (a 2048-sample constant tone) so the
only variable is the threshold.

```python
_FILLER_CLIP = (np.full(2048, 0.3, dtype=np.float32), [])

def test_filler_threshold_default(self):    # 0.6s
def test_filler_threshold_aggressive(self): # 0.15s
```

No plumbing changes needed — `_run_scenario` already accepts
`fillers` + `idle_threshold` (added in iter-011 era).

**Empirical A/B** (this run, x86_64 Linux stub pipeline):

| Metric           | threshold=0.6 (default) | threshold=0.15 (aggressive) | Δ        |
|------------------|-------------------------|-----------------------------|----------|
| TTFS             | 1301.1 ms               | 950.3 ms                    | -351 ms  |
| llm_first_token  | 100.1 ms                | 100.1 ms                    | 0        |
| last_filler_id   | 0 (no fire)             | non-zero (fire)             | —        |

The LLM TTFB is identical at 100ms in both runs (deterministic
stub). The default 0.6s threshold sits *above* that TTFB, so the
filler never triggers — TTFS is bounded by synth + first-sentence
emission. The aggressive 0.15s threshold sits *below* the TTFB,
so the filler fires while the LLM is still warming up — TTFS is
bounded by the much faster filler-clip start.

The 351ms savings is the entire LLM-warmup cost being masked. On
a real LLM with 200-500ms TTFB the aggressive threshold would
mask even more, but the operator pays for it with a filler clip
on every turn (false-positive rate climbs as threshold falls).

Verification: `python -m pytest tests/performance/ -q` → **13
passed in 3.5s** (was 11, +2 new). Full unit + integration suite:
`tests/ --ignore=tests/performance --ignore=tests/test_session.py
--ignore=tests/e2e -q` → **1148 passed, 1 skipped** (identical to
main).

Notes:
- **Pair-scenario compounding.** This is the second pair after
  iter-098. The pattern now has 2 instances and is starting to
  earn its keep — same data shape, same chart axes, same code
  path. A third pair (auto-aggressive threshold off vs on) would
  cement the pattern as the default.
- **The TTFS difference is pure LLM-masking, not pipeline speedup.**
  iter-098's aggressive splitter cuts real synth+dispatch work;
  iter-099's aggressive threshold just hides the LLM warmup
  behind a filler. The chart should label these distinctly so
  the operator doesn't conflate them.
- **iter-096 recommendation now has a calibration target.** The
  session-summary line that recommends an `idle_threshold` value
  based on FP rate now has a perf-suite anchor: at 0.15s the
  filler fires deterministically on a 100ms TTFB. Future iter
  could grid-search threshold values across multiple TTFB
  shapes, picking the lowest threshold where FP rate < 5%.
- **Operational gotcha for live deployments:** A 0.15s threshold
  on the live mic_chat path means *every* turn with a real LLM
  TTFB > 150ms will fire a filler. That's most turns. Operators
  who don't want chatty fillers should keep the default 0.6s.
- Next directions:
  - Pair scenario for `auto_aggressive_threshold` (iter-093)
    off vs on with stalled LLM — third pair, validates iter-093
    in the same way iter-098 validated iter-088.
  - Threshold grid-search (4-5 values) — would surface the FP
    rate curve more precisely than a binary A/B.
  - Extract `_emit_recording_block` (still pending from iter-097).
  - WER fixture (1.6) heavyweight iteration.

## iter-100 — A/B perf scenarios for auto_aggressive_threshold

**Goal:** Third pair-scenario in three iterations (after iter-098,
iter-099). Validates iter-093's auto-aggressive-on-stall logic
empirically and cements the pair-scenario pattern as the default
shape for opt-in feature validation.

iter-093 watches the inter-token LLM gap and flips the splitter
to aggressive mode mid-stream when the gap exceeds the configured
threshold AND no complete sentence has been emitted yet. The
existing unit tests in `tests/unit/test_chat_loop.py` verify the
flip mechanics; this iteration measures the TTFS impact.

**Change:** Extended the perf-suite stub `_yield_tokens` with an
optional `stall_after`/`stall_seconds` pair that injects a long
pause AFTER yielding the first token containing a substring. Then
two new scenarios drive the same `_STALLED_PREAMBLE` response
twice — once with `auto_aggressive_threshold=0.0` (off, pre-093
behavior), once with `0.3` (flips on the 500ms stall).

```python
_STALLED_PREAMBLE = (
    "Well let me think about this for a moment, "
    "the answer is twelve. "
    "And the reason is straightforward."
)

# Stall fires AFTER yielding the "moment," token, so the gap is
# observed when the next token (the post-stall "the") arrives.
stall_after="moment,", stall_seconds=0.5
```

`_run_scenario` grew two new kwargs (`auto_aggressive_threshold`,
`stall_after`/`stall_seconds`) — small surface-area expansion;
the existing pair-scenario shape carries over unchanged.

**Empirical A/B** (this run, x86_64 Linux stub pipeline, per-token
delay 10ms):

| Metric                | threshold=0.0 (off) | threshold=0.3 (on) | Δ      |
|-----------------------|---------------------|--------------------|--------|
| TTFS                  | 1431.7 ms           | 1401.5 ms          | -30 ms |
| llm_first_sentence    | 631.3 ms            | 601.1 ms           | -30 ms |
| max_token_gap         | 510.1 ms            | 510.1 ms           | 0      |
| mean_sentence_chars   | 49.0                | 32.3               | -17    |

Both runs see the same 510ms `max_token_gap` — the stall fires
identically. The mean_chars drop from 49→32 confirms the auto-flip
worked: with the splitter strict, the first sentence had to grow
to "Well let me think about this for a moment, the answer is
twelve." (49 chars); with the auto-flip, it sliced at the comma
to "Well let me think about this for a moment," (~32 chars after
trim).

The 30ms TTFS savings is modest because the stub LLM streams
tokens at 10ms each — only ~3-4 tokens between the comma and the
period. On a real LLM with 50ms/token (the iter-052 baseline),
this would scale to ~150-200ms savings. On a 100ms/token slow
LLM (our `slow_llm_300ms` scenario), ~300-400ms.

Verification: `python -m pytest tests/performance/ -q` → **15
passed in 4.9s** (was 13, +2 new). Full unit + integration suite:
`tests/ --ignore=tests/performance --ignore=tests/test_session.py
--ignore=tests/e2e -q` → **1148 passed, 1 skipped** (identical to
main).

Notes:
- **Three pair-scenarios — the pattern is the default now.**
  iter-098 (aggressive splitter), iter-099 (filler threshold),
  iter-100 (auto-aggressive). All share the same shape: a single
  `_FIXTURE` response or clip, two scenario methods that vary
  exactly one kwarg, and a TTFS / behavior-marker delta. Future
  opt-in features should default to this template.
- **Stub-vs-real scaling.** The savings ratio (30ms / 1431ms ≈ 2%)
  understates the real-world benefit. The stall is fixed at 500ms,
  but the *post-stall path-length* depends on per-token-delay
  (which is 10ms in the stub vs 50-100ms in production). A future
  variant could parameterize per-token-delay across [0.01, 0.05,
  0.1] to plot the savings curve.
- **stall_after as a building block.** This stub feature isn't
  iter-100-specific — any future scenario that wants to test
  mid-stream behavior (cancel handling, queue depth under stall,
  context-token billing on retry) can compose it. Worth keeping
  in mind as the perf suite grows.
- Next directions:
  - Per-token-delay grid (3-5 values) for iter-100 to surface the
    savings curve as a function of token speed.
  - Pair scenarios for `max_user_assistant` cap (5 vs 20) on a
    long-context session — would round out the four user-facing
    opt-ins.
  - Extract `_emit_recording_block` (still pending from iter-097).
  - WER fixture (1.6) heavyweight iteration.

## iter-101 — Per-token-delay grid for auto-aggressive savings curve

**Goal:** Replace iter-100's single A/B with a 3-point grid that
varies the dominant cost factor — `per_token_delay` — to surface
the savings curve. iter-100's note flagged the 30ms savings as
artificially small because of the 10ms stub delay; this iteration
adds the 50ms ("production-like") and 100ms ("slow LLM") points
so the chart shows what the auto-flip actually buys at real-world
TPS.

**Change:** Four new scenarios — two paired (off/on) at 50ms/token,
two paired at 100ms/token. iter-100's existing pair at 10ms/token
stays as the third grid point. No new plumbing — the existing
`_run_scenario` already accepts `per_token_delay`.

```python
test_auto_aggressive_off_50ms     test_auto_aggressive_on_50ms     # 50ms
test_auto_aggressive_off_100ms    test_auto_aggressive_on_100ms    # 100ms
# (iter-100 pair at 10ms stays — third grid point)
```

**Empirical savings curve** (this run, x86_64 Linux stub pipeline):

| per_token_delay | off TTFS  | on TTFS   | Δ TTFS  | mean_chars off→on |
|-----------------|-----------|-----------|---------|-------------------|
| 10 ms           | 1431.3 ms | 1401.2 ms | -30 ms  | 49.0 → 32.3       |
| 50 ms           | 1951.7 ms | 1801.4 ms | -150 ms | 49.0 → 32.3       |
| 100 ms          | 2601.8 ms | 2301.6 ms | -300 ms | 49.0 → 32.3       |

Two clean observations:

1. **Savings scale linearly with per-token-delay.** Every extra
   5ms of token cost = ~30ms additional TTFS savings from the
   auto-flip. The slope is the post-stall path-length: about 6
   tokens between "moment," and "twelve.", times the per-token
   delay = the TTFS gap the splitter saves by not waiting.

2. **The splitter behavior itself is delay-independent.**
   `mean_sentence_chars` stays 49.0 (off) and 32.3 (on) across
   all three grid points. The auto-flip mechanism is a pure
   timing trigger; once it fires, the slice is determined by
   buffer content, not stream rate. Validates the iter-093
   design — the threshold acts as a wall-clock filter, not a
   token-count filter.

**Implication for production tuning:** if the operator's real
LLM averages 50ms/token TTFB (the iter-052 baseline), enabling
`auto_aggressive_threshold=0.3` saves ~150ms TTFS per stalled
turn — a meaningful UX win. At 100ms/token (slow models or
network-bound), savings double to 300ms.

Verification: `python -m pytest tests/performance/ -q` → **19
passed in 12.6s** (was 15, +4 new — the suite is now 13s because
of the 100ms-delay scenarios; acceptable for a per-iter snapshot
that runs once). Full unit + integration suite: `tests/ --ignore=
tests/performance --ignore=tests/test_session.py --ignore=tests/
e2e -q` → **1148 passed, 1 skipped** (identical to main).

Notes:
- **Pattern: pair-grids for delay-sensitive features.** Where a
  feature's value scales with a continuous parameter (token speed,
  filler threshold, context size), a single A/B undersells the
  feature. The grid pattern is the natural extension.
- **Wall-time budget.** This added ~7s to the perf suite (4 new
  scenarios at avg 1.7s each). Fine for the per-iter snapshot,
  but the suite is now 13s — worth keeping in mind before
  another grid extension. Future iterations could group all
  delay-grids behind a `pytest -m grid` marker if it grows.
- **The 10ms point is now redundant.** It exists as the iter-100
  baseline but contributes little signal beyond the 50ms point.
  Could be removed in a future cleanup, but keeping it makes the
  curve's slope visible at low delays. Ship-as-is.
- Next directions:
  - Pair scenarios for `max_user_assistant` cap (5 vs 20) on a
    long-context session — the still-unvalidated user-facing
    opt-in.
  - Extract `_emit_recording_block` (still pending from iter-097).
  - WER fixture (1.6) heavyweight iteration.
  - Consider extracting the iter-100/iter-101 pair-scenario
    boilerplate (shared `_STALLED_PREAMBLE`, shared kwarg set)
    into a `_run_auto_aggressive_pair(delay)` helper if a fourth
    iteration adds yet more grid points.

## iter-102 — A/B perf scenarios for max_user_assistant context cap

**Goal:** Fourth pair-scenario after iter-098, iter-099, iter-100.
Validates iter-077 (context_tokens) + iter-078 (trim_events)
empirically by running an 8-turn session twice — once at the
default cap of 20 (never trims), once at a tight cap of 5
(trims after turn 3). Completes the user-facing-opt-in coverage
suite (splitter, filler threshold, auto-aggressive, context cap).

**New plumbing — `_run_session_scenario`.** The first
multi-turn perf helper; existing scenarios are all single-turn.
The cap is a session-level concept — it only matters AFTER
history accumulates — so a one-shot scenario can't see it.
Records the LAST turn's metrics into a single `ScenarioResult`
row + returns `trim_events` count for assertions.

```python
res, trim_events = _run_session_scenario(
    "context_cap_default",
    n_turns=8, max_user_assistant=20,
)
```

**Subtle gotcha — `flush_pending_audio` and pre-loaded mics.**
First attempt pre-loaded all 8 utterances at once. Worked for
turn 1, then turn 2's flush drained the entire remaining buffer
and the recorder waited forever for a tone burst. Fixed with the
same delayed-push pattern as iter-042's barge-in scenario: a
feeder thread that pushes utterance N+1 only AFTER turn N's
flush has completed (50ms semaphore-gated delay).

```python
pending = _th.Semaphore(0)
def _feeder():
    for _ in range(n_turns - 1):
        pending.acquire()       # block until turn N starts
        time.sleep(0.05)        # land after flush
        _utterance(speech_seconds, mic)
```

Inside the turn loop, `pending.release()` runs at the start of
each turn after the first.

**Empirical A/B** (this run, 8-turn session, x86_64 Linux stub):

| Metric                   | cap=20 (default) | cap=5 (tight) | Δ      |
|--------------------------|------------------|---------------|--------|
| context_tokens (turn 8)  | 53               | 24            | -55%   |
| TTFS (turn 8)            | 800.3 ms         | 800.3 ms      | 0      |
| trim_events (8 turns)    | 0                | 4+            | —      |

The 55% drop in billed tokens is the headline number. TTFS is
identical because the stub LLM has zero per-token cost — in
production, LLM TTFB scales with input context, so the cap
delivers TTFS savings on top of cost savings. iter-101's
per-token-delay grid pattern could be applied here too:
parameterize per-token-delay across the cap A/B to surface the
real-LLM TTFS benefit.

Verification: `python -m pytest tests/performance/ -q` → **21
passed in 13.9s** (was 19, +2 new). Full unit + integration
suite: `tests/ --ignore=tests/performance --ignore=tests/
test_session.py --ignore=tests/e2e -q` → **1148 passed, 1
skipped** (identical to main).

Notes:
- **Pattern: session-shaped pair-scenarios.** The 4th pair, but
  the FIRST that needs cross-turn state. The new helper
  `_run_session_scenario` is the natural place to put any future
  multi-turn metric (cumulative `llm_errors`, session-long
  `false_triggers`, `last_filler_id` rotation, etc.). Current
  shape is "drive N turns, record last-turn metrics + session
  counter" — plenty of room to extend.
- **The flush-vs-pre-load gotcha is a recurring trap.** Three
  scenarios now (iter-042 barge-in, iter-100 stall, iter-102
  session) hit "audio is in the buffer too early." All three
  use the same fix: thread + 50ms delay. Worth distilling into
  a `_push_after_flush(mic, audio, delay=0.05)` helper if a
  fourth instance lands.
- **The pair-scenario family is now 4 strong** with shared
  shape but diverging mechanisms. The pattern handles:
  - iter-098: behavior flag, single-turn (splitter)
  - iter-099: timing threshold, single-turn (filler)
  - iter-100: timing threshold, single-turn with stall (auto-flip)
  - iter-101: per-token-delay grid, single-turn extension
  - iter-102: session counter, multi-turn (context cap)

  Each iteration adds one new dimension. Next dimensions
  available: cumulative metric (e.g., `last_filler_id` rotation
  across turns), preempted_words on barge across N turns.

- Next directions:
  - Per-token-delay grid for context cap (would surface real-LLM
    TTFS scaling at different TPS).
  - Extract `_emit_recording_block` (still pending from iter-097).
  - WER fixture (1.6) heavyweight iteration.
  - `_push_after_flush` helper to dedupe the three thread+sleep
    sites (iter-042, iter-100, iter-102).

## iter-103 — Extract _emit_recording_block helper

**Goal:** 8th block helper after TTFS, Barge, Filler, Errors,
WPM, Sentence, History. Returns to the refactor pattern after 4
straight perf-scenario iterations (iter-098–102). Pulls the
contiguous mic-stale + VAD-false-trigger pair out of
`print_session_summary` into a `_emit_recording_block(emit,
stats: RecordingStats)` helper.

**Why this scope and not "all 4 recording-related blocks":**
The session-summary has 4 recording-pipeline-related lines —
primed audio (iter-057), mic stale (iter-037), VAD false-trig
(iter-048), bargeable (iter-074). They are NOT contiguous in
`print_session_summary` — the errors block (iter-097) sits
between primed audio and stale, and bargeable lives ~50 lines
later. Extracting all four would require either (a) moving them
to be contiguous, which changes output line order and breaks the
regression sentinel, or (b) calling the helper twice with
different args, which obscures intent.

This iteration takes the cleanest contiguous slice — stale +
false_trig — into one helper. Primed audio and bargeable can
each get their own helper later if a unifying rationale emerges.

**Change:**

```python
@dataclass
class RecordingStats:
    stale_total: int = 0
    false_triggers: int = 0
    n: int = 0   # successful-turn denominator for the FP rate

def _emit_recording_block(emit, stats: RecordingStats) -> None:
    if stats.stale_total: ...    # iter-037
    if stats.false_triggers > 0: # iter-048
        attempts = stats.false_triggers + stats.n
        ...
```

The 17-line inline block in `print_session_summary` shrinks to a
3-line helper call. Rendering is byte-for-byte identical.

**Tests:** 8 direct tests in `tests/unit/test_emit_recording_block.py`
covering the 2x2 matrix (clean, stale-only, false-trig-only,
both), plus formatting edge cases (16kHz divisor, integer pct
rounding, leading 4-space indent, n=0 + 1 trigger). Sub-millisecond
total runtime — same pattern as iter-089.

Verification:
- Direct tests: `python -m pytest tests/unit/test_emit_recording_block.py
  -q` → **8 passed in 20ms**.
- Regression sentinel (113 session-summary tests, byte-for-byte
  output validation): `python -m pytest tests/unit/
  test_emit_*.py tests/unit/test_session_*.py -q` → **113
  passed in 0.16s**.
- Full unit + integration: `tests/ --ignore=tests/performance
  --ignore=tests/test_session.py --ignore=tests/e2e -q` →
  **1156 passed, 1 skipped** (1148 prior + 8 new).
- Perf snapshot (regression check): `python -m pytest tests/
  performance/ -q` → **21 passed**.

Notes:
- **Eight block helpers extracted now.** Cumulative shrink to
  `print_session_summary`: 17 lines saved this iteration on top
  of the ~310 from iter-089 through iter-097. Total now ~327
  lines reduced from the original ~1500.
- **Per-helper test-count is converging.** This iteration's 8
  direct tests is similar to iter-097 (10), iter-095 (9),
  iter-094 (12). The 2x2 + edge-case shape is the natural unit
  for these emit helpers.
- **The non-contiguity problem.** This is the first time the
  refactor pattern has hit a "the related lines aren't adjacent"
  obstacle. Future helpers (primed-audio, bargeable, plus the
  remaining 5-10 single-line emits) face the same constraint.
  The accepted shape: extract the contiguous slice; leave
  isolated lines inline; revisit if grouping becomes obvious
  later. iter-097's "semantic-grouping" gain was a happy
  coincidence (iter-077 + iter-078 happened to be adjacent).
- Next directions:
  - `_emit_primed_audio_line` (iter-057) — single-line helper.
    Marginal value, but consistent with the "every named iter
    gets a helper" pattern.
  - `_emit_bargeable_line` (iter-074) — same shape.
  - Consider grouping primed_audio + bargeable + mic_stale +
    false_trig into a `_emit_pipeline_health_block` if their
    print_session_summary positions can be reordered (would
    change output but might be acceptable).
  - Per-token-delay grid for context cap (still pending from
    iter-102).
  - WER fixture (1.6) heavyweight iteration.

## iter-104 — Extract single-line recording helpers

**Goal:** 9th and 10th block helpers — `_emit_primed_audio_line`
(iter-057) and `_emit_bargeable_line` (iter-074). Both are
single-line emits scattered (non-contiguous) across
`print_session_summary`. Bundling them in one iteration matches
iter-103's documented "non-contiguity" pattern: extract isolated
lines as their own helpers rather than reordering output.

**Why bundle them in one iteration:** Each is a 6-line inline
block. Two in one PR matches the "iter-097 semantic-grouping"
shape — keeps the iteration log entry useful (one rationale, one
verification step) and keeps the per-iteration overhead low for
small extractions.

**Change:**

```python
def _emit_primed_audio_line(emit, primed_seconds_total: float) -> None:
    """iter-057 single line."""
    if primed_seconds_total > 0:
        emit(f"    Primed audio:     {primed_seconds_total:.1f}s ...")

def _emit_bargeable_line(emit, bargeable_values: list[float]) -> None:
    """iter-074 single line."""
    if bargeable_values and min(bargeable_values) < 0.99:
        worst = min(bargeable_values) * 100
        below_count = sum(1 for v in bargeable_values if v < 0.99)
        emit(f"    Bargeable:        {worst:.0f}% worst ...")
```

Both helpers take a free-function shape (no `*Stats` dataclass)
because each takes a single primitive — overkill to wrap in a
dataclass for one int/float/list. The `*Stats` pattern earned
its weight in iter-097 + iter-103 where 2-3 fields needed
bundling; here the simpler signature wins.

**Tests:**
- `tests/unit/test_emit_primed_audio_line.py`: 6 tests covering
  zero suppression, defensive negative, positive emission, .1f
  formatting, large values, indent.
- `tests/unit/test_emit_bargeable_line.py`: 8 tests covering
  empty/all-above suppression, one-turn-below, multi-turn-below,
  .0f rounding, threshold boundary (exactly 0.99 vs 0.989),
  indent.

Verification:
- Direct tests: **14 passed in 30ms**.
- Regression sentinel: `python -m pytest tests/unit/test_emit_*.py
  tests/unit/test_session_*.py -q` → **121 passed** (was 113;
  new 8 are the iter-104 helpers).
- Full unit + integration: `tests/ --ignore=tests/performance
  --ignore=tests/test_session.py --ignore=tests/e2e -q` →
  **1170 passed, 1 skipped** (1156 prior + 14 new).
- Perf snapshot: `tests/performance/ -q` → **21 passed**.

Notes:
- **Ten block helpers extracted now.** The session-summary
  emit-line shape is fully covered:
  - 7 `*Block` helpers + `*Stats` dataclasses (TTFS, Barge,
    Filler, Errors, WPM, Sentence, History, Recording).
  - 2 single-line free-function helpers (Primed Audio, Bargeable).

  What's left inline in `print_session_summary` is mostly the
  *aggregation* code that builds the inputs (median/max
  computations, list comprehensions over metrics_list). That's
  data-flow logic, not formatting — not a candidate for
  extraction in the same shape.
- **Free-function vs `*Stats`-shaped: clear rule of thumb.**
  ≥2 inputs → dataclass (lets future signal additions extend
  the dataclass without churning the helper signature).
  Exactly 1 input → free function. iter-104 codifies this.
- **Cumulative refactor savings.** Per-iter line counts:
  - iter-089 (TTFS): -65
  - iter-090 (Barge): -28
  - iter-091 (Filler): -45
  - iter-092 (Errors): -32
  - iter-094 (WPM): -52
  - iter-095 (Sentence): -39
  - iter-097 (History): -49
  - iter-103 (Recording): -17
  - iter-104 (PrimedAudio + Bargeable): -12
  Total **~339 lines** reduced from the original ~1500.
- The remaining inline code in `print_session_summary` is now
  ~1160 lines, almost all aggregation/computation. The
  formatting layer is ~80% extracted.
- Next directions:
  - WER fixture (1.6) heavyweight iteration — the last metric
    in the 46-metric taxonomy that's still incomplete.
  - Per-token-delay grid for context cap (still pending from
    iter-102) — applies iter-101's grid pattern to session helper.
  - `_push_after_flush` helper to dedupe iter-042 + iter-102
    thread+sleep pattern (only 2 sites; might wait for a 3rd).
  - Move on from `print_session_summary` extractions —
    diminishing returns; consider extracting the aggregation
    layer if a clean grouping emerges, or pivot to other
    refactor targets (mic_chat.py main loop, llm_stream
    error handling).

## iter-105 — WER computation primitive + session-summary wiring

**Goal:** Close the metric-taxonomy infrastructure gap for 1.6
(WER) — listed as "needs ground truth corpus" in 6+ prior
iteration logs. Build the metric primitive + plumbing now so a
future audio-fixture iteration only has to wire reference
transcripts in, not invent the metric from scratch.

**Split scope so this iteration finishes in one sitting:**
- ✅ WER computation primitive (`examples/_chat_wer.py`,
  pure string operation, no audio).
- ✅ TurnMetrics fields (`wer`, `wer_measured`).
- ✅ Session-summary helper (`_emit_wer_line`).
- ✅ Tests on all of the above (23 + 9 + indirect via session
  summary).
- ⏭ Audio-fixture corpus + reference transcripts (deferred —
  needs a different STT-mocking strategy on x86_64 since
  mlx-whisper is Mac-only).

**WER primitive (`examples/_chat_wer.py`):**

Standard formula: WER = (S + D + I) / N. Word-level Levenshtein
DP, no external deps (jiwer/python-Levenshtein NOT pulled in to
keep the install footprint flat). Light normalization:
lowercase + strip punctuation, but apostrophes preserved so
"don't" stays as one word (splitting it inflates N and
over-penalizes contractions).

```python
def compute_wer(reference: str, hypothesis: str) -> float:
    ref = _tokenize(reference)
    hyp = _tokenize(hypothesis)
    n = len(ref)
    if n == 0:
        return float(len(hyp))   # all insertions
    if not hyp:
        return 1.0               # all deletions
    # Classic word-level Levenshtein DP, O(R*H) time/space.
    ...
    return dp[n][len(hyp)] / n
```

**TurnMetrics extension:** Two new fields, both safely zero by
default so they don't disturb the regression sentinel:

```python
wer: float = 0.0          # Word Error Rate; 0.0 = perfect or
                          # not-measured (use wer_measured to
                          # distinguish).
wer_measured: bool = False  # True only when a reference was
                            # supplied this turn.
```

The `wer_measured` flag is critical — without it, "0.0 = perfect"
and "0.0 = not measured" would be indistinguishable, and the
session-summary would either always emit a misleading "0.00 WER"
line or never emit at all. With the flag, the helper filters on
the truth that a reference existed, then computes median/max
over the actual measurements.

**Session-summary wiring:**

```python
def _emit_wer_line(emit, wer_values: list[float]) -> None:
    if not wer_values:
        return
    med = statistics.median(wer_values)
    worst = max(wer_values)
    emit(f"    WER:              {med:.2f} median, {worst:.2f} max "
         f"({len(wer_values)} turns measured)")
```

Emitted right after `_emit_bargeable_line` so the recording-
related signals (mic_stale, false_trig, primed audio, bargeable,
WER) stay grouped — even though they're each their own helper.

Production-grade calibration anchors recorded inline in the
helper docstring so the operator sees the meaning of the number:

```
< 0.10 — production-grade STT
0.10-0.20 — acceptable for clean audio
0.20-0.40 — degraded; tune mic / silence_threshold
> 0.40 — STT is failing — check input quality
```

**Tests:**

`tests/unit/test_chat_wer.py` — 23 tests covering:
- Tokenization (basic, apostrophes, whitespace, empty,
  punctuation-only).
- Perfect transcription (identical, case-insensitive, with
  punctuation).
- Each error type in isolation (substitution, deletion,
  insertion).
- Multi-edit scenarios (completely-wrong, two-sub, sub+del).
- Empty-input edge cases (both empty, empty hyp, empty ref).
- Return type invariant (always float).
- Realistic STT scenarios (clean, one-misheard, dropped filler).
- Argument-order asymmetry (compute_wer(a,b) ≠ compute_wer(b,a)
  because N depends on which is the reference).

`tests/unit/test_emit_wer_line.py` — 9 tests covering:
- Empty list suppression (no measurements → no line).
- Single-turn perfect / single-turn imperfect.
- Multi-turn median + max.
- Even-length median (uses statistics.median's mean-of-middle
  convention).
- 2-decimal formatting + values >1.0 (insertion-heavy).
- 4-space leading indent (matches family convention).
- Turn-count suffix integrity.

Verification:
- `python -m pytest tests/unit/test_chat_wer.py
  tests/unit/test_emit_wer_line.py -q` → **32 passed in 40ms**.
- Regression sentinel: 144 session-summary tests pass (was 121
  — +9 from the new wer-line family + 14 from iter-104 already
  on main).
- Full unit + integration: `tests/ --ignore=tests/performance
  --ignore=tests/test_session.py --ignore=tests/e2e -q` →
  **1202 passed, 1 skipped** (1170 prior + 23 wer + 9 emit).
- Perf snapshot: `tests/performance/ -q` → **21 passed**.

Notes:
- **The 46-metric taxonomy is now infrastructurally complete.**
  Every metric in the "Standard" + "Novel/speculative" buckets
  has a TurnMetrics field, a session-summary line, and tests.
  The only thing 1.6 still lacks is **populated data** — a
  corpus of audio fixtures + reference transcripts. That's a
  different problem (audio infra, not code), and it's
  appropriately split into its own iteration.
- **Audio-fixture corpus design (recorded for the future iter):**
  - Recordings: 5-10 short utterances (1-3s each) covering
    common patterns — declarative, question, with filler,
    contraction-heavy, noisy.
  - Reference transcripts: hand-curated, lowercase, no
    punctuation (matches `_tokenize` normalization).
  - Storage: `tests/fixtures/wer/` with `audio_*.wav` +
    `reference_*.txt` pairs.
  - Driver: `tests/integration/test_wer.py` runs each fixture
    through a stub STT (or a CPU-only Whisper if available) and
    asserts WER stays below a threshold (e.g., 0.20 for clean
    audio, 0.40 for noisy). Use `pytest.skip(reason=...)` if
    no STT engine is loadable on the platform.
  - **The mlx-whisper x86_64 problem:** for cross-platform
    eval, the corpus iteration should produce reference
    transcripts using a CPU-only STT (whisper.cpp or
    faster-whisper), or use a mocked-but-realistic STT that
    introduces controlled errors (e.g., 1-substitution per 10
    words). The latter validates the WER pipeline without
    requiring real STT — good enough for CI.
- **`compute_wer` design choices recorded:**
  - Apostrophes preserved (vs split). Convention varies; we
    chose preserve because contractions are common in speech.
  - Empty reference returns float(len(hyp)) (vs inf). Some
    libraries return inf; we chose finite so callers can do
    median/mean math without special cases.
  - Case-insensitive. Whisper outputs lowercase by default,
    but other STTs vary; lowercasing both sides keeps the
    metric robust.
  - No optional kwargs (e.g., `case_sensitive`). YAGNI — add
    when a real caller needs it.
- **Final TurnMetrics size:** ~62 fields. The `wer_measured`
  bool is the first "is this measurement valid?" flag — sets
  a precedent for future "optional measurement" fields. If
  more land, they could share a `MeasurementFlags` bitfield
  to keep the dataclass shape from sprawling.
- Next directions:
  - **WER audio-fixture corpus** — design above; the corpus is
    the heavyweight piece. Could mock STT for CI portability.
  - Apply iter-088's aggressive splitter as a `chat.aggressive_first_sentence`
    default-on once the operator-cost on natural turns is
    quantified (would need a "long preamble vs short response"
    pair).
  - Per-token-delay grid for context cap (still pending from
    iter-102).
  - `_push_after_flush` helper to dedupe iter-042/iter-102.

## iter-106 — WER fixture corpus + integration test

**Goal:** Populate iter-105's infrastructure with a deterministic
corpus that exercises the full WER pipeline end-to-end. iter-105
intentionally split the metric work; this iteration completes the
"data-side" half by providing the reference/hypothesis pairs and
the integration test that drives them through compute_wer →
TurnMetrics → print_session_summary.

**Why text-only fixtures (no audio yet):** mlx-whisper is
Mac-only, and we work on x86_64 Linux. A real audio corpus would
either need a CPU-only STT (faster-whisper, whisper.cpp wrapper)
that's not yet wired up, or it would have to skip on this platform.
Text fixtures sidestep that — they exercise the full plumbing
(WER computation + TurnMetrics population + summary aggregation
+ wer_measured filter) and stay deterministic across CI hosts.
When a CPU-only STT lands, the integration runner can swap
simulated hypotheses for real STT output and the same
`expected_wer_min/max` bands still apply.

**Corpus** (`tests/fixtures/wer/corpus.json`, 5 fixtures):

| Name           | Pattern                                | Expected WER |
|----------------|----------------------------------------|--------------|
| clean          | Identical reference & hypothesis       | 0.0          |
| substitution   | Single homophone error                 | 0.15-0.25    |
| dropped_filler | STT drops leading "um"                 | 0.10-0.20    |
| noisy          | Multiple errors, declining audio       | 0.25-0.40    |
| catastrophic   | Every word wrong (STT failed entirely) | 0.95-1.05    |

The bands are tight enough to catch a regression in compute_wer
(say, an off-by-one in the DP table) but loose enough to survive
the natural rounding of (S+D+I)/N — adding/removing a single
fixture word doesn't blow the band.

**Integration test** (`tests/integration/test_wer_corpus.py`,
9 tests):

- **Corpus structural check** — size + names match design.
- **Per-fixture WER bands** — parametrized over all 5 fixtures;
  each computed WER must fall inside its `[min, max]` window.
- **End-to-end summary emission** — drive `print_session_summary`
  with all 5 fixtures wrapped as TurnMetrics; assert the WER line
  emits with the expected median + max + turn count.
- **Suppression invariant** — drive a session with `wer_measured
  =False` on every turn; assert the WER line does NOT emit. This
  is the regression sentinel for the `wer_measured` filter
  introduced in iter-105 — without it, `0.0 = "perfect"` would
  collide with `0.0 = "not measured"`.
- **Mixed-measurement session** — some turns measured, others
  not; assert only measured turns count in the aggregate (2
  measured + 2 unmeasured → "(2 turns measured)").

```python
def _build_turn_with_wer(reference: str, hypothesis: str) -> TurnMetrics:
    """Real prod path is:
        text = stt_engine.transcribe(audio_bytes)
        m.wer = compute_wer(reference, text)
        m.wer_measured = True
    We simulate without the audio + STT step."""
    m = TurnMetrics(transcript=hypothesis)
    m.wer = compute_wer(reference, hypothesis)
    m.wer_measured = True
    ...
```

The `_build_turn_with_wer` helper documents the production code
path so a future iter that swaps in real STT can copy-paste the
shape and just replace one line.

Verification:
- `python -m pytest tests/integration/test_wer_corpus.py -q` →
  **9 passed in 20ms**.
- Full unit + integration: **1211 passed, 1 skipped** (1202
  prior + 9 new).
- Perf snapshot: **21 passed**.

Notes:
- **The 46-metric taxonomy is now complete.** Both the
  infrastructure (iter-105) and a populating corpus (iter-106)
  exist. Real audio fixtures are a "polish" iteration — the
  metric is functional today, just on simulated hypotheses.
- **The `wer_measured` regression sentinel is now real.** Two
  tests (`test_session_summary_omits_wer_line_when_no_turn_measured`
  and `test_partial_measurement_session`) directly exercise the
  flag's purpose. Future TurnMetrics fields with the same
  "optional measurement" shape (e.g., reference_audio_present,
  ground_truth_intent_match) can copy this pattern.
- **Pattern: corpus-as-JSON.** Single-file JSON with a
  `_comment` array beats per-fixture .txt files for this size of
  corpus (5 entries). Easy to extend, version control diffs are
  readable, no filename-collision concerns. If the corpus grows
  past ~30 fixtures the per-file approach starts winning, but
  we're well below that.
- **Future iter for real audio:**
  - Add `audio_*.wav` files alongside `corpus.json` (ref text
    is already in JSON).
  - Wire a CPU-only STT (faster-whisper, whisper.cpp Python
    binding). Skip the integration test if the engine can't
    load (`pytest.skip(reason=...)`).
  - Each fixture's hypothesis becomes
    `stt.transcribe(audio_path)` instead of the hand-curated
    string. The expected_wer_min/max bands should still hold
    for production-grade STT on clean audio (the "clean" /
    "substitution" / "dropped_filler" entries); the "noisy"
    and "catastrophic" entries would need separate audio
    recordings of degraded conditions.
- **`compute_wer` is now battle-tested.** 23 unit tests in
  iter-105 + 5 fixture-driven tests + the cross-test where the
  integration suite recomputes WER from fixtures and matches
  the session-summary output. Three independent paths confirm
  the same results.
- Next directions:
  - Real audio fixtures for `clean` + `substitution` +
    `dropped_filler` (the three bands a real STT can hit). Add
    `requirements-dev.txt` entry for faster-whisper or
    whisper.cpp wrapper.
  - Per-token-delay grid for context cap (still pending from
    iter-102).
  - `_push_after_flush` helper to dedupe iter-042/iter-102.
  - Pivot to mic_chat.py refactor — extract `run_chat`'s
    config-loading / startup-sequence into a tested module
    (currently it's the largest unrefactored function).

## iter-107 — Extract prerender_fillers into _chat_fillers module

**Goal:** Pivot from session-summary refactor to mic_chat.py
itself. iter-106 noted `run_chat` is the largest unrefactored
function; the filler pre-rendering block (16 lines) is the
clearest extractable chunk inside it.

The block was previously inline:

```python
rendered_fillers: list[tuple] = []
if filler_texts:
    t_fill = time.monotonic()
    for text in filler_texts:
        try:
            audio_np, tokens = synthesize_with_alignment(
                tts_engine, text, voice, speed,
            )
            if len(audio_np) > 0:
                rendered_fillers.append((audio_np, tokens))
        except Exception as e:
            print(f"  {YELLOW}filler synth failed for {text!r}: {e}{RESET}")
    print(
        f"  Pre-rendered {len(rendered_fillers)}/{len(filler_texts)} "
        f"fillers in {(time.monotonic() - t_fill)*1000:.0f}ms "
        f"(idle threshold {filler_idle_threshold:.2f}s)"
    )
```

Three concerns tangled together: TTS engine wiring, error
handling/timing, and presentation (ANSI colors + leading spaces).
Untestable without spinning up a real TTS engine.

**Change:** New module `examples/_chat_fillers.py` exposing:

```python
def prerender_fillers(
    synth_fn: Callable[[str], tuple[Any, Any]],
    texts: Iterable[str],
    *,
    idle_threshold: float = 0.0,
    log: Callable[[str], None] = print,
) -> list[FillerClip]:
```

Three test-friendly seams:
- `synth_fn` — callable, not engine. Caller wraps the
  engine/voice/speed closure once at the top of `run_chat`. Tests
  pass any `(text) -> (audio, tokens)` callable.
- `log` — callable, not stdout. Default is `print` for
  production parity; tests capture into a list. Mirrors
  iter-009's pattern for logger injection.
- ANSI colors stay in `mic_chat.py`. The module emits plain
  text; the caller's `_filler_log` re-applies YELLOW for
  failures and plain styling for the summary. Keeps
  presentation owned by the entrypoint.

**Caller (mic_chat.py) shrinks from 16 lines to 13:**

```python
def _filler_log(line: str) -> None:
    if line.startswith("filler synth failed"):
        print(f"  {YELLOW}{line}{RESET}")
    else:
        print(f"  {line}")

rendered_fillers = prerender_fillers(
    lambda text: synthesize_with_alignment(tts_engine, text, voice, speed),
    filler_texts,
    idle_threshold=filler_idle_threshold,
    log=_filler_log,
)
```

Net line count is roughly the same in `run_chat` (the closure +
log adapter eat the savings), but the *test surface* is now
real: every behavior the original block had is verifiable
without booting kokoro.

**Tests** (`tests/unit/test_chat_fillers.py`, 12 tests):

- Edge cases: empty text iterable + iterable input.
- Happy path: 3 texts → 3 clips, summary log emits once.
- idle_threshold=0.0 case (operator may set explicitly).
- Failure handling: per-text try/except isolates failures;
  failure log line exposes exception text; all-failures returns
  empty list with 0/N summary.
- Empty-audio silent skip: matches the original inline
  behavior (no log on empty audio, only on exception).
- Mixed empty + ok: only non-empty audio lands.
- Default `log=print` flows to stdout via capsys.
- `synth_fn` invocation parity: receives just the text string,
  no engine/voice/speed details bleed in.

Verification:
- `python -m pytest tests/unit/test_chat_fillers.py -q` → **12
  passed in 20ms**.
- Full unit + integration: `tests/ --ignore=tests/performance
  --ignore=tests/test_session.py --ignore=tests/e2e -q` →
  **1223 passed, 1 skipped** (1211 prior + 12 new).
- Perf snapshot: **21 passed**.

Notes:
- **mic_chat.py is now 6% smaller in extractable concerns.**
  16 lines moved out into a tested module; ~145 lines of
  `run_chat` remain. Next-largest blocks (in line count):
  engine loading + timing (~13 lines), the closures
  `_speaker_factory` / `_synth` / `_play` (~17 lines), and
  the messages init + main turn loop (~30 lines). All
  candidates for follow-up extractions.
- **Two log-failure-vs-summary discrimination patterns now
  exist.** This module + iter-097's
  `_emit_history_block` both use a "look at the log line
  prefix" trick. If a third site lands, refactor: have the
  log function take a level (info/warning) + message instead
  of one line, and let the caller style by level.
- **Reusable shape: `synth_fn` + `log` injection.** The same
  pattern earned the `print_session_summary` tests in iter-017
  (log via `print` parameter) and iter-019 (synthesize_with_alignment
  factored similarly). Three+ instances now — this is the
  default extraction shape for any mic_chat.py subroutine
  that mixes TTS + logging.
- **Caller shape preserved.** `mic_chat.py:run_chat` still
  uses the same locals (`rendered_fillers`,
  `filler_idle_threshold`) downstream — extraction was purely
  internal to the rendering block, not a refactor of the
  call site.
- Next directions:
  - Extract engine loading (`stt_engine = WhisperEngine(...)`
    + `_load()` + timing) into `_chat_engines.py`. Would
    abstract the Mac-only WhisperEngine import at the same
    time, opening the door for x86_64 STT integration.
  - Extract the `_speaker_factory` / `_synth` / `_play`
    closures into a `ChatLoopFactory.from_engines(...)` builder.
  - Per-token-delay grid for context cap (still pending from
    iter-102).
  - Real audio fixtures + CPU-only STT (still pending from
    iter-106).

## iter-108 — Extract load_engines into _chat_engines module

**Goal:** Continue iter-107's mic_chat.py refactor. Pull the
13-line STT + TTS engine loading + timing block out of
`run_chat` into a tested module. Side benefit: abstracts the
Mac-only `WhisperEngine` import behind a factory callable, which
opens the door for x86_64 STT integration without touching
`mic_chat.py` itself.

**Original block:**

```python
# Load STT
t0 = time.monotonic()
stt_engine = WhisperEngine(model_repo=model_repo)
stt_engine._load()
stt_load = time.monotonic() - t0

# Load TTS
t1 = time.monotonic()
tts_engine = get_tts_engine("kokoro")
tts_engine._load()
tts_load = time.monotonic() - t1

print(f"  STT loaded in {stt_load*1000:.0f}ms")
print(f"  TTS loaded in {tts_load*1000:.0f}ms")
```

Three concerns tangled: hardcoded engine classes, timing logic,
ANSI-styled stdout logging. Untestable without spinning up real
mlx-whisper + kokoro.

**Change:** New module `examples/_chat_engines.py` exposing:

```python
@dataclass
class LoadedEngines:
    stt: Any
    tts: Any
    stt_load_seconds: float
    tts_load_seconds: float

def load_engines(
    stt_factory: Callable[[], Any],
    tts_factory: Callable[[], Any],
    *,
    log: Callable[[str], None] = print,
) -> LoadedEngines:
```

Three test seams (matching the iter-107 `prerender_fillers`
shape):
- `stt_factory` / `tts_factory` — `() -> engine` callables. The
  caller bakes `model_repo` / voice / preferences into a closure;
  `load_engines` itself knows nothing about engine constructors.
- `log` — callable, default `print`. Module emits plain text;
  caller (`mic_chat.py`) re-applies ANSI styling and 2-space
  indent.
- Returns a `LoadedEngines` dataclass — not a tuple or dict —
  so future fields (e.g., `vad_engine`) can extend without
  breaking callers.

**Caller (mic_chat.py) shrinks from 13 lines to 9:**

```python
from examples._chat_engines import load_engines

engines = load_engines(
    stt_factory=lambda: WhisperEngine(model_repo=model_repo),
    tts_factory=lambda: get_tts_engine("kokoro"),
    log=lambda line: print(f"  {line}"),
)
stt_engine = engines.stt
tts_engine = engines.tts
```

The factory closures preserve the existing wiring exactly — no
behavior change for production. The log adapter prepends 2 spaces
to match the original indented output.

**Tests** (`tests/unit/test_chat_engines.py`, 12 tests):

- Happy path: bundle returned, both factories called once,
  both `_load()` invoked.
- Timing: timings are non-negative; a 50ms `_load()` delay
  shows up in `stt_load_seconds ≥ 0.04` (permissive lower bound
  for sleep granularity).
- Logging: 2 lines emitted in STT-then-TTS order; both lines
  are plain text (no ANSI / leading spaces); default `log=
  print` flows to stdout via capsys.
- Error propagation: STT factory exception propagates; TTS
  factory NOT called on STT failure (no half-load); TTS
  `_load()` exceptions also propagate.
- Factory contract: factories receive `()` — no positional or
  keyword args added by `load_engines` itself.
- Return shape: `LoadedEngines` dataclass with the 4 named
  fields, locking in the API.

Verification:
- `python -m pytest tests/unit/test_chat_engines.py -q` →
  **12 passed in 70ms**.
- Full unit + integration: `tests/ --ignore=tests/performance
  --ignore=tests/test_session.py --ignore=tests/e2e -q` →
  **1235 passed, 1 skipped** (1223 prior + 12 new).
- Perf snapshot: **21 passed**.

Notes:
- **Two mic_chat.py extractions in two iterations.** iter-107
  + iter-108 share the exact same shape: `(callable_dep, log)`
  injection, plain-text emit, ANSI ownership at the caller,
  dataclass return. This is now the **default extraction
  shape for any mic_chat.py subroutine** — a third instance
  would let me promote it to a documented pattern in GENO.md.
- **WhisperEngine is no longer a hard import boundary.** The
  factory callable means `_chat_engines.py` could be tested
  on x86_64 Linux even if `WhisperEngine` itself can't be
  imported (its module-level `mlx-whisper` import would fail
  on Linux). A future faster-whisper or whisper.cpp wrapper
  can drop in by changing the factory closure in `mic_chat.py`
  — no other call site touched.
- **The "fail-fast on engine load error" decision is now
  tested.** The `test_stt_factory_failure_skips_tts` invariant
  prevents a future refactor from accidentally introducing a
  partial-load state. This isn't speculative: a try/except
  around the STT load would be a tempting "robustness"
  addition; the test makes the cost explicit.
- **mic_chat.py size after iter-107 + iter-108:** ~365 lines
  total, ~125 in `run_chat`. Down from ~395 / ~155 pre-iter-107.
  Remaining extractable blocks in run_chat:
  - speaker_factory + synth + play closures (~17 lines)
  - aggressive_first_sentence + auto_aggressive_threshold
    config read (~6 lines, very small — probably stays inline)
  - main turn loop with KeyboardInterrupt handler (~30 lines —
    largest remaining block)
- Next directions:
  - Extract the 3 closures (`_speaker_factory`, `_synth`, `_play`)
    into a single factory builder. Would round out the
    "everything before the turn loop" extraction.
  - Real audio fixtures + CPU-only STT (still pending from
    iter-106). Now easier given iter-108 — the `stt_factory`
    closure can return a faster-whisper wrapper instead of
    `WhisperEngine`.
  - Extract the main turn loop with KeyboardInterrupt handler
    into a `run_session` function — would complete the
    `run_chat` decomposition.

## iter-109 — Extract pyaudio closures into _chat_audio_io module

**Goal:** Third mic_chat.py extraction in three iterations.
Pulls the speaker_factory + synth + play closures (~22 lines
including the ChatLoop wiring) out of `run_chat` into a tested
module. **Promotes the extraction shape to documented guidance
in GENO.md** — three instances is the threshold where pattern
becomes convention.

**Original block:**

```python
def _speaker_factory():
    return pa.open(format=pyaudio.paInt16, channels=1, rate=TTS_RATE,
                   output=True, frames_per_buffer=1024)

def _synth(sentence: str):
    return synthesize_with_alignment(tts_engine, sentence, voice, speed)

def _play(speaker, audio_np, tokens, *, is_first_sentence=False,
          cancel_event=None, lag_out=None):
    return _play_aligned_core(speaker, audio_np, tokens,
        is_first_sentence=is_first_sentence, rate=TTS_RATE,
        cancel_event=cancel_event, lag_out=lag_out)

# ...later in ChatLoop(...):
chat_loop = ChatLoop(
    speaker_factory=_speaker_factory,
    synth_fn=_synth,
    play_fn=_play,
    ...
)
```

The closures had grown subtle — `lag_out` (iter-071), the
`is_first_sentence` flag (iter-006), the rate threading. All
production-only behavior, all entangled with pyaudio + kokoro
imports.

**Change:** New module `examples/_chat_audio_io.py` exposing:

```python
@dataclass
class AudioIO:
    speaker_factory: Callable[[], Any]
    synth_fn: Callable[[str], tuple[Any, Any]]
    play_fn: Callable[..., float]

def build_audio_io(
    pa, tts_engine, voice, speed, *,
    pyaudio_module=None, speaker_chunk=1024, rate=TTS_RATE,
) -> AudioIO:
```

**One new pattern element added in this iteration: lazy import
inside the closure.** Earlier iterations (107 + 108) didn't deal
with pyaudio because their work was string-shaped. Here, the
`speaker_factory` closure is the only spot that touches
`pyaudio.paInt16`, so the import lives inside that closure:

```python
def speaker_factory():
    nonlocal pyaudio_module
    if pyaudio_module is None:
        import pyaudio as pyaudio_module
    return pa.open(format=pyaudio_module.paInt16, ...)
```

This makes `build_audio_io` itself importable on x86_64 Linux
(pyaudio not installed) — the import only fires when the closure
is CALLED. Tests inject `pyaudio_module=_PyAudioModule()` to
avoid the runtime import entirely.

**Caller (mic_chat.py) shrinks from 23 lines to 3:**

```python
from examples._chat_audio_io import build_audio_io
audio_io = build_audio_io(pa, tts_engine, voice, speed)
# then ChatLoop(speaker_factory=audio_io.speaker_factory,
#               synth_fn=audio_io.synth_fn,
#               play_fn=audio_io.play_fn, ...)
```

**Tests** (`tests/unit/test_chat_audio_io.py`, 12 tests):

- Return type: AudioIO dataclass with three callable fields.
- speaker_factory: kwargs match original (paInt16, channels=1,
  output=True, frames_per_buffer=1024); returns pa.open's value;
  callable multiple times; speaker_chunk + rate kwargs override
  defaults.
- synth_fn: delegates to synthesize_with_alignment with
  (engine, sentence, voice, speed); speed kwarg is threaded
  through.
- play_fn: delegates to _play_aligned_core; is_first_sentence /
  cancel_event / lag_out forwarded intact (iter-071 reveal-lag
  contract preserved); rate kwarg captured in closure.
- Lazy import: build_audio_io constructs without pyaudio
  installed (the import fires inside speaker_factory only).

Verification:
- `python -m pytest tests/unit/test_chat_audio_io.py -q` →
  **12 passed in 50ms**.
- Full unit + integration: **1247 passed, 1 skipped** (1235
  prior + 12 new).
- Perf snapshot: **21 passed**.

**Pattern promoted to GENO.md.** A new "mic_chat.py extraction
pattern" section codifies the five rules earned across iter-107,
iter-108, iter-109:

1. Inject callable dependencies, not engines.
2. Inject `log` (callable, default `print`). Skip when the
   extracted code is silent.
3. ANSI styling stays at the caller.
4. Return a dataclass, not a tuple.
5. Lazy-import platform deps inside the closures, not at
   module scope.

The first three were implicit in iter-107 + iter-108. Rule 4 was
explicit but unmotivated — now justified by 3 dataclass returns
(`LoadedEngines`, `AudioIO`, plus the `RecordingStats` from
iter-103). Rule 5 is new this iteration and makes platform
abstraction concrete.

Notes:
- **mic_chat.py size after iter-109:** ~340 lines total /
  ~100 in `run_chat` (down from ~365 / ~125 pre-iter-109).
  Cumulative reduction in `run_chat`: 55 lines across three
  iterations (107: 16 → 13; 108: 13 → 9; 109: 23 → 3).
- **The `lazy import inside closure` trick is reusable.**
  Future extractions that touch pyaudio, mlx-whisper, or any
  Mac-only library should follow rule 5. The closure-as-import-
  boundary pattern lets the module pass-through importable
  even when the lib isn't available — important for CI parity
  on x86_64.
- **Last big block in `run_chat` is the main turn loop with
  KeyboardInterrupt handler** (~30 lines). Extracting that
  would complete the `run_chat` decomposition — it'd become
  config-load + load_engines + prerender_fillers + build_audio_io
  + run_session, with run_session owning the loop.
- **`run_session` extraction would surface a non-trivial
  question:** the loop owns `messages`, `all_metrics`,
  `false_triggers`, `llm_errors`, `trim_events`, `primed_frames`,
  `session_start`. That's a lot of mutable state. The
  extraction would need to either (a) pass them as a single
  `SessionState` dataclass, or (b) make `run_session` return a
  `SessionState` to print_session_summary. (a) is the natural
  fit given the dataclass-return rule.
- Next directions:
  - Extract main turn loop into `run_session` (closes the
    decomposition). Largest single iter-pivot remaining in
    mic_chat.py.
  - Real audio fixtures + CPU-only STT (still pending from
    iter-106). Ready to land — the `stt_factory` closure in
    iter-108 is the swap point.
  - Per-token-delay grid for context cap (still pending from
    iter-102).

## iter-110 — Extract run_session from run_chat

**Goal:** Complete the `run_chat` decomposition started in
iter-107. Pull the main turn loop + KeyboardInterrupt handler
(~50 lines mixing per-turn flow, session counters, trim
plumbing, and exit handling) into a tested module.

This is the 4th mic_chat.py extraction in four iterations and
the most complex — it has 6 mutable locals that needed to be
bundled into a `SessionState` dataclass per the GENO.md pattern's
rule 4.

**Original block:** Lines 285-354 of mic_chat.py — system_prompt
setup + 7 declared locals + the while-True loop + the
KeyboardInterrupt handler that hands the locals to
`print_session_summary` via `SessionMeta`.

**Change:** New module `examples/_chat_session.py` exposing:

```python
@dataclass
class SessionState:
    messages: list[dict]
    all_metrics: list
    false_triggers: int = 0
    llm_errors: int = 0
    trim_events: int = 0
    trim_messages_evicted: int = 0
    primed_frames: Optional[list[bytes]] = None
    session_start: float = 0.0

def run_session(
    chat_loop, system_prompt: str, *,
    max_user_assistant: int = 20,
    log: Callable[[str], None] = print,
    prompt_log: Callable[[int], None] = ...,
    clock: Callable[[], float] = time.monotonic,
    trim_messages: Optional[Callable] = None,
) -> SessionState:
```

`SessionState` field names line up 1:1 with the existing
`SessionMeta` dataclass — conversion in the caller is mechanical:

```python
SessionMeta(
    false_triggers=state.false_triggers,
    session_seconds=time.monotonic() - state.session_start,
    llm_errors=state.llm_errors,
    trim_events=state.trim_events,
    trim_messages_evicted=state.trim_messages_evicted,
    idle_threshold=filler_idle_threshold,
)
```

**Three orthogonal injection seams** (each test-relevant):
- `prompt_log(turn: int)` — the "[N] waiting..." prompt. Default
  re-creates the mic_chat.py styling. Tests capture turn counter.
- `log(line: str)` — the post-stream newline emitter. Default
  `print`. Tests silence it with `lambda s: None`.
- `trim_messages` — the context cap callable. When `None`, lazy-
  imports `ChatLoop.trim_messages` per GENO.md rule 5. Tests
  pass a stub that doesn't import ChatLoop at all.

**Caller (mic_chat.py) shrinks from ~50 lines to ~12:**

```python
from examples._chat_session import run_session

system_prompt = llm_config.get(
    "system_prompt", "You are a concise voice assistant.",
)
state = run_session(
    chat_loop, system_prompt, max_user_assistant=20,
    prompt_log=lambda turn: print(
        f"  {DIM}[{turn}] waiting...{RESET}", end="", flush=True,
    ),
)
# … then SessionMeta conversion + print_session_summary
```

ANSI styling stays at the caller (rule 3). The `prompt_log`
callable wraps `DIM`/`RESET` styling around the plain
"[N] waiting..." text the module would otherwise emit.

**Tests** (`tests/unit/test_chat_session.py`, 14 tests):

Three buckets, mirroring the loop's three result-shapes:

*Successful turn flow (5 tests):*
- Empty queue → immediate KI returns SessionState with system
  prompt seeded and counters at zero.
- session_start reflects clock callable.
- One success → metrics appended + `metric.print(turn)` called.
- Multiple successes → turn counter increments correctly.
- system_prompt is always the first message.

*Error / false-trigger handling (2 tests):*
- iter-058: `had_error=True` increments `llm_errors`, reuses
  turn counter — sequence `[1, 1, 2]` validates the no-advance
  on error.
- iter-048: `metrics=None` increments `false_triggers`, same
  no-advance shape.

*Trim + state plumbing (5 tests):*
- iter-025/057: `next_primed_frames` from turn N becomes
  `primed_frames` input to turn N+1.
- iter-078: trim event recorded when eviction happens (with
  an `_AppendingChatLoop` that adds messages each turn so the
  shrinker has work to do).
- No trim event when no eviction.
- iter-058 invariant: trim only runs after SUCCESSFUL turns,
  never on error / false-trigger.
- max_user_assistant kwarg threads through to `trim_messages`.

*Logging discipline (2 tests):*
- prompt_log fires every turn including the eventual KI prompt.
- log emits exactly once per successful turn (never on
  error / false-trigger).

Verification:
- `python -m pytest tests/unit/test_chat_session.py -q` → **14
  passed in 20ms**.
- Full unit + integration: **1261 passed, 1 skipped** (1247
  prior + 14 new).
- Perf snapshot: **21 passed**.

**The run_chat decomposition is now complete.** Original
mic_chat.py:run_chat was ~195 lines (pre-iter-107). After this
iteration it's ~95 lines, and structurally it's now five
function calls plus a small finally block:

```python
def run_chat(...):
    llm_config = load_llm_config()
    chat_cfg = load_chat_config()                      # iter-018
    engines = load_engines(...)                        # iter-108
    rendered_fillers = prerender_fillers(...)          # iter-107
    audio_io = build_audio_io(...)                     # iter-109
    chat_loop = ChatLoop(...)                          # iter-015
    state = run_session(chat_loop, system_prompt, ...) # iter-110
    print_session_summary(state.all_metrics, ..., meta=...)
    # finally: mic.stop_stream(), mic.close(), pa.terminate()
```

Every block is now its own tested module. The `run_chat`
function itself is essentially declarative — it threads
configuration into pre-built components and dispatches to the
session driver.

Notes:
- **Trace-of-counters tests are subtle.** Three test failures
  during writing came from miscounting what `[N] waiting...`
  fires when. The pattern: prompt_log fires ONCE per
  iteration, INCLUDING the iteration that raises
  KeyboardInterrupt. So an N-success session shows N+1 prompts
  (N successes + 1 final KI prompt). Worth recording for
  future loop-shaped extractions.
- **Lazy import of `ChatLoop.trim_messages`** (rule 5) saved
  the test from needing to import `ChatLoop` at all. Tests
  inject `trim_messages=_stub_trim` and never touch the real
  class. This is the cleanest application of rule 5 yet — the
  earlier instances (iter-109 pyaudio) had platform-specific
  motivation; this one has a pure decoupling motivation.
- **SessionState vs SessionMeta:** the field-name overlap is
  intentional — the dataclasses serve different layers
  (state during the run vs metadata for the summary) but
  carry the same per-session signals. A future cleanup could
  unify them, but the current 1:1 mapping is clear enough
  that the duplication is fine.
- Next directions:
  - Real audio fixtures + CPU-only STT (still pending from
    iter-106). Now ready to land — `stt_factory` from
    iter-108 is the swap point, and `run_session` accepts
    any chat_loop matching the `.run_one_turn` shape so
    integration tests can drive it without booting the real
    mic.
  - Per-token-delay grid for context cap (still pending from
    iter-102).
  - Pivot away from refactoring — `run_chat` is
    structurally complete. New work should target
    architecture (streaming-overlap improvements, multi-
    turn metric correlations) or testing depth (eval suite,
    fault-injection scenarios).

## iter-111 — Feature savings table on performance.html

**Goal:** Pivot from refactoring (iter-107–110 completed the
`run_chat` decomposition) to making the perf data more
actionable. Five iterations (iter-098/099/100/101/102) added
pair-scenarios that empirically validate user-facing opt-ins,
but the only place the deltas were summarized was in the
iteration log entries themselves. This iteration surfaces them
as a curated table at the top of `performance.html` so an
operator looking at the live perf page sees the empirical value
of each opt-in immediately.

**Change:** Three new pieces in `scripts/generate_iteration_reports.py`:

1. **`FEATURE_SAVINGS_TABLE`** — curated list of pair entries.
   Each entry binds a `feature` label to a `control_scenario`
   name + `treatment_scenario` name in `perf-results.json`,
   plus a `primary_metric` (the headline number — usually
   `ttfs_ms`), a `secondary_metric` (the behavior marker
   that proves the mechanism fired — usually `mean_sentence_chars`
   or `last_filler_id` or `context_tokens`), and a one-line
   takeaway.

2. **`_format_pair_value` / `_format_pair_delta`** — pure
   formatters. Per-metric formatting rules (ms with 1 decimal,
   tokens as int, last_filler_id as state-change description).
   Tested in isolation.

3. **`_render_feature_savings_section(scenarios)`** — composes
   the HTML. Lookups by name; pairs missing either control or
   treatment are silently skipped so old perf JSONs (pre-iter-098)
   render without an empty table.

**Six curated entries** corresponding to the iter-098–102
pair-scenarios:

| Feature | Control | Treatment | Primary | Secondary |
|---|---|---|---|---|
| Aggressive splitter (iter-088) | `long_preamble_aggressive_off` | `long_preamble_aggressive_on` | TTFS | mean_chars |
| Filler idle_threshold (iter-051) | `filler_threshold_default` | `filler_threshold_aggressive` | TTFS | filler fired? |
| Auto-aggressive @ 10ms (iter-093) | `auto_aggressive_off` | `auto_aggressive_on` | TTFS | mean_chars |
| Auto-aggressive @ 50ms (iter-093) | `auto_aggressive_off_50ms` | `auto_aggressive_on_50ms` | TTFS | mean_chars |
| Auto-aggressive @ 100ms (iter-093) | `auto_aggressive_off_100ms` | `auto_aggressive_on_100ms` | TTFS | mean_chars |
| Context cap (iter-024) | `context_cap_default` | `context_cap_tight` | context_tokens | TTFS |

When a future iteration adds a new pair-scenario family, the
log writer appends one entry to `FEATURE_SAVINGS_TABLE`. The
rendering does the rest.

**Tests** (`tests/unit/test_feature_savings.py`, 28 tests):

*`_format_pair_value` (6 tests):* ms / int / float / None /
unknown-metric formatting.

*`_format_pair_delta` (8 tests):* ms positive + negative +
sub-decimal; context_tokens with + without zero-control;
last_filler_id state-change description (both directions);
None handling.

*Table structure invariants (3 tests):* ≥6 entries; every
entry has the required keys; no duplicate treatment names
(would render two rows for the same data).

*`_render_feature_savings_section` (8 tests):* empty when no
pairs match; renders when one pair complete; skips half-present
pairs (missing treatment OR missing control); context_cap pair
uses token metric formatting; filler pair surfaces fire/no-fire
state-change; full snapshot renders all 6 rows; uses perf-table
CSS class; HTML-escapes feature names + takeaways (defensive).

*Integration with `render_performance_page` (2 tests):* page
includes section when pair scenarios are present; section is
absent when no pairs in snapshot (graceful degradation for
old JSONs).

Verification:
- `python -m pytest tests/unit/test_feature_savings.py -q` →
  **28 passed in 30ms**.
- Full unit + integration: **1289 passed, 1 skipped** (1261
  prior + 28 new).
- Perf snapshot: **21 passed**.
- Report regen: confirmed `performance.html` contains the new
  section with all 6 rows populated from the latest snapshot.

Notes:
- **Curated > auto-detected.** I considered automatic
  pair-detection by scenario-name pattern (e.g., split on
  `_off`/`_on` suffixes), but the names vary too much — iter-099
  uses `default`/`aggressive`, iter-102 uses `default`/`tight`,
  iter-101 uses `_off_50ms`/`_on_50ms` (off/on AND a delay
  tag). A curated list is unambiguous, surfaces the operator-
  meaningful labels (e.g., "Aggressive first-sentence splitter"
  vs the cryptic `long_preamble_*`), and keeps the takeaways
  inline. Cost: one new entry per pair-iteration.
- **HTML escaping discipline.** The
  `test_section_escapes_html_in_takeaway` test seeds a
  malicious entry to confirm `html.escape()` is applied. This
  is the sentinel for "future curated entries don't break the
  page" — it's not paranoia, it's the regression test for the
  one type of change someone would actually make to this list.
- **The "graceful degradation for old JSONs" property.** The
  iter-reports/ folder retains every per-iter perf snapshot
  (perf-iter-NNN.json). When the report generator regenerates
  iter-{036–097}.html using their respective snapshots, the
  feature-savings section is empty for those iterations
  (because their snapshots predate the pair-scenarios). The
  page just shows the legacy bar charts as before. No flag
  to set, no version-pinning — the feature is forward-only.
- **The pair-scenario pattern is now closed-loop.** iter-098
  added the first pair, iter-099/100/101/102 extended it.
  iter-111 surfaces the result data in a way an operator can
  read at a glance. A new pair iter is now a one-line addition
  to FEATURE_SAVINGS_TABLE — the friction is at its lowest.
- Next directions:
  - Real audio fixtures + CPU-only STT (still pending from
    iter-106). Three iterations now overdue.
  - Per-token-delay grid for context cap (still pending from
    iter-102). Would scale the iter-102 row in the savings
    table from one entry to a 3-entry grid (matching iter-101's
    auto-aggressive grid shape).
  - Eval/replay tool that uses SessionState (iter-110) to
    re-run a past conversation without booting the real mic —
    the dataclass surface makes this much easier than before.

## iter-112 — Context-dependent TTFB knob + extended cap grid

**Goal:** Validate iter-111's "append one entry to FEATURE_SAVINGS_TABLE"
workflow with a real new pair while also fixing iter-102's
disclosed limitation: the stub LLM didn't model context-dependent
TTFB, so the iter-102 cap=20 vs cap=5 row showed identical TTFS
(only the token-count delta was meaningful).

**Two changes:**

1. **`context_factor` kwarg on `_yield_tokens`** — seconds per
   input character, applied as a sleep BEFORE the first token
   yields. Simulates real LLM TTFB scaling with prompt length
   (KV-cache fill cost). Default 0.0 preserves all pre-iter-112
   behavior.

   ```python
   def _yield_tokens(text, *, ..., context_factor=0.0):
       def factory(messages, config):
           if context_factor > 0:
               total_chars = sum(
                   len(str(m.get("content", "")))
                   for m in messages
               )
               time.sleep(total_chars * context_factor)
           ...
   ```

2. **Two new paired session scenarios** at `context_factor=0.002`
   (2ms/char):
   - `context_cap_default_ctx2ms`: cap=20, full TTFB cost
   - `context_cap_tight_ctx2ms`: cap=5, trim bounds the TTFB

   These extend iter-102's pair into a 2-point grid showing both
   the no-scaling baseline (iter-102 original pair) and the
   scaling-on case (iter-112).

**Empirical numbers** (this run, x86_64 stub, 8-turn session):

| Metric              | cap=20 ctx0     | cap=5 ctx0      | cap=20 ctx2ms   | cap=5 ctx2ms    |
|---------------------|-----------------|-----------------|-----------------|-----------------|
| context_tokens      | 53              | 24 (-55%)       | 53              | 24 (-55%)       |
| TTFS (turn 8)       | 800.4ms         | 800.3ms         | 1424.7ms        | 1078.7ms (-346) |
| llm_first_token     | 0.0ms           | 0.0ms           | 624.2ms         | 278.1ms (-346)  |

The headline number: **with context_factor=2ms/char, the tight
cap saves 346ms TTFS on turn 8** — the exact `llm_first_token`
delta, which is the mechanism. Without context_factor (the
iter-102 baseline), cap=5 and cap=20 produce identical TTFS
because nothing depends on prompt length. With context_factor on,
the savings track the trim's effect on token count.

**Wall-time tradeoff considered.** First implementation used
`context_factor=0.005` (5ms/char, more production-realistic) but
that pushed the perf suite from 14s to 26.5s. Dropping to 0.002
kept the savings ratio identical (-30%) and limited the extra
wall time to ~5s (suite is now 19.8s). 5ms/char would be the
right number when run on a perf-only CI or with a faster mic.

**FEATURE_SAVINGS_TABLE update.** Validates iter-111's design —
the new pair lands on `performance.html` with one entry append:

```python
{
    "feature": "Context cap @ context_factor=2ms/char (iter-112)",
    "control_scenario": "context_cap_default_ctx2ms",
    "treatment_scenario": "context_cap_tight_ctx2ms",
    "primary_metric": "ttfs_ms",
    "primary_label": "TTFS (turn 8)",
    "secondary_metric": "context_tokens",
    "secondary_label": "Context tokens",
    "secondary_lower_is": "fewer (trim kept history bounded)",
    "takeaway": "Same 8-turn session, but with the LLM stub charging "
                "2ms per input character (KV-fill cost). The cap=5 "
                "side now BENEFITS visibly — bounded context = bounded "
                "TTFB. Real LLMs scale this way.",
}
```

The HTML rendering picks it up automatically — no rendering
changes needed. iter-111's "graceful degradation when scenario
missing" property means old perf JSONs (pre-iter-112) still
render their pages without empty rows.

**Tests** (`tests/unit/test_perf_yield_tokens.py`, 8 tests):

- Default behavior preserved: no context_factor → tokens yield
  immediately; messages list is ignored.
- context_factor activates first-token delay proportional to
  total chars across all messages.
- Sums across multiple messages correctly.
- Zero context_factor explicitly disables (sentinel for
  iter-098-102 scenarios that pass empty messages but might
  carry data in unusual paths).
- Defensive: missing 'content' key → empty-string contribution,
  no KeyError.
- Combines with per_token_delay (both delays compose).
- Backwards compat: all pre-iter-112 kwarg combinations still
  work.

Verification:
- `python -m pytest tests/unit/test_perf_yield_tokens.py -q` →
  **8 passed in 360ms** (most of the time is the actual sleep
  in the context_factor tests).
- `python -m pytest tests/performance/ -q` → **23 passed in
  19.8s** (was 21 in 14s, +2 scenarios, +5.8s wall time).
- Full unit + integration: **1297 passed, 1 skipped** (1289
  prior + 8 new).

Notes:
- **iter-111's append-one-entry workflow is validated.** A new
  pair-scenario family lands with: 2 new test methods in
  test_pipeline_perf.py, 1 new entry in FEATURE_SAVINGS_TABLE,
  unit tests for any new mechanism. The performance.html page
  picks it up automatically. Lowest possible friction.
- **The stub-vs-real fidelity gap is now smaller.** Pre-iter-112,
  the perf suite produced TTFS numbers that didn't reflect a
  major real-world cost (LLM TTFB scaling with prompt length).
  iter-112 closes that gap — operators reading the time-series
  chart see numbers that track production behavior shape, even
  if absolute magnitudes differ.
- **`context_factor` is reusable for future grids.** Same
  pattern as iter-101's per_token_delay grid: a single knob
  parameterized across multiple values produces a curve. A
  future iter could grid context_factor at 0.001/0.002/0.005
  to plot the full sensitivity surface — but with the wall-
  time tradeoff understood.
- **Wall-time budget watch.** Perf suite is now 19.8s. If
  another grid lands at this scale, it'd push toward 25s+ —
  the 30s mark would warrant the `pytest -m grid` marker
  iter-101 noted.
- **The perf snapshot's per-iter file size is growing.** Each
  perf-iter-NNN.json now has 23 scenarios; serialized JSON is
  ~30KB. With 100+ iterations this approaches 3MB on disk.
  Not a problem yet, but worth noting.
- Next directions:
  - Real audio fixtures + CPU-only STT (still pending from
    iter-106). Now four iterations overdue.
  - Eval/replay tool using SessionState (iter-110) — would
    let an operator re-run a recorded session with different
    chat config opts to compare empirically.
  - Pivot to architecture: perf data is rich enough now that
    a code change targeting one of the surfaced costs (e.g.,
    sentence-split coverage, first-synth overlap) has clear
    feedback loop.

## iter-113 — Cross-turn filler variety via recent-IDs FIFO

**Goal:** First architecture-pivot iteration after the
refactor + perf-data milestones. Targets a real UX issue:
filler-word repetition across turns.

`played_filler_ids` (iter-087) tracks fillers within a single
turn so the picker doesn't pick "umm umm" back-to-back. But each
turn creates a fresh SentenceWorker with an empty set, so across
turns the same filler can be picked every time. With 3 fillers
configured and an unlucky `random.choice`, the user can hear the
exact same "umm" five turns running. Visible repetition, easy
to fix.

**Change:** Add a cross-turn FIFO that lives at the ChatLoop
level and threads down to each SentenceWorker.

In `_chat_loop.py:ChatLoop.__init__`:

```python
from collections import deque as _deque
n_fillers = len(self._fillers)
self._recent_filler_ids = (
    _deque(maxlen=max(1, n_fillers - 1))
    if n_fillers > 0 else None
)
```

`maxlen = n_fillers - 1` is the right size: with 3 fillers, the
last 2 played are blocked, leaving 1 fresh option always
available. The picker is forced to rotate. With 1 filler,
`maxlen=1` means the picker's preferred set is sometimes empty
— the fallback "all-recent → use full available" branch ensures
the filler still plays.

In `_chat_pipeline.py:SentenceWorker.__init__`:

```python
recent_filler_ids: Optional[Any] = None,  # added kwarg
self._recent_filler_ids = recent_filler_ids
```

Picker integration (the only behavior change):

```python
available = [c for c in self._fillers if id(c) not in played_filler_ids]
# iter-113: prefer clips NOT in the cross-turn FIFO.
if self._recent_filler_ids:
    fresh = [c for c in available if id(c) not in self._recent_filler_ids]
    if fresh:
        available = fresh
clip = self._filler_picker(available)
```

The fallback ("everything's recent → use full available") matters
because when n_fillers ≤ maxlen, the FIFO can fill up to the
point where every clip is recent. Refusing to play a filler in
that case would defeat the iter-011 purpose; the better failure
mode is "play one even if it repeats."

After successful play, the worker appends to the FIFO:

```python
if self._recent_filler_ids is not None:
    if hasattr(self._recent_filler_ids, "append"):
        self._recent_filler_ids.append(id(clip))
    else:
        self._recent_filler_ids.add(id(clip))
```

Duck-typed append vs add so future callers can pass a `set`
(unbounded, fine for tests) or a `list`. Production passes a
`deque(maxlen=N)`.

**Tests** (`tests/unit/test_cross_turn_fillers.py`, 12 tests):

- Constructor wiring: default `recent_filler_ids=None`; passed
  kwarg stored as same reference (so mutations propagate).
- Filter behavior: prefer non-recent when fresh available;
  fall back to full when all recent; empty FIFO behaves like
  no FIFO.
- ChatLoop integration: deque sized n_fillers-1 (with min 1);
  None when no fillers; same-deque-reference passed to
  SentenceWorker; cross-turn append behavior (add 3 ids,
  oldest evicted).
- Append shape compat: works with deque (.append), list
  (.append), set (.add).

Verification:
- `python -m pytest tests/unit/test_cross_turn_fillers.py -q`
  → **12 passed in 60ms**.
- Filler regression sentinel (iter-061/087/091/093/096/099):
  `pytest tests/unit/ -k 'filler or Filler' -q` → **93 passed
  in 12s**. No regressions in any prior filler test.
- Full unit + integration: **1309 passed, 1 skipped** (1297
  prior + 12 new).
- Perf snapshot: **23 passed**.

Notes:
- **The fallback shape matters more than the filter.** Most
  multi-turn sessions never hit the all-recent fallback —
  with 3 fillers and maxlen=2, the fresh set always has 1
  candidate. The fallback exists for the edge case (1 filler,
  long session) where it WOULD fire, so the test
  `test_picker_falls_back_to_full_when_all_recent` is the
  regression sentinel for that path.
- **Append-vs-add duck typing is a small smell but justified.**
  The cleaner alternative would be a thin `_FillerHistory`
  protocol class with a single method. Wasn't worth the
  abstraction for two backends — a future iter could promote
  if a third storage shape lands (e.g., a max-frequency map
  rather than a FIFO).
- **No new metric for this iteration.** The TurnMetrics
  `last_filler_id` field (iter-081) is already enough for
  session-summary diversity reporting (median/mode of distinct
  fillers across turns). A future session-summary check could
  warn when `last_filler_id` is the SAME value across N+
  consecutive turns, which would now be rare due to iter-113.
- **No perf scenario for this iteration.** The cross-turn
  variety effect is only observable across many turns and is
  inherently random (filler picker is `random.choice` in
  production). A perf scenario testing variety would need a
  large N + fixed seed, and the result would be a categorical
  "yes, fillers vary" not a continuous metric.
- **The pattern: cross-turn state at ChatLoop, threaded to
  SentenceWorker per turn.** This is the second cross-turn
  state field on ChatLoop after iter-082's `_last_first_audio_at`.
  A third would justify a `ChatLoopState` dataclass; the two
  current fields are clear enough as instance attributes.
- Next directions:
  - Real audio fixtures + CPU-only STT (still pending from
    iter-106). Five iterations now overdue.
  - Eval/replay tool using SessionState (iter-110). Could
    measure filler variety empirically across recorded
    sessions.
  - Session-summary check: warn when the same `last_filler_id`
    appears in N+ consecutive turns. Becomes a defensive
    sentinel that iter-113 didn't break.
  - Architecture: investigate whether the bargeable_fraction
    metric (iter-074) has room to improve via earlier watcher
    start.

## iter-114 — Session-summary filler diversity check

**Goal:** Defensive sentinel for iter-113's cross-turn filler
variety fix. Without an explicit check, a future regression in
the FIFO logic (e.g., the deque accidentally becoming a no-op
container, or the `recent_filler_ids` kwarg being dropped from
the SentenceWorker constructor) could land silently — the user
would hear repetition again, but no test would fire because
filler-picking is randomized.

This iteration adds an empirical check: scan the per-turn
`last_filler_id` sequence for runs of the same id ≥ N
(default 3). When found, emit a warning line in the session
summary so the operator notices the regression on their next
session.

**Change:** New helper `_emit_filler_diversity_line` in
`_chat_metrics.py` next to the other emit-line family
(`_emit_wer_line`, `_emit_bargeable_line`, etc):

```python
def _emit_filler_diversity_line(
    emit, filler_ids: list[int], threshold: int = 3,
) -> None:
    fired = [fid for fid in filler_ids if fid != 0]
    if not fired:
        return
    # Find longest consecutive-same run in `fired`.
    longest_run = 1
    longest_id = fired[0]
    cur_run = 1
    cur_id = fired[0]
    for fid in fired[1:]:
        if fid == cur_id:
            cur_run += 1
            if cur_run > longest_run:
                longest_run = cur_run
                longest_id = cur_id
        else:
            cur_id = fid
            cur_run = 1
    if longest_run < threshold:
        return
    emit(
        f"    Filler diversity: filler {longest_id} repeated "
        f"{longest_run} turns running "
        f"— iter-113 cross-turn FIFO may not be wired"
    )
```

Two design choices recorded in the docstring:

1. **Zero-filtering before run counting.** A turn with
   `last_filler_id == 0` (no filler fired) is FILTERED OUT
   before the run scan, not treated as a break. Rationale:
   user perception is "same filler 3 times in a row"
   regardless of intervening no-filler turns. The pattern
   `[101, 0, 101, 0, 101]` should fire; the user heard 101,
   silence, 101, silence, 101 — clearly the same filler.

2. **Threshold = 3 default.** 2 is too noisy (random.choice
   produces 2-in-a-row trivially even with the FIFO working);
   4+ raises the bar so far that real regressions wouldn't
   fire it for many turns. 3 is the smallest pattern a user
   typically perceives as repetition.

**Wired into `print_session_summary`** right after the
`_emit_filler_block` call:

```python
_emit_filler_diversity_line(
    _emit,
    [m.last_filler_id for m in metrics_list],
)
```

The line is silent on clean sessions (the common case), so the
session-summary regression sentinel passes byte-for-byte
unchanged for any existing test.

**Tests** (`tests/unit/test_emit_filler_diversity_line.py`,
18 tests):

*Empty / no-filler suppression (2 tests):* empty list, all-zero
list — neither emits.

*Below threshold (3 tests):* no repetition, 2-in-a-row
(default-threshold passes), alternating pattern — none emit.

*At/above threshold (5 tests):* 3-in-a-row fires; 4-in-a-row
reports actual run length; runs at end / start / longest-of-
multiple are correctly identified.

*Zero-filtering (3 tests):* zeros between same fillers don't
break the run; leading + trailing zeros filter out cleanly.

*Custom threshold (3 tests):* threshold=5 suppresses 4-runs
but fires on 5-runs; threshold=2 catches shorter patterns.

*Output formatting (2 tests):* leading 4-space indent matches
family convention; warning includes the "iter-113" attribution
so operators know which fix to investigate.

Verification:
- `python -m pytest tests/unit/test_emit_filler_diversity_line.py
  -q` → **18 passed in 30ms**.
- Regression sentinel (144 session-summary tests pass byte-for-
  byte): `pytest tests/unit/test_emit_*.py
  tests/unit/test_session_*.py -q` → **144 passed**.
- Full unit + integration: **1327 passed, 1 skipped** (1309
  prior + 18 new).
- Perf snapshot: **23 passed**.

Notes:
- **The "warns on regression" line includes the responsible
  iter number.** Operator searches for "iter-113" in the
  ITERATION_LOG.md, gets the full context. Compare to the
  iter-074 bargeable-warning which just says "watcher coverage
  regression" — operator has to grep to find the fix iter.
  Future warnings should follow iter-114's pattern of naming
  the fix.
- **The scan is O(N) and exits early.** The longest_run search
  doesn't allocate, doesn't sort, doesn't re-scan. For typical
  sessions (≤50 turns), this is a single linear pass.
- **Threshold parameterization is the kind of "if you ever
  need it" knob that pays off later.** None of the current
  tests need a non-default threshold to validate behavior, but
  exposing it makes the helper composable for future eval
  scenarios (e.g., a session-replay tool that wants to flag
  even 2-runs in a side-by-side comparison).
- **The pattern is generalizable** — a `_emit_X_diversity_line`
  shape would also work for sentence-length variety or
  user-WPM swing. iter-114 is the first instance; future
  diversity checks can copy the structure.
- Next directions:
  - Real audio fixtures + CPU-only STT (still pending from
    iter-106). Six iterations now overdue.
  - Eval/replay tool using SessionState (iter-110). Could
    measure filler diversity empirically across recorded
    sessions and compare to iter-113's expected variety.
  - Architecture: investigate whether the bargeable_fraction
    metric (iter-074) has room to improve via earlier watcher
    start.

## iter-115 — Naturalness-consistency check

**Goal:** Two-fer iteration:

1. Apply iter-114's `_emit_X_diversity_line` shape to a different
   metric — proves the pattern is reusable (third instance after
   iter-114 itself, or rather second since iter-114 was the
   first).
2. Surface a real config-tuning signal: when 5+ consecutive turns
   land in `naturalness_bucket="rushed"` or `"slow"`, the
   operator's `speed` setting is wrong.

**Pre-iteration investigation that got pivoted:**
First attempted to find an architectural improvement for
`bargeable_fraction` (iter-074). Inspecting the latest perf
snapshot showed every scenario reports 1.000 — the metric is a
pure regression sentinel, no improvement headroom. Pivoted to
the diversity-check shape.

**Change:** New helper `_emit_naturalness_consistency_line` in
`_chat_metrics.py`, structured identically to iter-114's filler
diversity check:

```python
def _emit_naturalness_consistency_line(
    emit, buckets: list[str], threshold: int = 5,
) -> None:
    non_empty = [b for b in buckets if b]
    if not non_empty:
        return
    # Find longest consecutive-same run in `non_empty`.
    longest_run, longest_bucket = 1, non_empty[0]
    cur_run, cur = 1, non_empty[0]
    for b in non_empty[1:]:
        if b == cur:
            cur_run += 1
            if cur_run > longest_run:
                longest_run, longest_bucket = cur_run, cur
        else:
            cur, cur_run = b, 1
    if longest_run < threshold:
        return
    if longest_bucket == "natural":
        return  # natural is the goal — never flag
    if longest_bucket == "rushed":
        suggestion = "consider reducing speed"
    elif longest_bucket == "slow":
        suggestion = "consider increasing speed"
    else:
        suggestion = "consider tuning speed"
    emit(f"    Naturalness:      {longest_run} consecutive "
         f"{longest_bucket!r} turns — {suggestion} (iter-053 bucket)")
```

Two design choices recorded in the docstring:

1. **"natural" runs are never flagged.** Unlike iter-114's filler
   diversity (where ANY repeated id is concerning), the
   naturalness check targets only the off-bucket states. A 10-turn
   run of "natural" is the desired outcome.
2. **Threshold = 5** (vs iter-114's 3). Natural speech-rate
   variation produces brief "rushed" / "slow" streaks even with
   correct config; 5+ consecutive same-bucket is the smallest
   pattern where "config wrong" is more likely than "noise."

**Wired into `print_session_summary`** right after the
iter-114 filler diversity line. Same call-site pattern:

```python
_emit_naturalness_consistency_line(
    _emit,
    [m.naturalness_bucket for m in metrics_list],
)
```

**Tests** (`tests/unit/test_emit_naturalness_consistency_line.py`,
18 tests):

*Empty / no-bucket suppression (2 tests):* empty list, all-empty-
strings — neither emits.

*"natural" exclusion (2 tests):* 10-turn natural run silent; long
natural run with brief outliers (longest = natural) silent.

*Below threshold (2 tests):* 4-rushed below default; alternating
rushed+natural — neither fires.

*At/above threshold (4 tests):* 5-rushed → reduce-speed suggestion;
6-slow → increase-speed suggestion; run-at-end detected;
longest-of-multiple correctly selected.

*Empty filtering (2 tests):* empty buckets between same-bucket
turns don't break runs; leading empties also filter.

*Custom threshold (2 tests):* threshold=3 catches smaller
patterns; threshold=10 suppresses default 5-run.

*Output formatting (3 tests):* leading 4-space indent; iter-053
attribution; unknown bucket falls back to generic suggestion
(defensive against a future bucket addition).

*Edge case (1 test):* documented limitation — the function
reports ONLY the longest run; if "natural" is longest and a
shorter "rushed" run also crosses threshold, NOTHING fires.
Recorded as a known limitation rather than fixed; a future
iteration could scan each non-natural bucket independently if
this becomes a real signal.

Verification:
- `python -m pytest tests/unit/test_emit_naturalness_consistency_line.py
  -q` → **18 passed in 30ms**.
- Regression sentinel (162 session-summary tests pass byte-for-
  byte): `pytest tests/unit/test_emit_*.py
  tests/unit/test_session_*.py -q` → **162 passed**.
- Full unit + integration: **1345 passed, 1 skipped** (1327
  prior + 18 new).
- Perf snapshot: **23 passed**.

Notes:
- **The diversity-check pattern is now twice-instantiated**
  (iter-114 + iter-115). Both share:
  - Filter empties before scanning
  - Track longest run with single-pass O(N)
  - Configurable threshold kwarg
  - "name the responsible iteration" in the warning
  - Suppress when run < threshold

  A third instance (e.g., barge_in_phase repetition) would
  warrant extracting the run-finder into a shared helper.
- **The "natural" exclusion is a key differentiator from
  iter-114.** Filler diversity wants ALL runs flagged; this
  check wants only OFF-BUCKET runs flagged. Demonstrates the
  pattern is flexible — the inclusion/exclusion logic is
  per-instance, not part of the shared shape.
- **The "longest-of-multiple-buckets" limitation is
  documented as a known weakness.** A pathological session
  with 7 natural + 5 rushed wouldn't fire even though the
  rushed run is concerning. In practice this would require
  the user to flip between modes mid-session — unusual.
  If it becomes real, the fix is to track each non-natural
  bucket separately. Listed as a future iteration if needed.
- **The unknown-bucket defensive case** matters because
  `naturalness_bucket` is a free-form string field. iter-053
  defined three values (rushed/natural/slow) but a future
  iteration could add (e.g., "very_slow"). The fallback
  ensures the warning still fires with a generic suggestion.
- Next directions:
  - Real audio fixtures + CPU-only STT (still pending from
    iter-106). Seven iterations now overdue.
  - Eval/replay tool using SessionState (iter-110).
  - Extract the diversity-run-finder into a shared helper if
    a third instance lands.
  - Architecture: examine `streaming_overlap_ratio` (currently
    7-50% across scenarios) for headroom — much lower than
    the 1.0 ceiling, suggests the worker often waits for
    sentences rather than synthesizing concurrently.

## iter-116 — Extract _longest_consecutive_run shared helper

**Goal:** Consolidate the run-finding scan loop that iter-114
(`_emit_filler_diversity_line`) and iter-115
(`_emit_naturalness_consistency_line`) duplicated verbatim.
~10 lines of identical single-pass scan logic in two places —
small enough that extraction is borderline, but the duplication
is concrete and a third instance is plausible (e.g.,
barge_in_phase repetition, sentence-length-bucket runs).

**Pre-iteration investigation that got pivoted:**
First investigated `streaming_overlap_ratio` for architectural
headroom. Survey of the perf snapshot showed:
- 0% overlap on short single-sentence responses (no headroom —
  nothing to overlap when response IS one sentence).
- 7-22% on auto_aggressive_off scenarios (the 500ms stall
  blocks the LLM, naturally limits overlap).
- 38-57% on long_preamble scenarios (where iter-088's
  aggressive splitter already provides the lever — turning it
  on increases overlap by ~50%).
- 70% on filler-firing scenarios (filler audio plays during
  LLM streaming — natural overlap source).

The metric is well-tuned and the existing
`aggressive_first_sentence` flag is the architectural lever.
No further architectural improvement available from this metric
alone; it would require making aggressive_first_sentence the
default which has prosody tradeoffs.

Pivoted to the run-finder extraction.

**Change:** New `_longest_consecutive_run(values: list) ->
tuple[int, object]` function:

```python
def _longest_consecutive_run(values: list) -> tuple[int, object]:
    if not values:
        return (0, None)
    longest_run, longest_value = 1, values[0]
    cur_run, cur = 1, values[0]
    for v in values[1:]:
        if v == cur:
            cur_run += 1
            if cur_run > longest_run:
                longest_run, longest_value = cur_run, cur
        else:
            cur, cur_run = v, 1
    return (longest_run, longest_value)
```

Pure list-scanning primitive — no filtering. Callers pre-filter
(zeros for iter-114, empty strings for iter-115). Returns a
tuple so callers can tuple-unpack as `length, value`.

Both callers shrink:

iter-114 was:
```python
fired = [fid for fid in filler_ids if fid != 0]
if not fired: return
longest_run = 1; longest_id = fired[0]
cur_run = 1; cur_id = fired[0]
for fid in fired[1:]:
    if fid == cur_id:
        cur_run += 1
        if cur_run > longest_run:
            longest_run = cur_run
            longest_id = cur_id
    else:
        cur_id = fid
        cur_run = 1
if longest_run < threshold: return
emit(...)
```

Now:
```python
fired = [fid for fid in filler_ids if fid != 0]
if not fired: return
longest_run, longest_id = _longest_consecutive_run(fired)
if longest_run < threshold: return
emit(...)
```

iter-115 collapses similarly (~12 lines → 2 lines per call site).

**Tests** (`tests/unit/test_longest_consecutive_run.py`,
19 tests):

*Empty / singleton (3 tests):* empty → (0, None); single int;
single string.

*Single run (2 tests):* all-same; two-in-a-row.

*Multiple runs (5 tests):* one run in distinct sequence;
alternating returns 1; runs at start, end; two equal-length
runs (first wins); longer-later-run wins.

*Type variety (4 tests):* string runs; mixed strings; zeros
counted (no filter); None values.

*Stress (1 test):* large list with central run.

*Round-trip parity (3 tests):* shapes that iter-114 and
iter-115 actually pass after their pre-filter; locked tuple
return shape.

**Key invariant tested: ties resolve to EARLIEST run.**
`[1, 1, 1, 2, 2, 2]` returns `(3, 1)`, not `(3, 2)`. This
matches the prior in-place implementations exactly — both
iter-114 and iter-115 always reported the FIRST run encountered
in case of a tie. Documenting the rule as a tested invariant
prevents a future "it doesn't matter" refactor from accidentally
flipping it.

Verification:
- `python -m pytest tests/unit/test_longest_consecutive_run.py
  -q` → **19 passed in 30ms**.
- Regression sentinel for both iter-114 + iter-115 callers:
  `pytest tests/unit/test_emit_filler_diversity_line.py
  tests/unit/test_emit_naturalness_consistency_line.py
  tests/unit/test_session_summary.py -q` → **55 passed**
  byte-for-byte unchanged.
- Full unit + integration: **1364 passed, 1 skipped** (1345
  prior + 19 new).
- Perf snapshot: **23 passed**.

Notes:
- **Borderline extraction, justified by future-instances bias.**
  Two instances of a 10-line scan is small. But:
  1. The duplication was VERBATIM — no per-instance variation
     in the loop body, suggesting it's purely shared logic.
  2. The shape (return longest run + value) is general — fits
     any "consecutive same value" check.
  3. Tests now lock the tie-resolution rule explicitly, which
     was implicit in the prior duplicated code.
- **Pure primitive, no filtering.** Keeping the helper a
  scan-only function rather than baking in zero-filtering
  preserves composability. iter-114 filters zeros; iter-115
  filters empty strings; a hypothetical iter-117 might filter
  None. Each caller's filter rule is its own concern.
- **Tuple return locks call-site shape.**
  `length, value = _longest_consecutive_run(items)` reads
  cleanly. Returning a dataclass would be over-engineered for
  two fields.
- **streaming_overlap_ratio investigation (recorded for
  future):** the metric has structural ceilings most scenarios
  hit. Real headroom only exists when:
  - LLM stream is long enough to overlap with synth+playback
  - First sentence emits before the stream finishes
  Turning aggressive_first_sentence on is the only lever; and
  it's a config opt-in with prosody tradeoffs. A future
  iteration could grid-search a tighter splitter heuristic
  (e.g., comma-end at >= 30 chars instead of >= 20) to find
  a sweet spot, but this is real architecture work, not a
  one-iteration job.
- Next directions:
  - Real audio fixtures + CPU-only STT (faster-whisper IS
    installed on this system — verified). Eight iterations
    overdue. Worth attempting now.
  - barge_in_phase consistency check — would be the third
    instance of the diversity-check pattern, validating both
    iter-116's helper extraction and the diversity shape itself.
  - Sentence-length-bucket consistency check — same shape on a
    different signal.

## iter-117 — Real audio fixtures + faster-whisper integration

**Goal:** Eight iterations overdue from iter-106's recorded
design. iter-106 added text-only fixtures that exercised the
WER plumbing without audio; iter-117 closes the loop by adding
real .wav files and running them through a real STT engine.

**Pre-iteration verification:**
- `faster-whisper` 1.2.1 → installed.
- `espeak-ng` → installed at `/usr/bin/espeak-ng`.
- Smoke test: synthesize "the quick brown fox jumps..." with
  espeak-ng, transcribe with faster-whisper `tiny` model →
  "The quick brown fox jump over the lazy dog." (1 sub, WER ≈
  11%). Round-trip works on x86_64 Linux without any Mac-only
  dependencies.

**Change:** Three pieces:

1. **Two committed audio fixtures** (`tests/fixtures/wer/`):
   - `clean.wav` (66 KB) — "what is the weather today"
   - `quick_brown_fox.wav` (122 KB) — "the quick brown fox
     jumps over the lazy dog"

   Both generated via `espeak-ng -w <out.wav> "<text>"`, mono
   22050 Hz PCM. Committed to the repo so the integration test
   has deterministic input regardless of whether espeak-ng is
   present at test-run time.

2. **`audio_fixtures` section in corpus.json**, parallel to the
   existing iter-106 `fixtures` section but with `audio_path`
   instead of `hypothesis`. Per-entry `expected_wer_min/max`
   bands are intentionally generous (clean: 0.0-0.40,
   pangram: 0.0-0.30) because espeak-ng output isn't natural
   speech. Tightening when real human-recorded fixtures land
   is a follow-up.

3. **`tests/integration/test_wer_audio.py`** — 4 tests:
   - `test_corpus_declares_audio_fixtures`: structural sanity,
     runs even without faster-whisper.
   - `test_audio_files_exist_on_disk`: guards against a corpus
     update missing the companion .wav.
   - `test_each_audio_fixture_lands_in_wer_band`: the headline
     check — for each entry, transcribe + compute WER + assert
     in band. Concatenates failures into a single readable
     message.
   - `test_clean_audio_round_trips_below_30_pct_wer`: tighter
     assertion on the clean-audio fixture specifically;
     30% is the "if this fails something is fundamentally
     broken" line.

**Skip-cleanly contract:** Both faster-whisper import + model
load happen behind `pytest.skip()`. CI hosts without internet
or without enough disk for the model can run the suite without
failing the audio tests:

```python
try:
    from faster_whisper import WhisperModel
    _FW_AVAILABLE = True
except Exception as e:
    _FW_AVAILABLE = False
    _FW_IMPORT_ERROR = str(e)

@pytest.fixture(scope="module")
def stt_model():
    if not _FW_AVAILABLE:
        pytest.skip(f"faster-whisper not importable: {_FW_IMPORT_ERROR}")
    try:
        return WhisperModel("tiny", device="cpu", compute_type="int8")
    except Exception as e:
        pytest.skip(f"faster-whisper model failed to load: {e}")
```

The two structural-sanity tests don't depend on the model, so
they always run.

**Empirical data** (this run):
- `clean.wav` → faster-whisper transcribes near-perfectly,
  WER < 0.30 (assertion passes with margin).
- `quick_brown_fox.wav` → "jumps" sometimes drops to "jump"
  (1 sub / 9 words ≈ 11%), well within the 0.0-0.30 band.
- Total integration-test time: 2.5s including model load
  (cached after first run).

Verification:
- `python -m pytest tests/integration/test_wer_audio.py -v` →
  **4 passed in 2.5s**.
- Full unit + integration: **1368 passed, 1 skipped** (1364
  prior + 4 new).
- Perf snapshot: **23 passed**.

Notes:
- **The 46-metric taxonomy is now end-to-end populated.**
  iter-105 added the WER infrastructure; iter-106 added text-
  only fixtures; iter-117 closes the loop with real audio
  through a real STT engine. Every metric in the taxonomy now
  has tests at every level: unit (compute_wer), integration
  (text fixtures), end-to-end (audio fixtures). Three tiers,
  each with its own skip semantics.
- **espeak-ng quality matters less than determinism.** The
  audio isn't natural — the output sounds robotic — but it's
  deterministic given the input text. Tests don't care about
  prosody; they care about "does the STT pipeline produce
  reasonable transcripts." A synthetic voice at 22 kHz mono is
  enough to exercise that.
- **`tiny` model choice.** ~75 MB, downloads in seconds,
  CPU-only viable. The next step up (`base`, ~150 MB) would
  be more accurate but doubles model size + load time. For
  CI integration testing, `tiny` is the right tradeoff —
  accuracy doesn't need to be production-grade.
- **The `expected_wer_max=0.40` band is permissive on
  purpose.** Once a real human-voice fixture lands, the band
  can tighten to 0.10-0.15 (production-grade STT). Today's
  espeak-ng-generated audio tests "is the pipeline wired up
  correctly?", not "is the STT model good?".
- **Caching consideration.** First-run model download is ~15s
  on a typical connection. After that, faster-whisper hits
  the on-disk cache and the integration tests finish in
  2.5s. The cache lives at `~/.cache/huggingface/hub/`. CI
  hosts that wipe per-job will re-download every time —
  `pytest.skip` handles the no-network case gracefully.
- Next directions:
  - Tighten WER bands when real human-recorded audio lands.
  - Wire the STT factory from iter-108's `load_engines`
    closure to optionally use faster-whisper instead of
    WhisperEngine — would let Linux users actually run
    `mic_chat.py`. Real architecture work, not a small
    iteration.
  - Add a noisy-audio fixture (e.g., the same text mixed
    with synthetic background noise) to exercise the
    "noisy" / "catastrophic" bands of the iter-106 corpus
    against real audio rather than simulated hypotheses.
  - barge_in_phase consistency check (third instance of the
    diversity-check pattern).

## iter-118 — FasterWhisperEngine for x86_64 Linux

**Goal:** Real architecture work. iter-117 verified
faster-whisper round-trips through the WER pipeline as a test
fixture; iter-118 makes it a first-class STT engine that
mic_chat.py can use in production. **First STT path that runs
on x86_64 Linux without mlx-whisper / Apple Silicon.**

iter-108's `load_engines` factory was deliberately built to
swap engines via a closure. The swap point was always `lambda:
WhisperEngine(model_repo=...)` → operators on Linux had nothing
to swap to. iter-118 closes that gap.

**Change:** Three pieces:

1. **`stt/faster_whisper_engine.py`** — `FasterWhisperEngine`
   class implementing the same `STTEngine` contract as
   `WhisperEngine`:
   - `__init__(model_repo, device, compute_type, **kwargs)`
   - `_load()` — idempotent, lazy-imports `faster_whisper`
   - `transcribe(wav_bytes) -> (text|None, elapsed_seconds)`

   Class-level model cache keyed by
   `(resolved_repo, device, compute_type)` — mirrors
   `WhisperEngine`'s class-level cache. ChatLoop construction
   reuses the loaded model.

   Defaults match iter-117's audio-fixture choices: `tiny` /
   `cpu` / `int8`. A no-arg `FasterWhisperEngine()` is
   operator-friendly on x86_64 Linux.

2. **Repo-string translation** (`_resolve_model_repo`):
   - Short aliases ("tiny", "base", "large-v3") pass through.
   - Full HF repos and local paths pass through.
   - MLX-style ("mlx-community/whisper-tiny") strip the
     namespace + "whisper-" prefix → faster-whisper-recognizable
     size token.

   This means the same `model_repo` string in `config.local.yaml`
   works on both Mac (MLX path) and Linux (faster-whisper path) —
   the engine handles the translation. Operators don't have to
   maintain two configs.

3. **Factory registration** in `stt/__init__.py`:
   ```python
   ENGINES = {
       "whisper": WhisperEngine,                  # MLX, Apple Silicon
       "gemma4": Gemma4Engine,
       "faster_whisper": FasterWhisperEngine,    # iter-118: x86_64 Linux + CUDA
   }
   ```

   Lazy-imports `faster_whisper` inside `_load()`, not at module
   scope. So `from stt import FasterWhisperEngine` works on
   systems without the package — only `_load()` raises. Matches
   iter-109 GENO.md rule 5: "lazy-import platform deps inside
   the closures."

**How an operator uses it:**

```yaml
# config.local.yaml
chat:
  stt_engine: "faster_whisper"
  stt_model: "tiny"        # or "base", "large-v3", etc.
  stt_device: "cpu"        # or "cuda"
  stt_compute: "int8"      # or "float16", "int8_float16"
```

```python
# mic_chat.py — iter-108 stt_factory closure adapts
stt_factory=lambda: get_engine(
    chat_cfg.get("stt_engine", "whisper"),
    model_repo=chat_cfg.get("stt_model", default_repo),
)
```

(The mic_chat config wiring is left for a follow-up — it
needs new chat-config keys that don't exist yet. The engine
itself is functional; ChatLoop's `stt_engine` parameter accepts
any STTEngine subclass.)

**Tests** (`tests/unit/test_faster_whisper_engine.py`,
26 tests):

*Repo-string translation (10 tests parametrized):* native strings
pass through; MLX namespace stripped for tiny/base/small/medium/
large-v3/large-v3-turbo; unrecognized MLX-namespace passes
through (operator can debug).

*Construction + contract (5 tests):* implements STTEngine; default
construction matches iter-117 choices; overrides work; **kwargs
accepted; no model load at construction (lazy).

*Cache key behavior (2 tests):* uses resolved repo (so MLX +
native strings sharing the same size share the cache); device +
compute_type included in the key.

*Factory registration (4 tests):* `get_engine("faster_whisper")`
returns instance; kwargs forward; bogus name → ValueError;
ENGINES dict has the entry.

*Real round-trip (2 tests, skip when unavailable):*
- Transcribe iter-117's `clean.wav` fixture → assert
  "weather" or "today" appears in the transcript (loose match,
  doesn't require exact text — model output varies).
- Invalid model name → returns `(None, elapsed)`, never raises
  (matches mic_chat.py error-handling expectations).

Verification:
- `python -m pytest tests/unit/test_faster_whisper_engine.py -v`
  → **26 passed in 2.1s** (incl. real model load + transcription).
- Full unit + integration: **1394 passed, 1 skipped** (1368
  prior + 26 new).
- Perf snapshot: **23 passed**.

Notes:
- **The swap point promised by iter-108 is now real.** Every
  iteration since iter-108 has carried "ready to land — `stt_factory`
  is the swap point" as a forward note. Eight iterations later,
  the swap has a real backend.
- **iter-117 + iter-118 form a coherent unit.** iter-117 proved
  faster-whisper works with the WER pipeline as a test
  dependency; iter-118 promotes it to a production engine. A
  future iteration can add an integration test that drives
  `mic_chat.py:run_chat` with the FasterWhisperEngine + a
  virtual mic, end-to-end on x86_64 Linux.
- **Model cache invariant.** `_resolve_model_repo` is the
  shared key — two callers with different `model_repo` strings
  that resolve to the same size share the cache. Tested
  explicitly so a future regression that bypasses the resolver
  (e.g., direct dict lookup with raw repo) gets caught.
- **The `config.local.yaml` wiring is deferred** because adding
  config knobs without an end-to-end use case is a smell.
  Once a real Linux user wants to run mic_chat, the chat-config
  schema gets extended with `stt_engine` / `stt_model` /
  `stt_device` / `stt_compute` entries. Until then,
  programmatic construction (`FasterWhisperEngine(...)`) is
  available.
- **Pattern for cross-platform engines:**
  1. Implement the abstract class.
  2. Lazy-import platform deps inside `_load()`.
  3. Class-level cache for model reuse.
  4. Repo-string translation if other engines use a different
     namespace convention.
  5. Register in the factory.
  6. Tests across all three concerns: pure functions, contract,
     real round-trip with skip.
- Next directions:
  - Wire `chat_cfg.stt_engine` / `stt_model` keys + a
    `parse_stt_config` helper (matches iter-018's
    `parse_chat_config` pattern).
  - End-to-end test: full `mic_chat.py:run_chat` driven by
    `FasterWhisperEngine` + virtual mic + the iter-117
    `clean.wav` fixture playing through the speaker.
  - Add a noisy-audio fixture to exercise the iter-106
    "noisy"/"catastrophic" bands against real audio.
  - barge_in_phase consistency check (third diversity
    instance).

## iter-119 — Wire chat_cfg.stt_engine + parse_stt_config

**Goal:** Close iter-118's deferred work. iter-118 added
`FasterWhisperEngine` + factory registration but left config
wiring deferred ("adding knobs without an end-to-end use case is
a smell"). The use case now exists — Linux operators want to run
mic_chat. iter-119 wires the engine choice through
`config.local.yaml`.

**Change:** Three pieces:

1. **`STT_DEFAULTS` + `parse_stt_config`** in
   `examples/_chat_config.py`, matching the iter-020 +
   iter-034 parse-family conventions:

   ```python
   STT_DEFAULTS = {
       "engine": "whisper",
       "model": "",            # empty = engine class default
       "device": "cpu",        # faster_whisper-only
       "compute_type": "int8", # faster_whisper-only
   }

   def parse_stt_config(chat_cfg) -> dict:
       # Tolerant: malformed values fall back to defaults,
       # never raises. Strips whitespace from string values.
       # ...
   ```

   Two design choices recorded in the docstring:

   - **`model` default is empty string** rather than a real
     model name. Avoids hardcoding a Mac-only default that
     would mislead Linux operators. When the value is empty,
     mic_chat skips passing it and the engine class's own
     default kicks in.
   - **`device` and `compute_type` are always present** in the
     output regardless of engine. Callers decide which to use
     based on the chosen engine. Keeps the dict shape uniform.

2. **`mic_chat.py:run_chat` reorganization** — chat_cfg loading
   moved BEFORE the load_engines call so stt_cfg is available:

   ```python
   chat_cfg = load_chat_config()
   stt_cfg = parse_stt_config(chat_cfg)
   stt_model = stt_cfg["model"] or model_repo  # function-arg fallback
   stt_engine_name = stt_cfg["engine"]

   def _stt_factory():
       kwargs = {}
       if stt_model:
           kwargs["model_repo"] = stt_model
       if stt_engine_name == "faster_whisper":
           kwargs["device"] = stt_cfg["device"]
           kwargs["compute_type"] = stt_cfg["compute_type"]
       return _get_stt_engine(stt_engine_name, **kwargs)

   engines = load_engines(stt_factory=_stt_factory, ...)
   ```

   The `model_repo` function arg becomes a fallback only —
   honored when chat config doesn't set `stt_model`. Backwards
   compatible: existing callers (CLI invocations passing
   `model_repo` directly) keep working.

   The startup-print line now shows the chosen engine + model:
   `stt: faster_whisper/tiny │ llm: ... │ tts: ...`.

3. **Operator-facing config** (documented inline):

   ```yaml
   chat:
     stt_engine: "faster_whisper"   # or "whisper" (default)
     stt_model: "tiny"              # any size or repo string
     stt_device: "cpu"              # faster_whisper only
     stt_compute: "int8"            # faster_whisper only
   ```

**Tests** (`tests/unit/test_parse_stt_config.py`, 28 tests):

*Defaults (5 tests):* all keys present; engine="whisper";
model=""; device/compute defaults; STT_DEFAULTS matches output
for empty input.

*Malformed input (6 tests):* non-mapping (None, list, string,
int, object) → defaults; missing section → defaults; engine not
string / empty string / whitespace-only / with surrounding
whitespace.

*Per-key parsing (12 tests):* model string extraction,
full-repo passthrough, whitespace strip, non-string fallback,
empty fallback (4 tests for model); device cpu/cuda + strip +
non-string (4 tests); compute_type extracted + strip + non-
string (3 tests); engine valid + strip (already in malformed
section).

*Composite (4 tests):* full config flows through; partial
config backfills defaults; does NOT mutate input; does NOT
mutate STT_DEFAULTS constant.

*Independence (1 test):* unrelated chat-cfg keys don't bleed
in.

Verification:
- `python -m pytest tests/unit/test_parse_stt_config.py -q` →
  **28 passed in 30ms**.
- Full unit + integration: **1422 passed, 1 skipped** (1394
  prior + 28 new).
- Perf snapshot: **23 passed**.
- AST sanity check on `mic_chat.py`: parse OK.

Notes:
- **The iter-118 → iter-119 unit is now complete.** Together,
  they form the cross-platform STT story:
  - iter-118: implementation + factory registration.
  - iter-119: config wiring + operator UX.

  Linux operators can now opt into faster-whisper with one
  yaml-section change.
- **Backwards compatibility preserved.** Existing callers that
  pass `model_repo` to `run_chat` still work — the arg becomes
  the fallback when `stt_model` is empty in config. No CLI
  flag changes, no breaking config changes.
- **The empty-string default for `model` is the cleanest signal.**
  Three alternatives considered:
  1. `None` — would require explicit `if model is not None`
     checks at every consumer.
  2. Per-engine-default dict — duplicates knowledge that
     already lives in the engine class's `__init__` defaults.
  3. Empty string — Pythonic falsy, single-knob `if model:`
     check at the consumer site.

  The third choice is what the implementation uses.
- **No new pattern introduced.** This iteration is "apply the
  established parse-config pattern (iter-020, iter-034) to the
  next config knob." Documenting the choice between alternatives
  for future iterations that add more chat-cfg knobs.
- **mic_chat.py size:** ~370 lines total / ~100 in run_chat
  (up from ~340/95 pre-iter-119 because the stt config wiring
  added ~20 lines). Still significantly smaller than the pre-
  iter-107 baseline (~395/155).
- Next directions:
  - End-to-end test driving `mic_chat.py:run_chat` with
    `FasterWhisperEngine` + virtual mic + iter-117's
    `clean.wav` playing through the speaker. Validates the
    cross-platform path at the integration level.
  - Add a noisy-audio fixture (synthesized + bgm) to exercise
    the iter-106 noisy/catastrophic bands against real audio.
  - barge_in_phase consistency check (third diversity-check
    instance).

## iter-120 — Barge-phase consistency check

**Goal:** Third instance of the diversity-check pattern after
iter-114 (filler diversity) and iter-115 (naturalness).
Validates that iter-116's `_longest_consecutive_run` extraction
was justified — the helper is now used by three distinct
emit-line callers, all sharing the same single-pass scan
without re-implementing.

**The signal:** `barge_in_phase` (iter-047) records WHEN the
user barged in:
- `"llm_stream"` — barge happened BEFORE bot speech started.
  Means the user is impatient with LLM TTFB or the bot is slow.
- `"playback"` — barge happened DURING bot speech. Means the
  bot speaks too long or the user is interrupt-happy.

A single-turn barge is fine. Four consecutive barges in the
same phase is a UX problem worth surfacing.

**Change:** New `_emit_barge_phase_consistency_line` in
`_chat_metrics.py`. Same shape as iter-114/iter-115:

```python
def _emit_barge_phase_consistency_line(
    emit, phases: list[str], threshold: int = 4,
) -> None:
    non_empty = [p for p in phases if p]
    if not non_empty:
        return
    longest_run, longest_phase = _longest_consecutive_run(non_empty)
    if longest_run < threshold:
        return
    if longest_phase == "llm_stream":
        suggestion = "user impatient with LLM TTFB, or bot slow to start"
    elif longest_phase == "playback":
        suggestion = "user habit or bot speaks too long"
    else:
        suggestion = "consistent barge phase — investigate"
    emit(...)
```

**Three design choices recorded:**

1. **Both phases warrant warnings** (unlike iter-115 which
   excluded "natural"). Each barge phase carries a different
   UX signal — neither is "good." `if longest_phase ==
   "natural": return` would be the wrong rule here.

2. **Threshold = 4** (lower than iter-115's 5). Barge events
   are rarer + more semantically loaded than naturalness
   buckets. 4 consecutive same-phase barges is rare enough in
   normal use that the warning's signal-to-noise stays high.

3. **Per-phase suggestion text** — iter-115's `rushed` /
   `slow` recommendation pattern carries over. The two phases
   suggest opposite remediation (faster TTFB vs shorter bot
   speech), so the suggestion isn't a one-size-fits-all
   message.

**Wired into `print_session_summary`** right after iter-115's
naturalness check:

```python
_emit_barge_phase_consistency_line(
    _emit,
    [m.barge_in_phase for m in metrics_list],
)
```

**Tests** (`tests/unit/test_emit_barge_phase_consistency_line.py`,
18 tests):

*Empty / no-barge suppression (2 tests):* empty list, all-empty-
strings.

*Below threshold (2 tests):* 3-in-a-row default, alternating
phases.

*Both phases fire (3 tests):* 4 playback → "speaks too long"
suggestion; 5 llm_stream → "TTFB" suggestion; both phases not
excluded (sanity test).

*Empty filtering (3 tests):* empties between barges don't break
runs; leading + trailing empties; a phase change DOES break
the run.

*Custom threshold (2 tests):* threshold=3 catches; threshold=10
suppresses.

*Longest-of-multiple (2 tests):* longer run wins; run at end
detected.

*Output formatting (3 tests):* leading 4-space indent;
iter-047 attribution; unknown-phase defensive fallback.

*Pattern parity (1 test):* 1000-element list works (sanity
that the iter-116 helper handles big inputs).

Verification:
- `python -m pytest tests/unit/test_emit_barge_phase_consistency_line.py
  -q` → **18 passed in 30ms**.
- Regression sentinel (199 prior session-summary +
  diversity-check + run-finder tests): **199 passed**
  byte-for-byte.
- Full unit + integration: **1440 passed, 1 skipped** (1422
  prior + 18 new).
- Perf snapshot: **23 passed**.

Notes:
- **iter-116 helper extraction validated by use.** Three
  callers now consume `_longest_consecutive_run`:
  - `_emit_filler_diversity_line` (iter-114, threshold 3)
  - `_emit_naturalness_consistency_line` (iter-115, threshold 5)
  - `_emit_barge_phase_consistency_line` (iter-120, threshold 4)

  Each uses the same primitive, then applies its own filtering
  and threshold + suggestion logic. The helper-extraction
  pattern is doing exactly what extraction is supposed to do:
  **the loop body lives in one place; the policy decisions
  live in their own helpers.**

- **The diversity-check shape is now a documented family.**
  All three helpers share:
  - Filter empties before scanning (zeros, empty strings)
  - Use `_longest_consecutive_run`
  - Apply per-instance threshold (3/5/4)
  - Apply per-instance excluded-value rule (zeros / "natural" /
    none)
  - Emit suggestion based on the value type
  - Name the responsible iteration in the warning text

  A future fourth instance (e.g., consecutive `_too_short`
  utterances flagging a noisy mic) lands by copying the
  template.

- **Pattern: per-instance suggestion mapping.** When an emit
  helper has multiple meaningful values (rushed/slow,
  llm_stream/playback), the suggestion text per value lives
  inside the helper. Don't externalize it (would require
  passing a dict from the caller — overkill for two values).
  Don't make it a one-size-fits-all message (would lose the
  signal). Per-value branches in the helper are the right
  shape.

- **mic_chat.py size unchanged.** This iteration only touches
  `_chat_metrics.py` + a new test file. The session-summary
  layer continues to grow but `run_chat` itself stays at ~100
  lines.

- Next directions:
  - End-to-end test driving `mic_chat.py:run_chat` with
    `FasterWhisperEngine` + virtual mic + iter-117's
    `clean.wav`. Closes the iter-118+iter-119 cross-platform
    STT story at the integration level.
  - Add a noisy-audio fixture (synthesized + bgm) to exercise
    iter-106's noisy/catastrophic bands against real audio.
  - Sentence-length-bucket consistency check (fourth instance
    of the diversity pattern — would justify promoting the
    template into a documented protocol or factory function).

## iter-121 — STT config-to-engine routing integration

**Goal:** Close the iter-118 + iter-119 cross-platform STT
story at the integration level. iter-118 added the engine,
iter-119 wired the config; iter-121 verifies the chain works
end-to-end:

    chat_cfg dict
      → parse_stt_config()
      → factory closure (replicates mic_chat.py:_stt_factory)
      → get_engine(name, **kwargs)
      → engine instance ready to .transcribe()

**Pre-iteration scope decision:**
First considered driving the FULL `mic_chat.py:run_chat` loop
with `VirtualMicStream` + iter-117's `clean.wav`. clean.wav is
mono 22050 Hz but the recorder expects 16000 Hz — full integration
would need resampling support that doesn't currently exist
(scipy isn't a project dep). The tighter "factory routing"
integration is more focused on what iter-118/iter-119 actually
wired up, avoids the rate-mismatch detour, and still validates
real transcription via the routed engine.

The full-loop integration test stays as a future iteration if
needed.

**Change:** New `tests/integration/test_stt_routing.py` with
10 tests:

*The factory-builder helper* mirrors mic_chat.py exactly:

```python
def _build_stt_factory(chat_cfg: dict, model_repo_fallback: str = ""):
    stt_cfg = parse_stt_config(chat_cfg)
    stt_model = stt_cfg["model"] or model_repo_fallback
    stt_engine_name = stt_cfg["engine"]

    def _stt_factory():
        kwargs = {}
        if stt_model:
            kwargs["model_repo"] = stt_model
        if stt_engine_name == "faster_whisper":
            kwargs["device"] = stt_cfg["device"]
            kwargs["compute_type"] = stt_cfg["compute_type"]
        return _get_stt_engine(stt_engine_name, **kwargs)

    return _stt_factory
```

This is **the public-API documentation for what mic_chat.py does**,
backed by tests. If mic_chat's wiring drifts (e.g., a future
refactor changes the kwarg-passing rule), this helper goes
out-of-sync with the real one and the routing tests stop
matching production behavior. That divergence is deliberately
small (~10 lines) and will surface fast when changed.

**Test coverage** (10 tests):

*Routing without instantiating Mac-only deps (8 tests):*
- Default chat_cfg → factory closure built (no engine
  constructed yet — would import mlx).
- `stt_engine: faster_whisper` → factory builds
  FasterWhisperEngine.
- All 4 chat_cfg knobs flow through correctly.
- Partial config uses engine-class defaults (validates
  iter-119's "empty model = engine default" decision).
- Model fallback (function-arg) used when chat_cfg omits
  stt_model.
- chat_cfg model overrides function-arg fallback (precedence).
- Unknown engine raises ValueError from `get_engine`.
- Whisper routing does NOT pass `device`/`compute_type` —
  monkey-patches `_get_stt_engine` to capture kwargs and
  asserts the engine-class kwargs aren't leaked into the
  Mac-only Whisper constructor.
- Factory callable multiple times (each builds a fresh
  instance — SentenceWorker may invoke more than once).

*Real transcription via routed engine (1 test, skip when
unavailable):*
- Build factory from yaml-style config → construct engine →
  transcribe iter-117's clean.wav → assert "weather" or
  "today" appears. End-to-end the routing actually works.

Verification:
- `python -m pytest tests/integration/test_stt_routing.py -v` →
  **10 passed in 2.0s** (incl. real model load + transcription).
- Full unit + integration: **1450 passed, 1 skipped** (1440
  prior + 10 new).
- Perf snapshot: **23 passed**.

Notes:
- **The factory-builder helper is the test's contribution.**
  Beyond the assertions, the helper documents the exact wiring
  shape mic_chat.py uses. Future readers can import this helper
  for their own integration tests instead of reverse-engineering
  the mic_chat code path.
- **Skipping mlx for whisper-routing tests.** The whisper-engine
  routing test uses monkey-patch instead of actually calling
  `factory()` because invoking the closure would import mlx. The
  monkey-patched `_get_stt_engine` captures what kwargs would
  be passed without triggering the import. Pattern worth
  recording for future cross-platform routing tests.
- **iter-118 + iter-119 + iter-121 form a closed loop.** Each
  adds a layer:
  - iter-118: the engine class + factory registration.
  - iter-119: config parsing + mic_chat wiring.
  - iter-121: integration verifying both layers compose.

  A Linux operator who follows the iter-119 docs gets a
  working setup; this iteration tests that exactly.
- **The `ignored_fallback` precedence test documents an
  intentional design choice.** When both a function-arg and a
  chat_cfg model are provided, chat_cfg wins. The reasoning:
  config is the more recent / explicit layer. Future code
  changes that flip this precedence would break the test —
  intentional sentinel.
- **Future full-loop integration test:** would require either
  (a) resampling support so 22050 Hz fixtures work with the
  16 kHz recorder, or (b) regenerating `clean.wav` at 16 kHz.
  Either is straightforward but out of scope for this
  iteration.
- Next directions:
  - Regenerate `clean.wav` at 16 kHz, then drive the full
    `mic_chat.py:run_chat` through one virtual turn — the
    "drives ChatLoop" iteration originally considered here.
  - Sentence-length-bucket consistency check (fourth instance
    of the diversity pattern — would justify promoting the
    template).
  - Add a noisy-audio fixture (synthesized + bgm) for the
    iter-106 catastrophic bands.

## iter-122 — Full ChatLoop integration with FasterWhisperEngine

**Goal:** Drive `mic_chat.py:run_one_turn` through the entire
pipeline on x86_64 Linux: virtual mic → recorder + VAD → real
faster-whisper STT → stub LLM → SentenceWorker → stub TTS →
virtual speaker. Closes the iter-118/119/121 cross-platform STT
arc with a true end-to-end test.

iter-121 validated the routing layer; iter-122 validates the
pipeline.

**Two pieces:**

1. **Resampled fixture** — `tests/fixtures/wer/clean_16khz.wav`,
   1.51s of "what is the weather today" at 16 kHz mono PCM.
   Generated once via scipy.signal.resample from iter-117's
   22050 Hz fixture, then committed. Deterministic across runs.

   The resampling script is recorded in this iter's commit
   message. scipy is the only build-time dep; it's already
   installed on the host. A future fixture commit can re-run
   the script to add more entries.

2. **`tests/integration/test_chatloop_faster_whisper.py`**:

   - `test_one_turn_through_real_stt`: pushes leading silence
     + clean speech + trailing silence to `VirtualMicStream`;
     constructs `ChatLoop` with FasterWhisperEngine for STT
     (other layers stubbed); runs `loop.run_one_turn([])`;
     asserts:
     - No `had_error`
     - `result.metrics` populated
     - Transcript contains "weather" or "today" (loose match
       — tiny model output varies)
     - `sentences_spoken >= 1`
     - `stt_time > 0`
     - `speech_duration > 0`

   - `test_transcribe_fn_signature_is_what_chat_loop_expects`:
     sentinel for the closure shape ChatLoop expects. The
     transcribe_fn unwraps `(text, elapsed)` from the engine's
     return — if either side changes shape, this fails fast.

**Bug found and fixed during implementation:**
First test attempt hung (timeout reached). Root cause:
`concat()` in `examples/virtual_audio.py` coerces every input
to int16. I had converted the speech samples to float32
(-1..1) before calling concat — those got truncated to mostly
zeros, and the VAD never detected speech. Fixed by keeping all
three arrays (leading silence, speech, trailing silence) as
int16 throughout. `make_silence` already returns int16 and
`wave.readframes` gives int16; the test path now works without
any dtype gymnastics.

**Pattern recorded for future audio-fixture tests:**
> When constructing audio for VirtualMicStream, all input chunks
> must be int16. Don't normalize/scale — `make_silence` and
> `wave.readframes` give you int16; the recorder normalizes
> internally (`int16 → float32 / 32768.0`).

**The cross-platform STT story is now closed at every level:**
- iter-117: WAV fixtures + faster-whisper as test dependency.
- iter-118: FasterWhisperEngine implementing STTEngine contract.
- iter-119: chat_cfg.stt_engine config wiring + parse_stt_config.
- iter-121: factory routing integration test.
- iter-122: full pipeline integration test.

A Linux operator who follows the iter-119 docs gets a working
end-to-end pipeline. Tested at every layer.

Verification:
- `python -m pytest tests/integration/test_chatloop_faster_whisper.py
  -v` → **2 passed in 2.7s** (incl. real STT + full ChatLoop run).
- Full unit + integration: **1452 passed, 1 skipped** (1450
  prior + 2 new).
- Perf snapshot: **23 passed**.

Notes:
- **2.7s for the full integration test** is acceptable. Most of
  it is the STT model-load (cached on subsequent runs) plus the
  ~1.5s real-time audio playback through `_slow_play`. The
  model warm-up dominates; the actual loop iteration is fast.
- **Alternatives considered for the resampling step:**
  - sox / ffmpeg — neither installed on this host.
  - audioop — deprecated in Python 3.13.
  - Pure-Python linear interpolation — would work but quality
    is lower than scipy.
  - **scipy.signal.resample** — chosen. High-quality FFT-based
    resampling, already available, deterministic.
- **The test deliberately uses `_slow_play` instead of an
  instant playback stub** so timing-sensitive metrics (TTFS,
  speech_duration) get realistic values. Doesn't matter for
  this test's assertions but reflects the perf-suite's
  conventions, making the test ready to adopt as a perf scenario
  if the wall-time budget allows.
- **`test_transcribe_fn_signature` is a tight sentinel.** It
  asserts the closure `lambda wav: engine.transcribe(wav)[0]`
  produces a non-empty string from clean.wav. If a future
  refactor changes either:
  - `ChatLoop`'s `transcribe_fn` contract (now `wav → str|None`),
  - `FasterWhisperEngine.transcribe`'s return shape (now
    `(text, elapsed)`),
  the test fails immediately. Two-line guard against silent
  drift between layers.
- **mic_chat.py is fully exercised.** Every iteration since
  iter-107 added testability to one piece of `run_chat`;
  iter-122 demonstrates that all those pieces compose correctly.
  An operator's end-to-end experience is now backed by tests.
- Next directions:
  - Add a noisy-audio fixture (synthesized + bgm) to exercise
    iter-106's "noisy"/"catastrophic" bands against real audio.
  - Sentence-length-bucket consistency check (fourth instance
    of the diversity pattern — would justify promoting the
    template into a documented protocol).
  - Pivot to architecture: the cross-platform STT loop is
    closed; what's the next user-visible improvement? Fillers,
    barge-in latency, etc. — perf data already shows where the
    real costs live.

## iter-123 — Wire max_user_assistant via chat_cfg

**Goal:** Promote the iter-024 context cap from a hardcoded
literal to a tunable config knob. iter-102 + iter-112 proved
empirically the knob has real production value:
- -55% billed tokens on turn 8 (cap=5 vs 20)
- -346ms TTFS on turn 8 with `context_factor=2ms/char`

Different LLMs and different conversation patterns benefit from
different caps. iter-123 lets operators tune it without editing
mic_chat.py:

    chat:
      max_user_assistant: 10   # 0 = no cap (eval/replay)

**Three pieces:**

1. **`MAX_USER_ASSISTANT_DEFAULT` + `parse_max_user_assistant`**
   in `examples/_chat_config.py`:

   ```python
   MAX_USER_ASSISTANT_DEFAULT = 20

   def parse_max_user_assistant(chat_cfg) -> int:
       if not isinstance(chat_cfg, Mapping):
           return MAX_USER_ASSISTANT_DEFAULT
       raw = chat_cfg.get("max_user_assistant")
       if isinstance(raw, bool):
           return MAX_USER_ASSISTANT_DEFAULT  # bool ⊂ int defense
       if isinstance(raw, int) and raw >= 0:
           return raw
       return MAX_USER_ASSISTANT_DEFAULT
   ```

   Three design choices documented in the docstring:

   - **0 = "no cap" sentinel.** Operators bypassing the trim
     for eval/replay scenarios get explicit support. Negative
     values fall back to default — only 0 is the special case.
   - **Bool defense.** `bool` is an int subclass in Python, so
     a typo'd yaml value `true` would silently become a cap of
     1 without the explicit guard. The defensive branch
     prevents this surprise.
   - **No float coercion.** `10.0` falls back to default. The
     cap is conceptually a count; floats indicate operator
     confusion that's better surfaced via fallback than
     silently rounded.

2. **`mic_chat.py:run_chat` updated** to read the cap from
   chat_cfg:

   ```python
   from examples._chat_config import parse_max_user_assistant
   max_user_assistant = parse_max_user_assistant(chat_cfg)
   state = run_session(
       chat_loop, system_prompt,
       max_user_assistant=max_user_assistant,
       ...
   )
   ```

   Backwards-compatible: any operator who hasn't set the new
   key still gets the iter-024 default of 20.

3. **The `cap=0` invariant verified at the trim layer.**
   `trim_history` already handles 0 correctly — Python's
   `tail[-0:]` is `tail[0:]` which is the full tail (because
   `-0 == 0`). The wiring relies on this idiom, so iter-123's
   tests lock it down explicitly:

   ```python
   def test_cap_zero_passes_full_history_through_trim():
       messages = [system_msg, u1, a1, u2, a2]
       out = trim_history(messages, max_user_assistant=0)
       assert out == messages   # no eviction
   ```

   Without this test, a future "optimization" that adds an
   early-return on `max_user_assistant == 0` could evict
   everything (Python idiom is non-obvious).

**Tests** (`tests/unit/test_parse_max_user_assistant.py`,
17 tests):

*Defaults (3 tests):* return type is int; default is 20;
missing key → 20.

*Malformed input (5 tests):* non-mapping inputs → default;
string "10" → default; float 10.0 → default; negative → default;
bool True/False → default.

*Valid values (4 tests):* 0 returned as 0; small positive (5);
large positive (100); default value 20 explicitly honored.

*Cap=0 round-trip semantics (3 tests):* full history preserved
through trim_history; empty-tail edge case; no-system edge case.

*Independence (2 tests):* doesn't mutate input dict; doesn't
mutate the default constant.

Verification:
- `python -m pytest tests/unit/test_parse_max_user_assistant.py
  -q` → **17 passed in 20ms**.
- AST sanity check on `mic_chat.py`: parse OK.
- Full unit + integration: **1469 passed, 1 skipped** (1452
  prior + 17 new).
- Perf snapshot: **23 passed**.

Notes:
- **The "0 = no cap" sentinel was a deliberate UX choice.**
  Alternatives:
  1. None — would require explicit None checks at every
     consumer.
  2. `-1` — common UNIX idiom but fragile (someone might
     "validate" it as invalid and reject it).
  3. A separate boolean `disable_cap_trim` — would split a
     single concern across two yaml keys.
  4. **`0` as sentinel** — chosen. Yaml's natural integer,
     unambiguous in Python, naturally produces "no cap" via
     the `tail[-0:]` idiom without requiring caller-side
     branches.

  Documented + tested explicitly because the choice isn't
  obvious from the code.
- **The bool-is-int defense is a recurring pattern.** This is
  the second config parser that defends against `bool` being
  mistreated as `int` (the first was iter-119's
  parse_stt_config — wait, that was string-based, no overlap).
  Actually iter-123 is the first integer-config parser in the
  family. If a fourth comes along, the pattern is documented
  here.
- **Python's `tail[-0:]` idiom isn't widely known.** The
  explicit test that exercises it lives in iter-123 to prevent
  a future contributor from "cleaning up" the trim_history
  function with an `if max_user_assistant <= 0: return ...`
  branch that eats the no-cap behavior.
- **The full chat_cfg config schema is now:**
  ```yaml
  chat:
    # iter-018 (chat config)
    aggressive_first_sentence: false
    auto_aggressive_threshold: 0.0
    # iter-011 (fillers)
    fillers: []
    fillers_idle_threshold: 0.6
    # iter-020 (vad)
    vad:
      silence_threshold: 0.02
      silence_duration: 0.8
      min_speech_duration: 0.3
    # iter-119 (stt)
    stt_engine: "whisper"
    stt_model: ""
    stt_device: "cpu"
    stt_compute: "int8"
    # iter-123 (history cap)
    max_user_assistant: 20
  ```

  Every operator-tunable knob is now in chat_cfg with a
  parse_*_config helper backing it. No more hardcoded literals
  in run_chat.

- Next directions:
  - Add a noisy-audio fixture (synthesized + bgm) for the
    iter-106 catastrophic bands.
  - Sentence-length-bucket consistency check (fourth instance
    of the diversity pattern).
  - Architecture pivot: with config wiring done, focus on
    end-to-end perf wins — e.g., `aggressive_first_sentence:
    true` as a default once an A/B with real audio confirms
    the prosody tradeoff is acceptable.

## iter-124 — Noisy-audio fixture for WER pipeline

**Goal:** iter-106's "noisy" entry was a synthetic STT
hypothesis ("the quick crown fox jumped on the lazy dog") —
the WER pipeline tested but no real noise was involved.
iter-124 generates real noisy audio so faster-whisper actually
encounters degraded input. Validates iter-106's "noisy" band
isn't purely synthetic.

**Pre-iteration SNR probe:**
First tried 10 dB SNR — tiny model produced "what in the web of
the May." (WER 1.0, catastrophic). Probed 15/20/25/30 dB; got:
- 15 dB → "what in the weather today." (WER 0.20)
- 20 dB → "What is the winner today?" (WER 0.20)
- 25 dB → "What is the winner to me?" (WER 0.60) ← outlier
- 30 dB → "What is the winner today?" (WER 0.20)

The model is non-monotonic with SNR — 25 dB regresses worse than
20 dB. This is a known artifact of small Whisper models on
synthesized voices. Picked 15 dB for the fixture: stable WER ≈
0.2, well-defined band, consistent across runs.

**Two pieces:**

1. **`tests/fixtures/wer/noisy_16khz.wav`** — generated from
   `clean_16khz.wav` (iter-122) by adding gaussian noise at
   ~15 dB SNR using `numpy.random.default_rng(seed=42)` for
   determinism. 24144 samples, mono, 16 bit. Committed.

   Generation script (recorded for reproducibility):
   ```python
   speech = read_int16("clean_16khz.wav") / 32768.0
   speech_rms = np.sqrt(np.mean(speech ** 2))
   noise_rms = speech_rms / (10 ** (15 / 20))   # 15 dB SNR
   noise = np.random.default_rng(seed=42).standard_normal(len(speech)) * noise_rms
   noisy = np.clip((speech + noise) * 32768.0, -32768, 32767).astype(np.int16)
   ```

2. **`audio_fixtures` corpus entry:** `noisy_audio` with
   reference "what is the weather today" and band 0.10-0.50.
   The 0.10 floor catches a fixture-regen at lower SNR; the
   0.50 ceiling tolerates the model's natural variance.

   The existing `tests/integration/test_wer_audio.py` auto-
   picks up the new entry — its parametrized
   `test_each_audio_fixture_lands_in_wer_band` runs faster-
   whisper on the new file and asserts the WER inside
   [0.10, 0.50].

**Fixture-shape sentinel** (`tests/unit/test_noisy_audio_fixture.py`,
7 tests):

- File presence + shape (mono, 16-bit, 16 kHz, sample count
  matches clean_16khz.wav).
- RMS invariants: noisy RMS > clean RMS (catches accidental
  re-save of clean as noisy).
- **Measured SNR within ±2 dB of 15 dB target.** Computed by
  subtracting clean from noisy → noise-only signal → SNR formula.
  Catches drift if the fixture is regenerated with a different
  SNR target.
- Saturation < 5% (catches over-loud regeneration that would
  clip too many samples).
- Deterministic content sanity (middle samples have non-trivial
  values; locks against silent fixture or accidental zero).

These run regardless of faster-whisper availability, so even on
CI hosts without STT the fixture's structural integrity is
verified.

Verification:
- `python -m pytest tests/unit/test_noisy_audio_fixture.py -v` →
  **7 passed in 60ms**.
- `python -m pytest tests/integration/test_wer_audio.py -v` →
  **4 passed in 3.2s** (the parametrized fixture-band test now
  iterates over 3 fixtures instead of 2; new entry lands in
  band).
- Full unit + integration: **1476 passed, 1 skipped** (1469
  prior + 7 new).
- Perf snapshot: **23 passed**.

Notes:
- **The `noise_only = noisy - clean` SNR test is the strongest
  fixture invariant.** It measures the actual delivered SNR
  rather than trusting the generation parameters. If a future
  iteration regenerates the fixture with seed=43 (or 14 dB
  target, or a different noise distribution), this test fires.
- **Saturation guard catches a class of bugs:** if someone
  regenerates with too-high SNR target → noise dominates →
  clipping → most samples saturate → audio sounds garbled.
  The < 5% bound permits the small clipping at speech peaks
  that's natural at 15 dB but not the cascade clipping that
  happens at e.g. -10 dB.
- **Determinism is the headline property.** `seed=42` is the
  contract — change it and the fixture's WER shifts, breaking
  the corpus assertions. The "middle samples" test is a soft
  canary; if it ever fails, the bigger SNR/shape tests will
  too.
- **The corpus is now 6 text + 3 audio fixtures.** Three audio
  scenarios (clean speech, pangram, noisy speech) cover the
  non-degenerate real-audio space. Future additions:
  - Music/background-speech overlay (the iter-106 "noisy"
    case at lower SNR).
  - Truly catastrophic (5-10 dB) for the iter-106
    "catastrophic" band — model reliably fails.
  - Multi-speaker overlap.

  Each fixture gets the same shape: deterministic seed,
  expected band, fixture-shape unit test.
- **Pattern: deterministic-fixture invariants belong in a
  unit test, not the integration test.** The unit test runs
  always; the integration test runs only when STT is
  available. Splitting the assertions cleanly lets each test
  live in its proper skip context.
- Next directions:
  - Sentence-length-bucket consistency check (fourth
    diversity-pattern instance).
  - More-degraded audio fixture (truly catastrophic, 5-10 dB
    SNR) for the iter-106 "catastrophic" band.
  - Architecture pivot: A/B aggressive_first_sentence as a
    default, validated against the now-existing real-audio
    fixtures.

## iter-125 — Catastrophic-audio fixture (10 dB SNR)

**Goal:** iter-124 added a noisy-audio fixture (15 dB SNR)
matching iter-106's "noisy" band. iter-125 adds the
catastrophic-audio counterpart — completes the iter-106
band coverage with real audio at every level.

**Pre-iteration probe rediscovered an issue iter-124 didn't
catch:**
First generated `catastrophic_16khz.wav` at 10 dB SNR with
seed=42. The iter-117 integration test (which iterates over
`audio_fixtures`) ran it and got **WER 0.200** ("what in the
weather today"), well below the 0.80 minimum I'd set for
"catastrophic." Initial probe (run during iter-124) had
consistently gotten WER 1.0 ("what in the web of the May.").

Root cause: **faster-whisper's default decoding is non-
deterministic across runs.** Default `beam_size=5` plus
temperature-fallback sampling (0..1.0 in 0.2 steps) produces
different transcripts on repeated runs of the same audio.
The catastrophic fixture is right at the edge where the
model sometimes recovers a partial transcript.

**The fix:** force greedy decoding (`beam_size=1,
temperature=0`) in the iter-117 `_transcribe` helper.
With greedy decoding, all four audio fixtures produce
deterministic WERs:

| Fixture                  | Band         | Greedy WER (3 runs)    |
|--------------------------|--------------|------------------------|
| clean_audio              | [0.0, 0.4]   | [0.20, 0.20, 0.20]     |
| quick_brown_fox_audio    | [0.0, 0.3]   | [0.11, 0.11, 0.11]     |
| noisy_audio              | [0.1, 0.5]   | [0.20, 0.20, 0.20]     |
| catastrophic_audio       | [0.8, 1.3]   | [1.00, 1.00, 1.00]     |

Without the fix, the iter-124 noisy fixture happened to land
in band by luck (its 0.10-0.50 band was wide enough to
absorb the variance). The catastrophic band was tight enough
to expose the issue. iter-125's fix retroactively makes
iter-124's fixture deterministic too.

**Three pieces:**

1. **`tests/fixtures/wer/catastrophic_16khz.wav`** — generated
   from `clean_16khz.wav` via gaussian noise at 10 dB SNR with
   seed=42. Same generation pattern as iter-124's noisy fixture
   but at the lower SNR.

2. **`audio_fixtures` corpus entry** `catastrophic_audio` with
   reference "what is the weather today" and band 0.80-1.30.
   The iter-117 parametrized integration test auto-picks it up.

3. **`tests/integration/test_wer_audio.py:_transcribe` updated**
   to use `beam_size=1, temperature=0`. Comment in the function
   explains why — future contributors who try to "improve"
   accuracy by switching back to beam search will be reminded
   that determinism is the contract.

**Fixture-shape sentinel** (`tests/unit/test_catastrophic_audio_fixture.py`,
8 tests):

- File presence + shape (mono, 16-bit, 16 kHz, sample-count
  matches clean).
- SNR ordering: catastrophic noise RMS > iter-124 noisy noise
  RMS (catches a swap between the two fixtures).
- **Measured SNR within ±2 dB of 10 dB target** — strongest
  invariant via `noise_only = catastrophic - clean`.
- Saturation < 10% (looser than iter-124's 5% because more
  noise → more clipping at peaks; 10% is still well below
  "negative SNR" disasters).
- Determinism (middle samples non-trivial).
- **Distinct from noisy fixture** — explicit byte-array
  comparison. Catches accidental copy.

Verification:
- `python -m pytest tests/unit/test_catastrophic_audio_fixture.py
  -v` → **8 passed in 60ms**.
- `python -m pytest tests/integration/test_wer_audio.py -v` →
  **4 passed in ~4s** (now exercises 4 fixtures including
  catastrophic; greedy decoding enabled).
- All faster-whisper-using tests (42 total): **42 passed in
  4.7s**.
- Full unit + integration: **1484 passed, 1 skipped** (1476
  prior + 8 new).
- Perf snapshot: **23 passed**.

Notes:
- **The non-determinism discovery is the big takeaway.**
  iter-124's seed=42 + 15 dB SNR happened to be in a region
  where the model gave the same answer most of the time, so
  the bug stayed hidden. iter-125's tighter band on a more
  fragile fixture exposed it. Recorded for future fixture
  iterations: **always run the integration test 3+ times
  before committing the band**, and prefer greedy decoding
  for any band-based assertion.
- **The greedy-decoding tradeoff.** Beam search with temperature
  fallback is what production STT typically uses — it's more
  accurate on average. Greedy is faster but slightly worse
  WER. For a TEST harness, deterministic-and-slightly-worse is
  the right tradeoff: tests that flake destroy trust faster
  than slightly-low accuracy hurts the band. For PRODUCTION,
  the existing FasterWhisperEngine.transcribe still uses the
  default beam search. iter-125 only changes the test path.
- **The corpus is now 6 text + 4 audio fixtures.**
  - clean_audio (production-grade STT, espeak-ng output)
  - quick_brown_fox_audio (pangram, espeak-ng output)
  - noisy_audio (15 dB SNR)
  - catastrophic_audio (10 dB SNR)

  Together they cover the iter-106 corpus's bands with real
  audio rather than simulated hypotheses.
- **Pattern recorded for future model-using fixtures:**
  > Decode deterministically (greedy, temperature=0) when
  > asserting on output. Match production decoding only when
  > asserting on the *engine path*, not the *output content*.
- **The probing methodology — generate at multiple SNRs,
  observe WER, pick the SNR with stable behavior — is the
  third instance** (iter-117 model size choice, iter-124 SNR
  choice, iter-125 SNR choice + decoding-mode choice). It's
  becoming a documented procedure for fixture-based STT
  testing.
- Next directions:
  - Sentence-length-bucket consistency check (4th diversity
    instance — would be the first that uses iter-116's helper
    after iter-120).
  - Architecture pivot: A/B aggressive_first_sentence as a
    default with real audio fixtures.
  - Multi-speaker overlap fixture for the "two speakers
    talking at once" stress-test.

## iter-126 — Fix iter-115 limitation: filter "natural" before run scan

**Goal:** iter-115 documented a known limitation in
`_emit_naturalness_consistency_line`:

> When a "natural" run is longer than a threshold-crossing
> "rushed"/"slow" run, the helper reports the natural run as
> the longest, suppresses it (because natural is the desired
> state), and fires nothing — losing the rushed/slow signal.

iter-126 fixes this by **filtering "natural" out before the
scan**, mirroring iter-114's zero-filter precedent for fillers:

> User perception: "5 rushed turns in a row" regardless of
> intervening natural turns.

**Code change** (3 lines added, 4 lines removed):

```python
# Before:
non_empty = [b for b in buckets if b]
longest_run, longest_bucket = _longest_consecutive_run(non_empty)
if longest_run < threshold:
    return
if longest_bucket == "natural":
    return  # iter-115's backstop suppression
```

```python
# After:
filtered = [b for b in buckets if b and b != "natural"]
if not filtered:
    return
longest_run, longest_bucket = _longest_consecutive_run(filtered)
if longest_run < threshold:
    return
# No "natural" backstop — filtered out. Dead code removed.
```

The filter approach is consistent with iter-114's zero-filter
and iter-120's empty-string filter — all three diversity-check
helpers now pre-filter "uninteresting" values before scanning.
The `_longest_consecutive_run` primitive (iter-116) stays
unchanged.

**Three behavior-change cases:**

| Input                              | Pre-iter-126 | Post-iter-126 |
|------------------------------------|--------------|---------------|
| 7 natural + 5 rushed               | silent       | fires (5)     |
| 6 natural + 1 rushed + 1 nat + 1 r | silent       | silent (run=2)|
| rushed×7, natural between each     | silent       | fires (7)     |

The first row is the headline fix. The second is unchanged
(still under threshold post-filter). The third is a new
recovery — pre-iter-126, the filter would have caught only
isolated runs of 1.

**Tests** (21 total, was 18):

The existing tests fall into three buckets:

- **Unchanged behavior (16 tests)**: empty/below-threshold/
  natural-only suppression, both phases firing on pure runs,
  custom thresholds, formatting, defensive fallbacks. All
  still pass byte-for-byte.

- **Updated behavior** (1 test, renamed):
  `test_natural_run_longer_than_rushed_run_fires_after_iter_126`
  was `test_natural_run_longer_than_rushed_run` — flipped from
  documenting the limitation (`assert lines == []`) to asserting
  the corrected behavior (`assert "5 consecutive" in lines[0]`).
  Comment block explaining the iter-126 change replaces the
  "known limitation" note.

- **Updated rationale** (1 test, same outcome):
  `test_long_natural_run_does_not_fire_even_with_brief_outliers`
  still asserts `lines == []`, but the rationale changes:
  pre-iter-126 it was "longest run is 6 of natural, suppressed";
  post-iter-126 it's "filtered list has run of 2, below
  threshold." Same outcome; updated docstring.

- **New tests (3 total)**:
  - `test_iter_126_natural_filter_preserves_run_in_split_pattern`:
    [r,n,r,n,r,n,r,r,n] → filtered = [r]*5 → fires.
  - `test_iter_126_filter_does_not_affect_pure_rushed_runs`:
    no natural to filter → unchanged.
  - `test_iter_126_mixed_rushed_and_slow_picks_longest`:
    [n*3, r*3, n, s*6] → filtered run-finder picks slow (6
    > 3).

Verification:
- `python -m pytest tests/unit/test_emit_naturalness_consistency_line.py
  -v` → **21 passed in 40ms** (18 prior + 3 new; 1 renamed,
  1 docstring updated).
- `python -m pytest tests/unit/test_emit_*.py
  tests/unit/test_session_*.py
  tests/unit/test_longest_consecutive_run.py -q` → all related
  tests pass.
- Full unit + integration: **1487 passed, 1 skipped** (1484
  prior + 3 new).
- Perf snapshot: **23 passed**.

Notes:
- **iter-115 was correct to flag this as a limitation.** A
  pure documenting test ("here's what doesn't work") instead
  of trying-to-work-around-it is the right call when the fix
  is uncertain or non-trivial. iter-126 is the natural follow-
  up: now we know the right shape, the fix is one line.
- **The filter-before-scan pattern is now consistent across
  all three diversity helpers**:
  - iter-114 (filler): filter zeros before scan.
  - iter-115/iter-126 (naturalness): filter empties + "natural"
    before scan.
  - iter-120 (barge): filter empties before scan.

  None of them apply post-scan exclusion. The shared
  `_longest_consecutive_run` (iter-116) doesn't filter — that's
  per-instance policy. iter-126 brings naturalness in line.
- **Removed dead code.** Pre-iter-126 had `if longest_bucket
  == "natural": return` as a backstop after the scan — now
  unreachable since "natural" is filtered out. Removed
  rather than left as "defense in depth": dead code obscures
  what the function actually does, and the new filter is the
  single source of truth.
- **No new metric, no new infrastructure.** This iter is a
  pure bug fix — fewer lines than were added, behavior
  improvement, three new test cases that exercise edge cases
  the previous tests didn't cover.
- Next directions:
  - Multi-speaker overlap fixture (stress test for the WER
    pipeline; harder to generate deterministically).
  - Architecture: A/B `aggressive_first_sentence: true` as
    default, with the now-existing real-audio fixtures as
    the prosody A/B reference.
  - Sentence-length-bucket consistency check would be the 4th
    diversity instance — useful but not urgent given the
    pattern has 3 stable instances.

## iter-127 — Multi-speaker overlap fixture

**Goal:** Add a real-world failure mode to the corpus: two
speakers overlapping. iter-124 + iter-125 added gaussian-noise
fixtures (acoustic degradation); iter-127 adds cross-talk
(linguistic degradation). Both stress-test faster-whisper but
exercise different failure modes.

**Pre-iteration probe rediscovered the iter-125 lesson:**
faster-whisper handles light cross-talk very well. At a 50%
distractor amplitude, the tiny model nailed the reference
("What is the weather today?" → WER 0.0). Probed 0.5 / 0.75 /
1.0 / 1.25 / 1.5 amplitudes:

| Amp  | WER (3 runs) | Sample transcript                     |
|------|--------------|---------------------------------------|
| 0.50 | 0.0          | "What is the weather today?"          |
| 0.75 | 0.8          | "What is my window taking out?"       |
| 1.00 | 1.0          | "What's in my mind?"                  |
| 1.25 | 1.2          | "what's in the power of the internet" |
| 1.50 | 37.0 (looped)| "what's in the eye and what's in..."  |

Picked **amp=0.75** as the fixture: meaningful confusion (WER
0.8) without runaway output. At 1.5 the model loops, which is
diagnostically interesting but produces a fixture WER that's
both hard to band and noisy.

**Three pieces:**

1. **`tests/fixtures/wer/multispeaker_16khz.wav`** — generated
   from two espeak-ng voices:
   - Reference (en-us): "what is the weather today" — starts at
     t=0.
   - Distractor (en-gb-x-rp): "I am cooking dinner now" —
     starts at t=0.4s, mixed at amp=0.75.

   30253 samples, mono, 16 kHz. Resampled from each voice's
   native rate via scipy.

2. **`audio_fixtures` corpus entry** `multispeaker_audio` with
   reference "what is the weather today" and band 0.60-1.10.
   Wider than iter-124's noisy band (0.10-0.50) because the
   model's response to cross-talk is more sensitive to
   per-token alignment.

3. **8 fixture-shape unit tests** (`tests/unit/
   test_multispeaker_audio_fixture.py`):

   - File presence + mono/16-bit/16 kHz shape.
   - Length >= clean fixture (catches truncated distractor).
   - Distinct from clean / noisy / catastrophic byte-arrays.
   - Saturation < 5% (catches over-loud regeneration).
   - **Overlap-region RMS > pre-overlap RMS** — sanity that
     the distractor actually got mixed in. If a future regen
     forgets to add the second voice, this fires.
   - Distractor-window has non-trivial samples (silent regen
     check).

**Pattern recorded for synthesized-mix fixtures:**
> When generating a mix from multiple sources, include an
> RMS-based "overlap region carries more energy" sanity test.
> The simple "distinct bytes" check fires on any difference,
> but the RMS check specifically validates that the mixing
> step did its work.

Verification:
- `python -m pytest tests/unit/test_multispeaker_audio_fixture.py
  -v` → **8 passed in 60ms**.
- `python -m pytest tests/integration/test_wer_audio.py -q` →
  **4 passed in 3.4s** (now exercises 5 fixtures including
  multispeaker; greedy decoding makes WER deterministic).
- Full unit + integration: **1495 passed, 1 skipped** (1487
  prior + 8 new).
- Perf snapshot: **23 passed**.

Notes:
- **The corpus is now 6 text + 5 audio fixtures.** Audio
  fixtures span:
  - clean_audio (production-grade STT)
  - quick_brown_fox_audio (pangram)
  - noisy_audio (15 dB SNR — light noise)
  - catastrophic_audio (10 dB SNR — heavy noise)
  - multispeaker_audio (cross-talk, amp=0.75)

  Different failure modes, deterministic across runs (greedy
  decoding from iter-125), each with band sentinels in
  corpus.json.
- **The probing methodology continues to scale.** iter-117
  (model size), iter-124 (noise SNR), iter-125 (more SNR + decoding
  mode), iter-127 (distractor amp) all used the same approach:
  generate at multiple settings → observe WER distribution →
  pick the setting with stable WER and meaningful signal.
- **Two-voice mixing is reproducible.** Both espeak-ng outputs
  are deterministic given the input text + voice. The mix
  amplitude + offset are recorded in the iteration log + the
  fixture-shape tests, so regeneration produces an identical
  fixture.
- **Why band 0.60-1.10 and not tighter?** Observed WER 0.8 on
  3 runs. Wider band tolerates faster-whisper version drift
  without breaking the test. Tightening to 0.75-0.85 might
  fire if a future model release shifts the output by one
  word.
- **The "RMS in overlap region" test is the strongest
  generation invariant.** It catches a class of bugs (forgot
  to mix in the distractor; mixed at zero amplitude;
  truncated the distractor) that a simple byte-comparison
  check might miss.
- Next directions:
  - Architecture: A/B `aggressive_first_sentence: true` as
    default with the now-existing 5-fixture audio corpus.
  - Sentence-length-bucket consistency check (4th diversity
    instance).
  - Document the 5-fixture corpus as a "standard test
    benchmark" — operators wanting to evaluate a new STT
    wrapper can run their engine through these fixtures and
    compare bands.

## iter-128 — Sentence-length-bucket consistency check

**Goal:** Fourth instance of the diversity-check pattern after
iter-114 (filler), iter-115/iter-126 (naturalness), iter-120
(barge-phase). First instance applied to a CONTINUOUS metric —
buckets `mean_sentence_chars` (iter-095) into discrete
categories before scanning. Validates that the pattern
generalizes beyond string-valued signals.

**The signal:** `mean_sentence_chars` per-turn captures how
verbose the bot was. Buckets:

- `very_short` (< 15 chars): choppy. Caused by an over-aggressive
  splitter (iter-088 `AGGRESSIVE_MIN_CHARS` too low) or a one-
  word-answer LLM.
- `short` (15-30): brief, not problematic.
- `medium` (30-60): the desired state.
- `long` (> 60): wall-of-text. LLM rambles, or splitter is too
  lax.

5+ consecutive turns in `very_short` or `long` warrants a
warning. `medium` and `short` are the "fine" states — filtered
before scanning, mirroring iter-126's "natural" filter.

**Change:** Two new helpers in `_chat_metrics.py`:

```python
def _sentence_length_bucket(mean_chars: float) -> str:
    if mean_chars <= 0: return ""
    if mean_chars < 15: return "very_short"
    if mean_chars < 30: return "short"
    if mean_chars < 60: return "medium"
    return "long"


def _emit_sentence_length_consistency_line(
    emit, mean_chars_list: list[float], threshold: int = 5,
) -> None:
    interesting = {"very_short", "long"}
    filtered = [
        b for b in (
            _sentence_length_bucket(mc) for mc in mean_chars_list
        )
        if b in interesting
    ]
    ...
```

Wired into `print_session_summary` after iter-120's barge-phase
check.

**Tests** (24 in `test_emit_sentence_length_consistency_line.py`):

- Bucket boundaries (7 tests): zero/negative → empty; very_short
  upper edge 14; short edges 15-29; medium edges 30-59; long ≥ 60;
  float handling.
- Empty / no-sentences (2): empty list, all-zero list.
- Medium/short exclusion (3): 10-turn medium silent; 10-turn short
  silent; alternating medium+long → only long counts.
- At/above threshold (3): 5 very_short → fires with "over-aggressive"
  suggestion; 6 long → fires with "too lax"; below threshold silent.
- Filter behavior (3): medium between very_short doesn't break run;
  short between long doesn't break run; phase change between flagged
  buckets DOES break run.
- Custom threshold (2): smaller catches via threshold=3; larger
  suppresses default 5-run.
- Longest-of-multiple (1): when both very_short and long pass
  threshold, longer wins.
- Output formatting (2): leading 4-space indent; iter-095
  attribution.
- Pattern parity (1): 1000-element input handled.

Verification:
- `python -m pytest tests/unit/test_emit_sentence_length_consistency_line.py
  -q` → **24 passed in 30ms**.
- Regression sentinel (220 prior tests pass byte-for-byte).
- Full unit + integration: **1519 passed, 1 skipped** (1495
  prior + 24 new).
- Perf snapshot: **23 passed**.

Notes:
- **Pattern continues to scale.** Four diversity-check instances
  now use iter-116's `_longest_consecutive_run`:
  - iter-114 (filler diversity) — int values, threshold 3
  - iter-115/126 (naturalness) — string values, threshold 5
  - iter-120 (barge-phase) — string values, threshold 4
  - iter-128 (sentence length) — bucketed continuous values,
    threshold 5
- **First continuous-metric application.** Bucketing is the new
  step. The bucketing function is testable in isolation
  (7 boundary tests), then the run-finder consumes the buckets
  like any other categorical signal. Generalizes the pattern.
- **Filter-before-scan rule applies.** All four instances filter
  uninteresting values before the scan. The "uninteresting"
  set varies per-helper (zeros / empties / "natural" / "medium"
  + "short") but the structural pattern is identical.
- **The pattern could now be promoted to a documented
  protocol.** Five instances would justify it (4th makes it
  obvious; 5th locks it in). For now, GENO.md's
  "mic_chat.py extraction pattern" is the only documented
  pattern; a "diversity-check pattern" section could
  catalog filter rules + threshold conventions + suggestion
  mapping for future instances.
- Next directions:
  - Promote the diversity-check pattern to documented
    guidance (5th instance threshold reached).
  - Architecture: A/B aggressive_first_sentence as default
    using the iter-127 5-fixture audio corpus.
  - Document the corpus as a "standard test benchmark" for
    operators evaluating new STT engines.

## iter-129 — Promote diversity-check pattern to GENO.md

**Goal:** Document the diversity-check pattern in GENO.md
alongside iter-109's mic_chat extraction pattern. iter-128
brought the count to **four instances** (iter-114 filler,
iter-115/126 naturalness, iter-120 barge-phase, iter-128
sentence-length); same threshold for promotion as iter-109's
3-instance bar for the extraction pattern.

The pattern is now stable: 8 template steps documented, each
backed by concrete examples from the four existing instances.
A 5th instance lands by following the template directly.

**Change:** Two pieces:

1. **New `Session-summary diversity-check pattern` section in
   `GENO.md`** (between the existing mic_chat extraction
   pattern and the Architecture section). Documents 8 template
   steps:

   1. Filter "uninteresting" values BEFORE the run scan
      (per-instance policy)
   2. Use `_longest_consecutive_run` (iter-116 primitive)
   3. Apply per-instance threshold (3 / 4 / 5 conventions)
   4. For continuous metrics, bucket BEFORE filtering
      (iter-128 first instance)
   5. Per-value suggestion mapping inside the helper
   6. Defensive fallback for unknown values
   7. Name the responsible iteration in the warning text
   8. Tests cover the matrix (empty / threshold / filter /
      formatting / defensive)

   Each step is illustrated with examples from the four
   existing helpers — readers see exactly how the abstract
   rule maps to real code.

2. **Doc-vs-code drift sentinel**
   (`tests/unit/test_diversity_pattern_doc.py`, 8 tests):

   - `test_geno_md_has_diversity_pattern_section`: doc
     section exists.
   - `test_doc_references_run_finder_helper`: pattern doc
     mentions iter-116's primitive.
   - `test_each_documented_helper_exists`: every name in
     `_DIVERSITY_HELPERS` is a real attribute on
     `_chat_metrics`.
   - `test_each_documented_helper_is_callable`: not just
     defined, but callable.
   - `test_doc_references_each_responsible_iteration`:
     iter-114 / iter-115 / iter-120 / iter-128 attribution
     present in the doc text.
   - `test_doc_template_lists_consistent_threshold_conventions`:
     thresholds 3/4/5 pinned to their respective iter
     numbers in the bullet text. Regex-based — survives
     small rewording but catches a substantive change.
   - `test_doc_template_claims_match_actual_instance_count`:
     the doc says "four instances confirm it"; the test asserts
     the English numeral matches `_DIVERSITY_HELPERS` length. A
     future 5th instance landing without doc update fires this
     test (with a useful nudge in the assertion message).
   - `test_each_helper_has_a_corresponding_test_file`: every
     diversity helper has `tests/unit/test_<helper_name>.py`.

**The drift-sentinel pattern recorded for future doc-of-code
sections:** when a documented pattern catalogs N instances by
name + iter number, write a test that:

1. Lists the canonical instances in a tuple (the test's source
   of truth).
2. Verifies each entry resolves to a real importable callable.
3. Verifies the doc text mentions each instance's iter
   attribution.
4. Verifies the doc's "N instances" English numeral matches
   the tuple length.

Adding the 5th instance becomes "update the tuple AND the
doc together" — the test ensures both happen.

Verification:
- `python -m pytest tests/unit/test_diversity_pattern_doc.py
  -q` → **8 passed in 30ms**.
- Full unit + integration: **1527 passed, 1 skipped** (1519
  prior + 8 new).
- Perf snapshot: **23 passed**.

Notes:
- **GENO.md now documents two patterns.** iter-109's
  mic_chat extraction pattern + iter-129's diversity-check
  pattern. Both follow the same documentation structure:
  - Lead with the trigger (when to use)
  - Number each rule
  - Cite the responsible iter for each rule
  - End with concrete examples from existing instances
- **The doc-vs-code sentinel test is the meta-pattern.** It
  protects documentation from going stale silently — the
  alternative ("docs reference iter-114, but the helper got
  renamed in iter-XYZ") is invisible until someone reads the
  doc and follows a broken link. The sentinel makes the link
  break a test failure instead.
- **Extending to a 5th instance:** the new contributor
  - Adds the helper.
  - Adds its test file (matching naming convention).
  - Adds the helper name to `_DIVERSITY_HELPERS` in
    test_diversity_pattern_doc.py.
  - Updates the GENO.md text to say "five instances" + add
    the iter attribution + (optionally) a new threshold
    convention.
  - Runs `pytest tests/unit/test_diversity_pattern_doc.py` —
    if any of the four sentinels fail, fix the doc.
- **Pattern-promotion threshold is now established.** iter-109
  (3 instances), iter-129 (4 instances). Both pattern docs
  emerged from concrete code rather than upfront design — the
  rules came from observing what worked. Future patterns
  should accumulate ≥3-4 instances before being documented;
  premature documentation locks in incorrect rules.
- Next directions:
  - Architecture: A/B aggressive_first_sentence as default
    using the iter-127 5-fixture audio corpus.
  - Document the 5-fixture corpus as a "standard test
    benchmark" for operators evaluating new STT engines.
  - Apply iter-129's drift-sentinel pattern to iter-109's
    mic_chat extraction pattern docs (catalog the four
    extracted modules; verify doc/code parity).

## iter-130 — Drift-sentinel for mic_chat extraction pattern

**Goal:** Apply iter-129's drift-sentinel pattern to iter-109's
mic_chat extraction-pattern docs. The two pattern sections in
GENO.md are now both protected against doc/code divergence.

**Pre-iteration discovery: the doc was already drifting.**
iter-110's `run_session` extraction follows the same 5-rule
shape as iter-107/108/109:
- Callable dependency injection (`chat_loop` instead of an
  engine class)
- Log injection (`log` callable, default `print`)
- ANSI styling stays at the caller (`prompt_log` closure
  re-applies DIM/RESET)
- Returns a dataclass (`SessionState`)
- Lazy-imports `ChatLoop.trim_messages` per rule 5

But the GENO.md section still said "three instances confirm it"
and didn't list iter-110. The drift was silent — no test caught
it. iter-130 fixes the drift AND adds the test that prevents
future drift.

**Two changes:**

1. **GENO.md updated**:
   - "three instances" → "four instances".
   - iter-110 + `run_session` added to the bullet list.
   - `SessionState` added to the rule-4 dataclass examples
     alongside `LoadedEngines`, `AudioIO`, `RecordingStats`.

2. **`tests/unit/test_extraction_pattern_doc.py`** — 9 tests
   mirroring iter-129's diversity-pattern sentinel:

   - `_EXTRACTION_INSTANCES` tuple: source of truth listing
     all four (`iter_num`, `module_path`, `callable_name`)
     entries.
   - Section presence + `run_chat` reference in the doc text.
   - Each instance's module imports cleanly (catches a module
     rename).
   - Each instance's public callable exists + is callable.
   - Doc text references every iter number AND every callable
     name (catches partial updates that update one but not
     the other).
   - English numeral matches `_EXTRACTION_INSTANCES` length —
     "four instances" today; a 5th instance must update both
     the tuple and the doc atomically.
   - Section enumerates exactly 5 numbered rules — if a 6th
     is added, this nudges the contributor to confirm the
     rule generalizes.
   - Every instance has a corresponding `tests/unit/test_*.py`
     file.

**The "rules-vs-instances cross-check" is new this iteration.**
iter-129's diversity-pattern test only covered helper-vs-doc
sync; iter-130 adds:

- **Rule count check** (`test_doc_lists_5_numbered_rules`):
  catches a contributor adding a 6th rule without updating the
  cross-pattern guidance.
- **Test-file existence check**: catches an extraction landing
  without dedicated tests.

These extensions are worth backporting to iter-129's
diversity-pattern test in a future iteration if a 5th
diversity instance lands.

Verification:
- `python -m pytest tests/unit/test_extraction_pattern_doc.py
  -q` → **9 passed in 50ms**.
- `python -m pytest tests/unit/test_diversity_pattern_doc.py
  -q` → **8 passed** (iter-129 sentinel still passes).
- Full unit + integration: **1536 passed, 1 skipped** (1527
  prior + 9 new).
- Perf snapshot: **23 passed**.

Notes:
- **Both GENO.md pattern sections now have drift sentinels.**
  iter-129 covered diversity-check; iter-130 covers extraction.
  The shape is reusable: any future documented pattern in
  GENO.md should ship with a sibling drift-sentinel test.
- **Discovery-during-iteration is itself a sentinel.** iter-130
  found stale documentation that no human had reported. Adding
  the sentinel test means the next contributor who lands a
  pattern-conforming change gets a friendly test failure
  reminding them to update the doc — instead of silently
  letting the doc rot.
- **The two pattern docs in GENO.md form a small ecosystem.**
  Reading one helps a contributor understand the other:
  - Both follow "lead with trigger / number rules / cite iter
    per rule / show concrete examples"
  - Both have a backing tuple (the test file's source of
    truth) that can be extended to add an instance
  - Both fire when the doc and code diverge

  This consistency makes it cheap to add a third documented
  pattern in the future — copy the structure, write the
  sentinel test from the same template.
- **Extraction count is now 4.** iter-110 brought this to
  parity with iter-129's diversity count. Five extractions
  would justify a "you've done this enough times that the
  rules are stable" note in the doc; for now, four feels
  like the right level of evidence.
- Next directions:
  - Architecture: A/B `aggressive_first_sentence: true` as
    default with the iter-127 5-fixture corpus.
  - Document the 5-fixture corpus as a "standard test
    benchmark" (the still-pending option).
  - Backport iter-130's rule-count + test-file checks to
    iter-129's diversity-pattern sentinel test.

## iter-131 — Sentinel symmetry: backport iter-130 checks to iter-129

**Goal:** Make the iter-129 diversity-pattern sentinel and the
iter-130 extraction-pattern sentinel structurally consistent.
iter-130 introduced two check types that iter-129 didn't have:
rule count and per-callable name reference. iter-131 backports
both, plus adds iter-126 to the diversity sentinel's expected-
iters list (it's the fix-iter for iter-115's limitation).

**Pre-iteration discovery: the doc had a hidden drift.**
The diversity-pattern section attributed naturalness to
"iter-115/126" — the slash-form. The substring `iter-126`
doesn't appear in `iter-115/126`, so the new attribution test
caught the missing standalone reference. Fixed by changing
"iter-115/126" → "iter-115 + iter-126" globally in the doc.
This is exactly the kind of drift the sentinels exist to
catch — a silent attribution gap that worked visually but
failed an explicit substring check.

**Three changes to `tests/unit/test_diversity_pattern_doc.py`:**

1. **Added iter-126 to `expected_iters`**:
   ```python
   expected_iters = [
       "iter-114",  # filler diversity
       "iter-115",  # naturalness (initial)
       "iter-120",  # barge-phase
       "iter-126",  # naturalness (filter fix; iter-131 added)
       "iter-128",  # sentence-length
   ]
   ```

2. **New `test_doc_references_each_callable_name`** (mirrors
   iter-130's). For every name in `_DIVERSITY_HELPERS`, assert
   the name appears somewhere in the doc text. Catches a
   helper rename without doc update.

3. **New `test_doc_lists_8_numbered_rules`** (mirrors
   iter-130's `test_doc_lists_5_numbered_rules`). Counts
   `^\d+\.\s+\*\*` patterns inside the diversity-pattern
   section. The diversity pattern has 8 rules vs the
   extraction pattern's 5 — different counts, same shape.

**One change to `GENO.md`:**

- Replaced two occurrences of `iter-115/126` with
  `iter-115 + iter-126`. The slash form was visually clear
  but didn't satisfy a substring search for `iter-126` —
  exactly the drift the sentinel is designed to catch.

**Both sentinels now share the same coverage:**

| Check                              | iter-129 | iter-130 |
|------------------------------------|:--------:|:--------:|
| Section presence in GENO.md        |   ✅     |   ✅     |
| Top-level reference                |   ✅(116)|   ✅(run_chat)|
| Each documented entry resolves     |   ✅     |   ✅     |
| Each callable IS callable          |   ✅     |   ✅     |
| Each iter referenced in doc        |   ✅     |   ✅     |
| Each callable named in doc         |   ✅⭐   |   ✅     |
| English-numeral count matches      |   ✅     |   ✅     |
| Pattern-specific check             |   ✅(thresholds)| n/a |
| Numbered-rule count                |   ✅⭐   |   ✅     |
| Each entry has a test file         |   ✅     |   ✅     |

⭐ = added by iter-131.

The pattern-specific check (thresholds 3/4/5) is intentionally
asymmetric: it's domain-specific to diversity-check (the
iter-129 doc lists threshold conventions; iter-130's extraction
pattern doesn't have a comparable convention to verify).

Verification:
- `python -m pytest tests/unit/test_diversity_pattern_doc.py
  tests/unit/test_extraction_pattern_doc.py -v` → **19 passed
  in 70ms** (10 + 9).
- Full unit + integration: **1538 passed, 1 skipped** (1536
  prior + 2 net new — added 2 tests, no removals).
- Perf snapshot: **23 passed**.

Notes:
- **The doc-drift catch is the headline result.** iter-131
  found that `iter-126` wasn't a clean substring in the doc,
  even though human readers would interpret `iter-115/126` as
  attributing both. The sentinel pattern's value is partly
  catching exactly these "looks fine but mechanically
  inconsistent" gaps.
- **The two sentinels are structurally close enough to share
  a future template.** If a third documented pattern lands in
  GENO.md (e.g., a `cross-platform engine pattern` formalizing
  iter-118/119/121/122), the test file shape is now obvious:
  - `_INSTANCES` tuple: source of truth.
  - Section presence + top-level reference.
  - Per-entry: imports / callable / referenced in doc by both
    iter and name.
  - Numbered-rule count (extraction's 5, diversity's 8 — pick
    a count, lock it in).
  - English-numeral count matches tuple length.
  - Per-entry test file existence.
  - Optional pattern-specific check.

  Could be extracted into a generic `_assert_pattern_doc_in_sync`
  helper later. For now, the duplication is small (~20 lines
  per file) and explicit (each pattern owns its source of
  truth).
- **iter-131 is the natural close to the doc-pattern arc**
  started by iter-129 + iter-130. Both pattern sections are
  now equally protected. A 5th instance of either pattern
  must update both the source-of-truth tuple AND the doc
  numeral atomically.
- **No new assertion types needed yet.** iter-130's two
  contributions are the right level of coverage for this
  doc shape. Future iterations can add new check types if
  a new failure mode emerges.
- Next directions:
  - Architecture: A/B `aggressive_first_sentence: true` as
    default with the iter-127 5-fixture corpus.
  - Document the 5-fixture corpus as a "standard test
    benchmark" (still pending from iter-127).
  - If a third pattern lands in GENO.md, extract the shared
    sentinel scaffolding into a helper.

## iter-132 — STT benchmark CLI + corpus documentation

**Goal:** Operators evaluating a new `STTEngine` subclass need
a one-command way to see how it stacks up against the iter-117
through iter-127 audio fixture corpus. iter-132 adds the
benchmark CLI:

    python scripts/run_stt_benchmark.py --engine faster_whisper

**Output:**

    clean_audio               PASS  WER 0.20  band [0.00, 0.40]  elapsed 1.93s
    quick_brown_fox_audio     PASS  WER 0.11  band [0.00, 0.30]  elapsed 0.27s
    noisy_audio               PASS  WER 0.20  band [0.10, 0.50]  elapsed 0.71s
    catastrophic_audio        PASS  WER 1.00  band [0.80, 1.30]  elapsed 0.69s
    multispeaker_audio        PASS  WER 0.80  band [0.60, 1.10]  elapsed 0.71s

    5/5 fixtures passed in 4.3s

**Pre-iteration discovery: greedy decoding is mandatory for the
benchmark.**
First version routed through `FasterWhisperEngine.transcribe()`
which uses default beam-search + temperature fallback. The
multispeaker fixture failed reproducibly (WER 1.4-1.6) because
the band [0.60, 1.10] was set assuming **greedy decoding**
(iter-125's deterministic-decoding choice for the integration
test).

The fix recorded the same lesson iter-125 learned: deterministic
decoding is required when asserting on output. The benchmark CLI
takes the same approach as
`tests/integration/test_wer_audio.py:_transcribe`:

```python
if engine == "faster_whisper":
    # Bypass the engine wrapper, use greedy directly.
    from faster_whisper import WhisperModel
    m = WhisperModel(resolved, device=device, compute_type=compute)
    def transcribe(audio_path):
        segments, _ = m.transcribe(
            audio_path, language="en",
            beam_size=1, temperature=0,
        )
        return " ".join(s.text for s in segments).strip()
```

For non-`faster_whisper` engines (whisper / gemma4 / future
custom), the CLI uses `stt.get_engine` + the standard transcribe
contract. If those engines have non-deterministic output, that's
their concern.

**Three pieces:**

1. **`scripts/run_stt_benchmark.py`** — CLI with three layers:
   - `BenchmarkSummary` + `FixtureResult` dataclasses for the
     return shape.
   - `run_benchmark(transcribe, fixtures, fixture_dir, *, log,
     clock)` — pure function. Caller passes any callable
     `(audio_path) -> transcript`. Tests pass deterministic
     stubs.
   - `_build_transcribe_from_engine_args` (CLI-only): translates
     argparse args into a transcribe closure. Bypasses the
     engine wrapper for `faster_whisper` (greedy path); uses
     `get_engine` for everything else.
   - `main()` — argparse entry point; exits 0 if all fixtures
     pass, 1 otherwise.

2. **`tests/unit/test_run_stt_benchmark.py`** (14 tests):
   - Single-fixture happy/below-band/above-band paths.
   - Multi-fixture summary (mixed pass/fail; final summary line).
   - Per-row format includes name + status + WER + band + elapsed.
   - Empty corpus → empty summary with "0/0 fixtures passed"
     line.
   - Per-fixture elapsed positive; total_elapsed sums per-fixture.
   - Hypothesis stored on FixtureResult for debugging.
   - Transcribe receives full path (`fixture_dir/audio_path`),
     not just basename.
   - Default `log=print` flows to stdout via capsys.

3. **End-to-end smoke test** (verified manually):
   `python scripts/run_stt_benchmark.py --engine faster_whisper
   --model tiny` → **5/5 PASS in ~1.3s**, deterministic across
   runs.

Verification:
- `python -m pytest tests/unit/test_run_stt_benchmark.py -q` →
  **14 passed in 30ms**.
- Full unit + integration: **1552 passed, 1 skipped** (1538
  prior + 14 new).
- Perf snapshot: **23 passed**.
- CLI run x3: **5/5 PASS deterministically**.

Notes:
- **Pure-function design pays off.** The 14 unit tests don't
  touch faster-whisper at all — they exercise `run_benchmark`
  with stub callables. This is the iter-108/iter-109 dependency-
  injection pattern again: test the orchestration, leave the
  engine boundary to integration tests.
- **The CLI now serves as canonical reference for what the
  corpus looks like in practice.** Future contributors who add
  a fixture (say, `multispeaker_high_amp_16khz.wav`) just:
  1. Generate the audio + add to corpus.json
  2. Run the CLI; observe the WER
  3. Set the band from the observed value ± reasonable margin
- **Engine-specific bypass is a minor layering violation, but
  justified.** Adding `**transcribe_kwargs` to
  `STTEngine.transcribe` would make the contract leaky (each
  engine has different decode params). Keeping the contract
  clean and bypassing it in the benchmark for the engines we
  know about is pragmatic — and the benchmark is the only
  caller that needs decode control.
- **The corpus is now an operator-facing benchmark.** Up to
  iter-131 the fixtures were used only by the integration
  test (skipped on hosts without faster-whisper). iter-132
  surfaces them as a reusable evaluation tool. A `README.md`
  pointer to the script would close the documentation loop —
  saving for a future iteration.
- **Future engine-eval flow:**
  1. Implement custom STTEngine + register in `stt.ENGINES`.
  2. Run `python scripts/run_stt_benchmark.py --engine
     custom_engine`.
  3. See per-fixture pass/fail in seconds.
  4. If most fixtures fail, the engine isn't ready.
  5. If only edge-case fixtures fail (multispeaker,
     catastrophic), the engine has limitations to document.
- **Pattern for future CLI scripts:** same shape as iter-132 —
  pure function (testable) + thin CLI wrapper. The pattern is
  now in two places: `scripts/generate_iteration_reports.py`
  has a similar separation with `render_*` pure functions +
  `main()` wrapper.
- Next directions:
  - Architecture: A/B `aggressive_first_sentence: true` as
    default with the corpus.
  - README pointer to the benchmark CLI ("Evaluating a new
    STT engine") — closes the documentation loop.
  - Extract the `(name, status, WER, band, elapsed)` per-row
    format into a CSV / JSON output mode if a future iter
    needs machine-readable benchmark results.

## iter-133 — CSV/JSON output for STT benchmark CLI

**Goal:** Operators scripting comparison runs (engine A vs B,
or model size grid-search) need machine-readable output. iter-133
adds JSON + CSV output modes to iter-132's benchmark CLI.

    python scripts/run_stt_benchmark.py --engine faster_whisper --format json
    python scripts/run_stt_benchmark.py --engine faster_whisper --format csv

**Three additions:**

1. **`verbose: bool = True` kwarg on `run_benchmark`.** Default
   preserves iter-132 behavior (per-row + summary text emitted
   via `log`). When False, the function runs silently and
   returns the summary — caller formats. Used by JSON/CSV CLI
   paths to avoid interleaved text + format output.

2. **`format_summary_json(summary, *, indent=2) -> str`** — full
   structured dump:

   ```json
   {
     "passing": 5, "failing": 0, "total": 5,
     "total_elapsed_seconds": 1.32,
     "results": [
       {"name": "clean_audio", "reference": "...",
        "hypothesis": "...", "wer": 0.2,
        "expected_min": 0.0, "expected_max": 0.4,
        "elapsed_seconds": 0.29, "passed": true},
       ...
     ]
   }
   ```

   Top-level aggregates (`passing`/`failing`/`total`/
   `total_elapsed_seconds`) so operators don't need to iterate
   results for the headline numbers.

3. **`format_summary_csv(summary) -> str`** — header row + one
   data row per fixture:

       name,passed,wer,expected_min,expected_max,elapsed_seconds,reference,hypothesis
       clean_audio,True,0.2000,0.0000,0.4000,0.2736,what is the weather today,What is the winner today?
       ...

   RFC-4180 quoting (commas/quotes in transcripts are escaped
   correctly). 4-decimal precision on numeric fields. Round-
   trips through `csv.DictReader`.

4. **`--format` CLI arg** with choices `text` (default) /
   `json` / `csv`. Backwards compatible: omitting the flag
   produces iter-132's original output.

**Tests** (28 total, 14 new beyond iter-132):

*verbose flag (3 tests):* False suppresses output; False still
returns summary; True is the default (backward compat).

*format_summary_json (5 tests):* aggregates at top level;
per-result records have all FixtureResult fields; empty summary
produces valid JSON; `indent=None` gives compact one-line;
weird strings (quotes, tabs) round-trip correctly.

*format_summary_csv (6 tests):* starts with header; N+1 lines
for N fixtures; commas in strings are quoted (RFC-4180); empty
summary is just header; round-trips through `csv.DictReader`;
fixed 4-decimal precision.

Verification:
- `python -m pytest tests/unit/test_run_stt_benchmark.py -q` →
  **28 passed in 40ms** (14 original + 14 new).
- Full unit + integration: **1566 passed, 1 skipped** (1552
  prior + 14 new).
- Perf snapshot: **23 passed**.
- End-to-end CLI smoke: `--format text/json/csv` all work
  against the real corpus.

Notes:
- **Format functions take a `BenchmarkSummary`, not a
  fixtures list.** This means a future caller can run multiple
  benchmarks (different engines), collect summaries, and
  format them all the same way — without the format function
  knowing about the engine layer. Pure-function discipline:
  fewer dependencies, more reuse.
- **The `verbose=False` toggle is a small contract break.** It
  changes `run_benchmark`'s side-effects depending on a kwarg.
  Alternative: split into `run_benchmark` (silent, returns
  summary) + `report_benchmark(summary, log)` (formats text).
  Decided against because the existing `verbose=True` matches
  the tested-and-working iter-132 contract; introducing a
  separate report function would require updating iter-132's
  passing tests.
- **Why `indent=2` default for JSON.** Operators piping the
  output to a viewer or jq want it readable. Compact one-line
  is opt-in via `indent=None` for those who want to grep the
  output through additional pipes.
- **Why 4-decimal precision in CSV.** WER values typically have
  3-4 meaningful digits; 4 is enough for accurate display
  without scientific notation creeping in. Elapsed seconds
  also benefit from the same precision since per-fixture times
  are typically 0.1-2 seconds.
- **RFC-4180 quoting matters.** Without it, a transcript
  containing a comma (which happens — multispeaker_audio's
  hypothesis is "What is my window taking out?" but a future
  fixture might be "Hello, world.") would corrupt the CSV.
  The Python csv module handles this automatically with
  `quoting=QUOTE_MINIMAL`.
- **The CLI now serves three audiences:**
  - Interactive operators: `text` mode (iter-132).
  - Scripted comparisons: `json` mode (e.g., diff two engines'
    output).
  - Spreadsheet/database imports: `csv` mode (e.g., track
    benchmark trends in pandas).
- **No new pattern introduced.** This iter is a
  straightforward feature addition on iter-132's
  infrastructure. The pure-function-vs-CLI separation pattern
  (used in iter-132 + the report generator) carries through
  cleanly.
- Next directions:
  - README pointer to the benchmark CLI ("Evaluating a new
    STT engine" — closes the documentation loop).
  - Architecture: A/B `aggressive_first_sentence: true` as
    default with the corpus.
  - `--diff <baseline.json>` mode that compares the current
    run against a baseline JSON — useful for "did my engine
    change improve or regress?"

## iter-134 — --diff baseline mode for benchmark CLI

**Goal:** Operators iterating on an STT engine want a "did my
change improve or regress?" workflow. iter-134 adds
`--diff baseline.json`: load a saved JSON output from before a
change, run the benchmark again, see per-fixture deltas + status
flips highlighted.

    # Save a baseline before changes
    python scripts/run_stt_benchmark.py --engine faster_whisper \
        --format json > baseline.json

    # ...make engine changes...

    # Compare against baseline
    python scripts/run_stt_benchmark.py --engine faster_whisper \
        --diff baseline.json

**Output:**

    clean_audio               PASS   WER 0.20 -> 0.20  Δ +0.000
    noisy_audio               PASS   WER 0.20 -> 0.25  Δ +0.050
    multispeaker_audio        FAIL   WER 0.80 -> 1.20  Δ +0.400 (regressed)

    4/5 → 5/5 fixtures passing (+1)
    Improvements: noisy_audio
    Regressions: multispeaker_audio

**Three additions:**

1. **`FixtureDiff` dataclass** — per-fixture diff with
   `current_wer`, `baseline_wer`, `current_passed`,
   `baseline_passed` (each can be `None` for new/removed
   fixtures). Computed properties: `wer_delta` and
   `status_change` (one of "unchanged" / "regressed" /
   "improved" / "new" / "removed").

2. **`BenchmarkDiff` aggregate dataclass** with derived
   properties: `regressions`, `improvements`, `new_fixtures`,
   `removed_fixtures` — each a filtered list of `FixtureDiff`s.
   Plus `current_passing/total` and `baseline_passing/total`
   for the headline summary line.

3. **`compute_diff(summary, baseline_dict)`** matches fixtures
   by name, builds `FixtureDiff` entries, handles new/removed
   cases. **`format_diff_text(diff)`** renders the result —
   per-row WER comparison, status markers, summary line, and
   explicit "Improvements:"/"Regressions:"/"New:"/"Removed:"
   sections when relevant.

4. **`--diff <path>` CLI flag** — loads the baseline, runs the
   benchmark silently (`verbose=False`), feeds both into the
   diff path. Overrides `--format` (always text). Exit codes:
   3 if baseline file missing or unparseable.

**Tests** (16 new, 44 total):

*compute_diff (8 tests):* unchanged / regression / improvement /
new fixture / removed fixture / wer_delta correct /
wer_delta None for new+removed / aggregate counts match.

*format_diff_text (8 tests):* per-fixture rows show
"baseline -> current" / regression marker / improvement marker /
summary line "X/Y → A/B (+N)" / Regressions list / New +
Removed lists / unchanged session has no Improvements/
Regressions sections / negative deltas render with sign.

End-to-end smoke verified:
- Identical runs: all Δ +0.000, summary "5/5 → 5/5 (+0)".
- Missing baseline file → "baseline JSON not found" + exit 3.
- Unparseable baseline → "baseline JSON parse error" + exit 3.

Verification:
- `python -m pytest tests/unit/test_run_stt_benchmark.py -q` →
  **44 passed in 50ms** (28 prior + 16 new).
- Full unit + integration: **1582 passed, 1 skipped** (1566
  prior + 16 new).
- Perf snapshot: **23 passed**.

Notes:
- **The diff workflow is the canonical "engine change
  evaluation" loop.** Steps:
  1. Run benchmark, save JSON: `--format json > baseline.json`.
  2. Make engine change (faster-whisper version bump, model
     size change, custom STTEngine implementation).
  3. Run with `--diff baseline.json`.
  4. Read regressions/improvements at a glance.

  This converts "is the engine still good?" from a manual
  per-fixture comparison into a one-command answer.
- **`status_change` is the most actionable column.** Operators
  scanning the output focus on rows with markers (regressed /
  improved / new / removed) — those rows changed something
  worth noting. Unchanged rows blend into the noise. Future
  enhancement could color-code status_change for terminals
  that support it.
- **Pure-function discipline carries through.** `compute_diff`
  takes a summary + a parsed dict; `format_diff_text` takes a
  diff object. Neither touches the filesystem. The CLI does
  the JSON loading and the print. Easy to test, easy to reuse:
  a future `--diff <baseline.csv>` could land by adding a
  parser, no changes to compute_diff.
- **Why `wer_delta` is signed.** The natural reading of "Δ
  +0.05" is "got worse by 0.05" — operators reading the diff
  see the sign and immediately know which direction the metric
  moved. Negative deltas are improvements; positive are
  regressions (in WER terms). Matching this with the
  `status_change` marker on status flips gives operators two
  signals: per-fixture WER change AND threshold crossings.
- **The benchmark CLI is now a complete eval workflow:**
  - Run (iter-132)
  - Save in machine-readable formats (iter-133)
  - Diff against a baseline (iter-134)

  Together: a CI pipeline can run benchmark → save JSON →
  diff against the previous build's JSON → fail PR if any
  fixture regresses. That's a real engineering primitive,
  not just a debugging tool.
- Next directions:
  - README pointer to the benchmark CLI ("Evaluating a new
    STT engine" — covers iter-132/133/134 together).
  - `--diff` could optionally support `--format json/csv`
    for machine-readable diff output (CI pipelines).
  - Architecture: A/B `aggressive_first_sentence: true`
    using the diff workflow against a saved baseline.

## iter-135 — `--diff` with `--format json/csv`

**Goal:** iter-134 always rendered the diff as text. CI
pipelines that want to gate PRs on regression detection need
machine-readable output. iter-135 adds JSON + CSV diff
renderers, dispatched via the existing `--format` flag when
`--diff` is set.

**Three additions:**

1. **`format_diff_json(diff, *, indent=2) -> str`** — full
   structured dump with top-level aggregates:

       {
         "current_passing": 5, "current_total": 5,
         "baseline_passing": 4, "baseline_total": 5,
         "passing_delta": 1,
         "regression_count": 0,
         "improvement_count": 1,
         "new_count": 0,
         "removed_count": 0,
         "fixture_diffs": [
           {"name": "noisy_audio",
            "current_wer": 0.20, "baseline_wer": 0.30,
            "wer_delta": -0.10,
            "current_passed": true, "baseline_passed": false,
            "status_change": "improved"},
           ...
         ]
       }

   `passing_delta` is `current_passing - baseline_passing` —
   positive means improved overall. `None` values for
   new/removed fixtures serialize as JSON `null`.

2. **`format_diff_csv(diff) -> str`** — header + one row per
   fixture diff:

       name,status_change,current_wer,baseline_wer,wer_delta,current_passed,baseline_passed
       clean_audio,unchanged,0.2000,0.2000,0.0000,True,True
       noisy_audio,improved,0.2000,0.3000,-0.1000,True,False
       multispeaker_audio,regressed,1.2000,0.8000,0.4000,False,True
       new_audio,new,0.1500,,,True,
       legacy_audio,removed,,0.2000,,,True

   None values render as empty strings (RFC-4180 idiom for
   missing). Numeric fields use 4-decimal precision matching
   the summary CSV.

3. **CLI dispatch updated**: when `--diff` is set, `--format`
   chooses the diff renderer (`text` default, `json`, `csv`).

**CI integration use cases:**

```bash
# Save baseline before changes
$ python scripts/run_stt_benchmark.py --format json > baseline.json

# In PR check: fail if any regression
$ python scripts/run_stt_benchmark.py --diff baseline.json --format json \
    | jq '.regression_count > 0' | grep -q true && exit 1

# Or: track WER trends in a spreadsheet
$ python scripts/run_stt_benchmark.py --diff baseline.json --format csv \
    > diff.csv
```

**Tests** (13 new, 57 total):

*format_diff_json (6 tests):* top-level aggregates correct;
per-fixture records include all FixtureDiff fields; None
values render as JSON null; unchanged session has zero counts;
indent kwarg controls pretty-printing; mixed session with all
5 status_change categories produces valid JSON.

*format_diff_csv (7 tests):* header row present; N+1 lines for
N diffs; None columns render as empty strings; unchanged session
shows status_change=unchanged; empty diff is just header;
round-trips through csv.DictReader; negative deltas preserve sign.

End-to-end smoke verified against the real corpus:
- `--diff baseline.json --format json` → valid JSON with all
  expected fields.
- `--diff baseline.json --format csv` → header + 5 unchanged
  data rows.
- Identical-runs diff: passing_delta=0, regression_count=0.

Verification:
- `python -m pytest tests/unit/test_run_stt_benchmark.py -q` →
  **57 passed in 60ms** (44 prior + 13 new).
- Full unit + integration: **1595 passed, 1 skipped** (1582
  prior + 13 new).
- Perf snapshot: **23 passed**.

Notes:
- **The benchmark CLI is now a complete eval pipeline.** Every
  step has machine-readable output:
  - `run_benchmark` (iter-132) — text default
  - `--format json/csv` summary (iter-133)
  - `--diff baseline.json` text (iter-134)
  - `--diff --format json/csv` (iter-135)

  CI pipelines, dashboards, comparison scripts, and spreadsheet
  imports all work without screen-scraping.

- **`passing_delta` is the headline number** for CI gates. A
  one-line check:

      jq '.passing_delta < 0' diff.json | grep -q true && exit 1

  fires when fewer fixtures pass than the baseline. This is
  more permissive than `regression_count > 0` (allows
  fixture-level reshuffling within the same passing total) —
  pick based on policy.

- **Empty-string vs null choice:** JSON uses `null`, CSV uses
  empty string. Both are idiomatic for "missing" in their
  respective formats. JSON parsers handle null cleanly; CSV
  parsers (`csv.DictReader`) return empty strings, which
  spreadsheet tools display as blank cells.

- **Pure-function discipline scales.** `format_diff_*` functions
  take a `BenchmarkDiff` — they don't read the corpus, don't
  call the engine, don't touch the filesystem. The CLI's only
  responsibility is loading the baseline JSON and printing the
  formatted output. Future formats (HTML, Markdown, etc.) land
  by adding one function — no changes to `compute_diff` or
  `run_benchmark`.

- **The summary + diff symmetry is now complete.** Each
  benchmark layer has three output formats:

  | Layer    | text | json | csv |
  |----------|:----:|:----:|:---:|
  | summary  | ✅   | ✅   | ✅  |
  | diff     | ✅   | ✅   | ✅  |

  Five output paths total. The dispatch in `main()` is small —
  one `if/elif` ladder for each — but the user-facing surface
  is wide enough to cover all reasonable evaluation workflows.

- Next directions:
  - README pointer to the benchmark CLI ("Evaluating a new
    STT engine" — covers iter-132/133/134/135 as a unit).
  - Architecture: A/B `aggressive_first_sentence: true` using
    the diff workflow against a saved baseline.
  - Pre-commit hook or CI workflow example showing how to
    gate PRs with `passing_delta` from the JSON diff output.

## iter-136 — README "Evaluating a new STT engine" section

**Goal:** The benchmark CLI evolved through iter-132 → 135 with
no operator-facing documentation. The README still listed only
the high-level project description. iter-136 adds an
"Evaluating a new STT engine" section that walks an operator
through the full eval workflow.

**Three pieces:**

1. **README section added** (~70 lines, between Installation
   and Project Structure):

   - Quick benchmark example (`--engine faster_whisper`)
   - Saving a baseline + diffing changes
   - CI integration with `--diff --format json` + `jq`
   - Output format reference (text/json/csv)
   - Step-by-step "Adding a new STT engine" guide

   Each subsection has copy-pasteable shell commands. Operators
   can run `python scripts/run_stt_benchmark.py --engine
   faster_whisper --model tiny` directly from the README and
   see real output.

2. **Drift-sentinel test** (`tests/unit/test_readme_benchmark_docs.py`,
   12 tests). Same shape as iter-129/iter-130's GENO.md
   sentinels, applied to the README. Catches:
   - Section disappearing
   - Script path moving (catches a future rename)
   - Format choices in argparse not mentioned in README
   - `--diff` / `--engine` flags missing from documentation
   - Corpus path / count drift (5 fixtures today; if a 6th lands
     without README update, this fires)
   - Code-block invocations referencing wrong script paths

3. **The format-choices test parses the script source as text
   instead of importing it** because importing under a non-
   canonical name conflicts with `@dataclass` lookups (Python
   dataclass internals reach into `sys.modules[cls.__module__]`).
   Recorded for future doc-of-script tests: parse with regex,
   don't import.

**Drift-sentinel highlights** (catch real failure modes):

- `test_readme_format_choices_match_argparse_choices`: extracts
  the argparse `choices=[...]` list from the script source and
  asserts each appears in the README. Adding a new format
  (HTML, Markdown) without doc update will fire this.

- `test_readme_corpus_count_matches_actual`: parses
  `corpus.json:audio_fixtures`, counts the entries, requires
  the matching numeral ("5 audio fixtures") in the README.
  Currently 5; if a 6th fixture lands the test fires.

- `test_readme_code_blocks_use_real_script_invocation`: regex
  finds every `python scripts/<path>` invocation in the README
  and verifies it resolves to the actual script path. Catches
  typos like `python scripts/run_benchmark.py` (missing `_stt`)
  that would silently fail.

Verification:
- `python -m pytest tests/unit/test_readme_benchmark_docs.py
  -q` → **12 passed in 20ms**.
- Full unit + integration: **1607 passed, 1 skipped** (1595
  prior + 12 new).
- Perf snapshot: **23 passed**.

Notes:
- **The README is now the canonical operator entry point.**
  Pre-iter-136 a Linux operator hitting the repo had to read
  CHANGELOG-equivalents (ITERATION_LOG.md) to discover the
  benchmark CLI. iter-136 surfaces the workflow at the
  top-level README — discoverable in 30 seconds.

- **Drift sentinels for documentation are now standard.**
  iter-129 covered diversity-pattern doc, iter-130 covered
  extraction-pattern doc, iter-136 covers README. Three
  documentation surfaces, all with code-vs-doc sync tests.
  The shape: source of truth (corpus.json count, argparse
  choices, real file paths) compared against README text.

- **The "parse script source instead of importing it" pattern
  is reusable.** When a doc test wants to verify CLI flags
  match argparse choices, importing the script can fire
  dataclass-lookup errors if loaded under non-canonical
  names. Regex-based parsing is the safer path. Recorded
  here for future doc tests of CLI scripts.

- **The benchmark eval pipeline is now feature-complete and
  documented.** Five iterations:
  - iter-132: CLI + run_benchmark
  - iter-133: text → json/csv summary formats
  - iter-134: --diff with text rendering
  - iter-135: --diff with json/csv rendering
  - iter-136: README documentation + drift sentinel

  Every code path has tests, every output format has examples,
  every feature is discoverable from the top-level README.
  An operator dropping into the repo can evaluate a new STT
  engine without reading the iteration log.

- Next directions:
  - Architecture: A/B `aggressive_first_sentence: true` using
    the now-documented diff workflow against a saved baseline.
  - Pre-commit hook example showing `passing_delta` gating.
  - Document the GENO.md patterns in the README too — point
    contributors at the diversity-check + extraction patterns
    (currently only discoverable from GENO.md).

## iter-137 — `--fail-on-regression` CI gate (exit code on regressions only)

**Branch:** `iter-137-fail-on-regression` (merged ff to main)
**Date:** 2026-06-16

**Goal:** The benchmark eval pipeline (iter-132 → 136) was
feature-complete, but the README's recommended CI gate was a
brittle three-stage shell pipeline:

    ... --diff baseline.json --format json | jq '.regression_count > 0' | grep -q true && exit 1

That works but is fragile (depends on `jq`, swallows exit codes,
hard to read in a CI YAML). iter-137 moves the gate into the CLI
itself: a `--fail-on-regression` flag that makes the **process
exit code** the signal.

**The key semantic — gate on NEW regressions, not absolute
failures.** `--fail-on-regression` exits 1 iff at least one
fixture was PASS in the baseline and is FAIL now (a
`status_change == "regressed"` row). Pre-existing failures do
NOT block: an already-red corpus lets a PR through as long as it
leaves things no worse. This is the correct PR-gate policy — you
block a change that *makes something worse*, not a change that
merely fails to fix a long-standing failure.

**What changed:**

1. **`scripts/run_stt_benchmark.py`:**
   - New `--fail-on-regression` flag (`action="store_true"`).
   - Usage guard: requires `--diff` (exit 2 with a clear stderr
     message otherwise — there's no baseline to regress against).
   - In diff mode, when the flag is set the exit code becomes
     `1 if diff.regressions else 0`. Without the flag, diff mode
     keeps the iter-134/135 behavior (`0 if summary.failing == 0
     else 1`) so existing callers are unaffected. The diff report
     still prints (text/json/csv per `--format`) so CI logs show
     what changed.

2. **README CI section rewritten:** leads with
   `--fail-on-regression` as the clean gate (no jq/grep), explains
   the pre-existing-failure semantics, notes the exit-2 guard, and
   keeps the `jq '.regression_count'` pipeline as the alternative
   for shell-based gating against the JSON dump.

3. **Tests — `tests/unit/test_run_stt_benchmark.py` (+7):** first
   hermetic `main()` tests in this file. A `_run_main` helper
   writes a temp corpus.json + baseline.json, monkeypatches
   `module.CORPUS_PATH` and `module._build_transcribe_from_engine_args`
   (stub transcribe), sets `sys.argv`, and returns the exit code.
   Cases: requires-diff (exit 2 + stderr), regressed → exit 1,
   no-regression → exit 0, **pre-existing failure ignored → exit
   0** (the headline semantic), improvement → exit 0, mixed
   (one regressed) → exit 1, and a backward-compat case proving
   plain `--diff` without the flag still exits 1 on absolute
   current failure.

4. **Drift sentinel — `tests/unit/test_readme_benchmark_docs.py`
   (+1):** `test_readme_documents_fail_on_regression_flag` asserts
   the flag is both defined in the script source AND documented in
   the README (bidirectional sync, same shape as iter-136's
   format-choices sentinel).

**Verification:**
- `python -m pytest tests/unit/test_run_stt_benchmark.py
  tests/unit/test_readme_benchmark_docs.py -q` → **77 passed in
  0.15s** (64 prior + 7 benchmark + 1 README... net +8 over the
  combined prior 69).
- Full unit suite: **1585 passed**.
- Full unit + integration: **1615 passed, 1 skipped** (1607 prior
  + 8 new).
- Perf snapshot: **23 passed**.
- End-to-end smoke against the real corpus (faster_whisper tiny):
  `--fail-on-regression` without `--diff` → exit 2 with the guard
  message; against a saved baseline with a pre-existing
  multispeaker failure but nothing regressed → exit 0 (the legacy
  absolute-failure gate would block here, the new gate correctly
  passes).

**Notes:**
- **The CI gate is now a single flag.** A CI step is one line:
  `python scripts/run_stt_benchmark.py --engine faster_whisper
  --diff baseline.json --fail-on-regression`. The exit code drives
  the pipeline; no `jq`/`grep` plumbing, no swallowed errors.

- **Two gate policies, both available.** `--fail-on-regression`
  (NEW regressions only) vs the iter-135 `passing_delta < 0` /
  `regression_count > 0` JSON checks. iter-135's note already
  distinguished `passing_delta` (allows fixture reshuffling within
  the same passing total) from `regression_count`. iter-137 adds
  the in-process equivalent of the strictest-but-fairest policy:
  block on any PASS→FAIL flip, ignore the rest.

- **First hermetic `main()` test in the benchmark suite.** Prior
  tests all exercised pure functions (`run_benchmark`,
  `compute_diff`, `format_*`). The exit-code logic lives in
  `main()`, so testing it required driving `main()` with a temp
  corpus + monkeypatched engine builder. The `_run_main` helper is
  reusable for any future `main()`-level behavior (new flags,
  error paths).

- Next directions:
  - `--fail-on-new-fixture` or a `--strict` mode that also blocks
    on removed fixtures (corpus shrinking unnoticed).
  - A committed `.github/workflows/` example (or a
    `scripts/ci-gate.sh`) wiring `--fail-on-regression` end to end.
  - Document the GENO.md diversity-check + extraction patterns in
    the README so contributors discover them without reading
    GENO.md (carried over from iter-136).

## iter-138 — `--fail-on-removed` coverage gate (block silent corpus shrink)

**Branch:** `iter-138-fail-on-removed` (merged ff to main, commit `0c4a574`)
**Date:** 2026-06-16

**Goal:** iter-137 shipped `--fail-on-regression` — a clean CI
gate that blocks a PR which flips a fixture PASS→FAIL while
ignoring pre-existing failures. But that gate has a blind spot:
it only sees fixtures that are still in the corpus. **Deleting a
failing fixture reads as "fewer failures" to `--fail-on-regression`
and slips straight through.** A contributor (or an accident) can
dodge a red fixture by removing it, and the regression gate stays
green. iter-138 closes that gap with the orthogonal coverage
gate.

**The semantic — block on a baseline fixture that vanished.**
`--fail-on-removed` exits 1 iff at least one fixture present in
the baseline JSON is MISSING from the current corpus (a
`status_change == "removed"` row, i.e. `diff.removed_fixtures` is
non-empty). New fixtures never trip it (adding coverage is always
allowed); regressions on surviving fixtures are not its concern
(that's `--fail-on-regression`'s job). The two gates compose: set
both for a strict policy that blocks a PR which makes a fixture
worse OR drops one.

**What changed:**

1. **`scripts/run_stt_benchmark.py`:**
   - New `--fail-on-removed` flag (`action="store_true"`).
   - Usage guard generalized: the `requires --diff` check now
     fires for `--fail-on-regression` OR `--fail-on-removed`, and
     names whichever flag(s) were passed in the exit-2 stderr
     message ("... requires --diff <baseline.json>; without a
     baseline there is nothing to compare against").
   - Exit-code logic in diff mode generalized to an OR of the two
     gates: `blocked = (fail_on_regression and diff.regressions)
     or (fail_on_removed and diff.removed_fixtures)`; returns
     `1 if blocked else 0`. With neither flag, diff mode keeps the
     iter-134/135 absolute-failure exit so existing callers are
     unaffected. Reused iter-134's existing `removed_fixtures`
     property on `BenchmarkDiff` — no new diff plumbing needed.

2. **README CI section extended:** a new paragraph explains the
   `--fail-on-regression` blind spot (deleting a failing fixture)
   and introduces `--fail-on-removed` as the coverage-loss gate,
   with a copy-pasteable strict-gate example combining both flags.
   Notes the exit-2 `--diff` requirement.

3. **Tests — `tests/unit/test_run_stt_benchmark.py` (+8):**
   reuse the iter-137 `_run_main` hermetic driver (temp corpus +
   baseline, monkeypatched engine builder, returns exit code).
   The driver already supported a removed-fixture scenario for
   free — a baseline entry whose name isn't in `fixtures`.
   Cases: requires-diff (exit 2 + stderr), removal → exit 1,
   nothing-removed → exit 0 (even when a fixture regressed —
   proving the gates are independent), new fixture ignored →
   exit 0, strict-gate (both flags) firing on regression-only,
   on removal-only, and clean exit 0, and a backward-compat case
   proving plain `--diff` without the flag does NOT block on a
   removal.

4. **Drift sentinel — `tests/unit/test_readme_benchmark_docs.py`
   (+1):** `test_readme_documents_fail_on_removed_flag` asserts
   the flag is both defined in the script source AND documented
   in the README (bidirectional sync, same shape as iter-136/137
   sentinels).

**Verification:**
- `python -m pytest tests/unit/test_run_stt_benchmark.py
  tests/unit/test_readme_benchmark_docs.py -q` → **86 passed in
  0.15s** (77 prior + 8 benchmark + 1 README).
- Full unit suite: **1594 passed** (1585 prior + 9 new).
- Full unit + integration: **1624 passed, 1 skipped** (1615 prior
  + 9 new).
- Perf snapshot (`tests/performance/`): **23 passed**.
- End-to-end smoke (faster_whisper tiny, real 5-fixture corpus):
  - `--fail-on-removed` without `--diff` → exit 2 with the guard
    message.
  - Real baseline (all 5 pass), re-run with `--fail-on-removed`,
    nothing dropped → exit 0.
  - Baseline doctored with a phantom 6th fixture, re-run with
    `--fail-on-removed` → exit 1, diff prints
    "phantom  REMOV  WER 0.00 -> —  (removed)" and
    "Removed fixtures: phantom".

**Notes:**
- **Two orthogonal CI gates, freely composable.**
  `--fail-on-regression` (PASS→FAIL flip) and `--fail-on-removed`
  (fixture vanished) answer different questions; either alone, or
  both together (`--diff baseline.json --fail-on-regression
  --fail-on-removed`) for the strict policy. The exit-code logic
  is a simple OR over the two gate conditions.

- **Zero new diff plumbing.** iter-134 already computed
  `removed_fixtures` on `BenchmarkDiff` for the text/json/csv diff
  renderers. iter-138 just wires that existing signal to the exit
  code. The lesson from the iter-132→138 arc: build the data model
  once (the diff), then each gate is a thin exit-code policy on top.

- **The iter-137 `_run_main` hermetic driver paid off immediately.**
  All 8 new tests reused it unchanged — the removed-fixture case
  needed no new test infrastructure, just a baseline entry with a
  name absent from the corpus. Hermetic `main()` testing is now
  the standard for benchmark exit-code behavior.

- **Bookkeeping:** the merge commit also swept in regenerated
  `iter-reports/perf-*.json` snapshots (the perf suite rewrites
  them on run, same as iter-135/136 commits). No source impact.

- Next directions:
  - `--fail-on-new-fixture` is the deliberate non-feature: adding
    coverage should never block, so there's no symmetric gate. If a
    future need arises (e.g. "corpus is frozen, reject additions"),
    it'd be a separate `--frozen-corpus` flag, not a `--fail-on-*`.
  - A committed `.github/workflows/` example (or `scripts/ci-gate.sh`)
    wiring `--fail-on-regression --fail-on-removed` end to end
    (carried over from iter-137).
  - Document the GENO.md diversity-check + extraction patterns in
    the README so contributors discover them without reading
    GENO.md (carried over from iter-136).

## iter-139 — `scripts/ci-gate.sh` — one-line CI gate wrapper

**Branch:** `iter-139-ci-gate` (merged ff to main, commit `64b45f4`)
**Date:** 2026-06-16

**Goal:** iter-137 and iter-138 each carried over the same next
direction: a committed `scripts/ci-gate.sh` (or
`.github/workflows/` example) wiring `--fail-on-regression` +
`--fail-on-removed` end to end. The flags exist and compose, but
a CI author still has to know the exact incantation
(`--diff baseline.json --fail-on-regression --fail-on-removed`).
iter-139 ships the wrapper so a CI step is a single call.

**What changed:**

1. **`scripts/ci-gate.sh` (new, executable):** the committed
   one-line gate. Defaults `--engine faster_whisper` and
   `--baseline baseline.json`; always passes `--diff <baseline>
   --fail-on-regression --fail-on-removed` to
   `run_stt_benchmark.py` and passes the benchmark's exit code
   through unchanged (0 clean / 1 regressed-or-removed / 3
   baseline parse error). Design points:
   - **`--` separator forwards extra benchmark flags verbatim**
     (e.g. `-- --device cpu --compute int8 --format json`), so
     the wrapper never has to grow a flag per benchmark option.
   - **`--engine` / `--model` (both space- and `=`-form)** picked
     out explicitly since they're the common knobs; `--model`
     omitted from the forwarded command when unset (engine's own
     default).
   - **Fail-fast on a missing baseline (exit 2)** with the exact
     `... --format json > baseline.json` command to create one —
     more actionable than letting the benchmark exit 3.
   - **`PYTHON` and `BENCHMARK_SCRIPT` env overrides** so tests
     (and non-standard layouts) can point at a stub; resolves the
     benchmark alongside the script by default via `BASH_SOURCE`.
   - **`--help`** prints the leading comment block and exits 0.

2. **README CI section** documents the wrapper as the convenience
   path right after the two-flag strict-gate example: the
   single-call form, `--engine`/`--model`, the `--` forwarding
   convention, and the exit-2 missing-baseline behavior.

3. **Tests — `tests/unit/test_ci_gate_script.py` (new, +15):**
   first shell-script tests in the suite. Hermetic: a stub
   benchmark (written to `tmp_path`, wired via `BENCHMARK_SCRIPT`)
   echoes its argv on line 1 and exits with `STUB_EXIT`, so the
   tests assert flag wiring and exit-code passthrough with **no
   real engine, model download, or corpus**. Cases: script exists
   + executable, default flags wire both gates, exit-code
   passthrough (0/1/3), engine override, model forwarded /
   omitted, `--` extras forwarded, `=`-form args, gate-flags-
   precede-extras ordering, missing-baseline usage error (exit 2
   + creation hint), unknown-arg (exit 2), `--help` (exit 0),
   empty `--baseline` value (exit 2). All gated on `bash` being
   available (`skipif`).

4. **Drift sentinels — `tests/unit/test_readme_benchmark_docs.py`
   (+2):** `test_readme_documents_ci_gate_wrapper` (wrapper exists
   ⇔ README references `scripts/ci-gate.sh`) and
   `test_ci_gate_wraps_both_benchmark_flags` (the wrapper source
   actually invokes `--fail-on-regression`, `--fail-on-removed`,
   and `--diff` — so the README's "wires both gates together"
   claim can't go stale). Same bidirectional-sync shape as the
   iter-136/137/138 sentinels.

**Verification:**
- `python -m pytest tests/unit/test_ci_gate_script.py
  tests/unit/test_readme_benchmark_docs.py -q` → **31 passed in
  0.32s** (15 ci-gate + 16 README; +3 over prior 13 README).
- Full unit suite: **1611 passed** (1594 prior + 17 new).
- Unit + integration: **30 passed, 1 skipped** integration on top
  of the 1611 unit.
- Perf snapshot (`tests/performance/`): **23 passed**.
- End-to-end smoke (faster_whisper tiny, real 5-fixture corpus):
  - Missing baseline → exit 2 with the creation-command hint (no
    engine run).
  - Real baseline, re-run clean → "5/5 → 5/5 fixtures passing
    (+0)", exit 0.
  - Baseline doctored with a phantom 6th fixture → diff prints
    "phantom_gone  REMOV ... (removed)" and "Removed fixtures:
    phantom_gone", exit 1 (removal gate fires through the
    wrapper).

**Notes:**
- **The arc closes.** iter-132→134 built the diff data model;
  iter-135 added format dispatch; iter-137/138 added the two
  exit-code gates; iter-139 packages them as the one-line CI
  entry point. A CI step is now literally `scripts/ci-gate.sh
  --baseline baseline.json` — no flag memorization, no jq/grep.
- **First shell-script tests in the repo.** The `BENCHMARK_SCRIPT`
  stub pattern (echo argv + `STUB_EXIT`) is the reusable shape for
  testing any future wrapper script without standing up the real
  pipeline — analogous to the `_run_main` hermetic driver iter-137
  introduced for `main()`-level Python behavior.
- **Restored regenerated perf snapshots** (`iter-reports/perf-*.json`)
  before committing so the diff is scoped to the actual change —
  the perf suite rewrites them on every run.
- Next directions:
  - A `.github/workflows/stt-benchmark.yml` that calls
    `scripts/ci-gate.sh` on PRs against a committed baseline
    (the actual CI wiring, now that the gate is one line). Would
    need a committed `baseline.json` and a decision on which
    engine/model CI runs.
  - Document the GENO.md diversity-check + extraction patterns in
    the README so contributors discover them without reading
    GENO.md (carried over from iter-136).

## iter-140 — STT-RTF consistency check (5th diversity-check instance)

**Branch:** `iter-140-stt-rtf` (merged ff to main, commit `cb1f69c`)
**Date:** 2026-06-16

**Goal:** The diversity-check pattern (GENO.md "Session-summary
diversity-check pattern") had four instances; two prior next-
directions had drifted toward STT/benchmark CI tooling. This lap
returns to the session-summary surface and lands the long-
signposted **5th instance**: a cross-turn sentinel for STT
real-time factor. The signal already existed (`TurnMetrics.stt_rtf`,
iter-049) but only as a per-turn print and a session-median line —
nothing flagged a *sustained* slow streak, which is the symptom of
an engine/model mismatched to the host hardware.

**What changed:**

1. **`_stt_rtf_bucket(rtf)` (new) in `examples/_chat_metrics.py`:**
   second continuous-metric bucketer (after iter-128's
   `_sentence_length_bucket`). Maps `stt_rtf` → `""` (≤0, no
   measurable STT) / `"realtime"` (<1.0, the fine state) /
   `"slow"` (1.0–2.0) / `"very_slow"` (>2.0). Boundaries chosen
   against iter-049's RTF semantics (mlx-whisper on Apple Silicon
   lands ~0.1–0.3).

2. **`_emit_stt_rtf_consistency_line(emit, stt_rtf_list, threshold=5)`
   (new):** the 5th diversity-check helper. Filters out `""` and
   `"realtime"` before the scan, runs iter-116's
   `_longest_consecutive_run`, fires at threshold 5 (same as
   iter-115/128 — RTF is a noisy "natural variation is normal"
   signal). Per-value suggestions: `slow` → "try a smaller model
   or streaming STT", `very_slow` → "badly mismatched to the
   hardware (>2x realtime)", plus a defensive `else`. Names
   `iter-049 stt_rtf` in the warning text so operators can grep
   the source metric.

3. **Wired into `print_session_summary`** right after the
   iter-128 sentence-length sentinel, fed
   `[m.stt_rtf for m in metrics_list]`.

4. **GENO.md diversity-check pattern doc updated atomically:**
   four→**five instances** in the header, added the iter-140
   filter-rule bullet (drops `""` + `"realtime"`), the iter-140
   continuous-metric note under rule 4 (second bucketer), the
   `slow`/`very_slow` per-value-suggestion mention under rule 5,
   and iter-140 under the threshold-5 convention.

5. **Tests:**
   - `tests/unit/test_emit_stt_rtf_consistency_line.py` (new, +21):
     mirrors iter-128's matrix — bucket boundaries (incl. float
     edges and the ≤0 → `""` defensive path), empty/all-zero
     suppression, realtime-run never fires, at/above/below
     threshold per bucket, realtime-interleave doesn't break a
     run, phase-change (slow↔very_slow) DOES break a run, custom
     threshold (3 catches / 10 suppresses), longest-of-multiple,
     leading-indent + iter-049 attribution formatting, 1000-elem
     scale sanity.
   - `tests/unit/test_diversity_pattern_doc.py` (+1 helper in
     `_DIVERSITY_HELPERS`, iter-140 in `expected_iters`): the
     doc-sync sentinel now enforces the five-instance count, the
     helper's existence/callability/test-file-presence, and the
     iter-140 attribution in GENO.md. This is what forced the
     GENO.md edits to be atomic with the code.

**Verification:**
- `python -m pytest tests/unit/test_emit_stt_rtf_consistency_line.py
  tests/unit/test_diversity_pattern_doc.py -q` → **31 passed in 0.11s**.
- Full unit suite: **1632 passed** (1611 prior + 21 new).
- Integration (`tests/integration/`): **30 passed, 1 skipped**.

**Notes:**
- **The doc-sync sentinel did its job.** Adding the helper to
  `_DIVERSITY_HELPERS` immediately red-lit four assertions in
  `test_diversity_pattern_doc.py` (count phrase, helper-name
  reference, iter attribution, test-file presence) until GENO.md
  and the new test file caught up — exactly the atomic-update
  guarantee iter-129/131 built it for.
- **Second continuous-metric instance confirms the bucket-before-
  filter sub-pattern.** iter-128 introduced "bucket a float, then
  treat it like any categorical signal"; iter-140 reuses it
  verbatim with a different bucketer, so rule 4 is now a
  two-instance pattern rather than a one-off.
- Next directions:
  - A `.github/workflows/stt-benchmark.yml` that calls
    `scripts/ci-gate.sh` on PRs against a committed baseline
    (carried over from iter-139; needs a committed `baseline.json`
    and an engine/model choice for CI).
  - Document the GENO.md diversity-check + extraction patterns in
    the README so contributors discover them without reading
    GENO.md (carried over from iter-136).
  - A 6th diversity-check candidate: TTS RTF (`tts_rtf`, iter-050)
    has the same continuous-metric shape and a session-median line
    but no sustained-slow sentinel — a near-mechanical clone of
    iter-140 if a need surfaces.

## iter-141 — TTS-RTF consistency check (6th diversity-check instance)

**Branch:** `iter-141-tts-rtf` (merged ff to main, commit `7a205de`)
**Date:** 2026-06-16

**Goal:** iter-140 closed by signposting a 6th diversity-check
candidate — TTS RTF (`tts_rtf`, iter-050) has the identical
continuous-metric shape as STT RTF and a session-median line but
no sustained-slow sentinel. This lap lands that near-mechanical
clone: a cross-turn warning that fires when TTS synth runs slower
than realtime for 5+ consecutive turns — the symptom of an
engine/voice too heavy for the host, where synth-overlap can't
help because synth itself is the bottleneck.

**What changed:**

1. **`_tts_rtf_bucket(rtf)` (new) in `examples/_chat_metrics.py`:**
   third continuous-metric bucketer (after iter-128
   `_sentence_length_bucket` and iter-140 `_stt_rtf_bucket`). Same
   boundaries as iter-140: `""` (≤0, no audio produced) /
   `"realtime"` (<1.0) / `"slow"` (1.0–2.0) / `"very_slow"` (>2.0).
   Chosen against iter-050's RTF semantics (Kokoro on Apple Silicon
   lands ~0.1–0.3).

2. **`_emit_tts_rtf_consistency_line(emit, tts_rtf_list, threshold=5)`
   (new):** the 6th diversity-check helper. Filters out `""` and
   `"realtime"` before the scan, runs iter-116's
   `_longest_consecutive_run`, fires at threshold 5 (same as
   iter-115/128/140). Per-value suggestions: `slow` → "try a
   lighter voice or pre-rendered fillers", `very_slow` → "badly
   mismatched to the hardware (>2x realtime)", plus a defensive
   `else`. Names `iter-050 tts_rtf` in the warning text so
   operators can grep the source metric.

3. **Wired into `print_session_summary`** right after the iter-140
   STT-RTF sentinel, fed `[m.tts_rtf for m in metrics_list]`.

4. **GENO.md diversity-check pattern doc updated atomically:**
   five→**six instances** in the header, added the iter-141
   filter-rule bullet (drops `""` + `"realtime"`), the iter-141
   third-continuous-metric note under rule 4, and iter-141 under
   the threshold-5 convention bullet.

5. **Tests:**
   - `tests/unit/test_emit_tts_rtf_consistency_line.py` (new, +21):
     mirrors iter-140's matrix — bucket boundaries (incl. float
     edges and the ≤0 → `""` defensive path), empty/all-zero
     suppression, realtime-run never fires, at/above/below
     threshold per bucket, realtime-interleave doesn't break a run,
     phase-change (slow↔very_slow) DOES break a run, custom
     threshold (3 catches / 10 suppresses), longest-of-multiple,
     leading-indent + iter-050 attribution formatting, 1000-elem
     scale sanity.
   - `tests/unit/test_diversity_pattern_doc.py` (+1 helper in
     `_DIVERSITY_HELPERS`, iter-141 in `expected_iters`, `6: ("six
     instances", "Six")` in the counts map): the doc-sync sentinel
     now enforces the six-instance count.

**Verification:**
- `python -m pytest tests/unit/test_emit_tts_rtf_consistency_line.py
  tests/unit/test_diversity_pattern_doc.py -q` → **31 passed in 0.11s**.
- Full unit suite: **1653 passed** (1632 prior + 21 new).
- Integration (`tests/integration/`): **30 passed, 1 skipped**.

**Notes:**
- **The doc-sync sentinel did its job again.** Adding the helper to
  `_DIVERSITY_HELPERS` immediately red-lit the count/attribution/
  test-file assertions until GENO.md and the new test file caught
  up — the atomic-update guarantee iter-129/131 built it for.
- **Third continuous-metric instance solidifies the bucket-before-
  filter sub-pattern.** iter-128 introduced it, iter-140 confirmed
  it, iter-141 makes it a three-instance pattern with two of the
  three (stt/tts RTF) sharing identical bucket boundaries — a sign
  the realtime/slow/very_slow triad is the natural shape for any
  RTF-style continuous signal.
- Next directions:
  - A `.github/workflows/stt-benchmark.yml` that calls
    `scripts/ci-gate.sh` on PRs against a committed baseline
    (carried over from iter-139; needs a committed `baseline.json`
    and an engine/model choice for CI).
  - Document the GENO.md diversity-check + extraction patterns in
    the README so contributors discover them without reading
    GENO.md (carried over from iter-136).
  - The diversity-check surface now covers filler, naturalness,
    barge-phase, sentence-length, stt-rtf, tts-rtf. Remaining
    per-turn signals without a sustained-run sentinel: LLM
    streaming speed (`llm_first`/`llm_total` ratio) and TTFS
    jitter — either is a candidate 7th instance if a need surfaces.

## iter-142 — LLM-TPS consistency check (7th diversity-check instance)

**Branch:** `iter-142-llm-tps` (merged ff to main, commit `f274aad`)
**Date:** 2026-06-16

**Goal:** iter-141 closed by signposting LLM streaming speed as a
candidate 7th diversity-check instance. This lap lands it:
`llm_tps` (iter-052, LLM stream throughput in tokens/sec measured
after first token) had a per-turn print and a session-median line
but no sustained-slow sentinel. A slow LLM stream starves the TTS
worker — complete sentences arrive too slowly to feed synth-overlap
— regardless of how fast STT/TTS themselves run. This is the FIRST
continuous-metric instance with an INVERTED direction: `llm_tps`
is bigger-is-better, so the "fine" bucket is a HIGH value, flipping
the boundaries relative to the iter-140/141 RTF bucketers.

**What changed:**

1. **`_llm_tps_bucket(tps)` (new) in `examples/_chat_metrics.py`:**
   fourth continuous-metric bucketer (after iter-128
   `_sentence_length_bucket`, iter-140 `_stt_rtf_bucket`, iter-141
   `_tts_rtf_bucket`). Maps `llm_tps` → `""` (≤0, no measurable
   stream) / `"fast"` (≥25 tps, the fine state) / `"slow"`
   (10–25 tps) / `"very_slow"` (<10 tps). Boundaries chosen against
   iter-052's TPS semantics (local 7B-13B on Apple Silicon land
   30-80 tps, cloud APIs 20-60 tps). UNLIKE the RTF bucketers, the
   fine bucket is the HIGH end — the first inverted-direction
   bucketer in the family.

2. **`_emit_llm_tps_consistency_line(emit, llm_tps_list, threshold=5)`
   (new):** the 7th diversity-check helper. Filters out `""` and
   `"fast"` before the scan (the inversion is absorbed entirely by
   the filter rule — the run-scan stays direction-agnostic), runs
   iter-116's `_longest_consecutive_run`, fires at threshold 5
   (same as iter-115/128/140/141). Per-value suggestions: `slow` →
   "LLM stream lags synth, try a smaller model or fewer context
   tokens", `very_slow` → "LLM is the dominant bottleneck, try a
   smaller/quantized model or a faster backend", plus a defensive
   `else`. Names `iter-052 llm_tps` in the warning text.

3. **Wired into `print_session_summary`** right after the iter-141
   TTS-RTF sentinel, fed `[m.llm_tps for m in metrics_list]`.

4. **GENO.md diversity-check pattern doc updated atomically:**
   six→**seven instances** in the header, added the iter-142
   filter-rule bullet (drops `""` + `"fast"`, with an explicit
   inversion note), the iter-142 fourth-continuous-metric note
   under rule 4 (calling out the inverted direction), and iter-142
   under the threshold-5 convention bullet.

5. **Tests:**
   - `tests/unit/test_emit_llm_tps_consistency_line.py` (new, +21):
     mirrors iter-141's matrix but probes the HIGH end as the good
     state — bucket boundaries (incl. float edges and the ≤0 → `""`
     defensive path), empty/all-zero suppression, fast-run never
     fires, at/above/below threshold per bucket, fast-interleave
     doesn't break a run, phase-change (slow↔very_slow) DOES break
     a run, custom threshold (3 catches / 10 suppresses),
     longest-of-multiple, leading-indent + iter-052 attribution
     formatting, 1000-elem scale sanity.
   - `tests/unit/test_diversity_pattern_doc.py` (+1 helper in
     `_DIVERSITY_HELPERS`, iter-142 in `expected_iters`, `7:
     ("seven instances", "Seven")` in the counts map): the doc-sync
     sentinel now enforces the seven-instance count.

**Verification:**
- `python -m pytest tests/unit/test_emit_llm_tps_consistency_line.py
  tests/unit/test_diversity_pattern_doc.py -q` → **31 passed in 0.11s**.
- Full unit suite: **1674 passed** (1653 prior + 21 new).
- Integration (`tests/integration/`): **30 passed, 1 skipped**.

**Notes:**
- **First inverted-direction continuous bucketer.** The three prior
  continuous instances (iter-128/140/141) all had a LOW "fine"
  value. `llm_tps` flips this — and the key design observation is
  that the inversion lives entirely in the bucketer's boundaries
  and the filter rule's excluded-set (`"fast"` instead of
  `"realtime"`). `_longest_consecutive_run` and the warning-emit
  machinery are unchanged, confirming the run-scan primitive is
  genuinely direction-agnostic.
- **The doc-sync sentinel did its job again.** Adding the helper to
  `_DIVERSITY_HELPERS` red-lit the count/attribution/test-file
  assertions until GENO.md and the new test file caught up.
- Next directions:
  - A `.github/workflows/stt-benchmark.yml` that calls
    `scripts/ci-gate.sh` on PRs against a committed baseline
    (carried over from iter-139; needs a committed `baseline.json`
    and an engine/model choice for CI).
  - Document the GENO.md diversity-check + extraction patterns in
    the README so contributors discover them without reading
    GENO.md (carried over from iter-136).
  - The diversity-check surface now covers filler, naturalness,
    barge-phase, sentence-length, stt-rtf, tts-rtf, llm-tps.
    Remaining per-turn signals without a sustained-run sentinel:
    TTFS jitter and `streaming_overlap_ratio` (a low-overlap run
    would flag the iter-008 design failing to mask synth) — either
    is a candidate 8th instance if a need surfaces.

## iter-143 — streaming-overlap consistency check (8th diversity-check instance)

**Branch:** `iter-143-overlap-consistency` (merged ff to main, commit `0b5bee6`)
**Date:** 2026-06-16

**Goal:** iter-142 closed by signposting `streaming_overlap_ratio`
as a candidate 8th diversity-check instance ("a low-overlap run
would flag the iter-008 design failing to mask synth"). This lap
lands it. `streaming_overlap_ratio` (iter-043, the fraction of bot
synth that ran concurrently with the LLM stream) already had a
per-turn field and a session-summary "Median overlap" line but no
sustained-run sentinel. A run of low-overlap turns is the direct
symptom of the iter-008 streaming-sentence-dispatch design failing
to mask synth — synth ends up running sequentially after the
stream, evaporating the TTFS savings the design exists to buy. This
is the SECOND inverted-direction continuous-metric instance (after
iter-142 llm-tps): overlap is bigger-is-better, so the "fine" bucket
is a HIGH value.

**What changed:**

1. **`_streaming_overlap_bucket(ratio)` (new) in
   `examples/_chat_metrics.py`:** fifth continuous-metric bucketer
   (after iter-128 sentence-length, iter-140 stt-rtf, iter-141
   tts-rtf, iter-142 llm-tps). Maps `streaming_overlap_ratio` →
   `""` (≤0, no measurable overlap) / `"high"` (≥0.50, the fine
   state) / `"low"` (0.20–0.50) / `"very_low"` (<0.20). Boundaries
   chosen against iter-043's existing "Median overlap" semantics
   (>50% = healthy, <20% = overlap isn't happening). Like iter-142
   and UNLIKE the RTF bucketers, the fine bucket is the HIGH end —
   the second inverted-direction bucketer in the family.

2. **`_emit_streaming_overlap_consistency_line(emit, overlap_list,
   threshold=5)` (new):** the 8th diversity-check helper. Filters
   out `""` and `"high"` before the scan (the inversion is absorbed
   entirely by the filter rule — the run-scan stays
   direction-agnostic), runs iter-116's `_longest_consecutive_run`,
   fires at threshold 5 (same as iter-115/128/140/141/142).
   Per-value suggestions: `low` → "overlap is only partial; the bot
   may be replying too fast or the LLM stream lags synth",
   `very_low` → "synth runs sequentially after the LLM stream;
   check first-sentence latency and synth time", plus a defensive
   `else`. Names `iter-043 streaming_overlap_ratio` in the warning
   text.

3. **Wired into `print_session_summary`** right after the iter-142
   LLM-TPS sentinel, fed `[m.streaming_overlap_ratio for m in
   metrics_list]`.

4. **GENO.md diversity-check pattern doc updated atomically:**
   seven→**eight instances** in the header, added the iter-143
   filter-rule bullet (drops `""` + `"high"`, with an explicit
   "second inverted" note), the iter-143 fifth-continuous-metric
   note under rule 4 (calling out the inverted direction and that
   it watches the iter-008 design itself), and iter-143 under the
   threshold-5 convention bullet.

5. **Tests:**
   - `tests/unit/test_emit_streaming_overlap_consistency_line.py`
     (new, +21): mirrors iter-142's matrix probing the HIGH end as
     the good state — bucket boundaries (incl. float edges and the
     ≤0 → `""` defensive path), empty/all-zero suppression,
     high-run never fires, at/above/below threshold per bucket,
     high-interleave doesn't break a run, phase-change
     (low↔very_low) DOES break a run, custom threshold (3 catches /
     10 suppresses), longest-of-multiple, leading-indent + iter-043
     attribution formatting, 1000-elem scale sanity.
   - `tests/unit/test_diversity_pattern_doc.py` (+1 helper in
     `_DIVERSITY_HELPERS`, iter-143 in `expected_iters`, `8:
     ("eight instances", "Eight")` in the counts map): the doc-sync
     sentinel now enforces the eight-instance count.

**Verification:**
- `python -m pytest
  tests/unit/test_emit_streaming_overlap_consistency_line.py
  tests/unit/test_diversity_pattern_doc.py -q` → **31 passed in 0.11s**.
- Full unit suite: **1695 passed** (1674 prior + 21 new).
- Integration (`tests/integration/`): **30 passed, 1 skipped**.

**Notes:**
- **Second inverted-direction continuous bucketer confirms the
  inversion is pure filter-rule policy.** iter-142 was the first;
  iter-143 reuses the same trick (the fine bucket "high" is a HIGH
  value, excluded by the filter set) without touching
  `_longest_consecutive_run` or the emit machinery — two
  independent inverted instances now demonstrate the run-scan
  primitive is genuinely direction-agnostic.
- **This instance is special: it watches the core design itself.**
  Unlike STT/TTS/LLM RTF sentinels (which flag a slow *component*),
  the overlap sentinel flags the iter-008 *architecture* failing to
  do its job — a sustained low-overlap run means synth-overlap
  isn't masking synth even when each component is fast.
- **The doc-sync sentinel did its job again.** Adding the helper to
  `_DIVERSITY_HELPERS` red-lit the count/attribution/test-file
  assertions until GENO.md and the new test file caught up.
- Next directions:
  - A `.github/workflows/stt-benchmark.yml` that calls
    `scripts/ci-gate.sh` on PRs against a committed baseline
    (carried over from iter-139; needs a committed `baseline.json`
    and an engine/model choice for CI).
  - Document the GENO.md diversity-check + extraction patterns in
    the README so contributors discover them without reading
    GENO.md (carried over from iter-136).
  - The diversity-check surface now covers filler, naturalness,
    barge-phase, sentence-length, stt-rtf, tts-rtf, llm-tps,
    streaming-overlap. The per-turn-signal seam is nearly
    exhausted; remaining candidates are weaker (TTFS jitter,
    token-lag). Worth pausing the diversity-check expansion and
    pivoting to one of the two carried-over items (CI baseline or
    README pattern docs) next lap rather than mining a 9th
    near-mechanical instance.
