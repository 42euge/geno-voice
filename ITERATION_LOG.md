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
